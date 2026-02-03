import math
import torch
from einops import rearrange
from itertools import pairwise
from torch import nn
from .model import Attention

def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)

def Upsample(ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        torch.nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
    )

def Downsample(ch):
    return nn.Sequential(
        nn.ConstantPad2d((0, 1, 0, 1), 0),
        torch.nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0),
    )

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        # Removed: temb_proj
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3,1,1) if self.use_conv_shortcut else (1,1,0)
            self.shortcut = torch.nn.Conv2d(in_ch, out_ch, *kernel_stride_padding)

    def forward(self, x):
        h = x
        h = self.layer1(h)
        # Removed: temb addition
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        self.norm = Normalize(ch)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h_)

class plainUnet(nn.Module):
    def __init__(self, in_dim, in_ch, out_ch, 
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 num_res_blocks   = 2,
                 attn_resolutions = (16,),
                 dropout          = 0.1,
                 ):
        super().__init__()

        self.ch = ch
        self.in_dim = in_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, dropout=dropout
        )

        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,)+ch_mult]
        self.conv_in = torch.nn.Conv2d(in_ch, self.ch, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList()
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in, block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(nn.Sequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.downs.append(down)

        # Middle
        self.mid = nn.Sequential(
            make_block(block_in, block_in),
            AttnBlock(block_in),
            make_block(block_in, block_in)
        )

        # Upsampling
        self.ups = nn.ModuleList()
        for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
            up = nn.Module()
            up.blocks = nn.ModuleList()
            skip_in = block_out
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = next_skip_in
                block = [make_block(block_in + skip_in, block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                up.blocks.append(nn.Sequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.in_dim
        
        # Downsampling
        hs = [self.conv_in(x)]
        for down in self.downs:
            for block in down.blocks:
                hs.append(block(hs[-1]))
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1]))

        # Middle
        h = self.mid(hs[-1])

        # Upsampling
        for up in self.ups:
            for block in up.blocks:
                h = block(torch.cat([h, hs.pop()], dim=1))
            if hasattr(up, 'upsample'):
                h = up.upsample(h)

        return self.out_layer(h)
    

from torch.utils.data import DataLoader
from typing import Optional
from accelerate import Accelerator
from tqdm.auto import tqdm
from types import SimpleNamespace  
def masked_training_loop_plain(loader      : DataLoader,
                  model       : nn.Module,
                  accelerator : Optional[Accelerator] = None,
                  epochs      : int = 10000,
                  lr          : float = 1e-3,
                  start_epoch : int = 0):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    global_step = 0
    for epoch in (pbar := tqdm(range(start_epoch+1, start_epoch+epochs+1))):
        for x, f, mask in loader:
            model.train()
            optimizer.zero_grad()
            # Add mask to noise if provided (mask shape: (1, 1, H, W))
            if mask is not None:
                pred = model.forward(f)
                loss = nn.MSELoss()(x*mask, pred*mask)
            else:
                pred = model.forward(f)
                loss = nn.MSELoss()(x, pred)

            accelerator.backward(loss)
            optimizer.step()
            yield SimpleNamespace(
                loss=loss.detach(), step=global_step, epoch=epoch, pbar=pbar,
            )
            global_step += 1 
            
@torch.no_grad()
def evaluate_plain(
    model: nn.Module,
    ema: nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    ) -> torch.Tensor:

    model.eval()
    total_loss, count = 0.0, 0   
    with ema.average_parameters():
        for x, f, mask in loader:
            x = x.to(accelerator.device)
            f = f.to(accelerator.device)
            mask = mask.to(accelerator.device)            
            if mask is not None:
                pred = model.forward(f)
                loss = nn.MSELoss()(x*mask, pred*mask)
            else:
                pred = model.forward(f)
                loss = nn.MSELoss()(x, pred)
            total_loss += loss.item() * x.shape[0]
            count += x.shape[0]
    model.train()
    val_loss_tensor = torch.tensor(total_loss / count, device=next(model.parameters()).device)
    return val_loss_tensor