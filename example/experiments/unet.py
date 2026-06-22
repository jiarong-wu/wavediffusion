# Adapted from PNDM implmentation (https://github.com/luping-liu/PNDM)
# which is adapted from DDIM implementation (https://github.com/ermongroup/ddim)
# Test different options for U-Net architecture, such as periodic padding and attention. See `example/unet_experiments.ipynb` for experiments with these options.

import torch
from einops import rearrange
from itertools import pairwise
from torch import nn
import torch.nn.functional as F
from wavediffusion.model import (
    alpha, Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
)

down_stride = 4
down_kernel = 5

# down_stride = 2
# down_kernel = 3

# =============================================================================
# Periodic Padding Utilities
# =============================================================================

class PeriodicPadW(nn.Module):
    """Circular padding in W (width), zero padding in H (height)"""
    def __init__(self, pad_w, pad_h=None):
        super().__init__()
        self.pad_w = pad_w
        self.pad_h = pad_h if pad_h is not None else pad_w
    
    def forward(self, x):
        # x: (B, C, H, W)
        # F.pad takes (left, right, top, bottom)
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')  # W: periodic
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='constant', value=0)  # H: zero
        return x
    
def PeriodicConv2d(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv2d with periodic BC in width, zero padding in height"""
    pad = kernel_size // 2
    return nn.Sequential(
        PeriodicPadW(pad_w=pad, pad_h=pad),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0),
    )

def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)

def Upsample(ch):
    """2x upsample with periodic conv"""
    pad = down_kernel // 2
    return nn.Sequential(
        nn.Upsample(scale_factor=down_stride, mode='nearest'),
        PeriodicPadW(pad_w=pad, pad_h=pad),
        nn.Conv2d(ch, ch, kernel_size=down_kernel, stride=1, padding=0),
    )

def Downsample(ch):
    """ 2x downsample with periodic conv and stride 2 """
    
    return PeriodicConv2d(ch, ch, kernel_size=down_kernel, stride=down_stride)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            PeriodicConv2d(in_ch, out_ch, kernel_size=3, stride=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            PeriodicConv2d(out_ch, out_ch, kernel_size=3, stride=1),
        )
        if self.in_ch != self.out_ch:
            kernel_stride_padding = (3,1,1) if self.use_conv_shortcut else (1,1,0)
            self.shortcut = nn.Conv2d(in_ch, out_ch, *kernel_stride_padding)

    def forward(self, x, temb):
        h = x
        h = self.layer1(h)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.layer2(h)
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        # Normalize input along the channel dimension
        self.norm = Normalize(ch)
        # Attention over D: (B, N, D) -> (B, N, D)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        # Apply 1x1 convolution for projection
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        # temb is currently not used, but included for CondSequential to work
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h_)

# class Unet(nn.Module, ModelMixin):
#     def __init__(self, in_dim, in_ch, out_ch,
#                  ch               = 128,
#                  ch_mult          = (1,2,2,2),
#                  embed_ch_mult    = 4,
#                  num_res_blocks   = 2,
#                  attn_resolutions = (16,),
#                  dropout          = 0.1,
#                  resamp_with_conv = True,
#                  sig_embed        = None,
#                  cond_embed       = None,
#                  ):
#         super().__init__()

#         self.ch = ch
#         self.in_dim = in_dim
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.input_dims = (in_ch, in_dim, in_dim)
#         self.temb_ch = self.ch * embed_ch_mult

#         # Embeddings
#         self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch)
#         make_block = lambda in_ch, out_ch: ResnetBlock(
#             in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout
#         )
#         self.cond_embed = cond_embed

#         # Downsampling
#         curr_res = in_dim
#         in_ch_dim = [ch * m for m in (1,)+ch_mult]
#         self.conv_in = PeriodicConv2d(in_ch, self.ch, kernel_size=3, stride=1)
#         self.downs = nn.ModuleList()
#         for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
#             down = nn.Module()
#             down.blocks = nn.ModuleList()
#             for _ in range(self.num_res_blocks):
#                 block = [make_block(block_in,block_out)]
#                 if curr_res in attn_resolutions:
#                     block.append(AttnBlock(block_out))
#                 down.blocks.append(CondSequential(*block))
#                 block_in = block_out
#             if i < self.num_resolutions - 1: # Not last iter
#                 down.downsample = Downsample(block_in)
#                 curr_res = curr_res // down_stride
#             self.downs.append(down)

#         # Middle
#         self.mid = CondSequential(
#             make_block(block_in, block_in),
#             AttnBlock(block_in),
#             make_block(block_in, block_in)
#         )

#         # Upsampling
#         self.ups = nn.ModuleList()
#         for i_level, (block_out, next_skip_in) in enumerate(pairwise(reversed(in_ch_dim))):
#             up = nn.Module()
#             up.blocks = nn.ModuleList()
#             skip_in = block_out
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = next_skip_in
#                 block = [make_block(block_in+skip_in, block_out)]
#                 if curr_res in attn_resolutions:
#                     block.append(AttnBlock(block_out))
#                 up.blocks.append(CondSequential(*block))
#                 block_in = block_out
#             if i_level < self.num_resolutions - 1: # Not last iter
#                 up.upsample = Upsample(block_in)
#                 curr_res = curr_res * down_stride
#             self.ups.append(up)

#         # Out
#         self.out_layer = nn.Sequential(
#             Normalize(block_in),
#             nn.SiLU(),
#             PeriodicConv2d(block_in, out_ch, kernel_size=3, stride=1),
#         )

#     def forward(self, x, sigma, cond=None):
#         assert x.shape[2] == x.shape[3] == self.in_dim

#         # Embeddings
#         emb = self.sig_embed(x.shape[0], sigma.squeeze())
#         if self.cond_embed is not None:
#             assert cond is not None and x.shape[0] == cond.shape[0], \
#                 'Conditioning must have same batches as x!'
#             emb += self.cond_embed(cond)

#         # downsampling
#         hs = [self.conv_in(x)]
#         for down in self.downs:
#             for block in down.blocks:
#                 h = block(hs[-1], emb)
#                 hs.append(h)
#             if hasattr(down, 'downsample'):
#                 hs.append(down.downsample(hs[-1]))

#         # middle
#         h = self.mid(hs[-1], emb)

#         # upsampling
#         for up in self.ups:
#             for block in up.blocks:
#                 h = block(torch.cat([h, hs.pop()], dim=1), emb)
#             if hasattr(up, 'upsample'):
#                 h = up.upsample(h)

#         # out
#         return self.out_layer(h)
    
### Takes spatial field as preconditioning ###
from typing import Tuple
class myUnet(nn.Module, ModelMixin):
    def __init__(self, in_dim, in_ch, out_ch, precond_ch,
                 scale            : Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
                 ch               = 128,
                 ch_mult          = (1,2,2,2),
                 embed_ch_mult    = 4,
                 num_res_blocks   = 2,
                 attn_resolutions = (16,),
                 dropout          = 0.1,
                 sig_embed        = None,
                 ):
        super().__init__()

        self.ch = ch
        self.in_dim = in_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.precond_ch = precond_ch
        self.input_dims = (in_ch, in_dim, in_dim)
        self.temb_ch = self.ch * embed_ch_mult
        
        # Saving scales for construction of dataset
        self.register_buffer('meanx', scale[0])
        self.register_buffer('stdx', scale[1])
        self.register_buffer('meanf', scale[2])
        self.register_buffer('stdf', scale[3])

        # Embeddings
        self.sig_embed = sig_embed or SigmaEmbedderSinCos(self.temb_ch)
        make_block = lambda in_ch, out_ch: ResnetBlock(
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout
        )

        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,)+ch_mult]
        # Only the first conv layer takes precond channels
        self.conv_in = PeriodicConv2d(in_ch+precond_ch, self.ch, kernel_size=3, stride=1)
        self.downs = nn.ModuleList()
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            print(curr_res)
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in,block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in)
                curr_res = curr_res // down_stride
            self.downs.append(down)

        # Middle
        self.mid = CondSequential(
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
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = next_skip_in
                block = [make_block(block_in+skip_in, block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                up.blocks.append(CondSequential(*block))
                block_in = block_out
            if i_level < self.num_resolutions - 1: # Not last iter
                up.upsample = Upsample(block_in)
                curr_res = curr_res * down_stride
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            PeriodicConv2d(block_in, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x, sigma, cond=None): # Here cond has shape (B, C, H, W)
        assert x.shape[2] == x.shape[3] == self.in_dim

        # Embeddings
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        # Conditions as maps
        assert cond is not None and x.shape[0] == cond.shape[0] and x.shape[2] == cond.shape[2] and x.shape[3] == cond.shape[3], \
            'Conditioning must have same shape as x!'
        inputs = torch.cat([x, cond], dim=1)

        # downsampling
        hs = [self.conv_in(inputs)]
        for down in self.downs:
            for block in down.blocks:
                h = block(hs[-1], emb)
                hs.append(h)
            if hasattr(down, 'downsample'):
                hs.append(down.downsample(hs[-1]))

        # middle
        h = self.mid(hs[-1], emb)

        # upsampling
        for up in self.ups:
            for block in up.blocks:
                h = block(torch.cat([h, hs.pop()], dim=1), emb)
            if hasattr(up, 'upsample'):
                h = up.upsample(h)

        # out
        return self.out_layer(h)
