# Adapted from PNDM implmentation (https://github.com/luping-liu/PNDM)
# which is adapted from DDIM implementation (https://github.com/ermongroup/ddim)

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from itertools import pairwise
from torch import nn
from .model import (
    alpha, Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
)

def Normalize(ch):
    return torch.nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)

class GeoConv2d(nn.Module):
    ''' Conv2d that wraps circularly along width (longitude, last dim) and
        pads with zeros along height (latitude), since the globe is periodic
        in longitude but not in latitude (poles are real boundaries). '''
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, periodic=True):
        super().__init__()
        self.periodic = periodic
        self.pad = kernel_size // 2
        padding = 0 if periodic else self.pad
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.periodic and self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, self.pad, self.pad), mode='constant', value=0)
        return self.conv(x)

def Upsample(ch, periodic=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        GeoConv2d(ch, ch, kernel_size=3, stride=1, periodic=periodic),
    )

class GeoDownsample(nn.Module):
    def __init__(self, ch, periodic=False):
        super().__init__()
        self.periodic = periodic
        self.conv = torch.nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.periodic:
            x = F.pad(x, (0, 1, 0, 0), mode='circular')
        else:
            x = F.pad(x, (0, 1, 0, 0), mode='constant', value=0)
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        return self.conv(x)

def Downsample(ch, periodic=False):
    return GeoDownsample(ch, periodic=periodic)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False,
                 dropout, temb_channels=512, periodic=False):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.layer1 = nn.Sequential(
            Normalize(in_ch),
            nn.SiLU(),
            GeoConv2d(in_ch, out_ch, kernel_size=3, stride=1, periodic=periodic),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            torch.nn.Linear(temb_channels, out_ch),
        )
        self.layer2 = nn.Sequential(
            Normalize(out_ch),
            nn.SiLU(),
            torch.nn.Dropout(dropout),
            GeoConv2d(out_ch, out_ch, kernel_size=3, stride=1, periodic=periodic),
        )
        if self.in_ch != self.out_ch:
            if self.use_conv_shortcut:
                self.shortcut = GeoConv2d(in_ch, out_ch, kernel_size=3, stride=1, periodic=periodic)
            else:
                self.shortcut = torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

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
                 periodic         = False,
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
            in_ch=in_ch, out_ch=out_ch, temb_channels=self.temb_ch, dropout=dropout, periodic=periodic
        )

        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,)+ch_mult]
        # Only the first conv layer takes precond channels
        self.conv_in = GeoConv2d(in_ch+precond_ch, self.ch, kernel_size=3, stride=1, periodic=periodic)
        self.downs = nn.ModuleList()
        for i, (block_in, block_out) in enumerate(pairwise(in_ch_dim)):
            down = nn.Module()
            down.blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block = [make_block(block_in,block_out)]
                if curr_res in attn_resolutions:
                    block.append(AttnBlock(block_out))
                down.blocks.append(CondSequential(*block))
                block_in = block_out
            if i < self.num_resolutions - 1: # Not last iter
                down.downsample = Downsample(block_in, periodic=periodic)
                curr_res = curr_res // 2
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
                up.upsample = Upsample(block_in, periodic=periodic)
                curr_res = curr_res * 2
            self.ups.append(up)

        # Out
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            GeoConv2d(block_in, out_ch, kernel_size=3, stride=1, periodic=periodic),
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
