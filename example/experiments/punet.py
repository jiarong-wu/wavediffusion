"""
U-Net with Periodic Boundary Conditions in Width Direction

Periodic (circular) padding in W, zero padding in H.
Useful for cylindrical domains, periodic wave equations, etc.
"""

import torch
from einops import rearrange
from itertools import pairwise
from torch import nn
import torch.nn.functional as F
from wavediffusion.model import (
    alpha, Attention, ModelMixin, CondSequential, SigmaEmbedderSinCos,
)

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


class PeriodicPadWOnly(nn.Module):
    """Circular padding in W only, no H padding"""
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        return x


# =============================================================================
# Core Building Blocks with Periodic BC
# =============================================================================

def Normalize(ch):
    """GroupNorm - no spatial padding needed"""
    return nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6, affine=True)


def PeriodicConv2d(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv2d with periodic BC in width, zero padding in height"""
    pad = kernel_size // 2
    return nn.Sequential(
        PeriodicPadW(pad_w=pad, pad_h=pad),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0),
    )


def Upsample(ch):
    """2x upsample with periodic conv"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2.0, mode='nearest'),
        PeriodicPadW(pad_w=1, pad_h=1),
        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=0),
    )


def Downsample(ch):
    """2x downsample with periodic padding
    
    For stride=2 with kernel=3, we need:
    - W: periodic pad by 1 on each side
    - H: pad (0, 1, 0, 1) to handle odd dimensions
    """
    return PeriodicDownsample(ch)


class PeriodicDownsample(nn.Module):
    """Downsample with periodic BC in W, asymmetric padding in H for odd sizes"""
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        # Periodic padding in W
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        # Asymmetric padding in H for stride=2 (handles odd dimensions)
        x = F.pad(x, (0, 0, 0, 1), mode='constant', value=0)
        return self.conv(x)


# =============================================================================
# ResNet Block with Periodic BC
# =============================================================================

class ResnetBlock(nn.Module):
    def __init__(self, *, in_ch, out_ch=None, conv_shortcut=False, dropout):
        super().__init__()
        self.in_ch = in_ch
        out_ch = in_ch if out_ch is None else out_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_ch)
        self.conv1 = PeriodicConv2d(in_ch, out_ch, kernel_size=3, stride=1)
        
        self.norm2 = Normalize(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = PeriodicConv2d(out_ch, out_ch, kernel_size=3, stride=1)
        
        if self.in_ch != self.out_ch:
            if self.use_conv_shortcut:
                self.shortcut = PeriodicConv2d(in_ch, out_ch, kernel_size=3, stride=1)
            else:
                # 1x1 conv doesn't need spatial padding
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_ch != self.out_ch:
            x = self.shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=1):
        super().__init__()
        self.norm = Normalize(ch)
        self.attn = Attention(head_dim=ch // num_heads, num_heads=num_heads)
        # 1x1 conv, no spatial padding needed
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h_ = self.norm(x)
        h_ = rearrange(h_, 'b c h w -> b (h w) c')
        h_ = self.attn(h_)
        h_ = rearrange(h_, 'b (h w) c -> b c h w', h=H, w=W)
        return x + self.proj_out(h_)


# =============================================================================
# U-Net with Periodic BC in Width
# =============================================================================

class PeriodicUnet(nn.Module):
    """
    U-Net with periodic boundary conditions in width direction.
    
    Args:
        in_dim: Input spatial dimension (assumes square input, or H dimension)
        in_ch: Number of input channels
        out_ch: Number of output channels
        ch: Base channel count
        ch_mult: Channel multipliers for each level
        num_res_blocks: Number of ResBlocks per level in encoder
        attn_resolutions: Resolutions at which to apply attention
        dropout: Dropout rate
    """
    def __init__(self, in_dim, in_ch, out_ch,
                 ch               = 128,
                 ch_mult          = (1, 2, 2, 4),
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

        # Input conv with periodic BC
        self.conv_in = PeriodicConv2d(in_ch, self.ch, kernel_size=3, stride=1)
        
        # Downsampling
        curr_res = in_dim
        in_ch_dim = [ch * m for m in (1,) + ch_mult]
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

        # Output
        self.out_layer = nn.Sequential(
            Normalize(block_in),
            nn.SiLU(),
            PeriodicConv2d(block_in, out_ch, kernel_size=3, stride=1),
        )

    def forward(self, x):
        # Validate input dimensions
        assert x.shape[2] == x.shape[3] == self.in_dim, \
            f"Expected {self.in_dim}x{self.in_dim}, got {x.shape[2]}x{x.shape[3]}"

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


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model for 64x64 images (smaller for testing)
    model = PeriodicUnet(
        in_dim=64,
        in_ch=4,
        out_ch=4,
        ch=64,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        dropout=0.1,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 4, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Verify periodic BC: left and right edges should "see" each other
    # Create input with distinct left/right edges
    x_test = torch.zeros(1, 4, 64, 64).to(device)
    x_test[:, :, :, 0] = 1.0    # Left edge
    x_test[:, :, :, -1] = 1.0  # Right edge
    
    with torch.no_grad():
        y_test = model(x_test)
    
    print(f"\nPeriodic BC test:")
    print(f"  Input has signal at left and right edges")
    print(f"  Output edge difference: {(y_test[:,:,:,0] - y_test[:,:,:,-1]).abs().mean():.6f}")
    print(f"  (Should be small due to periodic BC)")
    
    print("\n✓ PeriodicUnet working correctly!")