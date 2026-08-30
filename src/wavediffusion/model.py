import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


## Basic functions used by all models

class ModelMixin:
    def rand_input(self, batchsize):
        assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
        return torch.randn((batchsize,) + self.input_dims)

    # Currently predicts eps, override following methods to predict, for example, x0
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond))

    # Only predict on not masked part (since eps also is masked    
    def get_loss_masked(self, x0, sigma, eps, cond=None, mask=None, loss=nn.MSELoss):
        if mask != None:
            return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond)*mask)
        else:
            print('Should provide mask!')

    def predict_eps(self, x, sigma, cond=None):
        return self(x, sigma, cond=cond)

    def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
        if cond is None or cfg_scale == 0:
            return self.predict_eps(x, sigma, cond=cond)
        assert sigma.shape == tuple(), 'CFG sampling only supports singleton sigma!'
        uncond = torch.full_like(cond, self.cond_embed.null_cond) # (B,)
        eps_cond, eps_uncond = self.predict_eps(                  # (B,), (B,)
            torch.cat([x, x]), sigma, torch.cat([cond, uncond])   # (2B,)
        ).chunk(2)
        return eps_cond + cfg_scale * (eps_cond - eps_uncond)

def get_sigma_embeds(batches, sigma, scaling_factor=0.5, log_scale=True):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    if log_scale:
        sigma = torch.log(sigma)
    s = sigma.unsqueeze(1) * scaling_factor
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)

# A simple embedding that works just as well as usual sinusoidal embedding
class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5, log_scale=True):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.log_scale = log_scale
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, batches, sigma):
        sig_embed = get_sigma_embeds(batches, sigma,
                                     self.scaling_factor,
                                     self.log_scale)                      # (B, 2)
        return self.mlp(sig_embed)                                        # (B, D)


## Modifiers for models, such as including scaling or changing model predictions

def alpha(sigma):
    return 1/(1+sigma**2)

# Scale model input so that its norm stays constant for all sigma
def Scaled(cls: ModelMixin):
    def forward(self, x, sigma, cond=None):
        return cls.forward(self, x * alpha(sigma).sqrt(), sigma, cond=cond)
    return type(cls.__name__ + 'Scaled', (cls,), dict(forward=forward))

# Train model to predict x0 instead of eps
def PredX0(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(x0, self(x0 + sigma * eps, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        x0_hat = self(x, sigma, cond=cond)
        return (x - x0_hat)/sigma
    def get_loss_masked(self, x0, sigma, eps, cond=None, mask=None, loss=nn.MSELoss):
        if mask != None:
            return loss()(x0*mask, self(x0 + sigma * eps, sigma, cond=cond)*mask)
        else:
            print('Should provide mask!')
    return type(cls.__name__ + 'PredX0', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps, get_loss_masked=get_loss_masked))

## Common functions for other models

class CondSequential(nn.Sequential):
    def forward(self, x, cond):
        for module in self._modules.values():
            x = module(x, cond)
        return x

class Attention(nn.Module):
    def __init__(self, head_dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # (B, N, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * head_dim
        q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
                            h=self.num_heads, k=self.head_dim)
        x = rearrange(F.scaled_dot_product_attention(q, k, v),
                      'b h n k -> b n (h k)')
        return self.proj(x)

