import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from functools import partial


def repeat_interleave(x, n):
    x = x.unsqueeze(2)
    x = x.repeat(1, 1, n, 1, 1)
    x = x.reshape(x.shape[0], x.shape[1] * n, x.shape[3], x.shape[4])
    return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        return x


class FFNifiedAttention(nn.Sequential):   # token mixer 
    def __init__(self, dim, norm='LN'):
        norm_ = [LayerNorm2d(dim)] if norm == 'LN' else [nn.BatchNorm2d(dim)] if norm == 'BN' else []
        proj = [nn.Conv2d(dim, dim, 1, 1, 0)]
        agg = []
        agg.append(nn.Conv2d(dim, dim, 3, 1, 1, groups = dim))
        agg.append(nn.GELU())
        agg.append(nn.Conv2d(dim, dim, 3, 1, 1, groups = dim))
        forward = norm_ + proj + agg if norm == 'LN' else proj + norm_ + agg  # LN before proj
        super().__init__(*forward)
        
        
class ConvFFN(nn.Sequential):   # channel mixer
    def __init__(self, dim, expansion_ratio, norm='LN'):
        norm_ = [LayerNorm2d(dim)] if norm == 'LN' else [nn.BatchNorm2d(dim)] if norm == 'BN' else []
        proj = [nn.Conv2d(dim, dim, 3, 1, 1, groups = dim)]
        agg = [nn.Conv2d(dim, int(dim * expansion_ratio), 1, 1, 0), nn.GELU(), nn.Conv2d(int(dim * expansion_ratio), dim, 1, 1, 0)]
        forward = norm_ + proj + agg if norm == 'LN' else proj + norm_ + agg
        super().__init__(*forward)


class FFBlock(nn.Module):
    def __init__(self, dim, expansion_ratio, norm):
        super().__init__()
        
        self.token_mixer = FFNifiedAttention(dim, norm)
        self.channel_mixer = ConvFFN(dim, expansion_ratio, norm)
        
    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


@ARCH_REGISTRY.register()
class FFNetSR(nn.Module):
    def __init__(
        self, scale, dim, depth, expansion_ratio, norm=None,
        is_coreml=False
    ):
        super().__init__()
        self.scale = scale

        # Shallow feature extractor
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        
        # Deep feature extractor
        self.body = nn.Sequential(*[
            FFBlock(dim, expansion_ratio, norm)
            for _ in range(depth)
        ])
        self.conv_bef_up = nn.Conv2d(dim, 3 * scale ** 2, 3, 1, 1)
        self.repeat_op = partial(repeat_interleave, n=scale ** 2) if is_coreml else partial(torch.repeat_interleave, repeats=scale ** 2, dim=1)
        
        # Upscaling Module
        self.up = nn.PixelShuffle(scale)
        
    def forward(self, x):
        x = self.conv_bef_up(
            self.body(self.proj(x))
        ) + self.repeat_op(x)
        x = self.up(x)
        return x
