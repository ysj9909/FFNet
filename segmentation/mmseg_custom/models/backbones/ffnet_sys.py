# code for backbone of system-level segmentors.

import torch
import torch.nn as nn
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from timm.models.layers import trunc_normal_, LayerNorm2d
import numpy as np
from pathlib import Path

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from mmcv.cnn.bricks import DropPath


class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, use_sync_bn = False):
        super().__init__()
        self.in_channels = a
        self.groups = groups
        self.kernel_size = ks
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        if use_sync_bn:
            self.add_module('bn', torch.nn.SyncBatchNorm(b))
        else:
            self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepVGGDW(torch.nn.Module):
    def __init__(self, in_dim, out_dim = None, ks = 7, stride = 1, use_sync_bn = False):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ks = ks
        self.stride = stride
        
        if ks > 3:
            ks_ = ks - 4
            self.conv1 = Conv2d_BN(in_dim, out_dim, ks, stride, ks // 2, groups = in_dim, use_sync_bn = use_sync_bn)
            self.conv2 = Conv2d_BN(in_dim, out_dim, ks_, stride, ks_ // 2, groups = in_dim, use_sync_bn = use_sync_bn)
            self.strip_conv1 = Conv2d_BN(in_dim, out_dim, (1, ks), pad = (0, ks//2), groups = in_dim, use_sync_bn = use_sync_bn)
            self.strip_conv2 = Conv2d_BN(in_dim, out_dim, (ks, 1), pad = (ks//2, 0), groups = in_dim, use_sync_bn = use_sync_bn)
        else:
            self.conv1 = Conv2d_BN(in_dim, out_dim, ks, stride, ks // 2, groups = in_dim, use_sync_bn = use_sync_bn)
            self.strip_conv1 = Conv2d_BN(in_dim, out_dim, (1, ks), pad = (0, ks//2), groups = in_dim, use_sync_bn = use_sync_bn)
            self.strip_conv2 = Conv2d_BN(in_dim, out_dim, (ks, 1), pad = (ks//2, 0), groups = in_dim, use_sync_bn = use_sync_bn)

    
    def forward(self, x):
        if self.ks > 3:
            return self.conv1(x) + self.conv2(x) + self.strip_conv1(x) + self.strip_conv2(x)
        else:
            return self.conv1(x) + self.strip_conv1(x) + self.strip_conv2(x) 
    
    @torch.no_grad()
    def fuse(self):
        if self.ks > 3:
            ks = self.ks
            conv1 = self.conv1.fuse()
            conv2 = self.conv2.fuse()
            strip_conv1 = self.strip_conv1.fuse()
            strip_conv2 = self.strip_conv2.fuse()
            
            conv1_w = conv1.weight
            conv1_b = conv1.bias
            conv2_w = conv2.weight
            conv2_b = conv2.bias
            strip_conv1_w = strip_conv1.weight
            strip_conv1_b = strip_conv1.bias
            strip_conv2_w = strip_conv2.weight
            strip_conv2_b = strip_conv2.bias
            
            conv2_w = torch.nn.functional.pad(conv2_w, [2,2,2,2])
            strip_conv1_w = torch.nn.functional.pad(strip_conv1_w, [0, 0, ks//2, ks//2])
            strip_conv2_w = torch.nn.functional.pad(strip_conv2_w, [ks//2, ks//2, 0, 0])

            final_conv_w = conv1_w + conv2_w + strip_conv1_w + strip_conv2_w
            final_conv_b = conv1_b + conv2_b + strip_conv1_b + strip_conv2_b
            
            fin_conv = nn.Conv2d(self.in_dim, self.out_dim, self.ks, stride = self.stride,
                             padding = self.ks // 2, groups = self.in_dim, bias = True)

            fin_conv.weight.data.copy_(final_conv_w)
            fin_conv.bias.data.copy_(final_conv_b)
            self.__delattr__('conv1')
            self.__delattr__('conv2')
            self.__delattr__('strip_conv1')
            self.__delattr__('strip_conv2')
            return fin_conv
        else:
            ks = self.ks
            conv1 = self.conv1.fuse()
            strip_conv1 = self.strip_conv1.fuse()
            strip_conv2 = self.strip_conv2.fuse()
            
            conv1_w = conv1.weight
            conv1_b = conv1.bias
            strip_conv1_w = strip_conv1.weight
            strip_conv1_b = strip_conv1.bias
            strip_conv2_w = strip_conv2.weight
            strip_conv2_b = strip_conv2.bias
            
            strip_conv1_w = torch.nn.functional.pad(strip_conv1_w, [0, 0, ks//2, ks//2])
            strip_conv2_w = torch.nn.functional.pad(strip_conv2_w, [ks//2, ks//2, 0, 0])

            final_conv_w = conv1_w + strip_conv1_w + strip_conv2_w
            final_conv_b = conv1_b + strip_conv1_b + strip_conv2_b
            
            fin_conv = nn.Conv2d(self.in_dim, self.out_dim, self.ks, stride = self.stride,
                             padding = self.ks // 2, groups = self.in_dim, bias = True)

            fin_conv.weight.data.copy_(final_conv_w)
            fin_conv.bias.data.copy_(final_conv_b)
            self.__delattr__('conv1')
            self.__delattr__('strip_conv1')
            self.__delattr__('strip_conv2')
            return fin_conv


class ResidualBlock(torch.nn.Module):
    def __init__(self, main, dim, drop_path = 0., use_layer_scale = True, 
            layer_scale_init_value = 1e-5, post_act = None, pre_norm = None):
        super().__init__()
        self.main = main
        self.dim = dim
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad = True
            )
        self.pre_norm = pre_norm
        self.post_act = post_act

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_main(self, x):
        if self.use_layer_scale:
            if self.pre_norm is None:
                return self.main(x) * self.layer_scale
            else:
                return self.main(self.pre_norm(x)) * self.layer_scale
        else:
            if self.pre_norm is None:
                return self.main(x)
            else:
                return self.main(self.pre_norm(x))

    def forward(self, x):
        if self.main is None:
            res = x
        else:
            res = self.drop_path(self.forward_main(x)) + x
            if self.post_act:
                res = self.post_act(res)
        return res


class Token_Mixer(torch.nn.Module):
    def __init__(self, dim, ks1, ks2, use_gelu = True, use_sync_bn = False):
        super().__init__()
        
        self.fc = Conv2d_BN(dim, dim, use_sync_bn = use_sync_bn)
        self.conv1 = RepVGGDW(dim, dim, ks1, use_sync_bn = use_sync_bn)
        self.act = torch.nn.GELU() 
        self.conv2 = RepVGGDW(dim, dim, ks2, use_sync_bn = use_sync_bn)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(self.fc(x))))


class Channel_Mixer(torch.nn.Module):
    def __init__(self, dim, ks, expansion_ratio, use_gelu = True, use_sync_bn = False):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        self.conv = Conv2d_BN(dim, dim, ks, 1, ks // 2, groups = dim, use_sync_bn = use_sync_bn)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = torch.nn.GELU() 
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.conv(x))))


class FFBlock(torch.nn.Module):
    def __init__(self, dim, kernel_size, use_se, use_gelu, expansion_ratio, ks1, ks2,
                 drop_path = 0., use_layer_scale = True, layer_scale_init_value = 1e-5, use_sync_bn = False):
        super(FFBlock, self).__init__()

        self.token_mixer = torch.nn.Sequential(
            ResidualBlock(Token_Mixer(dim, ks1, ks2, use_gelu, use_sync_bn = use_sync_bn),
             dim, drop_path, use_layer_scale, layer_scale_init_value),
            SqueezeExcite(dim, 0.25) if use_se else nn.Identity(),
        ) 

        self.channel_mixer = ResidualBlock(Channel_Mixer(dim, kernel_size, expansion_ratio, use_gelu, use_sync_bn = use_sync_bn),
            dim, drop_path, use_layer_scale, layer_scale_init_value)

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, use_sync_bn = False):
        super().__init__()
        layers = []
        
        layers.append(Conv2d_BN(dim, out_dim * 2, 3, 2, 1, use_sync_bn = use_sync_bn))
        layers.append(torch.nn.GELU())
        layers.append(Conv2d_BN(out_dim * 2, out_dim, use_sync_bn = use_sync_bn))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class FastForwardNetwork(torch.nn.Module):
    def __init__(self,
                 in_chans = 3,
                 num_classes = 1000,
                 embed_dim=[64, 128, 320, 512],
                 in_dim = 32,
                 depth=[1, 2, 14, 2],
                 in_channels = 3,
                 kernel_sizes = [3, 3, 3, 3],
                 kernel_sizes1 = [3, 3, 9, 9],
                 kernel_sizes2 = [3, 3, 9, 9],
                 use_gelus = [True, True, True, True],
                 use_ses = [False, False, False, False],
                 expansion_ratios = [3, 3, 3, 3],
                 down_ops=[[''], ['subsample', 2], ['subsample', 2], ['subsample', 2]],
                 use_layer_scale = False,
                 layer_scale_init_value = 1e-5,
                 head_init_scale = 1.,
                 drop_path_rate = 0.2, 
                 pretrained = None,
                 use_sync_bn = False,
                 init_cfg = None,
                 **kwargs):
        super().__init__()

        num_features = embed_dim[-1]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        # Patch embedding
        self.patch_embed = nn.Sequential(
            Conv2d_BN(3, in_dim, 3, 2, 1, use_sync_bn = use_sync_bn),
            torch.nn.GELU(),
            Conv2d_BN(in_dim, in_dim, 3, 1, 1, use_sync_bn = use_sync_bn),
            torch.nn.GELU(),
            Conv2d_BN(in_dim, embed_dim[0], 3, 1, 1, use_sync_bn = use_sync_bn),
            Conv2d_BN(embed_dim[0], embed_dim[0], 3, 2, 1, use_sync_bn = use_sync_bn)
        )

        # Build FastForwardNetwork blocks
        self.stages = nn.ModuleList()
        cur = 0
        for i, (ed, dpth, ks, ks1, ks2, do, ug, use, er) in enumerate(
                zip(embed_dim, depth, kernel_sizes, kernel_sizes1, kernel_sizes2, down_ops, use_gelus, use_ses, expansion_ratios)):
            stage = []
            if do[0] == 'subsample':
                # Build FastForwardNetwork downsample block
                stage.append(PatchMerging(*embed_dim[i - 1:i + 1], use_sync_bn))
            for d in range(dpth):
                if d % 2 == 0:
                    stage.append(FFBlock(ed, ks, use, ug, er, ks1, ks2, dpr[cur + d], use_layer_scale, layer_scale_init_value, use_sync_bn))
                else:
                    stage.append(FFBlock(ed, ks, False, ug, er, ks1, ks2, dpr[cur + d], use_layer_scale, layer_scale_init_value, use_sync_bn))
            
            self.stages.append(nn.Sequential(*stage))
            cur += depth[i]
        
        self.for_pretrain = init_cfg is None
        if self.for_pretrain:
            self.init_cfg = None
            self.head1 = nn.Sequential(
                Conv2d_BN(embed_dim[-1], 1024, use_sync_bn = use_sync_bn),
                nn.Hardswish(),
                nn.AdaptiveAvgPool2d(output_size=1),
            )
            self.head2 = nn.Sequential(
                nn.Linear(1024, 1280),
                nn.LayerNorm(1280, eps = 1e-6),
                nn.Hardswish(),
                nn.Linear(1280, num_classes),
            )
            self.apply(self._init_weights)
            self.head2[-1].weight.data.mul_(head_init_scale)
            self.head2[-1].bias.data.mul_(head_init_scale)
        else:
            self.init_cfg = init_cfg
            self.init_weights()
    
    def init_weights(self):

        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            load_state_dict(self, _state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.for_pretrain:
            x = self.patch_embed(x)
            for idx, stage in enumerate(self.stages):
                x = stage(x)
            x = self.head1(x)
            x = torch.flatten(x, 1)
            x = self.head2(x)
            return x
        else:
            outs = []
            x = self.patch_embed(x)
            for idx, stage in enumerate(self.stages):
                x = stage(x)
                if idx > 0:
                    outs.append(x)
            return outs


@BACKBONES.register_module()
class ffnet_seg(FastForwardNetwork):
    def __init__(self, **kwargs):
        super().__init__(depth = [3, 4, 18, 5],
                         embed_dim = [96, 192, 384, 768],
                         in_dim = 64, 
                         kernel_sizes1 = [3, 3, 9, 9],
                         kernel_sizes2 = [3, 3, 9, 9],
                         expansion_ratios = [4, 4, 4, 4],
                        #  drop_path_rate = 0.35,
                         use_layer_scale = True,
                         use_sync_bn = True,
                         **kwargs,)

