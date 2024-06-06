import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
# from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


def resize(
    x: torch.Tensor,
    size= None,
    scale_factor= None,
    mode= "bicubic",
    align_corners= False,
):
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size= None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor):
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


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


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main,
        shortcut,
        drop_path = 0.,
        post_act=None,
        pre_norm=None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_main(self, x: torch.Tensor):
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor):
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.drop_path(self.forward_main(x)) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


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


class Token_Mixer(torch.nn.Module):
    def __init__(self, dim, ks1, ks2, use_sync_bn = False):
        super().__init__()
        
        self.fc = Conv2d_BN(dim, dim, use_sync_bn = use_sync_bn)
        self.conv1 = RepVGGDW(dim, dim, ks1, use_sync_bn = use_sync_bn)
        self.act = torch.nn.GELU() 
        self.conv2 = RepVGGDW(dim, dim, ks2, use_sync_bn = use_sync_bn)
    
    def forward(self, x):
        return self.conv2(self.act(self.conv1(self.fc(x))))


class Channel_Mixer(torch.nn.Module):
    def __init__(self, dim, ks, expansion_ratio, use_sync_bn = False):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        self.conv = RepVGGDW(dim, dim, ks, use_sync_bn = use_sync_bn)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = torch.nn.GELU() 
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.conv(x))))


class FFNetBlock(torch.nn.Module):
    def __init__(self, dim, kernel_size, expansion_ratio, ks1, ks2, drop_path = 0., use_sync_bn = False):
        super(FFNetBlock, self).__init__()

        self.token_mixer = ResidualBlock(Token_Mixer(dim, ks1, ks2, use_sync_bn), nn.Identity(), drop_path)
        self.channel_mixer = ResidualBlock(Channel_Mixer(dim, kernel_size, expansion_ratio, use_sync_bn), nn.Identity(), drop_path)

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))



@HEADS.register_module()
class FFNetHead(BaseDecodeHead):
    def __init__(self, 
                 spatial_ks,
                 channel_ks,
                 head_stride,
                 head_width,
                 head_depth,
                 expansion_ratio,
                 final_expansion_ratio = None,
                 stride_list = [8, 16, 32],
                 dropout = 0,
                 drop_path = 0,
                 use_sync_bn = True,
                 **kwargs):
        super(FFNetHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.upsampler = nn.Sequential(
            nn.Identity(),
            UpSampleLayer(factor = 2, align_corners = self.align_corners),
            UpSampleLayer(factor = 4, align_corners = self.align_corners),
        )

        self.squeeze = nn.Sequential(
            Conv2d_BN(sum(self.in_channels), sum(self.in_channels), use_sync_bn = use_sync_bn),
            nn.GELU(),
            Conv2d_BN(sum(self.in_channels), head_width, use_sync_bn = use_sync_bn)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, head_depth)]
        blocks = []
        for idx in range(head_depth):
            block = FFNetBlock(head_width, channel_ks, expansion_ratio, spatial_ks, spatial_ks, dpr[idx], use_sync_bn)
            blocks.append(block)
        self.head_blocks = nn.Sequential(*blocks)

        self.align = nn.Identity() if final_expansion_ratio is None else \
                nn.Sequential(
                    Conv2d_BN(head_width, head_width * final_expansion_ratio, use_sync_bn = use_sync_bn),
                    nn.GELU(),
                ) 

    def forward(self, inputs):
        """"Forward function."""
        inputs = [self.upsampler[i](level) for i, level in enumerate(inputs)]
        inputs = torch.cat(inputs, dim = 1)

        x = self.squeeze(inputs)
        x = self.head_blocks(x)
        x = self.align(x)
        x = self.cls_seg(x)
        return x

