import torch
from torch import nn as nn
from torch.nn import functional as F
from typing import Dict, Optional, Tuple, Union
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from einops import rearrange, repeat

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 1, stride = 1,
                 padding = None, dilation = 1, groups = 1, bias = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, autopad(kernel_size, padding), dilation, groups, bias)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x


class SelfAttentionBlock(nn.Module):
    """mmcv-base General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, BasicBlock):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                BasicBlock(
                    in_channels,
                    channels,
                    1)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    BasicBlock(
                        channels,
                        channels,
                        1))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context

class CSSModule(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super(CSSModule, self).__init__()
        self.channels = channels

        self.pred_proj = nn.Conv1d(num_classes, num_classes, 1, bias=False, groups=num_classes)

        self.attn = SelfAttentionBlock(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True)


    def forward(self, feat, pred):
        b, cls, h, w = pred.shape
        _, c, _, _ = feat.shape
        _pred = pred.view(b, cls, -1)
        _pred = F.softmax(self.pred_proj(_pred), dim=2) # b, cls, l
        _feat = feat.view(b, c, -1)
        _feat = _feat.permute(0, 2, 1) # b, l, c
        feat_cls_relation = torch.matmul(_pred, _feat).permute(0, 2, 1).contiguous().unsqueeze(3) # b, c, cls, 1
        out = self.attn(feat, feat_cls_relation)
        return out

# compensatory attention
class CA(nn.Module):
    def __init__(self, in_channels, channels, heads=8):
        super(CA, self).__init__()

        self.heads = heads
        self.in_channels = in_channels
        self.channels = channels

        self.MHSA1 = MHA(in_channels, channels)

    def forward(self, x, aux):
        b, _, h, w = x.shape
        out = self.MHSA1(x, aux)
        return out

# simple multi-head attention
class MHA(nn.Module):
    def __init__(self, in_channels, channels, num_heads=8):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.channels = channels
        self.project_out_1 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.project_out_2 = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.project_out_3 = nn.Conv2d(channels, in_channels, kernel_size=1)


    def forward(self, x, aux):
        b, c, h, w = x.shape
        out1 = self.project_out_1(x)
        out2 = self.project_out_2(aux)

        q1 = rearrange(out1, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        k1 = rearrange(out2, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)
        v1 = rearrange(out2, 'b (head c) h w -> b head c (h w)', head=self.num_heads, h=h, w=w)

        q1 = torch.nn.functional.normalize(q1, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        out3 = rearrange(out3, 'b head c (h w) -> b (head c) h w ', head=self.num_heads, h=h, w=w)

        out = self.project_out_3(out3) + x
        return out


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

# Global Extraction
class GE(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.horizontal_proj = nn.Conv2d(dim, dim, kernel_size=(window_size, 1), padding=(window_size // 2 - 1, 0))
        self.vertical_proj = nn.Conv2d(dim, dim, kernel_size=(1, window_size), padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (0, self.ws - W % self.ws, 0, self.ws - H % self.ws), mode='reflect')
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        proj1 = self.horizontal_proj(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect'))
        proj2 = self.vertical_proj(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = self.proj(F.pad(proj1+proj2, pad=(0, 1, 0, 1), mode='reflect'))
        out = out[:, :, :H, :W]

        return out

class GlobalBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16, window_size=8):
        super().__init__()
        self.attn = GE(dim, num_heads=num_heads, window_size=window_size)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(x)
        x = self.norm(x)

        return x

# Spatial Pyramid Pooling
class SPP(nn.Module):
    def __init__(self, ch_in, ch_out, size_mps=(5, 9, 13)):
        super(SPP, self).__init__()
        dim = ch_out

        self.conv1 = nn.Sequential(
            BasicBlock(ch_in, dim, 1, 1),
            BasicBlock(dim, dim, 3, 1),
            BasicBlock(dim, dim, 1, 1))
        self.mps = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in size_mps])
        self.conv2 = nn.Sequential(
            BasicBlock(4 * dim, dim, 1, 1),
            BasicBlock(dim, dim, 3, 1)
        )
        self.conv3 = BasicBlock(ch_in, dim, 1, 1)
        self.output = BasicBlock(2 * dim, ch_out, 1, 1)

    def forward(self, x):
        x_ = self.conv1(x)
        out = self.conv2(torch.cat([x_] + [mp(x_) for mp in self.mps], 1))
        res = self.conv3(x)
        return self.output(torch.cat((out, res), dim=1))

class LocalBlock(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.spp = SPP(dim, dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.spp(x)
        x = self.norm(x)

        return x