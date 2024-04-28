import os
from typing import Optional, Tuple, Union, Dict, Any
import torch
import logging
import math
import json
import re
import torch.nn.functional as F
from pathlib import Path
from math import pi
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as t_F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
from einops import rearrange, repeat
from functools import partial
from dataclasses import dataclass
from copy import deepcopy

# try:
#     from apex.normalization import FusedLayerNorm
# except:
#     FusedLayerNorm = LayerNorm
#     print("Please 'pip install apex'")

try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_

try:
    import xformers.ops as xops # ==0.0.20
except ImportError:
    xops = None
    print("\n Import xformers.ops  ERROR. Please 'pip install xformers")
    logging.info("\n Import xformers.ops  ERROR. Please 'pip install xformers")

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent / f"configs/eva/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, "r", encoding="utf8") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = dict(sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0])))


_rescan_model_configs()  # initial populate of model config registry


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
FusedLayerNorm = LayerNorm

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0. # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    drop_path_rate: Optional[float] = None  # drop path rate
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    eva_model_name: str = None # a valid eva model name overrides layers, width, patch_size
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16   # 224/14
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        patch_dropout = 0.
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print(f'Shape of rope freq: {self.freqs_cos.shape}')
        logging.info(f'Shape of rope freq: {self.freqs_cos.shape}')

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.size()[0]
            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            freqs_cos = repeat(self.freqs_cos, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])
            freqs_sin = repeat(self.freqs_sin, 'i j -> n i m j', n=t.shape[0], m=t.shape[1])

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, 'n i m j -> n m i j')
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, 'n i m j -> n m i j')

            return  t * freqs_cos + rotate_half(t) * freqs_sin

        return  t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,

        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else:

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale,
                )
            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x





class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        print(f"os.getenv('RoPE')={os.getenv('RoPE')}")
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        if self.training and os.getenv('RoPE') == '1':
            return x, patch_indices_keep

        return x


class EVAVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):
        
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        return x


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    logging.info("\n=====  Vision Config:    =====")
    logging.info(vision_cfg)
    logging.info("\n==============================")
    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool, #False
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer= partial(FusedLayerNorm, eps=1e-6) if vision_cfg.fusedLN else partial(norm_layer, eps=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len= vision_cfg.pt_hw_seq_len,   # 224/14
            intp_freq= vision_cfg.intp_freq,
            naiveswiglu= vision_cfg.naiveswiglu,
            subln= vision_cfg.subln
        )


    return visual


class CustomCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            # text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            itm_task: bool = False,
    ):
        super().__init__()
        print("Constructing CustomCLIP...")
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)



    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def encode_image(self, image, normalize: bool = False):

        features = self.visual(image)

        return F.normalize(features, dim=-1) if normalize else features


    def forward(self, image):
        image_features = self.encode_image(image, normalize=True)

        return image_features


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype

class ResizeMaxSize(nn.Module):

    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == 'min' else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[:2]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = t_F.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = t_F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(
        image_size: int,
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_longest_max: bool = False,
        fill_color: int = 0,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)
    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        if resize_longest_max:
            transforms = [
                ResizeMaxSize(image_size, fill=fill_color)
            ]
        else:
            transforms = [
                Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_size),
            ]
        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def create_model_and_transforms_for_eva(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_clip: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_model: str = None,
        pretrained_text_model: str = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        cache_dir: Optional[str] = None,
        skip_list: list = [],
):
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_clip=force_custom_clip,
        force_patch_dropout=force_patch_dropout,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        pretrained_hf=pretrained_hf,
        pretrained_visual_model=pretrained_visual_model,
        pretrained_text_model=pretrained_text_model,
        cache_dir=cache_dir,
        skip_list=skip_list,
    )

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = image_transform(
        model.visual.image_size,
        # model.model.image_size,
        is_train=True,
        mean=image_mean,
        std=image_std
    )
    preprocess_val = image_transform(
        model.visual.image_size,
        # model.model.image_size,
        is_train=False,
        mean=image_mean,
        std=image_std
    )

    return model, preprocess_train, preprocess_val


# loading openai CLIP weights when is_openai=True for training
def load_state_dict(checkpoint_path: str, map_location: str='cpu', model_key: str='model|module|state_dict', is_openai: bool=False, skip_list: list=[]):
    if is_openai:
        model = torch.jit.load(checkpoint_path, map_location="cpu").eval()
        state_dict = model.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        for mk in model_key.split('|'):
            if isinstance(checkpoint, dict) and mk in checkpoint:
                state_dict = checkpoint[mk]
                break
            else:
                state_dict = checkpoint
        if next(iter(state_dict.items()))[0].startswith('module'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    for k in skip_list:
        if k in list(state_dict.keys()):
            print(f"Removing key {k} from pretrained checkpoint")
            logging.info(f"Removing key {k} from pretrained checkpoint")
            del state_dict[k]

    if os.getenv('RoPE') == '1':
        for k in list(state_dict.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del state_dict[k]
    return state_dict


def resize_evaclip_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    all_keys = list(state_dict.keys())
    # interpolate position embedding
    if 'visual.pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['visual.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches

        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)

        # class_token and dist_token are kept unchanged
        if orig_size != new_size:

            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

            # Because torch.nn.functional.interpolate can not process fp16 data type
            orig_type = pos_tokens.dtype
            pos_tokens = pos_tokens.type(torch.float32)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.type(orig_type)

            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)

            state_dict['visual.pos_embed'] = new_pos_embed

            # TODO: When patch size is not changed, there is no need to interpolate visual.patch_embed.proj.weight
            # patch_embed_proj = state_dict['visual.patch_embed.proj.weight']
            # patch_size = model.visual.patch_embed.patch_size
            # orig_type = patch_embed_proj.dtype
            # patch_embed_proj = patch_embed_proj.type(torch.float32)
            # patch_embed_proj = torch.nn.functional.interpolate(
            #     patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
            # patch_embed_proj = patch_embed_proj.type(orig_type)
            # state_dict['visual.patch_embed.proj.weight'] = patch_embed_proj

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, state_dict):
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)

            # Because torch.nn.functional.interpolate can not process fp16 data type
            orig_type = pos_tokens.dtype
            pos_tokens = pos_tokens.type(torch.float32)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.type(orig_type)

            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed


def state_dict_compatible(state_dict, prefix="visual"):
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix) and "logit_scale" not in key:
            rename_key = prefix + "." + key
        else:
            rename_key = key
        new_state_dict[rename_key] = value
    del state_dict
    return new_state_dict

def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict", strict=True):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key, is_openai=False)
    if 'visual.pos_embed' in state_dict:
        resize_evaclip_pos_embed(state_dict, model)
    elif 'pos_embed' in state_dict:
        interpolate_pos_embed(model, state_dict)


    # TODO
    # To be compatible with this code (pretrain vision encoder like vit/eva does not have 'model' or 'visual' before its params key)
    state_dict = state_dict_compatible(state_dict, prefix="visual")

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)

    logging.info(f"incompatible_keys.missing_keys: {incompatible_keys.missing_keys}")
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_clip: bool = False,
        force_patch_dropout: Optional[float] = None,
        pretrained_image: str = '',
        pretrained_text: str = '',
        pretrained_hf: bool = True,
        pretrained_visual_model: str = None,
        pretrained_text_model: str = None,
        cache_dir: Optional[str] = None,
        skip_list: list = [],
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = get_model_config(model_name)

    logging.info("\n=====  Vision Model Config    =====")
    logging.info(model_cfg)
    logging.info("\n===================================")

    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if 'rope' in model_cfg.get('vision_cfg', {}):
        logging.info("Using RoPE...")
        if model_cfg['vision_cfg']['rope']:
            os.environ['RoPE'] = "1"
    else:
        logging.info("Not using RoPE...")
        os.environ['RoPE'] = "0"

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True
    
    if force_patch_dropout is not None:
        # override the default patch dropout value
        logging.info("Force patch dropout...")
        model_cfg['vision_cfg']["patch_dropout"] = force_patch_dropout

    cast_dtype = get_cast_dtype(precision)
    custom_clip = model_cfg.pop('custom_text', False) or force_custom_clip or ('hf_model_name' in model_cfg['text_cfg'])


    if custom_clip:
        # if 'hf_model_name' in model_cfg.get('text_cfg', {}):
        #     model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf
        del model_cfg['text_cfg']
        model = CustomCLIP(**model_cfg, cast_dtype=cast_dtype)
    # else:
    #     model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    pretrained_cfg = {}
    if pretrained:
        checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights from [{pretrained}]')
            load_checkpoint(model,
                            checkpoint_path,
                            model_key="model|module|state_dict",
                            strict=False
                            ) 

    model.to(device=device)

    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = pretrained_cfg.get('mean', None) or OPENAI_DATASET_MEAN
    model.visual.image_std = pretrained_cfg.get('std', None) or OPENAI_DATASET_STD

    logging.info("Using pretrained_cfg.mean :{}".format(pretrained_cfg.get('mean', None)) if pretrained_cfg.get('mean',None) else "Using OPENAI_DATASET_MEAN: {}".format(OPENAI_DATASET_MEAN))
    return model

