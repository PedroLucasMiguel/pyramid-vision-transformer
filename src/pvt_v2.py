import torch
import torch.nn as nn

from typing import Dict
from functools import partial
from torch import Tensor, linspace
from collections import OrderedDict

class OverlapPatchEmbed(nn.Module):
    '''
        The main advantage of this embedding process to preserve the edges information in the image :)
        source: 
        https://www.researchgate.net/publication/361414319_Transformer_Help_CNN_See_Better_A_Lightweight_Hybrid_Apple_Disease_Identification_Model_Based_on_Transformers
    '''
    def __init__(self, patch_size:int=7,
                 stride:int=4,
                 in_channels:int=3,
                 embed_dim:int=768) -> None:
        super().__init__()

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x:Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W
    
class DepthWiseConv(nn.Module):
    def __init__(self, dim:int=768) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
    
    def forward(self, x:Tensor, H:int, W:int) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features:int,
                 hidden_features:int=None,
                 out_features:int=None,
                 act_layer:nn.Module=nn.GELU,
                 drop:float=0.0,
                 linear:bool=False) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthWiseConv(hidden_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x:Tensor, H:int, W:int) -> Tensor:
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
    
class Attention(nn.Module):
    def __init__(self, dim:int,
                 num_heads:int=8,
                 qkv_bias:bool=False,
                 qk_scale:float=None,
                 attn_drop:float=0.0,
                 proj_drop:float=0.0,
                 sr_ratio:int=1,
                 linear:bool=False) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x:Tensor, H:int, W:int) -> Tensor:
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Spacial reduction
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # TODO: Maybe there's some space for otptimization here, but i'm not sure
        # Maybe we can use MultiHeadedAttention? Don't know...
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DropPath(nn.Module):
    '''
        Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
        Implementation from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    '''
    def __init__(self, drop_prob: float = 0.0, 
                 scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __drop_path(self, x:Tensor, drop_prob:float = 0.0, training:bool = False, scale_by_keep:bool = True):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x:Tensor) -> Tensor:
        return self.__drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Block(nn.Module):
    def __init__(self, dim:int,
                 num_heads:int,
                 mlp_ratio:float=4.0,
                 qkv_bias:bool=False,
                 qk_scale:float=None,
                 drop:float=0.0,
                 attn_drop:float=0.0,
                 drop_path:float=0.0,
                 act_layer:nn.Module=nn.GELU,
                 norm_layer:nn.Module=nn.LayerNorm,
                 sr_ratio:int=1,
                 linear:bool=False) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio,
                              linear=linear)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, 
                       hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, 
                       drop=drop, 
                       linear=linear)
    
    def forward(self, x:Tensor, H:int, W:int) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
    
class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.stages = nn.ModuleList()

        for i in range(num_stages):
            self.stages.append(nn.Sequential(
                OrderedDict(
                    [
                        (f"patch_embed", OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                            stride=4 if i == 0 else 2,
                                                            in_channels=in_channels if i == 0 else embed_dims[i - 1],
                                                            embed_dim=embed_dims[i])),

                        (f"block", nn.ModuleList([Block(dim=embed_dims[i], 
                                                                num_heads=num_heads[i], 
                                                                mlp_ratio=mlp_ratios[i], 
                                                                qkv_bias=qkv_bias, 
                                                                qk_scale=qk_scale,
                                                                drop=drop_rate, 
                                                                attn_drop=attn_drop_rate, 
                                                                drop_path=dpr[cur + j], 
                                                                norm_layer=norm_layer,
                                                                sr_ratio=sr_ratios[i], 
                                                                linear=linear) for j in range(depths[i])])),

                        (f"norm", norm_layer(embed_dims[i]))
                    ]
                )
            ))

            cur += depths[i]
        
        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

    def freeze_patch_emb(self) -> None:
        self.stages[0].patch_embed.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self) -> Dict:
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x:Tensor) -> None:
        B = x.shape[0]

        for i in range(self.num_stages):
            x, H, W = self.stages[i].patch_embed(x)
            for blk in self.stages[i].block:
                x = blk(x, H, W)
            x = self.stages[i].norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.head(x)

        return x
    
def pvt_v2_b5():

    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1], drop_path_rate=0.3)

    return model