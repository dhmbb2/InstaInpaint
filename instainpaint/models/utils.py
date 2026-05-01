import torch
from flash_attn import flash_attn_qkvpacked_func
from torch import nn

from ..geometry.utils import zero_module


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * mask.floor_()


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
        use_weight_norm=False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        if use_weight_norm:
            self.fc1 = nn.utils.weight_norm(self.fc1)
            self.fc2 = nn.utils.weight_norm(self.fc2)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        bias=True,
        use_weight_norm=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        if use_weight_norm:
            self.qkv = nn.utils.weight_norm(self.qkv)
            self.proj = nn.utils.weight_norm(self.proj)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop)
        return self.proj_drop(self.proj(x.reshape(batch, tokens, channels)))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_use_bias=True,
        norm_use_bias=True,
        norm_use_affine=True,
        use_weight_norm=False,
        **unused,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=norm_use_affine, bias=norm_use_bias)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            bias=attn_use_bias,
            use_weight_norm=use_weight_norm,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=norm_use_affine, bias=norm_use_bias)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=attn_use_bias,
            use_weight_norm=use_weight_norm,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbedPlucker(nn.Module):
    def __init__(
        self,
        patch_size=8,
        in_chans=4,
        embed_dim=1024,
        use_bias=False,
        input_image_num=None,
        temporal_subsampling=False,
        temporal_subsample_freq=8,
        **unused,
    ):
        super().__init__()
        if temporal_subsampling:
            raise ValueError("Temporal subsampling is not part of the open-source runtime path.")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=use_bias)
        self.proj_plucker = zero_module(
            nn.Conv2d(6, embed_dim, kernel_size=patch_size, stride=patch_size, bias=use_bias)
        )

    def forward(self, image, plucker_rays):
        tokens = self.proj(image) + self.proj_plucker(plucker_rays)
        return tokens.flatten(2).transpose(1, 2)
