import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .utils import Block, trunc_normal_


class AeGaussianTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        cp_freq=1,
        attn_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        drop_schedule = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_schedule[i],
                norm_layer=norm_layer,
                attn_use_bias=attn_use_bias,
                norm_use_bias=norm_use_bias,
                norm_use_affine=norm_use_affine,
                use_weight_norm=use_weight_norm,
            )
            for i in range(depth)
        )
        self.cp_freq = int(cp_freq)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, tokens, cross_attn_x=None):
        for idx, block in enumerate(self.blocks):
            if self.cp_freq > 0 and idx % self.cp_freq == 0:
                tokens = checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens
