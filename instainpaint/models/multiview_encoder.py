from functools import partial

import torch
from torch import nn
import torch.utils.checkpoint as checkpoint

from .utils import Block, PatchEmbedPlucker


class MultiviewTransformerPlucker(nn.Module):
    def __init__(
        self,
        patch_size=8,
        in_chans=4,
        embed_dim=1024,
        depth=0,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        cp_freq=1,
        emb_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        input_image_num=4,
        temporal_subsampling=False,
        temporal_subsample_freq=8,
        **unused,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.input_image_num = input_image_num
        self.in_chans = in_chans
        self.patch_embed = PatchEmbedPlucker(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            use_bias=emb_use_bias,
            input_image_num=input_image_num,
            temporal_subsampling=temporal_subsampling,
            temporal_subsample_freq=temporal_subsample_freq,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        drop_schedule = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_schedule[i],
                norm_layer=norm_layer,
                norm_use_bias=norm_use_bias,
                norm_use_affine=norm_use_affine,
            )
            for i in range(depth)
        )
        self.norm = norm_layer(embed_dim, elementwise_affine=norm_use_affine, bias=norm_use_bias)
        self.cp_freq = int(cp_freq)

    def forward(self, image, plucker_rays, x_bg=None):
        tokens = self.patch_embed(image, plucker_rays)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        for idx, block in enumerate(self.blocks):
            if self.cp_freq > 0 and idx % self.cp_freq == 0:
                tokens = checkpoint.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.norm(tokens)


class ExtraTokenEmbed(nn.Module):
    def __init__(
        self,
        embed_dim,
        input_image_num,
        use_time_embed=False,
        use_triplane=False,
        triplane_num_tokens=None,
    ):
        super().__init__()
        self.input_image_num = input_image_num
        self.use_time_embed = use_time_embed
        self.use_triplane = use_triplane
        if use_time_embed:
            self.time_tokens = nn.Parameter(torch.randn(1, input_image_num, embed_dim) * 0.02)
        if use_triplane:
            self.triplane_tokens = nn.Parameter(torch.randn(1, triplane_num_tokens, embed_dim) * 0.02)

    def forward(self, tokens):
        batch = tokens.shape[0]
        if self.use_time_embed:
            tokens = torch.cat((tokens, self.time_tokens.expand(batch, -1, -1)), dim=1)
        if self.use_triplane:
            tokens = torch.cat((tokens, self.triplane_tokens.expand(batch, -1, -1)), dim=1)
        return tokens


def mvencoder_base(
    type,
    patch_size=8,
    in_chans=4,
    drop_path_rate=0.0,
    cp_freq=1,
    depth=0,
    embed_dim=1024,
    qkv_use_bias=True,
    emb_use_bias=False,
    norm_use_bias=False,
    norm_use_affine=False,
    input_image_num=4,
    temporal_subsampling=False,
    temporal_subsample_freq=8,
    **unused,
):
    if type != "plucker":
        raise ValueError("Only the Plucker multiview encoder is included.")
    return MultiviewTransformerPlucker(
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=qkv_use_bias,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        cp_freq=cp_freq,
        drop_path_rate=drop_path_rate,
        emb_use_bias=emb_use_bias,
        norm_use_bias=norm_use_bias,
        norm_use_affine=norm_use_affine,
        input_image_num=input_image_num,
        temporal_subsampling=temporal_subsampling,
        temporal_subsample_freq=temporal_subsample_freq,
    )
