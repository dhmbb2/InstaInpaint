import numpy as np
import torch
import torch.nn as nn


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def build_pytorch_mlp(input_dim, hidden_dim, output_dim, act_layer=nn.GELU, depth=10, bias=False, use_weight_norm=False):
    if depth == 0:
        mlp = [
            nn.Linear(input_dim, output_dim, bias=bias)
        ]
    else:
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        mlp.append(act_layer())
        for _ in range(depth - 1):
            mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            mlp.append(act_layer())
        mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    if use_weight_norm:
        mlp = [
            torch.nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer 
                for layer in mlp]
    mlp = nn.Sequential(*mlp)
    return mlp


class CondMlp(nn.Module):

    def __init__(self, pre_cond_mlp, post_cond_mlp, cond_embed):
        super().__init__()
        self.pre_cond_mlp = pre_cond_mlp
        self.post_cond_mlp = post_cond_mlp
        self.cond_embed = cond_embed

    def forward(self, x, query):
        # x: (B*P, num_in, C)
        # query: (B*P, num_out, 1)
        #     B = batch size
        #     P = num patches
        #     C = channel size

        batch_size = x.shape[0]
        num_in     = x.shape[1]
        num_out    = query.shape[1]
        
        x = self.pre_cond_mlp(x)
        c = self.cond_embed(query)

        x = x.unsqueeze(2).repeat(1, 1, num_out, 1)
        c = c.unsqueeze(1).repeat(1, num_in, 1, 1)

        x = torch.cat([x, c], -1)
        x = self.post_cond_mlp(x) # [B, num_in, num_out, C]

        return x
    

class TimeEmbedder:
    def __init__(self, num_freq, log_sampling=True, include_input=True):

        self.num_freq = num_freq
        periodic_fns = [torch.sin, torch.cos]
        max_freq = num_freq # self.kwargs['max_freq_log2'] <= NeRF
        N_freqs = num_freq # self.kwargs['num_freqs']

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        freq_bands = freq_bands * torch.pi * 2
            
        embed_fns = []
        if include_input:
            embed_fns.append(lambda x : x)
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                    
        self.embed_fns = embed_fns
        
    def forward(self, x):
        # What we want here is map [0, 1] to N frequency bands, while F(x=0) = F(x=1)
        # For band N of sin: sin(x * 2*pi * N), we pre-compute 2*pi*N during init.
        return torch.cat([fn(x) for fn in self.embed_fns], -1)
    
    def __call__(self, x):
        return self.forward(x)


def build_pytorch_cond_mlp(
        input_dim, 
        output_dim, 
        pre_cond_hidden_dim, 
        pre_cond_depth,
        post_cond_hidden_dim, 
        post_cond_depth,
        time_embed_num_freqs,
        act_layer=nn.GELU, 
        bias=False, 
        use_weight_norm=False):
    
    time_embed_dim = time_embed_num_freqs * 2 + 1
    time_embed = TimeEmbedder(time_embed_num_freqs)
    time_embed_dim = len(time_embed.embed_fns)

    pre_cond_mlp = build_pytorch_mlp(
        input_dim=input_dim, 
        hidden_dim=pre_cond_hidden_dim, 
        output_dim=post_cond_hidden_dim, 
        act_layer=act_layer, depth=pre_cond_depth, bias=bias, 
        use_weight_norm=use_weight_norm)
    
    post_cond_mlp = build_pytorch_mlp(
        input_dim=post_cond_hidden_dim+time_embed_dim, 
        hidden_dim=post_cond_hidden_dim, 
        output_dim=output_dim, 
        act_layer=act_layer, depth=post_cond_depth, bias=bias, 
        use_weight_norm=use_weight_norm)
    
    return CondMlp(pre_cond_mlp, post_cond_mlp, time_embed)


def grid_sample_3d(voxel, index):
    """
    Modified from 2d grid sample of tensoIR
    Aligned corner, repetitive padding
    image: Float[Tensor, B C VZ VY VX]
    index: Float[Tensor, B Z Y X 3]
    """
    N, C, VZ, VY, VX = voxel.shape
    _, Z, Y, X, _ = index.shape

    ix = index[..., 0]
    iy = index[..., 1]
    iz = index[..., 2]

    ix = (ix + 1) / 2 * (VX - 1)
    iy = (iy + 1) / 2 * (VY - 1)
    iz = (iz + 1) / 2 * (VZ - 1)
    with torch.no_grad():
        ix_d = torch.clamp(torch.floor(ix), 0, VX - 1).long()
        iy_d = torch.clamp(torch.floor(iy), 0, VY - 1).long()
        iz_d = torch.clamp(torch.floor(iz), 0, VZ - 1).long()

        ix_u = torch.clamp(ix_d + 1, 0, VX - 1).long()
        iy_u = torch.clamp(iy_d + 1, 0, VY - 1).long()
        iz_u = torch.clamp(iz_d + 1, 0, VZ - 1).long()

        index_ddd = (iz_d * VY * VX + iy_d * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_ddu = (iz_u * VY * VX + iy_d * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_dud = (iz_d * VY * VX + iy_u * VX + ix_d).long().view(N, 1, Z * Y * X)
        index_duu = (iz_u * VY * VX + iy_u * VX + ix_d).long().view(N, 1, Z * Y * X)

        index_udd = (iz_d * VY * VX + iy_d * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_udu = (iz_u * VY * VX + iy_d * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_uud = (iz_d * VY * VX + iy_u * VX + ix_u).long().view(N, 1, Z * Y * X)
        index_uuu = (iz_u * VY * VX + iy_u * VX + ix_u).long().view(N, 1, Z * Y * X)

    w_ddd = (ix - ix_d) * (iy - iy_d) * (iz - iz_d)
    w_ddu = (ix - ix_d) * (iy - iy_d) * (iz_u - iz)
    w_dud = (ix - ix_d) * (iy_u - iy) * (iz - iz_d)
    w_duu = (ix - ix_d) * (iy_u - iy) * (iz_u - iz)

    w_udd = (ix_u - ix) * (iy - iy_d) * (iz - iz_d)
    w_udu = (ix_u - ix) * (iy - iy_d) * (iz_u - iz)
    w_uud = (ix_u - ix) * (iy_u - iy) * (iz - iz_d)
    w_uuu = (ix_u - ix) * (iy_u - iy) * (iz_u - iz)

    voxel = voxel.reshape(N, C, VX * VY * VZ)

    v_ddd = torch.gather(voxel, 2, index_ddd.repeat(1, C, 1))
    v_ddu = torch.gather(voxel, 2, index_ddu.repeat(1, C, 1))
    v_dud = torch.gather(voxel, 2, index_dud.repeat(1, C, 1))
    v_duu = torch.gather(voxel, 2, index_duu.repeat(1, C, 1))

    v_udd = torch.gather(voxel, 2, index_udd.repeat(1, C, 1))
    v_udu = torch.gather(voxel, 2, index_udu.repeat(1, C, 1))
    v_uud = torch.gather(voxel, 2, index_uud.repeat(1, C, 1))
    v_uuu = torch.gather(voxel, 2, index_uuu.repeat(1, C, 1))

    out_val = (
        w_ddd.view(N, 1, Z, Y, X) * v_ddd.view(N, C, Z, Y, X)
        + w_ddu.view(N, 1, Z, Y, X) * v_ddu.view(N, C, Z, Y, X)
        + w_dud.view(N, 1, Z, Y, X) * v_dud.view(N, C, Z, Y, X)
        + w_duu.view(N, 1, Z, Y, X) * v_duu.view(N, C, Z, Y, X)
        + w_udd.view(N, 1, Z, Y, X) * v_udd.view(N, C, Z, Y, X)
        + w_udu.view(N, 1, Z, Y, X) * v_udu.view(N, C, Z, Y, X)
        + w_uud.view(N, 1, Z, Y, X) * v_uud.view(N, C, Z, Y, X)
        + w_uuu.view(N, 1, Z, Y, X) * v_uuu.view(N, C, Z, Y, X)
    )

    return out_val


def grid_sample_2d(image, index):
    """
    Mostly copy from tensoIR
    Aligned corner, repetitive padding
    image: Float[Tensor, B C IH IW]
    index: Float[Tensor, B H W 2]
    """

    N, C, IH, IW = image.shape
    _, H, W, _ = index.shape

    ix = index[..., 0]
    iy = index[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    with torch.no_grad():
        ix_nw = torch.floor(ix)
        iy_nw = torch.floor(iy)
        ix_ne = ix_nw + 1
        iy_ne = iy_nw
        ix_sw = ix_nw
        iy_sw = iy_nw + 1
        ix_se = ix_nw + 1
        iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.contiguous().view(N, C, IH * IW)

    nw_val = torch.gather(
        image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)
    )

    out_val = (
        nw_val.view(N, C, H, W) * nw.view(N, 1, H, W)
        + ne_val.view(N, C, H, W) * ne.view(N, 1, H, W)
        + sw_val.view(N, C, H, W) * sw.view(N, 1, H, W)
        + se_val.view(N, C, H, W) * se.view(N, 1, H, W)
    )

    return out_val
