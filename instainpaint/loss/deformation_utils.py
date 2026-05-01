import torch
import torch.nn.functional as F
from ..geometry.quaternion_utils import quaternion_multiply



def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def split_deformation(
    deformation, 
    deform_parametrization, 
    deform_format, 
    dynamic_model_pretraining,
    scale_t_bias,
    deform_opacity_bias=-4.6,
    channel_multiplier=None,
):

    assert deformation.ndim == 5, "(B, N, C, H, W), got {}".format(deformation.shape)

    if deform_parametrization == "implicit-decoder-only":
        deformation_dict = {}

    elif deform_parametrization == "dxyz":
        deformation = deformation * 0.1
        deformation_dict = {
            "translation": deformation,
        }

    elif deform_parametrization == "dxyzr":
        B, N, _, H, W = deformation.shape
        deformation_translation = deformation[:, :, 0*channel_multiplier:3*channel_multiplier, :, :]
        deformation_rotation    = deformation[:, :, 3*channel_multiplier:7*channel_multiplier, :, :]
        deformation_rotation    = F.normalize(deformation_rotation, dim=2, eps=1e-5)

        deformation_translation = deformation_translation * 0.1

        deformation_dict = {
            "rotation": deformation_rotation,
            "translation": deformation_translation,
        }

    elif deform_parametrization == "dxyzro":
        
        B, N, _, H, W = deformation.shape
        deformation_translation = deformation[:, :, 0*channel_multiplier:3*channel_multiplier, :, :]
        deformation_rotation    = deformation[:, :, 3*channel_multiplier:7*channel_multiplier, :, :]
        deformation_opacity     = deformation[:, :, 7*channel_multiplier:8*channel_multiplier, :, :]

        deformation_translation = deformation_translation * 0.1
        deformation_rotation[:, :, 0] = deformation_rotation[:, :, 0] * 10 # Make is more like identity
        deformation_rotation = F.normalize(deformation_rotation, dim=2, eps=1e-5)

        # This multiplies to the base opacity, therefore we want the init value ~1
        deformation_opacity = (1 - torch.sigmoid(deformation_opacity + deform_opacity_bias))

        deformation_dict = {
            "rotation": deformation_rotation,
            "translation": deformation_translation,
            "opacity": deformation_opacity,
        }

    elif deform_parametrization == "se3":

        B, N, _, H, W = deformation.shape

        deformation = deformation.reshape(B, N, -1, 7, H, W) # quaternion 4 + translation 3
        deformation_quaternion  = deformation[:, :, :, :4*channel_multiplier, :, :]
        deformation_translation = deformation[:, :, :, 4*channel_multiplier:, :, :]
        deformation_quaternion = F.normalize(deformation_quaternion, dim=3, eps=1e-5)
        
        deformation_translation = deformation_translation * 0.1
        deformation_quaternion = deformation_quaternion.permute(0, 1, 2, 4, 5, 3)
        deformation_rotation = quaternion_to_matrix(deformation_quaternion)
        deformation_rotation = deformation_rotation.reshape(*deformation_quaternion.shape[:-1], -1) 
        deformation_rotation = deformation_rotation.permute(0, 1, 2, 5, 3, 4)

        deformation_dict = {
            "translation": deformation_translation,
            "rotation_so3": deformation_rotation,
        }
        
    elif deform_parametrization == "rotor4d":

        assert deform_format == "singular"
        
        rotor_lower = rotation.permute(0, 1, 3, 4, 2)
        rotor_upper = deformation[:, :, :4, :, :].permute(0, 1, 3, 4, 2)
        if dynamic_model_pretraining:
            # Note: rotor_upper may not be zero after normalization, but this better simulates the 
            #       early stage behavior when we initializes dynamic model from static pretrain
            rotor_upper = rotor_upper * 0 
        rotor_lower, rotor_upper = rotornorm(rotor_lower, rotor_upper)
        rotor_lower = rotor_lower.permute(0, 1, 4, 2, 3)
        rotor_upper = rotor_upper.permute(0, 1, 4, 2, 3) # B, N, C, H, W
        rotation = torch.cat([rotor_lower, rotor_upper], 2)

        scale_t = deformation[:, :, 4:5, :, :]
        scale_t = (scale_t + scale_t_bias).exp().clamp(min=0.0, max=1.0)

        deformation_dict = {}

    else:
        raise NotImplementedError(deform_parametrization)

    return deformation_dict


def apply_image_warp(
    image_warp,
    *tensors,
):
    B, N, C, H, W = image_warp.shape
    ret = []
    image_warp = image_warp.reshape(B*N, -1, H, W).permute(0, 2, 3, 1)
    for t in tensors:
        t = t.reshape(B*N, -1, H, W)
        ori_dtype = t.dtype
        t = t.to(torch.float32)
        t = F.grid_sample(t, image_warp)
        t = t.to(ori_dtype)
        t = t.reshape(B, N, -1, H, W)
        ret.append(t)
    return ret


def apply_deformation(
    xyz_selected, 
    rgb_selected,
    opacity_selected, 
    scale_selected, 
    rotation_selected, 
    t_in_selected, 
    t_out_selected, 
    t_un_in_selected,
    t_un_out_idx,
    deformation_selected, 
    args):

    # if args.deform_detach_gs:
    #     xyz_selected = xyz_selected.detach()
    #     rgb_selected = rgb_selected.detach()
    #     opacity_selected = opacity_selected.detach()
    #     scale_selected = scale_selected.detach()
    #     rotation_selected = rotation_selected.detach()

    if getattr(args, "deform_flat_force_canonical", False):
        if args.deform_flat_canonical_def == "video":
            if (t_un_out_idx == 0).item():
                for k,v in deformation_selected.items():
                    xyz_selected = xyz_selected + (v.mean() * 0) # For DDP
                return xyz_selected, rgb_selected, opacity_selected, scale_selected, rotation_selected
        elif args.deform_flat_canonical_def == "token":
            canonical_mask = (t_un_in_selected == t_un_out_idx).float()
            for k,v in deformation_selected.items():
                if k == "translation":
                    deformation_selected[k] = v * (1-canonical_mask)
                elif k == "rotation":
                    # For quaternion, the identity is (1, 0, 0, 0)
                    deformation_selected[k] = torch.cat([
                        v[..., 0:1] * (1-canonical_mask) + canonical_mask, # to 1
                        v[..., 1:4] * (1-canonical_mask), # to 0
                    ], 1)
                else:
                    raise NotImplementedError(k)
        else:
            raise NotImplementedError()

    if args.deform_parametrization == "dxyz":
        xyz_selected = xyz_selected + deformation_selected["translation"]
    elif args.deform_parametrization == "dxyzr":
        xyz_selected = xyz_selected + deformation_selected["translation"]
        rotation_selected = quaternion_multiply(deformation_selected["rotation"], rotation_selected)
    elif args.deform_parametrization == "dxyzro":
        xyz_selected = xyz_selected + deformation_selected["translation"]
        rotation_selected = quaternion_multiply(deformation_selected["rotation"], rotation_selected)
        opacity_selected = opacity_selected * deformation_selected["opacity"]
    elif args.deform_parametrization == "se3":
        dims = deformation_selected.shape
        deformation_selected_R = deformation_selected["rotation_so3"].reshape(*dims[:-1], 3, 3)
        deformation_selected_T = deformation_selected["translation"].reshape(*dims[:-1], 3, 1)
        xyz_selected = (xyz_selected.unsqueeze(-2) @ deformation_selected_R + deformation_selected_T)[..., 0]
    elif args.deform_parametrization == "rotor4d":
        xyzt_selected = torch.cat([xyz_selected, t_in_selected], -1)
        xyz_selected, cov3d_selected, speed, w = slice_4d_to_3d(
            xyzt_selected, 
            scale_selected, 
            rotation_selected[..., :4], 
            rotation_selected[..., 4:], 
            t_current=t_out_selected)

    else:
        raise ValueError(args.deform_parametrization)
            
    return xyz_selected, rgb_selected, opacity_selected, scale_selected, rotation_selected


def deformation_selection(deformation_d, rays_t_un_input, rays_t_un_output, n, m, args, force_singular=False):
    
    deform_num_ch_ret = {}
    deformation_selected = {}
    deform_zero_reg_tgts = []
    for deformation_key, deformation_el in deformation_d.items():

        exception_keys = {
            'triplane', 'triplane_project_fn', 'canonical_warp'
        }

        if deformation_key == "translation":
            deform_num_ch = 3
        elif deformation_key == "rotation":
            deform_num_ch = 4
        elif deformation_key == "rotation_so3":
            deform_num_ch = 9
        elif deformation_key == "opacity":
            deform_num_ch = 1
        elif deformation_key in exception_keys:
            continue
        else:
            raise ValueError(deformation_key)
        deform_num_ch_ret[deformation_key] = deform_num_ch

        if (args.deform_format in {"flat", "triplane"}) and (not force_singular):
            out_idx = rays_t_un_output[n, m]
            ch_st = deform_num_ch * out_idx
            ch_ed = deform_num_ch * (out_idx+1)
        else:
            ch_st = deform_num_ch * m
            ch_ed = deform_num_ch * (m+1)

        deformation_selected[deformation_key] = deformation_el[:, :, ch_st:ch_ed]

        if (getattr(args, "deform_zero_regularization", 0.0) > 0.0) and (deformation_key == "translation"):
            in_inds = rays_t_un_input[n]
            out_idx = rays_t_un_output[n, m]
            deform_ts = deformation_selected[deformation_key].reshape(in_inds.shape[0], -1, deformation_selected[deformation_key].shape[-1])
            if args.deform_flat_canonical_def == "video":
                for in_idx, deform_t in zip(in_inds, deform_ts):
                    if (out_idx == 0).item():
                        deform_zero_reg_tgts.append(deform_t)
            elif args.deform_flat_canonical_def == "token":
                for in_idx, deform_t in zip(in_inds, deform_ts):
                    if (out_idx == in_idx).item():
                        deform_zero_reg_tgts.append(deform_t)
            else:
                raise NotImplementedError(args.deform_flat_canonical_def)
            # if out_idx % args.temporal_subsample_freq == 0:
            #     deform_zero_reg_tgts.append()

    return deformation_selected, deform_num_ch_ret, deform_zero_reg_tgts
