import torch
import numpy as np
import matplotlib.pyplot as plt
from ..loss.deformation_utils import deformation_selection, apply_deformation
from scipy.spatial.transform import Rotation, Slerp
from .camera_paths import get_circle_extrinsics, look_at


def spatial_subsample(t, sample_freq, n_samples):
    return t[:, :, :, ::sample_freq, ::sample_freq].permute(0, 1, 3, 4, 2).reshape(n_samples, -1)


def subsample_mask(mask, num):
    # mask: [N, -1]
    device = mask.device
    candidate_inds = torch.nonzero(mask)
    rnd_inds = torch.randperm(candidate_inds.shape[0], device=device)
    return candidate_inds[rnd_inds[:num]]


def tracking_point_colorization(
    xyz, 
    depth,
    rgb, 
    opacity, 
    scale, 
    rotation, 
    deformation, 
    cams_output,
    rays_t_un_output, 
    num_vids,
    args, 
    colorize_by="centroid",
    num_centroids=10):

    # xyz: [1, M, 3, H, W]
    # rgb: [1, M, 3, H, W]

    # Warnings:
    # 0. The xyz needs to be the extra xyz if `args.deform_extra_gs` is enabled
    # 1. This is designed for scene, which does not filter points by mask
    # 2. This only considers the cases with certain deformation formats

    assert colorize_by in {"centroid", "input_view"}

    n = 0
    device = xyz.device
    _, M, _, H, W = xyz.shape

    sample_freq = 4
    nh = H // sample_freq
    nw = W // sample_freq
    n_samples = M * nh * nw

    density_query_radius = 0.05
    density_quantile_candidates = np.arange(0.7, 0.1, -0.05)
    deform_quantile_candidates = np.arange(0.95, 0.1, -0.05)


    # Warp GS to first frame (the xyz in the canonical space is different)
    m = 0
    
    if args.deform_parametrization == "rotor4d":
        xyz_d      = xyz[n].permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_d      = rgb[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
        opacity_d  = opacity[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
        scale_d    = scale[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
        rotation_d = rotation[n, :].permute(0, 2, 3, 1).reshape(-1, 8)
    else:
        xyz_d      = xyz[n].permute(0, 2, 3, 1).reshape(-1, 3)
        rgb_d      = rgb[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
        opacity_d  = opacity[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
        scale_d    = scale[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
        rotation_d = rotation[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
        
    # if t_in is not None:
    #     t_in_d  = t_in.permute(0, 2, 3, 1).reshape(-1, 1)
    #     t_out_d = t_out.permute(0, 2, 3, 1).reshape(-1, 1)
    # else:
    #     t_in_d, t_out_d = None, None

    cur_deformation_d, deform_num_ch, _ = deformation_selection(deformation, rays_t_un_output, n, m, args)
    cur_deformation_d = {
        k: v[n].permute(0, 2, 3, 1).reshape(-1, deform_num_ch[k])
            for k,v in cur_deformation_d.items()}
            
    t_in_d, t_out_d = None, None
    deformed_results = apply_deformation(xyz_d, rgb_d, opacity_d, scale_d, rotation_d, t_in_d, t_out_d, cur_deformation_d, args)
    cur_xyz, cur_rgb, cur_opacity, cur_scale, cur_rotation = deformed_results

    cur_xyz     = cur_xyz.reshape(1, M, H, W, 3).permute(0, 1, 4, 2, 3)
    cur_rgb     = cur_rgb.reshape(1, M, H, W, 3).permute(0, 1, 4, 2, 3)
    cur_opacity = cur_opacity.reshape(1, M, H, W, 1).permute(0, 1, 4, 2, 3)
    
    xyz_subsampled     = spatial_subsample(cur_xyz, sample_freq, n_samples)
    rgb_subsampled     = spatial_subsample(cur_rgb, sample_freq, n_samples)
    opacity_subsampled = spatial_subsample(cur_opacity, sample_freq, n_samples)
    # rgb_subsampled[17*32+19] = -1

    # xyz_dists = (
    #     (xyz_subsampled[None, ...] - xyz_subsampled[:, None, ...]) ** 2
    # ).mean(-1).sqrt() # shape: [M, M]
    # density = (xyz_dists < density_query_radius).sum(-1).float() # Just ranking, don't care the ratio

    deformation_diff_list = []
    seq_len = cams_output.shape[1] // num_vids
    for m in range(seq_len-1):
        deformation_selected_st, _, _ = \
            deformation_selection(deformation, rays_t_un_output, n, m, args)
        deformation_selected_ed, _, _ = \
            deformation_selection(deformation, rays_t_un_output, n, m+1, args)
        if args.deform_parametrization == "dxyz":
            deformation_diff = (deformation_selected_ed["translation"] - deformation_selected_st["translation"])
            deformation_diff_sub = spatial_subsample(deformation_diff, sample_freq, n_samples)
        else:
            raise NotImplementedError(deformation_diff_sub)
        deformation_dist_sub = (deformation_diff_sub ** 2).mean(-1).sqrt()
        deformation_diff_list.append(deformation_dist_sub)
    deformation_diff_list = torch.stack(deformation_diff_list, 0) # [M-1, num_points]
    deformation_diff_list_raw = deformation_diff_list.clone()

    # centroid_selected_inds = deformation_dist_sub.argsort()[-num_centroids:]

    # We don't care low opacity deformation
    opacity_threshold = 0.3
    opacity_subsampled = opacity_subsampled[..., 0]
    opacity_mask = opacity_subsampled < opacity_threshold
    deformation_diff_list[:, opacity_mask] = 0

    # deformation_diff_list = deformation_diff_list.sort()
    # deformation_diff_list = deformation_diff_list[..., :-3] # Remove largest deformation, for cases of sudden jump
    deformation_diff_list_median = deformation_diff_list.median(0).values # [num_points, ]
    centroid_selected_inds = deformation_diff_list_median.argsort()[-num_centroids:]


    # centroid_selected_inds[0] = 16 * 32 + 13

    

    # for density_quantile_candidate, deform_quantile_candidate in \
    #         zip(density_quantile_candidates, deform_quantile_candidates):

    #     density_quantile = torch.quantile(density, density_quantile_candidate)
    #     is_valid_density = density > density_quantile
    #     deformation_dist_quantile = torch.quantile(deformation_dist_sub, deform_quantile_candidate)
    #     is_valid_deform = deformation_dist_sub > deformation_dist_quantile

    #     is_valid = is_valid_deform # is_valid_density & is_valid_deform
    #     if is_valid.sum().item() > num_centroids * 10: # We need more diversed sources
    #         break
    # centroid_selected_inds = subsample_mask(is_valid, num_centroids)

    c_xyz_arr = xyz_subsampled[centroid_selected_inds]

    cmap = plt.get_cmap("gist_rainbow")
    xyz_tmp   = cur_xyz.permute(0, 1, 3, 4, 2)
    c_rgb     = torch.zeros(1, M, H, W, 3, device=device, dtype=torch.float32)
    c_counter = torch.zeros(1, M, H, W, 1, device=device, dtype=torch.float32)
    group_elements_num = 10
    group_radius_candidates = \
        np.arange(0.02, 0, -0.01).tolist() + \
        np.arange(0.01, 0, -0.001).tolist()

    for i in range(num_centroids):
        try:
            if num_centroids == 1:
                c_color = cmap(0)[:3]
            else:
                c_color = cmap(i / (num_centroids-1))[:3]
            c_color_t = torch.tensor(c_color, device=device, dtype=torch.float32)
            centroid = c_xyz_arr[i].reshape(1, 1, 1, 1, 3)
            for group_radius_candidate in group_radius_candidates:
                is_in_group = ((xyz_tmp - centroid) ** 2).mean(-1).sqrt() < group_radius_candidate
                group_size = is_in_group.sum().item()
                break
                # if group_size > group_elements_num and group_size < group_elements_num * 10:
                #     break
            
            # is_in_group_flat = is_in_group.reshape(-1, 1)
            # group_sub_inds = subsample_mask(is_in_group_flat, group_elements_num)
            # is_in_group_sub = torch.zeros_like(is_in_group_flat)
            # is_in_group_sub[group_sub_inds] = True
            # is_in_group_sub = is_in_group_sub.reshape(1, M, H, W)
            is_in_group_sub = is_in_group.reshape(1, M, H, W)

            c_rgb[is_in_group_sub] = c_rgb[is_in_group_sub].reshape(-1, 3) + c_color_t
            c_counter[is_in_group_sub] = c_counter[is_in_group_sub] + 1

        except Exception as e:
            raise RuntimeError("Failed to colorize tracking points") from e
    
    valid_mask = (c_counter > 0)
    c_rgb = c_rgb / c_counter
    c_rgb[~valid_mask.repeat(1, 1, 1, 1, 3)] = 0

    c_opacity = opacity.clone()
    c_opacity[~valid_mask.permute(0, 1, 4, 2, 3)] = 0
    # c_opacity = valid_mask.type(xyz.dtype)
    # c_opacity = c_opacity.permute(0, 1, 4, 2, 3)

    c_rgb = c_rgb.permute(0, 1, 4, 2, 3)

    # c_mask = valid_mask.permute(0, 1, 4, 2, 3).repeat(1, 1, 3, 1, 1)
    # c_rgb[~c_mask] = rgb[~c_mask]
    # c_opacity = torch.ones_like(c_opacity)

    return {
        "xyz": xyz,
        "depth": depth,
        "rgb": c_rgb,
        "opacity": c_opacity,
        "scale": scale,
        "rotation": rotation,
        "deformation": deformation,
    }


@torch.no_grad()
def render_tracking_traj(
    renderer, 
    depth,
    rgb, 
    opacity, 
    scale, 
    rotation, 
    deformation, 
    cams_output, 
    rays_o,
    rays_d_un,
    rays_t_input,
    rays_t_output,
    rays_t_un_input,
    rays_t_un_output,
    cropping_output,
    args):

    n = 0 # Assume the batch index is being handled outside
    preds_arr = {}
    num_input_tokens = args.image_num_per_batch // args.temporal_subsample_freq

    for m in range(cams_output.shape[1]):

        if cropping_output is not None:
            render_height = cropping_output[n,m,0]
            render_width  = cropping_output[n,m,1]
            final_height = cropping_output[n,m,3] - cropping_output[n,m,2]
            final_width  = cropping_output[n,m,5] - cropping_output[n,m,4]
        else:
            if isinstance(args.output_image_res, Sequence):
                render_height = args.output_image_res[0]
                render_width  = args.output_image_res[1]
            else: # Backward compatible
                render_height = args.output_image_res
                render_width  = args.output_image_res
            final_height = render_height
            final_width  = render_width

        """
        Hubert: to be depricated, duplicated codes
        """
        t_in, t_out = None, None
        if args.temporal_subsampling:
            if args.temporal_subsample_emission == "center":
                assert args.temporal_subsample_freq % 2 == 1, "Got even number {}".format(args.temporal_subsample_freq)
                mid_ind = args.temporal_subsample_freq // 2
                xyz   = rays_o[n, mid_ind::args.temporal_subsample_freq] + \
                        rays_d_un[n, mid_ind::args.temporal_subsample_freq] * depth[n, :]
                t_in  = rays_t_input[n, mid_ind::args.temporal_subsample_freq]
                t_out = rays_t_output[n, m:m+1].repeat(num_input_tokens, 1, 1, 1)
            else:
                xyz   = rays_o[n, ::args.temporal_subsample_freq] + rays_d_un[n, ::args.temporal_subsample_freq] * depth[n, :]
                t_in  = rays_t_input[n, ::args.temporal_subsample_freq]
                t_out = rays_t_output[n, m:m+1].repeat(num_input_tokens, 1, 1, 1)

        if args.deform_parametrization == "rotor4d":
            xyz_d      = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            rgb_d      = rgb[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
            opacity_d  = opacity[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
            scale_d    = scale[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
            rotation_d = rotation[n, :].permute(0, 2, 3, 1).reshape(-1, 8)
        else:
            xyz_d      = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            rgb_d      = rgb[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
            opacity_d  = opacity[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
            scale_d    = scale[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
            rotation_d = rotation[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
            
        if t_in is not None:
            t_in_d  = t_in.permute(0, 2, 3, 1).reshape(-1, 1)
            t_out_d = t_out.permute(0, 2, 3, 1).reshape(-1, 1)
        else:
            t_in_d, t_out_d = None, None

        cur_deformation_d, deform_num_ch, _ = deformation_selection(deformation, rays_t_un_output, n, m, args)
        cur_deformation_d = {
            k: v[n].permute(0, 2, 3, 1).reshape(-1, deform_num_ch[k])
                for k,v in cur_deformation_d.items()}

        deformed_results = apply_deformation(
            xyz_d, rgb_d, opacity_d, scale_d, rotation_d, t_in_d, t_out_d, cur_deformation_d, args)

        if args.deform_parametrization == "rotor4d":
            cur_xyz, cur_cov3d = deformed_results
            preds = renderer(
                camera=cams_output[n,m],
                im_height=render_height,
                im_width=render_width,
                xyz=cur_xyz,
                rgb=rgb,
                opacity=opacity,
                cov3d=cur_cov3d,
            )
        else:
            # xyz_selected, rgb_selected, opacity_selected, scale_selected, rotation_selected
            cur_xyz, cur_rgb, cur_opacity, cur_scale, cur_rotation = deformed_results
            preds = renderer(
                camera=cams_output[n,m], 
                im_height=render_height, 
                im_width=render_width, 
                xyz=cur_xyz, 
                rgb=cur_rgb, 
                opacity=cur_opacity, 
                scale=cur_scale, 
                rotation=cur_rotation)

        if cropping_output is not None:
            _, _, hs, he, ws, we = cropping_output[n,m]
            preds = {
                k: v[:, :, hs:he, ws:we] if v.ndim == 4 else v
                    for k,v in preds.items()}

        for key in preds.keys():
            if key not in preds_arr:
                preds_arr[key] = []
            preds_arr[key].append(preds[key])

    for key in preds_arr.keys():
        preds_arr[key] = torch.cat(preds_arr[key], dim=0)

    return preds_arr["rgb"], preds_arr["mask"]

    
def compose_traj(rgb, mask):
    
    backward_traj = 4
    opacity_thres = 0.1
    ret_rgb, ret_mask = [], []
    for m in range(rgb.shape[0]):
        backward_traj_st = max(m-backward_traj, 0)
        backward_traj_ed = m + 1

        rgb_local_stack, mask_local_stack = [], []
        for t in range(backward_traj_st, backward_traj_ed):
            rgb_slice = rgb[backward_traj_st:backward_traj_ed]
            mask_slice = mask[backward_traj_st:backward_traj_ed]
            rgb_local_stack.append(rgb_slice)
            mask_local_stack.append(mask_slice)
        rgb_local_stack = torch.cat(rgb_local_stack, 0)
        mask_local_stack = torch.cat(mask_local_stack, 0)

        rgb_local = (
            (rgb_local_stack * mask_local_stack).sum(0, keepdims=True) / mask_local_stack.sum(0, keepdims=True)
        ).clamp(0, 1)
        mask_local = (
            mask_local_stack.sum(0, keepdims=True) / (mask_local_stack > 1e-4).sum(0, keepdims=True)
        ).clamp(0, 1)
        # rgb_local[(mask_local < 1e-4).any(0, keepdim=True).repeat(1,3,1,1)] = 0
        rgb_local[rgb_local.isnan()] = 0
        mask_local[mask_local.isnan()] = 0

        ret_rgb.append(rgb_local)
        ret_mask.append(mask_local)

    ret_rgb = torch.cat(ret_rgb, 0)
    ret_mask = torch.cat(ret_mask, 0)

    return ret_rgb, ret_mask


def get_interpolate_extrinsics(pose1, pose2, n_frames):
    """
    在两个相机位姿之间生成平滑的插值轨迹
    
    Args:
        pose1: 起始相机位姿 [4, 4]
        pose2: 结束相机位姿 [4, 4]
        n_frames: 需要生成的帧数（包括起始和结束位姿）
    
    Returns:
        poses: 插值后的相机位姿序列 [n_frames, 4, 4]
    """
    # 提取平移向量
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    
    # 提取旋转矩阵并转换为四元数
    r1 = Rotation.from_matrix(pose1[:3, :3])
    r2 = Rotation.from_matrix(pose2[:3, :3])
    
    # 创建时间点序列
    times = np.linspace(0, 1, n_frames)
    
    # 对平移进行线性插值
    translations = np.array([(1-t)*t1 + t*t2 for t in times])
    
    # 对旋转进行球面线性插值(SLERP)
    key_rots = Rotation.from_matrix(np.stack([pose1[:3, :3], pose2[:3, :3]]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rotations = slerp(times)
    
    # 组合成完整的变换矩阵序列
    poses = np.zeros((n_frames, 4, 4))
    poses[:, :3, :3] = rotations.as_matrix()
    poses[:, :3, 3] = translations
    poses[:, 3, 3] = 1.0
    
    return poses


