import torch
import torch.nn as nn
from typing import Sequence

def filter_selected(tensor, mask_flatten): 
    return tensor[~mask_flatten] 

class inpaint_aegaussian_loss:
    def __init__(self, project_gaussian_mode="ref"):
        self.project_gaussian_mode = project_gaussian_mode
        super().__init__()

    def forward_and_backward(
        self,
        renderer,
        depth,
        rgb,
        opacity,
        scale,
        rotation,
        instance_masks,
        cams_output,
        cropping_output,
        gts,
        depth_gt,
        depth_masks,
        mask_gt,
        rays_o,
        rays_d_un,
        loss_weights_dict,
        loss_func_dict,
        args,
        scaler=None,
        filter_point=True,
        rays_t_un_input=None,
        rays_t_un_output=None,
        global_step=None,
        view_mode=None,
        **kwargs,
    ):

        batch_size, im_num, _, height, width = next(iter(gts.values())).shape
        keys = list(gts.keys())
        loss_keys = list(loss_func_dict.keys())

        depth_d = nn.Parameter(depth.detach().clone())
        rgb_d = nn.Parameter(rgb.detach().clone())
        opacity_d = nn.Parameter(opacity.detach().clone())
        scale_d = nn.Parameter(scale.detach().clone())
        rotation_d = nn.Parameter(rotation.detach().clone())

        # Compute mask and depth loss
        # NOTE_YJQ: this is not in re10k dataset
        loss_arr = {}
        gaussians_arr = []
        if "depth_input" in loss_weights_dict and loss_weights_dict["depth_input"] > 0:
            depth_input_loss = loss_weights_dict[
                "depth_input"
            ] * nn.functional.mse_loss(depth_d * depth_masks, depth_gt * depth_masks)
            if scaler is not None:
                scaler.scale(depth_input_loss).backward()
            else:
                depth_input_loss.backward()
            loss_arr["depth_input"] = depth_input_loss

        if "mask_input" in loss_weights_dict and loss_weights_dict["mask_input"] > 0:
            # Gaussian at background region should have 0 opacity
            bg_mask = (mask_gt == 0).to(dtype=torch.float32)
            mask_input_loss = loss_weights_dict["mask_input"] * torch.mean(
                opacity_d**2.0 * bg_mask
            )
            if scaler is not None:
                scaler.scale(mask_input_loss).backward()
            else:
                mask_input_loss.backward()
            loss_arr["mask_input"] = mask_input_loss

        # Compute rendering loss
        preds_arr = {}
        gaussians_arr = []
        

        for n in range(0, batch_size):
            mask = mask_gt[n].permute(0, 2, 3, 1).reshape(-1)
            for m in range(0, im_num):
                xyz_d = rays_o[n, :] + rays_d_un[n, :] * depth_d[n, :]

                if self.project_gaussian_mode == "ref":
                    # project all gaussian for the 0-th image and mask
                    # some gaussian for all images after according to instance masks
                    (xyz_selected, 
                    rgb_selected, 
                    opacity_selected, 
                    opacity_selected, 
                    scale_selected, 
                    rotation_selected) = ([],[],[],[],[],[])
                    # keep all pixels from the first image
                    xyz_selected.append(xyz_d[0].permute(1, 2, 0).reshape(-1, 3))
                    rgb_selected.append(rgb_d[n, 0].permute(1, 2, 0).reshape(-1, 3))
                    opacity_selected.append(opacity_d[n, 0].permute(1, 2, 0).reshape(-1, 1))
                    scale_selected.append(scale_d[n, 0].permute(1, 2, 0).reshape(-1, 3))
                    rotation_selected.append(rotation_d[n, 0].permute(1, 2, 0).reshape(-1, 4))
                    # remove some gaussian according to instance masks

                    instance_mask_flatten = instance_masks[n, 1:].view(-1).bool()
                    xyz_selected.append(filter_selected(
                        xyz_d[1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                    rgb_selected.append(filter_selected(
                        rgb_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                    opacity_selected.append(filter_selected(
                        opacity_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 1), instance_mask_flatten))
                    scale_selected.append(filter_selected(
                        scale_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                    rotation_selected.append(filter_selected(
                        rotation_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 4), instance_mask_flatten))

                    xyz_selected = torch.cat(xyz_selected, axis=0)
                    rgb_selected = torch.cat(rgb_selected, axis=0)
                    opacity_selected = torch.cat(opacity_selected, axis=0)
                    scale_selected = torch.cat(scale_selected, axis=0)
                    rotation_selected = torch.cat(rotation_selected, axis=0)
                elif view_mode is None or view_mode == "4v":
                    xyz_selected = xyz_d.permute(0, 2, 3, 1).reshape(-1, 3)
                    rgb_selected = rgb_d[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
                    opacity_selected = opacity_d[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
                    scale_selected = scale_d[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
                    rotation_selected = rotation_d[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
                elif view_mode == "3v":
                    (xyz_selected, 
                    rgb_selected, 
                    opacity_selected, 
                    opacity_selected, 
                    scale_selected, 
                    rotation_selected) = ([],[],[],[],[],[])

                    # remove some gaussian according to instance mask from the first image
                    instance_mask_flatten = instance_masks[n, 0].view(-1).bool()
                    xyz_selected.append(filter_selected(
                        xyz_d[0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                    rgb_selected.append(filter_selected(
                        rgb_d[n, 0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                    opacity_selected.append(filter_selected(
                        opacity_d[n, 0].permute(1, 2, 0).reshape(-1, 1), instance_mask_flatten))
                    scale_selected.append(filter_selected(
                        scale_d[n, 0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                    rotation_selected.append(filter_selected(
                        rotation_d[n, 0].permute(1, 2, 0).reshape(-1, 4), instance_mask_flatten))
                    
                    # keep all gaussians from other views
                    instance_mask_flatten = instance_masks[n, 1:].view(-1).bool()
                    xyz_selected.append(xyz_d[1:].permute(0, 2, 3, 1).reshape(-1, 3))
                    rgb_selected.append(rgb_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3))
                    opacity_selected.append(opacity_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 1))
                    scale_selected.append(scale_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3))
                    rotation_selected.append(rotation_d[n, 1:].permute(0, 2, 3, 1).reshape(-1, 4))

                    xyz_selected = torch.cat(xyz_selected, axis=0)
                    rgb_selected = torch.cat(rgb_selected, axis=0)
                    opacity_selected = torch.cat(opacity_selected, axis=0)
                    scale_selected = torch.cat(scale_selected, axis=0)
                    rotation_selected = torch.cat(rotation_selected, axis=0)


                if filter_point:
                    if torch.sum(mask) == 0:
                        mask[0] = 1

                    xyz_selected = xyz_selected[mask > 0, :]
                    rgb_selected = rgb_selected[mask > 0, :]
                    opacity_selected = opacity_selected[mask > 0, :]
                    scale_selected = scale_selected[mask > 0, :]
                    rotation_selected = rotation_selected[mask > 0, :]
    
                gaussians = {}
                gaussians["xyz"] = xyz_selected
                gaussians["rgb"] = rgb_selected
                gaussians["opacity"] = opacity_selected
                gaussians["scale"] = scale_selected
                gaussians["rotation"] = rotation_selected
                gaussians_arr.append(gaussians)
                
                if cropping_output is not None:
                    im_height = cropping_output[n,m,0]
                    im_width  = cropping_output[n,m,1]
                else:
                    if isinstance(args.output_image_res, Sequence):
                        im_height = args.output_image_res[0]
                        im_width  = args.output_image_res[1]
                    else: # Backward compatible
                        im_height = args.output_image_res
                        im_width  = args.output_image_res
                preds = renderer(
                    camera=cams_output[n, m],
                    im_height=im_height,
                    im_width=im_width,
                    xyz=xyz_selected,
                    rgb=rgb_selected,
                    opacity=opacity_selected,
                    scale=scale_selected,
                    rotation=rotation_selected,
                )
                if cropping_output is not None:
                    _, _, hs, he, ws, we = cropping_output[n,m]
                    preds = {
                        k: v[:, :, hs:he, ws:we] if v.ndim == 4 else v
                            for k,v in preds.items()}
                for key in keys:
                    if key not in preds_arr:
                        preds_arr[key] = []
                    preds_arr[key].append(preds[key])

                loss_sum = 0
                for key in keys:
                    for loss_key in loss_keys:
                        if loss_weights_dict[loss_key][key] != 0:
                            func = loss_func_dict[loss_key]
                            loss = loss_weights_dict[loss_key][key] * func(
                                gts[key][n, m : m + 1], preds[key]
                            )
                            loss = torch.mean(loss) / batch_size / im_num
                            loss_sum += loss
                            if f"{key}_{loss_key}" not in loss_arr:
                                loss_arr[f"{key}_{loss_key}"] = loss
                            else:
                                loss_arr[f"{key}_{loss_key}"] += loss
                if scaler is not None:
                    scaler.scale(loss_sum).backward()
                else:
                    loss_sum.backward()

        for key in keys:
            preds_arr[key] = torch.cat(preds_arr[key], dim=0).reshape(
                batch_size, im_num, -1, height, width
            )

        grad = torch.cat(
            [
                depth_d.grad,
                rgb_d.grad,
                opacity_d.grad,
                scale_d.grad,
                rotation_d.grad,
            ],
            dim=2,
        )
        gaussian = torch.cat([depth, rgb, opacity, scale, rotation], dim=2)
        gaussian.backward(grad)

        for key in loss_arr.keys():
            if torch.is_tensor(loss_arr[key]):
                loss_arr[key] = loss_arr[key].item()

        return preds_arr, loss_arr, gaussians_arr

    def forward(
        self,
        renderer,
        xyz,
        rgb,
        opacity,
        scale,
        rotation,
        instance_masks,
        cams,
        cropping_output,
        mask_gt,
        args,
        filter_point=True,
        view_mode=None,
        **kwargs,
    ):
        batch_size, _, _, height, width = rgb.shape
        im_num = cams.shape[1]

        # Compute rendering loss
        preds_arr = {}
        gaussians_arr = []
        with torch.no_grad():
            for n in range(0, batch_size):
                mask = mask_gt[n].permute(0, 2, 3, 1).reshape(-1)
                for m in range(0, im_num):

                    if self.project_gaussian_mode == "ref":
                        (xyz_selected, 
                        rgb_selected, 
                        opacity_selected, 
                        opacity_selected, 
                        scale_selected, 
                        rotation_selected) = ([],[],[],[],[],[])

                        # keep all pixels from the first image
                        xyz_selected.append(xyz[n, 0].permute(1, 2, 0).reshape(-1, 3))
                        rgb_selected.append(rgb[n, 0].permute(1, 2, 0).reshape(-1, 3))
                        opacity_selected.append(opacity[n, 0].permute(1, 2, 0).reshape(-1, 1))
                        scale_selected.append(scale[n, 0].permute(1, 2, 0).reshape(-1, 3))
                        rotation_selected.append(rotation[n, 0].permute(1, 2, 0).reshape(-1, 4))
                        # remove some gaussian according to instance masks

                        instance_mask_flatten = instance_masks[n, 1:].view(-1).bool()
                        xyz_selected.append(filter_selected(
                            xyz[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                        rgb_selected.append(filter_selected(
                            rgb[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                        opacity_selected.append(filter_selected(
                            opacity[n, 1:].permute(0, 2, 3, 1).reshape(-1, 1), instance_mask_flatten))
                        scale_selected.append(filter_selected(
                            scale[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3), instance_mask_flatten))
                        rotation_selected.append(filter_selected(
                            rotation[n, 1:].permute(0, 2, 3, 1).reshape(-1, 4), instance_mask_flatten))

                        xyz_selected = torch.cat(xyz_selected, axis=0)
                        rgb_selected = torch.cat(rgb_selected, axis=0)
                        opacity_selected = torch.cat(opacity_selected, axis=0)
                        scale_selected = torch.cat(scale_selected, axis=0)
                        rotation_selected = torch.cat(rotation_selected, axis=0)
                    elif view_mode is None or view_mode == "4v":
                        xyz_selected = xyz[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
                        rgb_selected = rgb[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
                        opacity_selected = opacity[n, :].permute(0, 2, 3, 1).reshape(-1, 1)
                        scale_selected = scale[n, :].permute(0, 2, 3, 1).reshape(-1, 3)
                        rotation_selected = rotation[n, :].permute(0, 2, 3, 1).reshape(-1, 4)
                    elif view_mode == "3v":
                        (xyz_selected, 
                        rgb_selected, 
                        opacity_selected, 
                        opacity_selected, 
                        scale_selected, 
                        rotation_selected) = ([],[],[],[],[],[])


                        # remove some gaussian according to instance mask from the first image
                        instance_mask_flatten = instance_masks[n, 0].view(-1).bool()
                        xyz_selected.append(filter_selected(
                            xyz[n, 0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                        rgb_selected.append(filter_selected(
                            rgb[n, 0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                        opacity_selected.append(filter_selected(
                            opacity[n, 0].permute(1, 2, 0).reshape(-1, 1), instance_mask_flatten))
                        scale_selected.append(filter_selected(
                            scale[n, 0].permute(1, 2, 0).reshape(-1, 3), instance_mask_flatten))
                        rotation_selected.append(filter_selected(
                            rotation[n, 0].permute(1, 2, 0).reshape(-1, 4), instance_mask_flatten))
                        
                        # keep all gaussians from other views
                        instance_mask_flatten = instance_masks[n, 1:].view(-1).bool()
                        xyz_selected.append(xyz[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3))
                        rgb_selected.append(rgb[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3))
                        opacity_selected.append(opacity[n, 1:].permute(0, 2, 3, 1).reshape(-1, 1))
                        scale_selected.append(scale[n, 1:].permute(0, 2, 3, 1).reshape(-1, 3))
                        rotation_selected.append(rotation[n, 1:].permute(0, 2, 3, 1).reshape(-1, 4))

                        xyz_selected = torch.cat(xyz_selected, axis=0)
                        rgb_selected = torch.cat(rgb_selected, axis=0)
                        opacity_selected = torch.cat(opacity_selected, axis=0)
                        scale_selected = torch.cat(scale_selected, axis=0)
                        rotation_selected = torch.cat(rotation_selected, axis=0)

                    if filter_point:
                        if torch.sum(mask) == 0:
                            mask[0] = 1

                        xyz_selected = xyz_selected[mask > 0, :]
                        rgb_selected = rgb_selected[mask > 0, :]
                        opacity_selected = opacity_selected[mask > 0, :]
                        scale_selected = scale_selected[mask > 0, :]
                        rotation_selected = rotation_selected[mask > 0, :]

                    gaussians = {}
                    gaussians["xyz"] = xyz_selected
                    gaussians["rgb"] = rgb_selected
                    gaussians["opacity"] = opacity_selected
                    gaussians["scale"] = scale_selected
                    gaussians["rotation"] = rotation_selected
                    gaussians_arr.append(gaussians)

                    if cropping_output is not None:
                        im_height = cropping_output[n,m,0]
                        im_width  = cropping_output[n,m,1]
                    else:
                        if isinstance(args.output_image_res, Sequence):
                            im_height = args.output_image_res[0]
                            im_width  = args.output_image_res[1]
                        else: # Backward compatible
                            im_height = args.output_image_res
                            im_width  = args.output_image_res
                    preds = renderer(
                        camera=cams[n, m],
                        im_height=im_height,
                        im_width=im_width,
                        xyz=xyz_selected,
                        rgb=rgb_selected,
                        opacity=opacity_selected,
                        scale=scale_selected,
                        rotation=rotation_selected,
                        render_depth=args.render_depth,
                    )
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
                preds_arr[key] = torch.cat(preds_arr[key], dim=0).reshape(
                    batch_size, im_num, -1, height, width
                )

        return preds_arr, gaussians_arr