import argparse
import datetime
import json
import os
import time
import random
from tqdm import tqdm
from pathlib import Path

import cv2

import numpy as np
import matplotlib.pyplot as plt
import torch

import instainpaint.misc.utils as utils
from instainpaint.misc import checkpoint as checkpoint_utils
from instainpaint.misc import geometry_io, image_io

from instainpaint.data_loader.spin_nerf_dataset import SpinNerfDataset
from instainpaint.loss.inpaint_aegaussian_loss import inpaint_aegaussian_loss
from instainpaint.misc.env_utils import fix_random_seeds, init_distributed_mode

from instainpaint.misc.io_helper import mkdirs, pathmgr
from instainpaint.models.aegaussian_decoder import AeGaussianTransformer
from instainpaint.models.gaussian_decoder import GaussianMlpUpsampler
from instainpaint.models.multiview_encoder import mvencoder_base, ExtraTokenEmbed
from instainpaint.renderer.gaussian_renderer import GaussianRenderer
from instainpaint.misc.eval_metrics import compute_psnr, compute_ssim, compute_lpips
from instainpaint.misc.camera_paths import (
    get_circle_extrinsics,
)

os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp"

EMBED_DIM = 1024
PATCH_SIZE = 8
TRANSFORMER_DEPTH = 24
DEPTH_BIAS = -4
MODEL_INPUT_CHANNELS = 4

def worker_init_fn(worker_id):                                                                                                                                
    seed = 42
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return


def get_args_parser():
    parser = argparse.ArgumentParser("SpinNeRF inpaint LRM evaluation")
    parser.add_argument("--exp_root", required=True, type=str)
    parser.add_argument("--exp_name", default="default_exp_name", type=str)
    parser.add_argument("--batch_size_per_gpu", default=1, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--fov", default=None, type=utils.parse_tuple_args)
    parser.add_argument("--image_num_per_batch", default=4, type=int)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_eval_json", action="store_true")
    parser.add_argument("--input_image_res", default=(512, 904), type=utils.parse_tuple_args)
    parser.add_argument("--output_image_res", default=(512, 904), type=utils.parse_tuple_args)
    parser.add_argument("--output_image_num", default=40, type=int)
    parser.add_argument("--dataset_type", default="spin_nerf_dataset", choices=["spin_nerf_dataset"])
    parser.add_argument("--dataset_formulation", default="scene", choices=["scene"])
    parser.add_argument("--interactive_session", action="store_true")
    parser.add_argument("--centralized_cropping", action="store_true")
    parser.add_argument("--eval_index_json_file", default=None, type=str)
    parser.add_argument("--save_model_num", default=30, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--camera_trajectory", default="circle", choices=["circle"])
    parser.add_argument("--mask_cache_path", default=None, type=str)
    parser.add_argument("--context_json", default=None, type=str)
    parser.add_argument("--vis_center_file", default=None, type=str)
    parser.add_argument("--preinpaint_json", default=None, type=str)
    parser.add_argument("--vis_scale", default=1.0, type=float)
    parser.add_argument("--output_has_gt", action="store_true")
    parser.add_argument("--depth_vis_mode", default="inpaint", choices=["whole", "inpaint"])
    parser.add_argument("--cal_metric_mode", action="store_true")
    parser.add_argument("--skip_image_saving", action="store_true")

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_to_eval", default=-1, type=int)
    parser.add_argument(
        "--eval_metrics",
        default="",
        type=lambda s: s.split(",") if s != "" else [],
    )
    parser.add_argument("--eval_mask", default=None)
    parser.add_argument("--save_inputs", action="store_true")
    parser.add_argument("--save_gts", action="store_true")
    parser.add_argument("--normalize_mode", default="all", choices=["all", "input"])
    parser.add_argument("--models", nargs="+", default=None, type=str)

    return parser


def create_output_dir(args):    
    args.output_dir = os.path.join(args.exp_root, args.exp_name)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.image_dir = os.path.join(args.output_dir, "images")

    mkdirs(args.output_dir)
    return args


def test_lrm(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus <=1 : 
        args.gpu = "cuda"
    else: 
        print("There are multiple GPUS detected. Using distributed mode.")
        if args.interactive_session:
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
        init_distributed_mode(args)

        print("*" * 80)
        print(
            f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
        )
        print(args)
        print("*" * 80)

    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    if utils.is_main_process():
        with pathmgr.open(os.path.join(args.output_dir, "args.txt"), "w") as fOut:
            fOut.write(
                "\n".join(
                    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
                )
            )

    loss_weights_file = os.path.join(
        os.path.dirname(__file__), "config", "aegaussian.yaml"
    )
    if utils.is_main_process():
        pathmgr.copy_from_local(
            loss_weights_file,
            os.path.join(args.output_dir, "weights.yaml"),
            overwrite=True,
        )

    mvencoder = mvencoder_base(
        type="plucker",
        with_bg=False,
        in_chans=MODEL_INPUT_CHANNELS,
        depth=0,
        embed_dim=EMBED_DIM,
        patch_size=PATCH_SIZE,
        use_pos_embed=False,
        emb_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        input_image_num=args.image_num_per_batch,
        temporal_subsampling=False,
        temporal_subsample_freq=8,
    )

    extra_token_embed = ExtraTokenEmbed(
        embed_dim=EMBED_DIM,
        input_image_num=args.image_num_per_batch,
        use_time_embed=False,
        use_triplane=False,
        triplane_num_tokens=32**2 * 3,
    )

    gaudecoder = AeGaussianTransformer(
        embed_dim=EMBED_DIM,
        depth=TRANSFORMER_DEPTH,
        attn_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
    )

    gauupsampler = GaussianMlpUpsampler(
        mlp_dim=1024,
        token_dim=EMBED_DIM, 
        patch_size=PATCH_SIZE,
        depth_bias=DEPTH_BIAS,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
        input_image_num=args.image_num_per_batch,
    )

    renderer = GaussianRenderer()

    loss = inpaint_aegaussian_loss("all")

    eva_input_views = list(range(args.image_num_per_batch))
    args.eva_output_views = list(range(args.output_image_num))
    
    dataset = SpinNerfDataset(
        root_dir=args.data_path,
        mode="TEST",
        input_image_num=args.image_num_per_batch,
        input_image_res=args.input_image_res,
        output_image_res=args.output_image_res,
        output_image_num=args.output_image_num,
        output_has_gt=args.output_has_gt,
        use_depth=False,
        cal_metric_mode=args.cal_metric_mode,
        context_json=args.context_json,
        preinpaint_json=args.preinpaint_json,
        vis_center_file=args.vis_center_file,
        normalize_mode=args.normalize_mode,
        models=args.models,
        mvinpaint_path=None,
        use_tri_value_mask=False,
        multimask_root=None,
    )

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    else: 
        sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"Data loaded: there are {len(dataset)} 3D models.")

    mvencoder = mvencoder.cuda().eval()
    
    for para in mvencoder.parameters():
        para.requires_grad = False
    extra_token_embed = extra_token_embed.cuda().eval()
    for para in extra_token_embed.parameters():
        para.requires_grad = False
    gaudecoder = gaudecoder.cuda().eval()
    for para in gaudecoder.parameters():
        para.requires_grad = False
    gauupsampler = gauupsampler.cuda().eval()
    for para in gauupsampler.parameters():
        para.requires_grad = False
    renderer = renderer.cuda()

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    checkpoint_utils.restart_from_checkpoint(
        args.checkpoint,
        run_variables=to_restore,
        mvencoder=mvencoder,
        extra_token_embed=extra_token_embed,
        gaudecoder=gaudecoder,
        gauupsampler=gauupsampler,
        load_weights_only=False,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting LRM inference !")
    if torch.torch.distributed.is_initialized():
        data_loader.sampler.set_epoch(start_epoch)
    # ============ testing one epoch of DINO ... ============
    test_one_epoch(
        mvencoder,
        extra_token_embed,
        gaudecoder,
        gauupsampler,
        renderer,
        loss,
        data_loader,
        eva_input_views,
        args.eva_output_views,
        start_epoch,
        args,
    )
    log_stats = {"epoch": start_epoch}
    if utils.is_main_process():
        with pathmgr.open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

    return


def save_imgs_one_sample(preds, batch, batch_id, input_im_ids, output_im_ids, rgb_names_output, folder, 
                         skip_image_saving=False, save_inputs=False, 
                         save_gts=False, args=None):
    eval_dict = [{"id": n} for n in range(0, len(output_im_ids))]
    pred_keys = []
    for key, pred in preds.items():

        if key == "gradient":
            continue  # to be finished.

        elif key == "depth":
            if not skip_image_saving:
                name = os.path.join(folder, "%03d_depth.png")
                image_io.save_single_depth(pred[batch_id, :], name, output_im_ids)

                for n in range(0, len(output_im_ids)):
                    eval_dict[n]["Predicted_Depth"] = os.path.join(
                        folder, f"{output_im_ids[0]}_depth.png"
                    )
                pred_keys.append("depth")
            
            if args.cal_metric_mode:
                if args.depth_vis_mode == "whole":
                    depth_output_folder = os.path.join(folder, "depth_output")
                    os.makedirs(depth_output_folder, exist_ok=True)
                    for depth, image_name in zip(pred[batch_id, :], rgb_names_output):
                        depth = 1 - (depth-depth.min())/(depth.max()-depth.min())
                        depth = depth.detach().cpu().numpy()[0]
                        depth = (255 * np.clip(depth, 0, 1)).astype(np.uint8)
                        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                        name = os.path.join(depth_output_folder, Path(image_name).name)
                        cv2.imwrite(name, depth)
                else:
                    depth_output_folder = os.path.join(folder, "depth_output")
                    output_masks = batch["instance_masks_output"][batch_id, :]
                    os.makedirs(depth_output_folder, exist_ok=True)

                    def get_bounding_box(mask):
                        rows = np.where(np.any(mask, axis=1))[0] 
                        cols = np.where(np.any(mask, axis=0))[0] 
                        if len(rows) == 0 or len(cols) == 0:
                            return None
                        y_min, y_max = rows[0], rows[-1]
                        x_min, x_max = cols[0], cols[-1]
                        return (x_min, y_min, x_max, y_max)
                    
                    for depth, image_name, output_mask in zip(pred[batch_id, :], rgb_names_output, output_masks):
                        depth = depth[0]
                        x_min, y_min, x_max, y_max = get_bounding_box(output_mask.cpu().numpy().astype(np.uint8))
                        depth = depth[y_min:y_max, x_min:x_max]
                        # depth[output_mask == 0] = 0
                        depth = 1 - (depth-depth.min())/(depth.max()-depth.min())
                        depth = depth.detach().cpu().numpy()
                        depth = (255 * np.clip(depth, 0, 1)).astype(np.uint8)
                        depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                        name = os.path.join(depth_output_folder, Path(image_name).name)
                        cv2.imwrite(name, depth)

        elif key == "rgb":
            name = os.path.join(folder, "%03d_rgb.png")
            
            if not skip_image_saving:
                image_io.save_single_png(pred[batch_id, :], name, output_im_ids)

            if not skip_image_saving:
                for n in range(0, len(output_im_ids)):
                    eval_dict[n]["Output"] = os.path.join(
                        folder, "%03d_rgb.png" % output_im_ids[n]
                    )
                pred_keys.append("rgb")

            if save_inputs and (not skip_image_saving):
                input_name = os.path.join(folder, "%03d_rgb_input.png")
                image_io.save_single_png(batch["rgb_input"][batch_id, :], input_name, input_im_ids)
                for n in range(len(output_im_ids)):
                    for i in range(len(input_im_ids)):
                        eval_dict[n][f"Input{i}"] = os.path.join(
                            folder, "%03d_rgb_input.png" % input_im_ids[i]
                        )
                # additionally save the input mask
                input_mask_name = os.path.join(folder, "%03d_mask_input.png")
                image_io.save_single_png(batch["instance_masks_input"][batch_id, 0:1], input_mask_name, input_im_ids[0:1])
                eval_dict[n][f"Input_mask"] = os.path.join(
                    folder, "%03d_mask_input.png" % input_im_ids[0]
                )

            if save_gts and (not skip_image_saving):
                for n in range(0, len(output_im_ids)):
                    gt_name = os.path.join(folder, "%03d_rgb_gt.png")
                    image_io.save_single_png(batch["rgb_output"][batch_id, :], gt_name, output_im_ids)
                    eval_dict[n][f"GT"] = os.path.join(
                        folder, "%03d_rgb_gt.png" % output_im_ids[n]
                    )
                    # saves output_masks
                    output_mask_name = os.path.join(folder, "%03d_mask_output.png")
                    image_io.save_single_png(batch["instance_masks_output"][batch_id, :], output_mask_name, output_im_ids)
                    eval_dict[n][f"Output_mask"] = os.path.join(
                        folder, "%03d_mask_output.png" % output_im_ids[n]
                    )
            
            if args.cal_metric_mode:
                metric_output_folder = os.path.join(folder, "metric_output")
                os.makedirs(metric_output_folder, exist_ok=True)
                for rgb, image_name in zip(pred[batch_id, :], rgb_names_output):
                    rgb = ((rgb + 1) / 2).cpu().numpy().transpose(1,2,0)
                    name = os.path.join(metric_output_folder, Path(image_name).name)
                    plt.imsave(name, rgb)

        elif key == "mask":
            if not skip_image_saving:
                name = os.path.join(folder, "%03d_mask.png")
                image_io.save_single_png(
                    pred[batch_id, :],
                    name,
                    output_im_ids,
                    is_gamma=True,
                )

    return eval_dict, pred_keys


def test_one_epoch(
    mvencoder,
    extra_token_embed,
    gaudecoder,
    gauupsampler,
    renderer,
    loss,
    data_loader,
    eva_input_views,
    eva_output_views,
    epoch,
    args,
):

    eval_dict_array = []
    save_model_inds = np.arange(len(data_loader))
    if args.save_model_num == 0:
        save_model_inds = []
    elif args.save_model_num > 0:
        save_freq = max(len(data_loader) // args.save_model_num, 1)
        save_model_inds = save_model_inds[::save_freq]

    metrics_record = {metric: [] for metric in args.eval_metrics}
    for it, batch in enumerate(tqdm(data_loader)):

        if (args.num_to_eval > 0) and (it > args.num_to_eval):
            break

        auto_cast_dtype = torch.bfloat16
        preds, gaussians, gaussians_arr = test_one_iteration(
            batch,
            mvencoder,
            extra_token_embed,
            gaudecoder,
            gauupsampler,
            renderer,
            loss,
            args,
            auto_cast_dtype=auto_cast_dtype,
        )
        # logging
        torch.cuda.synchronize()

        model_ids = batch["name"]

        batch_size = len(model_ids)
        for b in range(0, batch_size):
            print("Model Id: %s" % model_ids[b])
            if "fov" in batch:
                args.fov = batch["fov"][b].item()
            if "eva_input_views" in batch:
                if isinstance(batch["eva_input_views"], np.ndarray):
                    eva_input_views = batch["eva_input_views"][b, :]
                    eva_input_views = eva_input_views.numpy().tolist()
                else:
                    eva_input_views = batch["eva_input_views"][b]
            if "eva_output_views" in batch:
                if isinstance(batch["eva_output_views"], np.ndarray):
                    eva_output_views = batch["eva_output_views"][b, :]
                    eva_output_views = eva_output_views.numpy().tolist()
                else:
                    eva_output_views = batch["eva_output_views"][b]
            if "rgb_names_output" in batch:
                rgb_names_output = sum(batch["rgb_names_output"],[])
            model_id = model_ids[b]
            test_folder = "test-" + args.dataset_type
            model_dir = os.path.join(args.output_dir, test_folder, model_id)

            if (it in save_model_inds):
                mkdirs(model_dir)

                # if the output don't have gt, we can't possibly save gt
                if not args.output_has_gt:
                    args.save_gts = False

                eval_dict, pred_keys = save_imgs_one_sample(
                    preds, batch, b, eva_input_views, eva_output_views, rgb_names_output, model_dir, 
                    args.skip_image_saving, args.save_inputs, args.save_gts, args=args,
                )
                
                if args.save_video:
                    # save an interpolated gif
                    n_frames = 72
                    cam_loc_scale = batch["cam_loc_scale"][0].item()
                    if "vis_center_pose" in batch:
                        poses = get_circle_extrinsics(
                            batch["camera_poses"][0], 
                            n_frames, 
                            policy="selected",
                            center_pose=batch["vis_center_pose"][0],
                            vis_scale=args.vis_scale,
                            cam_loc_scale=cam_loc_scale,
                        )
                    else:
                        poses = get_circle_extrinsics(
                            batch["camera_poses"][0],
                            n_frames,
                            cam_loc_scale=cam_loc_scale,
                        )

                    camera_int = batch["cameras_output"][0][0][16:]
                    poses = torch.from_numpy(np.concatenate([poses.reshape(n_frames, 16), camera_int.reshape(1, 4).expand(n_frames, -1)], axis=1).astype(np.float32)).to(args.gpu)
                    all_preds = []
                    all_depths = []
                    all_masks = []
                    for i in range(n_frames):
                        poses[:, -4] = poses[:, -3]
                        my_preds = renderer(
                            camera=poses[i],
                            # im_height=args.output_image_res[0],
                            # im_width=args.output_image_res[1],
                            im_height=512,
                            im_width=512,
                            xyz=gaussians_arr[0]["xyz"],
                            rgb=gaussians_arr[0]["rgb"],
                            opacity=gaussians_arr[0]["opacity"],
                            scale=gaussians_arr[0]["scale"],
                            rotation=gaussians_arr[0]["rotation"],
                            render_depth=True
                        )
                        all_preds.append(my_preds["rgb"][0, 0:3])
                        all_depths.append(my_preds["rgb"][0, 3])
                        all_masks.append(my_preds["mask"][0, 0])
                    all_preds = torch.stack(all_preds, dim=0)
                    all_depths = torch.stack(all_depths, dim=0)
                    all_masks = torch.stack(all_masks, dim=0)

                    # normalize all depths based on all masks
                    # min_vals = all_depths.view(all_depths.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                    # max_vals = all_depths.view(all_depths.size(0), -1).max(dim=1)[0].view(-1, 1, 1)

                    alpha_theshold = 0.2
                    min_val = all_depths[all_masks > alpha_theshold].min()
                    max_val = all_depths[all_masks > alpha_theshold].max()

                    all_depths = torch.where(
                        all_masks > alpha_theshold, all_depths, max_val
                    )
                    all_depths = torch.tensor(geometry_io.replace_outliers(all_depths.cpu().numpy(), method="max"))
                    min_val = all_depths.min()
                    max_val = all_depths.max()
                    all_depths = 1 - (all_depths - min_val) / (max_val - min_val + 1e-8)

                    video_output_dir = Path(args.output_dir) / "video_output"
                    os.makedirs(video_output_dir, exist_ok=True)
                    # image_io.save_as_gif(all_preds, os.path.join(model_dir, "interpolated_rgb.gif"), fps=10, image_ids=np.arange(n_frames))
                    image_io.save_as_mp4(all_preds, video_output_dir / f"{model_id}.mp4", fps=24)
                    image_io.save_as_mp4(all_depths, video_output_dir / f"{model_id}_depth.mp4", fps=24, is_depth=True)

                    eval_dict_array.append(eval_dict)


        if args.output_has_gt:
            with torch.no_grad():
                is_eval_depth = any(["depth-" in metric for metric in args.eval_metrics])
                is_eval_flow  = any(["flow-" in metric for metric in args.eval_metrics])
                
                assert preds["rgb"].shape[0] == 1
                img_pred = (preds["rgb"][0].clamp(-1, 1) + 1) / 2
                img_real = (batch["rgb_output"][0].to(device=args.gpu).clamp(-1, 1) + 1) / 2

                if is_eval_depth:
                    depth_real = None
                    depth_pred = None
                    depth_diff = depth_real - depth_pred
                if is_eval_flow:
                    flow_real = None
                    flow_pred_sparse = None
                    flow_pred_dense  = None

                if args.eval_mask is not None:
                    eval_mask = batch[args.eval_mask][0].to(device=args.gpu)
                else:
                    eval_mask = None

                if "valid_mask_output" in batch:
                    assert batch_size == 1, "Did not consider otherwise"
                    valid_mask_output = batch["valid_mask_output"].to(device=args.gpu)
                    if valid_mask_output.sum().item() == 0:
                        continue
                    img_real = img_real[valid_mask_output[0]]
                    img_pred = img_pred[valid_mask_output[0]]
                    if eval_mask is not None:
                        eval_mask = eval_mask[valid_mask_output[0]]
                    if any(["flow" in metric for metric in args.eval_metrics]):
                        flow_pred_sparse = flow_pred_sparse[valid_mask_output[0]]
                        flow_real = flow_real[valid_mask_output[0]]
                    if any(["depth" in metric for metric in args.eval_metrics]):
                        depth_real = depth_real[valid_mask_output[0]]
                        depth_pred = depth_pred[valid_mask_output[0]]
                else:
                    assert args.dataset_type != "dycheck_dataset", "Just in case"

    if args.output_has_gt:
        if len(args.eval_metrics) > 0:
            with open(os.path.join(args.output_dir, "metrics.log"), "a") as fp:
                fp.write(f"[*] Eval epoch {epoch} w/ ckpt: {args.checkpoint}\n")
                for metric in args.eval_metrics:
                    avg_val = torch.cat(metrics_record[metric], 0).mean().item()
                    print(" [{}] {:.6f}".format(metric, avg_val))
                    fp.write("\t[{}] {:.6f}\n".format(metric, avg_val))
                fp.write("\n")

    return

def encode_plucker_rays(mvencoder, instance_mask, rgb, rays_o, rays_d, rays_t, batch_size, image_num, height, width, auto_cast_dtype, args):
    if rgb is not None:
        rgb_enc = rgb.reshape(batch_size * image_num, -1, height, width)
        rgb_enc = rgb_enc.to(device=args.gpu, dtype=auto_cast_dtype)
        instance_mask = instance_mask.reshape(batch_size * image_num, -1, height, width)
        instance_mask = instance_mask.to(device=args.gpu, dtype=auto_cast_dtype)
        rgb_enc = torch.cat([rgb_enc, instance_mask], dim=1)
    else:
        rgb_enc = None

    rays_o_enc = rays_o.reshape(
        batch_size * image_num, 3, height, width
    )
    rays_d_enc = rays_d.reshape(
        batch_size * image_num, 3, height, width
    )

    plucker_rays = torch.cat(
        [rays_d_enc, torch.cross(rays_o_enc, rays_d_enc, dim=1)], dim=1
    )

    tokens = mvencoder(
        rgb_enc,
        plucker_rays.to(device=args.gpu, dtype=auto_cast_dtype),
    )
    tokens = tokens[:, 1:, :]

    token_num = tokens.shape[1]

    tokens = tokens.reshape(batch_size, image_num, token_num, -1)
    tokens = tokens.flatten(1,2)

    return tokens


def test_one_iteration(
    batch,
    mvencoder,
    extra_token_embed,
    gaudecoder,
    gauupsampler,
    renderer,
    loss,
    args,
    auto_cast_dtype=torch.bfloat16,
):
    images = batch["rgb_input"]
    cams_output = batch["cameras_output"]
    cams_input = batch["cameras_input"]
    cropping_output = batch.get("cropping_output", None)
    view_mode = None

    batch_size, image_num, ch, height, width = images.shape

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        
        rgb_input    = batch["rgb_input"].to(device=args.gpu)
        instance_masks_input = batch["instance_masks_input"].to(device=args.gpu, dtype=auto_cast_dtype)
        rays_o_input = batch["rays_o_input"].to(device=args.gpu).permute(0, 1, 4, 2, 3)
        rays_d_input = batch["rays_d_input"].to(device=args.gpu).permute(0, 1, 4, 2, 3)
        rays_t_input, rays_t_output, rays_t_un_output = None, None, None

        # remove the mask of the reference image
        for i in range(batch_size):
            instance_masks_input[i, 0] = 0

        tokens = encode_plucker_rays(
            mvencoder, instance_masks_input, rgb_input, rays_o_input, rays_d_input, rays_t_input, 
            batch_size, image_num, height, width, auto_cast_dtype, args)

        tokens = extra_token_embed(tokens)

        tokens = gaudecoder(tokens)

        cams_input = cams_input.to(device=args.gpu, dtype=auto_cast_dtype)
        gaussians = gauupsampler(
            tokens, None, cams_input, images, 
            rays_o_input, rays_d_input, rays_t_input, rays_t_output, rays_t_un_output, mode=view_mode)

    end.record()
    torch.cuda.synchronize()
    print("Inference time: %.3fs" % (float(start.elapsed_time(end)) / 1000.0))

    for key in gaussians.keys():
        if isinstance(gaussians[key], torch.Tensor):
            gaussians[key] = gaussians[key].to(dtype=torch.float32)
        elif isinstance(gaussians[key], dict):
            gaussians[key] = {
                k: v.to(dtype=torch.float32) if isinstance(v, torch.Tensor) else v
            for k,v in gaussians[key].items()}
        else:
            assert gaussians[key] is None, "Got unknown type {}".format(type(gaussians[key]))

    mask_gt = batch["mask_input"]
    mask_gt = mask_gt.to(args.gpu)

    # from depth to xyz
    rays_o = batch["rays_o_input"].permute(0, 1, 4, 2, 3).to(device=args.gpu)
    rays_d_un = batch["rays_d_un_input"].permute(0, 1, 4, 2, 3).to(device=args.gpu)
    

    gaussians["xyz"] = rays_o + rays_d_un * gaussians["depth"]

    cams_output = cams_output.to(device=args.gpu)
    preds, gaussians_arr = loss.forward(
        renderer,
        gaussians["xyz"],
        gaussians["rgb"],
        gaussians["opacity"],
        gaussians["scale"],
        gaussians["rotation"],
        instance_masks_input,
        cams_output,
        cropping_output,
        mask_gt,
        args,
        filter_point=False,
        return_gs=True,
        view_mode=view_mode
    )
    if args.render_depth:
        preds["depth"] = preds["rgb"][:, :, -1:]
        preds["rgb"] = preds["rgb"][:, :, :-1]

    return preds, gaussians, gaussians_arr


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    test_lrm(args)
