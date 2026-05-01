import argparse
import datetime
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

import instainpaint.misc.dist_helper as dist_helper
import instainpaint.misc.logging as logging
import instainpaint.misc.utils as utils
from instainpaint.misc import checkpoint as checkpoint_utils
from instainpaint.misc import image_io, optim as optim_utils
from instainpaint.data_loader.dl3dv_dataset import DL3DVDataset
from instainpaint.loss.inpaint_aegaussian_loss import inpaint_aegaussian_loss
from instainpaint.loss.perceptual_loss import PerceptualLoss
from instainpaint.misc.env_utils import fix_random_seeds, init_distributed_mode
from instainpaint.misc.dist_helper import get_world_size
from instainpaint.misc.io_helper import mkdirs, pathmgr
from instainpaint.models.aegaussian_decoder import AeGaussianTransformer
from instainpaint.models.gaussian_decoder import GaussianMlpUpsampler
from instainpaint.models.multiview_encoder import mvencoder_base, ExtraTokenEmbed
from instainpaint.renderer.gaussian_renderer import GaussianRenderer


os.environ["TORCH_EXTENSIONS_DIR"] = "/tmp"
logger = logging.get_logger(__name__)

EMBED_DIM = 1024
MODEL_INPUT_CHANNELS = 4


def custom_restart_from_checkpoint(
    ckp_path, 
    **kwargs
):
    """
    Re-start from checkpoint
    """
    if not pathmgr.isfile(ckp_path):
        raise ValueError(f"Checkpoint not found at {ckp_path}")
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    if get_world_size() == 1:
        ckp_path = pathmgr.get_local_path(ckp_path)
    with pathmgr.open(ckp_path, "rb") as fb:
        checkpoint = torch.load(fb, map_location="cpu", weights_only=False)

    assert "mvencoder" in kwargs.keys(), "mvencoder not found in kwargs"
    mvencoder_module = getattr(kwargs["mvencoder"], "module", kwargs["mvencoder"])
    patch_weight_key = "module.patch_embed.proj.weight"
    if (
        mvencoder_module.in_chans == 4
        and "mvencoder" in checkpoint
        and patch_weight_key in checkpoint["mvencoder"]
        and checkpoint["mvencoder"][patch_weight_key].shape[1] == 3
    ):
        proj_weight = checkpoint["mvencoder"][patch_weight_key].clone()
        proj_weight_new = torch.mean(proj_weight, dim=1, keepdim=True)
        checkpoint["mvencoder"][patch_weight_key] = torch.cat(
            [proj_weight, proj_weight_new], dim=1
        )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in checkpoint and value is not None:
            try:
                msg = checkpoint_utils.load_ddp_state_dict(
                    value, checkpoint[key], key=key, 
                    rewrite_weights=[])
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

def create_output_dir(args):
    try:
        args.user = os.environ["USER"] or "default"
    except KeyError:
        args.user = "root"    
    args.output_dir = os.path.join(args.exp_root, args.exp_name)
        
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.image_dir = os.path.join(args.output_dir, "images")
    args.logging_save_path = os.path.join(args.output_dir, "log_verbose_rank_0.txt")

    mkdirs(args.output_dir)
    mkdirs(args.checkpoint_dir)
    mkdirs(args.image_dir)

    # Setup logging format.
    logging.setup_logging(
        args.logging_save_path,
        mode="w",
        buffering=1024,
    )
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("LRM Gaussian", add_help=False)

    parser.add_argument("--exp_root", required=True, type=str)
    parser.add_argument("--exp_name", default="default_exp_name", type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--pretrain_ckpt_path", default=None, type=str)
    parser.add_argument("--loss_weights_file", default=None, type=str)

    parser.add_argument("--dataset_type", default="dl3dv_dataset", choices=["dl3dv_dataset"])
    parser.add_argument("--dataset_formulation", default="scene", choices=["scene"])
    parser.add_argument("--batch_size_per_gpu", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--image_num_per_batch", default=4, type=int)
    parser.add_argument("--output_image_num", default=8, type=int)
    parser.add_argument("--input_image_res", default=(512, 512), type=utils.parse_tuple_args)
    parser.add_argument("--output_image_res", default=(512, 512), type=utils.parse_tuple_args)
    parser.add_argument("--centralized_cropping", action="store_true")

    parser.add_argument("--transformer_depth", default=24, type=int)
    parser.add_argument("--upsampler_mlp_dim", default=1024, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--depth_bias", default=-4, type=int)
    parser.add_argument("--weight_norm", action="store_true")
    parser.add_argument("--remove_attn_bias", action="store_true")
    parser.add_argument("--remove_emb_bias", action="store_true")
    parser.add_argument("--remove_norm_bias", action="store_true")
    parser.add_argument("--remove_norm_affine", action="store_true")

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=8e-5, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--warmup_iters", default=200, type=int)
    parser.add_argument("--seed", default=1024, type=int)
    parser.add_argument("--saveimg_iter_freq", default=300, type=int)
    parser.add_argument("--saveckp_epoch_freq", default=5, type=int)
    parser.add_argument("--backup_ckp_epoch_freq", default=10, type=int)
    parser.add_argument("--interactive_session", action="store_true")

    parser.add_argument("--mask_cache_path", default=None, type=str)
    parser.add_argument("--clip_len", type=int, choices=[5, 15], default=15)
    parser.add_argument("--mask_mode", default="instance+random+3dconsistent", type=str)
    parser.add_argument("--mask_prob", nargs="+", default=None, type=float)
    parser.add_argument("--stereo_depth_cache_path", default=None, type=str)
    parser.add_argument("--project_gaussian_mode", default="all", choices=["ref", "all"], type=str)
    parser.add_argument("--mask_multiple_objects", action="store_true")


    return parser


def get_optimizer_args(lr):
    return argparse.Namespace(
        lr=lr,
        weight_decay=0.05,
        deform_lr_multiplier=-1,
        deform_param_group_keywords="",
        freeze_backbone=False,
        freeze_transformer=False,
    )


def train_lrm(args):
    init_distributed_mode(args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    logger.info("*" * 80)
    logger.info(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    logger.info(args)
    logger.info("*" * 80)

    tb_writer = None
    if SummaryWriter is not None and utils.is_main_process():
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    if utils.is_main_process():
        with pathmgr.open(os.path.join(args.output_dir, "args.txt"), "w") as fOut:
            fOut.write(
                "\n".join(
                    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
                )
            )

    if args.loss_weights_file is None:
        loss_weights_file = os.path.join(
            os.path.dirname(__file__), "config", "aegaussian.yaml"
        )
    else:
        loss_weights_file = os.path.join(
            os.path.dirname(__file__), "config", f"{args.loss_weights_file}.yaml"
        )

    if utils.is_main_process():
        pathmgr.copy_from_local(
            loss_weights_file,
            os.path.join(args.output_dir, "weights.yaml"),
            overwrite=True,
        )

    with pathmgr.open(loss_weights_file, "r") as fIn:
        loss_weights_dict = yaml.safe_load(fIn)

    mvencoder = mvencoder_base(
        type="plucker",
        in_chans=MODEL_INPUT_CHANNELS,
        drop_path_rate=0.0,
        cp_freq=1,
        depth=0,
        embed_dim=EMBED_DIM,
        patch_size=args.patch_size,
        use_pos_embed=False,
        emb_use_bias=(not args.remove_emb_bias),
        norm_use_bias=(not args.remove_norm_bias),
        norm_use_affine=(not args.remove_norm_affine),
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
        depth=args.transformer_depth,
        drop_path_rate=0.0,
        cp_freq=1,
        attn_use_bias=(not args.remove_attn_bias),
        norm_use_bias=(not args.remove_norm_bias),
        norm_use_affine=(not args.remove_norm_affine),
        use_weight_norm=args.weight_norm,
    )
    gauupsampler = GaussianMlpUpsampler(
        mlp_dim=args.upsampler_mlp_dim,
        mlp_depth=1,
        token_dim=EMBED_DIM,
        patch_size=args.patch_size,
        depth_bias=args.depth_bias,
        norm_use_bias=(not args.remove_norm_bias),
        norm_use_affine=(not args.remove_norm_affine),
        use_weight_norm=args.weight_norm,
        input_image_num=args.image_num_per_batch,
        color_space="rgb",
        decode_method="encoder",
        vary_view_joint_train=False,
        use_rgb_shortcut=False,
    )

    renderer = GaussianRenderer()

    loss = inpaint_aegaussian_loss(args.project_gaussian_mode)

    mse_loss_func = F.mse_loss
    perceptual_loss_func = PerceptualLoss(loss_type="zhang")
    for param in perceptual_loss_func.parameters():
        param.requires_grad = False
    loss_func_dict = {}
    loss_func_dict["mse"] = mse_loss_func
    loss_func_dict["perceptual"] = perceptual_loss_func

    dataset = DL3DVDataset(
        root_dir=args.data_path,
        mode="TRAIN",
        input_image_num=args.image_num_per_batch,
        input_image_res=args.input_image_res,
        output_image_num=args.output_image_num,
        output_image_res=args.output_image_res,
        centralized_cropping=args.centralized_cropping,
        relative_cam_pose=False,
        training_epochs=args.epochs,
        world_size=args.world_size,
        batch_size_per_gpu=args.batch_size_per_gpu,
        num_dataloader_workers=args.num_workers,
        gpus_per_node=(args.world_size if args.interactive_session else 1),
        no_dataset_schedule=False,
        produce_masked_result=True,
        mask_cache_path=args.mask_cache_path,
        clip_length=args.clip_len,
        depth_cache_path=None,
        do_resize=False,
        mask_mode=args.mask_mode,
        mask_prob=args.mask_prob,
        stereo_depth_cache_path=args.stereo_depth_cache_path,
        mask_multiple_objects=args.mask_multiple_objects,
        use_tri_value_mask=False,
    )

    mvencoder = mvencoder.cuda()
    extra_token_embed = extra_token_embed.cuda()
    gaudecoder = gaudecoder.cuda()
    gauupsampler = gauupsampler.cuda()
    perceptual_loss_func = perceptual_loss_func.cuda()
    renderer = renderer.cuda()

    mvencoder = dist_helper.get_parallel_model(mvencoder, args.gpu)
    if len(list(extra_token_embed.parameters())) > 0:
        extra_token_embed = dist_helper.get_parallel_model(extra_token_embed, args.gpu)
    gaudecoder = dist_helper.get_parallel_model(gaudecoder, args.gpu)
    gauupsampler = dist_helper.get_parallel_model(gauupsampler, args.gpu)

    params_groups = optim_utils.get_params_groups(
        get_optimizer_args(args.lr),
        gauencoder=mvencoder,
        extra_token_embed=extra_token_embed,
        gaudecoder=gaudecoder,
        gauupsampler=gauupsampler,
    )
    optimizer = torch.optim.AdamW(params_groups, betas=(0.9, 0.95))

    if args.pretrain_ckpt_path is not None:
        custom_restart_from_checkpoint(
            args.pretrain_ckpt_path,
            mvencoder=mvencoder,
            extra_token_embed=extra_token_embed,
            gaudecoder=gaudecoder,
            gauupsampler=gauupsampler,
        )

    start_epoch = 0
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)

        data_loader = utils.RepeatedDataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )
    else:
        sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
    logger.info(f"Data loaded: there are {len(dataset)} 3D models.")

    # ============ init schedulers ... ============
    lr_schedule = optim_utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_iters=args.warmup_iters,
        relu_warmup=False,
    )
    logger.info("Loss, optimizer and schedulers ready.")

    start_time = time.time()
    logger.info("Starting LRM training !")
    for epoch in range(start_epoch, args.epochs):
        if torch.distributed.is_initialized():
            data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(
            mvencoder,
            extra_token_embed,
            gaudecoder,
            gauupsampler,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            data_loader,
            optimizer,
            lr_schedule,
            epoch,
            args,
            tb_writer=tb_writer,
        )

        # ============ writing logs ... ============
        save_dict = {
            "mvencoder": mvencoder.state_dict(),
            "extra_token_embed": extra_token_embed.state_dict(),
            "gaudecoder": gaudecoder.state_dict(),
            "gauupsampler": gauupsampler.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }

        should_save_checkpoint = epoch == args.epochs - 1 or (
            args.saveckp_epoch_freq > 0
            and (epoch + 1) % args.saveckp_epoch_freq == 0
        )
        if should_save_checkpoint:
            backup_ckp_epoch = (
                epoch
                if args.backup_ckp_epoch_freq > 0
                and (epoch + 1) % args.backup_ckp_epoch_freq == 0
                else -1
            )
            checkpoint_utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoint_dir, "last.pth"),
                backup_ckp_epoch=backup_ckp_epoch,
                max_to_backup=2,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with pathmgr.open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    return


def train_one_epoch(
    mvencoder,
    extra_token_embed,
    gaudecoder,
    gauupsampler,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    data_loader,
    optimizer,
    lr_schedule,
    epoch,
    args,
    tb_writer=None,
):

    metric_logger = utils.MetricLogger(
        delimiter="  ",
        logger=logger,
        tb_writer=tb_writer,
        epoch=epoch,
    )
    header = "Exp {}; Epoch: [{}/{}]".format(args.exp_name, epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        (
            preds,
            gts,
            gaussians,
            losses,
        ) = train_one_iteration(
            batch,
            mvencoder,
            extra_token_embed,
            gaudecoder,
            gauupsampler,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            args,
            auto_cast_dtype=torch.bfloat16,
        )
        optim_utils.clip_gradients(mvencoder, 1.0, adaptive=False)
        optim_utils.clip_gradients(extra_token_embed, 1.0, adaptive=False)
        optim_utils.clip_gradients(gaudecoder, 1.0, adaptive=False)
        optim_utils.clip_gradients(gauupsampler, 1.0, adaptive=False)
        optimizer.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(**losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        is_major_degrade = False
        if "rgb_mse" in losses:
            is_major_degrade = is_major_degrade | (
                losses["rgb_mse"] / metric_logger.meters["rgb_mse"].global_avg > 1.5
            )
        if "rgb_perceptual" in losses:
            is_major_degrade = is_major_degrade | (
                losses["rgb_perceptual"] / metric_logger.meters["rgb_perceptual"].global_avg > 1.5
            )

        optimizer.zero_grad()

        should_save_images = (
            args.saveimg_iter_freq > 0 and it % args.saveimg_iter_freq == 0
        )
        if is_major_degrade or should_save_images:
            with open(os.path.join(args.image_dir, f"train_image_names_rank_{args.global_rank}.txt"), "a") as img_id_f:
                num_inputs  = len(batch["rgb_names_input"])
                num_outputs = len(batch["rgb_names_output"])
                img_id_f.write(f"[Epoch {epoch}] [Global Iter {it}]\n")
                img_id_f.write("Input:\n")
                for sample_ind in range(args.batch_size_per_gpu):
                    for io_ind in range(num_inputs):
                        img_id_f.write("\t{}\n".format(batch["rgb_names_input"][io_ind][sample_ind]))
                img_id_f.write("Output:\n")
                for sample_ind in range(args.batch_size_per_gpu):
                    for io_ind in range(num_outputs):
                        img_id_f.write("\t{}\n".format(batch["rgb_names_output"][io_ind][sample_ind]))
                img_id_f.write("\n")

        if should_save_images:
            if utils.is_main_process():
                input_image_out = os.path.join(args.image_dir, f"{it:06d}_inputs.png")
                inputs = batch["rgb_input"].detach().cpu()
                image_io.save_image(inputs, input_image_out)

                if "depth" in gts:
                    depth_mask = batch["depth_masks_output"].detach().cpu()
                    depth_gt = batch["depth_output"].detach().cpu()
                    depth_mask = depth_mask[:, : args.image_num_per_batch]
                    depth_gt = depth_gt[:, : args.image_num_per_batch]
                    depth_gt = depth_gt.reshape(-1)[depth_mask.reshape(-1) > 0]
                    depth_min = depth_gt.min().item()
                    depth_max = depth_gt.max().item()
                else:
                    depth_mask = None
                    depth_min = None
                    depth_max = None

                for key in gts.keys():
                    gt = gts[key].detach().cpu()
                    gts_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_gts_{key}.png"
                    )
                    if key == "depth":
                        image_io.save_depth(
                            gt, depth_mask, gts_image_out, depth_min, depth_max
                        )
                    else:
                        image_io.save_image(gt, gts_image_out, is_gamma=False)


                for key in preds.keys():
                    if key == "gradient" or key == "points" or key == "xyz":
                        continue  # to be finished.
                    pred = preds[key]
                    preds_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_preds_{key}.png"
                    )
                    if key == "depth" and depth_mask is not None:
                        image_io.save_depth(
                            pred, depth_mask, preds_image_out, depth_min, depth_max
                        )
                    else:
                        image_io.save_image(pred.clamp(-1, 1), preds_image_out, is_gamma=False)

                cams = batch["cameras_output"].detach().cpu().numpy()
                with pathmgr.open(
                    os.path.join(args.image_dir, f"{it:06d}_cam.npy"), "wb"
                ) as fp:
                    np.save(fp, cams)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def encode_plucker_rays(mvencoder, instance_mask, rgb, rays_o, rays_d, batch_size, image_num, height, width, auto_cast_dtype, args):
    if rgb is not None:
        rgb_enc = rgb.reshape(batch_size * image_num, -1, height, width)
        rgb_enc = rgb_enc.to(device=args.gpu, dtype=auto_cast_dtype)
        if mvencoder.module.in_chans == 4:
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


def train_one_iteration(
    batch,
    mvencoder,
    extra_token_embed,
    gaudecoder,
    gauupsampler,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    args,
    auto_cast_dtype,
):

    assert auto_cast_dtype == torch.bfloat16, "Got {}".format(auto_cast_dtype)

    images = batch["rgb_input"]
    cams_output = batch["cameras_output"]
    cams_input = batch["cameras_input"]
    cropping_output = batch.get("cropping_output", None)

    batch_size, image_num, ch, height, width = images.shape
    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):

        rgb_input    = batch["rgb_input"].to(device=args.gpu)
        instance_masks_input = batch["instance_masks_input"].to(device=args.gpu, dtype=auto_cast_dtype)
        rays_o_input = batch["rays_o_input"].to(device=args.gpu).permute(0, 1, 4, 2, 3)
        rays_d_input = batch["rays_d_input"].to(device=args.gpu).permute(0, 1, 4, 2, 3)
        rays_t_input = None
        if "depth_input" in batch.keys():
            depth_input = batch["depth_inputs"].unsqueeze(2).to(device=args.gpu)
            rgb_input = torch.cat([rgb_input,depth_input], 2)

        tokens = encode_plucker_rays(
            mvencoder, instance_masks_input, rgb_input, rays_o_input, rays_d_input,
            batch_size, image_num, height, width, auto_cast_dtype, args)
            
        initial_tokens = tokens.clone()

        tokens = extra_token_embed(tokens)

        tokens = gaudecoder(tokens)

        cams_input = cams_input.to(device=args.gpu, dtype=auto_cast_dtype)
        # NOTE_YJQ: All None because I kind of don't want to change dynamicaegaussianupsampler
        rays_o_input, rays_d_input, rays_t_input, rays_t_output, rays_t_un_output = None, None, None, None, None
        gaussians = gauupsampler(
            tokens, initial_tokens, cams_input, images,
            rays_o_input, rays_d_input,
            rays_t_input, rays_t_output, 
            rays_t_un_output, mode=None)

    for key in gaussians.keys():
        if isinstance(gaussians[key], torch.Tensor):
            gaussians[key] = gaussians[key].to(dtype=torch.float32)
        elif isinstance(gaussians[key], dict):
            gaussians[key] = {
                k: v.to(dtype=torch.float32) if isinstance(v, torch.Tensor) else v
            for k,v in gaussians[key].items()}
        else:
            assert gaussians[key] is None, "Got unknown type {}".format(type(gaussians[key]))

    gts = {}
    keys = ["rgb", "mask"]
    for i, key in enumerate(keys):
        gts[key] = batch[f"{key}_output"]
        gts[key] = gts[key].to(device=args.gpu)

    mask_gt = batch["mask_input"]
    mask_gt = mask_gt.to(args.gpu)

    if "depth_output" in batch:
        depth_gt = batch["depth_output"]
        depth_masks = batch["depth_masks_output"]
        depth_gt = depth_gt.to(args.gpu)
        depth_masks = depth_masks.to(args.gpu)
    else:
        depth_gt = None
        depth_masks = None

    # from depth to xyz
    rays_o = batch["rays_o_input"].permute(0, 1, 4, 2, 3).to(device=args.gpu)
    rays_d_un = batch["rays_d_un_input"].permute(0, 1, 4, 2, 3).to(device=args.gpu)

    # Add geometry loss
    cams_output = cams_output.to(device=args.gpu)
    preds, losses, gaussians = loss.forward_and_backward(
        renderer,
        gaussians["depth"],
        gaussians["rgb"],
        gaussians["opacity"],
        gaussians["scale"],
        gaussians["rotation"],
        instance_masks_input,
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
        filter_point=False,
    )

    return preds, gts, gaussians, losses


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    train_lrm(args)
