import os
import sys
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import math
import json
import torch
import itertools
from pathlib import Path
from multiprocessing import Process, Manager
import torch.distributed as dist

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py

import instainpaint.misc.logging as logging

from instainpaint.misc.env_utils import init_distributed_mode
from instainpaint.misc import checkpoint as checkpoint_utils
from instainpaint.models.multiview_encoder import mvencoder_base, ExtraTokenEmbed
from instainpaint.models.aegaussian_decoder import AeGaussianTransformer
from instainpaint.models.gaussian_decoder import GaussianMlpUpsampler
import instainpaint.misc.utils as utils
from instainpaint.data_loader.common import (
    center_crop_frame,
    decode_rgb_path,
    load_dl3dv_cameras,
    normalize_camera_poses,
    resize_to_height,
)
from instainpaint.data_loader.utils import load_one_frame
logger = logging.get_logger(__name__)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dl3dv_960"
DEFAULT_DEPTH_CACHE_PATH = PROJECT_ROOT / "data" / "depth_cache.h5"
DEFAULT_MASK_CACHE_PATH = PROJECT_ROOT / "data" / "mask_cache.h5"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "pretraining_ckpt.pth"

class DL3DVDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
        model_lists=None,
        input_image_res=(256, 256),
        input_image_num=2,
        output_image_res=(256, 256),
        output_image_num=1,
        centralized_cropping=True,
        relative_cam_pose=False,
        clip_length=5,
        start_id=0,
        end_id=-1,
        produce_masked_result=True,
        num_max_obj_per_image=8,
        mask_cache_path=None,
        depth_cache_path=False,
        do_resize=True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.input_image_res = input_image_res
        self.input_image_num = input_image_num
        self.output_image_res = output_image_res
        self.output_image_num = output_image_num
        self.clip_length = clip_length
        self.depth_cache_path = depth_cache_path
        self.mask_cache_path = mask_cache_path
        self.do_resize = do_resize

        if model_lists is None:
            model_list = "train.txt" if mode.upper() == "TRAIN" else "test.txt"
            model_list = self.root_dir / model_list

        with open(model_list, "r") as f:
            models = [line.strip() for line in f.readlines()]
            start_id = min(max(start_id, 0), len(models) - 1)
            if end_id > start_id:
                end_id = min(end_id, len(models))
                models = models[start_id:end_id]
            else:
                models = models[start_id:]

        self.models = models
        self.input_image_res = input_image_res
        self.input_image_num = input_image_num
        self.output_image_res = output_image_res
        self.output_image_num = output_image_num 
        self.centralized_cropping = centralized_cropping
        self.relative_cam_pose = relative_cam_pose
        self.mask_path = mask_cache_path
        self.produce_masked_result = produce_masked_result
        self.num_max_obj_per_image = num_max_obj_per_image

        self.check_model_validation()

        self.training_pairs = self.prepare_pairs()

    def __len__(self):
        return len(self.training_pairs)

    def _camera_pose_normalizion(self, camera_poses):
        """
        camera_poses: c2w np.array in (B, 4, 4)

        Note: We can do this in data preprocessing, but not that a big issue now
        """
        return normalize_camera_poses(camera_poses)

    def _load_all_cameras(self, model_path):
        return load_dl3dv_cameras(model_path)

    def check_model_validation(self):
        def get_all_groups(h5_file):
            """获取 HDF5 文件中所有 Group 的路径"""
            groups = []

            def _collect_groups(name, obj):
                if isinstance(obj, h5py.Group):
                    groups.append(name)

            h5_file.visititems(_collect_groups)
            return groups
        with h5py.File(self.mask_path, "r") as h5f:
            models_name = [model.split("/")[-1] for model in self.models]
            groups = get_all_groups(h5f)
            new_models = []
            for model_id, model_name in zip(self.models, models_name):
                if model_name not in groups:
                    print(f"Model {model_name} not found in the mask cache")
                    continue
                frame_validate = h5f[f"{model_name}/frame_validate"][()]
                frame_validate = frame_validate[:math.floor(len(frame_validate) / self.clip_length) * self.clip_length]
                if not frame_validate.any():
                    print(f"Model {model_name} has no valid frame, remove it from the dataset")
                    continue
                new_models.append(model_id)
            self.models = new_models

    def _load_images(self, model_path, image_idx, image_names, do_resize):
        
        for sub_name in os.listdir(model_path):
            if os.path.isdir(os.path.join(model_path, sub_name)):
                img_dir_name = sub_name
                break
        
        paths = [Path(model_path) / img_dir_name / image_names[i] for i in image_idx]
        images, names = [], []
        for p in paths:
            image = decode_rgb_path(p)
            if do_resize:
                image = resize_to_height(image, self.input_image_res[0])
            images.append(image)
            names.append(p)
        return np.stack(images, axis=0), names

    def prepare_pairs(self):
        training_pair = []
        for model in tqdm(self.models):
            model_path = Path(self.root_dir) / model
            transform_path = Path(model_path) / "transforms.json"
            with open(transform_path, "r") as f:
                transform_data = json.load(f)
            model_total_len = len(transform_data["frames"])
            # each pair contains the model id and the 
            # first frame index of the clip of self.clip_length
            normed_len = math.floor(model_total_len / self.clip_length) * self.clip_length
            for i in range(0, normed_len, self.clip_length):
                training_pair.append((model, i))
        return training_pair

    def __getitem__(self, idx):
        model, frame_index = self.training_pairs[idx]
        clip_index = frame_index // self.clip_length
        model_id = model.split("/")[-1]
        model_path = Path(self.root_dir) / model
        camera_poses, fov, image_names = self._load_all_cameras(model_path)

        # split the whole sequence according to the clip length
        normed_length = math.floor(len(camera_poses) / self.clip_length) * self.clip_length
        image_names_l = [image_names[i:i+self.clip_length] 
            for i in range(0, len(image_names), self.clip_length)]
        
        image_names = image_names_l[clip_index]
        camera_poses = camera_poses[clip_index*self.clip_length:(clip_index+1)*self.clip_length]
        camera_poses = self._camera_pose_normalizion(camera_poses)

        unit_l = self.clip_length // 3
        input_ids = np.array([0, unit_l-1, 2*unit_l-1, 3*unit_l-1])

        input_pair1 = np.array([0, unit_l-1])
        input_pair2 = np.array([2*unit_l-1, 3*unit_l-1])

        input_image_num = 2
        
        (
            images_input_2,
            rays_o_input_2,
            rays_d_input_2,
            rays_d_un_input_2,
            cameras_input_2,
            rays_t_input_2,
            rays_t_un_input_2,
        ) = ([], [], [], [], [], [], [])


        for input_ids in [input_pair1, input_pair2]:
            (
                images_input,
                rays_o_input,
                rays_d_input,
                rays_d_un_input,
                cameras_input,
                rays_t_input,
                rays_t_un_input,
            ) = ([], [], [], [], [], [], [])
            input_frames, image_names_input = self._load_images(model_path, input_ids, image_names, self.do_resize)
            for n in range(input_image_num):
                image, rays_o, rays_d, camera, _, rays_d_un = load_one_frame(
                    fov=fov,
                    im=input_frames[n],
                    c2w=camera_poses[input_ids[n]],
                    image_res=self.input_image_res,
                    hdr_to_ldr=False,
                    resize=False,
                )

                if self.centralized_cropping:
                    image, rays_o, rays_d, rays_d_un, camera, _ = center_crop_frame(
                        image,
                        rays_o,
                        rays_d,
                        rays_d_un,
                        camera,
                        self.input_image_res,
                    )

                rays_t_un = n
                rays_t_ratio = rays_t_un / (input_image_num - 1)
                rays_t = np.ones((rays_o.shape[0], rays_o.shape[1], 1), dtype=np.float32) * rays_t_ratio

                images_input.append(image)
                rays_o_input.append(rays_o)
                rays_d_input.append(rays_d)
                rays_d_un_input.append(rays_d_un)
                rays_t_input.append(rays_t)
                rays_t_un_input.append(rays_t_un)
                cameras_input.append(camera)

            images_input = np.stack(images_input, axis=0)
            rays_o_input = np.stack(rays_o_input, axis=0)
            rays_d_input = np.stack(rays_d_input, axis=0)
            rays_d_un_input = np.stack(rays_d_un_input, axis=0)
            rays_t_input = np.stack(rays_t_input, axis=0)
            rays_t_un_input = np.stack(rays_t_un_input, axis=0)
            cameras_input = np.stack(cameras_input, axis=0)

            images_input_2.append(images_input)
            rays_o_input_2.append(rays_o_input)
            rays_d_input_2.append(rays_d_input)
            rays_d_un_input_2.append(rays_d_un_input)
            cameras_input_2.append(cameras_input)
            rays_t_input_2.append(rays_t_input)
            rays_t_un_input_2.append(rays_t_un_input)
        
        images_input_2 = np.stack(images_input_2, axis=0)
        rays_o_input_2 = np.stack(rays_o_input_2, axis=0)
        rays_d_input_2 = np.stack(rays_d_input_2, axis=0)
        rays_d_un_input_2 = np.stack(rays_d_un_input_2, axis=0)
        cameras_input_2 = np.stack(cameras_input_2, axis=0)
        rays_t_input_2 = np.stack(rays_t_input_2, axis=0)
        rays_t_un_input_2 = np.stack(rays_t_un_input_2, axis=0)

        out = {
            "model_id": model_id,
            "rgb_input": images_input_2,
            "mask_input": np.ones_like(images_input_2)[:, 0:1],
            "rays_o_input": rays_o_input_2,
            "rays_d_input": rays_d_input_2,
            "rays_d_un_input": rays_d_un_input_2,
            "cameras_input": cameras_input_2,
            "rays_t_input": rays_t_input_2.astype(np.float32),
            "rays_t_un_input": rays_t_un_input_2.astype(np.int32),
            "clip_index": clip_index,
        }
            
        return out 



def encode_plucker_rays(mvencoder, rgb, rays_o, rays_d, rays_t, batch_size, image_num, height, width, auto_cast_dtype, device):
    if rgb is not None:
        rgb_enc = rgb.reshape(batch_size * image_num, 3, height, width)
        rgb_enc = rgb_enc.to(device=device, dtype=auto_cast_dtype)
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
        plucker_rays.to(device=device, dtype=auto_cast_dtype),
    )
    tokens = tokens[:, 1:, :]

    token_num = tokens.shape[1]
    tokens = tokens.reshape(batch_size, image_num * token_num, -1)

    return tokens

def writer_process(filename, queue):
    """Writer process that write masks into h5py to avoid conflict"""
    with h5py.File(filename, 'r+') as f:
        while True:
            msg = queue.get()
            if msg is None:  # end
                break
            # clip index and stero depth of the quadrisection points
            model_ids, clip_idxs, depths = msg
            assert len(model_ids) == len(clip_idxs) == len(depths)
            for model_id, clip_idx, depth in zip(model_ids, clip_idxs, depths):
                if model_id not in f:
                    group = f.create_group(model_id)
                else:
                    group = f[model_id]
                clip_name = str(clip_idx)
                if clip_name not in group:
                    group.create_dataset(clip_name, data=depth, dtype=np.uint8)
                else:
                    print(f"Clip {clip_name} already exists in {model_id}, skipping")

def get_args_parser():
    parser = argparse.ArgumentParser("InstaInpaint stereo depth cache", add_help=False)

    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--depth_cache_path", type=str, default=str(DEFAULT_DEPTH_CACHE_PATH))
    parser.add_argument("--mask_cache_path", type=str, default=str(DEFAULT_MASK_CACHE_PATH))
    parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--interactive_session", action=argparse.BooleanOptionalAction, default=True)

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    init_distributed_mode(args)

    logger.info("*" * 80)
    logger.info(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    logger.info(args)
    logger.info("*" * 80)

    # initializing writer process in the main process
    if utils.is_main_process():
        logger.info("Initializing writer process")
        depth_cache_dir = os.path.dirname(args.depth_cache_path)
        if depth_cache_dir:
            os.makedirs(depth_cache_dir, exist_ok=True)
        if not os.path.exists(args.depth_cache_path):
            # create the h5 file
            with h5py.File(args.depth_cache_path, "w") as h5:
                pass
        writing_queue = Manager().Queue()
        writer = Process(target=writer_process, args=(args.depth_cache_path, writing_queue))
        writer.start()

    device = torch.device("cuda", args.gpu)
    dataset = DL3DVDataset(
        root_dir=args.data_root,
        mode="TRAIN",
        input_image_res=(512, 512),
        output_image_res=(512, 512),
        input_image_num=2,
        output_image_num=1,
        clip_length=15,
        start_id=0,
        end_id=-1,
        produce_masked_result=False,
        mask_cache_path=args.mask_cache_path,
        do_resize=False,
    )

    embed_dim = 1024
    mvencoder = mvencoder_base(
        type="plucker",
        with_bg=False,
        in_chans=3,
        depth=0,
        embed_dim=embed_dim,
        patch_size=8,
        use_pos_embed=False,
        emb_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        input_image_num=4,
    )
    extra_token_embed = ExtraTokenEmbed(
        embed_dim=embed_dim,
        input_image_num=4,
    )
    gaudecoder = AeGaussianTransformer(
        embed_dim=embed_dim,
        depth=24,
        attn_use_bias=False,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
    )
    gauupsampler = GaussianMlpUpsampler(
        mlp_dim=1024,
        token_dim=embed_dim, 
        patch_size=8,
        depth_bias=-4,
        norm_use_bias=False,
        norm_use_affine=False,
        use_weight_norm=True,
        input_image_num=4,
        color_space="rgb",
        decode_method="encoder",
    )
    mvencoder = mvencoder.to(device=device, dtype=torch.bfloat16).eval()
    for para in mvencoder.parameters():
        para.requires_grad = False
    extra_token_embed = extra_token_embed.to(device=device, dtype=torch.bfloat16).eval()
    for para in extra_token_embed.parameters():
        para.requires_grad = False
    gaudecoder = gaudecoder.to(device=device, dtype=torch.bfloat16).eval()
    for para in gaudecoder.parameters():
        para.requires_grad = False
    gauupsampler = gauupsampler.to(device=device, dtype=torch.bfloat16).eval()
    for para in gauupsampler.parameters():
        para.requires_grad = False

    to_restore = {"epoch": 0}
    checkpoint_utils.restart_from_checkpoint(
        args.checkpoint_path,
        run_variables=to_restore,
        mvencoder=mvencoder,
        extra_token_embed=extra_token_embed,
        gaudecoder=gaudecoder,
        gauupsampler=gauupsampler,
        load_weights_only=False,
    )

    if torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        raise RuntimeError("Distributed training is not initialized")
    
    if utils.is_main_process():
        pbar = tqdm(total=len(dataloader), desc="Training")

    with torch.no_grad():
        for ii, batch in enumerate(dataloader):
            batch_size, pair_num, image_num, ch, height, width = batch["rgb_input"].shape
            rgb_input    = batch["rgb_input"].flatten(0,1).to(device=device, dtype=torch.bfloat16)
            rays_o_input = batch["rays_o_input"].flatten(0,1).to(device=device, dtype=torch.bfloat16).permute(0, 1, 4, 2, 3)
            rays_d_input = batch["rays_d_input"].flatten(0,1).to(device=device, dtype=torch.bfloat16).permute(0, 1, 4, 2, 3)
            rays_t_input, rays_t_output, rays_t_un_output = None, None, None

            tokens = encode_plucker_rays(
                mvencoder, rgb_input, rays_o_input, rays_d_input, rays_t_input, 
                batch_size*pair_num, image_num, height, width, torch.bfloat16, device)

            tokens = extra_token_embed(tokens)

            tokens = gaudecoder(tokens)

            cams_input = batch["cameras_input"].flatten(0,1).to(device=device, dtype=torch.bfloat16)
            gaussians = gauupsampler(
                tokens, cams_input, rgb_input, 
                rays_o_input, rays_d_input, rays_t_input, rays_t_output, rays_t_un_output)

            for key in gaussians.keys():
                if isinstance(gaussians[key], torch.Tensor):
                    gaussians[key] = gaussians[key].to(dtype=torch.float32)
                elif isinstance(gaussians[key], dict):
                    gaussians[key] = {
                        k: v.to(dtype=torch.float32) if isinstance(v, torch.Tensor) else v
                    for k,v in gaussians[key].items()}
                else:
                    assert gaussians[key] is None, "Got unknown type {}".format(type(gaussians[key]))

            stereo_depth = gaussians["depth"]
            stereo_depth = stereo_depth.reshape(batch_size, pair_num, image_num, height, width)
            stereo_depth = stereo_depth.flatten(1, 2)
            stereo_depth_to_save = stereo_depth[:, [1, 2]]
            
            clip_indices = batch["clip_index"].to(device)

            gathered_depth = [torch.zeros_like(stereo_depth_to_save) for _ in range(args.world_size)]
            dist.all_gather(gathered_depth, stereo_depth_to_save)
            gathered_clip_indices = [torch.zeros_like(clip_indices) for _ in range(args.world_size)]
            dist.all_gather(gathered_clip_indices, clip_indices)
            gathered_model_ids = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_model_ids, batch["model_id"])

            if utils.is_main_process():
                all_depth = torch.cat(gathered_depth, dim=0).cpu().numpy()
                all_clip_indices = torch.cat(gathered_clip_indices, dim=0).cpu().numpy()
                all_model_ids = list(itertools.chain.from_iterable(gathered_model_ids))
                writing_queue.put((all_model_ids, all_clip_indices, all_depth))
                pbar.update(1)
    
        if utils.is_main_process():
            pbar.close()
            writing_queue.put(None)
            writer.join(timeout=30)
