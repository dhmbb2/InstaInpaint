import math
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from ..misc.io_helper import pathmgr
from .common import (
    center_crop_frame,
    decode_rgb_bytes,
    load_dl3dv_cameras,
    normalize_camera_poses,
    resize_to_height,
    stack_items,
)
from .utils import generate_random_ellipse_mask, load_one_frame


class DL3DVDataset(Dataset):
    """DL3DV training loader for the open-source InstaInpaint training script."""

    SUPPORTED_CLIP_LENGTHS = {5, 15}
    SUPPORTED_MASK_MODES = {"instance", "random", "3dconsistent"}
    MAX_OBJECTS_PER_IMAGE = 8

    def __init__(
        self,
        root_dir,
        mode,
        input_image_res=(512, 512),
        input_image_num=4,
        output_image_res=(512, 512),
        output_image_num=8,
        centralized_cropping=True,
        clip_length=15,
        training_epochs=-1,
        world_size=-1,
        batch_size_per_gpu=-1,
        num_dataloader_workers=-1,
        gpus_per_node=-1,
        mask_cache_path=None,
        mask_mode="instance",
        mask_prob=None,
        mask_multiple_objects=False,
        stereo_depth_cache_path=None,
        produce_masked_result=True,
        model_lists=None,
        start_id=0,
        end_id=-1,
        **unused,
    ):
        super().__init__()
        if mode.upper() != "TRAIN":
            raise ValueError("DL3DVDataset in this package only supports TRAIN mode.")
        if input_image_num != 4:
            raise ValueError("DL3DVDataset currently supports exactly 4 input views.")
        if clip_length not in self.SUPPORTED_CLIP_LENGTHS:
            raise ValueError(f"clip_length must be one of {sorted(self.SUPPORTED_CLIP_LENGTHS)}.")
        if mask_cache_path is None:
            raise ValueError("mask_cache_path is required for DL3DV training.")

        self.root_dir = Path(root_dir)
        self.mode = mode.upper()
        self.input_image_res = tuple(input_image_res)
        self.input_image_num = input_image_num
        self.output_image_res = tuple(output_image_res)
        self.output_image_num = output_image_num
        self.centralized_cropping = centralized_cropping
        self.clip_length = clip_length
        self.mask_path = mask_cache_path
        self.mask_modes = self._parse_mask_modes(mask_mode, mask_prob)
        self.stereo_depth_cache_path = stereo_depth_cache_path
        self.mask_multiple_objects = mask_multiple_objects
        self.produce_masked_result = produce_masked_result

        if any(mode == "3dconsistent" for mode, _ in self.mask_modes):
            if self.stereo_depth_cache_path is None:
                raise ValueError("stereo_depth_cache_path is required when using 3dconsistent masks.")
            if self.clip_length != 15:
                raise ValueError("3dconsistent masks only support clip_length=15.")

        self.training_epochs = training_epochs
        self.world_size = world_size
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_dataloader_workers = num_dataloader_workers
        self.gpus_per_node = gpus_per_node

        self.models = self._load_model_list(model_lists, start_id, end_id)
        self._filter_valid_models()
        self.total_training_steps = self._estimate_total_training_steps()

    def __len__(self):
        return len(self.models)

    @staticmethod
    def _read_image_bytes(path):
        with pathmgr.open(path, "rb") as f:
            return f.read()

    @classmethod
    def _parse_mask_modes(cls, mask_mode, mask_prob):
        modes = mask_mode.split("+")
        unknown_modes = set(modes) - cls.SUPPORTED_MASK_MODES
        if unknown_modes:
            raise ValueError(f"Unsupported mask modes: {sorted(unknown_modes)}.")
        if len(modes) > 1:
            if mask_prob is None or len(mask_prob) != len(modes):
                raise ValueError("mask_prob must match mask_mode when multiple mask modes are used.")
            probs = np.asarray(mask_prob, dtype=np.float64)
            probs = probs / probs.sum()
        else:
            probs = None
        return tuple(zip(modes, probs)) if probs is not None else tuple((mode, None) for mode in modes)

    def _load_model_list(self, model_lists, start_id, end_id):
        if model_lists is None:
            model_list_path = self.root_dir / "train.txt"
            with pathmgr.open(model_list_path, "r") as f:
                models = [line.strip() for line in f if line.strip()]
        else:
            models = list(model_lists)

        start_id = min(max(start_id, 0), len(models) - 1)
        if end_id > start_id:
            return models[start_id : min(end_id, len(models))]
        return models[start_id:]

    def _estimate_total_training_steps(self):
        if self.training_epochs <= 0 or self.world_size <= 0 or self.batch_size_per_gpu <= 0:
            return -1
        steps_per_epoch = len(self) // (self.world_size * self.batch_size_per_gpu)
        return self.training_epochs * steps_per_epoch

    def _filter_valid_models(self):
        with h5py.File(self.mask_path, "r") as h5f:
            available_models = {key.split("/")[-1] for key in h5f.keys()}
            valid_models = []
            for model in self.models:
                model_id = Path(model).name
                if model_id not in available_models:
                    print(f"Model {model_id} not found in the mask cache")
                    continue

                frame_validate = h5f[f"{model_id}/frame_validate"][()]
                frame_validate = self._trim_to_clip_length(frame_validate)
                if not frame_validate.any():
                    print(f"Model {model_id} has no valid frames; removing it from the dataset")
                    continue
                valid_models.append(model)

        self.models = valid_models

    def _load_all_cameras(self, model_path):
        return load_dl3dv_cameras(model_path)

    def _load_images(self, model_path, frame_ids, image_names):
        image_dir = self._find_image_dir(model_path)
        images = []
        paths = []
        for frame_id in frame_ids:
            path = model_path / image_dir / image_names[frame_id]
            image = decode_rgb_bytes(self._read_image_bytes(path))
            image = resize_to_height(image, self.input_image_res[0])
            images.append(image)
            paths.append(path)
        return np.stack(images, axis=0), paths

    @staticmethod
    def _find_image_dir(model_path):
        for path in sorted(model_path.iterdir()):
            if path.is_dir():
                return path.name
        raise FileNotFoundError(f"No image directory found under {model_path}")

    def _load_masks(self, model_id):
        with h5py.File(self.mask_path, "r") as h5f:
            masks = h5f[f"{model_id}/masks"][()]
            frame_validate = h5f[f"{model_id}/frame_validate"][()]
        return self._trim_to_clip_length(masks), self._trim_to_clip_length(frame_validate)

    def _trim_to_clip_length(self, array):
        length = math.floor(len(array) / self.clip_length) * self.clip_length
        return array[:length]

    def _choose_mask_mode(self):
        if len(self.mask_modes) == 1:
            return self.mask_modes[0][0]
        modes, probs = zip(*self.mask_modes)
        return np.random.choice(modes, p=probs)

    def _choose_clip_index(self, frame_validate, mask_mode):
        if mask_mode == "instance":
            valid_indices = [idx for idx, is_valid in enumerate(frame_validate) if is_valid]
            return int(np.random.choice(valid_indices)) // self.clip_length
        return int(np.random.choice(len(frame_validate))) // self.clip_length

    def _clip_image_names(self, image_names, clip_index):
        start = clip_index * self.clip_length
        end = start + self.clip_length
        return image_names[start:end]

    def _clip_camera_poses(self, camera_poses, clip_index):
        start = clip_index * self.clip_length
        if self.clip_length == 5 and start + 2 * self.clip_length < len(camera_poses):
            clip = camera_poses[start : start + 2 * self.clip_length]
        else:
            clip = camera_poses[start : start + self.clip_length]
        return normalize_camera_poses(clip)[: self.clip_length]

    def _sample_view_ids(self):
        if self.clip_length == 5:
            output_id = np.random.choice([1, 3])
            input_ids = [2, 0, 1, 3, 4]
            input_ids.remove(output_id)
            return np.asarray(input_ids), np.asarray([output_id]), None

        unit = self.clip_length // 3
        input_ids = np.asarray([0, unit - 1, 2 * unit - 1, 3 * unit - 1])
        selected_idx = np.random.choice([1, 2])
        inpaint_target = input_ids[selected_idx]
        input_ids = np.concatenate([[inpaint_target], np.delete(input_ids, selected_idx)])

        output_ids = np.concatenate(
            [
                np.arange(1, unit - 1),
                np.arange(unit, 2 * unit - 1),
                np.arange(2 * unit, 3 * unit - 1),
            ]
        )
        np.random.shuffle(output_ids)
        return input_ids, output_ids[: self.output_image_num], selected_idx

    def _decode_instance_masks(self, instance_mask):
        bits = 2 ** np.arange(self.MAX_OBJECTS_PER_IMAGE)
        instance_mask = (instance_mask[..., None].astype(np.uint8) & bits) > 0
        return instance_mask.astype(np.uint8).transpose(0, 3, 1, 2)

    @staticmethod
    def _dilate_mask(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.dilate(mask, kernel, iterations=1)

    def _resize_mask_to_input_height(self, mask):
        height, width = mask.shape
        new_width = int(width * (self.input_image_res[0] / height))
        mask = cv2.resize(mask, (new_width, self.input_image_res[0]), interpolation=cv2.INTER_NEAREST)
        return self._dilate_mask(mask)

    def _random_mask(self, count):
        mask = np.zeros(self.input_image_res, dtype=np.uint8)
        for _ in range(count):
            width = np.random.randint(self.input_image_res[1] // 6, self.input_image_res[1] // 4)
            height = np.random.randint(self.input_image_res[0] // 6, self.input_image_res[0] // 4)
            x = np.random.randint(0, self.input_image_res[1] - width)
            y = np.random.randint(0, self.input_image_res[0] - height)
            mask[y : y + height, x : x + width] = 1
        return mask

    def _build_instance_masks(self, raw_masks, clip_index, input_ids, crop):
        clip_start = clip_index * self.clip_length
        masks = self._decode_instance_masks(raw_masks[clip_start : clip_start + self.clip_length])
        if masks[0].sum() == 0:
            raise ValueError(f"Expected at least one object in clip {clip_start}")

        valid_objects = [idx for idx in range(masks.shape[1]) if masks[0, idx].sum() > 0]
        object_count = self._sample_object_count(len(valid_objects))
        selected_objects = random.sample(valid_objects, object_count)

        masks = np.any(masks[input_ids][:, selected_objects], axis=1).astype(np.uint8)
        return [self._crop_mask(self._resize_mask_to_input_height(mask), crop) for mask in masks]

    def _build_random_masks(self, input_count, object_count):
        mask = self._random_mask(object_count)
        return [mask.copy() for _ in range(input_count)]

    def _build_3d_consistent_masks(
        self,
        model_id,
        clip_index,
        selected_idx,
        rays_o_input,
        rays_d_un_input,
        cameras_input,
        mask_count,
    ):
        height, width = self.input_image_res
        with h5py.File(self.stereo_depth_cache_path, "r") as h5f:
            depth = h5f[f"{model_id}/{clip_index}"][()]

        lrm_depth = depth[selected_idx - 1]
        world_coord = rays_o_input[0] + lrm_depth[..., None] * rays_d_un_input[0]
        random_mask = generate_random_ellipse_mask(
            img_size=(height, width),
            angle_range=(0, 180),
            mask_num=mask_count,
        )
        world_coord = world_coord[random_mask]

        cameras_input = np.stack(cameras_input, axis=0)
        fov_x = cameras_input[0][16]
        fov_y = cameras_input[0][17]
        f_x = width / (2 * math.tan(fov_x / 2))
        f_y = height / (2 * math.tan(fov_y / 2))
        intrinsic = np.asarray(
            [
                f_x,
                0,
                width / 2,
                0,
                f_y,
                height / 2,
                0,
                0,
                1,
            ],
            dtype=np.float32,
        ).reshape(3, 3)

        cameras = cameras_input[..., :16].reshape(-1, 4, 4)
        cameras[:, :3, 1:3] *= -1
        homog_world = np.pad(world_coord, ((0, 0), (0, 1)), mode="constant", constant_values=1)

        instance_masks = []
        for camera in cameras:
            camera_coord = (np.linalg.inv(camera) @ homog_world.T).T
            camera_coord[:, 2:3] = np.clip(camera_coord[:, 2:3], 1e-3, 100)
            points_2d = (intrinsic[None] @ (camera_coord[:, :3] / camera_coord[:, 2:3]).T).T
            pixels = np.round(points_2d[:, :2, 0]).astype(np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            x_coords = pixels[:, 0]
            y_coords = pixels[:, 1]
            valid = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
            np.add.at(mask, (y_coords[valid], x_coords[valid]), 1)
            mask[mask > 1] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            instance_masks.append(cv2.dilate(mask, kernel, iterations=2))

        return instance_masks

    def _sample_object_count(self, valid_object_count):
        if not self.mask_multiple_objects:
            return 1
        return min(np.random.randint(1, 5), valid_object_count)

    @staticmethod
    def _crop_mask(mask, crop):
        _, _, width_start, width_end = crop
        return mask[:, width_start:width_end]

    def _load_frame(self, image, camera_pose, fov, image_res):
        return load_one_frame(
            fov=fov,
            im=image,
            c2w=camera_pose,
            image_res=image_res,
            hdr_to_ldr=False,
            resize=False,
        )

    def _make_input_sample(self, image, camera_pose, fov, temporal_id):
        image, rays_o, rays_d, camera, _, rays_d_un = self._load_frame(
            image, camera_pose, fov, self.input_image_res
        )
        crop = (0, image.shape[1], 0, image.shape[2])
        if self.centralized_cropping:
            image, rays_o, rays_d, rays_d_un, camera, crop = center_crop_frame(
                image, rays_o, rays_d, rays_d_un, camera, self.input_image_res
            )

        denom = max(self.input_image_num - 1, 1)
        rays_t = np.full((*rays_o.shape[:2], 1), temporal_id / denom, dtype=np.float32)
        return image, rays_o, rays_d, rays_d_un, camera, rays_t, temporal_id, crop

    def _make_output_sample(self, image, camera_pose, fov, output_id, input_ids):
        image, rays_o, rays_d, camera, _, rays_d_un = self._load_frame(
            image, camera_pose, fov, self.output_image_res
        )
        crop = None
        if self.centralized_cropping:
            image, rays_o, rays_d, rays_d_un, camera, crop = center_crop_frame(
                image, rays_o, rays_d, rays_d_un, camera, self.output_image_res
            )

        min_id, max_id = input_ids.min(), input_ids.max()
        rays_t = np.full((*rays_o.shape[:2], 1), (output_id - min_id) / (max_id - min_id), dtype=np.float32)
        rays_t_un = output_id - min_id
        return image, rays_o, rays_d, rays_d_un, camera, rays_t, rays_t_un, crop

    @staticmethod
    def _stack(items, dtype=None):
        return stack_items(items, dtype)

    def __getitem__(self, idx):
        model = self.models[idx % len(self.models)]
        model_id = Path(model).name
        model_path = self.root_dir / model

        camera_poses, fov, image_names = self._load_all_cameras(model_path)
        raw_masks, frame_validate = self._load_masks(model_id)
        mask_mode = self._choose_mask_mode()
        clip_index = self._choose_clip_index(frame_validate, mask_mode)
        image_names = self._clip_image_names(image_names, clip_index)
        camera_poses = self._clip_camera_poses(camera_poses, clip_index)
        input_ids, output_ids, selected_idx = self._sample_view_ids()

        mask_count = np.random.randint(1, 5) if self.mask_multiple_objects else 1

        input_frames, input_paths = self._load_images(model_path, input_ids, image_names)
        output_frames, output_paths = self._load_images(model_path, output_ids, image_names)

        inputs = [self._make_input_sample(input_frames[n], camera_poses[input_ids[n]], fov, n) for n in range(len(input_ids))]
        images_input, rays_o_input, rays_d_input, rays_d_un_input, cameras_input, rays_t_input, rays_t_un_input, crops = zip(*inputs)

        if mask_mode == "instance":
            instance_masks_input = self._build_instance_masks(raw_masks, clip_index, input_ids, crops[0])
        elif mask_mode == "random":
            instance_masks_input = self._build_random_masks(len(input_ids), mask_count)
        elif mask_mode == "3dconsistent":
            instance_masks_input = self._build_3d_consistent_masks(
                model_id,
                clip_index,
                selected_idx,
                rays_o_input,
                rays_d_un_input,
                cameras_input,
                mask_count,
            )
        else:
            raise NotImplementedError(mask_mode)

        images_input = list(images_input)
        if self.produce_masked_result:
            for i in range(1, len(images_input)):
                images_input[i] = images_input[i] * (1 - instance_masks_input[i])

        output_samples = [
            self._make_output_sample(output_frames[n], camera_poses[output_ids[n]], fov, output_ids[n], input_ids)
            for n in range(len(output_ids))
        ]
        (
            images_output,
            rays_o_output,
            rays_d_output,
            rays_d_un_output,
            cameras_output,
            rays_t_output,
            rays_t_un_output,
            output_crops,
        ) = zip(*output_samples)

        images_input = self._stack(images_input, np.float32)
        images_output = self._stack(images_output, np.float32)
        cameras_output = self._stack(cameras_output, np.float32)

        out = {
            "name": model_id,
            "rgb_input": images_input,
            "mask_input": np.ones_like(images_input)[:, 0:1],
            "rgb_names_input": [str(path) for path in input_paths],
            "rays_o_input": self._stack(rays_o_input, np.float32),
            "rays_d_input": self._stack(rays_d_input, np.float32),
            "rays_d_un_input": self._stack(rays_d_un_input, np.float32),
            "cameras_input": self._stack(cameras_input, np.float32),
            "rays_t_input": self._stack(rays_t_input, np.float32),
            "rays_t_un_input": self._stack(rays_t_un_input, np.int32),
            "instance_masks_input": self._stack(instance_masks_input, np.float32),
            "camera_poses": camera_poses.astype(np.float32),
            "rgb_output": images_output,
            "mask_output": np.ones_like(images_output)[:, 0:1],
            "rgb_names_output": [str(path) for path in output_paths],
            "rays_o_output": self._stack(rays_o_output, np.float32),
            "rays_d_output": self._stack(rays_d_output, np.float32),
            "cameras_output": cameras_output,
            "rays_t_output": self._stack(rays_t_output, np.float32),
            "rays_t_un_output": self._stack(rays_t_un_output, np.int32),
            "fov": [
                cameras_output[..., 17] / np.pi * 180,
                cameras_output[..., 16] / np.pi * 180,
            ],
        }

        if self.centralized_cropping:
            out["cropping_output"] = np.asarray(
                [
                    [
                        self.output_image_res[0],
                        self.output_image_res[1],
                        0,
                        self.output_image_res[0],
                        0,
                        self.output_image_res[1],
                    ]
                    for _ in output_crops
                ],
                dtype=np.int32,
            )

        return out
