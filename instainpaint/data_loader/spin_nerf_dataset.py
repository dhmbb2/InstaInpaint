import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from ..misc.io_helper import pathmgr
from .common import decode_rgb_path, normalize_camera_poses, resize_to_height, stack_items
from .utils import (
    load_one_frame,
    load_one_image,
    max_convex_hall_indices,
    project_to_plane,
    ptz_load,
)


class SpinNerfDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
        input_image_res=(512, 512),
        input_image_num=4,
        output_image_res=(512, 512),
        output_image_num=40,
        centralized_cropping=True,
        produce_masked_result=True,
        output_has_gt=False,
        cal_metric_mode=False,
        context_json=None,
        preinpaint_json=None,
        vis_center_file=None,
        normalize_mode="all",
        models=None,
        **unused,
    ):
        super().__init__()
        if mode != "TEST":
            raise ValueError("SpinNerfDataset in this package only supports TEST mode.")
        if normalize_mode not in {"all", "input"}:
            raise ValueError("normalize_mode must be 'all' or 'input'.")
        if input_image_res[0] % 8 != 0 or input_image_res[1] % 8 != 0:
            raise ValueError("input_image_res height and width must be multiples of 8.")

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.input_image_res = input_image_res
        self.input_image_num = input_image_num
        self.output_image_res = output_image_res
        self.output_image_num = output_image_num
        self.centralized_cropping = centralized_cropping
        self.produce_masked_result = produce_masked_result
        self.output_has_gt = output_has_gt
        self.cal_metric_mode = cal_metric_mode
        self.normalize_mode = normalize_mode

        model_names = sorted(os.listdir(root_dir)) if models is None else models
        self.models = [self.root_dir / model for model in model_names]
        self.pre_selected_contexts = self._load_json(context_json)
        self.pre_inpaint = self._load_preinpaint_json(preinpaint_json)
        self.vis_centers = self._load_json(vis_center_file)

    @staticmethod
    def _load_json(path):
        if path is None:
            return None
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_preinpaint_json(path):
        data = SpinNerfDataset._load_json(path)
        if data is None:
            return None
        base_dir = Path(path).resolve().parent
        return {
            model: image_path if os.path.isabs(image_path) else str(base_dir / image_path)
            for model, image_path in data.items()
        }

    def __len__(self):
        return len(self.models)

    @staticmethod
    def _decode_rgb(path):
        return decode_rgb_path(path)

    @staticmethod
    def _fov_from_ptz(data):
        width, height = data["image_size"][0], data["image_size"][1]
        fx, fy = data["intrinsics"][0], data["intrinsics"][1]
        return np.array([2 * np.arctan2(height / 2, fy), 2 * np.arctan2(width / 2, fx)])

    def _load_camera(self, camera_path):
        data = ptz_load(camera_path)
        pose = np.array(torch.cat([data["rotation"], data["translation"].unsqueeze(1)], dim=1))
        return pose, self._fov_from_ptz(data)

    def _load_cameras(self, model_path):
        camera_dir = model_path / "scenes" / "scene_0" / "camera"
        if not camera_dir.exists():
            raise FileNotFoundError(f"Camera folder not found: {camera_dir}")

        poses, fovs, names = [], [], []
        for camera_file in sorted(camera_dir.glob("*.ptz")):
            pose, fov = self._load_camera(camera_file)
            poses.append(pose)
            fovs.append(fov)
            names.append(camera_file.stem)

        poses = np.asarray(poses)
        if poses.shape[1] == 3:
            bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4), (poses.shape[0], 1, 1))
            poses = np.concatenate([poses, bottom], axis=1)
        poses = np.linalg.inv(poses)
        poses[:, :3, 1:3] *= -1
        return poses, np.asarray(fovs), names

    @staticmethod
    def _normalize_all(poses):
        locs = poses[:, :3, 3:4]
        mean = locs.mean(axis=0, keepdims=True)
        scale = np.abs(locs - mean).max()
        return normalize_camera_poses(poses), mean, scale

    @staticmethod
    def _normalize_from_inputs(poses, input_ids):
        input_locs = poses[input_ids, :3, 3:4]
        mean = input_locs.mean(axis=0, keepdims=True)
        scale = np.abs(input_locs - mean).max()
        poses = poses.copy()
        poses[:, :3, 3:4] = (poses[:, :3, 3:4] - mean) / scale
        return poses, mean, scale

    def _load_images_and_masks(self, model_path, names):
        scene_dir = model_path / "scenes" / "scene_0"
        images, masks, paths = [], [], []
        for name in names:
            image_path = scene_dir / "rgb" / f"{name}.png"
            mask_path = scene_dir / "instances" / f"{name}.ptz"
            image = self._decode_rgb(image_path)
            mask = ptz_load(mask_path).numpy().astype(np.uint8)

            image = resize_to_height(image, self.input_image_res[0])
            mask = resize_to_height(mask, self.input_image_res[0], interpolation=cv2.INTER_NEAREST)
            images.append(image)
            masks.append((mask > 0.5).astype(np.uint8))
            paths.append(image_path)
        return np.stack(images), np.stack(masks), paths

    @staticmethod
    def _read_split(path):
        with pathmgr.open(path, "r") as f:
            return [line.strip().split(",")[-1] for line in f]

    def _select_context_and_targets(self, model_path, poses, image_names):
        train = self._read_split(model_path / "splits" / "train.txt")
        valid = self._read_split(model_path / "splits" / "valid.txt")
        if self.input_image_num == 2:
            start = random.randint(0, len(train) - 8)
            context = [train[start], train[start + 8]]
        else:
            train_poses = np.stack([poses[image_names.index(name)] for name in train])
            positions = train_poses[:, :3, 3]
            normal = train_poses[:, :3, 2].mean(axis=0)
            projected = project_to_plane(positions, positions.mean(axis=0), normal)
            context = [train[i] for i in max_convex_hall_indices(projected)]
        return context, valid, train

    @staticmethod
    def _middle_pose(a, b):
        r1, t1 = a[:3, :3], a[:3, 3]
        r2, t2 = b[:3, :3], b[:3, 3]
        slerp = Slerp([0, 1], Rotation.from_quat([Rotation.from_matrix(r1).as_quat(), Rotation.from_matrix(r2).as_quat()]))
        pose = np.eye(4)
        pose[:3, :3] = slerp(0.5).as_matrix()
        pose[:3, 3] = 0.5 * (t1 + t2)
        return pose

    def _center_crop(self, image, rays_o, rays_d, rays_d_un, mask, width):
        start = (image.shape[2] - width) // 2
        end = start + width
        return (
            image[:, :, start:end],
            rays_o[:, start:end],
            rays_d[:, start:end],
            rays_d_un[:, start:end],
            mask[:, start:end],
        )

    def _input_sample(self, n, image, mask, pose, fov, preinpaint_path=None):
        image, rays_o, rays_d, camera, _, rays_d_un = load_one_frame(
            fov=fov,
            im=image,
            c2w=pose,
            image_res=self.input_image_res,
            hdr_to_ldr=False,
            resize=False,
        )
        if preinpaint_path is not None:
            image, _ = load_one_image(
                preinpaint_path,
                image_res=(image.shape[2], image.shape[1]),
                resize=True,
                normalize=True,
                hdr_to_ldr=False,
            )
        if self.centralized_cropping:
            image, rays_o, rays_d, rays_d_un, mask = self._center_crop(
                image, rays_o, rays_d, rays_d_un, mask, self.input_image_res[1]
            )
        denom = max(self.input_image_num - 1, 1)
        rays_t = np.full((*rays_o.shape[:2], 1), n / denom, dtype=np.float32)
        return image, rays_o, rays_d, rays_d_un, camera, rays_t, n, mask

    def _output_sample(self, n, image, mask, pose, fov, output_id, input_ids):
        image, rays_o, rays_d, camera, _, rays_d_un = load_one_frame(
            fov=fov,
            im=image,
            c2w=pose,
            image_res=self.output_image_res,
            hdr_to_ldr=False,
            resize=False,
        )
        if self.output_image_res[0] == self.output_image_res[1]:
            camera[16] = min(camera[16], camera[17])
            camera[17] = camera[16]
        if self.output_has_gt:
            image, rays_o, rays_d, rays_d_un, mask = self._center_crop(
                image, rays_o, rays_d, rays_d_un, mask, self.output_image_res[1]
            )
        min_id, max_id = input_ids.min(), input_ids.max()
        rays_t = np.full((*rays_o.shape[:2], 1), (output_id - min_id) / (max_id - min_id), dtype=np.float32)
        return image, rays_o, rays_d, rays_d_un, camera, rays_t, output_id - min_id, mask

    @staticmethod
    def _stack(items, dtype=None):
        return stack_items(items, dtype, check_shapes=True)

    def __getitem__(self, idx):
        model_path = self.models[idx % len(self.models)]
        model_id = model_path.name
        poses, fovs, image_names = self._load_cameras(model_path)
        contexts, targets, train_split = self._select_context_and_targets(model_path, poses, image_names)
        if self.pre_selected_contexts is not None:
            contexts = self.pre_selected_contexts[model_id]

        input_ids = np.array([image_names.index(name) for name in contexts])
        output_ids = np.array([image_names.index(name) for name in targets])
        if self.normalize_mode == "input":
            poses, cam_loc_mean, cam_loc_scale = self._normalize_from_inputs(poses, input_ids)
        else:
            poses, cam_loc_mean, cam_loc_scale = self._normalize_all(poses)

        input_frames, input_masks, input_names = self._load_images_and_masks(model_path, contexts)
        output_frames, output_masks, output_names = self._load_images_and_masks(model_path, targets)

        input_samples = []
        for n in range(self.input_image_num):
            preinpaint_path = self.pre_inpaint.get(model_id) if self.pre_inpaint is not None and n == 0 else None
            input_samples.append(
                self._input_sample(n, input_frames[n], input_masks[n], poses[input_ids[n]], fovs[input_ids[n]], preinpaint_path)
            )

        images_input = [sample[0] for sample in input_samples]
        instance_masks_input = [sample[7] for sample in input_samples]
        if self.produce_masked_result:
            for i in range(1, self.input_image_num):
                images_input[i] = images_input[i] * (1 - instance_masks_input[i])

        output_samples = []
        for n in range(self.output_image_num):
            pose = poses[output_ids[n]] if self.output_has_gt else self._middle_pose(poses[input_ids[0]], poses[input_ids[1]])
            output_samples.append(
                self._output_sample(n, output_frames[n], output_masks[n], pose, fovs[output_ids[n]], output_ids[n], input_ids)
            )

        train_camera_poses = np.stack([poses[image_names.index(name)] for name in train_split])
        out = {
            "name": model_id,
            "rgb_input": self._stack(images_input, np.float32),
            "mask_input": np.ones_like(self._stack(images_input))[:, 0:1],
            "rgb_names_input": [str(path) for path in input_names],
            "rays_o_input": self._stack([sample[1] for sample in input_samples], np.float32),
            "rays_d_input": self._stack([sample[2] for sample in input_samples], np.float32),
            "rays_d_un_input": self._stack([sample[3] for sample in input_samples], np.float32),
            "cameras_input": self._stack([sample[4] for sample in input_samples], np.float32),
            "rays_t_input": self._stack([sample[5] for sample in input_samples], np.float32),
            "rays_t_un_input": self._stack([sample[6] for sample in input_samples], np.int32),
            "instance_masks_input": self._stack(instance_masks_input),
            "camera_poses": train_camera_poses,
            "cam_loc_mean": cam_loc_mean,
            "cam_loc_scale": cam_loc_scale,
            "fov": list(fovs[0] / np.pi * 180),
            "eva_input_views": input_ids,
        }

        if self.vis_centers is not None:
            out["vis_center_pose"] = poses[image_names.index(self.vis_centers[model_id])]

        if self.output_image_num > 0:
            out.update({
                "rgb_names_output": [str(path) for path in output_names],
                "rays_o_output": self._stack([sample[1] for sample in output_samples], np.float32),
                "rays_d_output": self._stack([sample[2] for sample in output_samples], np.float32),
                "cameras_output": self._stack([sample[4] for sample in output_samples], np.float32),
                "rays_t_output": self._stack([sample[5] for sample in output_samples], np.float32),
                "rays_t_un_output": self._stack([sample[6] for sample in output_samples], np.int32),
                "instance_masks_output": self._stack([sample[7] for sample in output_samples], np.int32),
                "eva_output_views": output_ids,
            })
            if self.output_has_gt:
                images_output = self._stack([sample[0] for sample in output_samples], np.float32)
                out["rgb_output"] = images_output
                out["mask_output"] = np.ones_like(images_output)[:, 0:1]
                out["cropping_output"] = np.tile(
                    np.array([
                        self.output_image_res[0],
                        self.output_image_res[1],
                        0,
                        self.output_image_res[0],
                        0,
                        self.output_image_res[1],
                    ], dtype=np.int32),
                    (self.output_image_num, 1),
                )
        return out
