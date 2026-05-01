import json
from pathlib import Path

import cv2
import numpy as np

from ..misc.io_helper import pathmgr


def decode_rgb_bytes(encoded):
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, -1)
    if image.ndim == 3:
        image[:, :, :3] = image[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(image)


def decode_rgb_path(path):
    with pathmgr.open(path, "rb") as f:
        return decode_rgb_bytes(f.read())


def resize_to_height(image, target_height, interpolation=cv2.INTER_AREA):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    new_width = int(width * (target_height / height))
    return cv2.resize(image, (new_width, target_height), interpolation=interpolation)


def normalize_camera_poses(camera_poses):
    camera_poses = camera_poses.copy()
    locations = camera_poses[:, :3, 3:4]
    mean = locations.mean(axis=0, keepdims=True)
    scale = np.abs(locations - mean).max()
    camera_poses[:, :3, 3:4] = (locations - mean) / scale
    return camera_poses


def load_dl3dv_cameras(model_path):
    transform_path = Path(model_path) / "transforms.json"
    with pathmgr.open(transform_path, "r") as f:
        transform_data = json.load(f)

    fov = (
        2 * np.arctan2(transform_data["cy"], transform_data["fl_y"]),
        2 * np.arctan2(transform_data["cx"], transform_data["fl_x"]),
    )

    image_names = []
    camera_poses = []
    for frame in transform_data["frames"]:
        image_names.append(Path(frame["file_path"]).name)
        camera_poses.append(frame["transform_matrix"])

    camera_poses = np.asarray(camera_poses, dtype=np.float32)
    camera_poses[:, 2, :] *= -1
    camera_poses = camera_poses[:, np.array([1, 0, 2, 3]), :]
    return camera_poses, fov, image_names


def center_crop_frame(image, rays_o, rays_d, rays_d_un, camera, target_res, mask=None):
    _, height, width = image.shape
    target_height, target_width = target_res
    height_start = (height - target_height) // 2
    height_end = height_start + target_height
    width_start = (width - target_width) // 2
    width_end = width_start + target_width

    image = image[:, height_start:height_end, width_start:width_end]
    rays_o = rays_o[height_start:height_end, width_start:width_end]
    rays_d = rays_d[height_start:height_end, width_start:width_end]
    rays_d_un = rays_d_un[height_start:height_end, width_start:width_end]
    if mask is not None:
        mask = mask[height_start:height_end, width_start:width_end]

    scale_x = target_width / width
    scale_y = target_height / height
    camera[16] = 2 * np.arctan2(scale_x * np.tan(camera[16] / 2), 1.0)
    camera[17] = 2 * np.arctan2(scale_y * np.tan(camera[17] / 2), 1.0)

    crop = (height_start, height_end, width_start, width_end)
    if mask is None:
        return image, rays_o, rays_d, rays_d_un, camera, crop
    return image, rays_o, rays_d, rays_d_un, camera, mask, crop


def stack_items(items, dtype=None, check_shapes=False):
    if check_shapes:
        shapes = [np.shape(item) for item in items]
        if len(set(shapes)) != 1:
            raise ValueError(f"Cannot stack arrays with inconsistent shapes: {shapes}")
    arr = np.stack(items, axis=0)
    return arr.astype(dtype) if dtype is not None else arr
