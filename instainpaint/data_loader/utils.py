import json
import math
import os
import gzip

import cv2
import numpy as np
import torch

from collections.abc import Sequence

from ..misc.io_helper import pathmgr
from scipy.spatial import ConvexHull
import itertools
from torchvision.transforms import Compose
import io


def load_frames_dtc(
    obj_model_dirs,
    load_depth: bool = False,
    load_brdf: bool = False,
    load_normal: bool = False,
):
    obj_frames_dtc = []

    for obj_model in obj_model_dirs:
        camera_rig_path = os.path.join(obj_model, "CameraRig.json")
        camera_dict = json.load(open(camera_rig_path, "r"))

        cam_intrinsic = camera_dict["intrinsic"]
        c2w = camera_dict["to_world"]

        fov = cam_intrinsic["fov"] * np.pi / 180
        width = cam_intrinsic["width"]
        height = cam_intrinsic["height"]

        fl_x = width / (2 * math.tan(fov / 2))
        fl_y = height / (2 * math.tan(fov / 2))
        cx = (width - 1) / 2
        cy = (height - 1) / 2

        full_frames = []

        for idx in range(len(c2w)):
            frame = {
                "name": obj_model.split("/")[-1] + f"-frame{idx}",
                "dir": obj_model,
                "fov": fov,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "cx": cx,
                "cy": cy,
                "w": width,
                "h": height,
                "file_path": os.path.join(obj_model, "images", f"{idx}.png"),
                "c2w": np.array(c2w[idx]),
            }

            if load_depth:
                assert os.path.exists(
                    os.path.join(obj_model, "depth")
                ), f"Depth folder does not exist in the {obj_model}. Cannot load depth."
                frame["depth_path"] = os.path.join(obj_model, "depth", f"{idx}.exr")

            if load_brdf:
                assert os.path.exists(
                    os.path.join(obj_model, "albedo")
                ), f"Albedo folder does not exist in the {obj_model}. Cannot load albedo."
                assert os.path.exists(
                    os.path.join(obj_model, "metallic_roughness")
                ), f" folder does not exist in the {obj_model}. Cannot load metallic and roughness."

                frame["albedo_path"] = os.path.join(obj_model, "albedo", f"{idx}.png")
                frame["specular_path"] = os.path.join(
                    obj_model, "metallic_roughness", f"{idx}.png"
                )

            if load_normal:
                assert os.path.exists(
                    os.path.join(obj_model, "normal")
                ), f"Normal folder does not exist in the {obj_model}. Cannot load normal."
                frame["normal_path"] = os.path.join(obj_model, "normal", f"{idx}.exr")

            full_frames.append(frame)

        obj_frames_dtc.append(full_frames)

    return obj_frames_dtc


def linear_to_srgb(l):
    s = np.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def srgb_to_linear(s):
    l = np.zeros_like(s)
    m = s <= 0.0404482362771082
    l[m] = s[m] / 12.92
    l[~m] = ((s[~m] + 0.055) / 1.055) ** 2.4
    return l


def compute_rays(fov, matrix, res):
    """
    Screen space convention:
    x = right
    y = up
    z = into camera

    fov: Image space (h, w) tuple or int for square image.
         Note that (h, w) is in image space, which is (y,x) in screen space.
    """
    res_tuple = res if isinstance(res, Sequence) else  (res, res)
    matrix = np.array(matrix)
    rays_o = np.zeros((res_tuple[0], res_tuple[1], 3), dtype=np.float32) + matrix[0:3, 3].reshape(1, 1, 3)
    rays_o = rays_o

    # h_axis, w_axis is the 2D image space axis in height/width direction
    h_axis = np.linspace(0.5, res_tuple[0] - 0.5, res_tuple[0]) / res_tuple[0]
    w_axis = np.linspace(0.5, res_tuple[1] - 0.5, res_tuple[1]) / res_tuple[1]
    h_axis = 2 * h_axis - 1
    w_axis = 2 * w_axis - 1
    x, y = np.meshgrid(w_axis, h_axis) # Default indexing="xy" behavior
    if isinstance(fov, Sequence) or isinstance(fov, np.ndarray):
        x = x * np.tan(fov[1] / 2.0) # fov order is (w, h)
        y = y * np.tan(fov[0] / 2.0)
    else:
        x = x * np.tan(fov / 2.0)
        y = y * np.tan(fov / 2.0)

    # At this point
    #   x is im_width  axis = right
    #   y is im_height axis = down
    
    y = y * -1
    z = -np.ones(res_tuple)
    rays_d_un = np.stack([x, y, z], axis=-1)
    rays_d = rays_d_un / np.linalg.norm(rays_d_un, axis=-1)[:, :, None]
    rot = matrix[0:3, 0:3][None, None, :, :]
    rays_d_un = np.sum(rot * rays_d_un[:, :, None, :], axis=-1)
    rays_d = np.sum(rot * rays_d[:, :, None, :], axis=-1)

    return rays_o, rays_d, rays_d_un


def camera_pose_to_vec(c2w, fov, principal_points=None):
    if principal_points is None:
        principal_points = (0.5, 0.5)
    camera_ext = np.array(c2w).reshape(16)
    if isinstance(fov, Sequence) or isinstance(fov, np.ndarray):
        camera_int = np.array([fov[1], fov[0], principal_points[1], principal_points[0]])
    else:
        camera_int = np.array([fov, fov, principal_points[1], principal_points[0]])
    camera = np.concatenate([camera_ext, camera_int])
    return camera


def load_one_frame(fov, im=None, c2w=None, image_res=512, hdr_to_ldr=False, resize=True, normalize=True, principal_points=None):

    im, mask = load_one_image(
        im=im, resize=resize, image_res=image_res, normalize=normalize, hdr_to_ldr=hdr_to_ldr,
    )
    _, im_h, im_w = im.shape
    rays_o, rays_d, rays_d_un = compute_rays(fov, c2w, (im_h, im_w))
    camera = camera_pose_to_vec(c2w, fov, principal_points)

    return (
        im.astype(np.float32) if im is not None else None,
        rays_o.astype(np.float32),
        rays_d.astype(np.float32),
        camera.astype(np.float32),
        mask.astype(np.float32) if mask is not None else None,
        rays_d_un.astype(np.float32),
    )


def load_one_image(
    im, image_res=512, resize=True, normalize=False, hdr_to_ldr=False, ldr_to_hdr=False,
):
    if isinstance(im, str):
        with pathmgr.open(im, "rb") as fIn:
            buffer = fIn.read()
            buffer = np.asarray(bytearray(buffer), dtype=np.uint8)
            im = cv2.imdecode(buffer, -1)
            if len(im.shape) == 3:
                if im.shape[2] == 3:
                    im = np.ascontiguousarray(im[:, :, ::-1])
                else:
                    tmp_im = im[:, :, :3]
                    tmp_im = np.ascontiguousarray(im[:, :, ::-1])
                    tmp_mask = im[:, :, 3:4]
                    im = np.concatenate([tmp_im, tmp_mask], axis=-1)
            else:
                im = np.stack([im, im, im], axis=-1)

    if im.dtype != np.float32:
        if im.dtype == np.uint16:
            im = im.astype(np.float32) / 65535
        else:
            im = im.astype(np.float32) / 255.0

    if hdr_to_ldr:
        im = linear_to_srgb(im)
    if ldr_to_hdr:
        im = srgb_to_linear(im)

    if im.shape[2] == 4:
        mask = im[:, :, 3:4]
        im = im[:, :, 0:3] + (1.0 - mask)
        if resize:
            mask = cv2.resize(
                mask[:, :, 0], 
                (image_res, image_res) if isinstance(image_res, int) else image_res, 
                interpolation=cv2.INTER_AREA
            )
        mask = mask[None, :, :]
    else:
        if isinstance(image_res, int):
            mask = np.ones((1, image_res, image_res), dtype=np.float32)
        else:
            mask = np.ones((1, image_res[0], image_res[1]), dtype=np.float32)

    if normalize:
        im = 2 * im - 1
    if resize:
        im = cv2.resize(
            im, 
            (image_res, image_res) if isinstance(image_res, int) else image_res, 
            interpolation=cv2.INTER_AREA)
        if im.ndim == 2: # Grayscale image
            im = im[None, :, :]
    im = im.transpose(2, 0, 1)

    return im, mask


def load_specular(im, image_res):
    im, _ = load_one_image(im, image_res, normalize=True, ldr_to_hdr=True)
    metallic = im[0:1, :]
    roughness = im[1:2, :]
    return roughness, metallic


def calc_resize_short_edge(img, target_size):
    ori_h, ori_w = img.shape[0], img.shape[1]
    target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
    if ori_h < ori_w:
        new_h = target_size[0]
        new_w = round((target_size[0] / new_h) * ori_w)
    else:
        new_w = target_size[1]
        new_h = round((target_size[0] / new_w) * ori_h)
    return new_h, new_w


def load_depth(depth, image_res, resize=True):
    if resize:
        depth = cv2.resize(
            depth, 
            (image_res, image_res) if isinstance(image_res, int) else image_res, 
            interpolation=cv2.INTER_AREA)
    if len(depth.shape) == 3:
        depth = depth.transpose(2, 0, 1)
        depth = depth[0:1, :, :]
    elif len(depth.shape) == 2:
        depth = depth[None, :, :]
    return depth


def load_covisible(covisible, image_res, resize=True):
    if resize:
        covisible = cv2.resize(
            covisible, 
            (image_res, image_res) if isinstance(image_res, int) else image_res, 
            interpolation=cv2.INTER_NEAREST)
    if len(covisible.shape) == 3:
        covisible = covisible.transpose(2, 0, 1)
        covisible = covisible[0:1, :, :]
    elif len(covisible.shape) == 2:
        covisible = covisible[None, :, :]
    return covisible


def load_flow(flow, image_res):
    flow = np.stack([
        cv2.resize(
            flow[..., 0], 
            (image_res, image_res) if isinstance(image_res, int) else image_res, 
            interpolation=cv2.INTER_AREA),
        cv2.resize(
            flow[..., 1], 
            (image_res, image_res) if isinstance(image_res, int) else image_res, 
            interpolation=cv2.INTER_AREA)
    ], -1)
    flow = flow.transpose(2, 0, 1)
    return flow


def load_envmap(env, width, height):
    orig_height, orig_width = env.shape[0:2]
    env = np.ascontiguousarray(env[:, :, ::-1])

    env_mask = np.ones((orig_height, orig_width), dtype=np.float32)
    check_height = int(
        orig_height * 0.85
    )  # Hard coded a threshold as part of env is missing
    env_mask[check_height:, :] = np.max(env[check_height:, :, :], axis=2) > 0.05
    env_mask = env_mask[:, :, None]

    if orig_height != height and orig_width != width:
        theta = np.linspace(0, orig_height - 1, orig_height) + 0.5
        weight = np.sin(theta / orig_height * np.pi)[:, None, None]
        weight = np.tile(weight, [1, orig_width, 1])

        env = cv2.resize(env * weight, (width, height), interpolation=cv2.INTER_AREA)
        env_mask = cv2.resize(
            env_mask * weight, (width, height), interpolation=cv2.INTER_AREA
        )
        weight = cv2.resize(weight, (width, height), interpolation=cv2.INTER_AREA)

        env_mask = env_mask.reshape(height, width, 1)
        weight = weight.reshape(height, width, 1)

        env = env / np.maximum(weight, 1e-6)
        env_mask = env_mask / np.maximum(weight, 1e-6)

    env = env.transpose(2, 0, 1)
    env_mask = env_mask.transpose(2, 0, 1)
    return env, env_mask


def transform_cams(cam, cam_rot, flattened=True):
    cam_rot = cam_rot.transpose(1, 0)
    if flattened:
        ext_orig = cam[:16].reshape(4, 4)
    else:
        ext_orig = cam.reshape(4, 4)
    inv = np.eye(4).astype(cam.dtype)
    inv[0:3, 0:3] = cam_rot
    ext = np.matmul(inv, ext_orig)
    ret_cam = cam.copy()
    if flattened:
        ret_cam[:16] = ext.reshape(16)
    else:
        ret_cam = ext
    return ret_cam


def transform_rays_o(rays_o, cam_rot):
    rays_o = rays_o[:, :, :, None]
    cam_rot = cam_rot.reshape(1, 1, 3, 3)
    rays_o = np.sum(cam_rot * rays_o, axis=-2)
    return rays_o


def transform_rays_d(rays_d, cam_rot):
    rays_d = rays_d[:, :, :, None]
    cam_rot = cam_rot.reshape(1, 1, 3, 3)
    rays_d = np.sum(cam_rot * rays_d, axis=-2)
    return rays_d


def transform_normal(normal, mask, cam_rot):
    normal = normal[:, None, :, :]
    cam_rot = cam_rot[:, :, None, None]
    normal = np.sum(normal * cam_rot, axis=0)
    normal = normal * mask
    return normal


def importance_selection(mask, crop_size, offset=0.001):
    mask = mask.squeeze() + offset
    height, width = mask.shape
    m_row = np.sum(mask, axis=1)
    m_col = np.sum(mask, axis=0)
    m_row = np.concatenate([np.zeros(1, dtype=m_row.dtype), m_row])
    m_col = np.concatenate([np.zeros(1, dtype=m_row.dtype), m_col])

    m_row_acc = np.cumsum(m_row)
    m_col_acc = np.cumsum(m_col)
    m_row_int = m_row_acc[crop_size:] - m_row_acc[:-crop_size]
    m_col_int = m_col_acc[crop_size:] - m_col_acc[:-crop_size]

    m_row_cdf = np.cumsum(m_row_int)
    m_col_cdf = np.cumsum(m_col_int)

    m_row_cdf = m_row_cdf / np.maximum(np.max(m_row_cdf), 1e-6)
    m_col_cdf = m_col_cdf / np.maximum(np.max(m_col_cdf), 1e-6)

    row_rand = np.random.random()
    col_rand = np.random.random()

    hs = np.searchsorted(m_row_cdf, row_rand, side="right")
    he = hs + crop_size

    ws = np.searchsorted(m_col_cdf, col_rand, side="right")
    we = ws + crop_size

    return hs, he, ws, we


def crop_and_resize(im, hs, he, ws, we, height, width):
    im = im[hs:he, ws:we]
    im = cv2.resize(im, (width, height), interpolation=cv2.INTER_LINEAR)
    return im


def compute_cropping_from_mask(mask):
    height, width = mask.shape

    if np.sum(mask) == 0:
        return 0, height, 0, width

    mask_row = np.sum(mask, axis=0)
    mask_col = np.sum(mask, axis=1)

    m_row_nonzero = np.nonzero(mask_row)[0]
    m_row_s = m_row_nonzero.min()
    m_row_e = min(m_row_nonzero.max() + 1, width)
    m_row_len = m_row_e - m_row_s
    m_row_center = (m_row_s + m_row_e) // 2

    m_col_nonzero = np.nonzero(mask_col)[0]
    m_col_s = m_col_nonzero.min()
    m_col_e = min(m_col_nonzero.max() + 1, height)
    m_col_len = m_col_e - m_col_s
    m_col_center = (m_col_s + m_col_e) // 2

    if m_col_len > m_row_len:
        hs = m_col_s
        he = m_col_e
        sq_size = m_col_len

        ws = m_row_center - sq_size // 2
        we = ws + sq_size

        if ws < 0:
            ws = 0
            we = sq_size
        elif we > width:
            we = width
            ws = we - sq_size
    else:
        ws = m_row_s
        we = m_row_e
        sq_size = m_row_len

        hs = m_col_center - sq_size // 2
        he = hs + sq_size

        if hs < 0:
            hs = 0
            he = sq_size
        elif he > height:
            he = height
            hs = height - sq_size
    return hs, he, ws, we

# ==================== 2. 三维点投影到平面 ====================
def project_to_plane(points, centroid, normal):
    """
    将三维点投影到平面，返回局部二维坐标
    """
    # 构建局部坐标系
    u = np.array([1, 0, 0], dtype=np.float32)  # 任意非平行向量
    u -= np.dot(u, normal) * normal  # 确保与法向量正交
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    # 投影到局部坐标系
    vecs = points - centroid
    x_coords = np.dot(vecs, u)
    y_coords = np.dot(vecs, v)
    return np.column_stack((x_coords, y_coords))

# ==================== 3. 二维点关键点筛选 ====================
def find_key_pairs_2d(projected_points):
    """
    输入二维点，返回：
    - 主方向最远点对 (A,B)
    - 垂直方向最远点对 (C,D)
    """
    # 计算凸包加速搜索
    # hull = ConvexHull(projected_points)
    # hull_points = projected_points[hull.vertices]
    hull_points = projected_points
    
    # 主方向最远点对
    max_dist = 0
    A, B = 0, 1
    
    for i in range(len(hull_points)):
        for j in range(i+1, len(hull_points)):
            dist = np.linalg.norm(hull_points[i] - hull_points[j])
            if dist > max_dist:
                max_dist = dist
                A, B = i, j
    
    # 计算主方向向量
    main_dir = hull_points[B] - hull_points[A]
    main_dir /= np.linalg.norm(main_dir)
    
    # 垂直方向向量
    perp_dir = np.array([-main_dir[1], main_dir[0]])
    
    # 沿垂直方向的投影极值点
    projections = np.dot(projected_points - hull_points[A], perp_dir)
    C = np.argmax(projections)
    D = np.argmin(projections)
    
    return A, B, C, D

def convex_quad_area(a, b, c, d):
    # 按顺时针/逆时针顺序计算多边形面积
    return 0.5 * abs(
        (a[0]*b[1] + b[0]*c[1] + c[0]*d[1] + d[0]*a[1]) -
        (a[1]*b[0] + b[1]*c[0] + c[1]*d[0] + d[1]*a[0])
    )

def get_hull_indices(points):
    """返回凸包顶点在原始数组中的索引"""
    hull = ConvexHull(points)
    return hull.vertices  # 直接返回原始索引

def max_convex_hall_indices(points, num_vertices=4):
    hull_indices = get_hull_indices(points)
    max_area = 0
    best_quad_indices = []
    
    # 遍历所有四元组组合（基于原始索引）
    for quad_indices in itertools.combinations(hull_indices, num_vertices):
        quad_points = points[list(quad_indices)]  # 通过索引获取坐标
        try:
            # 检查是否为凸四边形
            quad_hull = ConvexHull(quad_points)
            if len(quad_hull.vertices) == num_vertices:
                area = quad_hull.volume  # 面积
                if area > max_area:
                    max_area = area
                    best_quad_indices = quad_indices  # 保存原始索引
        except:
            continue
    
    return best_quad_indices

def convex_hull_centroid(points):
    """
    计算点集凸包的质心
    
    参数:
        points: numpy数组，形状为 (N, 2) 或 (N, 3)
        
    返回:
        centroid: 凸包质心的坐标，形状与输入维度一致
    """
    # 计算凸包
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]  # 提取凸包顶点
    
    # 分解凸包为三角形（仅2D）
    if points.shape[1] == 2:
        centroid = _centroid_2d(hull_vertices)
    else:
        raise NotImplementedError("3D凸包质心计算需更复杂方法")
    return centroid

def _centroid_2d(vertices):
    """计算二维凸多边形的质心"""
    area_total = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    
    # 将多边形分解为以第一个顶点为公共顶点的三角形
    for i in range(1, len(vertices)-1):
        # 三角形顶点：0, i, i+1
        x1, y1 = vertices[0]
        x2, y2 = vertices[i]
        x3, y3 = vertices[i+1]
        
        # 计算三角形面积（向量叉积的一半）
        area = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        area_total += area
        
        # 计算三角形质心
        tri_centroid_x = (x1 + x2 + x3) / 3.0
        tri_centroid_y = (y1 + y2 + y3) / 3.0
        
        # 累加加权质心
        centroid_x += tri_centroid_x * area
        centroid_y += tri_centroid_y * area
    
    # 计算整体质心
    centroid = np.array([centroid_x / area_total, centroid_y / area_total])
    return centroid

def ptz_load(path):
    with open(path, "rb") as f:
        data = f.read()
    data = gzip.decompress(data)
    return torch.load(io.BytesIO(data), map_location="cpu")


def generate_random_ellipse_mask(img_size=(512, 512), 
                                center_range=None, 
                                angle_range=(0, 360),
                                mask_num=1):
    if center_range is None:
        center_range = ((0, img_size[0]),
                        (0, img_size[1]))

    if mask_num <= 2:
        axis_range = (img_size[0]//8, img_size[1]//6)
    else:
        axis_range = (img_size[0]//12, img_size[1]//8)

    def _generate_single_ellipse_mask():
        center = (
            np.random.randint(center_range[0][0], center_range[0][1]),
            np.random.randint(center_range[1][0], center_range[1][1])
        )
        major_axis = np.random.randint(axis_range[0]+1, axis_range[1])
        minor_axis = np.random.randint(axis_range[0], major_axis)
        angle = np.deg2rad(np.random.uniform(angle_range[0], angle_range[1]))
        
        h, w = img_size
        y, x = np.ogrid[:h, :w]
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        xx = x - center[1]
        yy = y - center[0]
        x_rot = xx * cos_angle - yy * sin_angle
        y_rot = xx * sin_angle + yy * cos_angle
        
        mask = ((x_rot / major_axis)**2 + (y_rot / minor_axis)**2) <= 1

        return mask

    for i in range(mask_num):
        mask = _generate_single_ellipse_mask()
        if i == 0:
            masks = mask
        else:
            masks = np.logical_or(masks, mask)
    
    return masks
