import numpy as np

from ..data_loader.utils import convex_hull_centroid, project_to_plane


def normalize(v):
    return v / np.linalg.norm(v)


def get_circle_extrinsics(
    camera_poses,
    num_points=30,
    policy="mean",
    center_pose=None,
    vis_scale=1.0,
    cam_loc_scale=1.0,
):
    assert policy in {"mean", "centroid", "selected"}
    camera_poses = camera_poses.cpu().numpy()
    positions = camera_poses[:, :3, 3]
    mean_position = np.mean(positions, axis=0)
    directions = -camera_poses[:, :3, 2]
    mean_direction = np.mean(directions, axis=0)

    if policy == "mean":
        center = mean_position
    elif policy == "centroid":
        projected_2d = project_to_plane(positions, mean_position, mean_direction)
        centroid = convex_hull_centroid(projected_2d)
        center = positions[np.argmin(np.linalg.norm(projected_2d - centroid, axis=1))]
    elif policy == "selected":
        assert center_pose is not None
        center_pose = center_pose.cpu().numpy()
        center = center_pose[:3, 3]

    diameter = 8 / cam_loc_scale
    radius = diameter / 2
    direction = normalize(mean_direction)

    nearest_view = np.argmin(np.linalg.norm(positions - center, axis=1))
    rotation = camera_poses[nearest_view, :3, :3]

    if not np.allclose(direction, [0, 0, 1]):
        u = normalize(np.cross(direction, [0, 0, 1]))
    else:
        u = normalize(np.cross(direction, [1, 0, 0]))
    v = normalize(np.cross(direction, u))

    poses = []
    for i in range(num_points):
        theta = 2 * np.pi * i / num_points
        position = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
        mat = np.eye(4)
        mat[:3, :3] = rotation
        mat[:3, 3] = position
        poses.append(mat)
    return np.array(poses)


def look_at(eye, target, up):
    z_axis = normalize(target - eye)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])
