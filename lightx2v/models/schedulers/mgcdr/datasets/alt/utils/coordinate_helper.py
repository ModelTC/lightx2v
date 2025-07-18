# Import from third library
import numpy as np

# Import from alt
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


def transform_box3d_to_point_cloud(dimension, location, roll, pitch, yaw, cam2lidar):
    width, height, lenght = dimension
    x, y, z = location

    x_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    y_corners = [height / 2, height / 2, -height / 2, -height / 2, height / 2, height / 2, -height / 2, -height / 2]
    z_corners = [lenght / 2, lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, -lenght / 2, -lenght / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    if roll is None or pitch is None:
        rotation = yaw
        R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)], [0, 1, 0], [-np.sin(rotation), 0, np.cos(rotation)]])
    else:
        R_matrix = Rotation.from_rotvec(np.array([roll, pitch, yaw])).as_matrix()

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # from camera coordinate to velodyne coordinate
    corners_3d = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
    corners_3d = np.dot(cam2lidar, corners_3d.T).T
    corners_3d = corners_3d[:, :3]

    return corners_3d


def transform_point_cloud_to_image(pcloud, cam2lidar, camera_intrinsic, camera_intrinsic_dist, camera_lens, img_w, img_h):
    # from lidar coordinate to camera coordinate
    lidar2cam = np.linalg.inv(cam2lidar)
    pcloud = pcloud.T[0:3, :]
    lidar_in_cam = lidar2cam @ np.vstack((pcloud[0:3][:], np.ones_like(pcloud[1])))
    # only show point cloud in front of the camera
    lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]
    pc_xyz_ = lidar_in_cam[:3, :]
    if camera_lens == "fisheye" and camera_intrinsic_dist.shape[0] == 2:
        image_point = fisheye_camera_to_image(pc_xyz_.transpose(), camera_intrinsic, camera_intrinsic_dist)
        image_point = image_point.transpose()
    else:
        pc_xyz = pc_xyz_ / pc_xyz_[2, :]
        image_point = camera_intrinsic @ pc_xyz
        image_point = image_point[:2, :]
    image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= img_w)
    image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= img_h)
    image_limit = np.logical_and(image_limit_h, image_limit_w)
    image_point = image_point[:, image_limit]
    cam_z = pc_xyz_[2, image_limit]
    cam_z = cam_z[np.newaxis, :]
    return image_point, cam_z


def poly_val(param, x):
    n = len(param)
    res = [0.0 for x in range(x.shape[0])]
    for i in range(x.shape[0]):
        for itr in range(len(param)):
            res[i] = param[n - itr - 1] + res[i] * x[i]
    return res


def fisheye_camera_to_image(p3ds, camera_intrinsic, camera_dist):
    aff_ = np.array([camera_intrinsic[0][0], camera_intrinsic[0][1], camera_intrinsic[1][0], camera_intrinsic[1][1]]).reshape(
        2, 2
    )
    xc_ = camera_intrinsic[0][2]
    yc_ = camera_intrinsic[1][2]

    norm = np.linalg.norm(p3ds[:, :2], axis=1)

    if camera_dist.shape[0] == 2:
        theta = np.arctan2(-p3ds[:, 2], norm)
        inv_poly_param_ = camera_dist[1]
        rho = poly_val(inv_poly_param_, theta)

    elif camera_dist.shape[0] == 1:  # KB
        theta = np.arctan2(norm, p3ds[:, 2])
        _camera_dist = camera_dist[0]

        rho = (
            theta
            + np.power(theta, 3) * _camera_dist[0]
            + np.power(theta, 5) * _camera_dist[1]
            + np.power(theta, 7) * _camera_dist[2]
            + np.power(theta, 9) * _camera_dist[3]
        )

    xn = np.ones((2, p3ds.shape[0]), dtype=p3ds.dtype)
    xn[0] = p3ds[:, 0] / norm * rho
    xn[1] = p3ds[:, 1] / norm * rho

    p2ds = (np.dot(aff_, xn) + np.tile(np.array([xc_, yc_]).reshape(2, -1), (xn.shape[1]))).T

    return p2ds


def transform_box3d_to_image(dimension, location, roll, pitch, yaw, camera_intrinsic, camera_intrinsic_dist, camera_lens):
    width, height, lenght = dimension
    x, y, z = location

    x_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    y_corners = [height / 2, height / 2, -height / 2, -height / 2, height / 2, height / 2, -height / 2, -height / 2]
    z_corners = [lenght / 2, lenght / 2, lenght / 2, lenght / 2, -lenght / 2, -lenght / 2, -lenght / 2, -lenght / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # Calculate the angle of rotation around the Y axis
    if roll is None or pitch is None:
        rotation = yaw
        R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)], [0, 1, 0], [-np.sin(rotation), 0, np.cos(rotation)]])
    else:
        R_matrix = Rotation.from_rotvec(np.array([roll, pitch, yaw])).as_matrix()
        # euler = Rotation.from_rotvec(np.array([roll, pitch, yaw])).as_euler('zyx')
        # euler[0] = euler[0] + np.pi
        # R_matrix = Rotation.from_euler('zyx', euler).as_matrix()

    # transform 3d box based on rotation along Y-axis
    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])
    # only show 3D bounding box for objects in front of the camera
    if np.any(corners_3d[:, 2] < 0):
        corners_3d_img = None
    else:
        # from camera coordinate to image coordinate
        if camera_lens == "fisheye":
            corners_3d_img = fisheye_camera_to_image(corners_3d, camera_intrinsic, camera_intrinsic_dist)
        else:
            corners_3d_img = np.matmul(corners_3d, camera_intrinsic.T)
            corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]

    return corners_3d_img


def scale_to_255(a, min, max, dtype=np.uint8):
    """Scales an array of values from specified min, max range to 0-255
    Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def transform_point_cloud_to_birdseye(
    points, res=0.1, side_range=(-50.0, 50.0), fwd_range=(-50.0, 50.0), height_range=(-5.0, 5.0)
):
    """Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # Extract the points for each axis
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # Filter, to return only indices of points within desired cube
    # Three filters for: front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # Keepers
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # Convert to pixel position values based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # Shift pixels to have minimum be (0, 0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # Clip height values to between min and max heights
    pixel_values = np.clip(a=z_points, a_min=height_range[0], a_max=height_range[1])

    # Rescale the height values to be between the range 0-255
    pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

    # Initialize empty array of dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # Fill pixel values in image array
    im[y_img, x_img] = pixel_values

    return im


def transform_corners_3d_to_birdseye(corners_3d, res, side_range, fwd_range):
    x_corners_3d = corners_3d[..., 0]
    y_corners_3d = corners_3d[..., 1]
    x_bbox2d = (-y_corners_3d / res).astype(np.int32)
    y_bbox2d = (-x_corners_3d / res).astype(np.int32)
    x_bbox2d -= int(np.floor(side_range[0] / res))
    y_bbox2d += int(np.ceil(fwd_range[1] / res))
    return x_bbox2d, y_bbox2d


def draw_corners_3d(corners_3d, name):
    x = corners_3d[:, 0]
    y = corners_3d[:, 1]
    z = corners_3d[:, 2]

    # the start and end point for each line
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]

    x_lines = list()
    y_lines = list()
    z_lines = list()

    for p in pairs:
        for i in range(2):
            x_lines.append(x[p[i]])
            y_lines.append(y[p[i]])
            z_lines.append(z[p[i]])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    corners_3d_line = go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode="lines", name=name, marker=dict(color="red"))
    return corners_3d_line
