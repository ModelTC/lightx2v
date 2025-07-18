# Standard Library
import copy

# Import from third library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import LineCollection

# Import from alt
from alt.coordinate.bbox import points_cam2img

# from .utils import random_cnames, plt2img
from alt.visualize.utils import plt2img

light_color_list = [
    "#FFD700",  # Gold
    "#ADFF2F",  # Green Yellow
    "#FF69B4",  # Hot Pink
    "#87CEFA",  # Light Sky Blue
    "#FFA07A",  # Light Salmon
    "#00FA9A",  # Medium Spring Green
    "#FF6347",  # Tomato
    "#BA55D3",  # Medium Orchid
    "#FF4500",  # Orange Red
    "#DA70D6",  # Orchid
    "#7FFFD4",  # Aquamarine
    "#FFFACD",  # Lemon Chiffon
    "#FFDAB9",  # Peach Puff
    "#FFB6C1",  # Light Pink
    "#FF8C00",  # Dark Orange
    "#98FB98",  # Pale Green
    "#AFEEEE",  # Pale Turquoise
    "#DB7093",  # Pale Violet Red
    "#F5DEB3",  # Wheat
    "#B0E0E6",  # Powder Blue
    "#FFC0CB",  # Pink
    "#FF1493",  # Deep Pink
    "#00CED1",  # Dark Turquoise
    "#40E0D0",  # Turquoise
    "#EE82EE",  # Violet
    "#9370DB",  # Medium Purple
    "#C71585",  # Medium Violet Red
    "#20B2AA",  # Light Sea Green
    "#87CEEB",  # Sky Blue
    "#778899",  # Light Slate Gray
    "#6A5ACD",  # Slate Blue
    "#00BFFF",  # Deep Sky Blue
]


def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in line_indices:
            cv2.line(
                img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), color, thickness, cv2.LINE_AA
            )
        for start, end in ((4, 6), (5, 7)):  # 与点云检测模型可视化对齐，叉叉画在车头
            cv2.line(
                img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), color, thickness, cv2.LINE_AA
            )

    return img.astype(np.uint8)


def expand_plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    original_height, original_width = img.shape[:2]

    # Calculate new dimensions
    new_height = original_height * 2
    new_width = original_width * 2

    # Create a new blank image with doubled dimensions and white background
    new_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_img.fill(255)  # Assuming a white background, adjust as needed

    # Calculate the top-left corner position to place the original image
    top_left_y = (new_height - original_height) // 2
    top_left_x = (new_width - original_width) // 2

    # Place the original image in the center of the new image
    new_img[top_left_y : top_left_y + original_height, top_left_x : top_left_x + original_width] = img

    rect_corners[..., 0] += top_left_x
    rect_corners[..., 1] += top_left_y

    new_img = plot_rect3d_on_img(new_img, num_rects, rect_corners, color, thickness)

    crop_img = new_img[top_left_y : top_left_y + original_height, top_left_x : top_left_x + original_width]

    return crop_img.astype(np.uint8)


def draw_camera_bbox3d_on_img(bboxes3d, raw_img, cam2img, color=(0, 255, 0), thickness=1, label=[]):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`CameraInstance3DBoxes`, shape=[M, 7]):
            3d bbox in camera coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    cam2img = copy.deepcopy(cam2img)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert cam2img.shape == torch.Size([3, 3]) or cam2img.shape == torch.Size([4, 4])
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img, clamp=True)

    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def draw_lidar_bbox3d_on_img(
    bboxes3d, raw_img, lidar2camera_rt, camera_intrinsic, camera_dist, img_metas=None, color=(0, 255, 0), thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate([corners_3d.reshape(-1, 3), np.ones((num_bbox * 8, 1))], axis=-1)
    # if not 'fisheye':
    lidar2img_rt = camera_intrinsic @ lidar2camera_rt

    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()

    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    # else:

    #     pcloud = pcloud.T[0:3, :]
    #     lidar_in_cam = lidar2camera_rt @ np.vstack((pcloud[0:3][:], np.ones_like(pcloud[1])))
    #     # only show point cloud in front of the camera
    #     lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]
    #     pc_xyz_ = lidar_in_cam[:3, :]

    #     image_point = fisheye_camera_to_image(pc_xyz_.transpose(), camera_intrinsic, camera_intrinsic_dist)
    #     image_point = image_point.transpose()

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)


def fast_draw_bev_bbox_on_mesh(
    bboxes3d,
    ids=None,
    speeds=None,
    save_path=None,
    figsize=(4, 10),
    x_locator=25,
    y_locator=25,
    xlim=[-50, 50],
    ylim=[0, 250],
    color="white",
):
    corners_3d = bboxes3d.corners
    corners_3d = copy.deepcopy(corners_3d).numpy()

    bev_points = []

    plt.rc("font")
    plt.figure(figsize=figsize)
    plt.title("BEV View", fontsize=15, color="white")

    plt.xlabel("Y(m)", fontsize=15, color="white")
    plt.xlabel("X(m)", fontsize=15, color="white")

    # 提前计算颜色
    colors = [light_color_list[int(trackid) % len(light_color_list)] for trackid in ids]

    # 存储所有绘图元素
    lines, angle_lines = [], []
    scatters_x, scatters_y, scatters_colors = [], [], []

    for item, color, speed in zip(corners_3d, colors, speeds):
        p1 = (float(item[1][0]), float(item[0][1]))
        p2 = (float(item[5][0]), float(item[5][1]))
        p3 = (float(item[6][0]), float(item[6][1]))
        p4 = (float(item[2][0]), float(item[2][1]))
        bev_points.append([p1, p2, p3, p4])
        _y = [p1[0], p2[0], p3[0], p4[0], p1[0]]
        _x = [-1.0 * p1[1], -1.0 * p2[1], -1.0 * p3[1], -1.0 * p4[1], -1.0 * p1[1]]

        lines.append(list(zip(_x, _y)))
        scatters_x.extend(_x[:-1])  # 不需要最后一个点，因为它是第一个点的重复
        scatters_y.extend(_y[:-1])  # 同上
        scatters_colors.extend([color] * 4)

        # 车头方向
        hy1, hy2 = _y[1] + (_y[2] - _y[1]) / 3, _y[1] + (_y[2] - _y[1]) / 3 * 2
        hx1, hx2 = _x[1] + (_x[2] - _x[1]) / 3, _x[1] + (_x[2] - _x[1]) / 3 * 2

        angle_lines.append([(hx1, hy1), (hx2, hy2)])

        if any(speed):  # 全部为0的不画速度
            speed_y, speed_x = speed[0] / 10, -1.0 * speed[1] / 10  # 按100ms的速度来画
            cx1, cy1 = (_x[1] + _x[2]) / 2, (_y[1] + _y[2]) / 2
            plt.arrow(cx1, cy1, speed_x, speed_y, head_width=1, head_length=2, fc=color, ec=color)

    # 绘制轮廓线
    line_segments = LineCollection(lines, colors=colors, linewidths=1)
    plt.gca().add_collection(line_segments)

    # 绘制方向
    angle_line_segments = LineCollection(angle_lines, colors=colors, linewidths=4)
    plt.gca().add_collection(angle_line_segments)

    # 绘制散点
    plt.scatter(scatters_x, scatters_y, color=scatters_colors, s=0.1)

    circle = plt.Circle((0, 0), 1, color="blue", fill=True)  # 自车位置

    plt.tick_params(axis="x", width=2, length=4)
    plt.tick_params(axis="y", width=2, length=4)

    x_major_locator = MultipleLocator(x_locator)
    y_major_locator = MultipleLocator(y_locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    ax.add_patch(circle)
    ax.grid(True)  # 默认显示内部网格

    ax.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 设置网格线颜色和样式
    ax.grid(color="white", linestyle="--", linewidth=0.5)

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
        plt.close()
    else:
        rgb_image = plt2img(plt)
        plt.close()
        return rgb_image


def draw_bev_bbox_on_mesh(
    bboxes3d,
    ids=None,
    speeds=None,
    save_path=None,
    figsize=(4, 10),
    x_locator=25,
    y_locator=25,
    xlim=[-50, 50],
    ylim=[0, 250],
    color="white",
):
    corners_3d = bboxes3d.corners
    corners_3d = copy.deepcopy(corners_3d).numpy()

    bev_points = []

    plt.rc("font")
    plt.figure(figsize=figsize)
    plt.title("BEV View", fontsize=15, color="white")

    plt.xlabel("Y(m)", fontsize=15, color="white")
    plt.xlabel("X(m)", fontsize=15, color="white")

    for item, trackid, speed in zip(corners_3d, ids, speeds):
        p1 = (float(item[1][0]), float(item[0][1]))
        p2 = (float(item[5][0]), float(item[5][1]))
        p3 = (float(item[6][0]), float(item[6][1]))
        p4 = (float(item[2][0]), float(item[2][1]))
        bev_points.append([p1, p2, p3, p4])
        _y = [p1[0], p2[0], p3[0], p4[0], p1[0]]
        _x = [-1.0 * p1[1], -1.0 * p2[1], -1.0 * p3[1], -1.0 * p4[1], -1.0 * p1[1]]

        color = light_color_list[int(trackid) % len(light_color_list)]

        plt.plot(_x, _y, color=color, linewidth=1)
        plt.scatter(_x, _y, color=color, s=0.1)

        # 车头方向
        hy1, hy2 = _y[1] + (_y[2] - _y[1]) / 3, _y[1] + (_y[2] - _y[1]) / 3 * 2
        hx1, hx2 = _x[1] + (_x[2] - _x[1]) / 3, _x[1] + (_x[2] - _x[1]) / 3 * 2

        plt.plot([hx1, hx2], [hy1, hy2], color=color, linewidth=4)

        if any(speed):  # 全部为0的不画速度
            speed_y, speed_x = speed[0] / 10, -1.0 * speed[1] / 10  # 按100ms的速度来画
            cx1, cy1 = (_x[1] + _x[2]) / 2, (_y[1] + _y[2]) / 2
            plt.arrow(cx1, cy1, speed_x, speed_y, head_width=1, head_length=2, fc=color, ec=color)

    circle = plt.Circle((0, 0), 1, color="blue", fill=True)  # 自车位置

    plt.tick_params(axis="x", width=2, length=4)
    plt.tick_params(axis="y", width=2, length=4)

    x_major_locator = MultipleLocator(x_locator)
    y_major_locator = MultipleLocator(y_locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    ax.add_patch(circle)
    ax.grid(True)  # 默认显示内部网格

    ax.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # 设置网格线颜色和样式
    ax.grid(color="white", linestyle="--", linewidth=0.5)

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
        plt.close()
    else:
        rgb_image = plt2img(plt)
        plt.close()
        return rgb_image
