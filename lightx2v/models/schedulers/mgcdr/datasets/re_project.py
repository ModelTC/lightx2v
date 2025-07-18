import numpy as np
from PIL import Image, ImageDraw
# from tools.datasets.alt.coordinate.bbox import LiDARInstance3DBoxes
# from tools.datasets.alt.coordinate.points.utils import Camera3DPointsTransfer
from alt.coordinate.bbox import LiDARInstance3DBoxes
from alt.coordinate.points.utils import Camera3DPointsTransfer
import cv2
import warnings
import torch
import json

CLASS_BBOX_COLOR_MAP = {
    'VEHICLE_SUV': (255, 0, 0), #"运动型多用途轿车" car #红
    'VEHICLE_CAR': (255, 0, 0), # car #红
    'VEHICLE_TRUCK': (255, 128, 0), #"大型货车" truck #橙
    'CYCLIST_BICYCLE': (0, 255, 0), #"非机动车骑行者"  bicycle #绿
    'PEDESTRIAN_NORMAL': (0, 0, 255), #"普通行人" pedestrian #蓝
    'CYCLIST_MOTOR': (0, 255, 0), #"机动车骑行者" motorcycle #绿
    'VEHICLE_BUS': (255, 255, 0), #"巴士" bus #黄
    'VEHICLE_PICKUP': (255, 128, 0), #"皮卡" truck #橙
    'VEHICLE_SPECIAL': (255, 128, 0), #"特种车" car #橙
    'VEHICLE_TRIKE': (0, 255, 0), #"三轮车" motorcycle #绿
    'VEHICLE_MULTI_STAGE': (255, 128, 0), #"多段车单节车体" car #橙
    'PEDESTRIAN_TRAFFIC_POLICE': (0, 0, 255), #"交警" pedestrian # 蓝
    'VEHICLE_POLICE': (255, 0, 0), #"警车" car #红
    "VEHICLE_CAR_CARRIER_TRAILER":(255, 128, 0), #"拖挂车" # 橙
    "VEHICLE_TRAILER":(255, 128, 0), #"拖车" # 橙
    "VEHICLE_RUBBISH":(255, 128, 0), #"垃圾车" # 橙
    "OTHER":(255, 255, 255) #"垃圾车" # 橙
}

# 0000 0000 0000 0000
CLASS_BBOX_ID_MAP = {
    'VEHICLE_SUV': 0, #"运动型多用途轿车" car #红
    'VEHICLE_CAR': 1, # car #红
    'VEHICLE_TRUCK': 2, #"大型货车" truck #橙
    'CYCLIST_BICYCLE': 3, #"非机动车骑行者"  bicycle #绿
    'PEDESTRIAN_NORMAL': 4, #"普通行人" pedestrian #蓝
    'CYCLIST_MOTOR': 5, #"机动车骑行者" motorcycle #绿
    'VEHICLE_BUS': 6, #"巴士" bus #黄
    'VEHICLE_PICKUP': 7, #"皮卡" truck #橙
    'VEHICLE_SPECIAL': 8, #"特种车" car #橙
    'VEHICLE_TRIKE': 9, #"三轮车" motorcycle #绿
    'VEHICLE_MULTI_STAGE': 10, #"多段车单节车体" car #橙
    'PEDESTRIAN_TRAFFIC_POLICE': 11, #"交警" pedestrian # 蓝
    'VEHICLE_POLICE': 12, #"警车" car #红
    "VEHICLE_CAR_CARRIER_TRAILER":13, #"拖挂车" # 橙
    "VEHICLE_TRAILER":14, #"拖车" # 橙
    "VEHICLE_RUBBISH":15, #"垃圾车" # 橙
    "OTHER":16
}

# 定义边框线的连接点
line_indices = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)
]

def plot_filled_rect3d_on_img(img, num_rects, rect_corners, colors=[(0, 255, 0)], polycolors=[(255, 255, 255)],
                              thickness=1, alpha=0.3):
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
    # pap
    '''
        7 ------- 6
       / |       / |
      4 ------- 5  |
      |  |      |  |
      |  3 -----|--2
      | /       | /
      0 ------- 1
    0-> 3　为前进方向
    '''
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0]  # Left face
    ]


    overlay = img.copy()
    if isinstance(colors, tuple): colors = [colors]
    # if len(colors) == 1:
    #     head_color = colors[0]
    # else:
    #     head_color = CAR_HEAD_COLOR

    p1, p2, p3, p4 = 4, 5, 6, 7
    for i in range(num_rects):

        corners = rect_corners[i].astype(np.int32)

        if i >= len(colors):
            color = colors[-1]
            polycolor = polycolors[-1]
        else:
            color = colors[i]
            polycolor = polycolors[i]

        for face in faces:
            face_points = corners[face]
            cv2.fillPoly(overlay, [face_points], color=polycolor)

        # Blend the overlay with the original image to make it semi-transparent
        # Transparency factor
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for i in range(num_rects):
        # Draw edges of the cube for better visibility
        if i >= len(colors):
            color = colors[-1]
            polycolor = polycolors[-1]
        else:
            color = colors[i]
            polycolor = polycolors[i]
        for face in faces:
            face_points = corners[face]
            for i in range(len(face_points)):
                start_point = tuple(face_points[i])
                end_point = tuple(face_points[(i + 1) % len(face_points)])
                cv2.line(img, start_point, end_point, color=color, thickness=thickness)

        head_color = color
        for start, end in ((p1, p3), (p2, p4)):  # 与点云检测模型可视化对齐，叉叉画在车头
            cv2.line(
                img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), head_color, thickness,
                cv2.LINE_AA
            )

    return img.astype(np.uint8)

def plot_filled_rect3d_on_img_pillow_fixed(
        img, num_rects, rect_corners, with_borderline=True,
        colors=[(0, 255, 0)], polycolors=[(0, 0, 0)], thickness=1, alpha=0.3
):
    """
    Plot the boundary lines of 3D rectangulars on 2D images using Pillow.

    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        with_borderline (bool): Whether to draw border lines.
        colors (list[tuple[int]]): Colors for the border lines.
        polycolors (list[tuple[int]]): Colors for the filled polygons.
        thickness (int): Thickness of border lines.
        alpha (float): Transparency for the filled polygons.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0]  # Left face
    ]

    # Convert the numpy image to a Pillow Image
    pil_img = Image.fromarray(img)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))  # Transparent overlay
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")

    p1, p2, p3, p4 = 4, 5, 6, 7

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int16)

        if i >= len(colors):
            polycolor = polycolors[-1]
        else:
            polycolor = polycolors[i]

        # Draw filled faces
        for face in faces:
            face_points = [tuple(corners[point]) for point in face]
            draw_overlay.polygon(face_points, fill=(*polycolor, int(255 * alpha)))  # Add transparency with alpha

    # Blend the overlay with the original image
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)

    if with_borderline:
        draw = ImageDraw.Draw(pil_img, "RGBA")
        for i in range(num_rects):
            if i >= len(colors):
                color = colors[-1]
            else:
                color = colors[i]

            # Draw edges of the cube
            for face in faces:
                face_points = [tuple(corners[point]) for point in face]
                for j in range(len(face_points)):
                    start_point = face_points[j]
                    end_point = face_points[(j + 1) % len(face_points)]
                    draw.line([start_point, end_point], fill=(*color, 255), width=thickness)

            # Draw additional lines for head marking
            head_color = color
            for start, end in ((p1, p3), (p2, p4)):
                draw.line(
                    [tuple(corners[start]), tuple(corners[end])],
                    fill=(*head_color, 255),
                    width=thickness
                )

    # Convert to RGB before transforming
    output_img = np.array(pil_img.convert("RGB"))

    return output_img


def plot_rect3d_on_img(img, num_rects, rect_corners, colors=[(0, 255, 0)], thickness=1, mode='cross', alpha=0.5):
    """Plot the boundary lines of 3D rectangular on 2D images.
    # pap
        7 ------- 6
       / |       / |
      3 ------- 2  |
      |  |      |  |
      |  4 -----|--5
      | /       | /
      0 ------- 1
    3 -> 7　为前进方向
    Args:
        img (numpy.array): The numpy array of image.
        num_rects (int): Number of 3D rectangulars.
        rect_corners (numpy.array): Coordinates of the corners of 3D
            rectangulars. Should be in the shape of [num_rect, 8, 2].
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    if mode == 'poly':
        # 创建一个透明图层
        overlay = img.copy()
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    if isinstance(colors, tuple): colors = [colors]
    # if len(colors) == 1:
    #     head_color = colors[0]
    # else:
    #     head_color = CAR_HEAD_COLOR
    p1, p2, p3, p4 = 4, 5, 6, 7
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        # 使用 warnings.catch_warnings 来捕获警告
        # with warnings.catch_warnings(record=True) as w:
        #     warnings.simplefilter("always")  # 设置过滤器来捕获所有警告
        #     corners = rect_corners[i].astype(np.int32)
        #     # 检查是否捕获到了 RuntimeWarning
        #     for warning in w:
        #         if issubclass(warning.category, RuntimeWarning):
        #             print("捕获到 RuntimeWarning:", warning.message)
        #             import pdb;pdb.set_trace()

        if i >= len(colors):
            color = colors[-1]
        else:
            color = colors[i]

        for start, end in line_indices:
            cv2.line(
                img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), color, thickness,
                cv2.LINE_AA
            )

        head_color = color
        if mode == 'cross':

            for start, end in ((p1, p3), (p2, p4)):  # 与点云检测模型可视化对齐，叉叉画在车头
                cv2.line(
                    img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), head_color,
                    thickness, cv2.LINE_AA
                )
        if mode == 'poly':
            point = np.asarray([corners[p1], corners[p2], corners[p3], corners[p4]], dtype=np.int32)
            # print(point.shape, corners.shape, corners)
            cv2.fillPoly(overlay, [point], head_color)

    if mode == 'poly':
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img.astype(np.uint8)


def plot_rect3d_on_img_pillow_fixed(
        img,
        num_rects,
        rect_corners,
        colors=[(0, 255, 0)],
        thickness=1,
        mode="cross",
        alpha=0.5,
):
    """
    使用 Pillow 高效绘制 3D 矩形，并修复透明度和变换处理后丢失线条的问题。

    Args:
        img (numpy.ndarray): 输入图像。
        num_rects (int): 3D 矩形数量。
        rect_corners (numpy.ndarray): 3D 矩形顶点坐标，形状为 [num_rects, 8, 2]。
        colors (list[tuple]): 每个矩形的边框颜色。
        thickness (int): 边框线的厚度。
        mode (str): 绘制模式（'cross' 或 'poly'）。
        alpha (float): 透明度（仅在 `poly` 模式下生效）。

    Returns:
        numpy.ndarray: 绘制后的图像。
    """
    # 将 NumPy 图像转换为 Pillow 图像
    pil_img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    # 定义边框线的连接点
    line_indices = [
        (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)
    ]
    p1, p2, p3, p4 = 4, 5, 6, 7  # 定义车头点索引

    for i in range(num_rects):
        # 获取当前矩形的顶点坐标
        corners = rect_corners[i].astype(int)

        # 获取当前矩形的颜色
        color = tuple(colors[i % len(colors)])

        # 绘制边框线
        for start, end in line_indices:
            start_point = tuple(corners[start])
            end_point = tuple(corners[end])
            draw.line([start_point, end_point], fill=color, width=thickness)

        # 绘制车头的叉叉线（仅在 mode='cross' 时）
        if mode == "cross":
            head_color = color
            draw.line(
                [tuple(corners[p1]), tuple(corners[p3])], fill=head_color, width=thickness
            )
            draw.line(
                [tuple(corners[p2]), tuple(corners[p4])], fill=head_color, width=thickness
            )

        # 绘制车头的多边形填充（仅在 mode='poly' 时）
        if mode == "poly":
            head_color = tuple(int(alpha * c) for c in color)
            poly_points = [tuple(corners[idx]) for idx in (p1, p2, p3, p4)]
            draw.polygon(poly_points, fill=head_color)

    # 转回 NumPy 格式并返回
    return np.array(pil_img)



def expand_plot_rect3d_on_img(img, num_rects, rect_corners, colors=[(0, 255, 0)], polycolors=[(255, 255, 255)], \
                              thickness=1, mode='cross', alpha=0.5, with_filed=False,
                              with_borderline_on_filed_bbox=True):
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
    new_img[top_left_y: top_left_y + original_height, top_left_x: top_left_x + original_width] = img

    rect_corners_temp = rect_corners.copy()
    rect_corners_temp[..., 0] += top_left_x  # (23, 8, 2)
    rect_corners_temp[..., 1] += top_left_y  # (23, 8, 2)

    # 使用 warnings.catch_warnings 来捕获警告
    #
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # 设置过滤器来捕获所有警告
        if with_filed:
            # print(f'---------------In--{polycolors}------------------')plot_filled_rect3d_on_img_pillow
            new_img = plot_filled_rect3d_on_img_pillow_fixed(img=new_img, num_rects=num_rects,
                                                             rect_corners=rect_corners_temp,
                                                             with_borderline=with_borderline_on_filed_bbox,
                                                             colors=colors, polycolors=polycolors, thickness=thickness,
                                                             alpha=alpha)
            # new_img_ori = plot_filled_rect3d_on_img(new_img, num_rects, rect_corners_temp,  with_borderline=with_borderline_on_filed_bbox, colors=colors, polycolors=polycolors, thickness=thickness, alpha=alpha)
            # mse = calculate_mse(new_img, new_img_ori)
            # print(f"filed均方误差: {mse}")
        else:
            new_img = plot_rect3d_on_img_pillow_fixed(new_img, num_rects, rect_corners_temp, colors, thickness, mode,
                                                      alpha)
            # new_img_ori = plot_rect3d_on_img(new_img, num_rects, rect_corners_temp, colors, thickness, mode, alpha)
            # mse = calculate_mse(new_img, new_img_ori)
            # print(f"line均方误差: {mse}")
        # # 检查是否捕获到了 RuntimeWarning
        # for warning in w:
        #     if issubclass(warning.category, RuntimeWarning):
        #         print("捕获到 RuntimeWarning:", warning.message)
        #         import pdb;pdb.set_trace()

    # new_img = plot_rect3d_on_img(new_img, num_rects, rect_corners, colors, thickness, mode=mode)

    crop_img = new_img[top_left_y: top_left_y + original_height, top_left_x: top_left_x + original_width]

    return crop_img.astype(np.uint8)  # , rect_corners_temp

class MyCamera3DPointsTransfer(Camera3DPointsTransfer):
    printlog = True
    # printlog = False
    @classmethod
    def setprintlog(cls, printlog):
        cls.printlog = printlog

    # @classmethod
    # def transfer_to_pinhole(cls, camera_3ds, camera_intrinsic, camera_dist=[]):
    #     point_2d = camera_3ds @ camera_intrinsic.T

    #     # point_depth = torch.clamp(point_2d[..., 2:3], min=1e-3, max=1e5)
    #     # points_clamp = torch.cat([point_2d[..., :2], point_depth], dim=1)
    #     points_clamp = point_2d
    #     point_2d_res = points_clamp[..., :2] / (points_clamp[..., 2:3] + 1e-8)
    #     # if (point_2d_res<0).sum():
    #     #     import pdb;pdb.set_trace()

    #     #if cls.printlog:print(f'point_2d:{point_2d}\npoint_depth:{point_depth}\npoints_clamp:{points_clamp}\npoint_2d_res:{point_2d_res}')
    #     return point_2d_res

    def transfer_to_pinhole_box(self, camera_3ds, camera_intrinsic, z_min=1e-3):
        """
        将 3D 点投影到针孔相机模型的像素坐标，处理负深度点和小深度点。
        同时确保 3D Bounding Box 的形状保持完整。

        Args:
            camera_3ds (np.ndarray): 3D 点，形状为 [8, 3]，表示多个 3D Bounding Box。
            camera_intrinsic (np.ndarray): 相机内参矩阵 (3x3)。
            z_min (float): 最小深度值，用于处理小深度点。

        Returns:
            list[np.ndarray]: 投影后的 2D 像素坐标，每个元素形状为 [8, 2]。
        """

        def clip_line_to_z_plane(p1, p2, z_min):
            """
            裁剪跨越 z=0 平面的线段，并投影到像素平面。

            Args:
                p1, p2 (np.ndarray): 线段的起点和终点 (3,)。
                z_min (float): 最小深度值。

            Returns:
                tuple: 裁剪后的起点和终点 (p1_clipped, p2_clipped)。
            """
            z1, z2 = p1[2], p2[2]

            # 如果线段完全在相机后方，丢弃
            if z1 <= 0 and z2 <= 0:
                return None, None

            # 如果线段跨越 z=0 平面，计算交点
            if z1 <= 0 or z2 <= 0:
                t = z1 / (z1 - z2)
                intersection = p1 + t * (p2 - p1)
                if z1 <= 0:
                    p1 = intersection
                else:
                    p2 = intersection
            # # 确保深度大于 z_min
            # p1[2] = max(p1[2].item(), z_min)
            # p2[2] = max(p2[2].item(), z_min)

            return p1, p2

        def project_point(point, intrinsic, z_min):
            """
            将 3D 点投影到像素平面。

            Args:
                point (np.ndarray): 3D 点，形状为 (3,)。
                intrinsic (np.ndarray): 相机内参矩阵。
                z_min (float): 最小深度值。

            Returns:
                np.ndarray: 投影后的 2D 像素坐标，形状为 (2,)。
            """
            z = point[2]
            # z = max(point[2], z_min)  # 确保深度不小于 z_min
            point_2d = intrinsic @ point  # 相机内参投影
            return point_2d[:2] / z  # 齐次归一化

        projected_boxes = -1 * torch.zeros(8, 2)
        print('camera_3ds:', camera_3ds.shape)
        box = camera_3ds
        projected_box = []
        line_indices = [
            (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
            (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)
        ]
        for start, end in line_indices:
            # 当前点和下一个点
            p1 = box[start]
            p2 = box[end]

            # 裁剪线段到可见范围
            p1_clipped, p2_clipped = clip_line_to_z_plane(p1, p2, z_min)
            if p1_clipped is None or p2_clipped is None:
                continue  # 跳过不可见线段

            # 投影起点和终点
            p1_2d = project_point(p1_clipped, camera_intrinsic, z_min)
            p2_2d = project_point(p2_clipped, camera_intrinsic, z_min)

            projected_boxes[start] = p1_2d
            projected_boxes[end] = p2_2d

        return projected_boxes


class MaskLiDARInstance3DBoxes(LiDARInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def intersect_plane(self, p1, p2, z_target=1e-5):
        """
        裁剪跨越 z=z_target 平面的线段，并投影到像素平面。

        Args:
            p1, p2 (np.ndarray): 线段的起点和终点 (3,)。
            z_target (float): 目标深度值。

        Returns:
            tuple: 裁剪后的起点和终点 (p1_clipped, p2_clipped)。
        """
        z1, z2 = p1[2], p2[2]

        # 如果线段跨越 z=z_target 平面，计算交点
        if (z1 - z_target) * (z2 - z_target) < 0:  # z1 和 z2 在 z_target 两侧
            t = (z_target - z1) / (z2 - z1)
            intersection = p1 + t * (p2 - p1)
            if z1 < z_target:
                p1 = intersection
            else:
                p2 = intersection

        return p1, p2

    def img_points(self, camera_type='pinhole', lidar2cam_rt=None, camera_intrinsic=None, camera_intrinsic_dist=None,
                   img_w=None, img_h=None):
        corners = self.corners.clone()

        points = []
        masks = []
        lidar_in_cams = []
        lidar_in_worlds = []
        for idx, corner_points in enumerate(corners):  # (n,8,3)
            # from lidar coordinate to camera coordinate
            corner_points = corner_points.T[0:3, :]  # [3, 8]
            lidar_in_cam = lidar2cam_rt @ np.vstack(
                (corner_points[0:3][:], np.ones_like(corner_points[1])))  # [3, 4]*[4, 8]->[3,8]
            # only show point cloud in front of the camera
            # lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]
            pc_xyz_ = lidar_in_cam[:3, :]

            pc_xyz_proj = pc_xyz_.transpose().copy()

            # if torch.Tensor(camera_intrinsic_dist).numel() == 0:  # 针孔相机
            for start, end in line_indices:
                # 当前点和下一个点
                p1 = pc_xyz_.transpose()[start]
                p2 = pc_xyz_.transpose()[end]

                # 裁剪线段到可见范围
                pc_xyz_proj[start], pc_xyz_proj[end] = self.intersect_plane(p1, p2, z_target=1e-3)

            image_point = MyCamera3DPointsTransfer.transfer_camera3d_to_image(
                camera_3ds=torch.Tensor(pc_xyz_proj),
                camera_intrinsic=torch.Tensor(camera_intrinsic),
                camera_dist=torch.Tensor(camera_intrinsic_dist))

            image_point = image_point.numpy().transpose()

            # if camera_type == 'fisheye':
            #     Camera3DPointsTransfer.transfer_camera3d_to_image()
            #     image_point = fisheye_camera_to_image(pc_xyz_.transpose(), camera_intrinsic, camera_intrinsic_dist)
            #     image_point = image_point.transpose()
            # else:
            #     pc_xyz = pc_xyz_ / pc_xyz_[2, :]
            #     image_point = camera_intrinsic @ pc_xyz
            #     image_point = image_point[:2, :]

            # image_point (3,8)
            image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= img_w)
            image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= img_h)
            image_limit = np.logical_and(image_limit_h, image_limit_w)
            if image_point[:, image_limit].size == 0:
                continue
            points.append(image_point[np.newaxis, :])
            masks.append(idx)
            lidar_in_cams.append(pc_xyz_.transpose())  # [8,3]
            lidar_in_worlds.append(corner_points.T)  # [8,3]

            # if (image_point<0).sum():
            #     import pdb;pdb.set_trace()

        if len(points) > 0:
            return np.concatenate(points), np.asarray(masks), np.stack(lidar_in_cams), np.stack(
                lidar_in_worlds)  # [n,8,3]
        return [], [], [], []  # [n,8,3]

def intersect_plane(p1, p2, z_target=1e-5):
    """
    裁剪跨越 z=z_target 平面的线段，并投影到像素平面。

    Args:
        p1, p2 (np.ndarray): 线段的起点和终点 (3,)。
        z_target (float): 目标深度值。

    Returns:
        tuple: 裁剪后的起点和终点 (p1_clipped, p2_clipped)。
    """
    z1, z2 = p1[2], p2[2]

    # 如果线段跨越 z=z_target 平面，计算交点
    if (z1 - z_target) * (z2 - z_target) < 0:  # z1 和 z2 在 z_target 两侧
        t = (z_target - z1) / (z2 - z1)
        intersection = p1 + t * (p2 - p1)
        if z1 < z_target:
            p1 = intersection
        else:
            p2 = intersection

    return p1, p2

def img_points8points(corners, camera_type='pinhole', lidar2cam_rt=None, camera_intrinsic=None, camera_intrinsic_dist=None, img_w=None, img_h=None):
    points = []
    masks = []
    lidar_in_cams = []
    lidar_in_worlds = []
    for idx, corner_points in enumerate(corners):  # (n,8,3)
        # from lidar coordinate to camera coordinate
        corner_points = corner_points.T[0:3, :]  # [3, 8]
        lidar_in_cam = lidar2cam_rt @ np.vstack(
            (corner_points[0:3][:], np.ones_like(corner_points[1])))  # [3, 4]*[4, 8]->[3,8]
        # only show point cloud in front of the camera
        # lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]
        pc_xyz_ = lidar_in_cam[:3, :]
        pc_xyz_proj = pc_xyz_.transpose().copy()
        # if torch.Tensor(camera_intrinsic_dist).numel() == 0:  # 针孔相机
        for start, end in line_indices:
            # 当前点和下一个点
            p1 = pc_xyz_.transpose()[start]
            p2 = pc_xyz_.transpose()[end]

            # 裁剪线段到可见范围
            pc_xyz_proj[start], pc_xyz_proj[end] = intersect_plane(p1, p2, z_target=1e-3)
        # image_point = Camera3DPointsTransfer.transfer_camera3d_to_image(camera_3ds=torch.Tensor(pc_xyz_proj.transpose()),
        #                                                                 camera_intrinsic=torch.Tensor(camera_intrinsic),
        #                                                                 camera_dist=torch.Tensor(camera_intrinsic_dist))

        image_point = MyCamera3DPointsTransfer.transfer_camera3d_to_image(
            camera_3ds=torch.Tensor(pc_xyz_proj),
            camera_intrinsic=torch.Tensor(camera_intrinsic),
            camera_dist=torch.Tensor(camera_intrinsic_dist))
        image_point = image_point.numpy().transpose()

        # if camera_type == 'fisheye':
        #     Camera3DPointsTransfer.transfer_camera3d_to_image()
        #     image_point = fisheye_camera_to_image(pc_xyz_.transpose(), camera_intrinsic, camera_intrinsic_dist)
        #     image_point = image_point.transpose()
        # else:
        #     pc_xyz = pc_xyz_ / pc_xyz_[2, :]
        #     image_point = camera_intrinsic @ pc_xyz
        #     image_point = image_point[:2, :]

        # image_point (3,8)
        image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= img_w)
        image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= img_h)
        image_limit = np.logical_and(image_limit_h, image_limit_w)
        if image_point[:, image_limit].size == 0:
            continue

        points.append(image_point[np.newaxis, :])
        masks.append(idx)
        lidar_in_cams.append(pc_xyz_.transpose())  # [8,3]
        lidar_in_worlds.append(corner_points.T)  # [8,3]

        # if (image_point<0).sum():
        #     import pdb;pdb.set_trace()

    if len(points) > 0:
        return np.concatenate(points), np.asarray(masks), np.stack(lidar_in_cams), np.stack(lidar_in_worlds)  # [n,8,3]
    return [], [], [], []  # [n,8,3]

#             low_percentile, high_percentile, max_velocity, min_velocity = -32, 32, 1067.7296305671541, -1067.6177489590264
#             #velocity_normalization_decorator(self.segment_annos_root)
def normalize_vector(vector, low_percentile = -32, high_percentile = 32, min_velocity = -1067.6177489590264,  max_velocity = 1067.7296305671541):
    """
    对一个输入的 3 维向量进行归一化。

    参数：
        vector (numpy.ndarray): 输入的 3 维向量。
        low_percentile (float): 主流数据的低百分位值。
        high_percentile (float): 主流数据的高百分位值。
        max_velocity (float): 数据的最大值。
        min_velocity (float): 数据的最小值。

    返回：
        numpy.ndarray: 归一化后的 3 维向量，范围为 [0, 255]。
    """
    # 初始化归一化后的数组
    normalized_vector = np.zeros_like(vector, dtype=np.float64)

    # 主流数据（线性映射到 [10, 245]）
    linear_mask = (vector >= low_percentile) & (vector <= high_percentile)
    normalized_vector[linear_mask] = (
        (vector[linear_mask] - low_percentile) / (high_percentile - low_percentile) * (245 - 10) + 10
    )

    # 下离群点（平滑映射到 [0, 10]）
    below_mask = vector < low_percentile
    if np.any(below_mask):
        normalized_vector[below_mask] = 10 * (
            1 - np.exp(-5 * (low_percentile - vector[below_mask]) / (low_percentile - min_velocity))
        )

    # 上离群点（平滑映射到 [245, 255]）
    above_mask = vector > high_percentile
    if np.any(above_mask):
        normalized_vector[above_mask] = 245 + 10 * (
            1 - np.exp(-5 * (vector[above_mask] - high_percentile) / (max_velocity - high_percentile))
        )

    # 保证所有数据范围在 [0, 255]
    normalized_vector = np.clip(normalized_vector, 0, 255)
    #print(vector, normalized_vector)
    return normalized_vector.astype(np.uint8)


def draw_bbox_alt_camera_meta(bbox_img, meta, camera_name, img_shape,sx = 1, sy = 1,  bboxmode ='8points', draw_bbox_mode='cross', colorful_box=False, thickness=6, alpha=0.5):
    targets = meta["Objects"]
    camera_meta = meta["sensors"]["cameras"][camera_name]
    ego2local = meta.get("ego2local_transformation_matrix", None)
    camera_meta = meta["sensors"]["cameras"][camera_name]
    camera_meta["camera_intrinsic"] = (
                np.array(camera_meta["camera_intrinsic"]) * np.array([sx, sy, 1])[:, np.newaxis]).tolist()
    # img_w = self.img_w_fisheye if "fov195" in camera_name else self.img_w_pinhole
    # img_h = self.img_h_fisheye if "fov195" in camera_name else self.img_h_pinhole
    # img_h, img_w = img_shape
    if img_shape is None:
        img_w = camera_meta["image_width"]
        img_h = camera_meta["image_height"]
    else:
        img_w, img_h = img_shape
    lidar2camera_rt = np.array(camera_meta["extrinsic"])
    camera_intrinsic = np.array(camera_meta["camera_intrinsic"])
    camera_intrinsic_dist = np.array(camera_meta["camera_dist"])
    camera_type = "pinhole" if "195" not in camera_name else "fisheye"

    boxes, classes, localworld_velocity_list = [], [], []
    colors = []
    for target in targets:
        boxes.append(target["bbox3d"])
        if target["label"] in CLASS_BBOX_COLOR_MAP.keys():
            co = CLASS_BBOX_COLOR_MAP[target["label"]]
            label_id = CLASS_BBOX_ID_MAP[target["label"]]
        elif 'VEHICLE_' in target["label"]:
            co = CLASS_BBOX_COLOR_MAP["VEHICLE_TRUCK"]
            label_id = CLASS_BBOX_ID_MAP["VEHICLE_TRUCK"]
        elif "CYCLIST_" in target["label"]:
            co = CLASS_BBOX_COLOR_MAP["CYCLIST_BICYCLE"]
            label_id = CLASS_BBOX_ID_MAP["CYCLIST_BICYCLE"]
        elif "PEDESTRIAN_" in target["label"]:
            co = CLASS_BBOX_COLOR_MAP["PEDESTRIAN_NORMAL"]
            label_id = CLASS_BBOX_ID_MAP["PEDESTRIAN_NORMAL"]
        else:
            print(target["label"])
            co = CLASS_BBOX_COLOR_MAP["OTHER"]
            label_id = CLASS_BBOX_ID_MAP["OTHER"]
        classes.append(label_id)
        if "color" in target.keys():
            colors.append(target["color"])
        else:
            colors.append(co)

        if ego2local is not None:
            # for vel
            velocity = np.asarray(target["velocity"])
            localworld_velocity = ego2local[:3, :3] @ velocity[:, None]
            localworld_velocity_list.append(np.squeeze(localworld_velocity[:3]))

    # colors = []
    # if targets is not None:
    #     for target in targets:
    #         # NOTE - Only Draw visualizable objects
    #         boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])
    # print(len(boxes))

    if len(boxes) > 0:
        if bboxmode == '8points':
            img_points, masks, lidar_in_cams, _ = img_points8points(
                corners = torch.Tensor(boxes),
                camera_type=camera_type,
                lidar2cam_rt=lidar2camera_rt,
                camera_intrinsic=camera_intrinsic,
                camera_intrinsic_dist=camera_intrinsic_dist,
                img_w=img_w,
                img_h=img_h,
            )
        else:
            lidar_boxes = MaskLiDARInstance3DBoxes(torch.Tensor(boxes), origin=(0.5, 0.5, 0.5))
            img_points, masks, lidar_in_cams, _ = lidar_boxes.img_points(
                camera_type=camera_type,
                lidar2cam_rt=lidar2camera_rt,
                camera_intrinsic=camera_intrinsic,
                camera_intrinsic_dist=camera_intrinsic_dist,
                img_w=img_w,
                img_h=img_h,
            )
        # _, camera_meta, lidar_in_cams, classes = info

        if len(img_points) > 0:
            if colorful_box:
                box_colors = [colors[i] for i in masks]
            else:
                box_colors = [(21, 217, 211)]

            box_classes = [classes[i] for i in masks]
            # ===========================================================================
            # *                            Draw Bbox
            # ===========================================================================
            draw_img = expand_plot_rect3d_on_img(
                bbox_img, img_points.shape[0], img_points.transpose(0, 2, 1), \
                thickness=thickness, colors=box_colors, mode=draw_bbox_mode, alpha=alpha, with_filed=False
            )
            # ===========================================================================
            # *                            Draw Bbox
            # ===========================================================================

            # ===========================================================================
            # *                            Draw Traj
            # ===========================================================================
            if ego2local is not None:
                box_localworld_velocity = [localworld_velocity_list[i] for i in masks]
                box_localworld_velocity = np.asarray(box_localworld_velocity)  # [n,3]
                '''
                min_box_localworld_velocity, max_box_localworld_velocity = self.min_localworld_velocity, self.max_localworld_velocity
                # max_box_localworld_velocity = 100#np.max(box_localworld_velocity)
                # min_box_localworld_velocity = -100#np.min(box_localworld_velocity)
                box_localworld_velocity = np.clip(box_localworld_velocity, min_box_localworld_velocity, max_box_localworld_velocity)
                #print(max_box_localworld_velocity, min_box_localworld_velocity)
                norm_box_localworld_velocity = (box_localworld_velocity - min_box_localworld_velocity)/(max_box_localworld_velocity - min_box_localworld_velocity + 1e-8)
                #np.clip(((colors) * 255.).astype(np.uint8), 0, 255)
                color_norm_box_localworld_velocity = np.clip((norm_box_localworld_velocity*255).astype(np.uint8), 0, 255)
                '''
                color_norm_box_localworld_velocity = normalize_vector(box_localworld_velocity).astype(np.uint8)
                # print(f"color_norm_box_localworld_velocity: \n min {color_norm_box_localworld_velocity.min()}, max {color_norm_box_localworld_velocity.max()}, array {color_norm_box_localworld_velocity}")
                polycolors = color_norm_box_localworld_velocity.tolist()
                # print(f'---------------Out--{polycolors}------------------')
                traj_image_map = expand_plot_rect3d_on_img(
                    traj_image, img_points.shape[0], img_points.transpose(0, 2, 1), \
                    thickness=thickness, colors=polycolors, polycolors=polycolors, mode=draw_bbox_mode, \
                    with_filed=True, with_borderline_on_filed_bbox=False, alpha=1
                )
                # traj_image_map = expand_plot_rect3d_on_img(
                #     traj_image, img_points.shape[0], img_points.transpose(0, 2, 1), \
                #     thickness=thickness, colors=box_colors, polycolors=box_colors, mode=draw_bbox_mode, \
                #     with_filed=False, with_borderline_on_filed_bbox=True, alpha=1
                # )
            else:
                pass
            # ===========================================================================
            # *                            Draw Traj
            # ===========================================================================
            # if box_colors != [(21, 217, 211)]:print('box_colors:', box_colors)
            # print('img_points.shape, box_colors.shape:', len(img_points), len(box_colors))

        else:
            draw_img = bbox_img
            lidar_in_cams = []
            box_classes = []
            rect_corners_temp = np.zeros((1, 8, 2))

    else:
        draw_img = bbox_img
        lidar_in_cams = []
        box_classes = []
        rect_corners_temp = np.zeros((1, 8, 2))
    # print(rect_corners_temp, rect_corners_temp.shape, type(rect_corners_temp), type(rect_corners_temp[0]))
    return np.uint8(draw_img), dict(
                                    camera_meta=camera_meta,
                                    lidar_in_cams=lidar_in_cams,
                                    box_classes=box_classes,
                                    )  # , rect_corners_temp

# self.cam_order2pap = {
#     "CAM_FRONT": "center_camera_fov120",
#     "CAM_FRONT_RIGHT": "right_front_camera",
#     "CAM_BACK_RIGHT": "right_rear_camera",
#     "CAM_BACK": "rear_camera",
#     "CAM_BACK_LEFT": "left_rear_camera",
#     "CAM_FRONT_LEFT": "left_front_camera"
# }
# # img_shape = CARLA_IMAGE_SHAPE
# # img_shape =
# targets = [] # ins x 8 x 3
# camera_name = "center_camera_fov120"
# with open('pap_cam_intri_extri.json', 'r') as f:
#     meta = json.load(f)
#
# out = draw_bbox_alt_camera_meta(targets, meta, camera_name, img_shape, bboxmode ='8points')
# save_local_dir = "/mnt/iag/user/xujin2/code_fj1/open_source/metavdt/debug_dir"