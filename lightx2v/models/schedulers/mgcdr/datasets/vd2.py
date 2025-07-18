import io
import os

from toolz.curried import peekn
from tqdm import tqdm
import json
import numpy as np
import torch
import cv2
from typing import Literal, List, Dict
from matplotlib import cm
# from calib_helper  import CameraIntrinsic
from PIL import Image, ImageDraw
from einops import rearrange
from torchvision.utils import save_image
import random
import pickle as pkl
from torchvision import transforms
import warnings
import time
from colorama import Fore, Style
import copy
import magicdrivedit.utils.aoss
from magicdrivedit.registry import DATASETS, build_module
from mmcv.parallel import DataContainer
import glob
import math
from math import cos, sin
import traceback

IMG_FPS = 100
#   语义尽可能对齐pap
# CLASS_BBOX_COLOR_MAP = {
#     # 未知/其他/结束类（灰色）
#     'UNKNOWN': (128, 128, 128),  # 灰色
#     'VEHICLE_OTHERS': (128, 128, 128),  # 灰色
#     'VEHICLE_END': (128, 128, 128),  # 灰色
#     'BIKE_END': (128, 128, 128),  # 灰色
#     'NO_PERSON_VEHICLE': (128, 128, 128),  # 灰色
#     'MOTOR': (128, 128, 128),  # 灰色 # 不知道为啥会有这个类。。。
#     # 行人（蓝色系）
#     'PEDESTRIAN': (0, 0, 255),  # 纯蓝
#     # 标准乘用车（红色系）
#     'VEHICLE': (255, 0, 0),  # 纯红（基础车辆）
#     'VEHICLE_CAR': (255, 0, 0),  # 亮红（轿车）
#     'VEHICLE_SUV': (255, 102, 102),  # 浅红（SUV）
#     'VEHICLE_TAXI': (204, 0, 0),  # 暗红（出租车）
#     'VEHICLE_POLICE': (255, 0, 127),  # 品红（警车）
#     # 商用/大型车辆（橙色系）
#     'VEHICLE_VAN': (255, 153, 51),  # 橙黄（厢式车）
#     'VEHICLE_TRUCK': (255, 128, 0),  # 标准橙（卡车）
#     'VEHICLE_PICKUP_TRUCK': (255, 178, 102),  # 浅橙（皮卡）
#     'VEHICLE_EMERGENCY': (255, 165, 0),  # 金色（应急车辆）
#     # 公共交通（黄色系）
#     'VEHICLE_BUS': (255, 255, 0),  # 纯黄（巴士）
#     'VEHICLE_SCHOOL_BUS': (255, 255, 153),  # 浅黄（校车）
#     # 非机动车（绿色系）
#     'BIKE': (0, 255, 0),  # 纯绿（基础非机动车）
#     'BIKE_BICYCLE': (102, 255, 102),  # 浅绿（自行车）
#     'BIKE_BIKEBIG': (0, 153, 0),  # 深绿（大型非机动车）
#     'BIKE_BIKESMALL': (153, 255, 153),  # 荧光绿（小型非机动车）
# }

CLASS_BBOX_COLOR_MAP = {
    # 未知/其他/结束类（灰色）
    'UNKNOWN': (128, 128, 128),
    'VEHICLE_OTHERS': (128, 128, 128),
    'VEHICLE_END': (128, 128, 128),
    'BIKE_END': (128, 128, 128),
    'NO_PERSON_VEHICLE': (128, 128, 128),
    'MOTOR': (0, 255, 0),  # 绿，对应 CYCLIST_MOTOR

    # 行人（蓝色系）
    'PEDESTRIAN': (0, 0, 255),  # 对应 PEDESTRIAN_NORMAL
    'PEDESTRIAN_TRAFFIC_POLICE': (0, 0, 255),  # 对应 PEDESTRIAN_TRAFFIC_POLICE

    # 乘用车（红色系）
    'VEHICLE': (255, 0, 0),  # 统一红色，基础车辆
    'VEHICLE_CAR': (255, 0, 0),  # 轿车
    'VEHICLE_SUV': (255, 0, 0),  # SUV
    'VEHICLE_TAXI': (255, 0, 0),  # 出租车（与轿车同色）
    'VEHICLE_POLICE': (255, 0, 0),  # 警车

    # 商用/大型车辆（橙色系）
    'VEHICLE_VAN': (255, 128, 0),  # 厢式车，对应 VEHICLE_SPECIAL
    'VEHICLE_TRUCK': (255, 128, 0),  # 卡车，对应 VEHICLE_TRUCK
    'VEHICLE_PICKUP_TRUCK': (255, 128, 0),  # 皮卡，对应 VEHICLE_PICKUP
    'VEHICLE_EMERGENCY': (255, 128, 0),  # 应急车辆，对应 VEHICLE_SPECIAL
    'VEHICLE_MULTI_STAGE': (255, 128, 0),  # 多段车单节车体
    'VEHICLE_TRAILER': (255, 128, 0),  # 拖车
    'VEHICLE_CAR_CARRIER_TRAILER': (255, 128, 0),  # 拖挂车
    'VEHICLE_RUBBISH': (255, 128, 0),  # 垃圾车

    # 公共交通（黄色系）
    'VEHICLE_BUS': (255, 255, 0),  # 巴士
    'VEHICLE_SCHOOL_BUS': (255, 255, 0),  # 校车，保持一致

    # 非机动车（绿色系）
    'BIKE': (0, 255, 0),  # 统一绿色，基础非机动车
    'BIKE_BICYCLE': (0, 255, 0),  # 自行车，对应 CYCLIST_BICYCLE
    'BIKE_BIKEBIG': (0, 255, 0),  # 大型非机动车
    'BIKE_BIKESMALL': (0, 255, 0),  # 小型非机动车
    'VEHICLE_TRIKE': (0, 255, 0)  # 三轮车，对应 VEHICLE_TRIKE
}


CLASS_PRED = {
    0: 'UNKNOWN',
    1: 'PEDESTRIAN',
    2: 'VEHICLE',
    3: 'VEHICLE_CAR',
    4: 'VEHICLE_SUV',
    5: 'VEHICLE_VAN',
    6: 'VEHICLE_TRUCK',
    7: 'VEHICLE_PICKUP_TRUCK',
    8: 'VEHICLE_BUS',
    9: 'VEHICLE_TAXI',
    10: 'VEHICLE_EMERGENCY',
    11: 'VEHICLE_SCHOOL_BUS',
    12: 'VEHICLE_OTHERS',
    13: 'VEHICLE_END',
    14: 'BIKE',
    15: 'NO_PERSON_VEHICLE',
    16: 'BIKE_BICYCLE',
    17: 'BIKE_BIKEBIG',
    18: 'BIKE_BIKESMALL',
    19: 'BIKE_END'
}
# 先改成16以内
BBOX_2_id = {
    # 未知/其他/结束类（灰色）
    'UNKNOWN': 0,  # 灰色
    'VEHICLE_OTHERS': 0,  # 灰色
    'VEHICLE_END': 0,  # 灰色
    'BIKE_END': 0,  # 灰色
    'NO_PERSON_VEHICLE': 0,  # 灰色
    'MOTOR': 0,  # 灰色 # 不知道为啥会有这个类。。。
    # 行人（蓝色系）
    'PEDESTRIAN': 1,  # 纯蓝
    # 标准乘用车（红色系）
    'VEHICLE': 3,  # 纯红（基础车辆）
    'VEHICLE_CAR': 3,  # 亮红（轿车）
    'VEHICLE_SUV': 4,  # 浅红（SUV）
    'VEHICLE_TAXI': 9,  # 暗红（出租车）
    'VEHICLE_POLICE': 4,  # 品红（警车）
    # 商用/大型车辆（橙色系）
    'VEHICLE_VAN': 5,  # 橙黄（厢式车）
    'VEHICLE_TRUCK': 6,  # 标准橙（卡车）
    'VEHICLE_PICKUP_TRUCK': 7,  # 浅橙（皮卡）
    'VEHICLE_EMERGENCY': 10,  # 金色（应急车辆）
    # 公共交通（黄色系）
    'VEHICLE_BUS': 8,  # 纯黄（巴士）
    'VEHICLE_SCHOOL_BUS': 11,  # 浅黄（校车）
    # 非机动车（绿色系）
    'BIKE': 14,  # 纯绿（基础非机动车）
    'BIKE_BICYCLE': 14,  # 浅绿（自行车）
    'BIKE_BIKEBIG': 14,  # 深绿（大型非机动车）
    'BIKE_BIKESMALL': 14,  # 荧光绿（小型非机动车）
}

# bgr 用于opencv 的颜色
GEO2COLOR = {
    'WHITE': (255, 255, 255), # RGB(255,255,255) → BGR无变化[4](@ref)，用于同向车道分隔线（虚线可越线变道，实线禁止越线）
    'RED': (0, 0, 255),       # RGB(255,0,0) → BGR(0,0,255)[4](@ref)，用于禁令标志边框
    'YELLOW': (0, 255, 255),  # RGB(255,255,0) → BGR(0,255,255)[4](@ref)，用于对向车道分隔线
    'GREEN': (0, 255, 0),     # RGB(0,255,0) → BGR无变化[4](@ref)，用于导向箭头填充色
    'BLUE': (255, 0, 0),      # RGB(0,0,255) → BGR(255,0,0)[4](@ref)，用于特殊车道标识
    'BLACK': (0, 0, 0),       # 所有通道为0[4](@ref)，用于文字符号描边
    'ORANGE': (0, 165, 255)   # RGB(255,165,0) → BGR(0,165,255)[4](@ref)，用于施工区标线
}

# r g b
GEO_ID2COLOR = {
    "LANELINE":  (0, 255, 0),             # 车道线     绿色
    "ROADSIDE":  (255, 0, 0),             # 路沿       红色
    "CROSSWALK": (255,255,255),        # 人行横道    白色
    "STOPLINE":  (255,255,255),         # 停止线     白色
}
# pap
GEO_COLOR = {
    "LANELINE": (0, 255, 0),
    "ROADSIDE": (255, 0, 0)
}

# 投影相关：
def proj_pvb(img, all_bbox, all_classes, camera_intrinsic, lidar_camera_extrinsic, id2color = None):
    all_bbox_8point_in_cam = []
    all_classes_id_in_cam = []
    # get corners
    vis_object = []
    for gt_box, gt_name in zip(all_bbox, all_classes):
        if len(gt_box) == 7: # x, y, z, l, w, h, yaw
            # get corners
            corners_gt_box = get_coners(gt_box)  # to 8定点
        elif len(gt_box) == 8:
            corners_gt_box = gt_box
        else:
            assert 0
        image_points, point_3d_in_cam = lidar2image(
            corners_gt_box, camera_intrinsic, lidar_camera_extrinsic)
        if image_points is not None:
            vis_object.append(image_points)
            all_bbox_8point_in_cam.append(point_3d_in_cam.T)
            all_classes_id_in_cam.append(gt_name)
    # 画图，画在img上
    out_h, out_w, _ = img.shape
    for obj, obj_id in zip(vis_object, all_classes_id_in_cam):
        if obj is not None:
            if id2color is not None:
                color = id2color[obj_id]
            else:
                color = (255, 0, 0)
            img = draw_rect3d_on_img(img, obj, out_h, out_w, color)
    return img, all_bbox_8point_in_cam, all_classes_id_in_cam

def lidar2image(points_3d, camera_intrinsic, lidar_camera_extrinsic):
    # projection
    corners_aug = np.column_stack((points_3d, np.ones((len(points_3d), 1))))
    image_points, are_projection_valids, points_3d_incam = project_to_camera(
        corners_aug.T, np.array(camera_intrinsic), lidar_camera_extrinsic, fov_x=None)
    image_points = image_points[np.array(are_projection_valids)[0]]
    if len(image_points) <= 3:
        return None, None
    image_points = np.delete(image_points, -1, axis=1)
    return image_points, points_3d_incam


def project_to_camera(points,
                      camera_intrinsic,
                      sensor_extrinsic,
                      fov_x=None):
    """ Project points into image.

    Args:
        points: 4xN numpy array
        camera_intrinsic: 3x3 numpy array
        sensor_extrinsic: 4x4 numpy array

    Returns:
        List of 2d points
        Z >= 0
    """
    # sensor to camera, 4xN
    camera_points = sensor_extrinsic.dot(np.array(points))

    # z is front in camera, 3xN
    camera_points = camera_points[:-1]
    # camera to image, 3xN
    img_points = camera_intrinsic.dot(camera_points)
    img_points = np.array(img_points)

    img_points[0, :] /= img_points[2, :]
    img_points[0, img_points[2, :] < 0] = -1
    img_points[1, :] /= img_points[2, :]
    img_points[1, img_points[2, :] < 0] = -1
    img_points[2, :] = 1
    img_points = img_points.transpose()  # Nx3 array
    return img_points, camera_points[2, :] >= 0, camera_points

def get_coners(bbox_3d):
    x, y, z, l, w, h, yaw = bbox_3d
    w_half, l_half, h_half = w / 2, l / 2, h / 2
    corners = []
    corners.append(np.array([l_half, w_half, h_half]))
    corners.append(np.array([l_half, -w_half, h_half]))
    corners.append(np.array([-l_half, -w_half, h_half]))
    corners.append(np.array([-l_half, w_half, h_half]))
    corners.append(np.array([l_half, w_half, -h_half]))
    corners.append(np.array([l_half, -w_half, -h_half]))
    corners.append(np.array([-l_half, -w_half, -h_half]))
    corners.append(np.array([-l_half, w_half, -h_half]))
    rot = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw),
                     math.cos(yaw), 0], [0, 0, 1]])
    obj_center = np.array([x, y, z])

    # project to image
    corners = np.array(corners)
    corners = np.dot(rot, corners.T).T + obj_center

    return corners

def draw_rect3d_on_img(image,
                       rect_corners,
                       image_w,
                       image_h,
                       color=(0, 255, 0),
                       thickness=1):
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

    if len(rect_corners) != 8:
        # draw outsider
        bbox_lu, bbox_rd = [image_w, image_h], [0, 0]
        for i in range(len(rect_corners)):
            bbox_lu[0] = max(min(bbox_lu[0], rect_corners[i][0]), 0)
            bbox_lu[1] = max(min(bbox_lu[1], rect_corners[i][1]), 0)
            bbox_rd[0] = min(
                max(bbox_rd[0], rect_corners[i][0]), image_w)
            bbox_rd[1] = min(
                max(bbox_rd[1], rect_corners[i][1]), image_h)

        if np.any((np.array(bbox_rd) - np.array(bbox_lu)) <= 0):
            return image.astype(np.uint8)
        cv2.rectangle(image, (int(bbox_lu[0]), int(bbox_lu[1])), (int(
            bbox_rd[0]), int(bbox_rd[1])), color, 2)
        return image.astype(np.uint8)

    #
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    # heading_line_indices = ((4, 6), (5, 7))
    heading_line_indices = ((0, 5), (1, 4))

    corners = rect_corners.astype(int)
    try:
        for start, end in line_indices:
            cv2.line(image, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)
    except:
        print(f'a wired bbox appeared: {corners}')
    try:
        for start, end in heading_line_indices:
            cv2.line(image, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)
    except:
        print(f'a wired heading line appeared: {corners}')
    return image.astype(np.uint8)


def proj_geo(img, all_lines, all_classes, camera_intrinsic, lidar2camera_rt, thickness):
    for line, line_id in zip(all_lines, all_classes):
        line = np.array(line)
        draw_lane_line(img, line, line_id, camera_intrinsic, lidar2camera_rt, THICKNESS = thickness)
    return img

def draw_lane_line(image, line, line_id, camera_intrinsic, lidar_camera_extrinsic, THICKNESS=2):
    points, _ = lidar2image(line, camera_intrinsic, lidar_camera_extrinsic)
    if points is not None:
        h, w, _ = image.shape
        # Coarse filtration
        # valid = (points[:, 2] > 0) & (points[:, 0] < 2 * w) & (points[:, 1] < 2 * h)
        # valid = (points[:, 0] < 2 * w) & (points[:, 1] < 2 * h)
        # points = points[valid]
        if points.size <= 1:
            return
        for i in range(len(points) - 1):
            x1 = int(points[i][0])
            y1 = int(points[i][1])
            x2 = int(points[i + 1][0])
            y2 = int(points[i + 1][1])
            try:
                cv2.line(
                    image,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    # color=GT_COLOR.get(line_id, (0, 0 , 0)),
                    color=line_id, # 这里id 就是color
                    thickness=THICKNESS,
                    lineType=cv2.LINE_AA,
                )
            except Exception as e:
                traceback.print_exc()
    return

def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    B, N_out = bbox_shape[:2]
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    if bboxes is not None and len(bboxes[0][0]) > 0 and max_len > 0:
        for _b in range(B):
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            for _n in range(N_out):
                if _bboxes[_n] is None:
                    continue  # empty for this view
                try:
                    this_box_num = len(_bboxes[_n])
                    if this_box_num > 0:
                        ret_bboxes[_b, _n, :this_box_num] = torch.tensor(_bboxes[_n]).to(ret_bboxes)
                        ret_classes[_b, _n, :this_box_num] = torch.tensor(_classes[_n]).to(ret_classes)
                        if masks is not None:
                            ret_masks[_b, _n, :this_box_num] = torch.tensor(masks[_b, _n]).to(ret_masks)
                        else:
                            ret_masks[_b, _n, :this_box_num] = True
                except Exception as e:
                    traceback.print_exc()
                    # print(e)

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,  # [B, N_out, max_len, 8 x 3]
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict

# for ego pose
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵
    假设欧拉角顺序为：yaw -> pitch -> roll
    """
    # 绕 Z 轴的旋转矩阵 (yaw)
    Rz = np.array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])

    # 绕 Y 轴的旋转矩阵 (pitch)
    Ry = np.array([[cos(pitch), 0, sin(pitch)],
                   [0, 1, 0],
                   [-sin(pitch), 0, cos(pitch)]])

    # 绕 X 轴的旋转矩阵 (roll)
    Rx = np.array([[1, 0, 0],
                   [0, cos(roll), -sin(roll)],
                   [0, sin(roll), cos(roll)]])

    # 总旋转矩阵为三个旋转矩阵的乘积
    R = Rz @ Ry @ Rx
    return R

def quaternion_to_rotation_matrix(q):
    """
    将单位四元数转换为3x3旋转矩阵
    输入：q = [w, x, y, z]（需保证四元数已归一化）
    输出：3x3旋转矩阵
    """
    w, x, y, z = q
    # 计算平方项（优化计算速度）
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    # 构建旋转矩阵（根据四元数分量推导）
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return rotation_matrix

def get_ego2global_transformation_matrix(R_quaternion, T):
    R = quaternion_to_rotation_matrix(R_quaternion)
    ego2global_transformation_matrix = np.eye(4)
    ego2global_transformation_matrix[:3, :3] = R  # 旋转矩阵
    ego2global_transformation_matrix[:3, 3] = T  # 平移向量
    return ego2global_transformation_matrix

def convert_matrix_b_to_a(K_b):
    """ 将 B 数据集的相机外参转换为 A 坐标系 """  #例如 B 数据集（右手坐标系） 的所有数据转换到 A 数据集（左手坐标系）
    M = np.array([[1,  0,  0,  0],
                  [0, -1,  0,  0],
                  [0,  0,  1,  0],
                  [0,  0,  0,  1]])
    K_a = M @ K_b @ np.linalg.inv(M)
    return K_a

def convert_camera_intrinsics_b_to_a(K_b, image_size):
    """
    将 B 数据集的相机内参 K (右手坐标系) 转换为 A 数据集 (左手坐标系)
    :param K_b: 3x3 相机内参矩阵 (B 数据集)
    :param image_height: 图像高度 (像素)
    :return: 转换后的 K_a (A 数据集)
    """
    w, h = image_size
    K_a = K_b.copy()
    K_a[1, 1] = -K_a[1, 1]  # 反转 f_y
    K_a[1, 2] = h - K_a[1, 2]  # 调整光心 c_y
    return K_a

def convert_3dbbox_batch_b_to_a(bboxs_b):
    """
    批量转换 B 数据集的 3D BBox (n x 7) 到 A 坐标系
    :param bboxs_b: (n, 7) 的列表或 NumPy 数组，每行 (x, y, z, l, w, h, yaw) (右手系 B)
    :return: (n, 7) NumPy 数组，每行 (x, -y, z, l, w, h, -yaw) (左手系 A)
    """
    bboxs_b = np.array(bboxs_b)  # 转换为 NumPy 数组，保证操作高效
    bboxs_a = bboxs_b.copy()

    # 变换 y 轴: y -> -y
    bboxs_a[:, 1] = -bboxs_a[:, 1]

    # 变换 yaw: yaw -> -yaw
    bboxs_a[:, 6] = -bboxs_a[:, 6]

    return bboxs_a

def calc_dist_by_e2w(matrix1, matrix2):
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    pos1 = matrix1[:3, 3]
    pos2 = matrix2[:3, 3]
    distance = np.linalg.norm(pos2 - pos1)
    return distance

def align_camera_poses(tensor):
    T1 = tensor[0]
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    T1_inv = np.eye(4)
    T1_inv[:3, :3] = R1.T
    T1_inv[:3, 3] = -R1.T @ t1  # -R^T * t

    transformed_tensors = np.zeros_like(tensor)
    for i in range(tensor.shape[0]):
        transformed_tensors[i] = T1_inv @ tensor[i]
    return transformed_tensors

@DATASETS.register_module()
class VD2VariableDataset(object):
    def __init__(self,
                 raw_meta_files,
                 caption_dir,
                 path_to_aoss_config,
                 s3_path,
                 camera_list=None,
                 sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"],
                 sequence_length=[1, 9, 17, 33],
                 fps_list=[10],
                 data_fps=10,
                 split="train", # * 'test'就不会随机采首帧
                 use_random_seed=False,
                 expected_vae_size=(256, 512),
                 full_size=None,
                 user_frame_emb = True,
                 **kwargs):
        self.reader = magicdrivedit.utils.aoss.AossSimpleFile(
            client_config_path=path_to_aoss_config, s3_path=s3_path)
        self.camera_list = camera_list
        self.sqrt_required_text_keys = sqrt_required_text_keys
        self.seq_length = sequence_length
        self.fps_list = fps_list
        self.data_fps = data_fps
        self.split = split
        self.use_random_seed = use_random_seed
        self.expected_vae_size = expected_vae_size
        self.full_size = full_size
        self.user_frame_emb = user_frame_emb
        self.with_bbox_coords = kwargs.get('with_bbox_coords', True)
        self.with_camera_param = kwargs.get("with_camera_param", False)
        self.save_annos = kwargs.get('save_annos', False) # infer 生产数据用
        self.convert_left_hand = kwargs.get('convert_left_hand', False)
        self.caption_join_func = kwargs.get("caption_join_func", None)
        if self.caption_join_func is None:
            self.caption_join_func = self.joint_caption
        elif self.caption_join_func == "format_caption":
            self.caption_join_func = self.format_caption
        else:
            print(f"error self.caption_join_func: {self.caption_join_func}")
            assert False
        self.add_controller = kwargs.get("add_controller", False)
        if max(self.expected_vae_size) > 768:
            self.thickness = 6
        else:
            self.thickness = 2
        self.pvb_bbox2id = BBOX_2_id
        # for k,v in CLASS_PRED.items():
        #     self.pvb_CLASS_PRED_inv[v] = k

        self.transforms = transforms.Compose([
            transforms.Resize(expected_vae_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.condition_transform = transforms.Compose([
            # transforms.Lambda(lambda img: self.resize_nearest(img, (self.expected_vae_size[0], self.expected_vae_size[1]))), # 暂时写死，分辨率可以比图像小
            transforms.Lambda(lambda img: self.resize_nearest(img, (256, 512))), # 暂时写死，分辨率可以比图像小
            transforms.ToTensor()
        ])
        self.index_info = {}
        if self.use_random_seed:
            random.seed(time.time()%100000)
        print('vd2 self.seq_length:', self.seq_length)
        print(Style.BRIGHT + Fore.RED + f'self.fps_list:{self.fps_list} in data_fps:{self.data_fps}' + Style.RESET_ALL)
        self.specific_video_segment = None
        self.segment_infos = []
        self.segment_data_infos = {}
        self.max_seq_length = max(self.seq_length) if isinstance(self.seq_length, list) else self.seq_length
        for annn_root in raw_meta_files:
            print(f"load : {annn_root}")
            if ".pkl" in annn_root:
                with open(annn_root, "rb") as f:
                    loaded_data = pkl.load(f)
                self.segment_infos += loaded_data
                continue
            elif ".json" in annn_root:
                with open(annn_root, "r") as f:
                    loaded_data = json.load(f)
                self.segment_infos += loaded_data
                continue

            # 读取远程s3地址
            if "s3://" in annn_root:
                pass # 预留
            # 读取本地afs
            else:
                case_list = [path for path in os.listdir(annn_root) if path.endswith('.json')]
                for case_name in tqdm(case_list):
                    case_root = annn_root
                    case_infor = json.load(open(os.path.join(annn_root, case_name), "rb"))
                    case_num_frames = len(case_infor)
                    if case_num_frames < self.max_seq_length:
                        print(f"continue: {case_name} : {case_num_frames} < {self.max_seq_length}")
                        continue
                    self.segment_infos.append({
                        "case_name": case_name.replace(".json", ""),
                        "case_root": case_root,
                        "case_num_frames": case_num_frames,
                    })
                # 存储一遍pkl文件下次读取直接读pkl
                with open(annn_root + ".json", "w",  encoding='utf-8') as f:
                    json.dump(self.segment_infos, f, indent=4)
                annn_root_save = annn_root +  ".json"
                print(f"save to json file =>>>: {annn_root_save}")
        # 预留，加載caption 匹配过滤
        self.scene_description_data = {}
        # 遍历目录下所有文件
        for filename in os.listdir(caption_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(caption_dir, filename)
                # with open(file_path, "r", encoding="utf-8") as f:
                #     data = json.load(f)
                with open(file_path, 'r') as f:
                    captions = f.readlines()
                for line in tqdm(captions, total=len(captions)):
                    item = json.loads(line.strip())
                    key = item["case_name"]
                    if key in self.scene_description_data.keys():
                        self.scene_description_data[key][(min(item['indexes']), max(item['indexes']))] = item
                    else:
                        self.scene_description_data[key] = {(min(item['indexes']), max(item['indexes'])): item}

                # 假设每个 json 文件中存储的是一个 list
                # total_caption += data
        # # 过滤只保留有caption的case
        # segment_infos_on_caption = []
        # for item in self.segment_infos:
        #     if item["case_name"] in self.scene_description_data.keys():
        #         segment_infos_on_caption.append(item)
        # self.segment_infos = segment_infos_on_caption

        all_case_num = sum([value["case_num_frames"] for value in self.segment_infos])
        print(f"load data clip : {len(self.segment_infos)}  frame num {all_case_num}")
        if self.split == 'train':
            self.segment_infos = self.segment_infos * 10000

    def joint_caption(self, caption_dict):
        return '.'.join(caption_dict.values())

    def format_caption(self, tags: dict) -> str:
        """
        Convert a tags dictionary into the improved structured caption format.
        ...
        """
        parts = []
        # ["weather", "time", "lighting", "road_type", "general"]
        for key in ['weather', 'time', 'lighting', 'road_type']:
            value = tags.get(key)
            if value:
                parts.append(f"[{key}: {value}]")
        result = ' | '.join(parts)
        general = tags.get('general')
        if general:
            result += f' Caption: "{general}"'
        if 'dataset' in tags:
            result += f' {{+dataset:{tags["dataset"]}}}'
        if 'controller' in tags:
            result += f' {{+controller:{tags["controller"]}}}'
        return result

    @property
    def possible_keys(self):
        keys = []
        for f, t in zip(self.fps_list, self.seq_length):
            for fps in f:
                keys.append((fps, t))
        return keys

    def key_len(self, key):
        """
        每种key对应的样本个数，一组fps和num_frames为一个key，由于PAP的样本以case为单位，每个key的长度都是case总数
        为了接口的统一性，仍然保留该函数，作用同self.__len__()
        """
        if isinstance(key, str):
            fps, t = key.split("-")
            fps = int(fps)
            t = t if t == "full" else int(t)
        elif isinstance(key, tuple):
            fps, t = key
        else:
            raise TypeError(key)
        return len(self.segment_infos)

    def resize_nearest(self, img, size):
        size = (size[1], size[0])
        return img.resize(size, Image.NEAREST)

    def find_best_interval(self, interval_list, target_interval):
        """
        在列表中找到包含目标区间的区间；如果没有，则找到相交部分最多的区间。
        :param interval_list: List[Tuple[int, int]] 区间列表，元素为 (start, end)
        :param target_interval: Tuple[int, int] 目标区间，(start, end)
        :return: Tuple[int, int] 匹配的区间
        """
        best_interval = None
        max_overlap = 0
        target_start, target_end = target_interval
        for interval in interval_list:
            start, end = interval
            # 检查是否完全包含目标区间
            if start <= target_start and end >= target_end:
                return interval  # 完全包含，直接返回
            # 计算相交部分的长度
            overlap_start = max(start, target_start)
            overlap_end = min(end, target_end)
            overlap_length = max(0, overlap_end - overlap_start)
            # 更新最大相交区间
            if overlap_length > max_overlap:
                max_overlap = overlap_length
                best_interval = interval
        return best_interval  # 如果没有完全包含的区间，返回最大相交区间

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index)

    # 记录信息，报错追溯
    def update_info(self, **kwargs):
        required_keys = ['index', 'case_name']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key: {key}")
        self.index_info = kwargs
    def get_info(self):
        return self.index_info

    def get_item(self, index) -> dict:
        index, seq_length, selected_fps = [int(item) for item in index.split("-")]
        infos = self.segment_infos[index]
        case_name, case_root, case_num_frames  = infos["case_name"], infos["case_root"], infos["case_num_frames"]
        max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        # start_id = 0 if self.split != "train" else random.randint(0, max_fps_id - seq_length - 1)
        start_id = 0 if self.split != "train" else random.randint(0, max_fps_id - seq_length)

        # updata info
        dataset_tag = "vd2"
        self.update_info(index = index, case_name = case_name, case_num_frames = case_name, start_id = start_id, dataset_tag = dataset_tag)

        # load anno
        clip_anno_path = os.path.join(case_root, case_name + ".json")
        clip_anno_list = json.load(open(clip_anno_path, "rb"))

        frame_id_start = int(np.floor((start_id + 0) / max_fps_id * case_num_frames))
        path_to_annos= clip_anno_list[frame_id_start:frame_id_start+seq_length]
        annos = [json.loads(self.reader.load_file(anno_json_path)) for anno_json_path in path_to_annos]

        controller = "unknown"
        scene_description = None
        # 匹配caption
        min_frame_id = int(np.floor((start_id + 0) / max_fps_id * case_num_frames))
        max_frame_id = int(np.floor((start_id + seq_length - 1) / max_fps_id * case_num_frames))
        if case_name in self.scene_description_data.keys():
            interval_list = []
            for k in self.scene_description_data[case_name].keys():
                min_, max_ = k
                min_, max_ = int(min_), int(max_)
                interval_list.append([min_, max_])
            match_key = self.find_best_interval(interval_list, (min_frame_id, max_frame_id))
            all_scene_description = self.scene_description_data[case_name][(match_key[0], match_key[1])]
            if all_scene_description:
                scene_description = dict()
                for camera_name in self.camera_list:
                    if self.sqrt_required_text_keys:
                        combine_text = {}
                        for key in self.sqrt_required_text_keys:
                            if key in all_scene_description['caption'].keys():
                                combine_text[key] = all_scene_description['caption'][key]
                            elif key in all_scene_description['caption'][camera_name].keys():
                                combine_text[key] = all_scene_description['caption'][camera_name][key]

                        # 添加域控信息
                        if self.add_controller:
                            combine_text.update({
                                "dataset": dataset_tag,
                                "controller": controller,
                            })
                        scene_description[camera_name] = self.caption_join_func(combine_text)
                    else:
                        scene_description[camera_name] = all_scene_description['caption'][camera_name]['general']
            else:
                print('No Text!', min_frame_id, max_frame_id, self.scene_description_data[case_name].keys())
        else:
            print('No Text!', case_name)

        # 初始化黑图
        black_image = Image.new('RGB', (self.expected_vae_size[1], self.expected_vae_size[0]), color='black')
        images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        bbox_images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]  # blank img with bbox
        black_image_path = f"/iag_ad_01/ad/xujin2/data/vd/other/black_{self.expected_vae_size[0]}_{self.expected_vae_size[1]}.png"
        img_paths = [[ black_image_path for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        if self.with_bbox_coords:
            bboxes_meta_list = [[{} for _ in range(seq_length)] for _ in range(len(self.camera_list))]

        for i in range(seq_length):
            if i>= len(annos):
                continue
            else:
                bboxs, classes = annos[i]['gt_boxes'], annos[i]['gt_names']
                all_lanelines_dict = annos[i]['lanelines']['car_center_3d']
                all_lanelines, all_colors, all_geo_labels = [], [], []

                for line in all_lanelines_dict:
                    all_lanelines.append(line['geo3d'])
                    # all_colors.append(GEO2COLOR[line['color']])
                    all_colors.append(GEO_ID2COLOR.get(line['label'], (255, 255, 255)))
                    all_geo_labels.append(line['label'])
                if self.convert_left_hand: # 转化手系：右手转左手
                    bboxs = convert_3dbbox_batch_b_to_a(bboxs).tolist()
                    all_lanelines_ = []
                    for laneline in all_lanelines:
                        new_laneline = np.array(laneline)
                        new_laneline[:,1] = -new_laneline[:,1]
                        all_lanelines_.append(new_laneline.tolist())
                    all_lanelines = all_lanelines_

                camera_params = annos[i]['cams']
                for cam_id, camera_name in enumerate(self.camera_list, 0):
                    img_load_path = camera_params[camera_name]['aws_path']
                    img_paths[cam_id][i] = img_load_path
                    img_byte_stream = self.reader.load_file(img_load_path)
                    img = Image.open(io.BytesIO(img_byte_stream))
                    camera_intrinsic = camera_params[camera_name]["cam_intrinsic"]
                    lidar2camera_rt = camera_params[camera_name]["extrinsic"]
                    img_dist_w, img_dist_h = img.size # org size
                    if self.save_annos:
                        annos[i]['cams'][camera_name]["image_size"] = (img_dist_w, img_dist_h)
                    img = img.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                    images[cam_id][i] = img
                    bbox_image = np.array(bbox_images[cam_id][i])
                    sx, sy = self.expected_vae_size[1] / img_dist_w, self.expected_vae_size[0] / img_dist_h
                    # 内外参前处理
                    # 缩放内参到最终size上
                    camera_intrinsic = np.array(camera_intrinsic) * np.array([sx, sy, 1])[:, np.newaxis]
                    # camera_intrinsic = np.array(camera_intrinsic)
                    # lidar2camera_rt = np.linalg.inv(np.array(lidar2camera_rt))
                    lidar_camera_extrinsic = np.matrix(np.array(lidar2camera_rt).reshape(4, 4))
                    if self.convert_left_hand:
                        camera_intrinsic = convert_camera_intrinsics_b_to_a(camera_intrinsic, image_size=(self.expected_vae_size[1], self.expected_vae_size[0]))
                        lidar_camera_extrinsic = convert_matrix_b_to_a(lidar_camera_extrinsic)

                    # add geo
                    if len(all_lanelines) > 0:
                        bbox_image = proj_geo(np.array(bbox_image), all_lanelines, all_colors, camera_intrinsic,
                                              lidar_camera_extrinsic, thickness = self.thickness // 2)
                    # add pvb
                    if len(bboxs) > 0:
                        bbox_image, bboxs_coners_in_cam, classes_in_cam = proj_pvb(np.array(bbox_image), bboxs, classes, camera_intrinsic, lidar_camera_extrinsic, id2color=CLASS_BBOX_COLOR_MAP)
                    else:
                        bboxs_coners_in_cam = bboxs
                        classes_in_cam = classes

                    bbox_images[cam_id][i] = Image.fromarray(bbox_image)
                    if self.with_bbox_coords:
                        classes_in_cam_number = [self.pvb_bbox2id.get(key, 0) for key in classes_in_cam]
                        # assert len(classes_in_cam_number) == len(bboxs_coners_in_cam)
                        camera_param = np.concatenate([camera_intrinsic[:3, :3], lidar_camera_extrinsic[:3]], axis=1)
                        bboxes_meta_list[cam_id][i] = {  # boxes_coords, camera_meta, lidar_in_cams
                            # "lidar_in_worlds":bboxes_meta[0], #[n,8,3]
                            "cam_params": camera_param,  # !要[3,7]
                            "bboxes": np.array(bboxs_coners_in_cam),  #这里是全局3d点，不是对应相机的3d点
                            # [n,8,3] # lidar_in_cams #[B, N_out, max_len, 8 x 3] #! 看清楚是要cam还是world: 是cam
                            # "classes": classes  # [n,] #!要类别号码
                            "classes": classes_in_cam_number
                        }
        result = {
            'pixel_values': torch.stack([torch.stack([self.transforms(i) for i in img]) for img in images]).permute(1, 0, 2, 3, 4),
            # [V, T, C, H, W] -> [V, C, T, H, W], # 11, frame_num
            # 'condition_images_visual': condition_images,
            'bbox': torch.stack([torch.stack([self.condition_transform(i) for i in bbox_img]) for bbox_img in bbox_images]),
            # [V, T, C, H, W]
            # "hdmap": hdmap_images,
            # 'pts': torch.zeros((self.sequence_length), dtype=torch.long),
            'fps': selected_fps if seq_length > 1 else IMG_FPS,
            'img_path': [[img_paths[v_i][t_i] for v_i in range(len(self.camera_list))] for t_i in
                         range(seq_length)],
            'full_height': self.full_size[0],
            'full_width': self.full_size[1],
            "ego_velocity": torch.tensor(np.array([1.0,1.0,1.0])), # vd 数据暂时没用
            "path_to_annos": path_to_annos
        }
        result["height"] = result["pixel_values"].shape[-2]
        result["width"] = result["pixel_values"].shape[-1]
        result["ar"] = result["width"] / result["height"]
        # result["captions"] = [result.pop('scene_description')] * seq_length
        result["num_frames"] = seq_length

        if scene_description: # 有caption
            # result["scene_description"] = scene_description['center_camera_fov120']  # [k + '.' + v for k, v in scene_description.items()]
            result["captions"] = [scene_description['center_camera_fov120']] * seq_length
        else:
            result["captions"] = [""] * seq_length # 暂时给个空

        if self.with_bbox_coords:

            # max_len = 0
            max_len = 1
            V = len(bboxes_meta_list)
            T = len(bboxes_meta_list[0])
            cam_param = [[] for _ in range(T)]
            bboxes = [[] for _ in range(T)]
            classes = [[] for _ in range(T)]
            try:
                for t in range(T):
                    for v in range(V):
                        cam_param[t].append(bboxes_meta_list[v][t]["cam_params"])
                        bboxes[t].append(bboxes_meta_list[v][t]["bboxes"])
                        classes[t].append(bboxes_meta_list[v][t]["classes"])
                        if bboxes[t][-1] is not None:
                            max_len = max(max_len, len(bboxes[t][-1]))
            except Exception as e:
                traceback.print_exc()
                # print(1)
            bbox_shape = (8, 3)  # 8 x 3
            ret_dict = pad_bboxes_to_maxlen([T, V, max_len, *bbox_shape], max_len, bboxes, classes)
            # max_len = 1
            # ret_dict = pad_bboxes_to_maxlen([T, V, max_len, *bbox_shape], max_len, None, None)

            ret_dict['cam_params'] = np.asarray(cam_param)  # [t,v,3,7]
            # if self.bbox_mode == 'cxyz':
            #     ret_dict["bboxes"] = ret_dict["bboxes"][:, :, :, :, [6, 5, 7, 2]]  # [B, N_out, max_len, 8 x 3]
            # result["bboxes_3d_data"] = {"data": ret_dict}
            result["bboxes_3d_data"] = DataContainer(ret_dict)
            result["camera_param"] = np.asarray(cam_param)

        if self.user_frame_emb:
            if self.convert_left_hand:
                ego2global = np.array([convert_matrix_b_to_a(get_ego2global_transformation_matrix(frame['ego2global_rotation'],frame['ego2global_translation'])) for frame in annos])
            else:
                ego2global = np.array([get_ego2global_transformation_matrix(frame['ego2global_rotation'],frame['ego2global_translation']) for frame in annos])
            result['frame_emb'] = align_camera_poses(ego2global)
        else:
            result['frame_emb'] = np.array([np.zeros((4, 4)) for frame in annos])

        # if
        result['bev_map_with_aux'] = result["bbox"] # # [V, T, C, H, W]
        if self.save_annos:
            result["annos"] = json.dumps(annos, indent=2, sort_keys=True)
        return result

def denormalize(y, normalized=True):
    # 1. 反标准化
    if normalized:
        # mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        # std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        mean = 0.5
        std = 0.5
        x = y * std + mean  # 反标准化操作
    else:
        x = y

    # 2. 将值裁剪到 [0.0, 1.0] 范围内
    x = torch.clamp(x, 0.0, 1.0)
    return x

def format_video(input_img_list):
    # list for V, T, C, H, W
    new_tensor = torch.concatenate(input_img_list, dim=3)
    # V, T, C, H, W = new_tensor.shape
    return rearrange(new_tensor, "v t c h w -> t c h (v w)")

if __name__ == "__main__":
    raw_meta_files = [
        # "/iag_ad_01/ad/xujin2/data/vd/vd2/train_data_1667clip_0304",
        # # "/iag_ad_01/ad/xujin2/data/vd/vd2/train_data_1667clip_0304.json",
        # "/iag_ad_01/ad/xujin2/data/vd/vd2/train_data_6285clip_20250421",
        # # "/iag_ad_01/ad/xujin2/data/vd/vd2/vd2_gop_collect_20250422", # gop
        "/iag_ad_01/ad/xujin2/data/vd/vd2/train_data_1667clip_0304.json",
        "/iag_ad_01/ad/xujin2/data/vd/vd2/train_data_6285clip_20250421.json",
    ]
    caption_dir = "/iag_ad_01/ad/xujin2/data/vd/vd2/caption_file"
    # 7v
    camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", "rear_camera",
                   "left_rear_camera", "left_front_camera", "center_camera_fov30"]
    sequence_length = [1, 17, 33, 65, 97]
    fps_list = [10]
    data_fps = 10
    image_size = (256, 512)
    full_size = (image_size[0], image_size[1] * len(camera_list))  # MULTI_VIEW
    split = "test"
    # split = "train"
    ADdataset = VD2VariableDataset(
                raw_meta_files = raw_meta_files,
                caption_dir = caption_dir,
                path_to_aoss_config = "/iag_ad_01/ad/xujin2/aoss.conf",
                s3_path = "",
                camera_list = camera_list,
                sequence_length = sequence_length,
                fps_list = fps_list,
                data_fps = data_fps,
                expected_vae_size = image_size,
                user_frame_emb = True, # for debug
                full_size = full_size,
                split = split,
                # save_annos = True,
                convert_left_hand=True,
    )
    print('len(AD dataset):', len(ADdataset))
    # 将 Tensor 转为帧列表
    to_pil = transforms.ToPILImage()  # 转换为 PIL Image
    from torch.utils.tensorboard import SummaryWriter
    loacl_show_dataset_tb_dir = "/iag_ad_01/ad/xujin2/code_hsy/magicdrivedit/show_dataset/dataset/tensorboard"
    if os.path.exists(loacl_show_dataset_tb_dir):
        import shutil
        shutil.rmtree(loacl_show_dataset_tb_dir)
    writer = SummaryWriter(loacl_show_dataset_tb_dir)
    # sample_id = [random.randint(0, len(ADdataset) - 1) for _ in range(10)]
    index_list = [
        # f"{random.randint(0, len(ADdataset) - 1)}-17-10"
        f"{random.randint(0, len(ADdataset) - 1)}-2-10"
        for _ in range(10)
    ]

    # # for debug
    # index_list = [
    #     "13796331-10-97",
    #     "13796331-10-97",
    #     "13796331-10-97",
    #     "13796331-10-97",
    #     "4213614-10-97",
    #     "4213614-10-97",
    #     "4213614-10-97",
    #     "4213614-10-97",
    # ]
    # # for debug
    # index_list = [
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    #     "89-10-97",
    # ]

    from torchvision.io import write_video
    def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(0, 1)):
        """
        Args:
            x (Tensor): shape [C, T, H, W]
        """
        assert x.ndim == 4

        if x.shape[1] == 1:  # T = 1: save as image
            save_path += ".png"
            x = x.squeeze(1)
            save_image([x], save_path, normalize=normalize, value_range=value_range)
        else:
            save_path += ".mp4"
            if normalize:
                low, high = value_range
                x.clamp_(min=low, max=high)
                x.sub_(low).div_(max(high - low, 1e-5))

            x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
            write_video(save_path, x, fps=fps, video_codec="h264")
        print(f"Saved to {save_path}")
    # for i in range(0, 100, 10):
    save_local = True
    for index in tqdm(index_list):
        result = ADdataset.get_item(index)
        # 全部转换成 [V, T, C, H, W]
        video = result["pixel_values"].permute(1, 0, 2, 3, 4) #  [T, V, C, H, W] -> [V, T, C, H, W]
        bbox = result["bbox"]
        video = denormalize(video)
        show_data_list = [video, bbox]
        show_video = format_video(show_data_list)
        if save_local:
            save_local_video = rearrange(show_video, "t c h w -> c t h w")
            save_name = loacl_show_dataset_tb_dir + f"/{index}"
            save_sample(save_local_video, save_path = save_name)
        show_video = show_video.unsqueeze(0)
        caption = result["captions"][0]
        print(f"cap: {caption}")
        caption = ""
        writer.add_video(
            "idx:{} cap:{}".format(index, caption),
            show_video,
            fps=2,
        )
    writer.close()
