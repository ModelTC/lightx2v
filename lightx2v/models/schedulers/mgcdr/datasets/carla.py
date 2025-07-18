import io
import os

from toolz.curried import peekn
from torch.onnx.symbolic_opset11 import arange
from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn.functional as F

import cv2
from typing import Literal, List, Dict
from matplotlib import cm
# from calib_helper  import CameraIntrinsic
from PIL import Image, ImageDraw
from einops import rearrange
from torchvision.utils import save_image
import random
import re
import pickle as pkl
from torchvision import transforms
import warnings
import time
from colorama import Fore, Style
import copy
import magicdrivedit.utils.aoss
from magicdrivedit.registry import DATASETS, build_module
from magicdrivedit.datasets.re_project import draw_bbox_alt_camera_meta
from mmcv.parallel import DataContainer
import glob
import math
from math import cos, sin
import traceback

IMG_FPS = 100

# 来自pap
CLASS_BBOX_COLOR_MAP = {
    'VEHICLE_SUV': (255, 0, 0),  # "运动型多用途轿车" car #红
    'VEHICLE_CAR': (255, 0, 0),  # car #红
    'VEHICLE_TRUCK': (255, 128, 0),  # "大型货车" truck #橙
    'CYCLIST_BICYCLE': (0, 255, 0),  # "非机动车骑行者"  bicycle #绿
    'PEDESTRIAN_NORMAL': (0, 0, 255),  # "普通行人" pedestrian #蓝
    'CYCLIST_MOTOR': (0, 255, 0),  # "机动车骑行者" motorcycle #绿
    'VEHICLE_BUS': (255, 255, 0),  # "巴士" bus # 黄
    'VEHICLE_PICKUP': (255, 128, 0),  # "皮卡" truck #橙
    'VEHICLE_SPECIAL': (255, 128, 0),  # "特种车" car #橙
    'VEHICLE_TRIKE': (0, 255, 0),  # "三轮车" motorcycle #绿
    'VEHICLE_MULTI_STAGE': (255, 128, 0),  # "多段车单节车体" car #橙
    'PEDESTRIAN_TRAFFIC_POLICE': (0, 0, 255),  # "交警" pedestrian # 蓝
    'VEHICLE_POLICE': (255, 0, 0),  # "警车" car #红
    "VEHICLE_CAR_CARRIER_TRAILER": (255, 128, 0),  # "拖挂车" # 橙
    "VEHICLE_TRAILER": (255, 128, 0),  # "拖车" # 橙
    "VEHICLE_RUBBISH": (255, 128, 0)  # "垃圾车" # 橙
}

# 0000 0000 0000 0000
CLASS_BBOX_ID_MAP = {
    'VEHICLE_SUV': 0,  # "运动型多用途轿车" car #红
    'VEHICLE_CAR': 1,  # car #红
    'VEHICLE_TRUCK': 2,  # "大型货车" truck #橙
    'CYCLIST_BICYCLE': 3,  # "非机动车骑行者"  bicycle #绿
    'PEDESTRIAN_NORMAL': 4,  # "普通行人" pedestrian #蓝
    'CYCLIST_MOTOR': 5,  # "机动车骑行者" motorcycle #绿
    'VEHICLE_BUS': 6,  # "巴士" bus #黄
    'VEHICLE_PICKUP': 7,  # "皮卡" truck #橙
    'VEHICLE_SPECIAL': 8,  # "特种车" car #橙
    'VEHICLE_TRIKE': 9,  # "三轮车" motorcycle #绿
    'VEHICLE_MULTI_STAGE': 10,  # "多段车单节车体" car #橙
    'PEDESTRIAN_TRAFFIC_POLICE': 11,  # "交警" pedestrian # 蓝
    'VEHICLE_POLICE': 12,  # "警车" car #红
    "VEHICLE_CAR_CARRIER_TRAILER": 13,  # "拖挂车" # 橙
    "VEHICLE_TRAILER": 14,  # "拖车" # 橙
    "VEHICLE_RUBBISH": 15  # "垃圾车" # 橙
}



def carla2group_car(car_type):
    if car_type in [
        "vehicle.carlamotors.carlacola", # Carla Cola（虚拟品牌车辆）
        "vehicle.carlamotors.european_hgv", #  European HGV（重型货车）
        "vehicle.carlamotors.firetruck", # Firetruck（消防车）
        "vehicle.ford.ambulance",
        "vehicle.volkswagen.t2",
        "vehicle.volkswagen.t2_2021",
    ]:    #大货车, 大体型车
        return "big_car"
    elif car_type in ["vehicle.mitsubishi.fusorosa"]:
        return "bus"
    elif car_type in [
        "vehicle.harley-davidson.low_rider",   #  哈雷·戴维森 Low Rider（摩托车）
        "vehicle.kawasaki.ninja", # 川崎 Ninja（摩托车）
        "vehicle.vespa.zx125", # 韦斯帕 ZX125（摩托车）
        "vehicle.yamaha.yzf", #雅马哈 YZF（摩托车）
        "vehicle.bh.crossbike", #  BH Crossbike（自行车）
        "vehicle.diamondback.century", # Diamondback Century（自行车）
        "vehicle.gazelle.omafiets", #  Gazelle Omafiets（荷兰城市自行车）
    ]:
        return "two_wheeler"
    # suv car
    elif car_type in [
        "vehicle.audi.etron",  # 奥迪 e-tron 是一款电动 SUV，属于SUV。",
        "vehicle.jeep.wrangler_rubicon",  # 吉普 Wrangler Rubicon 是一款经典的越野 SUV。",
        "vehicle.nissan.patrol",  # 日产 Patrol 是一款大型SUV，属于越野型 SUV。",
        "vehicle.nissan.patrol_2021",  # 日产 Patrol 2021款 也是一款大型SUV。",
        "vehicle.tesla.cybertruck",  # 特斯拉 Cybertruck 是一款电动皮卡，具备SUV的功能和越野性能。",
    ]:
        return "suv_car"
    elif "vehicle." in car_type:
        return "car"
    else:
        print(f"未分类的 car_type {car_type}")
        assert 0

def group2pap_car(group_name):
    if group_name == "car":
        return "VEHICLE_CAR"
    elif group_name == "suv_car":
        return random.choice(["VEHICLE_SUV", "VEHICLE_POLICE"])  #"运动型多用途轿车" car #红,  #"警车" car #红
    elif group_name == "two_wheeler": # 2轮车
        return random.choice(["CYCLIST_BICYCLE", "CYCLIST_MOTOR", "VEHICLE_TRIKE"]) #"非机动车骑行者bicycle" #"机动车骑行者" motorcycle #"三轮车" motorcycle #绿
    elif group_name == "big_car":
        return random.choice([
            "VEHICLE_TRUCK", #"大型货车" truck #橙
            "VEHICLE_PICKUP",#"皮卡" truck #橙
            # "VEHICLE_SPECIAL",#"特种车" car #橙
            # "VEHICLE_MULTI_STAGE",#"多段车单节车体" car #橙
            # "VEHICLE_CAR_CARRIER_TRAILER",#"拖挂车" # 橙
            # "VEHICLE_TRAILER",#"拖车" # 橙
            # "VEHICLE_RUBBISH",
            # "VEHICLE_BUS",   #先取消BUS 效果不太行
        ])  #"垃圾车" # 橙
    elif group_name == "bus":
        return "VEHICLE_BUS"
    else:
        print(f"不合格的 group_name {group_name}")
        assert 0

def bbox8point_get_size(bbox_0):
    ''' 默认carla 左手系 1->5 前进方向
         5 ------- 7
       / |       / |
      1 ------- 3  |
      |  |      |  |
      |  4 -----|--6
      | /       | /
      0 ------- 2
    '''
    x_size = np.linalg.norm(np.array(bbox_0[5]) - np.array(bbox_0[1]))
    y_size = np.linalg.norm(np.array(bbox_0[3]) - np.array(bbox_0[1]))
    z_size = np.linalg.norm(np.array(bbox_0[1]) - np.array(bbox_0[0]))
    return x_size, y_size, z_size

def convert_carla_8point_to_pap(bbox_list, self_car_size, conver_center = True):
    # 默认carla 是左手系，转换为pap的右手系  左手系转右手系 (x, y, z) = (x , -y ,z)
    # 默认carla ego坐标系原点在车bbox的底面中心 即(0.5,0.5, 0)
    # pap ego 坐标系原点在车bbox的后轴中心点 即(0, 0.5, 0.5) 两者差一个  x_size * 0.5
    x_size, y_size, z_size = self_car_size
    new_bbox_list = []
    for bbox in bbox_list:
        if conver_center:
            new_bbox_list.append([bbox[0] + 0.5 * x_size, -bbox[1], bbox[2]])
        else:
            # 只换 左手系 -> 右手系
            new_bbox_list.append([bbox[0], -bbox[1], bbox[2]])
    return new_bbox_list

def carla_8points_index_change_order_pap(bbox_list):
    # chiyu carla
    '''
    carla中的车框索引描述 但是是左手坐标系
       5 ------- 7
       / |       / |
      1 ------- 3  |
      |  |      |  |
      |  4 -----|--6
      | /       | /
      0 ------- 2

    1->5　为前进方向
    转换一下，变成右手坐标系：
       7 ------- 5
       / |       / |
      3 ------- 1  |
      |  |      |  |
      |  6 -----|--4
      | /       | /
      2 ------- 0

    '''
    # pap
    '''
        7 ------- 6
       / |       / |
      3 ------- 2  |
      |  |      |  |
      |  4 -----|--5
      | /       | /
      0 ------- 1
    3 -> 7　为前进方向
    '''
    new_bbox_list = [
        bbox_list[2],
        bbox_list[0],
        bbox_list[1],
        bbox_list[3],
        bbox_list[6],
        bbox_list[4],
        bbox_list[5],
        bbox_list[7],
    ]
    return new_bbox_list


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
    heading_line_indices = ((4, 6), (5, 7))
    # heading_line_indices = ((0, 5), (1, 4))

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

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,  # [B, N_out, max_len, 8 x 3]
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


# def corners2pap9points(vertices):
#     """
#     通过8个顶点坐标反推3D包围框参数 (x, y, z, l, w, h, yaw)
#     :param vertices: 8x3 numpy 数组，按PAP顺序存储
#     :return: (x, y, z, l, w, h, yaw)
#     """
#     vertices = np.array(vertices)
#
#     # 计算中心点
#     center = np.mean(vertices, axis=0)
#
#     # 计算长、宽、高
#     l = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
#     w = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
#     h = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
#
#     # 计算yaw角（使用3 -> 7作为前进方向）
#     p3, p7 = vertices[3], vertices[7]
#     yaw = np.arctan2(p7[1] - p3[1], p7[0] - p3[0])
#
#     return (*center, l, w, h, 0.0, 0.0, yaw)

def corners2pap9points(vertices):
    """
    通过8个顶点坐标反推3D包围框参数 (x, y, z, l, w, h, 0, 0, yaw)
    :param vertices: 8x3 numpy 数组，按PAP顺序存储
    :return: (x, y, z, l, w, h, 0, 0, yaw)
    """
    # pap
    '''
        7 ------- 6
       / |       / |
      3 ------- 2  |
      |  |      |  |
      |  4 -----|--5
      | /       | /
      0 ------- 1
    3 -> 7　为前进方向
    '''
    vertices = np.array(vertices)
    # 计算中心点（底面4个点中心 + 顶面4个点中心 平均）
    bottom_center = np.mean(vertices[[0, 1, 2, 3]], axis=0)
    top_center = np.mean(vertices[[4, 5, 6, 7]], axis=0)
    center = (bottom_center + top_center) / 2
    # 计算长度 (3 -> 7方向是车辆前进方向)
    l = np.linalg.norm(vertices[7] - vertices[3])
    # 计算宽度 (3 -> 2)
    w = np.linalg.norm(vertices[2] - vertices[3])
    # 计算高度 (3 -> 0)
    h = np.linalg.norm(vertices[3] - vertices[0])
    # 计算yaw角（基于3 -> 7方向）
    p3, p7 = vertices[3], vertices[7]
    yaw = np.arctan2(p7[1] - p3[1], p7[0] - p3[0])
    return (*center, l, w, h, 0.0, 0.0, yaw)

# 示例8个顶点（假设已经旋转）
vertices = np.array([
    [-1, -0.5, -0.5],  # 0
    [ 1, -0.5, -0.5],  # 1
    [ 1,  0.5, -0.5],  # 2
    [-1,  0.5, -0.5],  # 3
    [-1, -0.5,  0.5],  # 4
    [ 1, -0.5,  0.5],  # 5
    [ 1,  0.5,  0.5],  # 6
    [-1,  0.5,  0.5]   # 7
])





def convert_matrix_b_to_a(K_b):
    """ 将 B 数据集的相机外参转换为 A 坐标系 """  #例如 B 数据集（右手坐标系） 的所有数据转换到 A 数据集（左手坐标系）
    M = np.array([[1,  0,  0,  0],
                  [0, -1,  0,  0],
                  [0,  0,  1,  0],
                  [0,  0,  0,  1]])
    K_a = M @ K_b @ np.linalg.inv(M)
    return K_a

def moving_average_velocity(velocities, window_size=3):
    smooth_velocities = []
    for i in range(len(velocities)):
        # 确定平滑窗口
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        # 计算窗口内的平均速度
        smooth_velocity = np.mean(velocities[start_idx:end_idx], axis=0)
        smooth_velocities.append(smooth_velocity)
    return np.array(smooth_velocities)

def process_speed_with_yaw(speeds, yaws):
    speeds = moving_average_velocity(speeds, 20)
    # 根据yaw优化速度方向
    optimized_speeds = []
    for idx, speed in enumerate(speeds):
        # 计算yaw的正交方向分量
        yaw_x = np.cos(yaws[idx])
        yaw_y = np.sin(yaws[idx])
        yaw_direction = np.array([yaw_x, yaw_y, 0])
        # 将速度投影到yaw方向上
        speed_projection = np.dot(speed[:2], yaw_direction[:2]) / np.linalg.norm(yaw_direction[:2])
        optimized_speed = np.abs(speed_projection) * yaw_direction
        # optimized_speed = yaw_direction
        optimized_speeds.append(optimized_speed.tolist())
    return optimized_speeds

def read_json_lines(filepath):
    """兼容读取按行存储和整体存储的 JSON 对象"""
    try:
        # 先整体读取文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return []
        # 尝试整体解析 JSON
        data = json.loads(content)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            print(f"⚠️ JSON 内容既不是字典也不是列表：{filepath}")
            return []
    except json.JSONDecodeError:
        # 如果整体解析失败，则尝试按行解析
        dicts = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 跳过空行
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        dicts.append(data)
                    else:
                        print(f"⚠️ 第 {line_num} 行非字典：{filepath}")
                except json.JSONDecodeError:
                    print(f"❌ 第 {line_num} 行解析失败：{filepath}")
        return dicts

# 车道线相关
GEO_CARLA2COLOR = {
    "Dashed": (0, 255, 0),    # 虚线
    "Solid":  (0, 255, 0),    #  实线
    "Curb":    (255, 0, 0),    #  路缘石
}
def show_lanes_per_camera(
        img_pv,
        camera_intrinsic,
        lidar2camera_rt,
        gt_lane,
        gt_lane_cls,
        gt_indexs,
        scale_h,
        scale_w,
        thickness,
        show_cls_num = False,
    ):
    record_label = set()
    # if cam_name not in [
    #     "front_camera_fov195",
    #     "rear_camera_fov195",
    #     "left_camera_fov195",
    #     "right_camera_fov195",
    # ]:

    viewpad = np.eye(4)
    intrin = np.array(camera_intrinsic)
    viewpad[:3, :3] = intrin
    extrin = np.array(lidar2camera_rt)
    ego2img = viewpad @ extrin

    for idx, lane_line in enumerate(gt_lane):
        if gt_lane_cls[idx] in GEO_CARLA2COLOR.keys():
            label = draw_lane_line(
                img_pv,
                lane_line,
                gt_lane_cls[idx],
                gt_indexs[idx],
                ego2img,
                (scale_h, scale_w),
                THICKNESS = thickness,
                GT_COLOR=GEO_CARLA2COLOR,
                show_cls_num=show_cls_num,
            )
            record_label.add(label)
    return record_label

def draw_lane_line(
        image,
        lane,
        label,
        gt_idx,
        ego2img,
        scale,
        THICKNESS=1,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        GT_COLOR={"LANELINE": (0, 255, 0), "ROADSIDE": (255, 0, 0)},
        show_cls_num = True,
):
    points = _project(interp_arc(lane), ego2img)
    if points is None:
        return None
    points[:, 0] *= scale[1]
    points[:, 1] *= scale[0]

    h, w, _ = image.shape
    # Coarse filtration
    valid = (points[:, 2] > 0) & (points[:, 0] < 2 * w) & (points[:, 1] < 2 * h)
    points = points[valid]

    if points.size <= 1:
        return None

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
                color=GT_COLOR[label],
                thickness=THICKNESS,
                lineType=cv2.LINE_AA,
            )
        except Exception as e:
            traceback.print_exc()
            print("划线报错")
    if show_cls_num:
        # Fine filtration
        points = points[(points[:, 0] < w) & (points[:, 1] < h)]
        if points is None or points.size == 0:
            return None

        mid_point = len(points) // 8
        cv2.putText(
            image,
            str(gt_idx),
            points[mid_point][:2].astype(np.int32),
            font,
            0.8,
            GT_COLOR[label],
            thickness=2,
        )
    return label

def interp_arc(points, t=1000):
    r"""
    Linearly interpolate equally-spaced points along a polyline, either in 2d or 3d.
    Parameters
    ----------
    points : List
        List of shape (N,2) or (N,3), representing 2d or 3d-coordinates.
    t : array_like
        Number of points that will be uniformly interpolated and returned.
    Returns
    -------
    array_like
        Numpy array of shape (N,2) or (N,3)

    Notes
    -----
    Adapted from https://github.com/johnwlambert/argoverse2-api/blob/main/src/av2/geometry/interpolate.py#L120

    """

    # filter consecutive points with same coordinate
    temp = []
    for point in points:
        point = point.tolist()
        if temp == [] or point != temp[-1]:
            temp.append(point)
    if len(temp) <= 1:
        return None
    points = np.array(temp, dtype=points.dtype)

    assert points.ndim == 2

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp = anchors + offsets
    return points_interp

def _project(
        points,
        ego2img,
):
    if points is None:
        return None
    one = np.ones((points.shape[0], 1))
    new_points1 = np.concatenate((points, one), axis=-1)
    new_points2 = ego2img @ new_points1.T
    new_points3 = new_points2.T
    new_points3[:, :2] /= new_points3[:, 2:3]
    return new_points3


@DATASETS.register_module(force=True)
class CARLAVariableDataset(object):
    def __init__(self,
                 raw_meta_files,
                 pap_cam_init_path,
                 path_to_aoss_config,
                 s3_path,
                 camera_list=None,
                 sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"],
                 sequence_length=[1, 9, 17, 33],
                 fps_list=[10],
                 data_fps=10,
                 split="train", # * 'test'就不会随机采首帧
                 use_random_seed=False,
                 expected_vae_size=(288, 512),
                 full_size=None,
                 user_frame_emb = True,
                 # trans_scale = 100,
                 trans_scale = 1,
                 **kwargs):
        print(f'raw_meta_files: {raw_meta_files}'); print(f'pap_cam_init_path: {pap_cam_init_path}'); print(f'path_to_aoss_config: {path_to_aoss_config}')
        print(f's3_path: {s3_path}'); print(f'camera_list: {camera_list}'); print(f'sqrt_required_text_keys: {sqrt_required_text_keys}')
        print(f'sequence_length: {sequence_length}'); print(f'fps_list: {fps_list}'); print(f'data_fps: {data_fps}')
        print(f'split: {split}'); print(f'use_random_seed: {use_random_seed}'); print(f'expected_vae_size: {expected_vae_size}')
        print(f'full_size: {full_size}'); print(f'user_frame_emb: {user_frame_emb}'); print(f'trans_scale: {trans_scale}')
        print(f'CARLAVariableDataset kwargs: {kwargs}')
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
        self.trans_scale = trans_scale
        self.with_bbox_coords = kwargs.get('with_bbox_coords', True)
        # self.with_bbox_coords = kwargs.get('with_bbox_coords', False)
        self.with_camera_param = kwargs.get("with_camera_param", False)
        self.proj_func = kwargs.get("proj_func", "pap")
        self.with_original_info = kwargs.get('with_original_info', True)
        self.scene_description_file = kwargs.get('scene_description_file', None)
        self.s3_root = kwargs.get("s3_root", None)
        self.add_geo = kwargs.get('add_geo', False)
        self.random_caption_key = kwargs.get('random_caption_key', None)
        self.random_caption = kwargs.get('random_caption', "chose")  # ["chose", "random"]
        self.caption_join_func = kwargs.get("caption_join_func", None)
        if self.caption_join_func is None:
            self.caption_join_func = self.joint_caption
        elif self.caption_join_func == "format_caption":
            self.caption_join_func = self.format_caption
        else:
            print(f"error self.caption_join_func: {self.caption_join_func}")
            assert False
        self.add_controller = kwargs.get("add_controller", None)

        self.pap2cam = {
            "center_camera_fov120": "CAM_FRONT",
            "right_front_camera": "CAM_FRONT_RIGHT",
            "right_rear_camera": "CAM_BACK_RIGHT",
            "rear_camera": "CAM_BACK",
            "left_rear_camera": "CAM_BACK_LEFT",
            "left_front_camera": "CAM_FRONT_LEFT"
        }

        self.index_info = {}
        self.transforms = transforms.Compose([
            transforms.Resize(expected_vae_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.condition_transform = transforms.Compose([
            transforms.Lambda(lambda img: self.resize_nearest(img, (256, 512))), # 暂时写死，分辨率可以比图像小
            transforms.ToTensor()
        ])
        if max(self.expected_vae_size) > 768:
            self.thickness = 6
        else:
            self.thickness = 2
        if self.use_random_seed:
            random.seed(time.time()%100000)
        print('carla self.seq_length:', self.seq_length)
        print(Style.BRIGHT + Fore.RED + f'self.fps_list:{self.fps_list} in data_fps:{self.data_fps}' + Style.RESET_ALL)
        # self.fps_stride_list = fps_stride_list
        # self.start_zero_frame = start_zero_frame
        self.specific_video_segment = None
        self.segment_infos = []
        self.segment_data_infos = {}
        self.max_seq_length = max(self.seq_length) if isinstance(self.seq_length, list) else self.seq_length
        print('carla self.max_seq_length:', self.max_seq_length)

        if pap_cam_init_path is not None:
            self.cam_params = []
            for cam_json_name in [path for path in os.listdir(pap_cam_init_path) if path.endswith('.json')]:
                cam_json_path = os.path.join(pap_cam_init_path, cam_json_name)
                with open(cam_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cam_params.append(data)
        else:
            self.cam_params = None

        print(f"raw_meta_files : {raw_meta_files}")
        for annn_root in raw_meta_files:
            case_list = [path for path in os.listdir(annn_root) if os.path.isdir(os.path.join(annn_root, path))]
            print(f"carla load annn_root: {annn_root}, case_list: {case_list}")
            for case_name in tqdm(case_list):
                case_root = annn_root
                try:
                    try:
                        data_info = json.load(open(os.path.join(case_root, case_name, 'data_infos.json')))
                    except:
                        data_info = []
                        # data_info = read_json_lines(os.path.join(case_root, case_name, 'data_infos.json'))
                        with open(os.path.join(case_root, case_name, 'data_infos.json')) as f:
                            for line in f:
                                data_info.append(json.loads(line))
                except:
                    traceback.print_exc()
                    continue

                case_num_frames = len(data_info)
                if case_num_frames < self.max_seq_length:
                    continue

                self.segment_infos.append({
                    "case_name": case_name,
                    "case_root": case_root,
                    "case_num_frames": case_num_frames,
                })
                self.segment_data_infos[case_name] = data_info
        print(f'scene_description_file: {self.scene_description_file}')
        self.caption_dict, self.caption_all = self.read_caption(self.scene_description_file)
        print(f"len(self.caption_all)={len(self.caption_all)}, len(set(self.caption_all))={len(set(self.caption_all))}")

        # 更新按照权重来采样caption # TODO why
        if self.random_caption_key is not None:
            for key, value in self.caption_dict.items():
                vv = self.random_caption_key.get(key, None)
                if vv is not None:
                    self.caption_dict[key] = vv
                else:
                    n = len(value)
                    self.caption_dict[key] = {
                        "choices": value,
                        "weights": (np.ones(n) / n),
                    }

        all_case_num = sum([value["case_num_frames"] for value in self.segment_infos])
        print(f"carla load data clip : {len(self.segment_infos)}  frame num {all_case_num}")
        self.segment_infos = self.segment_infos * 1024 * 128 # TODO why
        print(f"carla load data clip : {len(self.segment_infos)}  frame num {all_case_num}")

    def get_caption(self):
        if self.random_caption == "chose":
            text = random.choice(self.caption_all)
        elif self.random_caption == "force":
            text = self.caption_all[0]
        elif self.random_caption == "random":
            caption_dict = {}
            for k, conf in self.caption_dict.items():
                gen_caption = random.choices(conf["choices"], weights=conf["weights"], k=1)[0]
                caption_dict[k] = gen_caption
            text = self.caption_join_func(caption_dict)
        else:
            print("self.random_caption : {}".format(self.random_caption))
            assert 0

        return text

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

    def read_caption(self, scene_description_file=None):
        if scene_description_file is None:
            scene_description_file = [
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_20241125.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_20241124.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_20241209.json",
                # "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_new_20250122.json",
                # "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_new_20250124.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pvb_caption_new_20250208.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pap_caption_20250407.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/pap_caption_20250319.json",
                "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption/caption_pap_20250411.json",
            ]
        # caption_root = "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption"
        # scene_description_file = [
        #     # os.path.abspath(os.path.join(caption_root, file))
        #     os.path.join(caption_root, file)
        #     for file in os.listdir(caption_root)
        #     if file.endswith('.json')
        # ]

        sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"]
        caption_dict = {}
        caption_all = []
        for k in sqrt_required_text_keys:
            caption_dict[k] = []

        for file in scene_description_file:
            data_list = []
            if ".pkl" in file:
                with open(file, 'rb') as f:
                    load_caption_data = pkl.load(f)
                # 转换格式
                for k, v in load_caption_data.items():
                    # scene_description_data[k] = v["all_captions"]
                    for all_100_clip_range, all_100_clip in v["all_captions"].items():
                        data_list.append(all_100_clip)

            elif ".json" in file:
                with open(file, 'r', encoding='utf-8') as f:
                    # content = f.read().strip()
                    # data_list = json.loads(content)
                    for line in f.readlines():
                        data_list.append(json.loads(line.strip()))
            else:
                print(f"read {file}  failed !!!!")
                assert False

            for data in tqdm(data_list):
                data_caption = data["caption"]
                if "center_camera_fov120" in data_caption.keys():
                    data_caption = data_caption["center_camera_fov120"]
                cap_list = {}
                for kk in sqrt_required_text_keys:
                    v = data_caption[kk]
                    if v not in caption_dict[kk]:
                        caption_dict[kk].append(v)
                    cap_list[kk] = v
                caption = self.caption_join_func(cap_list)
                caption_all.append(caption)
        return caption_dict, caption_all

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

    # 记录信息，报错追溯
    def update_info(self, index, case_name, case_num_frames, start_id):
        self.index_info = {
            "index":index,
            "case_name":case_name,
            "case_num_frames":case_num_frames,
            "start_id":start_id
        }
    def get_info(self):
        return self.index_info

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index)

    def get_item(self, index) -> dict:
        import pdb; pdb.set_trace()
        index, seq_length, selected_fps = [int(item) for item in index.split("-")]
        infos = self.segment_infos[index]
        case_name, case_root, case_num_frames  = infos["case_name"], infos["case_root"], infos["case_num_frames"]
        max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        start_id = 0 if self.split != "train" else random.randint(0, max_fps_id - seq_length)
        # load anno
        frame_id_statr = int(np.floor((start_id + 0) / max_fps_id * case_num_frames))
        annos = self.segment_data_infos[case_name][frame_id_statr:frame_id_statr+seq_length]
        path_to_annos = [os.path.join(case_root, case_name)] * len(annos)
        self.update_info(index, case_name, case_num_frames, start_id)

        # caption 重新给random 待补充

        # 创建车辆映射表 会保保证group一直，group内部在首帧随机后固定下来：在一段视频下color映射关系是固定的
        # gen npc index to color
        npc_id2color = {}
        npc_id2pap_name = {}
        for frame in annos:
            if "detection" in frame.keys():
                for ins_id, ins in enumerate(frame["detection"]):
                    if ins["type"] not in npc_id2color:
                        # npc_id2color[ins["type"]] = random_gen_color(self.color_bbox) # 随机给color
                        group_name = carla2group_car(ins["type"])
                        carla_id = ins["type"]
                        # print(f"{carla_id}  -> {group_name}")
                        pap_car_name = group2pap_car(group_name) # 1对多 映射 会引入随机性，必须先在首帧固定下来
                        npc_id2pap_name[ins["type"]] = pap_car_name
                        color = CLASS_BBOX_COLOR_MAP[pap_car_name]
                        npc_id2color[ins["type"]] = color
        # 确定选择那个geo
        # gep_key = random.choice(['Geo', 'Geo_double_lane','Geo_double_lane_both_sides','Geo_triple_lane','Geo_triple_lane_both_sides'])
        gep_key = random.choice(['Geo'])

        # 获取额外信息
        ego_bbox = np.array(annos[0]["detection_self"]["points"])
        if ego_bbox.shape[-1] > 3:
            ego_bbox = ego_bbox[...,:3]
        ego_x_size, ego_y_size, ego_z_size = bbox8point_get_size(ego_bbox)
        ego_bbox = convert_carla_8point_to_pap(ego_bbox, (ego_x_size, ego_y_size, ego_z_size), conver_center=True)  # (x, y, z) -> (x, -y, z) 左手转右手
        ego_bbox = carla_8points_index_change_order_pap(ego_bbox)
        ego_bbox_9point = corners2pap9points(ego_bbox)
        ego_yaw = [ego_bbox_9point[-3], ego_bbox_9point[-2], ego_bbox_9point[-1]]
        # timestep_start
        start_time = time.time() * 1e9

        # 初始化黑图
        black_image = Image.new('RGB', (self.expected_vae_size[1], self.expected_vae_size[0]), color='black')
        images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        bbox_images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]  # blank img with bbox
        img_paths = [[f"/kaiwu_vepfs/kaiwu/xujin2/data/vd/other/black_{self.expected_vae_size[0]}_{self.expected_vae_size[1]}.png" for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        if self.with_bbox_coords:
            bboxes_meta_list = [[{} for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        # 随机选取一个pap的相机参数文件
        if self.cam_params:
            cam_param_mate = copy.deepcopy(random.choice(self.cam_params))
            if "Objects" in cam_param_mate:
                cam_param_mate.pop("Objects")  # 丢掉原有的 obj
            if "original_info" in cam_param_mate.keys():
                cam_param_mate.pop("original_info")
            cam_param_mate["case_name"] = case_name
            cam_param_mate["org_sensors"] = copy.deepcopy(cam_param_mate["sensors"])
        else:
            cam_param_mate = None
        annos_out = []
        for i in range(seq_length):
            if i>= len(annos):
                continue
            else:
                # bboxs, classes, conf = get_all_pvb(annos[i])
                # all_lanelines, all_colors = get_all_geo(annos[i])
                frame = annos[i]
                timestamp = int(frame["timestamp"] * 1e9 + start_time) # 需要一个函数转 timesstep

                # 获取pvb
                bboxs = []
                bbox_colors = []
                pap_ids = []
                carla_npc_id = []
                bboxs_velocity = []
                if "detection" in frame.keys():
                    for ins_id, ins in enumerate(frame["detection"]):
                        bbox = ins["points"]
                        bbox = [b[:3] for b in bbox]
                        # if self.cam_params:
                        bbox = convert_carla_8point_to_pap(bbox, (ego_x_size, ego_y_size, ego_z_size), conver_center= self.cam_params is not None)  # (x, y, z) -> (x, -y, z) 左手转右手
                        bbox = carla_8points_index_change_order_pap(bbox)
                        bboxs.append(bbox)
                        bbox_colors.append(npc_id2color[ins["type"]])
                        pap_ids.append(npc_id2pap_name[ins["type"]])
                        npc_velocity = ins["npc_velocity"]
                        bboxs_velocity.append([npc_velocity[0], -npc_velocity[1], npc_velocity[2]])  #左手系转右手
                        if 'npc_id' in ins.keys():
                            carla_npc_id.append(ins["npc_id"])
                        else:
                            carla_npc_id.append(-1)

                # 如果添加车道线
                if self.add_geo:
                    gt_lanes = []
                    gt_lane_cls = []
                    # gt_lane_sub_cls = []
                    gt_indexs = []
                    if "Geo" in frame.keys():
                        for idx, lanes in enumerate(frame[gep_key]):
                            # gt_lanes.append(self.fix_pts_interpolate(np.array(lanes["geo"]), 10, 10))
                            if len(lanes["xyz"]) <= 1:
                                continue
                            # gt_lanes.append(np.array(lanes["xyz"]))
                            pap_lanes = convert_carla_8point_to_pap(lanes["xyz"], (ego_x_size, ego_y_size, ego_z_size))
                            # # z 设置为0试试
                            # pap_lanes = np.array(pap_lanes)
                            # pap_lanes[:, 2:3] = 0 #
                            gt_lanes.append(np.array(pap_lanes))
                            gt_lane_cls.append(lanes["label"])
                            # gt_lane_sub_cls.append(lanes["style"])
                            gt_indexs.append(idx)

                if self.proj_func == "pap":
                    meta_by_frame = copy.deepcopy(cam_param_mate)
                    # 弄一个Objects
                    meta_by_frame["Objects"] = []
                    for bbox3d, color, pap_id,npc_id,  bbox_velocity in zip(bboxs, bbox_colors, pap_ids, carla_npc_id,  bboxs_velocity):
                        meta_by_frame["Objects"].append({
                            "label": pap_id,
                            "bbox3d": bbox3d,
                            "color": color,
                            "id": npc_id,
                            "class_id": CLASS_BBOX_ID_MAP[pap_id],
                            # "velocity": velocity
                            "velocity": bbox_velocity,
                        })

                for cam_id, camera_name in enumerate(self.camera_list, 0):

                    if camera_name in self.pap2cam.keys() and frame["camera_infos"] != {}:
                    # if False:
                        # 修改bug
                        CAM_NAME = self.pap2cam[camera_name]
                        rgb_relative_path = frame['camera_infos'][CAM_NAME]['rgb']
                        if rgb_relative_path is not None:
                            case_root_laster_dir = case_root.split("/")[-1]
                            if case_root_laster_dir in rgb_relative_path:
                                rgb_relative_path = rgb_relative_path.replace(f"{case_root_laster_dir}/", "")

                            if case_name in rgb_relative_path:
                                rgb_path = os.path.join(case_root, rgb_relative_path)
                            else:
                                rgb_path = os.path.join(case_root, case_name, rgb_relative_path)

                            img_paths[cam_id][i] = rgb_path
                            try:
                                img = Image.open(rgb_path)
                                bbox_local = Image.open(rgb_path.replace("rgb", "3dbox"))
                            except:
                                img = black_image.copy()
                                bbox_local = black_image.copy()

                        else:
                            img = images[cam_id][i]
                            bbox_local = black_image.copy()
                    else: # 黑图
                        img = images[cam_id][i]
                        bbox_local = black_image.copy()

                    if self.cam_params:
                        camera_intrinsic = cam_param_mate['sensors']['cameras'][camera_name]['camera_intrinsic']
                        lidar2camera_rt = cam_param_mate['sensors']['cameras'][camera_name]['extrinsic']
                        img_dist_w = cam_param_mate['sensors']['cameras'][camera_name]['image_width']
                        img_dist_h = cam_param_mate['sensors']['cameras'][camera_name]['image_height']
                    else:
                        camera_intrinsic = frame['camera_infos'][CAM_NAME]['intrin']
                        lidar2camera_rt = frame['camera_infos'][CAM_NAME]['extrin']
                        img_dist_w, img_dist_h = img.size

                    img = img.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                    bbox_image = np.array(bbox_images[cam_id][i])
                    sx, sy = self.expected_vae_size[1] / img_dist_w, self.expected_vae_size[0] / img_dist_h

                    # 内外参前处理
                    # 缩放内参到最终size上
                    camera_intrinsic = np.array(camera_intrinsic) * np.array([sx, sy, 1])[:, np.newaxis]
                    # camera_intrinsic = np.array(camera_intrinsic)
                    # lidar2camera_rt = np.linalg.inv(np.array(lidar2camera_rt))
                    lidar2camera_rt = np.matrix(np.array(lidar2camera_rt).reshape(4, 4))

                    # add pvb
                    # if len(bboxs) > 0:
                    if True:
                        if self.proj_func == "pap":
                            bbox_image, bboxes_meta = draw_bbox_alt_camera_meta(np.array(bbox_image), meta_by_frame, camera_name, img_shape = None, sx=sx, sy=sy, bboxmode="8points", colorful_box=True, thickness=self.thickness)
                            bboxs_coners_in_cam = bboxes_meta['lidar_in_cams']
                            pap_ids_in_cam = bboxes_meta['box_classes']
                        else:
                            bbox_image, bboxs_coners_in_cam, pap_ids_in_cam = proj_pvb(np.array(bbox_image), bboxs, pap_ids, camera_intrinsic, lidar2camera_rt)

                        # # for debug
                        # img_, _, _ = proj_pvb(np.array(img), bboxs, pap_ids, camera_intrinsic, lidar_camera_extrinsic)
                        # img = Image.fromarray(img_)
                    else:
                        bboxs_coners_in_cam = bboxs
                        pap_ids_in_cam = pap_ids
                    # add geo
                    if self.add_geo:
                        reshape_height, reshape_width = self.expected_vae_size
                        img_pv = np.zeros((reshape_height, reshape_width, 3), dtype=np.uint8)
                        # scale_h, scale_w = reshape_height / src_h, reshape_width / src_w
                        scale_h, scale_w = 1.0, 1.0  # 内参已经被归一化到了self.expected_vae_size 上了
                        _ = show_lanes_per_camera(img_pv, camera_intrinsic, lidar2camera_rt, gt_lanes, gt_lane_cls, gt_indexs, scale_h, scale_w, thickness=self.thickness // 2)
                        # bbox_image = bbox_image.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                        assert bbox_image.shape == img_pv.shape
                        bbox_image = np.clip(img_pv + np.array(bbox_image), 0, 255)
                        # bbox_image = Image.fromarray(bbox_image)

                    images[cam_id][i] = img
                    # bbox_images[cam_id][i] = Image.fromarray(bbox_image)
                    bbox_images[cam_id][i] = Image.fromarray(bbox_image) if self.cam_params else bbox_local
                    if self.with_bbox_coords:
                        # assert len(classes_in_cam_number) == len(bboxs_coners_in_cam)
                        camera_param = np.concatenate([camera_intrinsic[:3, :3], lidar2camera_rt[:3]], axis=1)
                        if self.proj_func == "pap":
                            classes = pap_ids_in_cam
                        else:
                            classes = [CLASS_BBOX_ID_MAP[data] for data in pap_ids_in_cam] # 映射成id 1,2,3,这种。
                        bboxes_meta_list[cam_id][i] = {  # boxes_coords, camera_meta, lidar_in_cams
                            # "lidar_in_worlds":bboxes_meta[0], #[n,8,3]
                            "cam_params": camera_param,  # !要[3,7]
                            "bboxes": np.array(bboxs_coners_in_cam),  #这里是全局3d点，不是对应相机的3d点
                            # [n,8,3] # lidar_in_cams #[B, N_out, max_len, 8 x 3] #! 看清楚是要cam还是world: 是cam
                            # "classes": classes  # [n,] #!要类别号码
                            "classes": classes
                        }

                    # 后处理anno_out 对齐输出pap格式
                    if self.cam_params: # per camera
                        # 把缩放内参写回去
                        # camera_intrinsic = cam_param_mate['sensors']['cameras'][camera_name]['camera_intrinsic']
                        meta_by_frame['sensors']['cameras'][camera_name]['camera_intrinsic'] = camera_intrinsic.tolist()
                        meta_by_frame['sensors']['lidar'] = {}
                        meta_by_frame['sensors']['radars'] = {}
                        meta_by_frame['sensors']['cameras'][camera_name]['data_path'] = f"cameras/{camera_name}/{timestamp}.jpg"
                        if self.s3_root:
                            meta_by_frame['sensors']['cameras'][camera_name]['s3_data_path'] = os.path.join(self.s3_root, meta_by_frame['sensors']['cameras'][camera_name]['data_path'])
                        meta_by_frame['sensors']['cameras'][camera_name]['timestamp'] = int(timestamp % 1e6)
                        meta_by_frame['sensors']['cameras'][camera_name]['image_width'] = self.expected_vae_size[1]
                        meta_by_frame['sensors']['cameras'][camera_name]['image_height'] = self.expected_vae_size[0]

                if self.cam_params: # per frames
                    for Obj in meta_by_frame["Objects"]:
                        Obj["bbox3d"] = corners2pap9points(Obj["bbox3d"])
                        Obj["token"] = "token"
                        Obj["score3d"] = 1.0
                        Obj["attribute"] = {
                            "OPEN_STATUS": None,
                            "SELF_LANE_OWNERSHIP": 1,
                            "SELF_LANE_CROSS": 1,
                            "SHIELD": False,
                            "MULTI_BBOX": None,
                            "VEHICLE_DOOR_STARUS": True,
                            "VEHICLE_DOOR_OPEN_SIDE": False,
                            "VEHICLE_TAILGATE_STARUS": False,
                            "ACCIDENTA_STATUS": False,
                            "LIGHT_STATUS": "NORMAL",
                            "VEHICLE_TRIKE_MANNED_STATUS": False
                        }
                        Obj["info2d"] = {}
                    meta_by_frame["vehicle_id"] = "AIGC-001"
                    meta_by_frame["valid"] = True
                    ego_velocity = frame["detection_self"]["npc_velocity"]
                    meta_by_frame["ego_velocity"] = [ego_velocity[0], -ego_velocity[1], ego_velocity[2]] # 左手转右手
                    if 'npc_acceleration' in frame["detection_self"].keys():
                        ego_acceleration = frame["detection_self"]["npc_acceleration"]
                        meta_by_frame["ego_acceleration"] = [ego_acceleration[0], -ego_acceleration[1], ego_acceleration[2]]  # 左手转右手
                    meta_by_frame["timestamp"] = timestamp
                    meta_by_frame["velocit_refine_yaw"] = process_speed_with_yaw([meta_by_frame["ego_velocity"]], ego_yaw)
                    # 加一个carla_case_name
                    meta_by_frame["carla_case_name"] = case_name
                    add_key = ['ego_velocity_ego', 'ego_velocity_world', 'ego_target_waypoint', 'ego_target_waypoint_ego', 'ego_target_road_option']
                    add_key += ["Geo"]
                    add_key += ["navi_pts"] # zhiji所需的导航点
                    add_key += ["steering_angle", "imu_data", "ego_control"] # zhiji所需的导航点
                    for key in add_key:
                        if key in frame.keys():
                            meta_by_frame[key] = frame[key]
            annos_out.append(meta_by_frame)
        import pdb; pdb.set_trace()
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
        # 添加caption
        caption_ = self.get_caption()
        # 添加域控信息
        if self.add_controller is not None:
            dataset_tag = self.add_controller["dataset_tag"]
            controller = self.add_controller["controller"]
            caption_ += f' {{+dataset:{dataset_tag}}}'
            caption_ += f' {{+controller:{controller}}}'
        result["captions"] = [caption_] * seq_length

        if self.with_bbox_coords:
            # max_len = 0
            max_len = 1
            V = len(bboxes_meta_list)
            T = len(bboxes_meta_list[0])
            cam_param = [[] for _ in range(T)]
            bboxes = [[] for _ in range(T)]
            classes = [[] for _ in range(T)]
            for t in range(T):
                for v in range(V):
                    cam_param[t].append(bboxes_meta_list[v][t]["cam_params"])
                    bboxes[t].append(bboxes_meta_list[v][t]["bboxes"])
                    classes[t].append(bboxes_meta_list[v][t]["classes"])

                    if bboxes[t][-1] is not None:
                        max_len = max(max_len, len(bboxes[t][-1]))
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
            ego2global_transformation_matrix = []
            for frame in annos:
                if "ego_pose" in frame:
                    pose = frame["ego_pose"]
                elif "detection_self" in frame:
                    pose = frame["detection_self"]["npc_transform"]
                else:
                    assert 0
                pose = convert_matrix_b_to_a(pose) # 左手系转右手系
                ego2global_transformation_matrix.append(pose)
            ego2global = np.array(ego2global_transformation_matrix)
            align_pose = align_camera_poses(ego2global)
            for anno, ego2global in zip(annos_out, align_pose):
                anno["ego2global_transformation_matrix"] = ego2global.tolist()
                anno["ego2world"] = ego2global.tolist()
            result['frame_emb'] = align_pose
            # result['frame_emb'][:, :3, 3] = result['frame_emb'][:, :3, 3] / self.trans_scale # 取消 100
        else:
            result['frame_emb'] = np.array([np.zeros((4, 4)) for frame in annos])

        if self.with_original_info:
            result["annos"] = json.dumps(annos_out, indent=2, sort_keys=True)
            result["case_name"] = case_name
        # if
        result['bev_map_with_aux'] = result["bbox"] # # [V, T, C, H, W]
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
    # read_path = "/kaiwu_vepfs/kaiwu/xujin2/data/carla/zhiji_cam_params/0000.json"
    # save_path = read_path.replace(".json", "_format.json")
    # with open(read_path, "r") as f:
    #     json_data = json.load(f)
    # with open(save_path, "w") as f:
    #     json.dump(json_data, f, indent=4)

    raw_meta_files = [
        # 继续车道线尝试
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/cutin_20250415",
        # 公交车绕行
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/bigcar_detour_20250418",
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/yuxin_bus_20250424_v1",
        # 多车道线选择
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/geo_edit_demo2",
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/busovertake_20250513", # 公交车绕行
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/frontbrake_20250515" # 高速突然刹车
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_parkedcarcutin", # 左转
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_parkedcarcutin_triple_lane",  #
        # 0522突然刹停车
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_frontbrake_town04", # 急刹车
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_parkedcarcutin_0522/", # 右侧起步别车 #带导航点
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/cutin_20250512", # cutin
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_cutin3_0522/", # cutin2
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_FrontBrake1_20250528144750",  # 急刹车
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_FrontBrake1_20250528122519",  ## 急刹车
        # "/kaiwu_vepfs/kaiwu/xujin2/data/carla/old_0303/data_CutIn3_20250528193257_checked",  # cutin100
        # 0611 前车刹停
        # "/kaiwu_vepfs/kaiwu/panze2/carla/data/data_FrontBrake1_20250610122702_checked/",
        # "/kaiwu_vepfs/kaiwu/panze2/carla/data/data_FrontBrake1_20250610095247_checked/",
        # 0612
        # "/kaiwu_vepfs/kaiwu/panze2/carla/data/data_FrontBrake1_20250612201114_checked",
        # 0614
        "/kaiwu_vepfs/kaiwu/panze2/carla/data/data_CutIn3_20250612103719_checked", # 95 cutin case
    ]
    # pap_cam_init_path = "/kaiwu_vepfs/kaiwu/xujin2/data/carla/pap_cam_params"    # pap 交付
    pap_cam_init_path = "/kaiwu_vepfs/kaiwu/xujin2/data/carla/zhiji_cam_params2"    # zhiji
    # pap_cam_init_path = None
    # pap_cam_init_path = None
    # 7v
    camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", "rear_camera",
                   "left_rear_camera", "left_front_camera", "center_camera_fov30"]
    scene_description_file = [
        # "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption_collect/pap.pkl",
        "/kaiwu_vepfs/kaiwu/xujin2/data/pap_pvb/caption_collect/zhiji.pkl",
    ]
    caption_join_func = "format_caption"
    add_controller = {
        "dataset_tag": "zhiji",
        "controller": "zhiji",
    }

    random_caption = "random" # ["chose", "random"] 选择模式和随机模式
    random_caption_key = {
        "weather": {
            "choices": [
                'sunny',  # 晴天
                'foggy',  # 有雾
                'rainy',  # 下雨
                'cloudy',  # 多云
                'clear',  # 晴朗无云
                'snowy'  # 下雪
            ],
            "weights": [0.5, 0.1, 0.15, 0.10, 0.1, 0.05],
        },
        "time": {
            "choices": [
                'daytime',  # 白天
                'night time',  # 夜间（表达方式一）
                'evening',  # 傍晚
                'night',  # 夜晚（表达方式二）
                'dusk',  # 黄昏
                'nighttime',  # 夜间（表达方式三）
                'early morning',  # 清晨
                'time is irrelevant in the provided context'  # 当前场景中时间无关紧要
            ],
            "weights": [0.55, 0.10, 0.05, 0.10, 0.1, 0.05, 0.1, 0.00],
        },
        "lighting": {
            "choices": [
                'bright',  # 明亮
                'evening with street light',  # 傍晚有路灯
                'evening without street light',  # 傍晚无路灯
                'backlighting'  # 逆光
            ],
            "weights": [1.0, 0.0, 0.0, 0.0],
        },
        "road_type": {
            "choices": [
                'urban',  # 城市道路
                'highway',  # 高速公路
                'countryside',  # 乡村道路
                'tunnel',  # 隧道
                'road'  # 一般道路（不具体）
            ],
            "weights": [0.25, 0.25, 0.2, 0.05, 0.25],
        }
    }

    add_geo = True

    sequence_length = [1, 17, 33, 65]
    fps_list = [10]
    data_fps = 10
    image_size = (256, 512)
    # image_size = (512, 1024)
    full_size = (image_size[0], image_size[1] * len(camera_list))  # MULTI_VIEW
    pd = CARLAVariableDataset(
                raw_meta_files = raw_meta_files,
                pap_cam_init_path = pap_cam_init_path,
                path_to_aoss_config = "/kaiwu_vepfs/kaiwu/xujin2/aoss.conf",
                s3_path = "",
                camera_list = camera_list,
                sequence_length = sequence_length,
                fps_list = fps_list,
                data_fps = data_fps,
                expected_vae_size = image_size,
                full_size = full_size,
                add_geo = add_geo,
                scene_description_file = scene_description_file,
                random_caption =random_caption,
                random_caption_key = random_caption_key,
                caption_join_func = caption_join_func,
                add_controller = add_controller,
            )
    print('carla:', len(pd))
    # 将 Tensor 转为帧列表
    to_pil = transforms.ToPILImage()  # 转换为 PIL Image
    from torch.utils.tensorboard import SummaryWriter
    loacl_show_dataset_tb_dir = "/kaiwu_vepfs/kaiwu/xujin2/code_hsy/magicdrivedit/show_dataset/dataset/tensorboard"
    if os.path.exists(loacl_show_dataset_tb_dir):
        import shutil
        shutil.rmtree(loacl_show_dataset_tb_dir)
    writer = SummaryWriter(loacl_show_dataset_tb_dir)

    from torchvision.io import write_video
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

    def format_video2(input_img_list, num_rows=2):
        # list for V, T, C, H, W
        x = input_img_list[0]
        maps = []
        for map in input_img_list[1:]:
            if x.shape != map.shape:
                map = F.interpolate(
                    map,
                    scale_factor=2,
                    mode="nearest",
                )
            maps.append(map)

        blended = x
        for map in maps:
            blended = torch.where(map > 0, map, blended)

        # blended shape: V, T, C, H, W
        v, t, c, h, w = blended.shape

        # 补齐到 num_rows 的整数倍
        pad_len = (num_rows - v % num_rows) % num_rows
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, t, c, h, w), dtype=blended.dtype, device=blended.device)
            blended = torch.cat([blended, pad_tensor], dim=0)  # << 关键修改
            v += pad_len

        return rearrange(blended, '(r v2) t c h w -> t c (r h) (v2 w)', r=num_rows)


    from torchvision.io import write_video


    def save_sample(x: torch.Tensor,
                    fps: int = 8,
                    save_path: str = None,
                    normalize: bool = True,
                    value_range: tuple = (0, 1),
                    video_codec: str = "ffv1",
                    ffmpeg_options: dict = None):
        """
        Save a tensor as an image or losslessly compressed video.

        Args:
            x (Tensor): shape [C, T, H, W]
            fps (int): Frames per second for video.
            save_path (str): Path (without extension) to save the file.
            normalize (bool): Whether to normalize x to [0,1] before scaling.
            value_range (tuple): (min, max) expected range of x.
            video_codec (str): FFmpeg codec (e.g., 'ffv1', 'libx264rgb').
            ffmpeg_options (dict): Additional FFmpeg options (values as str).
        """
        assert x.ndim == 4, f"Expected 4D tensor [C,T,H,W], got {x.shape}"

        # Single-frame: save as image
        if x.shape[1] == 1:
            out_path = save_path + ".png"
            img = x.squeeze(1)
            save_image([img], out_path, normalize=normalize, value_range=value_range)
            print(f"Saved image to {out_path}")
            return out_path

        # Multi-frame: prepare video tensor
        # Normalize if needed
        if normalize:
            low, high = value_range
            x = x.clone().clamp(min=low, max=high)
            x = (x - low) / (max(high - low, 1e-5))

        # Convert to uint8 [T,H,W,C]
        vid = (x * 255).add(0.5).clamp(0, 255).permute(1, 2, 3, 0).to(torch.uint8).cpu()

        # Determine extension based on codec
        ext = ".mkv" if video_codec.lower() == "ffv1" else ".mp4"
        out_path = save_path + ext

        # Default FFV1 options if none provided
        if ffmpeg_options is None:
            if video_codec.lower() == "ffv1":
                ffmpeg_options = {"level": "3", "g": "1"}
            elif video_codec.lower() in ("libx264rgb", "h264"):  # H264 lossless RGB
                ffmpeg_options = {"crf": "0", "preset": "veryslow", "pix_fmt": "rgb24"}
            else:
                ffmpeg_options = {}

        write_video(out_path, vid, fps=fps, video_codec=video_codec, options=ffmpeg_options)
        print(f"Saved video to {out_path} with codec={video_codec}")
        return out_path


    def clean_caption_for_filename(s, max_length=150):
        # 1. 去掉 Caption 内容
        s = re.sub(r'\s*Caption: ".*?"\s*', ' ', s)
        # 2. 去掉左右空格
        s = s.strip()
        # 3. 替换非法字符（比如 []{}|:+ 空格都换成 _）
        s = re.sub(r'[\[\]\{\}\|\:\+\s]', '_', s)
        # 4. 多个连续的 _ 合并成一个
        s = re.sub(r'_+', '_', s)
        # 5. 限制文件名长度
        if len(s) > max_length:
            s = s[:max_length]
        return s
    index_list = [
        # f"{random.randint(0, len(ADdataset) - 1)}-17-10"
        f"{id}-17-10"
        # f"{id}-2-10"
        for id in range(10)
    ]
    # for debug
    # index_list = [
    #     "7383918-10-17",
    #     "24716883-10-17",
    #     "5998949-10-17",
    # ]
    save_local = True
    for index in tqdm(index_list):
        result = pd.get_item(index)
        # result = ADdataset.get_item("0-65-10")
        # 全部转换成 [V, T, C, H, W]
        video = result["pixel_values"].permute(1, 0, 2, 3, 4) #  [T, V, C, H, W] -> [V, T, C, H, W]
        bbox = result["bbox"]
        video = denormalize(video)
        show_data_list = [video, bbox]
        show_video = format_video2(show_data_list)
        # caption
        caption = result["captions"][0]
        print(f"cap: {caption}")
        bug_info = pd.get_info()
        bug_info_print_str = json.dumps(bug_info, indent=4, ensure_ascii=False)
        print(
            Style.BRIGHT + Fore.RED + bug_info_print_str + Style.RESET_ALL
        )
        # 用正则去掉Caption部分
        caption_clean = clean_caption_for_filename(caption)
        if save_local:
            save_local_video = rearrange(show_video, "t c h w -> c t h w")
            save_name = loacl_show_dataset_tb_dir + f"/{index}{caption_clean}"
            save_sample(save_local_video, save_path=save_name)
        show_video = show_video.unsqueeze(0)
        writer.add_video(
            "idx:{} cap:{}".format(index, caption_clean),
            show_video,
            fps=2,
        )
    writer.close()

