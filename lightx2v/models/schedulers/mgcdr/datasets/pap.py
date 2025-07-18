import io
import os
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
import random
import re
try:
    from alt.coordinate.bbox import LiDARInstance3DBoxes
    from alt.coordinate.points.utils import Camera3DPointsTransfer
except:
    from magicdrivedit.datasets.alt.coordinate.bbox import LiDARInstance3DBoxes
    from magicdrivedit.datasets.alt.coordinate.points.utils import Camera3DPointsTransfer
# from dwm.datasets.alt.visualize.object.vis_camera_3d import expand_plot_rect3d_on_img
import pickle as pkl
from torchvision import transforms
import warnings
import time
from colorama import Fore, Style
import copy
import magicdrivedit.utils.aoss
from magicdrivedit.registry import DATASETS, build_module
from mmcv.parallel import DataContainer
from magicdrivedit.utils.inference_utils import edit_pos
import traceback
from pathlib import Path
CLASS_BBOX_COLOR_MAP = {
    'VEHICLE_SUV': (255, 0, 0),  # "运动型多用途轿车" car #红
    'VEHICLE_CAR': (255, 0, 0),  # car #红
    'VEHICLE_TRUCK': (255, 128, 0),  # "大型货车" truck #橙
    'CYCLIST_BICYCLE': (0, 255, 0),  # "非机动车骑行者"  bicycle #绿
    'PEDESTRIAN_NORMAL': (0, 0, 255),  # "普通行人" pedestrian #蓝
    'CYCLIST_MOTOR': (0, 255, 0),  # "机动车骑行者" motorcycle #绿
    'VEHICLE_BUS': (255, 255, 0),  # "巴士" bus #黄
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

IMG_FPS = 100

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
# 定义边框线的连接点
line_indices = [
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)
]

CAR_HEAD_COLOR = (255, 255, 255)
from itertools import islice

# GOP:CONE, ISOLATION_BARRER, POLE, BARRIER, TRIANGLE_WARNING, CONSTRUCTION_SIGN, PARKING_LOCK, BARRIER_GATE, CART, ANIMAL.

DEFAULT_CUBE = np.array(
    [
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=np.float64
)
BBOX_LINES = [[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [
    6, 7], [7, 8], [8, 5], [1, 5], [2, 6], [3, 7], [4, 8]]
CROSS_LINES = [[3, 8], [4, 7]]

def smart_read_json(filepath):
    """
    智能读取 JSON 文件，返回所有字典对象的列表
    - 兼容单字典、字典列表、按行存储的字典
    - 自动处理空文件和格式错误
    """
    try:
        # 检查文件是否存在
        if not Path(filepath).exists():
            print(f"⚠️ 文件不存在：{filepath}")
            return []

        # 尝试标准 JSON 加载
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 统一返回字典列表
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                print(f"⚠️ 非字典结构：{filepath}")
                return []

    except json.JSONDecodeError:
        # 处理非标准 JSON（如按行存储）
        print(f"⚙️ 尝试按行解析：{filepath}")
        return read_json_lines(filepath)

    except Exception as e:
        print(f"❌ 未知错误：{filepath} - {str(e)}")
        return []

def read_json_lines(filepath):
    """处理按行存储的 JSON 对象"""
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

def rotz(t, dims=3):
    """Rotation about the z-axis."""

    is_array = isinstance(t, np.ndarray)
    c = np.cos(t)
    s = np.sin(t)
    zeros = np.zeros_like(t) if is_array else 0
    ones = np.ones_like(t) if is_array else 1
    if dims == 3:
        return np.array([[c, -s, zeros], [s, c, zeros], [zeros, zeros, ones]])
    elif dims == 2:
        return np.array([[c, -s], [s, c]])
    else:
        raise NotImplementedError

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

def box_dim2corners(dim):
    """
    dim: [h, w, l, x, y, z, yaw]

    8 corners: np.array = n*8*3(x, y, z)
    #         7 -------- 6
    #        /|         /|
    #       4 -------- 5 .
    #       | |        | |
    #       . 3 -------- 2
    #       |/         |/
    #       0 -------- 1

                ^ x(l)
                |
                |
                |
    y(w)        |
    <-----------O
    """
    h = dim[0]
    w = dim[1]
    l = dim[2]
    x = dim[3]
    y = dim[4]
    z = dim[5]
    yaw = dim[6]

    # 3d bounding box corners
    Box = DEFAULT_CUBE.copy()
    Box[:, 0] *= l
    Box[:, 1] *= w
    Box[:, 2] *= h
    Box = Box.T

    if isinstance(yaw, float):
        R = rotz(yaw)
    else:
        R = np.array(yaw)
    corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return np.transpose(corners_3d)


def color_depth(depths: np.ndarray, scale=None, cmap='viridis',
                out: Literal['uint8,0,255', 'float,0,1'] = 'uint8,0,255'):
    if scale is None:
        scale = depths.max() + 1e-10
    colors = cm.get_cmap(cmap)(depths / scale)[..., :3]
    if out == 'uint8,0,255':
        return np.clip(((colors) * 255.).astype(np.uint8), 0, 255)
    elif out == 'float,0,1':
        return colors
    else:
        raise RuntimeError(f"Invalid out={out}")


def convert_pts_in_image(H, W, c2w, intr, pts):
    pts_ex = np.concatenate((pts, np.ones((*pts.shape[:-1], 1))), axis=-1)
    # w2i = c2i @ w2c
    w2c = np.linalg.inv(c2w)[:3, :]
    # w2c = c2w.cpu().inverse().to(intr.device)
    w2i = np.einsum('...ik,...kj->...ij', np.array(intr), w2c)
    # intr @ c2w.inverse()
    # w2i @ pts_exwc
    # print(w2i.shape,w2i.dtype,pts_ex.shape,pts_ex.dtype)
    uvd1 = np.einsum('ij,...j->...i', w2i, pts_ex)
    d = uvd1[..., 2]
    uv = uvd1[..., 0:2] / np.abs(uvd1[..., 2:3])
    u, v = np.moveaxis(uv, -1, 0)
    # print(u,v,u.shape)
    mask = (d > 0) & (u < W - 1) & (u >= 0) & (v < H - 1) & (v >= 0)
    # -1的原因：有时候由于运算精度的原因导致计算filter_mask的时候会溢出
    n = mask.sum()
    u = u[mask]
    v = v[mask]
    d = d[mask]
    return mask, n, u, v, d


def project_to_camera(points, camera_intrinsic, sensor_extrinsic, return_deep=False):
    # sensor to camera, 4xN
    camera_points = sensor_extrinsic.dot(np.array(points))
    # z is front in camera, 3xN
    camera_points = camera_points[:-1]
    # camera to image, 3xN
    img_points = camera_intrinsic.dot(camera_points)
    img_points = img_points[..., img_points[2, :] >= 0]  # 只保留深度为正的点
    img_points[0, :] /= img_points[2, :]
    img_points[1, :] /= img_points[2, :]
    img_points = img_points.transpose()  # Nx3 array
    if return_deep:
        return img_points
    return img_points[:, :2]


def convert_pts_in_fisheye_image(H, W, c2w: torch.Tensor, intr: torch.Tensor, pts: torch.Tensor, D):
    # pts_ex = torch.cat([pts, torch.ones([*pts.shape[:-1], 1])], axis=-1)
    pts_ex = np.concatenate((pts, np.ones((*pts.shape[:-1], 1))), axis=-1)
    w2c = c2w.cpu().inverse().to(intr.device)

    uvd1 = torch.einsum('ij,...j->...i', w2c, pts_ex)
    uv = uvd1[..., 0:2] / torch.abs(uvd1[..., 2:3])
    x, y = torch.movedim(uv, -1, 0)

    cx = intr[0, 2]
    cy = intr[1, 2]
    fx = intr[0, 0]  # Assuming fx = fy
    fy = intr[1, 1]
    # Calculate polar coordinates

    r = np.sqrt(x ** 2 + y ** 2)  # 距离光轴的距离
    theta = np.arctan(r)  # 光轴到点的角度
    d_theta = theta + np.power(theta, 3) * D[0] \
              + np.power(theta, 5) * D[1] \
              + np.power(theta, 7) * D[2] \
              + np.power(theta, 9) * D[3]

    # Convert polar coordinates to image coordinates
    # 使用等距鱼眼投影模型计算图像坐标 (u, v)
    u1 = d_theta * fx * x / r + cx
    v1 = d_theta * fy * y / r + cy

    # print(u,v,u.shape)
    mask = (u1 < W - 1) & (u1 >= 0) & (v1 < H - 1) & (v1 >= 0)
    # -1的原因：有时候由于运算精度的原因导致计算filter_mask的时候会溢出
    n = mask.sum().item()
    u1 = u1[mask]
    v1 = v1[mask]
    return mask, n, u1, v1, None


def interp_two_points(pt1, pt2, thres=1.0, step=1.0):
    # determin major axes
    if np.abs(pt1[0] - pt2[0]) > np.abs(pt1[1] - pt2[1]):
        major, minor = 0, 1
    else:
        major, minor = 1, 0

    # don't need interp
    if np.abs(pt1[major] - pt2[major]) < thres:
        return pt1[np.newaxis, ...]

    if pt1[major] < pt2[major]:
        x = np.array([pt1[major], pt2[major]])
        y = np.array([pt1[minor], pt2[minor]])
        z = np.array([pt1[2], pt2[2]])
        xval = np.arange(pt1[major], pt2[major], step)
        yval = np.interp(xval, x, y)
        zval = np.interp(xval, x, z)
    else:
        x = np.array([pt2[major], pt1[major]])
        y = np.array([pt2[minor], pt1[minor]])
        z = np.array([pt2[2], pt1[2]])
        xval = np.arange(pt2[major], pt1[major], step)
        yval = np.interp(xval, x, y)
        zval = np.interp(xval, x, z)
        xval, yval, zval = xval[::-1], yval[::-1], zval[::-1]

    if major == 1:
        xval, yval, zval = yval, xval, zval

    return np.c_[xval, yval, zval]


def linear_interp_3d(geo3d, cam):
    interp_res = list()
    for i in range(1, len(geo3d)):
        if cam in ["center_camera_fov120", "center_camera_fov30"]:
            interp_geo2d = interp_two_points(geo3d[i - 1], geo3d[i], step=1)
        else:
            interp_geo2d = interp_two_points(geo3d[i - 1], geo3d[i], thres=0.2, step=0.2)
        # if geo3d[i-1][0] < 0 or geo3d[i][0] < 0:
        #     interp_geo2d = interp_two_points(geo3d[i - 1], geo3d[i], step=step)
        # else:
        #     interp_geo2d = geo3d[i-1][np.newaxis, ...]
        interp_res.append(interp_geo2d)
    interp_res.append(geo3d[-1][None, ...])
    return np.concatenate(interp_res, axis=0)


# region[previous code]
# def project_to_camera(points, camera_intrinsic, sensor_extrinsic, cam_dist, cam):
#     if len(cam_dist) == 1:
#         if "fov195" in cam:
#             img_dist_type = "oriK"
#         else:
#             img_dist_type = "oriK"
#             cam_dist = np.zeros((4, 1))
#     else:
#         img_dist_type = "scaramuzza"
#
#     config = {
#         "0": {
#             "sensor_name": cam,
#             "param": {
#                 "img_dist_type": img_dist_type,
#                 "cam_K": {
#                     "data": camera_intrinsic,
#                 },
#                 "cam_dist": {
#                     "data": cam_dist
#                 },
#                 "cam_K_new": {
#                     "data": camera_intrinsic,
#                 },
#             }
#         }
#     }
#     # sensor to camera, 4xN
#     # Nx3 -> 4xN
#     sensor_extrinsic = sensor_extrinsic.T
#     points = np.concatenate([points.T, np.ones([1, *points.shape[:-1]])], axis=0)
#     camera_points = sensor_extrinsic.dot(np.array(points))
#     camera_points = linear_interp_3d(camera_points.T[:, :3], cam)
#
#     # remove points which depth < 0
#     valid_depth = (camera_points[:, 2] > 0) & (camera_points[:, 2] < 100)
#     if not np.any(valid_depth):
#         return np.zeros((0, 2))
#     camera_points = camera_points[valid_depth, :]
#
#     camera_intrinsic_new = CameraIntrinsic(config)
#     img_points_tt = camera_intrinsic_new.projectPoints(camera_points, )
#     return img_points_tt

# def proj(mat_3x3: np.ndarray, xyz: np.ndarray):
#     uvd = np.sum(mat_3x3 * np.expand_dims(xyz, axis=-2), axis=-1)
#     uv = uvd[..., 0:2] / np.abs(uvd[..., 2:3]).clip(1e-5)
#     return uv[..., 0], uv[..., 1], uvd[..., 2]
# endregion

def transfer_to_pinhole(cls, camera_3ds, camera_intrinsic, camera_dist=[]):
    point_2d = camera_3ds @ camera_intrinsic.T
    point_depth = torch.clamp(point_2d[..., 2:3], min=1e-5, max=1e5)
    points_clamp = torch.cat([point_2d[..., :2], point_depth], dim=1)

    point_2d_res = points_clamp[..., :2] / points_clamp[..., 2:3]
    return point_2d_res


def proj(mat_3x3: np.ndarray, xyz: np.ndarray, dist_matrix: np.ndarray = None):
    # 投影到图像平面
    uvd = np.sum(mat_3x3 * np.expand_dims(xyz, axis=-2), axis=-1)
    uv = uvd[..., 0:2] / np.abs(uvd[..., 2:3]).clip(1e-5)
    x, y = uv[..., 0], uv[..., 1]

    if dist_matrix is not None:
        # 从 dist_matrix 提取畸变系数
        k1, k2, p1, p2, k3 = dist_matrix[1, :5]

        # 计算径向畸变
        r2 = x ** 2 + y ** 2
        radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # 计算切向畸变
        x = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y = y * radial_distortion + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

    return x, y, uvd[..., 2]


def draw_3Dbbox_proj(img: np.ndarray,
                     bbox_list: list,
                     intr: np.ndarray,
                     ext: np.ndarray,  # c2e
                     inplace=True,
                     color_list: list = None,
                     thickness: int = 2):
    H, W, *_ = img.shape
    if not inplace:
        img = img.copy()

    ego2cam = np.linalg.inv(ext)  # 4*4

    for idx_box, box in enumerate(bbox_list):
        dim = [0] * 7
        # x, y, z, l, w, h, yaw
        dim[0] = box[5]  # h
        dim[1] = box[4]  # w
        dim[2] = box[3]  # l
        dim[3] = box[0]  # x
        dim[4] = box[1]  # y
        dim[5] = box[2]  # z
        dim[6] = box[-1]  # yaw
        box_corners = box_dim2corners(dim)
        pts_ex = np.concatenate((box_corners, np.ones((*box_corners.shape[:-1], 1))), axis=-1)
        pts_ex_reshape = pts_ex.reshape(-1, pts_ex.shape[-1])
        # cube_pts_in_cam = project_to_camera(np.transpose(pts_ex_reshape), np.array(intr), ego2cam, True)
        cube_pts_in_cam = np.einsum('ij,...j->...i', ego2cam, pts_ex)[:, :-1]  # (8, 3)
        # u1, v1, _ = proj(intr, cube_pts_in_cam)
        # _, _, u11, v11, _ = convert_pts_in_image(H, W, ext, intr, box_corners)

        # for LINES, LINE_COLOR in zip([BBOX_LINES, CROSS_LINES], [(0, 255, 0), (255, 0, 0)]):
        for LINES, LINE_COLOR in zip([BBOX_LINES, CROSS_LINES], [color_list[idx_box], (255, 0, 0)]):
            # for LINES, LINE_COLOR in zip([BBOX_LINES], [(0, 255, 0)]):
            # cube_edge_indices = cube_pts_in_cam.new_tensor(LINES, dtype=np.int64) - 1 # (12, 2)
            cube_edge_indices = np.array(LINES, dtype=np.int64) - 1  # (12, 2)
            cube_edge_pts_in_cam = cube_pts_in_cam[cube_edge_indices]  # (12, 2, 3)
            for i in range(len(LINES)):
                p1 = cube_edge_pts_in_cam[i, 0].copy()
                p2 = cube_edge_pts_in_cam[i, 1].copy()
                if p1[2] > p2[2]:
                    p1 = cube_edge_pts_in_cam[i, 1].copy()
                    p2 = cube_edge_pts_in_cam[i, 0].copy()
                if p1[2] < 0.01 and p2[2] >= 0.01:
                    k = (0.01 - p1[2]) / (p2[2] - p1[2])
                    p1 += k * (p2 - p1)
                cube_edge_pts_in_cam[i, 0] = p1
                cube_edge_pts_in_cam[i, 1] = p2
            u1, v1, _ = proj(intr, cube_edge_pts_in_cam)
            # _, _, u1, v1, _ = convert_pts_in_image(H, W, ext, intr, cube_edge_pts_in_cam)
            d1 = cube_edge_pts_in_cam[..., 2]
            if (d1 < 0.01).all() or (((u1 < 0) | (u1 >= W)) | ((v1 < 0) | (v1 >= H))).all():
                continue
            u1 = np.floor(u1).astype(int)
            v1 = np.floor(v1).astype(int)

            for i in range(len(LINES)):
                if d1[i, 0] < 0.01 and d1[i, 1] < 0.01:
                    continue
                p1 = (u1[i, 0], v1[i, 0])
                p2 = (u1[i, 1], v1[i, 1])
                cv2.line(img, p1, p2, LINE_COLOR, thickness)
    return img


class MyCamera3DPointsTransfer(Camera3DPointsTransfer):
    printlog = False

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

        if len(points) > 0:
            return np.concatenate(points), np.asarray(masks), np.stack(lidar_in_cams), np.stack(
                lidar_in_worlds)  # [n,8,3]
        return [], [], [], []  # [n,8,3]

def plot_filled_rect3d_on_img_numpy(
        img,
        num_rects,
        rect_corners,
        with_borderline=True,
        colors=[(0, 255, 0)],
        polycolors=[(0, 0, 0)],
        thickness=1,
        alpha=0.3,
):
    """
    使用 NumPy 替代 OpenCV 部分操作，加速绘制 3D 矩形的填充与边界。
    """
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0],  # Left face
    ]
    overlay = np.zeros_like(img, dtype=np.float32)  # 创建透明图层

    if isinstance(colors, tuple):
        colors = [colors]

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)

        # 确定当前多边形填充颜色
        polycolor = polycolors[i % len(polycolors)]

        # 使用 NumPy 填充多边形
        for face in faces:
            face_points = corners[face]
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool_)

            # 使用 matplotlib 的 Path 创建多边形区域
            poly_path = Path(face_points)
            y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
            points = np.stack((x.ravel(), y.ravel()), axis=-1)
            mask = poly_path.contains_points(points).reshape(img.shape[:2])

            # 应用填充颜色到透明图层
            for c in range(3):  # R, G, B 通道
                overlay[..., c] += mask * polycolor[::-1][c]  # RGB 转 BGR 顺序

    # 使用 NumPy 进行透明叠加
    img = (1 - alpha) * img + alpha * overlay
    img = img.astype(np.uint8)  # 转换回 uint8 类型

    if with_borderline:
        # 绘制边界线
        for i in range(num_rects):
            corners = rect_corners[i].astype(np.int32)

            # 边界线颜色
            color = colors[i % len(colors)]
            for face in faces:
                face_points = corners[face]
                for j in range(len(face_points)):
                    start_point = face_points[j]
                    end_point = face_points[(j + 1) % len(face_points)]
                    img = draw_line_numpy(img, start_point, end_point, color, thickness)

            # 画车头的叉叉线
            head_color = color
            for start, end in ((4, 6), (5, 7)):
                img = draw_line_numpy(img, corners[start], corners[end], head_color, thickness)

    return img


def draw_line_numpy(img, start, end, color, thickness=1):
    """
    使用 NumPy 绘制线条。
    """
    x1, y1 = start
    x2, y2 = end

    # 计算线段上的点
    num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
    x, y = np.linspace(x1, x2, num_points).astype(np.int32), np.linspace(y1, y2, num_points).astype(np.int32)

    # 处理线条宽度
    for dx in range(-thickness // 2, thickness // 2 + 1):
        for dy in range(-thickness // 2, thickness // 2 + 1):
            xx = np.clip(x + dx, 0, img.shape[1] - 1)
            yy = np.clip(y + dy, 0, img.shape[0] - 1)
            img[yy, xx, 0] = color[0]
            img[yy, xx, 1] = color[1]
            img[yy, xx, 2] = color[2]

    return img


import numpy as np
import cv2
from matplotlib.path import Path

from PIL import Image, ImageDraw
import numpy as np


def plot_filled_rect3d_resized_pillow(
        img,
        num_rects,
        rect_corners,
        with_borderline=True,
        colors=[(0, 255, 0)],
        polycolors=[(0, 0, 0)],
        thickness=1,
        alpha=0.3,
        resize_factor=0.5
):
    """
    使用 Resize-Draw-Upsample 方法优化 3D 矩形绘制。

    Args:
        img (numpy.ndarray): 输入图像。
        num_rects (int): 多边形数量。
        rect_corners (numpy.ndarray): 多边形的顶点坐标，形状为 [num_rects, 8, 2]。
        with_borderline (bool): 是否绘制边界线。
        colors (list[tuple]): 每个多边形的边界线颜色。
        polycolors (list[tuple]): 每个多边形的填充颜色。
        thickness (int): 边界线宽度。
        alpha (float): 填充透明度。
        resize_factor (float): 缩小比例（如 0.5 表示缩小到原始大小的一半）。

    Returns:
        numpy.ndarray: 绘制后的图像。
    """
    # Step 1: 缩小图像
    original_size = img.shape[:2]  # (height, width)
    resized_size = (int(original_size[1] * resize_factor), int(original_size[0] * resize_factor))  # (width, height)

    pil_img = Image.fromarray(img).convert("RGBA")  # 转换为 Pillow 图像
    resized_img = pil_img.resize(resized_size, Image.BILINEAR)  # 缩小图像

    # 缩小后的顶点坐标
    resized_corners = (rect_corners * resize_factor).astype(int)

    # 创建透明图层
    overlay = Image.new("RGBA", resized_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # 定义矩形的各个面
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0],  # Left face
    ]

    # Step 2: 在缩小图像上绘制多边形
    for i in range(num_rects):
        corners = resized_corners[i]
        polycolor = tuple(polycolors[i % len(polycolors)]) + (int(alpha * 255),)  # 修复拼接问题

        # 绘制每个面的填充
        for face in faces:
            face_points = [tuple(corners[point]) for point in face]
            draw.polygon(face_points, fill=polycolor)

        if with_borderline:
            color = tuple(colors[i % len(colors)]) + (255,)  # 修复拼接问题
            for face in faces:
                face_points = [tuple(corners[point]) for point in face]
                draw.line(face_points + [face_points[0]], fill=color, width=max(1, int(thickness * resize_factor)))

            # 绘制车头的叉叉线
            p1, p2, p3, p4 = 4, 5, 6, 7
            head_color = color
            draw.line([tuple(corners[p1]), tuple(corners[p3])], fill=head_color,
                      width=max(1, int(thickness * resize_factor)))
            draw.line([tuple(corners[p2]), tuple(corners[p4])], fill=head_color,
                      width=max(1, int(thickness * resize_factor)))

    # 将透明层叠加到缩小的图像上
    resized_img = Image.alpha_composite(resized_img, overlay)

    # Step 3: 上采样回原始大小
    upsampled_img = resized_img.resize((original_size[1], original_size[0]), Image.BILINEAR)

    # Step 4: 合并到原始图像
    final_img = Image.alpha_composite(pil_img, upsampled_img)
    return np.array(final_img.convert("RGB"))


def plot_filled_rect3d_optimized(img, num_rects, rect_corners, with_borderline=True, colors=[(0, 255, 0)],
                                 polycolors=[(0, 0, 0)], thickness=1, alpha=0.3):
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0]  # Left face
    ]

    # 创建透明图层
    height, width, _ = img.shape
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")

    for i in range(num_rects):
        corners = rect_corners[i].astype(int)

        # 获取填充颜色
        if i >= len(polycolors):
            polycolor = polycolors[-1]
        else:
            polycolor = polycolors[i]

        # 绘制每个面
        for face in faces:
            face_points = [tuple(corners[point]) for point in face]
            draw_overlay.polygon(face_points, fill=(*polycolor, int(255 * alpha)))

    # 叠加透明层到原图
    pil_img = Image.fromarray(img).convert("RGBA")
    img_with_overlay = Image.alpha_composite(pil_img, overlay)

    if with_borderline:
        draw = ImageDraw.Draw(img_with_overlay, "RGBA")
        for i in range(num_rects):
            corners = rect_corners[i].astype(int)

            # 获取边界颜色
            if i >= len(colors):
                color = colors[-1]
            else:
                color = colors[i]

            # 绘制边界线
            for face in faces:
                face_points = [tuple(corners[point]) for point in face]
                for j in range(len(face_points)):
                    start_point = face_points[j]
                    end_point = face_points[(j + 1) % len(face_points)]
                    draw.line([start_point, end_point], fill=(*color, 255), width=thickness)

    # 转回 NumPy
    return np.array(img_with_overlay.convert("RGB"))


def plot_filled_rect3d_on_img_pillow_fixed_vertex(
        img, num_rects, rect_corners, with_borderline=True,
        colors=[(0, 255, 0)], polycolors=[(0, 0, 0)], thickness=1, alpha=0.3, printlog=False
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

    height, width = img.shape[:2]  # Get image dimensions

    def is_point_inside_image(point):
        """Check if a point is inside the image boundary."""
        x, y = point
        return 0 <= x < width and 0 <= y < height

    def clip_polygon_to_image(polygon):
        """Clip polygon to the image boundary by discarding faces fully outside the image."""
        return any(is_point_inside_image(point) for point in polygon)

    for i in range(num_rects):
        if printlog: print(f"******* num_rects {i} start ********")
        corners = rect_corners[i]  # .astype(np.int16)

        if i >= len(colors):
            polycolor = polycolors[-1]
        else:
            polycolor = polycolors[i]

        # Draw filled faces
        for fi, face in enumerate(faces):
            if printlog: print(f"******* face {fi} _points start ********")

            face_points = [tuple(corners[point]) for point in face]
            if printlog: print(face_points)
            if clip_polygon_to_image(face_points):  # Only draw if part of the face is within the image
                draw_overlay.polygon(face_points, fill=(*polycolor, int(255 * alpha)))  # Add transparency with alpha
            if printlog: print(f"******* face {fi} _points start ********")
        if printlog: print(f"******* num_rects {i} end ********")
    # Blend the overlay with the original image
    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)

    if with_borderline:
        draw = ImageDraw.Draw(pil_img, "RGBA")
        for i in range(num_rects):
            corners = rect_corners[i]

            if i >= len(colors):
                color = colors[-1]
            else:
                color = colors[i]

            # Draw edges of the cube
            for face in faces:
                face_points = [tuple(corners[point]) for point in face]
                if clip_polygon_to_image(face_points):  # Only draw lines for visible faces
                    for j in range(len(face_points)):
                        start_point = face_points[j]
                        end_point = face_points[(j + 1) % len(face_points)]
                        if is_point_inside_image(start_point) or is_point_inside_image(end_point):
                            draw.line([start_point, end_point], fill=(*color, 255), width=thickness)

            # Draw additional lines for head marking
            head_color = color
            for start, end in ((4, 6), (5, 7)):
                start_point, end_point = tuple(corners[start]), tuple(corners[end])
                if is_point_inside_image(start_point) or is_point_inside_image(end_point):
                    draw.line(
                        [start_point, end_point],
                        fill=(*head_color, 255),
                        width=thickness
                    )

    # Convert to RGB before transforming
    output_img = np.array(pil_img.convert("RGB"))

    return output_img

def plot_filled_rect3d_on_img_opencv(
        img,
        num_rects,
        rect_corners,
        with_borderline=True,
        colors=[(0, 255, 0)],
        polycolors=[(0, 0, 0)],
        thickness=1,
        alpha=0.3,
):
    """
    使用 OpenCV 逐个绘制 3D 矩形的填充和边框。
    """
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [1, 2, 6, 5],  # Right face
        [4, 7, 3, 0],  # Left face
    ]
    overlay = img.copy()

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        polycolor = polycolors[i % len(polycolors)]

        for face in faces:
            face_points = corners[face]
            cv2.fillPoly(overlay, [face_points], color=polycolor[::-1])

        if with_borderline:
            color = colors[i % len(colors)]
            for face in faces:
                face_points = corners[face]
                for j in range(len(face_points)):
                    start = tuple(face_points[j])
                    end = tuple(face_points[(j + 1) % len(face_points)])
                    cv2.line(img, start, end, color=color[::-1], thickness=thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def plot_filled_rect3d_on_img(img, num_rects, rect_corners, with_borderline=True, colors=[(0, 255, 0)],
                              polycolors=[(0, 0, 0)], thickness=1, alpha=0.3):
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

    p1, p2, p3, p4 = 4, 5, 6, 7
    for i in range(num_rects):

        corners = rect_corners[i].astype(np.int32)
        try:
            if i >= len(colors):
                polycolor = polycolors[-1]
            else:
                polycolor = polycolors[i]

            for face in faces:
                face_points = corners[face]
                cv2.fillPoly(overlay, [face_points], color=polycolor[::-1])  # rgb->bgr
        except Exception as e:
            print(repr(e))
            # import pdb;
            # pdb.set_trace()
        # Blend the overlay with the original image to make it semi-transparent
        # Transparency factor
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    if with_borderline:
        for i in range(num_rects):
            # Draw edges of the cube for better visibility
            if i >= len(colors):
                color = colors[-1]
            else:
                color = colors[i]
            for face in faces:
                face_points = corners[face]
                for i in range(len(face_points)):
                    start_point = tuple(face_points[i])
                    end_point = tuple(face_points[(i + 1) % len(face_points)])
                    cv2.line(img, start_point, end_point, color=color, thickness=thickness)

            head_color = color
            for start, end in ((p1, p3), (p2, p4)):  # 与点云检测模型可视化对齐，叉叉画在车头
                cv2.line(
                    img, (corners[start, 0], corners[start, 1]), (corners[end, 0], corners[end, 1]), head_color,
                    thickness, cv2.LINE_AA
                )

    return img.astype(np.uint8)


from PIL import Image, ImageDraw
import numpy as np

from PIL import Image, ImageDraw
import numpy as np
import cv2


def plot_rect3d_on_img_combined(
        img,
        num_rects,
        rect_corners,
        colors=[(0, 255, 0)],
        thickness=1,
        mode="cross",
        alpha=0.5,
        threshold=2,
):
    """
    根据多边形数量动态切换绘制方法：逐个绘制或批量绘制。

    Args:
        img (numpy.ndarray): 输入图像。
        num_rects (int): 3D 矩形数量。
        rect_corners (numpy.ndarray): 3D 矩形顶点坐标，形状为 [num_rects, 8, 2]。
        colors (list[tuple]): 每个矩形的边框颜色。
        thickness (int): 边框线的厚度。
        mode (str): 绘制模式（'cross' 或 'poly'）。
        alpha (float): 透明度（仅在 `poly` 模式下生效）。
        threshold (int): 切换方法的阈值（多边形数量超过此值时切换到 PIL 批量绘制）。

    Returns:
        numpy.ndarray: 绘制后的图像。
    """

    if num_rects <= threshold:
        # 多边形数量少，使用逐个画的方式（基于 OpenCV）
        return plot_rect3d_on_img_opencv(
            img, num_rects, rect_corners, colors, thickness, mode, alpha
        )
    else:
        # 多边形数量多，使用批量画的方式（基于 Pillow）
        return plot_rect3d_on_img_pillow(
            img, num_rects, rect_corners, colors, thickness, mode, alpha
        )


def plot_rect3d_on_img_opencv(
        img, num_rects, rect_corners, colors, thickness=1, mode="cross", alpha=0.5
):
    """
    使用 OpenCV 逐个绘制 3D 矩形。

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
    line_indices = [
        (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)
    ]
    p1, p2, p3, p4 = 4, 5, 6, 7

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)

        if i >= len(colors):
            color = colors[-1]
        else:
            color = colors[i]

        for start, end in line_indices:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )

        head_color = color
        if mode == "cross":
            for start, end in ((p1, p3), (p2, p4)):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    head_color,
                    thickness,
                    cv2.LINE_AA,
                )

        if mode == "poly":
            overlay = img.copy()
            points = np.asarray([corners[p1], corners[p2], corners[p3], corners[p4]], dtype=np.int32)
            cv2.fillPoly(overlay, [points], head_color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


def plot_rect3d_on_img_pillow_fixed(
        img,
        num_rects,
        rect_corners,
        colors=[(0, 255, 0)],
        thickness=1,
        mode="cross",
        alpha=0.5,
        printlog=False
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

    width, height = pil_img.size
    for i in range(num_rects):
        # 获取当前矩形的顶点坐标
        corners = rect_corners[i]  # .astype(np.int16)

        # 获取当前矩形的颜色
        color = tuple(colors[i % len(colors)])

        # 绘制边框线
        if printlog: print(f"******* num_rects {i} start ********")
        for start, end in line_indices:

            start_point = tuple(corners[start])
            end_point = tuple(corners[end])
            if printlog: print(f"******* line {start_point}/{end_point} ********")
            clipped_line = [start_point, end_point]  # clip_line_to_canvas(start_point, end_point, width, height)
            if clipped_line:
                draw.line(clipped_line, fill=color, width=thickness)
        if printlog: print(f"******* num_rects {i} end ********")
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


def plot_rect3d_on_img(img, num_rects, rect_corners, colors=[(0, 255, 0)], thickness=1, mode='cross', alpha=0.5):
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


def calculate_mse(img1, img2):
    """
    计算两幅图像的均方误差 (MSE)。

    Args:
        img1, img2 (numpy.ndarray): 两幅图像，尺寸和通道数必须一致。

    Returns:
        float: 均方误差。
    """
    diff = (img1.astype(np.float32) - img2.astype(np.float32)) ** 2
    mse = np.mean(diff)  # 平均平方误差
    return mse


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
            new_img = plot_filled_rect3d_on_img_pillow_fixed_vertex(img=new_img, num_rects=num_rects, rect_corners=rect_corners_temp,
                                                       with_borderline=with_borderline_on_filed_bbox, colors=colors,
                                                       polycolors=polycolors, thickness=thickness, alpha=alpha)
        else:
            new_img = plot_rect3d_on_img_pillow_fixed(new_img, num_rects, rect_corners_temp, colors, thickness, mode, alpha)
        # # 检查是否捕获到了 RuntimeWarning
        # for warning in w:
        #     if issubclass(warning.category, RuntimeWarning):
        #         print("捕获到 RuntimeWarning:", warning.message)
        #         import pdb;pdb.set_trace()

    # new_img = plot_rect3d_on_img(new_img, num_rects, rect_corners, colors, thickness, mode=mode)

    crop_img = new_img[top_left_y: top_left_y + original_height, top_left_x: top_left_x + original_width]

    return crop_img.astype(np.uint8)


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    B, N_out = bbox_shape[:2]
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    if bboxes is not None:

        for _b in range(B):
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            for _n in range(N_out):
                if _bboxes[_n] is None:
                    continue  # empty for this view
                try:
                    this_box_num = len(_bboxes[_n])
                    ret_bboxes[_b, _n, :this_box_num] = torch.tensor(_bboxes[_n]).to(ret_bboxes)
                    ret_classes[_b, _n, :this_box_num] = torch.tensor(_classes[_n]).to(ret_classes)
                    if masks is not None:
                        ret_masks[_b, _n, :this_box_num] = torch.tensor(masks[_b, _n]).to(ret_masks)
                    else:
                        ret_masks[_b, _n, :this_box_num] = True
                except:
                    traceback.print_exc()
                    # import pdb;
                    # pdb.set_trace()
    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,  # [B, N_out, max_len, 8 x 3]
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


def Ego2GlobalToEgo2LocalWorldFrame(ego2global_transformation_list):
    local2global_world = ego2global_transformation_list[0]
    ego2local_list = []
    for ego2global in ego2global_transformation_list:
        ego2local = np.linalg.inv(local2global_world).dot(ego2global)
        ego2local_list.append(ego2local)
    return ego2local_list


# 超时处理器
import signal


def timeout_handler(signum, frame):
    raise TimeoutError("Read TimeOut!")

# 可视化 Geo 车道线
# bgr 用于opencv 的颜色 # vd2的颜色
# GEO2COLOR = {
#     'WHITE': (255, 255, 255), # RGB(255,255,255) → BGR无变化[4](@ref)，用于同向车道分隔线（虚线可越线变道，实线禁止越线）
#     'RED': (0, 0, 255),       # RGB(255,0,0) → BGR(0,0,255)[4](@ref)，用于禁令标志边框
#     'YELLOW': (0, 255, 255),  # RGB(255,255,0) → BGR(0,255,255)[4](@ref)，用于对向车道分隔线
#     'GREEN': (0, 255, 0),     # RGB(0,255,0) → BGR无变化[4](@ref)，用于导向箭头填充色
#     'BLUE': (255, 0, 0),      # RGB(0,0,255) → BGR(255,0,0)[4](@ref)，用于特殊车道标识
#     'BLACK': (0, 0, 0),       # 所有通道为0[4](@ref)，用于文字符号描边
#     'ORANGE': (0, 165, 255)   # RGB(255,165,0) → BGR(0,165,255)[4](@ref)，用于施工区标线
# }
LANELINE_INDEXES = {
    "SOLID_LANE": 0,
    "SOLID_LANE_NonMotor": 0,
    "DASHED_LANE": 1,
    "DASHED_LANE_NonMotor": 1,
    "LEFT_DASHED_RIGHT_SOLID": 2,
    "LEFT_SOLID_RIGHT_DASHED": 3,
    "DOUBLE_SOLID": 4,
    "DOUBLE_DASHED": 5,
    "FISHBONE_SOLID": 6,
    "FISHBONE_DASHED": 7,
    "THICK_SOLID": 8,
    "THICK_DASHED": 9,
    "VARIABLE_LANE": 10,
    "DIVERSION_BOUNDARY": 11,
    "WAITING_AREA": 12,
}

CURB_INDEXES = {
    "CURB": 13,
    "FENCE": 14,
    "WALL": 15,
    "DITCH_OR_PLANE": 16,
    "OTHER": 17,
}

COLOR_INDEXES = {
    "WHITE": 0,
    "YELLOW": 1,
    "RED": 2,
    "BLUE": 3,
}


GEO_COLOR = {
    "LANELINE": (0, 255, 0),
    "ROADSIDE": (255, 0, 0)
}
def show_lanes_per_camera(
        img_pv,
        anno,
        cam_name,
        gt_lane,
        gt_lane_cls,
        gt_indexs,
        scale_h,
        scale_w,
        thickness,
        show_cls_num = False,
    ):
    record_label = set()
    camera_infos = anno["sensors"]["cameras"]
    calib = camera_infos[cam_name]
    # if cam_name not in [
    #     "front_camera_fov195",
    #     "rear_camera_fov195",
    #     "left_camera_fov195",
    #     "right_camera_fov195",
    # ]:

    viewpad = np.eye(4)
    intrin = np.array(calib["camera_intrinsic"])
    viewpad[:3, :3] = intrin
    extrin = np.array(calib["extrinsic"])
    ego2img = viewpad @ extrin

    for idx, lane_line in enumerate(gt_lane):
        if gt_lane_cls[idx] in GEO_COLOR.keys():
            label = draw_lane_line(
                img_pv,
                lane_line,
                gt_lane_cls[idx],
                gt_indexs[idx],
                ego2img,
                (scale_h, scale_w),
                THICKNESS = thickness,
                GT_COLOR=GEO_COLOR,
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
    one = np.ones((points.shape[0], 1))
    new_points1 = np.concatenate((points, one), axis=-1)
    new_points2 = ego2img @ new_points1.T
    new_points3 = new_points2.T
    new_points3[:, :2] /= new_points3[:, 2:3]
    return new_points3



def get_dataset(case_name, root_path):
    # dataset ["pap", "dazhuo", "vd2", "zhiji"]
    if "clip_" in case_name and "dazhuo" in root_path:
        return "dazhuo"
    elif "zhiji" in case_name or "zhiji" in root_path:
        return "zhiji"
    else:
        return "pap"

def from_vehicle_id_get_controller(vehicle_id):
    # controller : ["J6M", "J6M-T68", "J6E", "MDC"]
    controller = "unknown"
    if vehicle_id is not None:
        vehicle_id_prefix = vehicle_id.split("-")[0]
        if vehicle_id_prefix in ["T68", "T28"]:
            controller = "J6M-T68"
        elif vehicle_id_prefix == "E03":
            controller = "J6E"
        elif vehicle_id == "A02-048":
            controller = "MDC"
        elif vehicle_id == "A02-528": # A02-528这辆车在2.19之前是J6E的域控
            controller = "J6E"
        elif "zhiji" in vehicle_id:
            controller = "zhiji"
        elif "CN" in vehicle_id:
            controller = "J6M"
        elif "A" in vehicle_id: #  A02~A13
            controller = "J6M"
        else:
            pass
    return controller


@DATASETS.register_module()
class PAPVariableDataset(object):
    def __init__(self,
                 raw_meta_files,
                 path_to_aoss_config,
                 s3_path,
                 processed_meta_file=None,
                 camera_list=None,
                 sequence_length=8,
                 fps_list=[10],
                 data_fps=10,
                 split="train",  # * 'test'就不会随机采首帧
                 enable_scene_description=False,
                 exclude_cameras=None,
                 edit_type=None,
                 draw_bbox_mode='cross',
                 split_file=None,
                 scene_description_file=None,
                 CaseName_PAPCaseName_Table_file=None,
                 sqrt_required_text_keys=None,
                 must_text=False,
                 expected_vae_size=(256, 512),
                 full_size=None,
                 # colorful_box=False,
                 colorful_box=True,
                 use_random_seed=False,
                 without_gt=False,
                 fps_stride_list=None,
                 start_zero_frame=False,
                 specific_video_segment_file=None,
                 enable_layout=False,
                 aws_path_transform=None,
                 **kwargs):
        self.draw_bbox_mode = draw_bbox_mode
        self.reader = magicdrivedit.utils.aoss.AossSimpleFile(
            client_config_path=path_to_aoss_config,  s3_path=s3_path)
        self.seq_length = sequence_length
        self.fps_list = fps_list
        self.data_fps = data_fps
        self.split = split
        self.with_original_info = kwargs.get('with_original_info', False)
        print('self.seq_length:', self.seq_length)
        print(Style.BRIGHT + Fore.GREEN + f"with_original_info:{self.with_original_info}" + Style.RESET_ALL)
        print(Style.BRIGHT + Fore.RED + f'self.fps_list:{self.fps_list} in data_fps:{self.data_fps}' + Style.RESET_ALL)
        self.without_gt = without_gt and (split != 'train')
        self.fps_stride_list = fps_stride_list
        self.start_zero_frame = start_zero_frame
        self.specific_video_segment = None
        if specific_video_segment_file:
            self.specific_video_segment = json.load(open(specific_video_segment_file, 'r'))
            print('self.specific_video_segment:', self.specific_video_segment)
        print(
            Style.BRIGHT + Fore.RED + f'self.without_gt:{self.without_gt} self.start_zero_frame:{self.start_zero_frame}' + Style.RESET_ALL)

        self.enable_scene_description = enable_scene_description
        self.exclude_cameras = exclude_cameras
        self.edit_type = edit_type
        self.segment_annos = []
        self.segment_infos = []
        self.debug = kwargs.get('debug', False)
        self.must_text = must_text
        self.expected_vae_size = expected_vae_size
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
        self.full_size = full_size
        self.colorful_box = colorful_box
        self.use_random_seed = use_random_seed
        self.with_filled_bbox = kwargs.get('with_filled_bbox', False)
        self.add_velocity_to_text = kwargs.get('add_velocity_to_text', False)
        self.with_bbox_coords = kwargs.get('with_bbox_coords', True)
        self.bbox_mode = kwargs.get('bbox_mode', 'all-xyz')

        self.bbox_drop_prob = kwargs.get('bbox_drop_prob', 0)
        self.with_traj_map = kwargs.get("with_traj_map", False)
        self.with_camera_param = kwargs.get("with_camera_param", False)
        self.simple_read_data = kwargs.get('simple_read_data', False)
        # self.simple_read_data = kwargs.get('simple_read_data', True)
        self.add_geo = kwargs.get('add_geo', False)
        self.add_canny = kwargs.get('add_canny', False)
        self.user_frame_emb = kwargs.get('user_frame_emb', True)
        self.traj_edit_type = kwargs.get("traj_edit_type", None)
        self.traj_param1 = kwargs.get("traj_param1", None)
        self.traj_param2 = kwargs.get("traj_param2", None)
        self.traj_param3 = kwargs.get("traj_param3", None)
        self.infer_mode = kwargs.get('infer_mode', False)
        self.stop_filter_e2w = kwargs.get("stop_filter_e2w", False)
        self.drop_3d_injection = kwargs.get("drop_3d_injection", 0.0)
        cam_init_path = kwargs.get('cam_init_path', None)
        self.aksk = kwargs.get('aksk', None)
        if cam_init_path is not None:
            self.cam_params = []
            for cam_json_name in [path for path in os.listdir(cam_init_path) if path.endswith('.json')]:
                cam_json_path = os.path.join(cam_init_path, cam_json_name)
                with open(cam_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cam_params.append(data)
        else:
            self.cam_params = None


        self.random_caption = kwargs.get("random_caption", None)
        self.random_caption_key = kwargs.get("random_caption_key", None)
        self.set_strat_id = kwargs.get("set_strat_id", False)
        self.re_sample = kwargs.get("re_sample", None)
        self.caption_join_func = kwargs.get("caption_join_func", None)
        if self.caption_join_func is None:
            self.caption_join_func = self.joint_caption
        elif self.caption_join_func == "format_caption":
            self.caption_join_func = self.format_caption
        else:
            print(f"error self.caption_join_func: {self.caption_join_func}")
            assert False
        self.add_controller = kwargs.get("add_controller", None)

        print("traj_edit_type: ", self.traj_edit_type)
        print("traj_edit_params: ", self.traj_param1, self.traj_param2, self.traj_param3)

        self.scene_description_data = None
        self.sqrt_required_text_keys = sqrt_required_text_keys

        self.enable_layout = enable_layout
        self.aws_path_transform = aws_path_transform
        if aws_path_transform is not None:
            print(f"Apply aws_path_transform: {aws_path_transform}")
            print(f"Please ensure the new aws path is exsited.")
        # for 大卓
        self.camera_map = {
            "center_camera_fov120":"front_wide_camera",
            "rear_camera":"rear_main_camera",
            "center_camera_fov30":"front_main_camera",
        }


        if self.enable_scene_description:
            self.scene_description_data = {}
            if self.random_caption is None:
                assert scene_description_file is not None
                assert CaseName_PAPCaseName_Table_file is not None
                with open(CaseName_PAPCaseName_Table_file, 'r') as f:
                    CaseName_PAPCaseName_Table = json.load(f)
                    self.CaseName_PAPCaseName_Table = CaseName_PAPCaseName_Table
                    self.CaseName_PAPCaseName_Table_reverse = {v: k for k, v in self.CaseName_PAPCaseName_Table.items()}

                # 读取pkl 加速
                if ".pkl" in scene_description_file[0]:
                    for caption_file in scene_description_file:
                        with open(caption_file, 'rb') as f:
                            load_caption_data = pkl.load(f)
                        # 转换格式
                        for k, v in load_caption_data.items():
                            self.scene_description_data[k] = v["all_captions"]
                # 读取json caption 保留这部分逻辑兼容旧格式
                else:
                    captions = []
                    for caption_file in scene_description_file:
                        with open(caption_file, 'r') as f:
                            captions += f.readlines()
                        for line in tqdm(captions, total=len(captions)):
                            item = json.loads(line.strip())
                            #key = PAPCaseName_CaseName_Table[item['case_name']]
                            if self.simple_read_data:
                                # 获取真正的clip case name
                                if 'root' in item.keys():
                                    if "s3://" in item['root']:
                                        aws_path = item['root']
                                        key = aws_path.split('/')[-1]
                                    else:
                                        key = item['case_name']
                                else:
                                    key = self.CaseName_PAPCaseName_Table_reverse[item['case_name']]
                                if key in [
                                    "2024_11_28_22_26_23_L2_E03-339_15019_pvbGt", # '20241123110216_20241123110246-ShangHai-A02-548'
                                    "2025_01_15_12_01_12_L2_CN-006_17238_pvbGt",
                                    "2023_11_27_08_03_07_GAC_A02-459_13367_pvbGt",
                                    "2024_11_23_11_02_16_L2_A02-548_15321_pvbGt",
                                    "2025_01_02_17_16_39_L2_CN-002_15982_pvbGt",
                                ]:
                                    print(1)

                            elif 'root' in item.keys():
                                aws_path = item['root']
                                aws_path = aws_path.replace(
                                    aws_path[:aws_path.index('/', aws_path.index('s3://') + len('s3://')) + 1], ""
                                )
                                key = aws_path
                            else:
                                key = item['case_name']
                            if key in self.scene_description_data.keys():
                                self.scene_description_data[key][(min(item['indexes']), max(item['indexes']))] = item
                            else:
                                self.scene_description_data[key] = {(min(item['indexes']), max(item['indexes'])): item}



            # 使用random caption 给t2v用
            else:
                print(Style.BRIGHT + Fore.RED + f'user random_caption !!!!!! ' + Style.RESET_ALL)

                self.caption_dict, self.caption_all = self.read_caption(scene_description_file, self.sqrt_required_text_keys)
                # 按照过滤原则过滤self.caption_all
                if self.random_caption == "chose" and self.random_caption_key is not None:
                    flter_caption = []
                    for caption in self.caption_all:
                        chose = True
                        for idx, key in enumerate(["weather", "time", "lighting", "road_type"]):
                            vv = self.random_caption_key.get(key, None)
                            if vv is not None:
                                if isinstance(vv, str):
                                    vv = [vv]
                                if isinstance(vv, dict):
                                    vv = vv["choices"]
                                match_found = any(word in caption.lower() for word in vv)
                                if match_found:
                                    chose = True
                                else:
                                    chose = False
                        if chose:
                            flter_caption.append(caption)
                    print(Style.BRIGHT + Fore.RED + f" select caption_all: {len(self.caption_all)} =====================> {len(flter_caption)}" + Style.RESET_ALL)
                    # self.caption_all = flter_caption
                    if len(self.caption_all) <= 0:
                        assert 0

                elif self.random_caption == "random" and self.random_caption_key is not None:
                    # print(Style.BRIGHT + Fore.RED + f" org caption_dict : {self.caption_dict}" + Style.RESET_ALL)
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

                    # print(Style.BRIGHT + Fore.RED + f" new caption_dict : {self.caption_dict}" + Style.RESET_ALL)

        self.segment_annos_root = []
        segment_infos = []
        segment_annos_root = []
        self.split_file = split_file

        if split is not None and self.split_file is not None:
            self.split_file = pkl.load(open(self.split_file, 'rb'))[split]

        if processed_meta_file is not None:  # 正式训练时必须传入该参数
            print('processed_meta_file:', processed_meta_file)
            for meta_file in processed_meta_file:
                data = pkl.load(open(meta_file[0], 'rb'))
                for seg in data:
                    if seg[2] < max(self.seq_length):
                        continue
                    # segment_infos.append((seg[0], seg[1], seg[2], meta_file[0].split('/')[-1][:-4]))
                    # base_info = (seg[0], seg[1], seg[2])
                    base_info = (seg[0], meta_file[1], seg[2]) # seg[1] 暂时无用， 直接meta_file[1]替代
                    # 附加信息：文件名部分
                    # extra_info = [meta_file[0].split('/')[-1][:-4]]
                    extra_info = [meta_file[0].split('/')[-1]]
                    # 如果 seg 中有超过 3 个元素，把后面的也加进去
                    if len(seg) > 3:
                        extra_info.extend(seg[3:])
                    # 合并成一个 tuple
                    segment_infos.append(base_info + tuple(extra_info))
                segment_annos_root.append(meta_file[1])
            self.segment_annos_root = segment_annos_root

            frame_cnt = 0
            #print(len(self.split_file), len(segment_infos))
            #print(segment_infos[0][0], self.split_file[0])
            no_match_caption_clip = []
            for seg in tqdm(segment_infos, total=len(segment_infos), desc='Load Data'):
                if self.split_file is not None and seg[0] in self.split_file:
                    if must_text:
                        # key = self.CaseName_PAPCaseName_Table[seg[0]]
                        if seg[1] not in self.scene_description_data.keys():
                            if seg[0] not in self.CaseName_PAPCaseName_Table.keys():
                                # print('1:', seg[0])
                                continue
                            elif self.CaseName_PAPCaseName_Table[seg[0]] not in self.scene_description_data.keys()  and self.random_caption is None:
                                # print('2:', self.CaseName_PAPCaseName_Table[seg[0]])
                                continue
                    self.segment_infos.append(seg)
                    frame_cnt += seg[2]
                elif self.split_file is None:
                    if must_text:
                        #key = self.CaseName_PAPCaseName_Table[seg[0]]
                        if self.simple_read_data:
                            if seg[0] not in self.scene_description_data.keys() and self.random_caption is None:
                                # print("no match caption : ", seg[0])
                                no_match_caption_clip.append((seg[0], seg[1], seg[2]))
                                continue
                        else:
                            if seg[1] not in self.scene_description_data.keys()  and self.random_caption is None:
                                if seg[0] not in self.CaseName_PAPCaseName_Table.keys():
                                    print('1:',seg[0])
                                    continue
                                elif self.CaseName_PAPCaseName_Table[seg[0]] not in self.scene_description_data.keys()  and self.random_caption is None:
                                    print('2:',self.CaseName_PAPCaseName_Table[seg[0]])
                                    continue
                    self.segment_infos.append(seg)
                    frame_cnt += seg[2]
            # print(f"num_cases={len(self)}, num_frames={frame_cnt}, no match cases = {no_match_caption_clip}")
            print(Style.BRIGHT + Fore.RED + f"num_cases={len(self)}, num_frames={frame_cnt}, no match cases = {len(no_match_caption_clip)}" + Style.RESET_ALL)

        else:
            print("process data meta files...")
            if self.debug:
                raw_meta_files = raw_meta_files[:1]
            for meta_file in tqdm(raw_meta_files):
                data = json.load(open(meta_file))
                if self.debug:
                    data = data[:5]
                for case in data:
                    aws_path = case["root"]
                    aws_path = aws_path.replace(
                        aws_path[:aws_path.index('/', aws_path.index('s3://') + len('s3://')) + 1], "")
                    annos = []
                    for line in io.BytesIO(reader.load_file(sub_dir=aws_path + "/gt.jsonl")).readlines():
                        annos.append(json.loads(line))
                    # if len(annos) < sequence_length[0]:  # 去掉帧数不足的case
                    #     continue
                    self.segment_annos.append(annos)
                    self.segment_infos.append((case["case_name"], aws_path, len(annos)))

        self.camera_list = camera_list
        if self.without_gt and self.split != 'train':
            new_segment_infos = []
            for item in self.segment_infos:
                for fps, stride in self.fps_stride_list:
                    new_segment_infos.extend(self.enumerate_segement(item, fps, stride))
            self.segment_infos = new_segment_infos

        # re_sample
        if self.re_sample is not None:
            re_sample_path = self.re_sample["path"]
            all_clip_data = []
            for root, _, files in os.walk(re_sample_path):
                for file in files:
                    if file.endswith(".json"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        all_clip_data += data
            re_sample_clip_case = []
            for case in all_clip_data:
                case_name = case["case_name"]
                re_sample_clip_case.append(case_name)

            segment_infos_resample = []
            for data in self.segment_infos:
                if data[0] in re_sample_clip_case:
                    segment_infos_resample.append(data)
            segment_infos_resample = segment_infos_resample * self.re_sample["sample"]
            self.segment_infos += segment_infos_resample
            print(Style.BRIGHT + Fore.GREEN + f"re_sample num_cases={len(self)}, num_frames={frame_cnt}, no match cases = {len(no_match_caption_clip)}" + Style.RESET_ALL)
        if self.split == 'train' and (not self.debug) and not self.infer_mode:
            # self.segment_infos = self.segment_infos * 10000
            self.segment_infos = self.segment_infos * 500

    def read_caption(self, scene_description_file, sqrt_required_text_keys):
        # sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"]
        caption_dict = {}
        caption_all = []
        for k in sqrt_required_text_keys:
            caption_dict[k] = []

        # 读取pkl 加速
        if ".pkl" in scene_description_file[0]:
            scene_description_data = {}
            for caption_file in scene_description_file:
                with open(caption_file, 'rb') as f:
                    load_caption_data = pkl.load(f)
                # 转换格式
                for k, v in load_caption_data.items():
                    scene_description_data[k] = v["all_captions"]

            for case_name, all_cap in scene_description_data.items():
                if (0, 99) in all_cap.keys():
                    data = all_cap[(0, 99)]
                    data_caption = data["caption"]
                    if "center_camera_fov120" in data_caption.keys():
                        data_caption = data_caption["center_camera_fov120"]
                    cap_dict = {}
                    for kk in sqrt_required_text_keys:
                        v = data_caption[kk]
                        if v not in caption_dict[kk]:
                            caption_dict[kk].append(v)
                        cap_dict[kk] = v
                    caption = self.caption_join_func(cap_dict)
                    caption_all.append(caption)
        else:
            for file in scene_description_file:
                data_list = []
                with open(file, 'r', encoding='utf-8') as f:
                    # content = f.read().strip()
                    # data_list = json.loads(content)
                    for line in f.readlines():
                        data_list.append(json.loads(line.strip()))

                for data in tqdm(data_list):
                    data_caption = data["caption"]
                    if "center_camera_fov120" in data_caption.keys():
                        data_caption = data_caption["center_camera_fov120"]
                    cap_dict = {}
                    for kk in sqrt_required_text_keys:
                        v = data_caption[kk]
                        if v not in caption_dict[kk]:
                            caption_dict[kk].append(v)
                        cap_dict[kk] = v
                    caption =  self.caption_join_func(cap_dict)
                    caption_all.append(caption)

        return caption_dict, caption_all

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
            print("self.description_mode : {}".format(self.random_caption))
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

    def enumerate_segement(self, segment_infos, fps, stride):
        # print('self.scene_description_data.keys():',self.scene_description_data.keys())

        case_name, segment_root, case_num_frames, tag = segment_infos
        selected_fps = fps
        # randomly select start frame id
        max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        assert max_fps_id > 1, f"max_fps_id too small!! Check input data original fps!!! len(segment_json_files): {len(info)}"
        if self.without_gt:
            num = max_fps_id // stride
        else:
            num = (max_fps_id - self.seq_length) // stride
        # mod = (max_fps_id - self.seq_length)%stride

        start_id_list = list(i * stride for i in range(num))

        if len(start_id_list) == 0:
            if max_fps_id:
                start_id_list = [0]

        print(f'case_num_frames:{case_num_frames}/max_fps_id:{max_fps_id}/len(start_id_list):{len(start_id_list)}')
        return [(case_name, segment_root, case_num_frames, tag, start_id, fps, stride) for start_id in start_id_list]

    def resize_nearest(self, img, size):
        size = (size[1], size[0])
        return img.resize(size, Image.NEAREST)

    def save_processed_meta_file(self, info_save_path, anno_save_root):
        with open(info_save_path, 'wb') as fp:
            pkl.dump(self.segment_infos, fp)
        print(f"save meta info to {info_save_path}")
        for idx, (case_name, _, _) in tqdm(enumerate(self.segment_infos)):
            case_root = os.path.join(anno_save_root, case_name)
            os.makedirs(case_root, exist_ok=True)
            for i in range(len(self.segment_annos[idx])):
                with open(os.path.join(case_root, f"{i:04d}.json"), "w") as fp:
                    json.dump(self.segment_annos[idx][i], fp)

    def draw_bbox(self, objects, init_image, camera_intrinsic, camera_dist, extrinsic):
        # [x, y, z, l, w, h, yaw]
        bbox_list = [obj["bbox3d"] for obj in objects]
        trackid_list = [obj["id"] for obj in objects]
        class_list = [obj["label"] for obj in objects]
        class_color_list = [CLASS_BBOX_COLOR_MAP[cname] for cname in class_list]
        final_color_list = class_color_list
        # final_color_list = []
        # for i in range(len(class_color_list)):
        #     color_tuple = class_color_list[i]
        #     trackid_color = trackid_color_list[i]
        #     modified_color = (*color_tuple[:-1], trackid_color)
        #     final_color_list.append(modified_color)

        img = draw_3Dbbox_proj(init_image, bbox_list, camera_intrinsic, extrinsic, color_list=final_color_list,
                               thickness=8)

        return img

    def draw_bbox_alt(self, meta):
        targets = meta["Objects"]
        imgs = {}
        for camera_name, camera_meta in meta["sensors"]["cameras"].items():
            img_w = self.img_w_fisheye if "fov195" in camera_name else self.img_w_pinhole
            img_h = self.img_h_fisheye if "fov195" in camera_name else self.img_h_pinhole
            lidar2camera_rt = np.array(camera_meta["extrinsic"])

            camera_intrinsic = np.array(camera_meta["camera_intrinsic"])
            camera_intrinsic_dist = np.array(camera_meta["camera_dist"])
            camera_type = "pinhole" if "195" not in camera_name else "fisheye"

            boxes, boxes_2d = [], []
            if targets is not None:
                for target in targets:
                    if "BARRICADE" in target["label"]:  # 路栏画起来过于奇怪，暂时先不要
                        continue

                    # NOTE - Only Draw visualizable objects
                    if target["info2d"] is None or (not target["info2d"]["visible"]) or (
                    target["info2d"]["bbox2d"]):  # 针对每个相机只画可见的
                        continue
                    if camera_name not in target["info2d"] or (not target["info2d"][camera_name]["visible"]):
                        continue

                    # boxes_2d.append(target.info2d[camera_name]["bbox2d"])
                    boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])

            bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            if len(boxes) > 0:
                lidar_boxes = LiDARInstance3DBoxes(torch.Tensor(boxes), origin=(0.5, 0.5, 0.5))
                img_points = lidar_boxes.img_points(
                    camera_type=camera_type,
                    lidar2cam_rt=lidar2camera_rt,
                    camera_intrinsic=camera_intrinsic,
                    camera_intrinsic_dist=camera_intrinsic_dist,
                    img_w=img_w,
                    img_h=img_h,
                )

                if len(img_points) > 0:
                    draw_img = expand_plot_rect3d_on_img(
                        bbox_img, img_points.shape[0], img_points.transpose(0, 2, 1), thickness=4, color=(21, 217, 211)
                    )
                else:
                    draw_img = bbox_img
            else:
                draw_img = bbox_img
            imgs[camera_name] = Image.fromarray(np.uint8(draw_img))
        return imgs

    def get_min_max_velocity(self, meta_list):
        localworld_velocity_list = []
        for meta in meta_list:
            for cam_id, camera_name in enumerate(self.camera_list, 0):
                targets = meta["Objects"]
                ego2local = meta.get("ego2local_transformation_matrix", None)
                if targets is not None:
                    for target in targets:
                        if "BARRICADE" in target["label"]:  # 路栏画起来过于奇怪，暂时先不要
                            continue

                        # NOTE - Only Draw visualizable objects
                        if target["info2d"] is None:  # 针对每个相机只画可见的
                            continue
                        # if not target["attribute"]["SHIELD"]:
                        #     continue
                        if camera_name not in target["info2d"] or (not target["info2d"][camera_name]["visible"]):
                            continue
                        if target["info2d"][camera_name]["bbox2d"] is None or (
                        not len(target["info2d"][camera_name]["bbox2d"])):
                            continue

                        velocity = np.asarray(target["velocity"])
                        localworld_velocity = ego2local[:3, :3] @ velocity[:, None]
                        localworld_velocity_list.append(np.squeeze(localworld_velocity[:3]))

        if len(localworld_velocity_list):
            localworld_velocity_list = np.concatenate(localworld_velocity_list, axis=0)
            return np.min(localworld_velocity_list), np.max(localworld_velocity_list)
        return None, None

    def draw_bbox_alt_camera_meta(self, meta, cam_id, camera_name, img_shape, sx=1, sy=1, draw_bbox_mode='cross', colorful_box=False,
                                  thickness=6, alpha=0.5):
        targets = meta["Objects"]
        camera_meta = meta["sensors"]["cameras"][camera_name]
        camera_meta["camera_intrinsic"] = (np.array(camera_meta["camera_intrinsic"]) * np.array([sx, sy, 1])[:, np.newaxis]).tolist()
        ego2local = meta.get("ego2local_transformation_matrix", None)

        # img_w = self.img_w_fisheye if "fov195" in camera_name else self.img_w_pinhole
        # img_h = self.img_h_fisheye if "fov195" in camera_name else self.img_h_pinhole
        # img_h, img_w = img_shape
        img_w, img_h = img_shape
        lidar2camera_rt = np.array(camera_meta["extrinsic"])
        camera_intrinsic = np.array(camera_meta["camera_intrinsic"])
        camera_intrinsic_dist = np.array(camera_meta["camera_dist"])
        camera_type = "pinhole" if "195" not in camera_name else "fisheye"

        boxes, classes, localworld_velocity_list = [], [], []
        colors = []
        if targets is not None:
            for target in targets:
                if "new_e2w" in meta and cam_id==0: # bbox3d与相机无关，只操作一次
                    for i in range(3):
                        delta = meta["new_e2w"][i][3] - meta["ego2global_transformation_matrix"][i][3]
                        if False and delta < -2:
                            print("delta: ", delta)
                        target["bbox3d"][i] -= delta

                if "BARRICADE" in target["label"]:  # 路栏画起来过于奇怪，暂时先不要
                    continue

                # NOTE - Only Draw visualizable objects
                if target["info2d"] is None:  # 针对每个相机只画可见的
                    continue
                # if not target["attribute"]["SHIELD"]:
                #     continue
                # if camera_name not in target["info2d"] or (not target["info2d"][camera_name]["visible"]):
                #     continue
                # if target["info2d"][camera_name]["bbox2d"] is None or (
                # not len(target["info2d"][camera_name]["bbox2d"])):
                #     continue

                # # boxes_2d.append(target.info2d[camera_name]["bbox2d"])
                # #* Drop Bbox in bbox_drop_probability
                # if random.random() < self.bbox_drop_prob and self.split=='train':
                #     continue

                if target["label"] in CLASS_BBOX_COLOR_MAP.keys():
                    co = CLASS_BBOX_COLOR_MAP[target["label"]]
                    label_id = CLASS_BBOX_ID_MAP[target["label"]]
                elif 'VEHICLE_' in target["label"]:
                    co = (255, 128, 0)
                    label_id = CLASS_BBOX_ID_MAP["VEHICLE_TRUCK"]
                elif "CYCLIST_" in target["label"]:
                    co = (0, 255, 0)
                    label_id = CLASS_BBOX_ID_MAP["CYCLIST_BICYCLE"]
                elif "PEDESTRIAN_" in target["label"]:
                    co = (0, 0, 255)
                    label_id = CLASS_BBOX_ID_MAP["PEDESTRIAN_NORMAL"]
                else:
                    continue
                classes.append(label_id)
                boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])

                if self.colorful_box:
                    colors.append(co)

                if self.with_traj_map:
                    velocity = np.asarray(target["velocity"])
                    localworld_velocity = ego2local[:3, :3] @ velocity[:, None]
                    localworld_velocity_list.append(np.squeeze(localworld_velocity[:3]))
                    # print('velocity:', velocity)

        # print(len(boxes))
        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        traj_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        if len(boxes) > 0:
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

                if self.enable_layout:
                    draw_img = expand_plot_rect3d_on_img(
                        bbox_img, img_points.shape[0], img_points.transpose(0, 2, 1), \
                        thickness=thickness, colors=box_colors, mode=draw_bbox_mode, alpha=alpha, with_filed=False
                    )
                else: # 返回空的bbox img，加快速度
                    draw_img = bbox_img

                if self.with_traj_map:
                    box_localworld_velocity = [localworld_velocity_list[i] for i in masks]
                    box_localworld_velocity = np.asarray(box_localworld_velocity)  # [n,3]

                    min_box_localworld_velocity, max_box_localworld_velocity = self.min_localworld_velocity, self.max_localworld_velocity
                    # max_box_localworld_velocity = 100#np.max(box_localworld_velocity)
                    # min_box_localworld_velocity = -100#np.min(box_localworld_velocity)
                    box_localworld_velocity = np.clip(box_localworld_velocity, min_box_localworld_velocity,
                                                      max_box_localworld_velocity)
                    # print(max_box_localworld_velocity, min_box_localworld_velocity)
                    norm_box_localworld_velocity = (box_localworld_velocity - min_box_localworld_velocity) / (
                                max_box_localworld_velocity - min_box_localworld_velocity + 1e-8)
                    # np.clip(((colors) * 255.).astype(np.uint8), 0, 255)
                    color_norm_box_localworld_velocity = np.clip((norm_box_localworld_velocity * 255).astype(np.uint8),
                                                                 0, 255)
                    polycolors = color_norm_box_localworld_velocity.tolist()
                    # print(f'---------------Out--{polycolors}------------------')
                    traj_image_map = expand_plot_rect3d_on_img(
                        traj_image, img_points.shape[0], img_points.transpose(0, 2, 1), \
                        thickness=thickness, colors=polycolors, polycolors=polycolors, mode=draw_bbox_mode, \
                        with_filed=True, with_borderline_on_filed_bbox=False, alpha=1
                    )
                else:
                    traj_image_map = traj_image

            else:
                draw_img = bbox_img
                lidar_in_cams = None
                box_classes = None
                traj_image_map = traj_image

        else:
            draw_img = bbox_img
            lidar_in_cams = None
            box_classes = None
            traj_image_map = traj_image

        return Image.fromarray(np.uint8(draw_img)), dict(camera_meta=camera_meta, lidar_in_cams=lidar_in_cams,
                                                         box_classes=box_classes,
                                                         traj_image_map=Image.fromarray(np.uint8(traj_image_map)))

    def draw_bbox_alt_camera_meta_filled(self, meta, camera_name, img_shape, draw_bbox_mode='cross', colorful_box=False,
                                         thickness=18, alpha=0.2):
        targets = meta["Objects"]
        camera_meta = meta["sensors"]["cameras"][camera_name]
        # img_w = self.img_w_fisheye if "fov195" in camera_name else self.img_w_pinhole
        # img_h = self.img_h_fisheye if "fov195" in camera_name else self.img_h_pinhole
        # img_h, img_w = img_shape
        img_w, img_h = img_shape
        lidar2camera_rt = np.array(camera_meta["extrinsic"])

        camera_intrinsic = np.array(camera_meta["camera_intrinsic"])
        camera_intrinsic_dist = np.array(camera_meta["camera_dist"])
        camera_type = "pinhole" if "195" not in camera_name else "fisheye"

        boxes, classes = [], []
        colors = []
        if targets is not None:
            for target in targets:
                if "BARRICADE" in target["label"]:  # 路栏画起来过于奇怪，暂时先不要
                    continue

                # NOTE - Only Draw visualizable objects
                if target["info2d"] is None:  # 针对每个相机只画可见的
                    continue
                # if not target["attribute"]["SHIELD"]:
                #     continue
                # if camera_name not in target["info2d"] or (not target["info2d"][camera_name]["visible"]):
                #     continue
                # if target["info2d"][camera_name]["bbox2d"] is None or (not len(target["info2d"][camera_name]["bbox2d"])):
                #     continue

                # boxes_2d.append(target.info2d[camera_name]["bbox2d"])
                boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])
                if self.colorful_box:
                    if target["label"] in CLASS_BBOX_COLOR_MAP.keys():
                        co = CLASS_BBOX_COLOR_MAP[target["label"]]
                        label_id = CLASS_BBOX_ID_MAP[target["label"]]
                    elif 'VEHICLE_' in target["label"]:
                        co = (255, 128, 0)
                        label_id = CLASS_BBOX_ID_MAP["VEHICLE_TRUCK"]
                    elif "CYCLIST_" in target["label"]:
                        co = (0, 255, 0)
                        label_id = CLASS_BBOX_ID_MAP["CYCLIST_BICYCLE"]
                    elif "PEDESTRIAN_" in target["label"]:
                        co = (0, 0, 255)
                        label_id = CLASS_BBOX_ID_MAP["PEDESTRIAN_NORMAL"]
                    else:
                        # 给个默认值 car
                        co = (255, 0, 0)
                        label_id = CLASS_BBOX_ID_MAP["VEHICLE_CAR"]

                    colors.append(co)
                    classes.append(label_id)

        bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        if len(boxes) > 0:
            lidar_boxes = MaskLiDARInstance3DBoxes(torch.Tensor(boxes), origin=(0.5, 0.5, 0.5))
            img_points, masks, lidar_in_cams, _ = lidar_boxes.img_points(
                camera_type=camera_type,
                lidar2cam_rt=lidar2camera_rt,
                camera_intrinsic=camera_intrinsic,
                camera_intrinsic_dist=camera_intrinsic_dist,
                img_w=img_w,
                img_h=img_h,
            )

            if len(img_points) > 0:
                if colorful_box:
                    box_colors = colors[masks]
                else:
                    box_colors = [(21, 217, 211)]
                # if box_colors != [(21, 217, 211)]:print('box_colors:', box_colors)

                draw_img = expand_plot_rect3d_on_img(
                    bbox_img, img_points.shape[0], img_points.transpose(0, 2, 1), \
                    thickness=thickness, colors=box_colors, mode=draw_bbox_mode, alpha=alpha, with_filed=True
                )
            else:
                draw_img = bbox_img

        else:
            draw_img = bbox_img

        return Image.fromarray(np.uint8(draw_img)), (camera_meta, lidar_in_cams, classes)

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

    # 记录信息，报错追溯
    def update_info(self, **kwargs):
        required_keys = ['index', 'case_name']
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key: {key}")
        self.index_info = kwargs

    def get_info(self):
        return self.index_info

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index)

    def get_item(self, index) -> dict:
        index, seq_length, selected_fps = [int(item) for item in index.split("-")]
        if self.set_strat_id and self.split != 'train':
            # case_name, segment_root, case_num_frames, tag, start_id, _, _ = self.segment_infos[index]
            case_name, segment_root, case_num_frames, tag, start_id = self.segment_infos[index]
            max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        else:
            case_name, segment_root, case_num_frames, tag = self.segment_infos[index]
            # selected_fps = self.fps_list[0] if self.split == "test" else random.choice(self.fps_list)
            # randomly select start frame id
            max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
            # print("max_fps_id: ", max_fps_id, " selected_fps: ", selected_fps, " len(segment_json_files): ", len(segment_json_files))
            if self.use_random_seed:
                local_random = random.Random()
                local_random.seed(time.time() + index)
                print('use local random seed')
                start_id = 0 if self.split == "test" or self.debug else local_random.randint(0, max_fps_id - 1)
            else:
                start_id = 0 if self.split == "test" or self.debug or self.start_zero_frame else random.randint(0,
                                                                                                                max_fps_id - 1)

            # start_id = min(start_id, max_fps_id - seq_length - 1)  # make sure within range 这里是bug
            start_id = min(start_id, max_fps_id - seq_length)  # make sure within range
            if self.specific_video_segment is not None:
                start_id = self.specific_video_segment[case_name][0]
        # updata info
        dataset_tag = get_dataset(case_name, segment_root)
        self.update_info(index = index, case_name = case_name, case_num_frames = case_name, start_id = start_id, dataset_tag = dataset_tag)

        # 随机选取一个pap的相机参数文件
        if self.cam_params:
            cam_param_mate = random.choice(self.cam_params)
        else:
            cam_param_mate = None
        images = [[] for _ in range(len(self.camera_list))]
        bbox_images = [[] for _ in range(len(self.camera_list))]  # blank img with bbox
        hdmap_images = [[] for _ in range(len(self.camera_list))]  # blank img with laneline
        img_paths = [[] for _ in range(len(self.camera_list))]
        filled_bbox_images = [[] for _ in range(len(self.camera_list))]
        bboxes_meta_list = [[] for _ in range(len(self.camera_list))]
        ego_velocity = []
        if self.with_traj_map:
            traj_images = [[] for _ in range(len(self.camera_list))]

        annos = []
        path_to_annos = []
        for i in range(seq_length):
            frame_id = int(np.floor((start_id + i) / max_fps_id * case_num_frames))
            found_path = None
            # for segment_annos_root in self.segment_annos_root:
            #     file_ = os.path.join(segment_annos_root, case_name, f"{frame_id:04d}.json")
            #     if os.path.exists(file_):
            #         found_path = file_
            #         break

            file_ = os.path.join(segment_root, case_name, f"{frame_id:04d}.json")
            if os.path.exists(file_):
                found_path = file_
            assert found_path is not None, f"cannot find case {case_name} in all given anno roots."
            with open(found_path) as fp:
                annos.append(json.load(fp))
                path_to_annos.append(found_path)

        vehicle_id = annos[0].get("vehicle_id", None)
        controller = from_vehicle_id_get_controller(vehicle_id) # 车辆域控信息
        scene_description = None
        all_scene_description = None
        if self.random_caption is not None:
            caption_ = self.get_caption()
            # 添加域控信息
            if self.add_controller:
                if "dataset_tag" in self.add_controller.keys():
                    caption_ += f' {{+dataset:{self.add_controller["dataset_tag"]}}}'
                else:
                    caption_ += f' {{+dataset:{dataset_tag}}}'
                if "controller" in self.add_controller.keys():
                    caption_ += f' {{+controller:{self.add_controller["controller"]}}}'
                else:
                    caption_ += f' {{+controller:{controller}}}'
            scene_description = {
                "center_camera_fov120": caption_,
            }
        elif self.enable_scene_description:
            assert self.scene_description_data is not None

            min_frame_id = int(np.floor((start_id + 0) / max_fps_id * case_num_frames))
            max_frame_id = int(np.floor((start_id + seq_length - 1) / max_fps_id * case_num_frames))


            if self.simple_read_data:
                key = case_name
            else:
                if segment_root in self.scene_description_data.keys():
                    key = segment_root
                else:
                    key = self.CaseName_PAPCaseName_Table[case_name]
            if key in self.scene_description_data.keys():
                interval_list = []
                for k in self.scene_description_data[key].keys():
                    min_, max_ = k
                    min_, max_ = int(min_), int(max_)
                    interval_list.append([min_, max_])

                match_key = self.find_best_interval(interval_list, (min_frame_id, max_frame_id))
                all_scene_description = self.scene_description_data[key][(match_key[0], match_key[1])]
                if all_scene_description:
                    scene_description = dict()
                    for camera_name in self.camera_list:
                        if self.sqrt_required_text_keys:
                            # combine_text = []
                            combine_text = {}
                            for key in self.sqrt_required_text_keys:
                                if key in all_scene_description['caption'].keys():
                                    combine_text[key] = all_scene_description['caption'][key]
                                elif key in all_scene_description['caption'][camera_name].keys():
                                    combine_text[key] = all_scene_description['caption'][camera_name][key]
                            # 添加域控信息
                            if self.add_controller:
                                combine_text.update({
                                    "dataset": dataset_tag if "dataset_tag" not in self.add_controller else self.add_controller["dataset_tag"],
                                    "controller":controller if "controller" not in self.add_controller else self.add_controller["controller"],
                                })
                            scene_description[camera_name] = self.caption_join_func(combine_text)
                        else:
                            scene_description[camera_name] = all_scene_description['caption'][camera_name]['general']
                else:
                    print('No Text!', min_frame_id, max_frame_id, self.scene_description_data[key].keys())
            else:
                print('No Text!', key)

        raw_img_size_dict = {}
        normalized_camera_intrinsic = {}

        if self.with_traj_map:
            ego2global_transformation_list = []
            for i in range(seq_length):
                ego2global_transformation_list.append(annos[i]["ego2global_transformation_matrix"])
            ego2local_transformation_matrix = Ego2GlobalToEgo2LocalWorldFrame(ego2global_transformation_list)

            for i in range(seq_length):
                annos[i]["ego2local_transformation_matrix"] = ego2local_transformation_matrix[i]

        if self.with_traj_map:
            self.min_localworld_velocity, self.max_localworld_velocity = self.get_min_max_velocity(annos)
        if self.traj_edit_type is not None:
            e2w_seq = []
            for i in range(seq_length):
                e2w_seq.append(annos[i]["ego2global_transformation_matrix"])
                annos[i]["new_e2w"] = copy.deepcopy(annos[i]["ego2global_transformation_matrix"])
            e2w_seq, _ = edit_pos(e2w_seq, edit_type=self.traj_edit_type, trans_scale=1.0,
                                  edit_param1=self.traj_param1, edit_param2=self.traj_param2, edit_param3=self.traj_param3
                                  )
            for i in range(seq_length):
                annos[i]["new_e2w"] = e2w_seq[i]

        for i in range(seq_length):
            if i >= len(annos):
                global_velocity = annos[-1]["ego_velocity"]
            else:
                global_velocity = annos[i]["ego_velocity"]
            try:
                ego_velocity.append(
                    np.linalg.inv(annos[i]["ego2global_transformation_matrix"])[:3, :3] @ np.asarray(global_velocity)[:,None])
            except Exception as err:
                print(f"case={case_name}, frame_no={start_id + i}")
                raise err
            if self.add_geo:
                gt_lanes = []
                gt_lane_cls = []
                gt_lane_sub_cls = []
                gt_indexs = []
                if "Geo" in annos[i].keys():
                    for idx, lanes in enumerate(annos[i]["Geo"]):
                        # gt_lanes.append(self.fix_pts_interpolate(np.array(lanes["geo"]), 10, 10))
                        if len(lanes["xyz"]) <= 1:
                            continue
                        xyz = np.array(lanes["xyz"])
                        if "new_e2w" in annos[i]:
                            delta = np.array(annos[i]["new_e2w"])[:3, 3] - np.array(annos[i]["ego2global_transformation_matrix"])[:3, 3]
                            xyz -= delta

                        gt_lanes.append(xyz)
                        gt_lane_cls.append(lanes["type"])
                        gt_lane_sub_cls.append(lanes["style"])
                        gt_indexs.append(idx)
            for cam_id, camera_name in enumerate(self.camera_list, 0):
                if i >= len(annos):
                    raw_img_size = raw_img_size_dict[camera_name]
                    img = np.zeros((self.expected_vae_size[0], self.expected_vae_size[1], 3), dtype=np.uint8)
                    img = Image.fromarray(np.uint8(img))
                    images[cam_id].append(img)

                    bbox_image = np.zeros((self.expected_vae_size[0], self.expected_vae_size[1], 3), dtype=np.uint8)
                    bbox_image_map = Image.fromarray(np.uint8(bbox_image))
                    bbox_images[cam_id].append(bbox_image_map)

                    hdmap_image = np.zeros((self.expected_vae_size[0], self.expected_vae_size[1], 3), dtype=np.uint8)
                    hdmap_image = Image.fromarray(np.uint8(hdmap_image))
                    hdmap_images[cam_id].append(hdmap_image)

                    if self.with_traj_map:
                        traj_image = np.zeros((self.expected_vae_size[0], self.expected_vae_size[1], 3), dtype=np.uint8)
                        traj_image = Image.fromarray(np.uint8(traj_image))
                        traj_images[cam_id].append(traj_image)

                    img_paths[cam_id].append(
                        annos[-1]['sensors']['cameras'][camera_name]["data_path"][:-4] + f'_vdummy_{i}.jpg')
                    continue
                # 兼容大卓数据
                if camera_name not in annos[i]['sensors']['cameras'].keys():
                    camera_name = self.camera_map[camera_name]

                cam_info = annos[i]['sensors']['cameras'][camera_name]
                if 'root' in annos[i].keys():
                    anno_root = annos[i]['root']
                else:
                    anno_root = annos[i]["original_info"]['root']
                # img_paths[cam_id].append(os.path.join(segment_root, cam_info["data_path"]))
                if self.simple_read_data:
                    if "s3://" in cam_info["data_path"]:
                        read_img_path = cam_info["data_path"]
                    else:
                        read_img_path = os.path.join(anno_root, cam_info["data_path"])
                else:
                    read_img_path = os.path.join(segment_root, cam_info["data_path"])
                img_paths[cam_id].append(read_img_path)
                befor_aws_tran_path = copy.deepcopy(read_img_path)

                if self.without_gt and i >= 100:
                    img = np.zeros((self.expected_vae_size[0], self.expected_vae_size[1], 3), dtype=np.uint8)
                    img = Image.fromarray(np.uint8(img))
                else:
                    start = time.perf_counter()
                    if self.aws_path_transform is not None:
                        read_img_path = self.aws_path_transform(read_img_path)
                    try_read_path = [read_img_path, befor_aws_tran_path]
                    if self.aksk:
                        try_read_path.append(f"{self.aksk}:{befor_aws_tran_path}")
                    img_byte_stream = None
                    # 多级路径尝试读取图
                    for path in try_read_path:
                        try:
                            img_byte_stream = self.reader.load_file(sub_dir=path)
                            if img_byte_stream is not None:
                                break  # 成功就退出循环
                        except:
                            pass
                            # print(f"Failed to load from {path}: {e}")
                    if img_byte_stream is None:
                        print(f" faild read data:  All attempts failed. Tried paths: {try_read_path}")
                        assert 0

                    end = time.perf_counter() - start
                    if end > 10:
                        print(f" Elapsed time: {end:.6f} seconds")

                    img = Image.open(io.BytesIO(img_byte_stream))
                raw_img_size = img.size
                src_w, src_h = raw_img_size
                # annos[i]['sensors']['cameras'][camera_name]["image_width"] = src_w
                # annos[i]['sensors']['cameras'][camera_name]["image_height"] = src_h
                # print(f"{camera_name} {src_w, src_h}")
                # image_size
                if cam_param_mate:
                    annos[i]["sensors"]["cameras"][camera_name]["camera_intrinsic"] = cam_param_mate["sensors"]["cameras"][camera_name]["camera_intrinsic"]
                    annos[i]["sensors"]["cameras"][camera_name]["extrinsic"] = cam_param_mate["sensors"]["cameras"][camera_name]["extrinsic"]
                    src_w, src_h = cam_param_mate["sensors"]["cameras"][camera_name]["image_width"],  cam_param_mate["sensors"]["cameras"][camera_name]["image_height"]
                    sx, sy = self.expected_vae_size[1] / src_w, self.expected_vae_size[0] / src_h

                else:
                    sx, sy = self.expected_vae_size[1] / img.size[0], self.expected_vae_size[0] / img.size[1]
                img = img.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                images[cam_id].append(img)
                # 投影完会改内参矩阵

                bbox_image_map, bboxes_meta = self.draw_bbox_alt_camera_meta(annos[i], cam_id, camera_name, img.size, sx, sy, \
                                                                             draw_bbox_mode=self.draw_bbox_mode,
                                                                             colorful_box=self.colorful_box, thickness=self.thickness)

                if self.add_canny:
                    blurred_image = np.array(img.resize((self.expected_vae_size[1], self.expected_vae_size[0])))
                    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
                    # edges = Image.fromarray(edges)
                    bbox_image_map = bbox_image_map.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                    bbox_image_map =  np.repeat(edges[:, :, np.newaxis], 3, axis=2) + np.array(bbox_image_map)
                    bbox_image_map = np.clip(bbox_image_map,0,255)
                    bbox_image_map = Image.fromarray(bbox_image_map)

                if self.add_geo:
                    reshape_height, reshape_width = self.expected_vae_size
                    img_pv = np.zeros((reshape_height, reshape_width, 3), dtype=np.uint8)
                    # scale_h, scale_w = reshape_height / src_h, reshape_width / src_w
                    scale_h, scale_w = 1.0, 1.0 #内参已经被归一化到了self.expected_vae_size 上了
                    _ = show_lanes_per_camera(img_pv, annos[i], camera_name, gt_lanes, gt_lane_cls, gt_indexs, scale_h, scale_w, thickness = self.thickness // 2)
                    bbox_image_map = bbox_image_map.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                    bbox_image_map = np.clip(img_pv + np.array(bbox_image_map), 0, 255)
                    bbox_image_map = Image.fromarray(bbox_image_map)

                bbox_images[cam_id].append(bbox_image_map)
                hdmap_image = np.zeros((img.size[1], img.size[0], 3), dtype=np.uint8)
                hdmap_image = Image.fromarray(np.uint8(hdmap_image))
                hdmap_images[cam_id].append(hdmap_image)
                if self.with_traj_map:
                    traj_images[cam_id].append(bboxes_meta["traj_image_map"])

                if self.with_bbox_coords:
                    lidar2camera_rt = np.array(bboxes_meta["camera_meta"]["extrinsic"])
                    camera_intrinsic = np.array(bboxes_meta["camera_meta"]["camera_intrinsic"])
                    camera_param = np.concatenate([camera_intrinsic[:3, :3], lidar2camera_rt[:3]], axis=1)

                    # camera_meta=camera_meta, lidar_in_cams=lidar_in_cams, box_classes=box_classes, draw_traj_img=draw_traj_img

                    bboxes_meta_list[cam_id].append(
                        {  # boxes_coords, camera_meta, lidar_in_cams
                            # "lidar_in_worlds":bboxes_meta[0], #[n,8,3]
                            "cam_params": camera_param,  # !要[3,7]
                            "bboxes": bboxes_meta["lidar_in_cams"],
                            # [n,8,3] # lidar_in_cams #[B, N_out, max_len, 8 x 3] #! 看清楚是要cam还是world: 是cam
                            "classes": bboxes_meta["box_classes"]  # [n,] #!要类别号码
                        }
                    )

                if self.with_filled_bbox:
                    filled_bbox_image_map = self.draw_bbox_alt_camera_meta_filled(annos[i], camera_name, img.size, \
                                                                                   draw_bbox_mode=self.draw_bbox_mode,
                                                                                  colorful_box=self.colorful_box)
                    filled_bbox_images[cam_id].append(filled_bbox_image_map)

                if i == 0:
                    raw_img_size_dict[camera_name] = raw_img_size # 3840, 2160
                    normalized_camera_intrinsic[camera_name] = camera_intrinsic # * np.array([sx, sy, 1])[:, np.newaxis]
        result = {
            'pixel_values': torch.stack([torch.stack([self.transforms(i) for i in img]) for img in images]).permute(1, 0, 2, 3, 4),
            # [V, T, C, H, W] -> [V, C, T, H, W], # 11, frame_num
            # 'condition_images_visual': condition_images,
            'bbox': torch.stack(
                [torch.stack([self.condition_transform(i) for i in bbox_img]) for bbox_img in bbox_images]),
            # [V, T, C, H, W]
            # "hdmap": hdmap_images,
            # 'pts': torch.zeros((self.sequence_length), dtype=torch.long),
            'fps': selected_fps if seq_length>1 else IMG_FPS,
            'img_path': [[img_paths[v_i][t_i] for v_i in range(len(self.camera_list))] for t_i in
                         range(seq_length)],
            'full_height': self.full_size[0],
            'full_width': self.full_size[1],
            "ego_velocity": torch.tensor(np.array(ego_velocity)),
            "path_to_annos": path_to_annos
        }
        if scene_description:
            # print(scene_description.keys())
            result['scene_description'] = scene_description['center_camera_fov120']  # [k + '.' + v for k, v in scene_description.items()]
        elif self.must_text:
            # result['scene_description'] = '.'.join(self.camera_list)#[camera_name for camera_name in self.camera_list]
            raise KeyError
        result["height"] = result["pixel_values"].shape[-2]
        result["width"] = result["pixel_values"].shape[-1]
        result["ar"] = result["width"] / result["height"]
        result["captions"] = [result.pop('scene_description') if scene_description else ""] * seq_length
        result["num_frames"] = seq_length

        if self.split != 'train':
            result["tag"] = tag

        if self.with_original_info:
            result["annos"] = json.dumps(annos, indent=2, sort_keys=True)
            result["original_info"] = json.dumps(annos[0]["original_info"], indent=2, sort_keys=True)

        if self.with_filled_bbox:
            result["filled_bbox_images"] = torch.stack(
                [torch.stack([self.condition_transform(i) for i in bbox_img]) for bbox_img in filled_bbox_images])

        if self.with_traj_map:
            result["traj"] = torch.stack(
                [torch.stack([self.condition_transform(i) for i in bbox_img]) for bbox_img in traj_images])

        if self.with_camera_param:
            V = len(bboxes_meta_list)
            T = len(bboxes_meta_list[0])

            cam_param = [[] for _ in range(T)]

            for t in range(T):
                for v in range(V):
                    try:
                        cam_param[t].append(bboxes_meta_list[v][t]["cam_params"])
                    except:
                        traceback.print_exc()
                        # import pdb;
                        # pdb.set_trace()

            result["camera_param"] = np.asarray(cam_param)

        if self.with_bbox_coords:
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
            # drop 3dbbox injection
            if self.drop_3d_injection > 0 and random.random() < self.drop_3d_injection:
                bboxes = None
            ret_dict = pad_bboxes_to_maxlen([T, V, max_len, *bbox_shape], max_len, bboxes, classes)

            ret_dict['cam_params'] = np.asarray(cam_param)  # [t,v,3,7]
            if self.bbox_mode == 'cxyz':
                ret_dict["bboxes"] = ret_dict["bboxes"][:, :, :, :, [6, 5, 7, 2]]  # [B, N_out, max_len, 8 x 3]

            result["bboxes_3d_data"] = DataContainer(ret_dict)
            result["camera_param"] = np.asarray(cam_param)

        # for cam_id, camera_name in enumerate(self.camera_list, 0):
        #     result["camera_param"][:, cam_id, :3, :3] = normalized_camera_intrinsic[camera_name]
        if self.user_frame_emb:
            result['frame_emb'] = align_camera_poses(np.array([frame['ego2global_transformation_matrix'] for frame in annos]))
            if not self.stop_filter_e2w:
                for idx in range(0, len(result['frame_emb']) - 1):
                    dist_by_e2w = calc_dist_by_e2w(
                        result['frame_emb'][idx], result['frame_emb'][idx + 1]
                    )
                    if dist_by_e2w > 4:  # 正常的超高行驶速度120 km/h = 33.3 m/s，4 * FPS = 40 m/s，超过40m/s认定为异常值
                        raise ValueError(f"unormal e2w matrix because the distance between two frame achieve {dist_by_e2w} m/s,"
                                         f" skip the case: {case_name}.")
            # result['frame_emb'][:, :3, 3] = result['frame_emb'][:, :3, 3] / 100 # 取消写死 / 100
        else:
            result['frame_emb'] = np.array([np.eye(4) for _ in range(len(annos))])

        if self.enable_layout:
            result['bev_map_with_aux'] = result["bbox"] # # [V, T, C, H, W]
        else:
            result['bev_map_with_aux'] = torch.zeros((seq_length, 8, 400, 400))

        # 排除nan或者inf 看是否能避免nan
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"{key} contains NaN"
                assert not torch.isinf(value).any(), f"{key} contains Inf"
            elif isinstance(value, np.ndarray):
                assert not np.isnan(value).any(), f"{key} contains NaN"
                assert not np.isinf(value).any(), f"{key} contains Inf"
            else:
                pass
        return result


def split_list(pkl_filename_list: List, splits: Dict):
    file_list = []
    data_list = []
    for pkl_filename in pkl_filename_list:
        with open(pkl_filename, 'rb') as f:
            data_list += pkl.load(f)
        print(pkl_filename, len(data_list))
    for data in data_list:
        file_list.append(data[0])
    # 打乱文件名列表
    random.shuffle(file_list)

    # 计算每个 split 对应的数量
    total_files = len(file_list)
    split_indices = []
    start_idx = 0
    for split_name, ratio in splits.items():
        split_size = int(total_files * ratio)
        print(split_name, split_size)
        split_indices.append((split_name, start_idx, start_idx + split_size))
        start_idx += split_size

    # 创建结果字典
    split_dict = {}
    for split_name, start_idx, end_idx in split_indices:
        split_dict[split_name] = file_list[start_idx:min(end_idx, total_files)]
    print('Split Finish')
    return split_dict


def save_pkl(data, savefilename):
    with open(savefilename, 'wb') as f:
        pkl.dump(data, f)


if __name__ == "__main__":

    from einops import rearrange
    from torchvision.utils import save_image
    import torchvision.transforms as T
    from torchvision.io import write_video
    num_frames = 16
    loop = 1
    processed_meta_file = [
        # # # v3
        # [
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250328/pvb_5kclip_bevlinev3_20250401_geo_v2.pkl",
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250328/pvb_5kclip_bevlinev3_20250401_geo_v2",
        # ],
        # # 0407新增2000左右clip
        # [
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/pvb_2kclip_bevlinev3_20250407_geo.pkl",
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/pvb_2kclip_bevlinev3_20250407_geo",
        # ],
        # # # 预标数据
        # [
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/preanno_3577clip.pkl",
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/preanno_3577clip"
        # ],
        # # # 大卓
        # [
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250418/pvb_deliver_20250418_dazhuo_9503clip_xingche.pkl",
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250418/pvb_deliver_20250418_dazhuo_9503clip.pkl",
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250418/pvb_deliver_20250418_dazhuo_9503clip",
        # ],
        # # #  # cutin 施工场景 带车道线
        # [
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/cutin_and_construction.pkl",
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/cutin_far.pkl",  # 远
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/cutin_mid.pkl",  # 中
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/cutin_close.pkl", # 近
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/lane_pread_20250407/cutin_and_construction",
        # ],
        # # zhiji
        # [
        #     # "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip.pkl",
        #     # "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip"
        #     # 带车道线 183clip 49361 frames
        #     "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo.pkl",
        #     "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo"
        # ],
        # # zhiji 精标 183clip 49361 frames
        # [
        #     "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo.pkl",
        #     "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo",
        # ],
        # zhiji 预标处理 all clip case: 6012  all num_frames: 1634075
        [
            "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250428/zhiji_20250428_preanno_addgeo.pkl",
            "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250428/zhiji_20250428_preanno_addgeo",
        ]
        # # # #cutin
        # [
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/cutin_20250321/data/cutin_data_20250321.pkl",
        #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/cutin_20250321/data/cutin_data_20250321",
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/aigc_data/aigc_for_cutin_20250521/for_convert_night_bigcutin_20250522.pkl", # 用于泛化夜晚cutin 原数据
        #     "/iag_ad_01/ad/xujin2/data/pap_pvb/aigc_data/aigc_for_cutin_20250521/for_convert_night_bigcutin_20250522"
        # ]
        # aigc 测试
        # [
            # "/iag_ad_01/ad/xujin2/data/pap_pvb/aigc_data/cutin_20250417/infer_0415_carlagen_65f_v4.pkl",
            # "/iag_ad_01/ad/xujin2/data/pap_pvb/aigc_data/cutin_20250417/infer_0415_carlagen_65f_v4",
        # ],
        # [
        #     "/iag_ad_01/ad/xujin2/data/zhiji/aigc_gen_0521/from_pap2zhiji_0520_v1/gen_ft_data_v1.pkl",
        #     "/iag_ad_01/ad/xujin2/data/zhiji/aigc_gen_0521/from_pap2zhiji_0520_v1/gen_ft_data_v1",
        # ],

    ]
    # zhiji
    # zhiji_data = [
    #         # "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip.pkl",
    #         # "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip"
    #         # 带车道线 183clip 49361 frames
    #         "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo.pkl",
    #         "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/data/zhiji_20250319_362clip/zhiji_20250319_362clip_addgeo",
    # ]
    # for _ in range(5):
    #     processed_meta_file.append(zhiji_data)
    # aws_path_transform = None
    aws_path_transform = lambda s: f"guoxi-business-v2-internal:s3://pap-move/{s.split(':', 1)[0]}/{s.split('s3://')[1].split('/', 1)[1]}"
    aksk = None
    # aksk = "kaiwu2"
    enable_scene_description = True
    caption_root = "/iag_ad_01/ad/xujin2/data/pap_pvb/caption" #pap
    # caption_root = "/iag_ad_01/ad/xujin2/data/pap_pvb/caption_dazhuo" #pap_dazhuo
    # caption_root = "/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/caption" # zhiji
    scene_description_file = [
        # os.path.abspath(os.path.join(caption_root, file))
        os.path.join(caption_root, file)
        for file in os.listdir(caption_root)
        if file.endswith('.json')
    ]
    # 大卓
    scene_description_file += ["/iag_ad_01/ad/xujin2/data/pap_pvb/caption_dazhuo/caption_pap_20250415.json"]
    # zhiji
    scene_description_file += ["/iag_ad_01/ad/xujin2/data/zhiji/20250319_362clip/caption/zhiji_caption_20250319.json"]
    # scene_description_file = [
    #     "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241125.json",
    #     "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241124.json",
    #     "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241209.json",
    #     "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pap_caption_20250319.json",
    #     "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_new_20250208.json",
    # ]
    add_geo = True
    caption_join_func = "format_caption"
    # add_controller = None   # 不加
    add_controller = {}     # 自动
    # add_controller = {        # 强制设置
    #     "dataset_tag": "zhiji",
    #     "controller": "zhiji",
    # }
    scene_description_file = [
        "/iag_ad_01/ad/xujin2/data/pap_pvb/caption_collect/pap.pkl",
        # "/iag_ad_01/ad/xujin2/data/pap_pvb/caption_collect/dazhuo.pkl",
        "/iag_ad_01/ad/xujin2/data/pap_pvb/caption_collect/zhiji.pkl",
    ]
    # cam_init_path = "/iag_ad_01/ad/xujin2/data/carla/zhiji_cam_params"    # zhiji
    cam_init_path = None

    drop_3d_injection = 1.0
    split = "test"
    # split = "train"
    # set_strat_id = True
    set_strat_id = False
    # random_caption = "random"
    random_caption = "chose"
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
    # random_caption = None
    # random_caption_key = None
    re_sample = None
    # re_sample = {
    #     "path": "/iag_ad_01/ad/xujin2/data/pap_pvb/pack_20250402/re_sample/snowy",
    #     "sample": 4,    # 重采样比率， 设置1 则不重采样
    # }
    CaseName_PAPCaseName_Table_file = "/iag_ad_01/ad/xujin2/data/pap_pvb/CaseName_PAPCaseName_Table_0705_0801.json"
    sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"]
    # sqrt_required_text_keys = ["weather", "time", "lighting", "road_type"]
    fps_list = [10]
    sequence_length = num_frames * loop
    data_fps = fps = 10
    exclude_cameras = None
    edit_type = None
    draw_bbox_mode = 'cross'
    split_file = None  # '/mnt/iag/user/tangweixuan/metavdt/opensora/datasets/data_process/pap_240806-241118_split.json'
    must_text = enable_scene_description
    with_filled_bbox = False
    colorful_box = True
    add_velocity_to_text = False
    with_traj_map = False
    enable_layout = True
    stop_filter_e2w = True
    # *-----------bbox-----------------
    with_bbox_coords = True  # *默认with_camera_param=True
    bbox_drop_prob = 0.05
    drop_bbox_coords_prob = 0
    bbox_mode = 'all-xyz'
    # *-----------bbox-----------------
    # 11v
    # camera_list = ['left_front_camera', 'center_camera_fov120', 'right_front_camera', 'right_rear_camera', \
    #                'rear_camera', 'left_rear_camera', 'center_camera_fov30', 'front_camera_fov195', \
    #                'left_camera_fov195', 'rear_camera_fov195', 'right_camera_fov195']
    # 7v
    camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", \
                   "rear_camera", "left_rear_camera", "left_front_camera", "center_camera_fov30"]

    # # 7v 大卓
    # camera_list = ["front_wide_camera", "rear_main_camera",
    #                   "left_front_camera", "left_rear_camera",
    #                   "right_front_camera", "right_rear_camera",
    #                   "front_main_camera"]

    image_size = (256, 512)
    # image_size = (512, 1024)
    full_size = (image_size[0], image_size[1] * len(camera_list))
    cfg_sequence_length = [1, 9, 17, 33]
    cfg_fps_list = [[100, ], [10, ], [10], [10]]
    pd = PAPVariableDataset(
        None,
        path_to_aoss_config = "/iag_ad_01/ad/xujin2/aoss.conf",
        s3_path = "",
        processed_meta_file=processed_meta_file,
        camera_list=camera_list,
        sequence_length=cfg_sequence_length,
        fps_list=cfg_fps_list,
        data_fps=10,
        split=split,  # * 'test'就不会随机采首帧
        set_strat_id = set_strat_id,
        enable_scene_description=enable_scene_description,
        exclude_cameras=None,
        edit_type=None,
        draw_bbox_mode='cross',
        split_file=split_file,
        scene_description_file=scene_description_file,
        CaseName_PAPCaseName_Table_file=CaseName_PAPCaseName_Table_file,
        sqrt_required_text_keys=sqrt_required_text_keys,
        must_text=must_text,
        expected_vae_size=image_size,
        full_size=full_size,
        colorful_box=colorful_box,
        use_random_seed=False,
        with_filled_bbox=with_filled_bbox,
        add_velocity_to_text=add_velocity_to_text,
        with_traj_map=with_traj_map,
        simple_read_data = True, # 火山云 debug
        add_geo = add_geo,
        re_sample = re_sample,
        # *-----------bbox-----------------
        with_bbox_coords=with_bbox_coords,
        bbox_mode=bbox_mode,
        bbox_drop_prob= bbox_drop_prob,
        enable_layout = enable_layout,
        drop_3d_injection = drop_3d_injection,
        # traj_edit_type = "self_expolate_views", # "speed_up", #"change_road", #"self_expolate_views", # "self_trans",
        traj_edit_type = None,
        random_caption = random_caption,
        random_caption_key = random_caption_key,
        stop_filter_e2w = stop_filter_e2w,
        # *-----------bbox-----------------
        debug=False,
        # aws_path_transform= lambda s: f"guoxi-business-v2-internal:s3://pap-move/{s.split(':',1)[0]}/{s.split('s3://')[1].split('/',1)[1]}"
        aws_path_transform = aws_path_transform,
        caption_join_func = caption_join_func,
        add_controller = add_controller,
        cam_init_path = cam_init_path,
        aksk = aksk,
    )
    print('len(dataset):', len(pd))
    print("constructed PAP dataset.")
    # 将 Tensor 转为帧列表
    to_pil = T.ToPILImage()  # 转换为 PIL Image
    from torch.utils.tensorboard import SummaryWriter
    loacl_show_dataset_tb_dir = "/iag_ad_01/ad/xujin2/code_hsy/magicdrivedit/show_dataset/dataset/tensorboard"
    if os.path.exists(loacl_show_dataset_tb_dir):
        import shutil
        shutil.rmtree(loacl_show_dataset_tb_dir)
    writer = SummaryWriter(loacl_show_dataset_tb_dir)

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
        # f"{random.randint(0, len(pd) - 1)}-97-10"
        f"{random.randint(0, len(pd) - 1)}-33-10"
        # f"{random.randint(0, len(pd) - 1)}-17-10"
        for _ in range(20)
    ]

    # # debug
    # index_list = [
    #     "52255756-17-10",
    #     "3244442-17-10",
    #     "9209808-17-10",
    #     "48738103-17-10",
    #     # "58717843-17-10",
    # ]

    # for i in range(0, 100, 10):
    save_local = True
    for index in tqdm(index_list):
        # data = pd[i]
        # data = pd.get_item(f"{i}-1-100")
        result = pd.get_item(index)
        # 全部转换成 [V, T, C, H, W]
        video = result["pixel_values"].permute(1, 0, 2, 3, 4)  # [T, V, C, H, W] -> [V, T, C, H, W]
        bbox = result["bbox"]
        video = denormalize(video)
        show_data_list = [video, bbox]
        # show_video = format_video(show_data_list)
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
            # save_sample(save_local_video, save_path = save_name) # mkv 无损
            save_sample(save_local_video, save_path = save_name, video_codec = "libx264rgb") # mp4 有损
        show_video = show_video.unsqueeze(0)
        writer.add_video(
            "idx:{} cap:{}".format(index, caption_clean),
            show_video,
            fps=2,
        )
    writer.close()