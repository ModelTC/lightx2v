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
    'VEHICLE_BUS': (255, 255, 0),  # "巴士" bus #黄
    'VEHICLE_PICKUP': (255, 128, 0),  # "皮卡" truck #橙
    'VEHICLE_SPECIAL': (255, 128, 0),  # "特种车" car #橙
    'VEHICLE_TRIKE': (0, 255, 0),  # "三轮车" motorcycle #绿
    'VEHICLE_MULTI_STAGE': (255, 128, 0),  # "多段车单节车体" car #橙
    'PEDESTRIAN_TRAFFIC_POLICE': (0, 0, 255),  # "交警" pedestrian # 蓝
    'VEHICLE_POLICE': (255, 0, 0),  # "警车" car #红
    # "VEHICLE_CAR_CARRIER_TRAILER": (255, 128, 0),  # "拖挂车" # 橙
    "VEHICLE_CAR_CARRIER_TRAILER": (255, 153, 51),  # "拖挂车" # 橙 vd2
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
    # "VEHICLE_CAR_CARRIER_TRAILER": 13,  # "拖挂车" # 橙
    "VEHICLE_CAR_CARRIER_TRAILER": 5,  # "拖挂车" #  # 临时改
    "VEHICLE_TRAILER": 14,  # "拖车" # 橙
    "VEHICLE_RUBBISH": 15  # "垃圾车" # 橙
}

def read_caption(scene_description_file):
    # scene_description_file = [
    #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241125.json",
    #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241124.json",
    #     # "/iag_ad_01/ad/xujin2/data/pap_pvb/caption/pvb_caption_20241209.json",
    #     caption_path
    # ]
    sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"]
    caption_dict = {}
    caption_all = []
    for k in sqrt_required_text_keys:
        caption_dict[k] = []

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
            cap_list = []
            for kk in sqrt_required_text_keys:
                v = data_caption[kk]
                if v not in caption_dict[kk]:
                    caption_dict[kk].append(v)
                cap_list.append(v)
            caption = ".".join(cap_list)
            caption_all.append(caption)
    return caption_dict, caption_all

def carla2group_car(car_type):
    if car_type in [
        "vehicle.carlamotors.carlacola", # Carla Cola（虚拟品牌车辆）
        "vehicle.carlamotors.european_hgv", #  European HGV（重型货车）
        "vehicle.carlamotors.firetruck", # Firetruck（消防车）
        "vehicle.ford.ambulance",
        "vehicle.volkswagen.t2",
        "vehicle.volkswagen.t2_2021",
        "vehicle.mitsubishi.fusorosa",
    ]:    #大货车, 大体型车
        return "big_car"
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
            # "VEHICLE_TRUCK", #"大型货车" truck #橙
            # "VEHICLE_PICKUP",#"皮卡" truck #橙
            # "VEHICLE_SPECIAL",#"特种车" car #橙
            # "VEHICLE_MULTI_STAGE",#"多段车单节车体" car #橙
            "VEHICLE_CAR_CARRIER_TRAILER",#"拖挂车" # 橙
            # "VEHICLE_TRAILER",#"拖车" # 橙
            # "VEHICLE_RUBBISH"
        ])  #"垃圾车" # 橙
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
                    print(e)

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,  # [B, N_out, max_len, 8 x 3]
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict

# def pad_bboxes_to_maxlen(
#         bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
#     B, N_out = bbox_shape[:2]
#     ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
#     # we set unknown to -1. since we have mask, it does not matter.
#     ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
#     ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
#     if bboxes is not None and len(bboxes[0][0]) > 0 and max_len > 0:
#         for _b in range(B):
#             _bboxes = bboxes[_b]
#             _classes = classes[_b]
#             for _n in range(N_out):
#                 if _bboxes[_n] is None:
#                     continue  # empty for this view
#                 try:
#                     this_box_num = len(_bboxes[_n])
#                     if this_box_num > 0:
#                         ret_bboxes[_b, _n, :this_box_num] = torch.tensor(_bboxes[_n]).to(ret_bboxes)
#                         ret_classes[_b, _n, :this_box_num] = torch.tensor(_classes[_n]).to(ret_classes)
#                         if masks is not None:
#                             ret_masks[_b, _n, :this_box_num] = torch.tensor(masks[_b, _n]).to(ret_masks)
#                         else:
#                             ret_masks[_b, _n, :this_box_num] = True
#                 except Exception as e:
#                     traceback.print_exc()
#
#     # assemble as input format
#     ret_dict = {
#         "bboxes": ret_bboxes,  # [B, N_out, max_len, 8 x 3]
#         "classes": ret_classes,
#         "masks": ret_masks
#     }
#     return ret_dict



# @DATASETS.register_module()
class CARLA2VariableDataset(object):
    def __init__(self,
                 raw_meta_files,
                 pap_cam_init_path,
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
        # self.with_bbox_coords = kwargs.get('with_bbox_coords', False)
        self.with_camera_param = kwargs.get("with_camera_param", False)
        self.proj_func = kwargs.get("proj_func", "pap")
        self.pap2cam = {
            "center_camera_fov120": "CAM_FRONT",
            "right_front_camera": "CAM_FRONT_RIGHT",
            "right_rear_camera": "CAM_BACK_RIGHT",
            "rear_camera": "CAM_BACK",
            "left_rear_camera": "CAM_BACK_LEFT",
            "left_front_camera": "CAM_FRONT_LEFT"
        }


        self.transforms = transforms.Compose([
            transforms.Resize(expected_vae_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.condition_transform = transforms.Compose([
            transforms.Lambda(lambda img: self.resize_nearest(img, (256, 512))), # 暂时写死，分辨率可以比图像小
            transforms.ToTensor()
        ])
        if self.use_random_seed: #可能音响起开帧采样
            random.seed(time.time()%100000)
        print('carla2 self.seq_length:', self.seq_length)
        print(Style.BRIGHT + Fore.RED + f'self.fps_list:{self.fps_list} in data_fps:{self.data_fps}' + Style.RESET_ALL)
        # self.fps_stride_list = fps_stride_list
        # self.start_zero_frame = start_zero_frame
        self.specific_video_segment = None
        self.segment_infos = []
        self.segment_data_infos = {}
        self.max_seq_length = max(self.seq_length) if isinstance(self.seq_length, list) else self.seq_length

        if pap_cam_init_path is not None:
            self.cam_params = []
            for cam_json_name in [path for path in os.listdir(pap_cam_init_path) if path.endswith('.json')]:
                cam_json_path = os.path.join(pap_cam_init_path, cam_json_name)
                with open(cam_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.cam_params.append(data)
        else:
            self.cam_params = None

        for annn_root in raw_meta_files:
            print(f"load : {annn_root}")
            case_list = [path for path in os.listdir(annn_root) if os.path.isdir(os.path.join(annn_root, path))]
            for case_name in tqdm(case_list):
                case_root = annn_root
                try:
                    try:
                        data_info = json.load(open(os.path.join(case_root, case_name, 'data_infos.json')))
                    except:
                        data_info = []
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
        self.caption_dict, self.caption_all = read_caption(caption_dir)
        self.description_mode = "random"
        all_case_num = sum([value["case_num_frames"] for value in self.segment_infos])
        print(f"load data clip : {len(self.segment_infos)}  frame num {all_case_num}")

    def get_caption(self):
        if self.description_mode == "chose":
            text = random.choice(self.caption_all)
        elif self.description_mode == "force":
            text = self.caption_all[0]
        elif self.description_mode == "random":
            caption_list = []
            for k, v in self.caption_dict.items():
                caption_list.append(random.choice(v))
            text = ".".join(caption_list)
        else:
            print("self.description_mode : {}".format(self.description_mode))
            assert 0
        return text

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

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index)

    def get_item(self, index) -> dict:
        index, seq_length, selected_fps = [int(item) for item in index.split("-")]
        infos = self.segment_infos[index]
        case_name, case_root, case_num_frames  = infos["case_name"], infos["case_root"], infos["case_num_frames"]
        max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        start_id = 0 if self.split != "train" else random.randint(0, max_fps_id - seq_length - 1)
        # load anno
        frame_id_statr = int(np.floor((start_id + 0) / max_fps_id * case_num_frames))
        annos = self.segment_data_infos[case_name][frame_id_statr:frame_id_statr+seq_length]
        path_to_annos = [os.path.join(case_root, case_name)] * len(annos)

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

        # 获取额外信息
        ego_bbox = np.array(annos[0]["detection_self"]["points"])
        if ego_bbox.shape[-1] > 3:
            ego_bbox = ego_bbox[...,:3]
        ego_x_size, ego_y_size, ego_z_size = bbox8point_get_size(ego_bbox)

        # 初始化黑图
        black_image = Image.new('RGB', (self.expected_vae_size[1], self.expected_vae_size[0]), color='black')
        images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        bbox_images = [[black_image.copy() for _ in range(seq_length)] for _ in range(len(self.camera_list))]  # blank img with bbox
        img_paths = [[f"/iag_ad_01/ad/xujin2/data/vd/other/black_{self.expected_vae_size[0]}_{self.expected_vae_size[1]}.png" for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        if self.with_bbox_coords:
            bboxes_meta_list = [[{} for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        # 随机选取一个pap的相机参数文件
        if self.cam_params:
            cam_param_mate = random.choice(self.cam_params)
            if "Objects" in cam_param_mate:
                cam_param_mate.pop("Objects")  # 丢掉原有的 obj
            if "original_info" in cam_param_mate.keys():
                cam_param_mate.pop("original_info")
            cam_param_mate["case_name"] = case_name

        else:
            cam_param_mate = None

        for i in range(seq_length):
            if i>= len(annos):
                continue
            else:
                # bboxs, classes, conf = get_all_pvb(annos[i])
                # all_lanelines, all_colors = get_all_geo(annos[i])
                frame = annos[i]
                # 获取pvb
                bboxs = []
                bbox_colors = []
                pap_ids = []
                if "detection" in frame.keys():
                    for ins_id, ins in enumerate(frame["detection"]):
                        bbox = ins["points"]
                        bbox = [b[:3] for b in bbox]
                        if self.cam_params:
                            bbox = convert_carla_8point_to_pap(bbox, (ego_x_size, ego_y_size, ego_z_size), conver_center= self.cam_params is not None)  # (x, y, z) -> (x, -y, z) 左手转右手
                            bbox = carla_8points_index_change_order_pap(bbox)
                        bboxs.append(bbox)
                        bbox_colors.append(npc_id2color[ins["type"]])
                        pap_ids.append(npc_id2pap_name[ins["type"]])
                if self.proj_func == "pap":
                    meta_by_frame = copy.deepcopy(cam_param_mate)
                    # 弄一个Objects
                    meta_by_frame["Objects"] = []
                    for bbox3d, color, pap_id in zip(bboxs, bbox_colors, pap_ids):
                        meta_by_frame["Objects"].append({
                            "label": pap_id,
                            "bbox3d": bbox3d,
                            "color": color,
                            # "velocity": velocity
                            "velocity": [0, 0, 0]
                        })

                for cam_id, camera_name in enumerate(self.camera_list, 0):
                    if camera_name in self.pap2cam.keys() and frame['camera_infos'] != {}:
                        CAM_NAME = self.pap2cam[camera_name]
                        rgb_relative_path = frame['camera_infos'][CAM_NAME]['rgb']
                        if case_name in rgb_relative_path:
                            rgb_path = os.path.join(case_root, rgb_relative_path)
                        else:
                            rgb_path = os.path.join(case_root, case_name, rgb_relative_path)
                        img_paths[cam_id][i] = rgb_path
                        img = Image.open(rgb_path)
                        try:
                            bbox_local = Image.open(rgb_path.replace("rgb", "3dbox"))
                        except:
                            bbox_local = black_image.copy()
                    else: # 黑图
                        img = images[cam_id][i]
                        bbox_local = black_image.copy()

                    #
                    if self.cam_params:
                        camera_intrinsic = cam_param_mate['cams'][camera_name]['cam_intrinsic']
                        lidar2camera_rt = cam_param_mate['cams'][camera_name]['extrinsic']
                        img_dist_w = cam_param_mate['cams'][camera_name]['img_width']
                        img_dist_h = cam_param_mate['cams'][camera_name]['img_width']

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
                    lidar_camera_extrinsic = np.matrix(
                        np.array(lidar2camera_rt).reshape(4, 4))

                    # add pvb
                    if len(bboxs) > 0:
                        # if self.proj_func == "pap":
                        #     bbox_image, bboxes_meta = draw_bbox_alt_camera_meta(np.array(bbox_image), meta_by_frame, camera_name, img_shape = None, sx=sx, sy=sy, bboxmode="8points", colorful_box=True)
                        #     bboxs_coners_in_cam = bboxes_meta['lidar_in_cams']
                        #     pap_ids_in_cam = bboxes_meta['box_classes']
                        # else:
                        bbox_image, bboxs_coners_in_cam, pap_ids_in_cam = proj_pvb(np.array(bbox_image), bboxs, pap_ids, camera_intrinsic, lidar_camera_extrinsic, id2color = CLASS_BBOX_COLOR_MAP)
                        # # for debug
                        # img_, _, _ = proj_pvb(np.array(img), bboxs, pap_ids, camera_intrinsic, lidar_camera_extrinsic)
                        # img = Image.fromarray(img_)
                    else:
                        bboxs_coners_in_cam = bboxs
                        pap_ids_in_cam = pap_ids

                    images[cam_id][i] = img
                    # bbox_images[cam_id][i] = Image.fromarray(bbox_image)
                    bbox_images[cam_id][i] = Image.fromarray(bbox_image) if self.cam_params else bbox_local
                    if self.with_bbox_coords:
                        # assert len(classes_in_cam_number) == len(bboxs_coners_in_cam)
                        camera_param = np.concatenate([camera_intrinsic[:3, :3], lidar2camera_rt[:3]], axis=1)
                        # if self.proj_func == "pap":
                        #     classes = pap_ids_in_cam
                        # else:

                        classes = [CLASS_BBOX_ID_MAP[data] for data in pap_ids_in_cam] # 映射成id 1,2,3,这种。
                        bboxes_meta_list[cam_id][i] = {  # boxes_coords, camera_meta, lidar_in_cams
                            # "lidar_in_worlds":bboxes_meta[0], #[n,8,3]
                            "cam_params": camera_param,  # !要[3,7]
                            "bboxes": np.array(bboxs_coners_in_cam),  #这里是全局3d点，不是对应相机的3d点
                            # [n,8,3] # lidar_in_cams #[B, N_out, max_len, 8 x 3] #! 看清楚是要cam还是world: 是cam
                            # "classes": classes  # [n,] #!要类别号码
                            "classes": classes
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

        result["captions"] = [self.get_caption()] * seq_length

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
                ego2global_transformation_matrix.append(pose)
            ego2global = np.array(ego2global_transformation_matrix)
            result['frame_emb'] = align_camera_poses(ego2global)
        else:
            result['frame_emb'] = np.array([np.zeros((4, 4)) for frame in annos])

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
    raw_meta_files = [
        "/iag_ad_01/ad/xujin2/data/carla/old_0303/bigcarin1",
        "/iag_ad_01/ad/xujin2/data/carla/old_0303/bigcarin2",
    ]
    # pap_cam_init_path = "/iag_ad_01/ad/xujin2/data/carla/pap_cam_params"
    pap_cam_init_path = "/iag_ad_01/ad/xujin2/data/carla/vd2_params"
    # pap_cam_init_path = None
    # pap_cam_init_path = None
    caption_dir = ["/iag_ad_01/ad/xujin2/data/vd/vd2/caption_file/vd2_caption_20250310.json"]
    # 7v
    camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", "rear_camera",
                   "left_rear_camera", "left_front_camera", "center_camera_fov30"]

    sequence_length = [1, 17, 33]
    fps_list = [10]
    data_fps = 10
    image_size = (256, 512)
    # image_size = (512, 1024)
    full_size = (image_size[0], image_size[1] * len(camera_list))  # MULTI_VIEW
    ADdataset = CARLA2VariableDataset(
                raw_meta_files = raw_meta_files,
                pap_cam_init_path = pap_cam_init_path,
                caption_dir = caption_dir,
                path_to_aoss_config = "/iag_ad_01/ad/xujin2/aoss.conf",
                s3_path = "",
                camera_list = camera_list,
                sequence_length = sequence_length,
                fps_list = fps_list,
                data_fps = data_fps,
                expected_vae_size = image_size,
                full_size = full_size
                )
    print('carla:', len(ADdataset))
    # 将 Tensor 转为帧列表
    to_pil = transforms.ToPILImage()  # 转换为 PIL Image
    from torch.utils.tensorboard import SummaryWriter
    loacl_show_dataset_tb_dir = "/iag_ad_01/ad/xujin2/code_hsy/magicdrivedit/show_dataset/dataset/tensorboard"
    if os.path.exists(loacl_show_dataset_tb_dir):
        import shutil
        shutil.rmtree(loacl_show_dataset_tb_dir)
    writer = SummaryWriter(loacl_show_dataset_tb_dir)
    # sample_id = [random.randint(0, len(ADdataset) - 1) for _ in range(10)]
    # index_list = [
    #     # f"{random.randint(0, len(ADdataset) - 1)}-17-10"
    #     f"{random.randint(0, len(ADdataset) - 1)}-2-10"
    #     for _ in range(4)
    # ]
    index_list = [
        # f"{random.randint(0, len(ADdataset) - 1)}-17-10"
        f"{id}-10-10"
        for id in range(5)
    ]
    # for debug
    # index_list = [
    #     "7383918-10-17",
    #     "24716883-10-17",
    #     "5998949-10-17",
    # ]
    for index in tqdm(index_list):
        result = ADdataset.get_item(index)
        # 全部转换成 [V, T, C, H, W]
        video = result["pixel_values"].permute(1, 0, 2, 3, 4) #  [T, V, C, H, W] -> [V, T, C, H, W]
        bbox = result["bbox"]
        video = denormalize(video)
        show_data_list = [video, bbox]
        show_video = format_video(show_data_list)
        show_video = show_video.unsqueeze(0)
        caption = result["captions"][0]
        # caption = ""
        writer.add_video(
            "idx:{} cap:{}".format(index, caption),
            show_video,
            fps=2,
        )
