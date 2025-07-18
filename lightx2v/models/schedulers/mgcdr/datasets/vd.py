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

IMG_FPS = 100

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


def proj_pvb(img, all_bbox, all_classes, camera_intrinsic, lidar_camera_extrinsic):
    all_bbox_8point_in_cam = []
    all_bbox_4point_in_cam_classes = []
    # get corners
    vis_object = []
    for gt_box, gt_name in zip(all_bbox, all_classes):
        # get corners
        corners_gt_box = get_coners(gt_box)  # to 8定点
        image_points, point_3d_in_cam = lidar2image(
            corners_gt_box, camera_intrinsic, lidar_camera_extrinsic)
        if image_points is not None:
            vis_object.append(image_points)
            all_bbox_8point_in_cam.append(point_3d_in_cam.T)
            all_bbox_4point_in_cam_classes.append(gt_name)
    # 画图，画在img上
    out_h, out_w, _ = img.shape
    for obj in vis_object:
        if obj is not None:
            img = draw_rect3d_on_img(img,obj, out_h, out_w)
    return img, all_bbox_8point_in_cam, all_bbox_4point_in_cam_classes

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

def get_all_pvb(data):
    all_bbox = []
    all_conf = []
    all_classes = []
    if data["pvb"]!= {} and data["pvb"]["perception_object_list"] is not None:
        for _, obj in enumerate(data["pvb"]["perception_object_list"]):
            conf = obj["type_confidence"]
            if conf < 0.3:
                continue
            center = obj['motion_info']['center']
            size = obj['size_info']['size']
            yaw = obj['direction_info']['yaw']
            label = obj['label']
            velocity = obj['motion_info']['velocity'] # 暂时没用到 vx vy
            label_map = CLASS_PRED.get(int(label), None)

            if label_map is None:
                print(f"pred label id {str(label)} map_res {str(label_map)} is not in record")
                continue
            all_classes.append(str(label_map))
            all_bbox.append([
                center['x'],
                center['y'],
                center['z'] + 1 / 2 * size['z'],
                size['y'],
                size['x'],
                size['z'],
                yaw,
            ])
            all_conf.append(conf)
    return all_bbox, all_classes, all_conf

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

def get_all_geo(data):
    all_lanelines, all_colors = [], []
    if data["geo"] != {} and data["geo"]["road_geometry"]["laneline_results"] is not None:
        for line in data["geo"]["road_geometry"]["laneline_results"]:
            points = []
            for point in line["world_points_reproj"]:
                points.append([point["x"], point["y"], 0.0])
            all_lanelines.append(points)
            all_colors.append(line["color_id"])
    return all_lanelines, all_colors

def proj_geo(img, all_lines, all_classes, camera_intrinsic, lidar2camera_rt):
    for line, line_id in zip(all_lines, all_classes):
        line = np.array(line)
        draw_lane_line(img, line, line_id, camera_intrinsic, lidar2camera_rt)
    return img

def draw_lane_line(image, line, line_id, camera_intrinsic, lidar_camera_extrinsic, THICKNESS=2):
    GT_COLOR = {
        2: (0, 255, 0), # "LANELINE"
        3: (255, 0, 0)  # "ROADSIDE"
    }
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
                    color=GT_COLOR.get(line_id, (0, 0 , 0)),
                    thickness=THICKNESS,
                    lineType=cv2.LINE_AA,
                )
            except Exception as e:
                print(e)
                print("划线报错")
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
                    print(e)

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

def convert_to_ego2global_transformation_matrix(anno):
    ego2global_transformation_matrix = np.eye(4)
    if anno["pose"] != {}:
        """
        将 7 个数（位置坐标和姿态角度）转换为 ego2global 转换矩阵
        输入：odometry_info = [x, y, z, vx, vy, vz, yaw]
        输出：4x4 转换矩阵
        """
        odometry_info = anno["pose"]["data"]
        # 提取位置和姿态信息
        x, y, z, vx, vy, vz, yaw = odometry_info  # 现在只提取位置(x, y, z)和姿态(yaw)
        # 计算旋转矩阵
        R = euler_to_rotation_matrix(0, 0, yaw)  # 假设只有yaw角度，不需要roll和pitch
        # 创建平移向量
        t = np.array([x, y, z])
        # 创建 4x4 的齐次变换矩阵
        ego2global_transformation_matrix[:3, :3] = R  # 旋转矩阵
        ego2global_transformation_matrix[:3, 3] = t  # 平移向量

    return ego2global_transformation_matrix

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
class VDVariableDataset(object):
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
                 expected_vae_size=(288, 512),
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
        self.pvb_CLASS_PRED_inv = {}
        for k,v in CLASS_PRED.items():
            self.pvb_CLASS_PRED_inv[v] = k

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
        print('vd self.seq_length:', self.seq_length)
        print(Style.BRIGHT + Fore.RED + f'self.fps_list:{self.fps_list} in data_fps:{self.data_fps}' + Style.RESET_ALL)
        # self.fps_stride_list = fps_stride_list
        # self.start_zero_frame = start_zero_frame
        self.specific_video_segment = None
        self.segment_infos = []
        self.max_seq_length = max(self.seq_length) if isinstance(self.seq_length, list) else self.seq_length
        for annn_root in raw_meta_files:
            print(f"load : {annn_root}")
            if ".pkl" in annn_root:
                with open(annn_root, "rb") as f:
                    loaded_data = pkl.load(f)
                self.segment_infos += loaded_data
                continue

            # 读取远程s3地址
            if "s3://" in annn_root:
                pass # 预留
            # 读取本地afs
            else:
                case_list = [path for path in os.listdir(annn_root) if os.path.isdir(os.path.join(annn_root, path))]
                for case_name in tqdm(case_list):
                    case_root = annn_root
                    case_num_frames = len(glob.glob(os.path.join(os.path.join(annn_root, case_name), '*.json')))
                    if case_num_frames < self.max_seq_length:
                        continue

                    self.segment_infos.append({
                        "case_name": case_name,
                        "case_root": case_root,
                        "case_num_frames": case_num_frames,
                    })
            # 存储一遍pkl文件下次读取直接读pkl
            with open(annn_root + ".pkl", "wb") as f:
                pkl.dump(self.segment_infos, f)

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
        # 过滤只保留有caption的case
        segment_infos_on_caption = []
        for item in self.segment_infos:
            if item["case_name"] in self.scene_description_data.keys():
                segment_infos_on_caption.append(item)
        self.segment_infos = segment_infos_on_caption

        all_case_num = sum([value["case_num_frames"] for value in self.segment_infos])
        print(f"load data clip : {len(self.segment_infos)}  frame num {all_case_num}")
        if self.split == 'train':
            self.segment_infos = self.segment_infos * 10000


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

    def get_item(self, index) -> dict:
        index, seq_length, selected_fps = [int(item) for item in index.split("-")]
        infos = self.segment_infos[index]
        case_name, case_root, case_num_frames  = infos["case_name"], infos["case_root"], infos["case_num_frames"]
        max_fps_id = int(np.floor(case_num_frames / self.data_fps * selected_fps))
        start_id = 0 if self.split != "train" else random.randint(0, max_fps_id - seq_length - 1)
        # load anno
        annos = []
        path_to_annos = []
        for i in range(seq_length):
            frame_id = int(np.floor((start_id + i) / max_fps_id * case_num_frames))
            anno_json_path = os.path.join(case_root,case_name, f"{frame_id:04d}.json")
            if "s3://" in anno_json_path:
                anno_json = json.loads(self.reader.load_file(anno_json_path))
            else:
                with open(anno_json_path) as fp:
                    anno_json = json.load(fp)
            annos.append(anno_json)
            path_to_annos.append(anno_json_path)

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
                        combine_text = []
                        for key in self.sqrt_required_text_keys:
                            if key in all_scene_description['caption'].keys():
                                combine_text.append(all_scene_description['caption'][key])
                            elif key in all_scene_description['caption'][camera_name].keys():
                                combine_text.append(all_scene_description['caption'][camera_name][key])

                        scene_description[camera_name] = '.'.join(combine_text)
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
        img_paths = [[f"/iag_ad_01/ad/xujin2/data/vd/other/black_{self.expected_vae_size[0]}_{self.expected_vae_size[1]}.png" for _ in range(seq_length)] for _ in range(len(self.camera_list))]
        if self.with_bbox_coords:
            bboxes_meta_list = [[{} for _ in range(seq_length)] for _ in range(len(self.camera_list))]

        for i in range(seq_length):
            if i>= len(annos):
                continue
            else:
                bboxs, classes, conf = get_all_pvb(annos[i])
                all_lanelines, all_colors = get_all_geo(annos[i])
                root = annos[i]["root"]
                camera_params = annos[i]["camera_params"]
                for cam_id, camera_name in enumerate(self.camera_list, 0):
                    img_load_path = os.path.join(root, annos[i]["boundle_img_paths"][camera_name])
                    img_paths[cam_id][i] = img_load_path
                    img_byte_stream = self.reader.load_file(img_load_path)
                    img = Image.open(io.BytesIO(img_byte_stream))
                    camera_intrinsic = camera_params[camera_name]["intrinsic"]["param"]["cam_K_new"]["data"]
                    lidar2camera_rt = camera_params[camera_name]["extrinsic"]["param"]["sensor_calib"]["data"]
                    # camera_intrinsic_dist = camera_params[c_name]["intrinsic"]["param"]["cam_dist"]["data"]
                    img_dist_w = camera_params[camera_name]["intrinsic"]["param"]["img_dist_w"]
                    img_dist_h = camera_params[camera_name]["intrinsic"]["param"]["img_dist_h"]
                    assert img.size == (img_dist_w, img_dist_h) # org size
                    img = img.resize((self.expected_vae_size[1], self.expected_vae_size[0]))
                    images[cam_id][i] = img
                    bbox_image = np.array(bbox_images[cam_id][i])
                    sx, sy = self.expected_vae_size[1] / img_dist_w, self.expected_vae_size[0] / img_dist_h
                    # 内外参前处理
                    # 缩放内参到最终size上
                    camera_intrinsic = np.array(camera_intrinsic) * np.array([sx, sy, 1])[:, np.newaxis]
                    # camera_intrinsic = np.array(camera_intrinsic)
                    lidar2camera_rt = np.linalg.inv(np.array(lidar2camera_rt))
                    lidar_camera_extrinsic = np.matrix(
                        np.array(lidar2camera_rt).reshape(4, 4))

                    # add geo
                    if len(all_lanelines) > 0:
                        bbox_image = proj_geo(np.array(bbox_image), all_lanelines, all_colors, camera_intrinsic,
                                              lidar_camera_extrinsic)
                    # add pvb
                    if len(bboxs) > 0:
                        bbox_image, bboxs_coners_in_cam, classes_in_cam = proj_pvb(np.array(bbox_image), bboxs, classes, camera_intrinsic, lidar_camera_extrinsic)
                    else:
                        bboxs_coners_in_cam = bboxs
                        classes_in_cam = classes

                    bbox_images[cam_id][i] = Image.fromarray(bbox_image)
                    if self.with_bbox_coords:
                        classes_in_cam_number = [self.pvb_CLASS_PRED_inv.get(key, 0) for key in classes_in_cam]
                        # assert len(classes_in_cam_number) == len(bboxs_coners_in_cam)
                        camera_param = np.concatenate([camera_intrinsic[:3, :3], lidar2camera_rt[:3]], axis=1)
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
            ego2global = np.array([convert_to_ego2global_transformation_matrix(frame) for frame in annos])
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
        # "/iag_ad_01/ad/xujin2/data/vd/debug_20250219/debug_save_v3",
        "/iag_ad_01/ad/xujin2/data/vd/debug_20250219/debug_save_v3.pkl",
    ]
    caption_dir = "/iag_ad_01/ad/xujin2/data/vd/caption_file/caption_0226"

    # 7v
    camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", "rear_camera",
                   "left_rear_camera", "left_front_camera", "center_camera_fov30"]
    sequence_length = [1, 17, 33]
    fps_list = [10]
    data_fps = 10
    image_size = (256, 512)
    full_size = (image_size[0], image_size[1] * len(camera_list))  # MULTI_VIEW

    ADdataset = VDVariableDataset(
                raw_meta_files = raw_meta_files,
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
        f"{random.randint(0, len(ADdataset) - 1)}-10-10"
        for _ in range(10)
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
        # caption = result["captions"][0]
        caption = ""
        writer.add_video(
            "idx:{} cap:{}".format(index, caption),
            show_video,
            fps=2,
        )
