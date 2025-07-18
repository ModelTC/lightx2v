# Standard Library
import os

# Import from third library
import cv2
import torch
import copy
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

# Import from alt
# import open3d
from alt.utils.coordinate_helper import (
    transform_box3d_to_image,
    transform_box3d_to_point_cloud,
    transform_corners_3d_to_birdseye,
)
from alt.utils.load_helper import AutolabelObjectLoader
from alt.utils.petrel_helper import global_petrel_helper
from alt.visualize.object.visualize import Visualize
from easydict import EasyDict as edict

from alt.coordinate.points.utils import Camera3DPointsTransfer

os.environ["DISPLAY"] = ":99"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


class BEVVisualizer(Visualize):
    box3d_color = (0, 0, 255)  # red
    box3d_no_match_color = (0, 0, 0)  # black
    box2d_color = (0, 255, 0)  # green
    box2d_no_match_color = (190, 190, 190)  # grey
    forward_color = (255, 255, 0)  # cyan

    def __init__(self,
                 meta_json: str,
                 box2d_threshold=0.05,
                 box3d_threshold=0.05,
                 lidar_side_range=(-50.0, 50.0),
                 lidar_fwd_range=(-100.0, 200.0),
                 num_horizontal_bins=20,
                 save_path=None,
                 vis_unmatch=False,
                 path_mapping={}) -> None:

        self.loader = AutolabelObjectLoader(meta_json, build_bev_meta=False)
        self.prepare_save_path(save_path=save_path)

        self.box2d_threshold = box2d_threshold
        self.box3d_threshold = box3d_threshold
        self.lidar_side_range = lidar_side_range
        self.lidar_fwd_range = lidar_fwd_range
        self.num_horizontal_bins = num_horizontal_bins
        self.vis_unmatch = vis_unmatch

        self.base_bev_imgs = {}
        self.path_mapping = path_mapping

        self.cap = None  # 写视频，如果需要

    def prepare_save_path(self, save_path):
        self.video_path = os.path.join(save_path, 'vis.mp4')

        if save_path is None:
            self.save_path = "{}/visualize/sensebee/images".format(self.loader.data_annotation_dir)
        else:
            self.image_path = os.path.join(save_path, 'images')
            if "s3:" not in self.image_path:
                os.makedirs(self.image_path, exist_ok=True)

    def get_merge_bev_img(self, frames, img_h=1080):
        outputs = []

        res = (self.lidar_fwd_range[1] - self.lidar_fwd_range[0]) / img_h
        for camera_name, gt_frame in frames.items():
            cam2lidar, camera_intrinsic, camera_intrinsic_dist = self.loader.get_camera_parameters(camera_name)
            for obj in gt_frame['objects']:
                bbox3d = obj['bbox3d']
                vel = obj['vel']

                if bbox3d is not None and obj['score3d'] >= self.box3d_threshold:
                    roll_cam, pitch_cam, yaw_cam = None, None, None
                    if len(bbox3d) == 7:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, yaw_cam] = bbox3d
                    else:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, roll_cam, pitch_cam, yaw_cam] = bbox3d

                    dimension = [w, h, l]
                    location = [cam_box_x, cam_box_y, cam_box_z]

                    corners_3d = transform_box3d_to_point_cloud(dimension, location, roll_cam, pitch_cam, yaw_cam, cam2lidar)
                    if obj['conf']:  # 证明有匹配
                        x_bbox2d, y_bbox2d = transform_corners_3d_to_birdseye(corners_3d, res, self.lidar_side_range, self.lidar_fwd_range)

                        center_x = int((int(x_bbox2d[2]) + int(x_bbox2d[3])) / 2)
                        center_y = int((int(y_bbox2d[2]) + int(y_bbox2d[3])) / 2)
                        if vel is None:
                            tgt_point = (int(center_x), int(center_y))
                        else:
                            # import ipdb; ipdb.set_trace()
                            vel_lidar = cam2lidar[:-1, :-1] @ vel
                            tgt_point = (int(center_x - vel_lidar[1] / res), int(center_y - vel_lidar[0] / res))

                        obj.update({"bev_corners_3d": corners_3d,
                                    "location": np.mean(corners_3d, axis=0),
                                    "dimension": dimension,
                                    "bev_vel": vel,
                                    "timestamp": gt_frame["timestamp"],
                                    "filename": gt_frame["filename"],
                                    "camera_name": camera_name,
                                    "bev_img_metas": (x_bbox2d, y_bbox2d, center_x, center_y, tgt_point)})
                        outputs.append(edict(obj))

        # 计算相互之间的关系, # 默认只存一个相机的信息, TODO
        target_cache_ids = []
        keep_outputs = []
        for target in outputs:
            if target.id in target_cache_ids:
                continue
            target_cache_ids.append(target.id)
            keep_outputs.append(target['bev_img_metas'])

        img_bev = self.prepare_base_bev(img_h=img_h)
        for (x_bbox2d, y_bbox2d, center_x, center_y, tgt_point) in keep_outputs:
            thickness = int(img_h * 0.001 + 0.5)
            cv2.line(img_bev, (x_bbox2d[2], y_bbox2d[2]), (x_bbox2d[3], y_bbox2d[3]), color=self.forward_color, thickness=thickness)
            cv2.line(img_bev, (x_bbox2d[3], y_bbox2d[3]), (x_bbox2d[7], y_bbox2d[7]), color=self.box3d_color, thickness=thickness)
            cv2.line(img_bev, (x_bbox2d[7], y_bbox2d[7]), (x_bbox2d[6], y_bbox2d[6]), color=self.box3d_color, thickness=thickness)
            cv2.line(img_bev, (x_bbox2d[6], y_bbox2d[6]), (x_bbox2d[2], y_bbox2d[2]), color=self.box3d_color, thickness=thickness)

            cv2.arrowedLine(img_bev, pt1=(center_x, center_y), pt2=tgt_point, color=self.box3d_color, thickness=thickness, line_type=cv2.LINE_8, shift=0, tipLength=0.1)

        return img_bev

    def draw_lidars(self, cur_timestamp, img_shape=(720, 2160)):
        lidar_path = self.loader._find_latest_lidar_file(cur_timestamp)
        if lidar_path.endswith(".bin"):
            pcd_array = global_petrel_helper.load_bin(lidar_path)
        elif lidar_path.endswith(".pcd"):
            pcd_array = global_petrel_helper.load_pcd(lidar_path)[1]

        self.source_data['lidar'] = copy.deepcopy(pcd_array)

        pcd_array = pcd_array[
            (pcd_array[:, 0] <= 200) & (pcd_array[:, 0] >= -100) & (pcd_array[:, 1] >= -50) & (pcd_array[:, 1] <= 50)
        ]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_array[:, :3])
        intensities = pcd_array[:, 3]

        # 将反射强度归一化到0-1范围
        if intensities.max() != intensities.min():
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        # 将归一化的反射强度映射到颜色，使用Matplotlib的colormap
        colormap = cm.get_cmap("viridis")  # 使用viridis颜色映射，可以根据需要选择其他映射
        # 将反射强度转换为颜色
        colors = colormap(intensities)[:, :3]  # 获取RGB值
        pcd.colors = open3d.utility.Vector3dVector(colors)

        rotation_matrix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))  # 绕z轴旋转
        pcd.rotate(rotation_matrix, center=(0, 0, 0))

        # 创建OffscreenRenderer对象
        width, height = img_shape
        renderer = open3d.visualization.rendering.OffscreenRenderer(width, height)

        # 设置渲染的背景色为黑色
        renderer.scene.set_background([0, 0, 0, 1])

        # 添加点云到场景中
        material = open3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 0.1
        renderer.scene.add_geometry("point_cloud", pcd, material)

        # 设置摄像机参数
        # center = pcd.get_center()
        center = [0, 0, 0]
        eye = center + np.array([0, 0, 80])
        renderer.scene.camera.look_at(center, eye, [0, 1, 0])

        image = renderer.render_to_image()
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv2_img

    def init_video(self, image_width, image_height):
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # 视频编解码器
        self.cap = cv2.VideoWriter(self.video_path, fourcc, 10, (image_width, image_height))  # 写入视频

    def draw_lidar_points(self, cur_timestamp):
        for sensor_name, sensor_data in self.source_data.items():
            if sensor_name == "lidar":
                continue
            cam2lidar, camera_intrinsic, camera_intrinsic_dist = self.loader.get_camera_parameters(sensor_name)

            camera_intrinsic = np.array(camera_intrinsic)[:3, :3]
            lidar2camera_rt = np.linalg.inv(cam2lidar)

            pc_velo = np.insert(self.source_data['lidar'][:, :3], 3, values=1, axis=1)
            # 相机坐标系下的XYZ
            cam_points = (lidar2camera_rt @ pc_velo.T).T
            cam_points = cam_points[:, :3]

            limit_cam_z = np.logical_and(cam_points[:, 2] >= 0.1, cam_points[:, 2] <= np.inf)
            cam_points = torch.Tensor(cam_points[limit_cam_z])

            if '195' not in sensor_name:
                cam_2ds = Camera3DPointsTransfer.transfer_to_pinhole(cam_points, camera_intrinsic)
            else:
                cam_2ds = Camera3DPointsTransfer.transfer_to_fisheye(cam_points, camera_intrinsic, camera_intrinsic_dist[0], "KB")

            image_shape = sensor_data.shape
            in_cam = (cam_2ds[:, 0] >= 0) & (cam_2ds[:, 0] <= image_shape[1]) & (cam_2ds[:, 1] >= 0) & (cam_2ds[:, 1] <= image_shape[0])

            for idx, point in enumerate(cam_2ds[in_cam]):
                sensor_data = cv2.circle(sensor_data, [int(x) for x in point], 1, color=[255, 0, 0])

            save_path = os.path.join(self.image_path, 'vis_lidar_point', str(cur_timestamp))
            os.makedirs(save_path, exist_ok=True)

            cv2.imwrite(f'{save_path}/{sensor_name}.png', sensor_data)

    def process(self, select_timestamps=None, ext='.png'):
        results = {}

        if select_timestamps is None:
            vis_timestamps = self.loader.intersect_timstamps
        else:
            vis_timestamps = [item for item in self.loader.intersect_timstamps if item in select_timestamps]

        for cur_timestamp in tqdm(vis_timestamps, desc="total timestamps", total=len(vis_timestamps)):
            imgs, gt_frames = {}, {}
            self.source_data = {}
            for camera_name in self.loader.camera_names:
                gt_frames[camera_name] = self.loader.select_frame_by_timestamp_and_camera_name(camera_name, cur_timestamp)

                vis_cam_img = self.visualize_single_image(camera_name, gt_frames[camera_name], with_bev=False)
                imgs[camera_name] = cv2.resize(vis_cam_img, (960, 540))

            img_bev = self.get_merge_bev_img(gt_frames, img_h=2160)

            lidar_img = self.draw_lidars(cur_timestamp)
            # self.draw_lidar_points(cur_timestamp)

            merge_img = np.zeros((2160, 3840, 3), np.uint8)

            merge_img[0: 540, 0: 960, :] = imgs['center_camera_fov120']
            merge_img[0: 540, 960: 1920, :] = imgs['center_camera_fov30']
            merge_img[540: 1080, 0: 960, :] = imgs['left_front_camera']
            merge_img[540: 1080, 960: 1920, :] = imgs['right_front_camera']
            merge_img[1080: 1620, 0: 960, :] = imgs['left_rear_camera']
            merge_img[1080: 1620, 960: 1920, :] = imgs['right_rear_camera']
            merge_img[1620: 2160, 960: 1920, :] = imgs['rear_camera']

            merge_img[0: 540, -960:, :] = imgs['front_camera_fov195']
            merge_img[540: 1080, -960:, :] = imgs['left_camera_fov195']
            merge_img[1080: 1620, -960:, :] = imgs['right_camera_fov195']
            merge_img[1620: 2160, -960:, :] = imgs['rear_camera_fov195']

            merge_img[:, 1980: 1980 + 720, :] = img_bev

            merge_img = cv2.hconcat([merge_img, lidar_img])

            if self.cap is None:
                self.init_video(image_width=merge_img.shape[1], image_height=merge_img.shape[0])

            save_path = os.path.join(self.image_path, str(int(cur_timestamp) * 1000 * 1000) + ext)
            global_petrel_helper.imwrite(save_path, merge_img, ext=ext)
            self.cap.write(merge_img)
            # logger.info("save to {}".format(save_path))
            results[cur_timestamp] = save_path

        if self.cap is not None:
            self.cap.release()
        return results

    def prepare_base_bev(self, img_h):
        if img_h in self.base_bev_imgs:
            return self.base_bev_imgs[img_h].copy()

        thickness = int(img_h * 0.001 + 0.5)
        lidar_scale = (self.lidar_fwd_range[1] - self.lidar_fwd_range[0]) / (self.lidar_side_range[1] - self.lidar_side_range[0])
        img_bev_w = int(img_h / lidar_scale)
        img_bev_tmp = np.zeros((img_h, img_bev_w, 3), np.uint8)
        num_horizontal_bins = self.num_horizontal_bins
        gap_lines = int(img_h / num_horizontal_bins)
        line_color = (32, 165, 218)
        for i in range(num_horizontal_bins + 1):
            if i % 5 == 0:
                line_thickness = thickness + 1
            else:
                line_thickness = thickness
            pt1 = (0, i * gap_lines)
            pt2 = (img_bev_w, i * gap_lines)
            cv2.line(img_bev_tmp, pt1, pt2, color=line_color, thickness=line_thickness)
        num_vertical_bins = int(img_bev_w / gap_lines)
        for i in range(num_vertical_bins + 1):
            if i % 5 == 0:
                line_thickness = thickness  # noqa
            else:
                line_thickness = thickness
            pt1 = (i * gap_lines, 0)
            pt2 = (i * gap_lines, img_h)
            cv2.line(img_bev_tmp, pt1, pt2, color=line_color, thickness=line_thickness)

        center_y = int(img_h / (self.lidar_fwd_range[1] - self.lidar_fwd_range[0]) * self.lidar_fwd_range[1])
        center_x = int(img_bev_w / (self.lidar_side_range[1] - self.lidar_side_range[0]) * self.lidar_side_range[1])
        cv2.circle(img_bev_tmp, (center_x, center_y), thickness * 6, line_color, thickness=-1)
        self.base_bev_imgs[img_h] = img_bev_tmp
        return self.base_bev_imgs[img_h].copy()

    def visualize_single_image(self, camera_name, camera_frame, with_bev=True):
        image_path = camera_frame["filename"]

        for key, value in self.path_mapping.items():
            if key in image_path:
                image_path = image_path.replace(key, value)

        img = global_petrel_helper.read_img(image_path)  # BGR image format

        self.source_data[camera_name] = copy.deepcopy(img)
        img_w, img_h = img.shape[1], img.shape[0] # noqa

        image_name = image_path.split('/')[-1]
        timestamp = camera_frame['timestamp']
        font_scale = img_h * 0.0003
        thickness = int(img_h * 0.001 + 0.5)
        cv2.putText(img, text=f'img: {image_name}', org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.0005)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale * 2, color=(0, 255, 0), thickness=thickness)

        try:
            lidar_file, time_diff = self._find_latest_lidar_file(timestamp)
            lidar_name = lidar_file.split('/')[-1]
        except Exception as e:  # noqa
            lidar_name = "car-center"
            time_diff = 0
        cv2.putText(img, text=f'bin: {lidar_name}', org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.001)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale * 2, color=(0, 255, 0), thickness=thickness)
        cv2.putText(img, text=f'time_diff: {time_diff:7.3f} ms', org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.0015)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale * 2, color=(0, 255, 0), thickness=thickness)

        cam2lidar, camera_intrinsic, camera_intrinsic_dist = self.loader.get_camera_parameters(camera_name)

        if with_bev:
            img_bev = self.prepare_base_bev(img_h)

        # load and show object bboxes
        for obj in camera_frame['objects']:
            bbox2d = obj['bbox2d']
            if bbox2d is not None and obj['score2d'] >= self.box2d_threshold:
                bbox = [int(p) for p in bbox2d]
                if not self.vis_unmatch and obj['conf'] is None:
                    continue

                bbox_color = self.box2d_color if obj['conf'] else self.box2d_no_match_color
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)
                txt = obj['label'] if obj['id'] is None else obj['label'] + '-ID: ' + str(obj['id'])
                cv2.putText(img, text=txt, org=(bbox[0], bbox[1] - thickness - 3), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=bbox_color, thickness=thickness)

            bbox3d = obj['bbox3d']
            if bbox3d is not None and obj['score3d'] >= self.box3d_threshold:
                vel = obj['vel']

                if not self.vis_unmatch and obj['conf'] is None:
                    continue

                bbox_color = self.box3d_color if obj['conf'] else self.box3d_no_match_color
                roll_cam, pitch_cam, yaw_cam = None, None, None

                if len(bbox3d) == 7:
                    [cam_box_x, cam_box_y, cam_box_z, w, h, l, yaw_cam] = bbox3d
                else:
                    [cam_box_x, cam_box_y, cam_box_z, w, h, l, roll_cam, pitch_cam, yaw_cam] = bbox3d

                dimension = [w, h, l]
                location = [cam_box_x, cam_box_y, cam_box_z]

                corners_3d_img = transform_box3d_to_image(dimension, location, roll_cam, pitch_cam, yaw_cam, camera_intrinsic, camera_intrinsic_dist, self.loader.multi_camera_lens[camera_name])

                # None means object is behind the camera, and ignore this object.
                if corners_3d_img is not None:
                    corners_3d_img = corners_3d_img.astype(int)
                    # draw lines in the image
                    # p0-p1, p1-p2, p2-p3, p3-p0
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]), (corners_3d_img[1, 0], corners_3d_img[1, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]), (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]), (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]), (corners_3d_img[0, 0], corners_3d_img[0, 1]), color=bbox_color, thickness=thickness)

                    # p4-p5, p5-p6, p6-p7, p7-p4
                    cv2.line(img, (corners_3d_img[4, 0], corners_3d_img[4, 1]), (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[5, 0], corners_3d_img[5, 1]), (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[6, 0], corners_3d_img[6, 1]), (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[7, 0], corners_3d_img[7, 1]), (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)

                    # p0-p4, p1-p5, p2-p6, p3-p7
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]), (corners_3d_img[4, 0], corners_3d_img[4, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]), (corners_3d_img[5, 0], corners_3d_img[5, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[2, 0], corners_3d_img[2, 1]), (corners_3d_img[6, 0], corners_3d_img[6, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[3, 0], corners_3d_img[3, 1]), (corners_3d_img[7, 0], corners_3d_img[7, 1]), color=bbox_color, thickness=thickness)

                    # draw front lines
                    cv2.line(img, (corners_3d_img[0, 0], corners_3d_img[0, 1]), (corners_3d_img[2, 0], corners_3d_img[2, 1]), color=bbox_color, thickness=thickness)
                    cv2.line(img, (corners_3d_img[1, 0], corners_3d_img[1, 1]), (corners_3d_img[3, 0], corners_3d_img[3, 1]), color=bbox_color, thickness=thickness)

                    txt = obj['label'] if obj['id'] is None else obj['label'] + '-ID: ' + str(obj['id'])

                    cv2.putText(img, text=txt, org=(corners_3d_img[4, 0], corners_3d_img[4, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=bbox_color, thickness=thickness)

                corners_3d = transform_box3d_to_point_cloud(dimension, location, roll_cam, pitch_cam, yaw_cam, cam2lidar)
                # BEV
                if obj['conf'] and with_bev:
                    # draw lines
                    res = (self.lidar_fwd_range[1] - self.lidar_fwd_range[0]) / img_h
                    x_bbox2d, y_bbox2d = transform_corners_3d_to_birdseye(corners_3d, res, self.lidar_side_range, self.lidar_fwd_range)
                    cv2.line(img_bev, (x_bbox2d[2], y_bbox2d[2]), (x_bbox2d[3], y_bbox2d[3]), color=self.forward_color, thickness=thickness)
                    cv2.line(img_bev, (x_bbox2d[3], y_bbox2d[3]), (x_bbox2d[7], y_bbox2d[7]), color=bbox_color, thickness=thickness)
                    cv2.line(img_bev, (x_bbox2d[7], y_bbox2d[7]), (x_bbox2d[6], y_bbox2d[6]), color=bbox_color, thickness=thickness)
                    cv2.line(img_bev, (x_bbox2d[6], y_bbox2d[6]), (x_bbox2d[2], y_bbox2d[2]), color=bbox_color, thickness=thickness)

                    center_x = int((int(x_bbox2d[2]) + int(x_bbox2d[3])) / 2)
                    center_y = int((int(y_bbox2d[2]) + int(y_bbox2d[3])) / 2)
                    if vel is None:
                        tgt_point = (int(center_x), int(center_y))
                    else:
                        vel_lidar = cam2lidar[:-1, :-1] @ vel
                        # if rotation_bev:
                        #     vel_lidar = R_matrix @ vel_lidar
                        tgt_point = (int(center_x - vel_lidar[1] / res), int(center_y - vel_lidar[0] / res))
                    cv2.arrowedLine(img_bev, pt1=(center_x, center_y), pt2=tgt_point, color=bbox_color, thickness=thickness, line_type=cv2.LINE_8, shift=0, tipLength=0.1)
                    # cv.putText(img_bev, text=tmp.txt, org=(x_bbox2d[6], y_bbox2d[6] + thickness + 8), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=bbox_color, thickness=thickness)
        if with_bev:
            return cv2.hconcat([img, img_bev])
        else:
            return img


if __name__ == "__main__":
    meta_json = "ad_system_common:s3://sdc_adas/autolabel/task/GAC/autolabel_20240410_768/meta_2024_03_21_02_44_00_gacGtParser.json"

    visualizer = BEVVisualizer(meta_json, save_path='result/vis_debug')
    results = visualizer.process()
