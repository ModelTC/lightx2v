# Standard Library
import os

# Import from third library
import numpy as np
import torch
from collections import OrderedDict

# Import from alt
from alt import smart_exists, smart_listdir
from alt.calibration.calibration import CalibrationAdapter
from alt.utils.coordinate_helper import transform_box3d_to_point_cloud
from alt.utils.file_helper import load_autolabel_meta_json, load
from alt.utils.petrel_helper import global_petrel_helper
from easydict import EasyDict as edict
from loguru import logger
from decimal import Decimal
# from mmcv.ops import box_iou_rotated, nms_rotated
from scipy.spatial.transform import Rotation


class AutolabelObjectLoader(object):
    def __init__(self, meta_json=None, gt_path=None, build_bev_meta=True, fusion_result=True) -> None:
        self.gt_path, self.meta_json = gt_path, meta_json
        self.fusion_result = fusion_result

        if meta_json is None:
            self.meta_json = gt_path.strip("/").replace("gt_labels", "meta.json")
            assert smart_exists(meta_json), meta_json

        self.build_from_meta_json(build_bev_meta)

    def reorg_frames_by_timestamps(self):
        multi_timestamp_frames = {}
        for timestamp in self.intersect_timstamps:
            assert timestamp not in multi_timestamp_frames, timestamp
            cur_metas = {name: self.select_frame_by_timestamp_and_camera_name(name, timestamp) for name in self.camera_names}
            multi_timestamp_frames[timestamp] = cur_metas

        return multi_timestamp_frames

    def build_from_meta_json(self, build_bev_meta):
        self.meta = load_autolabel_meta_json(self.meta_json)
        self.data_annotation = self.meta["data_annotation"]
        self.data_annotation_dir = self.meta["data_annotation"]

        self.camera_names = self._get_camera_names()
        self.lidar_root = self.meta['data_structure']['lidar']['car_center']

        self.lidar_name = sorted(self.meta["data_structure"]["lidar"].keys())[0]
        self.lidar_path = self.meta['data_structure']['lidar'][self.lidar_name]
        self.multi_gt_frames, self.multi_camera_lens, self.intersect_timstamps = self._load_multi_sensor_frames()
        self.calibration = CalibrationAdapter(config_path=self.meta["data_structure"]["config"])

        self.multi_timestamp_frames = self.reorg_frames_by_timestamps()
        logger.info("build multi_timestamp_bev_metas ...")

        self.lidar_timestamp_map = self._get_lidar_map()

        if build_bev_meta:
            self.multi_timestamp_bev_metas = {
                timestamp: self.get_bev_metas(timestamp)["fusion"] for timestamp in self.intersect_timstamps
            }
            logger.info("build multi_trackid_bev_metas ...")
            self.multi_trackid_bev_metas = self.reorg_bev_metas_by_trackid(self.multi_timestamp_bev_metas)

    def _get_lidar_map(self):
        lidar_timestamp_map = {}
        sensor_dir = os.path.join(self.data_annotation_dir, 'cache', 'sensor')

        for sensor_name in smart_listdir(sensor_dir):
            if self.lidar_name not in sensor_name:
                continue

            lidar_dir = os.path.join(sensor_dir, sensor_name)
            lidar_files = smart_listdir(lidar_dir)

            for file in sorted(lidar_files):
                lidar_suffix = '.bin'
                assert file.endswith(lidar_suffix)
                timestamp_ms = self.get_ms_timestamp(file[: -len(lidar_suffix)])
                lidar_timestamp_map[timestamp_ms] = os.path.join(lidar_dir, file)

        # 如果cache里面没有雷达数据，尝试从产线读取
        if not lidar_timestamp_map:
            lidar_files = smart_listdir(self.lidar_path)
            for file in sorted(lidar_files):
                lidar_suffix = '.pcd'
                assert file.endswith(lidar_suffix)
                timestamp_ms = self.get_ms_timestamp(file[: -len(lidar_suffix)])
                lidar_timestamp_map[timestamp_ms] = os.path.join(self.lidar_path, file)

        return lidar_timestamp_map

    def _find_latest_lidar_file(self, timestamp, return_time_error=False, raw_data=False):
        lidar_timstamps = np.array(list(self.lidar_timestamp_map.keys()))

        idx = (np.abs(lidar_timstamps - timestamp)).argmin()
        lidar_timestamp = lidar_timstamps[idx]
        time_diff = abs(float(lidar_timestamp) - float(timestamp))
        # logger.warning('lidar timediff: {}'.format(time_diff))
        if time_diff > 15:
            logger.warning(
                f'The time difference between point cloud ({lidar_timestamp}) and image ({timestamp}) '
                f'exceeds {time_diff} ms.'
            )

        lidar_path = self.lidar_timestamp_map[lidar_timestamp]

        if raw_data:
            filename = lidar_path.split('/')[-1].strip('.bin').replace('.', '') + '.pcd'  # 很怪很怪
            lidar_path = os.path.join(self.lidar_root, filename)
            assert smart_exists(lidar_path)

        if return_time_error:
            return lidar_path, time_diff
        return self.lidar_timestamp_map[lidar_timestamp]

    def reorg_bev_metas_by_trackid(self, multi_timestamp_bev_metas):
        outs = {}
        for _, targets in multi_timestamp_bev_metas.items():
            for target in targets:
                if target["id"] not in outs:
                    outs[target["id"]] = []
                outs[target["id"]].append(target)
        return outs

    def _get_image_len(self, camera_name=None):
        camera_name = self.camera_name if camera_name is None else camera_name
        camera_lens = None
        if "lens" in self.meta["data_structure"]["camera"][camera_name]:
            camera_lens = self.meta["data_structure"]["camera"][camera_name]["lens"]
        return camera_lens

    def _load_gt_frames(self, camera_name):
        if self.fusion_result:
            fusion_path = os.path.join(self.data_annotation_dir, f"{self.lidar_name}-to-{camera_name}#object.pkl")
        else:
            fusion_path = os.path.join(self.data_annotation_dir, f"{camera_name}#object.pkl")
        fusion_info = global_petrel_helper.load_pk(fusion_path)
        return fusion_info["frames"]

    def _get_camera_names(self):
        camera_names = []
        for k, v in self.meta["data_structure"]["camera"].items():
            if v["video"]:
                camera_names.append(k)
        return sorted(camera_names)

    def rebuild_timestamps(self, multi_gt_frames):
        multi_device_timestamps = [np.array(self.get_frame_timestamps(frames)) for name, frames in multi_gt_frames.items()]

        master_timestamps = multi_device_timestamps[0]
        auxiliary_timestamps = multi_device_timestamps[1:]

        self.multi_timestamps_mapping = {name: {} for name in multi_gt_frames.keys()}

        align_timestamps = []
        for timestamp in master_timestamps:
            bundle_list = [timestamp]

            for other_timestamps in auxiliary_timestamps:
                timestamp_gaps = np.abs(timestamp - other_timestamps)
                # logger.warning("min timestamp gap: {}".format(timestamp_gaps.min()))
                if timestamp_gaps.min() > 15:
                    continue
                bundle_list.append(other_timestamps[timestamp_gaps.argmin()])

            if len(bundle_list) == len(multi_gt_frames):
                align_timestamps.append(bundle_list)

        for align_timestamp_list in align_timestamps:
            # mean_timestamp = int(np.mean(align_timestamp_list) * 1000 * 1000)
            # 再去还原剩余的时间戳
            mean_timestamp = Decimal(float(np.mean(align_timestamp_list)) * 1000)  # us

            for sensor_timestamp, sensor_name in zip(align_timestamp_list, multi_gt_frames.keys()):
                for idx, frame in enumerate(multi_gt_frames[sensor_name]):
                    if self.get_ms_timestamp(frame["timestamp"]) == sensor_timestamp:
                        self.multi_timestamps_mapping[sensor_name][self.get_ms_timestamp(mean_timestamp)] = frame["timestamp"]
                        multi_gt_frames[sensor_name][idx]["timestamp"] = mean_timestamp

        return multi_gt_frames

    def _load_multi_sensor_frames(self):
        logger.info("_load_multi_sensor_frames_information ...")
        multi_gt_frames, multi_camera_lens = {}, {}
        for camera_name in self.camera_names:
            multi_camera_lens[camera_name] = self._get_image_len(camera_name=camera_name)
            multi_gt_frames[camera_name] = self._load_gt_frames(camera_name=camera_name)

        # 在这里加个offset的硬修改，然后备注每份相机的整体时间戳差异项
        multi_gt_frames = self.rebuild_timestamps(multi_gt_frames)

        # 并非每个时间戳都有所有相机的数据，取交集
        intersect_timstamps = self.get_intersect_camera_timestamps(multi_gt_frames)
        intersect_timstamps.sort()

        intersect_multi_gt_frames = {}
        frame_nums = {}
        for camera_name in self.camera_names:
            intersect_multi_gt_frames[camera_name] = []
            for frame in multi_gt_frames[camera_name]:
                if self.get_ms_timestamp(frame["timestamp"]) in intersect_timstamps:
                    intersect_multi_gt_frames[camera_name].append(frame)
            frame_nums[camera_name] = len(intersect_multi_gt_frames[camera_name])

        assert len(set(list(frame_nums.values()))) == 1, frame_nums
        return intersect_multi_gt_frames, multi_camera_lens, intersect_timstamps

    def select_frame_by_timestamp_and_camera_name(self, camera_name, timestamp):
        for frame in self.multi_gt_frames[camera_name]:
            if self.get_ms_timestamp(frame["timestamp"]) == timestamp:
                return frame
        raise NotImplementedError

    def get_ms_timestamp(self, μs_timestamp):
        return int(str(μs_timestamp)[:13])

    def get_frame_timestamps(self, frames):
        timestamps = [self.get_ms_timestamp(frame["timestamp"]) for frame in frames]
        return timestamps

    def get_intersect_camera_timestamps(self, res):
        # 创建一个空集合用于存放并集
        intersect_set = None
        # 遍历所有的列表
        for name, frames in res.items():
            cur_timestamps = self.get_frame_timestamps(frames)
            if intersect_set is None:
                intersect_set = set(cur_timestamps)
                continue
            intersect_set = intersect_set & set(cur_timestamps)
        return list(intersect_set)

    def get_bev_metas(self, timestamp, unmatch_target=False):
        # 非常临时的办法
        frames = self.multi_timestamp_frames[timestamp]

        outputs = []  # 融合结果
        outputs_3d = []  # 纯lidar框
        outputs_2d = {}  # 纯2D框
        for camera_name, gt_frame in frames.items():
            outputs_2d[camera_name] = []
            cam2lidar, camera_intrinsic, camera_intrinsic_dist = self.get_camera_parameters(camera_name)
            for obj in gt_frame["objects"]:
                bbox3d = obj["bbox3d"]
                vel = obj["vel"]
                bbox2d = obj["bbox2d"]

                if bbox3d is None:
                    outputs_2d[camera_name].append(obj)
                else:
                    if len(bbox3d) == 7:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, yaw_cam] = bbox3d
                        corners_3d = transform_box3d_to_point_cloud([w, h, l], [cam_box_x, cam_box_y, cam_box_z], None, None, yaw_cam, cam2lidar)
                    else:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, rot1, rot2, rot3] = bbox3d
                        corners_3d = transform_box3d_to_point_cloud([w, h, l], [cam_box_x, cam_box_y, cam_box_z], rot1, rot2, rot3, cam2lidar)
                        yaw_cam = Rotation.from_rotvec([rot1, rot2, rot3]).as_euler("zxy", degrees=False)[2]

                    if vel is not None:
                        vel = cam2lidar[:-1, :-1] @ vel

                    point_cam = np.array([[cam_box_x, cam_box_y, cam_box_z, 1]]).transpose()
                    x_lidar, y_lidar, z_lidar = np.array(cam2lidar @ point_cam)[:3, 0]

                    lidar2cam = np.linalg.inv(cam2lidar)
                    yaw_lidar = np.pi / 2 - np.arctan2(lidar2cam[2, 0], lidar2cam[0, 0]) - yaw_cam
                    yaw_lidar = np.mod(yaw_lidar + np.pi, 2 * np.pi) - np.pi

                    obj.update(
                        {
                            "bev_corners_3d": corners_3d,
                            "location": [x_lidar, y_lidar, z_lidar],
                            "dimension": [l, h, w],
                            "length": l,
                            "width": w,
                            "height": h,
                            "yaw": yaw_lidar,
                            "bev_vel": vel,
                            "timestamp": gt_frame["timestamp"],
                            "filename": gt_frame["filename"],
                            "camera_name": camera_name,
                        }
                    )

                    if bbox2d is None:  # 2D框都没有
                        outputs_3d.append(edict(obj))
                    else:
                        outputs.append(edict(obj))

        keep_outputs = self.fusion(outputs)
        return {"fusion": keep_outputs, "2d": outputs_2d, "3d": None}

    def fusion(self, outputs, iou_threshold=0.15):
        if len(outputs) == 0:
            return []

        bev_bboxes_3d, scores, labels = [], [], []
        for item in outputs:
            bev_bboxes_3d.append([item.location[0], item.location[1], item.length, item.width, item.yaw])
            scores.append(item.score3d)
            labels.append(1.0)  # 按照同一个label 来处理

        bev_ious = box_iou_rotated(torch.Tensor(bev_bboxes_3d), torch.Tensor(bev_bboxes_3d))
        _, keep_inds = nms_rotated(torch.Tensor(bev_bboxes_3d), torch.Tensor(scores), iou_threshold=iou_threshold, labels=torch.Tensor(labels))

        keep_outputs = []
        _debug_num_ = 0
        cache_indexes = []
        for box_idx in keep_inds.tolist():
            mapping_indexs = torch.nonzero(bev_ious[box_idx] > iou_threshold).reshape(-1).tolist()
            _debug_num_ += len(mapping_indexs)
            cur_target = {
                "bev_corners_3d": outputs[box_idx]["bev_corners_3d"],
                "location": outputs[box_idx]["location"],
                "dimension": outputs[box_idx]["dimension"],
                "length": outputs[box_idx]["length"],
                "width": outputs[box_idx]["width"],
                "height": outputs[box_idx]["height"],
                "bev_vel": outputs[box_idx]["bev_vel"],
                "id": outputs[box_idx]["id"],
                "yaw": outputs[box_idx]["yaw"],  # TODO
                "label": outputs[box_idx]["label"],
                "camera_metas": [outputs[item_idx] for item_idx in mapping_indexs if item_idx not in cache_indexes],
            }
            keep_outputs.append(edict(cur_target))
            cache_indexes.extend(mapping_indexs)
        return keep_outputs

    def get_camera_parameters(self, camera_name):
        cam2lidar = None
        if self.lidar_name == "car_center":
            cam2lidar = self.calibration.extrinsic[f"{camera_name}-to-car_center"]
        else:
            cam2car = self.calibration.extrinsic[f"{camera_name}-to-car_center"]
            lidar2car = self.calibration.extrinsic[f"{self.lidar_name}-to-car_center"]
            if isinstance(cam2car, np.ndarray) and isinstance(lidar2car, np.ndarray):
                cam2lidar = np.dot(np.linalg.inv(lidar2car), cam2car)
        if not isinstance(cam2lidar, np.ndarray):
            cam2lidar = self.calibration.extrinsic[f"{camera_name}-to-{self.lidar_name}"]
        if not isinstance(cam2lidar, np.ndarray):
            cam2lidar = np.linalg.inv(self.calibration.extrinsic[f"{self.lidar_name}-to-{camera_name}"])
        camera_intrinsic = self.calibration.intrinsic_new[camera_name]
        if camera_intrinsic is None:
            camera_intrinsic = self.calibration.intrinsic[camera_name]
        camera_intrinsic_dist = self.calibration.intrinsic_dist[camera_name]

        return cam2lidar, camera_intrinsic, camera_intrinsic_dist

    def get_lidar_results(self):
        lidar_res_path = os.path.join(self.data_annotation, 'car_center-to-car_center#object.pkl')
        lidar_res = load(lidar_res_path)

        outs = OrderedDict()
        for single_res in lidar_res['frames']:
            timestamp = self.get_ms_timestamp(single_res['timestamp'])
            outs[timestamp] = single_res['objects']

        return outs

    def get_camera_results(self):
        outs = OrderedDict()
        for camera_name, camera_gts in self.multi_gt_frames.items():
            outs[camera_name] = OrderedDict()
            for single_gt in camera_gts:
                timestamp = self.get_ms_timestamp(single_gt['timestamp'])
                outs[camera_name][timestamp] = single_gt["objects"]
        return outs
