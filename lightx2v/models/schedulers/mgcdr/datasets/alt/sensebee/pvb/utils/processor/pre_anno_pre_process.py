# Standard Library
import os
import random
import string

# Import from third library
import json
import numpy as np
from tqdm import tqdm

# Import from alt
import open3d
from alt import dump, smart_copy, smart_exists, smart_glob
from alt.calibration.calibration import CalibrationAdapter
from alt.schema.object import LabelMapper
from alt.sensebee.base.base_process import BaseSenseBeeProcessor
from alt.utils.file_helper import (
    _prepare_folder,
    dump_json_lines,
    load_autolabel_meta_json,
)
from alt.utils.load_helper import AutolabelObjectLoader
from alt.utils.petrel_helper import global_petrel_helper
from loguru import logger


class PVBSenseBeeProducer(BaseSenseBeeProcessor):
    load_fusion = True

    def __init__(self, meta_path, save_path="sensebee", skip_num=0) -> None:
        self.meta, self.root = self.safe_load_meta_json(meta_path)

        self.data_annotation = self.meta["data_annotation"]
        self.calibration = CalibrationAdapter(config_path=self.meta["data_structure"]["config"])
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.skip_num = skip_num

        self.autolabel_loader = AutolabelObjectLoader(meta_path, build_bev_meta=False, fusion_result=self.load_fusion)

    def remove_prefix(self, ceph_path):
        return "s3://" + ceph_path.split("s3://")[-1]

    def get_frames(self, path):
        frames = global_petrel_helper.load_pk(path)["frames"]
        if path.endswith("car_center-to-car_center#object.pkl"):
            # 是lidar
            name = "lidar"
        else:
            name = path.split("-")[-1].split("#")[0]
        return name, frames

    def get_timestamps(self, frames):
        timestamps = [int(str(frame["timestamp"])[:13]) for frame in frames]
        return timestamps

    def uni_camera_timestamps(self, res):
        # 创建一个空集合用于存放并集
        union_set = None
        # 遍历所有的列表
        for name, frames in res.items():
            if name == "lidar":
                continue
            cur_timestamps = self.get_timestamps(frames)
            if union_set is None:
                union_set = set(cur_timestamps)
                continue
            union_set = union_set & set(cur_timestamps)

        return list(union_set)

    def find_item(self, timestamp, frames, mode="nearest"):
        if mode == "nearest":
            time_errors = []
            for frame in frames:
                time_error = abs(int(str(frame["timestamp"])[:13]) - timestamp)
                time_errors.append(time_error)

            if np.min(time_errors) < 10:
                return frames[np.argmin(time_errors)]
            else:
                return None
        elif mode == "equal":
            for frame in frames:
                if int(str(frame["timestamp"])[:13]) == timestamp:
                    return frame
            raise NotImplementedError
        else:
            raise NotImplementedError

    def get_sensebee_camera_calibration(self, sensor_name, timestamp):
        if sensor_name in self.calibration.intrinsic_new and self.calibration.intrinsic_new[sensor_name] is not None:
            _camera_intrinsic = self.calibration.intrinsic_new[sensor_name]
        else:
            _camera_intrinsic = self.calibration.intrinsic[sensor_name]

        assert _camera_intrinsic is not None, sensor_name
        camera_intrinsic = np.zeros((3, 4))
        camera_intrinsic[:3, :3] = _camera_intrinsic
        camera_extrinsic = self.calibration.extrinsic[f"{sensor_name}-to-car_center"]

        lidar2camera_rt = np.linalg.inv(camera_extrinsic)

        if "fov195" in sensor_name:
            camera_dist = self.calibration.intrinsic_dist[sensor_name]
            if len(camera_dist) == 1:  # KB
                cameraType = 2
                fisheyeDistortion = camera_dist[0]
            else:  # Ocam
                cameraType = 3
                fisheyeDistortion = camera_dist[1]

            calib = {
                "calName": sensor_name,
                "cameraType": cameraType,  # 对齐sensebee
                "timestamp": timestamp,
                "P": camera_intrinsic.tolist(),
                "T": lidar2camera_rt[:3, :].tolist(),
                "fisheyeDistortion": fisheyeDistortion.tolist(),
                "SourceFisheyeDistortion": camera_dist.tolist(),
            }
        else:
            calib = {
                "calName": sensor_name,
                "cameraType": 0,  # 对齐sensebee
                "timestamp": timestamp,
                "P": camera_intrinsic.tolist(),
                "R": np.eye(3).tolist(),
                "T": lidar2camera_rt[:3, :].tolist(),
            }
        return calib

    def process(self, num_limit=10):
        uni_timestamps = self.autolabel_loader.intersect_timstamps

        total_labels = []
        NUM = 0
        for time_index, timestamp in tqdm(enumerate(uni_timestamps), desc="sample data", total=len(uni_timestamps)):
            NUM += 1
            label_json = {"name": str(timestamp), "lidar": "", "cameras": []}

            lidar_path, time_diff = self.autolabel_loader._find_latest_lidar_file(timestamp, return_time_error=True, raw_data=True)
            if time_diff > 15:
                continue

            sensebee_lidar_relative_path = os.path.join("lidar", lidar_path.split("/")[-1])
            smart_copy(lidar_path, os.path.join(self.save_path, sensebee_lidar_relative_path))
            label_json["lidar"] = sensebee_lidar_relative_path

            for camera_name, frame in self.autolabel_loader.multi_timestamp_frames[timestamp].items():
                sensor_type = "cameras"

                image_path = frame["filename"]
                assert smart_exists(image_path), image_path

                sensebee_relative_path = os.path.join(
                    sensor_type, camera_name, image_path.split("/")[-1].replace(".png", ".jpg")
                )
                camera_img = global_petrel_helper.imread(image_path)

                camera_img_path = os.path.join(self.save_path, sensebee_relative_path)
                _prepare_folder(camera_img_path)
                global_petrel_helper.imwrite(camera_img_path, camera_img)

                calib = self.get_sensebee_camera_calibration(sensor_name=camera_name, timestamp=timestamp)
                os.makedirs(os.path.join(self.save_path, "calib"), exist_ok=True)

                json_path = os.path.join(self.save_path, "calib", f"{camera_name}.json")

                if not os.path.exists(json_path):
                    json.dump(calib, open(json_path, "w"))

                label_json["cameras"].append(
                    {"image": sensebee_relative_path, "calib": os.path.join("calib", f"{camera_name}.json")}
                )

            total_labels.append(label_json)

        self.dump_key_frame_label_json(total_labels)

        label_timestamps = {int(item["name"]): item for item in total_labels}
        return label_timestamps

    def dump_key_frame_label_json(self, total_labels):
        # 全份数据记录提前dump
        dump_json_lines(os.path.join(self.save_path, "source_label.jsonl"), total_labels)

        key_labels = total_labels[:: (self.skip_num + 1)]
        if total_labels[-1] not in key_labels:
            key_labels.append(total_labels[-1])

        dump_json_lines(os.path.join(self.save_path, "label.jsonl"), key_labels)


class PVBPreAnnoPreProcessor(BaseSenseBeeProcessor):
    def __init__(self, meta_json, save_path, skip_num=0, **kwargs) -> None:
        self.meta_json = meta_json
        self.save_path = save_path
        self.skip_num = skip_num

        self.raw_data_producer = PVBSenseBeeProducer(meta_json, save_path=save_path, skip_num=skip_num)

        # self.autolabel_loader = AutolabelObjectLoader(meta_json, build_bev_meta=False)
        self.autolabel_loader = self.raw_data_producer.autolabel_loader

    def process(self):
        # S1. 生成裸数据, ms时间戳 int值
        self.label_timestamps = self.raw_data_producer.process()

        for timestamp in self.label_timestamps:
            cur_bev_metas = self.autolabel_loader.get_bev_metas(timestamp)
            sensebee_result = [
                self.sensebee_target3d_format(target, self.label_timestamps[timestamp]) for target in cur_bev_metas["fusion"]
            ]

            if sensebee_result:
                sensebee_resultRect = self.sensebee_target2d_format(cur_bev_metas["2d"], self.label_timestamps[timestamp])
            else:
                sensebee_resultRect = []

            timestamp_result = self.sensebee_format(result=sensebee_result, resultRect=sensebee_resultRect)
            dump(os.path.join(self.save_path, "pre_sensebee", "{}.json".format(timestamp)), timestamp_result)

    def sensebee_format(self, result, resultRect, toolName="pointCloudWebTool"):
        return {
            "step_1": {
                "dataSourceStep": 0,
                "result": result,
                "resultRect": resultRect,
                "resultLine": [],
                "resultPoint": [],
                "resultPolygon": [],
                "segmentation": [],
                "toolName": toolName,
            },
            "valid": True,
        }

    def find_path(self, camera_name, timestamp_labels):
        for meta_path in timestamp_labels["cameras"]:
            if camera_name in meta_path["image"]:
                return meta_path["image"]
        raise NotImplementedError

    def sensebee_target2d_format(self, targets, timestamp_labels, only_one=False):
        def find_path(camera_name):
            for meta_path in timestamp_labels["cameras"]:
                if camera_name in meta_path["image"]:
                    return meta_path["image"]
            raise NotImplementedError

        rects = []
        caches = []
        for camera_name, camera_metas in targets.items():
            for obj in camera_metas:
                if only_one and camera_name in caches:
                    continue

                if obj["bbox2d"] is None:
                    continue

                x1, y1, x2, y2 = obj["bbox2d"].tolist()

                rects.append({"height": y2 - y1, "imageName": find_path(camera_name), "width": x2 - x1, "x": x1, "y": y1})
                caches.append(camera_name)
        return rects

    def sensebee_target3d_format(self, target, timestamp_labels):
        location_x, location_y, location_z = target["location"]
        sensebee_width, sensebee_depth, sensebee_height = target["dimension"]

        targets_2d = {}
        for cam_meta in target["camera_metas"]:
            if cam_meta["camera_name"] not in targets_2d:
                targets_2d[cam_meta["camera_name"]] = []
            targets_2d[cam_meta["camera_name"]].append(cam_meta)
        rects = self.sensebee_target2d_format(targets_2d, timestamp_labels, only_one=True)

        return {
            "attribute": LabelMapper.E2C_LABELS[target["label"]],
            "center": {"x": location_x, "y": location_y, "z": location_z},
            # "count": 1369,
            "depth": sensebee_depth,
            "height": sensebee_height,
            "id": self.generate_random_string(),
            "rects": rects,
            "rotation": target["yaw"],
            "subAttribute": {
                "三轮车是否有人": "否",
                "事故车属性": "否",
                "自车道压线": "否",
                "自车道归属": "否",
                "车尾门(后备箱)开合": "否",
                "车灯": "关闭",
                "车舱门开启方位": "关闭",
                "遮挡": "是",
            },
            "trackID": int(target["id"]),
            "valid": True,
            "width": sensebee_width,
        }

    def generate_random_string(self, length=8):
        # 定义字符集，包括大小写字母和数字
        characters = string.ascii_letters + string.digits
        # 生成随机字符串
        random_string = "".join(random.choice(characters) for _ in range(length))
        return random_string


if __name__ == "__main__":
    # Import from alt
    from alt.utils.env_helper import env

    input_path = "result/pvb_labels/0603/input.tmp.txt"
    for idx, meta_json in enumerate(open(input_path).readlines()):
        # if not env.is_my_showtime(idx):
        #     continue
        meta_json = "ad_system_common:" + meta_json.strip()
        came_name = meta_json.split("/")[-2]
        save_path = "result/pvb_labels/0603/pre_label/{}".format(came_name)

        PVBPreAnnoPreProcessor(meta_json, save_path).process()
