# Standard Library
import os

# Import from third library
import json
from loguru import logger
import numpy as np

# Import from alt
from alt.files import smart_exists
from alt.utils.file_helper import load, load_autolabel_meta_json
from alt.utils.petrel_helper import global_petrel_helper


class BaseSenseBeeProcessor:
    def __init__(self) -> None:
        pass

    def process(self):
        raise NotImplementedError

    def safe_load_meta_json(self, meta_path):
        meta = load_autolabel_meta_json(meta_path)

        # 补上缺失的雷达路径，但有些真的不存在，补了后面还是会报错
        if (
            meta["data_structure"]["lidar"]["car_center"] is None
            or meta["data_structure"]["lidar"]["car_center"].rsplit("/", 2)[0] == ""
        ):
            camera_paths = list(meta["data_structure"]["camera"].values())
            root = camera_paths[0]["video"].rsplit("/", 2)[0]
            meta["data_structure"]["lidar"]["car_center"] = os.path.join(root, "lidar/top_center_lidar")
        else:
            root = meta["data_structure"]["lidar"]["car_center"].rsplit("/", 2)[0]

        # check location tmp.txt
        raw_location_path = meta["data_structure"]["location"]["local"]
        if raw_location_path == "":
            meta["data_structure"]["location"]["local"] = self.get_location_path(root)

        if meta["data_structure"]["config"] is None:
            meta["data_structure"]["config"] = os.path.join(root, "calib")

        return meta, root

    def get_location_path(self, root):
        for sub_path in ["ved/pose/location.tmp.txt", "ved/localization.tmp.txt", "ved/local_pose.tmp.txt", 'ved/pose/location_bag.tmp.txt']:
            cur_abs_path = os.path.join(root, sub_path)

            if smart_exists(cur_abs_path):
                return cur_abs_path

        raise NotImplementedError

    def generate_mapping_metas(self, max_time_error=150):
        timestamps = list(self.sensbee_labels.keys())
        self.sensbee_labels = {key: self.sensbee_labels[key] for key in timestamps}
        self.gt_targets = {key: self.gt_targets[key] for key in self.gt_targets if key in timestamps}

        self.source_sensbee_labels = {}
        self.source_label_json = self.label_json.replace("label.jsonl", "source_label.jsonl")
        assert global_petrel_helper.check_exists(self.source_label_json), self.source_label_json

        cache_calibs = {}
        for item in global_petrel_helper.readlines(self.source_label_json):
            data = json.loads(item)
            data["path"] = {}

            for idx, cam_meta in enumerate(data["cameras"]):
                calib_path = data["cameras"][idx]["calib"]
                if calib_path not in cache_calibs:
                    cache_calibs[calib_path] = load(os.path.join(self.sensebee_root, calib_path))
                data["cameras"][idx]["calib"] = cache_calibs[calib_path]
                data["path"][cam_meta["calib"]["calName"]] = cam_meta["image"]
            self.source_sensbee_labels[int(data["name"])] = data

        labeling_names = list(set(self.sensbee_labels.keys()) & set(self.gt_targets.keys()))
        unlabeling_names = list(set(self.source_sensbee_labels.keys()) - set(labeling_names))

        logger.info("labeling length: {}".format(len(labeling_names)))
        logger.info("unlabeling length: {}".format(len(unlabeling_names)))

        labeling_names.sort()
        unlabeling_names.sort()

        generate_mapping_metas = []
        for timestamp in unlabeling_names:
            timestamp_gap = np.abs(np.array(labeling_names) - timestamp)
            if (timestamp_gap.min() > max_time_error):  # 这份数据找不到在接受生成的时间范围内，1)可能没有原始数据 2)可能被标注无效直接丢掉
                continue

            src_timestamp = labeling_names[timestamp_gap.argmin()]
            dst_timestamp = timestamp
            generate_mapping_metas.append([src_timestamp, dst_timestamp])

        return generate_mapping_metas
