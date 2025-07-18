# Standard Library
import copy
import os

# Import from third library
import cv2
import numpy as np
from tqdm import tqdm

# Import from alt
import open3d
import threading
from alt.adaptor.location_adaptor import LocaltionFileAdaptor
from alt.decoder.base_folder_decoder import BaseFolderDecoder
from alt.files import smart_copy
from alt.sensebee.base.base_process import BaseSenseBeeProcessor
from alt.sensebee.gop.merge_pcd_by_location import MergePcdByLocation
from alt.sensebee.pvb.utils.processor.pre_anno_pre_process import PVBSenseBeeProducer
from alt.utils.file_helper import (
    ads_cli_download_folder,
    dump,
    dump_json_lines,
)
from alt.utils.petrel_helper import global_petrel_helper
from loguru import logger


def uni_camera_timestamps(camera_timestamps):
    # 创建一个空集合用于存放并集
    union_set = None
    # 遍历所有的列表
    for cur_timestamps in camera_timestamps:
        if union_set is None:
            union_set = set(cur_timestamps)
            continue
        union_set = union_set & set(cur_timestamps)
    return list(union_set)


class GOPSenseBeeProducer(PVBSenseBeeProducer):
    load_fusion = False
    
    def __init__(self, meta_path, save_path="sensebee", skip_num=0, reconstruction_num=3) -> None:
        super().__init__(meta_path, save_path, skip_num)

        self.vehicle_id = None
        if "A02-548" in meta_path:
            self.vehicle_id = "A02-548"

        self.readers, self.align_metas = self.prepare_data()

        # 点云拼接重建
        self.reconstruction_num = reconstruction_num

        if skip_num != 0:
            self.static_lidar_reconstruction()

    def dump_key_frame_label_json(self, total_labels):
        # 全份数据记录提前dump
        dump_json_lines(os.path.join(self.save_path, "source_label.jsonl"), total_labels)
        if self.skip_num == 0:
            key_labels = total_labels
        else:
            key_labels = total_labels[1 :: (self.skip_num + 1)]
        logger.info(f"source label len:{len(total_labels)}, key label len:{len(key_labels)}")
        dump_json_lines(os.path.join(self.save_path, "label.jsonl"), key_labels)

    def static_lidar_reconstruction(self):
        # S1.source_lidar_path 里面点云聚合后的结果，dst_save_path, 保证时间戳前后一致
        location_path = self.meta["data_structure"]["location"]["local"]
        try:
            location_path = LocaltionFileAdaptor.update_location(location_path)
        except Exception as e:
            logger.warning(f"replace location fail:{e}, use origin location file")
        source_lidar_path = self.readers["lidar"].path
        dst_save_path = os.path.join(self.save_path, f"lidar_reconstruction_{self.reconstruction_num}")
        os.makedirs(dst_save_path, exist_ok=True)

        merged_pcds = MergePcdByLocation(
            location_path=location_path,
            pcd_root_path=source_lidar_path,
            rm_dynamic=True,
            merged_num=self.reconstruction_num,
            merged_step=1,
        )
        for t, points in merged_pcds.merged_points.items():
            # open3d.io.write_point_cloud(os.path.join(dst_save_path, f'{t}.pcd'), open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points)))
            pcd = open3d.t.geometry.PointCloud()  # 带有强度的点云得这么存
            pcd.point["positions"] = open3d.core.Tensor(points[:, :3], open3d.core.float32)
            pcd.point["intensity"] = open3d.core.Tensor(points[:, 3:], open3d.core.float32)
            open3d.t.io.write_point_cloud(os.path.join(dst_save_path, f"{t}.pcd"), pcd)

        # S2. 替换reader 里面的lidar reader
        self.readers["origin_lidar"] = copy.deepcopy(self.readers["lidar"])
        self.readers["lidar"] = BaseFolderDecoder(path=dst_save_path, ext=".pcd")

    def single_video_process(self, camera_name, video_path, timestamp_path, image_path):
        os.makedirs(os.path.join(image_path, camera_name), exist_ok=True)
        cur_video = cv2.VideoCapture(video_path)
        timestamps = []
        for time_index, item in enumerate(global_petrel_helper.readlines(timestamp_path)):
            timestamp_ns = int(item.split(",")[-1].strip())
            timestamps.append(timestamp_ns)

        frame_index = 0
        while True:
            flag, img = cur_video.read()
            if not flag:
                break

            timestamp = timestamps[frame_index]
            timestamp_ms_str = str(timestamp // 1000 // 1000)
            if not timestamp_ms_str.endswith("99") and self.vehicle_id not in ["A02-548"]:  # 必须走这个
                frame_index += 1
                continue

            cur_save_path = os.path.join(image_path, camera_name, f"{timestamp}.jpg")
            cv2.imwrite(cur_save_path, img)
            frame_index += 1
        assert len(timestamps) == frame_index

    def rebuild_timestamps(self, readers, limit=15):
        # J6E数据需要 重写一下
        camera_readers = {name: single_reader for name, single_reader in readers.items() if name != "lidar"}
        camera_names = list(camera_readers.keys())

        master_timestamps = np.array(camera_readers[camera_names[0]].timestamps)
        auxiliary_timestamps = [np.array(camera_readers[name].timestamps) for name in camera_names[1:]]
        self.multi_timestamps_mapping = {name: {} for name in camera_names}

        align_timestamps = []
        for timestamp in master_timestamps:
            bundle_list = [timestamp]

            for other_timestamps in auxiliary_timestamps:
                timestamp_gaps = np.abs(timestamp - other_timestamps)
                # logger.warning("min timestamp gap: {}".format(timestamp_gaps.min()))
                if timestamp_gaps.min() > limit:
                    continue
                bundle_list.append(other_timestamps[timestamp_gaps.argmin()])

            if len(bundle_list) == len(camera_names):
                align_timestamps.append(bundle_list)

        rebuild_readers = copy.deepcopy(readers)
        for align_timestamp_list in align_timestamps:
            mean_timestamp = int(np.mean(align_timestamp_list))
            # 再去还原剩余的时间戳
            # mean_timestamp = Decimal(float(np.mean(align_timestamp_list)) * 1000)  # us

            for sensor_timestamp, sensor_name in zip(align_timestamp_list, camera_names):
                for source_timestamp in readers[sensor_name].timestamps:
                    if source_timestamp == sensor_timestamp:
                        source_value = copy.deepcopy(readers[sensor_name].get(source_timestamp))
                        rebuild_readers[sensor_name].remove(source_timestamp)
                        rebuild_readers[sensor_name].add(mean_timestamp, source_value)

        return rebuild_readers

    def prepare_data(self):
        self.raw_data = os.path.join(self.save_path, "raw_data")
        os.makedirs(self.raw_data, exist_ok=True)

        self.camera_datas = os.path.join(self.raw_data, "cameras")
        ads_cli_download_folder(os.path.join(self.root, "camera"), self.camera_datas)

        self.camera_metas = {}

        self.image_path = os.path.join(self.save_path, "cameras")
        os.makedirs(self.image_path, exist_ok=True)

        threads = []
        for camera_name, camera_meta in tqdm(self.meta["data_structure"]["camera"].items(), desc="camera_parsing"):
            suffix = camera_meta["video"].split(".")[-1]
            video_path = os.path.join(self.camera_datas, f"{camera_name}.{suffix}")
            timestamp_path = os.path.join(self.camera_datas, f"{camera_name}.txt")
            t = threading.Thread(target=self.single_video_process, args=(camera_name, video_path, timestamp_path, self.image_path))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        readers = {}
        readers["lidar"] = BaseFolderDecoder(self.meta["data_structure"]["lidar"]["car_center"], ext=".pcd")
        assert len(readers["lidar"].timestamps) != 0

        for camera_name in os.listdir(self.image_path):
            readers[camera_name] = BaseFolderDecoder(os.path.join(self.image_path, camera_name), ext=".jpg")

        if self.vehicle_id in ["A02-548"]:
            readers = self.rebuild_timestamps(readers)

        camera_timestamps = [readers[name].timestamps for name in readers if name != "lidar"]   # ms时间戳

        self.camera_uni_timestamps = uni_camera_timestamps(camera_timestamps)
        if len(self.camera_uni_timestamps) == 0:
            raise ValueError("no uni camera timestamps")
        self.camera_uni_timestamps.sort()

        align_metas = []
        for lidar_timestamp in readers["lidar"].timestamps:
            time_gaps = np.abs(np.array(self.camera_uni_timestamps) - lidar_timestamp)
            if time_gaps.min() > 15:  # 不能大于15ms
                continue
            align_metas.append({"lidar": lidar_timestamp, "camera": self.camera_uni_timestamps[time_gaps.argmin()]})

        return readers, align_metas

    def get_timestamps(self, frames):
        timestamps = [int(str(frame["timestamp"])[:13]) for frame in frames]
        return timestamps

    def process(self):
        total_labels = []
        for single_align_meta in tqdm(self.align_metas, desc="sample data..."):
            label_json = {"name": str(single_align_meta["camera"]), "lidar": "", "cameras": []}
            for sensor_name in self.readers:
                if "lidar" in sensor_name:
                    pcd_path, time_gap = self.readers[sensor_name].get_nearst(single_align_meta["lidar"])
                    assert time_gap == 0, time_gap
                    if "s3://" in pcd_path:
                        sensebee_relative_path = os.path.join(sensor_name, pcd_path.rsplit("/")[-1])
                        smart_copy(pcd_path, os.path.join(self.save_path, sensebee_relative_path))
                    else:
                        sensebee_relative_path = pcd_path.replace(self.save_path, "").strip("/")
                    label_json[sensor_name] = sensebee_relative_path
                else:
                    img_path, time_gap = self.readers[sensor_name].get_nearst(single_align_meta["camera"])
                    assert time_gap == 0, time_gap
                    sensebee_relative_path = img_path.replace(self.save_path, "").strip("/")

                    calib = self.get_sensebee_camera_calibration(sensor_name=sensor_name, timestamp=single_align_meta["camera"])

                    os.makedirs(os.path.join(self.save_path, "calib"), exist_ok=True)

                    json_path = os.path.join(self.save_path, "calib", f"{sensor_name}.json")
                    if not os.path.exists(json_path):
                        dump(json_path, calib)

                    label_json["cameras"].append(
                        {"image": sensebee_relative_path, "calib": os.path.join("calib", f"{sensor_name}.json")}
                    )

            total_labels.append(label_json)

        self.dump_key_frame_label_json(total_labels)
        label_timestamps = {int(item["name"]): item for item in total_labels}
        return label_timestamps


class GOPPreAnnoPreProcessor(BaseSenseBeeProcessor):
    def __init__(self, meta_json, save_path, skip_num=0, reconstruction_num=3) -> None:
        self.meta_json = meta_json
        self.save_path = save_path
        self.skip_num = skip_num

        self.raw_data_producer = GOPSenseBeeProducer(
            meta_json, save_path=save_path, skip_num=skip_num, reconstruction_num=reconstruction_num
        )

    def process(self):
        # 后续在这个位置增加预标注等内容，暂时没有，先留空
        return self.raw_data_producer.process()


if __name__ == "__main__":
    meta_json = "ad_system_common_auto:s3://sdc3-gt-label-2/pvbGt/2024_07/2024_07_08/A02-548/autolabel-10313/2024_07_05_15_16_45_L2_A02-548_10313_pvbGt/meta.json"
    GOPPreAnnoPreProcessor(meta_json=meta_json, save_path="/workspace/auto-labeling-tools/result/sensebee/gop_j6e_test").process()
