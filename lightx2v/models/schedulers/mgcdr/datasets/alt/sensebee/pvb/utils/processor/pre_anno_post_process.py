# Standard Library
import copy
import datetime
import os
import zipfile
from multiprocessing import Pool

# Import from third library
import json
import numpy as np
from tqdm import tqdm

# Import from alt
from alt.calibration.calibration import CalibrationAdapter
from alt.decoder import LocationDecoder, RadarDecoder
from alt.files import smart_isdir, smart_listdir, smart_exists
from alt.generator.lidar.dynamic_lidar_generator import DynamicLidarGenerator
from alt.refiner.box_refiner.box_optim import GradientOptimizeBoxes3D
from alt.schema.object import BundleFrameTargets
from alt.sensebee.base.base_process import BaseSenseBeeProcessor
from alt.sensebee.pvb.utils.tracker import BEVTracker
from alt.utils.file_helper import dump_json_lines, load
from alt.utils.load_helper import AutolabelObjectLoader
from alt.utils.petrel_helper import global_petrel_helper
from loguru import logger
from scipy import ndimage


try:
    # pytracking
    from alt.generator.camera.tracker.tamos import TaMOsGenerator
except Exception as e:
    logger.warning(e)
    TaMOsGenerator = None


def trans_fix_calib_to_bee_format(fix_calib_path, calibration):
    for sensor_name in calibration.intrinsic.keys():
        if sensor_name in calibration.intrinsic_new and calibration.intrinsic_new[sensor_name] is not None:
            _camera_intrinsic = calibration.intrinsic_new[sensor_name]
        else:
            _camera_intrinsic = calibration.intrinsic[sensor_name]

        assert _camera_intrinsic is not None, sensor_name
        camera_intrinsic = np.zeros((3, 4))
        camera_intrinsic[:3, :3] = _camera_intrinsic
        camera_extrinsic = calibration.extrinsic[f"{sensor_name}-to-car_center"]

        lidar2camera_rt = np.linalg.inv(camera_extrinsic)

        if "fov195" in sensor_name:
            camera_dist = calibration.intrinsic_dist[sensor_name]
            if len(camera_dist) == 1:  # KB
                cameraType = 2
                fisheyeDistortion = camera_dist[0]
            else:  # Ocam
                cameraType = 3
                fisheyeDistortion = camera_dist[1]

            calib = {
                "calName": sensor_name,
                "cameraType": cameraType,  # 对齐sensebee
                "timestamp": None,
                "P": camera_intrinsic.tolist(),
                "T": lidar2camera_rt[:3, :].tolist(),
                "fisheyeDistortion": fisheyeDistortion.tolist(),
                "SourceFisheyeDistortion": camera_dist.tolist(),
            }
        else:
            calib = {
                "calName": sensor_name,
                "cameraType": 0,  # 对齐sensebee
                "timestamp": None,
                "P": camera_intrinsic.tolist(),
                "R": np.eye(3).tolist(),
                "T": lidar2camera_rt[:3, :].tolist(),
            }
        global_petrel_helper.save_json(f'{fix_calib_path}/bee_format/calib/{sensor_name}.json', calib)
    return f'{fix_calib_path}/bee_format/'


class PVBPreAnnoPostProcessor(BaseSenseBeeProcessor):
    task = "pvb"

    track_parameters = {
        "VEHICLE": {"match_thresh": 0.99, "track_buffer": 20, "mode": "iou"},
        "TRUCK": {"match_thresh": 0.99, "track_buffer": 20, "mode": "iou"},
        "PEDESTRIAN": {"match_thresh": 2.0, "track_buffer": 20, "mode": "center"},
        "CYCLIST": {"match_thresh": 2.0, "track_buffer": 20, "mode": "center"},
    }

    def __init__(
        self, meta_json, sensebee_zip, label_json, sensebee_root, gt_path=None, key_frame_engine=False, projection_optim=False, invalid=True, fix_calib_path = None
    ) -> None:
        self.meta_json = meta_json
        self.metas, self.root = self.safe_load_meta_json(meta_json)

        self.vehicle_id = self.metas["data_structure"]["info"]["vehicle_name"]
        self.case_name = meta_json.split("/")[-2]
        self.invalid = invalid

        self.sensebee_zip = sensebee_zip
        self.label_json = label_json
        self.sensebee_root = sensebee_root[:-1] if sensebee_root.endswith("/") else sensebee_root
        self.gt_path = gt_path
        self.key_frame_engine = key_frame_engine

        self.fix_calib_path = fix_calib_path
        # 临时, 目前存量数据鱼眼去畸变参数不完整
        if fix_calib_path is None:
            self.calibration = CalibrationAdapter(config_path=self.metas["data_structure"]["config"])
        else:
            self.calibration = CalibrationAdapter(config_path=fix_calib_path)
            self.calib_fix_bee_format_path = trans_fix_calib_to_bee_format(fix_calib_path, self.calibration)

        self.location_decoder = self.build_location()
        self.sensbee_labels = self.build_sensebee_labels()

        self.gt_targets = self.build_sensebee_results()

        if projection_optim:
            self.gt_targets = self.refine_gt_targets()

        if key_frame_engine:
            self.sensbee_labels, self.gt_targets = self.generate_gt_targets()

        # 目前单独处理radar
        self.radar_decoders = self.build_radars()

    def build_radars(self):
        logger.info("build radar metas ...")
        radar_root = self.metas["data_structure"]["lidar"]["car_center"].split("lidar")[0] + "radar"
        res = {}
        if not smart_exists(radar_root):
            logger.error(f"randar error: {radar_root} not exist")
            return res

        for radar_name in smart_listdir(radar_root):
            cur_radar_path = os.path.join(radar_root, radar_name)
            if smart_isdir(cur_radar_path):
                res[radar_name] = RadarDecoder(cur_radar_path)
        return res

    def single_refine_target(self, target, calibrations, timestamp, idx):
        loc, dims, yaw = GradientOptimizeBoxes3D.refine(target, calibrations)
        target.set_location(loc.tolist())
        target.set_dimension(dims.tolist())
        target.set_rotation(yaw.item())
        return target, timestamp, idx

    def refine_gt_targets(self):
        self.gt_path = self.gt_path.replace("gt.jsonl", "gradient_optimized_refined_gt.jsonl")

        num_processor = 0
        if num_processor in [0, 1]:
            for timestamp, gt_target in tqdm(self.gt_targets.items(), desc="box_refine ..."):
                for idx, target in enumerate(tqdm(gt_target.bundle_targets_3d, desc=str(timestamp))):
                    # if "BARRICADE" in target.label: # 这个类别不做refine处理
                    #     continue
                    loc, dims, yaw = GradientOptimizeBoxes3D.refine(target, gt_target.calibrations)

                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_location(loc.tolist())
                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_dimension(dims.tolist())
                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_rotation(yaw.item())
        else:
            p = Pool(num_processor)
            res = []
            for timestamp, gt_target in tqdm(self.gt_targets.items(), desc="box refine stage1 ..."):
                for idx, target in enumerate(gt_target.bundle_targets_3d):
                    # if "BARRICADE" in target.label: # 这个类别不做refine处理
                    #     continue
                    res.append(p.apply_async(self.single_refine_target, args=(target, gt_target.calibrations, timestamp, idx)))
            p.close()
            p.join()
            for single_res in tqdm(res, desc="box refine stage2 ..."):
                optim_target, timestamp, idx = single_res.get()
                self.gt_targets[timestamp].bundle_targets_3d[idx] = optim_target
        return self.gt_targets

    def get_calib_path(self, calib_path):
        if self.fix_calib_path is None:
            return os.path.join(self.sensebee_root, calib_path)
        else:
            return os.path.join(self.calib_fix_bee_format_path, calib_path)

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
                    cache_calibs[calib_path] = load(self.get_calib_path(calib_path))
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
            timestamp_gap = np.array(labeling_names) - timestamp

            middle_flag = len(timestamp_gap[timestamp_gap < 0]) and len(timestamp_gap[timestamp_gap > 0])
            if not middle_flag:  # 边界帧被标注了无效
                logger.warning("{} not in middle".format(timestamp))
                continue

            # 找后面最近的一帧
            nearst_behind_gap = timestamp_gap[timestamp_gap > 0].min()
            nearst_before_gap = timestamp_gap[timestamp_gap < 0].max()

            nearst_behind_timestamp = labeling_names[list(timestamp_gap).index(nearst_behind_gap)]
            nearst_before_timestamp = labeling_names[list(timestamp_gap).index(nearst_before_gap)]

            if np.abs(nearst_before_gap) > max_time_error or np.abs(nearst_behind_gap) > max_time_error:
                continue  # 找不到符合范围的有效帧
            generate_mapping_metas.append([nearst_before_timestamp, timestamp, nearst_behind_timestamp])

        return generate_mapping_metas

    def generate_gt_targets(self, max_time_error=150):
        loader = AutolabelObjectLoader(self.meta_json, build_bev_meta=False)
        # S1. 生成lidar的预标注结果, 可以直接在预生产任务里面来读取
        pvb_lidar_dets, pvb_camera_dets = loader.get_lidar_results(), loader.get_camera_results()

        camera_generator = TaMOsGenerator()

        # S2. 基于最大时间误差，生成向前向后的mapping信息
        mapping_metas = self.generate_mapping_metas(max_time_error=max_time_error)

        # S3. 给当前的gt_target 补充ID信息
        track_parameters = {
            "VEHICLE": {"match_thresh": 6, "track_buffer": 20, "mode": "center"},
            "TRUCK": {"match_thresh": 6, "track_buffer": 20, "mode": "center"},
            "PEDESTRIAN": {"match_thresh": 4.0, "track_buffer": 20, "mode": "center"},
            "CYCLIST": {"match_thresh": 4.0, "track_buffer": 20, "mode": "center"},
        }
        self.process_tracking(class_cfg=track_parameters)

        # S4.
        lidar_generator = DynamicLidarGenerator(lidar_res=pvb_lidar_dets, gt_targets=copy.deepcopy(self.gt_targets))

        gene_metas = {}
        for before_timestamp, timestamp, after_timestamp in tqdm(mapping_metas, desc=f"{self.case_name} generate ..."):
            dst_timestamp = timestamp

            before_targets = self.gt_targets[before_timestamp]  # 在这里
            before_targets_2d, src_targets_3d = {}, []

            for target_3d in before_targets.bundle_targets_3d:
                if target_3d.info2d is None:
                    continue
                for target_2d in target_3d.info2d:
                    cur_target_2d = target_2d.generate_format
                    if cur_target_2d["camera_name"] not in before_targets_2d:
                        before_targets_2d[cur_target_2d["camera_name"]] = []

                    before_targets_2d[cur_target_2d["camera_name"]].append(cur_target_2d)

            for target_2d in before_targets.bundle_targets_2d:
                cur_target_2d = target_2d.generate_format
                if cur_target_2d["camera_name"] not in before_targets_2d:
                    before_targets_2d[cur_target_2d["camera_name"]] = []
                before_targets_2d[cur_target_2d["camera_name"]].append(cur_target_2d)

            # 3D层面的生成
            lidar_gene_res = lidar_generator.generate({"before_timestamp": before_timestamp,
                                                       "timestamp": timestamp,
                                                       "after_timestamp": after_timestamp})

            # 2D层面的生成, 按相机重组，避免多次读图
            camera_res = {}
            for camera_name, camera_targets_2d in before_targets_2d.items():
                src_filename = self.source_sensbee_labels[before_timestamp]["path"][camera_name]
                # assert src_filename in camera_targets_2d[0]['filename']
                src_image_path = self.join_image_path(src_filename)
                src_img = global_petrel_helper.imread(src_image_path)

                dst_filename = self.source_sensbee_labels[timestamp]["path"][camera_name]
                dst_image_path = self.join_image_path(dst_filename)
                dst_img = global_petrel_helper.imread(dst_image_path)

                camera_generator.initialize()
                outs = camera_generator.generate({"src_image": src_img, "dst_image": dst_img, "info": camera_targets_2d})
                camera_res[camera_name] = outs

            # 还原到要生成的目标上
            dst_targets = copy.deepcopy(before_targets)  # 属性层面直接继承上一帧
            dst_targets.set_timestamp(timestamp)

            cur_loc = self.location_decoder.retrieve_item(float(timestamp * 1000))[0]
            dst_targets.set_ego2world(LocationDecoder.convert_coordinate(cur_loc, mode="ego2wld"))
            dst_targets.set_ego_vels(cur_loc["data"]["vel"])
            dst_targets.ego2global_transformation_matrix = dst_targets.ego2world @ np.linalg.inv(self.first_ego2world)

            # 把没有关联上的直接remove
            dst_targets.bundle_targets_3d = [_target_ for _target_ in dst_targets.bundle_targets_3d if _target_.id in lidar_gene_res]

            for target_id, target_box_3d in lidar_gene_res.items():
                for _target_idx_, _target_ in enumerate(dst_targets.bundle_targets_3d):
                    if int(_target_.id) == int(target_id):
                        dst_targets.bundle_targets_3d[_target_idx_].set_location(target_box_3d[:3].tolist())
                        dst_targets.bundle_targets_3d[_target_idx_].set_dimension(target_box_3d[3:6].tolist())
                        dst_targets.bundle_targets_3d[_target_idx_].set_rotation(target_box_3d[6].item())

            # 2D
            for idx, target_3d in enumerate(dst_targets.bundle_targets_3d):
                token = target_3d.token

                # 还原对应的2D
                gene_targets_2d = []
                for camera_name, camera_preds in camera_res.items():
                    for cur_token, preds in camera_preds.items():
                        if cur_token == token:
                            __image_path__ = self.source_sensbee_labels[dst_timestamp]["path"][camera_name]
                            filename = __image_path__.replace(self.sensebee_root + "/", "")
                            reset_meta = [preds[0], preds[1], token, camera_name, filename]
                            gene_targets_2d.append(reset_meta)
                dst_targets.bundle_targets_3d[idx].reset_info2d(gene_targets_2d)

            for jdx, target_2d in enumerate(dst_targets.bundle_targets_2d):
                token = target_2d.token
                for camera_name, camera_preds in camera_res.items():
                    for cur_token, preds in camera_preds.items():
                        if cur_token == token:
                            __image_path__ = self.source_sensbee_labels[dst_timestamp]["path"][camera_name]
                            filename = __image_path__.replace(self.sensebee_root + "/", "")
                            reset_meta = [preds[0], preds[1], token, camera_name, filename]
                            dst_targets.bundle_targets_2d[jdx].reset_info2d(reset_meta)
            gene_metas[dst_timestamp] = dst_targets

        logger.info("src_gt_targets length: {}".format(len(self.gt_targets)))
        for timestamp in gene_metas:
            assert timestamp not in self.gt_targets
            self.gt_targets[timestamp] = gene_metas[timestamp]
        self.gt_targets = {k: self.gt_targets[k] for k in sorted(self.gt_targets)}

        logger.info("src_gt_targets length: {}".format(len(self.gt_targets)))
        return self.source_sensbee_labels, self.gt_targets

    def join_image_path(self, filename):
        if self.sensebee_root.split('/')[-1] == filename.split('/')[0]:
            image_path = os.path.join(self.sensebee_root, filename.split('/', 1)[-1])
        else:
            image_path = os.path.join(self.sensebee_root, filename)
        return image_path

    def build_sensebee_labels(self):
        logger.info("build_sensebee_labels ...")
        sensbee_labels = {}
        cache_calibs = {}

        for item in global_petrel_helper.readlines(self.label_json):
            data = json.loads(item)
            data["path"] = {}
            for idx, cam_meta in enumerate(data["cameras"]):
                calib_path = data["cameras"][idx]["calib"]
                if calib_path not in cache_calibs:
                    cache_calibs[calib_path] = load(self.get_calib_path(calib_path))

                data["cameras"][idx]["calib"] = cache_calibs[calib_path]
                data["path"][cache_calibs[calib_path]["calName"]] = cam_meta["image"]
            sensbee_labels[int(data["name"])] = data

        self.manual_labels = copy.deepcopy(sensbee_labels)  # 用于判断是否是关键帧
        return sensbee_labels

    def build_location(self):
        logger.info("build_location ...")
        if self.fix_calib_path is None:
            self.calibration_path = self.metas["data_structure"]["config"]
        else:
            self.calibration_path = self.fix_calib_path
        self.location_path = self.metas["data_structure"]["location"]["local"]
        self.vehicle_name = self.metas["data_structure"]["info"]["vehicle_name"]

        location_decoder = LocationDecoder(
            datapath={"path": self.location_path, "config": self.calibration_path, "vehicle": self.vehicle_name},
            sensor_name="local",
            source_sensor="car_center",
            cal_motion=True,
            intrpl_degree=3,
            cal_rotmat=True,
        )
        return location_decoder

    def sensebee_tmp_wrapper(self, result):
        if not result["valid"]:
            return result

        pure2d_results = result["step_1"]["resultRect"]

        remove_indexes = []
        for idx_3d, target in enumerate(result["step_1"]["result"]):
            if "rects" not in target:
                target["rects"] = []

            if len(target["rects"]) > 0:  # 不为空的也不处理
                continue

            identification = target["id"]  # 是否是用这个唯一的ID？ 对应下面的extId

            for idx_2d, target_2d in enumerate(pure2d_results):
                if target_2d.get("extId", None) == identification:
                    result["step_1"]["result"][idx_3d]["rects"].append(target_2d)
                    remove_indexes.append(idx_2d)

        result["step_1"]["resultRect"] = [pure2d_results[idx] for idx, _ in enumerate(pure2d_results) if idx not in remove_indexes]
        return result

    def build_sensebee_results(self):
        logger.info("build_sensebee_results ...")
        z = zipfile.ZipFile(self.sensebee_zip, "r")
        res = {}

        self.first_ego2world = None

        first_flag = True
        for frame_idx, target in tqdm(enumerate(z.infolist()), desc="parse sensebee results", total=len(z.infolist())):
            if target.filename in ["packagePageInfo.json"]:
                continue
            image_name = target.filename.strip(".json")  # ms timestamp
            timestamp_ms = int(image_name)

            if timestamp_ms not in self.sensbee_labels:
                continue

            cur_gt = json.loads(z.open(target.filename, "r").read())
            cur_gt = self.sensebee_tmp_wrapper(cur_gt)  # 非常临时
            cur_targets = BundleFrameTargets.build_from_sensebee(cur_gt, task=self.task)
            if not cur_targets.valid:
                continue

            cur_loc = self.location_decoder.retrieve_item(float(timestamp_ms * 1000))[0]
            cur_targets.set_ego2world(LocationDecoder.convert_coordinate(cur_loc, mode="ego2wld"))
            cur_targets.set_timestamp(timestamp_ms)
            cur_targets.set_ego_vels(cur_loc["data"]["vel"])
            cur_targets.set_calibration(self.sensbee_labels[timestamp_ms])

            if first_flag:
                # 当前到第一帧位置的变换矩阵，名字比较奇怪
                cur_targets.ego2global_transformation_matrix = np.eye(4)
                self.first_ego2world = cur_targets.ego2world
                first_flag = False
            else:
                cur_targets.ego2global_transformation_matrix = cur_targets.ego2world @ np.linalg.inv(self.first_ego2world)

            res[timestamp_ms] = cur_targets

        return {k: res[k] for k in sorted(res)}

    def process_tracking(self, class_cfg=None):
        # tracking
        if class_cfg is None:
            class_cfg = self.track_parameters

        tracker = BEVTracker(class_cfg=class_cfg)

        mapping_res = {}
        for timestamp_ms, bundle_targets in tqdm(self.gt_targets.items(), desc="tracking"):
            cur_tracklets = tracker(bundle_targets.bundle_targets_3d)

            for tracklet in cur_tracklets:
                token, _timestamp = tracklet.additional_info
                mapping_res[token] = tracklet.track_id

            for idx, item in enumerate(self.gt_targets[timestamp_ms].bundle_targets_3d):
                assert item.token in mapping_res
                self.gt_targets[timestamp_ms].bundle_targets_3d[idx].track_id = mapping_res[item.token]

    def process_speed(self):
        # speed
        total_tracklets = {}
        for timestamp_ms, bundle_targets in tqdm(self.gt_targets.items(), desc="reorg tracklets..."):
            for idx, item in enumerate(self.gt_targets[timestamp_ms].bundle_targets_3d):
                assert item.track_id is not None
                if item.track_id not in total_tracklets:
                    total_tracklets[item.track_id] = []

                total_tracklets[item.track_id].append(item)

        speed_res = {}
        for track_id, tracklets in total_tracklets.items():
            locations = [item.global_location for item in tracklets]
            tokens = [item.token for item in tracklets]
            timestamps = [item.timestamp for item in tracklets]
            labels = [item.label for item in tracklets]

            speeds = []

            if len(locations) <= 1:
                speed_res[tokens[0]] = [0, 0, 0]
            else:
                for idx in range(1, len(locations)):
                    loc_gap = np.array(locations[idx]) - np.array(locations[idx - 1])
                    time_gap = (timestamps[idx] - timestamps[idx - 1]) / 1000
                    assert time_gap > 0
                    enu_coord_speed = loc_gap / time_gap
                    ego_coord_speed = np.linalg.inv(item.ego2world)[:3, :3] @ enu_coord_speed
                    speeds.append(ego_coord_speed.tolist())

                speeds = [item for item in [speeds[0]] + speeds]
                assert len(speeds) == len(locations)

                # vx_list = ndimage.gaussian_filter([item[0] for item in speeds], sigma=5)
                # vy_list = ndimage.gaussian_filter([item[1] for item in speeds], sigma=5)
                # vz_list = ndimage.gaussian_filter([item[2] for item in speeds], sigma=5)

                vx_list = [item[0] for item in speeds]
                vy_list = [item[1] for item in speeds]
                vz_list = [item[2] for item in speeds]

                for token, vx, vy, vz in zip(tokens, vx_list, vy_list, vz_list):
                    speed_res[token] = [vx, vy, vz]

        for timestamp_ms, bundle_targets in tqdm(self.gt_targets.items(), desc="calcu speed"):
            for idx, item in enumerate(self.gt_targets[timestamp_ms].bundle_targets_3d):
                assert item.token in speed_res
                self.gt_targets[timestamp_ms].bundle_targets_3d[idx].vels = speed_res[item.token]

    @property
    def pap_case_name(self):
        def get_time_string(ms_timestamp):
            s_timestamp = ms_timestamp / 1000.0
            dt_object = datetime.datetime.fromtimestamp(s_timestamp)
            return dt_object.strftime("%Y%m%d%H%M%S")

        if not hasattr(self, "cache_case_name") or self.cache_case_name is None:
            timestamps_list = list(self.gt_targets.keys())
            first_timestamp, last_timestamp = timestamps_list[0], timestamps_list[-1]
            self.cache_case_name = "{}_{}-{}-{}".format(
                get_time_string(first_timestamp), get_time_string(last_timestamp), "ShangHai", self.vehicle_id
            )
        return self.cache_case_name

    def build_radar_metas(self, timestamp_ms):
        def remove_prefix(ceph_path):
            return "s3://" + ceph_path.split("s3://")[-1]

        res = {}
        for radar_name in self.radar_decoders:
            data_path, time_gap = self.radar_decoders[radar_name].get_nearst(timestamp_ms)
            timestamp_ns = int(data_path.split("/")[-1].split(".")[0])
            res[radar_name] = {"data_path": remove_prefix(data_path), "timestamp": timestamp_ns}
        return res

    def create_sensor_metas(self, sensors_cameras, cur_labels, timestamp_ms):
        return {
            "cameras": sensors_cameras,
            "lidar": {"car_center": {"data_path": cur_labels["lidar"], "timestamp": timestamp_ms}},
            "radars": self.build_radar_metas(timestamp_ms),
        }

    def process(self):
        self.process_tracking()
        self.process_speed()

        clip_metas = []

        for timestamp_ms, cur_labels in tqdm(self.sensbee_labels.items(), desc="label format"):
            if timestamp_ms in self.gt_targets:  # 有效帧
                bundle_targets = self.gt_targets[timestamp_ms]
                cur_metas = bundle_targets.pap_format()
                cur_metas['valid'] = True
            else:
                if not self.invalid:  # 是否保留无效帧选项，默认invalid is True
                    continue
                cur_loc = self.location_decoder.retrieve_item(float(timestamp_ms * 1000))[0]
                ego2global_transformation_matrix = LocationDecoder.convert_coordinate(cur_loc, mode="ego2wld") @ np.linalg.inv(self.first_ego2world)
                cur_metas = BundleFrameTargets.empty_format(timestamp=timestamp_ms,
                                                            ego_vels=cur_loc["data"]["vel"],
                                                            ego2global_transformation_matrix=ego2global_transformation_matrix)
                cur_metas['valid'] = False

            cur_metas["manual"] = bool(timestamp_ms in self.manual_labels)  # 是否是关键帧的flag

            sensors_cameras = {}
            for camera_meta in cur_labels["cameras"]:
                camera_name = camera_meta["calib"]["calName"]

                lidar2camera_rt = np.eye(4)
                lidar2camera_rt[:3, :] = np.array(camera_meta["calib"]["T"])

                camera_dist = camera_meta["calib"].get("SourceFisheyeDistortion", [])

                if "fov195" in camera_name and len(camera_dist) == 0:
                    camera_dist = self.calibration.intrinsic_dist[camera_name].tolist()

                camera_timestamp = camera_meta["calib"].get("timestamp", timestamp_ms)
                if not str(camera_timestamp).startswith("17"):
                    camera_timestamp = camera_meta["image"].split("/")[-1].split(".")[0]
                sensors_cameras[camera_name] = {
                    "data_path": camera_meta["image"],
                    "timestamp": camera_timestamp,
                    "camera_intrinsic": np.array(camera_meta["calib"]["P"])[:3, :3].tolist(),
                    "camera_dist": camera_dist,
                    "extrinsic": lidar2camera_rt.tolist(),
                }

            cur_metas["sensors"] = self.create_sensor_metas(sensors_cameras, cur_labels, timestamp_ms)

            cur_metas["vehicle_id"] = self.vehicle_id
            cur_metas["case_name"] = self.pap_case_name
            clip_metas.append(cur_metas)

        dump_json_lines(self.gt_path, clip_metas)

        return {
            "root": self.sensebee_root,
            "anno": self.gt_path,
            "case_name": self.case_name,
            "pap_case_name": self.pap_case_name,
            "vehicle_id": self.vehicle_id,
            "frame_num": len(self.gt_targets),
            "meta_json": self.meta_json,
        }


if __name__ == "__main__":
    path = "result/pvb_labels/0521_filter/0521_tmp.json"

    zip_path = "result/all_attrs/22204/22204-202405301401172.zip"
    sensbee_labels = "result/sensebee/0522/10_meta_2024_04_01_00_44_16_gacGtParser/label.jsonl"
    sensebee_root = (
        "ad_system_common:s3://sdc_gt_label/ql_test/sensebee/pvb/pvb_all_attr/0522/10_meta_2024_04_01_00_44_16_gacGtParser"
    )

    for idx, meta_json in enumerate(load(path)):
        if "2024_04_01_00_44_16_gacGtParser" in meta_json:
            print(idx)
            break
    assert "2024_04_01_00_44_16_gacGtParser" in meta_json
    gt_path = "ad_system_common:s3://sdc_gt_label/ql_test/sensebee/pvb/pvb_all_attr/0522/10_meta_2024_04_01_00_44_16_gacGtParser/pvb_gt_20240530.jsonl"

    PVBPreAnnoPostProcessor(meta_json, zip_path, sensbee_labels, sensebee_root, gt_path).process()
