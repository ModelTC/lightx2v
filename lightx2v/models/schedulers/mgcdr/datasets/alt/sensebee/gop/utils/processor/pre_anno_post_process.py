import os
import re
import json
import copy
import numpy as np
from tqdm import tqdm
from loguru import logger
from alt import load
from alt.sensebee.pvb.utils.processor.pre_anno_post_process import PVBPreAnnoPostProcessor
from alt.utils.petrel_helper import global_petrel_helper
from alt.refiner.box_refiner.box_optim import GradientOptimizeBoxes3D
from multiprocessing import Pool

try:
    # pytracking
    from alt.generator.camera.tracker.tamos import TaMOsGenerator
    from alt.generator.lidar.static_lidar_generator import StaticLidarGenerator
except Exception as e:
    logger.warning(e)
    TaMOsGenerator, StaticLidarGenerator = None, None


class GOPPreAnnoPostProcessor(PVBPreAnnoPostProcessor):
    task = 'gop'
    track_parameters = {'GOP': {"match_thresh": 1.0, "track_buffer": 20, 'mode': 'center'}}

    def __init__(self, meta_json, sensebee_zip, label_json, sensebee_root, gt_path=None, key_frame_engine=False, projection_optim=False, fix_calib_path=None) -> None:
        super().__init__(meta_json, sensebee_zip, label_json, sensebee_root, gt_path, key_frame_engine, projection_optim, fix_calib_path=fix_calib_path)

    def process(self):
        return super().process()

    def refine_gt_targets(self):
        self.gt_path = self.gt_path.replace('gt.jsonl', 'gradient_optimized_refined_gt.jsonl')

        num_processor = 0
        if num_processor in [0, 1]:
            for timestamp, gt_target in tqdm(self.gt_targets.items(), desc='box_refine ...'):
                for idx, target in enumerate(gt_target.bundle_targets_3d):
                    if "BARRICADE" in target.label:  # 这个类别不做refine处理
                        continue
                    loc, dims, yaw = GradientOptimizeBoxes3D.refine(target, gt_target.calibrations)

                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_location(loc.tolist())
                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_dimension(dims.tolist())
                    self.gt_targets[timestamp].bundle_targets_3d[idx].set_rotation(yaw.item())
        else:
            p = Pool(num_processor)
            res = []
            for timestamp, gt_target in self.gt_targets.items():
                for idx, target in enumerate(gt_target.bundle_targets_3d):
                    if "BARRICADE" in target.label:  # 这个类别不做refine处理
                        continue
                    res.append(p.apply_async(self.single_refine_target, args=(target, gt_target.calibrations, timestamp, idx)))
            p.close()
            p.join()
            for single_res in tqdm(res, desc='box refine ...'):
                optim_target, timestamp, idx = single_res.get()
                self.gt_targets[timestamp].bundle_targets_3d[idx] = optim_target
        return self.gt_targets

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
            timestamp_gap = np.abs(np.array(labeling_names) - timestamp)
            if (timestamp_gap.min() > max_time_error):  # 这份数据找不到在接受生成的时间范围内，1)可能没有原始数据 2)可能被标注无效直接丢掉
                continue

            src_timestamp = labeling_names[timestamp_gap.argmin()]
            dst_timestamp = timestamp
            generate_mapping_metas.append([src_timestamp, dst_timestamp])

        return generate_mapping_metas

    def generate_gt_targets(self, max_time_error=150):
        mapping_metas = self.generate_mapping_metas(max_time_error=max_time_error)
        camera_generator = TaMOsGenerator()
        lidar_generator = StaticLidarGenerator(location_path=self.location_path)

        gene_metas = {}
        for src_timestamp, dst_timestamp in tqdm(mapping_metas, desc=f'{self.case_name} generate ...'):
            src_targets = self.gt_targets[src_timestamp]
            src_targets_2d, src_targets_3d = {}, []

            for target_3d in src_targets.bundle_targets_3d:
                src_targets_3d.append({'token': target_3d.token,
                                       'location': target_3d.location,
                                       'length': target_3d.length,
                                       'width': target_3d.width,
                                       'height': target_3d.height,
                                       'yaw': target_3d.yaw})

                if target_3d.info2d is None:
                    continue
                for target_2d in target_3d.info2d:
                    cur_target_2d = target_2d.generate_format
                    if cur_target_2d['camera_name'] not in src_targets_2d:
                        src_targets_2d[cur_target_2d['camera_name']] = []

                    src_targets_2d[cur_target_2d['camera_name']].append(cur_target_2d)

            for target_2d in src_targets.bundle_targets_2d:
                cur_target_2d = target_2d.generate_format
                if cur_target_2d['camera_name'] not in src_targets_2d:
                    src_targets_2d[cur_target_2d['camera_name']] = []
                src_targets_2d[cur_target_2d['camera_name']].append(cur_target_2d)

            # 3D层面的生成
            lidar_gene_res = lidar_generator.generate({'targets': src_targets_3d, 'src_timestamp': src_timestamp, 'dst_timestamp': dst_timestamp})
            if lidar_gene_res is None:
                continue

            # 2D层面的生成, 按相机重组，避免多次读图
            camera_res = {}
            for camera_name, camera_targets_2d in src_targets_2d.items():
                src_filename = self.source_sensbee_labels[src_timestamp]['path'][camera_name]
                # assert src_filename in camera_targets_2d[0]['filename']
                src_image_path = self.join_image_path(src_filename)
                src_img = global_petrel_helper.imread(src_image_path)

                dst_filename = self.source_sensbee_labels[dst_timestamp]['path'][camera_name]
                dst_image_path = self.join_image_path(dst_filename)
                dst_img = global_petrel_helper.imread(dst_image_path)

                camera_generator.initialize()
                outs = camera_generator.generate({'src_image': src_img, 'dst_image': dst_img, 'info': camera_targets_2d})
                camera_res[camera_name] = outs

            # 还原到要生成的目标上
            dst_targets = copy.deepcopy(src_targets)
            dst_targets.set_timestamp(dst_timestamp)

            # 还原3D + 2D
            for idx, target_3d in enumerate(dst_targets.bundle_targets_3d):
                token = target_3d.token

                # 还原对应的3D
                for single_lidar_target in lidar_gene_res:
                    if single_lidar_target['token'] == token:
                        dst_targets.bundle_targets_3d[idx].reset_info3d(single_lidar_target)

                # 还原对应的2D
                gene_targets_2d = []
                for camera_name, camera_preds in camera_res.items():
                    for cur_token, preds in camera_preds.items():
                        if cur_token == token:
                            __image_path__ = self.source_sensbee_labels[dst_timestamp]['path'][camera_name]
                            filename = __image_path__.replace(self.sensebee_root + '/', '')
                            reset_meta = [preds[0], preds[1], token, camera_name, filename]
                            gene_targets_2d.append(reset_meta)
                dst_targets.bundle_targets_3d[idx].reset_info2d(gene_targets_2d)

            for jdx, target_2d in enumerate(dst_targets.bundle_targets_2d):
                token = target_2d.token
                for camera_name, camera_preds in camera_res.items():
                    for cur_token, preds in camera_preds.items():
                        if cur_token == token:
                            __image_path__ = self.source_sensbee_labels[dst_timestamp]['path'][camera_name]
                            filename = __image_path__.replace(self.sensebee_root + '/', '')
                            reset_meta = [preds[0], preds[1], token, camera_name, filename]
                            dst_targets.bundle_targets_2d[jdx].reset_info2d(reset_meta)
            gene_metas[dst_timestamp] = dst_targets

        logger.info('src_gt_targets length: {}'.format(len(self.gt_targets)))
        for timestamp in gene_metas:
            assert timestamp not in self.gt_targets
            self.gt_targets[timestamp] = gene_metas[timestamp]
        self.gt_targets = {k: self.gt_targets[k] for k in sorted(self.gt_targets)}

        logger.info('src_gt_targets length: {}'.format(len(self.gt_targets)))
        return self.source_sensbee_labels, self.gt_targets

    def process_speed(self):
        for timestamp_ms, _ in tqdm(self.gt_targets.items(), desc='calcu speed'):
            for idx, _ in enumerate(self.gt_targets[timestamp_ms].bundle_targets_3d):
                self.gt_targets[timestamp_ms].bundle_targets_3d[idx].vels = [0, 0, 0]

    def create_sensor_metas(self, sensors_cameras, cur_labels, timestamp_ms):
        lidar_relative_path = re.sub(r'lidar_reconstruction_.+?/', 'origin_lidar/', cur_labels['lidar'])
        return {'cameras': sensors_cameras,
                'lidar': {'car_center': {'data_path': lidar_relative_path, 'timestamp': timestamp_ms}},
                'radars': self.build_radar_metas(timestamp_ms)}


if __name__ == '__main__':
    from rich import print
    meta_json = '/workspace/auto-labeling-tools/result/sensebee/GOP/22058/single/meta.json'
    sensebee_zip = 'result/sensebee/GOP/22058/single/22058-2024061102472419.zip'
    label_json = '/workspace/auto-labeling-tools/result/sensebee/GOP/22058/single/label.jsonl'
    sensebee_root = 'ad_system_common:s3://sdc_gt_label/GAC/sensebee/gop_label_1w_new/gop_sensebee_clip_7_19'

    res = GOPPreAnnoPostProcessor(meta_json, sensebee_zip, label_json, sensebee_root, gt_path='gop_test.jsonl').process()
    print(res)
