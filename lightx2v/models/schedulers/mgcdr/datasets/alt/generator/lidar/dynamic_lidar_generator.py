# Import from third library
import math
import numpy as np
import torch

# Import from alt
from alt.generator.base import Generator
from loguru import logger
from mmcv.ops import box_iou_rotated


def normalize_angle(angle):
    a = math.fmod(angle + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += (2.0 * math.pi)
    return a - math.pi


def angle_interpolation(angle1, angle2):
    # 规范化角度到 -π 到 π 之间
    a1 = normalize_angle(angle1)
    a2 = normalize_angle(angle2)

    # 计算角度差，并考虑循环边界问题
    delta = a2 - a1
    if delta > math.pi:
        delta -= 2.0 * math.pi
    elif delta < -math.pi:
        delta += 2.0 * math.pi

    # 计算中间角度
    mid_angle = a1 + delta / 2.0
    return normalize_angle(mid_angle)


class DynamicLidarGenerator(Generator):
    def __init__(self, lidar_res, gt_targets):
        self.lidar_res = lidar_res
        self.gt_targets = gt_targets

        self.lidar_results = {}
        for timestamp, targets in self.lidar_res.items():
            self.lidar_results[timestamp] = []
            for target in targets:
                box = target["bbox3d"][:6].tolist() + [target["bbox3d"][8]]
                self.lidar_results[timestamp].append(box)

    def initialize(self):
        pass

    def reorg_box3d(self, timestamp):
        targets = self.gt_targets[timestamp]

        bbox3d = {}
        for target_3d in targets.bundle_targets_3d:
            cur_box = target_3d.location + [target_3d.length, target_3d.width, target_3d.height, target_3d.yaw, target_3d.id]
            bbox3d[target_3d.id] = cur_box
        return bbox3d

    def calcu_angle_error(self, yaw1, yaw2):
        if yaw1 is None or yaw2 is None:
            return 0
        diff = yaw1 / np.pi * 180 - yaw2 / np.pi * 180
        while diff < -180:
            diff += 360
        while diff > 180:
            diff -= 360

        angle_error = abs(diff)
        assert angle_error <= 180, angle_error
        return angle_error

    def get_lidar_result(self, timestamp):
        lidar_timestamps = np.array(list(self.lidar_results.keys()))
        timestamp_gap = np.abs(lidar_timestamps - timestamp)

        assert timestamp_gap.min() < 15
        select_index = lidar_timestamps[timestamp_gap.min()]
        return self.lidar_results[select_index]

    def generate(self, metas):
        before_targets = self.reorg_box3d(metas["before_timestamp"])
        after_targets = self.reorg_box3d(metas["after_timestamp"])

        before_targets_ids, after_targets_ids = list(before_targets.keys()), list(after_targets.keys())

        match_ids = set(before_targets_ids) & set(after_targets_ids)

        logger.warning("len: {}, ids: {}".format(len(before_targets_ids), len(match_ids)))

        res = {}
        for track_id in match_ids:
            ratios = [
                (metas["after_timestamp"] - metas["timestamp"]) / (metas["after_timestamp"] - metas["before_timestamp"]),
                (metas["timestamp"] - metas["before_timestamp"]) / (metas["after_timestamp"] - metas["before_timestamp"]),
            ]

            interp_target = np.array(before_targets[track_id]) * ratios[0] + np.array(after_targets[track_id]) * ratios[1]
            interp_target[6] = angle_interpolation(before_targets[track_id][6], after_targets[track_id][6])

            # 跟检测结果去判断IOU
            # lidar_boxes = torch.Tensor(np.array(self.get_lidar_result(metas["timestamp"])))
            # lidar_bev_boxes = lidar_boxes[:, [0, 1, 3, 4, 6]]

            # cur_boxes = torch.Tensor(interp_target[np.newaxis, :])[:, [0, 1, 3, 4, 6]]
            # ious = box_iou_rotated(lidar_bev_boxes, cur_boxes)

            # if ious.max() > 0.6:
            #     select_box = lidar_boxes[ious.argmax()].numpy()

            #     select_box[3:7] = interp_target[3:7]  # 角度还是采用和原来的角度
            #     interp_target = select_box

            last_angle, interp_angle = before_targets[track_id][6], interp_target[6]

            if self.calcu_angle_error(last_angle, interp_angle) >= 45:  # 会有自己标注错误的问题
                interp_target[6] = interp_angle = last_angle

            assert self.calcu_angle_error(last_angle, interp_angle) < 90

            res[int(track_id)] = interp_target[:7]

        return res


if __name__ == "__main__":

    root = "ad_system_common:s3://sdc_gt_label/ql_test/sensebee/pvb/pvb_all_attr/0522/23_meta_2024_03_21_02_44_00_gacGtParser"
    label_json = "ad_system_common:s3://sdc_gt_label/ql_test/sensebee/pvb/pvb_all_attr/0522/23_meta_2024_03_21_02_44_00_gacGtParser/label.jsonl"

    lidar_generator = DynamicLidarGenerator(root=root, labels=label_json)

    lidar_generator.initialize()
