# Standard Library
from collections import OrderedDict

# Import from third library
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import cv2
from io import BytesIO

# Import from alt
from alt.coordinate.bbox.lidar_box3d import LiDARInstance3DBoxes
from alt.coordinate.points import Camera3DPointsTransfer
from alt.evaluation.base_evaluator import BaseEvaluator
from alt.evaluation.stable_eval.stable_eval import StableEvaluator
from alt.utils.petrel_helper import global_petrel_helper
from alt.visualize.utils.plt_helper import plt2img
from easydict import EasyDict


class ALIGNED_FLAG:
    aligned = False
    offset = 1


def bbox_iou_overlaps(b1, b2, aligned=False, return_union=False, eps=1e-9):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + ALIGNED_FLAG.offset) * (b1[:, 3] - b1[:, 1] + ALIGNED_FLAG.offset)
    area2 = (b2[:, 2] - b2[:, 0] + ALIGNED_FLAG.offset) * (b2[:, 3] - b2[:, 1] + ALIGNED_FLAG.offset)
    # only for giou loss
    lt1 = torch.max(b1[:, :2], b2[:, :2])
    rb1 = torch.max(b1[:, 2:4], b2[:, 2:4])
    lt2 = torch.min(b1[:, :2], b2[:, :2])
    rb2 = torch.min(b1[:, 2:4], b2[:, 2:4])
    # en = (lt1 < rb2).type(lt1.type()).prod(dim=1)
    # inter_area = inter_area * en
    wh1 = (rb2 - lt1 + ALIGNED_FLAG.offset).clamp(min=0)
    wh2 = (rb1 - lt2 + ALIGNED_FLAG.offset).clamp(min=0)
    inter_area = wh1[:, 0] * wh1[:, 1]
    union_area = area1 + area2 - inter_area

    iou = inter_area / torch.clamp(union_area, min=1)

    ac_union = wh2[:, 0] * wh2[:, 1] + eps
    giou = iou - (ac_union - union_area) / ac_union
    return giou, iou


COLORS = [
    "#FF6F61",  # Coral
    "#6B5B95",  # Purple
    "#88B04B",  # Green
    "#FFA07A",  # Light Salmon
    "#20B2AA",  # Light Sea Green
    "#FFB347",  # Pastel Orange
    "#FFCC33",  # Goldenrod
    "#FF69B4",  # Hot Pink
    "#87CEEB",  # Sky Blue
    "#32CD32",  # Lime Green
    "#FFD700",  # Gold
    "#6495ED",  # Cornflower Blue
    "#DC143C",  # Crimson
    "#8A2BE2",  # Blue Violet
    "#00CED1",  # Dark Turquoise
    "#F08080",  # Light Coral
    "#C71585",  # Medium Violet Red
    "#3CB371",  # Medium Sea Green
    "#FF4500",  # Orange Red
    "#2E8B57",  # Sea Green
    "#DAA520",  # Goldenrod
]


class ObjectEvaluatorV2(BaseEvaluator):
    def __init__(self, anno=None) -> None:
        super().__init__()

        self.anno = anno
        self.metas = self.build()

    def build(self):
        metas = OrderedDict()
        for item in global_petrel_helper.readlines(self.anno):
            data = json.loads(item)
            if data.get('valid', True):
                metas[data["timestamp"]] = EasyDict(data)
        return metas

    def targets_evaluate(self):
        target_res = {}
        for _, metas in self.metas.items():
            for target in metas["Objects"]:
                if target.info2d is None:  # 针对每个相机只画可见的
                    continue

                if target["label"] not in target_res:
                    target_res[target["label"]] = 0
                target_res[target["label"]] += 1
        return target_res

    def vertical_distribution_evaluate(self):
        location_xs = []
        for _, metas in self.metas.items():
            for target in metas["Objects"]:
                location_x = target["bbox3d"][0]
                location_xs.append(location_x)
        return location_xs

    def projection_evaluate(self):
        total_similarity = {}
        for meta in self.metas.values():
            targets = meta["Objects"]

            for camera_name, camera_meta in meta["sensors"]["cameras"].items():
                lidar2camera_rt = torch.Tensor(camera_meta["extrinsic"])

                camera_intrinsic = torch.Tensor(camera_meta["camera_intrinsic"])
                camera_dist = torch.Tensor(camera_meta["camera_dist"])

                for target in targets:

                    if target.info2d is None:  # 针对每个相机只画可见的
                        continue

                    if camera_name not in target.info2d:
                        continue

                    box_2d = torch.Tensor([target.info2d[camera_name]["bbox2d"]])
                    lidar_bbox3d = [target["bbox3d"][:6] + [target["bbox3d"][8]]]

                    lidar_bbox3d = LiDARInstance3DBoxes(lidar_bbox3d, box_dim=7, origin=(0.5, 0.5, 0.5))
                    lidar_bbox3d_corners = lidar_bbox3d.corners

                    pts_4d = torch.cat([lidar_bbox3d_corners.reshape(-1, 3), torch.ones((8, 1))], dim=-1)
                    pts_camera_4d = pts_4d @ lidar2camera_rt.T
                    pts_camera_3d = pts_camera_4d[:, :3]

                    point_2d = Camera3DPointsTransfer.transfer_camera3d_to_image(pts_camera_3d, camera_intrinsic, camera_dist)

                    x1, x2 = point_2d[:, 0].min(), point_2d[:, 0].max()
                    y1, y2 = point_2d[:, 1].min(), point_2d[:, 1].max()

                    projection_bbox = torch.stack([x1, y1, x2, y2]).unsqueeze(0)
                    giou, iou = bbox_iou_overlaps(box_2d, projection_bbox)

                    if target["label"] not in total_similarity:
                        total_similarity[target["label"]] = []
                    total_similarity[target["label"]].extend(iou)
            return total_similarity

    def precison_and_recall_evaluate(self):
        precision, recall = {}, {}
        TP, FN, FP = {}, {}, {}

        for meta in self.metas.values():
            for target_3d in meta["Objects"]:
                if target_3d.info2d is None:  # 针对每个相机只画可见的
                    continue

                if target_3d["label"] not in TP:
                    TP[target_3d["label"]], FN[target_3d["label"]], FP[target_3d["label"]] = 0, 0, 0
                TP[target_3d["label"]] += 1

            for _, target_2d in meta["Pure2DObjects"].items():
                for label in target_2d["label2d"]:
                    if label not in TP:
                        TP[label], FN[label], FP[label] = 0, 0, 0
                FN[label] += 1

        for label in TP.keys():
            recall[label] = TP[label] / (TP[label] + FN[label] + 1e-10)
            precision[label] = TP[label] / (TP[label] + FP[label] + 1e-10)

        return recall, precision

    def stable_evaluate(self):
        targets = {}
        for timestamp, meta in self.metas.items():
            targets[timestamp] = []
            for target_3d in meta["Objects"]:
                cur_target = {
                    "location": target_3d["bbox3d"][:3],
                    "dimension": target_3d["bbox3d"][3:6],
                    "length": target_3d["bbox3d"][3],
                    "width": target_3d["bbox3d"][4],
                    "height": target_3d["bbox3d"][5],
                    "bev_vel": target_3d["velocity"],
                    "id": target_3d["id"],
                    "yaw": target_3d["bbox3d"][8],
                    "label": target_3d["label"],
                }
                targets[timestamp].append(cur_target)

        metrics = StableEvaluator().evaluate(targets)
        return metrics

    def clip_evaluate(self):
        target_res = {"frame": len(self.metas)}

        num_3d, num_2d, empty = 0, 0, 0
        for meta in self.metas.values():
            attention_objects = [item for item in meta["Objects"] if "barricade" not in item['label'].lower()]
            if len(attention_objects) == 0:
                empty += 1

            for target_3d in meta["Objects"]:
                if target_3d.info2d is None:
                    continue
                num_3d += 1
            for _, target_2d in meta["Pure2DObjects"].items():
                num_2d += len(target_2d["label2d"])

        target_res["empty"] = empty
        target_res["num_3d"] = num_3d
        target_res["num_2d"] = num_2d

        return target_res

    def eval(self, output_path="demo.png"):

        metrics = dict(
            clip_metrics=self.clip_evaluate(),
            targets_metrics=self.targets_evaluate(),
            vertical_metrics=self.vertical_distribution_evaluate(),
            projection_metrics=self.projection_evaluate(),
            precision_recall_metrics=self.precison_and_recall_evaluate(),
            stable_metrics=self.stable_evaluate(),
        )

        self.draw_img(output_path=output_path, **metrics)

        return metrics

    def draw_img(
        self,
        clip_metrics,
        targets_metrics,
        vertical_metrics,
        projection_metrics,
        precision_recall_metrics,
        stable_metrics,
        output_path,
    ):
        plt.figure(figsize=(20, 10))

        plt.subplot(2, 4, (1, 2))  # clip 的数量，每个类别目标的数量
        labels, data = [item.lower() for item in list(targets_metrics.keys())], list(targets_metrics.values())

        bar = plt.bar(range(len(data)), data, color=COLORS[: len(labels)], tick_label=labels)
        plt.bar_label(bar, label_type="edge")
        # plt.legend()
        plt.xticks(rotation=-30)

        plt.subplot(2, 4, 3)
        for _label in projection_metrics.keys():
            label = _label.lower()
            plt.hist(projection_metrics[_label], np.arange(0, 1, 0.1), density=True, alpha=0.6, label=label)
        plt.xlabel("projection similarity")
        plt.legend()

        plt.subplot(2, 4, 4)
        recall, precision = precision_recall_metrics
        x = np.arange(len(recall))
        total_width, n = 1.0, 2
        width = total_width / n
        # x = x - (total_width - width) / 2
        tick_label = [item.split("_")[-1].lower() for item in list(recall.keys())]
        plt.bar(x, list(recall.values()), width=width, label="3d recall", tick_label=tick_label)
        plt.bar(x + width, list(precision.values()), width=width, label="3d precision")
        plt.ylim((0, 1))
        plt.xticks(rotation=-30)
        plt.legend()

        plt.subplot(2, 4, 5)  # 下面的语句绘制第四个子图

        colors = ["#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#9B59B6", "#34495E"]
        labels, data = [item.lower() for item in list(clip_metrics.keys())], list(clip_metrics.values())
        bar = plt.bar(range(len(data)), data, color=colors[: len(labels)], tick_label=labels)
        plt.bar_label(bar, label_type="edge")
        # plt.legend()
        plt.xlabel("clip metrics")

        plt.subplot(2, 4, 6)  # 下面的语句绘制第四个子图
        plt.hist(vertical_metrics, np.arange(-100, 200, 20), density=True, alpha=0.6, color="g")
        plt.xlabel("vertical_distributon")

        plt.subplot(2, 4, (7, 8))  # 下面的语句绘制第四个子图
        color_list = ["b", "c", "g", "k", "m", "r", "y", "purple", "orange", "pink", "brown", "grey",
                      "olive", "cyan", "#FF5733", "#33FF57", "#3357FF", "#F1C40F", "#9B59B6", "#34495E"]
        total_width, n = len(stable_metrics) * 0.1, len(stable_metrics)
        width = total_width / (n + 1e-10)

        ratios_keys = [key for key in list(stable_metrics.values())[0].keys() if "ratio" in key] if stable_metrics else []
        x_axis = np.arange(len(ratios_keys))
        for idx, (label, cur_metrics) in enumerate(stable_metrics.items()):
            data = [round(cur_metrics[key], 3) for key in ratios_keys]
            if idx == 0:
                tick_label = [key.rsplit("_", 1)[0] for key in ratios_keys]
                bar = plt.bar(x_axis, data, width=width, label=label.lower(), tick_label=tick_label, color=color_list[idx])
            else:
                bar = plt.bar(x_axis + idx * width, data, width=width, label=label.lower(), color=color_list[idx])
        plt.bar_label(bar, label_type="edge")
        plt.legend()
        plt.xticks(rotation=-30)
        plt.xlabel("stabel information")

        plt.suptitle(self.anno)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight")
        plt.close()
        buf.seek(0)

        image_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        image_cv = cv2.imdecode(image_array, 1)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        global_petrel_helper.imwrite(output_path, image_cv, ext=output_path[-4:])


if __name__ == "__main__":
    anno = "ad_system_common:s3://sdc_gt_label/data.infra.gt_labels/gop/sensebee/GAC/autolabel_20240418_804/2024_04_01_01_14_09_gacGtParser/gt.jsonl"
    ObjectEvaluatorV2(anno).eval(output_path='demo.png')
