import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from alt.evaluation.base_evaluator import BaseEvaluator
from alt.utils.load_helper import AutolabelObjectLoader
from alt.evaluation.stable_eval.stable_eval import StableEvaluator


class ObjectEvaluator(BaseEvaluator):
    def __init__(self,
                 meta_json: str,
                 sequential_continuous_num: int = 200,
                 front_maximum_distance: int = 70,
                 projection_threshold: float = 0.05,
                 box2d_threshold: float = 0.3,
                 recall_threshold: float = 0.7,
                 precision_threshold: float = 0.7) -> None:

        self.meta_json = meta_json
        self.sequential_continuous_num = sequential_continuous_num  # 必须在这个数值内保持时序稳定
        self.front_maximum_distance = front_maximum_distance  # 整份clip必须有这个距离外的3D目标
        self.projection_threshold = projection_threshold  # 存在某个目标的投影阈值低于
        self.box2d_threshold = box2d_threshold  # 基于这个2D阈值做判定，若存在高于此阈值的2D框找不到雷达框对应, 则认为是误报
        self.recall_threshold = recall_threshold  # 多设备单帧平均的recall，需大于这个值才认为是合格，需要配合box2d_threshold
        self.precision_threshold = precision_threshold  # 多设备单帧平均的precision, 同上
        self.loader = AutolabelObjectLoader(meta_json)
        
        super().__init__()

    def stable_evaluate(self):
        targets = self.loader.multi_timestamp_bev_metas
        metrics = StableEvaluator().evaluate(targets)
        return metrics

    def collision_evaluate(self):
        from mmcv.ops import box_iou_rotated
        timestamps = self.loader.multi_timestamp_bev_metas.keys()
        max_vals = []

        for timestamp in timestamps:
            targets = self.loader.multi_timestamp_bev_metas[timestamp]

            boxes = []
            for target in targets:
                enu_box = target["location"][:2] + target["dimension"][:2] + [target["yaw"]]
                boxes.append(enu_box)
            boxes = torch.Tensor(boxes)

            ious = box_iou_rotated(boxes, boxes, mode="iou")
            mask = torch.triu(torch.ones_like(ious), diagonal=1).bool()
            max_vals.extend(ious[mask].view(-1).tolist())
        res = torch.Tensor(max_vals).view(-1)
        collision_ratio = torch.sum(res > 0).item() / res.numel()  * 100
        collision_cnt = torch.sum(res > 0).item()
        return collision_ratio, collision_cnt

    def get_similarity(self, target):
        confs = [item["conf"] for item in target["camera_metas"]]
        confs = [item for item in confs if item is not None]
        return np.max(confs) if len(confs) > 0 else None
    
    def projection_evaluate(self):
        total_similarity = {}
        for timestamp, targets in self.loader.multi_timestamp_bev_metas.items():
            for target in targets:
                confs = [item["conf"] for item in target["camera_metas"]]
                confs = [item for item in confs if item is not None]
                if target['label'] not in total_similarity:
                    total_similarity[target['label']] = []
                total_similarity[target['label']].extend(confs)
        return total_similarity

    def precison_and_recall_evaluate(self):

        precision, recall = {}, {}

        TP, FN, FP = {}, {}, {}
        for timestamp in self.loader.intersect_timstamps:
            for camera_name in self.loader.camera_names:
                cur_meta = self.loader.select_frame_by_timestamp_and_camera_name(camera_name, timestamp)
                for target in cur_meta["objects"]:
                    if target['label'] not in TP:
                        TP[target['label']] = 0
                    if target['label'] not in FN:
                        FN[target['label']] = 0
                    if target['label'] not in FP:
                        FP[target['label']] = 0

                    if target["conf"]:
                        if (target["conf"] < 0.2) and (target["score2d"] < self.box2d_threshold):
                            FP[target['label']] += 1
                        else:
                            TP[target['label']] += 1
                    else:
                        if target["score2d"] and (target["score2d"] > self.box2d_threshold) and (target["bbox3d"] is None):
                            FP[target['label']] += 1
                        # else: 其他本身就是已经被过滤掉的
                        #     FP += 1

        labels = list(set(TP.keys()) & set(FN.keys()) & set(FP.keys()))
    
        for label in labels:
            precision[label] = TP[label] / (TP[label] + FN[label] + 1e-10)
            recall[label] = TP[label] / (TP[label] + FP[label] + 1e-10)

        return recall, precision

    def vertical_distribution_evaluate(self):
        location_xs = []
        for _, metas in self.loader.multi_timestamp_bev_metas.items():
            for meta in metas:
                location_x = meta['location'][0]
                location_xs.append(location_x)
        return location_xs
    
    def targets_evaluate(self):
        target_res = {'frame_num': len(self.loader.multi_timestamp_bev_metas)}
        for _, metas in self.loader.multi_timestamp_bev_metas.items():
            for meta in metas:
                if meta['label'] not in target_res:
                    target_res[meta['label']] = 0
                target_res[meta['label']] += 1
        
        # for label in list(target_res.keys()):
        #     if label in ['frame_num']:
        #         continue
        #     target_res[f'{label}_avg_num'] = round(target_res[label] / len(self.loader.multi_timestamp_bev_metas), 2)

        return target_res
    
    
    def eval(self, output_path='demo.png'):
        # multi_timestamp_bev_metas, multi_trackid_bev_metas
        # TODO，位置突变、速度突变、轨迹闪烁比例、
        stable_metrics = self.stable_evaluate()

        # TODO，碰撞率指标, 尽量不引入torch
        collision_metrics = 0 # self.collision_evaluate()

        # TODO，投影指标
        projection_metrics = self.projection_evaluate()

        # TODO，precision & recall 指标
        precision_recall_metrics = self.precison_and_recall_evaluate()

        # TODO，纵向距离统计 指标
        vertical_metrics = self.vertical_distribution_evaluate()
        
        # 计算数量
        targets_metrics = self.targets_evaluate()

        # TODO，生成一张图表，记录每个metrics
        self.draw_img(targets_metrics=targets_metrics,
                      vertical_metrics=vertical_metrics,
                      projection_metrics=projection_metrics,
                      precision_recall_metrics=precision_recall_metrics,
                      stable_metrics=stable_metrics,
                      collision_metrics=collision_metrics,
                      output_path=output_path)


    def draw_img(self, targets_metrics, vertical_metrics, projection_metrics, precision_recall_metrics, stable_metrics, collision_metrics, output_path):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 4, 1) # clip 的数量，每个类别目标的数量

        labels, data = [item.split('_')[-1] for item in list(targets_metrics.keys())], list(targets_metrics.values())
        bar = plt.bar(range(len(data)), data, tick_label=labels, color=['g', 'b', 'b', 'b'])
        plt.bar_label(bar, label_type='edge')
        plt.xlabel('targets information')

        plt.subplot(2, 4, 2)  #下面的语句绘制第二个子图
        for _label in projection_metrics.keys():
            label = _label.lower()
            plt.hist(projection_metrics[_label], np.arange(0, 1, 0.1), density=True, alpha=0.6,  label=label)
        plt.xlabel('projection similarity')
        plt.legend()

        plt.subplot(2, 4, 3)	#下面的语句绘制第三个子图
        recall, precision = precision_recall_metrics
        x = np.arange(len(recall))
        total_width, n = 1.0, 2
        width = total_width / n
        # x = x - (total_width - width) / 2
        tick_label = [item.split('_')[-1].lower() for item in list(recall.keys())]
        plt.bar(x, list(recall.values()),  width=width, label='recall', tick_label=tick_label)
        plt.bar(x + width, list(precision.values()), width=width, label='precision')
        plt.ylim((0, 1))
        plt.xticks(rotation=-30) 
        plt.legend()
        
        plt.subplot(2, 4, 4)  # TODO
        x3 = [2, 5, 7, 8, 10, 11]
        y3 = [3, 5, 4, 1, 15, 10]
        plt.plot(x3, y3, '-.')
        plt.plot(x3, y3, 's')
        plt.xlabel('other information')

        plt.subplot(2, 4, (5, 6))	#下面的语句绘制第四个子图
        plt.hist(vertical_metrics, np.arange(-100, 200, 20), density=True, alpha=0.6, color='g')
        plt.xlabel('vertical_distributon')
        
        plt.subplot(2, 4, (7, 8))	#下面的语句绘制第四个子图
        color_list = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        total_width, n = len(stable_metrics) * 0.3, len(stable_metrics)
        width = total_width / n

        ratios_keys = [key for key in list(stable_metrics.values())[0].keys() if 'ratio' in key] if stable_metrics else []
        x_axis = np.arange(len(ratios_keys))
        for idx, (label, cur_metrics) in enumerate(stable_metrics.items()):
            data = [round(cur_metrics[key], 3) for key in ratios_keys]
            if idx == 0:
                tick_label = [key.rsplit('_', 1)[0] for key in ratios_keys]
                bar = plt.bar(x_axis, data,  width=width, label=label.lower(), tick_label=tick_label, color=color_list[idx])
            else:
                bar = plt.bar(x_axis + idx * width, data, width=width, label=label.lower(), color=color_list[idx])
        plt.bar_label(bar, label_type='edge')
        plt.legend()
        plt.xticks(rotation=-30)
        plt.xlabel('stabel information')

        plt.suptitle(self.meta_json)
        plt.savefig(output_path, bbox_inches='tight')


if __name__ == '__main__':
    meta_json = 'ad_system_common:s3://sdc_gt_label/GAC/autolabel_20240410_768/Data_Collection/GT_data/gacGtParser/drive_gt_collection/PVB_gt/2024-03/A02-290/2024-03-21/2024_03_21_15_26_34_gacGtParser/meta.json'

    evaluator = ObjectEvaluator(meta_json)
    evaluator.eval()