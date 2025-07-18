# Import from third library
import numpy as np
from tqdm import tqdm

# Import from alt
from alt import smart_exists, smart_glob
from alt.utils.file_helper import load_autolabel_meta_json
from alt.utils.load_helper import AutolabelObjectLoader
from alt.utils.petrel_helper import global_petrel_helper
from alt.sensebee.base.base_process import BaseSenseBeeProcessor
from loguru import logger


class ObjectQualityInspection(object):
    def __init__(
        self,
        meta_json: str,
        sequential_continuous_num: int = 200,
        front_maximum_distance: int = 70,
        projection_threshold: float = 0.05,
        box2d_threshold: float = 0.3,
        recall_threshold: float = 0.7,
        precision_threshold: float = 0.7,
    ) -> None:

        self.meta_json = meta_json
        self.sequential_continuous_num = sequential_continuous_num  # 必须在这个数值内保持时序稳定
        self.front_maximum_distance = front_maximum_distance  # 整份clip必须有这个距离外的3D目标
        self.projection_threshold = projection_threshold  # 存在某个目标的投影阈值低于
        self.box2d_threshold = box2d_threshold  # 基于这个2D阈值做判定，若存在高于此阈值的2D框找不到雷达框对应, 则认为是误报
        self.recall_threshold = recall_threshold  # 多设备单帧平均的recall，需大于这个值才认为是合格，需要配合box2d_threshold
        self.precision_threshold = precision_threshold  # 多设备单帧平均的precision, 同上
        self.loader = AutolabelObjectLoader(meta_json)

    def sequential_continuous_filter(self, timestamps):
        """
        Returns: continuous timstamps, continuous ratio
        """
        if len(timestamps) < self.sequential_continuous_num:
            return [], 0

        timestamp_gaps = [timestamps[idx + 1] - timestamps[idx] for idx in range(len(timestamps) - 1)]
        timestamp_gaps = np.array(timestamp_gaps)

        left_point, right_point = 0, len(timestamp_gaps) - 1

        while right_point - left_point > self.sequential_continuous_num - 1:
            _timestamp_gaps = timestamp_gaps[left_point : right_point + 1]
            mask = (_timestamp_gaps >= 90) & (_timestamp_gaps <= 110)
            if np.all(mask):
                return timestamps[left_point : right_point + 1 + 1], (right_point - left_point + 1) / len(_timestamp_gaps)
            elif timestamp_gaps[left_point] > 110 or timestamp_gaps[left_point] < 90:
                left_point += 1
            elif timestamp_gaps[right_point] > 110 or timestamp_gaps[right_point] < 90:
                right_point -= 1
            else:
                left_point += 1
                right_point -= 1

        return [], 0

    def velocity_anomaly_filter(self, timestamps):
        """
        Returns: velocity normal timstamps, velocity normal ratio
        """

        def vector_magnitude(vector):
            # 计算向量的模长
            return np.linalg.norm(vector)

        def vector_angle(vector1, vector2):
            # 计算两个向量之间的夹角（以度为单位）
            dot_product = np.dot(vector1, vector2)
            magnitude_product = vector_magnitude(vector1) * vector_magnitude(vector2)
            if magnitude_product == 0:
                return np.nan  # 如果有任一向量的模长为0，则返回NaN
            cosine_angle = dot_product / magnitude_product
            angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            return np.degrees(angle_rad)

        def vector_change_rate(vector1, vector2):
            # 计算向量的变化率，包括数值变化和角度变化
            delta_vector = vector2 - vector1
            magnitude_change_rate = vector_magnitude(delta_vector)
            angle_change_rate = vector_angle(vector1, vector2)
            return magnitude_change_rate, angle_change_rate

        # for track_id, track_targets in self.loader.multi_trackid_bev_metas.items():
        #     velocity_list = [target.bev_vel for target in track_targets]
        #     time_list = [target.timestamp for target in track_targets]

        return timestamps, 1.0  # TODO

    def front_distance_filter(self, timestamps):
        """
        Returns: velocity normal timstamps, velocity normal ratio
        """
        res = []
        for timestamp in timestamps:
            targets = self.loader.multi_timestamp_bev_metas[timestamp]
            for target in targets:
                target["bev_corners_3d"]
            center_points = np.array([np.mean(item["bev_corners_3d"], axis=0) for item in targets])

            flag = True
            if center_points.size > 0:
                flag = center_points[:, 0].max() >= self.front_maximum_distance
            res.append(flag)

        if any(res):
            return timestamps, sum(res) / len(res)
        else:
            return [], 0

    def get_similarity(self, target):
        confs = [item["conf"] for item in target["camera_metas"]]
        confs = [item for item in confs if item is not None]
        return np.max(confs) if len(confs) > 0 else None

    def camera_projection_filter(self, timestamps):
        """
        Returns: normal projection timstamps, normal projection ratio
        """
        total_similarity = []
        success_timestamps = []
        for timestamp in timestamps:
            targets = self.loader.multi_timestamp_bev_metas[timestamp]

            _boxes_similarity = [self.get_similarity(target) for target in targets]
            _boxes_similarity = [item for item in _boxes_similarity if item]

            mask = np.array(_boxes_similarity) > self.projection_threshold

            if all(mask):
                success_timestamps.append(timestamp)

            total_similarity.extend(_boxes_similarity)

        total_mask = np.array(total_similarity) > self.projection_threshold
        return success_timestamps, sum(total_mask) / (len(total_mask) + 1e-10)

    def precison_and_recall_filter(self, timestamps):
        success_timestamps = []
        pr_list = []

        for timestamp in timestamps:
            TP, FN, FP = 0, 0, 0
            for camera_name in self.loader.camera_names:
                cur_meta = self.loader.select_frame_by_timestamp_and_camera_name(camera_name, timestamp)
                for target in cur_meta["objects"]:

                    if target["conf"]:
                        if (target["conf"] < 0.2) and (target["score2d"] < self.box2d_threshold):
                            FP += 1
                        else:
                            TP += 1
                    else:
                        if target["score2d"] and (target["score2d"] > self.box2d_threshold) and (target["bbox3d"] is None):
                            FN += 1
                        # else: 其他本身就是已经被过滤掉的
                        #     FP += 1

            if TP == 0:  # 空图也OK
                success_timestamps.append(timestamp)
                continue

            recall = TP / (TP + FN + 1e-10)
            precision = TP / (TP + FP + 1e-10)
            pr_list.append([precision, recall])

            if recall >= self.recall_threshold and precision >= self.precision_threshold:
                success_timestamps.append(timestamp)

        mean_precision, mean_recall = np.nan, np.nan
        if len(pr_list) > 0:
            mean_precision, mean_recall = np.array(pr_list).mean(axis=0).tolist()

        return success_timestamps, (mean_precision, mean_recall)

    # def process(self):
    #     # S1. 选择稳定连续的时间戳区间
    #     select_timestamps, continuous_ratio = self.sequential_continuous_filter(self.loader.intersect_timstamps)
    #     logger.info("S1. sequential_continuous_ratio[{}]: {}, len: {}".format(self.sequential_continuous_num, continuous_ratio, len(select_timestamps)))

    #     # # S2. 过滤速度过于异常的相关问题
    #     # select_timestamps, normal_speed_ratio = self.velocity_anomaly_filter(select_timestamps)
    #     # logger.info("S2. normal_speed_ratio[{}]: {}, len: {}".format("None", normal_speed_ratio, len(select_timestamps)))

    #     # # S3. 纵向分布问题
    #     # select_timestamps, front_distance_ratio = self.front_distance_filter(select_timestamps)
    #     # logger.info("S3. front_distance_ratio[{}]: {}, len: {}".format(self.front_maximum_distance, front_distance_ratio, len(select_timestamps)))

    #     # # S4. 目标滞后等投影问题
    #     # select_timestamps, normal_projection_ratio = self.camera_projection_filter(select_timestamps)
    #     # logger.info("S4. normal_projection_ratio[{}]: {}, len: {}".format(self.projection_threshold, normal_projection_ratio, len(select_timestamps)))

    #     # # S5. 漏检&误报问题问题
    #     # select_timestamps, (precision, recall) = self.precison_and_recall_filter(select_timestamps)
    #     # logger.info("S5. precison: {}, recall: {}, len: {}".format(precision, recall , len(select_timestamps)))

    #     # # S6. 需要在最后再做一遍时序的验证
    #     # select_timestamps, continuous_ratio = self.sequential_continuous_filter(select_timestamps)
    #     # logger.info("S6. sequential_continuous_ratio[{}]: {}, len: {}".format(self.sequential_continuous_num, continuous_ratio, len(select_timestamps)))

    #     return select_timestamps

    def speed_filter(self, speed_limit=60):
        local_file = self.loader.meta["data_structure"]["location"]["local"]

        speeds = []
        for data in global_petrel_helper.readlines(local_file):
            _, _, _, _, _, vx, vy, vz, _ = data.split(" ")
            speed = np.sqrt(float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2)
            speed = speed * 3.6
            speeds.append(speed)

        return np.mean(speeds) <= speed_limit

    def process(self):
        select_timestamps = self.loader.intersect_timstamps
        if not self.speed_filter():
            select_timestamps = []

        return select_timestamps


def calcu_speeds(metas):
    local_file = metas["data_structure"]["location"]["local"]
    speeds = []
    for data in global_petrel_helper.readlines(local_file):
        _, _, _, _, _, vx, vy, vz, _ = data.split(" ")
        speed = np.sqrt(float(vx) ** 2 + float(vy) ** 2 + float(vz) ** 2)
        speed = speed * 3.6
        speeds.append(speed)
    return np.mean(speeds)


class GopQualityInspection(object):
    def __init__(
        self,
        meta_json: str,
        gop1v_upper_thres: int = 8,  # 单张图像超过8个gop目标进行统计
        gop11v_upper_thres: float = 0.1,  # 整个clip11v，这种图像数量超过多少比例，则不送标
        gop1v_lower_thres: int = 0,  # 单张图像超过0个gop目标进行统计
        gop11v_lower_thres: float = 0.1,  # 整个clip11v，这种图像数量少于多少比例，则不送标
        gop_det_thres: float = 0.4,  # 0.4的检测结果阈值
    ) -> bool:
        self.meta_json = meta_json
        self.gop1v_upper_thres = gop1v_upper_thres
        self.gop11v_upper_thres = gop11v_upper_thres
        self.gop1v_lower_thres = gop1v_lower_thres
        self.gop11v_lower_thres = gop11v_lower_thres
        self.gop_det_thres = gop_det_thres
        self.target_labels = [
            "Animals",
            "Gate-Rod",
            "Retractable-Door",
            "Cone",
            "Pole",
            "Isolation-Barrel",
            "Triangle-Warning",
            "Barrier",
            "Permanent-Barricade",
            "Temporary-Barricade",
            "Ground-Sign",
            "Speed-Bump",
            "Open-Parking-Lock",
            "Closed-Parking-Lock",
            "Parking-Limiter",
            "Column",
            "Charging Pile",
            "Handcart",
            "Obstacles",
        ]

    def gop_num_filter(self):
        """
        Returns: whether this clip has suitable gop object num
        """
        metas = BaseSenseBeeProcessor().safe_load_meta_json(self.meta_json)[0]
        data_annotation = metas["data_annotation"]

        if calcu_speeds(metas) > 60:
            logger.warning('speed over 60km/h, drop clip')
            return False

        pkl_names = smart_glob("{}/*object.pkl".format(data_annotation))
        over_upper_11v = 0
        over_lower_11v = 0
        frame_num11v = 0
        for pkl_name in pkl_names:
            frames = global_petrel_helper.load_pk(pkl_name)["frames"]
            frame_num11v += len(frames)
            for frame in frames:
                tmp = 0
                for obj_info in frame["objects"]:
                    if obj_info["label"] in self.target_labels and obj_info["score2d"] > self.gop_det_thres:
                        tmp += 1
                if tmp > self.gop1v_lower_thres:
                    over_lower_11v += 1
                if tmp > self.gop1v_upper_thres:
                    over_upper_11v += 1

        if over_upper_11v > self.gop11v_upper_thres * frame_num11v:
            logger.info(f"{over_upper_11v} over {self.gop11v_upper_thres} imgs have too many gop objects, drop clip")
            return False
        elif over_lower_11v < self.gop11v_lower_thres * frame_num11v:
            logger.info(f"{over_lower_11v} less than {self.gop11v_lower_thres} imgs have enough gop objects, drop clip")
            return False
        else:
            return True

    def process(self):
        flag = self.gop_num_filter()
        return flag


if __name__ == "__main__":
    # Standard Library
    import pickle as pkl

    # Import from alt
    from alt import dump
    from alt.utils.env_helper import env

    input_path = "input.tmp.txt"
    gt_paths = [item.strip() for item in open(input_path).readlines()]
    gt_paths = ["ad_system_common:" + item for item in gt_paths]

    res = []
    for idx, gt_path in tqdm(enumerate(gt_paths), desc=f"rank: {env.rank}"):
        if not env.is_my_showtime(idx):
            continue
        meta_json = gt_path.strip("/").replace("gt_labels", "meta.json")
        assert smart_exists(meta_json), meta_json
        select_timestamps = ObjectQualityInspection(meta_json).process()
        res.append(select_timestamps)
    env.barrier()

    res = env.gather(res)

    if env.is_master():
        dump("res.pkl", res)

    # gop filter example
    flag = GopQualityInspection(
        meta_json="s3://sdc_gt_label/GAC/autolabel_20240411_772/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-24/2024_03_24_01_47_18_gacGtParser/meta.json"
    ).process()
    print(flag)
