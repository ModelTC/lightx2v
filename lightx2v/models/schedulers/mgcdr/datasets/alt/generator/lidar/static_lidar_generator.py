# Standard Library
import copy

# Import from third library
import numpy as np

# Import from alt
from alt.decoder.simple_location_decoder import SimpleLocationDecoder
from alt.generator.base import Generator
from alt.sensebee.gop.merge_pcd_by_location import get_RT_from_enu_yaw
from loguru import logger


class StaticLidarGenerator(Generator):
    def __init__(self, location_path):
        self.location_decoder = SimpleLocationDecoder(location_path)

    def initialize(self):
        pass

    def generate(self, metas):
        if len(metas["targets"]) == 0:
            return []

        targets = metas["targets"]
        src_timestamp = metas["src_timestamp"] * 1000 * 1000
        dst_timestamp = metas["dst_timestamp"] * 1000 * 1000
        dst_targets = copy.deepcopy(targets)
        time_error_src, src_location = self.location_decoder.search(src_timestamp)
        time_error_dst, dst_location = self.location_decoder.search(dst_timestamp)
        if max(time_error_src, time_error_dst) > 10 * 1000 * 1000:
            logger.warning(
                f"the timestamp in location file and the src/dst timestamps are not aligned, {max(time_error_src, time_error_dst) / 1000000}ms over 10ms"
            )
            return None
        else:
            T_o_src = get_RT_from_enu_yaw(enu=src_location["location"], yaw=src_location["yaw"])
            T_o_dst = get_RT_from_enu_yaw(enu=dst_location["location"], yaw=dst_location["yaw"])
            T_dst_src = np.linalg.inv(T_o_dst) @ T_o_src
            src_targets_location = np.array([target["location"] + [1.0] for target in targets])
            new_location = (src_targets_location @ T_dst_src.T)[:, :3].tolist()  # 行向量

            yaw_offset = dst_location["yaw"] - src_location["yaw"]  # 时序接近，就简单点算角度变化了
            new_yaw = [target["yaw"] - yaw_offset for target in targets]
            assert len(dst_targets) == len(new_location) == len(new_yaw)
            for i in range(len(dst_targets)):
                dst_targets[i]["location"] = new_location[i]
                dst_targets[i]["yaw"] = new_yaw[i]
        return dst_targets


if __name__ == "__main__":
    location_path = "ad_system_common_sdc:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-19/2024_03_19_07_37_49_gacGtParser/ved/pose/location.tmp.txt"
    lidar_generator = StaticLidarGenerator(location_path=location_path)
    targets = [
        {"token": "abcd", "location": [1.0, 2.0, 0.0], "length": 4.2, "weight": 1.7, "height": 1.3, "yaw": 0.1},
        {"token": "efg", "location": [1.0, -2.0, 0.0], "length": 4.2, "weight": 1.7, "height": 1.3, "yaw": -0.1},
    ]
    metas = {"targets": targets, "src_timestamp": 1710833869199, "dst_timestamp": 1710833869999}
    dst_targets = lidar_generator.generate(metas)
    print(dst_targets)
