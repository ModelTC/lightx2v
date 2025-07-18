import os
import numpy as np

from alt.utils.petrel_helper import global_petrel_helper


class RadarDecoder(object):
    def __init__(self, radar_path):
        self.radar_path = radar_path

        self.build()

    def build(self):
        metas = {}
        for item in global_petrel_helper.listdir(self.radar_path):
            if not item.endswith('.json'):
                continue
            timestamp_ns = int(item.split('.')[0])
            timestamp_ms = timestamp_ns // 1000 // 1000
            metas[timestamp_ms] = os.path.join(self.radar_path, item)
        
        self.metas = {k: metas[k] for k in sorted(metas)}
        self.timestamps = list(self.metas.keys())

    def get_nearst(self, timestamp_ms):
        timestamp_gaps = np.abs(np.array(self.timestamps) - timestamp_ms)
        select_index = timestamp_gaps.argmin()
        return self.metas[self.timestamps[select_index]], timestamp_gaps[select_index]



if __name__ == '__main__':
    path = 'ad_system_common:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-23/2024_03_23_08_06_26_gacGtParser/radar/front_left_radar'
    RadarDecoder(path).get(1711181203871)