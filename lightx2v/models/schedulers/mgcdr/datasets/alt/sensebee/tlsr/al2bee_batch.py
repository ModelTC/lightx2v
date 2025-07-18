import os
import copy
import random
import string
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from alt.utils.petrel_helper import global_petrel_helper, expand_s3_path
from alt import load_autolabel_meta_json
from alt import load
from alt.decoder.simple_location_decoder import SimpleLocationDecoder
from alt.sensebee.base.base_process import BaseSenseBeeProcessor


tlr_model2bee = {
    'Black': {'color': 4, 'shape': 19},
    'Enter': {'color': 2, 'shape': 14},
    'G-C': {'color': 2, 'shape': 4},
    'G-L': {'color': 2, 'shape': 2},
    'G-L-UTurn': {'color': 2, 'shape': 7},
    'G-R': {'color': 2, 'shape': 3},
    'G-R-T': {'color': 2, 'shape': 9},
    'G-T': {'color': 2, 'shape': 1},
    'G-UTurn': {'color': 2, 'shape': 6},
    'Number': {'shape': 10, 'have_number': 0},
    'Others': {'have_number': 1},
    'R-C': {'color': 1, 'shape': 4},
    'R-L': {'color': 1, 'shape': 2},
    'R-R': {'color': 1, 'shape': 3},
    'R-T': {'color': 1, 'shape': 1},
    'R-UTurn': {'color': 1, 'shape': 6},
    'X-Enter': {'color': 1, 'shape': 11},
    'Y-C': {'color': 3, 'shape': 4},
    'Y-L': {'color': 3, 'shape': 2},
    'Y-R': {'color': 3, 'shape': 3},
    'Y-T': {'color': 3, 'shape': 1},
    'Y-UTurn': {'color': 3, 'shape': 6}
}
tlr_attr_temp = {
                'color': '5',  # 其他
                'shape': '17',  # 其他
                'have_number': '1',  # 无数字
                'boader_type': '0',  # 0-底座框
                'boader_quality': '100',  # 灯框质量100%
                'shape_quality': '100'  # 类别质量100%
                }

tsr_model2bee = {
    'ElecSpeed': {'Label': 17},
    'Min_speed_70': {'Label': 36},
    'Min_speed_100': {'Label': 39},
    'Min_speed_110': {'Label': 40},
    'Min_speed_60': {'Label': 35},
    'Min_speed_80': {'Label': 37},
    'Min_speed_90': {'Label': 38},
    'Min_speed_50': {'Label': 34},
    'bus-lane': {'Label': 72},
    'construction': {'Label': 62},
    'crosswalk': {'Label': 64},
    'diversion': {'Label': 63},
    'lane-merge': {'Label': 61},
    'left-U-turn-lane': {'Label': 71},
    'left-straight-turn-lane': {'Label': 69},
    'left-turn-lane': {'Label': 67},
    'lift-10': {'Label': 19},
    'lift-100': {'Label': 31},
    'lift-110': {'Label': 32},
    'lift-120': {'Label': 33},
    'lift-20': {'Label': 21},
    'lift-30': {'Label': 23},
    'lift-40': {'Label': 25},
    'lift-50': {'Label': 26},
    'lift-60': {'Label': 27},
    'lift-70': {'Label': 28},
    'lift-80': {'Label': 29},
    'lift-90': {'Label': 30},
    'lift-pass': {'Label': 42},
    'merge-left': {'Label': 57},
    'merge-right': {'Label': 58},
    'notice-child': {'Label': 60},
    'notice-ped': {'Label': 59},
    'others': {'Label': 9999},
    'ramp': {'Label': 74},
    'right-straight-turn-lane': {'Label': 70},
    'right-turn-lane': {'Label': 68},
    'slow-down': {'Label': 47},
    'speed-10': {'Label': 2},
    'speed-100': {'Label': 14},
    'speed-110': {'Label': 15},
    'speed-120': {'Label': 16},
    'speed-15': {'Label': 3},
    'speed-20': {'Label': 4},
    'speed-30': {'Label': 6},
    'speed-40': {'Label': 8},
    'speed-5': {'Label': 1},
    'speed-50': {'Label': 9},
    'speed-60': {'Label': 10},
    'speed-70': {'Label': 11},
    'speed-80': {'Label': 12},
    'speed-90': {'Label': 13},
    'stop': {'Label': 46},
    'turn-left': {'Label': 67},
    'turn-right': {'Label': 68},
    'uturn-lane': {'Label': 66},
    'x-Uturn': {'Label': 55},
    'x-enter': {'Label': 49},
    'x-height': {'Label': 43},
    'x-left': {'Label': 52},
    'x-parking': {'Label': 48},
    'x-pass': {'Label': 41},
    'x-right': {'Label': 53},
    'x-weight': {'Label': 44},
    'x-whistle': {'Label': 56},
    'lift-5': {'Label': 18},
}
tsr_attr_temp = {
                'Label': '9999'  # 其他
                }


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


class TLSRPreAnnoPreProcessor():
    def __init__(self) -> None:
        pass

    def prepare_autolabel_meta(self, meta_path, mode):
        # 加载真值生产结果
        self.mode = mode
        self.meta_json = BaseSenseBeeProcessor().safe_load_meta_json(meta_path)[0]
        self.cls_metas = {}
        self.cls_metas[120] = load(os.path.join(self.meta_json['data_annotation'], f'center_camera_fov120#objattr-{mode}.pkl'))
        self.cls_metas[30] = load(os.path.join(self.meta_json['data_annotation'], f'center_camera_fov30#objattr-{mode}.pkl'))
        self.det_metas = {}
        self.det_metas[120] = load(os.path.join(self.meta_json['data_annotation'], f'center_camera_fov120#object-{mode}.pkl'))
        self.det_metas[30] = load(os.path.join(self.meta_json['data_annotation'], f'center_camera_fov30#object-{mode}.pkl'))

    def clip_filter(self, keep_empty):
        # 过滤目标太少的clip
        filter_thres = {'tlr': 50, 'tsr': 50}
        if len(self.cls_metas[120]['frames']) + len(self.cls_metas[30]['frames']) < filter_thres[self.mode]:
            print(f'30&120总有效帧太少，舍弃当前批次')
            print(len(self.cls_metas[120]['frames']) + len(self.cls_metas[30]['frames']))
            return False

        # 求出120,30联合时间戳，要求严格对齐
        timestamps_120 = [int(i['timestamp'] / 1000) for i in self.det_metas[120]['frames']]
        timestamps_30 = [int(i['timestamp'] / 1000) for i in self.det_metas[30]['frames']]
        self.uni_timestamps = uni_camera_timestamps([timestamps_120, timestamps_30])
        if len(self.uni_timestamps) < 25:  # 对齐数量太少，直接舍弃
            print('30, 120联合时间戳太少，舍弃当前批次')
            print(len(self.uni_timestamps))
            return False
        if not keep_empty:
            new_uni_timestamps = []
            cls_timestamps = [int(i['timestamp'] / 1000) for i in self.cls_metas[30]['frames']] + [int(i['timestamp'] / 1000) for i in self.cls_metas[120]['frames']]
            for t in self.uni_timestamps:
                if t in cls_timestamps:
                    new_uni_timestamps.append(t)
            self.uni_timestamps = new_uni_timestamps
        self.uni_timestamps.sort()
        return True

    def prepare_bundle_priority(self):
        if self.mode == 'tsr':
            self.cls_p0 = ['speed-90', 'speed-100', 'speed-110', 'speed-120', 'speed-5', 'speed-10', 'speed-15', 'speed-20', 'speed-25', 'speed-30', 'speed-35', 'speed-40']
            self.cls_p0.extend(['lift-90', 'lift-100', 'lift-110', 'lift-120', 'lift-50', 'lift-60', 'lift-70', 'lift-80', 'lift-5', 'lift-10', 'lift-15', 'lift-20', 'lift-25', 'lift-30', 'lift-35', 'lift-40'])
            self.cls_p0.extend(['x-height', 'x-weight'])
            self.cls_p1 = ['speed-50', 'speed-60', 'speed-70', 'speed-80']
            self.cls_p1.extend(['Min_speed_50', 'Min_speed_60', 'Min_speed_70', 'Min_speed_80', 'Min_speed_90', 'Min_speed_100', 'Min_speed_110'])
            self.cls_p1.extend(['x-whistle', 'x-parking', 'x-enter', 'x-pass', 'x-left', 'x-right'])
        elif self.mode == 'tlr':
            self.cls_p0 = ['Y-C', 'Y-L', 'Y-R', 'Y-T', 'Y-UTurn']
            self.cls_p0.extend(['G-L-UTurn', 'G-R-T']),
            self.cls_p0.extend(['G-L', 'R-L', 'G-R', 'R-R', 'G-UTurn', 'R-UTurn'])
            self.cls_p0.extend(['Number', 'Enter', 'X-Enter'])
            self.cls_p1 = ['G-T', 'R-T']
        self.drop_prob = {'tsr': {'p0': 1.0, 'p1': 0.8, 'p2': 0.1}, 'tlr': {'p0': 1.0, 'p1': 0.8, 'p2': 0.1}}
        self.timestamp_to_cls_res = {}
        for res in self.cls_metas.values():
            for i in res['frames']:
                labels = []
                for j in i['objattrs']:
                    labels.append(j['label'])
                self.timestamp_to_cls_res[int(i['timestamp'] / 1000)] = self.timestamp_to_cls_res.get(int(i['timestamp'] / 1000), []) + labels

    def get_bundle_priority(self, timestamp):
        ins = self.timestamp_to_cls_res.get(timestamp, [])
        for i in self.cls_p0:
            if i in ins:
                return 'p0'
        for i in self.cls_p1:
            if i in ins:
                return 'p1'
        return 'p2'

    def sample_timestamp(self):
        if self.mode == 'tsr':  # 仅tsr才需要过滤掉停着的帧
            location_decoder = SimpleLocationDecoder(self.meta_json['data_structure']['location']['local'])
        self.to_label_timestamps = [self.uni_timestamps[0]]  # 实际送标时间戳
        t_pre = self.uni_timestamps[0]
        for t in self.uni_timestamps[1:]:
            if  t - t_pre < 300:  # 确保两帧之间间隔不小于300ms
                continue
            if self.mode == 'tsr':
                error, location = location_decoder.search(t * 1000 * 1000)
                if error < 10 * 1000 * 1000 and abs(location['speed'][0]) + abs(location['speed'][1]) < 0.1:
                    continue  # 如果东向，北向，速度绝对值之和小于0.1m/s，则认为静止，不送标
            prio = self.get_bundle_priority(t)
            if random.random() > self.drop_prob[self.mode][prio]:
                continue
            self.to_label_timestamps.append(t)
            t_pre = t

    def process(self, meta_paths, pesudo_root_path, width, height, fovs, mode, keep_empty=False):
        all_imgs = []
        for meta_path in meta_paths:

            self.prepare_autolabel_meta(meta_path, mode)

            if not self.clip_filter(keep_empty):
                continue

            self.prepare_bundle_priority()

            self.sample_timestamp()

            for fov in fovs:  # 30和120必定联合送标
                bee_temp = {'width': width, 'height': height, 'valid': True, 'rotate': 0,
                            'step_1': {'toolName': 'rectTool', 'dataSourceStep': 0, 'result': []},
                            'step_3': {'toolName': 'tagTool', 'dataSourceStep': 1, 'result': []}}
                metas = self.cls_metas[fov]
                not_empty_ts = set()
                for i in metas['frames']:
                    if int(i['timestamp'] / 1000) not in self.to_label_timestamps:
                        continue
                    pesudo_meta = copy.deepcopy(bee_temp)
                    for object in i['objattrs']:
                        ins_det = {'valid': True}
                        if mode == 'tlr':
                            ins_det['attribute'] = 'attention_current'  # 默认为关注灯_当前，让人工标注
                        elif mode == 'tsr':
                            ins_det['attribute'] = 'tsr'  # tsr检测框不分类别，都是tsr
                        ins_det['x'] = object['bbox2d'][0]
                        ins_det['y'] = object['bbox2d'][1]
                        ins_det['width'] = object['bbox2d'][2] - object['bbox2d'][0]
                        ins_det['height'] = object['bbox2d'][3] - object['bbox2d'][1]
                        ins_det['id'] = ''.join(random.sample(string.digits, 8))
                        pesudo_meta['step_1']['result'].append(ins_det)

                        ins_attr = {}
                        ins_attr['sourceID'] = ins_det['id']
                        ins_attr['id'] = ''.join(random.sample(string.digits, 8))
                        if mode == 'tlr':  # tlr有6种属性，通过tlr_model2bee完成真值模型到部分属性的映射
                            attr_temp = copy.deepcopy(tlr_attr_temp)
                            for k, v in tlr_model2bee[object['label']].items():
                                attr_temp[k] = str(v)
                        elif mode == 'tsr':  # tsr就一个属性，模型是66类，需要映射到需求的80类
                            attr_temp = copy.deepcopy(tsr_attr_temp)
                            for k, v in tsr_model2bee[object['label']].items():
                                attr_temp[k] = str(v)
                        ins_attr['result'] = attr_temp
                        pesudo_meta['step_3']['result'].append(ins_attr)

                    not_empty_ts.add(int(i['timestamp'] / 1000))
                    pesudo_json_path = expand_s3_path(os.path.join(pesudo_root_path, '/'.join(i['filename'].split('/')[3:]) + '.json'))
                    global_petrel_helper.save_json(pesudo_json_path, pesudo_meta)
                    all_imgs.append('/'.join(i['filename'].split('/')[3:]))

                metas = self.det_metas[fov]
                for i in metas['frames']:
                    if int(i['timestamp'] / 1000) not in self.to_label_timestamps:  # 丢弃
                        continue
                    if int(i['timestamp'] / 1000) in not_empty_ts:  # 单单根据检测结果无法判断是否是空图，因为检测结果里有很低阈值的，需要查询not_empty_ts判断
                        continue
                    pesudo_json_path = expand_s3_path(os.path.join(pesudo_root_path, '/'.join(i['filename'].split('/')[3:]) + '.json'))
                    global_petrel_helper.save_json(pesudo_json_path, bee_temp)
                    all_imgs.append('/'.join(i['filename'].split('/')[3:]))
        i = self.det_metas[30]['frames'][0]
        img_root_path = '/'.join(i['filename'].split('/')[:3])
        pesudo_label_path = expand_s3_path(pesudo_root_path)

        return all_imgs, img_root_path, pesudo_label_path


if __name__ == '__main__':
    all_imgs, img_root_path, pesudo_label_path = TLSRPreAnnoPreProcessor().process(meta_paths=[
        's3://sdc_gt_label/GAC/autolabel_20240430_891/Data_Collection/GT_data/gacGtParser/drive_gt_collection/120m_TLR/2024-04/A02-290/2024-04-26/2024_04_26_03_02_01_gacGtParser/meta.json',
        's3://sdc_gt_label/GAC/autolabel_20240430_891/Data_Collection/GT_data/gacGtParser/drive_gt_collection/release_speed_limit_sign/2024-04/A02-290/2024-04-26/2024_04_26_03_03_13_gacGtParser/meta.json'
    ],
                width=3840,
                height=2160,
                fovs=[30, 120],
                mode='tsr',
                keep_empty=True)
    with open('/workspace/kongzelong2/pillar_test/2405/try_batch_bee/filelist.tmp.txt', 'w') as fw:
        fw.write('\n'.join(all_imgs))
    print('img_root_path')
    print(img_root_path)
    print('pesudo_label_path')
    print(pesudo_label_path)
