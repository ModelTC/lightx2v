import sys
sys.path.insert(0, '/workspace/kongzelong2/auto-labeling-tools')
import copy
import os
import json
import click
import numpy as np
from alt import load
from alt.calibration.calibration import CalibrationAdapter
from alt.decoder.simple_location_decoder import SimpleLocationDecoder
from alt.sensebee.gop.merge_pcd_by_location import get_RT_from_enu_yaw
from alt.sensebee.tlsr.auto_quality.cal_valid import is_valid_anno
from alt.sensebee.base.base_process import BaseSenseBeeProcessor
from alt.sensebee.tlsr.al2bee_batch import uni_camera_timestamps
from alt.inference.odd import get_scene_infer_res
from alt.utils.env_helper import env
from alt.utils.petrel_helper import global_petrel_helper, expand_s3_path
from alt.utils.bee_helper import BeeHelper

def fill_scene(cur_bundle, scene_infer_res, scene_res_transform_dict):
    for sub_scene in ['location', 'period', 'weather']:
        cur_scene = scene_res_transform_dict.get(scene_infer_res[sub_scene]['label'], {})
        for k, v in cur_scene.items():
            cur_bundle['scene'][k][v] = True

def fill_calib(cur_bundle, cam_info, calibration):
    cur_bundle['camera_infos'][cam_info]['calibration_info']['intrin'] = calibration.intrinsic_new[cam_info].tolist()
    cur_bundle['camera_infos'][cam_info]['calibration_info']['cam_dist'] = calibration.intrinsic_dist[cam_info].tolist()
    cur_bundle['camera_infos'][cam_info]['calibration_info']['extrin'] = calibration.extrinsic[f'{cam_info}-to-car_center'].tolist()
    cur_bundle['camera_infos'][cam_info]['calib_path'] = calibration.config_path

def fill_location(location_decoder, cur_bundle, timestamp):
    if location_decoder is not None:
        t_error, location_device_t = location_decoder.search(timestamp * 1000 * 1000)
        if t_error < 10 * 1000 * 1000:  # 时间容许误差10ms
            cur_bundle['localization']['enu_x'] = location_device_t['location'][0]
            cur_bundle['localization']['enu_y'] = location_device_t['location'][1]
            cur_bundle['localization']['enu_z'] = location_device_t['location'][2]
            cur_bundle['localization']['enu_yaw'] = location_device_t['yaw']
            cur_bundle['localization']['pose'] = np.linalg.inv(get_RT_from_enu_yaw(location_device_t['location'], location_device_t['yaw'])).tolist()

def fill_other(cur_bundle, meta_info, case_name, sensebee_task):
    cur_bundle['vehicle_id'] = meta_info['data_structure']['info']['vehicle_name']
    cur_bundle['case_name'] = case_name[:19] + '-' + cur_bundle['vehicle_id']
    cur_bundle['sensebee_task'] = sensebee_task
    cur_bundle['raw_data'] = meta_info['data_structure']['camera']['center_camera_fov120']['video']  # 放了个fov120的源视频路径
    cur_bundle['autolabel_data'] = meta_info['data_annotation']

def get_cam_info(img_path):
    if 'center_camera_fov120' in img_path:
        return 'center_camera_fov120'
    elif 'center_camera_fov30' in img_path:
        return 'center_camera_fov30'
    else:
        AssertionError('only fov 30 and 120 are supported')


def process(local_anno_path, img_s3_root_path, pre_bee_input_txt, sensebee_task, output_path, mode):
    scene_res_transform_dict = {
        'daytime': {'light': 'Daytime'},
        'night_under_light': {'light': 'Night_with_streetlights'},
        'night_in_dark': {'light': 'Night_without_streetlights'},
        'tunnel': {'road_scene': 'Tunnel'},
        'overcast': {'weather': 'Cloudy'},
        'clear': {'weather': 'Sunny'},
        'rainy': {'weather': 'Rainy'},
        'foggy': {'weather': 'Foggy'},
        'snowy': {'weather': 'Snowy'},
        'other': {'weather': 'Other'}
    }
    bundle_template = {
        'timestamp': 0,
        'scene': {
            'pilot_park': {
                'Pilot': False,
                'Park': False
            },
            'weather': {
                'Other': False,
                'Sunny': False,
                'Rainy': False,
                'Cloudy': False,
                'Foggy': False,
                'Snowy': False
            },
            'road_scene': {
                'Other': False,
                'Urban_non_intersection': False,
                'Urban_intersection': False,
                'Highway': False,
                'Tunnel': False,
                'Updown_ramps': False,
                'Country': False,
                'Indoor_parkinglot': False,
                'Outdoor_parkinglot': False,
                'Roundabout': False
            },
            'light':{
                'Other': False,
                'Daytime': False,
                'Dusk_Early_morning': False,
                'Backlighting': False,
                'Night_with_streetlights': False,
                'Night_without_streetlights': False,
                'Indoor_with_light': False,
                'Indoor_without_light': False
            }
        },
        'camera_infos':{
            'center_camera_fov30': {
                'filename': '',
                'image_id': -1,
                'img_timestamp': 0,
                'image_width': 3840,
                'image_height': 2160,
                'calibration_info': {
                    'intrin': [],
                    'cam_dist': [],
                    'extrin': []
                },
                'calib_path': ''
            },
            'center_camera_fov120': {
                'filename': '',
                'image_id': -1,
                'img_timestamp': 0,
                'image_width': 3840,
                'image_height': 2160,
                'calibration_info': {
                    'intrin': [],
                    'cam_dist': [],
                    'extrin': []
                },
                'calib_path': ''
            }
        },
        'localization': {
            'enu_x': None,
            'enu_y': None,
            'enu_z': None,
            'enu_yaw': None,
            'longitude': None,
            'latitude': None,
            'altitude': None,
            'yaw': None,
            'pose': []
        },
        'vehicle_id': '',
        'case_name': '',
        'data_source': '',
        'anno_infos': {},
        'manual_refine': {
            'fov30': '0',
            'fov120': '0'
        },
        'sensebee_task': 0,
        'raw_data': '',
        'autolabel_data': ''
    }

    case2meta = {}
    for i in open(pre_bee_input_txt, 'r').readlines():
        case2meta[i.split('/')[-2]] = i.strip()

    scene_infer_res = get_scene_infer_res([line.strip() for line in open(pre_bee_input_txt, 'r').readlines()])

    meta_info_cache = {}
    final_res = {}
    bee_img_num = 0
    json_num = 0
    manual_metas = set()
    dumped_timestamps = {}
    for one_img_anno in load(os.path.join(local_anno_path, 'packagePageInfo.json')):
        bee_img_num += 1
        img_path = one_img_anno['path']

        if not os.path.exists(os.path.join(local_anno_path, img_path) + '.json'):
            continue
        json_num += 1

        timestamp = int(int(img_path.split('/')[-1].replace('.', '')[:19]) / 1000 / 1000)  # 对齐120和30采用13位时间戳
        case_name = max([i for i in img_path.split('/') if i.startswith('202')], key=len)
        if f'{timestamp}_{case_name}' not in final_res:
            cur_bundle = copy.deepcopy(bundle_template)
            cur_bundle['timestamp'] = timestamp * 1000 * 1000
            cam_info = get_cam_info(img_path)
            cur_bundle['camera_infos'][cam_info]['filename'] = os.path.join(img_s3_root_path, img_path)
            cur_bundle['camera_infos'][cam_info]['image_id'] = one_img_anno['id']
            cur_bundle['camera_infos'][cam_info]['img_timestamp'] = int(img_path.split('/')[-1].replace('.', '')[:19])

            meta_path = case2meta[case_name]
            manual_metas.add(meta_path)
            if meta_path not in meta_info_cache:  # meta不多，缓存下
                meta_info = BaseSenseBeeProcessor().safe_load_meta_json(meta_path)[0]
                calibration = CalibrationAdapter(config_path=meta_info['data_structure']['config'])
                try:
                    location_decoder = SimpleLocationDecoder(meta_info['data_structure']['location']['local'])
                except:
                    location_decoder = None
                meta_info_cache[meta_path] = {'meta_info': meta_info, 'calibration': calibration, 'location_decoder': location_decoder}
            else:
                meta_info = meta_info_cache[meta_path]['meta_info']
                calibration = meta_info_cache[meta_path]['calibration']
                location_decoder = meta_info_cache[meta_path]['location_decoder']

            # 场景部分
            fill_scene(cur_bundle, scene_infer_res[meta_path], scene_res_transform_dict)
            # 标定部分
            fill_calib(cur_bundle, cam_info, calibration)
            # 定位部分
            fill_location(location_decoder, cur_bundle, timestamp)
            # 杂项部分
            fill_other(cur_bundle, meta_info, case_name, sensebee_task)

            # 核心标注
            cur_bundle['anno_infos'][cam_info.replace('center_camera', 'anno_info_path')] = json.load(open(os.path.join(local_anno_path, img_path) + '.json', 'r'))  # 暂时标注内容保持sensebee格式
            cur_bundle['manual_refine'][cam_info.split('_')[-1]] = '1'

            final_res[f'{timestamp}_{case_name}'] = cur_bundle
            dumped_timestamps[case_name] = np.concatenate((dumped_timestamps.get(case_name, np.array([])), np.array([timestamp])))
        else:  # 已经有一个视角的结果
            cam_info = get_cam_info(img_path)
            cur_bundle = final_res[f'{timestamp}_{case_name}']
            cur_bundle['camera_infos'][cam_info]['filename'] = os.path.join(img_s3_root_path, img_path)
            cur_bundle['camera_infos'][cam_info]['image_id'] = one_img_anno['id']
            cur_bundle['camera_infos'][cam_info]['img_timestamp'] = int(img_path.split('/')[-1].replace('.', '')[:19])

            meta_path = case2meta[case_name]
            assert meta_path in meta_info_cache
            calibration = meta_info_cache[meta_path]['calibration']
            # 标定部分
            fill_calib(cur_bundle, cam_info, calibration)
            # 核心标注
            cur_bundle['anno_infos'][cam_info.replace('center_camera', 'anno_info_path')] = json.load(open(os.path.join(local_anno_path, img_path) + '.json', 'r'))  # 暂时标注内容保持sensebee格式
            cur_bundle['manual_refine'][cam_info.split('_')[-1]] = '1'

    for meta in manual_metas:  # 补上非人工标注真值生产结果
        meta_info = meta_info_cache[meta]['meta_info']
        gt_infos = {30: {}, 120: {}}  # 加载真值生产结果
        for fov in [30, 120]:
            for gt_cls in ['objattr', 'object']:
                tmp = {}
                frames = load(os.path.join(meta_info['data_annotation'], f'center_camera_fov{fov}#{gt_cls}-{mode}.pkl'))['frames']
                for frame in frames:
                    instances = []
                    for ins in frame[gt_cls + 's']:
                        instances.append({'label': ins['label'], 'bbox2d': ins['bbox2d'].tolist(), 'conf': ins['conf']})
                    tmp[int(frame['timestamp'] / 1000)] = {'filename': frame['filename'], 'instances': instances}
                gt_infos[fov][gt_cls] = tmp

        uni_timestamps = uni_camera_timestamps([gt_infos[30]['object'].keys(), gt_infos[120]['object'].keys()])
        new_uni_timestamps = []
        cls_timestamps = list(gt_infos[30]['objattr'].keys()) + list(gt_infos[120]['objattr'].keys())
        for t in uni_timestamps:  # 先检测联合时间戳，再必须某个视角有分类目标
            if t in cls_timestamps:
                new_uni_timestamps.append(t)
        uni_timestamps = new_uni_timestamps
        uni_timestamps.sort()

        case_name = max([i for i in meta.split('/') if i.startswith('202')], key=len)
        for timestamp in uni_timestamps:
            if f'{timestamp}_{case_name}' in final_res:  # 人工精修的帧
                continue
            else:  # 把预标注真值丢进去
                if np.min(np.abs(dumped_timestamps[case_name] - timestamp)) < 300:
                    continue  # 附近300ms已经有数据了，当前预标注丢弃
                else:
                    dumped_timestamps[case_name] = np.concatenate((dumped_timestamps[case_name], np.array([timestamp])))

                    cur_bundle = copy.deepcopy(bundle_template)
                    cur_bundle['timestamp'] = timestamp * 1000 * 1000
                    calibration = meta_info_cache[meta]['calibration']
                    location_decoder = meta_info_cache[meta]['location_decoder']
                    for fov in [30, 120]:
                        cam_info = f'center_camera_fov{fov}'
                        cur_bundle['camera_infos'][cam_info]['filename'] = gt_infos[fov]['object'][timestamp]['filename']
                        cur_bundle['camera_infos'][cam_info]['img_timestamp'] = int(cur_bundle['camera_infos'][cam_info]['filename'].split('/')[-1].replace('.', '')[:19])
                        # 标定部分
                        fill_calib(cur_bundle, cam_info, calibration)
                        # 核心标注
                        if timestamp in gt_infos[fov]['objattr']:
                            cur_bundle['anno_infos'][cam_info.replace('center_camera', 'anno_info_path')] = gt_infos[fov]['objattr'][timestamp]['instances']  # 暂时保持真值生产格式
                    # 场景部分
                    fill_scene(cur_bundle, scene_infer_res[meta], scene_res_transform_dict)
                    # 定位部分
                    fill_location(location_decoder, cur_bundle, timestamp)
                    # 杂项部分
                    fill_other(cur_bundle, meta_info, case_name, sensebee_task)

                    final_res[f'{timestamp}_{case_name}'] = cur_bundle

    with open(os.path.join(output_path, f'{sensebee_task}_output.jsonl'), 'w') as fw:
        valid_num = 0
        pure_annoed_bundle_num = 0
        union_dict = {}
        for timestamp_case, one_bundle in final_res.items():
            if one_bundle['manual_refine']['fov120'] == '1' and one_bundle['manual_refine']['fov30'] == '1':
                pure_annoed_bundle_num += 1
            for anno in one_bundle['anno_infos'].values():
                if 'step_1' not in anno:  # 非人工标注
                    continue
                if is_valid_anno(anno):
                    valid_num += 1
                    union_dict[timestamp_case] = union_dict.get(timestamp_case, 0) + 1
            fw.write(json.dumps(one_bundle) + '\n')
        print(f'bee_img: {bee_img_num}')
        print(f'json_num: {json_num}')
        print(f'valid_annoed_img: {valid_num}...............')
        print(f'all_bundle_num: {len(final_res)}...............')
        print(f'pure_annoed_bundle_num: {pure_annoed_bundle_num}')
        print(f'valid_bundle: {len(union_dict)}...............')
    global_petrel_helper.save_jsonl(f'ad_system_common_auto:s3://sdc3-gt-label-2/data.infra.gt_labels/tlsr_deliver/{mode[:2]}/sensebee/{sensebee_task}/{sensebee_task}_output.jsonl', list(final_res.values()))
    print(f'ad_system_common_auto:s3://sdc3-gt-label-2/data.infra.gt_labels/tlsr_deliver/{mode[:2]}/sensebee/{sensebee_task}/{sensebee_task}_output.jsonl')


@click.command()
@click.option('-a', '--local_anno_path', default='/workspace/kongzelong2/pillar_test/2407/0717/24238post')
@click.option('--img_s3_root_path', default='ad_system_common_auto:s3://sdc3-gt-label-2')
@click.option('--pre_bee_input_txt', default='/workspace/kongzelong2/pillar_test/2407/0717/24238post/input.tmp.txt')
@click.option('--sensebee_task', default=24238)
@click.option('--mode', default='tlr')
@click.option('-o', '--output_path', default='/workspace/kongzelong2/pillar_test/2407/0717/tlsr_post_non_key/')
def main(local_anno_path, img_s3_root_path, pre_bee_input_txt, sensebee_task, output_path, mode):
    process(local_anno_path, img_s3_root_path, pre_bee_input_txt, sensebee_task, output_path, mode)


if __name__ == '__main__':
    ids = [24595, 24477, 24477, 24477, 24477, 24477]
    modes = ['tsr', 'tlr']
    input_txt = '/workspace/kongzelong2/pillar_test/2407/0725/24595post/input.tmp.txt'
    root_path = '/workspace/kongzelong2/pillar_test/2407/0726/new_post'


    for i, (id, mode) in enumerate(zip(ids, modes)):
        if not env.is_my_showtime(i):
            continue
        local_anno_path = os.path.join(root_path, str(id))
        bee = BeeHelper(sensebee_id=id)
        img_s3_root_path = expand_s3_path(bee.root)
        zip_file = bee.get_result(save_path=local_anno_path, skip_exist=True)
        os.system(f'cd {local_anno_path}; unzip {zip_file}')

        process(local_anno_path, img_s3_root_path, input_txt, id, local_anno_path, mode)
        print(os.path.join(local_anno_path, f'{id}_output.jsonl').replace('/workspace', 'http://10.4.196.77:8081'))
