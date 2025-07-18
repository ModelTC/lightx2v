import sys
sys.path.insert(0, '/workspace/kongzelong2/auto-labeling-tools')
import os
import json
import cv2
import math
import shutil
import csv
import mpi4py
from alt.utils.petrel_helper import global_petrel_helper, expand_s3_path
from multiprocessing import Pool
from alt.utils.bee_helper import BeeHelper
from alt.utils.env_helper import env


def save_crop_img(ind, one_json, package_info, attr_num2name, det_num2name, img_root, vis_path, mode):
    if not one_json['valid']:
        return None
    package_page = str(package_info[ind]['packageID']) + '_' + str(package_info[ind]['page'])
    img_path = os.path.join(img_root, package_info[ind]['path'])
    img = global_petrel_helper.imread(img_path)
    id2bbox = {}
    for det in one_json['step_1']['result']:
        if not det['valid']:
            continue
        id2bbox[det['id']] = [int(det['x']), int(det['y']), int(det['width']), int(det['height'])]
        if mode == 'tlr':
            cv2.imwrite(os.path.join(vis_path, '关注度', det_num2name[det['attribute']], '{}_{}.jpg'.format(package_page, det['id'])), img[int(det['y']):int(det['y'] + det['height']), int(det['x']):int(det['x'] + det['width'])])
    if 'step_3' not in one_json:
        return None
    for attr_ins in one_json['step_3']['result']:
        if attr_ins['sourceID'] not in id2bbox:
            continue
        for attr, value in attr_ins['result'].items():
            cls = attr_num2name[attr][value]
            bbox = id2bbox[attr_ins['sourceID']]
            if cls == '其他' or cls == '易混淆类':
                continue
            if cls == '看不清' and bbox[2] < 20 and bbox[3] < 20:  # 看不清太小的不要
                continue
            os.makedirs(os.path.join(vis_path, attr, cls), exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, attr, cls, '{}_{}.jpg'.format(package_page, attr_ins['id'])), img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]])

def crop_vis(package_path, label_path, img_root, vis_path, det_dict_path, attr_dict_path, mode, process_num=10):
    p = Pool(process_num)
    attr_num2name = {}
    name = json.load(open(attr_dict_path, 'r'))
    for one_attr in name:
        num2name = {}
        for i in one_attr['subSelected']:
            num2name[i['value']] = i['key']
        attr_num2name[one_attr['value']] = num2name

    det_num2name = {}
    if mode == 'tlr':
        name = json.load(open(det_dict_path, 'r'))
        for one_det in name:
            det_num2name[one_det['value']] = one_det['key']
            os.makedirs(os.path.join(vis_path, '关注度', one_det['key']), exist_ok=True)

    package_info = json.load(open(package_path, 'r'))
    for ind, one_info in enumerate((package_info)):
        if os.path.exists(os.path.join(label_path, one_info['path'] + '.json')):
            one_json = json.load(open(os.path.join(label_path, one_info['path'] + '.json')))
            p.apply_async(save_crop_img, (ind, one_json, package_info, attr_num2name, det_num2name, img_root, vis_path, mode))
            # save_crop_img(ind, one_json, package_info, attr_num2name, det_num2name, img_root, vis_path, mode)
    p.close()
    p.join()

    if mode == 'tsr':
        imgs_path = os.path.join(vis_path, 'Label')
        cates = os.listdir(imgs_path)
        for cate in cates:
            imgs = os.listdir(os.path.join(imgs_path, cate))
            if len(imgs) <= 2000:
                continue
            else:
                new_folder_num = math.ceil(len(imgs) / 2000) - 1
                for folder_ind in range(new_folder_num):
                    os.makedirs(os.path.join(imgs_path, f'{cate}_{folder_ind}'), exist_ok=True)
                    for img in imgs[2000 * (folder_ind + 1): 2000 * (folder_ind + 2)]:
                        shutil.move(os.path.join(imgs_path, cate, img), os.path.join(imgs_path, f'{cate}_{folder_ind}', img))


def is_digit(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, str) and x.isdigit():
        return True
    else:
        return False

def check_track_id(img_path, one_json, mode):
    if not one_json['valid']:
        return []
    no_tracks = []
    if mode == 'tlr':
        for det in one_json['step_1']['result']:
            if det['attribute'] == 'confused':
                continue
            if 'textAttribute' not in det or not is_digit(det['textAttribute']):
                no_tracks.append([img_path, (int(det['x'] + det['width'] / 2), int(det['y'] + det['height'] / 2)), '漏标id'])
    elif mode == 'tsr':
        no_track_ids = []
        id2bbox = {}
        for det in one_json['step_1']['result']:
            if 'textAttribute' not in det or not is_digit(det['textAttribute']):
                no_track_ids.append(det['id'])
            id2bbox[det['id']] = (int(det['x'] + det['width'] / 2), int(det['y'] + det['height'] / 2))
        if 'step_3' not in one_json:
            return []
        for attr in one_json['step_3']['result']:
            if attr['result'] == {}:
                no_tracks.append([img_path, id2bbox[attr['sourceID']], '漏标属性类别'])
                continue
            if attr['result']['Label'] == '78' or attr['result']['Label'] == '9999':
                continue
            if attr['sourceID'] in no_track_ids:
                no_tracks.append([img_path, id2bbox[attr['sourceID']], '漏标id'])

    return no_tracks

def find_no_id(package_path, label_path, save_path, mode):
    package_info = json.load(open(package_path, 'r'))
    todump = []
    for one_json in package_info:
        img_path = one_json['path']
        if os.path.exists(os.path.join(label_path, img_path + '.json')):
            one_json = json.load(open(os.path.join(label_path, img_path + '.json'), 'r'))
            todump.extend(check_track_id(img_path, one_json, mode))
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for row in todump:
            writer.writerow(row)


if __name__ == '__main__':

    ids = [24704, 24791]
    process_num = 8
    root_save_path = '/workspace/kongzelong2/pillar_test/2407/0726/new_vis'

    to_print = []
    for i, id in enumerate(ids):
        if not env.is_my_showtime(i):
            continue
        label_path = os.path.join(root_save_path, str(id))
        os.makedirs(label_path, exist_ok=True)
        bee = BeeHelper(sensebee_id=id)
        img_root = expand_s3_path(bee.root)
        zip_file = bee.get_result(save_path=label_path, skip_exist=True)
        os.system(f'cd {label_path}; unzip {zip_file}')
        vis_path = os.path.join(label_path, 'imgs')
        # det_dict_path = '/workspace/kongzelong2/pillar_test/240531/tlr22059/tlr_focus.json'
        # attr_dict_path = '/workspace/kongzelong2/pillar_test/240531/tlr22059/tlr.json'
        # mode = 'tlr'
        det_dict_path = ''
        attr_dict_path = '/workspace/kongzelong2/pillar_test/2407/0717/24242vis/tsr_attr_v6.4.json'
        mode = 'tsr'
        save_path = os.path.join(label_path, f'{id}_no_trackid.csv')
        package_path = os.path.join(label_path, 'packagePageInfo.json')
        crop_vis(package_path, label_path, img_root, vis_path, det_dict_path, attr_dict_path, mode, process_num=process_num)
        find_no_id(package_path, label_path, save_path, mode)
        os.system(f'cd {label_path}; zip -r {id}.zip imgs')
        to_print.append(os.path.join(label_path.replace('/workspace', 'http://10.4.196.77:8081'), f'{id}.zip'))
        to_print.append(os.path.join(label_path.replace('/workspace', 'http://10.4.196.77:8081'), f'{id}_no_trackid.csv'))
    env.barrier()
    to_print = env.gather(to_print)
    if env.is_master():
        for i in to_print:
            print(i)
    env.barrier()
