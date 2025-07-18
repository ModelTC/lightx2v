import sys
sys.path.insert(0, '/workspace/kongzelong2/auto-labeling-tools')
import json
import os
import csv
from alt.utils.bee_helper import BeeHelper

def process(task, label_path, ori_label_path):
    package_path = os.path.join(label_path, 'packagePageInfo.json')
    save_path = os.path.join(label_path, f'{task}_vis_check.csv')
    ori_package_path = os.path.join(ori_label_path, 'packagePageInfo.json')
    package_page2img = {}
    all_annos_ori = json.load(open(ori_package_path, 'r'))
    for anno in all_annos_ori:
        package_page2img['{}_{}'.format(anno['packageID'], anno['page'])] = anno['path']

    reason_dict = {0: '小图可视化发现类别错误', 2: '小图可视化发现疑似类别错误'}
    todump = []
    all_annos = json.load(open(package_path, 'r'))
    for anno in all_annos:
        img_path = anno['path']
        if int(img_path.split('/')[0]) != task:
            continue
        ori_info = img_path.split('/')[-1].split('.')[0]
        ori_package_page = ori_info.rsplit('_', 1)[0]
        ori_id_attr = ori_info.rsplit('_', 1)[1]
        if os.path.exists(os.path.join(label_path, img_path + '.json')):
            one_json = json.load(open(os.path.join(label_path, img_path + '.json'), 'r'))
            valid = int(one_json['step_1']['result'][0]['filterLabel'])
            if valid == 1:
                continue
            elif valid == 0 or valid == 2:
                ori_anno_json = json.load(open(os.path.join(ori_label_path, package_page2img[ori_package_page] + '.json'), 'r'))
                det_id2bbox = {}
                for det in ori_anno_json['step_1']['result']:
                    det_id2bbox[det['id']] = (int(det['x'] + det['width'] / 2), int(det['y'] + det['height'] / 2))
                attr_id2det_id = {}
                for attr in ori_anno_json['step_3']['result']:
                    attr_id2det_id[attr['id']] = attr['sourceID']
                if valid == 2:
                    if '限' not in img_path:
                        continue
                todump.append([package_page2img[ori_package_page], det_id2bbox[attr_id2det_id[ori_id_attr]], reason_dict[valid]])

    with open(save_path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        for row in todump:
            writer.writerow(row)


if __name__ == '__main__':
    tasks = [24632, 24705]
    ori_label_paths = [
        '/workspace/kongzelong2/pillar_test/2407/0724/24632vis',
        '/workspace/kongzelong2/pillar_test/2407/0724/24705vis'
    ]
    second_id = 24912
    root_path = '/workspace/kongzelong2/pillar_test/2407/0726/new_comm'


    bee = BeeHelper(sensebee_id=second_id)
    zip_file = bee.get_result(save_path=root_path, skip_exist=True)
    os.system(f'cd {root_path}; unzip {zip_file}')

    for task, ori_label_path in zip(tasks, ori_label_paths):
        process(task, root_path, ori_label_path)
        print(os.path.join(root_path.replace('/workspace', 'http://10.4.196.77:8081'), f'{task}_vis_check.csv'))
