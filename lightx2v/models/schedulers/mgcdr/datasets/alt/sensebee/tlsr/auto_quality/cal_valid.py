import os
import json
from alt import load


def is_valid_anno(anno):
    if not anno['valid']:
        return False
    elif 'step_1' not in anno:
        return False
    elif 'step_3' not in anno:
        return False
    valid = False
    for i in anno['step_1']['result']:
        valid = valid or i['valid']
    return valid


def main():
    package_path = '/workspace/kongzelong2/pillar_test/2406/0619/cal_invalid/22283/packagePageInfo.json'
    root_path = '/workspace/kongzelong2/pillar_test/2406/0619/cal_invalid/22283'

    bee_img_num = 0
    json_num = 0
    valid_num = 0
    union_dict = {}
    for one_img_anno in load(package_path):
        bee_img_num += 1
        img_path = one_img_anno['path']
        if not os.path.exists(os.path.join(root_path, img_path) + '.json'):
            continue
        anno = json.load(open(os.path.join(root_path, img_path) + '.json', 'r'))
        json_num += 1

        if is_valid_anno(anno):
            valid_num += 1
            timestamp = int(int(img_path.split('/')[-1].replace('.', '')[:19]) / 1000 / 1000)
            union_dict[timestamp] = union_dict.get(timestamp, 0) + 1

    print(f'bee_img: {bee_img_num}')  # 上传了多少张图片到bee上
    print(f'json_num: {json_num}')  # 有多少图片真的有个标注json，有些图会直接安排给运营，跳过了不标注
    print(f'image_valid: {valid_num}')  # 图片级有效数量，去掉了全图无效，或者里面没有有效目标的
    print(f'bundle_valid: {len(union_dict)}')  # 联合有效bundle数量，图片级有效再进一步合并同一时间戳的30和120的


if __name__ == '__main__':
    main()
