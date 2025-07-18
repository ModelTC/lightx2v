import click
import os
import random
from loguru import logger
from alt.inference.odd import get_scene_infer_res
from alt.sensebee.tlsr.al2bee_batch import TLSRPreAnnoPreProcessor

def scene_filter(scene_info, mode):
    mode2prob = {'tsr': 0.6, 'tlr': 0.3}
    location_p0 = ['close_road', 'tunnel']
    period_p0 = ['night_under_light', 'night_in_dark']
    weather_p0 = ['rainy', 'foggy', 'snowy', 'other']
    location = scene_info['location']['label']
    if location in location_p0:  # 有重要场景必定送标
        return True
    period = scene_info['period']['label']
    if period in period_p0:
        return True
    weather = scene_info['weather']['label']
    if weather in weather_p0:
        return True
    if random.random() > mode2prob[mode]:  # 普通场景以特定概率送标
        return True
    else:
        return False


@click.command()
@click.option('-i', '--input_gt_txt', default='/workspace/kongzelong2/pillar_test/2407/0717/tlsr_scene_filter/input.tmp.txt')
@click.option('--pesudo_root_path', default='s3://sdc3-gt-label-2/data.infra.gt_labels/tsr/sensebee/autolabel-debug')
@click.option('--width', default=3840)
@click.option('--height', default=2160)
@click.option('--fovs', multiple=True, default=(30, 120))
@click.option('-m', '--mode', default='tsr')
@click.option('--keep_empty', default=False)
@click.option('--filter_night_hour', default=0)
@click.option('-o', '--to_label_txt', default='/workspace/kongzelong2/pillar_test/2407/0717/tlsr_scene_filter/out.tmp.txt')
def main(input_gt_txt, pesudo_root_path, width, height, fovs, mode, keep_empty, filter_night_hour, to_label_txt):
    per_task_num = {'tsr': 10000, 'tlr': 5000}
    with open(input_gt_txt, 'r') as fr:
        all_lines = fr.readlines()

    scene_infer_res = get_scene_infer_res([line.strip() for line in all_lines])

    to_label = []
    to_label_group = []
    for ind, line in enumerate(all_lines):
        print(f'processing {ind + 1}/{len(all_lines)}')
        if len(to_label) > per_task_num[mode]:
            to_label_group.append(to_label)
            to_label = []

        time_str = max([i for i in line.split('/') if i.startswith('202')], key=len)
        time_str = int(time_str.split('_')[3])
        if time_str < filter_night_hour and time_str > filter_night_hour - 14:
            logger.info(f'not night data, drop data')
            continue

        if not scene_filter(scene_infer_res[line.strip()], mode):
            logger.info(f'not important scene, drop data')
            continue

        try:
            all_imgs, img_root_path, pesudo_label_path = TLSRPreAnnoPreProcessor().process(meta_paths=[line.strip()],
                        pesudo_root_path=pesudo_root_path,
                        width=width,
                        height=height,
                        fovs=list(fovs),
                        mode=mode,
                        keep_empty=keep_empty)
            if len(all_imgs) == 0:
                continue
            to_label.extend(all_imgs)
        except Exception as e:
            logger.error(e)
    if len(to_label) > 2000:
        to_label_group.append(to_label)
    else:
        to_label_group[-1].extend(to_label)
    for i, to_label in enumerate(to_label_group):
        with open(to_label_txt.replace('.tmp.txt', f'_{i}.tmp.txt'), 'w') as fw:
            fw.write('\n'.join(to_label))
    print(img_root_path)
    print(pesudo_label_path)

if __name__ == '__main__':
    main()
