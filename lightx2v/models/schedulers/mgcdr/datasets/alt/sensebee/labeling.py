# Import from third library
import os
import click
import json
import copy
import zipfile
from tqdm import tqdm
from loguru import logger

# Import from alt
from alt import load, dump, smart_exists, smart_copy
from alt.utils.file_helper import dump_json_lines, ads_cli_upload_folder
from alt.utils.env_helper import env
from alt.quality.quality_inspection import ObjectQualityInspection, GopQualityInspection
from alt.sensebee.pvb.utils.processor import PVBPreAnnoPostProcessor, PVBPreAnnoPreProcessor
from alt.sensebee.gop.utils.processor import GOPPreAnnoPostProcessor, GOPPreAnnoPreProcessor
from alt.sensebee.base.utils import split_group_labels_clip_num, merge_pre_sensebee

from alt.visualize.object.sensebee_bev import SenseBeeBEV
from alt.utils.petrel_helper import global_petrel_helper, expand_s3_path
from alt.utils.bee_helper import BeeHelper, BeeUploader
from alt.evaluation.object_evaluator_v2 import ObjectEvaluatorV2
from alt.apis.autolabel_platform import AutolabelerPlatformHelper


@click.group()
def cli():
    pass


def parse_meta_json(input_path, save_path):
    if input_path.endswith('.tmp.txt'):
        meta_jsons = [item.strip() for item in open(input_path).readlines()]
    elif input_path.endswith('meta.json'):
        meta_jsons = [input_path]
    elif input_path.endswith('.json'):
        meta_jsons = load(input_path)
    elif input_path.isdigit():
        meta_jsons = AutolabelerPlatformHelper.get_result(task_id=int(input_path))
        # 需要保存本地记录
        dump(os.path.join(save_path, f'{input_path}.json'), meta_jsons)
    else:
        raise NotImplementedError
    return meta_jsons


def deduplicate_list_of_dicts(lst):
    # 创建一个空的集合用于存储唯一的字典
    seen = set()
    result = []

    for d in lst:
        # 将字典转换为可哈希的元组
        t = tuple(sorted(d.items()))
        if t not in seen:
            seen.add(t)
            result.append(d)

    return result


@click.command()
@click.option('-t', '--task', default='pvb')
@click.option('-i', '--input_path', default='result/pvb_labels/0603/input.tmp.txt')
@click.option('-s', '--save_path', default='result/pvb_labels/0603')
@click.option('--final_root', default='s3://sdc3-gt-label-2/data.infra.gt_labels/gop/sensebee/GAC/autolabel_20240410_769')
@click.option('--ads_cli_threads', default=10, help='ads_cli threads num')
@click.option('--ads_cli_listers', default=10, help='ads_cli listers num')
@click.option('--aws_ak', default='45DJ69SBDGDFITJT9PCD')
@click.option('--aws_sk', default='oiWI7NP3O3g1UOkEzuzJSZLWkBJpv2a5mcuShCRd')
@click.option('--endpoint', default='http://auto-business.st-sh-01.sensecoreapi-oss.cn')
@click.option('--skip_num', default=0, type=int, help='data sample step')
@click.option('--clip_num_per_task', default=3, type=int, help='per task clip num')
@click.option('--reconstruction_num', default=3)
@click.option("--upload_sensebee/--no-upload_sensebee", default=False)
@click.option("--pre_skip_num", default=0, type=int, help="pre sample ratio")
@click.option("--max_task_num", default=10, type=int, help="max sensebee task num")
def preprocess(task, input_path, save_path, final_root, ads_cli_threads, ads_cli_listers, aws_ak, aws_sk, endpoint, skip_num, clip_num_per_task, reconstruction_num, upload_sensebee, pre_skip_num, max_task_num):
    logger.info('task: {}'.format(task))
    meta_jsons = parse_meta_json(input_path, save_path)
    logger.info("source meta_jsons length {}".format(len(meta_jsons)))
    meta_jsons = meta_jsons[::(pre_skip_num + 1)]
    logger.info("used meta_jsons length {}".format(len(meta_jsons)))

    pre_label_path = os.path.join(save_path, 'pre_sensebee')
    os.makedirs(pre_label_path, exist_ok=True)

    filter = {'pvb': ObjectQualityInspection, 'gop': GopQualityInspection}
    processer = {'pvb': PVBPreAnnoPreProcessor, 'gop': GOPPreAnnoPreProcessor}

    labeling_cache = []
    for idx, meta_json in tqdm(enumerate(meta_jsons), desc='preprocess', total=len(meta_jsons)):
        if not env.is_my_showtime(idx):  # 单进程启动默认不起作用
            continue

        logger.info(meta_json)
        case_name = meta_json.split('/')[-2]
        cur_save_path = os.path.join(save_path, case_name)

        labeling_cache.append({'meta_json': expand_s3_path(meta_json),
                               'root': expand_s3_path(os.path.join(final_root, case_name)),
                               'label_json': expand_s3_path(os.path.join(final_root, case_name, 'label.jsonl')),
                               'root_local': cur_save_path,
                               'label_json_local': os.path.join(cur_save_path, 'label.jsonl'),
                               'case_name': case_name
                               })

        if os.path.exists(os.path.join(cur_save_path, 'label.jsonl')):
            continue  # 类似resume功能

        try:
            if filter[task](meta_json).process():
                processer[task](meta_json, cur_save_path, skip_num, reconstruction_num=reconstruction_num).process()
                # 把所有预标注汇总到一个文件夹内
                if task in ['pvb']:
                    merge_pre_sensebee(case_name, os.path.join(cur_save_path, 'pre_sensebee'), pre_label_path)
        except Exception as e:
            logger.error(e)  # 并不是所有的meta json都是完全没问题的
            logger.error(meta_json)  # 报错出来，便于单独调试debug看看报错原因

    # 若MPI启动, 等待多个进程一起结束，若否, 则直接跳过
    env.barrier()
    labeling_cache = env.gather(labeling_cache, root=0)

    if env.is_master():
        logger.info('merge label_json...')
        # 把所有的label.jsonl合在一起
        label_res = []
        for folder in tqdm(os.listdir(save_path)):
            label_path = os.path.join(save_path, folder, 'label.jsonl')

            if not os.path.exists(label_path):
                continue

            source_label_path = os.path.join(save_path, folder, 'source_label.jsonl')
            if len(open(source_label_path).readlines()) < 10:
                continue

            for item in open(label_path).readlines():
                data = json.loads(item)
                new_data = copy.deepcopy(data)

                new_data['lidar'] = os.path.join(folder, data['lidar'])
                for idx in range(len(data['cameras'])):
                    assert 'image' in new_data['cameras'][idx]
                    assert 'calib' in new_data['cameras'][idx]
                    new_data['cameras'][idx]['image'] = os.path.join(folder, data['cameras'][idx]['image'])
                    new_data['cameras'][idx]['calib'] = os.path.join(folder, data['cameras'][idx]['calib'])
                label_res.append(new_data)

        logger.info('dump label_json ...')
        dump_json_lines(os.path.join(save_path, 'total_label.jsonl'), label_res)

        bee_label_txt = split_group_labels_clip_num(os.path.join(save_path, 'total_label.jsonl'), clip_num_per_task=clip_num_per_task)

        dump(os.path.join(save_path, 'labels/labeling_cache.json'), labeling_cache)

        # 需要把本地文件都上传到ceph，方便送标
        logger.info('sync data to ceph ...')
        ads_cli_upload_folder(save_path, final_root, ads_cli_listers, ads_cli_threads, aws_ak, aws_sk, endpoint=endpoint)

        logger.warning('SenseBee Root: {}'.format(final_root))
        logger.warning('SenseBee PreLabel: {}'.format('{}/pre_sensebee'.format(final_root)))
        logger.warning('label json: {}'.format(os.path.join(save_path, 'total_label.jsonl')))

        if upload_sensebee:
            if len(bee_label_txt) > max_task_num:
                bee_label_txt = bee_label_txt[::(len(bee_label_txt) // max_task_num)]

            for idx, (sub_label_json, sub_label_txt) in enumerate(bee_label_txt):
                pre_sensebee = None if task == 'gop' else '{}/pre_sensebee'.format(final_root)
                task_name = 'j6e_' + task.lower() + "_" + final_root.split('/')[-1] + '_sub_{}'.format(idx)
                task_id = getattr(BeeUploader, f"upload_{task}")(task_name=task_name, root=final_root, label_json=sub_label_json, file_txt=sub_label_txt, pre_sensebee=pre_sensebee)
                logger.warning("{}: {}".format(task_name, task_id))


def postprocess_task(task, label_cache, zip_file, save_path, key_frame_engine=False, projection_optim=False, fix_calib_path=None):
    if isinstance(label_cache, str):
        label_cache = load(label_cache)

    res, error_res = [], []
    processer = {'pvb': PVBPreAnnoPostProcessor, 'gop': GOPPreAnnoPostProcessor}

    SenseBeeID = zip_file.split('/')[-1].split('-')[0]
    os.makedirs(save_path, exist_ok=True)
    for idx, single_meta in tqdm(enumerate(label_cache)):
        if not env.is_my_showtime(idx):
            continue
        meta_json, root, label_json = single_meta['meta_json'], single_meta['root'], single_meta['label_json']
        gt_path = os.path.join(root, 'gt.jsonl')

        case_name = single_meta['case_name']
        if os.path.exists(os.path.join(save_path, f'{case_name}.json')):
            single_res = load(os.path.join(save_path, f'{case_name}.json'))
        else:
            single_res = processer[task](meta_json, zip_file, label_json, root, gt_path, key_frame_engine, projection_optim, fix_calib_path=fix_calib_path).process()

        video_name = f'{SenseBeeID}_{case_name}_key_frame_engine.{key_frame_engine}_projection_optim.{projection_optim}.mp4'
        res_video = os.path.join(save_path, 'vis', video_name)
        ceph_video = os.path.join(root, 'visualize', video_name)
        os.makedirs(os.path.join(save_path, 'vis'), exist_ok=True)

        if smart_exists(res_video) and smart_exists(ceph_video):
            pass
        else:
            SenseBeeBEV(root, single_res['anno'], res_video).process()
            smart_copy(res_video, ceph_video)
        single_res['video_path'] = ceph_video
        single_res['video_url'] = global_petrel_helper.get_url(ceph_video)

        # 评测
        eval_img_path = os.path.join(root, f'evaluation/{case_name}_key_frame_engine.{key_frame_engine}_projection_optim.{projection_optim}.png')
        ObjectEvaluatorV2(single_res['anno']).eval(output_path=eval_img_path)
        single_res['eval_image_path'] = eval_img_path
        single_res['eval_image_url'] = global_petrel_helper.get_url(eval_img_path)

        single_res['level'] = None  # TODO

        dump(os.path.join(save_path, f'{case_name}.json'), single_res)

        res.append(single_res)

    logger.warning("barrier ...")
    env.barrier()

    logger.warning("gather result ...")
    res = env.gather(res)
    error_res = env.gather(error_res)

    if env.is_master():
        logger.info(res)
        dump(os.path.join(save_path, 'total.json'), res)
        if error_res:
            dump(os.path.join(save_path, 'total_error.json'), error_res)


def batch_postprocess_task(task, root, zip_file, save_path, key_frame_engine, projection_optim, fix_calib_path=None):
    sub_labeling_cache = list()
    if env.is_master():
        logger.info('task: {}'.format(task))
        logger.info('root: {}'.format(root))
        logger.info('zip_file: {}'.format(zip_file))

        # load sensebee标注的子文件
        z = zipfile.ZipFile(zip_file, 'r')
        sensebee_labeling_names = []
        for target in tqdm(z.infolist()):
            if target.filename in ['packagePageInfo.json']:
                continue
            image_name = target.filename.strip('.json')
            sensebee_labeling_names.append(image_name)

        # 根据sensebee的结果，自动生成子任务的labeling_cache
        total_labeling_cache_file = os.path.join(root, 'labels/labeling_cache.json')
        assert smart_exists(total_labeling_cache_file), root

        label_cache_metas = deduplicate_list_of_dicts(load(total_labeling_cache_file))

        for single_meta in label_cache_metas:
            case_name = single_meta['case_name']
            cur_label_path = os.path.join(root, case_name, 'label.jsonl')

            if not smart_exists(cur_label_path):
                continue

            hit_flags = []
            for item in global_petrel_helper.readlines(cur_label_path):
                cur_frame_name = json.loads(item)['name'].strip()
                hit_flags.append(cur_frame_name in sensebee_labeling_names)

            if hit_flags and all(hit_flags):
                sub_labeling_cache.append(single_meta)

        logger.info(sub_labeling_cache)

        dump(zip_file + '.labeling_cache.json', sub_labeling_cache)

    env.barrier()
    sub_labeling_cache = env.bcast(sub_labeling_cache)

    postprocess_task(task, sub_labeling_cache, zip_file, save_path, key_frame_engine, projection_optim, fix_calib_path=fix_calib_path)


@click.command()
@click.option('-t', '--task', default='pvb', type=click.Choice(['pvb', 'gop']))
@click.option('--label_cache', default='/workspace/auto-labeling-tools/result/sensebee/GOP/22058/labeling_22058.json')
@click.option('-z', '--zip_file', default='result/sensebee/GOP/22058/0613/22058-2024061213130078.zip')
@click.option('-s', '--save_path', default='/workspace/auto-labeling-tools/result/sensebee/GOP/22058/0613')
@click.option('--key_frame_engine/--no-key_frame_engine', default=False)
@click.option('--projection_optim/--no-projection_optim', default=False)
def postprocess(task, label_cache, zip_file, save_path, key_frame_engine, projection_optim):
    logger.info('task: {}'.format(task))
    logger.info('label_cache: {}'.format(label_cache))
    logger.info('zip_file: {}'.format(zip_file))

    postprocess_task(task, label_cache, zip_file, save_path, key_frame_engine, projection_optim)


@click.command()
@click.option('-t', '--task', default='pvb', type=click.Choice(['pvb', 'gop']))
@click.option('-r', '--root', default='/workspace/auto-labeling-tools/result/sensebee/GOP/22058/labeling_22058.json')
@click.option('-z', '--zip_file', default='result/sensebee/GOP/22058/0613/22058-2024061213130078.zip')
@click.option('-s', '--save_path', default='/workspace/auto-labeling-tools/result/sensebee/GOP/22058/0613')
@click.option('--key_frame_engine/--no-key_frame_engine', default=False)
@click.option('--projection_optim/--no-projection_optim', default=False)
def batch_postprocess(task, root, zip_file, save_path, key_frame_engine, projection_optim):
    batch_postprocess_task(task, root, zip_file, save_path, key_frame_engine, projection_optim)


@click.command()
@click.option('-t', '--task', default='pvb', type=click.Choice(['pvb', 'gop']))
@click.option('-i', '--sensebee_id', default=24397)
@click.option('-s', '--save_path', default='/workspace/kongzelong2/pillar_test/2407/0724/pvb_post/24397')
@click.option('--key_frame_engine/--no-key_frame_engine', default=True)
@click.option('--projection_optim/--no-projection_optim', default=False)
@click.option('--fix_calib_path', default=None)
def bee_batch_postprocess(task, sensebee_id, save_path, key_frame_engine, projection_optim, fix_calib_path=None):
    root, zip_file = None, None
    if env.is_master():
        bee = BeeHelper(sensebee_id=sensebee_id)
        # root = '{}:{}'.format('ad_system_common_oss3', bee.root)
        root = expand_s3_path(bee.root)
        logger.info('task {}, ID {}, root: {}'.format(task, sensebee_id, root))
        zip_file = bee.get_result(save_path=save_path, skip_exist=True)

        logger.info('zip_file: {}'.format(zip_file))

    env.barrier()
    root, zip_file = env.bcast(root), env.bcast(zip_file)

    batch_postprocess_task(task, root, zip_file, save_path, key_frame_engine, projection_optim, fix_calib_path=fix_calib_path)


cli.add_command(preprocess)
cli.add_command(postprocess)
cli.add_command(batch_postprocess)
cli.add_command(bee_batch_postprocess)

if __name__ == "__main__":
    cli()
