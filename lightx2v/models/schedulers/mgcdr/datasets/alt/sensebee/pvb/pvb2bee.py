from alt import smart_exists, dump
from alt.quality import ObjectQualityInspection
from alt.visualize import BEVVisualizer
from alt import dump, load
from tqdm import tqdm
import click
from loguru import logger


from alt.quality.pvb_sensebee_filter import PVBSensebeeFilter


@click.group()
def cli():
    pass


@click.command()
@click.option('-i', '--input_path', default='result/pvb_quality_check_data/pvb_input.tmp.txt')
@click.option('--num_limit', default=-1)
@click.option('--vis_path', default='ad_system_common:s3://sdc_gt_label/ql_test/sensebee_quality/20240517_debug')
@click.option('--cache_file', default='sensebee_cache/pvb_quality_0517.pkl')
@click.option('--autolabel_platform/--no-autolabel_platform', default=False)
def preprocess(input_path, num_limit, vis_path, cache_file, autolabel_platform):
    # S1. 处理下真值生产的路径
    gt_paths = [item.strip() for item in open(input_path).readlines()]
    gt_paths = ["ad_system_common:" + item for item in gt_paths]

    # S2. 根据所要求的CLIP数量和要求进行自动过滤
    filter_res = []
    for filter_idx, gt_path in tqdm(enumerate(gt_paths), desc='Auto Filter'):
        if gt_path.endswith('gt_labels'):
            meta_json = gt_path.strip("/").replace("gt_labels", "meta.json")
        meta_json = gt_path
        assert smart_exists(meta_json), meta_json

        select_timestamps = ObjectQualityInspection(meta_json).process()
        if select_timestamps: # select_timestamps 为[], 则不满足要求
            filter_res.append([meta_json, select_timestamps])

        if num_limit > 0 and len(filter_res) >= num_limit:
            break

    # S3. 根据筛选的结果进行可视化
    sensebee_cache_metas = {}
    for (meta_json, select_timestamps) in tqdm(filter_res, desc='Clip Visualizer'):
        s3_index = meta_json.find('s3://')
        bucket = meta_json[s3_index:].replace('s3://', '').split('/')[0]

        save_path = meta_json.replace(meta_json[:s3_index] + f's3://{bucket}', vis_path)
        if autolabel_platform:
            save_path = save_path.replace('meta.json', 'sensebee/pvbGtQc/pvb_bev_11v/pre_sensebee/sensor')
        vis_res = BEVVisualizer(meta_json=meta_json, save_path=save_path).process(select_timestamps)
        sensebee_cache_metas[meta_json] = {"select_timestamps": select_timestamps, "label": vis_res}
    dump(cache_file, sensebee_cache_metas)
    
    logger.info('[attention]: cache file: {}'.format(cache_file))


@click.command()
@click.option('-r', '--sensebee_result', default='result/pvb_quality_check_data/21747-2024050814225275.zip')
@click.option('--cache_file', default='sensebee_cache/quality_sensebee.pkl')
@click.option('--vis_path', default='ad_system_common:s3://sdc_gt_label/ql_test/sensebee_quality/20240517_debug')
@click.option('--deliver_result', default='pvb_quality_deliver.json')
@click.option('--threshold', default=0.0, help='clip pass rate threshold')
def postprocess(sensebee_result, cache_file, vis_path, deliver_result, threshold):    
    cache_labeling_metas = load(cache_file)

    sensebee_res = PVBSensebeeFilter.process(sensebee_result, vis_path)

    image_root = vis_path  # vis_path 是为了和preprocess做个统一
    case_metas = {}
    for meta_json, metas in cache_labeling_metas.items():
        case_metas[meta_json] = {'clip': None, 'frames': {}}
        for timestamp, image_path in metas['label'].items():
            _image_root_ = image_root if image_root.endswith('/') else image_root + '/'
            sensebee_path = image_path.replace(_image_root_, '')

            assert sensebee_path in sensebee_res, sensebee_path
            case_metas[meta_json]['frames'][timestamp] = sensebee_res[sensebee_path]

        cur_out = case_metas[meta_json]['frames']
        bev_vis_pass_rate = len([item['bev_vis'] for item in cur_out.values() if item['bev_vis']]) / len(cur_out)
        pinhole_pass_rate = len([item['pinhole'] for item in cur_out.values() if item['pinhole']]) / len(cur_out)
        fisheye_pass_rate = len([item['fisheye'] for item in cur_out.values() if item['fisheye']]) / len(cur_out)
        total_pass_rate = len([item['pass'] for item in cur_out.values() if item['pass']]) / len(cur_out)
        
        case_metas[meta_json]['clip'] = {'pinhole_pass_rate': pinhole_pass_rate,
                                         'fisheye_pass_rate': fisheye_pass_rate,
                                         'bev_vis_pass_rate': bev_vis_pass_rate,
                                         'total_pass_rate': total_pass_rate}    
    
    success_case_metas = {key : value for key, value in case_metas.items() if value['clip']['total_pass_rate'] >= threshold}
    
    dump(deliver_result, success_case_metas)


cli.add_command(preprocess)
cli.add_command(postprocess)

if __name__ == "__main__":
    cli()
    
    
