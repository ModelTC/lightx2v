import os
import json
import datetime
from alt import smart_exists
from alt.utils.token_helper import generate_random_string
from alt.inference.lidar import LidarInferencer
from alt import load, smart_glob
from alt.sensebee.base.base_process import BaseSenseBeeProcessor


class ODDInferencer(LidarInferencer):
    CACHE = '.cache_odd'
    MODEL_NAME = 'odd'

    def __init__(self,
                 timeout=36000,  # 从向云端提交任务开始计时，超时后跳过，问题是一旦提交，就算我们这边跳过，云端依旧会推理，云端待推理会变得越来越多，资源会变得更紧张，所以给予充足时间运行
                 **kwargs):
        super().__init__(**kwargs)
        import swagger_client
        model_factory_config = swagger_client.configuration.Configuration()
        self.factory_api = swagger_client.ModelServingApi(swagger_client.ApiClient(configuration=model_factory_config))

        os.makedirs(self.CACHE, exist_ok=True)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        random_string = generate_random_string()
        self.dataset_path = os.path.join(self.CACHE, f'{self.__class__.__name__}_{current_time}_{random_string}.txt')
        self.timeout = timeout
        self.sleep_time = 10

    def forward(self, input_paths, **kwargs):
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        with open(self.dataset_path, 'w') as f:
            for single_path in input_paths:
                smart_exists(single_path), single_path
                single_mata = {"ceph_path": single_path, "bbox": []}
                f.writelines(json.dumps(single_mata) + '\n')

        results = self.infer_by_model_factory()
        return results


def get_wanted_pkl(meta):
    meta_json = BaseSenseBeeProcessor().safe_load_meta_json(meta)[0]
    pkls = smart_glob("{}/*fov120*.pkl".format(meta_json['data_annotation']))
    for pkl in pkls:
        for t in ['object-tsr', 'object-tlr']:
            if t in pkl:
                return pkl


def get_scene_infer_res(metas):
    meta_scene_infos = {}
    inferer = ODDInferencer()
    to_infer_img_paths = []
    for meta in metas:
        pkl = get_wanted_pkl(meta)
        gt_res = load(pkl)
        img_path = gt_res['frames'][int(len(gt_res['frames']) / 2)]['filename']
        to_infer_img_paths.append(img_path)
    res = inferer.process(to_infer_img_paths)
    assert len(metas) == len(to_infer_img_paths) == len(res)
    for i, meta in enumerate(metas):
        meta_res = res[i]['result']
        reorg_res ={}
        for k, v in meta_res.items():
            reorg_res[k] = {'label': v.split(':')[0], 'score': float(v.split(':')[1])}
        meta_scene_infos[meta] = reorg_res
    return meta_scene_infos


if __name__ == '__main__':
    infer = ODDInferencer()
    lidar_path = 'ad_system_common_sdc:s3://sdc_gt_label/GAC/autolabel_20240430_891/Data_Collection/GT_data/gacGtParser/drive_gt_collection/traffic_light_in_junction/2024-04/A02-290/2024-04-23/2024_04_23_11_15_06_gacGtParser/gt_labels/cache/sensor/center_camera_fov30#s3/1713870907349999.840.png'
    res = infer.process([lidar_path])
    print(res)
