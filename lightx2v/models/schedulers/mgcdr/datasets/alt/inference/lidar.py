import os
import time
import json
import ast
import datetime
from loguru import logger
from alt import smart_exists
from alt.utils.token_helper import generate_random_string

from .base import BaseModelFactoryInference


class LidarInferencer(BaseModelFactoryInference):
    CACHE = '.cache'
    MODEL_NAME = 'lod_autolabel'

    def __init__(self,
                 timeout=36000,  # 从向lod提交任务开始计时，超时后跳过，问题是一旦提交，就算我们这边跳过，云端依旧会推理，云端待推理会变得越来越多，资源会变得更紧张，所以给予充足时间运行
                 threshold=dict(Car=0.2, Pedestrian=0.2, Cyclist=0.2, Truck=0.2),
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
        self.threshold = threshold

    def forward(self, input_paths, **kwargs):
        if isinstance(input_paths, str):
            input_paths = [input_paths]

        with open(self.dataset_path, 'w') as f:
            for single_path in input_paths:
                smart_exists(single_path), single_path
                single_mata = {"ceph_path": single_path, "bbox": []}
                f.writelines(json.dumps(single_mata) + '\n')

        results = self.infer_by_model_factory()

        results = self.filter_by_threshold(results)
        return results

    def filter_by_threshold(self, results):
        # TODO
        return results

    def infer_by_model_factory(self):
        add_dataset_response = self.factory_api.add_dataset(
            input_list_file=self.dataset_path,
            input_type='list_json_ceph_anno',
            dataset_name=f'{self.__class__.__name__}.{time.time()}',
            desc='autolabel',
            label='inference')
        if add_dataset_response.status != 'SUCCESS':
            raise RuntimeError(f'model factory add dataset failed: {add_dataset_response}')

        dataset_id = add_dataset_response.result['id']

        models = self.factory_api.query_model()
        model_id = None
        for model in models:
            if model['name'] == self.MODEL_NAME:
                model_id = model['id']
        if model_id is None:
            raise Exception(f'model package {self.package} not found in model factory')

        add_task_response = self.factory_api.add_task(model_id=model_id, dataset_id=dataset_id, sub_size=150, max_fail_time=2)
        add_task_response = ast.literal_eval(add_task_response)
        task_id = add_task_response['result']['task']

        t = 0
        result = None
        while t < self.timeout // self.sleep_time:
            logger.info(f'waiting task_id: {task_id}, cost {t * self.sleep_time} s...')
            query_task_response = self.factory_api.query_task(task_id=task_id, with_log=True)

            if query_task_response.result[0]['task']['status'] != 'SUCCESS':
                logger.info(query_task_response.result[0]['task']['status'])
                if query_task_response.result[0]['task']['status'] == 'FAILURE':
                    logger.error(query_task_response.result[0]['logs'])
                    return None
                time.sleep(self.sleep_time)
                t += 1
                continue
            query_task_response = self.factory_api.query_task(task_id=task_id, subtask_num=-1, with_result=True)

            assert query_task_response.status == 'SUCCESS', f'get model factory result failed: {query_task_response}'
            result = [result for results in query_task_response.result[0]['results'] for result in results]
            break
        if result is None:
            revoke_task_response = self.factory_api.revoke_task(task_id=task_id)  # noqa
            logger.error(f'timeout, revoke task:{task_id}')
            assert result is not None, 'timeout'  # 超时还没跑出来就取消掉上传的任务，并报错
        return result


if __name__ == '__main__':
    infer = LidarInferencer()

    # 暂时只接受ceph路径，最好整个list一起，效率比较高，不要单循环, 目前只支持sdc这个前缀
    lidar_path = 'ad_system_common_sdc:s3://sdc_gt_label/GAC/autolabel_20240410_768/Data_Collection/GT_data/gacGtParser/drive_gt_collection/PVB_gt/2024-03/A02-290/2024-03-22/2024_03_22_06_54_59_gacGtParser/gt_labels/cache/sensor/car_center#s1/1711090527099000.000.bin'

    res = infer.process([lidar_path])

    print(res)
