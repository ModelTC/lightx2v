import torch
from collections import OrderedDict

from alt.generator.base import Generator
from alt.utils.env_helper import env

from pytracking.parameter.tamos.tamos_swin_base import parameters
from pytracking.tracker.tamos import TaMOs


class TaMOsGenerator(Generator):
    def __init__(self, net_path='/root/checkpoints/TaMOs/tamos_swin_base.pth.tar'):
        self.track_parameters = parameters(net_path=net_path)

        if env.distributed:
            torch.cuda.set_device(int(env.rank % torch.cuda.device_count()))

        self.tracker = TaMOs(params=self.track_parameters)

    def initialize(self, metas=None, **kwargs):
        pass

    def process_in_batches(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def generate(self, metas):
        src_image = metas['src_image']
        dsr_image = metas['dst_image']

        res = {}
        batch_size = 8
        for batch_idx, target_2d_list in enumerate(self.process_in_batches(metas['info'], batch_size=batch_size)):
            init_bbox, init_object_ids, object_ids, sequence_object_ids = OrderedDict(), list(), list(), list()
            caches = {}
            for idx, target_2d in enumerate(target_2d_list):
                obj_idx = idx + 1
                init_bbox[obj_idx] = target_2d['box2d']
                init_object_ids.append(obj_idx)
                object_ids.append(obj_idx)
                sequence_object_ids.append(obj_idx)
                caches[obj_idx] = target_2d['token']
            
            out = self.tracker.initialize(src_image, {'init_bbox': init_bbox, 'init_object_ids': init_object_ids, 'object_ids': object_ids, 'sequence_object_ids': sequence_object_ids})
            track_res = self.tracker.track(dsr_image)
            
            for obj_idx, obj_token in caches.items():
                pred_bbox = track_res['target_bbox'][obj_idx]
                pred_score = track_res['object_presence_score'][obj_idx]
                assert obj_token not in res
                res[obj_token] = [pred_bbox, pred_score]

        return res
    

if __name__ == '__main__':
    net_path = '/workspace/auto-labeling-tools/result/ckpts/tamos_swin_base.pth.tar'
    generator = TaMOsGenerator(net_path=net_path)

    info = {'init_bbox': [0, 0, 100, 100]}
    generator.tracker.initialize(image=None, info={'init_bbox': [0, 0, 100, 100]})
    generator.tracker.track(image=None)
