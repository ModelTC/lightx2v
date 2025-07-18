from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
import mmcv
from mmengine.config import ConfigDict
import pickle as pkl
from magicdrivedit.utils.misc import format_numel_str
from magicdrivedit.registry import DATASETS, build_module
# from .pap import PAPDataset
from .utils import IMG_FPS
import json
import io
import time
from torchvision import transforms
import random
import os
from PIL import Image
import traceback
from colorama import Fore, Style
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Read TimeOut!")

# load_data_timeout = 90 * 1000 # 90s
# load_data_timeout = 60 * 5 # 300s
load_data_timeout = 110000 # 110000

@DATASETS.register_module()
class PAPMultiResDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        self.datasets = OrderedDict()
        for (res, d_cfg) in cfg:
            dataset = build_module(d_cfg, DATASETS)
            self.datasets[res] = dataset

    def as_buckets(self):
        buckets = OrderedDict()  # str: list of indexes
        for res, v in self.datasets.items():
            for key in v.possible_keys:
                buckets["-".join(map(str, [*res, *key]))] = list(
                    range(v.key_len("-".join(map(str, key)))))
        return buckets

    def rand_another_key(self, t=None):
        buckets = self.as_buckets()
        key = np.random.choice(list(buckets.keys()))
        idx = np.random.choice(buckets[key])
        return f"{idx}-{key}"

    def parse_index(self, index: str):
        idx, real_h, real_w, fps = map(int, index.split("-")[:-1])
        real_t = index.split("-")[-1]
        real_t = real_t if real_t == "full" else int(real_t)
        return idx, real_h, real_w, fps, real_t

    def get_info(self, ):
        return ""

    def __len__(self):
        return sum(len(v) for v in self.datasets.values())

    def __getitem__(self, index):
        import pdb; pdb.set_trace()
        idx, real_h, real_w, fps, real_t = self.parse_index(index)
        for try_idx in range(80):
            # sub_index = f"{idx}-{real_t}-{fps}"
            # return self.datasets[(real_h, real_w)][sub_index]

            # 设置超时信号处理器
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(load_data_timeout)  # 设置超时时间

            try:
                sub_index = f"{idx}-{real_t}-{fps}"
                return self.datasets[(real_h, real_w)][sub_index]
            except Exception as err:
                # 进入异常处理，不会停止程序
                if isinstance(err, TimeoutError):
                    print(Style.BRIGHT + Fore.RED + f"get item data time out {load_data_timeout}  s" + Style.RESET_ALL)
                else:
                    traceback.print_exc()
                new_idx = random.randint(0, len(self.datasets[(real_h, real_w)]))
                # print(f"error segment index={idx}, another random index={new_idx}")
                bug_info = self.datasets[(real_h, real_w)].get_info()
                bug_info_print_str = json.dumps(bug_info, indent=4, ensure_ascii=False)
                print(
                    Style.BRIGHT + Fore.RED + bug_info_print_str + Style.RESET_ALL
                )
                print(
                    Style.BRIGHT + Fore.RED + f"error segment index={idx}, another random index={new_idx}" + Style.RESET_ALL
                )
                idx = new_idx
                signal.alarm(0)  # 确保计时器被取消
            finally:
                signal.alarm(0)  # 确保计时器被取消
        raise Exception("exceed the maximum 80 retry times.")

class PAPVariableBatchSampler(torch.utils.data.DistributedSampler):
    def __init__(
        self,
        dataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.bs_config = bucket_config
        self.verbose = verbose
        self.last_micro_batch_access_index = 0

        self._default_bucket_sample_dict = self.dataset.as_buckets()
        self._default_bucket_micro_batch_count = OrderedDict()
        self.approximate_num_batch = 0
        # process the samples
        for bucket_id, data_list in self._default_bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bs_config[bucket_id]
            if bs_per_gpu == -1:
                logging.warning(f"Got bs=-1, we drop {bucket_id}.")
                continue
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            self._default_bucket_sample_dict[bucket_id] = data_list
            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            self._default_bucket_micro_batch_count[bucket_id] = num_micro_batches
            self.approximate_num_batch += num_micro_batches
        self._print_bucket_info(self._default_bucket_sample_dict)

    def __iter__(self) -> Iterator[List[str]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_last_consumed = OrderedDict()

        bucket_sample_dict = OrderedDict({k: v for k, v in self._default_bucket_sample_dict.items()})
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in self._default_bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        # NOTE: all the dict/indexes should be the same across all devices.
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bs_config[bucket_id]
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bs_config[bucket_id]
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # encode t, h, w into the sample index
            cur_micro_batch = [f"{idx}-{bucket_id}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def reset(self):
        self.last_micro_batch_access_index = 0

    def get_num_batch(self) -> int:
        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(self._default_bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        full_dict = defaultdict(lambda: [0, 0])
        num_img_dict = defaultdict(lambda: [0, 0])
        num_vid_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            real_h, real_w, fps = map(int, k.split("-")[:-1])
            real_t = k.split("-")[-1]
            real_t = real_t if real_t == "full" else int(real_t)
            num_batch = size // self.bs_config[k]

            total_samples += size
            total_batch += num_batch

            full_dict[k][0] += size
            full_dict[k][1] += num_batch

            if real_t == 1:
                num_img_dict[k][0] += size
                num_img_dict[k][1] += num_batch
            else:
                num_vid_dict[k][0] += size
                num_vid_dict[k][1] += num_batch

        # log
        if dist.get_rank() == 0:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(full_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

def save_pkl(data, savefilename):
    with open(savefilename, 'wb') as f:
        pkl.dump(data, f)

if __name__ == "__main__":

    from opensora.aoss import AossSimpleFile
    from einops import rearrange
    from torchvision.utils import save_image
    import torchvision.transforms as T
    from torchvision.io import write_video

    reader = AossSimpleFile(
        client_config_path="/mnt/iag/user/tangweixuan/aoss.conf",  # "/mnt/iag/user/ouwenxuan/aoss.conf",
        s3_path='iaginfra:s3://uniad-infra/pap_pvb/'  # 'iaginfra:s3://uniad-infra/pap_pvb/'
    )
    num_frames = 16
    loop = 1
    processed_meta_file = [
        [
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_240705_240801/240705_240801_pap_segment_infos.pkl",
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_240705_240801/240705_240801_pap_segment_annos"
        ],
        [
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_240806-241118/240806-241118_pap_segment_infos.pkl",
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_240806-241118/240806-241118_pap_segment_annos"
        ],
        [
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_241119-241203/241119-241203_pap_segment_infos.pkl",
            "/mnt/iag/user/tangweixuan/metavdt/data/data_processed_meta_241119-241203/241119-241203_pap_segment_annos"
        ]
    ]
    enable_scene_description = True
    scene_description_file = [
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_caption_20241121.json",
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_caption_20241123.json",
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_sample_metas.json",
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_caption_20241124.json",
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_caption_20241125.json",
        "/mnt/iag/user/tangweixuan/datasets/pap/text/pvb_caption_20241209.json"
    ]
    CaseName_PAPCaseName_Table_file = "/mnt/iag/user/tangweixuan/metavdt/data/CaseName_PAPCaseName_Table_0705_0801.json"
    sqrt_required_text_keys = ["weather", "time", "lighting", "road_type", "general"]
    fps_list = [10]
    sequence_length = num_frames * loop
    data_fps = fps = 10
    exclude_cameras = None
    edit_type = None
    draw_bbox_mode = 'cross'
    split_file = None  # '/mnt/iag/user/tangweixuan/metavdt/opensora/datasets/data_process/pap_240806-241118_split.json'
    must_text = True
    with_filled_bbox = False
    colorful_box = True
    add_velocity_to_text = True
    with_traj_map = True
    # *-----------bbox-----------------
    with_bbox_coords = False  # *默认with_camera_param=True
    bbox_drop_prob = 0.05
    drop_bbox_coords_prob = 0
    bbox_mode = 'all-xyz'
    # *-----------bbox-----------------

    camera_list = ['left_front_camera', 'center_camera_fov120', 'right_front_camera', 'right_rear_camera', \
                   'rear_camera', 'left_rear_camera', 'center_camera_fov30', 'front_camera_fov195', \
                   'left_camera_fov195', 'rear_camera_fov195', 'right_camera_fov195']
    image_size = (256, 448)
    full_size = (image_size[0], image_size[1] * len(camera_list))

    pd = PAPDataset(
        None,
        reader,
        processed_meta_file=processed_meta_file,
        camera_list=camera_list,
        sequence_length=2,
        fps_list=[10],
        data_fps=10,
        split="train",  # * 'test'就不会随机采首帧
        enable_scene_description=enable_scene_description,
        exclude_cameras=None,
        edit_type=None,
        draw_bbox_mode='cross',
        split_file=split_file,
        scene_description_file=scene_description_file,
        CaseName_PAPCaseName_Table_file=CaseName_PAPCaseName_Table_file,
        sqrt_required_text_keys=None,
        must_text=must_text,
        expected_vae_size=image_size,
        full_size=full_size,
        colorful_box=colorful_box,
        use_random_seed=False,
        with_filled_bbox=with_filled_bbox,
        add_velocity_to_text=add_velocity_to_text,
        with_traj_map=with_traj_map,
        # *-----------bbox-----------------
        with_bbox_coords=with_bbox_coords,
        bbox_mode=bbox_mode,
        bbox_drop_prob=bbox_drop_prob,
        # *-----------bbox-----------------
        debug=True
    )
    print('len(dataset):', len(pd))
    print("constructed PAP dataset.")
    # 将 Tensor 转为帧列表
    to_pil = T.ToPILImage()  # 转换为 PIL Image

    # for i in range(0, 100, 10):
    for i in range(10, len(pd)):
        data = pd[i]

        img = data["video"].permute(0, 2, 1, 3, 4)  # [V, C, T, H, W] -> [V, T, C, H, W]
        bbox = data["bbox"]  # [V, T, C, H, W]
        filled_bbox_images = data["traj"]  # [V, T, C, H, W]

        V, T, C, H, W = img.shape
        savevideo = torch.cat((rearrange(img, 'V T C H W -> T C H (V W)'), rearrange(bbox, 'V T C H W -> T C H (V W)'),
                               rearrange(filled_bbox_images, 'V T C H W -> T C H (V W)')), dim=-2)

        for idx, img in enumerate(savevideo):
            save_image(img, f"./debug/testtraj_{i}_{idx}.png")

        # write_video(f"./debug/testtraj_{i}.mp4", savevideo.permute(0, 2,3,1), fps=fps, video_codec="h264", options={'crf': '0'})

        with open(f'./debug/test_text_{i}.txt', 'w') as f:
            f.write(data["text"])