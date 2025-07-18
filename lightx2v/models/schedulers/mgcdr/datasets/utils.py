from typing import List, Dict, Any
import logging
import os
import re
import time

import numpy as np
import json
from copy import deepcopy
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image
from trimesh.scene import cameras
from tqdm import tqdm
from ..mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from einops import repeat, rearrange
import cv2
import math
try:
    from moviepy import ImageSequenceClip
except:
    from moviepy.editor import ImageSequenceClip
import traceback

IMG_FPS = 120
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)
# 11v
# camera_list = ['left_front_camera', 'center_camera_fov120', 'right_front_camera', 'right_rear_camera', \
#                'rear_camera', 'left_rear_camera', 'center_camera_fov30', 'front_camera_fov195', \
#                'left_camera_fov195', 'rear_camera_fov195', 'right_camera_fov195']
# 7v
camera_list = ["center_camera_fov120", "right_front_camera", "right_rear_camera", "rear_camera",
               "left_rear_camera", "left_front_camera", "center_camera_fov30"]
camera_list_195 = ["front_camera_fov195", "left_camera_fov195", "rear_camera_fov195","right_camera_fov195"]
def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, 
                high_quality=False, verbose=True, with_postfix=True, force_image=False, save_per_n_frame=-1, tag=None):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    try:
        assert x.ndim == 4, f"Input dim is {x.ndim}/{x.shape}"
        x = x.to("cpu")
        if with_postfix:
            save_path += f"_{x.shape[-2]}x{x.shape[-1]}"
        if tag is not None:
            save_path += f"_{tag}"

        if not force_video and x.shape[1] == 1:  # T = 1: save as image
            if not is_img(save_path):
                save_path += ".png"
            x = x.squeeze(1)
            save_image([x], save_path, normalize=normalize, value_range=value_range)
        else:
            if with_postfix:
                save_path += f"_f{x.shape[1]}_fps{fps}.mp4"
            elif not is_vid(save_path):
                save_path += ".mp4"
            if normalize:
                low, high = value_range
                x.clamp_(min=low, max=high)
                x.sub_(low).div_(max(high - low, 1e-5))

            x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
            imgList = [xi for xi in x.numpy()]
            if force_image:
                os.makedirs(save_path)
                for xi, _x in enumerate(imgList):
                    _save_path = os.path.join(save_path, f"f{xi:05d}.png")
                    Image.fromarray(_x).save(_save_path)
            elif high_quality:
                if save_per_n_frame > 0 and len(imgList) > save_per_n_frame:
                    single_value = len(imgList) % 2
                    for i in range(0, len(imgList) - single_value, save_per_n_frame):
                        if i == 0:
                            _save_path = f"_f0-{save_per_n_frame + 1}".join(os.path.splitext(save_path))
                            vid_len = save_per_n_frame + single_value
                        else:
                            vid_len = save_per_n_frame
                            _save_path = f"_f{i + 1}-{i + 1 + vid_len}".join(os.path.splitext(save_path))
                            i += single_value
                        if len(imgList[i:i+vid_len]) < vid_len:
                            logging.warning(f"{len(imgList)} will stop at frame {i}.")
                            break
                        clip = ImageSequenceClip(imgList[i:i+vid_len], fps=fps)
                        clip.write_videofile(
                            _save_path, verbose=verbose, bitrate="4M",
                            logger='bar' if verbose else None)
                        clip.close()
                else:
                    clip = ImageSequenceClip(imgList, fps=fps)
                    clip.write_videofile(
                        save_path, verbose=verbose, bitrate="4M",
                        logger='bar' if verbose else None)
                    clip.close()
            else:
                write_video(save_path, x, fps=fps, video_codec="h264")
        # if verbose:
        if True:
            print(f"Saved to {save_path}")
    except Exception as e:
        traceback.print_exc()
    return save_path


def inverse_concat_n_views_pt(cat_tensor, NC, oneline=False):
    num_cams = NC  # This is the number of cameras before concat
    half = math.ceil(num_cams / 2)
    T, H, W , C = cat_tensor.shape
    if oneline:
        # If the concat was done in a single line, we need to reshape back to the original shape
        # We reverse the rearrange operation
        imgs = rearrange(cat_tensor, "C T H (NC W) -> NC C T H W", NC=NC)
    else:
        # Reverse the splitting and concatenation done in the vertical direction
        imgs_up, imgs_down = torch.split(cat_tensor, [H//2, H//2], dim=1)
        if NC % 2 == 1:
            imgs_down = torch.split(imgs_down, [W - W//half, W//half], dim=2)[0]
        imgs_up = rearrange(imgs_up, "T H (NC W) C -> NC T H W C", NC=half)
        imgs_down = rearrange(imgs_down, "T H (NC W) C -> NC T H W C", NC=num_cams - half)

        imgs = torch.cat([imgs_up, imgs_down], dim=0)

    return imgs


def extract_camera_relative_path(path: str, keyword: str = "camera") -> str:
    """
    从给定路径中提取从关键字开始的相对路径。

    :param path: 完整的文件路径
    :param keyword: 关键字，默认为 "camera"
    :return: 关键字及其后的相对路径
    """
    parts = path.split('/')

    for i, part in enumerate(parts):
        if keyword in part:
            return '/'.join(parts[i:])

    return path

# 全局缓存（或类成员变量）
g_mapping_cache = {}
# 全局缓存（或类成员变量）
_mapping_cache = {}

def new_re_distort_image_geometrically(pil_undistorted_img, config):
    width = config['image_width']
    height = config['image_height']
    cache_key = (width, height, tuple(config['cam_K'].flatten()), tuple(config['camera_dist'][0]))
    # 检查缓存
    if cache_key not in _mapping_cache:
        raw_K = np.array(config['cam_K'], dtype=np.float64)
        D = np.array(config['camera_dist'][0], dtype=np.float64)
        new_K = np.array(config['cam_K_new'], dtype=np.float64)
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)
        
        uv_undistorted = cv2.fisheye.undistortPoints(
            uv.reshape(-1, 1, 2), raw_K, D, R=np.eye(3), P=new_K
        ).reshape(-1, 2)
        
        map_x = uv_undistorted[:, 0].reshape((height, width)).astype(np.float32)
        map_y = uv_undistorted[:, 1].reshape((height, width)).astype(np.float32)
        _mapping_cache[cache_key] = (map_x, map_y)
    else:
        map_x, map_y = _mapping_cache[cache_key]

    # 剩余处理...

def re_distort_image_geometrically(pil_undistorted_img, config):
    """
    使用 fisheye 模型的几何逆映射从去畸变图恢复原始畸变图，保证无黑边、无尺寸变化。

    参数:
        pil_undistorted_img (PIL.Image): 去畸变图像
        config (dict): 包含以下字段：
            - image_width (int)
            - image_height (int)
            - camera_intrinsic (list[list]): 原始畸变图内参
            - camera_dist (list[list]): 畸变参数（4项）
            - new_intrinsic (list[list]): 去畸变时使用的新内参（newCamMat）

    返回:
        PIL.Image: 还原的 fisheye 畸变图像
    """
    width = config['image_width']
    height = config['image_height']
    raw_K = np.array(config['cam_K'], dtype=np.float64)
    D = np.array(config['camera_dist'][0], dtype=np.float64)
    new_K = np.array(config['cam_K_new'], dtype=np.float64)

    img_undist = np.array(pil_undistorted_img.convert("RGB"))
    img_undist = cv2.cvtColor(img_undist, cv2.COLOR_RGB2BGR)

    # 1. 构建去畸变图像上的像素网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(np.float32)

    # 2. 调用 fisheye::undistortPoints 得到反向映射
    uv_undistorted = cv2.fisheye.undistortPoints(
        uv.reshape(-1, 1, 2), raw_K, D, R=np.eye(3), P=new_K
    ).reshape(-1, 2)

    map_x = uv_undistorted[:, 0].reshape((height, width)).astype(np.float32) # 相同的分辨率和畸变参数的图像可以复用
    map_y = uv_undistorted[:, 1].reshape((height, width)).astype(np.float32)

    # 3. 重映射：去畸变图 → 畸变图
    restored_img = cv2.remap(
        img_undist,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # 一般不会出现黑边
    )

    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_img)


def save_carla_misc(save_root, save_s3_root, video=None, value_range=(-1, 1), text=None, annos = None,
                        path_to_annos=None, img_path=None, resize_org = False, re_dist = False):
    os.makedirs(save_root, exist_ok=True)
    num_views = len(img_path[0])
    save_clip_name = save_root.split("/")[-1]
    
    tt = time.time()
    if video is not None:
        t0 = time.time()
        low = value_range[0]
        high = value_range[1]
        video.clamp_(min=low, max=high)
        video.sub_(low).div_(max(high - low, 1e-5))
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        assert img_path is not None
        video = inverse_concat_n_views_pt(video, num_views) # NC T H W C
        print('process video cost time:{} s'.format(time.time() - tt))    

        for idx_frame, anno in enumerate(annos):
            for idx_view, cam in enumerate(camera_list):
                img_local_path = anno['sensors']['cameras'][cam]['data_path']
                if "s3://" in img_local_path: # 抗 虚拟相机给的是s3 绝对路径地址 处理成相对路径
                    img_local_path = extract_camera_relative_path(img_local_path,"camera")
                    anno['sensors']['cameras'][cam]['data_path'] = img_local_path # 写回去相对路径
                _save_path = os.path.join(save_root, img_local_path)
                os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                save_img = Image.fromarray(video[idx_view, idx_frame].cpu().numpy())
                if resize_org and "org_sensors" in anno.keys():
                    anno['sensors']['cameras'][cam]['image_width'] = anno['org_sensors']['cameras'][cam]['image_width']
                    anno['sensors']['cameras'][cam]['image_height'] = anno['org_sensors']['cameras'][cam]['image_height']
                    anno['sensors']['cameras'][cam]['camera_intrinsic'] = anno['org_sensors']['cameras'][cam]['camera_intrinsic']
                    image_width = anno['org_sensors']['cameras'][cam]['image_width']
                    image_height = anno['org_sensors']['cameras'][cam]['image_height']
                    save_img = save_img.resize((image_width, image_height))
                    if re_dist and "dist_params" in anno['sensors']['cameras'][cam]: # 反去畸变
                        dist_params = anno['sensors']['cameras'][cam]['dist_params']
                        save_img = re_distort_image_geometrically(save_img, dist_params)
                        anno['sensors']['cameras'][cam]['camera_intrinsic'] = dist_params['cam_K']
                        anno['sensors']['cameras'][cam]['camera_dist'] = dist_params['camera_dist']
                save_img.save(_save_path)
    print('save video cost time:{} s, _save_path:{}'.format(time.time() - tt, _save_path))    
    
    tt = time.time()
    if annos is not None:
        # add s3_data_path
        for anno in annos:
            # for cameras, cameras_dict in anno['sensors']['cameras'].items():
            for cam in camera_list:
                anno['sensors']['cameras'][cam]['s3_data_path'] = os.path.join(save_s3_root, save_clip_name, anno['sensors']['cameras'][cam]['data_path'])
                # if resize_org and "org_sensors" in anno.keys():
                #     anno['sensors']['cameras'][cam]['image_width'] = anno['org_sensors']['cameras'][cam]['image_width']
                #     anno['sensors']['cameras'][cam]['image_height'] = anno['org_sensors']['cameras'][cam]['image_height']
                #     anno['sensors']['cameras'][cam]['camera_intrinsic'] = anno['org_sensors']['cameras'][cam]['camera_intrinsic']

            for cam_195 in camera_list_195: # 目前7v 不支持广角鱼眼镜头
                if cam_195 in anno['sensors']['cameras'].keys():
                    anno['sensors']['cameras'].pop(cam_195)
            anno["root"] = os.path.join(save_s3_root, save_clip_name) # clip 里的root
        save_anno_path = os.path.join(save_root, "annos.jsonl")
        with open(save_anno_path, "w") as f:
            for item in tqdm(annos, total=len(annos), desc=f"dumping to {save_anno_path}"):
                f.writelines(json.dumps(item, ensure_ascii=False) + "\n")
    print('save annos cost time:{} s, save_anno_path:{}'.format(time.time() - tt, save_anno_path))  

    tt = time.time()
    # save clip
    with open(os.path.join(save_root, "clip.json"), "w") as f:
        json.dump({
            "anno": os.path.join(save_s3_root, save_clip_name, "annos.jsonl"),
            "root": os.path.join(save_s3_root, save_clip_name),
            "case_name": save_clip_name,
            "frame_num": len(annos),
            "caption": os.path.join(save_s3_root, save_clip_name, "prompt.tmp.txt"),
            "video_path": "",
            "video_url": "",
            "meta_json": ""
        }, f)
    print('save clip cost time:{} s, save_clip_path:{}'.format(time.time() - tt, os.path.join(save_root, "clip.json")))

    tt = time.time()
    if text is not None:
        with open(os.path.join(save_root, "prompt.tmp.txt"), "w") as f:
            f.write(text[0])
    if path_to_annos is not None:
        with open(os.path.join(save_root, "annos_path.json"), "w") as f:
            json.dump(path_to_annos, f)
    print('save prompt.tmp + annos_path cost time:{} s, save_root:{}'.format(time.time() - tt, save_root))

def save_misc(save_root, video=None, value_range=(-1, 1), text=None, layout=None, path_to_annos=None, img_path=None):
    os.makedirs(save_root, exist_ok=True)
    num_views = len(img_path[0])
    if video is not None:
        low = value_range[0]
        high = value_range[1]
        video.clamp_(min=low, max=high)
        video.sub_(low).div_(max(high - low, 1e-5))
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        assert img_path is not None
        video = inverse_concat_n_views_pt(video, num_views) # NC T H W C
        for idx_frame, frame in enumerate(img_path):
            for idx_view, view in enumerate(frame):
                _save_path = os.path.join(save_root, "cameras", *view[0].split("/")[-2:])
                os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                Image.fromarray(video[idx_view, idx_frame].cpu().numpy()).save(_save_path)

    if text is not None:
        with open(os.path.join(save_root, "prompt.tmp.txt"), "w") as f:
            f.write(text[0])

    if layout is not None:
        assert img_path is not None
        # layout = rearrange(layout[0], "NC T C H W -> NC T H W C").mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        # pass

    if path_to_annos is not None:
        with open(os.path.join(save_root, "annos_path.json"), "w") as f:
            json.dump(path_to_annos, f)

def save_misc_vd2(save_root, save_s3_root, video=None, value_range=(-1, 1), text=None, annos=None, path_to_annos=None, img_path=None):
    os.makedirs(save_root, exist_ok=True)
    save_clip_name = save_root.split("/")[-1]
    num_views = len(img_path[0])
    if video is not None:
        low = value_range[0]
        high = value_range[1]
        video.clamp_(min=low, max=high)
        video.sub_(low).div_(max(high - low, 1e-5))
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to(torch.uint8)
        assert img_path is not None
        video = inverse_concat_n_views_pt(video, num_views) # NC T H W C
        for idx_frame, frame in enumerate(img_path):
            for idx_view, view in enumerate(frame):
                camera_name = view[0].split("/")[-2]
                _save_path = os.path.join(save_root, "cameras", *view[0].split("/")[-2:])
                os.makedirs(os.path.dirname(_save_path), exist_ok=True)
                save_img = Image.fromarray(video[idx_view, idx_frame].cpu().numpy())
                save_img.save(_save_path)
                W, H = save_img.size
                if annos is not None:
                    # annos[idx_frame]['cams'][camera_name]['data_path'] = os.path.join(save_clip_name, "cameras", *view[0].split("/")[-2:])
                    annos[idx_frame]['cams'][camera_name]['data_path'] = os.path.join("cameras", *view[0].split("/")[-2:])
                    # 变换内参到对应的size
                    camera_intrinsic = annos[idx_frame]['cams'][camera_name]["cam_intrinsic"]
                    img_org_w, img_org_h = annos[idx_frame]['cams'][camera_name]["image_size"] # org size
                    sx, sy = W / img_org_w, H / img_org_h
                    camera_intrinsic = np.array(camera_intrinsic) * np.array([sx, sy, 1])[:, np.newaxis]
                    annos[idx_frame]['cams'][camera_name]["cam_intrinsic"] = camera_intrinsic.tolist()
                    annos[idx_frame]['cams'][camera_name]["image_size"] = (W, H) # 修改图像size

    if text is not None:
        with open(os.path.join(save_root, "prompt.tmp.txt"), "w") as f:
            f.write(text[0])

    if path_to_annos is not None:
        with open(os.path.join(save_root, "org_annos_path.json"), "w") as f:
            json.dump(path_to_annos, f)

    if annos is not None:
        with open(os.path.join(save_root, "annos.json"), "w") as f:
            json.dump(annos, f, indent=4)

    # save clip
    with open(os.path.join(save_root, "clip.json"), "w") as f:
        json.dump({
            "annos": os.path.join(save_s3_root, save_clip_name, "annos.json"),
            "root": os.path.join(save_s3_root, save_clip_name),
            "case_name": save_clip_name,
        }, f)

def unsqueeze_tensors_in_dict(in_dict: Dict[str, Any], dim) -> Dict[str, Any]:
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.unsqueeze(dim)
        elif isinstance(v, dict):
            out_dict[k] = unsqueeze_tensors_in_dict(v, dim)
        elif isinstance(v, list):
            if dim == 0:
                out_dict[k] = [v]
            elif dim == 1:
                out_dict[k] = [[vi] for vi in v]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            out_dict[k] = None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return out_dict


def stack_tensors_in_dicts(
        dicts: List[Dict[str, Any]], dim, holder=None) -> Dict[str, Any]:
    """stack any Tensor in list of dicts. If holder is provided, dicts will be
    stacked ahead of holder tensor. Make sure no dict is changed in place.

    Args:
        dicts (List[Dict[str, Any]]): dicts to stack, without the desired dim.
        dim (int): dim to add for stack.
        holder (_type_, optional): dict to hold, with the desired dim. Defaults
        to None. 

    Raises:
        TypeError: if the datatype for values are not Tensor or dict.

    Returns:
        Dict[str, Any]: stacked dict.
    """
    if len(dicts) == 1:
        if holder is None:
            return unsqueeze_tensors_in_dict(dicts[0], dim)
        else:
            this_dict = dicts[0]
            final_dict = deepcopy(holder)
    else:
        this_dict = dicts[0]  # without dim
        final_dict = stack_tensors_in_dicts(dicts[1:], dim)  # with dim
    for k, v in final_dict.items():
        if isinstance(v, torch.Tensor):
            # for v in this_dict, we need to add dim before concat.
            if this_dict[k].shape != v.shape[1:]:
                print("Error")
            final_dict[k] = torch.cat([this_dict[k].unsqueeze(dim), v], dim=dim)
        elif isinstance(v, dict):
            final_dict[k] = stack_tensors_in_dicts(
                [this_dict[k]], dim, holder=v)
        elif isinstance(v, list):
            if dim == 0:
                final_dict[k] = [this_dict[k]] + v
            elif dim == 1:
                final_dict[k] = [
                    [this_vi] + vi for this_vi, vi in zip(this_dict[k], v)]
            else:
                raise ValueError(
                    f"cannot handle {k}:{v} ({v.__class__}) with dim={dim}")
        elif v is None:
            assert final_dict[k] is None
        else:
            raise TypeError(f"Unknow dtype for {k}:{v} ({v.__class__})")
    return final_dict


def box_center_shift(bboxes: LiDARInstance3DBoxes, new_center):
    raw_data = bboxes.tensor.numpy()
    new_bboxes = LiDARInstance3DBoxes(
        raw_data, box_dim=raw_data.shape[-1], origin=new_center)
    return new_bboxes


def trans_boxes_to_view(bboxes, transform, aug_matrix=None, proj=True):
    """2d projection with given transformation.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transform (np.array): 4x4 matrix
        aug_matrix (np.array, optional): 4x4 matrix. Defaults to None.

    Returns:
        np.array: (N, 8, 3) normlized, where z = 1 or -1
    """
    if len(bboxes) == 0:
        return None

    bboxes_trans = box_center_shift(bboxes, (0.5, 0.5, 0.5))
    trans = transform
    if aug_matrix is not None:
        aug = aug_matrix
        trans = aug @ trans
    corners = bboxes_trans.corners
    num_bboxes = corners.shape[0]

    coords = np.concatenate(
        [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
    )
    trans = deepcopy(trans).reshape(4, 4)
    coords = coords @ trans.T

    coords = coords.reshape(-1, 4)
    # we do not filter > 0, need to keep sign of z
    if proj:
        z = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= z
        coords[:, 1] /= z
        coords[:, 2] /= np.abs(coords[:, 2])

    coords = coords[..., :3].reshape(-1, 8, 3)
    return coords


def trans_boxes_to_views(bboxes, transforms, aug_matrixes=None, proj=True):
    """This is a wrapper to perform projection on different `transforms`.

    Args:
        bboxes (LiDARInstance3DBoxes): bboxes
        transforms (List[np.arrray]): each is 4x4.
        aug_matrixes (List[np.array], optional): each is 4x4. Defaults to None.

    Returns:
        List[np.array]: each is Nx8x3, where z always equals to 1 or -1
    """
    if len(bboxes) == 0:
        return None

    coords = []
    for idx in range(len(transforms)):
        if aug_matrixes is not None:
            aug_matrix = aug_matrixes[idx]
        else:
            aug_matrix = None
        coords.append(
            trans_boxes_to_view(bboxes, transforms[idx], aug_matrix, proj))
    return coords

