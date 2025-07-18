from typing import Tuple, List
from functools import partial
import logging

import os
import cv2
import mmcv
import torch
import random
import numpy as np
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


from pyquaternion import Quaternion
from transformers import CLIPTokenizer
from mmengine.config import ConfigDict

from magicdrivedit.registry import DATASETS
from mmcv.parallel import DataContainer
from ..mmdet_plugin.datasets import NuScenesDataset
from ..mmdet_plugin.core.bbox import LiDARInstance3DBoxes
from .utils import trans_boxes_to_views, IMG_FPS


def transform_bbox(first_frame_boxes, next2top: np.ndarray):
    gt_bboxes_3d = first_frame_boxes['gt_bboxes_3d'].data.tensor
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
    )
    gt_labels_3d = torch.clone(first_frame_boxes['gt_labels_3d'].data)

    gt_bboxes_3d.rotate(next2top[:3, :3])
    gt_bboxes_3d.translate(next2top[:3, 3])

    return {
        "gt_bboxes_3d": DataContainer(gt_bboxes_3d),
        "gt_labels_3d": DataContainer(gt_labels_3d),
    }


def obtain_next2top(first, current, epsilon=1e-6, v2=True):
    l2e_r = first["lidar2ego_rotation"]
    l2e_t = first["lidar2ego_translation"]
    e2g_r = first["ego2global_rotation"]
    e2g_t = first["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    l2e_r_s = current["lidar2ego_rotation"]
    l2e_t_s = current["lidar2ego_translation"]
    e2g_r_s = current["ego2global_rotation"]
    e2g_t_s = current["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    next2lidar_rotation = R.T  # points @ R.T + T
    next2lidar_translation = T
    if v2:
        # inverse, point trans from lidar to next
        _R = np.concatenate([next2lidar_rotation.T, np.array(
            [[0.,] * 3], dtype=T.dtype)], axis=0)
        _T = -next2lidar_rotation.T @ next2lidar_translation
        _T = np.concatenate(
            [_T[..., np.newaxis], np.array([[1.]], dtype=T.dtype)], axis=0)
        # shape like:
        # | R T |
        # | 0 1 |
        # A @ point lidar -> point next
        next2lidar = np.concatenate([_R, _T], axis=1)
    else:
        _R = np.concatenate(
            [next2lidar_rotation, np.array([[0.,]] * 3, dtype=T.dtype)], axis=1)
        _T = np.concatenate(
            [next2lidar_translation, np.array([1.], dtype=T.dtype)], axis=0)
        # shape like:
        # | R 0 |
        # | T 1 |.T
        next2lidar = np.concatenate(
            [_R, _T[np.newaxis, ...]], axis=0,
        ).T  # A @ [points, 1].T
    if epsilon is not None:
        next2lidar[np.abs(next2lidar) < epsilon] = 0.
    return next2lidar


META_KEY_LIST = [
    "gt_bboxes_3d",
    "gt_labels_3d",
    "camera_intrinsics",
    "camera2ego",
    "lidar2ego",
    "lidar2camera",
    "camera2lidar",
    "lidar2image",
    "img_aug_matrix",
    "metas",
]


def _tokenize_captions(examples, template, tokenizer=None, is_train=True):
    captions = []
    for example in examples:
        caption = template.format(**example["metas"].data)
        captions.append(caption)
    captions.append("")
    if tokenizer is None:
        return None, captions

    # pad in the collate_fn function
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="do_not_pad",
        truncation=True,
    )
    input_ids = inputs.input_ids
    # pad to the longest of current batch (might differ between cards)
    padded_tokens = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids
    return padded_tokens, captions


def ensure_canvas(coords, canvas_size: Tuple[int, int]):
    """Box with any point in range of canvas should be kept.

    Args:
        coords (_type_): _description_
        canvas_size (Tuple[int, int]): _description_

    Returns:
        np.array: mask on first axis.
    """
    (h, w) = canvas_size
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    w_mask = np.any(np.logical_and(
        coords[..., 0] > 0, coords[..., 0] < w), axis=1)
    h_mask = np.any(np.logical_and(
        coords[..., 1] > 0, coords[..., 1] < h), axis=1)
    c_mask = np.logical_and(c_mask, np.logical_and(w_mask, h_mask))
    return c_mask


def ensure_positive_z(coords):
    c_mask = np.any(coords[..., 2] > 0, axis=1)
    return c_mask


def random_0_to_1(mask: np.array, num):
    assert mask.ndim == 1
    inds = np.where(mask == 0)[0].tolist()
    random.shuffle(inds)
    mask = np.copy(mask)
    mask[inds[:num]] = 1
    return mask


def _transform_all(examples, matrix_key, proj):
    """project all bbox to views, return 2d coordinates.

    Args:
        examples (List): collate_fn input.

    Returns:
        2-d list: List[List[np.array]] for B, N_cam. Can be [].
    """
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    # lidar2image (np.array): lidar to image view transformation
    trans_matrix = np.stack([example[matrix_key].data.numpy()
                            for example in examples], axis=0)
    # img_aug_matrix (np.array): augmentation matrix
    img_aug_matrix = np.stack([example['img_aug_matrix'].data.numpy()
                               for example in examples], axis=0)
    B, N_cam = trans_matrix.shape[:2]

    bboxes_coord = []
    # for each keyframe set
    for idx in range(B):
        # if zero, add empty list
        if len(gt_bboxes_3d[idx]) == 0:
            # keep N_cam dim for convenient
            bboxes_coord.append([None for _ in range(N_cam)])
            continue

        coords_list = trans_boxes_to_views(
            gt_bboxes_3d[idx], trans_matrix[idx], img_aug_matrix[idx], proj)
        bboxes_coord.append(coords_list)
    return bboxes_coord


def _preprocess_bbox_keep_all(
        bbox_mode, canvas_size, examples, is_train=True, view_shared=False,
        use_3d_filter=True, bbox_add_ratio=0, bbox_add_num=0, bbox_drop_ratio=0,
        keyframe_rate=1):
    """Pre-processing for bbox. NOTE: B should be T in this level.
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, masks is the same along `N_cam` dim; otherwise,
            use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): T, 1, max_len, ...; same for each cam view
            classes (LongTensor): T, 1, max_len; same for each cam view
            masks: 1, N_cam, max_len; 1 for data, 0 for mask-out, -1 for drop
    """
    # init data
    # bboxes = []
    # classes = []
    # max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]  # len T list
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]
    # gt_bboxes_token: List[str] = [bbox.tokens for bbox in gt_bboxes_3d]

    possible_tokens = set()
    for example in examples:
        _tokens = set([t for t in example['gt_bboxes_3d'].data.tokens])
        possible_tokens = possible_tokens.union(_tokens)
    possible_tokens = sorted(list(possible_tokens))

    max_len = len(possible_tokens)
    if max_len == 0:
        return None, None
    token_idx_dict = {token: idx for idx, token in enumerate(possible_tokens)}

    # params
    B = len(gt_bboxes_3d)
    N_out = N_cam = len(examples[0]['lidar2image'].data.numpy())

    # construct data
    if bbox_mode == 'cxyz':
        # x1y1z1, x1y0z1, x1y1z0, x0y1z1
        out_shape = [4, 3]
    elif bbox_mode == 'all-xyz':
        out_shape = [8, 3]
    elif bbox_mode == 'owhr':
        raise NotImplementedError("Not sure how to do this.")
    else:
        raise NotImplementedError(f"Wrong mode {bbox_mode}")
    # bboxes and classes are always view-shared
    bboxes_out = torch.zeros(B, 1, max_len, *out_shape, dtype=torch.float32)
    classes_out = -torch.ones(B, 1, max_len, dtype=torch.int32)
    # mask = 0 if:
    #    1. there is no such a box token in frame annotations
    #    2. not view_shared, and unseen in this view (aug by bbox_add_ratio)
    # mask = -1 if: (only happens to where mask == 1)
    #    1. non-keyframe dropping (with bbox_drop_ratio)
    #    2. any further mask?
    mask_out = torch.zeros(B, N_cam, max_len)

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # set value for boxes
    for bi in range(B):
        bboxes_kf = gt_bboxes_3d[bi]
        classes_kf = gt_labels_3d[bi]
        tokens = bboxes_kf.tokens

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0:
            continue

        # == start to set mask ==
        # if drop box, we set mask = -1 to mark they are dropped.
        elif bi % keyframe_rate != 0 and is_train:  # only for non-keyframes
            if random.random() < bbox_drop_ratio:
                set_box_to_none = True
            else:
                set_box_to_none = False
        else:
            set_box_to_none = False

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[bi]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[bi]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)] * N_out
        for ci, index_list_c in enumerate(index_list):
            tokens_c = [tokens[i] for i in np.where(index_list_c)[0]]
            for token in tokens_c:
                idx = token_idx_dict[token]
                mask_out[bi, ci, idx] = 1
        # if drop box, we set mask = -1 to mark they are dropped.
        if set_box_to_none:
            mask_out[bi] = -mask_out[bi]  # set 1 to -1
        # == mask all done here ==

        # == bboxes & classes, same across the whole batch ==
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        for box, cls, token in zip(bboxes_pt, classes_kf, tokens):
            idx = token_idx_dict[token]
            bboxes_out[bi, 0, idx] = box
            classes_out[bi, 0, idx] = cls

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_dict = {
        "bboxes": bboxes_out,
        "classes": classes_out,
        "masks": mask_out,
    }
    return ret_dict, bboxes_coord


def _preprocess_bbox(bbox_mode, canvas_size, examples, is_train=True,
                     view_shared=False, use_3d_filter=True, bbox_add_ratio=0,
                     bbox_add_num=0, bbox_drop_ratio=0, keyframe_rate=1):
    """Pre-processing for bbox
    .. code-block:: none

                                       up z
                        front x           ^
                             /            |
                            /             |
              (x1, y0, z1) + -----------  + (x1, y1, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)

    Args:
        bbox_mode (str): type of bbox raw data.
            cxyz -> x1y1z1, x1y0z1, x1y1z0, x0y1z1;
            all-xyz -> all 8 corners xyz;
            owhr -> center, l, w, h, z-orientation.
        canvas_size (2-tuple): H, W of input images
        examples: collate_fn input
        view_shared: if enabled, all views share same set of bbox and output
            N_cam=1; otherwise, use projection to keep only visible bboxes.
    Return:
        in form of dict:
            bboxes (Tensor): B, N_cam, max_len, ...
            classes (LongTensor): B, N_cam, max_len
            masks: 1 for data, 0 for padding
    """
    # init data
    bboxes = []
    classes = []
    max_len = 0
    gt_bboxes_3d: List[LiDARInstance3DBoxes] = [
        example["gt_bboxes_3d"].data for example in examples]
    gt_labels_3d: List[torch.Tensor] = [
        example["gt_labels_3d"].data for example in examples]

    # params
    B = len(gt_bboxes_3d)
    N_cam = len(examples[0]['lidar2image'].data.numpy())
    N_out = 1 if view_shared else N_cam

    bboxes_coord = None
    if not view_shared and not use_3d_filter:
        bboxes_coord = _transform_all(examples, 'lidar2image', True)
    elif not view_shared:
        bboxes_coord_3d = _transform_all(examples, 'lidar2camera', False)

    # for each keyframe set
    for idx in range(B):
        bboxes_kf = gt_bboxes_3d[idx]
        classes_kf = gt_labels_3d[idx]

        # if zero, add zero length tensor (for padding).
        if len(bboxes_kf) == 0:
            set_box_to_none = True
        elif idx % keyframe_rate != 0 and is_train:  # only for non-keyframes
            if random.random() < bbox_drop_ratio:
                set_box_to_none = True
            else:
                set_box_to_none = False
        else:
            set_box_to_none = False
        if set_box_to_none:
            bboxes.append([None] * N_out)
            classes.append([None] * N_out)
            continue

        # whether share the boxes across views, filtered by 2d projection.
        if not view_shared:
            index_list = []  # each view has a mask
            if use_3d_filter:
                coords_list = bboxes_coord_3d[idx]
                filter_func = ensure_positive_z
            else:
                # filter bbox according to 2d projection on image canvas
                coords_list = bboxes_coord[idx]
                # judge coord by cancas_size
                filter_func = partial(ensure_canvas, canvas_size=canvas_size)
            # we do not need to handle None since we already filter for len=0
            for coords in coords_list:
                c_mask = filter_func(coords)
                if random.random() < bbox_add_ratio and is_train:
                    c_mask = random_0_to_1(c_mask, bbox_add_num)
                index_list.append(c_mask)
                max_len = max(max_len, c_mask.sum())
        else:
            # we use as mask, torch.bool is important
            index_list = [torch.ones(len(bboxes_kf), dtype=torch.bool)]
            max_len = max(max_len, len(bboxes_kf))

        # construct data
        if bbox_mode == 'cxyz':
            # x1y1z1, x1y0z1, x1y1z0, x0y1z1
            bboxes_pt = bboxes_kf.corners[:, [6, 5, 7, 2]]
        elif bbox_mode == 'all-xyz':
            bboxes_pt = bboxes_kf.corners  # n x 8 x 3
        elif bbox_mode == 'owhr':
            raise NotImplementedError("Not sure how to do this.")
        else:
            raise NotImplementedError(f"Wrong mode {bbox_mode}")
        bboxes.append([bboxes_pt[ind] for ind in index_list])
        classes.append([classes_kf[ind] for ind in index_list])
        bbox_shape = bboxes_pt.shape[1:]

    # there is no (visible) boxes in this batch
    if max_len == 0:
        return None, None

    # pad and construct mask
    # `bbox_shape` should be set correctly
    ret_dict = pad_bboxes_to_maxlen(
        [B, N_out, max_len, *bbox_shape], max_len, bboxes, classes)
    return ret_dict, bboxes_coord


def pad_bboxes_to_maxlen(
        bbox_shape, max_len, bboxes=None, classes=None, masks=None, **kwargs):
    B, N_out = bbox_shape[:2]
    ret_bboxes = torch.zeros(B, N_out, max_len, *bbox_shape[3:])
    # we set unknown to -1. since we have mask, it does not matter.
    ret_classes = -torch.ones(B, N_out, max_len, dtype=torch.long)
    ret_masks = torch.zeros(B, N_out, max_len, dtype=torch.bool)
    if bboxes is not None:
        for _b in range(B):
            _bboxes = bboxes[_b]
            _classes = classes[_b]
            for _n in range(N_out):
                if _bboxes[_n] is None:
                    continue  # empty for this view
                this_box_num = len(_bboxes[_n])
                ret_bboxes[_b, _n, :this_box_num] = _bboxes[_n]
                ret_classes[_b, _n, :this_box_num] = _classes[_n]
                if masks is not None:
                    ret_masks[_b, _n, :this_box_num] = masks[_b, _n]
                else:
                    ret_masks[_b, _n, :this_box_num] = True

    # assemble as input format
    ret_dict = {
        "bboxes": ret_bboxes,
        "classes": ret_classes,
        "masks": ret_masks
    }
    return ret_dict


def draw_cube_mask(canvas_size, coords):
    """draw bbox in cube as mask

    Args:
        canvas_size (Tuple): (w, h) output sparital shape
        coords (np.array): (N, 8, 3) or (N, 8, 2), bbox

    Returns:
        np.array: canvas_size shape, binary mask
    """
    canvas = np.zeros((*canvas_size, 3))
    for index in range(len(coords)):
        for p1, p2, p3, p4 in [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [3, 2, 6, 7],
            [0, 4, 7, 3],
        ]:
            cv2.fillPoly(
                canvas,
                [coords[index, [p1, p2, p3, p4]].astype(np.int32)[..., :2]],
                (255, 0, 0),
            )
    # only draw on first channel, here we take first channel
    canvas[canvas > 0] = 1
    return canvas[..., 0]


def _get_fg_cube_mask(bbox_view_coord, canvas_size, examples):
    """get foreground mask according to bbox

    Args:
        bbox_view_coord (np.array): 2d coordinate of bbox on each view
        examples (_type_): raw_data, will use if bbox_view_coord is None.

    Returns:
        torch.FloatTensor: binary mask with shape (B, N_cam, W, H)
    """
    # TODO: this method is problematic, some off-canvas points are not handled
    # correctly. It should consider viewing frustum.
    if bbox_view_coord is None:
        bbox_view_coord = _transform_all(examples, 'lidar2image', True)
    B = len(bbox_view_coord)
    N_cam = len(bbox_view_coord[0])
    view_fg_mask = np.zeros((B, N_cam, *canvas_size))
    for _b in range(B):
        for _n in range(N_cam):
            coords = bbox_view_coord[_b][_n]
            if coords is None:
                break  # one cam is None, all cams are None, just skip
            mask = ensure_canvas(coords, canvas_size)
            coords = coords[mask][..., :2]  # Nx8x2
            view_fg_mask[_b, _n] = draw_cube_mask(canvas_size, coords)
    view_fg_mask = torch.from_numpy(view_fg_mask)
    return view_fg_mask


def collate_fn_single_clip(
    examples: Tuple[dict, ...],
    template: str,
    frame_emb=None,
    tokenizer: CLIPTokenizer = None,
    is_train: bool = True,
    bbox_mode: str = None,
    bbox_view_shared: bool = False,
    bbox_drop_ratio: float = 0,
    bbox_add_ratio: float = 0,
    bbox_add_num: int = 3,
    foreground_loss_mode: str = None,
    keyframe_rate: int = 1,
    bbox_processor_type: int = 1,
    return_raw_data = False,
):
    """
    We need to handle:
    1. make multi-view images (img) into tensor -> [N, 6, 3, H, W]
    2. make masks (gt_masks_bev, gt_aux_bev) into tensor
        -> [N, 25 = 8 map + 10 obj + 7 aux, 200, 200]
    3. make caption (location, desctiption, timeofday) and tokenize, padding
        -> [N, pad_length]
    4. extract camera parameters (camera_intrinsics, camera2lidar)
        camera2lidar: A @ v_camera = v_lidar
        -> [N, 6, 3, 7]
    We keep other meta data as original.
    """
    if return_raw_data:
        return examples
    if bbox_add_ratio > 0:
        assert bbox_view_shared == False, "You cannot add any box on view shared."
    if bbox_drop_ratio > 0 and is_train:
        if keyframe_rate == 1:
            logging.warning("You set bbox_drop_ratio>0, but with keyframe_rate=1, nothing will be dropped. Please check!")
    bbox_processors = {
        1: _preprocess_bbox,
        2: _preprocess_bbox_keep_all,
    }

    # multi-view images
    pixel_values = torch.stack([example["img"].data for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    # mask
    if "gt_aux_bev" in examples[0] and examples[0]["gt_aux_bev"] is not None:
        keys = ["gt_masks_bev", "gt_aux_bev"]
        assert bbox_drop_ratio == 0, "map is not affected in bbox_drop"
    else:
        keys = ["gt_masks_bev"]
    # fmt: off
    try:
        bev_map_with_aux = torch.stack([torch.from_numpy(np.concatenate([
            example[key] for key in keys  # np array, channel-last
        ], axis=0)).float() for example in examples], dim=0)  # float32
    except:
        bev_map_with_aux = torch.zeros((pixel_values.shape[0], 8, 400, 400,))
    # fmt: on

    # camera param
    # TODO: camera2lidar should be changed to lidar2camera
    # fmt: off
    camera_param = torch.stack([torch.cat([
        example["camera_intrinsics"].data[:, :3, :3],  # 3x3 is enough
        example["camera2lidar"].data[:, :3],  # only first 3 rows meaningful
    ], dim=-1) for example in examples], dim=0)
    # fmt: on
    camera_int = torch.stack([
        example["camera_intrinsics"].data[:, :3, :3]  # 3x3 is enought
        for example in examples], dim=0)
    camera_ext = torch.stack([
        example["lidar2camera"].data for example in examples], dim=0)
    # aug is either eye or has values
    camera_aug = torch.stack([
        example["img_aug_matrix"].data for example in examples], dim=0)

    ret_dict = {
        "pixel_values": pixel_values,
        "bev_map_with_aux": bev_map_with_aux,
        "camera_param": camera_param,
        "camera_param_raw": {
            "int": camera_int,
            "ext": camera_ext,
            "aug": camera_aug,
        },
    }

    if "next2top" in examples[0]:
        next2top = []
        for example in examples:
            _ego_v = example["next2top"].data  # 4x4
            next2top.append(_ego_v)
        next2top = torch.stack(next2top, dim=0)

    if frame_emb == "next2top":
        ret_dict['frame_emb'] = next2top
    else:
        assert frame_emb == None
        ret_dict['frame_emb'] = None

    # bboxes_3d, convert to tensor
    # here we consider:
    # 1. do we need to filter bboxes for each view? use `view_shared`
    # 2. padding for one batch of data if need (with zero), and output mask.
    # 3. what is the expected output format? dict of kwargs to bbox embedder
    # TODO: should we change to frame's coordinate?
    canvas_size = pixel_values.shape[-2:]
    if bbox_mode is not None:
        # NOTE: both can be None
        bboxes_3d_input, bbox_view_coord = bbox_processors[int(bbox_processor_type)](
            bbox_mode, canvas_size, examples, is_train=is_train,
            view_shared=bbox_view_shared, bbox_add_ratio=bbox_add_ratio,
            bbox_add_num=bbox_add_num, bbox_drop_ratio=bbox_drop_ratio,
            keyframe_rate=keyframe_rate)
        # if bboxes_3d_input is not None:
        #     bboxes_3d_input["cam_params"] = camera_param
        ret_dict["bboxes_3d_data"] = DataContainer(bboxes_3d_input)
    else:
        bbox_view_coord = None

    if foreground_loss_mode == "bbox":
        ret_dict['view_fg_mask'] = _get_fg_cube_mask(
            bbox_view_coord, canvas_size, examples)
    elif foreground_loss_mode == "pc_seg":
        raise NotImplementedError(foreground_loss_mode)
    elif foreground_loss_mode is not None:
        raise TypeError(foreground_loss_mode)

    # captions: one real caption with one null caption
    input_ids_padded, captions = _tokenize_captions(
        examples, template, tokenizer, is_train)
    ret_dict["captions"] = captions[:-1]  # list of str
    if tokenizer is not None:
        # real captions in head; the last one is null caption
        # we omit "attention_mask": padded_tokens.attention_mask, seems useless
        ret_dict["input_ids"] = input_ids_padded[:-1]
        ret_dict["uncond_ids"] = input_ids_padded[-1:]

    # other meta data
    meta_list_dict = dict()
    for key in META_KEY_LIST:
        try:
            meta_list = [example[key] for example in examples]
            meta_list_dict[key] = meta_list
        except KeyError:
            continue
    ret_dict['meta_data'] = meta_list_dict

    return ret_dict


@DATASETS.register_module()
class NuScenesTDataset(NuScenesDataset):
    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
        force_all_boxes=False,
        video_length=None,
        start_on_keyframe=True,
        start_on_firstframe=False,
        next2topv2=True,
        trans_box2top=False,
        fps=12,
        img_collate_param={},
        micro_frame_size=None,
        balance_keywords=None,
        drop_ori_imgs=False,
        additional_image_annotations: dict=None,
        **kwargs,
    ) -> None:
        self.video_length = video_length
        self.start_on_keyframe = start_on_keyframe
        self.start_on_firstframe = start_on_firstframe
        self.micro_frame_size = micro_frame_size
        self.balance_keywords = balance_keywords
        super().__init__(
            ann_file, pipeline, dataset_root, object_classes, map_classes,
            load_interval, with_velocity, modality, box_type_3d,
            filter_empty_gt, test_mode, eval_version, use_valid_flag,
            force_all_boxes)
        if "12Hz" in ann_file and start_on_keyframe:
            logging.warning("12Hz should use all starting frame to train, please "
                         "double-check!")
        self.next2topv2 = next2topv2
        self.trans_box2top = trans_box2top
        self.allow_class = kwargs.pop("allow_class", None)
        if self.allow_class is not None:
            logging.info(f"Your allow_class = {self.allow_class}")
        self.del_box_ratio = kwargs.pop("del_box_ratio", 0.0)
        self.drop_nearest_car = kwargs.pop("drop_nearest_car", 0)
        if self.del_box_ratio > 0 or self.drop_nearest_car > 0:
            logging.info(f"Your del_box_ratio = {self.del_box_ratio}, "
                         f"drop_nearest_car = {self.drop_nearest_car}")
        self.img_collate_param = img_collate_param
        if isinstance(self.img_collate_param, ConfigDict):
            self.img_collate_param = img_collate_param.to_dict()
        self.fps = fps
        self.drop_ori_imgs = drop_ori_imgs
        self.additional_image_annotations = additional_image_annotations

    @property
    def num_frames(self):
        return self.video_length

    def balance_annotations(self, data_infos, scene_tokens):
        keywords_dict = {k: [] for k in self.balance_keywords}
        if "none" in keywords_dict:  # we care about none: will force add "daytime"
            prepend_daytime = True
            logging.warning("[Balance] force add daytime keyword.")
        else:
            prepend_daytime = False
            keywords_dict['none'] = []  # for no keywords
        for scene in scene_tokens:
            # NOTE: we assume annotations are the same for the whole scene clip 
            first_frame: dict = data_infos[self.token_data_dict[scene[0]]]
            first_frame_anno: str = first_frame['description'].lower()
            matched_keywords = [keyword for keyword in self.balance_keywords if keyword in first_frame_anno]
            if matched_keywords:
                for keyword in matched_keywords:
                    assert keyword != "none"
                    keywords_dict[keyword].append(scene)
            else:
                # TODO: hard-coded keyword!
                if prepend_daytime and "daytime" not in first_frame_anno:
                    for _s in scene:
                        data_infos[self.token_data_dict[_s]]['description'] = "Daytime. " + data_infos[self.token_data_dict[_s]]['description']
                keywords_dict["none"].append(scene)

        counts = {key: len(items) for key, items in keywords_dict.items()}
        all_count = sum(counts.values())
        if all_count != len(scene_tokens):
            # TODO: consider overlapping?
            logging.warning(
                f"[Balance] You have {len(scene_tokens)} clips while we "
                f"categorized {all_count} clips. Keywords have overlaps.")
        max_count = max(counts.values())

        balanced_data = []
        for keyword, items in keywords_dict.items():
            repeat_times = max_count // counts[keyword]
            items = items * repeat_times
            logging.info(
                f"[Balance] We repeat {keyword} for {repeat_times} times: "
                f"{counts[keyword]} -> {len(items)}")
            balanced_data.extend(items)

        return data_infos, balanced_data

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene 

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        skip1, skip2 = 0, 0
        for sid, scene in enumerate(scene_tokens):
            if self.video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    # trim to micro_frame_size
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
            else:
                assert isinstance(self.video_length, int)
                if sid in []:
                    logging.info(f"Got {len(all_clips)} for sid={sid}.")
                if self.start_on_firstframe:
                    first_frames = [0]
                else:
                    first_frames = range(len(scene) - self.video_length + 1)
                for start in first_frames:
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue  # this is not a keyframe
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue  # this is not a keyframe
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + self.video_length]]
                    all_clips.append(clip)
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames. "
                     f"start_on_firstframe={self.start_on_firstframe}")
        return all_clips

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.clip_infos = self.build_clips(data_infos, data['scene_tokens'])
        return data_infos

    def resize_nearest(self, img, size):
        # size 应该是 (height, width)
        return img.resize((size[1], size[0]), Image.NEAREST)

    def __len__(self):
        return len(self.clip_infos)

    def load_clip(self, clip):
        frames = []
        first_info = self.data_infos[clip[0]]
        for frame in clip:
            frame_info = super().get_data_info(frame)
            info = self.data_infos[frame]
            next2top = obtain_next2top(first_info, info, v2=self.next2topv2)
            frame_info['next2top'] = next2top
            frames.append(frame_info)
        return frames

    def get_data_info(self, index):
        """We should sample from clip_infos
        """
        clip = self.clip_infos[index]
        frames = self.load_clip(clip)
        return frames

    def get_ann_info(self, index):
        anns_results, mask = super().get_ann_info(index)
        info = self.data_infos[index]
        if "gt_box_ids" not in info:
            if self.allow_class is not None or self.del_box_ratio > 0:
                logging.warning(
                    "There is no 'gt_box_ids', your filter is disabled.")
                self.del_box_ratio = 0.0
            return anns_results, mask

        gt_bboxes_3d = anns_results['gt_bboxes_3d'].tensor

        # add token
        gt_bboxes_token = [info["gt_box_ids"][i] for i in np.where(mask)[0]]
        token_idxes = torch.arange(
            len(gt_bboxes_token),
            dtype=gt_bboxes_3d.dtype)
        gt_bboxes_3d = torch.cat([
            gt_bboxes_3d, token_idxes.unsqueeze(-1)], dim=-1)

        # rebuild boxes
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0),
            tokens=gt_bboxes_token,
        )
        anns_results['gt_bboxes_3d'] = gt_bboxes_3d
        return anns_results, mask

    def rand_del_box(self, examples, drop_ratio, allow_class, drop_nearest_car,
                     dist_use_frame=-1):
        # allow_class = [0, 1, 2, 3, 4, 5, 9]
        if examples[0]['gt_bboxes_3d'].data.raw_tokens is None:
            for example in examples:
                indexes = []
                for idx in range(example['gt_bboxes_3d'].data.tensor.shape[0]):
                    if allow_class is None:
                        class_in = True
                    else:
                        class_in = int(
                            example['gt_labels_3d'].data[idx]) in allow_class
                    if class_in:
                        indexes.append(idx)
                example['gt_bboxes_3d'] = DataContainer(
                    example['gt_bboxes_3d'].data[indexes])
                example['gt_labels_3d'] = DataContainer(
                    example['gt_labels_3d'].data[indexes])
            return

        possible_tokens = set()
        for example in examples:
            _tokens = set([t for t in example['gt_bboxes_3d'].data.tokens])
            possible_tokens = possible_tokens.union(_tokens)
        possible_tokens = sorted(list(possible_tokens))
        random.shuffle(possible_tokens)

        # drop first tokens
        possible_tokens = possible_tokens[int(len(possible_tokens) * drop_ratio):]

        if drop_ratio > 0 or drop_nearest_car > 0:
            # get distance according to one frame
            if dist_use_frame == -1:
                dist_use_frame = len(examples) // 2
            logging.info(f"[Random Drop] Use dist from {dist_use_frame} frame")
            all_tokens = examples[dist_use_frame]['gt_bboxes_3d'].data.tokens
            car_tokens = [all_tokens[i] for i in
                        torch.where(examples[dist_use_frame]['gt_labels_3d'].data == 0)[0]]
            centers = examples[dist_use_frame]['gt_bboxes_3d'].data.center
            car_centers = [centers[i] for i in
                        torch.where(examples[dist_use_frame]['gt_labels_3d'].data == 0)[0]]
            if len(car_centers) == 0:
                car_tokens = []
            else:
                dist = (torch.stack(car_centers, dim=0) ** 2).sum(-1)
                assert len(dist) == len(car_tokens)
                car_tokens = [car_tokens[i] for i in torch.argsort(dist)]
            # default: make sure all 3-nearest cars are here
            possible_tokens = set(possible_tokens) | set(car_tokens[:3])
            if drop_nearest_car > 0:  # if set, make sure they are dropped
                possible_tokens = set(possible_tokens) - set(car_tokens[:drop_nearest_car])

        # gather annotations after dropping (will change in-place)
        for example in examples:
            indexes = []
            for idx, token in enumerate(example['gt_bboxes_3d'].data.tokens):
                token_in = token in possible_tokens
                if allow_class is None:
                    class_in = True
                else:
                    class_in = int(
                        example['gt_labels_3d'].data[idx]) in allow_class
                if token_in and class_in:
                    indexes.append(idx)
            example['gt_bboxes_3d'] = DataContainer(
                example['gt_bboxes_3d'].data[indexes])
            example['gt_labels_3d'] = DataContainer(
                example['gt_labels_3d'].data[indexes])

    def load_frames(self, frames):
        if None in frames:
            return None
        examples = []
        first_frame_boxes = None
        for frame in frames:
            self.pre_pipeline(frame)
            example = self.pipeline(frame)
            if self.filter_empty_gt and frame['is_key_frame'] and (
                example is None or ~(example["gt_labels_3d"]._data != -1).any()
            ):
                return None
            if self.trans_box2top:
                if first_frame_boxes is None:
                    first_frame_boxes = {
                        'gt_bboxes_3d': example['gt_bboxes_3d'],
                        'gt_labels_3d': example['gt_labels_3d'],
                    }
                else:
                    this_frame_boxes = transform_bbox(
                        first_frame_boxes, frame['next2top'])
                    example['gt_bboxes_3d'] = this_frame_boxes['gt_bboxes_3d']
                    example['gt_labels_3d'] = this_frame_boxes['gt_labels_3d']

            if self.additional_image_annotations is not None:
                # for data_dict in self.additional_image_annotations:
                for key, root_path in self.additional_image_annotations.items():
                    result_str = []
                    for img_path in example["metas"].data['filename']:
                        relative_path = '/'.join(img_path.split('/')[-2:])
                        new_filename = os.path.join(root_path, relative_path)
                        # 挖坑, hdmap 是 xxxx.png
                        if "hdmap" in key:
                            new_filename = new_filename.replace(".jpg", ".png")
                        if os.path.exists(new_filename):
                            try:
                                result_str.append(Image.open(new_filename))
                            except UnidentifiedImageError:
                                print(
                                    f"UnidentifiedImageError: cannot identify image file '{new_filename}', using black image instead.")
                                result_str.append(Image.new('RGB', (100, 100), 'black'))
                            except Exception as e:
                                print(f"Unexpected error with file '{new_filename}': {str(e)}")
                                result_str.append(Image.new('RGB', (100, 100), 'black'))
                        else:
                            print(f"File '{new_filename}' does not exist, using black image instead.")
                            result_str.append(Image.new('RGB', (100, 100), 'black'))

                    example[key] = result_str
            examples.append(example)
        if self.del_box_ratio > 0 or self.allow_class is not None or self.drop_nearest_car > 0:
            # will change in-place
            self.rand_del_box(
                examples, self.del_box_ratio, self.allow_class, self.drop_nearest_car)
        ret_dicts = collate_fn_single_clip(examples, **self.img_collate_param)
        if self.img_collate_param.get("return_raw_data", False):
            return ret_dicts
        ret_dicts['height'] = ret_dicts['pixel_values'].shape[-2]
        ret_dicts['width'] = ret_dicts['pixel_values'].shape[-1]

        # cond
        bbox = None
        for key in self.additional_image_annotations.keys():
            if key in examples[0].keys():
                cond = torch.stack(
                    [torch.stack([self.condition_transform(i) for i in exp[key]]) for exp in examples])
                if bbox is not None:
                    # import pdb; pdb.set_trace()
                    bbox = torch.clamp(cond + bbox, 0.0, 1.0)
                else:
                    bbox = cond
        bbox = bbox.permute(1, 0, 2, 3, 4)  # [T, V, C, H, W] -> [V, T, C, H, W]
        ret_dicts['bbox'] = bbox
        ret_dicts["bev_map_with_aux"] = ret_dicts['bbox']
        # import pdb; pdb.set_trace()         # todo
        if self.drop_ori_imgs:
            ret_dicts["pixel_values_shape"] = torch.IntTensor(
                list(ret_dicts['pixel_values'].shape))
            ret_dicts.pop("pixel_values")
        return ret_dicts

    def prepare_train_data(self, index):
        """This is called by `__getitem__`
        """
        frames = self.get_data_info(index)
        ret_dicts = self.load_frames(frames)
        if ret_dicts is None:
            return None
        if self.img_collate_param.get("return_raw_data", False):
            return ret_dicts
        ret_dicts['fps'] = self.fps if self.num_frames == "full" or self.num_frames > 1 else IMG_FPS
        ret_dicts['num_frames'] = len(frames)
        if self.num_frames != "full":
            assert self.num_frames == len(frames)
        return ret_dicts
        
    def prepare_test_data(self, index):
        raise NotImplementedError()