"""Miscellaneous Tools"""

import addict
import ctypes
from collections.abc import Iterable
from contextlib import contextmanager
import threading
import subprocess
import traceback
from decimal import Decimal
from urllib.parse import urlparse
import cv2
import numpy as np
try:
    import torch
except ImportError:
    IS_TORCH_AVAILABLE = False
else:
    IS_TORCH_AVAILABLE = True
try:
    from mmcv.ops import bbox_overlaps, box_iou_rotated
except ImportError:
    IS_MMCV_AVAILABLE = False
else:
    IS_MMCV_AVAILABLE = True


def parse_decimal(var):
    """Parse decimal information"""
    assert var.replace('.', '', 1).isnumeric(), 'invalid decimal'
    var = str(var)
    digit = var.find('.')
    if digit == -1:
        digit, point = len(var), 0
    else:
        point = len(var) - digit - 1
    return dict(digit=digit, point=point)


def convert_numeric(var):
    """Convert numeric variable"""
    if var.isnumeric():
        return int(var)
    if var.replace('.', '', 1).isnumeric():
        return float(var)
    return var


def decode_image(buf, dtype='uint8', flags=cv2.IMREAD_UNCHANGED):
    """Decode image"""
    if isinstance(buf, (bytes, memoryview)):
        buf = np.frombuffer(buffer=buf, dtype=np.dtype(dtype))
    else:
        assert isinstance(buf, np.ndarray), \
            'invalid buffer to be decoded as array'
    img = cv2.imdecode(buf=buf, flags=flags)
    assert img is not None, f'failed to decode {dtype} buffer'
    return img


def encode_image(img, extension='.jpg', **kwargs):
    """Encode image"""
    assert isinstance(img, np.ndarray), \
        'invalid array to be encoded as buffer'
    sig, buf = cv2.imencode(ext=extension, img=img, **kwargs)
    assert sig, f'failed to encode {extension} image'
    buf = buf.tobytes()
    return buf


def modulo_axis(axis, dims=3):
    """Modulo axis"""
    return np.mod(np.arange(axis + 1, axis + dims), dims)


def normalize_homo(homo):
    """Normalize homogeneous coordinates"""
    return homo / np.where(homo[..., -1:] == 0., 1., homo[..., -1:])


def homo2euclid(homo):
    """Convert homogeneous to euclidean coordinates"""
    return normalize_homo(homo)[..., :-1]


def xywh2xyxy(bbox, px_type='point'):
    """Convert bbox format from xywh to xyxy"""
    assert bbox.ndim == 2 and bbox.shape[1] == 4, 'invalid bbox'
    segments = np.hsplit(bbox, 2)
    origin, size = segments[0], segments[1]
    if px_type == 'point':
        end = origin + size
    elif px_type == 'square':
        end = origin + size - 1
    else:
        raise ValueError('invalid pixel type')
    bbox = np.c_[origin, end]
    return bbox


def xyxy2xywh(bbox, px_type='point'):
    """Convert bbox format from xyxy to xywh"""
    assert bbox.ndim == 2 and bbox.shape[1] == 4, 'invalid bbox'
    segments = np.hsplit(bbox, 2)
    origin, end = segments[0], segments[1]
    if px_type == 'point':
        size = end - origin
    elif px_type == 'square':
        size = end - origin + 1
    else:
        raise ValueError('invalid pixel type')
    bbox = np.c_[origin, size]
    return bbox


def convert_arr(arr, device=None):
    """Conversion between numpy array and torch tensor"""
    assert IS_TORCH_AVAILABLE, 'torch is unavailable'
    if isinstance(arr, np.ndarray):
        assert isinstance(device, str), 'invalid device'
        tensor = dict(cpu=torch.FloatTensor, cuda=torch.cuda.FloatTensor)[
            device.lower()]
        return tensor(arr)
    if not isinstance(arr, torch.Tensor):
        raise TypeError('invalid array to convert')
    if arr.device.type == 'cuda':
        arr = arr.cpu()
    if arr.device.type == 'cpu':
        return arr.numpy()
    raise ValueError('array with invalid device to convert')


def boxes_iou2d(bbox1: np.ndarray, bbox2: np.ndarray, aligned=False, device='cpu', **kwargs):
    """Calculate boxes 2d IoU"""
    bbox1, bbox2 = convert_arr(bbox1, device), convert_arr(bbox2, device)
    assert IS_MMCV_AVAILABLE, 'mmcv is unavailable'
    iou = bbox_overlaps(bbox1, bbox2, aligned=aligned, mode='iou', **kwargs)
    return convert_arr(iou)


def boxes_iou3d(bbox1: np.ndarray, bbox2: np.ndarray, h_axis=2, aligned=False, device='cpu'):
    """Calculate boxes 3d IoU"""
    bbox1, bbox2 = convert_arr(bbox1, device), convert_arr(bbox2, device)
    assert bbox1.ndim == bbox2.ndim == 2 and bbox1.shape[1] == bbox2.shape[1] == 7, 'invalid bboxes'
    bev_axes = modulo_axis(h_axis, 3)
    bev_dims = [*bev_axes, *(bev_axes + 3), 6]
    h_dim = h_axis + 3
    assert IS_MMCV_AVAILABLE, 'mmcv is unavailable'
    iou_bev = box_iou_rotated(
        bbox1[:, bev_dims], bbox2[:, bev_dims], aligned=aligned, clockwise=h_axis == 1)
    area1, area2 = bbox1[:, bev_dims[2:4]].prod(
        dim=1), bbox2[:, bev_dims[2:4]].prod(dim=1)
    hoff1, hoff2 = bbox1[:, h_dim] / 2, bbox2[:, h_dim] / 2
    if aligned:
        area = area1 + area2
        min_h = torch.max(bbox1[:, h_axis] - hoff1, bbox2[:, h_axis] - hoff2)
        max_h = torch.min(bbox1[:, h_axis] + hoff1, bbox2[:, h_axis] + hoff2)
        volumes = area1 * bbox1[:, h_dim] + area2 * bbox2[:, h_dim]
    else:
        area = area1[:, None] + area2[None, :]
        min_h = torch.max(bbox1[:, h_axis, None] - hoff1[:, None],
                          bbox2[None, :, h_axis] - hoff2[None, :])
        max_h = torch.min(bbox1[:, h_axis, None] + hoff1[:, None],
                          bbox2[None, :, h_axis] + hoff2[None, :])
        volumes = (area1 * bbox1[:, h_dim])[:, None] + \
            (area2 * bbox2[:, h_dim])[None, :]
    overlaps_bev = area * iou_bev / (1 + iou_bev)
    overlaps_h = torch.clamp(max_h - min_h, min=0)
    overlaps = overlaps_bev * overlaps_h
    iou = overlaps / torch.clamp(volumes - overlaps, min=1e-6)
    return convert_arr(iou)


def boxes_iou_bev(bbox1: np.ndarray, bbox2: np.ndarray, aligned=False, device='cpu'):
    """Calculate boxes IoU in bird's-eye view"""
    bbox1, bbox2 = convert_arr(bbox1, device), convert_arr(bbox2, device)
    assert bbox1.ndim == bbox2.ndim == 2 and bbox1.shape[1] == bbox2.shape[1] == 5, 'invalid bboxes'
    bbox1 = torch.cat([.5 * (bbox1[:, 0:2] + bbox1[:, 2:4]),
                      bbox1[:, 2:4] - bbox1[:, 0:2], bbox1[:, 4:]], dim=-1)
    bbox2 = torch.cat([.5 * (bbox2[:, 0:2] + bbox2[:, 2:4]),
                      bbox2[:, 2:4] - bbox2[:, 0:2], bbox2[:, 4:]], dim=-1)
    assert IS_MMCV_AVAILABLE, 'mmcv is unavailable'
    iou = box_iou_rotated(bbox1, bbox2, aligned=aligned)
    return convert_arr(iou)


def normalize_vec(vec, axis=-1):
    """Normalize vector"""
    norm = np.linalg.norm(vec, axis=axis)
    vec /= norm
    return vec


def normalize_rad(rad):
    """Normalize radian"""
    return np.mod(rad + np.pi, 2 * np.pi) - np.pi


def convert_orien(orien, center, axes, tgt_type='local'):
    """Convert orientation format between global yaw and local alpha"""
    assert orien.ndim == 2 and orien.shape[1] == 1, 'invalid orientation'
    assert center.ndim == 2 and orien.shape[0] == center.shape[0], 'invalid center'
    assert len(axes) == 2, 'invalid axes'
    slope = np.arctan2(center[:, axes[1]], center[:, axes[0]])[..., None]
    if tgt_type == 'local':
        orien -= slope
    elif tgt_type == 'global':
        orien += slope
    else:
        raise ValueError('invalid conversion type')
    return normalize_rad(orien)


def radian2matrix(rad, axes=None):
    """Convert orientation radian to rotation matrix"""
    assert rad.ndim == 2 and rad.shape[1] == 1, 'invalid rad'
    sin_rot, cos_rot = np.sin(rad), np.cos(rad)
    rot_tmplt = np.concatenate(
        [cos_rot, sin_rot, -sin_rot, cos_rot], axis=-1).reshape(-1, 2, 2)
    if axes is None:
        return rot_tmplt.transpose(0, 2, 1)
    assert len(axes) == 2, 'invalid axes'
    rot_mat = np.tile(np.eye(3), reps=(rot_tmplt.shape[0], 1, 1))
    rot_idx = np.stack(np.meshgrid(axes, axes), axis=0)
    rot_mat[:, rot_idx[0], rot_idx[1]] = rot_tmplt
    return rot_mat


def validate_url(url, print_exc=False):
    """Validate url"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        if print_exc:
            print(traceback.format_exc())
        return False


def check_output(cmd_str, decode_output=True, **kwargs):
    """Check output"""
    if isinstance(cmd_str, str):
        cmd_str = cmd_str.split(' ')
    elif not isinstance(cmd_str, (list, tuple)):
        raise TypeError('invalid command')
    output = subprocess.check_output(
        cmd_str, stderr=subprocess.PIPE, **kwargs).strip()
    if decode_output:
        output = output.decode(encoding='utf-8')
    return output


def validate_exe(exe, cmds=None, print_exc=False, return_output=False):
    """Validate exe"""
    if cmds is None:
        # cmds = ['-V', '-v']
        output = check_output(['bash', '-c', f'command -v {exe}'])
        if not output:
            output = None
        return output if return_output else output is not None
    if isinstance(cmds, str):
        cmds = [cmds]
    elif not isinstance(cmds, (list, tuple, set)):
        raise TypeError('invalid command')
    for _cmd in cmds:
        try:
            output = check_output([exe, _cmd])
        except FileNotFoundError as exc:
            if print_exc:
                print(exc)
        else:
            break
    else:
        output = None
    return output if return_output else output is not None


@contextmanager
def limit_timeout(timeout):
    """Limit timeout"""
    if timeout is None:
        yield
        return
    thread = ctypes.c_long(threading.current_thread().ident)

    def _handler():
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread, ctypes.py_object(TimeoutError))
        if ret == 0:
            raise ValueError(f'invalid thread id: {thread}')
        if ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread, None)
            raise RuntimeError('failed to set async exception')

    timer = threading.Timer(timeout, _handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def update_kwargs(default, update=None, keepattr=False, recursive=True):
    """Update default keyword arguments with update"""
    if default is None:
        default = dict()
    elif not isinstance(default, dict):
        raise TypeError('invalid default type')
    if update is None:
        return dict(default)
    assert isinstance(update, dict), 'invalid update type'
    assert not keepattr or len(update.keys() - default) == 0, \
        'invalid keyword contained'
    kwargs = dict(default)
    if not recursive:
        kwargs.update(update)
        return kwargs

    def _iterable(_arg):
        return not isinstance(_arg, str) and isinstance(_arg, Iterable)

    for key in update:
        _def, _upd = kwargs.setdefault(key, None), update[key]
        if _upd is None:
            continue
        if isinstance(_upd, dict):
            kwargs[key] = update_kwargs(_def, _upd, keepattr, recursive)
            continue
        if not _iterable(_upd):
            kwargs[key] = _upd
            continue
        assert not list(filter(_iterable, _upd)), 'nested non-dict detected'
        kwargs[key] = _upd.copy()
    return kwargs
