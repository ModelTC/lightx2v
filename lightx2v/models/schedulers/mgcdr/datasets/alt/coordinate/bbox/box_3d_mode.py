from enum import IntEnum, unique

import numpy as np
import torch

from .base_box3d import BaseInstance3DBoxes
from .cam_box3d import CameraInstance3DBoxes, RoadCamInstance3DBoxes
from .depth_box3d import DepthInstance3DBoxes
from .lidar_box3d import LiDARInstance3DBoxes
from .utils import limit_period

# from scipy.spatial.transform import Rotation as R

__all__ = ['BaseInstance3DBoxes', 'CameraInstance3DBoxes', 'RoadCamInstance3DBoxes',
           'DepthInstance3DBoxes', 'LiDARInstance3DBoxes', 'Box3DMode']


@unique
class Box3DMode(IntEnum):
    r"""Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is [0.5, 1.0, 0.5],
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth mode:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2
    ROADCAM = 3

    @staticmethod
    def convert(box, src, dst, rt_mat=None, with_yaw=True, gplane=None):
        """Convert boxes from `src` mode to `dst` mode.

        Args:
            box (tuple | list | np.ndarray |
                torch.Tensor | :obj:`BaseInstance3DBoxes`):
                Can be a k-tuple, k-list or an Nxk array/tensor, where k = 7.
            src (:obj:`Box3DMode`): The src Box mode.
            dst (:obj:`Box3DMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.
            with_yaw (bool, optional): If `box` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.

        Returns:
            (tuple | list | np.ndarray | torch.Tensor |
                :obj:`BaseInstance3DBoxes`):
                The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                'Box3DMode.convert takes either a k-tuple/list or '
                'an Nxk array/tensor, where k >= 7')
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        if is_Instance3DBoxes:
            with_yaw = box.with_yaw

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if with_yaw:
            yaw = arr[..., 6:7]
        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.LIDAR and dst == Box3DMode.ROADCAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                # 采用投影来计算
                # 雷达坐标系下目标与xy平面平行，构造朝向向量的z坐标直接用0
                point_y = torch.cat((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)), dim=1)  # noqa
                point_o = torch.zeros_like(point_y)
                point_o[:, 3] = 1
                # 朝向向量转到相机坐标系下
                point_y_cam = point_y @ rt_mat.t()
                point_o_cam = point_o @ rt_mat.t()
                cam_dir = point_y_cam - point_o_cam
                # 此时朝向向量与相机坐标系各个轴均不平行，投影至xz平面求新的yaw
                yaw = torch.atan2(-cam_dir[:, 2], cam_dir[:, 0]).unsqueeze(1)

        elif src == Box3DMode.ROADCAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                # 采用投影来计算
                # 通过外参计算出相机坐标系下地面方程系数
                gplane = torch.tensor([0, 0, 1, 0], dtype=torch.float32) @ rt_mat
                # 在相机坐标系中构造朝向向量，但真实目标在地面方程平面，而不在xz平面
                # 通过满足地面方程求出端点的y坐标
                point_y_y = -(gplane[0] * torch.cos(yaw) - gplane[2] * torch.sin(yaw) + gplane[3]) / gplane[1]
                point_y = torch.cat((torch.cos(yaw), point_y_y, -torch.sin(yaw), torch.ones_like(yaw)), dim=1)
                point_o = torch.zeros_like(point_y)
                point_o[:, 1] = -gplane[3] / gplane[1]
                point_o[:, 3] = 1
                # 朝向向量转到雷达坐标系下
                point_y_lidar = point_y @ rt_mat.t()
                point_o_lidar = point_o @ rt_mat.t()
                lidar_dir = point_y_lidar - point_o_lidar
                # 此时朝向向量与雷达坐标系的xy平面平行，求出新的yaw
                yaw = torch.atan2(lidar_dir[:, 1], lidar_dir[:, 0]).unsqueeze(1)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                yaw = -yaw
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                yaw = yaw + np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                yaw = yaw - np.pi / 2
                yaw = limit_period(yaw, period=np.pi * 2)
        else:
            raise NotImplementedError(
                f'Conversion from Box3DMode {src} to {dst} '
                'is not supported yet')

        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[..., :3] @ rt_mat.t()

        if with_yaw:
            remains = arr[..., 7:]
            arr = torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)
        else:
            remains = arr[..., 6:]
            arr = torch.cat([xyz[..., :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
                return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw, origin=(0.5, 1.0, 0.5))
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            elif dst == Box3DMode.ROADCAM:
                target_type = RoadCamInstance3DBoxes
                assert gplane is not None
                return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw, origin=(0.5, 1.0, 0.5), gplane=gplane)
            else:
                raise NotImplementedError(
                    f'Conversion to {dst} through {original_type}'
                    ' is not supported yet')
            return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw)
        else:
            return arr
