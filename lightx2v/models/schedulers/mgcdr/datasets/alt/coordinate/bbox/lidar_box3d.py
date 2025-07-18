# Import from third library
import numpy as np
import torch

# Import from alt
from alt.coordinate.points import BasePoints
from alt.coordinate.points.utils import Camera3DPointsTransfer

# Import from local
from .base_box3d import BaseInstance3DBoxes
from .utils import rotation_3d_in_axis, fisheye_camera_to_image


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    """3D boxes of instances in LIDAR coordinates.

    Coordinates in LiDAR:

    .. code-block:: none

                                up z    x front (yaw=0)
                                   ^   ^
                                   |  /
                                   | /
       (yaw=0.5*pi) left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and increases from
    the positive direction of x to the positive direction of y.

    A refactor is ongoing to make the three coordinate systems
    easier to understand and convert between each other.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicating the dimension of a box.
            Each row is (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

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
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims #scale [b,3]
        # [1, 8, 3]
        corners_norm = torch.from_numpy(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
            device=dims.device, dtype=dims.dtype
        )

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin [0.5, 0.5, 0]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate around z axis
        corners = rotation_3d_in_axis(corners, self.tensor[:, 6], axis=self.YAW_AXIS)
        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners # (n,8,3)

    def rotate(self, angle, points=None):
        """Rotate boxes with points (optional) with the given angle or rotation
        matrix.

        Args:
            angles (float | torch.Tensor | np.ndarray):
                Rotation angle or rotation matrix.
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            tuple or None: When ``points`` is None, the function returns
                None, otherwise it returns the rotated points and the
                rotation matrix ``rot_mat_T``.
        """
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, f"invalid rotation angle shape {angle.shape}"

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3], angle, axis=self.YAW_AXIS, return_mat=True
            )
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if self.tensor.shape[1] == 9:
            # rotate velo vector
            self.tensor[:, 7:9] = self.tensor[:, 7:9] @ rot_mat_T[:2, :2]

        if points is not None:
            if isinstance(points, torch.Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def flip(self, bev_direction="horizontal", points=None):
        """Flip the boxes in BEV along given BEV direction.

        In LIDAR coordinates, it flips the y (horizontal) or x (vertical) axis.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
            points (torch.Tensor | np.ndarray | :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or None: Flipped points.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1::7] = -self.tensor[:, 1::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        elif bev_direction == "vertical":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            if isinstance(points, (torch.Tensor, np.ndarray)):
                if bev_direction == "horizontal":
                    points[:, 1] = -points[:, 1]
                elif bev_direction == "vertical":
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points

    def convert_to(self, dst, rt_mat=None, **kwargs):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`Box3DMode`): the target Box mode
            rt_mat (np.ndarray | torch.Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None.
                The conversion from ``src`` coordinates to ``dst`` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BaseInstance3DBoxes`:
                The converted box of the same type in the ``dst`` mode.
        """
        # Import from local
        from .box_3d_mode import Box3DMode

        return Box3DMode.convert(box=self, src=Box3DMode.LIDAR, dst=dst, rt_mat=rt_mat, **kwargs)

    def enlarged_box(self, extra_width):
        """Enlarge the length, width and height boxes.

        Args:
            extra_width (float | torch.Tensor): Extra width to enlarge the box.

        Returns:
            :obj:`LiDARInstance3DBoxes`: Enlarged boxes.
        """
        enlarged_boxes = self.tensor.clone()
        enlarged_boxes[:, 3:6] += extra_width * 2
        # bottom center z minus extra_width
        enlarged_boxes[:, 2] -= extra_width
        return self.new_box(enlarged_boxes)

    def img_points(self, camera_type='pinhole', lidar2cam_rt=None, camera_intrinsic=None, camera_intrinsic_dist=None, img_w=None, img_h=None):
        corners = self.corners.clone()

        points = []
        for corner_points in corners: # (n,8,3)
            # from lidar coordinate to camera coordinate
            corner_points = corner_points.T[0:3, :]
            lidar_in_cam = lidar2cam_rt @ np.vstack((corner_points[0:3][:], np.ones_like(corner_points[1])))
            # only show point cloud in front of the camera
            # lidar_in_cam = lidar_in_cam[:, lidar_in_cam[2] > 0]
            pc_xyz_ = lidar_in_cam[:3, :]

            image_point = Camera3DPointsTransfer.transfer_camera3d_to_image(camera_3ds=torch.Tensor(pc_xyz_.transpose()),
                                                                            camera_intrinsic=torch.Tensor(camera_intrinsic),
                                                                            camera_dist=torch.Tensor(camera_intrinsic_dist))
            image_point = image_point.numpy().transpose()

            # if camera_type == 'fisheye':
            #     Camera3DPointsTransfer.transfer_camera3d_to_image()
            #     image_point = fisheye_camera_to_image(pc_xyz_.transpose(), camera_intrinsic, camera_intrinsic_dist)
            #     image_point = image_point.transpose()
            # else:
            #     pc_xyz = pc_xyz_ / pc_xyz_[2, :] 
            #     image_point = camera_intrinsic @ pc_xyz
            #     image_point = image_point[:2, :]

            # image_point (3,8)
            image_limit_w = np.logical_and(image_point[0, :] >= 0, image_point[0, :] <= img_w)
            image_limit_h = np.logical_and(image_point[1, :] >= 0, image_point[1, :] <= img_h)
            image_limit = np.logical_and(image_limit_h, image_limit_w)
            if image_point[:, image_limit].size == 0:
                continue
            points.append(image_point[np.newaxis, :])
            
            # if (image_point<0).sum():
            #     import pdb;pdb.set_trace()

        if len(points) > 0:
            
            return np.concatenate(points)
        return []
