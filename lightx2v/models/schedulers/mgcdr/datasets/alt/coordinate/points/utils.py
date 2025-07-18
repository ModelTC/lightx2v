# Import from third library
import torch


class Camera3DPointsTransfer(object):
    @classmethod
    def transfer_camera3d_to_image(cls, camera_3ds, camera_intrinsic, camera_dist):
        if camera_dist.numel() == 0:  # 针孔相机
            point_2d_res = cls.transfer_to_pinhole(camera_3ds, camera_intrinsic, camera_dist)
        elif camera_dist.numel() == 4:
            point_2d_res = cls.transfer_to_fisheye(camera_3ds, camera_intrinsic, camera_dist, camera_type="KB")
        elif camera_dist.numel() in [11, 13]:
            point_2d_res = cls.transfer_to_fisheye(camera_3ds, camera_intrinsic, camera_dist, camera_type="OCAM")
        elif camera_dist.numel() in [22, 26]:
            point_2d_res = cls.transfer_to_fisheye(camera_3ds, camera_intrinsic, camera_dist[1], camera_type="OCAM")
        else:
            raise NotImplementedError("invalid camera_dist")

        return point_2d_res

    @classmethod
    def transfer_to_pinhole(cls, camera_3ds, camera_intrinsic, camera_dist=[]):
        point_2d = camera_3ds @ camera_intrinsic.T
        point_depth = torch.clamp(point_2d[..., 2:3], min=1e-5, max=1e5)
        points_clamp = torch.cat([point_2d[..., :2], point_depth], dim=1)

        point_2d_res = points_clamp[..., :2] / points_clamp[..., 2:3]
        # if (point_2d_res<0).sum():
        #     import pdb;pdb.set_trace()

        return point_2d_res

    @classmethod
    def transfer_to_fisheye(cls, camera_3ds, camera_intrinsic, camera_dist, camera_type="KB"):
        aff_ = torch.tensor(
            [camera_intrinsic[0][0], camera_intrinsic[0][1], camera_intrinsic[1][0], camera_intrinsic[1][1]]
        ).reshape(2, 2)
        xc_ = camera_intrinsic[0][2]
        yc_ = camera_intrinsic[1][2]

        inv_poly_param_ = camera_dist

        # 计算每个点的二维平面上的范数（欧几里得距离）
        norm = torch.norm(camera_3ds[:, :2], dim=1)
        # 计算范数的倒数，避免除以零的情况
        invNorm = torch.where(norm == 0, torch.tensor(0.0), 1.0 / norm)

        if camera_type == "KB":
            camera_dist = camera_dist.squeeze()
            theta = torch.atan2(norm, camera_3ds[:, 2])
            rho = (
                theta
                + torch.pow(theta, 3) * camera_dist[0]
                + torch.pow(theta, 5) * camera_dist[1]
                + torch.pow(theta, 7) * camera_dist[2]
                + torch.pow(theta, 9) * camera_dist[3]
            )
        else:
            # 计算每个点的 theta 值，使用 arctan2 来计算反正切
            theta = torch.atan2(-camera_3ds[:, 2], norm)
            # 使用反多项式参数计算 rho
            rho = cls.poly_val(inv_poly_param_, theta)

        # 计算归一化坐标
        xn = torch.ones((2, camera_3ds.shape[0]), dtype=torch.float32)
        xn[0] = camera_3ds[:, 0] * invNorm * rho
        xn[1] = camera_3ds[:, 1] * invNorm * rho

        # 应用仿射变换，并加上主点坐标，得到二维图像坐标
        p2ds = (torch.matmul(aff_.float(), xn) + torch.tensor([xc_, yc_]).reshape(2, 1)).T

        return p2ds

    @classmethod
    def poly_val(cls, param, x):
        n = len(param)

        res = torch.zeros_like(x)
        for i in range(n):
            res = res * x + param[n - i - 1]
        return res
