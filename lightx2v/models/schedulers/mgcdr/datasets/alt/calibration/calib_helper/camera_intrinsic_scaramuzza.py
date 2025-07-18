# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:09:45 2024

@author: zck
"""
import numpy as np


class CameraIntrinsicScaramuzza:
    """
    Fisheye model tagged by scaramuzza.

    Details can be found:
        https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab

    """
    img_dist_type = "scaramuzza"

    def __init__(self, cfg_js_dict: dict):
        assert len(cfg_js_dict) == 1
        cfg = list(cfg_js_dict.values())[0]
        assert cfg["param"]["img_dist_type"] == self.img_dist_type
        self.cam_K = cfg["param"]["cam_K"]["data"]
        self.cam_D = cfg["param"]["cam_dist"]["data"]
        self.cfg = cfg

    @classmethod
    def poly_val(cls, param, x):
        """
        y = sum_{k=0}^{n-1}(param[k] * x**k)
        """
        n = len(param)
        res = [0.0 for x in range(x.shape[0])]
        for i in range(x.shape[0]):
            for itr in range(len(param)):
                res[i] = param[n - itr - 1] + res[i] * x[i]
        return res

    @classmethod
    def _poly_val(cls, param, x):
        p = np.poly1d(param[::-1])
        res = p(x)
        return res

    @classmethod
    def fisheye_camera_to_image(cls, p3ds, camera_intrinsic, camera_dist):
        aff_ = np.array([
            camera_intrinsic[0][0], camera_intrinsic[0][1],
            camera_intrinsic[1][0], camera_intrinsic[1][1]
        ]).reshape(2, 2)
        xc_ = camera_intrinsic[0][2]
        yc_ = camera_intrinsic[1][2]
        inv_poly_param_ = camera_dist[1]

        norm = np.linalg.norm(p3ds[:, :2], axis=1)

        invNorm = 1 / norm

        theta = np.arctan2(-p3ds[:, 2], norm)

        rho = cls._poly_val(inv_poly_param_, theta)

        xn = np.ones((2, p3ds.shape[0]))
        xn[0] = p3ds[:, 0] * invNorm * rho
        xn[1] = p3ds[:, 1] * invNorm * rho

        p2ds = (np.dot(aff_, xn) +
                np.tile(np.array([xc_, yc_]).reshape(2, -1), (xn.shape[1]))).T

        return p2ds

    def projectPoints(self, objectPoints: np.ndarray) -> np.ndarray:
        # assert and reshape
        shape = list(objectPoints.shape)
        assert 3 == shape[-1]
        objectPoints = objectPoints.reshape((-1, 3))

        # project
        imagePoints = self.fisheye_camera_to_image(
            p3ds=objectPoints,
            camera_intrinsic=self.cam_K,
            camera_dist=self.cam_D,
        )

        # reshape
        shape[-1] = 2
        imagePoints = imagePoints.reshape(shape)

        return imagePoints
