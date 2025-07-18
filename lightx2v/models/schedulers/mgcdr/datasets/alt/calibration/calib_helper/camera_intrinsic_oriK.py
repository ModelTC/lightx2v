# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:09:45 2024

@author: zck
"""
import cv2
import numpy as np

def fisheye_camera_to_image(p3ds, camera_intrinsic, kb_param):
    r = np.linalg.norm(p3ds[:, :2], axis=1).reshape(-1, 1)
    theta = np.arctan2(r, p3ds[:, 2:3])
    d_theta = theta + np.power(theta, 3)*kb_param[0] \
        + np.power(theta, 5)*kb_param[1] \
        + np.power(theta, 7)*kb_param[2] \
        + np.power(theta, 9)*kb_param[3]
    p2ds = camera_intrinsic @ np.concatenate([d_theta*p3ds[:, 0:1]/r, d_theta*p3ds[:, 1:2]/r, np.ones_like(p3ds[:, 0:1])], 1).transpose()

    image_points = p2ds[:2]

    return image_points.T

class CameraIntrinsicOriK:
    """
    Pinhole model tagged by oriK.

    Details can be found:
        https://docs.opencv.org/3.4.9/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d

    """
    img_dist_type = "oriK"

    def __init__(self, cfg_js_dict: dict):
        assert len(cfg_js_dict) == 1
        cfg = list(cfg_js_dict.values())[0]
        assert cfg["param"]["img_dist_type"] == self.img_dist_type
        
        if "cam_K_new" not in cfg["param"]:
            self.cam_K = np.float64(cfg["param"]["cam_K"]["data"])
            self.cam_D = np.float64(cfg["param"]["cam_dist"]["data"])
        else:
            self.cam_K = np.float64(cfg["param"]["cam_K_new"]["data"])
            self.cam_D = np.zeros((4, 1))
        self.cfg = cfg

    def projectPoints(self, objectPoints: np.ndarray) -> np.ndarray:
        shape = list(objectPoints.shape)
        assert 3 == shape[-1]

        sensor_name = self.cfg["sensor_name"]
        if "195" in sensor_name:
            # self.cam_D = np.float64(self.cfg["param"]["cam_dist"]["data"])
            self.cam_D = np.float64(self.cfg["param"]["cam_dist"]["data"]).transpose()
            imagePoints = fisheye_camera_to_image(objectPoints, self.cam_K, self.cam_D)
        else:
            objectPoints = objectPoints.reshape((-1, 1, 3))

            # project
            imagePoints = cv2.projectPoints(
                objectPoints=objectPoints,
                rvec=np.zeros((3, 1)),
                tvec=np.zeros((3, 1)),
                cameraMatrix=self.cam_K,
                distCoeffs=self.cam_D,
            )[0]

        # reshape
        shape[-1] = 2
        imagePoints = imagePoints.reshape(shape)

        return imagePoints
