# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:09:45 2024

@author: zck
"""
import json
import numpy as np
from typing import Union
from .camera_intrinsic_scaramuzza import CameraIntrinsicScaramuzza
from .camera_intrinsic_oriK import CameraIntrinsicOriK
__register__ = {
    "scaramuzza": CameraIntrinsicScaramuzza,
    "oriK": CameraIntrinsicOriK,
}


class CameraIntrinsic:

    def __new__(self, config: Union[str, dict]):
        # if config is file, load as json
        if type(config) is str:
            cfg_js_dict = json.load(open(config, "r"))
        else:
            cfg_js_dict = config

        # schema check
        assert len(cfg_js_dict) == 1
        cfg = list(cfg_js_dict.values())[0]
        img_dist_type = cfg["param"]["img_dist_type"]
        assert img_dist_type in __register__, f"{img_dist_type} not supported"

        # return child class ref to __register__
        cls = __register__[img_dist_type]
        return cls(cfg_js_dict)

    def projectPoints(objectPoints: np.ndarray) -> np.ndarray:
        """
        Projects 3D points to an image plane.

        Similar to opencv api, but camera parameters are stored in object:
            https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c


        Parameters
        ----------
        objectPoints : np.ndarray, N1 x ... x 3, float64
            3D points in the camera coordinate.

        Returns
        -------
        imagePoints : np.ndarray, N1 x ... x 2, float64.
            2D points mapped to the image coordinate.

        """
        raise NotADirectoryError()
