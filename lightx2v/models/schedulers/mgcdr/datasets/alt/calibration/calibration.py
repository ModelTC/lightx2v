import json
import os

import numpy as np
from addict import Dict

from alt.utils.petrel_helper import global_petrel_helper

class CalibrationAdapter(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.sensor_list = []
        self.intrinsic = Dict()
        self.intrinsic_new = Dict()
        self.intrinsic_w = Dict()
        self.intrinsic_h = Dict()
        self.intrinsic_dist = Dict()
        self.extrinsic = Dict()
        self.init_sensor()

    def init_sensor(self):
        if 's3://' in self.config_path:
            contents = global_petrel_helper.listdir(self.config_path)
            fold_list = [content.rstrip('/') for content in contents if content.endswith('/')]
        else:
            fold_list = []
            for fold in os.listdir(self.config_path):
                if os.path.isdir(os.path.join(self.config_path, fold)):
                    fold_list.append(fold)
        for fold in fold_list:
            if (
                'cam' in fold
                or 'car_center' in fold
                or 'gnss' in fold
                or 'lidar' in fold
                or 'back' in fold
                or 'front' in fold
            ):
                self.sensor_list.append(fold)
        # init camera intrinsic
        for sensor in self.sensor_list:
            if (
                ('cam' not in sensor and 'back' not in sensor and 'front' not in sensor)
                or 'radar' in sensor
                or 'lidar' in sensor
            ):
                continue
            intrinsic, intrinsic_new, img_dist_w, img_dist_h, dist = self.read_cam_intrinsic(self.config_path, sensor)
            self.intrinsic[sensor] = intrinsic
            self.intrinsic_new[sensor] = intrinsic_new
            self.intrinsic_w[sensor] = img_dist_w
            self.intrinsic_h[sensor] = img_dist_h
            self.intrinsic_dist[sensor] = dist

        for src in self.sensor_list:
            for tgt in self.sensor_list:
                if src == tgt:
                    continue
                self.extrinsic['{}-to-{}'.format(src, tgt)] = self.read_sensor_extrinsic(self.config_path, src, tgt)

    def read_cam_intrinsic(self, config_path, sensor):
        camera_intrinsic_path = os.path.join(config_path, sensor, '{}-intrinsic.json'.format(sensor))
        infos = global_petrel_helper.load_json(camera_intrinsic_path)

        config_name = ''
        for key in infos:
            config_name = key
        img_dist_w = infos[config_name]['param']['img_dist_w']
        img_dist_h = infos[config_name]['param']['img_dist_h']
        intrinsic = infos[config_name]['param']['cam_K']['data']
        dist = infos[config_name]['param']['cam_dist']['data']
        if 'cam_K_new' in infos[config_name]['param']:
            intrinsic_new = infos[config_name]['param']['cam_K_new']['data']
            return (
                np.array(intrinsic).reshape(3, 3),
                np.array(intrinsic_new).reshape(3, 3),
                img_dist_w,
                img_dist_h,
                np.array(dist),
            )
        return np.array(intrinsic).reshape(3, 3), None, img_dist_w, img_dist_h, np.array(dist)

    def read_sensor_extrinsic(self, config_path, src, tgt):
        extrinsic_path = os.path.join(config_path, src, '{}-to-{}-extrinsic.json'.format(src, tgt))
        if 's3://' in extrinsic_path:
            if not global_petrel_helper.check_exists(extrinsic_path):
                return None
            infos = global_petrel_helper.load_json(extrinsic_path)
        else:
            if not os.path.exists(extrinsic_path):
                return None
            infos = json.load(open(extrinsic_path))

        key_name = ''
        for key in infos:
            key_name = key
        extrinsic = infos[key_name]['param']['sensor_calib']['data']
        return np.array(extrinsic).reshape(4, 4)


if __name__ == '__main__':
    config_path = 'config/vehicle/CN-006'
    calib = CalibrationAdapter(config_path)
    print(calib.__dict__)
