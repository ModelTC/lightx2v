import os
from decimal import Decimal

import cv2 as cv
import numpy as np
import plotly
import plotly.graph_objects as go
from loguru import logger

from ...calibration.calibration import CalibrationAdapter
from alt.utils.petrel_helper import global_petrel_helper
from alt.utils.coordinate_helper import (
    draw_corners_3d,
    transform_box3d_to_image,
    transform_box3d_to_point_cloud,
    transform_corners_3d_to_birdseye,
    transform_point_cloud_to_birdseye,
    transform_point_cloud_to_image,
)


class Visualize(object):
    def __init__(self, meta_path, camera_name=None, conf_path='~/petreloss.conf'):
        super(Visualize, self).__init__()
        self.ceph_prefix = 's3://'
        self.meta_path = meta_path
        self.camera_name = camera_name
        self.case_path = self.meta_path.split(self.ceph_prefix)[-1][: -len('/meta.json')]
        self.petrel_tool = global_petrel_helper
        if self.ceph_prefix in self.meta_path:
            cluster_names = list(self.petrel_tool.conf.keys())
            cluster_names.remove('DEFAULT')
            cluster_names.remove('mc')
            cluster_names.remove('dfs')
            cluster_name = self.meta_path.split(':' + self.ceph_prefix)[0]
            if cluster_name not in cluster_names:
                raise ValueError(
                    'The path of meta.json is wrong. '
                    f'Please enter a path starting with the cluster name in {cluster_names}.'
                )
        self.meta = self.petrel_tool.map_bucket(self.petrel_tool.load_json(meta_path))
        self.data_annotation_dir = self.meta['data_annotation']
        self.camera_names = self._get_camera_names()
        if self.camera_name not in self.camera_names:
            raise ValueError(f'The camera name is wrong. Please enter a camera name from {self.camera_names}.')
        self.camera_lens = self._get_image_len()
        self.lidar_name = sorted(self.meta['data_structure']['lidar'].keys())[0]
        logger.info(f'Visualize {self.lidar_name} and {self.camera_name}')
        self.gt_frames = self._load_gt_frames(camera_name=self.camera_name)
        self.calibration = CalibrationAdapter(
            config_path=self.meta['data_structure']['config'], petrel_tool=self.petrel_tool
        )
        self.categories = []
        self.lidar_timestamp_map = self._get_lidar_map()
        self.lidar_timstamps = self._get_lidar_timestamps()
        self.box3d_color = (0, 0, 255)  # red
        self.box3d_no_match_color = (0, 0, 0)  # black
        self.box2d_color = (0, 255, 0)  # green
        self.box2d_no_match_color = (190, 190, 190)  # grey
        self.forward_color = (255, 255, 0)  # cyan

    def _get_camera_names(self):
        camera_names = []
        for k, v in self.meta['data_structure']['camera'].items():
            if v['video']:
                camera_names.append(k)
        return sorted(camera_names)

    def _get_image_len(self, camera_name=None):
        camera_name = self.camera_name if camera_name is None else camera_name
        camera_lens = None
        if 'lens' in self.meta['data_structure']['camera'][camera_name]:
            camera_lens = self.meta['data_structure']['camera'][camera_name]['lens']
        return camera_lens

    def _load_gt_frames(self, camera_name):
        fusion_path = os.path.join(self.data_annotation_dir, f'{self.lidar_name}-to-{camera_name}#object.pkl')
        fusion_info = self.petrel_tool.load_pickle(fusion_path)
        return fusion_info['frames']

    def _get_lidar_map(self):
        lidar_timestamp_map = {}
        sensor_dir = os.path.join(self.data_annotation_dir, 'cache', 'sensor')
        sensor_dirs, _ = self.petrel_tool.list_dir(sensor_dir)
        for dir in sensor_dirs:
            if self.lidar_name in dir:
                lidar_dir = os.path.join(sensor_dir, dir)
                _, lidar_files = self.petrel_tool.list_dir(lidar_dir)
                for file in sorted(lidar_files):
                    lidar_suffix = '.bin'
                    assert file.endswith(lidar_suffix)
                    lidar_timestamp_map[Decimal(file[: -len(lidar_suffix)])] = os.path.join(lidar_dir, file)
        return lidar_timestamp_map

    def _get_lidar_timestamps(self):
        timestamps = self.lidar_timestamp_map.keys()
        timestamps = np.array([Decimal(ts) for ts in timestamps])
        return timestamps

    def _find_latest_lidar_file(self, timestamp):
        idx = (np.abs(self.lidar_timstamps - timestamp)).argmin()
        lidar_timestamp = self.lidar_timstamps[idx]
        time_diff = abs(float(lidar_timestamp) - float(timestamp)) / 1e3
        if time_diff > 100:
            logger.warning(
                f'The time difference between point cloud ({lidar_timestamp}) and image ({timestamp}) '
                f'exceeds {time_diff} ms.'
            )
        return self.lidar_timestamp_map[lidar_timestamp], time_diff

    def vis_camera_and_BEV(
        self,
        show=True,
        vis_box2d=False,
        vis_box3d=False,
        vis_point_cloud=False,
        vis_point_cloud_3d=False,
        save_video=False,
        save_img=False,
        save_path=None,
        wait_time=30,
        fps=10,
        lidar_side_range=(-25.0, 25.0),
        lidar_fwd_range=(-50.0, 50.0),
        rotation_bev=0,
    ):
        image_path = self.gt_frames[0]['filename']
        image_path = (
            self.data_annotation_dir[: self.data_annotation_dir.index(self.ceph_prefix)]
            + image_path[image_path.index(self.ceph_prefix) :]
        )
        img = self.petrel_tool.load_img(image_path)  # BGR image format
        img_w, img_h = img.shape[1], img.shape[0]

        lidar_scale = (lidar_fwd_range[1] - lidar_fwd_range[0]) / (lidar_side_range[1] - lidar_side_range[0])
        font_scale = img_h * 0.0003
        thickness = int(img_h * 0.001 + 0.5)

        # create folder to save image if not existing
        if save_img or save_video or vis_point_cloud_3d:
            if save_path is None:
                save_path = os.path.join('./camera_BEV_vis', self.case_path, self.camera_name)
                if vis_box2d or vis_box3d:
                    save_path = save_path + '_box'
                if vis_point_cloud or vis_point_cloud_3d:
                    save_path = save_path + '_point_cloud'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        if save_img:
            save_img_path = os.path.join(save_path, 'img')
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
        if save_video:
            video_path = os.path.join(save_path, f'{self.camera_name}.mp4')
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(video_path, fourcc, fps, (int(img_w + img_h / lidar_scale), img_h))
        if vis_point_cloud_3d:
            save_html_path = os.path.join(save_path, 'html')
            if not os.path.exists(save_html_path):
                os.makedirs(save_html_path)

        img_bev_w = int(img_h / lidar_scale)
        img_bev_tmp = np.zeros((img_h, img_bev_w, 3), np.uint8)
        num_horizontal_bins = 15
        gap_lines = int(img_h / num_horizontal_bins)
        line_color = (32, 165, 218)
        for i in range(num_horizontal_bins + 1):
            if i % 5 == 0:
                line_thickness = thickness + 1
            else:
                line_thickness = thickness
            pt1 = (0, i * gap_lines)
            pt2 = (img_bev_w, i * gap_lines)
            cv.line(img_bev_tmp, pt1, pt2, color=line_color, thickness=line_thickness)
        num_vertical_bins = int(img_bev_w / gap_lines)
        for i in range(num_vertical_bins + 1):
            if i % 5 == 0:
                line_thickness = thickness + 1
            else:
                line_thickness = thickness
            pt1 = (i * gap_lines, 0)
            pt2 = (i * gap_lines, img_h)
            cv.line(img_bev_tmp, pt1, pt2, color=line_color, thickness=line_thickness)
        cv.circle(img_bev_tmp, (int(img_bev_w / 2), int(img_h / 2)), thickness * 6, line_color, thickness=-1)

        # visualization
        for gt_frame in self.gt_frames:
            img_bev = img_bev_tmp.copy()

            image_path = gt_frame['filename']
            image_name = image_path.split('/')[-1]
            image_path = (
                self.data_annotation_dir[: self.data_annotation_dir.index(self.ceph_prefix)]
                + image_path[image_path.index(self.ceph_prefix) :]
            )
            img = self.petrel_tool.load_img(image_path)  # BGR image format

            timestamp = gt_frame['timestamp']
            cv.putText(
                img,
                text=f'img: {image_name}',
                org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.0005)),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale * 2,
                color=(0, 255, 0),
                thickness=thickness,
            )

            cam2lidar = None
            if self.lidar_name == "car_center":
                cam2lidar = self.calibration.extrinsic[f'{self.camera_name}-to-car_center']
            else:
                cam2car = self.calibration.extrinsic[f'{self.camera_name}-to-car_center']
                lidar2car = self.calibration.extrinsic[f'{self.lidar_name}-to-car_center']
                if isinstance(cam2car, np.ndarray) and isinstance(lidar2car, np.ndarray):
                    cam2lidar = np.dot(np.linalg.inv(lidar2car), cam2car)
            if not isinstance(cam2lidar, np.ndarray):
                cam2lidar = self.calibration.extrinsic[f'{self.camera_name}-to-{self.lidar_name}']
            if not isinstance(cam2lidar, np.ndarray):
                cam2lidar = np.linalg.inv(self.calibration.extrinsic[f'{self.lidar_name}-to-{self.camera_name}'])
            camera_intrinsic = self.calibration.intrinsic_new[self.camera_name]
            if camera_intrinsic is None:
                camera_intrinsic = self.calibration.intrinsic[self.camera_name]
            camera_intrinsic_dist = self.calibration.intrinsic_dist[self.camera_name]

            lidar_file, time_diff = self._find_latest_lidar_file(timestamp)
            lidar_name = lidar_file.split('/')[-1]
            cv.putText(
                img,
                text=f'bin: {lidar_name}',
                org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.001)),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale * 2,
                color=(0, 255, 0),
                thickness=thickness,
            )
            cv.putText(
                img,
                text=f'time_diff: {time_diff:7.3f} ms',
                org=(int(5 * img_h * 0.0005), int(40 * img_h * 0.0015)),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale * 2,
                color=(0, 255, 0),
                thickness=thickness,
            )
            if vis_point_cloud or vis_point_cloud_3d:
                pcloud = self.petrel_tool.load_bin(lidar_file)
                pcloud_ = pcloud.copy()
                if rotation_bev:
                    rotation_bev_rad = np.deg2rad(rotation_bev)
                    R_matrix = np.array(
                        [
                            [np.cos(rotation_bev_rad), -np.sin(rotation_bev_rad), 0, 0],
                            [np.sin(rotation_bev_rad), np.cos(rotation_bev_rad), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                    pcloud = np.dot(R_matrix, pcloud.T).T
            if vis_point_cloud:
                res = (lidar_fwd_range[1] - lidar_fwd_range[0]) / img_h
                img_bev = transform_point_cloud_to_birdseye(pcloud, res, lidar_side_range, lidar_fwd_range)
                img_bev = cv.applyColorMap(img_bev, cv.COLORMAP_JET)
                img_bev = cv.resize(img_bev, (img_bev_w, img_h), interpolation=cv.INTER_CUBIC)

                image_point, cam_z = transform_point_cloud_to_image(
                    pcloud_, cam2lidar, camera_intrinsic, camera_intrinsic_dist, self.camera_lens, img_w, img_h
                )
                if image_point.shape[1] > 0:
                    im_color = cv.applyColorMap((cam_z / 200 * 255).astype(np.uint8), cv.COLORMAP_JET)
                    im_color = im_color.reshape((im_color.shape[1], im_color.shape[0], im_color.shape[2]))
                    for idx in range(image_point.shape[1]):
                        color = (int(im_color[idx][0][0]), int(im_color[idx][0][1]), int(im_color[idx][0][2]))
                        cv.circle(
                            img, (int(round(image_point[0][idx])), int(round(image_point[1][idx]))), 1, color, -1,
                        )

            if vis_point_cloud_3d:

                def trans_col(col):
                    if col == 'height':
                        return pcloud_[:, 2]
                    if col == 'distance':
                        return np.sqrt(pcloud_[:, 0] ** 2 + pcloud_[:, 1] ** 2)
                    if col == 'intensity':
                        assert pcloud_.shape[-1] == 4, "point cloud data here should be 4 dims with x,y,z and r"
                        return pcloud_[:, 3]

                pts = go.Scatter3d(
                    x=pcloud_[:, 0],
                    y=pcloud_[:, 1],
                    z=pcloud_[:, 2],
                    mode='markers',
                    name='point clouds',
                    marker=dict(
                        size=0.5,
                        color=trans_col('intensity'),  # set color to an array/list of desired values
                        colorscale='deep',  # choose a colorscale
                        opacity=1,
                    ),
                )
                data = [pts]

            # show point cloud
            res = (lidar_fwd_range[1] - lidar_fwd_range[0]) / img_h

            # load and show object bboxes
            for obj in gt_frame['objects']:
                bbox2d = obj['bbox2d']
                if vis_box2d and bbox2d is not None:
                    id = obj['id']
                    label = obj['label']
                    bbox = [int(p) for p in bbox2d]
                    if obj['conf']:
                        bbox_color = self.box2d_color
                    else:
                        bbox_color = self.box2d_no_match_color
                    cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=bbox_color, thickness=thickness)
                    if id is None:
                        txt = label
                    else:
                        txt = label + '-ID: ' + str(id)
                    cv.putText(
                        img,
                        text=txt,
                        org=(bbox[0], bbox[1] - thickness - 3),
                        fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=bbox_color,
                        thickness=thickness,
                    )

                bbox3d = obj['bbox3d']
                if vis_box3d and bbox3d is not None:
                    id = obj['id']
                    label = obj['label']
                    vel = obj['vel']
                    if obj['conf']:
                        bbox_color = self.box3d_color
                    else:
                        bbox_color = self.box3d_no_match_color

                    roll_cam, pitch_cam, yaw_cam = None, None, None
                    if len(bbox3d) == 7:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, yaw_cam] = bbox3d
                    else:
                        [cam_box_x, cam_box_y, cam_box_z, w, h, l, roll_cam, pitch_cam, yaw_cam] = bbox3d

                    dimension = [w, h, l]
                    location = [cam_box_x, cam_box_y, cam_box_z]

                    corners_3d_img = transform_box3d_to_image(
                        dimension,
                        location,
                        roll_cam,
                        pitch_cam,
                        yaw_cam,
                        camera_intrinsic,
                        camera_intrinsic_dist,
                        self.camera_lens,
                    )

                    # None means object is behind the camera, and ignore this object.
                    if corners_3d_img is not None:
                        corners_3d_img = corners_3d_img.astype(int)

                        # draw lines in the image
                        # p0-p1, p1-p2, p2-p3, p3-p0
                        cv.line(
                            img,
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )

                        # p4-p5, p5-p6, p6-p7, p7-p4
                        cv.line(
                            img,
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )

                        # p0-p4, p1-p5, p2-p6, p3-p7
                        cv.line(
                            img,
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[4, 0], corners_3d_img[4, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[5, 0], corners_3d_img[5, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            (corners_3d_img[6, 0], corners_3d_img[6, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            (corners_3d_img[7, 0], corners_3d_img[7, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )

                        # draw front lines
                        cv.line(
                            img,
                            (corners_3d_img[0, 0], corners_3d_img[0, 1]),
                            (corners_3d_img[2, 0], corners_3d_img[2, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img,
                            (corners_3d_img[1, 0], corners_3d_img[1, 1]),
                            (corners_3d_img[3, 0], corners_3d_img[3, 1]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        if id is None:
                            txt = label
                        else:
                            txt = label + '-ID: ' + str(id)
                        cv.putText(
                            img,
                            text=txt,
                            org=(corners_3d_img[4, 0], corners_3d_img[4, 1] - 2),
                            fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale,
                            color=bbox_color,
                            thickness=thickness,
                        )

                    corners_3d = transform_box3d_to_point_cloud(
                        dimension, location, roll_cam, pitch_cam, yaw_cam, cam2lidar
                    )
                    corners_3d_ = corners_3d.copy()
                    # BEV
                    if obj['conf']:
                        if rotation_bev:
                            rotation_bev_rad = np.deg2rad(rotation_bev)
                            R_matrix = np.array(
                                [
                                    [np.cos(rotation_bev_rad), -np.sin(rotation_bev_rad), 0],
                                    [np.sin(rotation_bev_rad), np.cos(rotation_bev_rad), 0],
                                    [0, 0, 1],
                                ]
                            )
                            corners_3d = np.dot(R_matrix, corners_3d.T).T

                        # draw lines
                        x_bbox2d, y_bbox2d = transform_corners_3d_to_birdseye(
                            corners_3d, res, lidar_side_range, lidar_fwd_range
                        )
                        cv.line(
                            img_bev,
                            (x_bbox2d[2], y_bbox2d[2]),
                            (x_bbox2d[3], y_bbox2d[3]),
                            color=self.forward_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img_bev,
                            (x_bbox2d[3], y_bbox2d[3]),
                            (x_bbox2d[7], y_bbox2d[7]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img_bev,
                            (x_bbox2d[7], y_bbox2d[7]),
                            (x_bbox2d[6], y_bbox2d[6]),
                            color=bbox_color,
                            thickness=thickness,
                        )
                        cv.line(
                            img_bev,
                            (x_bbox2d[6], y_bbox2d[6]),
                            (x_bbox2d[2], y_bbox2d[2]),
                            color=bbox_color,
                            thickness=thickness,
                        )

                        center_x = int((int(x_bbox2d[2]) + int(x_bbox2d[3])) / 2)
                        center_y = int((int(y_bbox2d[2]) + int(y_bbox2d[3])) / 2)
                        if vel is None:
                            tgt_point = (int(center_x), int(center_y))
                        else:
                            vel_lidar = cam2lidar[:-1, :-1] @ vel
                            if rotation_bev:
                                vel_lidar = R_matrix @ vel_lidar
                            tgt_point = (int(center_x - vel_lidar[1] / res), int(center_y - vel_lidar[0] / res))
                        cv.arrowedLine(
                            img_bev,
                            pt1=(center_x, center_y),
                            pt2=tgt_point,
                            color=bbox_color,
                            thickness=thickness,
                            line_type=cv.LINE_8,
                            shift=0,
                            tipLength=0.1,
                        )
                        # cv.putText(
                        #     img_bev,
                        #     text=tmp.txt,
                        #     org=(x_bbox2d[6], y_bbox2d[6] + thickness + 8),
                        #     fontFace=cv.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=font_scale,
                        #     color=bbox_color,
                        #     thickness=thickness,
                        # )

                    # point cloud 3D
                    if vis_point_cloud_3d:
                        if id is None:
                            txt = label
                        else:
                            txt = label + '-ID: ' + str(id)
                        corners_3d_line = draw_corners_3d(corners_3d_, txt)
                        data.append(corners_3d_line)

            img_concat = cv.hconcat([img, img_bev])

            # save visualization image if you want
            if save_img:
                cv.imwrite(os.path.join(save_img_path, '{0:f}'.format(timestamp) + '.png'), img_concat)
            if save_video:
                out.write(img_concat)

            if vis_point_cloud_3d:
                layout = dict(scene=dict(xaxis=dict(visible=True), yaxis=dict(visible=True), zaxis=dict(visible=True)))
                fig = go.Figure(data=data, layout=layout)
                axis_range = 100
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(tickmode='auto', nticks=10, range=[-axis_range, axis_range], autorange=False),
                        yaxis=dict(tickmode='auto', nticks=10, range=[-axis_range, axis_range], autorange=False),
                        zaxis=dict(tickmode='auto', nticks=10, range=[-axis_range, axis_range], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                    ),
                    margin=dict(r=0, l=0, b=0, t=0),
                )
                html_path = os.path.join(save_html_path, image_name.rsplit('.', 1)[0] + '.html')
                plotly.offline.plot(fig, filename=html_path, auto_open=False)

            if show:
                cv.imshow('Play {}'.format(self.camera_name), img_concat)
                cv.waitKey(wait_time)

        if show:
            cv.destroyAllWindows()
        if save_video:
            out.release()


if __name__ == '__main__':
    rotation_bev = 0

    # 3m1_aeb
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_test_230525/3m1_aeb/meta.json'
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_e2e_test_230531/3m1_aeb/meta.json'
    # camera_names = ['center_camera_fov30', 'center_camera_fov120']

    # gac_l4
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_test_230525/gac_l4/meta.json'
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_e2e_test_230531/gac_l4/meta.json'
    # camera_names = [
    #     'center_camera_fov30',
    #     'center_camera_fov120',
    #     'rear_camera',
    #     'left_front_camera',
    #     'left_rear_camera',
    #     'right_front_camera',
    #     'right_rear_camera',
    # ]

    # hozon
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_test_230525/hozon/meta.json'
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_e2e_test_230531/hozon/meta.json'
    # camera_names = ['front_narrow', 'front_wide', 'back_left', 'front_left', 'back_right', 'front_right', 'back']

    # pilot
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_test_230525/pilot/meta.json'
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_e2e_test_230531/pilot/meta.json'
    # camera_names = [
    #     'center_camera_fov30',
    #     'center_camera_fov120',
    #     'rear_camera',
    #     'left_front_camera',
    #     'left_rear_camera',
    #     'right_front_camera',
    #     'right_rear_camera',
    # ]

    # st_baidu_gac10
    meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_test_230525/st_baidu_gac10/meta.json'
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_test/autolabel_e2e_test_230531/st_baidu_gac10/meta.json'
    rotation_bev = 90
    camera_names = [
        'center_camera_fov30',
        'center_camera_fov120',
        'rear_camera',
        'left_front_camera',
        'left_rear_camera',
        'right_front_camera',
        'right_rear_camera',
    ]

    # parking_data
    # meta_path = 'ad_system_common_sh41:s3://sh41hdd_autolabel/GT_DATA/parking_data/autolabel_230518_036/' \
    #     '2023_0313_baidu/3/14.22.06/meta.json'
    # camera_names = [
    #     'front_camera_fov195',
    #     'center_camera_fov120',
    #     'rear_camera_fov195',
    #     'left_camera_fov195',
    #     'right_camera_fov195',
    # ]

    logger.info('Start')
    # conf_path = 'cfg/petreloss.conf'
    conf_path = 'cfg/petreloss_local.conf'
    camera_name = camera_names[1]
    visualize = Visualize(meta_path=meta_path, camera_name=camera_name, conf_path=conf_path)
    lidar_side = 80
    lidar_fwd = 120
    visualize.vis_camera_and_BEV(
        show=True,
        vis_box2d=True,
        vis_box3d=True,
        vis_point_cloud=False,
        vis_point_cloud_3d=False,
        save_video=False,
        save_img=False,
        save_path=None,
        wait_time=1,
        fps=10,
        lidar_side_range=(-lidar_side, lidar_side),
        lidar_fwd_range=(-lidar_fwd, lidar_fwd),
        rotation_bev=rotation_bev,
    )
    logger.info('End!')
