# Standard Library
import os

# Import from third library
import cv2
import re
import json
import time
import matplotlib.cm as cm
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

# Import from alt
# import open3d
from alt.coordinate.bbox import LiDARInstance3DBoxes
from alt.utils.petrel_helper import global_petrel_helper
from alt.visualize.object.vis_camera_3d import expand_plot_rect3d_on_img
from alt.visualize.utils.vis_camera_2d import plot_one_box
from easydict import EasyDict

from alt.utils.env_helper import env

os.environ["DISPLAY"] = ":99"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np


def arrange_images(image_dict, rows, cols):
    # 指定的排列顺序
    order = [
        'left_camera_fov195', 'front_camera_fov195', 'right_camera_fov195', 'rear_camera_fov195',
        'left_front_camera', 'center_camera_fov120', 'right_front_camera', 'right_rear_camera',
        'rear_camera', 'left_rear_camera', 'center_camera_fov30'
    ]

    # 获取图像的高和宽
    sample_img = next(iter(image_dict.values()))
    img_height, img_width = sample_img.shape[:2]

    # 创建一个全为0的数组来存放最终的图像
    arranged_image = np.zeros((rows * img_height, cols * img_width, 3), dtype=sample_img.dtype)

    idx = 0
    for camera_name in order:
        if camera_name in image_dict:
            if idx >= rows * cols:
                break
            img = image_dict[camera_name]
            row = idx // cols
            col = idx % cols
            arranged_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img
            # 在图像上打印相机名
            cv2.putText(arranged_image, camera_name,
                        (col * img_width + 10, row * img_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            idx += 1

    return arranged_image

class SenseBeeBEV(object):
    def __init__(self, root, anno, save_video=None) -> None:
        self.root = root
        self.anno = anno

        self.metas = self.build()
        self.cap = None
        self.save_video = save_video

        if env.distributed:
            torch.cuda.set_device(int(env.rank % torch.cuda.device_count()))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(env.rank % torch.cuda.device_count())

    def init_video(self, image_width, image_height):
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # 视频编解码器
        self.cap = cv2.VideoWriter(self.save_video, fourcc, 10, (image_width, image_height))  # 写入视频

    def build(self):
        metas = []
        for item in global_petrel_helper.readlines(self.anno):
            data = json.loads(item)
            if data.get('valid', True):  # 特定需求下保存无效帧，不可视化
                metas.append(EasyDict(data))
        return metas

    def draw_cameras(self, meta):
        targets = meta["Objects"]

        imgs = {}
        for camera_name, camera_meta in meta["sensors"]["cameras"].items():
            lidar2camera_rt = np.array(camera_meta["extrinsic"])

            camera_intrinsic = np.array(camera_meta["camera_intrinsic"])
            camera_intrinsic_dist = np.array(camera_meta["camera_dist"])
            img_path = os.path.join(self.root, camera_meta["data_path"])

            cur_img = global_petrel_helper.imread(img_path)
            img_h, img_w = cur_img.shape[:2]

            camera_type = "pinhole" if "195" not in camera_name else "fisheye"

            boxes, boxes_2d = [], []
            for target in targets:
                if "BARRICADE" in target["label"]:  # 路栏画起来过于奇怪，暂时先不要
                    continue

                if target.info2d is None:  # 针对每个相机只画可见的
                    continue

                if camera_name not in target.info2d:
                    continue

                boxes_2d.append(target.info2d[camera_name]["bbox2d"])
                boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])

            if len(boxes) > 0:
                lidar_boxes = LiDARInstance3DBoxes(torch.Tensor(boxes), origin=(0.5, 0.5, 0.5))
                img_points = lidar_boxes.img_points(
                    camera_type=camera_type,
                    lidar2cam_rt=lidar2camera_rt,
                    camera_intrinsic=camera_intrinsic,
                    camera_intrinsic_dist=camera_intrinsic_dist,
                    img_w=img_w,
                    img_h=img_h,
                )

                if len(img_points) > 0:
                    draw_img = expand_plot_rect3d_on_img(
                        cur_img, img_points.shape[0], img_points.transpose(0, 2, 1), thickness=4, color=(66,58, 249)
                    )
                else:
                    draw_img = cur_img
            else:
                draw_img = cur_img

            # 需要画2D信息
            if False and camera_name in meta["Pure2DObjects"] :
                Pure2DObjects = meta["Pure2DObjects"][camera_name]
                for single_box, single_label in zip(Pure2DObjects["bbox2d"], Pure2DObjects["label2d"]):
                    # The road barriers are drawn strangely and have received relatively little attention, not include them in the visualization for now.
                    if "BARRICADE" not in single_label:
                        boxes_2d.append(single_box)

            # for item2d in boxes_2d:
            #     draw_img = plot_one_box(draw_img, item2d, color=(255, 0, 0))

            imgs[camera_name] = cv2.resize(draw_img, (960, 540))

        return imgs

    def draw_lidars(self, meta, img_shape=(720, 2160)):
        relative_path = meta["sensors"]["lidar"]["car_center"]["data_path"]
        relative_path = re.sub(r'lidar_reconstruction_.+?/', 'origin_lidar/', relative_path)

        lidar_path = os.path.join(self.root, relative_path)
        valid, pcd_array = global_petrel_helper.load_pcd(lidar_path)
        assert valid, lidar_path

        pcd_array = pcd_array[
            (pcd_array[:, 0] <= 200) & (pcd_array[:, 0] >= -100) & (pcd_array[:, 1] >= -50) & (pcd_array[:, 1] <= 50)
        ]
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_array[:, :3])
        intensities = pcd_array[:, 3]

        # 将反射强度归一化到0-1范围
        if intensities.max() != intensities.min():
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        # 将归一化的反射强度映射到颜色，使用Matplotlib的colormap
        colormap = cm.get_cmap("viridis")  # 使用viridis颜色映射，可以根据需要选择其他映射
        # 将反射强度转换为颜色
        colors = colormap(intensities)[:, :3]  # 获取RGB值
        pcd.colors = open3d.utility.Vector3dVector(colors)

        rotation_matrix = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))  # 绕z轴旋转
        pcd.rotate(rotation_matrix, center=(0, 0, 0))

        # 创建OffscreenRenderer对象
        width, height = img_shape
        renderer = open3d.visualization.rendering.OffscreenRenderer(width, height)

        # 设置渲染的背景色为黑色
        renderer.scene.set_background([0, 0, 0, 1])

        # 添加点云到场景中
        material = open3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 0.1
        renderer.scene.add_geometry("point_cloud", pcd, material)

        # 设置摄像机参数
        # center = pcd.get_center()
        center = [0, 0, 0]
        # print(center)
        eye = center + np.array([0, 0, 80])
        renderer.scene.camera.look_at(center, eye, [0, 1, 0])

        image = renderer.render_to_image()
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return cv2_img

    def draw_bev(self, meta):
        lidar_boxes, ids, vels = [], [], []
        for target in meta["Objects"]:
            if target.info2d:
                lidar_boxes.append(target["bbox3d"][:6] + [target["bbox3d"][8]])
                ids.append(target["id"])
                vels.append(target['velocity'])

        lidar_boxes = LiDARInstance3DBoxes(torch.Tensor(lidar_boxes), origin=(0.5, 0.5, 0.5))
        img = fast_draw_bev_bbox_on_mesh(
            bboxes3d=lidar_boxes, ids=ids, speeds=vels, figsize=(10, 30), x_locator=20, y_locator=20, xlim=[-50, 50], ylim=[-100, 200]
        )

        return img[350:-300, 70:-70]

    def process(self):
        os.makedirs("./result", exist_ok=True)
        for frame_index, meta in tqdm(enumerate(self.metas), desc="vis ...", total=len(self.metas)):
            timestamp = meta["timestamp"]

            camera_imgs = self.draw_cameras(meta)
            merged_img = arrange_images(camera_imgs, 3, 4)
            cv2.imwrite(f"./result/{frame_index:03d}.jpg", merged_img)
            # [cv2.imwrite(f"./{k}.jpg", v) for (k, v) in camera_imgs.items()]
            # lidar_img = self.draw_lidars(meta)
            # bev_img = self.draw_bev(meta)

            # merge_img = np.zeros((2160, 3840, 3), np.uint8)
            #
            # merge_img[0:540, 0:960, :] = camera_imgs["center_camera_fov120"]
            # merge_img[0:540, 960:1920, :] = camera_imgs["center_camera_fov30"]
            # merge_img[540:1080, 0:960, :] = camera_imgs["left_front_camera"]
            # merge_img[540:1080, 960:1920, :] = camera_imgs["right_front_camera"]
            # merge_img[1080:1620, 0:960, :] = camera_imgs["left_rear_camera"]
            # merge_img[1080:1620, 960:1920, :] = camera_imgs["right_rear_camera"]
            # merge_img[1620:2160, 960:1920, :] = camera_imgs["rear_camera"]
            #
            # merge_img[0:540, -960:, :] = camera_imgs["front_camera_fov195"]
            # merge_img[540:1080, -960:, :] = camera_imgs["left_camera_fov195"]
            # merge_img[1080:1620, -960:, :] = camera_imgs["right_camera_fov195"]
            # merge_img[1620:2160, -960:, :] = camera_imgs["rear_camera_fov195"]
            #
            # merge_img[:, 1980 : 1980 + 720, :] = cv2.resize(bev_img, (720, 2160))
            #
            # img = merge_img[:, 1980 : 1980 + 720, :]

            # cv2.putText(
            #     img,
            #     text=str(timestamp),
            #     org=(10, 50),
            #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #     fontScale=2,
            #     color=(255, 255, 255),
            #     thickness=2,
            # )

            # merge_img = cv2.hconcat([merge_img, lidar_img])
            # if self.cap is None:
            #     self.init_video(image_width=merge_img.shape[1], image_height=merge_img.shape[0])
            # self.cap.write(merge_img)

        self.cap.release()


if __name__ == "__main__":
    root = "iaginfra:s3://uniad-infra/pap_pvb/ql_test/sensebee/pvb/pvb_all_attr/0522/18_meta_2024_03_21_06_42_00_gacGtParser"
    anno = "iaginfra:s3://uniad-infra/pap_pvb/ql_test/sensebee/pvb/pvb_all_attr/0522/18_meta_2024_03_21_06_42_00_gacGtParser/gt.jsonl"
    os.makedirs("result", exist_ok=True)
    SenseBeeBEV(root, anno, "result/10V/source_bev_vis/xx.mp4").process()

