# Standard Library
import os

# Import from third library
import numpy as np
from tqdm import tqdm

# Import from alt
import plotly
import plotly.graph_objects as go
from alt import load
from alt.utils.coordinate_helper import draw_corners_3d
from alt.utils.petrel_helper import global_petrel_helper


class LidarVisualizer(object):
    def __init__(self, xaxis=[-200, 200], yaxis=[-100, 100], zaxis=[-50, 50]):
        self.xazis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis

    def trans_col(self, pcd_arrays, col):
        if col == "height":
            return pcd_arrays[:, 2]
        if col == "distance":
            return np.sqrt(pcd_arrays[:, 0] ** 2 + pcd_arrays[:, 1] ** 2)
        if col == "intensity":
            assert pcd_arrays.shape[-1] == 4, "point cloud data here should be 4 dims with x,y,z and r"
            return pcd_arrays[:, 3]

    def rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def boxdim2corners(self, box):
        x, y, z, w, l, h, yaw = box[:7]
        z += h / 2

        # 3d bounding box corners
        Box = np.array(
            [
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [0, 0, 0, 0, h, h, h, h],
            ]
        )
        Box = np.array(
            [
                [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
            ]
        )
        R = self.rotz(yaw)
        corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)

        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z

        return np.transpose(corners_3d)

    # @classmethod
    def process(self, pcd_arrays, boxes_3d, html_path):
        pts = go.Scatter3d(
            x=pcd_arrays[:, 0],
            y=pcd_arrays[:, 1],
            z=pcd_arrays[:, 2],
            mode="markers",
            name="point clouds",
            marker=dict(
                size=0.5,
                color=self.trans_col(pcd_arrays, col="intensity"),  # set color to an array/list of desired values
                colorscale="deep",  # choose a colorscale
                opacity=1,
            ),
        )
        data = [pts]

        for box3d in boxes_3d:
            corners_3d = self.boxdim2corners(box3d)
            txt = "{} - {}".format(box3d[7], box3d[8])
            corners_3d_line = draw_corners_3d(corners_3d, txt)
            data.append(corners_3d_line)

        layout = dict(scene=dict(xaxis=dict(visible=True), yaxis=dict(visible=True), zaxis=dict(visible=True)))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(tickmode="auto", nticks=10, range=self.xazis, autorange=False),
                yaxis=dict(tickmode="auto", nticks=10, range=self.yaxis, autorange=False),
                zaxis=dict(tickmode="auto", nticks=10, range=self.zaxis, autorange=False),
                aspectratio=dict(x=4, y=2, z=1),
            ),
            margin=dict(r=0, l=0, b=0, t=0),
        )
        plotly.offline.plot(fig, filename=html_path, auto_open=False)
        
    
    def vis_autolabeler_inference(self, path, save_path, path_mapping={'/root/output': "ad_system_common:s3://sdc_gt_label",
                                                                       "/root/auto-label-run/output": "ad_system_common:s3://sdc_gt_label"}):
        for meta in tqdm(load(path)):
            filename = meta["img_meta"]["img_info"]["filename"]

            for src_path, dst_path in path_mapping.items():
                if src_path in filename:
                    filename = filename.replace(src_path, dst_path)

            pcd_array = global_petrel_helper.load_bin(filename)
            assert pcd_array is not None,  meta["img_meta"]["img_info"]["filename"]
            boxes_3d = []
            for box, score, label in zip(meta["img_bbox"]["boxes_3d"], meta["img_bbox"]["scores_3d"], meta["img_bbox"]["labels_3d"]):
                _box_ = box.tolist() + [score.item(), label.item()]
                boxes_3d.append(_box_)

            os.makedirs(save_path, exist_ok=True)
            html_save_path = os.path.join(save_path, filename.split("/")[-1] + ".html")
            self.process(pcd_array, boxes_3d, html_save_path)


if __name__ == "__main__":
    # input_path = "/workspace/qiaolei/auto-labeling-tools/result/lod_result/inference_results_0.2.pkl"
    input_path = "ad_system_common:s3://sdc_gt_label/SENSETIME/autolabel_20240423_833/data_collection/gt_data/pilotGtParser/drive_gt_collection/A19-PVL182/2024_04/2024_04_18/2024_04_18_15_35_15_pilotGtParser/gt_labels/cache/car_center/pillar.lod/inference_results.pkl"

    metas = load(input_path)
    viser = LidarVisualizer()
    for meta in tqdm(metas):
        filename = meta["img_meta"]["img_info"]["filename"]
        filename = filename.replace("/root/output/", "ad_system_common:s3://sdc_gt_label/")
        pcd_array = global_petrel_helper.load_bin(filename)
        boxes_3d = []
        for box, score, label in zip(meta["img_bbox"]["boxes_3d"], meta["img_bbox"]["scores_3d"], meta["img_bbox"]["labels_3d"]):
            _box_ = box.tolist() + [score.item(), label.item()]
            boxes_3d.append(_box_)

        os.makedirs("result/lidar_vis_debug", exist_ok=True)
        save_path = "result/lidar_vis_debug/" + filename.split("/")[-1] + ".html"
        viser.process(pcd_array, boxes_3d, save_path)
