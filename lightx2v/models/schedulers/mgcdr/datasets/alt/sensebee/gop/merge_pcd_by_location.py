import open3d
import os
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from alt.decoder.simple_location_decoder import NearestLocation
from alt.inference.lidar import LidarInferencer
from alt.utils.petrel_helper import global_petrel_helper
from alt.sensebee.gop.occ3d_utils_github import points_in_rbbox
from alt.files.smart import smart_listdir


def load_pcd_as_np(path, dim=3):
    # pc = np.asarray(open3d.io.read_point_cloud(path).points)  # 本地pcd
    # pc = global_petrel_helper.load_bin(path)[:,:3]  # s3的bin
    flag, pc = global_petrel_helper.load_pcd(path)  # s3的pcd
    assert flag
    if dim == 4:  # 默认读的就是4维带强度的
        return pc
    elif dim == 3:
        return pc[:, :3]

# 计算enu原点到当前自车enu的齐次变换矩阵
def get_RT_from_enu_yaw(enu, yaw, degrees=False):
    R_wo = Rotation.from_euler('zyx', [yaw, 0, 0], degrees=degrees).as_matrix()
    T_wo = np.array(enu)[:, np.newaxis]
    RT_wo = np.concatenate((R_wo, T_wo), axis=1)
    RT_wo = np.concatenate((RT_wo, np.array([[0, 0, 0, 1]])))
    return RT_wo

# 通过location.txt中信息将前后帧点云合并到当前帧点云，rm_dynamic决定是否调用模型推理去除动态目标
class MergePcdByLocation:
    def __init__(self, location_path, pcd_root_path, rm_dynamic=False, merged_num=5, merged_step=1):
        self.merged_points = {}
        ts = [int(i.replace('.pcd', '')) for i in smart_listdir(pcd_root_path)]
        # ts = ts[:3]

        pcd_path = [os.path.join(pcd_root_path, f'{t}.pcd') for t in ts]
        points_merge = [load_pcd_as_np(path, dim=4) for path in pcd_path]

        if rm_dynamic:
            lidar_detector = LidarInferencer()  # 雷达检测器初始化
            # 曾经lod接口只支持bin格式点云，所以写了一个临时转存bin的操作，现在支持直接pcd了，不需要了
            # bin_paths = []
            # for i, t in enumerate(ts):
            #     bin_path = os.path.join('/'.join(pcd_root_path.split('/')[:2]) + '/', 'sdc_gt_label/tmp_bin_for_lod_infer', '/'.join(pcd_root_path.split('/')[2:]), f'{t}.bin')
            #     bin_path = bin_path.replace('ad_system_common:s3://', 'ad_system_common_sdc:s3://')
            #     bin_paths.append(bin_path)
            #     global_petrel_helper.save_bin(bin_path, points_merge[i])  # 存的是带反射强度的bin点云，给api推理
            # lidar_res = lidar_detector.process(bin_paths)  # 传入bin路径，返回推理结果

            lidar_res = lidar_detector.process(pcd_path)  # 直接传入pcd路径，返回推理结果

        if rm_dynamic:
            # 准备去除动态目标
            points_filtered = []
            for points, det in zip(points_merge, lidar_res):
                dt = np.array(det['bbox'])
                dt[:,[3, 4]] = dt[:,[4, 3]]  # 修正长宽顺序
                dt[:, 3:6] = dt[:, 3:6] * 1.1  # 对检测框进行1.1倍放大
                dt[:, 6] = np.pi - dt[:, 6]  # 修正yaw
                point_indices = points_in_rbbox(points, dt)  # 看哪些点落在框中，王哲版本相关库太老，换github上新的实现方式，numba可以正常加速
                ind = np.array([], dtype=np.int64)
                for j in range(dt.shape[0]):
                    ind = np.concatenate((ind, np.where(point_indices[:, j] == 1)[0]))
                ind = list(ind)
                points = np.delete(points, ind, 0)
                points_filtered.append(points)
        else:
            points_filtered = points_merge

        nearest_location_decoder = NearestLocation(location_path)  # 用于查询某个时间戳最近的自车enu信息
        locations = []
        for t in ts:
            location, offset, _ = nearest_location_decoder(t)
            if offset is not None:
                if offset > 100 * 1000 * 1000:  # 若当前点云时间戳在Location.txt中找到最近的大于100ms，则认为没有该位置信息
                    location = None
            locations.append(location)

        RT_o_merge = []  # 计算enu原点到各个enu位置的变换
        for loc in locations:
            if loc is None:
                RT_o_merge.append(None)
            else:
                RT_o_merge.append(get_RT_from_enu_yaw(enu=loc['location'], yaw=loc['yaw']))

        for i in range(len(ts)):
            RT_o_master = RT_o_merge[i]
            points_master = points_merge[i]
            if RT_o_master is None:
                pass
            else:
                for j in range(i - (merged_num // 2) * merged_step, i + (merged_num // 2 + 1) * merged_step, merged_step):
                    if j < 0 or j >= len(ts) or j == i:
                        continue
                    if RT_o_merge[j] is None:
                        continue
                    RT = np.linalg.inv(RT_o_master) @ RT_o_merge[j]
                    points = points_filtered[j]
                    points_inten = copy.deepcopy(points[:, 3])  # 保留强度信息
                    points = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # 齐次变换操作的坐标第四维必须是1
                    points = points @ RT.T  # 将其他位置原点云转到主位置坐标系下
                    points[:, 3] = points_inten
                    points_master = np.vstack((points_master, points))
            self.merged_points[ts[i]] = points_master

    def get(self, timestamp):
        if timestamp not in self.merged_points:
            return None
        return self.merged_points[timestamp]


if __name__ == '__main__':
    location_path = 'ad_system_common_sdc:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-19/2024_03_19_07_37_49_gacGtParser/ved/pose/location.tmp.txt'
    pcd_root_path = 'ad_system_common_sdc:s3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-19/2024_03_19_07_37_49_gacGtParser/lidar/top_center_lidar/'

    points_merger = MergePcdByLocation(
                                 location_path=location_path,
                                 pcd_root_path=pcd_root_path,
                                 rm_dynamic=True,
                                 merged_num=5,
                                 merged_step=1
                                 )
    points = points_merger.get(1710833869199000000)

    save_path = "/workspace/kongzelong2/pillar_test/240611/test_pcd_infer/merged.pcd"

    pcd = open3d.t.geometry.PointCloud()
    pcd.point["positions"] = open3d.core.Tensor(points[:, :3], open3d.core.float32)
    pcd.point["intensity"] = open3d.core.Tensor(points[:, 3:], open3d.core.float32)
    open3d.t.io.write_point_cloud(save_path, pcd)
