from alt.inference.lidar import LidarInferencer

class PVBInferencer(LidarInferencer):
    CACHE = '.cache_pvb'
    MODEL_NAME = 'mb15_pvb'  # 模型细节找fuhaiwen


if __name__ == '__main__':
    infer = PVBInferencer()

    img_path = 'ad_system_common_sdc:s3://sdc_gt_label/GAC/autolabel_20240410_768/Data_Collection/GT_data/gacGtParser/drive_gt_collection/PVB_gt/2024-03/A02-290/2024-03-22/2024_03_22_06_54_59_gacGtParser/gt_labels/cache/sensor/center_camera_fov30#s2/1711090526499999.855.png'

    res = infer.process([img_path])

    print(res)
