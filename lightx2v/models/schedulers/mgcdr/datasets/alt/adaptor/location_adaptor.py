import os
from tqdm import tqdm
import copy
from loguru import logger
from alt import load, dump, smart_glob, smart_exists


class LocaltionFileAdaptor(object):
    SUCCESS_TYPES = ['WIDELANE', 'NARROWLANE', 'L1_INT', 'WIDE_INT', 'NARROW_INT', 'INS_RTKFIXED']
    FAILED_TYPES = ['NONE', 'L1_FLOAT', 'PSRDIFF', 'NARROW_FLOAT', 'SINGLE'] # 部分

    @classmethod
    def get_clip_gnss_status(cls, location_root):
        gnss_status_file = smart_glob('{}/*_bestgnsspos.csv'.format(location_root))

        assert len(gnss_status_file) == 1
        gnss_status_metas = load(gnss_status_file[0])

        failed_num = 0
        for idx, status_mata in gnss_status_metas.iterrows():
            if status_mata['position_type'] not in cls.SUCCESS_TYPES:
                failed_num += 1
        return failed_num / gnss_status_metas.shape[0]

    @classmethod
    def update(cls, src_meta, threshold=0.5):
        if isinstance(src_meta, str):
            src_meta = load(src_meta)

        location_path = src_meta['data_structure']['location']['local']
        
        if location_path.endswith('localization.tmp.txt'):  # 已经是局部定位
            return src_meta

        glabal_location_path = cls.map_bucket(src_meta, location_path)

        # 这个路径是固定的？
        location_root = glabal_location_path.rsplit('/', 2)[0]
        gnss_status = cls.get_clip_gnss_status(location_root)

        if gnss_status < threshold:
            logger.info('gnss_status: {}'.format(gnss_status))
            local_location_path = os.path.join(location_root, 'localization.tmp.txt')
            assert smart_exists(local_location_path), location_root
            local_location_path = cls.map_bucket(src_meta, local_location_path, reversed=True)

            dst_meta = copy.deepcopy(src_meta)
            dst_meta['data_structure']['location']['local'] = local_location_path
            return dst_meta

        return src_meta
        
    @classmethod
    def map_bucket(cls, src_meta, path, reversed=False):
        mapping_metas = {}
        for src_bucket, mapping_metas in src_meta['bucket_mapper'].items():
            src_string = '{' + src_bucket + '}'
            if reversed:
                if mapping_metas['name'] in path:
                    return path.replace(mapping_metas['name'], src_string)
            else:
                if src_string in path:
                    return path.replace(src_string, mapping_metas['name'])
        return path
    
    @classmethod
    def update_location(cls, global_location_path, threshold=0.5):
        if global_location_path.endswith('localization.tmp.txt'):  # 已经是局部定位
            return global_location_path

        location_root = global_location_path.rsplit('/', 2)[0]
        gnss_status = cls.get_clip_gnss_status(location_root)

        if gnss_status < threshold:
            logger.info('gnss_status: {}'.format(gnss_status))
            local_location_path = os.path.join(location_root, 'localization.tmp.txt')
            assert smart_exists(local_location_path), location_root
            
            return local_location_path
        return global_location_path

if __name__ == '__main__':
    # input_txt = 'result/pvb_labels/input.tmp.txt'

    # total_res = []
    # for clip_meta_json in tqdm(open(input_txt).readlines()):
    #     dst_meta = LocaltionFileAdaptor.update(clip_meta_json.strip(), threshold=0.5)
    #     # dump('meta.json', dst_meta)
    
    location_path = 's3://sdc_gac/Data_Collection/GT_data/gacGtParser/drive_gt_collection/GOP_gt/2024-03/A02-290/2024-03-19/2024_03_19_07_37_49_gacGtParser/ved/pose/location.tmp.txt'
    location_path = LocaltionFileAdaptor.update_location(location_path)
