import os
import zipfile
import json
from tqdm import tqdm
from alt import smart_exists
from shapely.geometry import Point, Polygon
from loguru import logger


class PVBSensebeeFilter(object):
    _pinhole_polygon = Polygon([[0, 0], [1920, 0], [1920, 2160], [0, 2160]])
    _bev_polygon = Polygon([[1920, 0], [2880, 0], [2880, 2160], [1920, 2160]])
    _fisheye_polygon = Polygon([[2880, 0], [3840, 0], [3840, 2160], [2880, 2160]])
    _image_width, _image_height = 3840, 2160

    def __init__(self) -> None:
        pass

    @classmethod
    def caclu_single_frame_pass_rate(cls, meta):
        res = {'pinhole': True, 'fisheye': True, 'bev_vis': True}

        assert meta['width'] == cls._image_width
        assert meta['height'] == cls._image_height
        
        for box in meta['step_1']['result']:
            center_x, center_y = box['x'] + 0.5 * box['width'], box['y'] + 0.5 * box['height']

            center_point = Point(center_x, center_y)
            
            if cls._bev_polygon.contains(center_point):
                res['bev_vis'] = False
            elif cls._pinhole_polygon.contains(center_point):
                res['pinhole'] = False
            elif cls._fisheye_polygon.contains(center_point):
                res['fisheye'] = False
            else:
                raise NotImplementedError
        
        res['pass'] = res['bev_vis'] & res['fisheye'] & res['pinhole']
            
        return res
    
    @classmethod
    def process(cls, input_path, image_root):
        z = zipfile.ZipFile(input_path, 'r')

        res = {}
        for target in tqdm(z.infolist()):
            if target.filename in ['packagePageInfo.json']:
                continue

            image_name = target.filename.strip('.json')
            image_path = os.path.join(image_root, image_name)
            assert smart_exists(image_path), image_path
            meta = json.loads(z.open(target.filename,'r').read())

            res[image_name] = cls.caclu_single_frame_pass_rate(meta)

        bev_vis_pass_rate = len([item['bev_vis'] for item in res.values() if item['bev_vis']]) / len(res)
        pinhole_pass_rate = len([item['pinhole'] for item in res.values() if item['pinhole']]) / len(res)
        fisheye_pass_rate = len([item['fisheye'] for item in res.values() if item['fisheye']]) / len(res)
        total_pass_rate = len([item['pass'] for item in res.values() if item['pass']]) / len(res)

        logger.info('bev_vis_pass_rate: {} %'.format(round(bev_vis_pass_rate * 100, 3)))
        logger.info('pinhole_pass_rate: {} %'.format(round(pinhole_pass_rate * 100, 3)))
        logger.info('fisheye_pass_rate: {} %'.format(round(fisheye_pass_rate * 100, 3)))
        logger.info('total_pass_rate: {} %'.format(round(total_pass_rate * 100, 3)))

        return res

