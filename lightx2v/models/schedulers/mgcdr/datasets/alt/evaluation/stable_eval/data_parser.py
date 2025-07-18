from alt.utils.global_helper import EVAL_DATA_LOADER_REGISTRY

from .data_loader import *  # noqa


class data_parser():

    def __init__(self, data_path, data_loader=None):

        self.data = data_path
        self.timestamp2frame = {}
        #  {type1:{track_id_i:[obj_i1, obj_i2, ...], track_id_j:[obj_j1, obj_j2, ...]},# noqa
        #  type2:{track_id_i:[obj_i1, obj_i2, ...], track_id_j:[obj_j1, obj_j2, ...]}} # noqa
        self.datas_org_by_type_track = None
        #  {type1:{frame_id_i:[obj1, obj2, ...], frame_id_j:[obj1, obj2, ...]}, # noqa
        #  type2:{frame_id_i:[obj1, obj2, ...], frame_id_j:[obj1, obj2, ...]}} # noqa
        self.datas_org_by_type_frame = None
        self.data_loader = data_loader

    def data_parse(self, types, data=None, **kwargs):
        """
           types: list or None. assign eval object's category,
                  use to filter uneval catogry,if types' value
                  is None, eval all category object.
        """
        if data:
            parse_data = data
        else:
            parse_data = self.data
        self.datas_org_by_type_track, self.datas_org_by_type_frame = self.data_loader(parse_data, types)

        self.obj_types = list(self.datas_org_by_type_track.keys())

    def get_data_org_by_frame(self, obj_type):

        datas = self.datas_org_by_type_frame[obj_type]
        return datas

    def get_data_org_by_track(self, obj_type):

        datas = self.datas_org_by_type_track[obj_type]
        return datas

    def get_eval_types(self):

        return self.obj_types


class data_parser_factory():

    def __init__(self, data_path, frame_2nd_timestamps_file=None):

        self.data_path = data_path

    @classmethod
    def creat_data_parser(cls, eval_item_data_loader, data_path):

        data_loader = EVAL_DATA_LOADER_REGISTRY.get(eval_item_data_loader)

        data_parser_instance = data_parser(data_path, data_loader)
        return data_parser_instance
