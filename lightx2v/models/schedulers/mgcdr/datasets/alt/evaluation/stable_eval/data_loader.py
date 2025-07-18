import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from alt.utils.global_helper import EVAL_DATA_LOADER_REGISTRY
from alt.utils.petrel_helper import global_petrel_helper


class Type_is:

    def __init__(self):

        self.type_cnt = defaultdict(dict)

    def get_type(self, track_id):

        max_type_id = -1
        max_times = -1
        for type_id, type_cnt in self.type_cnt[track_id].items():
            if type_cnt > max_times:
                max_times = type_cnt
                max_type_id = type_id

        return max_type_id

    def update_type(self, track_id, type_id):

        if self.type_cnt[track_id].get(type_id, None) is None:
            self.type_cnt[track_id][type_id] = 1
        else:
            self.type_cnt[track_id][type_id] += 1


def push_data_to_structA(obj_formated, obj_type, datas_org_by_trackIds):

    if datas_org_by_trackIds[obj_type].get(obj_formated["track_id"], None) is None:
        datas_org_by_trackIds[obj_type][obj_formated["track_id"]] = [
            obj_formated
        ]
    else:
        datas_org_by_trackIds[obj_type][obj_formated["track_id"]].append(
            obj_formated)

    return datas_org_by_trackIds


def push_data_to_structB(obj_dict, frame_id, datas_org_by_frameIds):

    for obj_type, obj_list in obj_dict.items():
        if len(obj_list) and datas_org_by_frameIds[obj_type].get(obj_list[0]["frame_id"], None) is None:
            datas_org_by_frameIds[obj_type][frame_id] = obj_list
        else:
            datas_org_by_frameIds[obj_type][frame_id].extend(obj_list)
    return datas_org_by_frameIds


@EVAL_DATA_LOADER_REGISTRY.register("bev_data_loader")
def bev_data_loader(data, types):

    type_is = Type_is()
    datas_org_by_trackIds = defaultdict(dict)
    datas_org_by_frameIds = defaultdict(dict)
    all_tracks_id = set()

    if isinstance(data, list):
        iter_data = data
    else:
        iter_data = global_petrel_helper.readlines(data)

    for idx, data_frame in tqdm(enumerate(iter_data)):
        if isinstance(data_frame, str):
            data_frame = json.loads(data_frame)
        timestamp = data_frame["timestamp"]
        frame_id = idx
        obj_dict = defaultdict(list)
        if "tracklets" not in data_frame.keys():
            continue
        # obj:[x, y, z, l, h, w, yaw, label, confidence, trackid]
        for obj in data_frame["tracklets"]:

            obj_type = obj[7]
            obj_id = obj[9]
            if types and not (obj_type in types) and not (obj_id in all_tracks_id):
                continue
            obj_formated = dict(track_id=obj[9],
                                type=obj[7],
                                subtype=None,
                                speed=None,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                x=obj[0],
                                y=obj[1],
                                z=obj[2],
                                heading=obj[6],
                                width=obj[5],
                                length=obj[3],
                                height=obj[4],
                                sensor_id=None,
                                type_name=None)

            all_tracks_id.add(obj_formated["track_id"])
            type_is.update_type(obj_formated["track_id"],
                                obj_formated["type"])
            cur_obj_type = type_is.get_type(obj_formated["track_id"])

            obj_dict[cur_obj_type].append(obj_formated)
            datas_org_by_trackIds = push_data_to_structA(
                obj_formated, cur_obj_type, datas_org_by_trackIds)
        datas_org_by_frameIds = push_data_to_structB(
            obj_dict, frame_id, datas_org_by_frameIds)

    return datas_org_by_trackIds, datas_org_by_frameIds


@EVAL_DATA_LOADER_REGISTRY.register("baidu_data_loader")
def baidu_data_loader(data, types):

    def convert_timestamp(date_string):
        date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
        timestamp = date_object.timestamp()
        return timestamp

    type_is = Type_is()
    datas_org_by_trackIds = defaultdict(dict)
    datas_org_by_frameIds = defaultdict(dict)
    all_tracks_id = set()

    if isinstance(data, list):
        iter_data = data
    else:
        iter_data = global_petrel_helper.readlines(data)

    for idx, data_frame in tqdm(enumerate(iter_data)):
        if isinstance(data_frame, str):
            data_frame = json.loads(data_frame)
        frame_id = idx
        obj_dict = defaultdict(list)

        for obj in data_frame:

            obj_type = obj["PtcType"]
            obj_id = obj["PtcID"]
            timestamp = convert_timestamp(obj["Timestamp"])
            if types and not (obj_type in types) and not (obj_id in all_tracks_id):
                continue
            obj_formated = dict(track_id=obj_id,
                                type=obj_type,
                                subtype=None,
                                speed=None,
                                frame_id=frame_id,
                                timestamp=timestamp,
                                x=obj["x"],
                                y=obj["y"],
                                z=obj["z"],
                                heading=obj["heading"],
                                width=obj["width"],
                                length=obj["length"],
                                height=obj["height"],
                                sensor_id=None,
                                type_name=None)

            all_tracks_id.add(obj_formated["track_id"])
            type_is.update_type(obj_formated["track_id"],
                                obj_formated["type"])
            cur_obj_type = type_is.get_type(obj_formated["track_id"])

            obj_dict[cur_obj_type].append(obj_formated)
            datas_org_by_trackIds = push_data_to_structA(
                obj_formated, cur_obj_type, datas_org_by_trackIds)
        datas_org_by_frameIds = push_data_to_structB(
            obj_dict, frame_id, datas_org_by_frameIds)

    return datas_org_by_trackIds, datas_org_by_frameIds


@EVAL_DATA_LOADER_REGISTRY.register("sdk_data_loader")
def sdk_data_loader(data, types):

    type_is = Type_is()
    datas_org_by_trackIds = defaultdict(dict)
    datas_org_by_frameIds = defaultdict(dict)
    all_tracks_id = set()

    if isinstance(data, list):
        iter_data = data
    else:
        iter_data = global_petrel_helper.readlines(data)

    for idx, data_frame in tqdm(enumerate(iter_data)):
        if isinstance(data_frame, str):
            data_frame = json.loads(data_frame)
        frame_id = idx
        obj_dict = defaultdict(list)
        if len(data_frame) == 0:
            continue
        data_frame = data_frame[0]
        # obj:[x, y, z, l, h, w, yaw, label, confidence, trackid]
        for obj in data_frame["targets"]:
            timestamp = obj["fuse"]["G_timestamp"]
            obj_type = obj["fuse"]["G_class"]
            obj_id = obj["fuse"]["G_track_id"]
            if types and not (obj_type in types) and not (obj_id in all_tracks_id):
                continue
            if (isinstance(obj["fuse"]["G_speed_x"], dict) or 
                isinstance(obj["fuse"]["G_speed_y"], dict) or
                isinstance(obj["fuse"]["G_speed"], dict)):

                v_x = None
                v_y = None
                v_z = None
            else:
                v_x = obj["fuse"]["G_speed_x"]
                v_y = obj["fuse"]["G_speed_y"]
                v_z = obj["fuse"]["G_speed"]
            obj_formated = dict(track_id=obj_id,
                                type=obj_type,
                                subtype=None,
                                speed={"x": v_x, "y": v_y, "z": v_z},# noqa
                                frame_id=frame_id,
                                timestamp=timestamp,
                                x=obj["fuse"]["G_enu"][0],
                                y=obj["fuse"]["G_enu"][0],
                                z=0.0,
                                heading=obj["fuse"]["G_heading"] / 180.0 * np.pi,
                                width=obj["fuse"]["G_size"][1],
                                length=obj["fuse"]["G_size"][0],
                                height=obj["fuse"]["G_size"][2],
                                sensor_id=None,
                                type_name=None)

            all_tracks_id.add(obj_formated["track_id"])
            type_is.update_type(obj_formated["track_id"],
                                obj_formated["type"])
            cur_obj_type = type_is.get_type(obj_formated["track_id"])

            obj_dict[cur_obj_type].append(obj_formated)
            datas_org_by_trackIds = push_data_to_structA(
                obj_formated, cur_obj_type, datas_org_by_trackIds)
        datas_org_by_frameIds = push_data_to_structB(
            obj_dict, frame_id, datas_org_by_frameIds)

    return datas_org_by_trackIds, datas_org_by_frameIds


@EVAL_DATA_LOADER_REGISTRY.register("Pusher_Msg_2011_data_loader")
def Pusher_Msg_2011_data_loader(data, types):

    type_is = Type_is()
    datas_org_by_trackIds = defaultdict(dict)
    datas_org_by_frameIds = defaultdict(dict)
    all_tracks_id = set()
    if isinstance(data, list):
        iter_data = data
    else:
        df = pd.read_excel(data)
        iter_data = df.groupby('帧号')

    # 遍历每一组
    for frame_id, data_frame in iter_data:
        obj_dict = defaultdict(list)
        for _, obj in data_frame.iterrows():

            obj_type = obj["PtcType交通参与者类型"]
            obj_id = obj["ObjID目标ID"]
            heading = obj["PtcHeading航向角"] / 180.0 * np.pi
            if types and not (obj_type in types) and not (obj_id in all_tracks_id):
                continue
            obj_formated = dict(track_id=obj_id,
                                type=obj_type,
                                subtype=None,
                                speed=None,
                                frame_id=frame_id,
                                timestamp=obj["CaptureTime"],
                                x=0.0,
                                y=0.0,
                                z=0.0,
                                heading=heading,
                                width=obj["VehW宽"],
                                length=obj["VehL长"],
                                height=obj["VehH高"],
                                sensor_id=None,
                                type_name=None)

            all_tracks_id.add(obj_formated["track_id"])
            type_is.update_type(obj_formated["track_id"],
                                obj_formated["type"])
            cur_obj_type = type_is.get_type(obj_formated["track_id"])

            obj_dict[cur_obj_type].append(obj_formated)
            datas_org_by_trackIds = push_data_to_structA(
                obj_formated, cur_obj_type, datas_org_by_trackIds)
        datas_org_by_frameIds = push_data_to_structB(
            obj_dict, frame_id, datas_org_by_frameIds)

    return datas_org_by_trackIds, datas_org_by_frameIds


@EVAL_DATA_LOADER_REGISTRY.register("alt_data_loader")
def alt_data_loader(data, types):

    type_is = Type_is()
    datas_org_by_trackIds = defaultdict(dict)
    datas_org_by_frameIds = defaultdict(dict)
    all_tracks_id = set()

    timestamps = list(data.keys())
    timestamps.sort()

    iter_data = data

    for idx, timestamp in tqdm(enumerate(timestamps)):
        data_frame = data[timestamp]

        frame_id = idx
        obj_dict = defaultdict(list)
        if len(data_frame) == 0:
            continue

        for obj in data_frame:
            timestamp = timestamp
            obj_type = obj["label"]
            obj_id = obj["id"]
            if types and not (obj_type in types) and not (obj_id in all_tracks_id):
                continue

            v_x = obj["bev_vel"][0]
            v_y = obj["bev_vel"][1]
            v_z = obj["bev_vel"][2]
            obj_formated = dict(track_id=obj_id,
                                type=obj_type,
                                subtype=None,
                                speed={"x": v_x, "y": v_y, "z": v_z},# noqa
                                frame_id=frame_id,
                                timestamp=timestamp,
                                x=obj["location"][0],
                                y=obj["location"][1],
                                z=obj["location"][2],
                                heading=obj["yaw"],
                                width=obj["dimension"][0],
                                length=obj["dimension"][2],
                                height=obj["dimension"][1],
                                sensor_id=None,
                                type_name=None)

            all_tracks_id.add(obj_formated["track_id"])
            type_is.update_type(obj_formated["track_id"],
                                obj_formated["type"])
            cur_obj_type = type_is.get_type(obj_formated["track_id"])

            obj_dict[cur_obj_type].append(obj_formated)
            datas_org_by_trackIds = push_data_to_structA(
                obj_formated, cur_obj_type, datas_org_by_trackIds)
        datas_org_by_frameIds = push_data_to_structB(
            obj_dict, frame_id, datas_org_by_frameIds)

    return datas_org_by_trackIds, datas_org_by_frameIds