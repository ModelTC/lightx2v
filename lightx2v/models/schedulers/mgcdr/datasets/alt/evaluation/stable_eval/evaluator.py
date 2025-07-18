import json

import numpy as np


class Abnormal_Type:

    Size_ab_h = 0  # height change abnormally
    Size_ab_w = 1  # width change abnormally
    Size_ab_l = 2  # length change abnormally
    Vel_ab = 3  # velocity change abnormally
    Azi_ab = 4  # Azimuth change abnormally
    Short_track_ab = 5  # short track abnormally
    Track_lost_ab = 6  # object lost abnormally
    Type_change = 7  # type change abnormally
    Curva_ab = 8  # curvtation change abnormally
    Vel_x_ab = 9
    Vel_y_ab = 10
    Vel_z_ab = 11
    Yaw_ab = 12
    Position_ab = 13


class Abnormal_Object:

    def __init__(self,
                 obj_type,
                 obj_subtype,
                 ab_type,
                 track_id,
                 ab_times_interval,
                 frame_id,
                 extra_info=None,
                 **kwargs):
        self.obj_type = obj_type
        self.obj_subtyep = obj_subtype
        self.abnormal_type = ab_type
        self.track_id = track_id
        self.ab_time_list = ab_times_interval
        self.frame_ids = frame_id
        self.extra_info = extra_info
        self.sensor_id = kwargs.get("sensor_id", [-1])
        self.type_name = kwargs.get("type_name", None)

    def __str__(self):

        tmp = {}
        tmp["obj_type"] = self.obj_type
        tmp["obj_subtype"] = self.obj_subtyep
        tmp["abnormal_type"] = self.abnormal_type
        tmp["track_id"] = self.track_id
        tmp["abnoraml_times"] = self.ab_time_list
        tmp["frame_ids"] = self.frame_ids
        tmp["extra_info"] = self.extra_info
        tmp["sensor_id"] = self.sensor_id
        tmp["type_name"] = self.type_name
        fmt_str = json.dumps(tmp)
        return fmt_str


class Evaluator():

    def __init__(self, cfgs):
        """
        init all thresh
        """
        # abnormal info records
        self.ab_records = []

        # eval items
        self.eval_type_jump = False if cfgs["evaluations_items"].get(
            "eval_type_jump") is None else True
        self.eval_speed_jump = False if cfgs["evaluations_items"].get(
            "eval_speed_jump") is None else True
        self.eval_short_track = False if cfgs["evaluations_items"].get(
            "eval_short_track") is None else True
        self.eval_length_jump = False if cfgs["evaluations_items"].get(
            "eval_length_jump") is None else True
        self.eval_height_jump = False if cfgs["evaluations_items"].get(
            "eval_height_jump") is None else True
        self.eval_width_jump = False if cfgs["evaluations_items"].get(
            "eval_width_jump") is None else True
        self.eval_heading_jump = False if cfgs["evaluations_items"].get(
            "eval_heading_jump") is None else True
        self.eval_track_lost = False if cfgs["evaluations_items"].get(
            "eval_track_lost") is None else True
        self.eval_freespace_jump = False if cfgs["evaluations_items"].get(
            "eval_freespace_jump") is None else True
        self.eval_curvature_jump = False if cfgs["evaluations_items"].get(
            "eval_curvature_jump") is None else True
        self.eval_yaw_jump = False if cfgs["evaluations_items"].get(
            "eval_yaw_jump") is None else True
        self.eval_position_jump = False if cfgs["evaluations_items"].get(
            "eval_position_jump") is None else True
        # eval config parameters
        self.speed_win_size = None
        self.heading_win_size = None
        self.width_win_size = None
        self.height_win_size = None
        self.length_win_size = None
        self.freespace_win_size = None
        self.curvature_win_size = None

        self.win_size = cfgs["eval_global_params"]["win_size"]

        if self.eval_heading_jump:
            self.thresh_heading = cfgs["evaluations_items"][
                "eval_heading_jump"]["thresh_heading"]
            self.heading_win_size = cfgs["evaluations_items"][
                "eval_heading_jump"].get("heading_win_size", None)
        if self.eval_short_track:
            self.thresh_short_track = cfgs["evaluations_items"][
                "eval_short_track"]["thresh_short_track"]
        if self.eval_track_lost:
            self.thresh_track_lost = cfgs["evaluations_items"][
                "eval_track_lost"]["skip_frame_time"]
        if self.eval_length_jump:
            self.thresh_length = cfgs["evaluations_items"]["eval_length_jump"][
                "thresh_length"]
            self.length_win_size = cfgs["evaluations_items"][
                "eval_length_jump"].get("length_win_size", None)
        if self.eval_height_jump:
            self.thresh_height = cfgs["evaluations_items"]["eval_height_jump"][
                "thresh_height"]
            self.height_win_size = cfgs["evaluations_items"][
                "eval_height_jump"].get("height_win_size", None)
        if self.eval_width_jump:
            self.thresh_width = cfgs["evaluations_items"]["eval_width_jump"][
                "thresh_width"]
            self.width_win_size = cfgs["evaluations_items"][
                "eval_width_jump"].get("width_win_size", None)
        if self.eval_freespace_jump:
            self.thresh_freespace = cfgs["evaluations_items"][
                "eval_freespace_jump"]["thresh_freespace"]
            self.freespace_win_size = cfgs["evaluations_items"][
                "eval_freespace_jump"].get("freespace_win_size", None)
        if self.eval_curvature_jump:
            self.thresh_curvature = cfgs["evaluations_items"][
                "eval_curvature_jump"]["thresh_curvature"]
            self.curvature_win_size = cfgs["evaluations_items"][
                "eval_curvature_jump"].get("curvature_win_size", None)
        if self.eval_yaw_jump:
            self.thresh_yaw = cfgs["evaluations_items"]["eval_yaw_jump"][
                "thresh_yaw"]
            self.yaw_win_size = cfgs["evaluations_items"]["eval_yaw_jump"].get(
                "yaw_win_size", None)

        if self.eval_position_jump:
            self.skip_frame_time = cfgs["evaluations_items"]["eval_position_jump"].get(
                "skip_frame_time", None)
            self.thresh_position = cfgs["evaluations_items"]["eval_position_jump"].get(
                "thresh_position", None)

        self.is_speed_vector = True
        if self.eval_speed_jump:

            self.thresh_speed_x = cfgs["evaluations_items"]["eval_speed_jump"][
                "thresh_vector_speed_x"]
            self.thresh_speed_y = cfgs["evaluations_items"]["eval_speed_jump"][
                "thresh_vector_speed_y"]
            self.thresh_speed_z = cfgs["evaluations_items"]["eval_speed_jump"][
                "thresh_vector_speed_z"]
            self.speed_win_size = cfgs["evaluations_items"][
                "eval_speed_jump"].get("speed_win_size", None)

    def _get_obj_total_num(self, datas_org_by_frame):

        obj_total_num = 0
        for frame_id, objects in datas_org_by_frame.items():

            obj_total_num += len(objects)
        return obj_total_num

    def eval(self, type, datas_org_by_frame, datas_org_by_track):
        """
        running eval
        """

        eval_result = {}

        datas_org_by_frame = datas_org_by_frame
        datas_org_by_track = datas_org_by_track

        self.eval_type_name = None

        track_total_num = len(datas_org_by_track)
        obj_total_num = self._get_obj_total_num(datas_org_by_frame)
        track_total_num = len(datas_org_by_track)
        eval_result["obj_total_num"] = obj_total_num
        eval_result["track_total_num"] = track_total_num

        if self.eval_short_track:

            short_track_num = self._short_track(datas_org_by_track)
            if track_total_num:
                eval_result[
                    "short_track_pass_ratio"] = 1 - short_track_num / track_total_num
            else:
                eval_result["short_track_pass_ratio"] = 0
            eval_result["short_track_exception_num"] = short_track_num
            eval_result["short_track_normal_num"] = eval_result[
                "track_total_num"] - short_track_num
            eval_result["short_track_" + "track_total_num"] = eval_result["track_total_num"]

        if self.eval_type_jump:

            type_jump_num = self._type_change(datas_org_by_track)
            if obj_total_num:
                eval_result[
                    "type_jump_pass_ratio"] = 1 - type_jump_num / obj_total_num
            else:
                eval_result["type_jump_pass_ratio"] = 0
            eval_result["type_jump_exception_num"] = type_jump_num
            eval_result["type_jump_normal_num"] = eval_result[
                "obj_total_num"] - type_jump_num
            eval_result["type_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_track_lost:

            track_lost_ratio, track_lost_num = self._track_lost(datas_org_by_track, self.thresh_track_lost)
            if track_total_num:
                eval_result[
                    "track_lost_ratio"] = track_lost_ratio / track_total_num
            else:
                eval_result["track_lost_ratio"] = 0
            eval_result["track_lost_exception_num"] = track_lost_num
            eval_result["track_lost_" + "track_total_num"] = eval_result["track_total_num"]

        if self.eval_height_jump:

            height_win_size = self.win_size
            if self.height_win_size:
                height_win_size = self.height_win_size

            h_jump_num = self._common_jump_eval(
                datas_org_by_track,
                ab_type=Abnormal_Type.Size_ab_h,
                eval_key="height",
                thresh=self.thresh_height,
                win_size=height_win_size)
            if obj_total_num:
                eval_result[
                    "height_jump_pass_ratio"] = 1 - h_jump_num / obj_total_num
            else:
                eval_result["height_jump_pass_ratio"] = 0
            eval_result["height_jump_exception_num"] = h_jump_num
            eval_result["height_jump_normal_num"] = eval_result[
                "obj_total_num"] - h_jump_num
            eval_result["height_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_width_jump:

            width_win_size = self.win_size
            if self.width_win_size:
                width_win_size = self.width_win_size

            w_jump_num = self._common_jump_eval(
                datas_org_by_track,
                ab_type=Abnormal_Type.Size_ab_w,
                eval_key="width",
                thresh=self.thresh_width,
                win_size=width_win_size)
            if obj_total_num:
                eval_result[
                    "width_jump_pass_ratio"] = 1 - w_jump_num / obj_total_num
            else:
                eval_result["width_jump_pass_ratio"] = 0
            eval_result["width_jump_exception_num"] = w_jump_num
            eval_result["width_jump_normal_num"] = eval_result[
                "obj_total_num"] - w_jump_num
            eval_result["width_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_length_jump:

            length_win_size = self.win_size
            if self.length_win_size:
                length_win_size = self.length_win_size

            l_jump_num = self._common_jump_eval(
                datas_org_by_track,
                ab_type=Abnormal_Type.Size_ab_l,
                eval_key="length",
                thresh=self.thresh_length,
                win_size=length_win_size)
            if obj_total_num:
                eval_result[
                    "length_jump_pass_ratio"] = 1 - l_jump_num / obj_total_num
            else:
                eval_result["length_jump_pass_ratio"] = 0
            eval_result["length_jump_exception_num"] = l_jump_num
            eval_result["length_jump_normal_num"] = eval_result[
                "obj_total_num"] - l_jump_num
            eval_result["length_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_heading_jump:

            heading_win_size = self.win_size
            if self.heading_win_size:
                heading_win_size = self.heading_win_size
            heading_jump_num = self._common_jump_eval(datas_org_by_track, ab_type=Abnormal_Type.Azi_ab,
                                                      eval_key="heading", thresh=self.thresh_heading,
                                                      win_size=heading_win_size, use_std=False)
            if obj_total_num:
                eval_result[
                    "heading_jump_pass_ratio"] = 1 - heading_jump_num / obj_total_num
            else:
                eval_result["heading_jump_pass_ratio"] = 0
            eval_result["heading_jump_exception_num"] = heading_jump_num
            eval_result["heading_jump_normal_num"] = eval_result[
                "obj_total_num"] - heading_jump_num
            eval_result["heading_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_yaw_jump:

            yaw_win_size = self.win_size
            if self.yaw_win_size:
                yaw_win_size = self.yaw_win_size # noqa

            yaw_jump_num = self._yaw_jump_eval(datas_org_by_track)
            # method2 for eval laneline yaw jump
            # yaw_jump_num = self._common_jump_eval(datas_org_by_track, ab_type=Abnormal_Type.Yaw_ab,
            #                                       eval_key = "coffts",thresh=self.thresh_yaw,
            #                                       win_size = yaw_win_size, use_std = True, sub_key=1)

            if obj_total_num:
                eval_result[
                    "yaw_jump_pass_ratio"] = 1 - yaw_jump_num / obj_total_num
            else:
                eval_result["yaw_jump_pass_ratio"] = 0
            eval_result["yaw_jump_exception_num"] = yaw_jump_num
            eval_result["yaw_jump_normal_num"] = eval_result[
                "obj_total_num"] - yaw_jump_num
            eval_result["yaw_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_freespace_jump:

            freespace_win_size = self.win_size
            if self.freespace_win_size:
                freespace_win_size = self.freespace_win_size # noqa
            freespace_jump_num = self._freespace_jump(datas_org_by_frame, freespace_win_size)
            if obj_total_num:
                eval_result[
                    "freespace_jump_pass_ratio"] = 1 - freespace_jump_num / obj_total_num
            else:
                eval_result["freespace_jump_pass_ratio"] = 0
            eval_result["freespace_jump_exception_num"] = freespace_jump_num
            eval_result["freespace_jump_normal_num"] = eval_result[
                "obj_total_num"] - freespace_jump_num
            eval_result["freespace_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_curvature_jump:

            curvature_win_size = self.win_size
            if self.curvature_win_size:
                curvature_win_size = self.curvature_win_size
            curvature_jump_num = self._curvature_jump_eval_v1(
                datas_org_by_track,
                curvature_win_size,
                eval_key="coffts",
                sub_key=-2)
            eval_result["curvature_jump_" + "exception_num"] = curvature_jump_num
            if obj_total_num > 0:
                eval_result["curvature_jump_" + "pass_ratio"] = 1 - curvature_jump_num / obj_total_num
            else:
                eval_result["curvature_jump_" + "pass_ratio"] = 0
            eval_result["curvature_jump_" + "obj_total_num"] = eval_result["obj_total_num"]
            eval_result["curvature_jump_" + "normal_num"] = eval_result[
                "obj_total_num"] - curvature_jump_num

        if self.eval_position_jump:

            position_jump_num = self._position_jump_eval(datas_org_by_track,
                                                         ab_type=Abnormal_Type.Position_ab,
                                                         thresh=self.thresh_position,
                                                         skip_frame_time=self.skip_frame_time)

            if obj_total_num:
                eval_result[
                    "position_jump_pass_ratio"] = 1 - position_jump_num / obj_total_num
            else:
                eval_result["position_jump_pass_ratio"] = 0
            eval_result["position_jump_exception_num"] = position_jump_num
            eval_result["position_jump_normal_num"] = eval_result[
                "obj_total_num"] - position_jump_num
            eval_result["position_jump_" + "obj_total_num"] = eval_result["obj_total_num"]

        if self.eval_speed_jump:

            speed_win_size = self.win_size
            if self.speed_win_size:
                speed_win_size = self.speed_win_size

            if self.is_speed_vector:
                x_speed_jump_num = self._common_jump_eval(
                    datas_org_by_track,
                    ab_type=Abnormal_Type.Vel_x_ab,
                    eval_key="speed",
                    thresh=self.thresh_speed_x,
                    win_size=speed_win_size,
                    use_std=False,
                    sub_key='x')
                y_speed_jump_num = self._common_jump_eval(
                    datas_org_by_track,
                    ab_type=Abnormal_Type.Vel_y_ab,
                    eval_key="speed",
                    thresh=self.thresh_speed_y,
                    win_size=speed_win_size,
                    use_std=False,
                    sub_key='y')
                z_speed_jump_num = self._common_jump_eval(
                    datas_org_by_track,
                    ab_type=Abnormal_Type.Vel_z_ab,
                    eval_key="speed",
                    thresh=self.thresh_speed_z,
                    win_size=speed_win_size,
                    use_std=False,
                    sub_key='z')
                if obj_total_num:
                    eval_result[
                        "speed_jump_x_pass_ratio"] = 1 - x_speed_jump_num / obj_total_num
                    eval_result[
                        "speed_jump_y_pass_ratio"] = 1 - y_speed_jump_num / obj_total_num
                    eval_result[
                        "speed_jump_z_pass_ratio"] = 1 - z_speed_jump_num / obj_total_num
                else:
                    eval_result["speed_jump_x_pass_ratio"] = 0
                    eval_result["speed_jump_y_pass_ratio"] = 0
                    eval_result["speed_jump_z_pass_ratio"] = 0

                eval_result["speed_jump_x_exception_num"] = x_speed_jump_num
                eval_result["speed_jump_x_normal_num"] = eval_result[
                    "obj_total_num"] - x_speed_jump_num
                eval_result["speed_jump_x_" + "obj_total_num"] = eval_result["obj_total_num"]

                eval_result["speed_jump_y_exception_num"] = y_speed_jump_num
                eval_result["speed_jump_y_normal_num"] = eval_result[
                    "obj_total_num"] - y_speed_jump_num
                eval_result["speed_jump_y_" + "obj_total_num"] = eval_result["obj_total_num"]

                eval_result["speed_jump_z_exception_num"] = z_speed_jump_num
                eval_result["speed_jump_z_normal_num"] = eval_result[
                    "obj_total_num"] - z_speed_jump_num
                eval_result["speed_jump_z_" + "obj_total_num"] = eval_result["obj_total_num"]
            else:

                speed_jump_num = self._common_jump_eval(
                    datas_org_by_frame,
                    ab_type=Abnormal_Type.Vel_ab,
                    eval_key="speed",
                    thresh=self.thresh_speed,
                    win_size=speed_win_size,
                    use_std=True)
                if obj_total_num:
                    eval_result[
                        "speed_jump_pass_ratio"] = 1 - speed_jump_num / obj_total_num
                else:
                    eval_result["speed_jump_pass_ratio"] = 0
                eval_result["speed_jump_exception_num"] = speed_jump_num
                eval_result["speed_jump_normal_num"] = eval_result[
                    "obj_total_num"] - speed_jump_num
                eval_result["speed_jump_" + "obj_total_num"] = eval_result["obj_total_num"]
        return eval_result

    def get_abnormal_info(self):

        return self.ab_records

    def _short_track(self, datas_org_by_track):

        short_track_num = 0
        for t_id, object_tracks in datas_org_by_track.items():
            tracks_num = len(object_tracks)

            if tracks_num < self.thresh_short_track:

                short_track_num += 1
                start_timestamp = self._get_time(datas_org_by_track[t_id][0])
                end_timestamp = self._get_time(datas_org_by_track[t_id][-1])

                obj_type = object_tracks[-1]["type"]
                obj_subtype = object_tracks[-1].get("sub_type", None)
                ab_type = Abnormal_Type.Short_track_ab
                track_id = object_tracks[-1]["track_id"]
                sensor_id = object_tracks[-1].get("sensor_id", [-1])
                abnormal_timestamp = [start_timestamp, end_timestamp]
                frame_ids = [
                    object_tracks[0]["frame_id"], object_tracks[-1]["frame_id"]
                ]
                extra_info = "object tracked num:{}".format(tracks_num)
                ab_obj = Abnormal_Object(obj_type,
                                         obj_subtype,
                                         ab_type,
                                         track_id,
                                         abnormal_timestamp,
                                         frame_ids,
                                         extra_info,
                                         sensor_id=sensor_id,
                                         type_name=self.eval_type_name)
                self.ab_records.append(ab_obj)

        return short_track_num

    def _type_change(self, datas_org_by_track):

        type_jump_num = 0
        for t_id, object_tracks in datas_org_by_track.items():

            if len(object_tracks) <= 1:
                continue

            for cur_obj_idx in range(1, len(object_tracks)):

                bef_obj_idx = cur_obj_idx - 1
                cur_obj = object_tracks[cur_obj_idx]
                bef_obj = object_tracks[bef_obj_idx]
                cur_obj_type = cur_obj["type"]
                bef_obj_type = bef_obj["type"]
                cur_obj_subtype = cur_obj.get("sub_type", None) # noqa
                bef_obj_subtype = bef_obj.get("sub_type", None) # noqa

                if not (cur_obj_type == bef_obj_type):
                    type_jump_num += 1
                    start_timestamp = self._get_time(bef_obj)
                    end_timestamp = self._get_time(cur_obj)

                    obj_type = bef_obj["type"]
                    obj_subtype = bef_obj.get("sub_type", None)
                    ab_type = Abnormal_Type.Type_change
                    track_id = t_id
                    sensor_id = bef_obj.get("sensor_id", [-1])
                    abnormal_timestamp = [start_timestamp, end_timestamp]
                    frame_ids = cur_obj["frame_id"]
                    extra_info = "(frame_id:{}, type:{}, sub_type:{}) -> (frame_id:{}, type:{}, sub_type:{})".\
                                   format(bef_obj["frame_id"], bef_obj["type"], bef_obj.get("sub_type", None), # noqa
                                   cur_obj["frame_id"], cur_obj["type"], cur_obj.get("sub_type", None)) # noqa
                    ab_obj = Abnormal_Object(obj_type,
                                             obj_subtype,
                                             ab_type,
                                             track_id,
                                             abnormal_timestamp,
                                             frame_ids,
                                             extra_info,
                                             sensor_id=sensor_id,
                                             type_name=self.eval_type_name)
                    self.ab_records.append(ab_obj)

        return type_jump_num

    def _track_lost(self, datas_org_by_track, thresh):

        track_lost_ab_num = 0
        skip_frame_time = thresh
        track_lost_ratio_total = 0
        for t_id, object_tracks in datas_org_by_track.items():
            track_size = len(object_tracks)
            if track_size <= 1:
                continue
            obj_track_lost_num = 0
            for cur_id in range(1, track_size):
                bef_id = cur_id - 1
                cur_obj = object_tracks[cur_id]
                bef_obj = object_tracks[bef_id]
                if cur_obj["timestamp"] - bef_obj["timestamp"] > 1.3 * skip_frame_time:

                    track_lost_ab_num += 1
                    obj_track_lost_num += 1
                    start_timestamp = self._get_time(
                        datas_org_by_track[t_id][bef_id])
                    end_timestamp = self._get_time(
                        datas_org_by_track[t_id][cur_id])
                    obj_type = object_tracks[-1]["type"]
                    obj_subtype = object_tracks[-1].get("sub_type", None)
                    ab_type = Abnormal_Type.Track_lost_ab
                    track_id = object_tracks[-1]["track_id"]
                    sensor_id = object_tracks[-1].get("sensor_id", [-1])
                    abnormal_timestamp = [start_timestamp, end_timestamp]
                    frame_ids = [
                        object_tracks[bef_id]["frame_id"],
                        object_tracks[cur_id]["frame_id"]
                    ]
                    extra_info = "object tracks lost too much from frame:{} to frame:{}".format(object_tracks[bef_id]["frame_id"], object_tracks[cur_id]["frame_id"]) # noqa
                    ab_obj = Abnormal_Object(obj_type,
                                             obj_subtype,
                                             ab_type,
                                             track_id,
                                             abnormal_timestamp,
                                             frame_ids,
                                             extra_info,
                                             sensor_id=sensor_id,
                                             type_name=self.eval_type_name)
                    self.ab_records.append(ab_obj)
            track_size_true = (object_tracks[-1]["timestamp"] - object_tracks[0]["timestamp"]) // skip_frame_time
            if track_size_true == 0:
                track_size_true = track_size
            lost_ratio = obj_track_lost_num / track_size_true
            track_lost_ratio_total += lost_ratio
        return track_lost_ratio_total, track_lost_ab_num

    # multi frames and single object colc statics
    def _common_jump_eval(self,
                          datas_org_by_track,
                          ab_type,
                          eval_key,
                          thresh,
                          win_size,
                          use_std=False,
                          sub_key=None):

        jump_num = 0
        multi_frames_objects = []
        for t_id, object_tracks in datas_org_by_track.items():

            track_size = len(object_tracks)
            if track_size == 0:
                continue
            for i in range(1, track_size // win_size):

                multi_frames_objects = object_tracks[(i - 1) * win_size:i * win_size]
                start_frame_idx, end_frame_idx = multi_frames_objects[0]["frame_id"], multi_frames_objects[-1]["frame_id"] # noqa
                if sub_key:
                    items = [multi_frames_objects[i][eval_key][sub_key] for i in range(len(multi_frames_objects)) if multi_frames_objects[i][eval_key][sub_key]]
                else:
                    items = [multi_frames_objects[i][eval_key] for i in range(len(multi_frames_objects)) if multi_frames_objects[i][eval_key]]

                items_std = np.std(items, ddof=1)
                items_mean_val = np.average(items)
                cur_thresh = thresh

                if use_std:
                    cur_thresh = float(thresh) * items_std
                for obj, item in zip(multi_frames_objects, items):
                    diff = item - items_mean_val
                    if ab_type == Abnormal_Type.Azi_ab:
                        if diff < -1.0 * np.pi:
                            diff += 2.0 * np.pi
                        if diff > np.pi:
                            diff -= 2.0 * np.pi
                    diff = abs(diff)

                    if abs(item - items_mean_val) > cur_thresh:

                        start_timestamp = self._get_time(
                            multi_frames_objects[0])
                        end_timestamp = self._get_time(
                            multi_frames_objects[-1])
                        jump_num += 1

                        obj_type = obj["type"]
                        obj_subtype = obj.get("sub_type", None)
                        track_id = obj["track_id"]
                        sensor_id = obj.get("sensor_id", [-1])
                        abnormal_timestamp = [start_timestamp, end_timestamp]
                        frame_ids = obj["frame_id"]
                        extra_info = "std:{}, mean:{}, obj:{}".format(items_std, items_mean_val, item)
                        ab_obj = Abnormal_Object(obj_type, obj_subtype, ab_type,
                                                 track_id, abnormal_timestamp,
                                                 frame_ids, extra_info,
                                                 sensor_id=sensor_id,
                                                 type_name=self.eval_type_name)
                        self.ab_records.append(ab_obj)
        return jump_num

    def _curvature_jump_eval(self, datas_org_by_frame, win_size):

        quad_idx = -2
        win_cnt = 0
        curva_jump_num = 0
        multi_frames_lines = []
        frame_ids = []
        for frame_id, lines in datas_org_by_frame.items():
            if win_cnt < win_size:
                if len(lines):
                    multi_frames_lines.extend(lines)
                    frame_ids.append(frame_id)
                win_cnt += 1
            elif len(multi_frames_lines):
                win_cnt = 0
                start_frame_idx, end_frame_idx = min(frame_ids), max(frame_ids)
                quadratics = [
                    multi_frames_lines[i]["coffts"][quad_idx]
                    for i in range(len(multi_frames_lines))
                ]

                curva_std = np.std(quadratics, ddof=1)
                mean_curva_val = np.average(quadratics)

                for line, quad in zip(multi_frames_lines, quadratics):
                    if abs(quad - mean_curva_val) > self.thresh_curvature * curva_std:
                        start_timestamp = self._get_time(
                            datas_org_by_frame[start_frame_idx][-1])
                        end_timestamp = self._get_time(
                            datas_org_by_frame[end_frame_idx][-1])
                        curva_jump_num += 1

                        obj_type = line["type"]
                        obj_subtype = line.get("sub_type", None)  #
                        ab_type = Abnormal_Type.Curva_ab
                        track_id = line["track_id"]
                        sensor_id = line.get("sensor_id", [-1])
                        abnormal_timestamp = [start_timestamp, end_timestamp]
                        frame_ids = line["frame_id"]
                        extra_info = "curvature_std:{}, curvature_mean:{}, line_curvature:{}".format(
                            curva_std, mean_curva_val, quad)
                        ab_obj = Abnormal_Object(obj_type,
                                                 obj_subtype,
                                                 ab_type,
                                                 track_id,
                                                 abnormal_timestamp,
                                                 frame_ids,
                                                 extra_info,
                                                 sensor_id=sensor_id,
                                                 type_name=self.eval_type_name)
                        self.ab_records.append(ab_obj)
                multi_frames_lines = []
                frame_ids = []

        return curva_jump_num

    def _yaw_jump_eval(self, datas_org_by_trackId):

        def colc_angle(vec_a, vec_b):

            a_norm = np.sqrt(np.sum(vec_a * vec_a))
            b_norm = np.sqrt(np.sum(vec_b * vec_b))
            cos_val = np.dot(vec_a, vec_b) / (a_norm * b_norm)
            cos_val = max(cos_val, -1)
            cos_val = min(cos_val, 1)
            arc_val = np.arccos(cos_val)
            angle_val = arc_val
            return angle_val

        yaw_jump_num = 0
        for t_id, line_track in datas_org_by_trackId.items():

            if len(line_track) <= 1:
                continue

            for cur_line_idx in range(1, len(line_track)):

                bef_line_idx = cur_line_idx - 1
                cur_line = line_track[cur_line_idx]
                bef_line = line_track[bef_line_idx]
                if cur_line["frame_id"] - bef_line["frame_id"] > 2:
                    continue

                cur_vec = np.array(cur_line["start_pt"]) - np.array(
                    cur_line["end_pt"])
                bef_vec = np.array(bef_line["start_pt"]) - np.array(
                    bef_line["end_pt"])

                angle_val = colc_angle(cur_vec, bef_vec)
                self.logger.debug(f"angle change:{angle_val}, eval thresh:{self.thresh_yaw}")
                if angle_val > self.thresh_yaw:

                    yaw_jump_num += 1
                    start_timestamp = self._get_time(bef_line)
                    end_timestamp = self._get_time(cur_line)

                    obj_type = bef_line["type"]
                    obj_subtype = bef_line.get("sub_type", None)  #
                    ab_type = Abnormal_Type.Yaw_ab
                    track_id = t_id
                    sensor_id = bef_line.get("sensor_id", [-1])
                    abnormal_timestamp = [start_timestamp, end_timestamp]
                    frame_ids = cur_line["frame_id"]
                    extra_info = "angle_change:{}".format(angle_val)
                    ab_obj = Abnormal_Object(obj_type,
                                             obj_subtype,
                                             ab_type,
                                             track_id,
                                             abnormal_timestamp,
                                             frame_ids,
                                             extra_info,
                                             sensor_id=sensor_id,
                                             type_name=self.eval_type_name)
                    self.ab_records.append(ab_obj)

        return yaw_jump_num

    def _curvature_jump_eval_v1(self,
                                datas_org_by_trackId,
                                win_size,
                                eval_key,
                                sub_key=None):

        curva_jump_num = 0
        frame_ids = []

        for t_id, object_tracks in datas_org_by_trackId.items():

            track_size = len(object_tracks)
            if track_size == 0:
                continue
            for i in range(1, np.int(np.ceil(track_size / win_size))):

                multi_frames_objects = object_tracks[(i - 1) * win_size:i * win_size]
                start_frame_idx, end_frame_idx = multi_frames_objects[0]["frame_id"], multi_frames_objects[-1]["frame_id"] # noqa
                if sub_key:
                    items = [multi_frames_objects[i][eval_key][sub_key] for i in range(len(multi_frames_objects))]
                else:
                    items = [multi_frames_objects[i][eval_key] for i in range(len(multi_frames_objects))]
                curva_std = np.std(items, ddof=1)
                mean_curva_val = np.average(items)

                for line, quad in zip(multi_frames_objects, items):
                    if abs(quad - mean_curva_val) > self.thresh_curvature * curva_std:
                        start_timestamp = self._get_time(
                            multi_frames_objects[0])
                        end_timestamp = self._get_time(
                            multi_frames_objects[-1])
                        curva_jump_num += 1

                        obj_type = line["type"]
                        obj_subtype = line.get("sub_type", None)  #
                        ab_type = Abnormal_Type.Curva_ab
                        track_id = line["track_id"]
                        sensor_id = line.get("sensor_id", [-1])
                        abnormal_timestamp = [start_timestamp, end_timestamp]
                        frame_ids = line["frame_id"]
                        extra_info = "curvature_std:{}, curvature_mean:{}, line_curvature:{}".format(
                            curva_std, mean_curva_val, quad)
                        ab_obj = Abnormal_Object(obj_type,
                                                 obj_subtype,
                                                 ab_type,
                                                 track_id,
                                                 abnormal_timestamp,
                                                 frame_ids,
                                                 extra_info,
                                                 sensor_id=sensor_id,
                                                 type_name=self.eval_type_name)
                        self.ab_records.append(ab_obj)
                frame_ids = []

        return curva_jump_num

    def _position_jump_eval(self,
                            datas_org_by_track,
                            ab_type,
                            thresh,
                            skip_frame_time=100):

        jump_num = 0
        for t_id, object_tracks in datas_org_by_track.items():

            track_size = len(object_tracks)
            if track_size == 0:
                continue
            for cur_idx in range(1, track_size):

                bef_idx = cur_idx - 1
                cur_obj = object_tracks[cur_idx]
                bef_obj = object_tracks[bef_idx]
                cur_thresh = thresh
                if cur_obj["timestamp"] - bef_obj["timestamp"] > 1.2 * skip_frame_time:
                    skip_frame = (cur_obj["timestamp"] - bef_obj["timestamp"]) / skip_frame_time
                    cur_thresh = skip_frame * thresh

                cur_obj_pos = [cur_obj['x'], cur_obj['y']]
                bef_obj_pos = [bef_obj['x'], bef_obj['y']]
                dist = np.sqrt(np.sum((np.array(cur_obj_pos) - np.array(bef_obj_pos))**2))

                if dist > cur_thresh:

                    start_timestamp = self._get_time(cur_obj)
                    end_timestamp = self._get_time(bef_obj)
                    jump_num += 1

                    obj = cur_obj
                    obj_type = obj["type"]
                    obj_subtype = obj.get("sub_type", None)
                    track_id = obj["track_id"]
                    sensor_id = obj.get("sensor_id", [-1])
                    abnormal_timestamp = [start_timestamp, end_timestamp]
                    frame_ids = obj["frame_id"]
                    extra_info = f"cur_obj:{cur_obj['x']}, {cur_obj['y']}, bef_obj:{bef_obj['x']}, {bef_obj['y']}"
                    ab_obj = Abnormal_Object(obj_type, obj_subtype, ab_type,
                                             track_id, abnormal_timestamp,
                                             frame_ids, extra_info,
                                             sensor_id=sensor_id, type_name=self.eval_type_name)
                    self.ab_records.append(ab_obj)
        return jump_num

    def _get_time(self, data_item):
        stamp = data_item["timestamp"]
        return stamp
