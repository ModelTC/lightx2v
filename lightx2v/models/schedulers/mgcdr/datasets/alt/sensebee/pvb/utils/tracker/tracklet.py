import numpy as np
import copy
from collections import deque

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter

from . import matching


class STrack(BaseTrack):
    def __init__(self, tlwh, score, label, temp_feat=None, buffer_size=20):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.label = label
        self.tracklet_len = 0
        self.track_conf = 1.0
        self.buffer_size = buffer_size

        # self.cur_feature = cur_feature
        self.smooth_feature = None

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.5

    # def update_features(self, feat):
    #     feat /= np.linalg.norm(feat)
    #     self.curr_feat = feat
    #     if self.smooth_feat is None:
    #         self.smooth_feat = feat
    #     else:
    #         self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
    #     self.features.append(feat)
    #     self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_features(self, feat):
        # feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        # self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def activate(self, frame_id, **kwargs):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.update_features(new_track.curr_feat)
        self._tlwh = new_track._tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._tlwh = new_track._tlwh
        # new_tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def dt_bboxes(self):
        return np.concatenate(
            (self.tlbr,
             np.array([self.score, self.label, self.track_conf, self.track_id
                       ])),
            axis=0)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                      self.end_frame)


class BEVSTrack(BaseTrack):
    shared_kalman = KalmanFilter(ndim=7)

    def __init__(self, data, score, label, additional_info=[], buffer_size=30):
        # wait activate
        self._data = np.asarray(data, dtype=np.float64)
        self.kalman_filter = KalmanFilter(ndim=7)
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.label = label
        self.additional_info = additional_info

        self.score = score
        self.tracklet_len = 0

    @property
    def coordinate(self):
        return copy.deepcopy(self._data)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = BEVSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.coordinate)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.coordinate)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        
        self.score = new_track.score
        self._data = new_track._data
        self.additional_info = new_track.additional_info

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_coordinate = new_track.coordinate
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_coordinate)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self._data = new_track._data
        self.additional_info = new_track.additional_info

    @property
    def bev(self):
        if self.mean is not None:
            bev_box = copy.deepcopy([self.mean[:7][_idx] for _idx in [0, 1, 3, 4, 6]])
            # bev_box = copy.deepcopy([self._data[_idx] for _idx in [0, 1, 3, 4, 6]])
        else:
            bev_box = copy.deepcopy([self._data[_idx] for _idx in [0, 1, 3, 4, 6]])
        return bev_box

    @property
    def origin_bev(self):
        return copy.deepcopy([self._data[_idx] for _idx in [0, 1, 3, 4, 6]])

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
