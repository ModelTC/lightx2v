# Import from third library
import numpy as np

# Import from alt
from easydict import EasyDict
from loguru import logger

# Import from local
from . import matching
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .tracklet import BEVSTrack, joint_stracks, sub_stracks


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.bev_iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < -1)
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


default_class_cfg = {'VEHICLE': {"match_thresh": 5, "track_buffer": 20},
                     'PEDESTRIAN': {"match_thresh": 5, "track_buffer": 20},
                     'CYCLIST': {"match_thresh": 5, "track_buffer": 20}}


class BEVTracker(object):
    VEHECLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2

    def __init__(self, mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], class_cfg=default_class_cfg, quiet=True):
        self.quiet = quiet
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

        self.class_cfg = class_cfg
        self.det_thresh = 0
        self.initialize()

    def initialize(self):
        self.stracks = {}

        for label in self.class_cfg.keys():
            self.stracks[label] = EasyDict(
                {"tracked_stracks": [], "lost_stracks": [], "removed_stracks": [], "kalman_filter": KalmanFilter(ndim=7)}
            )
        self.frame_id = 0
        BaseTrack._count = 0

    def forward(self, dets, label):
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) > 0:
            """Detections"""
            detections = [BEVSTrack(det.data[:7], 1, label, det.data[7:]) for det in dets]
        else:
            detections = []
        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[BEVSTrack]
        for track in self.stracks[label].tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        """ Step 2: association, with IOU"""
        strack_pool = joint_stracks(tracked_stracks, self.stracks[label].lost_stracks)
        # Predict the current location with KF
        # for strack in strack_pool:
        #     strack.predict()
        BEVSTrack.multi_predict(strack_pool)

        if self.class_cfg[label]['mode'] == 'iou':
            smooth_bev_dists = matching.bev_iou_and_euclidean_distance(strack_pool, detections, "bev")
            origin_bev_dists = matching.bev_iou_and_euclidean_distance(strack_pool, detections, "origin_bev")
        else:
            smooth_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'bev')
            origin_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'origin_bev')
        
        # smooth_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'bev')
        # origin_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'origin_bev')

        dists = np.minimum(origin_bev_dists, smooth_bev_dists)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.class_cfg[label]['match_thresh'])

        # smooth_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'bev')
        # origin_bev_dists = matching.euclidean_and_shape_distance(strack_pool, detections, 'origin_bev')

        # dists = np.minimum(smooth_bev_dists, origin_bev_dists)

        # dists = matching.iou_distance(strack_pool, detections)
        # dists = matching.fuse_motion(self.stracks[label].kalman_filter, dists, strack_pool, detections)
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=5)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.stracks[label].kalman_filter, self.frame_id)
            activated_starcks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if self.frame_id - track.end_frame > self.class_cfg[label]["track_buffer"]:
                track.mark_removed()
                removed_stracks.append(track)
            else:
                track.mark_lost()
                lost_stracks.append(track)

        # """ Step 5: Update state"""
        # for track in self.stracks[label].lost_stracks:
        #     if self.frame_id - track.end_frame > self.max_time_lost:
        #         track.mark_removed()
        #         removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.stracks[label].tracked_stracks = [t for t in self.stracks[label].tracked_stracks if t.state == TrackState.Tracked]
        self.stracks[label].tracked_stracks = joint_stracks(self.stracks[label].tracked_stracks, activated_starcks)
        self.stracks[label].tracked_stracks = joint_stracks(self.stracks[label].tracked_stracks, refind_stracks)
        self.stracks[label].lost_stracks = sub_stracks(self.stracks[label].lost_stracks, self.stracks[label].tracked_stracks)
        self.stracks[label].lost_stracks.extend(lost_stracks)
        self.stracks[label].lost_stracks = sub_stracks(self.stracks[label].lost_stracks, self.stracks[label].removed_stracks)
        self.stracks[label].removed_stracks.extend(removed_stracks)

        self.stracks[label].tracked_stracks, self.stracks[label].lost_stracks = remove_duplicate_stracks(
            self.stracks[label].tracked_stracks, self.stracks[label].lost_stracks
        )
        # get scores of lost tracks
        output_stracks = [track for track in self.stracks[label].tracked_stracks if track.is_activated]

        if not self.quiet:
            logger.info("===========Frame {}, Cls {}==========".format(self.frame_id, label))
            logger.info("Activated: {}".format([track.track_id for track in activated_starcks]))
            logger.info("Refind: {}".format([track.track_id for track in refind_stracks]))
            logger.info("Lost: {}".format([track.track_id for track in lost_stracks]))
            logger.info("Removed: {}".format([track.track_id for track in removed_stracks]))

        return output_stracks

    def __call__(self, targets, **kwargs):
        split_class_out = self.split_by_class(targets)

        self.frame_id += 1
        output = []
        for label in self.class_cfg.keys():
            cur_output = self.forward(split_class_out[label], label)
            assert len(split_class_out[label]) == len(cur_output)
            output.extend(cur_output)

        return output

    def split_by_class(self, input):
        def split(x):
            out = {}
            for key in self.class_cfg.keys():
                out[key] = []

            for item in x:
                if item.corse_label in self.class_cfg.keys():
                    out[item.corse_label].append(item)
            return out

        split_out = split(input)
        return split_out
