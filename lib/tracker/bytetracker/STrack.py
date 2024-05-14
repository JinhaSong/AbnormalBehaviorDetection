from lib.tracker.bytetracker.basetrack import BaseTrack, TrackState
from lib.tracker.bytetracker.kalman_filter import KalmanFilter
from itertools import combinations
import numpy as np

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.score = score
        self.cls=cls
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.cp = []
        self.overlap = []
        self.merge = []
        self.meta_label = 1
        self.switch_state = False
        self.action_state = None

    @staticmethod
    def record_center_point(frame, stracks):
        for strack in stracks:
            center_point = np.asarray([strack.xyah[0], strack.xyah[1]])
            strack.cp.append({'frame_count': frame, 'center_point': center_point})

    @staticmethod
    def check_overlap(stracks):
        if len(stracks) > 0:
            combination_stracks = list(combinations(stracks, 2))
            for a_strack, b_strack in combination_stracks:
                if a_strack.cls == 0 and b_strack.cls == 0:
                    if (
                            ((b_strack.tlbr[0] <= a_strack.tlbr[0] <= b_strack.tlbr[2] or b_strack.tlbr[0] <= a_strack.tlbr[2] <= b_strack.tlbr[2]) and (b_strack.tlbr[1] <= a_strack.tlbr[1] <= b_strack.tlbr[3] or b_strack.tlbr[1] <= a_strack.tlbr[3] <= b_strack.tlbr[3])) or
                            ((a_strack.tlbr[0] <= b_strack.tlbr[0] <= a_strack.tlbr[2] or a_strack.tlbr[0] <= b_strack.tlbr[2] <= a_strack.tlbr[2]) and (a_strack.tlbr[1] <= b_strack.tlbr[1] <= a_strack.tlbr[3] or a_strack.tlbr[1] <= b_strack.tlbr[3] <= a_strack.tlbr[3]))
                        ):
                        a_strack.overlap.append(b_strack)
                        set(a_strack.overlap)
                        b_strack.overlap.append(a_strack)
                        set(b_strack.overlap)

    @staticmethod
    def init_switch_action_state(stracks):
        for strack in stracks:
            strack.switch_state = False
            strack.action_state = None

    @staticmethod
    def init_overlap(stracks):
        for strack in stracks:
            strack.overlap = []

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)