from lib.tracker import Tracker
from lib.tracker.sort.utils import *
from lib.tracker.sort.KalmanBoxTracker import KalmanBoxTracker


class Sort(Tracker):
    def __init__(self, params):
        """
        Sets key parameters for SORT
        """
        super().__init__(params)
        self.score_threshold = float(params["score_threshold"])
        self.max_age = int(params["max_age"])
        self.min_hits = int(params["min_hits"])
        self.trackers = []
        self.frame_count = 0
        self.tracker_name = params["tracker_name"]

    @staticmethod
    def reformat_detection_result(detection_result):
        det_list = []
        for info_ in detection_result:
            dets = []
            if info_['label'][0]['description'] == "person":
                dets.append(float(info_['position']['x']))
                dets.append(float(info_['position']['y']))
                dets.append(float(info_['position']['x'] + info_['position']['w']))
                dets.append(float(info_['position']['y'] + info_['position']['h']))
                dets.append(float(info_['label'][0]['score']))
                det_list.append(dets)
        return det_list

    def update(self, detection_result):
        """
        Params:
          detection_result: object detection results in following format
            ※ refer to the following path for format:
              $PROJECT_ROOT/detector/object_detection/template/ObjectDetection.py
        Converted object detection result:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        detection_result = self.filter_object_result(detection_result, self.score_threshold)
        dets = self.reformat_detection_result(detection_result['results'][0]['detection_result'])
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
