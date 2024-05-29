from lib.tracker.bytetracker.kalman_filter import KalmanFilter
from lib.tracker.bytetracker.basetrack import TrackState
from lib.tracker.bytetracker.utils import *
from lib.tracker.bytetracker.STrack import STrack


class BYTETracker:
    def __init__(self, cfg):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.min_box_area = int(cfg["min_box_area"])

        self.frame_id = 0

        self.score_threshold = float(cfg["score_threshold"])
        self.match_thresh = float(cfg["match_threshold"])
        self.track_thresh = float(cfg["track_threshold"])
        self.det_thresh = float(cfg["track_threshold"]) + 0.1
        self.buffer_size = int(int(cfg["frame_rate"]) / 30.0 * int(cfg["track_buffer"]))
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.tracker_name = cfg["tracker_name"]

    @staticmethod
    def det_to_trk_data_conversion(detection_result):
        # detection_result = detection_result['results'][0]['detection_result']
        boxes_tmp_list = []
        boxes_list = []
        scores_tmp_list = []
        scores_list = []
        classes_tmp_list = []
        classes_list = []
        for info_ in detection_result:
            boxes_tmp_list.append(info_['position']['x'])
            boxes_tmp_list.append(info_['position']['y'])
            boxes_tmp_list.append(info_['position']['x'] + info_['position']['w'])
            boxes_tmp_list.append(info_['position']['y'] + info_['position']['h'])
            scores_tmp_list.append(info_['label'][0]['score'])
            classes_tmp_list.append(info_['label'][0]['class_idx'])
            boxes_list.append(boxes_tmp_list)
            scores_list.append(scores_tmp_list)
            classes_list.append(classes_tmp_list)
            boxes_tmp_list = []
            scores_tmp_list = []
            classes_tmp_list = []
        boxes_list_array = np.array(boxes_list)
        scores_list_array = np.array(scores_list)
        classes_list_array = np.array(classes_list)

        return boxes_list_array, scores_list_array, classes_list_array

    def update(self, detection_result):
        detection_result = self.filter_object_result(detection_result, self.score_threshold)
        bboxes, scores, classes = self.det_to_trk_data_conversion(detection_result)

        output = []
        self.frame_id += 1
        scores = scores.reshape(-1)
        classes = classes.reshape(-1)

        tracked_stracks = []
        lost_stracks = []
        removed_stracks = []
        unconfirmed_stracks = []
        activated_stracks = []
        refind_stracks = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed_stracks.append(track)
            else:
                tracked_stracks.append(track)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        is_human = classes == 0
        low_idxes_low_limit = 0.1 < scores
        low_idxes_high_limit = scores <= self.track_thresh
        det_low_idxes = np.logical_and(low_idxes_low_limit, low_idxes_high_limit)
        det_low_human_idxes = np.logical_and(det_low_idxes, is_human)
        det_high_idxes = scores > self.track_thresh
        det_high_human_idxes = np.logical_and(det_high_idxes, is_human)

        bboxes_high = bboxes[det_high_human_idxes]
        scores_high = scores[det_high_human_idxes]
        classes_high = classes[det_high_human_idxes]

        bboxes_low = bboxes[det_low_human_idxes]
        scores_low = scores[det_low_human_idxes]
        classes_low = classes[det_low_human_idxes]

        STrack.multi_predict(strack_pool)

        if len(bboxes_high) > 0:
            detections_high = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for (tlbr, s, cls) in zip(bboxes_high, scores_high, classes_high)]
        else:
            detections_high = []

        dists_first = matching.iou_distance(strack_pool, detections_high)
        dists_first = matching.fuse_score(dists_first, detections_high)
        matches_first, track_remain_idxes, detection_remain_idxes = matching.linear_assignment(dists_first, thresh=self.match_thresh)

        for track_idx, det_idx in matches_first:
            matched_track_high = strack_pool[track_idx]
            matched_det_high = detections_high[det_idx]
            if matched_track_high.state == TrackState.Tracked:
                matched_track_high.update(matched_det_high, self.frame_id)
                activated_stracks.append(matched_track_high)
            else:
                matched_track_high.re_activate(matched_det_high, self.frame_id, new_id=False)
                refind_stracks.append(matched_track_high)

        if len(bboxes_low) > 0:
            detections_low = [STrack(STrack.tlbr_to_tlwh(tlbr), s, cls) for (tlbr, s, cls) in zip(bboxes_low, scores_low, classes_low)]
        else:
            detections_low = []

        stracks_remain = [strack_pool[idx] for idx in track_remain_idxes if strack_pool[idx].state == TrackState.Tracked]
        dists_second = matching.iou_distance(stracks_remain, detections_low)
        matches_second, track_re_remain_idxes, detection_re_remain_idxes = matching.linear_assignment(dists_second, thresh=0.5)

        for track_idx, det_idx in matches_second:
            matched_track_remain = stracks_remain[track_idx]
            matched_det_low = detections_low[det_idx]
            if matched_track_remain.state == TrackState.Tracked:
                matched_track_remain.update(matched_det_low, self.frame_id)
                activated_stracks.append(matched_track_remain)
            else:
                matched_track_remain.re_activate(matched_det_low, self.frame_id, new_id=False)
                refind_stracks.append(matched_track_remain)

        for idx in track_re_remain_idxes:
            track_re_remain = stracks_remain[idx]
            if not track_re_remain.state == TrackState.Lost:
                track_re_remain.mark_lost()
                lost_stracks.append(track_re_remain)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        detections_remain = [detections_high[idx] for idx in detection_remain_idxes]
        dists_unconfirm = matching.iou_distance(unconfirmed_stracks, detections_remain)
        dists_unconfirm = matching.fuse_score(dists_unconfirm, detections_remain)
        matches_begin, u_unconfirmed, u_detection = matching.linear_assignment(dists_unconfirm, thresh=0.7)

        for track_idx, det_idx in matches_begin:
            matched_track_begin = unconfirmed_stracks[track_idx]
            matched_det_begin = detections_remain[det_idx]
            matched_track_begin.update(matched_det_begin, self.frame_id)
            activated_stracks.append(matched_track_begin)

        for it in u_unconfirmed:
            track = unconfirmed_stracks[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections_remain[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        STrack.record_center_point(self.frame_id, self.tracked_stracks)
        STrack.init_switch_action_state(self.tracked_stracks)
        STrack.init_overlap(self.tracked_stracks)
        STrack.check_overlap(self.tracked_stracks)

        output += self.tracked_stracks
        detection_result = self.assign_tracking_ids(detection_result, self.tracked_stracks)

        return detection_result

    def filter_object_result(self, detection_result, score_threshold):
        object_result = []
        for obj in detection_result:
            score = obj["label"][0]["score"]
            if score > score_threshold:
                object_result.append(obj)
        detection_result = object_result
        return detection_result

    def assign_tracking_ids(self, detection_result, tracked_stracks):
        """
        Assign tracking IDs to detection results.

        :param detection_result: List of detection results.
        :param tracked_stracks: List of tracked STrack objects.
        :return: List of detection results with assigned tracking IDs.
        """
        for det in detection_result:
            det_bbox = det['position']
            det_x1, det_y1 = int(det_bbox['x']), int(det_bbox['y'])
            det_x2, det_y2 = int(det_bbox['x'] + det_bbox['w']), int(det_bbox['y'] + det_bbox['h'])

            for track in tracked_stracks:
                track_bbox = track.tlbr
                track_x1, track_y1, track_x2, track_y2 = int(track_bbox[0]), int(track_bbox[1]), int(
                    track_bbox[2]), int(track_bbox[3])

                if det_x1 == track_x1 and det_y1 == track_y1 and det_x2 == track_x2 and det_y2 == track_y2:
                    det['tracking_id'] = track.track_id
                    break

        return detection_result