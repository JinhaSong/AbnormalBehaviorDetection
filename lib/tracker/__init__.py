class Tracker(object):
    def __init__(self, params):
        self.params = params

    def update(self, detection_result):
        return None

    @staticmethod
    def filter_object_result(detection_result, score_threshold):
        object_result = []
        for obj in detection_result["results"][0]["detection_result"]:
            score = obj["label"][0]["score"]
            if score > score_threshold:
                object_result.append(obj)
        detection_result["results"][0]["detection_result"] = object_result
        return detection_result