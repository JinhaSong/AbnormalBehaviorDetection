import os
import cv2
import time
import torch
import numpy as np

from lib.pose.alphapose.alphapose.alphapose.utils.transforms import get_func_heatmap_to_coord
from lib.pose.alphapose.alphapose.alphapose.utils.pPose_nms import pose_nms
from lib.pose.alphapose.alphapose.alphapose.utils.presets import SimpleTransform
from lib.pose.alphapose.alphapose.alphapose.models import builder
from lib.pose.alphapose.alphapose.alphapose.utils.config import update_config

class AlphaPose:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, cfg, device):
        self.model_name   = cfg["model_name"]
        self.model_config = cfg["model_config"]
        self.model_path   = cfg["model_path"]
        self.dataset      = cfg["dataset"]
        self.person_num = 0
        self.cfg = update_config(self.model_config)
        self.device = "cuda:" + device
        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self.eval_joints = list(range(self.cfg.DATA_PRESET.NUM_JOINTS))

        self.input_size = self.cfg.DATA_PRESET.IMAGE_SIZE
        self.output_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.sigma = self.cfg.DATA_PRESET.SIGMA

        self.pose_model.to(self.device)
        self.pose_model.eval()
        self.transformation = SimpleTransform(
            self.pose_dataset,
            scale_factor=0,
            input_size=self.input_size,
            output_size=self.output_size,
            rot=0, sigma=self.sigma,
            train=False, add_dpg=False, gpu_device=self.device)

    def process(self, image, result):
        array_scores, array_boxes, array_ids, final_result, person_num = self.reformat_od_result(result)
        pose = None
        if len(array_scores) > 0:
            orig_img = self.image_preprocess(image)
            inps, orig_img, boxes, scores, ids, cropped_boxes, final_result = self.preprocess_od_result(orig_img, array_scores, array_boxes, array_ids, final_result)
            with torch.no_grad():
                if orig_img is None or inps is None:
                    return final_result, pose
                else:
                    inps = inps.to(self.device)
                    hm = self.pose_model(inps)
                    hm = hm.cpu()
                    final_result, pose = self.reformat_pe_result(boxes, scores, ids, hm, cropped_boxes, orig_img, final_result)
            return final_result, pose, person_num  # , pose
        else:
            return final_result, pose, person_num

    def letterbox_image(self, img, inp_dim):
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

        return canvas

    def prep_frame(self, img, inp_dim):
        orig_im = img
        img = (self.letterbox_image(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_

    def image_preprocess(self, image):
        img = self.prep_frame(image, self.cfg.get('INP_DIM', 608))

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        orig_img = image

        return orig_img

    def image_postprocess(self, transformation, orig_img, boxes, scores, ids, inps, cropped_boxes):
        with torch.no_grad():
            if orig_img is None:
                pose = (None, None, None, None, None, None)
                return pose
            if boxes is None or boxes.nelement() == 0:
                pose = (None, orig_img, boxes, scores, ids, None)
                return pose

            for i, box in enumerate(boxes):
                inps[i], cropped_box = transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

        return inps, orig_img, boxes, scores, ids, cropped_boxes

    def reformat_od_result(self, result):
        final_result = []
        # get boxes
        array_boxes = []
        array_scores = []
        array_ids = []
        persons = []

        person_num = 0
        for obj in result:
            label = obj['label'][0]['description']
            if label == "person":
                persons.append(obj)
                person_num += 1
            else:
                final_result.append(obj)

        for person in persons:
            score = person['label'][0]['score'] / 100
            position = person['position']
            x = position['x'] - 10
            y = position['y'] - 10
            w = position['w'] + 10
            h = position['h'] + 10

            array_scores.append(score)
            array_boxes.append([x, y, x + w, y + h])
            array_ids.append(0.)

        return array_scores, array_boxes, array_ids, final_result, person_num

    def preprocess_od_result(self, orig_img, array_scores, array_boxes, array_ids,final_result):
        scores = torch.Tensor(np.array(array_scores))
        boxes = torch.Tensor(np.array(array_boxes))
        ids = torch.Tensor(np.array(array_ids))

        inps = torch.zeros(boxes.size(0), 3, *self.input_size)
        cropped_boxes = torch.zeros(boxes.size(0), 4)

        inps, orig_img, boxes, scores, ids, cropped_boxes = self.image_postprocess(self.transformation, orig_img, boxes, scores, ids, inps, cropped_boxes)

        return inps, orig_img, boxes, scores, ids, cropped_boxes, final_result

    def reformat_pe_result(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, final_result):
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0, 136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0, 26)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                               norm_type=None)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(boxes, scores, ids, preds_img, preds_scores, 0)

            _result = []
            keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle"]
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'bbox': [boxes[k][0], boxes[k][1], boxes[k][2] - boxes[k][0], boxes[k][3] - boxes[k][1]]
                    }
                )
            result = {
                'result': _result
            }

            for p in range(len(scores)):
                obj = {
                    "label": [{
                        "description": "person",
                        "class_idx": 0,
                        "score": float(scores[p]) * 100
                    }],
                    "position": {
                        "x": boxes[p][0],
                        "y": boxes[p][1],
                        "w": boxes[p][2] - boxes[p][0],
                        "h": boxes[p][3] - boxes[p][1]
                    },
                    "pose": {}
                }
                for k, keypoint_name in enumerate(keypoint_names):
                    obj["pose"][str(k)] = {
                        "name": keypoint_name,
                        "x": float(preds_img[p][k][0]),
                        "y": float(preds_img[p][k][1]),
                    }
                final_result.append(obj)

        return final_result, result

    def inference_image(self, image, detection_result):
        """
        :param image: input image(np array)
        :param dets: detection results(bounding box(x1, y1, x2, y2), score, class, class index)
            - format:
                [{"label": [{"description": cls, "score": score, "class_idx": cls_idx}],
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...]
        :return: dict format bounding box(x1, y1, x2, y2), score, class, class index
            - format:
                [{"label": [{"description": cls, "score": score, "class_idx": cls_idx}],
                 "position": {"x": x, "y": y, "w": w, "h": h}, "pose": [{"name": joint_name, "x": x, "y": y}, ...]}, ...]
        """
        start_time = time.time()
        result, pose, person_num = self.process(image, detection_result)
        self.person_num = person_num
        self.result = result
        end_time = time.time()
        self.analysis_time = end_time - start_time

        return self.result

    def inference_image_batch(self, images, detection_results):
        """
        :param image: input images(list in dict: [np array])
        :param dets: detection results(bounding box(x1, y1, x2, y2), score, class, class index) of each images
            - format:
                [[{"label": {"description": cls, "score": score, "class_idx": cls_idx},
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...], ...]
        :return: dict format bounding box(x1, y1, x2, y2), score, class, class index and pose
            - format:
                [[{"label": [{"description": cls, "score": score, "class_idx": cls_idx}],
                 "position": {"x": x, "y": y, "w": w, "h": h}, "pose": [{"name": joint_name, "x": x, "y": y}, ...]}, ...], ...]
        """
        results = []
        total_person_num = 0
        for i, (image, detection_result) in enumerate(zip(images, detection_results)):
            self.person_num = 0
            result = self.inference_image(image, detection_result)
            total_person_num += self.person_num
            results.append(result)
        self.person_num = total_person_num
        self.results = results

        return results

if __name__ == '__main__':
    model = AlphaPose()