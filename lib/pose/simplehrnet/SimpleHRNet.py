import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from lib.pose.simplehrnet.models.hrnet import HRNet
from lib.pose.simplehrnet.models.poseresnet import PoseResNet

class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=True,
                 return_heatmaps=False,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 device=torch.device("cpu")):

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.device = device

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            # print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                # print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                # print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            # print("device: 'cpu'")
            pass
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def predict(self, image, detections, prob_thresh=0.5):
        if len(image.shape) == 3:
            return self._predict_single(image, detections, prob_thresh)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    @staticmethod
    def correct_coordinates(image, x1, y1, x2, y2):
        h, w = image.shape[:2]
        x1 = np.clip(x1, 0, w)
        x2 = np.clip(x2, 0, w)
        y1 = np.clip(y1, 0, h)
        y2 = np.clip(y2, 0, h)
        return x1, y1, x2, y2

    def _predict_single(self, image, detections, prob_thresh):
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),  # (width, height)
                    interpolation=self.interpolation
                )

            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]
            heatmaps = np.zeros((1, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            if detections is not None:
                for i, detection in enumerate(detections):
                    x1 = int(round(detection["position"]["x"]))
                    x2 = int(round(detection["position"]["x"] + detection["position"]["w"]))
                    y1 = int(round(detection["position"]["y"]))
                    y2 = int(round(detection["position"]["y"] + detection["position"]["h"]))
                    x1, y1, x2, y2 = self.correct_coordinates(image, x1, y1, x2, y2)

                    # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                    correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        # 세로를 늘림 (마진을 줄이기 위해 비율 조정을 완화)
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor * 0.9))  # 마진을 줄임
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        # 가로를 늘림 (마진을 줄이기 위해 비율 조정을 완화)
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor * 0.9))  # 마진을 줄임
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)

                    # 이미지 경계를 넘지 않도록 추가 보정
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)

                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    boxes[i] = [x1, y1, x2, y2]
                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        person_ids = np.arange(len(pts), dtype=np.int32)


        point_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]

        results = []
        for i, (pt, pid, detection) in enumerate(zip(pts, person_ids, detections)):
            pose = {}
            for j, pt in enumerate(pt):
                pose["{}".format(j)] = {
                    "name": str(point_names[j]),
                    "x": float(pt[1]),
                    "y": float(pt[0])
                }
            detection["pose"] = pose
            results.append(detection)

        return results

    def _predict_batch(self, images):
        if not self.multiperson:
            old_res = images[0].shape

            if self.resolution is not None:
                images_tensor = torch.empty(images.shape[0], 3, self.resolution[0], self.resolution[1])
            else:
                images_tensor = torch.empty(images.shape[0], 3, images.shape[1], images.shape[2])

            for i, image in enumerate(images):
                if self.resolution is not None:
                    image = cv2.resize(
                        image,
                        (self.resolution[1], self.resolution[0]),  # (width, height)
                        interpolation=self.interpolation
                    )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images_tensor[i] = self.transform(image)

            images = images_tensor
            boxes = np.repeat(
                np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32), len(images), axis=0
            )  # [x1, y1, x2, y2]
            heatmaps = np.zeros((len(images), self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            image_detections = self.detector.predict(images)

            base_index = 0
            nof_people = int(np.sum([len(d) for d in image_detections if d is not None]))
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images_tensor = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            for d, detections in enumerate(image_detections):
                image = images[d]
                if detections is not None and len(detections) > 0:
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))
                        x1, y1, x2, y2 = self.correct_coordinates(image, x1, y1, x2, y2)

                        # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                        correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                        if correction_factor > 1:
                            # increase y side
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                            # increase x side
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)

                        boxes[base_index + i] = [x1, y1, x2, y2]

                        images_tensor[base_index + i] = self.transform(image[y1:y2, x1:x2, ::-1])

                    base_index += len(detections)

            images = images_tensor

        images = images.to(self.device)

        if images.shape[0] > 0:
            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

            if self.multiperson:
                # re-add the removed batch axis (n)
                if self.return_heatmaps:
                    heatmaps_batch = []
                if self.return_bounding_boxes:
                    boxes_batch = []
                pts_batch = []
                index = 0
                for detections in image_detections:
                    if detections is not None:
                        pts_batch.append(pts[index:index + len(detections)])
                        if self.return_heatmaps:
                            heatmaps_batch.append(heatmaps[index:index + len(detections)])
                        if self.return_bounding_boxes:
                            boxes_batch.append(boxes[index:index + len(detections)])
                        index += len(detections)
                    else:
                        pts_batch.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
                        if self.return_heatmaps:
                            heatmaps_batch.append(np.zeros((0, self.nof_joints, self.resolution[0] // 4,
                                                            self.resolution[1] // 4), dtype=np.float32))
                        if self.return_bounding_boxes:
                            boxes_batch.append(np.zeros((0, 4), dtype=np.float32))
                if self.return_heatmaps:
                    heatmaps = heatmaps_batch
                if self.return_bounding_boxes:
                    boxes = boxes_batch
                pts = pts_batch

            else:
                pts = np.expand_dims(pts, axis=1)

        else:
            boxes = np.asarray([], dtype=np.int32)
            if self.multiperson:
                pts = []
                for _ in range(len(image_detections)):
                    pts.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
            else:
                raise ValueError  # should never happen

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]
