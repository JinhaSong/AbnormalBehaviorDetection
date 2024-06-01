import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

COCO_CLASSES_LIST = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

OBSTACLE_15 = [
    "person",
    "bicycle",
    "bus",
    "car",
    "carrier",
    "motorcycle",
    "movable_signage",
    "truck",
    "bollard",
    "chair",
    "potted_plant",
    "table",
    "tree_trunk	",
    "pole",
    "fire_hydrant",
]
ALPHA = 1
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
RED_RGB = (255, 0, 0)
HANGEUL_FONT = os.path.join(os.getcwd(), 'utility/font/malgun.ttf')
OBJECT_DATASET_CLASSES = {
    "coco": COCO_CLASSES_LIST,
    "obstacle-15": OBSTACLE_15,
}



def gen_colors(num_colors):
    """Generate different colors.
    # Arguments
      num_colors: total number of colors/classes.
    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.
    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.
    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin + 1, h - margin - 2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1] + h, topleft[0]:topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class Visualization():
    def __init__(self):
        self.object_class = OBJECT_DATASET_CLASSES["obstacle-15"]
        self.colors = gen_colors(len(self.object_class) + 1)
        self.joints = {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]
            ]
        }
        self.person_colors = {}

    def get_color(self, person_id):
        if person_id not in self.person_colors:
            new_color = gen_colors(1)[0]
            self.person_colors[person_id] = new_color
        return self.person_colors[person_id]

    def draw_bboxes(self, img, detection_results, score_threshold=0.5):
        for detection_result in detection_results:
            score = detection_result["label"][0]["score"]
            cl = detection_result["label"][0]["class_idx"]
            cls_name = detection_result["label"][0]["description"]
            bbox = detection_result["position"]
            x_min = int(bbox["x"])
            y_min = int(bbox["y"])
            x_max = int(bbox["x"] + bbox["w"])
            y_max = int(bbox["y"] + bbox["h"])

            color = self.colors[cl]
            if score >= score_threshold:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
                txt = '{} {:.2f}'.format(cls_name, score)
                img = draw_boxed_text(img, txt, txt_loc, color)
        return img

    def draw_bboxes_person(self, img, detection_results, score_threshold=0.5):
        for detection_result in detection_results:
            score = detection_result["label"][0]["score"]
            cls_name = detection_result["label"][0]["description"]
            person_id = detection_result["tracking_id"]
            bbox = detection_result["position"]
            x_min = int(bbox["x"])
            y_min = int(bbox["y"])
            x_max = int(bbox["x"] + bbox["w"])
            y_max = int(bbox["y"] + bbox["h"])

            if score >= score_threshold and cls_name == "person":
                color = self.get_color(person_id)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
                txt = '{} {:.2f} ID:{}'.format(cls_name, score, person_id)
                img = draw_boxed_text(img, txt, txt_loc, color)
        return img

    def draw_points_and_skeleton(self, image, pose_results, score_threshold=0.5):
        points = self.reformat_pose_results(pose_results)
        skeleton = self.joints["skeleton"]
        image = self.draw_skeleton(image, points, skeleton, score_threshold=score_threshold)
        image = self.draw_points(image, points, score_threshold=score_threshold)
        return image

    @staticmethod
    def reformat_pose_results(pose_results):
        points = []
        for i, pose_result in enumerate(pose_results):
            if pose_result["label"][0]["description"] == "person":
                dict_pose = pose_result["pose"]
                person = []
                for p in range(0, 17):
                    x = dict_pose[str(p)]["x"]
                    y = dict_pose[str(p)]["y"]
                    score = pose_result["label"][0]["score"]
                    person.append([y, x])
                person.append(score)
                points.append(person)
        return points

    def draw_points(self, image, points, color_palette='tab20', palette_samples=16, score_threshold=0.5):
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection

        for p, person in enumerate(points):
            if person[-1] >= score_threshold:
                for i, point in enumerate(person[:-1]):
                    image = cv2.circle(image, (int(point[1]), int(point[0])), circle_size,
                                       tuple(colors[i % len(colors)]), -1)

        return image

    def draw_skeleton(self, image, points, skeleton, color_palette='Set2', palette_samples=8, score_threshold=0.5):
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        for p, person in enumerate(points):
            if person[-1] >= score_threshold:
                for i, joint in enumerate(skeleton):
                    pt1 = person[joint[0]]
                    pt2 = person[joint[1]]
                    image = cv2.line(
                        image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                        tuple(colors[0 % len(colors)]), 2
                    )

        return image