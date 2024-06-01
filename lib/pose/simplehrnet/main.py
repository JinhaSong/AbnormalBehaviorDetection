import os
import torch
import time

from lib.pose.simplehrnet.SimpleHRNet import SimpleHRNet

class SimpleHRNetWrapper:
    model = None
    result = None
    dataset_class = None
    prob_thresh = 0.0
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, cfg, device):
        super().__init__()
        self.model_name = cfg["model_name"]
        model_path = cfg["model_path"]
        hrnet_c = cfg["hrnet_c"]
        hrnet_j = cfg["hrnet_j"]
        self.model = SimpleHRNet(hrnet_c, hrnet_j, checkpoint_path=model_path, device=torch.device("cuda:" + str(device)))

    def inference_image(self, image, detection_result):
        result = self.model.predict(image, detection_result)

        return result