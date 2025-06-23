import torch
import torch.nn as nn
import torch.nn.functional as F

from models.c3d.module.basicblock3d import BasicBlock3D
from models.c3d.module.bottleneck3d import Bottleneck3D


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=32):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout3d(p=0.3)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.feature_maps = None
        self.gradients = None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        def save_feature_map_hook(module, input, output):
            if not output.requires_grad:
                output = output.detach().clone().requires_grad_(True)

            self.feature_maps = output
            output.register_hook(self.save_gradients)

        self.layer4.register_forward_hook(save_feature_map_hook)

        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def save_gradients(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_feature_maps(self):
        return self.feature_maps

    def generate_cam(self, class_indices=None):
        if self.gradients is None:
            print("Gradients and feature maps are required for Grad-CAM")
            return None
        if self.feature_maps is None:
            print("Feature maps are required for Grad-CAM")
            return None

        gradients = self.gradients
        feature_maps = self.feature_maps

        min_batch_size = min(gradients.size(0), feature_maps.size(0))
        gradients = gradients[:min_batch_size]
        feature_maps = feature_maps[:min_batch_size]

        if class_indices is None:
            class_indices = [0, 1, 2]  # 기본적으로 클래스 0 (assault), 1 (falldown), 2 (normal)

        cams = []
        for idx in class_indices:
            weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)

            cam = torch.sum(weights * feature_maps, dim=1)
            cam = F.relu(cam)  # ReLU를 통해 음수를 제거

            # Heatmap을 이미지 크기에 맞춰서 resize (frame_count, 224, 224)
            cam = F.interpolate(cam.unsqueeze(1), size=(cam.size(2), 224, 224), mode='trilinear', align_corners=False)
            cam = cam.squeeze(1)

            # heatmap 정규화
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min)

            cams.append(cam)

        # (class_count, frame_count, 224, 224)로 변환
        cams = torch.stack(cams, dim=0)

        return cams  # (3, frame_count, 224, 224)


def HeatmapResNetD4C(depth=18, in_channels=32):
    if depth == 18:
        return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels)
    elif depth == 34:
        return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels)
    elif depth == 50:
        return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels)
    elif depth == 101:
        return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels)
    elif depth == 152:
        return ResNet3D(Bottleneck3D, [3, 8, 36, 3], in_channels)

