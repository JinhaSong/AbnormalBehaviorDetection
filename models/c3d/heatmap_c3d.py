import torch
import torch.nn as nn
import pytorch_lightning as pl

class BasicBlock3D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class HeatmapC3D(nn.Module):
    def __init__(self, T):
        super(HeatmapC3D, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(BasicBlock3D, 64, stride=2)
        self.layer2 = self._make_layer(BasicBlock3D, 128, stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, 256, stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, stride):
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes
        layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        return out


def create_heatmap_c3d_model(T):
    return HeatmapC3D(T)