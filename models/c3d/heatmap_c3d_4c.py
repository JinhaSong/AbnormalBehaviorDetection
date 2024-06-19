import torch
import torch.nn as nn
import torch.nn.functional as F

from models.c3d.module.basicblock3d import BasicBlock3D
from models.c3d.module.bottleneck3d import Bottleneck3D


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=32, num_classes=512):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 1, 1, -1)
        return x


def HeatmapC3D4C(depth=18, in_channels=32, num_classes=512):
    if depth == 18:
        return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels, num_classes)
    elif depth == 34:
        return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels, num_classes)
    elif depth == 50:
        return ResNet3D(Bottleneck3D, [3, 4, 6, 3], in_channels, num_classes)
    elif depth == 101:
        return ResNet3D(Bottleneck3D, [3, 4, 23, 3], in_channels, num_classes)
    elif depth == 152:
        return ResNet3D(Bottleneck3D, [3, 8, 36, 3], in_channels, num_classes)
