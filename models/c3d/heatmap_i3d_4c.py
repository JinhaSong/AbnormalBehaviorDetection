import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV1Module3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionV1Module3D, self).__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV1I3D(nn.Module):
    def __init__(self, in_channels):
        super(InceptionV1I3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 192, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(192)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionV1Module3D(192, 64)
        self.inception3b = InceptionV1Module3D(256, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionV1Module3D(512, 128)
        self.inception4b = InceptionV1Module3D(512, 128)
        self.inception4c = InceptionV1Module3D(512, 128)
        self.inception4d = InceptionV1Module3D(512, 128)
        self.inception4e = InceptionV1Module3D(512, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.inception5a = InceptionV1Module3D(1024, 256)
        self.inception5b = InceptionV1Module3D(1024, 256)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

class InceptionV3Module3D(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels3_reduce, out_channels3, out_channels5_reduce, out_channels5, out_channels_pool):
        super(InceptionV3Module3D, self).__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels3_reduce, kernel_size=1),
            nn.Conv3d(out_channels3_reduce, out_channels3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels5_reduce, kernel_size=1),
            nn.Conv3d(out_channels5_reduce, out_channels5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionV3I3D(nn.Module):
    def __init__(self, in_channels):
        super(InceptionV3I3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv3d(64, 80, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm3d(80)
        self.conv5 = nn.Conv3d(80, 192, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm3d(192)
        self.pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionV3Module3D(192, 64, 48, 64, 64, 96, 32)
        self.inception3b = InceptionV3Module3D(256, 64, 48, 64, 64, 96, 64)
        self.inception3c = InceptionV3Module3D(320, 64, 48, 64, 64, 96, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionV3Module3D(320, 128, 128, 192, 32, 96, 64)
        self.inception4b = InceptionV3Module3D(576, 128, 128, 192, 32, 96, 64)
        self.inception4c = InceptionV3Module3D(576, 128, 128, 192, 32, 96, 64)
        self.inception4d = InceptionV3Module3D(576, 128, 128, 192, 32, 96, 64)
        self.inception4e = InceptionV3Module3D(576, 128, 128, 192, 32, 96, 64)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.inception5a = InceptionV3Module3D(576, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionV3Module3D(832, 256, 160, 320, 32, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def HeatmapI3D(version="v1", in_channels=32):
    if version == "v1":
        return InceptionV1I3D(in_channels=in_channels)
    else:
        return InceptionV3I3D(in_channels=in_channels)