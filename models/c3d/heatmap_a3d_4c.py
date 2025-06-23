import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(in_channels // 2)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)  # ReLU를 일관성 있게 정의

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = torch.sigmoid(self.bn2(self.conv2(x)))
        return x * residual


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)  # ReLU 통일

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)  # ReLU 통일

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class A3D(nn.Module):
    def __init__(self, block, layers, in_channels=32):
        super(A3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.attention = AttentionBlock3D(2048)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout3d(p=0.3)

        # Grad-CAM을 위한 변수
        self.feature_maps = None
        self.gradients = None

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Grad-CAM을 위한 hook을 설정
        def save_feature_map_hook(module, input, output):
            # output이 requires_grad를 갖도록 설정
            if not output.requires_grad:
                output = output.clone().detach().requires_grad_(True)
            self.feature_maps = output
            output.register_hook(self.save_gradients)

        # 마지막 레이어에서 Grad-CAM을 위해 feature maps를 저장
        self.layer4.register_forward_hook(save_feature_map_hook)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    # backward hook을 사용하여 gradient 저장
    def save_gradients(self, grad):
        self.gradients = grad

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

        cams = torch.stack(cams, dim=0)

        return cams  # (3, frame_count, 224, 224)


def HeatmapA3D(in_channels=32, depth=18):
    if depth == 18:
        return A3D(BasicBlock3D, [2, 2, 2, 2], in_channels)
    elif depth == 34:
        return A3D(BasicBlock3D, [3, 4, 6, 3], in_channels)
    elif depth == 50:
        return A3D(Bottleneck3D, [3, 4, 6, 3], in_channels)
    elif depth == 101:
        return A3D(Bottleneck3D, [3, 4, 23, 3], in_channels)
    elif depth == 152:
        return A3D(Bottleneck3D, [3, 8, 36, 3], in_channels)
