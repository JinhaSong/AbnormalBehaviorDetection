import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV1Module3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionV1Module3D, self).__init__()
        self.branch1 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv3d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1)
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

        self.dropout = nn.Dropout3d(p=0.3)

        self.inception3a = InceptionV1Module3D(192, 256)
        self.inception3b = InceptionV1Module3D(256, 512)
        self.pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionV1Module3D(512, 512)
        self.inception4b = InceptionV1Module3D(512, 512)
        self.inception4c = InceptionV1Module3D(512, 512)
        self.inception4d = InceptionV1Module3D(512, 512)
        self.inception4e = InceptionV1Module3D(512, 1024)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)  # 마지막 차원에 대한 pooling 생략

        self.inception5a = InceptionV1Module3D(1024, 1024)
        self.inception5b = InceptionV1Module3D(1024, 1024)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        def save_feature_map_hook(module, input, output):
            if not output.requires_grad:
                output = output.detach().clone().requires_grad_(True)

            self.feature_maps = output
            output.register_hook(self.save_gradients)

        self.inception5b.register_forward_hook(save_feature_map_hook)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

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
        x = torch.flatten(x, 1)

        return x

    # backward hook을 사용하여 gradient 저장
    def save_gradients(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_feature_maps(self):
        return self.feature_maps

    # Grad-CAM heatmap 생성 함수
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


def HeatmapInceptionV1I3D(in_channels=32):
    return InceptionV1I3D(in_channels)



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