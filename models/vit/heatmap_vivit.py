import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = attn_output + x
        x = x + self.mlp(self.norm2(x))
        return x


class HeatmapViViT(nn.Module):
    def __init__(self, num_frames, in_channels, embed_dim, patch_size=(2, 16, 16)):
        super(HeatmapViViT, self).__init__()
        if embed_dim == 384:
            depth = 12
            num_heads = 6
            mlp_dim = embed_dim * 4
        elif embed_dim == 768:
            depth = 12
            num_heads = 12
            mlp_dim = embed_dim * 4
        elif embed_dim == 1024:
            depth = 24
            num_heads = 16
            mlp_dim = embed_dim * 4
        else:
            raise ValueError("Unsupported embed_dim. Choose from 384, 768, or 1024.")

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, (num_frames // patch_size[0]) * (224 // patch_size[1]) * (224 // patch_size[2]) + 1, embed_dim))  # Positional Embedding
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)

        # Class Token 추가
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=x.size(0))
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding 추가
        x += self.pos_embed

        # Transformer Blocks
        for blk in self.transformer_blocks:
            x = blk(x)

        # feature_maps 저장 및 requires_grad 설정
        x.requires_grad_(True)  # Grad-CAM을 위해 requires_grad 설정
        self.feature_maps = x.detach()  # feature map 저장
        x.register_hook(self.save_gradients)  # backward에서 gradients 저장

        # Layer Normalization
        x = self.norm(x)

        # Class Token을 통한 최종 feature
        cls_token_final = x[:, 0]  # 첫 번째 토큰 (CLS token)만 사용
        return cls_token_final

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
            class_indices = [0, 1, 2]

        cams = []
        for idx in (class_indices if class_indices is not None else range(feature_maps.size(1))):
            weights = torch.mean(gradients, dim=1, keepdim=True)

            cam = torch.sum(weights * feature_maps, dim=2)  # (batch, num_tokens)
            cam = F.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min)
            cams.append(cam)

        cams = torch.stack(cams, dim=0)

        return cams