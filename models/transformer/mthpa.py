import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        chunk_size = min(64, seq_len)
        attn_output_chunks = []
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]
            
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if mask is not None:
                scores = scores.masked_fill(mask[:, i:end_i] == 0, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output_chunk = torch.matmul(attn_weights, v)
            attn_output_chunks.append(attn_output_chunk)
        
        attn_output = torch.cat(attn_output_chunks, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output), None


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        attn_output, _ = self.attention(x, mask)
        x = residual + self.dropout(attn_output)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(SpatialAttention, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        x = self.norm(x)
        attn_output, _ = self.attention(x, mask)
        x = residual + self.dropout(attn_output)
        return x


class ObjectPoseRelationAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, N_patches, O, D = x.shape
        x_flat = x.reshape(B * T * N_patches, O, D)
        x_norm = F.layer_norm(x_flat, (D,))
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = self.dropout(attn_out)
        attn_out = attn_out.view(B, T, N_patches, O, D)
        return attn_out


class CrossViewAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossViewAttention, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        residual = query
        query = self.norm(query)
        attn_output, _ = self.attention(query, mask)
        x = residual + self.dropout(attn_output)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        if x.dim() == 3:
            # (B*O*N, T, D): temporal attention만 적용
            x = self.temporal_attn(x, mask)
            return x
        elif x.dim() == 5:
            B, T, O, N, D = x.shape
            # Temporal attention
            x_temp = rearrange(x, 'b t o n d -> (b o n) t d')
            x_temp = self.temporal_attn(x_temp, mask)
            x = rearrange(x_temp, '(b o n) t d -> b t o n d', b=B, o=O, n=N)
            # Spatial attention
            x_spat = rearrange(x, 'b t o n d -> (b t o) n d')
            x_spat = self.spatial_attn(x_spat, mask)
            x = rearrange(x_spat, '(b t o) n d -> b t o n d', b=B, t=T, o=O)
            # MLP
            x_flat = rearrange(x, 'b t o n d -> (b t o n) d')
            x_flat = self.mlp(x_flat)
            x = rearrange(x_flat, '(b t o n) d -> b t o n d', b=B, t=T, o=O, n=N)
            return x
        else:
            raise ValueError(f"Unsupported input shape for TransformerBlock: {x.shape}")


class ObjectPoseRelationBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(ObjectPoseRelationBlock, self).__init__()
        self.object_pose_attn = ObjectPoseRelationAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        
    def forward(self, x, object_mask=None):
        x = self.object_pose_attn(x, object_mask)
        x = self.mlp(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        return x


class MTHPA(nn.Module):
    def __init__(self,
                 num_frames=64,
                 in_channels=5,
                 embed_dim=768,
                 patch_size=(2, 16, 16),
                 num_heads=12,
                 num_layers=12,
                 num_object_layers=4,
                 mlp_dim=None,
                 dropout=0.1):
        super(MTHPA, self).__init__()
        if mlp_dim is None:
            mlp_dim = embed_dim * 2
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, 1, 1, embed_dim))
        self.object_pos_embed = nn.Parameter(torch.randn(1, 1, in_channels, 1, embed_dim))
        self.spatial_pos_embed = None
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.object_pose_relation = ObjectPoseRelationAttention(embed_dim, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        B, T, O, H, W = x.shape
        x = x.view(B * T * O, 1, H, W)
        x = self.patch_embedding[0](x)
        BTO, embed_dim, H_p, W_p = x.shape
        N_patches = H_p * W_p

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, T, O, N_patches, embed_dim)

        temporal_pos_embed = self.temporal_pos_embed
        if temporal_pos_embed.shape[1] < T:
            temporal_pos_embed = temporal_pos_embed[:, :1, :, :, :].repeat(1, T, 1, 1, 1)
        else:
            temporal_pos_embed = temporal_pos_embed[:, :T, :, :, :]
        object_pos_embed = self.object_pos_embed
        if object_pos_embed.shape[2] < O:
            object_pos_embed = object_pos_embed[:, :, :1, :, :].repeat(1, 1, O, 1, 1)
        else:
            object_pos_embed = object_pos_embed[:, :, :O, :, :]
        
        spatial_pos_embed = nn.Parameter(torch.randn(1, 1, 1, N_patches, embed_dim)).to(x.device)
        x = x + temporal_pos_embed + object_pos_embed + spatial_pos_embed

        # 기존: x = rearrange(x, 'b t o n d -> (b o n) t d')
        for block in self.transformer_blocks:
            x = block(x)
        # 기존: x = rearrange(x, '(b o n) t d -> b t o n d', b=B, o=O, n=N_patches)

        x = x.permute(0, 1, 3, 2, 4)
        x = self.object_pose_relation(x)
        x = x.mean(dim=1)
        x = x.mean(dim=1)
        return x

    def save_gradients(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_feature_maps(self):
        return self.feature_maps
        
    def generate_cam(self, class_indices=None):
        if self.gradients is None or self.feature_maps is None:
            print("Gradients and feature maps are required for Grad-CAM")
            return None
            
        gradients = self.gradients
        feature_maps = self.feature_maps
        
        min_batch_size = min(gradients.size(0), feature_maps.size(0))
        gradients = gradients[:min_batch_size]
        feature_maps = feature_maps[:min_batch_size]
        
        if class_indices is None:
            class_indices = [0, 1]

        cams = []
        for idx in class_indices:
            weights = torch.mean(gradients, dim=1, keepdim=True)
            cam = torch.sum(weights * feature_maps, dim=2)
            cam = F.relu(cam)
            
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            
            cams.append(cam)
            
        return torch.stack(cams, dim=0)


def MTHPA_Tiny(num_frames=64, in_channels=5):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=192,
        patch_size=(16, 16),
        num_heads=3,
        num_layers=12,
        num_object_layers=3,
        mlp_dim=192*4
    )

def MTHPA_Small(num_frames=64, in_channels=5):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=384,
        patch_size=(16, 16),
        num_heads=6,
        num_layers=12,
        num_object_layers=6,
        mlp_dim=384*4
    )

def MTHPA_Base(num_frames=64, in_channels=5):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=96,
        patch_size=(16, 16),
        num_heads=3,
        num_layers=6,
        num_object_layers=3,
        mlp_dim=96*4
    )

def MTHPA_Large(num_frames=64, in_channels=5):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=1536,
        patch_size=(16, 16),
        num_heads=16,
        num_layers=24,
        num_object_layers=24,
        mlp_dim=1536*4
    )


