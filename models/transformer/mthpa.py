import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class MemoryEfficientMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Fused QKV projection for memory efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Single projection for Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, L, D
        q, k, v = qkv.unbind(0)  # Each is B, H, L, D
        
        # Use PyTorch's efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use memory-efficient attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale
            )
        else:
            # Fallback to chunked attention for long sequences
            if L > 1024:
                attn_output = self._chunked_attention(q, k, v, mask)
            else:
                attn_output = self._standard_attention(q, k, v, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output), None
    
    def _standard_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v)
    
    def _chunked_attention(self, q, k, v, mask=None, chunk_size=512):
        B, H, L, D = q.shape
        attn_chunks = []
        
        for i in range(0, L, chunk_size):
            end_i = min(i + chunk_size, L)
            q_chunk = q[:, :, i:end_i]
            
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                if mask.dim() == 2:
                    scores = scores.masked_fill(mask[:, i:end_i, None, :] == 0, -1e4)
            
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
            attn_weights = self.dropout(attn_weights)
            
            attn_chunk = torch.matmul(attn_weights, v)
            attn_chunks.append(attn_chunk)
        
        return torch.cat(attn_chunks, dim=2)


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MemoryEfficientMultiHeadAttention(embed_dim, num_heads, dropout)
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
        super().__init__()
        self.attention = MemoryEfficientMultiHeadAttention(embed_dim, num_heads, dropout)
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
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, N_patches, O, D)
        B, T, N_patches, O, D = x.shape
        
        # Process in chunks to save memory
        chunk_size = 8  # Process 8 time steps at once
        outputs = []
        
        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            x_chunk = x[:, t_start:t_end]  # (B, chunk, N, O, D)
            
            # Reshape for attention
            x_flat = x_chunk.reshape(-1, O, D)  # (B*chunk*N, O, D)
            x_norm = self.norm(x_flat)
            
            # Apply attention
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            attn_out = self.dropout(attn_out)
            
            # Reshape back
            attn_out = attn_out.view(B, t_end - t_start, N_patches, O, D)
            outputs.append(attn_out)
        
        return torch.cat(outputs, dim=1)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def _forward_impl(self, x, mask=None):
        if x.dim() == 3:
            # (B*O*N, T, D): temporal attention only
            x = self.temporal_attn(x, mask)
            x = self.mlp(x)
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
    
    def forward(self, x, mask=None):
        if self.use_checkpoint and self.training and x.requires_grad:
            return gradient_checkpoint(self._forward_impl, x, mask, use_reentrant=False)
        else:
            return self._forward_impl(x, mask)


class MTHPA(nn.Module):
    def __init__(self,
                 num_frames=64,
                 in_channels=32,
                 embed_dim=768,
                 patch_size=(16, 16),
                 num_heads=12,
                 num_layers=12,
                 num_object_layers=4,
                 mlp_dim=None,
                 dropout=0.1,
                 use_gradient_checkpointing=False):
        super(MTHPA, self).__init__()
        if mlp_dim is None:
            mlp_dim = embed_dim * 4
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        
        # Position embeddings
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, 1, 1, embed_dim))
        self.object_pos_embed = nn.Parameter(torch.randn(1, 1, in_channels, 1, embed_dim))
        self.spatial_pos_embed = None  # Will be created dynamically
        
        # Transformer blocks with optional gradient checkpointing
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_dim, dropout,
                use_checkpoint=(use_gradient_checkpointing and i % 2 == 0)
            )
            for i in range(num_layers)
        ])
        
        # Object pose relation attention
        self.object_pose_relation = ObjectPoseRelationAttention(embed_dim, dropout=dropout)
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For Grad-CAM compatibility
        self.feature_maps = None
        self.gradients = None

    def forward(self, x):
        if x.shape[1] == 32 and x.shape[-1] == 16:
            x = x.permute(0, 4, 1, 2, 3)
        B, T, O, H, W = x.shape
        
        x_reshaped = x.reshape(B * T * O, 1, H, W)
        x = self.patch_embedding[0](x_reshaped)

        x = x.view(B, T, O, self.embed_dim, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 2, 4, 5, 3)
        N_patches = x.shape[3] * x.shape[4]
        x = x.reshape(B, T, O, N_patches, self.embed_dim)
        
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
        
        if self.spatial_pos_embed is None or self.spatial_pos_embed.shape[3] != N_patches:
            self.spatial_pos_embed = nn.Parameter(
                torch.randn(1, 1, 1, N_patches, self.embed_dim)
            ).to(x.device)
        
        x = x + temporal_pos_embed + object_pos_embed + self.spatial_pos_embed

        for block in self.transformer_blocks:
            x = block(x)

        x = x.permute(0, 1, 3, 2, 4)
        x = self.object_pose_relation(x)
        
        x = x.mean(dim=1)
        x = x.mean(dim=1)
        x = x.mean(dim=1)
        
        return x

    def save_gradients(self, grad):
        self.gradients = grad

    def get_gradients(self):
        return self.gradients

    def get_feature_maps(self):
        return self.feature_maps


def MTHPA_Tiny(num_frames=64, in_channels=32):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=192,
        patch_size=(32, 32),  # Larger patch size for memory efficiency
        num_heads=3,
        num_layers=12,
        num_object_layers=3,
        mlp_dim=192*4,
        use_gradient_checkpointing=True
    )

def MTHPA_Small(num_frames=64, in_channels=32):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=384,
        patch_size=(32, 32),  # Larger patch size for memory efficiency
        num_heads=6,
        num_layers=12,
        num_object_layers=6,
        mlp_dim=384*4,
        use_gradient_checkpointing=True
    )

def MTHPA_Base(num_frames=64, in_channels=32):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=768,
        patch_size=(32, 32),  # Larger patch size for memory efficiency
        num_heads=12,
        num_layers=12,
        num_object_layers=6,
        mlp_dim=768*4,
        use_gradient_checkpointing=True
    )

def MTHPA_Large(num_frames=64, in_channels=32):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=1536,
        patch_size=(56, 56),  # Even larger patch size for Large model
        num_heads=16,
        num_layers=24,
        num_object_layers=24,
        mlp_dim=1536*4,
        use_gradient_checkpointing=True
    )