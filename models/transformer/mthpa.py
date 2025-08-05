import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from torch.utils.checkpoint import checkpoint as gradient_checkpoint


class PoseHeatmapEncoder(nn.Module):
    """Encode pose heatmaps with spatial awareness"""
    def __init__(self, embed_dim, patch_size=16):
        super().__init__()
        # Two-stage encoding for better feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),  # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.stage2 = nn.Conv2d(128, embed_dim, 4, stride=4)  # 56 -> 14
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # Normalize
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class SpatioTemporalAttention(nn.Module):
    """Balanced spatial-temporal attention"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        # Use fewer heads for efficiency
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads // 2, dropout=dropout, batch_first=True
        )
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads // 2, dropout=dropout, batch_first=True
        )
        
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Learnable weights for mixing
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x, mask=None):
        # x: (B, T, O, N, D)
        B, T, O, N, D = x.shape
        
        # 1. Spatial attention (per frame-object)
        x_spatial = rearrange(x, 'b t o n d -> (b t o) n d')
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = rearrange(x_spatial, '(b t o) n d -> b t o n d', b=B, t=T, o=O)
        
        # 2. Pool spatial for temporal attention
        x_pooled = x.mean(dim=3)  # (B, T, O, D)
        
        # 3. Temporal attention (per object)
        x_temporal = rearrange(x_pooled, 'b t o d -> (b o) t d')
        x_temporal = self.temporal_norm(x_temporal)
        
        if mask is not None:
            mask_temporal = rearrange(mask, 'b t o -> (b o) t')
            x_temporal, _ = self.temporal_attn(
                x_temporal, x_temporal, x_temporal,
                key_padding_mask=~mask_temporal
            )
        else:
            x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        
        x_temporal = rearrange(x_temporal, '(b o) t d -> b t o d', b=B, o=O)
        
        # 4. Mix spatial and temporal
        x_pooled = x_pooled + self.alpha * x_temporal
        x_out = x + x_pooled.unsqueeze(3)
        
        return x_out, x_pooled  # Return both for flexibility


class CrossPersonAttention(nn.Module):
    """Lightweight cross-person attention"""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Use standard multihead attention for stability
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        # x: (B, T, O, D)
        B, T, O, D = x.shape
        
        if O == 1:
            return x
        
        # Process each timestep
        x_out = []
        for t in range(T):
            x_t = x[:, t]  # (B, O, D)
            x_t_norm = self.norm(x_t)
            
            if mask is not None:
                mask_t = mask[:, t]  # (B, O)
                attn_out, _ = self.attn(
                    x_t_norm, x_t_norm, x_t_norm,
                    key_padding_mask=~mask_t
                )
            else:
                attn_out, _ = self.attn(x_t_norm, x_t_norm, x_t_norm)
            
            # Gated addition
            x_mean = x_t.mean(dim=1, keepdim=True).expand_as(x_t)
            gate_input = torch.cat([x_t, x_mean], dim=-1)
            gate = self.gate(gate_input)
            
            x_t = x_t + gate * attn_out
            x_out.append(x_t)
        
        x_out = torch.stack(x_out, dim=1)  # (B, T, O, D)
        return x_out


class MTHPABlock(nn.Module):
    """MTHPA block balancing efficiency and concept preservation"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1, use_cross_attn=True):
        super().__init__()
        
        # Spatial-temporal attention
        self.st_attn = SpatioTemporalAttention(embed_dim, num_heads, dropout)
        
        # Cross-person attention (can be disabled for efficiency)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = CrossPersonAttention(embed_dim, num_heads // 2, dropout)
        
        # FFN
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # x: (B, T, O, N, D) or (B, T, O, D) if pooled
        if x.dim() == 4:  # Already pooled
            B, T, O, D = x.shape
            
            # Cross-person attention
            if self.use_cross_attn and O > 1:
                x = x + self.cross_attn(x, mask)
            
            # FFN
            x = x + self.ffn(x)
            
            return x
        else:  # Full resolution
            B, T, O, N, D = x.shape
            
            # Spatial-temporal attention
            x, x_pooled = self.st_attn(x, mask)
            
            # Cross-person attention on pooled features
            if self.use_cross_attn and O > 1:
                x_pooled = x_pooled + self.cross_attn(x_pooled, mask)
                # Update full resolution
                x = x + (x_pooled.unsqueeze(3) - x.mean(dim=3, keepdim=True))
            
            # FFN (on pooled to save memory)
            x_pooled = x_pooled + self.ffn(x_pooled)
            
            # For next layer, return pooled
            return x_pooled


class MTHPA(nn.Module):
    def __init__(self,
                 num_frames=32,
                 in_channels=32,
                 embed_dim=384,
                 patch_size=(16, 16),
                 num_heads=8,
                 num_layers=12,
                 mlp_ratio=3.0,
                 dropout=0.1,
                 use_gradient_checkpointing=True):
        super().__init__()
        
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Pose encoder
        self.pose_encoder = PoseHeatmapEncoder(embed_dim, self.patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))  # 14x14
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.person_embed = nn.Parameter(torch.zeros(1, 1, in_channels, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MTHPABlock(
                embed_dim, num_heads, mlp_ratio, dropout,
                use_cross_attn=(i >= num_layers // 3)  # Cross-attn only in later layers
            )
            for i in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim // 4)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.person_embed, std=0.02)
        
    def forward(self, x, mask=None):
        # x: (B, O, T, H, W)
        B, O, T, H, W = x.shape
        
        # Handle mask
        if mask is None:
            mask = (x != -1).any(dim=[3, 4])  # (B, O, T)
            mask = mask.permute(0, 2, 1)  # (B, T, O)
        else:
            if mask.shape == (B, O, T):
                mask = mask.permute(0, 2, 1)
        
        # Prepare input
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, O, H, W)
        x = torch.where(x == -1, torch.zeros_like(x), x)
        
        # Encode
        x = rearrange(x, 'b t o h w -> (b t o) 1 h w')
        x = self.pose_encoder(x)  # (B*T*O, D, 14, 14)
        
        # Get dimensions
        _, D, H_feat, W_feat = x.shape
        N = H_feat * W_feat
        
        # Reshape and add position embeddings
        x = rearrange(x, 'bto d h w -> bto (h w) d')
        x = x + self.pos_embed[:, :N, :]
        x = rearrange(x, '(b t o) n d -> b t o n d', b=B, t=T, o=O)
        
        # Add temporal embeddings
        # self.temporal_embed shape: (1, num_frames, 1, embed_dim)
        # Need to expand to (B, T, O, N, D)
        temporal_embed = self.temporal_embed[:, :T, :, :]  # (1, T, 1, D)
        temporal_embed = temporal_embed.unsqueeze(2)  # (1, T, 1, 1, D)
        temporal_embed = temporal_embed.expand(B, T, O, N, D)
        x = x + temporal_embed
        
        # Add person embeddings  
        # self.person_embed shape: (1, 1, in_channels, embed_dim)
        # Need to expand to (B, T, O, N, D)
        person_embed = self.person_embed[:, :, :O, :]  # (1, 1, O, D)
        person_embed = person_embed.unsqueeze(3)  # (1, 1, O, 1, D)
        person_embed = person_embed.expand(B, T, O, N, D)
        x = x + person_embed
        
        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, N, D)
            x = x * mask_expanded.float()
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            if self.use_gradient_checkpointing and self.training and i % 2 == 0:
                x = gradient_checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        
        # Final pooling
        # x is already pooled to (B, T, O, D) after first block
        if x.dim() == 4:  # Already pooled
            if mask is not None:
                mask_float = mask.float()
                x_sum = (x * mask_float.unsqueeze(-1)).sum(dim=[1, 2])
                mask_count = mask_float.sum(dim=[1, 2], keepdim=True).clamp(min=1)
                x_global = x_sum / mask_count.squeeze(-1)
            else:
                x_global = x.mean(dim=[1, 2])
        else:  # Still has spatial dimension
            x_pooled = x.mean(dim=3)  # Pool spatial first
            if mask is not None:
                mask_float = mask.float()
                x_sum = (x_pooled * mask_float.unsqueeze(-1)).sum(dim=[1, 2])
                mask_count = mask_float.sum(dim=[1, 2], keepdim=True).clamp(min=1)
                x_global = x_sum / mask_count.squeeze(-1)
            else:
                x_global = x_pooled.mean(dim=[1, 2])
        
        # Output
        x_out = self.norm(x_global)
        x_out = self.head(x_out)
        
        return x_out


# Model configurations
def MTHPA_Tiny(num_frames=32, in_channels=32, patch_size=(16, 16)):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=256,
        patch_size=patch_size,
        num_heads=8,
        num_layers=6,
        mlp_ratio=2.5,
        use_gradient_checkpointing=True
    )

def MTHPA_Small(num_frames=32, in_channels=32, patch_size=(16, 16)):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=384,
        patch_size=patch_size,
        num_heads=8,
        num_layers=8,
        mlp_ratio=2.5,
        use_gradient_checkpointing=True
    )

def MTHPA_Base(num_frames=32, in_channels=32, patch_size=(16, 16)):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=512,  # Reduced for memory
        patch_size=patch_size,
        num_heads=8,
        num_layers=10,  # Reduced for speed
        mlp_ratio=3.0,
        use_gradient_checkpointing=True
    )

def MTHPA_Large(num_frames=32, in_channels=32, patch_size=(16, 16)):
    return MTHPA(
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim=768,  # Reduced from 1536
        patch_size=patch_size,
        num_heads=12,
        num_layers=12,  # Reduced from 16
        mlp_ratio=3.0,
        use_gradient_checkpointing=True
    )