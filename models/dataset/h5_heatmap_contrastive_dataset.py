import h5py
import random
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize
import pandas as pd
import numpy as np


class HDF5ContrastiveDataset(Dataset):
    def __init__(self, h5_file, contrastive_pairs_csv, video_list_csv, max_length, transform=None, log=False, depth=32, cache_size=0):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth
        self.cache_size = cache_size
        
        # Read contrastive pairs CSV
        self.pairs_df = pd.read_csv(contrastive_pairs_csv)
        # Read video list CSV
        self.video_df = pd.read_csv(video_list_csv)
        
        # Optional caching for frequently accessed items
        self.cache = {} if cache_size > 0 else None
        self.cache_hits = 0
        self.cache_misses = 0
        
        if log:
            self.log_dataset_statistics()

    def log_dataset_statistics(self):
        num_positive = (self.pairs_df['label'] == 1).sum()
        num_negative = (self.pairs_df['label'] == 0).sum()
        print(f"Total pairs: {len(self.pairs_df)}")
        print(f"Positive pairs: {num_positive}")
        print(f"Negative pairs: {num_negative}")

    def __len__(self):
        return len(self.pairs_df)

    def pad_or_crop(self, tensor, target_length):
        current_length = tensor.shape[0]
        if current_length == target_length:
            return tensor
        elif current_length > target_length:
            # Random crop for training variety
            if self.training and current_length > target_length:
                start_idx = random.randint(0, current_length - target_length)
                return tensor[start_idx:start_idx + target_length]
            else:
                return tensor[:target_length]
        else:
            # Pad with zeros
            padding = (0, 0, 0, 0, 0, 0, 0, target_length - current_length)
            return F.pad(tensor, padding, 'constant', 0)

    def _load_from_h5(self, idx_str):
        """Load data from HDF5 file with optional caching"""
        if self.cache is not None and idx_str in self.cache:
            self.cache_hits += 1
            return self.cache[idx_str].clone()  # Return a copy to avoid modifications
        
        self.cache_misses += 1
        with h5py.File(self.h5_file, 'r') as file:
            data = torch.tensor(file['videos'][idx_str][:], dtype=torch.float32)
        
        # Update cache if enabled
        if self.cache is not None:
            if len(self.cache) >= self.cache_size:
                # Remove oldest item (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[idx_str] = data.clone()
        
        return data

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        anchor_idx = str(row['anchor_idx'])
        pair_idx = str(row['pair_idx'])
        label = row['label']
        
        # Load data with caching
        anchor = self._load_from_h5(anchor_idx)
        pair = self._load_from_h5(pair_idx)
        
        # Check depth
        if anchor.shape[1] != self.depth or pair.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: anchor {anchor.shape[1]}, pair {pair.shape[1]}")
        
        # Pad or crop to target length
        anchor = self.pad_or_crop(anchor, self.max_length)
        pair = self.pad_or_crop(pair, self.max_length)
        
        # (T, C, H, W) → (C, T, H, W) for easier processing
        anchor = anchor.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        pair = pair.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        
        # Apply transforms if specified
        if self.transform:
            anchor = torch.stack([self.transform(anchor[c]) for c in range(anchor.size(0))])
            pair = torch.stack([self.transform(pair[c]) for c in range(pair.size(0))])
        
        # Check for NaN values
        if torch.isnan(anchor).any() or torch.isnan(pair).any():
            print(f"NaN detected in sample pair {idx}")
            # Replace NaN with zeros
            anchor = torch.nan_to_num(anchor, nan=0.0)
            pair = torch.nan_to_num(pair, nan=0.0)
        
        return anchor, pair, torch.tensor(label, dtype=torch.float32)

    def get_cache_stats(self):
        """Get cache statistics"""
        if self.cache is not None:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
            return {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache)
            }
        return None


class SmallHDF5ContrastiveDataset(Dataset):
    def __init__(self, h5_file, contrastive_pairs_csv, video_list_csv, max_length, transform=None, subset_size=10, depth=32):
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth
        self.subset_size = subset_size
        
        # Read contrastive pairs CSV
        self.pairs_df = pd.read_csv(contrastive_pairs_csv)
        # Read video list CSV
        self.video_df = pd.read_csv(video_list_csv)
        
        # Split into positive and negative pairs
        positive_pairs = self.pairs_df[self.pairs_df['label'] == 1]
        negative_pairs = self.pairs_df[self.pairs_df['label'] == 0]
        
        # Sample balanced subset
        half = subset_size // 2
        pos_sampled = positive_pairs.sample(n=min(half, len(positive_pairs)), random_state=42)
        neg_sampled = negative_pairs.sample(n=min(subset_size - len(pos_sampled), len(negative_pairs)), random_state=42)
        
        self.sampled_pairs = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Subset size: {len(self.sampled_pairs)}")
        print(f"Positive pairs: {len(pos_sampled)}")
        print(f"Negative pairs: {len(neg_sampled)}")

    def __len__(self):
        return len(self.sampled_pairs)

    def pad_or_crop(self, tensor, target_length):
        current_length = tensor.shape[0]
        if current_length == target_length:
            return tensor
        elif current_length > target_length:
            return tensor[:target_length]
        else:
            padding = (0, 0, 0, 0, 0, 0, 0, target_length - current_length)
            return F.pad(tensor, padding, 'constant', 0)

    def __getitem__(self, idx):
        row = self.sampled_pairs.iloc[idx]
        anchor_idx = str(row['anchor_idx'])
        pair_idx = str(row['pair_idx'])
        label = row['label']
        
        with h5py.File(self.h5_file, 'r') as file:
            anchor = torch.tensor(file['videos'][anchor_idx][:], dtype=torch.float32)
            pair = torch.tensor(file['videos'][pair_idx][:], dtype=torch.float32)
        
        # Check depth
        if anchor.shape[1] != self.depth or pair.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: anchor {anchor.shape[1]}, pair {pair.shape[1]}")
        
        anchor = self.pad_or_crop(anchor, self.max_length)
        pair = self.pad_or_crop(pair, self.max_length)
        
        # (T, C, H, W) → (C, T, H, W)
        anchor = anchor.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        pair = pair.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        
        if self.transform:
            anchor = torch.stack([self.transform(anchor[c]) for c in range(anchor.size(0))])
            pair = torch.stack([self.transform(pair[c]) for c in range(pair.size(0))])
        
        if torch.isnan(anchor).any() or torch.isnan(pair).any():
            print(f"NaN detected in sample pair {idx}")
            anchor = torch.nan_to_num(anchor, nan=0.0)
            pair = torch.nan_to_num(pair, nan=0.0)
        
        return anchor, pair, torch.tensor(label, dtype=torch.float32)


def get_transform():
    """Get data transformation pipeline"""
    return Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])


def contrastive_collate_fn(batch):
    """Custom collate function for contrastive learning"""
    anchors, pairs, labels = zip(*batch)
    
    # Stack tensors
    anchors = torch.stack(anchors)
    pairs = torch.stack(pairs)
    labels = torch.tensor(labels)
    
    # Memory optimization: ensure contiguous memory layout
    anchors = anchors.contiguous()
    pairs = pairs.contiguous()
    labels = labels.contiguous()
    
    return anchors, pairs, labels


def triplet_collate_fn(batch):
    """Custom collate function for triplet learning"""
    anchors, positives, negatives, is_normals = zip(*batch)
    return (
        torch.stack(anchors).contiguous(),
        torch.stack(positives).contiguous(),
        torch.stack(negatives).contiguous(),
        torch.tensor(is_normals).contiguous()
    )


def collate_fn(batch):
    """Custom collate function for single sample"""
    anchors, is_normals = zip(*batch)
    return (
        torch.stack(anchors).contiguous(),
        torch.tensor(is_normals).contiguous()
    )


# Memory-efficient data loader wrapper
class MemoryEfficientDataLoader:
    """Wrapper for DataLoader with memory optimization features"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=2, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Optimize worker settings
        if num_workers > 0:
            kwargs['persistent_workers'] = True
            kwargs['prefetch_factor'] = 2
        
        # Create data loader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=contrastive_collate_fn,
            **kwargs
        )
        
        self.epoch_count = 0
    
    def __iter__(self):
        self.epoch_count += 1
        
        # Periodic garbage collection
        if self.epoch_count % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)