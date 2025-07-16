import h5py
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize
import pandas as pd


class HDF5ContrastiveDataset(Dataset):
    def __init__(self, h5_file, contrastive_pairs_csv, video_list_csv, max_length, transform=None, log=False, depth=32):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth
        # Read contrastive pairs CSV
        self.pairs_df = pd.read_csv(contrastive_pairs_csv)
        # Read video list CSV
        self.video_df = pd.read_csv(video_list_csv)
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
            return tensor[:target_length]
        else:
            padding = (0, 0, 0, 0, 0, 0, 0, target_length - current_length)
            return F.pad(tensor, padding, 'constant', 0)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
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
        return anchor, pair, torch.tensor(label, dtype=torch.float32)


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
        return anchor, pair, torch.tensor(label, dtype=torch.float32)


def get_transform():
    return Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])


def triplet_collate_fn(batch):
    anchors, positives, negatives, is_normals = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        torch.tensor(is_normals)
    )

def contrastive_collate_fn(batch):
    anchors, pairs, labels = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(pairs),
        torch.tensor(labels)
    )

def collate_fn(batch):
    anchors, is_normals = zip(*batch)
    return (
        torch.stack(anchors),
        torch.tensor(is_normals)
    )