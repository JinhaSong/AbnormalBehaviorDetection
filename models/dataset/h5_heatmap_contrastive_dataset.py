import h5py
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize


class HDF5ContrastiveDataset(Dataset):
    def __init__(self, h5_file, max_length, transform=None, log=False, depth=32):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth

        with h5py.File(self.h5_file, 'r') as file:
            self.contrastive_pairs = file['contrastive_pairs'][()]  # (index1, index2, label)
            self.normal_indices = file['normal_indices'][()]
            self.abnormal_indices = file['abnormal_indices'][()]

        if log:
            self.log_dataset_statistics()

    def log_dataset_statistics(self):
        num_positive = sum(1 for pair in self.contrastive_pairs if pair[2] == 1)
        num_negative = len(self.contrastive_pairs) - num_positive
        print(f"Total pairs: {len(self.contrastive_pairs)}")
        print(f"Positive pairs: {num_positive}")
        print(f"Negative pairs: {num_negative}")

    def __len__(self):
        return len(self.contrastive_pairs)

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
        with h5py.File(self.h5_file, 'r') as file:
            idx1, idx2, label = self.contrastive_pairs[idx]

            sample1 = torch.tensor(file['videos'][str(idx1)][:], dtype=torch.float32)
            sample2 = torch.tensor(file['videos'][str(idx2)][:], dtype=torch.float32)

        # Check depth
        if sample1.shape[1] != self.depth or sample2.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: sample1 {sample1.shape[1]}, sample2 {sample2.shape[1]}")

        sample1 = self.pad_or_crop(sample1, self.max_length)
        sample2 = self.pad_or_crop(sample2, self.max_length)

        # (T, C, H, W) → (C, T, H, W)
        sample1 = sample1.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        sample2 = sample2.transpose(0, 1).transpose(1, 2).transpose(2, 3)

        if self.transform:
            sample1 = torch.stack([self.transform(sample1[c]) for c in range(sample1.size(0))])
            sample2 = torch.stack([self.transform(sample2[c]) for c in range(sample2.size(0))])

        if torch.isnan(sample1).any() or torch.isnan(sample2).any():
            print(f"NaN detected in sample pair {idx}")

        return sample1, sample2, torch.tensor(label, dtype=torch.float32)


class SmallHDF5ContrastiveDataset(Dataset):
    def __init__(self, h5_file, max_length, transform=None, subset_size=10, depth=32):
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth
        self.subset_size = subset_size

        with h5py.File(self.h5_file, 'r') as file:
            self.all_pairs = file['contrastive_pairs'][()]
            self.normal_indices = file['normal_indices'][()]
            self.abnormal_indices = file['abnormal_indices'][()]

        # Split into positive and negative pairs
        positive_pairs = [pair for pair in self.all_pairs if pair[2] == 1]
        negative_pairs = [pair for pair in self.all_pairs if pair[2] == 0]

        # Sample equally from both classes if possible
        half = subset_size // 2
        pos_sampled = random.sample(positive_pairs, min(half, len(positive_pairs)))
        neg_sampled = random.sample(negative_pairs, min(subset_size - len(pos_sampled), len(negative_pairs)))

        # Final sampled subset
        self.sampled_pairs = pos_sampled + neg_sampled
        self.sampled_pairs = self.sampled_pairs[:subset_size]  # ensure exact subset size

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
        with h5py.File(self.h5_file, 'r') as file:
            idx1, idx2, label = self.sampled_pairs[idx]

            sample1 = torch.tensor(file['videos'][str(idx1)][:], dtype=torch.float32)
            sample2 = torch.tensor(file['videos'][str(idx2)][:], dtype=torch.float32)

        # Depth check
        if sample1.shape[1] != self.depth or sample2.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: sample1 {sample1.shape[1]}, sample2 {sample2.shape[1]}")

        sample1 = self.pad_or_crop(sample1, self.max_length)
        sample2 = self.pad_or_crop(sample2, self.max_length)

        sample1 = sample1.transpose(0, 1).transpose(1, 2).transpose(2, 3)
        sample2 = sample2.transpose(0, 1).transpose(1, 2).transpose(2, 3)

        if self.transform:
            sample1 = torch.stack([self.transform(sample1[c]) for c in range(sample1.size(0))])
            sample2 = torch.stack([self.transform(sample2[c]) for c in range(sample2.size(0))])

        if torch.isnan(sample1).any() or torch.isnan(sample2).any():
            print(f"NaN detected in sample pair {idx}")

        return sample1, sample2, torch.tensor(label, dtype=torch.float32)


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

def collate_fn(batch):
    anchors, is_normals = zip(*batch)
    return (
        torch.stack(anchors),
        torch.tensor(is_normals)
    )