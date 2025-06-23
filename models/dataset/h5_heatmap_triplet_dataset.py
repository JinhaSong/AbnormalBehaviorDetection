import h5py
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize


class HDF5AnchorDataset(Dataset):
    def __init__(self, h5_file, max_length, transform=None, log=False, depth=32):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth

        with h5py.File(self.h5_file, 'r') as file:
            anchor_indices = file['triplet_info'][:, 0]  # Only use anchor indices
            self.anchor_info = list(set(anchor_indices))  # Remove duplicates
            self.normal_indices = file['normal_indices'][()]
            self.abnormal_indices = file['abnormal_indices'][()]

        if log:
            self.log_dataset_statistics()

    def log_dataset_statistics(self):
        normal_anchors = sum(1 for idx in self.anchor_info if idx in self.normal_indices)
        abnormal_anchors = len(self.anchor_info) - normal_anchors
        print(f"Total anchors: {len(self.anchor_info)}")
        print(f"Normal anchors: {normal_anchors}")
        print(f"Abnormal anchors: {abnormal_anchors}")

    def __len__(self):
        return len(self.anchor_info)

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
            anchor_idx = self.anchor_info[idx]
            anchor = torch.tensor(file['videos'][str(anchor_idx)][:], dtype=torch.float32)

        # Check depth
        if anchor.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: anchor depth {anchor.shape[1]}")

        anchor = self.pad_or_crop(anchor, self.max_length)
        anchor = anchor.transpose(0, 1).transpose(1, 2).transpose(2, 3)  # (C, T, H, W)

        if self.transform:
            anchor = torch.stack([self.transform(anchor[c]) for c in range(anchor.size(0))])

        if torch.isnan(anchor).any():
            # Handle NaN values
            anchor = torch.nan_to_num(anchor)
            print(f"NaN detected and replaced in sample {idx}")

        is_normal = 1 if anchor_idx in self.normal_indices else 0

        return anchor, torch.tensor([is_normal], dtype=torch.float32)


class HDF5TripletDataset(Dataset):
    def __init__(self, h5_file, max_length, transform=None, log=False, depth=32):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length
        self.depth = depth

        with h5py.File(self.h5_file, 'r') as file:
            self.triplet_info = file['triplet_info'][()]
            self.normal_indices = file['normal_indices'][()]
            self.abnormal_indices = file['abnormal_indices'][()]

        if log:
            self.log_dataset_statistics()

    def log_dataset_statistics(self):
        normal_triplets = sum(1 for t in self.triplet_info if t[0] in self.normal_indices)
        abnormal_triplets = len(self.triplet_info) - normal_triplets
        print(f"Total triplets: {len(self.triplet_info)}")
        print(f"Normal triplets: {normal_triplets}")
        print(f"Abnormal triplets: {abnormal_triplets}")

    def __len__(self):
        return len(self.triplet_info)

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
            anchor_idx, pos_idx, neg_idx = self.triplet_info[idx]

            anchor = torch.tensor(file['videos'][str(anchor_idx)][:], dtype=torch.float32)
            positive = torch.tensor(file['videos'][str(pos_idx)][:], dtype=torch.float32)
            negative = torch.tensor(file['videos'][str(neg_idx)][:], dtype=torch.float32)

        # Check depth
        if anchor.shape[1] != self.depth or positive.shape[1] != self.depth or negative.shape[1] != self.depth:
            raise ValueError(f"Depth mismatch at index {idx}: anchor depth {anchor.shape[1]}, positive depth {positive.shape[1]}, negative depth {negative.shape[1]}")

        anchor = self.pad_or_crop(anchor, self.max_length)
        positive = self.pad_or_crop(positive, self.max_length)
        negative = self.pad_or_crop(negative, self.max_length)

        anchor = anchor.transpose(0, 1).transpose(1, 2).transpose(2, 3)  # (C, T, H, W)
        positive = positive.transpose(0, 1).transpose(1, 2).transpose(2, 3)  # (C, T, H, W)
        negative = negative.transpose(0, 1).transpose(1, 2).transpose(2, 3)  # (C, T, H, W)

        if self.transform:
            anchor = torch.stack([self.transform(anchor[c]) for c in range(anchor.size(0))])
            positive = torch.stack([self.transform(positive[c]) for c in range(positive.size(0))])
            negative = torch.stack([self.transform(negative[c]) for c in range(negative.size(0))])

        if torch.isnan(anchor).any() or torch.isnan(positive).any() or torch.isnan(negative).any():
            print(f"NaN detected in sample {idx}")

        is_normal = 1 if anchor_idx in self.normal_indices else 0

        return anchor, positive, negative, torch.tensor([is_normal], dtype=torch.float32)


class SmallHDF5TripletDataset(HDF5TripletDataset):
    def __init__(self, h5_file, max_length, transform=None, subset_size=10):
        super().__init__(h5_file, max_length, transform, False)
        self.subset_size = subset_size

        # Determine the number of samples to draw from normal and abnormal indices
        half_size = subset_size // 2
        normal_sample_size = min(half_size, len(self.normal_indices))
        abnormal_sample_size = min(half_size, len(self.abnormal_indices))

        # Ensure that we have enough samples, adjust if necessary
        total_available_samples = len(self.normal_indices) + len(self.abnormal_indices)
        if subset_size > total_available_samples:
            subset_size = total_available_samples
            normal_sample_size = min(len(self.normal_indices), subset_size // 2)
            abnormal_sample_size = subset_size - normal_sample_size

        # Sample indices
        sampled_normal_indices = random.sample(list(self.normal_indices), normal_sample_size)
        sampled_abnormal_indices = random.sample(list(self.abnormal_indices), abnormal_sample_size)

        # Create a subset of triplet_info using sampled indices
        self.sampled_triplet_info = []
        normal_triplets = [triplet for triplet in self.triplet_info if triplet[0] in sampled_normal_indices]
        abnormal_triplets = [triplet for triplet in self.triplet_info if triplet[0] in sampled_abnormal_indices]

        self.sampled_triplet_info.extend(normal_triplets[:normal_sample_size])
        self.sampled_triplet_info.extend(abnormal_triplets[:abnormal_sample_size])

        # Ensure the final subset has exactly `subset_size` elements
        self.sampled_triplet_info = self.sampled_triplet_info[:subset_size]

        self.triplet_info = self.sampled_triplet_info

        # Count normal and abnormal in the subset
        self.sampled_normal_triplets = [triplet for triplet in self.triplet_info if triplet[0] in self.normal_indices]
        self.sampled_abnormal_triplets = [triplet for triplet in self.triplet_info if triplet[0] in self.abnormal_indices]

        print(f"Subset size: {self.subset_size}")
        print(f"Number of normal triplets in subset: {len(self.sampled_normal_triplets)}")
        print(f"Number of abnormal triplets in subset: {len(self.sampled_abnormal_triplets)}")

    def __len__(self):
        return len(self.triplet_info)


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