import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize


class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.dataset_keys = []
        with h5py.File(self.h5_file, 'r') as file:
            self.dataset_keys = list(file.keys())

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as file:
            video_data = file[self.dataset_keys[idx]]['video_data'][()]
            label = file[self.dataset_keys[idx]]['label'][()]

        video_data = video_data.transpose((1, 0, 2, 3))
        video_data = torch.tensor(video_data, dtype=torch.float32)

        if self.transform:
            video_data = torch.stack([self.transform(video_data[c]) for c in range(video_data.size(0))])

        label = torch.tensor(label, dtype=torch.float32)

        return video_data, label


class HDF5TripletDataset(Dataset):
    def __init__(self, h5_file, max_length, transform=None):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_length = max_length

        with h5py.File(self.h5_file, 'r') as file:
            self.triplet_info = file['triplet_info'][()]
            self.videos = [file['videos'][key][:] for key in file['videos']]
            self.normal_indices = file['normal_indices'][()]
            self.abnormal_indices = file['abnormal_indices'][()]

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
        anchor_idx, pos_idx, neg_idx = self.triplet_info[idx]

        anchor = torch.tensor(self.videos[anchor_idx], dtype=torch.float32)
        positive = torch.tensor(self.videos[pos_idx], dtype=torch.float32)
        negative = torch.tensor(self.videos[neg_idx], dtype=torch.float32)

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

        return anchor, positive, negative, torch.tensor([1 if idx in self.normal_indices else 0])


class SmallHDF5TripletDataset(HDF5TripletDataset):
    def __init__(self, h5_file, max_length, transform=None, subset_size=10):
        super().__init__(h5_file, max_length, transform)
        self.subset_size = subset_size
        self.triplet_info = self.triplet_info[:self.subset_size]  # Only keep a subset of the triplet info

    def __len__(self):
        return len(self.triplet_info)


def get_transform():
    return Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])


def collate_fn(batch):
    anchors, positives, negatives, is_normals = zip(*batch)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        torch.tensor(is_normals)
    )