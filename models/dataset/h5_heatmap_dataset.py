import json
import h5py
import torch
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize


class HDF5ShanghaiTechDataset(Dataset):
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

def get_transform():
    return Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])
