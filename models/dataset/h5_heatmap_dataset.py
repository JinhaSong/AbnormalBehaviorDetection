import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.dataset = h5py.File(h5_file, 'r')
        self.keys = list(self.dataset.keys())
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.dataset[key][()]
        if self.transform:
            data = self.transform(data)
        return torch.tensor(data, dtype=torch.float32)

    def __del__(self):
        self.dataset.close()
