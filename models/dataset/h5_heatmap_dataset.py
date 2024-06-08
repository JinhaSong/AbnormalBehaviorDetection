import json
import h5py
import torch
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import Compose, ToTensor, Normalize


class HDF5TrainDataset(Dataset):
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

        video_data = video_data.transpose((1, 0, 2, 3))
        video_data = torch.tensor(video_data, dtype=torch.float32)

        if self.transform:
            video_data = torch.stack([self.transform(video_data[c]) for c in range(video_data.size(0))])

        return video_data


class HDF5TestDataset(Dataset):
    def __init__(self, h5_file, transform=None, max_frames=128):
        super().__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.max_frames = max_frames
        self.dataset_keys = []

        with h5py.File(self.h5_file, 'r') as file:
            self.dataset_keys = list(file.keys())

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        video_key = self.dataset_keys[idx]

        with h5py.File(self.h5_file, 'r') as file:
            video_data = file[video_key]['video_data'][()]
            frame_numbers = file[video_key]['frame_numbers'][()]
            label = file[video_key]['label'][()]
            annotation = json.loads(file[video_key]['annotation'][()])

        video_data = video_data.transpose((1, 0, 2, 3))
        video_data = torch.tensor(video_data, dtype=torch.float32)

        # Pad video data to max_frames
        if video_data.shape[1] < self.max_frames:
            pad_size = self.max_frames - video_data.shape[1]
            padding = torch.zeros((video_data.shape[0], pad_size, video_data.shape[2], video_data.shape[3]))
            video_data = torch.cat((video_data, padding), dim=1)

        if self.transform:
            video_data = torch.stack([self.transform(video_data[c]) for c in range(video_data.size(0))])

        anomalies = []
        video_labels = annotation.get('anomalies', [])
        video_length = annotation.get('video_length', [])

        for anomaly in video_labels:
            for key, intervals in anomaly.items():
                anomalies.extend(intervals)

        return {
            'video_data': video_data,
            'frame_numbers': frame_numbers,
            'label': label,
            'anomalies': anomalies,
            'video_key': video_key,
            'video_length': video_length
        }



def get_transform():
    return Compose([
        Normalize(mean=[0.5], std=[0.5])
    ])
