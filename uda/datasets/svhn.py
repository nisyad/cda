import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SVHNDataset(Dataset):
    def __init__(self, data, transform=None):

        # self.data = data['X'] / 255  # convert to 0-1 range
        self.data = data['X']
        self.data = self.data.transpose(3, 0, 1, 2)  # [N,3,28,28]
        self.labels = data['y'].reshape(-1)
        self.transform = transform

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):

        X = self.data[idx]
        y = self.labels[idx].astype(np.int64)

        if self.transform:
            X = self.transform(X)

        return X, y


def fetch(data,
          batch_size=64,
          transform=None,
          shuffle=True,
          num_workers=1,
          pin_memory=True):

    # data = torch.load(data_dir)

    dataset = SVHNDataset(data=data, transform=transform)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
