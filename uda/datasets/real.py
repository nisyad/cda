import torch
from torch.utils.data import Dataset, DataLoader


class REALDataset(Dataset):
    def __init__(self, data, lbls, transform=None):

        self.data = data.permute(0, 3, 1, 2)
        self.labels = lbls
        self.transform = transform

    def __len__(self):
        return len(self.data)  # 60000 for training and 10000 for test

    def __getitem__(self, idx):

        X = self.data[idx].float()
        y = self.labels[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


def fetch(data,
          lbls,
          batch_size=64,
          transform=None,
          shuffle=True,
          num_workers=1,
          pin_memory=True):

    # data = torch.load(data_dir)

    dataset = REALDataset(data=data, lbls=lbls, transform=transform)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory)
