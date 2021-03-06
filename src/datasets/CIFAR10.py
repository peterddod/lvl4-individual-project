import torch
from torch.utils.data import Dataset
from nitools.utils import load_cifar10

class CIFAR10(Dataset):

    def __init__(self, train=True, scaled=False):
        self.data = load_cifar10(scaled)
        self._type = "test"

        if train:
            self._type = "train"

    def __len__(self):
        return len(self.data[f'{self._type}_X'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.data[f'{self._type}_X'][idx], self.data[f'{self._type}_y'][idx])

        return sample