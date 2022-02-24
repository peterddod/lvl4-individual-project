import torch
from torch.utils.data import Dataset
from nitools.utils import load_mnist

class MNIST(Dataset):

    def __init__(self, train=True, scaled=False):
        self.data = load_mnist(scaled)
        self._type = "test"

        if train:
            self._type = "train"

    def __len__(self):
        return len(self.data[f'{self._type}_X'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {f'{self._type}_X': self.data[f'{self._type}_X'][idx], f'{self._type}_y': self.data[f'{self._type}_y'][idx]}

        return sample