import torch
from torch.utils.data import Dataset
from nitools.utils import load_norb

class NORB(Dataset):

    def __init__(self, train=True, scaled=False, augment=False, label_smoothing=0.1):
        self.data = load_norb(scaled, augment=augment, label_smoothing=label_smoothing)
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