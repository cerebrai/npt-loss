import numpy as np
from utils import bin_loader
from torch.utils.data import Dataset


class BinDataset(Dataset):
    def __init__(self, bin_file, transform=None):
        self.img_lst, self.lbs = bin_loader(bin_file)
        self.num = len(self.img_lst)
        self.transform = transform

    def __len__(self):
        return self.num

    def _read(self, idx=None):
        if idx == None:
            idx = np.random.randint(self.num)
        try:
            img = self.img_lst[idx]
            return img
        except Exception as err:
            print('Read image[{}, {}] failed ({})'.format(idx, fn, err))
            return self._read()

    def __getitem__(self, idx):
        img = self._read(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img
