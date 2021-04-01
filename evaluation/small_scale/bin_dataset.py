import numpy as np
from torch.utils.data import Dataset
import pickle
import sys
import io
from PIL import Image


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
        return img


def bin_loader(path):
    '''load verification img array and label from bin file
    '''
    print(path)
    if sys.version_info >=(3, 5) :
       bins, lbs = pickle.load(open(path, 'rb'), encoding='bytes')
    else :
       bins, lbs = pickle.load(open(path, 'rb'))
    assert len(bins) == 2*len(lbs)
    imgs = [pil_loader(b) for b in bins]
    return imgs, lbs

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
