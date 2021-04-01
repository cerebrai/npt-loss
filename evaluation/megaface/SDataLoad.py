"""
Custom data loader
"""
from torch.utils.data.dataset import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
# -----  Custom Data Loader Class ------ #
class DataLoad(Dataset):
    def __init__(self, csv_file_path, csv_file_root, transforms=None, matfile_path=None):

        data = pd.read_csv(csv_file_path, header=None)
        self.img_paths = np.asarray(data.iloc[:, 0])
        self.root = csv_file_root
        #self.labels = np.asarray(data.iloc[:, 1])
        self.transforms = transforms
        self.matfile_path = matfile_path
        if self.matfile_path is not None:
            self.data = loadmat(matfile_path)


    def __getitem__(self, index):

        current_path = os.path.join(self.root, self.img_paths[index])
        #pudb.set_trace()
        #current_label = self.labels[index]
        if self.matfile_path is not None:
            img_as_img = Image.fromarray(self.data[current_path]).convert('RGB')
        else:
            img_as_img = Image.open(current_path).convert('RGB')



        if self.transforms is not None:
            img_transformed_hr = self.transforms(img_as_img)

        return img_transformed_hr, current_path

    
    def __len__(self):
        return len(self.img_paths)

