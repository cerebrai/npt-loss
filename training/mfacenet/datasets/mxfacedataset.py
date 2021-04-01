import os
import numbers
import numpy
#import cv2
import torch
from PIL import Image
import mxnet as mx
from torch.utils.data import Dataset

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.train = train
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        patn_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(patn_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            print("header0 label:", header.label)
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = list(range(1, int(header.label[0])))
            self.num_classes = int(self.header0[1] - self.header0[0])
            self.samples = len(self.imgidx)
            # print(self.imgidx)
        else:
            self.imgidx = list(self.imgrec.keys)
        print("Number of Samples:{} Number of Classes: {}".format(len(self.imgidx), int(self.header0[1] - self.header0[0])))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        # print(label)

        sample = mx.image.imdecode(img).asnumpy()  # RGB
        img = Image.fromarray(numpy.uint8(sample))
        if self.transform is not None:
            img = self.transform(img)

        # --  making return dictionary
        data = dict()
        data["img"] = img
        data["label"] = label

        return data


    def __len__(self):
        # print(len(self.imgidx))
        return len(self.imgidx)


if __name__ == '__main__':
    from PIL import Image
#    import torchvision.transforms as transforms
#    train_transforms = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize(mean = [0.5,  0.5, 0.5],
#                             std = [0.5,  0.5, 0.5]),
#    ])

    root_dir = '/scratch/cc0011/ms1m-retinaface-t1/'
    trainset = MXFaceDataset(root_dir)
    img,label=trainset[1]
#    print('img:{}'.format(img.size()))
    print(img)
#.convert('RGB')
    img.show()

#    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
#    for batch_idx, (sample, label) in enumerate(train_loader):
#        print(sample.shape, label)
