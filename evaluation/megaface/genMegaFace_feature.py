from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import struct
#from backbone.model_irse import IR_18, IR_50
from mega_utils import mega_extract
#from torchvision.datasets import FilesDataset
from model_load import load_model
from SDataLoad import DataLoad 




parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--arch', '-a', metavar='ARCH', default='IR_MFACENET',
                   )
parser.add_argument('--input-size', default=[112,112], type=int,
                    help='input size (default: 112x112)')
parser.add_argument('--batch-size', type=int, help='', default=1024)
parser.add_argument('--load-path', type=str)
parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
parser.add_argument('--output-facescrub', type=str, help='', default='./feature_out')
parser.add_argument('--output-megaface', type=str, help='', default='./feature_out')
def main():
    global args
    args = parser.parse_args()
    #INPUT_SIZE =[112,112]
    #======= model & loss & optimizer =======#
    #BACKBONE_DICT = {'IR_18': IR_18(INPUT_SIZE), 
                   #  'IR_50': IR_50(INPUT_SIZE)}

    #BACKBONE = BACKBONE_DICT[args.arch]

    #if args.load_path:
    #   BACKBONE.load_state_dict(torch.load(args.load_path))
    BACKBONE, val_transform = load_model(args.load_path)
    BACKBONE.cuda()
#    BACKBONE = torch.nn.DataParallel(BACKBONE, device_ids = [0,1,2,3]).cuda()
    #BACKBONE = torch.nn.DataParallel(BACKBONE, device_ids = [0]).cuda()
    print('transform')
    #val_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
#        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
#        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
     #   transforms.ToTensor(),
     #   transforms.Normalize(mean = [0.5, 0.5, 0.5],
   #                          std = [0.5, 0.5, 0.5]),
   # ])

    #val_dataset1 = FilesDataset(args.facescrub_lst, args.facescrub_root, val_transform)
    val_dataset1 = DataLoad(args.facescrub_lst, args.facescrub_root, transforms=val_transform)
    val_loader1 = torch.utils.data.DataLoader(
                 val_dataset1, 
                 batch_size=args.batch_size, shuffle=False,
                 num_workers=8, pin_memory=True)

    #val_dataset2 = FilesDataset(args.megaface_lst, args.megaface_root, val_transform)
    #"/vol/vssp/facer2vm/data/MegaFace/megaface_testpak_from_insightface/megaface_images.mat"
    val_dataset2 = DataLoad(args.megaface_lst, args.megaface_root, transforms=val_transform, matfile_path=None)
    val_loader2 = torch.utils.data.DataLoader(
                 val_dataset2, 
                 batch_size=args.batch_size, shuffle=False,
                 num_workers=64, pin_memory=True)

    BACKBONE.eval()
    tdevice=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mega_extract(args.arch, args.output_facescrub, val_loader1, BACKBONE)
    mega_extract(args.arch, args.output_megaface, val_loader2, BACKBONE)


if __name__ == '__main__':
    main()


