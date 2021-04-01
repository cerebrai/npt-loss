import torch.nn.functional as F
from smodels.model_resnet import getmodel_byname
from torchvision import transforms, utils
from Params import Params
import os
from STrain import Trainer
from SDataLoad import DataLoad
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn
from datasets import MXFaceDataset

def check_resume(save_dir):
    saved_models = os.listdir(save_dir)
    try:
        i = saved_models.index("train_results.pkl")
        del saved_models[i]
    except:
        pass

    if len(saved_models) > 0:
        saved_model_ints = [int(x.split("_")[2].split(".")[0]) for x in saved_models]
        saved_model_ints.sort(reverse=True)
        resume_checkpoint = save_dir + "/model_epoch_" + str(saved_model_ints[0]) + ".pth"
        return True, resume_checkpoint
    else:
        return False, None

def main():

    torch.manual_seed(1984)
    Param = Params[1]

    checkpoint = dict()

    # -- Defining Transforms
    INPUT_SIZE = [112, 112] # support: [112, 112] and [224, 224]
    RGB_MEAN = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    RGB_STD = [128./255., 128./255., 128./255.]
    train_transforms = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    val_transforms = transforms.Compose([
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = RGB_MEAN,
                             std = RGB_STD),
    ])

    # -- recfile dataloader
    data_loader_rec_san = MXFaceDataset(Param["data_root"], train_transforms)
    loader = DataLoader(data_loader_rec_san, batch_size=Param["batch_size"], shuffle=True, num_workers=32, pin_memory=True)

    checkpoint["train_trans"] = train_transforms
    checkpoint["val_trans"] = val_transforms
    checkpoint['num_classes'] = data_loader_rec_san.num_classes

    # --- Defining Model --- #
    model = getmodel_byname(Param['model_name'])
    checkpoint["model_name"] = Param['model_name']
   
    # -- Loading another old model as initial weights
    #old_checkpoint2 = torch.load("/vol/vssp/cvpnobackup/facer2vm/people/safwan/saved_models/EP8s/9-EP8/model_epoch_1.pth")
    #model.load_state_dict(old_checkpoint2["weight"])

    checkpoint["Param"] = Param

    # -- checking if resuming is required
    save_dir = os.path.join(Param['save_dir'], Param['exp_id'])
    resume_flag = False
    if Param['save_dir'] is not None:
        if os.path.isdir(save_dir):
            resume_flag, resume_checkpoint_path = check_resume(save_dir)

    
    if resume_flag:
        resume_checkpoint = torch.load(resume_checkpoint_path)
        print("Resuming from checkpoint : " + resume_checkpoint_path)
        #model = nn.DataParallel(model)
        #model.load_state_dict(resume_checkpoint["weight"])
        checkpoint = resume_checkpoint

    # ------- Creating and calling the trainer object -- #
    trainer = Trainer(model, Param, F.cross_entropy, checkpoint=checkpoint, save_dir=save_dir, save_freq=2, resume_flag=resume_flag)
    trainer.loop(Param['epochs'], loader)

    # -- For visualizing a batch
    # for data in train_loader:
    #     imshow(data[0])




if __name__ == '__main__': # if running as a script not being loaded as a module



    main()
