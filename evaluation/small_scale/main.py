import torchvision.transforms as transforms
from utils import perform_val, get_val_data
import torch
from model_load import load_model
import sys
import os
from config import config

def evaluation(checkpoint_path,DEVICE,EMBEDDING_SIZE, Val_DATA_ROOT):

   BACKBONE, test_transform = load_model(checkpoint_path)

   lfw, cfp_fp, agedb_30, calfw, cfp_ff, vgg2_fp, lfw_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cfp_ff_issame, vgg2_fp_issame = get_val_data(Val_DATA_ROOT, transform=test_transform)
   lfw_loader = torch.utils.data.DataLoader(
        lfw, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

   cfp_fp_loader = torch.utils.data.DataLoader(
        cfp_fp, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

   agedb_30_loader = torch.utils.data.DataLoader(
        agedb_30, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

   calfw_loader = torch.utils.data.DataLoader(
        calfw, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

   cfp_ff_loader = torch.utils.data.DataLoader(
        cfp_ff, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

   vgg2_fp_loader = torch.utils.data.DataLoader(
        vgg2_fp, 
        batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)



   #evaluate verifcation datasets
   print("Perform Evaluation on LFW,  CFP_FP, AgeDB ...")
   accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(DEVICE, EMBEDDING_SIZE, BACKBONE, lfw_loader, lfw_issame)
   print("accuracy_lfw:{}".format(accuracy_lfw))

   accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(DEVICE, EMBEDDING_SIZE,  BACKBONE, cfp_fp_loader, cfp_fp_issame)
   print("accuracy_cfp_fp:{}".format(accuracy_cfp_fp))    
    
   accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(DEVICE, EMBEDDING_SIZE,  BACKBONE, agedb_30_loader, agedb_30_issame)
   print("accuracy_agedb:{}".format(accuracy_agedb))

   accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(DEVICE, EMBEDDING_SIZE,  BACKBONE, calfw_loader, calfw_issame)
   print("accuracy_calfw:{}".format(accuracy_calfw))

   accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(DEVICE, EMBEDDING_SIZE,  BACKBONE, cfp_ff_loader, cfp_ff_issame)
   print("accuracy_cfp_ff:{}".format(accuracy_cfp_ff))

   accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(DEVICE, EMBEDDING_SIZE,  BACKBONE, vgg2_fp_loader, vgg2_fp_issame)
   print("accuracy_vgg2_fp:{}".format(accuracy_vgg2_fp))


if __name__ == "__main__":
    
    configs = config[1]
    Val_DATA_ROOT= configs["data_root"]
    EMBEDDING_SIZE=configs["embedding_size"]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_dir = sys.path[0]
    checkpoint_path = configs["checkpoint"] 
    evaluation(checkpoint_path, DEVICE, EMBEDDING_SIZE, Val_DATA_ROOT)


