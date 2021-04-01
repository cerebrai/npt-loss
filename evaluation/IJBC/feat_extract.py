import os
import numpy as np
import cv2
import sys
import glob
import pudb
from tqdm import tqdm 
from GetFeats import GetFeats
import joblib
import json
from config import config

def get_image_feature(getfeat_obj, img_path, img_list_path):
    img_list = open(img_list_path)
    files = img_list.readlines()
    #files  = files[:10]
    img_feats = []
    faceness_scores = []
    for img_index, each_line in enumerate(tqdm(files)):
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape( (5,2) )
        feat = getfeat_obj.get_feats(img, lmk)
        img_feats.append(feat)
        #pudb.set_trace()
        faceness_scores.append(name_lmk_score[-1])
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores

# -- Reading the config file
configs = config[1]
main_dir =  configs["main_dir"]
img_path = main_dir + '/IJBC/loose_crop'
img_list_path = main_dir + '/IJBC/meta/ijbc_name_5pts_score.txt'
checkpoint_path = configs['checkpoint']
getfeat_obj = GetFeats(checkpoint_path)
img_feats, faceness_scores = get_image_feature(getfeat_obj, img_path, img_list_path)
feats_dict = dict()
feats_dict['img_feats'] = img_feats
feats_dict['faceness_scores'] = faceness_scores
joblib.dump(feats_dict, sys.path[0] + "/results/feats.pkl")
