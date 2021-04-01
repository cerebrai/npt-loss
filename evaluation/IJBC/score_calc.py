import os
import numpy as np
#from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt
#import sklearn
import cv2
import sys
import joblib
#import glob
#from prettytable import PrettyTable
import pandas as pd

def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:,1].astype(np.int)
    medias = ijb_meta[:,2].astype(np.int)
    return templates, medias

def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, header=None, sep=" ")
    t1 = pairs.iloc[:, 0].astype(np.int)
    t2 = pairs.iloc[:, 1].astype(np.int)
    label = pairs.iloc[:, 2].astype(np.int)
    return t1, t2, label

def image2template_feature(img_feats = None, templates = None, medias = None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u,ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else: # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        #pudb.set_trace()
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.squeeze()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


main_dir = '/vol/vssp/facer2vm2/people/safwan/IJB_release'
templates, medias = read_template_media_list(os.path.join(main_dir + '/IJBC/meta', 'ijbc_face_tid_mid.txt'))
p1, p2, label = read_template_pair_list(os.path.join(main_dir + '/IJBC/meta', 'ijbc_template_pair_label.txt'))
feats_dict = joblib.load(sys.path[0] + "/results/feats.pkl")
img_feats = feats_dict['img_feats']
faceness_scores = feats_dict['faceness_scores'] 

use_norm_score = False # if Ture, TestMode(N1)  
use_detector_score = False # if Ture, TestMode(D1)
use_flip_test = False # if Ture, TestMode(F1)

if use_flip_test:
    # concat --- F1
    #img_input_feats = img_feats 
    # add --- F2
    img_input_feats = img_feats[:,0:int(img_feats.shape[1]/2)] + img_feats[:,int(img_feats.shape[1]/2):]
else:
    img_input_feats = img_feats[:,0:int(img_feats.shape[1]/2)]
    
if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))    
    
if use_detector_score:
    img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
score = verification(template_norm_feats, unique_templates, p1, p2)
score_save_name = sys.path[0] + "/results/scores.npy"
np.save(score_save_name, score)
