from verification import evaluate
import os
from bin_dataset import BinDataset
import torchvision.transforms as transforms

import torch
import numpy as np
import struct
import numbers
import io
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_val_pair(path, name, input_size, transform=None):

    fn = '{}.bin'.format(name)
    dataset = BinDataset(os.path.join(path, fn), transform)
    issame = dataset.lbs
    return dataset, issame

def get_val_data(data_path, transform=None, input_size =[112,112]):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw', input_size, transform)
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp', input_size, transform)
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30', input_size, transform)

    calfw, calfw_issame = get_val_pair(data_path, 'calfw', input_size, transform)
    cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff', input_size, transform)
    vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp', input_size, transform)



    return lfw, cfp_fp, agedb_30, calfw, cfp_ff, vgg2_fp, lfw_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cfp_ff_issame, vgg2_fp_issame



def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf

def normalize(feat, axis=1):
    if axis == 0:
        return feat / np.linalg.norm(feat, axis=0)
    elif axis == 1:
        return feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]

def extract_feature(device=None, embedding_size=None, backbone=None, grpface=None, test_loader=None, normalised = True):
    
    #pudb.set_trace()
    backbone = backbone.cuda()
    backbone.eval() # switch to evaluation mode
    idx = 0
    features = []
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            output = backbone(input.cuda())+backbone(torch.flip(input.cuda(),[3]))
            if normalised:
               norm = output.pow(2).sum(dim=1, keepdim=True).sqrt()+1e-10
               output = torch.div(output,norm)
            features.append(output.data.cpu().numpy())

    embeddings=np.vstack(features)
    return embeddings

def perform_val(device=None, embedding_size=None, backbone=None, test_loader=None, issame=None, metric='dist', normalised = True, nrof_folds = 10):

    embeddings=extract_feature(device=device, embedding_size=embedding_size, backbone=backbone, test_loader=test_loader, normalised=normalised)

#    embeddings = normalize(embeddings)
#    print('embeddings:{}'.format(embeddings.shape))
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds = nrof_folds, metric=metric)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)
#    print('Accuracies:{}'.format(accuracy))
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res
