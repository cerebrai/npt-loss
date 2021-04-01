import numpy as np
from prettytable import PrettyTable
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import roc_curve, auc
import os
import sys

def read_template_pair_list(path):
    pairs = np.loadtxt(path, dtype=str)
    t1 = pairs[:,0].astype(np.int)
    t2 = pairs[:,1].astype(np.int)
    label = pairs[:,2].astype(np.int)
    return t1, t2, label

main_dir = '/vol/vssp/facer2vm2/people/safwan/IJB_release'
p1, p2, label = read_template_pair_list(os.path.join(main_dir + '/IJBC/meta', 'ijbc_template_pair_label.txt'))
score_save_path = sys.path[0] + "/results"
files = glob.glob(score_save_path + '/*.npy')  
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file)) 
methods = np.array(methods)
scores = dict(zip(methods,scores))
#colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
#x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
#pudb.set_trace()
tpr_fpr_table = PrettyTable(['Methods'] + list(map(str, x_labels)))
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr
    #plt.plot(fpr, tpr, color=colours[method], lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
    plt.plot(fpr, tpr, lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
    tpr_fpr_row = []
    tpr_fpr_row.append(method)
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels) 
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True)) 
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB-C')
plt.legend(loc="lower right")
print(tpr_fpr_table)
#plt.show()

