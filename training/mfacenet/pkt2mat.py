import joblib
from scipy.io import savemat
import sys
import json
from Params import Params

Param = Params[1] 

file_path = Param["save_dir"] + "/" + Param["exp_id"] + "/train_results.pkl"
out_path = sys.path[0] + "/train_results.mat"

data_dict = joblib.load(file_path)
savemat(out_path, data_dict)


