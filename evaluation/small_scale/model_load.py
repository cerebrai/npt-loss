import torch
import torch.nn as nn

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_name = checkpoint["model_name"]
    if model_name == "mfacenet":
        from smodels.model_facemobilenet import getmodel_byname
    else:
        from smodels.model_resnet import getmodel_byname
  
    num_classes = checkpoint["num_classes"]
    trans = checkpoint["val_trans"]
    model = getmodel_byname(model_name)
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["weight"])
    return model, trans




