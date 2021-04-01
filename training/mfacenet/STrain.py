"""
Trainer class to be called in main for training and testing the model
"""
import sys
import torch
from tqdm import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.optim as optim
import numpy as np
import joblib

class Trainer(object):
    cuda = torch.cuda.is_available()
    if cuda:
        print("GPU detected: Running in GPU Mode")
    else:
        print("No GPU: Running in CPU Mode")
    
    np.random.seed(1984)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    def __init__(self, model, Param, loss_f, checkpoint={}, save_dir=None, save_freq=2, resume_flag=False):

        self.model = model
        self.device_id = list(range(0, torch.cuda.device_count()))
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.checkpoint = checkpoint
        self.batch_size = Param["batch_size"]
        self.r = Param["r"]
        self.delta = Param["delta"]
        self.top_k = Param["top_k"]
        if Param["multi_flag"]:
            self.multi_flag = True
        else:
            self.multi_flag = False

        self.loss_f = NPTLoss(r=self.r, delta=self.delta, top_k=self.top_k)
        #self.loss_f = loss_f #uncomment for arcface

        if not resume_flag:
            self.head = Linear(256, self.checkpoint["num_classes"], multi_flag=self.multi_flag, device_id=self.device_id)
            #self.head = ArcFace(256, self.checkpoint["num_classes"], s = self.r, m=self.delta, device_id=self.device_id)
        else:
            self.head = Linear(256, self.checkpoint["num_classes"], fc=self.checkpoint["head_fc"], multi_flag=self.multi_flag, device_id=self.device_id)
            #self.head = ArcFace(256, self.checkpoint["num_classes"], fc=self.checkpoint["head_fc"], s = self.r, m = self.delta, device_id=self.device_id)


        self.mat_train_result_path = os.path.join(self.save_dir, "train_results.pkl")
        if os.path.isfile(self.mat_train_result_path):
            self.train_results = joblib.load(self.mat_train_result_path)
        else:
            self.train_results = dict()


        # -- Defining optimizer
        ignored_params = list(map(id, self.model.output.parameters()))
        ignored_params += list(map(id, self.head.weight))
        prelu_params_id = []
        prelu_params = []
        for m in self.model.modules():
            if isinstance(m, nn.PReLU):
                ignored_params += list(map(id, m.parameters()))
                prelu_params += m.parameters()
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        self.optimizer = optim.SGD([
            {'params': base_params, 'weight_decay': 1e-5},
            {'params': self.model.output.parameters(), 'weight_decay': 1e-4},
            {'params': self.head.weight, 'weight_decay': Param['decay']},
            {'params': prelu_params, 'weight_decay': 0.0}
        ], lr=Param['lr'], momentum=Param['moment'], nesterov=True)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 45], gamma=0.1)

        # -- Moving to GPU 
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        self.model = nn.DataParallel(self.model)
        if resume_flag:
            self.model.load_state_dict(checkpoint["weight"])
        if len(self.device_id)>0:
            self.model.cuda()
        else:
            print("No GPUs !! Are you kidding me ?? I quit !")
            exit()

        if self.cuda and self.multi_flag is False:
            self.head.cuda(self.device_id[0])

        if resume_flag:
            self.optimizer.load_state_dict(checkpoint["optim_state"])
            self.scheduler.load_state_dict(checkpoint["sch_state"])


    def _iteration(self, data_loader, is_train=True):
        loop_loss, accuracy = list(), list()

        for data_dict in tqdm(data_loader, ncols=80):
        # -- unpacking data dict
            data = data_dict["img"]
            target = data_dict["label"]
            if self.cuda:
                data, target = data.cuda(), target.cuda()

            feats = self.model.forward(data)
            dot_p = self.head(feats, target)
            loss = self.loss_f(dot_p.clone(), target)
            loop_loss.append(loss.data.item())
            accuracy.append((dot_p.data.max(1)[1] == target.data).sum().item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print(f" loss: {sum(loop_loss) / (len(data_loader.dataset)/ self.batch_size):.4f} /accuracy: {sum(accuracy) / len(data_loader.dataset):.2%} LR_vis: {self.optimizer.param_groups[0]['lr']}")
        self.result_append("Train_loss", loop_loss)
        self.result_append("Train_loss_avg", sum(loop_loss) / (len(data_loader.dataset)/ self.batch_size))
        self.result_append("Train_acc", (sum(accuracy) / len(data_loader.dataset)))


    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            self._iteration(data_loader)

    def loop(self, epochs, train_data, scheduler=None):

        if scheduler is None:
            scheduler = self.scheduler
        last_epoch = scheduler.state_dict()["last_epoch"]

        for ep in range(last_epoch + 1, epochs + 1):

            print("epochs: {}".format(ep))
            self.train(train_data)
            if scheduler is not None:
                scheduler.step()
            if ep % self.save_freq:
                self.save(ep, scheduler)

    def save(self, epoch, scheduler, **kwargs):
        if self.save_dir is not None:
            model_out_path = self.save_dir
            self.checkpoint["epoch"] = epoch
            self.checkpoint["weight"] = self.model.state_dict()
            self.checkpoint["optim_state"] = self.optimizer.state_dict()
            self.checkpoint["sch_state"] = scheduler.state_dict()
            self.checkpoint["head_fc"] = self.head.weight
            if not os.path.isdir(model_out_path):
                os.makedirs(model_out_path)
            torch.save(self.checkpoint, model_out_path + "/model_epoch_{}.pth".format(epoch))

            joblib.dump(self.train_results, self.mat_train_result_path)
    

    def result_append(self, key, val):
        if not (key in self.train_results):
            self.train_results[key] = list()
        if isinstance(val, list):
            self.train_results[key].extend(val)
        else:
            self.train_results[key].append(val)


class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device_id=[0], s=64.0, m=0.50, easy_margin=False, fc=None):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        if fc is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            self.weight = Parameter(fc)



        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input

            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class Linear(nn.Module):
    r"""Implement of Softmax (normal classification head):
    Args:
           in_features: size of each input sample
           out_features: size of each output sample
           device_id: the ID of GPU where the model will be trained by model parallel. 
                      if device_id=None, it will be trained on CPU without model parallel.
    """
    def __init__(self, in_features, out_features, fc=None, device_id=[0], multi_flag=True, normalised = True, eval_flag=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.multi_flag = multi_flag # if true split class weight vector among GPUs
        self.normalised = normalised
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))

        #self.bias = Parameter(torch.FloatTensor(out_features))
        if fc is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            self.weight = Parameter(fc)

        #nn.init.zeros_(self.bias)

        if eval_flag:
            self.weight.requires_grad = False


    def forward(self, x, label, s=1.0):
        batch_size = label.size(0)
        if self.device_id == None or self.multi_flag is False:
            if self.normalised:
                #pudb.set_trace()
                out = F.linear(F.normalize(x), F.normalize(self.weight))
            else:
                out = F.linear(x, self.weight)
        else:

            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            if self.normalised:
               out = F.linear(F.normalize(temp_x), F.normalize(weight))
            else:
               out = F.linear(temp_x, weight)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                if self.normalised:
                   out = torch.cat((out, (F.linear(F.normalize(temp_x), F.normalize(weight))).cuda(self.device_id[0])), dim=1)
                else:
                   out = torch.cat((out, (F.linear(temp_x, weight)).cuda(self.device_id[0])), dim=1)


        return out*s





class NPTLoss(nn.Module):
    def __init__(self, r=1.0, delta=0.5, top_k=2):
        super(NPTLoss, self).__init__()
        self.delta = delta
        self.r = r
        self.top_k = int(top_k)

    def forward(self, dot_p, target):
        
        true_class_dist = dot_p[torch.arange(0, dot_p.shape[0]), target]
        dot_p[torch.arange(0, dot_p.shape[0]), target] = 0
        negative_max_sort, _ = torch.sort(dot_p, 1, descending=True)
        negative_max_top_k = negative_max_sort[:, :self.top_k]
        Temp = negative_max_top_k - true_class_dist[:,None] + self.delta
        Temp[Temp<0] = 0
        Temp = Temp * 2* self.r
        Temp = torch.sum(Temp,1)
        return torch.mean(Temp)

