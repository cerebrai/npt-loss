import numpy as np
import torch
import torch.distributed as dist
import os
import io
import pickle
from PIL import Image
import multiprocessing as mp
import sys

def init_processes(addr, port, gpu_num, backend):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(rank, size)
    if mp.get_start_method(allow_none=True) != 'spawn':
       try:
         mp.set_start_method('spawn')
       except RuntimeError:
         pass

    torch.cuda.set_device(rank % gpu_num)
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend)
    print('initialize {} successfully (rank {})'.format(backend, rank))
    return rank, size

def separate_bn_paras(modules):
#https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/utils.py
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)
            
    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn

#def separate_irse_bn_paras(modules):
#    if not isinstance(modules, list):
#        modules = [*modules.modules()]
#    paras_only_bn = []
#    paras_wo_bn = []
#    for layer in modules:
#        if 'model' in str(layer.__class__):
#            continue
#        if 'container' in str(layer.__class__):
#            continue
#        else:
#            if 'batchnorm' in str(layer.__class__):
#                paras_only_bn.extend([*layer.parameters()])
#            else:
#                paras_wo_bn.extend([*layer.parameters()])
#
#    return paras_only_bn, paras_wo_bn



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_ckpt(state, ckpt, epoch, is_best=False):
    folder = os.path.dirname(ckpt)
    fn = '{}_epoch_{}.pth.tar'.format(os.path.basename(ckpt), epoch)
    if folder != ''and not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, fn)
    print('saving to {}'.format(path))
    torch.save(state, '{}'.format(path))
    if is_best:
        bes_fn ='{}_model_best.pth.tar'.format(os.path.basename(ckpt))
        best_fn = os.path.join(folder, bes_fn)
        if os.path.exists(best_fn):
            os.unlink(best_fn)
        os.symlink(fn, best_fn)


def load_ckpt(path, model, ignores=[], strict=True, optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if len(ignores) > 0:
            assert optimizer == None
            keys = list(checkpoint['state_dict'].keys())
            for ignore in ignores:
                if ignore in keys:
                    print('ignoring {}'.format(ignore))
                    del checkpoint['state_dict'][ignore]
                else:
                    raise ValueError('cannot find {} in load_path'.format(ignore))
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if not strict:
            pretrained_keys = set(checkpoint['state_dict'].keys())
            model_keys = set([k for k, _ in model.named_parameters()])
            for k in model_keys - pretrained_keys:
                print('warning: {} not loaded'.format(k))
        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['step']))
            return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)


def normalize(feat, axis=1):
    if axis == 0:
        return feat / np.linalg.norm(feat, axis=0)
    elif axis == 1:
        return feat / np.linalg.norm(feat, axis=1)[:, np.newaxis]


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
        return img


def bin_loader(path):
    '''load verification img array and label from bin file
    '''
    print(path)
    if sys.version_info >=(3, 5) :
       bins, lbs = pickle.load(open(path, 'rb'), encoding='bytes')
    else :
       bins, lbs = pickle.load(open(path, 'rb'))
    assert len(bins) == 2*len(lbs)
    imgs = [pil_loader(b) for b in bins]
    return imgs, lbs


def save_imgs(imgs, ofolder):
    '''save pil image array to JPEG image file
    '''
    for i, img in enumerate(imgs):
        opath = os.path.join(ofolder, "{}.jpg".format(i))
        if not os.path.exists(os.path.dirname(opath)):
            print(opath)
            os.makedirs(os.path.dirname(opath))
        img.save(opath, "JPEG")
    else:
        raise TypeError('axis value should be 0 or 1(cannot handel axis {})'.format(axis))


