import torch

config = {
    1: dict(
        data_root = "", # where lfw.bin etc can be found
        embedding_size = 256, #256 for mfacenet; 512 for resnet
        checkpoint = "", # path of the stored model 
    ),
}

