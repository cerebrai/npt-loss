This code corresponds to the paper "NPT-Loss: Demystifying face recognition losses with Nearest Proxies Triplet". The paper can be found in the main directory. 
The repository contains training codes, links to training datasets, evaluation codes, links to evaluation datasets and pretrained models.
All codes use the Pytorch framework.

# Training
See requirement file for the required python libraries to run the training code. 
Alternatively, run the following commands (assuming that anaconda is already installed)
```

conda create -n pytorch_npt
```
Download the training datasets, i.e., CASIA and MS1M-v2 from the insightface repository [dataset-zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo).

Provide the path of the training dataset in Params.py. 

Provide a path where the trained models will be saved in Params.py.

Run trainer.sh.
