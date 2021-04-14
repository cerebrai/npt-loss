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

# Evaluation

Pretrained models can be found [here](https://drive.google.com/drive/folders/1avkjV0zr1y7x7B2_DIJbVkkkezcEfTdE?usp=sharing)

## Small Scale (i.e., LFW, CFP etc)
Open config.py

Set path for the evaluation datasets. The training dataset for MS1M-v2 also contains the evaluation datasets (i.e., .bin files) for evaluation on LFW, CFP, AgeDB etc.

Specify the embedding size(256 for Mobilefacenet and 512 for Resnet-50).

Specify the path to the trained model.

Activate the conda enviroment and run 
```
python main.py
```

## IJBB and IJBC
Download [IJB datasets](https://github.com/deepinsight/insightface/tree/master/evaluation/IJB) from the insightface repository. Remember to update the metadata.

Open config.py (in both IJBB and IJBC folders) and set paths for the datasets and the trained models.

Activate conda envroment and run the run_IJBB.sh and run_IJBC.sh scripts

## MegaFace
