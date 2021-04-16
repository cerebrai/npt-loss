This code corresponds to the paper "NPT-Loss: Demystifying face recognition losses with Nearest Proxies Triplet". The paper can be found in the main directory. 
The repository contains training codes, links to training datasets, evaluation codes, links to evaluation datasets and pretrained models.
All codes use the Pytorch framework.

# Training
See requirement file for the required python libraries to run the training code. 
Alternatively, run the following commands (assuming that anaconda is already installed)
```

conda create -n pytorch_npt python=3.7
conda activate pytorch_git
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c
pytorch
conda install -c anaconda scikit-learn=0.24.1
conda install -c conda-forge matplotlib=3.3.4
conda install -c anaconda opencv=3.4.2
conda install -c conda-forge tqdm=4.59.0
conda install -c anaconda pandas=1.2.3
conda install -c anaconda scikit-image=0.17.2
conda install -c conda-forge prettytable=2.1.0
conda install -c anaconda mxnet=1.5.0

conda install -c anaconda joblib=1.0.1
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
For evaluation on megaface, we need to make another python enviroment. Run the following commands 
```
conda create -n mxnet_git python=2.7
conda install -c anaconda mxnet=1.5.0
conda install -c anaconda opencv=3.4.2
```
Download the megaface protocol folder from [here](https://drive.google.com/file/d/1lYk84qLwIlUdHOnqSJmvarz6mZ_s7pZ6/view?usp=sharing)

Download the megaface testpack from insightface repository from [here](https://github.com/deepinsight/insightface/tree/master/evaluation/Megaface)

Unzip the megaface testpack inside the megaface protocol folder. 

Specify paths in run_feats.sh file

run run_feats.sh
