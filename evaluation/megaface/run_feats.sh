#!/usr/bin/env bash

ALGO="NPT"
trained_model="/vol/research/NOBACKUP/CVSSP/facer2vm/people/safwan_nobackup/saved_models/new_codes_ms1m_res/2_n_ms1m_res/model_epoch_21.pth"
data_path="/vol/research/facer2vm_metric/people/safwan/datasets/megaface"
anaconda_path="/vol/research/facer2vm_metric/people/safwan/toolboxes/anaconda3"
DIM=512
DEVKIT=$data_path/devkit/experiments
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

python -u genMegaFace_feature.py --batch-size 100 --load-path $trained_model --arch $ALGO --facescrub-lst $data_path/facescrub_images.csv --megaface-lst $data_path/megaface_images.csv --megaface-root $data_path/megaface_images/ --facescrub-root $data_path/facescrub_images --output-facescrub $DIR/facescrub/ --output-megaface $DIR/megaface/ 2>&1

source $anaconda_path/etc/profile.d/conda.sh
conda activate mxnet_git

python -u remove_noises.py --algo "$ALGO" --feature-dir-input $DIR --feature-dir-out $DIR/feature_out_clean --dim $DIM 2>&1

export LD_LIBRARY_PATH=$data_path/missing_opencv_lib:$LD_LIBRARY_PATH
cd "$DEVKIT"
python -u run_experiment.py $DIR"/feature_out_clean/megaface" $DIR"/feature_out_clean/facescrub_images" "_"$ALGO".bin" $DIR"/Result_clean" -s 1000000 -p ../templatelists/facescrub_features_list_orig.json 2>&1 

cd -


