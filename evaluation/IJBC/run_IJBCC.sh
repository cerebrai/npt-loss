#!/bin/bash

# --- activating conda enviroment

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

python feat_extract.py 2>&1
python score_calc.py 2>&1
python results_calc.py 2>&1

