#!/bin/bash


# -- Getting into script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

# -- calling the python script
echo "[INFO] Calling Python Script"
python Main.py 2>&1 

echo "Done"


