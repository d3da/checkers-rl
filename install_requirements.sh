#/bin/bash
set -eux

pip install -U pip tqdm numpy pydraughts pandas matplotlib scikit-learn scikit-optimize pyarrow

if [[ "$USER" = "jovyan" ]]; then
    pip install -U torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
else
    pip install -U torch
fi
