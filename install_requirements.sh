#/bin/bash
set -eux

pip install -U pip tqdm numpy pydraughts pandas matplotlib scikit-learn scikit-optimize

if [[ "$USER" = "joyvan" ]]; then
    pip install -U torch~=2.1.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
else
    pip install -U torch
fi
