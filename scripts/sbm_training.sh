#!/bin/sh

conda init higgs
conda activate higgs

python dgd/main.py dataset=sbm dataset.h=1 dataset.resolution=1
python dgd/main.py dataset=sbm dataset.h=1.5 dataset.resolution=1
# 500 epochs is the current experiment plan; whether it is a paper-specified
# value must be established separately before reporting formal results.
python dgd/main.py dataset=sbm dataset.h=2 dataset.resolution=1 model=beta_h2 model.eta_strategy=fixed train.n_epochs=500
