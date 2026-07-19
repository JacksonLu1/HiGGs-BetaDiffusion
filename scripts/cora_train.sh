#!/bin/sh

conda init higgs
conda activate higgs

python dgd/main.py dataset=cora dataset.h=2 dataset.resolution=3 model=beta_h2 general.name=cora_beta_h2_r2 train.n_epochs=500
python dgd/main.py dataset=cora dataset.h=1 dataset.resolution=1 general.name=cora_h1_r2 train.batch_size=4 train.n_epochs=1000
python dgd/main.py dataset=cora dataset.h=1.5 dataset.resolution=1 general.name=cora_glue_r2 train.batch_size=2 train.n_epochs=500
