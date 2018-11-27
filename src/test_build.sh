#!/bin/bash

module load python/3.6.2
# module load cuda/9.1
# module load cudnn/7.1
module load tensorflow/r1.10

python train.py \
-e vanilla_encoder \
-d vanilla_decoder \
-w 1 \
-o 0 \
-cp CC \
-m ../runs/tests/vanilla/ \
--data ../data/train \
--n-epoch 1
