#!/bin/bash

python train.py \
-e vanilla_conv \
-d vanilla_deconv \
-w 1 \
-o 0 \
-m ../runs/tests/vanilla/ \
--data-dir ../data/train \
--n-epoch 1
