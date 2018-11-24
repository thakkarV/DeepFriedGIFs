#!/bin/bash

python train.py \
-e vanilla_encoder \
-d vanilla_decoder \
-w 1 \
-o 0 \
-cp CC \
-m ../runs/tests/vanilla/ \
--data-dir ../data/train \
--n-epoch 1
