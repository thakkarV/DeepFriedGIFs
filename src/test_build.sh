#!/bin/bash -l

#$ -m beas
#$ -pe omp 4
#$ -l mem_total=64G
#$ -l gpus=0.25
#$ -l gpu_c=6.0

module load python/3.6.2
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
