#!/bin/bash -l

#$ -P dl-course
#$ -m beas
#$ -pe omp 4
#$ -l mem_total=64G
#$ -l gpus=0.25
#$ -l gpu_c=6.0
#$ -l h_rt=48:00:00

module load python/3.6.2
module load tensorflow/r1.10

python train.py \
-e vgg_small_encoder \
-d vgg_small_decoder \
-w 1 \
-o 0 \
-cp CC \
-ch 64 \
-cw 64 \
-m ../runs_11_30_v7/tests/vanilla/ \
--data ../data/train \
--n-epoch 10 \
--log-interval 10

