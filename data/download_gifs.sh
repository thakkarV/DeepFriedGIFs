#!/bin/bash -l

#$ -N tgif-dl
#$ -l h_rt=6:00:00

cd $1
xargs -n 1 curl -O -L --keepalive-time 60 < ./../tgif/data/splits/($1).txt
