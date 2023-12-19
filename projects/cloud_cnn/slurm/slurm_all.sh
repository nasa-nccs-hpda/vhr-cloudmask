#!/bin/bash

for i in {1..64}
do
    sbatch /explore/nobackup/people/jacaraba/development/vhr-cloudmask/projects/cloud_cnn/slurm/predict.sh;
    sleep 30s
done
