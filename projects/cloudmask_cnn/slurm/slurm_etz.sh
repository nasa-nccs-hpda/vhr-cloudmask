#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J 3sl --export=ALL
module load anaconda
conda activate ilab
export PYTHONPATH="/adapt/nobackup/people/jacaraba/development/tensorflow-caney"

#Run tasks sequentially without ‘&’
srun -G1 -n1 python /adapt/nobackup/people/jacaraba/development/vhr-cloudmask/projects/cloud_mask_cnn/scripts/predict.py \
	-c /adapt/nobackup/people/jacaraba/development/vhr-cloudmask/projects/cloud_mask_cnn/configs/cloud_mask_alaska_senegal_3sl_etz.yaml

