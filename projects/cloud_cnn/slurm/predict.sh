#!/bin/bash
#SBATCH -t05-00:00:00 -c10 --mem-per-cpu=10240 -G1 -J ethiopia --export=ALL -q ilab
module load singularity

srun -G1 -n1 singularity exec --env PYTHONPATH="$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/vhr-cloudmask" \
    --nv -B $NOBACKUP,/lscratch,/explore/nobackup/people,/explore/nobackup/projects \
    /explore/nobackup/projects/ilab/containers/above-shrubs.2023.07 \
    python /explore/nobackup/people/jacaraba/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py \
    -c /explore/nobackup/people/jacaraba/development/vhr-cloudmask/projects/cloud_cnn/configs/production/cloud_mask_alaska_senegal_3sl_cas.yaml \
    -s predict
