# Running Cloud Mask Software

## Building Singularity Sandbox

```bash
singularity build --sandbox /lscratch/jacaraba/container/tensorflow-caney docker://nasanccs/tensorflow-caney:latest
```

## Running Script Inside Singularity Container

```bash
singularity exec --nv -B /lscratch,/explore/nobackup/projects/3sl,/explore/nobackup/projects/ilab,$NOBACKUP /lscratch/jacaraba/container/tensorflow-caney python /explore/nobackup/people/jacaraba/development/vhr-cloudmask/projects/cloudmask_cnn/scripts/predict.py
```