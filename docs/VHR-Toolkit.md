# VHR Toolkit Development

We need specific data sources to perform regression testing on the
VHR Toolkit integration. The following are example sources to use
for development.

## Data for Testing

- Vietnam: /explore/nobackup/projects/ilab/projects/Vietnam/Sarah/data/Keelin00_20120130_data.tif
- Alaska: 
- Senegal: 

## Environment Setup

Container general command

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/vhr-cloudmask" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css/nga /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif pytest $NOBACKUP/development/vhr-cloudmask/tests
```


## Sprint #1

### Cloud Masking Testing

Vietnam data prediction:

```bash
singularity exec --env PYTHONPATH="$NOBACKUP/development/tensorflow-caney:$NOBACKUP/development/vhr-cloudmask" --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/css/nga /explore/nobackup/projects/ilab/containers/vhr-cloudmask.sif python $NOBACKUP/development/vhr-cloudmask/vhr_cloudmask/view/cloudmask_cnn_pipeline_cli.py -o '/explore/nobackup/projects/ilab/test/vhr-cloudmask' -r '/explore/nobackup/projects/ilab/projects/Vietnam/Sarah/data/Keelin00_20120130_data.tif' -s predict -ib B G R N G1 G2 -ob B G R N
```

### Cloud Shadow Testing

## Spring #2

