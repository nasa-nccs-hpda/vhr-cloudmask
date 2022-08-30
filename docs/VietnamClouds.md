# Alaska Clouds

## Dataset Location

- Data (Keelin Squares):
- Training Data:
- Training Labels:
- Validation Dataset:

## Artifacts Location

- Random Forest Models:
- Random Forest Predictions:
- CNN Models:
- CNN Predictions:

# Vietnam Clouds

Training data 8-bands:
Training data 4-bands:
Training data 4-bands + 3 indices: /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv

## Random Forest

Train

4-bands

```bash
python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/datasets/cloud_training_4band_BGRNIR1.csv --bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands.pkl --step train
```

4-bands

```bash

/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/datasets/cloud_training_4band_BGRNIR1.csv

/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl


python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/datasets/cloud_training_4band_BGRNIR1.csv --bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --step train --gpu


```


4-bands + indices

```bash
python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv --step train --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl
```

Prediction

```bash
python rf_pipeline.py --step predict --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl --rasters '/att/pubrepo/ILAB/projects/Vietnam/Sarah/data/*.tif' --output-dir /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/predictions/cloud_training_4band_rgb_fdi_si_ndwi
```

Explainable AI

```bash
```


python rf_pipeline.py --bands B G R NIR1 --input-bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/predictions/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1-BIG --rasters '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/pansharpen_TamNongClip/*.tif' --window-size 8192 --step predict
