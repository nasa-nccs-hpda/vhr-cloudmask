# Cloud Mask RF

Segmentation of clouds with Random Forest. The script CLI contains details related to the
input needed to run a Random Forest model over cloud data with and without GPUs.

## Preprocessing

```bash
python rf_pipeline.py --data-csv senegal/senegal-clouds-dataset-rf-2022-01.csv --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-clouds-dataset-rf-2022-01-4bands.csv --bands B G R NIR1 --step preprocess
```

## Training

```bash
python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/labels/cloud_training_4band_rgb_fdi_si_ndwi.csv --step train --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl
```

## Inference

Segmentation Maps

```bash
python rf_pipeline.py --step predict --output-model /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/models/cloud_training_4band_rgb_fdi_si_ndwi.pkl --rasters '/att/pubrepo/ILAB/projects/Vietnam/Sarah/data/*.tif' --output-dir /adapt/nobackup/projects/ilab/projects/Vietnam/Jordan/Vietnam_CLOUD/article/random_forest/predictions/cloud_training_4band_rgb_fdi_si_ndwi
```

Output Probability Maps

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/analysis --rasters '/explore/nobackup/projects/3sl/data/Tappan/Tappan04_WV02_20130921_M1BS_103001002706B900_data.tif' --step predict --gpu --output-probabilities
```
