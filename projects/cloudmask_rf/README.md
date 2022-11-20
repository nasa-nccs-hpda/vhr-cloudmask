# Cloud Mask RF

For the classification of clouds with the Random Forest we use
the rf_pipeline.py script.

## Requirements

The following are the requirements to run the script:
- CSV with data points in the format
```bash
```
- xxx

## Preprocessing

## Training

## Inference

## References



python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1 --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict --gpu



/explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/analysis


python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/analysis --rasters '/explore/nobackup/projects/3sl/data/Tappan/*_data.tif' --step predict --gpu --output-probabilities

python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/analysis2 --rasters '/explore/nobackup/projects/3sl/data/Tappan/Tappan24_WV03_20171001_M1BS_1040010034837400_data.tif' --step predict --gpu --output-probabilities


python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /explore/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/analysis --rasters '/explore/nobackup/projects/3sl/data/Tappan/Tappan04_WV02_20130921_M1BS_103001002706B900_data.tif' --step predict --gpu --output-probabilities