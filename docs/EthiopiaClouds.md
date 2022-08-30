# Ethiopia Clouds

## Dataset Location

- Data (Woubet Squares): /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data
- Data (Woubet Squares) for validation: /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data_fixed
- Training Data:
- Training Labels:
- Validation Dataset:

## Artifacts Location

- Random Forest Models:
- Random Forest Predictions:
- CNN Models:
- CNN Predictions:

### Experiment #1: Senegal-only Points

Predict

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/random_forest/predictions/senegal-clouds-dataset-rf-2022-01-4bands --rasters '/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data_fixed/*.tif' --step predict
```

### Experiment #2: Senegal + Vietnam Points

Predict

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands --rasters '/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data_fixed/*.tif' --step predict
```

## Woubet Squares

WV02_20100204_M1BS_1030010003515A00-toa.tif - has clouds
WV02_20100207_M1BS_10300100043C5100-toa.tif - has clouds
WV02_20110911_M1BS_103001000DBF5700-toa.tif - has clouds
WV03_20201214_M1BS_10400100649F8100-toa.tif - big clouds
WV03_20190626_M1BS_104001004E7F5100-toa.tif - has clouds
WV03_20190305_M1BS_1040010049593100-toa.tif - big clouds

## Tiles with Clouds

WV02_20100207_M1BS_10300100043C5100-toa_0-5000 - some
WV02_20100207_M1BS_10300100043C5100-toa_0-10000 - fully cloudy
WV02_20100207_M1BS_10300100043C5100-toa_0-15000 - fully cloudy
WV02_20100207_M1BS_10300100043C5100-toa_0-20000 - some clouds
WV02_20100207_M1BS_10300100043C5100-toa_0-25000 - some clouds
WV02_20100207_M1BS_10300100043C5100-toa_0-30000 - some clouds
WV02_20100207_M1BS_10300100043C5100-toa_0-35000 - some clouds
WV02_20100207_M1BS_10300100043C5100-toa_0-40000 - some clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-0 - some clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-5000 - some clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-10000 - some small clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-15000 - some nice clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-20000 - some thin, nice clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-25000 - some thin clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-30000 - some clouds
WV02_20110911_M1BS_103001000DBF5700-toa_0-35000 - really cloudy
WV02_20110911_M1BS_103001000DBF5700-toa_0-40000 - fully cloudy
WV02_20110911_M1BS_103001000DBF5700-toa_0-45000 - lots of thin clouds
WV03_20151120_M1BS_10400100141FA900-toa_0-0 - some clouds
WV03_20160923_M1BS_1040010021454200-toa_0-0 - some clouds
WV03_20190305_M1BS_1040010049593100-toa_0-0 - really cloudy on top
WV03_20190626_M1BS_104001004E7F5100-toa_0-5000 - some clouds, lot of buildings
WV03_20190626_M1BS_104001004E7F5100-toa_0-10000 - some clouds, lot of buildings
WV03_20201214_M1BS_10400100649F8100-toa_0-15000 - no clouds
WV03_20201214_M1BS_10400100649F8100-toa_0-20000 - no clouds


## Predict Random Forest

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1 --rasters '/adapt/nobackup/projects/ilab/projects/VHRCloudMask/Ethiopia/data_fixed/*.tif' --step predict --gpu
```


Foto
WV02_20110911_M1BS_103001000DBF5700-toa_0-50000_clouds