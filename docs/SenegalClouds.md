# Alaska Clouds

## Dataset Location

- Data (Tappan Squares):
- Training Data:
- Training Labels:
- Validation Dataset:

## Artifacts Location

- Random Forest Models: /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models
- Random Forest Predictions:
- CNN Models:
- CNN Predictions:

## Random Forest

There are two main experiments that we want to try here. The first experiment is a
random forest model with 4 bands and data only from Senegal (1M points). The second
experiment is a random forest model with 4 bands and data from Senegal and Vietnam
(2M points). The data csv are outlined below.

Senegal-only Points: /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-clouds-dataset-rf-2022-01-4bands.csv
Senegal + Vietnam Points: /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-vietnam-clouds-rf-2022-01-4bands.csv

The experiments inference results are stored in the following location:

Senegal-only Predictions:
Senegal + Vietnam Predictions:

To preprocess and generate the Senegal dataset:

```bash
python rf_pipeline.py --data-csv senegal/senegal-clouds-dataset-rf-2022-01.csv --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-clouds-dataset-rf-2022-01-4bands.csv --bands B G R NIR1 --step preprocess
```

### Experiment #1: Vietnam-only Points

Predict

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/vietnam-clouds-dataset-rf-2022-01-4bands --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict
```

Predict V2

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Vietnam/random_forest/models/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1 --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict
```

maybe some points Tappan09_WV03_20190203_M1BS_1040010047CA0300_data_clouds



Validation

```bash
```

### Experiment #2: Senegal-only Points

Train

```bash
python rf_pipeline.py --data-csv senegal/senegal-clouds-dataset-rf-2022-01.csv --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-clouds-dataset-rf-2022-01-4bands.csv --bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-clouds-dataset-rf-2022-01-4bands.pkl --step train
```

Predict

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/senegal-clouds-dataset-rf-2022-01-4bands --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict
```

Validation

```bash
```

### Experiment #3: Senegal + Vietnam Points

Train

```bash
python rf_pipeline.py --data-csv senegal/senegal-clouds-dataset-rf-2022-01.csv --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-vietnam-clouds-rf-2022-01-4bands.csv --bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands.pkl --step train
```

Train V2

```bash
python rf_pipeline.py --train-csv /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/datasets/senegal-vietnam-clouds-rf-2022-01-4bands.csv --bands B G R NIR1 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --step train --gpu
```

Predict

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict
```

Predict V2

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-BGRNIR1 --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict --gpu
```

Predict (NO POSTPROCESS)

```bash
python rf_pipeline.py --bands B G R NIR1 --input-bands CB B G Y R RE NIR1 NIR2 --output-model /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/models/senegal-vietnam-clouds-dataset-rf-2022-01-4bands.pkl --output-dir /adapt/nobackup/projects/ilab/projects/VHRCloudMask/Senegal/random_forest/predictions/senegal-vietnam-clouds-dataset-rf-2022-01-4bands-nopostprocess --rasters '/adapt/nobackup/projects/ilab/projects/Senegal/Tapan_data/*.tif' --step predict
```

Validation

```bash
python validation_senegal.py --image-regex '/Users/jacaraba/Desktop/holidays_work/senegal/validation_more_noclouds/*.gpkg' --label-regex '/Users/jacaraba/Desktop/holidays_work/senegal/senegal-vietnam-clouds-dataset-rf-2022-01-4bands' --output-dir /Users/jacaraba/Desktop/holidays_work/senegal/validation_scores
```

## Senegal Clouds Documentation

| Tappan Square | Notes |
| ------------- | ----- |
Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif | tiny clouds |
Tappan01_WV02_20121014_M1BS_103001001B793900_data.tif | tiny clouds |
Tappan01_WV02_20130414_M1BS_103001001F227000_data.tif | tiny to no clouds |
Tappan01_WV02_20161026_M1BS_103001005E913900_data.tif | tiny to no clouds |
Tappan01_WV02_20161026_M1BS_103001005FAB0500_data.tif | some thin clouds |
Tappan01_WV02_20181217_M1BS_1030010089CC6D00_data.tif | tiny to no clouds |
Tappan02_WV02_20120218_M1BS_103001001077BE00_data.tif | tiny to no clouds |
Tappan02_WV02_20121014_M1BS_103001001B793900_data.tif | tiny clouds |
Tappan02_WV02_20181217_M1BS_1030010089CC6D00_data.tif | tiny to no clouds |
Tappan02_WV03_20160123_M1BS_10400100173A0B00_data.tif | tiny to no clouds |
Tappan02_WV03_20160123_M1BS_1040010018A59100_data.tif | tiny to no clouds |
Tappan04_WV02_20120218_M1BS_103001001077BE00_data.tif | tiny to no clouds |
Tappan04_WV02_20121014_M1BS_103001001B793900_data.tif | tiny to no clouds |
Tappan04_WV02_20130921_M1BS_103001002706B900_data.tif | big clouds, several clouds |
Tappan04_WV02_20131106_M1BS_10300100299DF600_data.tif | tiny to no clouds |
Tappan04_WV02_20181217_M1BS_1030010089CC6D00_data.tif | tiny to no clouds |
Tappan04_WV03_20160123_M1BS_10400100173A0B00_data.tif | tiny to no clouds |
Tappan04_WV03_20160123_M1BS_1040010018A59100_data.tif | tiny to no clouds |
Tappan05_WV02_20110207_M1BS_1030010008B55200_data.tif | tiny to no clouds |
Tappan05_WV02_20110430_M1BS_103001000A27E100_data.tif | tiny to no clouds |
Tappan05_WV02_20181217_M1BS_1030010089CC6D00_data.tif | tiny to no clouds |
Tappan06_WV02_20120229_M1BS_1030010011AAC100_data.tif | tiny to no clouds |
Tappan06_WV02_20131106_M1BS_1030010029099500_data.tif | tiny to no clouds |
Tappan06_WV02_20131206_M1BS_10300100291F4300_data.tif | tiny to no clouds |
Tappan06_WV02_20131209_M1BS_103001002A192000_data.tif | tiny to no clouds |
Tappan06_WV02_20180111_M1BS_1030010076791B00_data.tif | tiny to no clouds |
Tappan06_WV02_20180122_M1BS_103001007854AE00_data.tif | some thin clouds |
Tappan06_WV02_20180122_M1BS_1030010078C30A00_data.tif | tiny to no clouds |
Tappan06_WV02_20190207_M1BS_103001008ACFCA00_data.tif | some thin clouds |
Tappan06_WV02_20190207_M1BS_103001008C679000_data.tif | some thin clouds |
Tappan06_WV03_20151209_M1BS_1040010014076300_data.tif | tiny to no clouds |
Tappan06_WV03_20151209_M1BS_10400100156ADA00_data.tif | tiny to no clouds |
Tappan06_WV03_20180509_M1BS_104001003C633600_data.tif | tiny to no clouds |
Tappan07_WV02_20120229_M1BS_1030010011AAC100_data.tif | tiny to no clouds |
Tappan07_WV02_20131106_M1BS_1030010029099500_data.tif | tiny to no clouds |
Tappan07_WV02_20180111_M1BS_1030010076791B00_data.tif | tiny to no clouds |
Tappan07_WV02_20180122_M1BS_103001007854AE00_data.tif | tiny to no clouds |
Tappan07_WV02_20180122_M1BS_1030010078C30A00_data.tif | tiny to no clouds |
Tappan07_WV02_20190207_M1BS_103001008ACFCA00_data.tif | tiny to no clouds |
Tappan07_WV02_20190207_M1BS_103001008C679000_data.tif | tiny to no clouds |
Tappan07_WV02_20191211_M1BS_103001009FB02300_data.tif | really thin clouds |
Tappan07_WV03_20151209_M1BS_1040010014076300_data.tif | tiny to no clouds |
Tappan07_WV03_20151209_M1BS_10400100156ADA00_data.tif | tiny to no clouds |
Tappan07_WV03_20180509_M1BS_104001003C633600_data.tif | tiny to no clouds |
Tappan09_WV02_20110121_M1BS_1030010009AC3C00_data.tif | some clouds |
Tappan09_WV02_20110427_M1BS_103001000A0E4100_data.tif | tiny to no clouds |
Tappan09_WV02_20150516_M1BS_10300100429F9400_data.tif | tiny to no clouds |
Tappan09_WV02_20160217_M1BS_10300100511E0500_data.tif | tiny to no clouds |
Tappan09_WV02_20160629_M1BS_10300100589A8F00_data.tif | tiny to no clouds |
Tappan09_WV02_20160629_M1BS_103001005A85B300_data.tif | tiny to no clouds |
Tappan09_WV02_20170211_M1BS_10300100621AA800_data.tif | several clouds |
Tappan09_WV02_20170211_M1BS_1030010062ABC400_data.tif | several clouds |
Tappan09_WV02_20190529_M1BS_1030010092488200_data.tif | tiny to no clouds |
Tappan09_WV02_20190529_M1BS_10300100937EBB00_data.tif | tiny to no clouds |
Tappan09_WV02_20191127_M1BS_103001009D5BF100_data.tif | tiny to no clouds |
Tappan09_WV03_20170530_M1BS_104001002D825600_data.tif | tiny to no clouds |
Tappan09_WV03_20170530_M1BS_104001002EBA0900_data.tif | tiny to no clouds |
Tappan09_WV03_20190203_M1BS_1040010047CA0300_data.tif | fully cloudy |
Tappan09_WV03_20190319_M1BS_1040010048D74F00_data.tif | tiny to no clouds |
Tappan09_WV03_20190319_M1BS_104001004A1FF400_data.tif | tiny to no clouds |
Tappan09_WV03_20191202_M1BS_1040010053383600_data.tif | tiny to no clouds |
Tappan09_WV03_20191202_M1BS_1040010055C0E600_data.tif | tiny to no clouds |
Tappan10_WV02_20110522_M1BS_103001000BC03B00_data.tif | tiny to no clouds |
Tappan10_WV02_20131114_M1BS_1030010029969A00_data.tif | fully cloudy |
Tappan10_WV02_20141014_M1BS_1030010038A78F00_data.tif | tiny to no clouds |
Tappan10_WV02_20160307_M1BS_1030010053556100_data.tif | tiny to no clouds |
Tappan10_WV02_20160629_M1BS_1030010058202900_data.tif | tiny to no clouds |
Tappan10_WV02_20160907_M1BS_103001005CAC1300_data.tif | fully cloudy |
Tappan10_WV02_20160910_M1BS_103001005C862800_data.tif | tiny to no clouds |
Tappan10_WV02_20161015_M1BS_103001005E358F00_data.tif | haze, cloudy |
Tappan10_WV02_20161021_M1BS_103001005EA9A000_data.tif | tiny to no clouds |
Tappan10_WV02_20161206_M1BS_1030010060A58600_data.tif | tiny to no clouds |
Tappan10_WV02_20170121_M1BS_1030010062444900_data.tif | tiny to no clouds |
Tappan10_WV02_20170217_M1BS_10300100659BEE00_data.tif | fully cloudy |
Tappan10_WV02_20170329_M1BS_1030010067C63500_data.tif | tiny to no clouds |
Tappan10_WV02_20180119_M1BS_1030010078814D00_data.tif | some thin clouds |
Tappan10_WV02_20180320_M1BS_103001007A9EE400_data.tif | tiny to no clouds |
Tappan10_WV02_20181214_M1BS_103001008853C300_data.tif | fully cloudy |
Tappan10_WV02_20181214_M1BS_103001008A2B1D00_data.tif | fully cloudy |
Tappan10_WV02_20190311_M1BS_103001008DA9FD00_data.tif | tiny to no clouds |
Tappan10_WV02_20190824_M1BS_10300100993D9D00_data.tif | fully cloudy |
Tappan10_WV02_20191216_M1BS_10300100A01C9100_data.tif | some thin clouds |
Tappan10_WV03_20150401_M1BS_104001000909AF00_data.tif | tiny to no clouds |
Tappan10_WV03_20150401_M1BS_1040010009A47A00_data.tif | tiny to no clouds |
Tappan11_WV02_20101014_M1BS_1030010007B7A900_data.tif | tiny to no clouds |
Tappan11_WV02_20111123_M1BS_103001000FA22C00_data.tif | tiny to no clouds |
Tappan11_WV02_20131114_M1BS_1030010029969A00_data.tif | fully thin clouds |
Tappan11_WV02_20160307_M1BS_1030010053556100_data.tif | tiny to no clouds |
Tappan11_WV02_20180320_M1BS_103001007A9EE400_data.tif | tiny to no clouds |
Tappan11_WV03_20151217_M1BS_104001001642F800_data.tif | tiny to no clouds |
Tappan11_WV03_20151223_M1BS_104001001569BE00_data.tif | tiny to no clouds |
Tappan11_WV03_20191018_M1BS_10400100520FD200_data.tif | tiny to no clouds |
Tappan11_WV03_20191018_M1BS_1040010052CE1E00_data.tif | tiny to no clouds |
Tappan12_WV02_20101105_M1BS_1030010007A0DE00_data.tif | tiny to no clouds |
Tappan12_WV02_20101116_M1BS_1030010007CC7300_data.tif | tiny to no clouds |
Tappan12_WV02_20140925_M1BS_103001003775F400_data.tif | fully cloudy |
Tappan12_WV02_20160304_M1BS_103001005051F600_data.tif | tiny to no clouds |
Tappan12_WV02_20160629_M1BS_1030010058202900_data.tif | tiny to no clouds |
Tappan12_WV02_20160907_M1BS_103001005CAC1300_data.tif | some clouds |
Tappan12_WV02_20160910_M1BS_103001005C862800_data.tif | tiny to no clouds |
Tappan12_WV02_20160918_M1BS_103001005D497600_data.tif | several clouds |
Tappan12_WV02_20160921_M1BS_103001005E0A2200_data.tif | tiny to no clouds |
Tappan12_WV02_20161015_M1BS_103001005E358F00_data.tif | tiny to no clouds |
Tappan12_WV02_20161021_M1BS_103001005EA9A000_data.tif | tiny to no clouds |
Tappan12_WV02_20161026_M1BS_103001005FD4F200_data.tif | fully cloudy, thin clouds |
Tappan12_WV02_20161109_M1BS_103001005F78BE00_data.tif | fully cloudy |
Tappan12_WV02_20161206_M1BS_1030010060A58600_data.tif | tiny to no clouds |
Tappan12_WV02_20170121_M1BS_1030010062444900_data.tif | tiny to no clouds |
Tappan12_WV02_20170211_M1BS_10300100621AA800_data.tif | several clouds |
Tappan12_WV02_20170211_M1BS_1030010062ABC400_data.tif | tiny to no clouds |
Tappan12_WV02_20170211_M1BS_103001006633DD00_data.tif | some thin clouds |
Tappan12_WV02_20170217_M1BS_10300100659BEE00_data.tif | several clouds |
Tappan12_WV02_20170329_M1BS_1030010067C63500_data.tif | tiny to no clouds |
Tappan12_WV02_20170428_M1BS_1030010069132600_data.tif | tiny to no clouds |
Tappan12_WV02_20170506_M1BS_10300100694D1200_data.tif | tiny to no clouds |
Tappan12_WV02_20180304_M1BS_103001007818FC00_data.tif | tiny to no clouds |
Tappan12_WV02_20181214_M1BS_103001008853C300_data.tif | several clouds |
Tappan12_WV02_20181214_M1BS_103001008A2B1D00_data.tif | several clouds |
Tappan12_WV02_20191216_M1BS_103001009F98AD00_data.tif | fully cloudy |
Tappan12_WV03_20150401_M1BS_104001000909AF00_data.tif | tiny to no clouds |
Tappan12_WV03_20150401_M1BS_1040010009A47A00_data.tif | tiny to no clouds |
Tappan12_WV03_20151217_M1BS_1040010015116A00_data.tif | tiny to no clouds |
Tappan12_WV03_20151223_M1BS_10400100157EED00_data.tif | tiny to no clouds |
Tappan12_WV03_20161031_M1BS_10400100224AF300_data.tif | tiny to no clouds |
Tappan12_WV03_20170102_M1BS_1040010027841C00_data.tif | some thin clouds |
Tappan12_WV03_20170127_M1BS_10400100285DBD00_data.tif | some thin clouds |
Tappan12_WV03_20170208_M1BS_10400100282C7400_data.tif | tiny to no clouds |
Tappan12_WV03_20170208_M1BS_104001002859FA00_data.tif | tiny to no clouds |
Tappan12_WV03_20170304_M1BS_10400100290E0D00_data.tif | tiny to no clouds |
Tappan12_WV03_20170411_M1BS_104001002C5FEE00_data.tif | fully thin clouds |
Tappan12_WV03_20190222_M1BS_1040010048538A00_data.tif | tiny to no clouds |
Tappan12_WV03_20191012_M1BS_10400100537D9100_data.tif | tiny to no clouds |

Selected for Validation

| Tappan Square | Notes |
| ------------- | ----- |
Tappan01_WV02_20110430_M1BS_103001000A27E100_data.tif | tiny clouds | good, no clouds - done
Tappan01_WV02_20121014_M1BS_103001001B793900_data.tif | tiny clouds | pretty good some clouds - done
Tappan01_WV02_20161026_M1BS_103001005FAB0500_data.tif | some thin clouds | good, no clouds - done
Tappan02_WV02_20121014_M1BS_103001001B793900_data.tif | tiny clouds | pretty good, some clouds - done
Tappan02_WV02_20181217_M1BS_1030010089CC6D00_data.tif | tiny to no clouds | - done
Tappan04_WV02_20131106_M1BS_10300100299DF600_data.tif | tiny to no clouds | - done
Tappan04_WV02_20130921_M1BS_103001002706B900_data.tif | big clouds, several clouds | missed some clouds, got others - done
Tappan05_WV02_20110207_M1BS_1030010008B55200_data.tif | tiny to no clouds | no clouds - done
Tappan07_WV02_20180111_M1BS_1030010076791B00_data.tif | tiny to no clouds | - done
Tappan09_WV02_20110121_M1BS_1030010009AC3C00_data.tif | some clouds | did okay - done
Tappan09_WV02_20170211_M1BS_1030010062ABC400_data.tif | several clouds | really bad, missed many - done
Tappan09_WV03_20190203_M1BS_1040010047CA0300_data.tif | fully cloudy | really good, lots of clouds -done
Tappan10_WV02_20170217_M1BS_10300100659BEE00_data.tif | fully cloudy | really good - done
Tappan10_WV02_20181214_M1BS_103001008A2B1D00_data.tif | fully cloudy | really good - done
Tappan11_WV03_20151223_M1BS_104001001569BE00_data.tif | tiny to no clouds | - done
Tappan11_WV02_20160307_M1BS_1030010053556100_data.tif | tiny to no clouds | - done

Tappan12_WV02_20140925_M1BS_103001003775F400_data.tif | fully cloudy | okay, some field - done
Tappan12_WV02_20160907_M1BS_103001005CAC1300_data.tif | some clouds | okay - done


Tappan12_WV02_20161109_M1BS_103001005F78BE00_data.tif | fully cloudy | okay, full
Tappan12_WV02_20170217_M1BS_10300100659BEE00_data.tif | several clouds | okay, some field
Tappan12_WV02_20181214_M1BS_103001008A2B1D00_data.tif | several clouds |
Tappan12_WV02_20191216_M1BS_103001009F98AD00_data.tif | fully cloudy |
Tappan12_WV03_20170102_M1BS_1040010027841C00_data.tif | some thin clouds |
Tappan12_WV03_20170127_M1BS_10400100285DBD00_data.tif | some thin clouds |
Tappan12_WV03_20170411_M1BS_104001002C5FEE00_data.tif | fully thin clouds |

# Training

visualize

Tappan12_WV02_20170217_M1BS_10300100659BEE00_data - picture



maybe
Tappan12_WV02_20191216_M1BS_103001009F98AD00_data.tif | fully cloudy |
