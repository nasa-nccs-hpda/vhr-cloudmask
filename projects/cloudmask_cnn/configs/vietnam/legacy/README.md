# Vietnam Cloud Masking

## 1. Labels Creation

We created the labels in small squares of 5000x5000 with corresponding data and label files. Original
labels are located under: /att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Mary/cloud_labels

## 2. Preprocessing Labels

Some of the labels did not come in the format expected by the model. For example, some labels include classes
from 0-4, which includes clouds, thin-clouds, shadows, and not-clouds. Our first step before generating the
dataset and training the model is to preprocess these label files.

```bash
python fix_labels.py --image-regex 'images/*.tif' --label-regex 'labels/*.tif' --output-dir labels_processed
```

## 3. Print Configuration File

Generate Configuration Options

```bash
dl_pipeline fit --data=SegmentationDataModule --model=UNetSegmentation --print_config > vietnam.yml
```

## 4. Training and Preprocessing

```bash
dl_pipeline fit --config vietnam.yml
```

## 5. Testing

## 6. Prediction

## References
