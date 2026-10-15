# FCIFNet

Official PyTorch implementation of FCIFNet for hyperspectral and LiDAR joint classification.

## Files

- `models/fcifnet.py`: network definition
- `models/train.py`: training and evaluation
- `data.py`: paper dataset splits and patch loading
- `requirements.txt`: Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Data

Place the original MAT files in the following structure:

```text
DATA/
  Trento/
    HSI_Trento.mat
    Lidar_Trento.mat
    GT_Trento.mat
  Houston/
    HSI.mat
    LiDAR.mat
    TRLabel.mat
    TSLabel.mat
  MUUFL/
    HSI.mat
    LiDAR.mat
    muufl_tr.mat
    muufl_ts.mat
  Augsburg/
    data_HS_LR.mat
    data_SAR_HR.mat
    TrainImage.mat
    TestImage.mat
```

## Training

```bash
python models/train.py --dataset Trento --data-root DATA --output runs/trento
```

Replace `Trento` with `Houston`, `MUUFL`, or `Augsburg` as needed. The default run uses 200 epochs and seed 0. Houston, MUUFL, and Augsburg use their supplied training and test label maps. Trento has only one ground-truth map, so the loader selects the paper's per-class training counts with seed 0 and uses the remaining labeled pixels for testing. The loader checks the per-class counts for every dataset before training. There is no alternative split option. Training writes the best checkpoint, epoch history, and final metrics to the output directory without printing each epoch.
