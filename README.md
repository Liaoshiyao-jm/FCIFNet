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

Replace `Trento` with `Houston`, `MUUFL`, or `Augsburg` as needed. 
