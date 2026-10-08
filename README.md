# FCIFNet

Official PyTorch implementation of FCIFNet for hyperspectral and LiDAR joint classification.

## Files

- `fcifnet.py`: network definition
- `data.py`: paper dataset splits and patch loading
- `train.py`: training and evaluation
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

Each dataset automatically uses the single split reported in the paper. The
loader verifies every per-class training and test count before training.

## Training

```bash
python train.py --dataset Trento --data-root DATA --output runs/trento
```

Replace `Trento` with `Houston`, `MUUFL`, or `Augsburg` when needed. There is no
split-method option. Use `--device cpu` to force CPU training or `--device
cuda` to force CUDA. 

