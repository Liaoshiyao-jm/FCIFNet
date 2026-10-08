from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


PAPER_COUNTS = {
    "Augsburg": (
        (146, 264, 21, 248, 52, 7, 23),
        (13361, 30065, 3830, 26609, 523, 1638, 1507),
    ),
    "Houston": (
        (198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187),
        (1053, 1064, 505, 1056, 1056, 143, 1072, 1053, 1059, 1036, 1054, 1041, 285, 247, 473),
    ),
    "MUUFL": (
        (1184, 216, 345, 91, 335, 23, 113, 313, 86, 58, 15),
        (22062, 4054, 6537, 1735, 6352, 443, 2120, 5927, 1299, 125, 254),
    ),
    "Trento": (
        (129, 125, 105, 154, 184, 122),
        (3905, 2778, 374, 8969, 10317, 3052),
    ),
}


class PatchDataset(Dataset):
    def __init__(self, hsi, lidar, labels, coordinates, patch_size=11, augment=False):
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("Patch size must be a positive odd integer")
        if hsi.ndim != 3:
            raise ValueError(f"Expected HSI array with three dimensions, got {hsi.shape}")
        if lidar.ndim == 2:
            lidar = lidar[..., None]
        if lidar.ndim != 3:
            raise ValueError(f"Expected LiDAR array with two or three dimensions, got {lidar.shape}")
        if hsi.shape[:2] != labels.shape or lidar.shape[:2] != labels.shape:
            raise ValueError("Data and label spatial dimensions do not match")
        self.margin = patch_size // 2
        self.patch_size = patch_size
        self.augment = augment
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        hsi = np.asarray(hsi, dtype=np.float32).transpose(2, 0, 1)
        lidar = np.asarray(lidar, dtype=np.float32).transpose(2, 0, 1)
        padding = ((0, 0), (self.margin, self.margin), (self.margin, self.margin))
        self.hsi = np.pad(hsi, padding, mode="reflect")
        self.lidar = np.pad(lidar, padding, mode="reflect")

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        row, column = self.coordinates[index]
        row += self.margin
        column += self.margin
        rows = slice(row - self.margin, row + self.margin + 1)
        columns = slice(column - self.margin, column + self.margin + 1)
        hsi = self.hsi[:, rows, columns]
        lidar = self.lidar[:, rows, columns]
        if self.augment:
            if random.random() < 0.5:
                turns = random.randint(1, 3)
                hsi = np.rot90(hsi, turns, axes=(1, 2))
                lidar = np.rot90(lidar, turns, axes=(1, 2))
            if random.random() < 0.5:
                hsi = np.flip(hsi, axis=1)
                lidar = np.flip(lidar, axis=1)
            if random.random() < 0.5:
                hsi = np.flip(hsi, axis=2)
                lidar = np.flip(lidar, axis=2)
        target = self.labels[row - self.margin, column - self.margin] - 1
        return (
            torch.from_numpy(np.ascontiguousarray(hsi)).unsqueeze(0),
            torch.from_numpy(np.ascontiguousarray(lidar)),
            torch.tensor(target, dtype=torch.long),
        )


def _array(path, key):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    content = loadmat(path)
    if key not in content:
        raise KeyError(f"Missing variable {key!r} in {path}")
    return content[key]


def _counts(labels, num_classes):
    return tuple(int(np.count_nonzero(labels == class_id)) for class_id in range(1, num_classes + 1))


def _validate_split(dataset, train_labels, test_labels):
    expected_train, expected_test = PAPER_COUNTS[dataset]
    if train_labels.shape != test_labels.shape:
        raise ValueError(f"{dataset} training and test label shapes differ")
    if np.any((train_labels > 0) & (test_labels > 0)):
        raise ValueError(f"{dataset} training and test labels overlap")
    actual_train = _counts(train_labels, len(expected_train))
    actual_test = _counts(test_labels, len(expected_test))
    if actual_train != expected_train or actual_test != expected_test:
        raise ValueError(
            f"{dataset} labels do not match the paper split: "
            f"train={actual_train}, test={actual_test}"
        )


def _official_labels(dataset, directory, seed):
    if dataset == "Augsburg":
        train_labels = _array(directory / "TrainImage.mat", "TrainImage").astype(np.int64)
        test_labels = _array(directory / "TestImage.mat", "TestImage").astype(np.int64)
    elif dataset == "Houston":
        train_labels = _array(directory / "TRLabel.mat", "TRLabel").astype(np.int64)
        test_labels = _array(directory / "TSLabel.mat", "TSLabel").astype(np.int64)
    elif dataset == "MUUFL":
        train_labels = _array(directory / "muufl_tr.mat", "training_map").astype(np.int64)
        test_labels = _array(directory / "muufl_ts.mat", "testing_map").astype(np.int64)
    else:
        labels = _array(directory / "GT_Trento.mat", "gt_trento").astype(np.int64)
        train_labels = np.zeros_like(labels)
        test_labels = labels.copy()
        generator = np.random.default_rng(seed)
        for class_id, count in enumerate(PAPER_COUNTS[dataset][0], start=1):
            coordinates = np.argwhere(labels == class_id)
            chosen = generator.choice(len(coordinates), size=count, replace=False)
            selected = coordinates[chosen]
            train_labels[selected[:, 0], selected[:, 1]] = class_id
            test_labels[selected[:, 0], selected[:, 1]] = 0
    _validate_split(dataset, train_labels, test_labels)
    return train_labels, test_labels


def _data_arrays(dataset, directory):
    if dataset == "Augsburg":
        return (
            _array(directory / "data_HS_LR.mat", "data_HS_LR"),
            _array(directory / "data_SAR_HR.mat", "data_SAR_HR"),
        )
    if dataset == "Houston":
        return _array(directory / "HSI.mat", "HSI"), _array(directory / "LiDAR.mat", "LiDAR")
    if dataset == "MUUFL":
        return _array(directory / "HSI.mat", "HSI"), _array(directory / "LiDAR.mat", "LiDAR")
    lidar_path = directory / "Lidar_Trento.mat"
    lidar_key = "lidar_trento"
    if not lidar_path.is_file():
        lidar_path = directory / "Lidar1_Trento.mat"
        lidar_key = "lidar1_trento"
    return _array(directory / "HSI_Trento.mat", "hsi_trento"), _array(lidar_path, lidar_key)


def load_data(dataset, root=Path("DATA"), batch_size=16, patch_size=11, workers=0, seed=42):
    if dataset not in PAPER_COUNTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    directory = Path(root) / dataset
    hsi, lidar = _data_arrays(dataset, directory)
    train_labels, test_labels = _official_labels(dataset, directory, seed)
    labels = train_labels + test_labels
    train_coordinates = np.argwhere(train_labels > 0)
    test_coordinates = np.argwhere(test_labels > 0)
    train_set = PatchDataset(hsi, lidar, labels, train_coordinates, patch_size, True)
    test_set = PatchDataset(hsi, lidar, labels, test_coordinates, patch_size, False)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    in_ch_hsi = int(hsi.shape[2])
    in_ch_lidar = 1 if lidar.ndim == 2 else int(lidar.shape[2])
    num_classes = len(PAPER_COUNTS[dataset][0])
    return train_loader, test_loader, in_ch_hsi, in_ch_lidar, num_classes
