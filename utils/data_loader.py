from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def load_wine_features(data_dir: str | Path) -> torch.Tensor:
    data_dir = Path(data_dir)
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    both = pd.concat((red, white), axis=0, ignore_index=True)
    x = both.drop(columns=["quality"]).to_numpy(dtype="float32", copy=True)
    x = StandardScaler().fit_transform(x)
    return torch.from_numpy(x)


def make_wine_dataloader(
    data_dir: str | Path, batch_size: int = 128, shuffle: bool = True
) -> DataLoader[torch.Tensor]:
    x = load_wine_features(data_dir)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)
