from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


@dataclass(slots=True)
class WineDataBundle:
    x_scaled: torch.Tensor
    x_raw: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler


def load_wine_bundle(data_dir: str | Path) -> WineDataBundle:
    data_dir = Path(data_dir)
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    both = pd.concat((red, white), axis=0, ignore_index=True)
    feature_frame = both.drop(columns=["quality"])
    x_raw = feature_frame.to_numpy(dtype="float32", copy=True)
    scaler = StandardScaler().fit(x_raw)
    x_scaled = scaler.transform(x_raw).astype("float32", copy=False)
    return WineDataBundle(
        x_scaled=torch.from_numpy(x_scaled),
        x_raw=x_raw,
        feature_names=list(feature_frame.columns),
        scaler=scaler,
    )


def denormalize_features(x_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(x_scaled)


def load_wine_features(data_dir: str | Path) -> torch.Tensor:
    return load_wine_bundle(data_dir).x_scaled


def make_wine_dataloader(
    data_dir: str | Path, batch_size: int = 128, shuffle: bool = True
) -> DataLoader[torch.Tensor]:
    x = load_wine_features(data_dir)
    return DataLoader(x, batch_size=batch_size, shuffle=shuffle)
