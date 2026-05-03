from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class TabularDatasetBundle:
    dataset_name: str
    x_raw: np.ndarray
    feature_names: list[str]
    feature_groups: list[str]
    sample_labels: np.ndarray | None = None
    label_name: str | None = None
    scaler: StandardScaler | None = None
    x_scaled: torch.Tensor | None = None


WineDataBundle = TabularDatasetBundle


def _resolve_dataset_dir(
    data_dir: str | Path, dataset_name: str, required_files: list[str]
) -> Path:
    root = Path(data_dir)
    if all((root / file_name).exists() for file_name in required_files):
        return root
    candidate = root / dataset_name
    if all((candidate / file_name).exists() for file_name in required_files):
        return candidate
    raise FileNotFoundError(
        f"Could not resolve dataset '{dataset_name}' under {root}."
    )


def fit_feature_scaler(x_train_raw: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x_train_raw.astype("float32", copy=False))
    return scaler


def transform_features(
    x_raw: np.ndarray, scaler: StandardScaler | None = None
) -> np.ndarray:
    x_raw = x_raw.astype("float32", copy=False)
    if scaler is None:
        return x_raw.copy()
    return scaler.transform(x_raw).astype("float32", copy=False)


def denormalize_features(
    x_scaled: np.ndarray, scaler: StandardScaler | None
) -> np.ndarray:
    if scaler is None:
        return x_scaled
    return scaler.inverse_transform(x_scaled)


def _attach_full_dataset_scaling(bundle: TabularDatasetBundle) -> TabularDatasetBundle:
    scaler = fit_feature_scaler(bundle.x_raw)
    bundle.scaler = scaler
    bundle.x_scaled = torch.from_numpy(transform_features(bundle.x_raw, scaler))
    return bundle


def load_wine_raw_bundle(data_dir: str | Path) -> TabularDatasetBundle:
    dataset_dir = _resolve_dataset_dir(
        data_dir,
        dataset_name="wine-quality",
        required_files=["winequality-red.csv", "winequality-white.csv"],
    )
    red = pd.read_csv(dataset_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(dataset_dir / "winequality-white.csv", sep=";")
    red["wine_type"] = "red"
    white["wine_type"] = "white"
    both = pd.concat((red, white), axis=0, ignore_index=True)
    feature_frame = both.drop(columns=["quality", "wine_type"])
    return TabularDatasetBundle(
        dataset_name="wine-quality",
        x_raw=feature_frame.to_numpy(dtype="float32", copy=True),
        feature_names=list(feature_frame.columns),
        feature_groups=list(feature_frame.columns),
        sample_labels=both["wine_type"].to_numpy(copy=True),
        label_name="wine_type",
    )


def load_mfeat_raw_bundle(data_dir: str | Path) -> TabularDatasetBundle:
    feature_specs = [
        ("fou", "mfeat-fou"),
        ("fac", "mfeat-fac"),
        ("kar", "mfeat-kar"),
        ("pix", "mfeat-pix"),
        ("zer", "mfeat-zer"),
        ("mor", "mfeat-mor"),
    ]
    dataset_dir = _resolve_dataset_dir(
        data_dir,
        dataset_name="mfeat",
        required_files=[file_name for _, file_name in feature_specs],
    )

    frames: list[pd.DataFrame] = []
    feature_names: list[str] = []
    feature_groups: list[str] = []
    for group_name, file_name in feature_specs:
        frame = pd.read_csv(
            dataset_dir / file_name,
            sep=r"\s+",
            header=None,
            engine="python",
        )
        frame.columns = [f"{group_name}_{index:03d}" for index in range(frame.shape[1])]
        frames.append(frame)
        feature_names.extend(frame.columns.tolist())
        feature_groups.extend([group_name] * frame.shape[1])

    return TabularDatasetBundle(
        dataset_name="mfeat",
        x_raw=pd.concat(frames, axis=1).to_numpy(dtype="float32", copy=True),
        feature_names=feature_names,
        feature_groups=feature_groups,
        sample_labels=np.repeat(np.arange(10, dtype=np.int64), 200),
        label_name="digit_class",
    )


def load_dataset_bundle(
    data_dir: str | Path = "data",
    dataset_name: str = "mfeat",
    fit_scaler: bool = False,
) -> TabularDatasetBundle:
    normalized_name = dataset_name.strip().lower()
    if normalized_name in {"wine", "wine-quality", "wine_quality"}:
        bundle = load_wine_raw_bundle(data_dir)
    elif normalized_name == "mfeat":
        bundle = load_mfeat_raw_bundle(data_dir)
    else:
        raise ValueError(
            "Unknown dataset_name. Supported values: 'mfeat' and 'wine-quality'."
        )
    return _attach_full_dataset_scaling(bundle) if fit_scaler else bundle


def load_wine_bundle(data_dir: str | Path) -> WineDataBundle:
    return load_dataset_bundle(
        data_dir=data_dir,
        dataset_name="wine-quality",
        fit_scaler=True,
    )


def load_mfeat_bundle(data_dir: str | Path) -> TabularDatasetBundle:
    return load_dataset_bundle(data_dir=data_dir, dataset_name="mfeat", fit_scaler=True)


def load_dataset_features(
    data_dir: str | Path = "data",
    dataset_name: str = "mfeat",
) -> torch.Tensor:
    bundle = load_dataset_bundle(
        data_dir=data_dir,
        dataset_name=dataset_name,
        fit_scaler=True,
    )
    if bundle.x_scaled is None:
        raise RuntimeError("Scaled features are missing from the dataset bundle.")
    return bundle.x_scaled


def load_wine_features(data_dir: str | Path) -> torch.Tensor:
    return load_dataset_features(data_dir=data_dir, dataset_name="wine-quality")


def make_dataset_dataloader(
    data_dir: str | Path = "data",
    dataset_name: str = "mfeat",
    batch_size: int = 128,
    shuffle: bool = True,
) -> DataLoader:
    x = load_dataset_features(data_dir=data_dir, dataset_name=dataset_name)
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=shuffle)


def make_wine_dataloader(
    data_dir: str | Path, batch_size: int = 128, shuffle: bool = True
) -> DataLoader:
    return make_dataset_dataloader(
        data_dir=data_dir,
        dataset_name="wine-quality",
        batch_size=batch_size,
        shuffle=shuffle,
    )
