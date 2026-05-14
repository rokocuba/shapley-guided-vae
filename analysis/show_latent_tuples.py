from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import VAE, VAEConfig
from utils import load_wine_bundle


def _load_run(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return metadata


def _parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _inverse_with_saved_scaler(x_scaled: np.ndarray, metadata: dict) -> np.ndarray:
    mean = np.array(metadata["data"]["scaler_mean"], dtype=np.float32)
    scale = np.array(metadata["data"]["scaler_scale"], dtype=np.float32)
    return x_scaled * scale + mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=str, default="0,1,2")
    parser.add_argument("--out", type=Path, default=Path("analysis/output/tuple_viz"))
    args = parser.parse_args()

    meta = _load_run(args.run_dir)
    run_id = meta["run_id"]
    training_type = meta["training_type"]
    shapley_tactic = meta["shapley_tactic"] or "none"
    cfg = meta["config"]
    data_dir = Path(meta["data"]["data_dir"])
    feature_names = meta["data"]["feature_names"]
    row_ids = _parse_indices(args.rows)

    bundle = load_wine_bundle(data_dir)
    model = VAE(
        VAEConfig(
            input_dim=meta["data"]["n_features"],
            hidden_dims=tuple(cfg.get("hidden_dims", (16, 8))),
            latent_dim=cfg["latent_dim"],
            input_dropout=cfg["input_dropout"],
        )
    )
    model.load_state_dict(
        torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    out_dir = args.out / f"{training_type}__{shapley_tactic}__{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_rows: list[dict[str, object]] = []
    latent_rows: list[dict[str, object]] = []
    for row_idx in row_ids:
        x = bundle.x_scaled[row_idx : row_idx + 1]
        with torch.no_grad():
            x_hat, mu, _ = model(x)
        x_np = x.numpy()
        x_hat_np = x_hat.numpy()
        x_denorm = _inverse_with_saved_scaler(x_np, meta)[0]
        xhat_denorm = _inverse_with_saved_scaler(x_hat_np, meta)[0]

        for i, name in enumerate(feature_names):
            feature_rows.append(
                {
                    "run_id": run_id,
                    "training_type": training_type,
                    "shapley_tactic": shapley_tactic,
                    "row": row_idx,
                    "feature": name,
                    "input_denorm": float(x_denorm[i]),
                    "output_denorm": float(xhat_denorm[i]),
                }
            )

        latent_row = {f"z_{j}": float(v) for j, v in enumerate(mu[0].tolist())}
        latent_rows.append(
            {
                "run_id": run_id,
                "training_type": training_type,
                "shapley_tactic": shapley_tactic,
                "row": row_idx,
                **latent_row,
            }
        )

        plt.figure(figsize=(12, 4))
        idx = np.arange(len(feature_names))
        plt.plot(idx, x_denorm, marker="o", linewidth=1.4, label="input")
        plt.plot(idx, xhat_denorm, marker="x", linewidth=1.2, label="output")
        plt.xticks(idx, feature_names, rotation=35, ha="right")
        plt.ylabel("value (denormalized)")
        plt.title(
            f"Input vs Output (row={row_idx}, type={training_type}, tactic={shapley_tactic})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"{run_id}__row_{row_idx}_input_output_denorm.png", dpi=150
        )
        plt.close()

    feature_path = out_dir / f"{run_id}__features_denorm.csv"
    latent_path = out_dir / f"{run_id}__latent_vectors.csv"
    pd.DataFrame(feature_rows).to_csv(feature_path, index=False)
    pd.DataFrame(latent_rows).to_csv(latent_path, index=False)
    pd.concat(
        (
            pd.DataFrame(feature_rows),
            pd.DataFrame(latent_rows).assign(feature="__latent__"),
        ),
        ignore_index=True,
        sort=False,
    ).to_csv(out_dir / f"{run_id}__tuples_denorm.csv", index=False)
    print(f"saved_tuple_viz={out_dir}")


if __name__ == "__main__":
    main()
