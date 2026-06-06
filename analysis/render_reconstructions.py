from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import VAE, VAEConfig
from shapley import FeatureBlockIndex
from utils import load_dataset_bundle, transform_features


PIX_SHAPE = (16, 15)


def resolve_model_paths(runs_dir: Path, run_ids: list[str]) -> list[Path]:
    if run_ids:
        model_paths: list[Path] = []
        for run_id in run_ids:
            candidate = Path(run_id)
            if candidate.is_file():
                model_paths.append(candidate)
                continue
            if candidate.is_dir():
                model_paths.extend(sorted(candidate.rglob("model.pt")))
                continue

            run_dir = runs_dir / run_id
            model_path = run_dir / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Could not find model.pt for run id {run_id!r} at {model_path}."
                )
            model_paths.append(model_path)
    else:
        model_paths = sorted(path for path in runs_dir.rglob("model.pt") if path.is_file())

    if not model_paths:
        raise FileNotFoundError(f"No model.pt files found under {runs_dir}.")
    return model_paths


def make_split_indices(
    n_rows: int,
    test_size: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_test = int(n_rows * test_size)
    n_test = max(1, min(n_rows - 1, n_test))
    generator = torch.Generator().manual_seed(int(split_seed))
    permutation = torch.randperm(n_rows, generator=generator).numpy()
    return permutation[n_test:].astype(int), permutation[:n_test].astype(int)


def load_metadata(model_path: Path) -> dict[str, object]:
    metadata_path = model_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected metadata next to model file: {metadata_path}"
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def model_label(metadata: dict[str, object], fallback: str) -> str:
    training_type = metadata.get("training_type")
    tactic = metadata.get("shapley_tactic")
    if training_type == "shapley" and tactic:
        return f"shapley-{tactic}"
    if isinstance(training_type, str) and training_type:
        return training_type
    return fallback


def scaled_to_raw(
    x_scaled: np.ndarray,
    *,
    mean: np.ndarray | None,
    scale: np.ndarray | None,
) -> np.ndarray:
    if mean is None or scale is None:
        return x_scaled
    return x_scaled * scale + mean


def pix_image(pix_vector: np.ndarray) -> np.ndarray:
    if pix_vector.size != PIX_SHAPE[0] * PIX_SHAPE[1]:
        raise ValueError(f"Expected 240 pixel features, got {pix_vector.size}.")
    return pix_vector.reshape(PIX_SHAPE)


def save_pair_image(
    *,
    input_pix: np.ndarray,
    recon_pix: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    vmin = float(min(input_pix.min(), recon_pix.min()))
    vmax = float(max(input_pix.max(), recon_pix.max()))
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    for ax, image, name in zip(axes, (input_pix, recon_pix), ("input", "reconstruction")):
        ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_grid_image(
    *,
    input_images: list[np.ndarray],
    recon_images: list[np.ndarray],
    titles: list[str],
    out_path: Path,
) -> None:
    n_rows = len(input_images)
    fig, axes = plt.subplots(n_rows, 2, figsize=(4.8, 2.0 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    for row_idx, (input_pix, recon_pix, title) in enumerate(
        zip(input_images, recon_images, titles)
    ):
        vmin = float(min(input_pix.min(), recon_pix.min()))
        vmax = float(max(input_pix.max(), recon_pix.max()))
        axes[row_idx, 0].imshow(input_pix, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(title, rotation=0, ha="right", va="center")
        axes[row_idx, 1].imshow(recon_pix, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].axis("off")
    axes[0, 0].set_title("input")
    axes[0, 1].set_title("reconstruction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_digit_model_grid(
    *,
    input_images: list[np.ndarray],
    recon_images_by_model: list[list[np.ndarray]],
    model_titles: list[str],
    sample_titles: list[str],
    out_path: Path,
) -> None:
    image_rows = [input_images, *recon_images_by_model]
    row_titles = ["raw", *model_titles]
    n_rows = len(image_rows)
    n_samples = len(sample_titles)
    fig, axes = plt.subplots(
        n_rows,
        n_samples,
        figsize=(1.36 * n_samples + 2.2, 1.45 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.2, right=0.995, top=0.92, bottom=0.02, wspace=0.04, hspace=0.08)
    values = [image for row_images in image_rows for image in row_images]
    vmin = float(min(image.min() for image in values))
    vmax = float(max(image.max() for image in values))
    for row_idx, (model_title, row_images) in enumerate(zip(row_titles, image_rows)):
        row_box = axes[row_idx, 0].get_position()
        fig.text(
            0.19,
            row_box.y0 + row_box.height / 2,
            model_title,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="semibold",
        )
        for col_idx, image in enumerate(row_images):
            axes[row_idx, col_idx].imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    sample_titles[col_idx],
                    rotation=0,
                    ha="center",
                    fontsize=8,
                )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def remove_stale_outputs(out_dir: Path) -> None:
    for pattern in ("pair_*.png", "digit_*.png", "reconstruction_pairs.png"):
        for path in out_dir.glob(pattern):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_ids",
        nargs="*",
        help=(
            "Run IDs under --runs-dir, run directories, or model.pt paths. "
            "Omit to render every model.pt under --runs-dir."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("analysis/output/training_runs"),
        help="Directory containing training run subfolders.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/pictures/reconstruction"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducibility seed. Omit it for fresh random samples each run.",
    )
    parser.add_argument(
        "--samples-per-digit",
        type=int,
        default=10,
        help="Number of validation examples to include in each digit composite.",
    )
    args = parser.parse_args()

    if args.samples_per_digit < 1:
        raise ValueError("--samples-per-digit must be >= 1.")

    model_paths = resolve_model_paths(args.runs_dir, args.run_ids)
    model_metadata = [(path, load_metadata(path)) for path in model_paths]
    metadata = model_metadata[0][1]
    config = metadata["config"]
    data = metadata["data"]

    bundle = load_dataset_bundle(data_dir=args.data_dir)
    block_index = FeatureBlockIndex.from_feature_groups(bundle.feature_groups)
    pix_block = block_index.blocks[block_index.names.index("pix")]

    scaler_mean = data.get("scaler_mean")
    scaler_scale = data.get("scaler_scale")
    mean = None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float32)
    scale = None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float32)
    x_scaled = torch.from_numpy(
        transform_features(
            bundle.x_raw,
            None,
        )
        if mean is None or scale is None
        else ((bundle.x_raw.astype("float32") - mean) / scale).astype("float32")
    )

    _, validation_idx = make_split_indices(
        n_rows=bundle.x_raw.shape[0],
        test_size=float(config.get("test_size", 0.2)),
        split_seed=int(config.get("split_seed", 555)),
    )
    if bundle.sample_labels is None:
        raise ValueError("Digit labels are required to sample one validation image per digit.")

    labels = np.asarray(bundle.sample_labels)
    rng = np.random.default_rng(args.seed)
    selected_by_digit: dict[int, np.ndarray] = {}
    for digit in range(10):
        digit_candidates = validation_idx[labels[validation_idx] == digit]
        if digit_candidates.size < args.samples_per_digit:
            raise ValueError(
                "Validation split has only "
                f"{digit_candidates.size} samples for digit {digit}; "
                f"need {args.samples_per_digit}."
            )
        selected_by_digit[digit] = rng.choice(
            digit_candidates,
            size=args.samples_per_digit,
            replace=False,
        ).astype(int)
    selected_array = np.concatenate([selected_by_digit[digit] for digit in range(10)])
    x_raw_reference: np.ndarray | None = None

    recon_by_model: list[np.ndarray] = []
    model_titles: list[str] = []
    for model_path, metadata in model_metadata:
        config = metadata["config"]
        data = metadata["data"]
        scaler_mean = data.get("scaler_mean")
        scaler_scale = data.get("scaler_scale")
        mean = None if scaler_mean is None else np.asarray(scaler_mean, dtype=np.float32)
        scale = None if scaler_scale is None else np.asarray(scaler_scale, dtype=np.float32)
        x_scaled = torch.from_numpy(
            transform_features(
                bundle.x_raw,
                None,
            )
            if mean is None or scale is None
            else ((bundle.x_raw.astype("float32") - mean) / scale).astype("float32")
        )

        model = VAE(
            VAEConfig(
                input_dim=int(data.get("n_features", x_scaled.shape[1])),
                hidden_dims=tuple(int(v) for v in config["hidden_dims"]),
                latent_dim=int(config["latent_dim"]),
                input_dropout=0.0,
                deterministic_latent=True,
                output_group_sizes=tuple(int(v) for v in block_index.sizes),
            )
        )
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            x_batch = x_scaled[selected_array]
            mu, _ = model.encoder(x_batch)
            x_hat = model.decoder(mu).cpu().numpy()

        if x_raw_reference is None:
            x_raw_reference = scaled_to_raw(x_batch.cpu().numpy(), mean=mean, scale=scale)
        x_hat_raw = scaled_to_raw(x_hat, mean=mean, scale=scale)
        recon_by_model.append(x_hat_raw)
        model_titles.append(model_label(metadata, model_path.parent.name))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(args.out_dir)
    saved_paths: list[Path] = []
    batch_position = {int(row_idx): out_idx for out_idx, row_idx in enumerate(selected_array)}
    if x_raw_reference is None:
        raise RuntimeError("No models were loaded.")
    for digit in range(10):
        input_images: list[np.ndarray] = []
        recon_images_by_model: list[list[np.ndarray]] = []
        sample_titles: list[str] = []
        for row_idx in selected_by_digit[digit]:
            out_idx = batch_position[int(row_idx)]
            input_images.append(
                pix_image(x_raw_reference[out_idx, pix_block.start : pix_block.stop])
            )
            sample_titles.append(f"sample {int(row_idx)}")
        for x_hat_raw in recon_by_model:
            recon_images: list[np.ndarray] = []
            for row_idx in selected_by_digit[digit]:
                out_idx = batch_position[int(row_idx)]
                recon_images.append(
                    pix_image(x_hat_raw[out_idx, pix_block.start : pix_block.stop])
                )
            recon_images_by_model.append(recon_images)
        out_path = args.out_dir / f"digit_{digit}_reconstructions.png"
        save_digit_model_grid(
            input_images=input_images,
            recon_images_by_model=recon_images_by_model,
            model_titles=model_titles,
            sample_titles=sample_titles,
            out_path=out_path,
        )
        saved_paths.append(out_path)

    print(f"models={len(model_paths)}")
    for path, title in zip(model_paths, model_titles):
        print(f"model={title}\t{path}")
    print(f"saved_reconstructions={args.out_dir}")
    for digit in range(10):
        sample_ids = ",".join(str(v) for v in selected_by_digit[digit])
        print(f"digit_{digit}_validation_samples={sample_ids}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
