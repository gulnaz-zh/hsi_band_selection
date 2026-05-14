"""Command-line runner for 3D RGA hyperspectral band prioritization.

Example:
    python run_band_search.py \
        --dataset salinas \
        --cube Salinas_corrected.mat --cube-key salinas_corrected \
        --labels Salinas_gt.mat --labels-key salinas_gt \
        --class-id 1 --seed-row 80 --seed-col 60
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.dataset_presets import (
    DATASET_PRESETS,
    band_to_wavelength_nm,
    preprocess_cube_for_dataset,
    validate_dataset_shape,
)
from src.hsi_region_growing import (
    aggregate_selected_bands,
    class_mask,
    contiguous_ranges,
    find_informative_bands,
    normalize_to_uint8,
    project_region_to_2d,
    region_grow_3d,
)


def load_mat_array(path: Path, key: str | None) -> np.ndarray:
    data = loadmat(path)
    if key is not None:
        if key not in data:
            available = ", ".join(k for k in data if not k.startswith("__"))
            raise KeyError(f"{key!r} not found in {path}. Available keys: {available}")
        return np.asarray(data[key])

    candidates = [(k, v) for k, v in data.items() if not k.startswith("__") and isinstance(v, np.ndarray)]
    if len(candidates) != 1:
        available = ", ".join(k for k, _ in candidates)
        raise ValueError(f"Please pass an explicit key for {path}. Available keys: {available}")
    return np.asarray(candidates[0][1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS),
        default="custom",
        help="Dataset preset for paper-specific preprocessing.",
    )
    parser.add_argument("--cube", required=True, type=Path, help="Path to .mat file containing HSI cube.")
    parser.add_argument("--cube-key", help="Variable name for the HSI cube inside --cube.")
    parser.add_argument("--labels", type=Path, help="Path to .mat file containing labels.")
    parser.add_argument("--labels-key", help="Variable name for labels inside --labels.")
    parser.add_argument("--class-id", type=int, help="Ground-truth class id to segment.")
    parser.add_argument("--seed-row", required=True, type=int, help="Seed row/y coordinate.")
    parser.add_argument("--seed-col", required=True, type=int, help="Seed column/x coordinate.")
    parser.add_argument("--band-start", type=int, default=0, help="First band to test, inclusive.")
    parser.add_argument("--band-stop", type=int, help="Last band to test, inclusive. Defaults to final band.")
    parser.add_argument("--out", type=Path, default=Path("outputs/band_scores.csv"))
    parser.add_argument("--no-normalize", action="store_true", help="Use raw cube values instead of uint8 scaling.")
    parser.add_argument("--python-region-grow", action="store_true", help="Disable the compiled Cython grower.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset]
    cube_key = args.cube_key or preset.default_cube_key
    labels_key = args.labels_key or preset.default_labels_key
    cube = load_mat_array(args.cube, cube_key)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube, got shape {cube.shape}.")
    validate_dataset_shape(cube, args.dataset)

    cube = preprocess_cube_for_dataset(cube, args.dataset)
    cube = cube if args.no_normalize else normalize_to_uint8(cube)
    stop = cube.shape[2] - 1 if args.band_stop is None else args.band_stop
    bands = range(args.band_start, stop + 1)

    use_cython = not args.python_region_grow and cube.dtype == np.uint8

    if args.labels is None:
        run_unlabelled_search(args, cube, bands, use_cython)
        return

    if args.class_id is None:
        raise ValueError("--class-id is required when --labels is provided.")

    labels = load_mat_array(args.labels, labels_key).squeeze()
    if labels.shape != cube.shape[:2]:
        raise ValueError(f"Label shape {labels.shape} does not match cube spatial shape {cube.shape[:2]}.")
    seed_label = int(labels[args.seed_row, args.seed_col])
    if seed_label != args.class_id:
        warnings.warn(
            f"Seed ({args.seed_row}, {args.seed_col}) has label {seed_label}, "
            f"but --class-id is {args.class_id}. Band scores will compare the grown "
            "region against a different class mask.",
            stacklevel=2,
        )

    target = class_mask(labels, args.class_id)
    result = find_informative_bands(
        cube,
        (args.seed_row, args.seed_col),
        target,
        bands=bands,
        use_cython=use_cython,
    )
    ranges = contiguous_ranges(result.selected_bands)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["band", "wavelength_nm", "nmi", "ari", "selected"])
        writer.writeheader()
        for score in result.scores:
            wavelength = band_to_wavelength_nm(score.band, args.dataset, cube.shape[2])
            writer.writerow(
                {
                    "band": score.band,
                    "wavelength_nm": "" if wavelength is None else f"{wavelength:.3f}",
                    "nmi": f"{score.nmi:.6f}",
                    "ari": f"{score.ari:.6f}",
                    "selected": int(score.selected),
                }
            )

    print(f"NMI threshold: {result.nmi_threshold:.4f}")
    print(f"ARI threshold: {result.ari_threshold:.4f}")
    print(f"Selected bands: {result.selected_bands}")
    print(f"Contiguous ranges: {ranges}")
    print(f"Band scores written to: {args.out}")

    if ranges:
        best_range = max(ranges, key=lambda item: item[1] - item[0])
        aggregated = aggregate_selected_bands(
            cube,
            (args.seed_row, args.seed_col),
            best_range,
            target_mask=target,
            use_cython=use_cython,
        )
        print(f"Aggregated range: {best_range}")
        print(f"ARC NMI/ARI: {aggregated.arc_nmi:.4f} / {aggregated.arc_ari:.4f}")
        print(f"MRR NMI/ARI: {aggregated.mrr_nmi:.4f} / {aggregated.mrr_ari:.4f}")


def run_unlabelled_search(args: argparse.Namespace, cube: np.ndarray, bands: range, use_cython: bool) -> None:
    """Run seed-band growth without ground truth, used for NERC-ARF-style data."""

    cube_mean = float(np.mean(cube))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["band", "wavelength_nm", "voxels", "projected_pixels"],
        )
        writer.writeheader()
        for band in bands:
            region = region_grow_3d(
                cube,
                (args.seed_row, args.seed_col, band),
                use_cython=use_cython,
                cube_mean=cube_mean,
            )
            projected = project_region_to_2d(region)
            wavelength = band_to_wavelength_nm(band, args.dataset, cube.shape[2])
            writer.writerow(
                {
                    "band": band,
                    "wavelength_nm": "" if wavelength is None else f"{wavelength:.3f}",
                    "voxels": int(np.sum(region)),
                    "projected_pixels": int(np.sum(projected)),
                }
            )

    print("No labels were provided, so NMI/ARI band selection was skipped.")
    print(f"Unlabelled per-band growth statistics written to: {args.out}")


if __name__ == "__main__":
    main()
