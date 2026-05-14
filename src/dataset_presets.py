"""Dataset-specific preprocessing used in the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    default_cube_key: str | None
    default_labels_key: str | None
    raw_bands: int | None
    processed_bands: int | None
    wavelength_min_nm: float
    wavelength_max_nm: float
    class_count: int | None


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "custom": DatasetPreset("custom", None, None, None, None, 0.0, 0.0, None),
    "salinas": DatasetPreset("salinas", "salinas_corrected", "salinas_gt", 224, 204, 400.0, 2500.0, 16),
    "indian_pines": DatasetPreset("indian_pines", "indian_pines_corrected", "indian_pines_gt", 224, 204, 400.0, 2500.0, 16),
    "pavia_centre": DatasetPreset("pavia_centre", "pavia", "pavia_gt", 102, 102, 430.0, 860.0, 9),
    "nerc_arf": DatasetPreset("nerc_arf", None, None, 622, 207, 380.0, 2500.0, None),
}


AVIRIS_WATER_ABSORPTION_BANDS_1_BASED = (
    range(104, 109),
    range(150, 164),
    range(220, 221),
)


def aviris_water_absorption_indices_zero_based() -> list[int]:
    indices: list[int] = []
    for band_range in AVIRIS_WATER_ABSORPTION_BANDS_1_BASED:
        indices.extend(band - 1 for band in band_range)
    return indices


def remove_aviris_water_absorption_bands(cube: np.ndarray) -> np.ndarray:
    """Remove the 20 AVIRIS water-absorption bands reported for Salinas/Indian Pines."""

    if cube.shape[2] != 224:
        return cube
    return np.delete(cube, aviris_water_absorption_indices_zero_based(), axis=2)


def downsample_nerc_arf_bands(cube: np.ndarray, bin_size: int = 3) -> np.ndarray:
    """Average fixed non-overlapping spectral bins for the NERC-ARF cube.

    The paper reduces 622 bands to 207 bins by averaging triples and dropping the
    final leftover band.
    """

    if cube.shape[2] != 622:
        return cube
    usable = (cube.shape[2] // bin_size) * bin_size
    reshaped = cube[:, :, :usable].reshape(cube.shape[0], cube.shape[1], usable // bin_size, bin_size)
    return reshaped.mean(axis=3)


def preprocess_cube_for_dataset(cube: np.ndarray, dataset: str) -> np.ndarray:
    """Apply the paper's dataset-specific spectral preprocessing."""

    dataset = dataset.lower()
    if dataset in {"salinas", "indian_pines"}:
        return remove_aviris_water_absorption_bands(cube)
    if dataset == "nerc_arf":
        return downsample_nerc_arf_bands(cube)
    return cube


def validate_dataset_shape(cube: np.ndarray, dataset: str) -> None:
    """Raise a helpful error if a known dataset has an unexpected band count."""

    preset = DATASET_PRESETS[dataset]
    if preset.name == "custom" or preset.raw_bands is None or preset.processed_bands is None:
        return
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube for {dataset}, got shape {cube.shape}.")
    if cube.shape[2] not in {preset.raw_bands, preset.processed_bands}:
        raise ValueError(
            f"{dataset} is expected to have {preset.raw_bands} raw bands or "
            f"{preset.processed_bands} processed bands, got {cube.shape[2]}."
        )


def band_to_wavelength_nm(band: int, dataset: str, processed_band_count: int) -> float | None:
    """Approximate wavelength for a processed band index."""

    preset = DATASET_PRESETS[dataset]
    if preset.name == "custom" or processed_band_count <= 1:
        return None
    step = (preset.wavelength_max_nm - preset.wavelength_min_nm) / (processed_band_count - 1)
    return preset.wavelength_min_nm + band * step
