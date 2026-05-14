"""3D region-growing band prioritization for hyperspectral images.

This module reconstructs the method described in:

    "A region-growing approach for spectral band prioritization in
    hyperspectral remote sensing"

The implementation is intentionally dependency-light. It operates on NumPy arrays
with shape ``(height, width, bands)`` and uses the paper's local 6-neighbour 3D
region growing criterion:

    seed_value - 0.1 * global_mean < neighbour_value <
    seed_value + 0.1 * global_mean

The grown 3D region is projected to a 2D mask before being compared with a class
mask using NMI and ARI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:
    from RegionGrowth import RegionGrowHSI3D as _cython_region_grow_hsi_3d
except ImportError:  # pragma: no cover - depends on local extension build
    _cython_region_grow_hsi_3d = None


ArrayLikeSeed = tuple[int, int, int]


@dataclass(frozen=True)
class BandScore:
    """Segmentation quality obtained from one spectral seed band."""

    band: int
    nmi: float
    ari: float
    selected: bool


@dataclass(frozen=True)
class BandSearchResult:
    """Complete band-prioritization result for one seed/class."""

    selected_bands: list[int]
    scores: list[BandScore]
    nmi_threshold: float
    ari_threshold: float


@dataclass(frozen=True)
class AggregatedSegmentation:
    """Final segmentation masks from a selected informative band range."""

    bands: list[int]
    arc_mask: np.ndarray
    mrr_mask: np.ndarray
    arc_nmi: float | None = None
    arc_ari: float | None = None
    mrr_nmi: float | None = None
    mrr_ari: float | None = None


def normalize_to_uint8(cube: np.ndarray) -> np.ndarray:
    """Scale an HSI cube to uint8 while preserving relative intensities."""

    cube = np.asarray(cube, dtype=np.float32)
    finite = np.isfinite(cube)
    if not finite.any():
        raise ValueError("Input cube does not contain finite values.")
    low = float(np.nanmin(cube[finite]))
    high = float(np.nanmax(cube[finite]))
    if high <= low:
        return np.zeros(cube.shape, dtype=np.uint8)
    scaled = (cube - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.rint(scaled * 255).astype(np.uint8)


def region_grow_3d(
    cube: np.ndarray,
    seed: ArrayLikeSeed,
    *,
    tolerance_fraction: float = 0.1,
    mask: np.ndarray | None = None,
    use_cython: bool = True,
    cube_mean: float | None = None,
) -> np.ndarray:
    """Grow a 3D region from ``seed`` using 6-connected neighbours.

    Parameters
    ----------
    cube:
        Hyperspectral cube with shape ``(height, width, bands)``.
    seed:
        ``(row, col, band)`` seed coordinate.
    tolerance_fraction:
        Fraction of the global mean intensity used around the seed value. The
        experiments in the paper use ``0.1``.
    mask:
        Optional boolean volume restricting voxels that may be visited.
    use_cython:
        Use the compiled Cython implementation when available. If the extension
        is not built, or if a custom ``mask`` is supplied, the pure-Python
        fallback is used.
    cube_mean:
        Optional precomputed cube mean. This is useful when scanning many seed
        bands from the same cube.

    Returns
    -------
    np.ndarray
        Boolean 3D mask with the same shape as ``cube``.
    """

    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError("Expected cube with shape (height, width, bands).")

    height, width, bands = cube.shape
    row, col, band = map(int, seed)
    if not (0 <= row < height and 0 <= col < width and 0 <= band < bands):
        raise ValueError(f"Seed {seed!r} is outside cube shape {cube.shape!r}.")

    if use_cython and mask is None and _cython_region_grow_hsi_3d is not None:
        if cube.dtype != np.uint8:
            raise TypeError("The Cython region grower expects a uint8 cube. Use normalize_to_uint8 first.")
        contiguous = np.ascontiguousarray(cube)
        return _cython_region_grow_hsi_3d(
            contiguous,
            row,
            col,
            band,
            float(tolerance_fraction),
            -1.0 if cube_mean is None else float(cube_mean),
        ).astype(bool, copy=False)

    allowed = np.ones(cube.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if allowed.shape != cube.shape:
        raise ValueError("mask must have the same shape as cube.")
    if not allowed[row, col, band]:
        return np.zeros(cube.shape, dtype=bool)

    seed_value = float(cube[row, col, band])
    mean_intensity = float(np.mean(cube)) if cube_mean is None else float(cube_mean)
    tolerance = tolerance_fraction * mean_intensity
    lower = seed_value - tolerance
    upper = seed_value + tolerance

    output = np.zeros(cube.shape, dtype=bool)
    output[row, col, band] = True
    queue: deque[ArrayLikeSeed] = deque([(row, col, band)])

    while queue:
        y, x, z = queue.pop()
        for ny, nx, nz in (
            (y - 1, x, z),
            (y + 1, x, z),
            (y, x - 1, z),
            (y, x + 1, z),
            (y, x, z - 1),
            (y, x, z + 1),
        ):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and 0 <= nz < bands
                and allowed[ny, nx, nz]
                and not output[ny, nx, nz]
            ):
                value = float(cube[ny, nx, nz])
                if lower < value < upper:
                    output[ny, nx, nz] = True
                    queue.append((ny, nx, nz))

    return output


def project_region_to_2d(region: np.ndarray) -> np.ndarray:
    """Project a 3D region to the spatial plane by union across bands."""

    region = np.asarray(region, dtype=bool)
    if region.ndim != 3:
        raise ValueError("Expected a 3D region mask.")
    return np.any(region, axis=2)


def class_mask(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Create a binary class mask from a ground-truth label image."""

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("Expected 2D label image.")
    return labels == class_id


def normalized_mutual_information(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute NMI for two binary masks using the paper's entropy normalization."""

    a = np.asarray(mask_a, dtype=np.int8).ravel()
    b = np.asarray(mask_b, dtype=np.int8).ravel()
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes.")

    contingency = np.zeros((2, 2), dtype=np.float64)
    np.add.at(contingency, (a, b), 1)
    total = contingency.sum()
    if total == 0:
        return 0.0

    pxy = contingency / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])))

    def entropy(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    denom = entropy(px) + entropy(py)
    return 0.0 if denom == 0 else float((2.0 * mi) / denom)


def adjusted_rand_index(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute ARI for two binary masks without scikit-learn."""

    a = np.asarray(mask_a, dtype=np.int8).ravel()
    b = np.asarray(mask_b, dtype=np.int8).ravel()
    if a.shape != b.shape:
        raise ValueError("Masks must have identical shapes.")

    contingency = np.zeros((2, 2), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)

    def comb2(values: np.ndarray) -> np.ndarray:
        return values * (values - 1) / 2.0

    n = contingency.sum()
    if n < 2:
        return 0.0
    sum_comb = float(comb2(contingency).sum())
    row_comb = float(comb2(contingency.sum(axis=1)).sum())
    col_comb = float(comb2(contingency.sum(axis=0)).sum())
    total_comb = float(n * (n - 1) / 2.0)
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return 0.0 if denom == 0 else float((sum_comb - expected) / denom)


def score_band(
    cube: np.ndarray,
    spatial_seed: tuple[int, int],
    band: int,
    target_mask: np.ndarray,
    *,
    tolerance_fraction: float = 0.1,
    use_cython: bool = True,
    cube_mean: float | None = None,
) -> tuple[float, float, np.ndarray]:
    """Run 3D RGA at one band and score its projected mask."""

    region = region_grow_3d(
        cube,
        (int(spatial_seed[0]), int(spatial_seed[1]), int(band)),
        tolerance_fraction=tolerance_fraction,
        use_cython=use_cython,
        cube_mean=cube_mean,
    )
    projected = project_region_to_2d(region)
    return (
        normalized_mutual_information(projected, target_mask),
        adjusted_rand_index(projected, target_mask),
        projected,
    )


def find_informative_bands(
    cube: np.ndarray,
    spatial_seed: tuple[int, int],
    target_mask: np.ndarray,
    *,
    bands: Iterable[int] | None = None,
    absolute_threshold: float = 0.75,
    threshold_fraction_of_max: float = 0.75,
    tolerance_fraction: float = 0.1,
    use_cython: bool = True,
) -> BandSearchResult:
    """Search bands whose NMI and ARI exceed the paper's selection threshold.

    The default threshold is 0.75. If no band reaches it for a metric, the threshold
    for that metric is relaxed to 75% of the maximum score reached by that class.
    """

    cube = np.asarray(cube)
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != cube.shape[:2]:
        raise ValueError("target_mask must match the spatial dimensions of cube.")

    band_list = list(range(cube.shape[2])) if bands is None else [int(b) for b in bands]
    raw: list[tuple[int, float, float]] = []
    cube_mean = float(np.mean(cube))
    for band in band_list:
        nmi, ari, _ = score_band(
            cube,
            spatial_seed,
            band,
            target_mask,
            tolerance_fraction=tolerance_fraction,
            use_cython=use_cython,
            cube_mean=cube_mean,
        )
        raw.append((band, nmi, ari))

    max_nmi = max((nmi for _, nmi, _ in raw), default=0.0)
    max_ari = max((ari for _, _, ari in raw), default=0.0)
    nmi_threshold = absolute_threshold if max_nmi >= absolute_threshold else threshold_fraction_of_max * max_nmi
    ari_threshold = absolute_threshold if max_ari >= absolute_threshold else threshold_fraction_of_max * max_ari

    scores = [
        BandScore(band=band, nmi=nmi, ari=ari, selected=nmi >= nmi_threshold and ari >= ari_threshold)
        for band, nmi, ari in raw
    ]
    return BandSearchResult(
        selected_bands=[score.band for score in scores if score.selected],
        scores=scores,
        nmi_threshold=float(nmi_threshold),
        ari_threshold=float(ari_threshold),
    )


def contiguous_ranges(values: Sequence[int]) -> list[tuple[int, int]]:
    """Collapse sorted integer values into inclusive contiguous ranges."""

    if not values:
        return []
    sorted_values = sorted(set(map(int, values)))
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_values[0]
    for value in sorted_values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


def sample_equidistant_bands(start: int, stop: int, count: int = 10) -> list[int]:
    """Sample up to ``count`` equidistant integer bands from an inclusive range."""

    if start > stop:
        raise ValueError("start must be less than or equal to stop.")
    if count <= 0:
        raise ValueError("count must be positive.")
    if stop - start + 1 <= count:
        return list(range(start, stop + 1))
    return sorted(set(np.rint(np.linspace(start, stop, count)).astype(int).tolist()))


def aggregate_selected_bands(
    cube: np.ndarray,
    spatial_seed: tuple[int, int],
    selected_bands: Sequence[int] | tuple[int, int],
    *,
    target_mask: np.ndarray | None = None,
    count: int = 10,
    tolerance_fraction: float = 0.1,
    use_cython: bool = True,
) -> AggregatedSegmentation:
    """Aggregate regions from selected bands using ARC and MRR.

    ``selected_bands`` may be an explicit sequence or an inclusive ``(start, stop)``
    range. ARC marks pixels present in all projected regions. MRR marks pixels
    present in at least half of the projected regions, matching the paper's
    "at least five of ten" majority rule.
    """

    if len(selected_bands) == 2 and isinstance(selected_bands, tuple):
        bands = sample_equidistant_bands(int(selected_bands[0]), int(selected_bands[1]), count=count)
    else:
        bands = list(map(int, selected_bands))
    if not bands:
        raise ValueError("At least one selected band is required for aggregation.")

    projected_masks = []
    cube_mean = float(np.mean(cube))
    for band in bands:
        _, _, projected = score_band(
            cube,
            spatial_seed,
            band,
            np.zeros(cube.shape[:2], dtype=bool),
            tolerance_fraction=tolerance_fraction,
            use_cython=use_cython,
            cube_mean=cube_mean,
        )
        projected_masks.append(projected)

    stack = np.stack(projected_masks, axis=0)
    arc = np.all(stack, axis=0)
    mrr = np.sum(stack, axis=0) >= int(np.ceil(len(bands) / 2.0))

    if target_mask is None:
        return AggregatedSegmentation(bands=bands, arc_mask=arc, mrr_mask=mrr)

    target_mask = np.asarray(target_mask, dtype=bool)
    return AggregatedSegmentation(
        bands=bands,
        arc_mask=arc,
        mrr_mask=mrr,
        arc_nmi=normalized_mutual_information(arc, target_mask),
        arc_ari=adjusted_rand_index(arc, target_mask),
        mrr_nmi=normalized_mutual_information(mrr, target_mask),
        mrr_ari=adjusted_rand_index(mrr, target_mask),
    )
