"""Generate a toy 3D HSI cube and visualize 3D region-growing output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from src.hsi_region_growing import project_region_to_2d, region_grow_3d


def make_toy_cube() -> tuple[np.ndarray, tuple[int, int, int]]:
    height, width, bands = 80, 100, 12
    y, x = np.mgrid[:height, :width]
    cube = np.zeros((height, width, bands), dtype=np.uint8)

    for band in range(bands):
        cube[:, :, band] = 25 + band * 3 + ((x + y) % 12)

    object_mask = ((x - 48) ** 2 / 22**2 + (y - 40) ** 2 / 16**2) <= 1
    for band in range(4, 9):
        cube[object_mask, band] = 138 + (band - 6) * 2

    distractor_mask = ((x - 22) ** 2 / 9**2 + (y - 20) ** 2 / 7**2) <= 1
    cube[distractor_mask, 4:9] = 92

    return cube, (40, 48, 6)


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    data = slice_2d.astype(np.float32)
    low = float(data.min())
    high = float(data.max())
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.rint((data - low) / (high - low) * 255).astype(np.uint8)


def save_overlay(base: np.ndarray, mask: np.ndarray, seed: tuple[int, int], path: Path) -> None:
    rgb = np.dstack([base, base, base]).astype(np.uint8)
    rgb[mask, 0] = 235
    rgb[mask, 1] = np.maximum(rgb[mask, 1] // 3, 30)
    rgb[mask, 2] = np.maximum(rgb[mask, 2] // 3, 30)
    row, col = seed
    rgb[max(0, row - 2) : row + 3, max(0, col - 2) : col + 3] = [40, 190, 90]
    Image.fromarray(rgb).save(path)


def main() -> None:
    out_dir = Path("images")
    out_dir.mkdir(exist_ok=True)

    cube, seed = make_toy_cube()
    region = region_grow_3d(cube, seed)
    projected = project_region_to_2d(region)
    seed_array = np.array(seed, dtype=np.int16)

    savemat(
        out_dir / "toy_hsi_cube.mat",
        {
            "toy_hsi_cube": cube,
            "seed": seed_array,
            "region_3d": region.astype(np.uint8),
            "projected_region_2d": projected.astype(np.uint8),
        },
    )

    band_image = normalize_slice(cube[:, :, seed[2]])
    Image.fromarray(band_image).save(out_dir / "toy_input_band.png")

    seed_rgb = np.dstack([band_image, band_image, band_image])
    row, col, _ = seed
    seed_rgb[max(0, row - 2) : row + 3, max(0, col - 2) : col + 3] = [40, 190, 90]
    Image.fromarray(seed_rgb.astype(np.uint8)).save(out_dir / "toy_seed.png")

    save_overlay(band_image, projected, (row, col), out_dir / "toy_region_growing_output.png")


if __name__ == "__main__":
    main()
