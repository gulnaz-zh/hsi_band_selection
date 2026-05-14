"""Export 3D RGA segmentation outputs for inspection in TuiView.

The script writes one multi-band GeoTIFF per seed band. Each GeoTIFF stores the
binary 3D region mask as a raster stack: rows and columns are the spatial
dimensions, and GeoTIFF bands are the HSI spectral dimension. TuiView can then be
used to inspect the grown region across bands.

Example:
    python visualize_tuiview.py \
        --dataset salinas \
        --cube Salinas_corrected.mat --cube-key salinas_corrected \
        --seed-row 282 --seed-col 16 \
        --seed-bands 0 25 50 75 100 \
        --out-dir outputs/tuiview_salinas \
        --open-tuiview
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

from run_band_search import load_mat_array
from src.dataset_presets import DATASET_PRESETS, preprocess_cube_for_dataset, validate_dataset_shape
from src.hsi_region_growing import normalize_to_uint8, project_region_to_2d, region_grow_3d


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
    parser.add_argument("--seed-row", required=True, type=int, help="Seed row/y coordinate.")
    parser.add_argument("--seed-col", required=True, type=int, help="Seed column/x coordinate.")
    parser.add_argument(
        "--seed-bands",
        nargs="+",
        required=True,
        type=int,
        help="One or more seed bands to grow and export.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tuiview"), help="Output directory.")
    parser.add_argument("--no-normalize", action="store_true", help="Use raw cube values instead of uint8 scaling.")
    parser.add_argument("--python-region-grow", action="store_true", help="Disable the compiled Cython grower.")
    parser.add_argument(
        "--open-tuiview",
        action="store_true",
        help="Open the generated region GeoTIFFs with the tuiview command.",
    )
    parser.add_argument(
        "--tuiview-command",
        default="tuiview",
        help="TuiView executable name or path. Defaults to 'tuiview'.",
    )
    return parser.parse_args()


def require_gdal():
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise SystemExit(
            "GDAL Python bindings are required to write Tuiview-compatible GeoTIFFs. "
            "Install TuiView/GDAL in a conda environment, for example: "
            "conda create -n tuiview-env -c conda-forge tuiview"
        ) from exc
    return gdal


def write_multiband_geotiff(path: Path, volume: np.ndarray, *, description: str) -> None:
    """Write ``volume`` shaped ``(height, width, bands)`` as a multi-band GeoTIFF."""

    gdal = require_gdal()
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError("Expected volume with shape (height, width, bands).")

    height, width, bands = volume.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        width,
        height,
        bands,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    if dataset is None:
        raise RuntimeError(f"Could not create {path}.")

    dataset.SetMetadataItem("TIFFTAG_IMAGEDESCRIPTION", description)
    for band_index in range(bands):
        raster_band = dataset.GetRasterBand(band_index + 1)
        raster_band.WriteArray(volume[:, :, band_index].astype(np.uint8, copy=False))
        raster_band.SetDescription(f"spectral_band_{band_index}")
        raster_band.SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None


def export_seed_band(
    cube: np.ndarray,
    *,
    seed_row: int,
    seed_col: int,
    seed_band: int,
    out_dir: Path,
    use_cython: bool,
    cube_mean: float,
) -> tuple[Path, Path]:
    region = region_grow_3d(
        cube,
        (seed_row, seed_col, seed_band),
        use_cython=use_cython,
        cube_mean=cube_mean,
    )
    projected = project_region_to_2d(region)

    region_path = out_dir / f"seed_r{seed_row}_c{seed_col}_b{seed_band}_region3d.tif"
    projection_path = out_dir / f"seed_r{seed_row}_c{seed_col}_b{seed_band}_projection2d.tif"

    write_multiband_geotiff(
        region_path,
        region.astype(np.uint8) * 255,
        description=f"3D RGA binary region for seed ({seed_row}, {seed_col}, {seed_band})",
    )
    write_multiband_geotiff(
        projection_path,
        projected[:, :, None].astype(np.uint8) * 255,
        description=f"2D projection of 3D RGA region for seed ({seed_row}, {seed_col}, {seed_band})",
    )
    return region_path, projection_path


def open_in_tuiview(command: str, paths: list[Path]) -> None:
    executable = shutil.which(command) if not Path(command).exists() else command
    if executable is None:
        raise SystemExit(f"Could not find TuiView command {command!r} on PATH.")
    subprocess.Popen([executable, *map(str, paths)])


def main() -> None:
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset]
    cube_key = args.cube_key or preset.default_cube_key
    cube = load_mat_array(args.cube, cube_key)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D HSI cube, got shape {cube.shape}.")
    validate_dataset_shape(cube, args.dataset)

    cube = preprocess_cube_for_dataset(cube, args.dataset)
    cube = cube if args.no_normalize else normalize_to_uint8(cube)

    for seed_band in args.seed_bands:
        if not 0 <= seed_band < cube.shape[2]:
            raise ValueError(f"Seed band {seed_band} is outside cube band range 0-{cube.shape[2] - 1}.")

    use_cython = not args.python_region_grow and cube.dtype == np.uint8
    cube_mean = float(np.mean(cube))
    generated_region_paths: list[Path] = []

    for seed_band in args.seed_bands:
        region_path, projection_path = export_seed_band(
            cube,
            seed_row=args.seed_row,
            seed_col=args.seed_col,
            seed_band=seed_band,
            out_dir=args.out_dir,
            use_cython=use_cython,
            cube_mean=cube_mean,
        )
        generated_region_paths.append(region_path)
        print(f"Wrote 3D region: {region_path}")
        print(f"Wrote 2D projection: {projection_path}")

    if args.open_tuiview:
        open_in_tuiview(args.tuiview_command, generated_region_paths)


if __name__ == "__main__":
    main()
