"""Render 3D RGA segmentation outputs with PyVista.

The script runs 3D region growing for one or more seed bands and renders each
grown 3D binary region as a perspective plot. Rows and columns are the spatial
dimensions, and the vertical axis is the spectral band index.

Example:
    python visualize_pyvista.py \
        --dataset salinas \
        --cube datasets/Salinas_corrected.mat \
        --seed-row 282 --seed-col 16 \
        --seed-bands 0 25 50 75 100 \
        --out-dir outputs/pyvista_salinas
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_band_search import load_mat_array
from src.dataset_presets import DATASET_PRESETS, preprocess_cube_for_dataset, validate_dataset_shape
from src.hsi_region_growing import normalize_to_uint8, region_grow_3d


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
        help="One or more seed bands to grow and render.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/pyvista"), help="Output directory.")
    parser.add_argument("--no-normalize", action="store_true", help="Use raw cube values instead of uint8 scaling.")
    parser.add_argument("--python-region-grow", action="store_true", help="Disable the compiled Cython grower.")
    parser.add_argument("--show", action="store_true", help="Open an interactive PyVista window while rendering.")
    parser.add_argument("--base-band", type=int, help="HSI band used as the grayscale base surface.")
    parser.add_argument("--region-opacity", type=float, default=1.0, help="Opacity of the colored region surfaces.")
    parser.add_argument("--base-opacity", type=float, default=0.85, help="Opacity of the grayscale base surface.")
    parser.add_argument(
        "--z-scale",
        type=float,
        default=2.0,
        help="Scale factor applied to the spectral axis in the 3D plot.",
    )
    return parser.parse_args()


def require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit(
            "PyVista is required for 3D visualization. Install it with: "
            "python -m pip install pyvista"
        ) from exc
    return pv


def image_surface(pv, image: np.ndarray, *, z: float):
    """Create a regular image plane whose scalar values shade the base scene."""

    height, width = image.shape
    rows, cols = np.mgrid[0:height, 0:width]
    surface = pv.StructuredGrid(
        cols.astype(np.float32),
        rows.astype(np.float32),
        np.full((height, width), z, dtype=np.float32),
    )
    surface["intensity"] = image.ravel(order="F").astype(np.float32)
    return surface


def mask_surface(pv, mask: np.ndarray, *, z: float):
    """Create a continuous surface from a 2D binary mask at one spectral slice."""

    height, width = mask.shape
    rows, cols = np.mgrid[0:height, 0:width]
    surface = pv.StructuredGrid(
        cols.astype(np.float32),
        rows.astype(np.float32),
        np.full((height, width), z, dtype=np.float32),
    )
    surface["mask"] = mask.ravel(order="F").astype(np.uint8)
    return surface.threshold(0.5, scalars="mask")


def render_region(
    cube: np.ndarray,
    region: np.ndarray,
    *,
    seed: tuple[int, int, int],
    base_band: int,
    color: str,
    out_path: Path,
    show: bool,
    region_opacity: float,
    base_opacity: float,
    z_scale: float,
) -> None:
    pv = require_pyvista()
    occupied_bands = np.flatnonzero(np.any(region, axis=(0, 1)))
    if occupied_bands.size == 0:
        raise ValueError(f"Region for seed {seed} is empty.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=not show, window_size=(1400, 900))
    plotter.set_background("white")

    base = image_surface(pv, cube[:, :, base_band], z=float(base_band) * z_scale)
    plotter.add_mesh(
        base,
        scalars="intensity",
        cmap="gray",
        opacity=base_opacity,
        show_scalar_bar=True,
        smooth_shading=False,
    )

    for band in occupied_bands:
        mesh = mask_surface(pv, region[:, :, int(band)], z=float(band) * z_scale)
        if mesh.n_cells == 0:
            continue
        plotter.add_mesh(
            mesh,
            color=color,
            opacity=region_opacity,
            show_edges=False,
            smooth_shading=False,
        )

    seed_row, seed_col, seed_band = seed
    seed_point = np.array([[seed_col, seed_row, seed_band * z_scale]], dtype=np.float32)
    plotter.add_points(seed_point, color="cyan", point_size=10.0, render_points_as_spheres=True)
    plotter.add_text(f"Seed point: ({seed_row}, {seed_col}, {seed_band})", position="upper_left", font_size=14)
    plotter.add_axes()
    plotter.show_grid(xlabel="X Axis", ylabel="Y Axis", ztitle="Z Axis")
    plotter.view_isometric()
    plotter.camera.zoom(1.35)
    plotter.show(screenshot=str(out_path), auto_close=not show)
    if show:
        plotter.close()


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
    base_band = args.seed_bands[0] if args.base_band is None else args.base_band
    if not 0 <= base_band < cube.shape[2]:
        raise ValueError(f"Base band {base_band} is outside cube band range 0-{cube.shape[2] - 1}.")

    use_cython = not args.python_region_grow and cube.dtype == np.uint8
    cube_mean = float(np.mean(cube))
    colors = ["red", "green", "yellow", "olive", "blue", "saddlebrown", "orange", "purple", "cyan"]

    for index, seed_band in enumerate(args.seed_bands):
        region = region_grow_3d(
            cube,
            (args.seed_row, args.seed_col, seed_band),
            use_cython=use_cython,
            cube_mean=cube_mean,
        )
        out_path = args.out_dir / f"seed_r{args.seed_row}_c{args.seed_col}_b{seed_band}_region3d.png"
        render_region(
            cube,
            region,
            seed=(args.seed_row, args.seed_col, seed_band),
            base_band=base_band,
            color=colors[index % len(colors)],
            out_path=out_path,
            show=args.show,
            region_opacity=args.region_opacity,
            base_opacity=args.base_opacity,
            z_scale=args.z_scale,
        )
        print(f"Wrote PyVista 3D render: {out_path}")


if __name__ == "__main__":
    main()
