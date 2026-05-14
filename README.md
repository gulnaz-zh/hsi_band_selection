# 3D Region-Growing Spectral Band Prioritization for HSI

This repository reconstructs the code for the submission:

**A region-growing approach for spectral band prioritization in hyperspectral remote sensing**

The original folder contained a generic grayscale region-growing Cython implementation. The recovered code in `src/hsi_region_growing.py` implements the hyperspectral workflow described in the paper/thesis. The Cython region-growing routine in `RegionGrowth.pyx` is motivated by Pengyi Zhang's RegionGrowth project: https://github.com/PengyiZhang/RegionGrowth

- 3D Region Growing Algorithm (3D RGA) on hyperspectral cubes.
- HSI cube convention: `(height, width, bands)`.
- Seed convention: `(row, column, band)`.
- 6-connected 3D neighbourhood.
- Seed-intensity stopping criterion:

```text
seed_value - 0.1 * mean(cube) < neighbour_value < seed_value + 0.1 * mean(cube)
```

- Projection of the grown 3D region to a 2D spatial mask by union across bands.
- Band scoring with Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI).
- Informative-band selection with a 0.75 NMI/ARI threshold, relaxed to `0.75 * max_score` when a class never reaches 0.75.
- Final segmentation aggregation over ten equidistant bands in the selected range:
  - ARC: all-regions consensus.
  - MRR: majority-regions rule.

## Files

- `src/hsi_region_growing.py`: main recovered implementation.
- `src/dataset_presets.py`: dataset-specific preprocessing for the paper datasets.
- `run_band_search.py`: command-line runner for `.mat` HSI datasets.
- `generate_toy_images.py`: creates the toy 3D HSI `.mat` cube and demo images.
- `images/`: toy `.mat` cube, input band, seed, and projected 3D region-growing output.
- `RegionGrowth.pyx`: fast Cython HSI 3D grower inspired by `PengyiZhang/RegionGrowth`.
- `setup.py`: builds the Cython extension in place.

## Install

```bash
python -m pip install numpy scipy pytest cython
python setup.py build_ext --inplace
```

The expensive 3D region-growing loop runs through Cython when the `RegionGrowth` extension is built. If the extension is missing, the Python API falls back to the pure Python implementation so tests and small examples still run. `scipy` is only needed by `run_band_search.py` to load `.mat` files.

## Toy Demo

Generate the toy 3D HSI example and projected region-growing output:

```bash
python generate_toy_images.py
```

The generated `images/toy_hsi_cube.mat` contains:

- `toy_hsi_cube`: synthetic 3D HSI matrix with shape `(80, 100, 12)`.
- `seed`: seed coordinate in `(row, column, band)` order.
- `region_3d`: binary 3D region-growing output.
- `projected_region_2d`: binary 2D projection of the grown 3D region.

Input band:

![Toy input band](images/toy_input_band.png)

Seed:

![Toy seed](images/toy_seed.png)

Output:

![Toy region growing output](images/toy_region_growing_output.png)

## Usage

Run a band search for one class and one spatial seed:

```bash
python run_band_search.py \
  --dataset salinas \
  --cube Salinas_corrected.mat --cube-key salinas_corrected \
  --labels Salinas_gt.mat --labels-key salinas_gt \
  --class-id 1 \
  --seed-row 80 --seed-col 60 \
  --out outputs/salinas_class_1_scores.csv
```

The script prints the selected band indices, contiguous selected ranges, and ARC/MRR scores for the widest selected range. It also writes per-band NMI/ARI scores to CSV.

By default, the runner normalizes the cube to `uint8` and uses the Cython grower. If you pass `--no-normalize` with a non-`uint8` cube, the runner uses the slower Python grower unless you convert the cube beforehand.

Use `--dataset` for the paper datasets:

- `salinas`: accepts raw 224-band AVIRIS cubes or already-corrected 204-band cubes. Raw cubes have the reported water-absorption bands removed.
- `indian_pines`: accepts raw 224-band AVIRIS cubes or corrected 204-band cubes. Raw cubes have the reported water-absorption bands removed.
- `pavia_centre`: accepts the 102-band ROSIS cube without spectral removal.
- `nerc_arf`: accepts the raw 622-band AisaFENIX cube or the 207-band processed cube. Raw cubes are downsampled by averaging non-overlapping triples and dropping the final leftover band.

For NERC-ARF or other unlabeled data, omit `--labels`. The runner then writes per-band grown-region sizes instead of NMI/ARI:

```bash
python run_band_search.py \
  --dataset nerc_arf \
  --cube nerc_arf.mat --cube-key cube \
  --seed-row 120 --seed-col 240 \
  --band-start 0 --band-stop 206 \
  --out outputs/nerc_seed_stats.csv
```

## Python API

```python
from src.hsi_region_growing import (
    class_mask,
    find_informative_bands,
    aggregate_selected_bands,
    normalize_to_uint8,
)

cube = normalize_to_uint8(cube)       # cube shape: (height, width, bands)
target = class_mask(labels, class_id=1)

result = find_informative_bands(
    cube,
    spatial_seed=(80, 60),
    target_mask=target,
)

final = aggregate_selected_bands(
    cube,
    spatial_seed=(80, 60),
    selected_bands=(20, 100),
    target_mask=target,
)

print(result.selected_bands)
print(final.arc_nmi, final.arc_ari)
print(final.mrr_nmi, final.mrr_ari)
```

## Experimental Settings Recovered From The Manuscript/Thesis

The code follows the reported settings for Salinas, Indian Pines, Pavia Centre, and NERC-ARF:

- Salinas: AVIRIS, `512x217x224`, 400-2500 nm, 16 classes; water absorption bands removed in the experiments.
- Indian Pines: AVIRIS, `145x145x224`, 400-2500 nm, 16 classes; 204 retained bands after water absorption removal.
- Pavia Centre: ROSIS, `1096x715x102`, 430-860 nm, 9 classes.
- NERC-ARF: AisaFENIX, `4759x8496x622`, 380-2500 nm; downsampled to 207 bands by averaging non-overlapping triples, rotated, cropped, and analyzed without ground truth.

For labelled datasets, the paper selects seed spatial coordinates from the region of interest using the ground truth, scans candidate seed bands, and compares each projected 3D RGA region to the seed class mask using NMI and ARI.

## Notes

Compiled extension files, generated C files, cache folders, and run outputs are intentionally ignored. Rebuild the extension locally with `python setup.py build_ext --inplace`.
