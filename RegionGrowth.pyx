# -*- coding: utf-8 -*-
"""Fast 3D region growing for hyperspectral image cubes.

This Cython implementation is motivated by the queue-based region-growing
extension in Pengyi Zhang's RegionGrowth project:
https://github.com/PengyiZhang/RegionGrowth

The original project supports generic 2D/3D grayscale region growing. This file
keeps only the HSI-specific 3D routine required by the spectral band
prioritization workflow in the paper.
"""

import numpy as np
cimport cython
cimport numpy as np
from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def RegionGrowHSI3D(np.ndarray[np.uint8_t, ndim=3, mode="c"] cube,
                    int row,
                    int col,
                    int band,
                    double tolerance_fraction=0.1,
                    double mean_intensity=-1.0):
    """
    Fast 6-connected 3D region growing for hyperspectral cubes.

    Parameters
    ----------
    cube:
        C-contiguous uint8 array with shape (height, width, bands).
    row, col, band:
        Seed coordinate in HSI order.
    tolerance_fraction:
        Fraction of the global cube mean used around the seed intensity.
    mean_intensity:
        Optional precomputed global cube mean. Passing it avoids recomputing the
        same value during a full spectral-band search.

    Returns
    -------
    np.ndarray[np.uint8_t, ndim=3]
        Binary 3D mask with 1 for grown voxels and 0 for background.
    """
    cdef int height = cube.shape[0]
    cdef int width = cube.shape[1]
    cdef int bands = cube.shape[2]
    cdef int y, x, z
    cdef int64_t total
    cdef int64_t idx, next_idx
    cdef int64_t head = 0
    cdef int64_t tail = 0
    cdef int64_t i
    cdef double total_intensity = 0.0
    cdef double seed_value
    cdef double lower
    cdef double upper
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] output
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] queue

    if row < 0 or row >= height or col < 0 or col >= width or band < 0 or band >= bands:
        raise ValueError("Seed is outside cube shape.")

    total = <int64_t>height * <int64_t>width * <int64_t>bands
    if mean_intensity < 0:
        for y in range(height):
            for x in range(width):
                for z in range(bands):
                    total_intensity += cube[y, x, z]
        mean_intensity = total_intensity / total

    seed_value = cube[row, col, band]
    lower = seed_value - tolerance_fraction * mean_intensity
    upper = seed_value + tolerance_fraction * mean_intensity

    output = np.zeros((height, width, bands), dtype=np.uint8)
    queue = np.empty(total, dtype=np.int64)

    idx = ((<int64_t>row * width) + col) * bands + band
    queue[tail] = idx
    tail += 1
    output[row, col, band] = 1

    while head < tail:
        idx = queue[head]
        head += 1

        y = <int>(idx // (<int64_t>width * bands))
        i = idx - (<int64_t>y * width * bands)
        x = <int>(i // bands)
        z = <int>(i - (<int64_t>x * bands))

        if y > 0 and output[y - 1, x, z] == 0:
            if lower < cube[y - 1, x, z] < upper:
                output[y - 1, x, z] = 1
                next_idx = (((<int64_t>y - 1) * width) + x) * bands + z
                queue[tail] = next_idx
                tail += 1

        if y + 1 < height and output[y + 1, x, z] == 0:
            if lower < cube[y + 1, x, z] < upper:
                output[y + 1, x, z] = 1
                next_idx = (((<int64_t>y + 1) * width) + x) * bands + z
                queue[tail] = next_idx
                tail += 1

        if x > 0 and output[y, x - 1, z] == 0:
            if lower < cube[y, x - 1, z] < upper:
                output[y, x - 1, z] = 1
                next_idx = ((<int64_t>y * width) + x - 1) * bands + z
                queue[tail] = next_idx
                tail += 1

        if x + 1 < width and output[y, x + 1, z] == 0:
            if lower < cube[y, x + 1, z] < upper:
                output[y, x + 1, z] = 1
                next_idx = ((<int64_t>y * width) + x + 1) * bands + z
                queue[tail] = next_idx
                tail += 1

        if z > 0 and output[y, x, z - 1] == 0:
            if lower < cube[y, x, z - 1] < upper:
                output[y, x, z - 1] = 1
                next_idx = ((<int64_t>y * width) + x) * bands + z - 1
                queue[tail] = next_idx
                tail += 1

        if z + 1 < bands and output[y, x, z + 1] == 0:
            if lower < cube[y, x, z + 1] < upper:
                output[y, x, z + 1] = 1
                next_idx = ((<int64_t>y * width) + x) * bands + z + 1
                queue[tail] = next_idx
                tail += 1

    return output
