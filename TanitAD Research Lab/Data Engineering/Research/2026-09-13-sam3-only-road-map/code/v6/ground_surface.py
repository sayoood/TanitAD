"""A SMOOTH ground surface for projecting pixels onto the road.

P.ground_grid gives one height per 2 m cell (10th percentile of LiDAR or path points) and P.lift / the inverse warps
looked it up per cell with no interpolation. Measured on the PI's arrow frame (ipm_native.py): the warped image shows
rectangular seams on the 2 m cell borders -- a height step of 5 cm between neighbouring cells moves the ground point by
~0.4 m at 10 m range for a camera 1.28 m up, which breaks painted arrows that cross a cell border.
Here: empty cells are filled from their nearest valid neighbours, the 50 x 50 grid is lightly smoothed (Gaussian,
sigma 1 cell), and heights are read by BILINEAR interpolation between cell centres.
"""
import numpy as np
from scipy import ndimage as ndi


def smooth_grid(grid, fb, sigma=1.0):
    g = grid.reshape(50, 50).astype(np.float64)
    valid = np.isfinite(g)
    if not valid.any():
        return np.full((50, 50), fb)
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = g[tuple(idx)]
    return ndi.gaussian_filter(filled, sigma=sigma, mode="nearest") if sigma > 0 else filled


def height(xy, sgrid):
    """Bilinear height at rig-frame xy (N, 2) from a smoothed 50x50 grid of 2 m cells centred at -49, -47, ... 49."""
    fx = (xy[:, 0] + 50.0) / 2.0 - 0.5; fy = (xy[:, 1] + 50.0) / 2.0 - 0.5
    fx = np.clip(fx, 0, 49); fy = np.clip(fy, 0, 49)
    i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
    i1 = np.minimum(i0 + 1, 49); j1 = np.minimum(j0 + 1, 49)
    ax = fx - i0; ay = fy - j0
    return (sgrid[i0, j0] * (1 - ax) * (1 - ay) + sgrid[i1, j0] * ax * (1 - ay) + sgrid[i0, j1] * (1 - ax) * ay + sgrid[i1, j1] * ax * ay)
