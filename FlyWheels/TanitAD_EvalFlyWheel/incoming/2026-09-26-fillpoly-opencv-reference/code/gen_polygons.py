#!/usr/bin/env python3
"""STEP 1 of 2 — generate the polygon cases, in the MAIN venv, and bank the VERTICES.

Why here and not in the OpenCV env: the old skipped test drew its quads from ``np.random.default_rng(0)``,
and numpy does not guarantee a Generator's stream is identical across versions (the OpenCV 4.5.4 env runs
numpy 1.26.4; the main venv runs 2.x). So the polygons are generated HERE — exactly the ones the old test
would have drawn — and their vertices are banked. The OpenCV env only draws them. No test then depends on
two random generators agreeing.

Groups
* ``quads``        — bit-for-bit the old test's 200 cases: seed 0, ``integers(-20, 60, (4, 2))``, 40x40 grid.
* ``boxes``        — the REAL use: rotated agent rectangles, vertices from the occupancy builders' own pixel
                     transform ``np.round((corners - BX + DX/2) / DX)`` on the 200x200, 0.5 m/px BEV grid.
* ``adversarial``  — hand-written: degenerate, concave, self-intersecting, boundary, off-grid, huge, reversed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, os.path.join(REPO, "stack"))
from adapters import nuscenes_planning as NP  # noqa: E402  (BX, DX, BEV -- the real grid)


def quads() -> list:
    rng = np.random.default_rng(0)                       # the old test's generator, unchanged
    return [rng.integers(-20, 60, size=(4, 2)).tolist() for _ in range(200)]


def boxes() -> list:
    """Rotated agent boxes through the builders' own metres -> pixel rounding."""
    rng = np.random.default_rng(1)
    sizes = [(4.6, 1.9), (10.0, 2.5), (0.7, 0.7), (1.8, 0.6), (12.0, 2.9), (5.2, 2.1)]   # car, truck, ped, bike, bus, van
    yaws = [0.0, np.pi / 4, np.pi / 2, np.pi / 3, -np.pi / 6]
    out = []
    for i in range(300):
        l, w = sizes[i % len(sizes)]
        yaw = yaws[i % len(yaws)] if i < 60 else float(rng.uniform(-np.pi, np.pi))
        # centres across the whole grid INCLUDING beyond its edges (clipping is part of the contract)
        cx, cy = (float(v) for v in rng.uniform(-55.0, 55.0, size=2))
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        corners = local @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
        px = np.round((corners - NP.BX + NP.DX / 2.0) / NP.DX).astype(np.int64)
        out.append(px.tolist())
    return out


def adversarial() -> list:
    sq = [[10, 10], [40, 10], [40, 40], [10, 40]]
    return [
        [[20, 20]] * 4,                                   # all four vertices identical
        [[5, 5], [30, 5]],                                # two points
        [[5, 20], [20, 20], [40, 20], [60, 20]],          # collinear, horizontal
        [[20, 5], [20, 25], [20, 45], [20, 60]],          # collinear, vertical
        [[0, 0], [21, 21], [42, 42]],                     # collinear, diagonal (zero area)
        [[10, 10], [50, 12], [30, 30], [50, 50], [10, 48]],             # concave (chevron)
        [[32, 2], [40, 25], [62, 32], [40, 39], [32, 62], [24, 39], [2, 32], [24, 25]],  # 8-point star
        [[10, 10], [50, 50], [50, 10], [10, 50]],         # bowtie, self-intersecting
        sq,                                               # square
        sq[::-1],                                         # same square, REVERSED winding
        [[10, 10], [10, 10], [40, 10], [40, 40], [40, 40], [10, 40]],  # duplicate consecutive vertices
        [[0, 0], [63, 0], [63, 63], [0, 63]],             # exactly the full 64x64 grid
        [[-10, -10], [80, -10], [80, 80], [-10, 80]],     # larger than the grid
        [[-30, 5], [-5, 5], [-5, 30], [-30, 30]],         # fully off-grid (left)
        [[-8, 20], [12, 20], [12, 40], [-8, 40]],         # straddles the left edge
        [[50, -8], [70, -8], [70, 12], [50, 12]],         # straddles the top-right corner
        [[-10000, 30], [10000, 31], [10000, 33], [-10000, 32]],         # huge coordinates, thin
        [[5, 5], [6, 5], [5, 6]],                         # 1-pixel triangle
        [[20, 10], [21, 10], [21, 50], [20, 50]],         # 1-2 px wide sliver
        [[0, 63], [63, 63], [63, 62]],                    # on the bottom boundary
    ]


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "polygons.json"
    doc = {
        "_what": "polygon cases for the cv2.fillPoly 4.5.4 reference; vertices banked so no RNG must agree across numpy versions",
        "generated_with": {"python": sys.version.split()[0], "numpy": np.__version__,
                           "grid_from": "taniteval/adapters/nuscenes_planning.py BEV/DX/BX",
                           "BEV": int(NP.BEV), "DX": float(NP.DX), "BX": float(NP.BX)},
        "groups": {
            "quads": {"hw": [40, 40], "source": "old test_fillpoly_parity_with_real_cv2: default_rng(0).integers(-20,60,(4,2)) x200",
                      "polys": quads()},
            "boxes": {"hw": [int(NP.BEV), int(NP.BEV)], "source": "rotated agent boxes via np.round((c-BX+DX/2)/DX), default_rng(1)",
                      "polys": boxes()},
            "adversarial": {"hw": [64, 64], "source": "hand-written edge cases", "polys": adversarial()},
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    n = {k: len(v["polys"]) for k, v in doc["groups"].items()}
    print(f"WROTE {out}: {n}  (numpy {np.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
