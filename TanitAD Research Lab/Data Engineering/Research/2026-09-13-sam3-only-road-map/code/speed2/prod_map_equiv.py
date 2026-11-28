"""Is the production map of a validation clip the same map on a shifted grid? (production poses are interpolated, the validation
sequence used nearest samples -> the renderer's map origin differs by a sub-cell offset.) Three numbers:
  CONTROL   the validation map against ITSELF resampled at the same sub-cell offset (nearest lookup) -- what grid re-quantization
            alone does to cell agreement and thin-class IoU
  CELLS     the production map against the validation map, world-aligned the same way
  CONSUMER  both maps exported through the production exporter on the SAME v2ep frames and poses: mean absolute difference of the
            soft class fractions on the BEV head's 0.5 m cartesian grid and polar48 grid, per channel, over all frames and cells;
            with a control that must be clearly larger: the production export against the validation export 40 frames later
Usage: prod_map_equiv.py <validation worldmap.npz> <production worldmap.npz> <production sam3mapgt.npz>"""
import json, sys
sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import sam3map_export_gt as X

A = dict(np.load(sys.argv[1], allow_pickle=True)); B = dict(np.load(sys.argv[2], allow_pickle=True)); G = np.load(sys.argv[3], allow_pickle=True)
ca, oa, ra = A["cls"], A["origin"], float(A["res"]); cb, ob = B["cls"], B["origin"]
ii, jj = np.meshgrid(np.arange(ca.shape[0]), np.arange(ca.shape[1]), indexing="ij")
wx = oa[0] + (ii + 0.5) * ra; wy = oa[1] + (jj + 0.5) * ra


def lookup_grid(cls, org, x, y):
    bi = np.floor((x - org[0]) / ra).astype(int); bj = np.floor((y - org[1]) / ra).astype(int)
    ok = (bi >= 0) & (bi < cls.shape[0]) & (bj >= 0) & (bj < cls.shape[1])
    out = np.full(x.shape, 255, np.uint8); out[ok] = cls[bi[ok], bj[ok]]
    return out


def agree(c1, c2):
    seen = (c1 != 255) | (c2 != 255)
    iou = {k: round(float(((c1 == k) & (c2 == k)).sum() / max(int(((c1 == k) | (c2 == k)).sum()), 1)), 3) for k in range(8) if (c1 == k).sum() >= 500}
    return round(float((c1[seen] == c2[seen]).mean()), 5), iou


off = ob - oa
ctrl = lookup_grid(ca, oa, wx + off[0], wy + off[1])                       # A resampled at B's sub-cell offset
print("sub-cell offset of the production grid (m):", np.round(off, 4))
print("CONTROL  validation vs itself at that offset:", agree(ca, ctrl))
print("CELLS    production vs validation (world-aligned):", agree(ca, lookup_grid(cb, ob, wx, wy)))
Ts = G["T_world_rig"]
grids = {"cart": (X.cart_points(X.CART), X.CART), "polar48": (X.polar_points(X.POLAR48), X.POLAR48)}
res = {}
for g, ((pts, n_cells), spec) in grids.items():
    fa, fb = [], []
    for T in Ts:
        Rm, t = T[:2, :2], T[:2, 3]; w = pts @ Rm.T + t
        fa.append(X.fractions(X.lookup(A, w), n_cells).reshape(9, *spec["shape"]).astype(np.float32) / 255)
        fb.append(X.fractions(X.lookup(B, w), n_cells).reshape(9, *spec["shape"]).astype(np.float32) / 255)
    fa, fb = np.stack(fa), np.stack(fb)
    stored = G[f"{g}_frac"].astype(np.float32) / 255
    same = np.abs(fa - fb).mean(axis=(0, 2, 3)); ctrl40 = np.abs(np.roll(fa, 40, axis=0) - fb).mean(axis=(0, 2, 3))
    res[g] = {"export_reproduces_stored": bool(np.allclose(fb, stored, atol=1.5 / 255)),
              "mean_abs_fraction_diff_per_channel": [round(float(x), 4) for x in same],
              "control_40_frames_later": [round(float(x), 4) for x in ctrl40]}
    print(f"CONSUMER {g}: stored export reproduced {res[g]['export_reproduces_stored']}")
    print(f"   channels {X.CLASS_NAMES}")
    print(f"   |production - validation| mean fraction diff: {res[g]['mean_abs_fraction_diff_per_channel']}")
    print(f"   control, 40 frames apart:                     {res[g]['control_40_frames_later']}")
print("ZZMAPEQUIV-DONEZZ")
