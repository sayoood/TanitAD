#!/usr/bin/env python3
"""Restrict marking detection to the ROAD — the fix for `R-2026-09-14-foliage`.

`ridge_cols` returns the brightest narrow features in an image row. On this recording
the sunlit vegetation and pale rock bank to the right of the carriageway are brighter
and far more textured than the paint, so an unmasked detector lands overwhelmingly on
the bank, and every measurement that needed the RIGHT lane boundary was measuring
shrubbery. `ground_calib.collect_road_tracks` already masks to a projected road
corridor; the probes in this directory did not.

⚠️ The mask consumes the calibration it helps produce, so it is deliberately COARSE:
a corridor `half_w` metres either side of the vehicle's axis. Both lane boundaries sit
within ~2.3 m even when the car is well off centre, while the bank is several metres
beyond the shoulder — so a 3 m half-width separates them with a factor of two to spare,
and a 0.7 deg yaw uncertainty (0.5 m at 40 m) cannot move a boundary out of it. It is a
gate, not a fit.

⚠️ The near end is cut at the bonnet row. `flow_scale` measured that the image below
~0.77 H is static (bonnet or windscreen reflection); it is not road and it is exactly
what broke `plane_calib` (§65).

RUN THIS FILE DIRECTLY to write an annotated preview and LOOK at it. That is the rule
the foliage retraction exists to enforce: before the first statistic, draw the
detector's input on the image.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bev_calib as BC, lag_scale as LS, run_real as RR                # noqa: E402
from overlay_far import ridge_cols, ridge_width_px                     # noqa: E402


def calib(fx=1533.0, height=1.586, horizon=448.4, yaw=-5.35, lateral=-0.126):
    P = dict(RR.NOMINAL)
    P.update(fx=fx, height=height, lateral=lateral, yaw=np.deg2rad(yaw))
    P["pitch"] = LS.pitch_for_horizon(P, horizon)
    return P


def road_mask(P, shape, y_left=4.2, y_right=3.0, x_range=(8.0, 45.0), bonnet_row=832,
              half_w=None):
    """Binary mask of the drivable corridor, projected from ``P``.

    ⚠️ ASYMMETRIC ON PURPOSE. A symmetric +/-3 m corridor, previewed on this recording,
    sat with its LEFT edge exactly on the ego lane's left boundary while its RIGHT edge
    ran past the shoulder — so the boundary that matters most was at the very edge of
    the mask, and the side with the contaminating bank had the most slack. The bank is
    on the RIGHT here (a cutting with a barrier), so the corridor is widened to the left
    and trimmed on the right. ``half_w`` still works and sets both.
    """
    if half_w is not None:
        y_left = y_right = half_w
    h, w = shape
    xs = np.linspace(x_range[0], x_range[1], 80)
    left = BC.project_ground(np.stack([xs, np.full_like(xs, y_left)], 1), P)
    right = BC.project_ground(np.stack([xs, np.full_like(xs, -y_right)], 1), P)
    ok = np.isfinite(left).all(1) & np.isfinite(right).all(1)
    m = np.zeros((h, w), np.uint8)
    if ok.sum() >= 2:
        poly = np.concatenate([left[ok], right[ok][::-1]])
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    if bonnet_row:
        m[int(bonnet_row):, :] = 0
    return m


def masked_ridges(gray, mask, rows, fh, horizon, fx):
    """Ridge points inside the mask only, with the physical width schedule."""
    pts = []
    H, W = gray.shape
    for v in rows:
        vi = int(round(v))
        if not (0 <= vi < H):
            continue
        row_mask = mask[vi]
        if row_mask.max() == 0:
            continue
        x = fh / max(v - horizon, 1e-3)
        for c in ridge_cols(gray[vi], ridge_width_px(x, fx)):
            ci = int(c)
            if 0 <= ci < W and row_mask[ci]:
                pts.append((float(v), float(c)))
    return np.asarray(pts, float) if pts else np.empty((0, 2))


def _preview(out_path, n=4, half_w=3.0):
    P = calib()
    fx, height, horizon = P["fx"], P["height"], 448.4
    fh = fx * height
    rows = np.arange(horizon + fh / 45.0, horizon + fh / 9.0, 1.0)
    recs = {int(r["frame"]): r for r in RR.load_records(RR.RUN)}
    fdir = RR.RUN / "frames"
    avail = [f for f in sorted(int(p.stem) for p in fdir.glob("*.jpg"))
             if (r := recs.get(f)) and r["complete"] and r["speed_ms"] > 12]
    out = []
    for f in avail[:: max(1, len(avail) // n)][:n]:
        g = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        m = road_mask(P, g.shape, y_left=half_w + 1.2, y_right=half_w)
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        im[m > 0] = (0.72 * im[m > 0] + np.array([70, 0, 0])).astype(np.uint8)
        un = 0
        for v in rows:
            vi = int(round(v))
            x = fh / max(v - horizon, 1e-3)
            for c in ridge_cols(g[vi], ridge_width_px(x, fx)):
                ci = int(c)
                if 0 <= ci < g.shape[1]:
                    inside = m[vi, ci] > 0
                    cv2.circle(im, (ci, vi), 1,
                               (0, 255, 0) if inside else (0, 0, 255), -1)
                    un += 0 if inside else 1
        pts = masked_ridges(g, m, rows, fh, horizon, fx)
        cv2.putText(im, f"frame {f}   GREEN = kept ({len(pts)})   RED = rejected ({un})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        out.append(im[400:1010])
    img = np.vstack(out)
    cv2.imwrite(str(out_path), cv2.resize(img, (1500, int(img.shape[0] * 1500 / img.shape[1]))))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    _preview(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "road_mask_preview.png"),
             half_w=float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)
