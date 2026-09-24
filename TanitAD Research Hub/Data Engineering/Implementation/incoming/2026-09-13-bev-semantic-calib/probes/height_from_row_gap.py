#!/usr/bin/env python3
"""Camera height from the PIXEL GAP between lane lines. No projection, no window.

WHY THE EARLIER HEIGHT MEASUREMENTS WERE CIRCULAR
--------------------------------------------------
Both lateral instruments used so far associate detected paint to a PROJECTED lane
edge inside a tolerance window, so when the assumed height changes the projection
moves and the associator simply finds whatever paint is now nearest. MEASURED, and
it is fatal: `lane_residual.py` reports the painted lane as **3.53 m at h = 1.17**
and **3.57 m at h = 1.427**. Lateral scale is LINEAR in height, so the same paint
must measure 4.31 m at the larger height. A ratio of 1.011 where 1.22 was required
means the instrument has ~zero sensitivity to the quantity it claims to measure --
it confirms whatever height it is given. `lane_calib`'s 2.6-4.6 m censoring window
is the same defect, and this module was written after finding I had rebuilt it.

THE NON-CIRCULAR FORM
---------------------
Back-projection of a single image ROW needs no focal length and no model lane::

    y = (u - cx) * h / (v - v_h)

so two lane lines seen in the same row v, separated by ``du`` pixels, are separated
on the ground by ``W = du * h / (v - v_h)``. Given the true lane width that inverts::

    h = W_true * (v - v_h) / du

Only three things enter: the pixel gap, the row, and the horizon. No focal, no
projected edge, no tolerance window, nothing that can drift to meet the answer.

⭐ AND IT CHECKS THE HORIZON FOR FREE. With the right ``v_h`` the height comes out the
same at every row; with a wrong one it trends with row, because ``(v - v_h)`` is
scaled wrongly at each range. Scanning ``v_h`` for a flat ``h(v)`` measures both.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_real as RR


def row_gaps(img, rows, min_sep_px=90, smooth=9):
    """Pixel gap between the two lane lines bracketing the lane, per row."""
    from trajlib import lane_calib as LC
    H, W = img.shape[:2]
    u, v = LC._ridge_points(img, int(0.50 * H), int(0.95 * H))
    if len(u) < 120:
        return {}
    out = {}
    for r in rows:
        sel = np.abs(v - r) <= 2
        if sel.sum() < 6:
            continue
        prof = np.bincount(np.clip(u[sel], 0, W - 1).astype(int), minlength=W).astype(float)
        k = np.ones(smooth) / smooth
        prof = np.convolve(prof, k, mode="same")
        if prof.max() <= 0:
            continue
        pk = [i for i in range(1, W - 1)
              if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.30 * prof.max()]
        if len(pk) < 2:
            continue
        # ⚠️ Do NOT try to identify the ego lane. MEASURED: bracketing the IMAGE
        # centre gave gaps of 1178 px at row 620 falling to 713 px at row 740 — a
        # lane must WIDEN downward, so the picker was grabbing different lines at
        # different rows. The -7 deg mount yaw puts the ego lane well off centre,
        # so "the pair straddling u = W/2" is not the ego lane at every row.
        #
        # Identification is unnecessary: lane lines are parallel and (mostly) one
        # lane apart, so the MEDIAN gap between ADJACENT lines is one lane width in
        # pixels regardless of which lines were found. Gaps below `min_sep_px` are
        # the two edges of a single painted stripe, not two lanes.
        pk = np.asarray(sorted(pk), dtype=float)
        gaps = np.diff(pk)
        gaps = gaps[gaps >= min_sep_px]
        if len(gaps) >= 1:
            out[r] = float(np.median(gaps))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--lane-width", type=float, default=3.50)
    ap.add_argument("--rows", type=int, nargs="+",
                    default=[620, 660, 700, 740, 780, 820])
    ap.add_argument("--horizons", type=float, nargs="+",
                    default=[420., 435., 450., 465., 480., 495., 510.])
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    keep = []
    for r in usable:
        t = np.asarray(r["t"], float); yw = np.asarray(r["yaw"], float)
        m = np.abs(t) < 0.4
        if m.sum() >= 3 and abs(np.rad2deg(np.polyfit(t[m], yw[m], 1)[0])) <= 0.5:
            keep.append(r)

    acc = {r: [] for r in a.rows}
    for pi in np.linspace(0, len(keep) - 1, a.frames).astype(int):
        f = int(keep[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for r, du in row_gaps(img, a.rows).items():
            acc[r].append(du)
    ok_rows = [r for r in a.rows if len(acc[r]) >= 15]
    if len(ok_rows) < 3:
        print("too few rows with a usable line pair")
        return 1
    med = {r: float(np.median(acc[r])) for r in ok_rows}
    print(f"lane width assumed {a.lane_width:.2f} m (Sayed: standard French motorway)\n")
    print(f"  {'row':>5} {'n':>5} {'line gap':>10}")
    for r in ok_rows:
        print(f"  {r:5d} {len(acc[r]):5d} {med[r]:9.1f} px")

    print(f"\n  h = W (v - v_h) / du, evaluated at each row. The RIGHT horizon makes it FLAT.\n")
    print(f"  {'horizon':>8} " + "".join(f"{r:>9}" for r in ok_rows) + f"{'spread':>9}{'mean h':>9}")
    best = None
    for vh in a.horizons:
        hs = [a.lane_width * (r - vh) / med[r] for r in ok_rows]
        if min(hs) <= 0:
            continue
        spread = (max(hs) - min(hs)) / np.mean(hs)
        print(f"  {vh:8.1f} " + "".join(f"{h:9.3f}" for h in hs)
              + f"{spread:8.1%}{np.mean(hs):9.3f}")
        if best is None or spread < best[1]:
            best = (vh, spread, float(np.mean(hs)))
    if best is None:
        print("  no horizon gave positive heights")
        return 1
    vh, spread, h = best
    print(f"\n  flattest at horizon {vh:.1f} px  ->  h = {h:.3f} m  (spread {spread:.1%})")
    print(f"  with f*h = 2872.7 (the pipeline's own flow measurement): "
          f"f = {2872.7/h:.0f} px, HFOV {np.rad2deg(2*np.arctan(1920/(2*2872.7/h))):.1f} deg")
    print(f"  with f*h = 2444.6 (the set rendered earlier):            "
          f"f = {2444.6/h:.0f} px, HFOV {np.rad2deg(2*np.arctan(1920/(2*2444.6/h))):.1f} deg")
    if a.json:
        a.json.write_text(json.dumps(dict(rows=ok_rows, gaps=med, best_horizon=vh,
                                          height=h, spread=spread,
                                          lane_width=a.lane_width), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
