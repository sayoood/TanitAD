#!/usr/bin/env python3
"""Independent check on f*h: identify the marking standard, then measure its period.

WHY THIS IS NOT CIRCULAR, AND WHY IT IS NOT THE EARLIER DASH MEASUREMENT
------------------------------------------------------------------------
The DUTY CYCLE -- mark length divided by period -- is a pure RATIO, so it is
invariant to the longitudinal scale. It therefore identifies WHICH standard
marking this is without assuming anything about `f*h`. Only then is the period,
in reconstructed metres, compared against that standard's true period. The
identification comes first and free; the test comes second.

⚠️ This is NOT the dash measurement that failed in §11. That one autocorrelated a
STACKED BEV and returned 4.30 m, which was proven to be a frame comb: it tracked
the frame spacing (0.10 s -> 4.40 m, 0.15 s -> 6.08 m, 0.20 s -> 4.04 m), because
each frame's reconstruction has hard near/far edges at the same range and stacking
them at the ego displacement builds a comb. Here every profile comes from ONE
frame, so no comb can form; runs are pooled across frames afterwards.

French marking types (Instruction interministerielle, 7e partie) -- the candidate
set is declared, and the duty cycle is what chooses among them.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bev_calib as BC, lag_scale as LS, run_real as RR

# name -> (mark_m, gap_m). Duty = mark/(mark+gap).
STANDARDS = {
    "T1  (3 + 10)":   (3.0, 10.0),
    "T'1 (20 + 6)":   (20.0, 6.0),
    "T2  (3 + 3.5)":  (3.0, 3.5),
    "T'2 (20 + 20)":  (20.0, 20.0),
    "T3  (3 + 1.33)": (3.0, 1.33),
}


def runs_one(img, P, y_centre, half_m=0.35, x_win=(7.0, 26.0), cell=0.10,
              close_m=0.0):
    """On/off run lengths along x inside one lane-line band, from ONE frame."""
    from trajlib import lane_calib as LC
    H, W = img.shape[:2]
    u, v = LC._ridge_points(img, int(0.50 * H), int(0.95 * H))
    if len(u) < 80:
        return [], []
    g = BC.ground_from_pixels(np.stack([u, v], 1).astype(float), P)
    m = (np.abs(g[:, 1] - y_centre) <= half_m) & (g[:, 0] >= x_win[0]) & (g[:, 0] <= x_win[1])
    if m.sum() < 25:
        return [], []
    nb = int((x_win[1] - x_win[0]) / cell)
    idx = ((g[m, 0] - x_win[0]) / cell).astype(int)
    idx = idx[(idx >= 0) & (idx < nb)]
    prof = np.bincount(idx, minlength=nb).astype(float)
    on = prof > 0
    # bridge DETECTOR fragmentation before measuring paint. The ridge detector
    # breaks a continuous line into pieces, and without this the run lengths
    # measure the detector, not the road.
    if close_m > 0:
        k = int(round(close_m / cell))
        i = 0
        while i < nb:
            if not on[i]:
                j = i
                while j < nb and not on[j]:
                    j += 1
                if i > 0 and j < nb and (j - i) <= k:
                    on[i:j] = True
                i = j
            else:
                i += 1
    marks, gaps, i = [], [], 0
    while i < nb:
        j = i
        while j < nb and on[j] == on[i]:
            j += 1
        L = (j - i) * cell
        # discard runs touching the window edge: they are truncated, not measured
        if i > 0 and j < nb:
            (marks if on[i] else gaps).append(L)
        i = j
    return marks, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--fh", type=float, default=2444.6)
    ap.add_argument("--height", type=float, default=1.427)
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--yaw", type=float, default=-7.299)
    ap.add_argument("--lateral", type=float, default=-0.126)
    ap.add_argument("--y-line", type=float, default=-1.75, help="the DASHED right line")
    ap.add_argument("--close-m", type=float, default=0.0,
                    help="bridge detector gaps shorter than this before measuring runs")
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    P = dict(RR.NOMINAL)
    P.update(height=a.height, fx=a.fh / a.height, lateral=a.lateral, yaw=np.deg2rad(a.yaw))
    P["pitch"] = LS.pitch_for_horizon(P, a.horizon)
    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 8.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    M, G = [], []
    for pi in np.linspace(0, len(usable) - 1, a.frames).astype(int):
        f = int(usable[pi]["frame"])
        img = cv2.imread(str(fdir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        m, g = runs_one(img, P, a.y_line, close_m=a.close_m)
        M += [x for x in m if x >= 0.5]          # below 0.5 m is speckle, not a mark
        G += [x for x in g if x >= 0.5]
    if len(M) < 20 or len(G) < 20:
        print(f"too few runs: {len(M)} marks, {len(G)} gaps — cannot identify a standard")
        return 1
    mk, gp = float(np.median(M)), float(np.median(G))
    duty = mk / (mk + gp)
    print(f"f*h {a.fh}  h {a.height}  line at y = {a.y_line} m   "
          f"{len(M)} marks, {len(G)} gaps over {a.frames} single frames\n")
    print(f"  measured mark {mk:.2f} m   gap {gp:.2f} m   period {mk+gp:.2f} m")
    print(f"  DUTY CYCLE {duty:.3f}  <- scale-invariant, so this identifies the standard "
          f"without assuming f*h\n")
    # ⛔ THE GUARD THIS PROBE EXISTS TO CARRY. No French marking has a period under
    # 4 m. A shorter one means the runs are RIDGE-DETECTOR GRANULARITY, not paint --
    # exactly the failure that made the stacked-BEV dash measurement return 1.88 m
    # and imply a 4.6 deg HFOV. Without this the duty-cycle matcher cheerfully
    # returned "T'2" and f*h x 26.7 from a 1.50 m period.
    if mk + gp < 4.0:
        print(f"  ⛔ REFUSED: a {mk+gp:.2f} m period is shorter than ANY marking standard.")
        print(f"     These runs are the ridge detector fragmenting a line, not paint.")
        print(f"     Try --close-m to bridge the fragmentation, or accept that this front")
        print(f"     end cannot measure dash geometry. NOT a statement about f*h.")
        if a.json:
            a.json.write_text(json.dumps(dict(mark_m=mk, gap_m=gp, period_m=mk+gp,
                                              duty=duty, refused=True,
                                              reason="period below any standard"), indent=2))
        return 2
    rows = []
    for name, (sm, sg) in STANDARDS.items():
        d = sm / (sm + sg)
        rows.append((abs(d - duty), name, d, sm + sg))
    rows.sort()
    for err, name, d, per in rows:
        print(f"    {name:16s} duty {d:.3f}  (|delta| {err:.3f})   true period {per:5.2f} m")
    _, best, dbest, per_true = rows[0]
    print(f"\n  best match: {best}")
    k = per_true / (mk + gp)
    print(f"  its true period {per_true:.2f} m vs measured {mk+gp:.2f} m  ->  f*h x {k:.3f}"
          f"  =>  f*h = {a.fh*k:.0f} px*m  (assumed {a.fh:.0f})")
    if rows[1][0] < 1.6 * rows[0][0]:
        print("  ⚠️ the duty cycle does NOT discriminate: the top two candidates are "
              "within 1.6x. Treat the period comparison as UNIDENTIFIED, not as a check.")
    if a.json:
        a.json.write_text(json.dumps(dict(mark_m=mk, gap_m=gp, duty=duty, n_marks=len(M),
                                          n_gaps=len(G), best=best, k_fh=float(k),
                                          fh_implied=float(a.fh * k)), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
