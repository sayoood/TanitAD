#!/usr/bin/env python3
"""Where does the visible ROAD end and the car's own bonnet begin?

This is not cosmetic. Everything the overlay draws nearer than this row is painted
over sheet metal, so it can be neither verified nor falsified by the road -- and it
is exactly the part of the corridor that looks widest and most wrong. Establishing
the limit turns "the near field is unmeasured" from a hedge into a fact with a row
number attached.

THE DETECTOR. Content fixed to the CAR is static in the image; the road is not. So
pool |I(t) - I(t+1)| per image row over many pairs: the road rows carry motion
energy, the bonnet rows carry almost none. No calibration is involved, so this
cannot be contaminated by the thing being checked.

⛔ THIS PROBE IS INVALID AND IS KEPT ONLY AS THE RECORD OF WHY.

It answers "which pixels CHANGE", not "which surface MOVES", and a glossy bonnet
reflecting moving scenery changes pixels enthusiastically while the surface itself
is bolted to the car. MEASURED 2026-09-13: it reported 100-130% of the road's
motion energy all the way down to row 928, and on that basis I briefly retracted
the (correct) finding that rows 830+ are bodywork.

The retraction was wrong. `lk_failure_vs_static.py` settles it properly by asking
the right question -- does a STRONGER tracker find motion there? Re-tracking the
same rows with win 51 / 7 pyramid levels instead of win 21 / 4 DOUBLES the tracking
rate (20% -> 41% at rows 830-900, 31% -> 52% at 900-960) and the measured dv stays
at -0.2 and -0.6 px against a predicted 52-67 px. Twice as many points found, and
they still do not move: the surface is static, the tracker was never the issue.

⇒ Same family as `df` on a pod, `free` on Thor, and cgroup `usage_in_bytes`: a
probe that aggregates the wrong quantity and reads like an answer. The row band cut
at 0.77 H stands, on the evidence it originally had.
"""
import sys, argparse, json, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import run_real as RR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--fh", type=float, default=2444.6, help="to convert rows to metres")
    ap.add_argument("--horizon", type=float, default=465.0)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    a = ap.parse_args()

    recs = RR.load_records(RR.RUN); fdir = RR.RUN / "frames"
    usable = [r for r in recs if r["complete"] and r["speed_ms"] > 12.0
              and (fdir / f"{int(r['frame']):06d}.jpg").exists()]
    acc, n = None, 0
    for pi in np.linspace(0, len(usable) - 1, a.pairs).astype(int):
        f = int(usable[pi]["frame"])
        p0, p1 = fdir / f"{f:06d}.jpg", fdir / f"{f+1:06d}.jpg"
        if not p1.exists():
            continue
        i0 = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
        i1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
        if i0 is None or i1 is None:
            continue
        d = np.abs(i0.astype(np.int16) - i1.astype(np.int16)).mean(axis=1)
        acc = d if acc is None else acc + d
        n += 1
    if not n:
        print("no usable pairs")
        return 1
    prof = acc / n
    H = len(prof)
    road = float(np.median(prof[int(0.60 * H):int(0.72 * H)]))   # definitely road
    print(f"{n} consecutive frame pairs, speed > 12 m/s")
    print(f"reference motion energy on known-road rows (0.60-0.72 H): {road:.2f}\n")
    print(f"{'row':>6} {'motion':>8} {'vs road':>8}   {'range at that row':>18}")
    for r in range(int(0.60 * H), H, 20):
        x = a.fh / (r - a.horizon) if r > a.horizon else float("inf")
        print(f"{r:6d} {prof[r]:8.2f} {100*prof[r]/road:7.0f}%   {x:15.2f} m")
    # the bonnet edge: the highest row below 0.70H where motion stays under 30%
    thr = 0.30 * road
    edge = None
    for r in range(int(0.70 * H), H):
        if prof[r] < thr and np.all(prof[r:min(r + 40, H)] < thr):
            edge = r
            break
    if edge is None:
        print("\nno static band found — the bonnet may not be in frame")
        return 0
    x_edge = a.fh / (edge - a.horizon)
    print(f"\n  BONNET EDGE at row {edge}  ({edge/H:.3f} of image height)")
    print(f"  => the road is visible no nearer than {x_edge:.2f} m")
    print(f"  => every overlay pixel below row {edge} is drawn over the car's own")
    print(f"     bodywork. It cannot be verified against the road, and it is the")
    print(f"     widest part of the corridor — which is why it reads as wrong.")
    if a.json:
        a.json.write_text(json.dumps(dict(bonnet_row=int(edge), frac=edge / H,
                                          nearest_road_m=float(x_edge),
                                          road_energy=road, n_pairs=n), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
