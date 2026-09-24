#!/usr/bin/env python3
"""Run BEV-agreement calibration on the REAL 2026-08-08 recording.

Classical front end on purpose: this de-risks the OPTIMISER before taking on a
SAM3 dependency (and this container has no GPU, no torch and no HF access). The
ink comes from ``lane_calib._ridge_points`` -- the pipeline's OWN marking
detector -- so nothing here is a new, unvalidated perception component and the
comparison against the pipeline's own numbers is fair.

What it measures, in order:

  1. Does the objective have a peak at all on real data, or is it flat/noisy?
     (Synthetic says it should be sharply peaked; real data has ego-motion error,
     rolling shutter, possible EIS and a non-planar road.)
  2. The f*h manifold scan -- the headline synthetic claim -- repeated on real ink.
  3. A joint optimisation of the parameters synthetic showed to be observable.

⚠️ Everything it prints is MEASURED-on-real but still rests on the pipeline's ego
poses being right; those carry 0.70 m position / 0.53 deg heading hold-out RMS.
Short baselines keep the RELATIVE motion far better than that, which is why the
anchors below span ~2 s rather than the whole clip.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bev_calib as BC                                              # noqa: E402

RUN = pathlib.Path("/root/trajdata/out_bev/2026-08-08_14-19-54-android")
NOMINAL = dict(yaw=0.0, pitch=0.0, roll=0.0, height=1.17, lateral=-0.35,
               longitudinal=2.10, fx=1478.3, cx=960.0, cy=540.0)


def load_records(run: pathlib.Path):
    return [json.loads(l) for l in (run / "trajectory.jsonl").open()]


def ridge_ink(path: pathlib.Path, max_pts: int, rng) -> np.ndarray:
    """Marking ink for one frame, via the pipeline's own ridge detector."""
    from trajlib import lane_calib as LC
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((0, 2))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H = g.shape[0]
    u, v = LC._ridge_points(g, int(0.55 * H), int(0.88 * H))
    uv = np.stack([u, v], axis=1).astype(float)
    if len(uv) > max_pts:                       # keep evaluations affordable
        uv = uv[rng.choice(len(uv), max_pts, replace=False)]
    return uv


def build_anchors(recs, run, n_anchors, span_s, step_s, max_pts, seed=0):
    """Pick anchors spread over the clip; return per-anchor (obs, poses).

    The pose of a later frame in the anchor's own frame comes straight from that
    anchor's trajectory window -- which is exactly the quantity the window holds,
    and is METRIC, which is what makes the scale observable at all.
    """
    rng = np.random.default_rng(seed)
    by_frame = {int(r["frame"]): r for r in recs}
    frames_dir = run / "frames"
    usable = [r for r in recs
              if r["complete"] and r["speed_ms"] > 8.0
              and (frames_dir / f"{int(r['frame']):06d}.jpg").exists()]
    if not usable:
        raise SystemExit("no usable frames — was the pipeline run with frame export?")
    picks = np.linspace(0, len(usable) - 1, n_anchors).astype(int)
    offsets = np.arange(0.0, span_s + 1e-9, step_s)

    anchors = []
    for pi in picks:
        a = usable[pi]
        t = np.asarray(a["t"], float)
        ax, ay, ayaw = (np.asarray(a[k], float) for k in ("x", "y", "yaw"))
        fps = 29.922
        obs, poses = [], []
        for dt in offsets:
            fno = int(round(a["frame"] + dt * fps))
            r = by_frame.get(fno)
            p = frames_dir / f"{fno:06d}.jpg"
            if r is None or not p.exists() or dt > t.max():
                continue
            uv = ridge_ink(p, max_pts, rng)
            if len(uv) < 200:
                continue
            obs.append(uv)
            poses.append((float(np.interp(dt, t, ax)),
                          float(np.interp(dt, t, ay)),
                          float(np.interp(dt, t, ayaw))))
        if len(obs) >= 4:
            anchors.append((obs, poses, int(a["frame"])))
    return anchors


def multi_agreement(anchors, P, grid, sigma=1.0) -> float:
    """Mean agreement over anchors — one scene quirk should not decide the fit."""
    v = [BC.agreement(o, p, P, grid, sigma) for o, p, _ in anchors]
    return float(np.mean(v)) if v else 0.0


def save_bev(anchors, P, grid, path: pathlib.Path, title: str):
    """Write the stacked BEV for one parameter set.

    The number is the evidence, but the picture is what makes a smear obvious:
    a correct calibration draws crisp lane lines, a wrong one draws a fan.
    """
    tot = None
    for obs, poses, _ in anchors[:1]:                 # one anchor, so lines stay legible
        for o, ps in zip(obs, poses):
            H = grid.accumulate_one(BC.to_anchor(BC.ground_from_pixels(o, P), ps))
            tot = H if tot is None else tot + H
    if tot is None:
        return
    img = tot.T[::-1]                 # after .T: rows = lateral y, cols = longitudinal x
                                      # so lane markings read as HORIZONTAL bands
    img = np.clip(img / max(1e-9, np.percentile(img, 99.5)), 0, 1)
    img = (255 * (1.0 - img)).astype(np.uint8)         # dark ink on light ground
    img = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (32, 32, 32), -1)
    cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=RUN)
    ap.add_argument("--anchors", type=int, default=8)
    ap.add_argument("--span", type=float, default=0.6,
                    help="seconds per anchor. ⚠️ MUST be short enough that frames still\n                         share ground: at 22 m/s a 2.0 s span is 44 m of travel against a\n                         ~35 m visible band, i.e. ZERO overlap — measured, and it made the\n                         objective score noise (a 54.7 deg yaw won). 0.6 s keeps ~60%.")
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--max-pts", type=int, default=4000)
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--bev-dir", type=pathlib.Path, default=None,
                    help="write stacked-BEV pictures for the nominal and fitted sets")
    a = ap.parse_args()

    recs = load_records(a.run)
    print(f"records {len(recs)}  frames {len(list((a.run/'frames').glob('*.jpg')))}")
    anchors = build_anchors(recs, a.run, a.anchors, a.span, a.step, a.max_pts)
    npts = sum(sum(len(o) for o in obs) for obs, _, _ in anchors)
    print(f"anchors {len(anchors)}  frames/anchor {np.mean([len(o) for o,_,_ in anchors]):.1f}  "
          f"ink points {npts}")
    grid = BC.BevGrid(x_range=(5.0, 40.0), y_range=(-10.0, 10.0), cell=0.08)
    out = {"anchors": len(anchors), "ink_points": int(npts)}

    # ---- 1. does the objective discriminate at all on real ink? -------------
    print("\n[1] does the objective discriminate on REAL ink?")
    base = dict(NOMINAL)
    probes = [("nominal (shipped)", {}),
              ("yaw -7.01", dict(yaw=np.deg2rad(-7.01))),
              ("yaw -7.01, pitch -0.64", dict(yaw=np.deg2rad(-7.01), pitch=np.deg2rad(-0.64))),
              ("yaw -7.01, pitch -2.93 (FOE)", dict(yaw=np.deg2rad(-7.01), pitch=np.deg2rad(-2.93))),
              ("height 1.6", dict(height=1.6)),
              ("height 0.05 (collapse)", dict(height=0.05))]
    scores = {}
    for name, d in probes:
        P = dict(base); P.update(d)
        s = multi_agreement(anchors, P, grid)
        scores[name] = s
        print(f"    {name:32s} {s:.6e}")
    out["probe_scores"] = scores

    # ---- 2. the f*h manifold, on real data ---------------------------------
    print("\n[2] f*h = const manifold on REAL ink (synthetic said PEAKED)")
    P_ref = dict(base); P_ref.update(yaw=np.deg2rad(-7.01), pitch=np.deg2rad(-0.64))
    fh = 1.21 * 1478.3
    hs = np.array([0.80, 0.95, 1.10, 1.21, 1.35, 1.50, 1.70, 1.90])
    man = []
    for h in hs:
        P = dict(P_ref); P.update(height=float(h), fx=float(fh / h))
        s = multi_agreement(anchors, P, grid)
        man.append(s)
        print(f"    h={h:5.2f} m  f={fh/h:7.1f} px   {s:.6e}")
    man = np.array(man)
    print(f"    -> peak at h = {hs[man.argmax()]:.2f} m,  max/min = {man.max()/man.min():.2f}x  "
          f"{'PEAKED' if man.max()/man.min() > 1.3 else 'FLAT'}")
    out["manifold"] = {"h": hs.tolist(), "score": man.tolist(),
                       "peak_h": float(hs[man.argmax()]),
                       "ratio": float(man.max() / man.min())}

    # ---- 3. joint fit of the observable parameters -------------------------
    print("\n[3] joint fit (yaw, pitch, height, fx) from the SHIPPED nominal")
    P0 = dict(base)
    free = ("yaw", "pitch", "height", "fx")
    P, info = BC.calibrate(None, None, P0, free=free,
                           bounds=dict(height=(0.5, 2.5), fx=(800.0, 2600.0)),
                           grid=grid,
                           objective=lambda PP: multi_agreement(anchors, PP, grid))
    print(f"    score {info['score0']:.4e} -> {info['score']:.4e}  ({info['nfev']} evals)")
    for k in free:
        val = np.rad2deg(P[k]) if k in ("yaw", "pitch", "roll") else P[k]
        unit = "deg" if k in ("yaw", "pitch", "roll") else ("px" if k == "fx" else "m")
        print(f"    {k:8s} -> {val:9.3f} {unit}")
    out["fit"] = {k: float(P[k]) for k in BC.PARAMS}
    out["fit_deg"] = {k: float(np.rad2deg(P[k])) for k in ("yaw", "pitch", "roll")}
    out["fit_score"] = info["score"]

    if a.bev_dir:
        a.bev_dir.mkdir(parents=True, exist_ok=True)
        save_bev(anchors, base, grid, a.bev_dir / "bev_nominal.png",
                 "SHIPPED nominal: yaw 0.00  pitch 0.00  h 1.17  f 1478")
        save_bev(anchors, P, grid, a.bev_dir / "bev_fitted.png",
                 f"BEV FIT: yaw {np.rad2deg(P['yaw']):.2f}  pitch {np.rad2deg(P['pitch']):.2f}  "
                 f"h {P['height']:.3f}  f {P['fx']:.0f}")
        print(f"wrote BEV pictures to {a.bev_dir}")

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
