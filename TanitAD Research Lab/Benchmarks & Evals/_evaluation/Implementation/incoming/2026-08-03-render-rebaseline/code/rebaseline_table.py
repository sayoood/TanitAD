#!/usr/bin/env python3
"""SUPERSEDED -> CORRECTED table for the NuRec render-fidelity absolutes (P2).

Reads the `render_quality.py` report.json pairs produced at reference offset 0 (the
shipped, wrong indexing) and at the per-scene corrected offset, and reports:

  * every arm's absolute grad-NCC before and after re-baselining;
  * the BEFORE->AFTER improvement claim (the shipped "+23.4 %" and "+35.1 %") at both
    offsets, so the question "does the improvement survive?" is answered with a number;
  * a PAIRED bootstrap over frames for every delta.

⚠️ ESTIMATOR, stated because this programme does not accept an interval without one:
**paired bootstrap over the probed FRAMES of one clip**, B=10000. The unit is the frame.
Frames of a single clip are autocorrelated, so this interval is OPTIMISTIC and it is a
within-scene statement only; it is NOT the episode-cluster bootstrap and must never be
quoted as one. Where a claim needs to generalise across scenes, the n is the number of
SCENES (here 2), and that is reported separately and is tiny.

⛔ grad-NCC only. PSNR, plain NCC and MAE are RETRACTED on these night clips (a wrong
reference frame outranked the correct one on both PSNR and NCC). They are printed for
context and decide nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """Paired bootstrap over frames of mean(a) - mean(b). Returns point, lo, hi, frac."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size > 0
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    draws = d[idx].mean(axis=1)
    return {"delta": round(float(d.mean()), 4),
            "lo": round(float(np.quantile(draws, 0.025)), 4),
            "hi": round(float(np.quantile(draws, 0.975)), 4),
            "separated": bool(np.quantile(draws, 0.025) > 0 or np.quantile(draws, 0.975) < 0),
            "frac_frames_positive": round(float((d > 0).mean()), 3),
            "n_frames": int(d.size),
            "estimator": "paired bootstrap over frames, B=%d (within-scene; frames are "
                         "autocorrelated so this is OPTIMISTIC and is NOT an "
                         "episode-cluster bootstrap)" % n_boot}


def arms_of(report: dict) -> dict:
    return {a["arm"]: a for a in report["arms"]}


def per_frame_gradncc(arm: dict) -> np.ndarray:
    return np.array([x["grad_ncc"] for x in arm["per_frame"]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True,
                    metavar="SCENE=OFF0_JSON,CORRECTED_JSON")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"metric": "gradient-NCC (the ONLY admissible fidelity metric on these clips)",
           "what": "superseded (reference offset 0) vs corrected (per-scene offset)",
           "scenes": {}}
    for spec in a.pair:
        scene, files = spec.split("=", 1)
        f0, f1 = files.split(",")
        r0, r1 = json.loads(Path(f0).read_text()), json.loads(Path(f1).read_text())
        A0, A1 = arms_of(r0), arms_of(r1)
        assert r0["frames"] == r1["frames"], "the two runs must score identical frames"
        row = {"ref_offset_superseded": r0["ref_offset"],
               "ref_offset_corrected": r1["ref_offset"],
               "frames": r0["frames"], "n_frames": len(r0["frames"]),
               "alignment_gate_corrected_run": (r1.get("alignment_gate") or {}).get("pass"),
               "arms": {}, "improvement_claims": {}}
        for name in A0:
            if name not in A1:
                continue
            v0, v1 = per_frame_gradncc(A0[name]), per_frame_gradncc(A1[name])
            row["arms"][name] = {
                "grad_ncc_superseded": round(float(v0.mean()), 4),
                "grad_ncc_corrected": round(float(v1.mean()), 4),
                "abs_change": round(float(v1.mean() - v0.mean()), 4),
                "pct_change": round(100 * float(v1.mean() - v0.mean()) / float(v0.mean()), 1),
                "neg_control_all_pass_superseded": A0[name]["neg_control_all_pass"],
                "neg_control_all_pass_corrected": A1[name]["neg_control_all_pass"],
                "raster_ms_median": A1[name]["raster_ms_median"],
                "mae_full_superseded_CONTEXT_ONLY": A0[name]["mae_full"],
                "mae_full_corrected_CONTEXT_ONLY": A1[name]["mae_full"],
                "paired_offset_effect": paired_boot(v1, v0)}
        base = "BEFORE_base"
        for after in [n for n in A0 if n.startswith("AFTER")]:
            if base not in A0 or after not in A1:
                continue
            b0, a0_ = per_frame_gradncc(A0[base]), per_frame_gradncc(A0[after])
            b1, a1_ = per_frame_gradncc(A1[base]), per_frame_gradncc(A1[after])
            row["improvement_claims"][f"{base} -> {after}"] = {
                "superseded": {"before": round(float(b0.mean()), 4),
                               "after": round(float(a0_.mean()), 4),
                               "pct": round(100 * (a0_.mean() - b0.mean()) / b0.mean(), 1),
                               "paired": paired_boot(a0_, b0)},
                "corrected": {"before": round(float(b1.mean()), 4),
                              "after": round(float(a1_.mean()), 4),
                              "pct": round(100 * (a1_.mean() - b1.mean()) / b1.mean(), 1),
                              "paired": paired_boot(a1_, b1)}}
        out["scenes"][scene] = row

    Path(a.out).write_text(json.dumps(out, indent=1))

    for scene, row in out["scenes"].items():
        print(f"\n=== {scene}  n_frames={row['n_frames']}  "
              f"offset {row['ref_offset_superseded']} -> {row['ref_offset_corrected']}  "
              f"gate_on_corrected={row['alignment_gate_corrected_run']}")
        print(f"{'arm':<28}{'SUPERSEDED':>12}{'CORRECTED':>11}{'abs':>9}{'%':>8}"
              f"{'negctl(sup/cor)':>18}{'ms':>9}")
        for n, v in row["arms"].items():
            print(f"{n:<28}{v['grad_ncc_superseded']:>12.4f}{v['grad_ncc_corrected']:>11.4f}"
                  f"{v['abs_change']:>+9.4f}{v['pct_change']:>+8.1f}"
                  f"{str(v['neg_control_all_pass_superseded'])[:5]+'/'+str(v['neg_control_all_pass_corrected'])[:5]:>18}"
                  f"{v['raster_ms_median']:>9.1f}")
        for claim, v in row["improvement_claims"].items():
            s, c = v["superseded"], v["corrected"]
            print(f"  IMPROVEMENT {claim}")
            print(f"    superseded: {s['before']:.4f} -> {s['after']:.4f}  "
                  f"= {s['pct']:+.1f} %   paired {s['paired']['delta']:+.4f} "
                  f"[{s['paired']['lo']:+.4f}, {s['paired']['hi']:+.4f}] "
                  f"sep={s['paired']['separated']}")
            print(f"    CORRECTED : {c['before']:.4f} -> {c['after']:.4f}  "
                  f"= {c['pct']:+.1f} %   paired {c['paired']['delta']:+.4f} "
                  f"[{c['paired']['lo']:+.4f}, {c['paired']['hi']:+.4f}] "
                  f"sep={c['paired']['separated']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
