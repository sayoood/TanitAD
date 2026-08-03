#!/usr/bin/env python3
"""Turn an `rs_sweep.py` run directory into the tables that go in the write-up.

WHY A GENERATOR AND NOT COPY-PASTE
----------------------------------
Two headline numbers in this programme have already had to be retracted because they
were transcribed out of a superseded run. Every table below is produced FROM
`report.json`, and prints the run directory it came from, so a number in the write-up
can always be traced to the directory that produced it.

⚠️ THE INTERVAL'S ESTIMATOR, STATED. The resampling unit here is the FRAME inside ONE
scene — there is no second scene and no episode structure to cluster over. So the
interval describes frame-to-frame variability WITHIN this clip and is NOT a scene-level
confidence interval; it may not be quoted as one. It is paired (same frames, both arms),
which is the part that matters for ranking two arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    """Percentile bootstrap of mean(a - b), resampling FRAMES (paired)."""
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return (float(d.mean()), float(np.percentile(m, 2.5)),
            float(np.percentile(m, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="report.json paths (or run dirs)")
    ap.add_argument("--baseline", default="g_p1.00")
    a = ap.parse_args()

    for p in a.reports:
        p = Path(p).expanduser()
        if p.is_dir():
            p = p / "report.json"
        d = json.loads(p.read_text())
        arms = {s["arm"]: s for s in d["arms"]}
        base = arms[a.baseline]
        bp = np.array(base["grad_ncc_per_frame"], float)
        print(f"\n{'=' * 108}")
        print(f"run_dir  {d['run_dir']}   panel={d.get('panel', 'rs')}  "
              f"config={d['config']}  n_frames={len(d['frames'])}  "
              f"git_sha={d.get('git_sha') or '(none)'}")
        print(f"scene    {d['scene_dir']}")
        print(f"shutter  {d['shutter_geometry']['shutter_type_declared']}  ego moves "
              f"{d['shutter_geometry']['ego_translation_m_min_max_mean']} m per readout")
        for k in ("selfcheck_s1_equals_g_p0.50_bitexact",
                  "selfcheck_ut_gate_restored_bitexact"):
            if k in d:
                print(f"selfcheck {k} = {d[k]}")
        print(f"{'=' * 108}")
        print(f"{'arm':<21}{'calls':>6}{'gradNCC':>9}{'delta':>9}{'95% CI (frames)':>21}"
              f"{'negctl':>8}{'margin':>9}{'meanA':>8}{'rmean':>8}"
              f"{'raster ms':>10}{'wall ms':>9}")
        for s in d["arms"]:
            v = np.array(s["grad_ncc_per_frame"], float)
            m, lo, hi = paired_boot(v, bp)
            ci = "[%+.4f,%+.4f]" % (lo, hi)
            ctl = "%d/%d" % (s["neg_control_pass_frames"], s["n_frames"])
            print(f"{s['arm']:<21}{s['n_render_calls_per_frame']:>6}"
                  f"{s['grad_ncc_mean']:>9.4f}{m:>+9.4f}{ci:>21}{ctl:>8}"
                  f"{s['neg_margin_mean']:>+9.4f}{s['mean_alpha']:>8.4f}"
                  f"{s['render_mean']:>8.4f}"
                  f"{s['raster_ms_median']:>10.1f}{s['wall_ms_median']:>9.1f}")
        print(f"reference frame mean = {base['ref_mean']:.4f}  "
              f"(render_mean far above it = over-brightening, not fidelity)")
        if any("mean_abs_diff_vs_native_u8" in s for s in d["arms"]):
            print(f"\nDISTANCE TO gsplat's NATIVE ROLLING SHUTTER (mean |diff| in u8 "
                  f"levels; smaller = closer):")
            for s in sorted((x for x in d["arms"]
                             if "mean_abs_diff_vs_native_u8" in x),
                            key=lambda x: x["mean_abs_diff_vs_native_u8"]):
                print(f"  {s['arm']:<21}{s['mean_abs_diff_vs_native_u8']:>8.3f}")


if __name__ == "__main__":
    main()
