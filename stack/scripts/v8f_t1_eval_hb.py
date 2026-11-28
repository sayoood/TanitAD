"""v8f_t1_eval_hb.py -- the T1-kin executor of record from 2026-11-20: the pinned T1 tool with a heading bound a car obeys.

PI ruling 2026-11-20 (decision card D16, recommendation D, "Approve all four", confirmed to the MM first-hand): the T1-kin tier gets a
heading bound, with one overlap read old/new executor; then the bounded executor is the tier of record.

Why: the pinned tool's ``hermite_frames`` holds the heading only while the tangent is below ``EPS_M`` = 1e-3 m per 0.2 s, so near a
standstill the executed heading follows millimetres of plan jitter (MEASURED, card ADD67, row 122 at bb061af6: heading offset below
2 m/s 52.9 deg without the bound, 6.6 deg with it). That heading sets the frame of the next plan and the yaw history the model reads.

The bound: frame by frame from the current knot's heading ``psi0`` and position ``knots[0]``, a heading step is at most
``kappa`` x the distance between the two frame positions (0.2 1/m = a 5 m turning radius). Positions and speeds pass through
unchanged. ``bound_heading`` is the function of card ADD67 (integ10k/v8f_t1_eval_reader_rule_rzgfh.py at bb061af6), copied
unchanged in its arithmetic so the tier of record and the study read share one definition.

The pinned tool (``stack/scripts/v8f_t1_eval.py``) is NOT edited. This wrapper replaces its module-level ``hermite_frames`` for
the duration of one run and restores it afterwards; any wrapper that imports the tool as the module ``v8f_t1_eval`` and calls its
``main`` (for example a study wrapper of a build tree) composes with it through ``--inner``.

Usage:
  python v8f_t1_eval_hb.py [--kappa-max 0.2|none] [--inner MODULE] [--hb-log JSON] <the argv of the inner tool, e.g. run ...>
  --kappa-max none runs the old executor (for the overlap read only); the default 0.2 is the tier of record.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import sys

import numpy as np

KAPPA_RECORD = 0.2                       # 1/m, the tier of record (PI 2026-11-20, D16-D; card ADD67)
KB_BINS = (("lt2", 0.0, 2.0), ("2to8", 2.0, 8.0), ("gt8", 8.0, float("inf")))


def wrap_pi(a):
    """angle into [-pi, pi)"""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def kb_new():
    """the counters of the heading bound, written to --hb-log"""
    return {"frames_total": 0, "frames_bound_acted": 0, "max_excess_rad": None, "max_abs_diff_where_not_acted": 0.0,
            "by_speed": {n: {"frames_total": 0, "frames_bound_acted": 0} for n, _, _ in KB_BINS}}


def bound_heading(xy, psi, v, xy0, psi0, kappa, rec=None):
    """the heading bound: frame by frame from psi0 and the position xy0 of the current knot; a heading step is at most kappa x the
    distance between the two frame positions. The output heading is psi[f] itself when its step from the previous output heading is
    within the bound, else the previous output heading turned by exactly the bound towards psi[f]. Counters into rec when given."""
    xy = np.asarray(xy, np.float64)
    psi = np.asarray(psi, np.float64)
    vv = np.asarray(v, np.float64)
    out = np.empty(len(psi), np.float64)
    prev, pxy = float(psi0), np.asarray(xy0, np.float64)
    for f in range(len(psi)):
        lim = float(kappa) * float(np.hypot(xy[f, 0] - pxy[0], xy[f, 1] - pxy[1]))
        d = wrap_pi(float(psi[f]) - prev)
        acted = not (abs(d) <= lim)
        o = wrap_pi(prev + (1.0 if d > 0 else -1.0) * lim) if acted else float(psi[f])
        if rec is not None:
            ex = abs(wrap_pi(o - prev)) - lim
            rec["max_excess_rad"] = ex if rec["max_excess_rad"] is None else max(rec["max_excess_rad"], ex)
            rec["frames_total"] += 1
            rec["frames_bound_acted"] += int(acted)
            if not acted:
                rec["max_abs_diff_where_not_acted"] = max(rec["max_abs_diff_where_not_acted"], abs(o - float(psi[f])))
            for n, lo, hi in KB_BINS:
                if lo <= float(vv[f]) < hi:
                    rec["by_speed"][n]["frames_total"] += 1
                    rec["by_speed"][n]["frames_bound_acted"] += int(acted)
        out[f] = o
        prev, pxy = o, xy[f]
    return out


def bounded_frames(orig, kappa, rec=None):
    """``orig``'s frames with the heading bound applied; ``kappa`` None returns ``orig`` itself (the old executor)."""
    if kappa is None:
        return orig

    def frames(k_prev, knots, m, psi0):
        xy, psi, v = orig(k_prev, knots, m, psi0)
        psi = bound_heading(xy, psi, v, np.asarray(knots, np.float64)[0], psi0, kappa, rec)
        return xy, psi, v
    frames.__wrapped__ = orig
    return frames


@contextlib.contextmanager
def installed(tool, kappa, rec=None):
    """replace ``tool.hermite_frames`` by the bounded frames for the duration of the block, then restore the original object."""
    orig = tool.hermite_frames
    tool.hermite_frames = bounded_frames(orig, kappa, rec)
    try:
        yield tool
    finally:
        tool.hermite_frames = orig


def parse_kappa(s):
    if str(s).lower() == "none":
        return None
    k = float(s)
    if not (math.isfinite(k) and k > 0.0):
        raise SystemExit(f"REFUSED: --kappa-max must be a positive finite number or none, got {s}")
    return k


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--kappa-max", default=str(KAPPA_RECORD))
    ap.add_argument("--inner", default="v8f_t1_eval")
    ap.add_argument("--hb-log", default=None)
    mine, rest = ap.parse_known_args(argv)
    kappa = parse_kappa(mine.kappa_max)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    inner = importlib.import_module(mine.inner)
    tool = sys.modules.get("v8f_t1_eval")
    if tool is None or not hasattr(tool, "hermite_frames"):
        raise SystemExit(f"REFUSED: --inner {mine.inner} did not import the T1 tool as the module v8f_t1_eval")
    if not hasattr(inner, "main"):
        raise SystemExit(f"REFUSED: --inner {mine.inner} has no main()")
    rec = kb_new() if kappa is not None else None
    log = {"executor": "T1-kin heading bound" if kappa is not None else "T1-kin old executor (no heading bound)",
           "kappa_max_per_m": kappa, "inner": mine.inner, "tool_file": tool.__file__, "tool_sha256": sha256_file(tool.__file__),
           "argv": argv}
    print(f"[v8f_t1_eval_hb] {log['executor']}; kappa_max={kappa}; tool sha256 {log['tool_sha256'][:12]}", flush=True)
    try:
        with installed(tool, kappa, rec):
            return inner.main(rest)
    finally:
        if mine.hb_log:
            log["bound_counters"] = rec
            with open(mine.hb_log, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
