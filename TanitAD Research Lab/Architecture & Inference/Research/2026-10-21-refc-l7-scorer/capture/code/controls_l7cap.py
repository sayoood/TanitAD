"""controls_l7cap.py -- the capture's controls, as functions (so tests and the real smoke use the SAME code).

  k1_identity(wrapped_dir, [unwrapped_dir, ...])   control (a): the capture changes nothing the roll emits. Reading rule = R8-PICKER ADDENDUM 2:
        PASS iff (1) the wrapped dump is bit-identical (equal_nan) to AT LEAST ONE unwrapped dump, (2) every INTEGER array is identical across
        ALL dumps (every pick), (3) the wrapped dump's max float |diff| against every unwrapped dump is <= the unwrapped-vs-unwrapped envelope
        (0 when only one unwrapped dump exists, which makes it a bit-identity requirement). Compared files: ep*.npz, decisions/ep*.npz,
        refcv6_extras.npz -- the files the UNWRAPPED roll writes. The capture's own files are not compared.
  machinery(dump_dir, gt6_dir=None, agents_npz=None)   control (b): run scorer9's `l7_targets_pai.build` on the captured windows, then
        recompute the per-window own-pick ADE two ways that do not touch the builder: from the dumped `anchor_traj[w, sel_idx[w]]` and from
        the roll's EMITTED plan `os` (ep*.npz), against GT `g` -- plain numpy, float64. All three must agree to 1e-6 m. A dump with the WRONG
        candidate index disagrees on the second (the emitted plan is not that candidate) and goes RED  (control (c)).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

SCORER9_CODE = os.environ.get("L7_SCORER9_CODE", "C:/Users/Admin/refcv8_pf/scorer9/code")
UNWRAPPED_FILES = ("ep*.npz", os.path.join("decisions", "ep*.npz"), "refcv6_extras.npz")


def _arrays(dump: str) -> dict:
    out = {}
    for pat in UNWRAPPED_FILES:
        for f in sorted(glob.glob(os.path.join(dump, pat))):
            rel = os.path.relpath(f, dump).replace("\\", "/")
            with np.load(f, allow_pickle=True) as z:
                for k in z.files:
                    out[(rel, k)] = np.asarray(z[k])
    return out


def _cmp(a: dict, b: dict) -> dict:
    missing = sorted(set(b) - set(a))
    extra = sorted(set(a) - set(b))
    differing, max_f, int_diff = [], 0.0, 0
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            differing.append((k, "shape/dtype"))
            continue
        if x.dtype.kind in "fc":
            same = np.array_equal(x, y, equal_nan=True)
            if not same:
                d = np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))) if x.size else 0.0
                max_f = max(max_f, float(d))
                differing.append((k, "float"))
        elif x.dtype.kind in "biuUSO":
            if not np.array_equal(x, y):
                differing.append((k, "int/str"))
                int_diff += 1
    return {"n_arrays": len(set(a) & set(b)), "missing_in_first": [list(m) for m in missing[:5]], "n_missing_in_first": len(missing), "n_missing_in_second": len(extra),
            "differing": [list(d[0]) + [d[1]] for d in differing[:8]], "n_differing": len(differing),
            "bit_identical": (not differing and not missing and not extra and len(set(a) & set(b)) > 0), "max_abs_float_diff": max_f, "n_int_arrays_differing": int_diff}


def k1_identity(wrapped: str, unwrapped: list) -> dict:
    w = _arrays(wrapped)
    us = [_arrays(u) for u in unwrapped]
    vs = [_cmp(w, u) for u in us]
    env = 0.0
    uu = []
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            c = _cmp(us[i], us[j])
            uu.append(c)
            env = max(env, c["max_abs_float_diff"])
    # an EMPTY comparison is INCONCLUSIVE, never a match (the empty-string-blob hole of CLAUDE.md): every pair must have compared >= 1 array and
    # carry the same array set on both sides
    nonempty = all(v["n_arrays"] > 0 and v["n_missing_in_first"] == 0 and v["n_missing_in_second"] == 0 for v in vs) and bool(vs)
    ints_ok = all(v["n_int_arrays_differing"] == 0 for v in vs) and all(c["n_int_arrays_differing"] == 0 for c in uu)
    some_bit = any(v["bit_identical"] for v in vs)
    within = all(v["max_abs_float_diff"] <= env for v in vs)
    return {"rule": "R8-PICKER Addendum 2 reading rule", "n_unwrapped": len(us), "unwrapped_vs_unwrapped_envelope_m": env,
            "wrapped_vs_each": vs, "bit_identical_to_at_least_one": some_bit, "all_int_arrays_identical": ints_ok,
            "float_diff_within_envelope": within, "nonempty_same_array_sets": nonempty, "PASS": bool(nonempty and some_bit and ints_ok and within)}


def machinery(dump: str, gt6_dir: str | None = None, agents_npz: str | None = None, tol: float = 1e-6) -> dict:
    sys.path.insert(0, SCORER9_CODE)
    import l7_targets_pai as TP
    res, ctl, timing = TP.build(dump, gt6_dir, agents_npz, None)
    with np.load(os.path.join(dump, "fan_extras.npz"), allow_pickle=True) as z:
        ci, ws = np.asarray(z["clip_index"]), np.asarray(z["ws"])
        fan = np.asarray(z["anchor_traj"], np.float64)
        sel = np.asarray(z["sel_idx"]).astype(np.int64).reshape(-1)
    n = len(ci)
    # the roll's own GT and emitted plan, straight from ep*.npz (independent of the builder and of fan_extras)
    own = {}
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        with np.load(f, allow_pickle=True) as e:
            c = int(np.asarray(e["clip_index"]).ravel()[0])
            for j, w in enumerate(np.asarray(e["ws"]).ravel()):
                own[(c, int(w))] = (np.asarray(e["g"][j], np.float64), np.asarray(e["os"][j], np.float64))
    ade_builder = np.asarray([res["ade_s2"][i, sel[i]] for i in range(n)], np.float64)
    ade_fan = np.empty(n)
    ade_os = np.empty(n)
    for i in range(n):
        g, os_ = own[(int(ci[i]), int(ws[i]))]
        ade_fan[i] = np.linalg.norm(fan[i, sel[i], :4] - g, axis=-1).mean()
        ade_os[i] = np.linalg.norm(os_ - g, axis=-1).mean()
    d1 = float(np.abs(ade_builder - ade_fan).max())
    d2 = float(np.abs(ade_builder - ade_os).max())
    c_os = ctl["C_OS_fan_at_sel_equals_bank_os_max_abs_m"]
    out = {"n_windows": n, "builder_vs_fan_recompute_max_abs_m": d1, "builder_vs_emitted_plan_recompute_max_abs_m": d2, "tol_m": tol,
           "builder_controls": ctl, "mean_own_pick_ade_s2_m": float(ade_builder.mean()),
           "PASS": bool(d1 <= tol and d2 <= tol and c_os["pass"] and ctl["C_GT_gt_as_candidate_cost_m"]["pass"]),
           "targets_timing": timing}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("k1")
    k.add_argument("--wrapped", required=True)
    k.add_argument("--unwrapped", nargs="+", required=True)
    k.add_argument("--out", required=True)
    m = sub.add_parser("machinery")
    m.add_argument("--dump", required=True)
    m.add_argument("--gt6", default=None)
    m.add_argument("--agents-npz", default=None)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "k1":
        r = k1_identity(a.wrapped, a.unwrapped)
    else:
        r = machinery(a.dump, a.gt6, a.agents_npz)
    json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(a.cmd, "PASS" if r["PASS"] else "RED", ("ZZL7CCTLOKZZ" if r["PASS"] else "ZZL7CCTLREDZZ"))
    return 0 if r["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
