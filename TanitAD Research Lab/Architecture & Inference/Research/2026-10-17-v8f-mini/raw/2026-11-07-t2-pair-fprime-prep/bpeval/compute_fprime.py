#!/usr/bin/env python3
"""compute_fprime.py -- F' for PREREG_V8F_LORA Amendment 4 item 3, by F's exact definition, behind an identity gate.

F (RESULT_V8F_GRID.md; v8f_mini_eval.py:559) = | mean_over_2399_windows(ade6 of seed 0) - mean_over_2399_windows(ade6 of seed 1) |,
the per-window `ade6` arrays of the two `v8f_mini_eval.py run` outputs (eval_<tag>.npz), ADE6 = mean over the 30 plan steps of the
step distance to the human path. The banked M2b gives F = 0.12074 (reads 0.1207 at 4 dp). F' is the same number from B'_s0 and B'_s1.

Inputs (all from the pulled box-2 run, --out-dir):
  eval_M2b_s0-idcheck.{json,npz}   the IDENTITY eval: the banked M2b_s0 checkpoint on box 2's ORIGINAL eval cache, the invocation of record
  eval_M2bp_s0.{json,npz}          B'_s0 on the pod eval cache, everything else identical
  eval_M2bp_s1.{json,npz}          B'_s1 likewise
Pinned reference: --banked-dir (default ./banked) = raw/full/eval/eval_M2b_s{0,1}.{json,npz} of the tip, sha256 in banked/SHA256.txt.

REFUSES (exit 3) when the identity eval misses its tolerance, when a file is missing, when the windows differ, or when the banked F does not
reproduce. The beside rows (B' - M2b per seed) are DESCRIPTIVE: they decide nothing. Estimator note: F' answers the TRAINING-seed question
only through two seeds (H-ESTIM-SEED-1); no bootstrap interval is attached to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F_BANKED_4DP = 0.1207
# Identity tolerances. Why these numbers: two banked evals of the SAME checkpoint on box 2 (raw/full eval_M2b_s0 and the later
# eval_M2b_s0-tacpred) agree on the per-window ADE6 array to 0.0 (MEASURED: means 2.175306138079546 and 2.175306138079546). A re-run on the same
# GPU, torch and checkpoint is therefore expected to be bit-equal; the tolerance only admits float-reduction jitter and is 0.08 % of F.
TOL_WIN = 1e-4          # max |delta| of any per-window ade6 / fde6 / ade2 value, metres
TOL_MEAN = 1e-5         # |delta| of the 2399-window mean, metres (a mean averages the jitter down)
TOL_JSON = 1.5e-4       # numeric JSON leaves are rounded to 4 dp by the harness: one rounding unit (1e-4) plus half for a flip at the boundary
BANKED_SHA = {}
for _l in open(os.path.join(HERE, "banked", "SHA256.txt"), encoding="utf-8") if os.path.exists(os.path.join(HERE, "banked", "SHA256.txt")) else []:
    _p = _l.split()
    if len(_p) == 2:
        BANKED_SHA[_p[1].lstrip("*")] = _p[0]

ROWS = [  # (label, family, json path under `raw`)  -- the four families, all DESCRIPTIVE
    ("ADE6 m", "TRAJ", ("traj", "ade6_m", "mean")), ("FDE6 m", "TRAJ", ("traj", "fde6_m", "mean")), ("ADE2 m", "TRAJ", ("traj", "ade2_m", "mean")),
    ("speed MAE m/s", "LONGITUDINAL", ("families", "longitudinal", "speed_mae_mps")),
    ("accel MAE m/s2", "LONGITUDINAL", ("families", "longitudinal", "accel_mae_mps2")),
    ("along-track MAE m", "LONGITUDINAL", ("families", "longitudinal", "along_mae_m")),
    ("progress ratio mean", "LONGITUDINAL", ("families", "longitudinal", "ego_progress", "progress_ratio_mean")),
    ("heading MAE deg", "LATERAL", ("families", "lateral", "heading_mae_deg")),
    ("yaw-rate MAE deg/s", "LATERAL", ("families", "lateral", "yaw_rate_mae_degps")),
    ("curvature MAE 1/m", "LATERAL", ("families", "lateral", "curvature_mae_1pm")),
    ("cross-track MAE m", "LATERAL", ("families", "lateral", "cross_mae_m")),
    ("lateral decision kappa", "TACTICAL", ("families", "tactical_from_trajectory", "lateral_decision", "kappa")),
    ("longitudinal decision kappa", "TACTICAL", ("families", "tactical_from_trajectory", "longitudinal_decision", "kappa")),
    ("goal point error m", "TACTICAL", ("families", "tactical_from_trajectory", "goal_setting", "goal_point_error_m")),
    ("T lat macro-F1", "TACTICAL", ("tactical", "lat", "macro_f1")), ("T lon macro-F1", "TACTICAL", ("tactical", "lon", "macro_f1")),
    ("S pos err @30 m", "STRATEGIC (PAUSED, PI ITEM 25; shown, decides nothing)", ("strategic", "pos_err_m@30", "mean")),
    ("S pos err @80 m", "STRATEGIC (PAUSED, PI ITEM 25; shown, decides nothing)", ("strategic", "pos_err_m@80", "mean")),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refuse(msg, code=3):
    print("ZZ FPRIME REFUSED: %s ZZ" % msg)
    sys.exit(code)


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, o


def pick(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def load(dirp, stem):
    j, n = os.path.join(dirp, stem + ".json"), os.path.join(dirp, stem + ".npz")
    for p in (j, n):
        if not os.path.isfile(p):
            refuse("missing file %s" % p, 2)
    z = np.load(n)
    return json.load(open(j, encoding="utf-8")), {k: z[k] for k in ("ade6", "fde6", "ade2", "eid")}, (sha256(j), sha256(n))


def f_of(a6, b6):
    """F's definition: |mean(ade6_s0) - mean(ade6_s1)| over the same windows (v8f_mini_eval.py:559)."""
    return float(abs(a6.mean() - b6.mean()))


def identity(ref, new, tag):
    (rj, rz, _), (nj, nz, _) = ref, new
    out = {"tag": tag, "tolerances": {"per_window_m": TOL_WIN, "mean_m": TOL_MEAN, "json_leaf": TOL_JSON}, "checks": [], "miss": []}
    def chk(name, ok, val):
        out["checks"].append({"name": name, "ok": bool(ok), "value": val})
        if not ok:
            out["miss"].append(name)
    chk("n_windows_equal", rz["ade6"].shape == nz["ade6"].shape == (2399,), [list(rz["ade6"].shape), list(nz["ade6"].shape)])
    if out["miss"]:
        return out
    chk("episode_ids_equal", bool(np.array_equal(rz["eid"], nz["eid"])), None)
    for k in ("ade6", "fde6", "ade2"):
        chk("%s_max_abs_window_delta" % k, float(np.abs(rz[k] - nz[k]).max()) <= TOL_WIN, float(np.abs(rz[k] - nz[k]).max()))
    chk("ade6_mean_delta", abs(float(rz["ade6"].mean() - nz["ade6"].mean())) <= TOL_MEAN, float(rz["ade6"].mean() - nz["ade6"].mean()))
    ra, na = dict(leaves(rj["raw"])), dict(leaves(nj["raw"]))
    chk("json_same_leaf_paths", set(ra) == set(na), sorted(set(ra) ^ set(na))[:10])
    bad = []
    for p in sorted(set(ra) & set(na)):
        a, b = ra[p], na[p]
        num = isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool)
        if num:
            if abs(a - b) > TOL_JSON:
                bad.append([p, a, b])
        elif a != b:
            bad.append([p, a, b])
    chk("json_leaves_within_tolerance", not bad, {"n_leaves": len(ra), "n_bad": len(bad), "first_bad": bad[:8]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="E:/Projects/TanitAD-artifacts/lorapod_eval/out")
    ap.add_argument("--banked-dir", default=os.path.join(HERE, "banked"))
    ap.add_argument("--result", default=None, help="default <out-dir>/FPRIME.json")
    ap.add_argument("--banked-selftest", action="store_true",
                    help="use the banked M2b files as identity, B'_s0 and B'_s1: F' must equal F = 0.1207 and every beside row 0 (the tool's own check)")
    a = ap.parse_args(argv)
    bd = a.banked_dir
    for f, h in BANKED_SHA.items():
        if sha256(os.path.join(bd, f)) != h:
            refuse("banked reference %s does not match its pinned sha256" % f)
    if len(BANKED_SHA) != 4:
        refuse("banked/SHA256.txt must pin 4 files (json+npz, seeds 0 and 1)")
    b0, b1 = load(bd, "eval_M2b_s0"), load(bd, "eval_M2b_s1")
    F = f_of(b0[1]["ade6"], b1[1]["ade6"])
    if round(F, 4) != F_BANKED_4DP:
        refuse("the banked F does not reproduce from F's definition: %.6f (want %.4f at 4 dp)" % (F, F_BANKED_4DP))
    if a.banked_selftest:
        idc, p0, p1 = b0, b0, b1
    else:
        idc, p0, p1 = load(a.out_dir, "eval_M2b_s0-idcheck"), load(a.out_dir, "eval_M2bp_s0"), load(a.out_dir, "eval_M2bp_s1")
    ident = identity(b0, idc, "M2b_s0 on box 2's original eval cache vs banked raw/full/eval/eval_M2b_s0")
    if ident["miss"]:
        print(json.dumps(ident, indent=1, default=float)[:3000])
        refuse("the IDENTITY eval missed: %s -- F' is not computed (the eval pipeline on box 2 does not reproduce the banked M2b_s0)" % ident["miss"])
    for nm, p in (("B'_s0", p0), ("B'_s1", p1)):
        if not (p[1]["ade6"].shape == (2399,) and np.array_equal(p[1]["eid"], b0[1]["eid"])):
            refuse("%s is not on the 2,399 windows / 141 episodes of the banked grid" % nm)
        if p[0].get("kin_mode") != "yaw" or p[0].get("ages") != "cycle" or p[0].get("plan") != "none" or "families" not in p[0]["raw"]:
            refuse("%s was not run as the invocation of record (kin yaw, ages cycle, plan none, --families)" % nm)
    Fp = f_of(p0[1]["ade6"], p1[1]["ade6"])
    Fp_json = abs(pick(p0[0]["raw"], ROWS[0][2]) - pick(p1[0]["raw"], ROWS[0][2]))
    if abs(Fp - Fp_json) > TOL_JSON:
        refuse("F' from the npz (%.6f) and from the 4-dp JSON means (%.6f) disagree" % (Fp, Fp_json))
    rows = []
    for lab, fam, path in ROWS:
        r = {"row": lab, "family": fam}
        for s, (bp, bm) in enumerate(((p0, b0), (p1, b1))):
            x, y = pick(bp[0]["raw"], path), pick(bm[0]["raw"], path)
            r["M2b_s%d" % s], r["Bp_s%d" % s] = y, x
            r["Bp_minus_M2b_s%d" % s] = None if (x is None or y is None) else round(x - y, 6)
        rows.append(r)
    res = {"definition": "F = |mean ade6(s0) - mean ade6(s1)| over the 2,399 eval-141 windows (v8f_mini_eval.py:559; RESULT_V8F_GRID.md)",
           "estimator_note": "training-seed variance from TWO seeds only (H-ESTIM-SEED-1); the episode-cluster bootstrap does not answer it; no interval is attached",
           "F_M2b_banked_m": F, "F_M2b_banked_4dp": round(F, 4), "2F_banked_m": 2 * F,
           "Fprime_m": Fp, "Fprime_4dp": round(Fp, 4), "2Fprime_m": 2 * Fp, "Fprime_from_json_4dp_means": Fp_json,
           "ADE6_mean_m": {"M2b_s0": float(b0[1]["ade6"].mean()), "M2b_s1": float(b1[1]["ade6"].mean()),
                           "Bp_s0": float(p0[1]["ade6"].mean()), "Bp_s1": float(p1[1]["ade6"].mean())},
           "identity": ident, "beside_rows_DESCRIPTIVE_decide_nothing": rows,
           "unavailable_in_this_eval": "distance keeping (headway / time-gap / TTC): the eval's own lead input is not built; v8f_dk_read.py reads it from the saved npz (a separate step)",
           "selftest": bool(a.banked_selftest),
           "file_sha256": {"identity_json_npz": list(idc[2]), "Bp_s0_json_npz": list(p0[2]), "Bp_s1_json_npz": list(p1[2])}}
    outp = a.result or os.path.join(a.out_dir, "FPRIME.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("identity: PASS (%d checks, tolerances window %.0e / mean %.0e / json %.1e)" % (len(ident["checks"]), TOL_WIN, TOL_MEAN, TOL_JSON))
    print("F  (banked M2b)  = %.5f m (reads %.4f)   2F  = %.4f m" % (F, round(F, 4), 2 * F))
    print("F' (B' s0, s1)   = %.5f m (reads %.4f)   2F' = %.4f m" % (Fp, round(Fp, 4), 2 * Fp))
    print("%-28s %-26s %9s %9s %9s %9s" % ("row", "family", "M2b_s0", "Bp_s0", "M2b_s1", "Bp_s1"))
    for r in rows:
        f = lambda v: "n/a" if v is None else "%.4f" % v
        print("%-28s %-26s %9s %9s %9s %9s" % (r["row"], r["family"][:26], f(r["M2b_s0"]), f(r["Bp_s0"]), f(r["M2b_s1"]), f(r["Bp_s1"])))
    print("written", outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
