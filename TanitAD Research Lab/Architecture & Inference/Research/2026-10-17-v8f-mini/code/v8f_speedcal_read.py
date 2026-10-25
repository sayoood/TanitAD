"""PREREG_V8F_SPEEDCAL reader, committed before its data: is T's speed error a HEAD BIAS (a map fitted on TRAINING windows undoes part of it) or information the inputs do not carry?

  python -I v8f_speedcal_read.py --eval-dir <speedcal evals> --fit-dir <speedcal fits> --kindx-dir-s0 <KINDX s0 evals> --kindx-dir-s1 <KINDX s1 evals> --expect-sha0 H --expect-sha1 H --out READ.json
  python -I v8f_speedcal_read.py ... --check-inputs        (every file, sha and structure refusal, NO number read)

ZERO TRAINING. The fits and evals are `stack/scripts/v8f_mini_eval_speedcal.py`'s: per seed s, `fit` (X_s's T forward on HBLEND's 141 training clips, closed-form masked least squares per speed cell) and six
evals of X_s (`own`, `identity`, `gt`, `cal`, `perm`, `ceiling`; see the driver's docstring). X_s = `M2b-mixo_s<s>`. The PREREG is binding and nothing here moves a cell set, a fit set, the map form, a bar or a
routing (a test compares every committed string with `Project Steering/PREREG_V8F_SPEEDCAL.md`).

THE READ (T1-class open loop, eval-141, paired episode-cluster bootstrap B = 2,000, 141 episodes / 2,399 windows; F = 0.1207 m, committed):
  PRIMARY    own-code ADE6, SPEEDCAL_s - X_s  (cal - own)                          <= -F AND separated, at BOTH seeds
  CEILING    (T0 diagnostic, NEVER a result) ADE6 of the same affine form fitted on the EVAL GT, minus X_s     beside; it routes the two "no" rows
  CAPTURE    (ADE6 X_s - SPEEDCAL_s) / gain_speed(X_s)                              beside
  SECONDARY  speed MAE, accel MAE, distance keeping, LEAD-VIOL: SPEEDCAL - X_s     not separated-worse at either seed
  beside     the four families (LONGITUDINAL / LATERAL by the paired family tool; TACTICAL / STRATEGIC by `v8f_paired_families.py`); per cell a, b and the RMSE before / after on train and on eval; the same map on B
CONTROLS (all BEFORE any number is read; a failed control is rc 2 and no outcome): 1 the identity map (a = 1, b = 0) reproduces X_s's own eval per window, delta 0.0; 2 the fitted map equals numpy `lstsq` on the
same masked training arrays to 1e-9 (the permuted-target map and the CEILING map too; the fit clips are disjoint from the eval clips); 3 the GT in the five speed cells reproduces KINDX's speed-only oracle per
window <= 1e-4 m; 4 the map fitted on training targets PERMUTED across windows is NOT separated-better than X_s at either seed. Added guards: X_s's own eval reproduces KINDX's per window; the hook leaves every
non-speed cell alone; T's decisions and S are X_s's own; the calibrated cells ARE the map applied to X_s's own cells; the arm is live (it moves the cells and the plan).

DECLARED INTERPRETATIONS (fixed before any SPEEDCAL number; none moves a bar):
  1. PRIMARY "separated": the paired CI excludes 0. "<= -F" is the UNROUNDED point (cal - own) <= -0.1207, on the improving side (point < 0).
  2. "not separated-worse" (SECONDARY): worse = the paired CI lies ABOVE 0 (lo > 0) for a metric where more is worse: speed MAE, accel MAE; DISTANCE KEEPING = the arm's |metric - the human's own value| for headway /
     time-gap / min-TTC (`abs_error_vs_human`, K = 10 lead block); LEAD-VIOL = the share of lead windows violating. Exactly SPEEDAVG's reading (its functions are imported, not copied).
  3. CEILING "<= -F": the UNROUNDED point (ceiling - own) <= -0.1207. The PREREG states the CEILING routing as a size comparison only (the PRIMARY row says "and separated" where it means it), so no separation is
     required; the separated flag is reported beside.
  4. "PRIMARY not separated at either seed" = the paired CI includes 0 at BOTH seeds (neither seed separated). A separated-WORSE PRIMARY is separated, so it falls to MIXED.
  5. The rows are tested IN THE TABLE'S ORDER. They are mutually exclusive by construction: HEAD BIAS PAYS and PARTIAL BIAS need a separated improvement at both seeds (and differ on "<= -F at both"), the two "no"
     rows need NO separated seed and split on CEILING ("<= -F at both" against "> -F at either"), MIXED is the rest. PARTIAL BIAS does not look at SECONDARY (the row names none); both seeds <= -F with a
     separated-worse SECONDARY is MIXED.
  6. gain_speed(X_s) is KINDX's (`eval_M2b-mixo_s<s>-kindx-own` - `-kindx-only-speed`); the driver's own-minus-gt is reported beside it.
STRATEGIC is in force for v8F (PI-V8F-2026-10-17, retractions R72 / R73). S is X_s's own in every arm, so S's checkpoint error is identical to X_s's by construction; the reader CHECKS that and reports the numbers.
Tier stamp: "T1-class open loop (own held outputs; no recorded future action)"; CEILING and the GT arm are stamped T0 (they read the eval GT).
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2,000, over EPISODES: it answers "would another draw of EPISODES say this?" and is BLIND to training variance (H-ESTIM-SEED-1). The second seed IS the
replicate the PREREG asks for; a separated CI at two seeds is still necessary, not sufficient, for a lever claim.
Refusals (rc 2, one prefix): a missing file, an eval that is not the mode / seed / checkpoint it claims, a limited (smoke) run, a failed control, a window list that moved."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PKG = HERE.parents[1]
RAW = PKG / "raw"
SCHEMA = "v8f-speedcal-read-v1"
TIER = "T1-class open loop (own held outputs; no recorded future action)"
TIER_T0 = "T0 (reads the eval GT): a diagnostic, never a result"
F_BAR = 0.1207                                   # PREREG_V8F_SPEEDCAL: F, committed
B_BOOT = 2000
N_WINDOWS, N_EPISODES = 2399, 141
SEEDS = (0, 1)
MODES = ("own", "identity", "gt", "cal", "perm", "ceiling")
TOL_M = 1e-4                                     # control 3's per-window bar (m)
TOL_LSTSQ = 1e-9                                 # control 2's bar: the fitted map against numpy lstsq
TOL_CELLS = 1e-5                                 # the same model run in two processes: the cells agree to float noise
X0_SHA = "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d"        # X_s0 = M2b-mixo_s0 (KINDX)
SPEED_CELLS = ("lon_v_target_ms", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
OUTCOMES = ("HEAD BIAS PAYS", "PARTIAL BIAS", "MAP DOES NOT TRANSFER", "INFORMATION-LIMITED", "MIXED")
#: the PREREG's outcome table, VERBATIM and in the PREREG's order (a test compares every cell with the file): condition, committed next step
TABLE = {
    "HEAD BIAS PAYS": ("PRIMARY and SECONDARY at both seeds",
                       "The calibration joins `forward_infer`: 10 buffers per model, deployable at zero training. The HSPEC dominance row is re-read on the calibrated model against P10. "
                       "Then a T-side debiasing arm, so one model needs no post-map, under its own PREREG"),
    "PARTIAL BIAS": ("PRIMARY < 0 and separated at both seeds, but not ≤ −F at both",
                     "Report CAPTURE. Next: a non-linear (isotonic, per-regime) calibration under its own PREREG"),
    "MAP DOES NOT TRANSFER": ("PRIMARY not separated at either seed, AND CEILING ≤ −F at both seeds",
                              "The error is affine-recoverable but the 141-clip fit does not carry to eval. Next: the same map fitted on all 4,512 training clips' windows"),
    "INFORMATION-LIMITED": ("PRIMARY not separated at either seed, AND CEILING > −F at either seed",
                            "T's speed output carries no affine-recoverable information about the right speed. The speed lever moves to perception (what T sees: temporal context, "
                            "encoder adaptation). That needs compute beyond box 2, so a PI go"),
    "MIXED": ("anything else", "reported per seed; seeds before windows"),
}


def refuse(msg):
    print(f"SPEEDCAL READ REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_RDA_PATH = HERE.parent / "v8f_speedavg_read.py"
#: SPEEDAVG's reader: its paired estimator wrapper, SECONDARY, CAPTURE and the family tools are IMPORTED (never copied), so SPEEDCAL reads exactly what SPEEDAVG read
RDA = _load("v8f_speedavg_read_for_speedcal", _RDA_PATH) if _RDA_PATH.exists() else None
if RDA is not None:
    RDA.refuse = refuse                          # their refusals carry this reader's prefix and rc 2
    sha256 = RDA.sha256
    paired, dev, same_json, pf_default, _paths = RDA.paired, RDA.dev, RDA.same_json, RDA.pf_default, RDA._paths
    primary_met, primary_negative_separated, separated_worse = RDA.primary_met, RDA.primary_negative_separated, RDA.separated_worse
    secondary_of, capture = RDA.secondary_of, RDA.capture
else:                                            # a stack-only tree (the lander): the module still imports; nothing here runs without the package code
    sha256 = paired = dev = same_json = pf_default = _paths = primary_met = primary_negative_separated = separated_worse = secondary_of = capture = None


def _jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ the pure pieces (numpy)
def lstsq_map(x, y, m):
    """numpy's own `lstsq` of  [x 1] [a b]^T = y  over the masked windows (control 2's REFERENCE: independent of the driver's closed form)."""
    x = np.asarray(x, np.float64)[np.asarray(m, bool)]
    y = np.asarray(y, np.float64)[np.asarray(m, bool)]
    sol = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    return float(sol[0]), float(sol[1])


def calibrate_std_np(std, a, b, mu, sd):
    """The driver's formula, restated here so the reader checks the hook against ITS OWN reading of the PREREG:  std' = a * std + ((a - 1) * mu + b) / sd  (float64, cast back)."""
    std = np.asarray(std)
    a, b, mu, sd = (np.asarray(v, np.float64) for v in (a, b, mu, sd))
    return (a * std.astype(np.float64) + ((a - 1.0) * mu + b) / sd).astype(std.dtype)


def to_phys(std, mu, sd):
    return np.asarray(std, np.float64) * np.asarray(sd, np.float64) + np.asarray(mu, np.float64)


def cell_rmse(x, y, has):
    """Per cell: n and the RMSE of x (physical, m/s) to the GT y on the windows with a v9 row and a finite GT."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    m = np.asarray(has, bool)[:, None] & np.isfinite(y)
    out = {}
    for i, nm in enumerate(SPEED_CELLS):
        k = m[:, i]
        out[nm] = {"n": int(k.sum()), "rmse": float(np.sqrt(np.mean((x[k, i] - y[k, i]) ** 2))) if k.any() else None}
    return out


# ------------------------------------------------------------------------------------------------ inputs
def load_eval(eval_dir, name, need_cells=False, need_json=True):
    pz, pj = os.path.join(eval_dir, f"eval_{name}.npz"), os.path.join(eval_dir, f"eval_{name}.json")
    if not os.path.exists(pz):
        refuse(f"eval_{name}.npz is missing from {eval_dir}")
    if need_json and not os.path.exists(pj):
        refuse(f"eval_{name}.json is missing from {eval_dir}")
    z = np.load(pz)
    for k in ("eid", "ade6", "pred"):
        if k not in z.files:
            refuse(f"eval_{name}.npz carries no {k}")
    ade = np.asarray(z["ade6"], np.float64)
    if not np.isfinite(ade).all():
        refuse(f"eval_{name}.npz holds a non-finite ade6")
    E = {"name": name, "eid": np.asarray(z["eid"]), "ade6": ade, "pred": np.asarray(z["pred"], np.float64), "path": pz,
         "ade2": np.asarray(z["ade2"], np.float64) if "ade2" in z.files else None, "fde6": np.asarray(z["fde6"], np.float64) if "fde6" in z.files else None,
         "json": _jl(pj) if os.path.exists(pj) else None, "cells": None}
    if need_cells:
        pc = os.path.join(eval_dir, f"speedcal_cells_{name}.npz")
        if not os.path.exists(pc):
            refuse(f"speedcal_cells_{name}.npz is missing from {eval_dir}")
        zc = np.load(pc)
        E["cells"] = {k: np.asarray(zc[k]) for k in zc.files}
    return E


def check_arm(E, seed, mode, sha, allow_smoke=False, target="x"):
    """One driver eval: the mode / seed / tag it claims, the checkpoint's sha256, no `limit`, and the relabelled carrier."""
    j = E["json"]
    b = j.get("speedcal")
    if not isinstance(b, dict) or b.get("schema") != "v8f-speedcal-eval-v1":
        refuse(f"eval_{E['name']}.json carries no speedcal block: it is not a v8f_mini_eval_speedcal.py output")
    base = f"M2b-mixo_s{seed}-speedcal" if target == "x" else f"M2b_s{seed}-speedcal"
    tag = f"{base}-{mode}"
    if (b.get("mode"), b.get("seed"), b.get("tag"), j.get("tag"), b.get("target")) != (mode, seed, tag, tag, target):
        refuse(f"eval_{E['name']}.json says mode/seed/tag/target {(b.get('mode'), b.get('seed'), b.get('tag'), j.get('tag'), b.get('target'))}, expected {(mode, seed, tag, tag, target)}")
    if b.get("limit") and not allow_smoke:
        refuse(f"eval_{E['name']}.json is a LIMITED run (first {b['limit']} windows): a smoke, never a result")
    if b.get("x_sha256") != sha:
        refuse(f"eval_{E['name']}.json scored checkpoint {str(b.get('x_sha256'))[:12]}, expected {sha[:12]}")
    if str(j.get("ckpt")) != str(b.get("x_ckpt")):
        refuse(f"eval_{E['name']}.json: ckpt {j.get('ckpt')!r} is not the checkpoint {b.get('x_ckpt')!r} the block names")
    if mode != "own" and (j.get("tac") != f"speedcal-{mode}" or (j.get("tac_manipulation") or {}).get("mode") != f"speedcal-{mode}"):
        refuse(f"eval_{E['name']}.json: the carrier labels were not relabelled to speedcal-{mode}")
    if mode == "own" and (j.get("tac") != "pred" or j.get("tac_manipulation")):
        refuse(f"eval_{E['name']}.json is not X_{seed}'s own code (tac {j.get('tac')!r})")
    if mode != "own" and (b.get("hook") or {}).get("mode") != mode:
        refuse(f"eval_{E['name']}.json: the hook record is missing or of another mode")
    if mode in ("identity", "cal", "perm", "ceiling") and not isinstance(b.get("map"), dict):
        refuse(f"eval_{E['name']}.json: a {mode} eval carries no map")
    return b


def load_fit(fit_dir, seed, sha, allow_smoke=False):
    """The seed's fit json and its cells npz (the arrays control 2 re-fits)."""
    tag = f"M2b-mixo_s{seed}-speedcal-fit"
    pj, pz = os.path.join(fit_dir, f"fit_{tag}.json"), os.path.join(fit_dir, f"fit_cells_{tag}.npz")
    for p in (pj, pz):
        if not os.path.exists(p):
            refuse(f"{os.path.basename(p)} is missing from {fit_dir}")
    fj = _jl(pj)
    if fj.get("schema") != "v8f-speedcal-fit-v1" or fj.get("seed") != seed:
        refuse(f"{pj} is not the v8f-speedcal-fit-v1 of seed {seed}")
    if fj.get("limit") and not allow_smoke:
        refuse(f"{pj} is a LIMITED (smoke) fit: never a result")
    if fj.get("x_sha256") != sha:
        refuse(f"{pj} was fitted on checkpoint {str(fj.get('x_sha256'))[:12]}, expected {sha[:12]}")
    z = np.load(pz)
    return {"json": fj, "cells": {k: np.asarray(z[k]) for k in z.files}, "path": pj, "cells_path": pz, "sha256": sha256(pj)}


def load_all(eval_dir, fit_dir, kindx_dirs, sha0, sha1, allow_smoke=False, first_n=None):
    """-> A[seed][mode] (the driver's evals + cells), Fit[seed], K[seed] = {'own', 'speed'} (KINDX's), eid. Every file on ONE window list; the committed 2,399 windows over 141 episodes."""
    shas = {0: sha0, 1: sha1}
    A, K, Fit = {}, {}, {}
    for s in SEEDS:
        A[s] = {}
        for m in MODES:
            E = load_eval(eval_dir, f"M2b-mixo_s{s}-speedcal-{m}", need_cells=(m != "own"))
            E["block"] = check_arm(E, s, m, shas[s], allow_smoke)
            A[s][m] = E
        Fit[s] = load_fit(fit_dir, s, shas[s], allow_smoke)
        cm = os.path.join(eval_dir, f"ceiling_map_M2b-mixo_s{s}-speedcal-ceiling.json")
        if not os.path.exists(cm):
            refuse(f"{os.path.basename(cm)} is missing from {eval_dir}")
        Fit[s]["ceiling_map"] = _jl(cm)
        K[s] = {}
        for nm, key in (("own", "own"), ("only-speed", "speed")):
            K[s][key] = load_eval(kindx_dirs[s], f"M2b-mixo_s{s}-kindx-{nm}", need_json=False)
    eid = A[0]["own"]["eid"]
    n = int(eid.size)
    for s in SEEDS:
        for E in list(A[s].values()):
            if not np.array_equal(E["eid"], eid):
                refuse(f"eval {E['name']} is not on one window list (eid differs): the window set or the episode list moved")
            if E["pred"].shape != A[s]["own"]["pred"].shape:
                refuse(f"eval {E['name']} has pred {E['pred'].shape}, the own eval has {A[s]['own']['pred'].shape}")
        for E in K[s].values():
            if not np.array_equal(E["eid"][:n] if (allow_smoke and first_n) else E["eid"], eid):
                refuse(f"eval {E['name']} is not on the driver evals' window list (eid differs): the window set or the episode list moved")
    ne = int(np.unique(eid).size)
    if (n, ne) != (N_WINDOWS, N_EPISODES) and not allow_smoke:
        refuse(f"the evals hold {n} windows over {ne} episodes, the committed eval-141 grid has {N_WINDOWS} over {N_EPISODES}")
    return A, Fit, K, eid


# ------------------------------------------------------------------------------------------------ the controls
def fit_control(Fs, eval_clips=None, tol=TOL_LSTSQ):
    """CONTROL 2 for one seed. The fitted map (and the permuted-target map) against numpy `lstsq` on the SAME masked training arrays, per cell; the permutation is a permutation of the true targets; the
    window counts; and the fit clips are disjoint from the eval clips (no eval window enters the fit)."""
    fj, z = Fs["json"], Fs["cells"]
    mu, sd = np.asarray(z["cons_mu"], np.float64), np.asarray(z["cons_sd"], np.float64)
    x = to_phys(z["own_std"], mu, sd)
    y = np.asarray(z["gt_phys"], np.float64)
    yp = np.asarray(z["perm_gt_phys"], np.float64)
    has = np.asarray(z["has"], bool)
    mask = has[:, None] & np.isfinite(y) & np.isfinite(x)
    out = {"bar": f"|a, b - numpy lstsq| <= {tol} on every cell", "cells": {}}
    worst = 0.0
    ok = True
    for c, nm in enumerate(SPEED_CELLS):
        a_ref, b_ref = lstsq_map(x[:, c], y[:, c], mask[:, c])
        ap_ref, bp_ref = lstsq_map(x[:, c], yp[:, c], mask[:, c])
        da, db = abs(fj["map"]["a"][c] - a_ref), abs(fj["map"]["b"][c] - b_ref)
        dpa, dpb = abs(fj["perm"]["a"][c] - ap_ref), abs(fj["perm"]["b"][c] - bp_ref)
        n_ok = bool(fj["map"]["n"][c] == int(mask[:, c].sum()) == fj["perm"]["n"][c])
        perm_ok = bool(np.array_equal(np.sort(yp[mask[:, c], c]), np.sort(y[mask[:, c], c])) and not np.array_equal(yp[mask[:, c], c], y[mask[:, c], c])
                       and np.array_equal(yp[~mask[:, c], c], y[~mask[:, c], c], equal_nan=True))
        out["cells"][nm] = {"n": int(mask[:, c].sum()), "a_json": fj["map"]["a"][c], "a_lstsq": a_ref, "b_json": fj["map"]["b"][c], "b_lstsq": b_ref, "max_abs_dev": max(da, db),
                            "perm_a_json": fj["perm"]["a"][c], "perm_a_lstsq": ap_ref, "perm_max_abs_dev": max(dpa, dpb), "n_equals_mask": n_ok, "perm_is_a_nonidentity_permutation_of_the_targets": perm_ok}
        worst = max(worst, da, db, dpa, dpb)
        ok = ok and da <= tol and db <= tol and dpa <= tol and dpb <= tol and n_ok and perm_ok
    out["max_abs_dev"] = worst
    out["fit_windows_match_json"] = bool(int(has.size) == fj["fit_set"]["n_windows"])
    clips = fj["fit_set"].get("clips")
    if eval_clips is not None:
        out["fit_clips_disjoint_from_eval_clips"] = bool(clips is not None and len(clips) > 0 and not (set(clips) & set(eval_clips)))
        out["n_fit_clips"], out["n_eval_clips"] = (len(clips) if clips else 0), len(eval_clips)
    out["PASS"] = bool(ok and out["fit_windows_match_json"] and out.get("fit_clips_disjoint_from_eval_clips", True))
    return out


def map_of(E):
    b = E["block"]["map"]
    return np.asarray(b["a"], np.float64), np.asarray(b["b"], np.float64)


def controls(A, Fit, K, eid, pf, banked=None, tol=TOL_M, first_n=None, n_boot=B_BOOT, smoke=False, eval_clips=None):
    """Controls 1-4 and the added guards, per seed (`first_n`: a SMOKE compares the first n windows of the full-size KINDX npz). Returns {'PASS': bool, ...}. Control 4 is the only one that reads a paired interval
    (the permuted arm against X_s); it decides nothing about the outcome."""
    banked = banked or {}
    out = {"tolerance_m": tol, "lstsq_tolerance": TOL_LSTSQ, "seeds": {}}
    sl = (slice(0, first_n) if first_n else slice(None))
    for s in SEEDS:
        a, k, fs = A[s], K[s], Fit[s]
        r = {}
        # CONTROL 1: the identity map reproduces X_s's own eval EXACTLY, per window (and the plan, bit for bit), and it IS the identity (a = 1, b = 0 exactly)
        ia, ib = map_of(a["identity"])
        d1 = dev(a["identity"]["ade6"], a["own"]["ade6"])
        p1 = paired(pf, a["identity"]["ade6"], a["own"]["ade6"], eid, n_boot)
        c1 = {"ade6_max_abs_dev_m": d1, "pred_equal": bool(np.array_equal(a["identity"]["pred"], a["own"]["pred"])), "paired_delta": [p1["delta"], p1["lo"], p1["hi"]],
              "map_is_identity": bool(np.all(ia == 1.0) and np.all(ib == 0.0)), "cells_unchanged": bool(np.array_equal(a["identity"]["cells"]["used_std"], a["identity"]["cells"]["own_std"])),
              "bar": "exactly 0.0 per window (PREREG: delta = 0.0), with a = 1 and b = 0"}
        c1["PASS"] = bool(d1 == 0.0 and c1["pred_equal"] and p1["delta"] == 0.0 and p1["lo"] == 0.0 and p1["hi"] == 0.0 and c1["map_is_identity"] and c1["cells_unchanged"])
        r["control_1_identity_map_is_identity"] = c1
        # CONTROL 2: the fitted map is numpy lstsq on the same masked training arrays; the maps the evals APPLIED are the fit json's, bit for bit; the CEILING map is lstsq on the identity eval's cells
        c2 = fit_control(fs, eval_clips)
        ca, cb = map_of(a["cal"])
        pa, pb = map_of(a["perm"])
        c2["cal_eval_applied_the_fit_json_map"] = bool(np.array_equal(ca, np.asarray(fs["json"]["map"]["a"], np.float64)) and np.array_equal(cb, np.asarray(fs["json"]["map"]["b"], np.float64)))
        c2["perm_eval_applied_the_fit_json_perm_map"] = bool(np.array_equal(pa, np.asarray(fs["json"]["perm"]["a"], np.float64)) and np.array_equal(pb, np.asarray(fs["json"]["perm"]["b"], np.float64)))
        zi = a["identity"]["cells"]
        xe = to_phys(zi["own_std"], zi["cons_mu"], zi["cons_sd"])
        ye = np.asarray(zi["gt_phys"], np.float64)
        me = np.asarray(zi["has"], bool)[:, None] & np.isfinite(ye) & np.isfinite(xe)
        cma, cmb = map_of(a["ceiling"])
        worst = 0.0
        for c in range(5):
            ar, br = lstsq_map(xe[:, c], ye[:, c], me[:, c])
            worst = max(worst, abs(cma[c] - ar), abs(cmb[c] - br))
        c2["ceiling_map_vs_lstsq_on_eval_cells_max_abs_dev"] = worst
        c2["ceiling_map_json_equals_applied"] = bool(np.array_equal(cma, np.asarray(fs["ceiling_map"]["a"], np.float64)) and np.array_equal(cmb, np.asarray(fs["ceiling_map"]["b"], np.float64)))
        c2["PASS"] = bool(c2["PASS"] and c2["cal_eval_applied_the_fit_json_map"] and c2["perm_eval_applied_the_fit_json_perm_map"] and worst <= TOL_LSTSQ and c2["ceiling_map_json_equals_applied"])
        r["control_2_fitted_map_is_lstsq"] = c2
        # CONTROL 3: the GT in the five speed cells reproduces KINDX's speed-only oracle per window; and the driver's own code-level comparison with the eval's oracle code is exactly 0.0
        gtdev = a["gt"]["block"]["hook"].get("oracle_code_max_abs_dev")
        d3 = dev(a["gt"]["ade6"], k["speed"]["ade6"][sl])
        r["control_3_gt_equals_kindx_speed_oracle"] = {"ade6_max_abs_dev_m": d3, "pred_max_abs_dev_m": dev(a["gt"]["pred"], k["speed"]["pred"][sl]), "driver_code_max_abs_dev": gtdev,
                                                       "bar": f"ade6 <= {tol} m per window AND the driver's code == the eval's oracle code exactly (0.0)"}
        r["control_3_gt_equals_kindx_speed_oracle"]["PASS"] = bool(d3 <= tol and gtdev == 0.0)
        # CONTROL 4: the deliberate regression. The map fitted on PERMUTED training targets must NOT be separated-better than X_s
        pp = paired(pf, a["perm"]["ade6"], a["own"]["ade6"], eid, n_boot)
        better = bool(pp["point"] < 0 and pp["separated"])
        r["control_4_permuted_map_is_not_separated_better"] = {"paired_delta": [pp["point"], pp["lo"], pp["hi"]], "separated": bool(pp["separated"]), "separated_better": better,
                                                              "ade6_perm": float(a["perm"]["ade6"].mean()), "ade6_own": float(a["own"]["ade6"].mean()),
                                                              "perm_slopes_a": [float(v) for v in fs["json"]["perm"]["a"]], "bar": "NOT (point < 0 and separated)"}
        r["control_4_permuted_map_is_not_separated_better"]["PASS"] = bool(not better)
        # added guards
        g = {}
        refs = {"kindx_own": k["own"]["ade6"][sl]}
        if s in banked and banked[s] is not None:
            refs["banked_standard_eval"] = banked[s]["ade6"][sl]
        g["own_equals_kindx_own_ade6_max_abs_dev_m"] = {nm: dev(a["own"]["ade6"], v) for nm, v in refs.items()}
        g["own_reproduces_the_references"] = bool(all(v <= tol for v in g["own_equals_kindx_own_ade6_max_abs_dev_m"].values()))
        hooked = ("identity", "gt", "cal", "perm", "ceiling")
        g["hook_leaves_nonspeed_cells_alone"] = bool(all(a[m]["block"]["hook"].get("nonspeed_max_abs_dev") == 0.0 for m in hooked))
        g["tactical_and_strategic_equal_to_X_s"] = bool(all(same_json(a["own"]["json"]["raw"].get(f), a[m]["json"]["raw"].get(f)) for m in hooked for f in ("tactical", "strategic")))
        own_cells = a["identity"]["cells"]["own_std"]
        g["every_mode_saw_the_same_own_cells"] = bool(all(dev(a[m]["cells"]["own_std"], own_cells) <= TOL_CELLS for m in hooked))
        applied = {}
        for m in ("identity", "cal", "perm", "ceiling"):
            zc = a[m]["cells"]
            am, bm = map_of(a[m])
            applied[m] = bool(np.array_equal(zc["used_std"], calibrate_std_np(zc["own_std"], am, bm, zc["cons_mu"], zc["cons_sd"])))
        g["used_cells_are_the_map_applied_to_own_cells"] = applied
        zg = a["gt"]["cells"]
        nh = ~zg["has"].astype(bool)
        g["gt_falls_back_to_own_where_no_v9_row"] = bool(np.array_equal(zg["used_std"][nh], zg["own_std"][nh]) and np.array_equal(a["gt"]["ade6"][nh], a["own"]["ade6"][nh]))
        zc = a["cal"]["cells"]
        g["cal_is_live_it_moves_the_cells_and_the_plan"] = bool(np.abs(zc["used_std"] - zc["own_std"]).max() > 0.0 and np.abs(a["cal"]["pred"] - a["own"]["pred"]).max() > 0.0)
        g["window_counts"] = {"n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size)}
        g["n_windows_without_v9_row"] = int(nh.sum())
        r["guards"] = g
        r["guards_PASS"] = bool(all(v for k_, v in g.items() if isinstance(v, bool)) and all(applied.values()))
        r["PASS"] = bool(c1["PASS"] and c2["PASS"] and r["control_3_gt_equals_kindx_speed_oracle"]["PASS"] and r["control_4_permuted_map_is_not_separated_better"]["PASS"] and r["guards_PASS"])
        out["seeds"][f"s{s}"] = r
    out["PASS"] = bool(all(out["seeds"][f"s{s}"]["PASS"] for s in SEEDS))
    return out


# ------------------------------------------------------------------------------------------------ the decision
def ceiling_le(g, F=F_BAR) -> bool:
    """CEILING "<= -F" (interpretation 3): the UNROUNDED point (ceiling - own) <= -F. No separation is required; the flag is reported beside."""
    return bool(g["point"] <= -F)


def decide(prim, ceil, sec_ok, F=F_BAR):
    """THE COMMITTED TABLE, rows tested in the PREREG's order. prim[s] / ceil[s] = the PRIMARY / CEILING paired block, sec_ok[s] = SECONDARY not separated-worse. See the module docstring (interpretations 1-5)."""
    met = {s: primary_met(prim[s], F) for s in SEEDS}
    negsep = {s: primary_negative_separated(prim[s]) for s in SEEDS}
    sep = {s: bool(prim[s]["separated"]) for s in SEEDS}
    cle = {s: ceiling_le(ceil[s], F) for s in SEEDS}
    if all(met.values()) and all(sec_ok.values()):
        out = "HEAD BIAS PAYS"
    elif all(negsep.values()) and not all(met.values()):
        out = "PARTIAL BIAS"
    elif not any(sep.values()) and all(cle.values()):
        out = "MAP DOES NOT TRANSFER"
    elif not any(sep.values()) and not all(cle.values()):
        out = "INFORMATION-LIMITED"
    else:
        out = "MIXED"
    return {"outcome": out, "primary_met": {f"s{s}": met[s] for s in SEEDS}, "primary_negative_and_separated": {f"s{s}": negsep[s] for s in SEEDS},
            "primary_separated": {f"s{s}": sep[s] for s in SEEDS}, "ceiling_le_minus_F": {f"s{s}": cle[s] for s in SEEDS},
            "ceiling_separated": {f"s{s}": bool(ceil[s]["separated"]) for s in SEEDS}, "secondary_not_separated_worse": {f"s{s}": bool(sec_ok[s]) for s in SEEDS},
            "per_seed_row": {f"s{s}": ("PRIMARY met" if met[s] else ("PRIMARY < 0, separated, above -F" if negsep[s] else ("PRIMARY not separated" if not sep[s] else "PRIMARY separated WORSE")))
                             + ("" if sec_ok[s] else "; SECONDARY separated-worse") for s in SEEDS},
            "F_m": F}


# ------------------------------------------------------------------------------------------------ the real family / distance-keeping tools (injected in tests)
def real_tools(args, eid):
    """-> (famfn, dkfn): SPEEDAVG's reader builds them from the programme's own instruments (imported, never copied)."""
    return RDA.real_tools(args, eid)


def family_reads(A_s, eid, famfn, dkfn, seed):
    """Per seed: the paired family blocks and the distance-keeping read of cal vs X_s's own."""
    na, nb = f"M2b-mixo_s{seed}-speedcal-cal", f"M2b-mixo_s{seed}-speedcal-own"
    pa, pb = A_s["cal"]["pred"], A_s["own"]["pred"]
    fam = famfn(pa, pb, eid)
    if not fam.get("longitudinal") or not fam.get("lateral"):
        refuse(f"the paired family block of seed {seed} has no longitudinal / lateral component: a missing family is refused")
    d = dkfn({na: pa, nb: pb}, {na: A_s["cal"]["ade6"], nb: A_s["own"]["ade6"]}, (na, nb))
    pair = d["read"]["pairs"].get(f"{na}__minus__{nb}")
    if not pair:
        refuse(f"the distance-keeping read has no pair block for {na} - {nb}")
    return fam, d["read"], pair


# ------------------------------------------------------------------------------------------------ the read
def build_result(A, Fit, K, eid, ctl, pf, fams, n_boot=B_BOOT, F=F_BAR):
    """The DECIDING block (PRIMARY, CEILING, CAPTURE, SECONDARY, outcome). `fams[s]` = (fam, dk_read, dk_pair)."""
    prim, ceil, sec, cap = {}, {}, {}, {}
    for s in SEEDS:
        a = A[s]
        prim[s] = paired(pf, a["cal"]["ade6"], a["own"]["ade6"], eid, n_boot)
        ceil[s] = paired(pf, a["ceiling"]["ade6"], a["own"]["ade6"], eid, n_boot)
        sec[s] = secondary_of(fams[s][0], fams[s][2])
        cap[s] = capture(a["own"]["ade6"], a["cal"]["ade6"], K[s]["own"]["ade6"], K[s]["speed"]["ade6"], eid, n_boot)
        cap[s]["driver_gain_speed_m"] = float(a["own"]["ade6"].mean() - a["gt"]["ade6"].mean())
    dec = decide(prim, ceil, {s: sec[s]["not_separated_worse"] for s in SEEDS}, F)
    out_name = dec["outcome"]
    keys = ("point", "delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_boot", "estimator")
    res = {"schema": SCHEMA, "prereg": "PREREG_V8F_SPEEDCAL.md", "tier": TIER,
           "estimator": f"paired episode-cluster bootstrap (taniteval/ci.py), B={n_boot}, {int(np.unique(eid).size)} episodes / {int(eid.size)} windows; answers 'another draw of EPISODES' only; "
                        "BLIND to training variance (H-ESTIM-SEED-1): the second seed is the replicate; a separated CI at two seeds is necessary, not sufficient, for a lever claim",
           "arm": "SPEEDCAL_s = X_s with T's five speed cells replaced by a_c * cell + b_c, the map fitted on TRAINING windows (HBLEND's 141 train clips) (X_s = M2b-mixo_s<s>)", "F_m": F,
           "speed_cells": list(SPEED_CELLS), "controls": {"PASS": ctl["PASS"], "detail": ctl},
           "ade6_m": {f"s{s}": {"X_own": float(A[s]["own"]["ade6"].mean()), "SPEEDCAL": float(A[s]["cal"]["ade6"].mean()), "identity": float(A[s]["identity"]["ade6"].mean()),
                                "perm_control": float(A[s]["perm"]["ade6"].mean()), "ceiling_T0": float(A[s]["ceiling"]["ade6"].mean()), "gt_speed_cells_T0": float(A[s]["gt"]["ade6"].mean())} for s in SEEDS},
           "PRIMARY": {f"s{s}": {k: prim[s][k] for k in keys if k in prim[s]} for s in SEEDS},
           "CEILING": {"tier": TIER_T0, "what": "the same affine form fitted on the EVAL GT itself (the best ANY affine map can do), minus X_s; routes the two 'no' rows; never a result",
                       **{f"s{s}": {k: ceil[s][k] for k in keys if k in ceil[s]} for s in SEEDS}},
           "CAPTURE": {f"s{s}": cap[s] for s in SEEDS},
           "SECONDARY": {f"s{s}": {"not_separated_worse": sec[s]["not_separated_worse"], "separated_worse": sec[s]["separated_worse"],
                                   "rows": {k: {x: v.get(x) for x in ("delta", "lo", "hi", "separated", "n_used", "n_windows") if x in v} for k, v in sec[s]["rows"].items()}} for s in SEEDS},
           "decision": dec, "OUTCOME": out_name,
           "outcome_table_verbatim": {o: {"condition": TABLE[o][0], "next": TABLE[o][1], "this_read": o == out_name} for o in OUTCOMES},
           "committed_next": TABLE[out_name][1],
           "provenance": {"fit_json_sha256": {f"s{s}": Fit[s]["sha256"] for s in SEEDS}, "fit_set_clips": {f"s{s}": Fit[s]["json"]["fit_set"]["n_clips"] for s in SEEDS},
                          "fit_set_windows": {f"s{s}": Fit[s]["json"]["fit_set"]["n_windows"] for s in SEEDS}}}
    res["reading"] = (f"PREREG_V8F_SPEEDCAL: {out_name}; PRIMARY (cal - own, ADE6 m): " + ", ".join(f"s{s} {prim[s]['point']:+.4f} [{prim[s]['lo']:+.4f}, {prim[s]['hi']:+.4f}]{'' if prim[s]['separated'] else ' ns'}" for s in SEEDS)
                      + f" vs -F = -{F}; CEILING (T0) " + ", ".join(f"s{s} {ceil[s]['point']:+.4f}" for s in SEEDS)
                      + "; CAPTURE " + ", ".join(f"s{s} " + (f"{cap[s]['point']:.2f}" if 'point' in cap[s] else "n/d") for s in SEEDS)
                      + "; SECONDARY not separated-worse: " + ", ".join(f"s{s} {sec[s]['not_separated_worse']}" for s in SEEDS) + ".")
    return res


def per_cell_block(A, Fit):
    """Per cell: a, b and the RMSE (m/s) to the GT before / after, on TRAIN (the fit set) and on EVAL, for the fitted map, the permuted-target map and the CEILING (T0) map."""
    out = {}
    for s in SEEDS:
        fj = Fit[s]["json"]
        zi = A[s]["identity"]["cells"]
        mu, sd = zi["cons_mu"], zi["cons_sd"]
        before = cell_rmse(to_phys(zi["own_std"], mu, sd), zi["gt_phys"], zi["has"])
        rows = {}
        for m, key in (("cal", "SPEEDCAL"), ("perm", "perm_control"), ("ceiling", "ceiling_T0")):
            zc = A[s][m]["cells"]
            after = cell_rmse(to_phys(zc["used_std"], mu, sd), zc["gt_phys"], zc["has"])
            am, bm = map_of(A[s][m])
            rows[key] = {nm: {"a": float(am[c]), "b": float(bm[c]), "eval_n": before[nm]["n"], "eval_rmse_before": before[nm]["rmse"], "eval_rmse_after": after[nm]["rmse"]} for c, nm in enumerate(SPEED_CELLS)}
        for c, nm in enumerate(SPEED_CELLS):
            rows["SPEEDCAL"][nm].update({"train_n": fj["map"]["n"][c], "train_rmse_before": fj["map"]["rmse_train_before"][c], "train_rmse_after": fj["map"]["rmse_train_after"][c],
                                         "train_mean_pred": fj["per_cell"][nm]["mean_pred"], "train_mean_gt": fj["per_cell"][nm]["mean_gt"], "train_sd_pred": fj["per_cell"][nm]["sd_pred"],
                                         "train_sd_gt": fj["per_cell"][nm]["sd_gt"]})
        out[f"s{s}"] = rows
    return out


def baseline_block(b_dir, A, Fit, shab, pf, eid, n_boot=B_BOOT):
    """DESCRIPTIVE: X_s's fitted map applied to the baseline B = M2b_s<s> (the non-MIXO arm): B's cal - own ADE6 paired, per seed. Absent -> a NOT RUN note."""
    out = {"decides": "nothing", "what": "X_s's fitted map applied to the baseline B = M2b_s<s> (non-MIXO), in physical units"}
    for s in SEEDS:
        try:
            Eo = load_eval(b_dir, f"M2b_s{s}-speedcal-own", need_cells=False)
            Ec = load_eval(b_dir, f"M2b_s{s}-speedcal-cal", need_cells=True)
        except SystemExit:
            out[f"s{s}"] = {"status": "NOT RUN: the baseline evals are absent"}
            continue
        Eo["block"] = check_arm(Eo, s, "own", shab[s], False, "b")
        Ec["block"] = check_arm(Ec, s, "cal", shab[s], False, "b")
        if not np.array_equal(Eo["eid"], eid) or not np.array_equal(Ec["eid"], eid):
            refuse(f"the baseline evals of seed {s} are not on the evals' window list")
        am, bm = map_of(Ec)
        out[f"s{s}"] = {"b_sha256": shab[s], "map_equals_X_s_fit": bool(np.array_equal(am, np.asarray(Fit[s]["json"]["map"]["a"], np.float64)) and np.array_equal(bm, np.asarray(Fit[s]["json"]["map"]["b"], np.float64))),
                        "ade6_B_own": float(Eo["ade6"].mean()), "ade6_B_cal": float(Ec["ade6"].mean()), "paired_cal_minus_own": {k: v for k, v in paired(pf, Ec["ade6"], Eo["ade6"], eid, n_boot).items() if k in ("point", "delta", "lo", "hi", "separated")}}
    return out


PAIRFAM_N_VALUES = 119          # the reproduction control's value count on a banked eval json that carries `raw.families` (measured: SPEEDAVG's substitute control)
PAIRFAM_SUB_MAX_DELTA = 1e-6


def _max_abs_delta(o):
    """The largest |delta| over every paired cell of a `v8f_paired_families.py` block (a cell is any dict with a numeric `delta`)."""
    m = 0.0
    if isinstance(o, dict):
        d = o.get("delta")
        if isinstance(d, (int, float)) and np.isfinite(d):
            m = abs(float(d))
        for v in o.values():
            m = max(m, _max_abs_delta(v))
    elif isinstance(o, list):
        for v in o:
            m = max(m, _max_abs_delta(v))
    return m


def _repro(j):
    """-> [(arm, n_checked, n_failed, ok)] of a pairfam json's JSON-reproduction controls."""
    return [(c.get("arm"), c.get("n_checked"), c.get("n_failed"), c.get("ok")) for c in (j.get("controls") or []) if "n_checked" in c]


def substitute_control(pf_dir, seed):
    """The paired-families tool's SUBSTITUTE control (coordinator, 2026-10-25): our `own`-mode npz against the banked X_s npz, both read against the banked eval json. PASS = controls_ok true, 119 / 119 values
    reproduced on both arms, and the paired max |delta| <= 1e-6. It tells whether the tool reads THIS driver's npz format and reproduces a banked eval; it decides nothing about the outcome."""
    p = os.path.join(pf_dir, f"subctl_own_vs_banked_s{seed}.json") if pf_dir else None
    if not p or not os.path.exists(p):
        return {"status": "NOT RUN: the substitute control's output is absent", "PASS": False, "file": p}
    j = _jl(p)
    rep = _repro(j)
    md = max(_max_abs_delta(j.get("tactical")), _max_abs_delta(j.get("strategic")))
    ok = bool(j.get("controls_ok") is True and len(rep) == 2 and all(n == PAIRFAM_N_VALUES and f == 0 and k is True for _a, n, f, k in rep) and md <= PAIRFAM_SUB_MAX_DELTA)
    return {"status": "own-mode npz vs the banked X_s npz, both against the banked eval json", "file": p, "controls_ok": j.get("controls_ok"), "reproduction": [
            {"arm": a, "n_checked": n, "n_failed": f, "ok": k} for a, n, f, k in rep], "paired_max_abs_delta": md,
            "bar": f"controls_ok, {PAIRFAM_N_VALUES}/{PAIRFAM_N_VALUES} values reproduced on both arms, paired max |delta| <= {PAIRFAM_SUB_MAX_DELTA}", "PASS": ok}


def pairfam_block(pf_dir, seed):
    """TACTICAL / STRATEGIC by `v8f_paired_families.py` (cal - own): its summary, its own controls verdict (NATIVE: the cal / own eval jsons carry `raw.families`, so its JSON-reproduction control can run) and the
    SUBSTITUTE control. An instrument gap (a native reproduction control that did not pass) is reported as a GAP, never as a pass. Absent -> a NOT RUN note (never silently dropped)."""
    p = os.path.join(pf_dir, f"pairfam_speedcal_s{seed}.json") if pf_dir else None
    sub = substitute_control(pf_dir, seed)
    if not p or not os.path.exists(p):
        return {"status": "NOT RUN: the paired-families tool's output is absent", "file": p, "substitute_control": sub}
    j = _jl(p)
    rep = _repro(j)
    gap = [{"arm": a, "n_checked": n, "n_failed": f} for a, n, f, k in rep if k is not True]
    return {"status": "v8f_paired_families.py, cal - own, paired episode-cluster bootstrap", "file": p, "controls_ok": j.get("controls_ok"), "native_reproduction": [{"arm": a, "n_checked": n, "n_failed": f, "ok": k} for a, n, f, k in rep],
            "instrument_gap": (gap or None), "substitute_control": sub, "summary": j.get("summary"), "n_windows": j.get("n_windows"), "n_episodes": j.get("n_episodes"),
            "variance_answered": j.get("variance_answered"), "tactical": j.get("tactical"), "strategic": j.get("strategic")}


def beside_block(A, Fit, K, eid, fams, pf, n_boot=B_BOOT, pf_dir=None, b_dir=None, shab=None):
    """Everything that decides nothing: the four families, the per-cell maps and RMSE, ADE2 / FDE6, the baseline B, S and T checked equal to X_s."""
    out = {"families": {}, "per_cell": per_cell_block(A, Fit), "other_horizons": {}, "decides": "nothing"}
    for s in SEEDS:
        fam, dkr, pair = fams[s]
        j_own, j_cal = A[s]["own"]["json"]["raw"], A[s]["cal"]["json"]["raw"]
        pfb = pairfam_block(pf_dir, s)
        out["families"][f"s{s}"] = {
            "LONGITUDINAL": {"paired_vs_own": fam.get("longitudinal"), "distance_keeping_k10": {"arm": dkr["arms"].get(f"M2b-mixo_s{s}-speedcal-cal"), "pair_vs_own": pair}},
            "LATERAL": {"paired_vs_own": fam.get("lateral")},
            "TACTICAL": {"heads_equal_to_X_s": same_json(j_own.get("tactical"), j_cal.get("tactical")), "head_block": j_cal.get("tactical"),
                         "status": "T's decision heads are X_s's own in every arm (checked bit-equal); the plan-derived manoeuvre and the goal point are read off the arm's PLAN, so they can move: see `paired`",
                         "paired": pfb},
            "STRATEGIC": {"status": "in force for v8F (PI-V8F-2026-10-17, R72 / R73). S is X_s's own: identical to X_s by construction (checked bit-equal)", "equal_to_X_s": same_json(j_own.get("strategic"), j_cal.get("strategic")),
                          "block": j_cal.get("strategic"), "paired": pfb.get("strategic") if isinstance(pfb, dict) else None}}
        out["other_horizons"][f"s{s}"] = {k: paired(pf, A[s]["cal"][k], A[s]["own"][k], eid, n_boot) for k in ("ade2", "fde6") if A[s]["cal"][k] is not None and A[s]["own"][k] is not None}
        for nm, mode in (("identity_minus_own_ade6", "identity"), ("perm_minus_own_ade6 (control 4)", "perm"), ("gt_speed_minus_own_ade6 (T0, driver)", "gt"), ("ceiling_minus_own_ade6 (T0)", "ceiling")):
            out["other_horizons"][f"s{s}"][nm] = paired(pf, A[s][mode]["ade6"], A[s]["own"]["ade6"], eid, n_boot)
    if b_dir and shab:
        out["baseline_B"] = baseline_block(b_dir, A, Fit, shab, pf, eid, n_boot)
    else:
        out["baseline_B"] = {"status": "NOT RUN: no baseline directory given", "decides": "nothing"}
    return out


def md_of(res) -> str:
    L = ["# SPEEDCAL_READ: PREREG_V8F_SPEEDCAL executed as written", "", f"**Outcome (the PREREG table applied verbatim): {res['OUTCOME']}.**", "",
         f"Tier: {res['tier']}. CEILING is stamped T0. Estimator: {res['estimator']}.", "", f"Committed next step, quoted from the PREREG: *\"{res['committed_next']}\"*", "",
         "| read | seed 0 | seed 1 | bar |", "|---|---|---|---|"]
    P, C, Z = res["PRIMARY"], res["CAPTURE"], res["CEILING"]
    f = lambda g: f"{g['point']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}]{' *' if g['separated'] else ''}"        # noqa: E731
    L.append(f"| PRIMARY ADE6 SPEEDCAL - X_s | {f(P['s0'])} | {f(P['s1'])} | <= -{res['F_m']} and separated at BOTH |")
    L.append(f"| CEILING (T0 diagnostic) ADE6 eval-GT map - X_s | {f(Z['s0'])} | {f(Z['s1'])} | beside; routes the two 'no' rows |")
    L.append("| CAPTURE | " + " | ".join((f"{C[f's{s}']['point']:.3f}" if "point" in C[f's{s}'] else "n/d") for s in SEEDS) + " | beside |")
    L.append("| SECONDARY not separated-worse | " + " | ".join(str(res["SECONDARY"][f"s{s}"]["not_separated_worse"]) for s in SEEDS) + " | both |")
    L += ["", "## The outcome table (verbatim from the PREREG)", "", "| outcome | condition | next (committed) | this read |", "|---|---|---|---|"]
    for o, v in res["outcome_table_verbatim"].items():
        L.append(f"| **{o}** | {v['condition']} | {v['next']} | {'**YES**' if v['this_read'] else ''} |")
    L += ["", "## Controls", "", f"PASS: {res['controls']['PASS']}"]
    for s in SEEDS:
        r = res["controls"]["detail"]["seeds"][f"s{s}"]
        L.append(f"* seed {s}: control 1 {r['control_1_identity_map_is_identity']['PASS']} (max dev {r['control_1_identity_map_is_identity']['ade6_max_abs_dev_m']}), "
                 f"control 2 {r['control_2_fitted_map_is_lstsq']['PASS']} (max dev {r['control_2_fitted_map_is_lstsq']['max_abs_dev']:.3g}), "
                 f"control 3 {r['control_3_gt_equals_kindx_speed_oracle']['PASS']} (max dev {r['control_3_gt_equals_kindx_speed_oracle']['ade6_max_abs_dev_m']:.3g} m), "
                 f"control 4 {r['control_4_permuted_map_is_not_separated_better']['PASS']} (perm - own {r['control_4_permuted_map_is_not_separated_better']['paired_delta'][0]:+.3f} m), guards {r['guards_PASS']}")
    if "beside" in res:
        L += ["", "## Per cell: the map (a, b) and the RMSE to the GT, m/s (beside; decides nothing)", "", "| seed | cell | a | b | train n | train RMSE before -> after | eval RMSE before -> after |", "|---|---|---|---|---|---|---|"]
        for s in SEEDS:
            for nm, v in res["beside"]["per_cell"][f"s{s}"]["SPEEDCAL"].items():
                L.append(f"| {s} | `{nm}` | {v['a']:.4f} | {v['b']:+.4f} | {v['train_n']} | {v['train_rmse_before']:.3f} -> {v['train_rmse_after']:.3f} | {v['eval_rmse_before']:.3f} -> {v['eval_rmse_after']:.3f} |")
    if "beside" in res:
        L += ["", "## TACTICAL / STRATEGIC (`v8f_paired_families.py`, SPEEDCAL - X_s; beside, decides nothing)", ""]
        for s in SEEDS:
            pfb = res["beside"]["families"][f"s{s}"]["TACTICAL"].get("paired") or {}
            sub = pfb.get("substitute_control") or {}
            L.append(f"* seed {s}: tool controls_ok {pfb.get('controls_ok')}; native reproduction {pfb.get('native_reproduction')}; "
                     f"INSTRUMENT GAP {pfb.get('instrument_gap') or 'none'}; substitute control PASS {sub.get('PASS')} (max |delta| {sub.get('paired_max_abs_delta')}); "
                     f"separated cells {((pfb.get('summary') or {}).get('tactical') or {}).get('separated')} / {((pfb.get('summary') or {}).get('strategic') or {}).get('separated')}")
    L += ["", f"Reading: {res['reading']}", ""]
    return "\n".join(L)


def check_inputs(args, A, Fit, K, eid):
    """Everything the read refuses on, no number read: every file present and on one window list, the shas, the modes, the lead block / ground truth for the family tools."""
    famfn, dkfn = real_tools(args, eid)
    return {"schema": "v8f-speedcal-check-inputs-v1", "n_windows": int(eid.size), "n_episodes": int(np.unique(eid).size), "seeds": list(SEEDS), "modes": list(MODES),
            "x0_sha256": args.expect_sha0, "x1_sha256": args.expect_sha1, "family_tools": "built", "decides": "nothing"}


def _dump(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def eval_clip_names(eval_grid=None):
    p = Path(eval_grid) if eval_grid else RAW / "eval_grid_windows.json"
    return list(_jl(p)["order"]) if p.exists() else None


def main(argv=None):
    if RDA is None:
        refuse(f"{_RDA_PATH} (SPEEDAVG's reader, which this reader imports) is absent")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--fit-dir", required=True)
    ap.add_argument("--kindx-dir-s0", required=True)
    ap.add_argument("--kindx-dir-s1", required=True)
    ap.add_argument("--banked-own-s0", default=None, help="the banked STANDARD eval npz of X_s0 (a second reference for the own-reproduction guard)")
    ap.add_argument("--banked-own-s1", default=None)
    ap.add_argument("--pairfam-dir", default=None, help="the dir holding pairfam_speedcal_s<s>.json (v8f_paired_families.py, cal - own)")
    ap.add_argument("--b-eval-dir", default=None, help="the dir holding the baseline B evals (X_s's map on M2b_s<s>)")
    ap.add_argument("--expect-sha-b0", default=None)
    ap.add_argument("--expect-sha-b1", default=None)
    ap.add_argument("--eval-grid", default=None)
    ap.add_argument("--expect-sha0", default=X0_SHA)
    ap.add_argument("--expect-sha1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lead-block", default=None)
    ap.add_argument("--n-boot", type=int, default=B_BOOT)
    ap.add_argument("--no-beside", action="store_true")
    ap.add_argument("--check-inputs", action="store_true")
    ap.add_argument("--smoke-first-n", type=int, default=0, help="SMOKE ONLY: the evals are the first N windows (a driver --limit run); the outcome is NOT a result and is not written")
    args = ap.parse_args(argv)
    _paths()
    pf = pf_default()
    smoke = bool(args.smoke_first_n)
    if args.expect_sha0 == args.expect_sha1:
        refuse("--expect-sha0 == --expect-sha1: there is no second seed")
    A, Fit, K, eid = load_all(args.eval_dir, args.fit_dir, {0: args.kindx_dir_s0, 1: args.kindx_dir_s1}, args.expect_sha0, args.expect_sha1, allow_smoke=smoke, first_n=args.smoke_first_n or None)
    if smoke and int(eid.size) != args.smoke_first_n:
        refuse(f"--smoke-first-n {args.smoke_first_n} but the evals hold {eid.size} windows")
    banked = {}
    for s, p in ((0, args.banked_own_s0), (1, args.banked_own_s1)):
        if p:
            if not os.path.exists(p):
                refuse(f"--banked-own-s{s} {p} does not exist")
            banked[s] = load_eval(os.path.dirname(p), os.path.basename(p)[len("eval_"):-len(".npz")], need_json=False)
    if args.check_inputs:
        _dump(args.out, check_inputs(args, A, Fit, K, eid))
        print("SPEEDCAL INPUTS OK")
        return 0
    ctl = controls(A, Fit, K, eid, pf, banked, first_n=args.smoke_first_n or None, n_boot=min(args.n_boot, B_BOOT), smoke=smoke, eval_clips=eval_clip_names(args.eval_grid))
    _dump(os.path.splitext(args.out)[0] + "_controls.json", ctl)
    if not ctl["PASS"]:
        refuse("a control failed (see the *_controls.json): no outcome is read")
    if smoke:
        print("SMOKE: controls 1-4 and the guards PASS on the first", int(eid.size), "windows; no outcome is read from a smoke")
        return 0
    famfn, dkfn = real_tools(args, eid)
    fams = {s: family_reads(A[s], eid, famfn, dkfn, s) for s in SEEDS}
    res = build_result(A, Fit, K, eid, ctl, pf, fams, args.n_boot)
    _dump(args.out, res)                                          # the deciding block is on disk BEFORE any beside read is computed
    print(res["reading"])
    print("committed next:", res["committed_next"])
    if args.no_beside:
        return 0
    shab = {0: args.expect_sha_b0, 1: args.expect_sha_b1} if args.expect_sha_b0 and args.expect_sha_b1 else None
    res["beside"] = beside_block(A, Fit, K, eid, fams, pf, args.n_boot, args.pairfam_dir, args.b_eval_dir, shab)
    _dump(args.out, res)
    with open(os.path.splitext(args.out)[0] + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(md_of(res))
    print("beside written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
