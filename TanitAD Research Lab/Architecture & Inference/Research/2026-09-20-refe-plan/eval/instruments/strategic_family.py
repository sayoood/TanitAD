#!/usr/bin/env python3
"""STRATEGIC family (STRAT-FAM) for REFe navtest reads. NEW instrument; it runs BESIDE the pinned families_run.py / p5c_families.py and edits neither.
0 GPU, CPU only. PI ruling 2026-11-16: "strategic is not paused anymore"; every eval reports the STRATEGIC family in full.

WHAT A REFe ARM HAS (it owns no strategic layer): the INPUT `driving_command` (LEFT / STRAIGHT / RIGHT = the route) and its own 8-pose, 0.5 s, 4 s plan.
  S-DEC   strategic decision = command adherence, IMPORTED from eval/command_adherence.py (plan_classes / confusion / grouped / cell / paired; nothing copied):
          per-command confusion (commanded vs executed direction), per-class adherence (= recall) with n, all-token and turn-active (LEFT u RIGHT) cells.
  S-GOAL  goal / route setting, plan vs the LOGGED human future (navtest `human_future_poses`, ego frame x forward / y left), at the 4 s endpoint (pose index 7):
          endpoint_error_m      |plan[7,:2] - gt[7,:2]|                      (mean per token)
          endpoint_heading_err  |wrap(plan[7,2] - gt[7,2])| rad              (mean per token; plan heading is the A7-repaired last heading)
          route_progress_ratio  plan arc length / logged arc length          (RATIO OF SUMS, origin -> 8 poses; defined whenever the class has logged motion)
          lateral_offset_m      (plan y - gt y) at the endpoint, signed so that + = further toward the COMMANDED side (LEFT +y, RIGHT -y; STRAIGHT has no side:
                                it is left-positive raw y); also |lateral offset| (unsigned).
          each per command class (LEFT, STRAIGHT, RIGHT; UNKNOWN has n = 0 on navtest) and for TURN_ACTIVE (LEFT u RIGHT, side-signed).
ESTIMATOR: paired log-cluster bootstrap, B = 10,000, 95 %, seed 20260927 = the resample stream of proxy_eval.boot (command_adherence.resample_weights, reused): one
`rng.integers(0, G, G)` per resample over the logs of the subset, models held fixed, arm and base on the SAME tokens and the SAME resamples. A per-token mean is the
resampled sum / resampled count; the ratio is the resampled sum / resampled sum. It answers "another draw of LOGS"; training variance (one seed per arm, H-ESTIM-SEED-1)
is unmeasured and inference is deterministic. "separated" = the 95 % interval of arm - base excludes 0: NECESSARY, not sufficient, for a lever claim.
NO pooled composite score exists here: every metric is its own row. NO decision bar: this family is REPORTED, never gating (PREREG_P5C froze its bars before the ruling).
TIER: T1 (NAVSIM v1.1 self-action open loop: the plan is the model's own; the command is an INPUT). Evidence class of every number: MEASURED (ours) on the named seams.

    python eval/instruments/strategic_family.py read --arm P7=<seam.npz> --base P6=<seam.npz> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = Path(os.environ.get("REFE_EVAL_DIR") or HERE.parent)           # holds command_adherence.py and proxy_eval.py (pinned, never edited)
sys.path.insert(0, str(EVAL))
import command_adherence as CA                                         # noqa: E402  (the adherence definition is IMPORTED, never copied)

N_BOOT, SEED = CA.N_BOOT, CA.SEED
CLASSES = ("LEFT", "STRAIGHT", "RIGHT")
SUBSETS = ("LEFT", "STRAIGHT", "RIGHT", "TURN_ACTIVE")
METRICS = ("endpoint_error_m", "endpoint_heading_err_rad", "route_progress_ratio", "lateral_offset_m", "abs_lateral_offset_m")
END = 7                                                                # pose index of the 4 s endpoint (8 poses at 0.5 s)
TIER = CA.TIER
AUTHORITY = ("NONE. STRATEGIC is a REPORTED family (PI 2026-11-16: not paused). It is not a decision bar and carries no authority over any PREREG; PREREG_P5C froze its bars "
             "before the ruling. Quote it only with the estimator and n.")
ESTIMATOR = ("paired log-cluster bootstrap over the navtest logs, 10,000 resamples, 95 %, seed 20260927: the resample stream of proxy_eval.boot "
             "(command_adherence.resample_weights), arm and base on identical tokens and identical resamples; per-token means = resampled sum / resampled count, "
             "route-progress = resampled sum of plan arc / resampled sum of logged arc. It answers 'another draw of LOGS'; training variance (one seed per arm, "
             "H-ESTIM-SEED-1) is unmeasured; inference is deterministic. A separated interval is necessary, not sufficient, for a lever claim.")


# ------------------------------------------------------------------ per-token geometry
def arc_length(poses) -> np.ndarray:
    """[N,8,>=2] -> [N] polyline length origin -> pose0 -> ... -> pose7 (ego frame, the ego sits at the origin)"""
    xy = np.asarray(poses, dtype=np.float64)[..., :2]
    z = np.zeros_like(xy[:, :1])
    pts = np.concatenate([z, xy], axis=1)
    return np.linalg.norm(np.diff(pts, axis=1), axis=-1).sum(-1)


def side_sign(cmd) -> np.ndarray:
    """+1 for LEFT and STRAIGHT/UNKNOWN (left-positive raw y), -1 for RIGHT: multiplies (plan y - gt y) so + = toward the commanded side"""
    return np.where(np.asarray(cmd) == CA.RIGHT, -1.0, 1.0)


def goal_token_metrics(plan, gt, cmd) -> dict:
    """per-token S-GOAL columns. plan, gt: [N,8,3] (x, y, heading), ego frame. Strict: shapes must match and xy be finite."""
    p, g = CA.check_plan_format(plan, "plan").astype(np.float64), CA.check_plan_format(gt, "logged future").astype(np.float64)
    if p.shape != g.shape or p.shape[2] < 3:
        raise ValueError(f"plan {p.shape} vs logged future {g.shape}: need identical [N,8,3] (heading column required)")
    d = p[:, END, :2] - g[:, END, :2]
    dh = p[:, END, 2] - g[:, END, 2]
    dy = d[:, 1] * side_sign(cmd)
    return {"endpoint_error_m": np.linalg.norm(d, axis=-1), "endpoint_heading_err_rad": np.abs(np.arctan2(np.sin(dh), np.cos(dh))),
            "lateral_offset_m": dy, "abs_lateral_offset_m": np.abs(dy), "plan_arc_m": arc_length(p), "gt_arc_m": arc_length(g)}


def subset_mask(cmd, name: str) -> np.ndarray:
    cmd = np.asarray(cmd)
    if name == "TURN_ACTIVE":
        return (cmd == CA.LEFT) | (cmd == CA.RIGHT)
    return cmd == CA.CMD_NAMES.index(name)


# ------------------------------------------------------------------ per-log sums -> resampled statistics
_COLS = ("endpoint_error_m", "endpoint_heading_err_rad", "lateral_offset_m", "abs_lateral_offset_m", "plan_arc_m", "gt_arc_m")


def per_log_sums(tok: dict, logs, mask):
    """-> (sorted logs of the subset, S [G, 7] = per-log sums of the six columns + the token count)"""
    logs = np.asarray(logs)
    sel = np.flatnonzero(mask)
    ul = sorted(set(logs[sel].tolist()))
    pos = {l: i for i, l in enumerate(ul)}
    g = np.array([pos[l] for l in logs[sel]], dtype=np.int64)
    S = np.zeros((len(ul), len(_COLS) + 1), dtype=np.float64)
    for j, c in enumerate(_COLS):
        np.add.at(S[:, j], g, tok[c][sel])
    np.add.at(S[:, -1], g, 1.0)
    return ul, S


def stats_from_sums(R) -> dict:
    """R [..., 7] sums (a point or a resample batch) -> the five metrics; NaN where the denominator is 0 (never a fake value)"""
    R = np.asarray(R, dtype=np.float64)
    n = R[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"endpoint_error_m": R[..., 0] / n, "endpoint_heading_err_rad": R[..., 1] / n, "lateral_offset_m": R[..., 2] / n,
                "abs_lateral_offset_m": R[..., 3] / n, "route_progress_ratio": np.where(R[..., 5] > 0, R[..., 4] / R[..., 5], np.nan)}


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def goal_block(S, n_boot=N_BOOT, seed=SEED) -> dict:
    """one arm, one subset: point + 95 % CI per metric from the registered resample stream"""
    point = stats_from_sums(S.sum(0))
    boot = stats_from_sums(np.einsum("bg,gk->bk", CA.resample_weights(S.shape[0], n_boot, seed), S))
    out = {"n": int(S[:, -1].sum()), "n_logs": int(S.shape[0])}
    for m in METRICS:
        ci, n_nan = CA._ci(boot[m])
        out[m] = {"point": _f(point[m]), "ci95": ci, "n_undefined_resamples": n_nan}
    return out


def goal_paired(Sa, Sb, n_boot=N_BOOT, seed=SEED) -> dict:
    """arm - base on IDENTICAL tokens and the SAME resamples"""
    assert Sa.shape == Sb.shape, (Sa.shape, Sb.shape)
    W = CA.resample_weights(Sa.shape[0], n_boot, seed)
    pa, pb = stats_from_sums(Sa.sum(0)), stats_from_sums(Sb.sum(0))
    ba, bb = stats_from_sums(np.einsum("bg,gk->bk", W, Sa)), stats_from_sums(np.einsum("bg,gk->bk", W, Sb))
    out = {"n": int(Sa[:, -1].sum()), "n_logs": int(Sa.shape[0])}
    for m in METRICS:
        ci, n_nan = CA._ci(ba[m] - bb[m])
        pt = _f(pa[m] - pb[m])
        out[m] = {"point": pt, "ci95": ci, "separated": None if ci[0] is None else bool(ci[0] > 0 or ci[1] < 0), "n_undefined_resamples": n_nan}
    return out


def mean_token_ratio(tok: dict, mask, min_gt_arc_m: float = 1.0) -> dict:
    """DESCRIPTIVE only: mean of per-token plan_arc / gt_arc over tokens whose logged arc >= 1 m (a validity mask for a ratio, NOT a bar); the headline is the ratio of sums"""
    ok = mask & (tok["gt_arc_m"] >= min_gt_arc_m)
    return {"mean_per_token_ratio": _f((tok["plan_arc_m"][ok] / tok["gt_arc_m"][ok]).mean()) if ok.any() else None, "n_defined": int(ok.sum()),
            "n_excluded_gt_arc_below_min": int((mask & ~ok).sum()), "min_gt_arc_m": min_gt_arc_m, "descriptive_only": True}


# ------------------------------------------------------------------ S-DEC
def sdec_arm(cmd, plan_cls, logs, n_boot=N_BOOT, seed=SEED, label="") -> dict:
    """per-command confusion + per-class adherence (recall) with n and CI + all-token / turn-active cells (command_adherence.cell)"""
    _, Ca, _, Ct, cells = CA._cells(cmd, plan_cls, logs, n_boot, seed, label)
    out = {"all_tokens": cells["all_tokens"], "turn_active": cells["turn_active"], "per_class": {}}
    for nm in CLASSES:
        m = subset_mask(cmd, nm)
        if not m.any():
            out["per_class"][nm] = {"n": 0, "status": "no tokens of this command class"}
            continue
        _, C = CA.grouped(cmd, plan_cls, logs, m)
        c = CA.cell(C, n_boot, seed, f"{label}: command {nm}")
        out["per_class"][nm] = {"n": c["n"], "n_logs": c["n_logs"], "adherence": c["accuracy"], "executed_direction_counts": c["confusion_command_rows_plan_cols"][CA.CMD_NAMES.index(nm)]}
    return out


def sdec_paired(cmd, cls_a, cls_b, logs, n_boot=N_BOOT, seed=SEED) -> dict:
    out = {}
    _, Caa, _, Cta, _ = CA._cells(cmd, cls_a, logs, 2, seed, "a")
    _, Cab, _, Ctb, _ = CA._cells(cmd, cls_b, logs, 2, seed, "b")
    out["all_tokens"], out["turn_active"] = CA.paired(Caa, Cab, n_boot, seed), CA.paired(Cta, Ctb, n_boot, seed)
    out["per_class"] = {}
    for nm in CLASSES:
        m = subset_mask(cmd, nm)
        if m.any():
            _, Ca = CA.grouped(cmd, cls_a, logs, m)
            _, Cb = CA.grouped(cmd, cls_b, logs, m)
            out["per_class"][nm] = CA.paired(Ca, Cb, n_boot, seed)["accuracy_delta"]
    return out


# ------------------------------------------------------------------ whole read
def family_arm(cmd, logs, plan, gt, n_boot=N_BOOT, seed=SEED, label="") -> dict:
    """one arm: S-DEC + S-GOAL, per command class"""
    cls, prov = CA.plan_classes(plan)
    tok = goal_token_metrics(plan, gt, cmd)
    goal = {}
    for nm in SUBSETS:
        m = subset_mask(cmd, nm)
        if not m.any():
            goal[nm] = {"n": 0, "status": "no tokens of this command class"}
            continue
        _, S = per_log_sums(tok, logs, m)
        goal[nm] = {**goal_block(S, n_boot, seed), "descriptive_mean_token_ratio": mean_token_ratio(tok, m)}
    return {"S_DEC": sdec_arm(cmd, cls, logs, n_boot, seed, label), "S_GOAL": goal, "labeller_provenance": prov, "_cls": cls, "_tok": tok}


def family_paired(cmd, logs, A, B, plan_a, plan_b, n_boot=N_BOOT, seed=SEED) -> dict:
    """A, B = family_arm results (same tokens). delta = A - B."""
    out = {"S_DEC": sdec_paired(cmd, A["_cls"], B["_cls"], logs, n_boot, seed), "S_GOAL": {}}
    for nm in SUBSETS:
        m = subset_mask(cmd, nm)
        if m.any():
            _, Sa = per_log_sums(A["_tok"], logs, m)
            _, Sb = per_log_sums(B["_tok"], logs, m)
            out["S_GOAL"][nm] = goal_paired(Sa, Sb, n_boot, seed)
    return out


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ------------------------------------------------------------------ controls (each reads a known value; each has a RED arm in the tests)
def mirror_lateral(poses) -> np.ndarray:
    """left-right mirror of a plan in the ego frame: y -> -y, heading -> -heading"""
    p = np.array(poses, dtype=np.float64, copy=True)
    p[..., 1] = -p[..., 1]
    if p.shape[-1] >= 3:
        p[..., 2] = -p[..., 2]
    return p


def control_logged_future(cmd, logs, gt, n_boot=200, seed=SEED) -> dict:
    """the logged human future AS an arm: every S-GOAL error is EXACTLY 0 and the route-progress ratio EXACTLY 1.0"""
    a = family_arm(cmd, logs, gt, gt, n_boot, seed, "logged-future control")
    bad = []
    for nm, g in a["S_GOAL"].items():
        if g.get("n", 0) == 0:
            continue
        for m, want in (("endpoint_error_m", 0.0), ("endpoint_heading_err_rad", 0.0), ("lateral_offset_m", 0.0), ("abs_lateral_offset_m", 0.0), ("route_progress_ratio", 1.0)):
            v = g[m]["point"]
            if g[m]["ci95"][0] is not None and (g[m]["ci95"][0] != want or g[m]["ci95"][1] != want):
                bad.append((nm, m, "ci", g[m]["ci95"]))
            if v is None or v != want:
                if not (m == "route_progress_ratio" and v is None):
                    bad.append((nm, m, v))
    return {"ok": not bad, "violations": bad, "expected": "endpoint error 0, heading error 0, lateral offset 0, route-progress ratio 1, CI degenerate at the same value"}


def control_mirror(cmd, logs, plan, n_boot=200, seed=SEED) -> dict:
    """deliberate regression: the arm's plan mirrored left-right must LOWER turn-active adherence. ok=False if the mirror changed nothing (a no-op mirror) or did not lower it."""
    m = mirror_lateral(plan)                                         # resolved at call time: a patched no-op mirror must turn this RED
    changed = float(np.abs(np.asarray(m) - np.asarray(plan)[..., : m.shape[-1]]).max()) if np.asarray(plan).size else 0.0
    ca, _ = CA.plan_classes(plan)
    cm, _ = CA.plan_classes(m)
    ta = subset_mask(cmd, "TURN_ACTIVE")
    acc = float((ca[ta] == cmd[ta]).mean())
    accm = float((cm[ta] == cmd[ta]).mean())
    return {"ok": bool(changed > 0.0 and accm < acc), "turn_active_adherence_real": acc, "turn_active_adherence_mirrored": accm, "max_abs_change_by_mirror": changed,
            "expected": "mirrored turn-active adherence strictly below the real one, and the mirror must actually change the plan"}


def control_self(cmd, logs, A, n_boot=200, seed=SEED) -> dict:
    """arm vs itself: every delta EXACTLY 0.0 with CI [0.0, 0.0], not separated"""
    P = family_paired(cmd, logs, A, A, None, None, n_boot, seed)
    bad = []

    def chk(path, d):
        pt, ci = d["point"], d["ci95"]
        if pt is None:                                                # undefined metric (e.g. no moving tokens): nothing to compare
            return
        if pt != 0.0 or ci[0] != 0.0 or ci[1] != 0.0 or d.get("separated") not in (False, None):
            bad.append((path, pt, ci))
    for sec in ("all_tokens", "turn_active"):
        chk(f"S_DEC/{sec}", P["S_DEC"][sec]["accuracy_delta"])
    for nm, d in P["S_DEC"]["per_class"].items():
        chk(f"S_DEC/{nm}", d)
    for nm, g in P["S_GOAL"].items():
        for m in METRICS:
            chk(f"S_GOAL/{nm}/{m}", g[m])
    return {"ok": not bad, "violations": bad, "expected": "delta 0.0, CI [0.0, 0.0] on every cell"}


# ------------------------------------------------------------------ real-data read
def commit_guard(min_gb: float = 4.0, max_wait_s: int = 1800) -> float:
    """wait while the free commit charge (page-file headroom) is below `min_gb` (the dev box is memory-tight); returns the free GB"""
    import ctypes
    t0 = time.time()
    while True:
        buf = ctypes.create_string_buffer(64)
        ctypes.c_uint32.from_buffer(buf).value = 64                   # MEMORYSTATUSEX.dwLength
        ctypes.windll.kernel32.GlobalMemoryStatusEx(buf)
        avail_pf = ctypes.c_uint64.from_buffer(buf, 32).value         # ullAvailPageFile = free commit charge
        gb = avail_pf / 2 ** 30
        if gb >= min_gb:
            return gb
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"COMMIT-LOW: {gb:.2f} GB free commit < {min_gb} GB after {max_wait_s} s; refusing to start (dev box memory-tight)")
        time.sleep(20)


def run_read(arms: dict, base: str, seams: dict, exp: dict, n_boot=N_BOOT, seed=SEED, controls=True) -> dict:
    """arms: name -> seam path; base = the name of the comparator. Reads every arm on the SAME (sorted) navtest tokens."""
    t0 = time.time()
    loaded = {n: CA.load_seam(p) for n, p in arms.items()}
    tokens = sorted(loaded[base]["tokens"])
    for n, s in loaded.items():
        if sorted(s["tokens"]) != tokens:
            return {"INVALID": [f"token set of {n} differs from the base's: pairing needs IDENTICAL tokens"]}
    cmd, logs = CA.commands_and_logs(tokens, exp)
    gt = np.asarray([exp[t]["human_future_poses"] for t in tokens], dtype=np.float64)
    counts = {CA.CMD_NAMES[i]: int((cmd == i).sum()) for i in range(4)}
    gates = {"G_COUNTS": {"ok": counts == CA.EXPECT_COUNTS and len(tokens) == CA.EXPECT_N and len(set(logs.tolist())) == CA.EXPECT_LOGS, "observed": counts,
                          "n_tokens": len(tokens), "n_logs": len(set(logs.tolist()))}}
    plans = {n: CA.align(s, tokens).astype(np.float64) for n, s in loaded.items()}
    fam = {n: family_arm(cmd, logs, plans[n], gt, n_boot, seed, n) for n in plans}
    res = {"instrument": "eval/instruments/strategic_family.py (STRAT-FAM)", "tier": TIER, "estimator": ESTIMATOR, "authority": AUTHORITY,
           "evidence_class": "MEASURED (ours) on the named seams; the logged future is the navtest export's human_future_poses",
           "tokens": {"n": len(tokens), "n_logs": len(set(logs.tolist())), "command_counts": counts, "turn_active_n": int(subset_mask(cmd, "TURN_ACTIVE").sum())},
           "seams": {n: s["path"] for n, s in loaded.items()}, "base": base,
           "arms": {n: strip(f) for n, f in fam.items()}, "paired_vs_base": {}, "gates": gates, "n_boot": n_boot, "seed": seed}
    for n in plans:
        if n != base:
            res["paired_vs_base"][f"{n}_minus_{base}"] = family_paired(cmd, logs, fam[n], fam[base], plans[n], plans[base], n_boot, seed)
    if controls:
        c = {"logged_future_as_arm": control_logged_future(cmd, logs, gt, 200, seed)}
        for n in plans:
            c[f"mirror_{n}"] = control_mirror(cmd, logs, plans[n])
        first = next(iter(fam))
        c[f"self_vs_self_{first}"] = control_self(cmd, logs, fam[first], 200, seed)
        res["controls"] = c
        for k, v in c.items():
            if not v["ok"]:
                res.setdefault("INVALID", []).append(f"control {k} did not read its known value")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _j(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("read")
    r.add_argument("--arm", action="append", required=True, help="NAME=seam.npz (repeatable); the arm(s) read against --base")
    r.add_argument("--base", required=True, help="NAME=seam.npz of the comparator")
    r.add_argument("--out", required=True)
    r.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    arms = dict(x.split("=", 1) for x in a.arm)
    bn, bp = a.base.split("=", 1)
    arms[bn] = bp
    commit_guard()
    res = run_read(arms, bn, arms, CA.load_export(), n_boot=a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=_j), encoding="utf-8")
    print("ZZSTRAT_DONE", a.out, "INVALID" if res.get("INVALID") else "ok", json.dumps(res.get("INVALID", [])))
    return 1 if res.get("INVALID") else 0


if __name__ == "__main__":
    sys.exit(main())
