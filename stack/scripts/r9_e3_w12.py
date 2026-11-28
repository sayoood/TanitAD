#!/usr/bin/env python3
"""r9_e3_w12.py -- the W1 / W2 input producer for the refcv9 E3 final-read judge (``r9_e3_verdict.py``).  CPU only, no GPU, no model.

    python r9_e3_w12.py --a-dump <refcv9 T1 S6 roll dump, inference seed S> --b-dump <refcv8 S6 dump of the SAME seed (GT + v0 + windows)> \
        --b-plan-npz <DEPLOY8_POS_<tree>_s<S>.npz> --seed S --split all --out w12_s<S>.json

WHAT.  One document per INFERENCE SEED (the judge reads W1 / W2 at seeds 0 AND 1 and needs both)::

    {"schema": "r9e3-w12-v1", "split": "all", "tier": "T1 ...", "infer_seed": S, "rows": {"FD": row, "MD": row}, "lon_indicators": {...}, ...}

and a row is the PAIRED PER-WINDOW form ``r9_e3_verdict.row_read`` bootstraps (``taniteval.ci.paired_episode_cluster_bootstrap``, B 2000, seed 0)::

    {"a": [refcv9 0/1 per window], "b": [baseline 0/1 per window], "eid": [episode id per window], "better": "lower"}

    FD  false deceleration: the plan decelerates on a window whose GROUND TRUTH is cruise     (population = GT-cruise windows)    W1: separated LOWER than the baseline
    MD  missed deceleration: the plan does not decelerate on a GT-decel window                (population = GT-decel windows)     W2: NOT separated worse than the baseline

a and b are the SAME windows (a row holds ONLY population windows: the bootstrap takes a plain mean, so a NaN outside the population would poison it).  delta = refcv9 - baseline.

DEFINITIONS ARE IMPORTED, NOT RE-IMPLEMENTED.  The plan class (decel / cruise / accel from the plan's LAST chord against the measured v0, +-1.5 m/s) is
``s6_lon_decomp.plan_lon_class``; the FD / MD indicator (GT class from the logged future ``g``, plan class from the plan) is
``cruise_brake_retime_r8.indicator``.  Both live in ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code`` (or ``--battery-code`` / ``$R9_W12_BATTERY_CODE``);
the SHIP-R8 H2cn verdict's own ``l2_common.lon_indicators`` is the same formula (the test pins the equality).

THE BASELINE = register row R8-SHIP-H2CN-1, refcv8 EMA-48,852 + the H2cn stage (arm ``H2CN_DEPLOY8``).  Its per-window PLAN is in the repo
(``.../2026-10-26-refcv8-ship-r8-h2cn-factored/out/arms/h2cn/H2CN_DEPLOY8/DEPLOY8_POS_step48852ema_pf_s<S>.npz``: ``S6_pos`` [N,6,2], ``S6_ws``, ``S6_clip``); its GT ``g``,
``v0`` and window keys are in the battery archive dump of the same tree (dev box ``C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/s6_s<S>``, NOT in the repo).
``--b-plan-npz`` replaces the b-dump's ``os`` plan by ``S6_pos`` after a row-for-row alignment check (``S6_ws`` / ``S6_clip`` == the dump's); without it the b-dump's ``os`` plan is
the baseline (any other baseline: pass its dump).  ``--verify-baseline`` reproduces the banked SHIP numbers (see below).

WINDOWS.  Split "all" = all 139 eval139 EPISODES (design 5.1: "four families, W1-W5 on all of eval139 (comparability with refcv8's registered table)").  The PRIMARY
population (``--population compliant``, the default) is refcv8's registered window set: the S6 CEILING-COMPLIANT windows (AMENDMENT A-R8-CLAMP-2 section 2.1: logged max
speed over the 6 s future <= the fed ceiling; "it depends on the logged data and the fed label only, never on a model"), read from the baseline npz's ``S6_compliant`` and
applied to BOTH arms on the paired windows -- so rows FD / MD are on the population of the registered SHIP-R8 H2cn cells (BARS_4C FD n 1,856 / MD n 639).  The same rates on
ALL S6 windows are written BESIDE as rows FD_all / MD_all (the judge reads FD / MD only; the beside rows never decide).  Pre-data reading, MM 4a3d05 2026-11-10, recorded before
any refcv9 E3 output exists.  ``--population all`` (no mask needed) writes all-window FD / MD rows: a beside read, not the W1 / W2 input.  The grid must be the S6 grid
(dt 1.0 s, k 6): the class definition is only defined there.

REFUSED (rc 2, nothing written): ``--split`` other than ``all`` (DEV / HOLD); a dump manifest stamped dev / hold; an input that does not cover all 139 eval episodes (a DEV-half dump);
a tier that is not T1 (read from the manifest ``tiers[arm]`` / ``tier``; absent = refused); a missing / mismatching inference seed (manifest ``inference_seed`` vs ``--seed``, also the
baseline npz meta); windows that do not align between a and b (a window in one arm only, duplicate keys); GT / v0 disagreement between the arms; a grid that is not dt 1.0 / k 6.

ESTIMATOR / CAVEAT (H-ESTIM-SEED-1): the paired interval answers "another draw of EPISODES" only; one call = one inference seed; ONE training run per arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCHEMA = "r9e3-w12-v1"
TIER = "T1 self-action OPEN LOOP on logged eval139 frames (not driving performance)"
ESTIMATOR = "taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, over EPISODES (clips); delta = refcv9 - baseline"
CAVEAT = ("H-ESTIM-SEED-1: the paired interval answers another draw of EPISODES, not another training run (one training run per arm) and not another INFERENCE draw; "
          "one document = one inference seed")
EVAL139_EPISODES = 139
S6_DT, S6_K = 1.0, 6
ROW_BETTER = {"FD": "lower", "MD": "lower"}
POPULATIONS = ("compliant", "all")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BATTERY_REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code"
BANKED_BARS_4C = ("FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv8-ship-r8-h2cn-factored/out_4c/BARS_4C.json")
DEFAULT_ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf"


class Refused(Exception):
    """An input the producer will not read (rc 2).  Never a quiet narrowing."""


def _sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ imported definitions
def battery_code_dir(override=None) -> Path:
    cand = [override, os.environ.get("R9_W12_BATTERY_CODE"), REPO / BATTERY_REL]
    for c in cand:
        if c and (Path(c) / "s6_lon_decomp.py").is_file():
            return Path(c)
    raise Refused(f"the battery code dir with s6_lon_decomp.py was not found (tried {[str(c) for c in cand if c]}): the FD / MD definitions are imported, never re-implemented")


def _defs(code_dir=None):
    d = battery_code_dir(code_dir)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import cruise_brake_retime_r8 as CB          # noqa: N812  (FD / MD indicator)
    import s6_lon_decomp as S6                   # noqa: N812  (plan_lon_class)
    return S6, CB, d


def _ci():
    try:
        from taniteval import ci
    except ModuleNotFoundError:
        for cand in (REPO / "taniteval", HERE.parent / "taniteval"):
            if (cand / "taniteval" / "ci.py").is_file():
                sys.path.insert(0, str(cand))
                break
        from taniteval import ci
    return ci


# ------------------------------------------------------------------------------------------------------------------ dump loading
def _manifest(dump_dir) -> dict:
    try:
        return json.loads((Path(dump_dir) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{dump_dir}: no readable manifest.json ({e})")


def _seed_of(man) -> object:
    s = man.get("inference_seed")
    return s.get("seed") if isinstance(s, dict) else s


def check_stamps(man: dict, where: str, arm: str, seed: int, expect_episodes: int) -> dict:
    """Refuse a DEV / HOLD stamp, a non-T1 or unstamped tier, a missing / wrong inference seed, a non-S6 grid, a partial episode set."""
    for k in ("split", "subset"):
        v = man.get(k)
        if isinstance(v, str) and v.lower() in ("dev", "hold"):
            raise Refused(f"{where}: manifest {k} = {v!r} -- W1 / W2 are read on ALL of eval139; a DEV / HOLD input is refused")
    tier = (man.get("tiers") or {}).get(arm) if isinstance(man.get("tiers"), dict) else None
    tier = tier if tier is not None else man.get("tier")
    if not (isinstance(tier, str) and tier.strip()):
        raise Refused(f"{where}: no tier stamp for arm {arm!r} (manifest `tiers[{arm}]` / `tier`): an unstamped input is not read")
    if not tier.startswith("T1"):
        raise Refused(f"{where}: tier {tier!r} -- W1 / W2 are T1 (action-closed) bars; a T0 / other-tier input is refused")
    s = _seed_of(man)
    if s is None:
        raise Refused(f"{where}: the manifest carries no inference_seed: a W1 / W2 document is per inference seed")
    if str(s) != str(seed):
        raise Refused(f"{where}: manifest inference_seed {s!r} but --seed {seed}")
    grid = man.get("grid") or {}
    if float(grid.get("dt_s", -1)) != S6_DT or int(grid.get("k", -1)) != S6_K:
        raise Refused(f"{where}: grid dt_s={grid.get('dt_s')!r} k={grid.get('k')!r} -- FD / MD are defined on the S6 grid (dt {S6_DT}, k {S6_K})")
    n_ep = len(man.get("episodes") or [])
    if n_ep != expect_episodes:
        raise Refused(f"{where}: {n_ep} episode(s), eval139 has {expect_episodes} -- a partial set (a DEV half?) is not split 'all'")
    return {"tier": tier, "infer_seed": s, "n_episodes": n_ep}


def load_dump(dump_dir, arm: str = "os") -> dict:
    """{keys [(clip_id, ws)], clip_index [N], P [N,K,2], G [N,K,2], v0 [N], dt, manifest}; windows in the dump's file order."""
    man = _manifest(dump_dir)
    d = Path(dump_dir)
    keys, cidx, P, G, v0 = [], [], [], [], []
    for e in man.get("episodes") or []:
        f = d / f"ep{int(e['file_index']):03d}.npz"
        try:
            z = np.load(f)
        except OSError as ex:
            raise Refused(f"{f}: unreadable ({ex})")
        for k in ("g", arm, "ws", "v0"):
            if k not in z.files:
                raise Refused(f"{f} has no key {k!r} (keys {z.files})")
        ws = np.asarray(z["ws"]).reshape(-1)
        if not (z["g"].shape[0] == ws.size == z["v0"].shape[0] and z[arm].shape == z["g"].shape):
            raise Refused(f"{f}: g / {arm} / ws / v0 shapes disagree")
        keys += [(str(e["clip_id"]), int(w)) for w in ws]
        cidx += [int(e.get("episode_index", e["file_index"]))] * ws.size
        P.append(np.asarray(z[arm], dtype=np.float64)[..., :2])
        G.append(np.asarray(z["g"], dtype=np.float64)[..., :2])
        v0.append(np.asarray(z["v0"], dtype=np.float64).reshape(-1))
    if not keys:
        raise Refused(f"{dump_dir}: the dump holds no windows")
    if len(set(keys)) != len(keys):
        raise Refused(f"{dump_dir}: duplicate (clip_id, ws) window keys")
    return {"keys": keys, "clip_index": np.asarray(cidx, np.int64), "P": np.concatenate(P), "G": np.concatenate(G), "v0": np.concatenate(v0),
            "dt": float((man.get("grid") or {}).get("dt_s", S6_DT)), "manifest": man, "dir": str(d), "arm": arm}


def apply_plan_npz(b: dict, npz_path, seed: int, prefix: str = "S6") -> dict:
    """Replace the b-dump's plan by ``<prefix>_pos`` of a DEPLOY8_POS npz after a row-for-row alignment check; returns {**b, P, compliant, plan_npz}."""
    z = np.load(npz_path, allow_pickle=True)
    for k in (f"{prefix}_pos", f"{prefix}_ws", f"{prefix}_clip"):
        if k not in z.files:
            raise Refused(f"{npz_path}: no key {k!r}")
    try:
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    except ValueError:
        meta = {}
    if str(meta.get("inference_seed")) != str(seed):
        raise Refused(f"{npz_path}: baseline npz meta inference_seed {meta.get('inference_seed')!r} but --seed {seed} (a missing or wrong seed is refused)")
    ws = np.asarray([w for _, w in b["keys"]], np.int64)
    if not (np.array_equal(z[f"{prefix}_ws"], ws) and np.array_equal(z[f"{prefix}_clip"], b["clip_index"])):
        raise Refused(f"{npz_path}: {prefix}_ws / {prefix}_clip do not align row for row with {b['dir']} (the plan would be scored against another window's GT)")
    if z[f"{prefix}_pos"].shape != b["P"].shape:
        raise Refused(f"{npz_path}: {prefix}_pos shape {z[f'{prefix}_pos'].shape} != dump plan shape {b['P'].shape}")
    comp = np.asarray(z[f"{prefix}_compliant"], bool).reshape(-1) if f"{prefix}_compliant" in z.files else None
    if comp is not None and comp.size != len(b["keys"]):
        raise Refused(f"{npz_path}: {prefix}_compliant has {comp.size} entries for {len(b['keys'])} windows")
    return {**b, "P": np.asarray(z[f"{prefix}_pos"], dtype=np.float64), "compliant": comp, "plan_npz": str(npz_path), "plan_npz_sha256": _sha256_file(npz_path)}


def _align(a: dict, b: dict, gt_tol: float = 1e-4, v0_tol: float = 1e-3):
    """Index vectors (ia, ib) pairing a and b on (clip_id, ws).  A window in one arm only, or a GT / v0 disagreement, REFUSES (never an intersection)."""
    pos_b = {k: i for i, k in enumerate(b["keys"])}
    only_a = [k for k in a["keys"] if k not in pos_b]
    set_a = set(a["keys"])
    only_b = [k for k in b["keys"] if k not in set_a]
    if only_a or only_b:
        raise Refused(f"window-set mismatch: {len(only_a)} window(s) only in a ({a['dir']}), {len(only_b)} only in b ({b['dir']}); the arms must be rolled on the SAME windows")
    ia = np.arange(len(a["keys"]))
    ib = np.asarray([pos_b[k] for k in a["keys"]])
    gd = float(np.abs(a["G"][ia] - b["G"][ib]).max())
    vd = float(np.abs(a["v0"][ia] - b["v0"][ib]).max())
    if not gd <= gt_tol:
        raise Refused(f"the two arms' GT `g` differ by up to {gd:.6g} m on the paired windows (tol {gt_tol}): different label clocks / corpora")
    if not vd <= v0_tol:
        raise Refused(f"the two arms' v0 differ by up to {vd:.6g} m/s on the paired windows (tol {v0_tol})")
    return ia, ib


# ------------------------------------------------------------------------------------------------------------------ the indicators
def window_indicators(P, G, v0, dt, mask=None, *, code_dir=None) -> dict:
    """{'FD': [N] 0/1/NaN, 'MD': [N] 0/1/NaN, 'gt': classes, 'pc': classes}: NaN outside the population.  Definitions imported (see module doc)."""
    S6, CB, _ = _defs(code_dir)
    mask = np.ones(len(v0), bool) if mask is None else np.asarray(mask, bool)
    gt = S6.plan_lon_class(np.asarray(G, np.float64), np.asarray(v0, np.float64), dt)
    pc = S6.plan_lon_class(np.asarray(P, np.float64), np.asarray(v0, np.float64), dt)
    return {"FD": CB.indicator(pc, gt, mask, "false_decel"), "MD": CB.indicator(pc, gt, mask, "missed_decel"), "gt": gt, "pc": pc}


def _row(a_vals, b_vals, eids, better: str) -> dict:
    """The paired per-window row ``row_read`` bootstraps.  Population windows only."""
    return {"a": [float(x) for x in a_vals], "b": [float(x) for x in b_vals], "eid": [str(e) for e in eids], "better": better,
            "n_windows": int(len(a_vals)), "n_episodes": int(len(set(str(e) for e in eids)))}


def paired_summary(row: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    r = _ci().paired_episode_cluster_bootstrap(row["a"], row["b"], row["eid"], n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes", "estimator")}


def build(a_dir, b_dir, *, seed: int, split: str = "all", arm_a: str = "os", arm_b: str = "os", b_plan_npz=None, expect_episodes: int = EVAL139_EPISODES,
          labels=("refcv9", "refcv8+H2cn"), code_dir=None, n_boot: int = 2000, population: str = "compliant") -> dict:
    if population not in POPULATIONS:
        raise Refused(f"--population {population!r}: one of {POPULATIONS}")
    if split != "all":
        raise Refused(f"--split {split!r}: W1 / W2 are read on ALL of eval139 (split 'all'); DEV / HOLD is refused")
    if seed is None:
        raise Refused("no --seed: a W1 / W2 document is per inference seed (the judge needs 0 AND 1)")
    A, B = load_dump(a_dir, arm_a), load_dump(b_dir, arm_b)
    sa = check_stamps(A["manifest"], str(a_dir), arm_a, seed, expect_episodes)
    sb = check_stamps(B["manifest"], str(b_dir), arm_b, seed, expect_episodes)
    if b_plan_npz:
        B = apply_plan_npz(B, b_plan_npz, seed)
    if population == "compliant" and B.get("compliant") is None:
        raise Refused("population 'compliant' (the default: refcv8's registered FD / MD table is on the S6 CEILING-COMPLIANT windows, AMENDMENT A-R8-CLAMP-2) needs the "
                      "baseline's S6_compliant mask: pass --b-plan-npz (or --population all for an all-window beside read, which is not the W1 / W2 input)")
    ia, ib = _align(A, B)
    keys = [A["keys"][i] for i in ia]
    eid_all = [k[0] for k in keys]
    dt = A["dt"]
    comp = np.asarray(B["compliant"], bool)[ib] if population == "compliant" else None
    pops = {"": comp, "_all": None} if population == "compliant" else {"": None}
    rows, rates = {}, {}
    for suffix, mask in pops.items():
        ind_a = window_indicators(A["P"][ia], A["G"][ia], A["v0"][ia], dt, mask=mask, code_dir=code_dir)
        ind_b = window_indicators(B["P"][ib], B["G"][ib], B["v0"][ib], dt, mask=mask, code_dir=code_dir)
        role = "PRIMARY (the W1 / W2 input)" if suffix == "" and population == "compliant" else "BESIDE (all S6 windows; never decides)"
        for k in ("FD", "MD"):
            name = k + suffix
            pop = np.isfinite(ind_a[k])
            if not np.array_equal(pop, np.isfinite(ind_b[k])):
                raise Refused(f"{name}: the population differs between a and b (same GT, same mask: this is a bug)")
            sel = np.flatnonzero(pop)
            rows[name] = _row(ind_a[k][sel], ind_b[k][sel], [eid_all[i] for i in sel], ROW_BETTER[k])
            rows[name]["role"] = role
            rates[name] = {"n_windows": int(sel.size), "n_episodes": rows[name]["n_episodes"], "role": role,
                           "rate_a": float(ind_a[k][sel].mean()) if sel.size else None, "rate_b": float(ind_b[k][sel].mean()) if sel.size else None}
            if sel.size:
                rows[name]["summary"] = paired_summary(rows[name], n_boot=n_boot)
    h = hashlib.sha256()
    for c, w in sorted(keys):
        h.update(f"{c}\t{w}\n".encode())
    hm = None
    if comp is not None:
        hm = hashlib.sha256()
        for (c, w), m in sorted(zip(keys, comp.tolist())):
            hm.update(f"{c}\t{w}\t{int(m)}\n".encode())
        hm = hm.hexdigest()
    S6, _CB, cdir = _defs(code_dir)
    pop_txt = ("S6 CEILING-COMPLIANT windows of all eval139 episodes (the baseline npz's S6_compliant, model-independent, = refcv8's registered FD / MD population); "
               "FD_all / MD_all beside on all S6 windows") if population == "compliant" else "all S6 windows of eval139 (a BESIDE read: not the W1 / W2 input)"
    return {"schema": SCHEMA, "split": "all", "tier": TIER, "infer_seed": int(seed), "labels": f"{labels[0]} - {labels[1]}", "delta": f"{labels[0]} - {labels[1]}",
            "estimator": ESTIMATOR, "rows": rows, "lon_indicators": rates, "n_windows_paired": len(keys), "n_episodes": len(set(eid_all)), "window_set_sha256": h.hexdigest(),
            "population": population, "n_compliant": int(comp.sum()) if comp is not None else None, "compliant_mask_sha256": hm,
            "definitions": {"plan_lon_class": "s6_lon_decomp.plan_lon_class (last chord vs v0, +-1.5 m/s)", "indicator": "cruise_brake_retime_r8.indicator",
                            "FD": "plan decel on GT-cruise windows", "MD": "plan not-decel on GT-decel windows", "population": pop_txt,
                            "code_dir": str(cdir), "s6_lon_decomp_sha256": _sha256_file(Path(cdir) / "s6_lon_decomp.py")},
            "inputs": {"a": {"dir": str(a_dir), "arm": arm_a, "manifest_sha256": _sha256_file(Path(a_dir) / "manifest.json"), **sa},
                       "b": {"dir": str(b_dir), "arm": arm_b, "manifest_sha256": _sha256_file(Path(b_dir) / "manifest.json"), **sb,
                             "plan_npz": B.get("plan_npz"), "plan_npz_sha256": B.get("plan_npz_sha256")}},
            "n_boot": n_boot, "boot_seed": 0, "caveat": CAVEAT}


# ------------------------------------------------------------------------------------------------------------------ baseline reproduction
def verify_baseline(b_dir, b_plan_npz, seed: int, banked_json=None, code_dir=None) -> dict:
    """Reproduce the banked SHIP-R8 H2CN_DEPLOY8 FD / MD rates (BARS_4C.json seeds.<s>.cells['H2CN_DEPLOY8-RETIME_DEPLOY8|FD'|'|MD'] keys mean_new, n_windows, n_episodes)
    on the baseline's own compliant S6 mask.  Returns {ok, cells}."""
    B = apply_plan_npz(load_dump(b_dir, "os"), b_plan_npz, seed)
    if B.get("compliant") is None:
        raise Refused(f"{b_plan_npz}: no S6_compliant mask")
    ind = window_indicators(B["P"], B["G"], B["v0"], B["dt"], mask=B["compliant"], code_dir=code_dir)
    bj = json.loads(Path(banked_json or (REPO / BANKED_BARS_4C)).read_text(encoding="utf-8"))
    cells, ok = {}, True
    for k in ("FD", "MD"):
        fin = np.isfinite(ind[k])
        ref = bj["seeds"][str(seed)]["cells"][f"H2CN_DEPLOY8-RETIME_DEPLOY8|{k}"]
        mine = {"mean_new": float(ind[k][fin].mean()), "n_windows": int(fin.sum()), "n_episodes": len(set(np.asarray(B['clip_index'])[fin].tolist()))}
        good = abs(mine["mean_new"] - ref["mean_new"]) < 1e-9 and mine["n_windows"] == ref["n_windows"]
        ok &= bool(good)
        cells[k] = {"mine": mine, "banked": {x: ref[x] for x in ("mean_new", "n_windows", "n_episodes")}, "equal": bool(good)}
    return {"ok": bool(ok), "cells": cells, "banked_json": str(banked_json or (REPO / BANKED_BARS_4C))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a-dump", help="refcv9 T1 S6 roll dump dir (manifest.json + ep###.npz {g, os, ws, v0})")
    ap.add_argument("--b-dump", required=True, help="baseline S6 dump dir of the SAME seed (GT, v0, windows; battery archive s6_s<S>)")
    ap.add_argument("--b-plan-npz", help="baseline DEPLOY8_POS npz (S6_pos replaces the b-dump's plan)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--population", default="compliant", choices=POPULATIONS,
                    help="compliant (default, the W1 / W2 input: refcv8's registered window set, needs --b-plan-npz) or all (a beside read)")
    ap.add_argument("--arm-a", default="os")
    ap.add_argument("--arm-b", default="os")
    ap.add_argument("--expect-episodes", type=int, default=EVAL139_EPISODES)
    ap.add_argument("--battery-code", help="dir with s6_lon_decomp.py / cruise_brake_retime_r8.py")
    ap.add_argument("--out", help="output w12 JSON")
    ap.add_argument("--verify-baseline", action="store_true", help="reproduce the banked SHIP-R8 H2CN_DEPLOY8 FD / MD rates (compliant mask) and exit")
    args = ap.parse_args(argv)
    try:
        if args.verify_baseline:
            if not args.b_plan_npz:
                raise Refused("--verify-baseline needs --b-plan-npz")
            r = verify_baseline(args.b_dump, args.b_plan_npz, args.seed, code_dir=args.battery_code)
            print(json.dumps(r, indent=1))
            return 0 if r["ok"] else 1
        if not (args.a_dump and args.out):
            raise Refused("--a-dump and --out are required")
        doc = build(args.a_dump, args.b_dump, seed=args.seed, split=args.split, arm_a=args.arm_a, arm_b=args.arm_b, b_plan_npz=args.b_plan_npz,
                    expect_episodes=args.expect_episodes, code_dir=args.battery_code, population=args.population)
    except Refused as e:
        print(f"[r9-e3-w12] REFUSED: {e}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    li = doc["lon_indicators"]
    print(f"[r9-e3-w12] seed {args.seed}: FD {li['FD']['rate_a']} vs {li['FD']['rate_b']} (n {li['FD']['n_windows']}), MD {li['MD']['rate_a']} vs {li['MD']['rate_b']} (n {li['MD']['n_windows']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
