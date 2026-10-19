"""tac_retime_r8.py -- the TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5), CPU only, zero GPU.

NEW module.  Imports RT / TFM / BD / DF (deploy_forward_r8) and edits none of them.

    python tac_retime_r8.py --tree step30000ema_unc      # -> raw/retime_deploy/TAC_<tree>.json + TAC_<tree>.md

AMENDMENT 2 (2026-10-19 ~16:10): the SCORED arm is `RETIME_DEPLOY8` = ONE re-timing call per window on all 8 model slots (0.5, 1, 1.5, 2, 3, 4, 5, 6 s) with Amendment 1's table, prior =
ha0_ext_pose at the same 8 slots; each surface is scored on that call's outputs at its own instants (S2 = slots 0-3, S6 = slots 1, 3-7).  Its positions come from the sibling build
`deploy8_forward_r8.py` -> raw/retime_deploy/DEPLOY8_POS_<tree>_s<seed>.npz (its documented consumer contract: `<S>_pos` at the surface's own instants, + clip / ws / eid / os / compliant); this module only
READS them (loader below; refuses a file it cannot align -> clause 5 INCOMPLETE).  The per-surface `RETIME_DEPLOY` (Amendment 1's form), `RETIME_TRAINFIT` and the unmodified os are printed BESIDE.

CLAUSE 5 (GOALS_AND_CLAIMS, D-R8-SHIP-RETIME): "TACTICAL TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan not separated worse than refcv7-50,400 at both
seeds (instrument being built 2026-10-19; if it cannot be computed, SHIP-R8 reads INCOMPLETE, not PASS)".

THE METRIC (the battery's own, refav1_arm._components): per window, the plan's (dy, dv, v0, v1) = four_families.maneuver_kinematics(plan, dt) go through the programme's canonical
labeller refc_tactical.factor_from_kinematics -> (lat in {keep, left, right}, lon in {steady, accelerate, brake_stop}); TAC_traj_{lat,lon}_correct = 1.0 iff the plan's label equals the
logged ground truth's label computed the same way ("trajectory-derived decision correct", HIGHER is better).  This module recomputes the labels DIRECTLY (not through
`_components`) and asserts the result array-equal to `_components`' TAC keys on os / refcv7 / the re-timed plan.

THE RE-TIMED PLAN: RT.retime_apply(os, ha0_ext_pose, g) with g = the DEPLOYED table (raw/retime_deploy/FROZEN_GAINS_RETIME_deploy.json); RETIME_TRAINFIT (per-surface train tables) and
the unmodified os are printed beside.  The re-timing moves points along the os polyline, so the lateral LABEL can change only through the end heading / dyaw; the longitudinal label
moves with dv (the speed change between the first and last plan segment) -- the cell exists because ADE improvement by re-timing could still cost the decision.

SEPARATED WORSE (an ACCURACY metric, higher is better; the bootstrap cell is arm - refcv7): the CI upper bound < 0.  (The ERROR-metric rule `lo > 0` of pair_read_r8.bar_NR_metric would
read an improvement as a failure on these rows -- the a0k0_headroom_r8.resummarise docstring records exactly that mistake.)  Clause 5 PASSES iff NO cell {S2, S6} x {lat, lon} x
{seed 0, seed 1} is separated worse.  SCOPE (an interpretation, stated): every surface on which the battery defines the rows (S2 and S6), the strict reading.

ESTIMATOR: paired episode-cluster bootstrap B 2000, bootstrap seed 0 (pair_read_r8.paired_cell -> taniteval.ci), models and gains FIXED: it answers 'would another draw of EPISODES say this?'
-- NOT another training run, NOT another inference run (seeds 0 and 1 are the replicate).  Tier: T1 self-action OPEN LOOP (not driving performance).  Compliant set = windows whose logged
human max speed <= the fed ceiling (compliant_reread_r8.compliant_mask), the set SHIP-R8 names.

CONTROLS (ALL must hold or ALL_PASS is False and ship_r8_verdict reads clause 5 INCOMPLETE):
  K1  the UNMODIFIED os reproduces the battery's own banked TAC values on the same windows: per (surface, seed) the all-window os / refcv7 TAC means equal PAIRED_READ_<tree>.json's
      `means` EXACTLY, and the all-window os - refcv7 cell equals its banked `os_minus_refcv7_50400` cell (delta/lo/hi/n/separated and mean_new/mean_ref).  The literals are quoted in the .md.
  K2  the same os - refcv7 cell on the ceiling-compliant windows equals the compliant tree json's banked cell (raw/compliant_reread/trees/<tree>.json) -- required when that file exists.
  K3  ANALYTIC: the ground truth scored against itself reads exactly 1.0 on both rows (every window).
  K4  CONST_G1 (all gains 1) re-timing returns os bit for bit: TAC identical, cell vs os exactly 0 with a zero-width CI.
  K5  the directly recomputed labels equal refav1_arm._components' TAC arrays (os, refcv7, RETIME_DEPLOY).
  K6  (cross-check, DETERMINISM not correctness) when FORWARD_DEPLOY_<tree>.json exists, its NR_minus_refcv7 TAC cells for the per-surface RETIME_DEPLOY equal the cells computed here.
  K7  (cross-check, DETERMINISM not correctness) when FORWARD_DEPLOY8_<tree>.json exists, its NR_minus_refcv7 TAC cells for RETIME_DEPLOY8 equal the cells computed here from the npz positions
      (the same positions scored by two code paths: RT.forward's and this module's direct labeller).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a0k0_headroom_retime_r8 as RT  # noqa: E402
import build_deploy_gains_r8 as BD  # noqa: E402
import deploy_forward_r8 as DF  # noqa: E402
import retime_trainfit_r8 as TFM  # noqa: E402

PKG = HERE.parent
DEP = PKG / "raw" / "retime_deploy"
ARCHIVE = "C:/Users/Admin/refcv8_eval_archive/battery"
SEEDS = (0, 1)
SURFS = ("S2", "S6")
SETS = ("compliant", "all")
KEYS = ("TAC_traj_lat_correct", "TAC_traj_lon_correct")
N_BOOT = RT.N_BOOT
TIER = RT.TIER
RULE = ("CLAUSE 5: TAC_traj_lat_correct and TAC_traj_lon_correct of the RE-TIMED plan (RETIME_DEPLOY8, AMENDMENT 2) not separated worse than refcv7-50,400 at both inference seeds, "
        "on ceiling-compliant windows, on every surface the battery defines them (S2, S6).  ACCURACY metric: separated worse <=> the paired-bootstrap CI upper bound < 0.")
SCORED_ARM = "RETIME_DEPLOY8"
CELL_EXACT_KEYS = ("delta", "lo", "hi", "n_windows", "n_episodes", "separated", "n_dropped_nonfinite", "mean_new", "mean_ref")


# ------------------------------------------------------------------------------------------ pure
def separated_worse(cell: dict) -> bool | None:
    """ACCURACY row, cell = arm - refcv7: worse AND separated <=> hi < 0.  None when the cell carries no interval (never counted as clean)."""
    if not cell or "hi" not in cell or cell.get("hi") is None:
        return None
    return bool(cell["hi"] < 0)


def clause5(cells: dict) -> dict:
    """cells = {'S2|TAC_traj_lat_correct|s0': cell (RETIME_DEPLOY - refcv7), ...}.  FAIL if any present cell is separated worse; else INCOMPLETE if a cell is missing / interval-less; else PASS (all 8 present)."""
    need = [f"{s}|{k}|s{sd}" for s in SURFS for k in KEYS for sd in SEEDS]
    miss = [n for n in need if n not in cells or separated_worse(cells[n]) is None]
    worse = [n for n in need if n in cells and separated_worse(cells[n]) is True]
    status = "FAIL" if worse else ("INCOMPLETE" if miss else "PASS")       # a separated-worse cell is conclusive even when another cell is missing
    return {"status": status, "n_cells": len(need), "missing_or_undetermined": miss, "separated_worse_cells": worse, "rule": RULE}


def cell_eq(mine: dict, banked: dict, keys=CELL_EXACT_KEYS) -> dict:
    """Exact equality on every key the BANKED cell carries (from `keys`); a key the banked cell lacks is not compared but listed."""
    diff = {k: (mine.get(k), banked.get(k)) for k in keys if k in banked and mine.get(k) != banked.get(k)}
    absent = [k for k in keys if k not in banked]
    return {"exact": bool(not diff and "delta" in banked), "diff": diff, "banked_lacks": absent}


def tac_from_labels(lat_p, lon_p, lat_g, lon_g) -> dict:
    return {"TAC_traj_lat_correct": (np.asarray(lat_p) == np.asarray(lat_g)).astype(np.float32),
            "TAC_traj_lon_correct": (np.asarray(lon_p) == np.asarray(lon_g)).astype(np.float32)}


# ------------------------------------------------------------------------------------------ labels (direct; same canonical functions `_components` calls)
def labels(P, dt: float) -> tuple:
    import torch
    from taniteval import four_families as ff
    from tanitad.refs.refc_tactical import factor_from_kinematics
    dy, dv, v0, v1, _ = ff.maneuver_kinematics(torch.as_tensor(P).float(), dt)
    lat, lon = factor_from_kinematics(dy, dv, v0, v1)
    return lat.numpy(), lon.numpy()


def tac_direct(P, G, dt: float) -> tuple:
    lp, op = labels(P, dt)
    lg, og = labels(G, dt)
    return tac_from_labels(lp, op, lg, og), (lp, op)


def fmt(c, dp=4) -> str:
    if not c or "delta" not in c:
        return "n/a"
    return f"{c['delta']:+.{dp}f} [{c['lo']:+.{dp}f}, {c['hi']:+.{dp}f}]" + (" SEP" if c.get("separated") else "")


# ------------------------------------------------------------------------------------------ the run

# ------------------------------------------------------------------------------------------ the DEPLOY8 positions (sibling build; READ ONLY here)
def select_deploy8(arrs: dict, surf: str, S: dict, tree: str | None = None, seed: int | None = None) -> tuple:
    """(P [n_rows, T_surface, 2] float32 | None, reason).  The npz layout is the sibling's documented CONTRACT (deploy8_forward_r8.CONTRACT): `<S>_pos` [n, T, 2] float32 = the re-timed plan AT THE
    SURFACE'S instants (`<S>_instants_s`), rows = ALL windows of the surface in RT.surface_inputs order, with `<S>_clip`, `<S>_ws`, `<S>_eid`, `<S>_os` (the panel's os) and `<S>_compliant`.
    Row alignment is VERIFIED, never assumed, and any failure REFUSES (-> clause 5 INCOMPLETE): the file's clip / ws / eid must equal this run's rows elementwise, its `<S>_os` must equal this
    run's os plan BIT FOR BIT, its `<S>_compliant` this run's mask, its instants the surface's, and (when present) its meta must name this tree, this inference seed and the arm RETIME_DEPLOY8."""
    need = [f"{surf}_{k}" for k in ("pos", "clip", "ws", "eid", "os", "compliant", "instants_s")]
    miss = [k for k in need if k not in arrs]
    if miss:
        return None, f"the npz lacks {miss} (consumer contract of deploy8_forward_r8)"
    if "meta_json" in arrs:
        try:
            meta = json.loads(str(np.asarray(arrs["meta_json"])))
        except ValueError:
            return None, "meta_json is not parseable"
        for k, want in (("arm", SCORED_ARM), ("tree", tree), ("inference_seed", seed)):
            if want is not None and meta.get(k) != want:
                return None, f"meta {k} is {meta.get(k)!r}, this run needs {want!r}"
    P = np.asarray(arrs[f"{surf}_pos"])
    n = len(S["eid"])
    if P.ndim != 3 or P.shape[0] != n or P.shape[2] != 2:
        return None, f"{surf}_pos has shape {P.shape}; this run's {surf} panel has {n} rows"
    want_t = [float(h) / 10.0 for h in S["hz"]]
    if [float(x) for x in np.asarray(arrs[f"{surf}_instants_s"]).tolist()] != want_t or P.shape[1] != len(want_t):
        return None, f"{surf}_instants_s {np.asarray(arrs[f'{surf}_instants_s']).tolist()} (T={P.shape[1]}) != the surface's {want_t}"
    for k, mine in (("clip", S["clip"]), ("ws", S["ws"]), ("eid", S["eid"])):
        got = np.asarray(arrs[f"{surf}_{k}"])
        if got.shape != (n,) or not np.array_equal(got.astype(np.int64), np.asarray(mine, np.int64)):
            return None, f"{surf}_{k} differs from this run's rows: NOT row-aligned"
    if not np.array_equal(np.asarray(arrs[f"{surf}_os"]), np.asarray(S["X"]["O"])):
        return None, f"{surf}_os differs from this run's os plan (not bit-equal): NOT row-aligned"
    if not np.array_equal(np.asarray(arrs[f"{surf}_compliant"], bool), np.asarray(S["mask"], bool)):
        return None, f"{surf}_compliant differs from this run's ceiling mask"
    return P.astype(np.float32), "ok"


def load_deploy8_surface(tree: str, seed: int, surf: str, S: dict, root: Path | None = None) -> tuple:
    p = (root or DEP) / f"DEPLOY8_POS_{tree}_s{seed}.npz"
    if not p.exists():
        return None, f"{p.name} does not exist (the sibling deploy8 build has not produced it)"
    with np.load(p, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    return select_deploy8(arrs, surf, S, tree, seed)


def run(tree: str) -> int:
    import compliant_reread_r8 as CR
    t0 = time.time()
    train_fz = json.load(open(BD.SOURCE, encoding="utf-8"))
    TFM.assert_frozen_is_train(train_fz)
    pin = DF._gains_pinned(BD.DEPLOY_JSON, BD.SOURCE)
    if not (pin["train_pinned"] and pin["deploy_matches_sha256_file"]):
        raise SystemExit(f"[tac-retime] REFUSING: a gains file differs from its pinned sha256 ({pin})")
    deploy = json.load(open(BD.DEPLOY_JSON, encoding="utf-8"))
    bad = BD.check_deploy(deploy)
    if bad:
        raise SystemExit("[tac-retime] REFUSING: the deploy table fails AMENDMENT 1's checks: " + "; ".join(bad))
    tools = RT.tools_load()
    PR, ra = tools["PR"], tools["ra"]
    pr_path = Path(f"{ARCHIVE}/{tree}/PAIRED_READ_{tree}.json")
    pr_banked = json.load(open(pr_path, encoding="utf-8")) if pr_path.exists() else None
    tj_path = PKG / "raw" / "compliant_reread" / "trees" / f"{tree}.json"
    tj = json.load(open(tj_path, encoding="utf-8")) if tj_path.exists() else None
    fd_path = DEP / f"FORWARD_DEPLOY_{tree}.json"
    fd = json.load(open(fd_path, encoding="utf-8")) if fd_path.exists() else None
    fd8_path = DEP / f"FORWARD_DEPLOY8_{tree}.json"
    fd8 = json.load(open(fd8_path, encoding="utf-8")) if fd8_path.exists() else None
    out = {"tool": "tac_retime_r8.py", "tree": f"{ARCHIVE}/{tree}", "tree_name": tree, "tier": TIER, "estimator": RT.QUESTION, "n_boot": N_BOOT, "bootstrap_seed": 0, "rule": RULE,
           "metric": "refav1_arm._components TAC_traj_{lat,lon}_correct = plan label == GT label under refc_tactical.factor_from_kinematics(four_families.maneuver_kinematics(.)); higher is better",
           "gains": {"deploy_json": str(BD.DEPLOY_JSON), "deploy_sha256": pin["deploy_sha256"], "train_sha256": pin["train_sha256"],
                     "deploy_tables": {s: deploy["surfaces"][s]["RETIME_DEPLOY"] for s in SURFS}, "train_tables": {s: train_fz["surfaces"][s]["all"]["RETIME"] for s in SURFS}},
           "banked_sources": {"PAIRED_READ": str(pr_path) if pr_banked else None, "compliant_tree_json": str(tj_path) if tj else None, "FORWARD_DEPLOY": str(fd_path) if fd else None, "FORWARD_DEPLOY8": str(fd8_path) if fd8 else None},
           "surfaces": {}, "controls": {}}
    k1, k2, k3, k4, k5, k6, k7 = {}, {}, {}, {}, {}, {}, {}
    clause_cells, beside_cells, first, d8_status, d8_info = {}, {}, {}, {}, {}
    for seed in SEEDS:
        ctx = CR.prepare_seed(tree, seed, tools, tools["rc"])
        for surf in SURFS:
            S = RT.surface_inputs(tools, ctx, surf, first, tree)
            gains = DF.surface_arm_gains(deploy, train_fz, surf)
            X, eid, dt, comp = S["X"], np.asarray(S["eid"]), S["dt"], np.asarray(S["mask"], bool)
            tag = f"{surf}|s{seed}"
            D8, d8_why = load_deploy8_surface(tree, seed, surf, S)
            d8_status[tag] = d8_why
            # ---- K3 analytic: GT vs GT
            tg, _ = tac_direct(X["G"], X["G"], dt)
            k3[tag] = {k: {"n": int(len(v)), "all_exactly_1": bool(np.all(v == 1.0))} for k, v in tg.items()}
            # ---- K1 all-window banked known values (os / refcv7 means and the os - refcv7 cell)
            Cos, Cr = ra._components(X["O"], X["G"], dt), ra._components(X["R"], X["G"], dt)
            tos, _ = tac_direct(X["O"], X["G"], dt)
            trf, _ = tac_direct(X["R"], X["G"], dt)
            k5[f"{tag}|os"] = all(np.array_equal(tos[k], Cos[k]) for k in KEYS)
            k5[f"{tag}|refcv7"] = all(np.array_equal(trf[k], Cr[k]) for k in KEYS)
            if pr_banked is not None:
                bsd = pr_banked["surfaces"][surf]["seeds"][str(seed)]
                for k in KEYS:
                    mine_os, mine_r = float(np.nanmean(tos[k])), float(np.nanmean(trf[k]))
                    b_os, b_r = bsd["means"]["os"][k], bsd["means"][ctx["refarm"]][k]
                    cell = PR.paired_cell(tos[k], trf[k], eid, N_BOOT)
                    bc = bsd["rows"][k]["os_minus_refcv7_50400"]
                    k1[f"{tag}|{k}"] = {"os_mean_mine": mine_os, "os_mean_banked": b_os, "refcv7_mean_mine": mine_r, "refcv7_mean_banked": b_r,
                                        "means_exact": bool(mine_os == b_os and mine_r == b_r), "n_windows_mine": int(len(tos[k])), "n_windows_banked": bsd["n_windows"],
                                        "cell_mine": RT.slim(cell), "cell_banked": {kk: bc.get(kk) for kk in CELL_EXACT_KEYS}, "cell": cell_eq(cell, bc)}
            else:
                k1[f"{tag}"] = {"status": "ABSENT", "reason": f"{pr_path} does not exist"}
            surf_rec = out["surfaces"].setdefault(surf, {}).setdefault(f"s{seed}", {})
            for set_name in SETS:
                sel = comp if set_name == "compliant" else np.ones(len(eid), bool)
                Xs = {k: v[sel] for k, v in X.items()}
                e = eid[sel]
                P_dep = RT.retime_apply(Xs["O"], Xs["HP"], gains[DF.DEPLOY_ARM])
                P_trn = RT.retime_apply(Xs["O"], Xs["HP"], gains[DF.TRAIN_ARM])
                P_g1 = RT.retime_apply(Xs["O"], Xs["HP"], gains["CONST_G1"])
                plans = {"os": Xs["O"], "RETIME_DEPLOY": P_dep, "RETIME_TRAINFIT": P_trn, "refcv7_50400": Xs["R"], "ha0_ext": Xs["HX"], "ha0_ext_pose": Xs["HP"]}
                if D8 is not None:
                    plans = {"os": plans["os"], SCORED_ARM: D8[sel], **{k: v for k, v in plans.items() if k != "os"}}
                    dd = np.linalg.norm(D8[sel].astype(np.float64) - P_dep.astype(np.float64), axis=-1)
                    d8_info[f"{tag}|{set_name}"] = {"n_windows": int(sel.sum()), "mean_abs_diff_vs_per_surface_m": float(dd.mean()), "max_abs_diff_vs_per_surface_m": float(dd.max())}
                tac, lab = {}, {}
                for a, P in plans.items():
                    tac[a], lab[a] = tac_direct(P, Xs["G"], dt)
                # K5 on the re-timed plan too
                Cdep = ra._components(P_dep, Xs["G"], dt)
                k5[f"{tag}|{set_name}|RETIME_DEPLOY"] = all(np.array_equal(tac["RETIME_DEPLOY"][k], Cdep[k]) for k in KEYS)
                # K4 CONST_G1 == os
                tg1, _ = tac_direct(P_g1, Xs["G"], dt)
                k4[f"{tag}|{set_name}"] = {"P_bit_equal_os": bool(np.array_equal(P_g1, Xs["O"])),
                                           "cells": {k: RT.slim(PR.paired_cell(tg1[k], tac["os"][k], e, N_BOOT)) for k in KEYS}}
                rec = {"n_windows": int(sel.sum()), "n_episodes": int(len(np.unique(e))), "means": {a: {k: float(np.nanmean(v[k])) for k in KEYS} for a, v in tac.items()}, "cells": {}}
                arms_here = [a for a in (SCORED_ARM, "os", "RETIME_DEPLOY", "RETIME_TRAINFIT") if a in plans]
                for arm in arms_here:
                    rec["cells"][f"{arm}_minus_refcv7_50400"] = {k: RT.slim(PR.paired_cell(tac[arm][k], tac["refcv7_50400"][k], e, N_BOOT)) for k in KEYS}
                for arm in [a for a in (SCORED_ARM, "RETIME_DEPLOY", "RETIME_TRAINFIT") if a in plans]:
                    rec["cells"][f"{arm}_minus_os"] = {k: RT.slim(PR.paired_cell(tac[arm][k], tac["os"][k], e, N_BOOT)) for k in KEYS}
                rec["label_changed_vs_os"] = {**({SCORED_ARM: {"lat": float(np.mean(lab[SCORED_ARM][0] != lab["os"][0])), "lon": float(np.mean(lab[SCORED_ARM][1] != lab["os"][1]))}} if SCORED_ARM in lab else {}),
                                              "RETIME_DEPLOY": {"lat": float(np.mean(lab["RETIME_DEPLOY"][0] != lab["os"][0])), "lon": float(np.mean(lab["RETIME_DEPLOY"][1] != lab["os"][1]))},
                                              "RETIME_TRAINFIT": {"lat": float(np.mean(lab["RETIME_TRAINFIT"][0] != lab["os"][0])), "lon": float(np.mean(lab["RETIME_TRAINFIT"][1] != lab["os"][1]))}}
                surf_rec[set_name] = rec
                if set_name == "compliant":
                    for k in KEYS:
                        beside_cells[f"{surf}|{k}|s{seed}"] = rec["cells"]["RETIME_DEPLOY_minus_refcv7_50400"][k]
                        if SCORED_ARM in plans:
                            clause_cells[f"{surf}|{k}|s{seed}"] = rec["cells"][f"{SCORED_ARM}_minus_refcv7_50400"][k]
                    # K2 compliant banked cell
                    if tj is not None:
                        bt = tj["seeds"][str(seed)]["surfaces"][surf]["compliant"]["uncapped"]["rows"]
                        for k in KEYS:
                            mine = rec["cells"]["os_minus_refcv7_50400"][k]
                            k2[f"{tag}|{k}"] = {"cell_mine": mine, "cell_banked": {kk: bt[k]["os_minus_refcv7_50400"].get(kk) for kk in CELL_EXACT_KEYS},
                                                "cell": cell_eq(mine, bt[k]["os_minus_refcv7_50400"])}
                    if fd8 is not None and SCORED_ARM in plans:
                        arm8 = fd8["seeds"][str(seed)]["surfaces"][surf]["sets"]["compliant"]["arms"][SCORED_ARM]["NR_minus_refcv7"]
                        for k in KEYS:
                            k7[f"{tag}|{k}"] = {"cell": cell_eq(rec["cells"][f"{SCORED_ARM}_minus_refcv7_50400"][k], arm8[k])}
                    # K6 determinism cross-check against FORWARD_DEPLOY
                    if fd is not None:
                        arm = fd["seeds"][str(seed)]["surfaces"][surf]["sets"]["compliant"]["arms"][DF.DEPLOY_ARM]["NR_minus_refcv7"]
                        for k in KEYS:
                            mine = rec["cells"]["RETIME_DEPLOY_minus_refcv7_50400"][k]
                            k6[f"{tag}|{k}"] = {"cell": cell_eq(mine, arm[k])}
        del ctx
    out["clause5"] = clause5(clause_cells)
    out["clause5"]["arm"] = SCORED_ARM
    out["clause5"]["cells"] = {k: {"cell": v, "separated_worse": separated_worse(v)} for k, v in clause_cells.items()}
    out["clause5"]["deploy8_load_status"] = d8_status
    out["clause5_beside_per_surface_RETIME_DEPLOY"] = {**clause5(beside_cells), "arm": "RETIME_DEPLOY",
                                                        "cells": {k: {"cell": v, "separated_worse": separated_worse(v)} for k, v in beside_cells.items()}}
    out["deploy8_vs_per_surface_INFORMATIONAL"] = {"note": "Euclidean point distance between the 8-slot call's outputs and the per-surface call's, same windows; AMENDMENT 2 quotes (INHERITED, not re-verified, other definition possible) "
                                                            "mean 0.0006 m / max 0.77 m on S2 and mean 0.0012 m / max 0.175 m on S6 at EMA-30k", "cells": d8_info}
    ctl = {"K1_os_reproduces_banked_PAIRED_READ_all_windows": bool(k1 and all("cell" in v and v["means_exact"] and v["cell"]["exact"] and v["n_windows_mine"] == v["n_windows_banked"] for v in k1.values())),
           "K2_os_reproduces_banked_compliant_tree_cell": (None if tj is None else bool(k2 and all(v["cell"]["exact"] for v in k2.values()))),
           "K3_gt_vs_gt_exactly_1": bool(k3 and all(x["all_exactly_1"] for v in k3.values() for x in v.values())),
           "K4_const_g1_equals_os": bool(k4 and all(v["P_bit_equal_os"] and all(c["delta"] == 0.0 and c["lo"] == 0.0 and c["hi"] == 0.0 for c in v["cells"].values()) for v in k4.values())),
           "K5_direct_labels_equal_components": bool(k5 and all(k5.values())),
           "K6_crosscheck_vs_FORWARD_DEPLOY_determinism_only": (None if fd is None else bool(k6 and all(v["cell"]["exact"] for v in k6.values()))),
           "K7_crosscheck_vs_FORWARD_DEPLOY8_determinism_only": (None if fd8 is None or not k7 else bool(all(v["cell"]["exact"] for v in k7.values())))}
    ctl["ALL_PASS"] = bool(ctl["K1_os_reproduces_banked_PAIRED_READ_all_windows"] and ctl["K2_os_reproduces_banked_compliant_tree_cell"] in (True, None) and ctl["K3_gt_vs_gt_exactly_1"]
                           and ctl["K4_const_g1_equals_os"] and ctl["K5_direct_labels_equal_components"] and ctl["K6_crosscheck_vs_FORWARD_DEPLOY_determinism_only"] in (True, None)
                           and ctl["K7_crosscheck_vs_FORWARD_DEPLOY8_determinism_only"] in (True, None))
    out["controls"] = {**ctl, "K1_detail": k1, "K2_detail": k2, "K3_detail": k3, "K4_detail": k4, "K5_detail": k5, "K6_detail": k6, "K7_detail": k7,
                       "note": "K2 / K6 / K7 are None (not applicable) when the compliant tree json / FORWARD_DEPLOY / FORWARD_DEPLOY8 file (or the DEPLOY8 npz) does not exist; K1 is required"}
    out["wall_s"] = round(time.time() - t0, 1)
    out["generated"] = time.strftime("%FT%T")
    out["script_sha256"] = TFM.sha256_file(__file__)
    jp = DEP / f"TAC_{tree}.json"
    RT.jdump(out, jp)
    chk = json.load(open(jp, encoding="utf-8"))                       # assert on the ARTIFACT
    assert chk["clause5"]["status"] in ("PASS", "FAIL", "INCOMPLETE") and chk["clause5"]["arm"] == SCORED_ARM
    assert len(chk["clause5_beside_per_surface_RETIME_DEPLOY"]["cells"]) == 8
    (DEP / f"TAC_{tree}.md").write_text(render(chk), encoding="utf-8")
    print(f"WROTE {jp} clause5={chk['clause5']['status']} controls_ALL_PASS={chk['controls']['ALL_PASS']} wall={out['wall_s']}s", flush=True)
    print(f"ZZTACOKZZ {tree}", flush=True)
    return 0 if chk["controls"]["ALL_PASS"] else 4


def render(j: dict) -> str:
    A = SCORED_ARM
    L = [f"# TAC_{j['tree_name']} -- TACTICAL read of the RE-TIMED plan (SHIP-R8 clause 5, scored arm {A})", "",
         f"tier: {j['tier']}", f"estimator: paired episode-cluster bootstrap, B {j['n_boot']}, bootstrap seed {j['bootstrap_seed']}; models and gains FIXED ('another draw of EPISODES' only; inference seeds 0 and 1 are the replicate).",
         f"gains: deploy sha256 `{j['gains']['deploy_sha256']}` (from the frozen TRAIN gains sha256 `{j['gains']['train_sha256']}`); script sha256 `{j['script_sha256']}`.", "", f"**{j['rule']}**", "",
         f"## clause 5 on this tree: **{j['clause5']['status']}**  (separated-worse cells: {j['clause5']['separated_worse_cells'] or 'none'}; missing: {j['clause5']['missing_or_undetermined'] or 'none'})", ""]
    if j["clause5"]["missing_or_undetermined"]:
        L += ["DEPLOY8 load status: " + "; ".join(f"{k}: {v}" for k, v in j["clause5"]["deploy8_load_status"].items()), ""]
    L += [f"Compliant windows. Clause-5 cell = {A} - refcv7-50,400 (accuracy, higher is better; worse+separated = CI upper < 0).", "",
          f"| surface | row | seed | n win / eps | os | {A} | per-surface DEPLOY | TRAINFIT | refcv7-50,400 | {A} - refcv7 (delta [lo, hi]) | separated worse |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for surf in SURFS:
        for k in KEYS:
            for sd in SEEDS:
                r = j["surfaces"][surf][f"s{sd}"]["compliant"]
                m = r["means"]
                c = r["cells"].get(f"{A}_minus_refcv7_50400", {}).get(k)
                a8 = f"{m[A][k]:.4f}" if A in m else "n/a"
                L.append(f"| {surf} | {k} | {sd} | {r['n_windows']} / {r['n_episodes']} | {m['os'][k]:.4f} | {a8} | {m['RETIME_DEPLOY'][k]:.4f} | {m['RETIME_TRAINFIT'][k]:.4f} | {m['refcv7_50400'][k]:.4f} | {fmt(c)} | {separated_worse(c)} |")
    b5 = j.get("clause5_beside_per_surface_RETIME_DEPLOY", {})
    L += ["", f"## beside (outside clause 5): the per-surface form of Amendment 1 on the same cells reads **{b5.get('status')}** (separated worse: {b5.get('separated_worse_cells') or 'none'})", "",
          f"| surface | row | seed | {A} - os | per-surface DEPLOY - os | TRAINFIT - os | os - refcv7 | per-surface DEPLOY - refcv7 | label changed vs os, {A} (lat / lon) |", "|---|---|---|---|---|---|---|---|---|"]
    for surf in SURFS:
        for k in KEYS:
            for sd in SEEDS:
                r = j["surfaces"][surf][f"s{sd}"]["compliant"]
                ch = r["label_changed_vs_os"].get(A) or {"lat": float("nan"), "lon": float("nan")}
                L.append(f"| {surf} | {k} | {sd} | {fmt(r['cells'].get(f'{A}_minus_os', {}).get(k))} | {fmt(r['cells']['RETIME_DEPLOY_minus_os'][k])} | {fmt(r['cells']['RETIME_TRAINFIT_minus_os'][k])} | "
                         f"{fmt(r['cells']['os_minus_refcv7_50400'][k])} | {fmt(r['cells']['RETIME_DEPLOY_minus_refcv7_50400'][k])} | {ch['lat']:.4f} / {ch['lon']:.4f} |")
    inf = j.get("deploy8_vs_per_surface_INFORMATIONAL", {}).get("cells", {})
    if inf:
        L += ["", "### informational: 8-slot call vs per-surface call, Euclidean point distance (m), compliant / all windows", ""]
        L += [f"* {k}: mean {v['mean_abs_diff_vs_per_surface_m']:.5f}, max {v['max_abs_diff_vs_per_surface_m']:.4f} (n {v['n_windows']})" for k, v in inf.items()]
    c = j["controls"]
    L += ["", f"## controls (ALL_PASS = **{c['ALL_PASS']}**)", "",
          f"* K1 unmodified os reproduces the banked PAIRED_READ (all windows, exact): **{c['K1_os_reproduces_banked_PAIRED_READ_all_windows']}**",
          f"* K2 unmodified os - refcv7 reproduces the banked compliant-tree cell: **{c['K2_os_reproduces_banked_compliant_tree_cell']}**",
          f"* K3 GT vs GT reads exactly 1.0 (analytic): **{c['K3_gt_vs_gt_exactly_1']}**", f"* K4 CONST_G1 == os (bit-equal plans, zero cell): **{c['K4_const_g1_equals_os']}**",
          f"* K5 directly recomputed labels == refav1_arm._components TAC arrays: **{c['K5_direct_labels_equal_components']}**",
          f"* K6 cross-check vs FORWARD_DEPLOY (per-surface) NR cells (determinism only, not correctness): **{c['K6_crosscheck_vs_FORWARD_DEPLOY_determinism_only']}**",
          f"* K7 cross-check vs FORWARD_DEPLOY8 NR cells for {A} (determinism only, not correctness): **{c['K7_crosscheck_vs_FORWARD_DEPLOY8_determinism_only']}**", "",
          "### K1 literals (os, all windows: mine vs the banked PAIRED_READ `means`)", "", "| surface | seed | row | os mine | os banked | refcv7 mine | refcv7 banked | n | exact |", "|---|---|---|---|---|---|---|---|---|"]
    for key, v in c["K1_detail"].items():
        if "os_mean_mine" in v:
            s_, sd, k = key.split("|")
            L.append(f"| {s_} | {sd} | {k} | {v['os_mean_mine']!r} | {v['os_mean_banked']!r} | {v['refcv7_mean_mine']!r} | {v['refcv7_mean_banked']!r} | {v['n_windows_mine']} | {v['means_exact'] and v['cell']['exact']} |")
    L += ["", "Single training seed; T1 open loop; EMA-30k reads are ORIENTATION ONLY unless the tree is step48852ema_unc.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    return run(ap.parse_args().tree)


if __name__ == "__main__":
    sys.exit(main())
