#!/usr/bin/env python3
"""gapdecomp_m.py -- H-GAPDECOMP-M-1: class M of the gap against P10, split by GT path shape and by speed (PREREG_FROZEN.md; zero training, CPU only).

Pure layer (numpy; unit-tested by test_gapdecomp_m.py): shape / speed subclasses, the three partitions of M, group decomposition over the LANDED draw stream, the verdict.
Heavy layer (main): reads the landed per-window arrays (dagland/gapdecomp, imported not copied), runs the known answers KA1-KA6, the split, the Beside block.

    python gapdecomp_m.py --signed F:/TanitAD-scratch/gapdecomp_m/signed --out raw/GAPDECOMP_M.json

Identity: a row is mean over its n valid windows of d = a - b; over the valid M windows Delta_M = sum_sub w_sub Delta_sub with w_sub = n_sub / n_M.
Bootstrap: the landed draw stream (taniteval.ci: default_rng(0), choice(uniq, n_ep) over the row's valid-pair episodes), subclass statistics recomputed on every draw.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
ART = "E:/Projects/TanitAD-artifacts"
LAND = f"{ART}/dagland/gapdecomp"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
PIN_LAND = {"gapdecomp.py": "824c177d8fc7b8ce3728c334bece1f2d628ea2ee684fefe8f62c2d3b7e310f7f",
            "raw/CLASS_SPLIT.json": "171165229262439a1e6749b6955aff9d3b68afb97a36eaf0a754b8da24676d3c",
            "raw/arrays/CTRL_s0.npz": "a23a03c7fb27b58691a83c30bc2d77fbb7c0484245796e940b188fc8d8d62df5",
            "raw/arrays/CTRL_s1.npz": "22cd3778e7e5fe8d35f50d4b38e96cb066fb86dce55ce9b9fac8fc30c6365cf3",
            "raw/arrays/SOUP_s0.npz": "a4f7b38fd3ab8d70d50ee275f417f724b92ae2def4a6562e5b988b33076eb64e",
            "raw/arrays/SOUP_s1.npz": "62d2a052ea700bdf1880d8aeb255ec93ec429bc20523db2b4e3de771483c5c1e"}

FLOOR_N, FLOOR_EP = 60, 20
W_MAX, SHARE_MIN, LO_MIN = 0.30, 0.5, 0.3
TURN_LO, TURN_HI = 10.0, 45.0
SPEED_EDGES = (3.0, 8.0, 15.0)
SHAPES = ("STRAIGHT", "GENTLE", "TURN")
SPEEDS = ("0.5-3", "3-8", "8-15", ">15")
VROWS = ("ADE6", "yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
BESIDE_ROWS = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
ARMS = ("CTRL", "SOUP")
ARM_NAME = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}
SEEDS = (0, 1)
TOL_KA2 = 1e-9
DT = 0.2


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_landed():
    spec = importlib.util.spec_from_file_location("gapdecomp_landed", f"{LAND}/gapdecomp.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["gapdecomp_landed"] = m
    spec.loader.exec_module(m)
    return m


def load_ff():
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", os.path.join(TREE, "stack").replace("/", os.sep))
    sys.path.insert(0, os.path.join(TREE, "taniteval"))
    from taniteval import four_families as FF  # noqa: E402
    return FF


GD = load_landed()


# ------------------------------------------------------------------------------------------------------------------------------ pure layer
def heading_change_deg(g, FF=None):
    """-> (dpsi_deg [N] signed net heading change over the pair-valid GT steps, n_pair_valid [N]).  The instrument's own heading and pair-valid mask (four_families._seq_geometry, dt 0.2)."""
    import torch
    FF = FF or load_ff()
    P = FF._seq_geometry(torch.as_tensor(np.asarray(g, np.float64)), DT)
    h, pv = P["heading"], P["pair_valid"]
    dh = h[:, 1:] - h[:, :-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    tot = torch.where(pv, dh, torch.zeros_like(dh)).sum(1)
    return np.degrees(tot.numpy()), pv.sum(1).numpy().astype(int)


def shape_index(dpsi_deg):
    """0 STRAIGHT |dpsi| < 10; 1 GENTLE 10 <= |dpsi| <= 45; 2 TURN |dpsi| > 45."""
    a = np.abs(np.asarray(dpsi_deg, np.float64))
    return np.where(a < TURN_LO, 0, np.where(a <= TURN_HI, 1, 2)).astype(int)


def speed_index(v0):
    """0 [0.5,3), 1 [3,8), 2 [8,15), 3 [15,inf) by v0 (m/s); a v0 below 0.5 is not in class M and raises."""
    v0 = np.asarray(v0, np.float64)
    if (v0 < 0.5).any():
        raise ValueError("speed_index is defined for class M only (v0 >= 0.5)")
    return np.searchsorted(np.asarray(SPEED_EDGES), v0, side="right").astype(int)


def _support(mask, ep):
    return int(mask.sum()), int(len(set(ep[mask])))


def build_partitions(shape_idx, speed_idx, eid, mmask, floor_n=FLOOR_N, floor_ep=FLOOR_EP):
    """The three partitions of M.  -> {pname: {"labels": [N] int (-1 outside M), "names": [...], "nameable": [...]}}; cross pools below-floor cells as PREREG_FROZEN says."""
    ep = np.asarray([str(x) for x in eid])
    mmask = np.asarray(mmask, bool)
    parts = {"shape": {"labels": np.where(mmask, np.asarray(shape_idx, int), -1), "names": [f"shape={s}" for s in SHAPES], "nameable": [True] * 3},
             "speed": {"labels": np.where(mmask, speed_idx, -1), "names": [f"speed={s}" for s in SPEEDS], "nameable": [True] * 4}}
    cell = np.asarray(shape_idx) * 4 + np.asarray(speed_idx)
    lab, names, nameable, pooled_all = -np.ones(len(ep), int), [], [], np.zeros(len(ep), bool)

    def at_floor(m):
        n, ne = _support(m, ep)
        return n >= floor_n and ne >= floor_ep
    for c in range(12):
        m = mmask & (cell == c)
        if at_floor(m):
            lab[m] = len(names)
            names.append(f"cross={SHAPES[c // 4]}/{SPEEDS[c % 4]}")
            nameable.append(True)
    for s in range(3):
        fails = [s * 4 + k for k in range(4) if not at_floor(mmask & (cell == s * 4 + k))]
        if not fails:
            continue
        m = mmask & np.isin(cell, fails)
        if at_floor(m):
            lab[m] = len(names)
            names.append(f"cross={SHAPES[s]}/other")
            nameable.append(True)
        else:
            pooled_all |= m
    if pooled_all.any():
        lab[pooled_all] = len(names)
        names.append("cross=ALL/other")
        nameable.append(False)
    parts["cross"] = {"labels": lab, "names": names, "nameable": nameable}
    return parts


def decompose_groups(a, b, eid, lab, names, nameable=None, n_boot=GD.N_BOOT, seed=GD.SEED):
    """Group split of Delta_M = mean(a - b) over the valid M windows ('masked' pool: the landed draws over the episodes holding a valid pair of the whole row).
    lab [N] int: group of each window, -1 outside M.  -> (dict, share_draws {name: [n_boot]}); floats unrounded."""
    a, b, lab = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(lab)
    if not (a.shape == b.shape == lab.shape == np.asarray(eid).shape):
        raise ValueError("a, b, lab, eid must be aligned")
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        raise ValueError("no valid pair")
    K = len(names)
    nameable = list(nameable) if nameable is not None else [True] * K
    ep = np.asarray([str(x) for x in eid])
    uniq = np.unique(ep[ok])
    n_ep = len(uniq)
    ei = np.searchsorted(uniq, ep)
    d = np.where(ok, a - b, 0.0)
    sel = ok & (lab >= 0)
    S, N, Nall = np.zeros((n_ep, K)), np.zeros((n_ep, K)), np.zeros(n_ep)
    np.add.at(S, (ei[sel], lab[sel]), d[sel])
    np.add.at(N, (ei[sel], lab[sel]), 1.0)
    np.add.at(Nall, ei[ok], 1.0)
    Sg, Ng = S.sum(0), N.sum(0)
    nM, SM, n_tot = float(Ng.sum()), float(Sg.sum()), float(Nall.sum())
    Mx = GD.cached_draws(uniq, n_boot, seed)
    Sb, Nb, Ntb = Mx @ S, Mx @ N, Mx @ Nall
    NMb, SMb = Nb.sum(1), Sb.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dM_b = np.where(NMb > 0, SMb / np.where(NMb > 0, NMb, 1.0), np.nan)
        dg_b = np.where(Nb > 0, Sb / np.where(Nb > 0, Nb, 1.0), np.nan)
        cM_b = SMb / Ntb
        cg_b = Sb / Ntb[:, None]
        sh_b = np.where(SMb[:, None] != 0, Sb / np.where(SMb != 0, SMb, 1.0)[:, None], np.nan)
    lo, hi, _ = GD._pct(dM_b)
    clo, chi, _ = GD._pct(cM_b)
    out = {"M": {"n": int(nM), "n_episodes": int((N.sum(1) > 0).sum()), "n_row_valid": int(n_tot), "delta": SM / nM, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0),
                 "contribution": SM / n_tot, "contribution_lo": clo, "contribution_hi": chi},
           "groups": {}, "n_boot": int(n_boot), "seed": int(seed)}
    draws = {}
    for i, nm in enumerate(names):
        dlo, dhi, dnan = GD._pct(dg_b[:, i])
        slo, shi, snan = GD._pct(sh_b[:, i])
        n_i, ne_i = int(Ng[i]), int((N[:, i] > 0).sum())
        w = float(Ng[i] / nM)
        sh = float(Sg[i] / SM) if SM != 0 else None
        out["groups"][nm] = {"n": n_i, "n_episodes": ne_i, "w": w, "delta": (float(Sg[i] / Ng[i]) if Ng[i] > 0 else None), "delta_lo": dlo, "delta_hi": dhi,
                             "contribution": float(Sg[i] / n_tot), "share": sh, "share_lo": slo, "share_hi": shi, "share_n_nan_draws": snan,
                             "enrichment": (sh / w) if (sh is not None and w > 0) else None, "support_ok": bool(n_i >= FLOOR_N and ne_i >= FLOOR_EP), "nameable": bool(nameable[i])}
        draws[nm] = sh_b[:, i]
    out["_share_sum_max_abs_dev_over_draws"] = float(np.nanmax(np.abs(sh_b.sum(1) - 1.0))) if np.isfinite(sh_b).any() else None
    out["_share_sum_point_abs_dev"] = float(abs(sum(g["share"] for g in out["groups"].values() if g["share"] is not None) - 1.0)) if SM != 0 else None
    out["_class_sum_identity_abs_dev"] = float(abs(sum(g["w"] * (g["delta"] or 0.0) for g in out["groups"].values()) - out["M"]["delta"]))
    return out, draws


def eligible(g):
    """A candidate: named by a nameable partition cell, supported on the row's valid windows, holding <= 30 % of the row's valid M windows."""
    return bool(g["nameable"] and g["support_ok"] and g["w"] <= W_MAX)


def qualifies(g):
    return bool(eligible(g) and g["share"] is not None and g["share_lo"] is not None and g["share"] >= SHARE_MIN and g["share_lo"] > LO_MIN)


def row_verdict(R, DRAWS=None):
    """R = {0: {"M": .., "groups": {key: ..}}, 1: ...} for one row, arm; keys 'partition|name'.  -> the PREREG_FROZEN verdict."""
    sep = {s: bool(R[s]["M"]["separated"]) for s in SEEDS}
    undefined = [f"s{s}" for s in SEEDS if not sep[s]]
    q = {s: {k for k, g in R[s]["groups"].items() if qualifies(g)} for s in SEEDS}
    both = sorted(q[0] & q[1], key=lambda k: -min(R[0]["groups"][k]["share"], R[1]["groups"][k]["share"]))
    if undefined:
        verdict, named = "NOT DEFINED at " + ", ".join(undefined), None
    elif both:
        verdict, named = "LOCALISED", both[0]
    else:
        verdict, named = "DIFFUSE", None
    elig = {s: [k for k, g in R[s]["groups"].items() if eligible(g)] for s in SEEDS}
    freq = {}
    if DRAWS is not None and named is not None:
        for s in SEEDS:
            M = np.stack([DRAWS[s][k] for k in elig[s]], axis=1)
            fin = np.isfinite(M).any(1)
            am = np.argmax(np.where(np.isfinite(M), M, -np.inf), axis=1)
            freq[f"s{s}"] = float((np.asarray(elig[s])[am][fin] == named).mean()) if fin.any() else None
    return {"verdict": verdict, "named": named, "qualifying_both_seeds": both, "qualifying": {f"s{s}": sorted(q[s]) for s in SEEDS}, "n_eligible": {f"s{s}": len(elig[s]) for s in SEEDS},
            "argmax_freq_of_named": freq, "M_separated": {f"s{s}": sep[s] for s in SEEDS}}


def pick_beside_set(verdicts, R_by_row_seed):
    """B: the subclasses named LOCALISED for any DAG-OACC row; else the leading candidate (labelled)."""
    named = []
    for lab in VROWS:
        n = verdicts[lab]["named"]
        if n and n not in named:
            named.append(n)
    if named:
        return named, "localised"
    best, best_v = None, -np.inf
    keys = set(R_by_row_seed[(BESIDE_ROWS[0], 0)]["groups"])
    for k in sorted(keys):
        if not all(eligible(R_by_row_seed[(lab, s)]["groups"][k]) for lab in BESIDE_ROWS for s in SEEDS):
            continue
        sh = [R_by_row_seed[(lab, s)]["groups"][k]["share"] for lab in BESIDE_ROWS for s in SEEDS]
        if any(x is None for x in sh):
            continue
        v = float(np.mean(sh))
        if v > best_v:
            best, best_v = k, v
    return ([best] if best else []), ("leading, NOT localised" if best else "none eligible")


def subclass_mask(parts, key):
    pn, nm = key.split("|", 1)
    return parts[pn]["labels"] == parts[pn]["names"].index(nm)


def zeroed_rows(GDm, z, mask, labels):
    """D2' arithmetic: Delta' of every one of the 19 rows with the windows of `mask` zeroed (n unchanged), the landed decompose with the subclass mapped to class L and the rest to M."""
    cls = np.where(mask, 0, 3).astype(np.int8)
    out = {}
    for i, lab in enumerate(labels):
        r = GDm.decompose(z[f"a_{i}"], z[f"b_{i}"], z["eid"], cls, GDm.POOL[lab])
        tz = r["trigger_zeroed"]
        out[lab] = {"delta_total": r["delta"], "delta_zeroed": tz["delta"], "lo": tz["lo"], "hi": tz["hi"], "separated": tz["separated"],
                    "separated_worse": GDm.sep_worse(lab, tz["delta"], tz["lo"], tz["hi"]), "separated_worse_total": GDm.sep_worse(lab, r["delta"], r["lo"], r["hi"])}
    return out


def plan_summary(sg, mask):
    """Window means over the subclass's windows of the signed per-window arrays (extract_signed.py); NaN windows dropped, counts reported."""
    def mean(x):
        v = np.asarray(x, np.float64)[mask]
        f = v[np.isfinite(v)]
        return (float(f.mean()) if len(f) else None), int(len(f))
    r = {}
    for who in ("x", "p10"):
        spd_p, n1 = mean(sg[f"spd_{who}"])
        spd_g, _ = mean(sg["spd_gt"])
        kp, n2 = mean(sg[f"kabs_{who}"])
        kg, _ = mean(sg[f"kabs_gt_{who}"])
        ke, _ = mean(sg[f"kerr_{who}"])
        r[who] = {"plan_speed": spd_p, "gt_speed": spd_g, "speed_err_plan_minus_gt": (spd_p - spd_g) if spd_p is not None and spd_g is not None else None,
                  "plan_abs_curv": kp, "gt_abs_curv": kg, "abs_curv_diff_plan_minus_gt": (kp - kg) if kp is not None and kg is not None else None, "signed_curv_err_plan_minus_gt": ke,
                  "n_windows_speed": n1, "n_windows_curv": n2}
    return r


# ------------------------------------------------------------------------------------------------------------------------------ heavy layer
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--signed", required=True, help="dir with signed_{CTRL,SOUP}_s{0,1}.npz from extract_signed.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frozen-dir", default=HERE)
    a_ = ap.parse_args(argv)
    t0 = time.time()
    fz = open(f"{a_.frozen_dir}/FROZEN_SHA.txt", encoding="utf-8").read()
    want = fz.split("sha256:")[1].split()[0]
    got = sha256_file(f"{a_.frozen_dir}/PREREG_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_FROZEN.md sha256 {got} != frozen {want}", file=sys.stderr)
        return 2
    for rel, sha in PIN_LAND.items():
        if sha256_file(f"{LAND}/{rel}") != sha:
            print(f"REFUSED: landed input {rel} has another sha256", file=sys.stderr)
            return 2
    landed = json.load(open(f"{LAND}/raw/CLASS_SPLIT.json", encoding="utf-8"))
    LABELS = GD.LABELS
    Z = {(arm, s): np.load(f"{LAND}/raw/arrays/{arm}_s{s}.npz") for arm in ARMS for s in SEEDS}
    z0 = Z[("CTRL", 0)]
    res = {"hypothesis": "H-GAPDECOMP-M-1", "tier": "T1-kin", "evidence_class": "MEASURED (ours; landed arrays dagland/gapdecomp/raw/arrays, shas in MANIFEST.txt); Beside D2' ESTIMATED",
           "estimator": f"paired episode-cluster bootstrap, B={GD.N_BOOT}, seed {GD.SEED}, the landed draw stream; answers another draw of episodes only",
           "prereg_sha256": got, "prereg_frozen_clock": [ln for ln in fz.splitlines() if ln.startswith("wall_clock:")][0].split(":", 1)[1].strip(), "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "landed_pins": PIN_LAND, "controls": {}, "partitions": {}, "rows": {}, "verdicts": {}, "beside": {}}
    ctl = res["controls"]
    # windows, class M, subclasses: identical across arms and seeds (the landed K1 already pinned the windows)
    for k, z in Z.items():
        if not all(np.array_equal(z[f], z0[f]) for f in ("eid", "v0", "g")):
            print(f"REFUSED: {k} windows differ from CTRL_s0", file=sys.stderr)
            return 2
    eid, v0, g = z0["eid"], z0["v0"], z0["g"]
    cls, masks = GD.classify(v0, g)
    mmask = masks["M"]
    FF = load_ff()
    dpsi, npv = heading_change_deg(g, FF)
    sh_i = shape_index(dpsi)
    sp_i = np.zeros(len(v0), int)
    sp_i[mmask] = speed_index(v0[mmask])
    parts = build_partitions(sh_i, sp_i, eid, mmask)
    ep = np.asarray([str(x) for x in eid])
    res["shape_geometry"] = {"windows_in_M_without_pair_valid_GT_step": int((mmask & (npv == 0)).sum()), "dpsi_deg_quantiles_M": {str(q): float(np.quantile(np.abs(dpsi[mmask]), q)) for q in (0.1, 0.5, 0.9, 0.99)}}
    # KA1
    ka1 = {"M_windows": int(mmask.sum()), "M_episodes": int(len(set(ep[mmask]))), "M_equals_landed": bool(int(mmask.sum()) == landed["arms"]["CTRL"]["s0"]["ADE6"]["classes"]["M"]["n"]), "partitions": {}}
    for pn, p in parts.items():
        lab = p["labels"]
        cnt = [(int((lab == i).sum()), len(set(ep[lab == i]))) for i in range(len(p["names"]))]
        cover = bool((lab[mmask] >= 0).all() and (lab[~mmask] == -1).all() and sum(c[0] for c in cnt) == int(mmask.sum()))
        named_ok = all(c[0] >= FLOOR_N and c[1] >= FLOOR_EP for c, nmb in zip(cnt, p["nameable"]) if nmb) if pn == "cross" else True   # shape / speed levels below the floor are simply not nameable per row
        ka1["partitions"][pn] = {"covers_M_once": cover, "named_cells_at_floor": bool(named_ok), "cells": {nm: {"windows": c[0], "episodes": c[1], "nameable": nmb} for nm, c, nmb in zip(p["names"], cnt, p["nameable"])}}
        res["partitions"][pn] = ka1["partitions"][pn]["cells"]
    ka1["PASS"] = bool(ka1["M_windows"] == 2032 and ka1["M_episodes"] == 140 and ka1["M_equals_landed"] and all(v["covers_M_once"] and v["named_cells_at_floor"] for v in ka1["partitions"].values()))
    ctl["KA1"] = ka1
    # the split of the 5 rows, with KA2 / KA3 / KA4 per arm, seed, row, partition
    R_all, D_all = {}, {}
    ka2, ka3, ka4 = {"PASS": True, "n_checked": 0, "max_abs_dev": 0.0, "failed": []}, {"PASS": True, "n_checked": 0, "failed": []}, {"PASS": True, "n_checked": 0, "max_dev_point": 0.0, "max_dev_draws": 0.0, "failed": []}
    for arm in ARMS:
        res["rows"][arm] = {}
        for s in SEEDS:
            z = Z[(arm, s)]
            res["rows"][arm][f"s{s}"] = {}
            for lab in VROWS:
                i = LABELS.index(lab)
                a, b = z[f"a_{i}"], z[f"b_{i}"]
                assert GD.POOL[lab] == "masked"
                lm = landed["arms"][arm][f"s{s}"][lab]["classes"]["M"]
                row = {"parts": {}}
                merged_groups, merged_draws, Mrow = {}, {}, None
                for pn, p in parts.items():
                    r, dr = decompose_groups(a, b, eid, p["labels"], p["names"], p["nameable"])
                    row["parts"][pn] = r
                    for nm, gg in r["groups"].items():
                        merged_groups[f"{pn}|{nm}"] = gg
                        merged_draws[f"{pn}|{nm}"] = dr[nm]
                    Mrow = r["M"]
                    # KA2 vs the landed class M
                    m = r["M"]
                    pairs = {"n": (m["n"], lm["n"]), "n_episodes": (m["n_episodes"], lm["n_episodes"]), "delta": (m["delta"], lm["delta"]), "lo": (m["lo"], lm["delta_lo"]), "hi": (m["hi"], lm["delta_hi"]),
                             "contribution": (m["contribution"], lm["contribution"]), "contribution_lo": (m["contribution_lo"], lm["contribution_lo"]), "contribution_hi": (m["contribution_hi"], lm["contribution_hi"])}
                    dev = max(abs(float(x) - float(y)) for x, y in pairs.values())
                    ka2["n_checked"] += 1
                    ka2["max_abs_dev"] = max(ka2["max_abs_dev"], dev)
                    if dev > TOL_KA2 or r["_class_sum_identity_abs_dev"] > TOL_KA2:
                        ka2["PASS"] = False
                        ka2["failed"].append(f"{arm}_s{s}|{lab}|{pn}")
                    # KA3 (X, X)
                    zr, _ = decompose_groups(a, a, eid, p["labels"], p["names"], p["nameable"])
                    z_ok = bool(zr["M"]["delta"] == 0.0 and zr["M"]["lo"] == 0.0 and zr["M"]["hi"] == 0.0 and all(gg["delta"] in (0.0, None) for gg in zr["groups"].values()))
                    ka3["n_checked"] += 1
                    if not z_ok:
                        ka3["PASS"] = False
                        ka3["failed"].append(f"{arm}_s{s}|{lab}|{pn}")
                    # KA4 shares sum to 1
                    pt, dd = r["_share_sum_point_abs_dev"], r["_share_sum_max_abs_dev_over_draws"]
                    ka4["n_checked"] += 1
                    ka4["max_dev_point"] = max(ka4["max_dev_point"], pt or 0.0)
                    ka4["max_dev_draws"] = max(ka4["max_dev_draws"], dd or 0.0)
                    if (pt is not None and pt > 1e-9) or (dd is not None and dd > 1e-9):
                        ka4["PASS"] = False
                        ka4["failed"].append(f"{arm}_s{s}|{lab}|{pn}")
                row["M"] = Mrow
                row["groups"] = merged_groups
                R_all[(arm, lab, s)], D_all[(arm, lab, s)] = row, merged_draws
                res["rows"][arm][f"s{s}"][lab] = {"M": Mrow, "parts": {pn: {"groups": row["parts"][pn]["groups"]} for pn in parts}}
    ctl["KA2"], ctl["KA3"], ctl["KA4"] = ka2, ka3, ka4
    # verdicts
    for arm in ARMS:
        res["verdicts"][arm] = {}
        for lab in VROWS:
            res["verdicts"][arm][lab] = row_verdict({s: R_all[(arm, lab, s)] for s in SEEDS}, {s: D_all[(arm, lab, s)] for s in SEEDS})
    # Beside: B, D2' and plan against GT
    vc = res["verdicts"]["CTRL"]
    Rbs = {(lab, s): R_all[("CTRL", lab, s)] for lab in VROWS for s in SEEDS}
    B, B_kind = pick_beside_set(vc, Rbs)
    res["beside"] = {"B": B, "B_kind": B_kind, "evidence_class_D2prime": "ESTIMATED (decomposition arithmetic, not a retrained model)", "D2prime": {}, "plan_vs_gt": {}}
    # KA6: an empty subclass reproduces each row's landed Delta and the landed D2_worse_rows
    ka6 = {"PASS": True, "n_checked": 0, "max_abs_dev_delta": 0.0, "failed": []}
    for arm in ARMS:
        for s in SEEDS:
            zr = zeroed_rows(GD, Z[(arm, s)], np.zeros(len(eid), bool), LABELS)
            worse_total = sorted(l for l in LABELS if zr[l]["separated_worse"])
            want_worse = sorted(landed["beside"][arm][f"s{s}"]["D2_worse_rows"])
            dev = max(abs(zr[l]["delta_zeroed"] - zr[l]["delta_total"]) for l in LABELS)
            ka6["n_checked"] += 1
            ka6["max_abs_dev_delta"] = max(ka6["max_abs_dev_delta"], dev)
            if dev > 1e-12 or worse_total != want_worse:
                ka6["PASS"] = False
                ka6["failed"].append(f"{arm}_s{s}: dev {dev}; worse {worse_total} vs landed {want_worse}")
    ctl["KA6"] = ka6
    for key in B:
        res["beside"]["D2prime"][key] = {}
        mask = subclass_mask(parts, key)
        for arm in ARMS:
            res["beside"]["D2prime"][key][arm] = {}
            for s in SEEDS:
                zr = zeroed_rows(GD, Z[(arm, s)], mask, LABELS)
                worse = [l for l in LABELS if zr[l]["separated_worse"]]
                res["beside"]["D2prime"][key][arm][f"s{s}"] = {"rows": zr, "D2prime_worse_rows": worse, "D2prime": not worse, "D2_total_worse_rows": [l for l in LABELS if zr[l]["separated_worse_total"]]}
            r0, r1 = res["beside"]["D2prime"][key][arm]["s0"], res["beside"]["D2prime"][key][arm]["s1"]
            res["beside"]["D2prime"][key][arm]["both_seeds_D2prime"] = bool(r0["D2prime"] and r1["D2prime"])
        res["beside"]["plan_vs_gt"][key] = {}
        for arm in ARMS:
            res["beside"]["plan_vs_gt"][key][arm] = {}
            for s in SEEDS:
                p = f"{a_.signed}/signed_{arm}_s{s}.npz"
                if not os.path.isfile(p):
                    res["beside"]["plan_vs_gt"][key][arm][f"s{s}"] = "UNVERIFIED: signed arrays missing"
                    continue
                sg = np.load(p)
                if not all(np.array_equal(sg[f], z0[f]) for f in ("eid", "v0")):
                    res["beside"]["plan_vs_gt"][key][arm][f"s{s}"] = "UNVERIFIED: signed arrays not aligned to the landed windows"
                    continue
                res["beside"]["plan_vs_gt"][key][arm][f"s{s}"] = plan_summary(sg, mask)
    res["controls_PASS"] = bool(all(v["PASS"] for v in ctl.values()))
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a_.out)), exist_ok=True)
    with open(a_.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("GAPDECOMP_M", "controls PASS" if res["controls_PASS"] else "CONTROLS FAILED", "->", a_.out)
    if not res["controls_PASS"]:
        for k, v in ctl.items():
            if not v["PASS"]:
                print("  FAILED", k, json.dumps(v, default=str)[:600])
    return 0 if res["controls_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
