#!/usr/bin/env python3
"""smooth_t1_read.py -- H-SMOOTH-T1-1: reads the five SMOOTH-T1 rollouts against PREREG_FROZEN.md (zero training, CPU only).

    python smooth_t1_read.py --out-dir F:/TanitAD-scratch/smooth_t1/out --raw <package>/raw [--skip-extract]

1. gate: PREREG_FROZEN.md must hash to FROZEN_SHA.txt.   2. extract the 19 per-window rows of every run (code/extract_smooth.py = the landed extractor + two path arguments).
3. known answers K1-K7 (PREREG_FROZEN.md), written to the JSON BEFORE any smoothed row.   4. rows X - P10, X - Y, Y - P10 by the landed ``gapdecomp.decompose`` (paired episode-cluster bootstrap B = 2,000, seed 0).
5. outcomes (DOMINANT > HARMS > MOVED > NO EFFECT) and the descriptive curvature-jitter block.  Y = the BANKED arrays of the H-GAPDECOMP-1 package; X = the extracted smoothed run.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
GD_DIR = f"{ART}/dagland/gapdecomp"                       # the landed H-GAPDECOMP-1 package: gapdecomp.py, raw/arrays (the banked Y arrays)
MANIFEST = "F:/TanitAD-scratch/soup_t1_out/runs_soup.json"
SOUP_READ, PIN_SOUP_READ = "F:/TanitAD-scratch/soup_t1_out/read/SOUP_READ.json", "dca10c07f5fabda9ddc9ea2725f425f53f1462bec8488d58ebc92ecd8d33de6a"
TREE = f"{ART}/dagt1_tree_b43c7a9a"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
P10_PRED = "F:/TanitAD-scratch/smooth_t1/p10_eval_pred6.npy"
RUNS = {"K1OFF": ("K1OFF_DAGOACC_s0", "CTRL", 0), "CTRL_s0": ("SMOOTH_DAGOACC_s0", "CTRL", 0), "CTRL_s1": ("SMOOTH_DAGOACC_s1", "CTRL", 1),
        "SOUP_s0": ("SMOOTH_SOUP_s0", "SOUP", 0), "SOUP_s1": ("SMOOTH_SOUP_s1", "SOUP", 1)}
ARM_NAME = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}
CRAWL = ("yaw-rate MAE", "curvature MAE", "L2 curvature MAE (GT>=5.2)", "M1 crawl share")
FAMILY = {"ADE6": "PRIMARY", "speed MAE": "LONGITUDINAL", "along-track MAE": "LONGITUDINAL", "accel MAE": "LONGITUDINAL", "DK-VIOL": "LONGITUDINAL", "DK-HW": "LONGITUDINAL", "DK-TG": "LONGITUDINAL",
          "heading MAE": "LATERAL", "yaw-rate MAE": "LATERAL", "curvature MAE": "LATERAL", "cross-track MAE": "LATERAL", "L1 lateral-accel MAE": "LATERAL", "L2 curvature MAE (GT>=5.2)": "LATERAL",
          "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL", "S-POS@30": "STRATEGIC", "S-POS@50": "STRATEGIC", "S-POS@80": "STRATEGIC", "M1 crawl share": "crawl"}
D1_POINT_M, ADE_GUARD_M = -0.2415, 0.0604
TOL_DUMP, TOL_ROW = 1e-4, 6e-5
BAND = (3.0, 8.0)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_gd():
    sys.path.insert(0, GD_DIR)
    import gapdecomp as GD
    return GD


# ------------------------------------------------------------------------------------------------ pure layer (unit-checkable)
def better(GD, lab, row):
    """Separated on the better side of the row's direction (TAC rows: higher is better; every other row: lower)."""
    sep = bool(row["lo"] > 0 or row["hi"] < 0)
    return bool(sep and ((row["delta"] > 0) if lab in GD.HIGHER_BETTER else (row["delta"] < 0)))


def worse(GD, lab, row):
    return GD.sep_worse(lab, row["delta"], row["lo"], row["hi"])


def read_a(GD, xp):
    """Read A (D1, D2, D3) of one seed from {label: row} of X - P10."""
    sb = {l: better(GD, l, r) for l, r in xp.items()}
    sw = {l: worse(GD, l, r) for l, r in xp.items()}
    d1 = bool(sb["ADE6"] and xp["ADE6"]["delta"] <= D1_POINT_M)
    d2 = not any(sw.values())
    fams = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
    d3 = {f: any(sb[l] for l in xp if FAMILY[l] == f) for f in fams}
    return {"D1": d1, "D2": d2, "D3": all(d3.values()), "D3_by_family": d3, "separated_worse_rows": sorted(l for l, v in sw.items() if v), "separated_better_rows": sorted(l for l, v in sb.items() if v),
            "DOMINANT": bool(d1 and d2 and all(d3.values()))}


def read_xy(GD, xy):
    """The X - Y reading of one seed."""
    crawl_sb = {l: better(GD, l, xy[l]) for l in CRAWL}
    non_crawl_sw = sorted(l for l in xy if l not in CRAWL and worse(GD, l, xy[l]))
    ade = xy["ADE6"]["delta"]
    return {"crawl_separated_better": crawl_sb, "crawl_all_separated_better": all(crawl_sb.values()), "ade6_delta": ade, "ade6_within_guard": bool(ade <= ADE_GUARD_M),
            "non_crawl_separated_worse": non_crawl_sw, "n_separated_worse_all": sum(worse(GD, l, r) for l, r in xy.items()),
            "MOVED_seed": bool(all(crawl_sb.values()) and ade <= ADE_GUARD_M), "HARMS_seed": bool(ade > ADE_GUARD_M or non_crawl_sw)}


def arm_outcome(per_seed):
    """per_seed {0: {'A': read_a, 'XY': read_xy}, 1: ...} -> the committed outcome (DOMINANT > HARMS > MOVED > NO EFFECT)."""
    dom = all(per_seed[s]["A"]["DOMINANT"] for s in (0, 1))
    harms = any(per_seed[s]["XY"]["HARMS_seed"] for s in (0, 1))
    moved = (not dom) and all(per_seed[s]["XY"]["MOVED_seed"] for s in (0, 1))
    out = "DOMINANT" if dom else ("HARMS" if harms else ("MOVED" if moved else "NO EFFECT"))
    return {"outcome": out, "DOMINANT": dom, "HARMS_condition": harms, "MOVED_condition": bool(moved or (not dom and all(per_seed[s]["XY"]["MOVED_seed"] for s in (0, 1)))),
            "moved_and_harms": bool(harms and not dom and all(per_seed[s]["XY"]["MOVED_seed"] for s in (0, 1)))}


# ------------------------------------------------------------------------------------------------ heavy layer
def extract(tag, arm, seed, out_dir, arr_dir):
    dst = f"{arr_dir}/{tag}.npz"
    if os.path.exists(dst):
        return dst
    os.makedirs(arr_dir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", f"{HERE}/extract_smooth.py", "--arm", arm, "--seed", str(seed), "--out", dst, "--dump-dir", f"{out_dir}/{tag}/dump_{tag}", "--ticks", f"{out_dir}/{tag}/ticks/ticks_{tag}.npz"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    open(f"{PK}/logs/extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    return dst


def load_dump(d):
    fs = sorted(glob.glob(os.path.join(d, "ep*.npz")))
    zs = [np.load(f) for f in fs]
    return len(fs), {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def cmp_arrays(a, b):
    """-> (bit_identical, max_abs_dev, n_keys, [differing keys]) over the keys of two dicts of arrays."""
    keys = sorted(set(a) & set(b))
    bad, mx = [], 0.0
    for k in keys:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append(k)
            continue
        if x.dtype.kind in "fc":
            same = np.array_equal(x, y, equal_nan=True)
            if not same:
                d = np.abs(np.where(np.isnan(x) & np.isnan(y), 0.0, x.astype(np.float64) - y.astype(np.float64)))
                mx = max(mx, float(np.nanmax(d)) if d.size else 0.0)
                bad.append(k)
        elif not np.array_equal(x, y):
            bad.append(k)
    return (not bad), mx, len(keys), bad, sorted(set(a) ^ set(b))


def curv_block(GD, d, tk_plan, tk_raw, tk_v0, torch, FF):
    """Descriptive: mean |kappa| and curvature roughness (mean squared first difference of kappa) in the v0 3-8 band of M, of the executed path and the issued plan."""
    cls, masks = GD.classify(d["v0"], d["g"])
    band = masks["M"] & (d["v0"] >= BAND[0]) & (d["v0"] < BAND[1])
    geo = lambda X: FF._seq_geometry(torch.as_tensor(np.asarray(X, np.float64)), 0.2)      # noqa: E731

    gt_geo = geo(d["g"])

    def stats(X, mask, ref=None):
        # READER_FIXES.md fix 5: an executed path is scored on the steps where it AND the GT are pair-valid (the mask of H-GAPDECOMP-M's extract_signed.curv_stats, the source of 0.1845 / 0.2621); an issued plan on its own pair-valid steps
        G = geo(X)
        k = G["curvature"]
        pv = G["pair_valid"] & ref["pair_valid"] if ref is not None else G["pair_valid"]
        kk = torch.where(pv, k.abs(), torch.zeros_like(k)).sum(1) / pv.sum(1).clamp_min(1)
        dk = (k[:, 1:] - k[:, :-1]) ** 2
        dv = (pv[:, 1:] & pv[:, :-1])
        rr = torch.where(dv, dk, torch.zeros_like(dk)).sum(1) / dv.sum(1).clamp_min(1)
        m = torch.as_tensor(mask)
        return {"n": int(m.sum()), "mean_abs_kappa": float(kk[m].mean()), "roughness_mean_sq_dkappa": float(rr[m].mean())}
    out = {"n_band_windows": int(band.sum()), "executed_x": stats(d["cl"], band, gt_geo), "executed_p10": stats(d["fl"], band, gt_geo), "executed_gt": stats(d["g"], band, gt_geo)}
    tb = (tk_v0 >= BAND[0]) & (tk_v0 < BAND[1])
    out["issued_plan_ticks_in_band"] = {"plan": stats(tk_plan, tb), **({"plan_raw": stats(tk_raw, tb)} if tk_raw is not None else {})}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    t_start = time.time()
    want = [ln.split()[2] for ln in open(f"{PK}/FROZEN_SHA.txt", encoding="utf-8") if ln.startswith("PREREG_FROZEN.md sha256")][0]
    got = sha256_file(f"{PK}/PREREG_FROZEN.md")
    if want != got:
        print(f"REFUSED: PREREG_FROZEN.md {got} != frozen {want}", file=sys.stderr)
        return 2
    if sha256_file(SOUP_READ) != PIN_SOUP_READ:
        print("REFUSED: SOUP_READ.json is not the pinned banked read", file=sys.stderr)
        return 2
    GD = load_gd()
    arr_dir = f"{a.raw}/arrays"
    os.makedirs(a.raw, exist_ok=True)
    if not a.skip_extract:
        for k, (tag, arm, seed) in RUNS.items():
            extract(tag, arm, seed, a.out_dir, arr_dir)
    man = json.load(open(MANIFEST, encoding="utf-8"))
    banked = json.load(open(SOUP_READ, encoding="utf-8"))["tables"]
    bk_tbl = {"CTRL": banked["CTRL_P10"], "SOUP": banked["SOUP_P10"]}
    res = {"hypothesis": "H-SMOOTH-T1-1", "tier": "T1-kin", "evidence_class": "MEASURED (ours; artifacts in MANIFEST.txt)", "prereg_sha256": got,
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0 (taniteval.ci draw stream, via gapdecomp.decompose); answers another draw of EPISODES only",
           "run_clock": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "controls": {}, "arms": {}, "descriptive": {}}
    ctl = res["controls"]
    # ---- K1: the --smooth off rerun of DAG-OACC s0 against the banked dump and ticks
    tag, arm, seed = RUNS["K1OFF"]
    ent = man["arms"]["CTRL"]["0"]["t1"]
    n1, dk = load_dump(f"{a.out_dir}/{tag}/dump_{tag}")
    n2, db = load_dump(ent["dump_dir"])
    same_d, dev_d, nk_d, bad_d, only_d = cmp_arrays(dk, db)
    tk, tb = dict(np.load(f"{a.out_dir}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True)), dict(np.load(ent["ticks"], allow_pickle=True))
    tk_cmp = {k: v for k, v in tk.items() if k not in ("tag", "ckpt")}
    tb_cmp = {k: v for k, v in tb.items() if k not in ("tag", "ckpt")}
    same_t, dev_t, nk_t, bad_t, only_t = cmp_arrays(tk_cmp, tb_cmp)
    done = json.load(open(f"{a.out_dir}/done_{tag}.txt", encoding="utf-8"))
    ctl["K1_off_reproduces_banked_DAGOACC_s0"] = {"dump_files": [n1, n2], "dump_bit_identical": same_d, "dump_max_abs_dev_m": dev_d, "dump_keys_compared": nk_d, "dump_differing_keys": bad_d, "dump_key_set_diff": only_d,
                                                  "ticks_bit_identical": same_t, "ticks_max_abs_dev": dev_t, "ticks_keys_compared": nk_t, "ticks_differing_keys": bad_t, "ticks_key_set_diff": only_t,
                                                  "threads": done.get("threads"), "PASS": bool((same_d and same_t) or (dev_d <= TOL_DUMP and dev_t <= TOL_DUMP and not only_d and not only_t))}
    # ---- arrays: Y banked, X smoothed, K1 off
    Z = {}
    for k, (tag, arm, seed) in RUNS.items():
        Z[k] = np.load(f"{arr_dir}/{tag}.npz", allow_pickle=True)
    Y = {(arm, s): np.load(f"{GD_DIR}/raw/arrays/{arm}_s{s}.npz", allow_pickle=True) for arm in ("CTRL", "SOUP") for s in (0, 1)}
    labels = tuple(str(x) for x in Y[("CTRL", 0)]["labels"])
    assert labels == GD.LABELS
    # K2: (X_off, Y) reads 0 on all 19 rows
    k2 = {}
    cls0, _m0 = GD.classify(Y[("CTRL", 0)]["v0"], Y[("CTRL", 0)]["g"])
    for i, lab in enumerate(labels):
        xa, ya = Z["K1OFF"][f"a_{i}"], Y[("CTRL", 0)][f"a_{i}"]
        m = np.isfinite(xa) & np.isfinite(ya)
        r0 = GD.decompose(xa, ya, Y[("CTRL", 0)]["eid"], cls0, GD.POOL[lab], a.n_boot)
        k2[lab] = {"same_nan_pattern": bool(np.array_equal(np.isfinite(xa), np.isfinite(ya))), "row_delta": r0["delta"], "row_lo": r0["lo"], "row_hi": r0["hi"], "row_separated": r0["separated"],
                   "per_window_max_abs_diff": float(np.abs(xa[m] - ya[m]).max()) if m.any() else 0.0}
    # READER_FIXES.md fix 3: the card reads "(X_off, Y) reads 0 on all 19 rows (|delta| <= 1e-4)", i.e. the ROW delta; the per-window max (the first draft's gate) is kept as information
    ctl["K2_Xoff_minus_Y_is_zero"] = {"rows": k2, "PASS": bool(all(v["same_nan_pattern"] and abs(v["row_delta"]) <= TOL_DUMP for v in k2.values())),
                                      "PASS_if_per_window_max_were_the_gate": bool(all(v["same_nan_pattern"] and v["per_window_max_abs_diff"] <= TOL_DUMP for v in k2.values())), "max_abs_row_delta": max(abs(v["row_delta"]) for v in k2.values())}
    # K3: Y - P10 reproduces the banked SOUP_READ rows; K4: arm against itself reads 0
    k3, k4 = {}, {}
    for (arm, s), z in Y.items():
        cls, masks = GD.classify(z["v0"], z["g"])
        for i, lab in enumerate(labels):
            r = GD.decompose(z[f"a_{i}"], z[f"b_{i}"], z["eid"], cls, GD.POOL[lab], a.n_boot)
            bk = bk_tbl[arm][f"s{s}"][lab]
            k3[f"{arm}_s{s}:{lab}"] = bool(abs(r["delta"] - bk["delta"]) <= TOL_ROW and abs(r["lo"] - bk["lo"]) <= TOL_ROW and abs(r["hi"] - bk["hi"]) <= TOL_ROW and r["separated"] == bk["separated"])
            zr = GD.decompose(z[f"a_{i}"], z[f"a_{i}"], z["eid"], cls, GD.POOL[lab], 200)
            k4[f"{arm}_s{s}:{lab}"] = bool(zr["delta"] == 0.0 and zr["lo"] == 0.0 and zr["hi"] == 0.0)
    ctl["K3_Y_minus_P10_reproduces_SOUP_READ"] = {"n_rows": len(k3), "n_pass": sum(k3.values()), "failed": [k for k, v in k3.items() if not v], "PASS": all(k3.values())}
    ctl["K4_self_minus_self_is_zero"] = {"n_rows": len(k4), "n_pass": sum(k4.values()), "PASS": all(k4.values())}
    # K5 / K7 on the smoothed ticks (needs scipy; the same operator as the tool, spelled inline)
    from scipy.signal import savgol_filter
    k5 = {}
    for k in ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"):
        tag, arm, seed = RUNS[k]
        t = dict(np.load(f"{a.out_dir}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True))
        eb = man["arms"][arm][str(seed)]["t1"]
        tbk = dict(np.load(eb["ticks"], allow_pickle=True))
        nb, dbk = load_dump(eb["dump_dir"])
        nx, dx = load_dump(f"{a.out_dir}/{tag}/dump_{tag}")
        rec_f64 = t["P"] + savgol_filter(t["plan_raw"] - t["P"], 7, 2, axis=1, mode="interp")
        rec = rec_f64.astype(np.float32).astype(np.float64)               # READER_FIXES.md fix 4: the tool returns the smoothed plan in the model's plan dtype (float32) -- mirror the cast
        k0, kb0 = t["tick"] == 0, tbk["tick"] == 0
        d_raw = float(np.abs(t["plan_raw"][k0] - tbk["plan"][kb0]).max())
        d_rec = float(np.abs(rec - t["plan"]).max())
        d_rec_uncast = float(np.abs(rec_f64 - t["plan"]).max())
        k5[k] = {"tick0_plan_raw_vs_Y_plan_max_abs_m": d_raw, "plan_vs_P_plus_SG_max_abs_m": d_rec, "plan_vs_P_plus_SG_float64_uncast_max_abs_m": d_rec_uncast, "max_abs_plan_value_m": float(np.abs(t["plan"]).max()), "S_changed_plan_max_abs_m": float(np.abs(t["plan"] - t["plan_raw"]).max()),
                 "floor_fl_equals_Y": bool(np.array_equal(dx["fl"], dbk["fl"])), "floor_fl_max_abs_dev_m": float(np.abs(dx["fl"] - dbk["fl"]).max()), "windows_equal_Y": bool(all(np.array_equal(dx[q], dbk[q]) for q in ("g", "v0", "eid", "ws", "clip_index"))), "smooth_field": str(t.get("smooth")),
                 "PASS": bool(d_raw <= TOL_DUMP and d_rec <= TOL_DUMP and np.array_equal(dx["fl"], dbk["fl"]) and float(np.abs(t["plan"] - t["plan_raw"]).max()) > 1e-3)}
    ctl["K5_smoothed_ticks_identities"] = {"runs": k5, "PASS": all(v["PASS"] for v in k5.values())}
    if os.path.exists(P10_PRED):
        p10 = np.load(P10_PRED).astype(np.float64)
        t = dict(np.load(f"{a.out_dir}/{RUNS['CTRL_s0'][0]}/ticks/ticks_{RUNS['CTRL_s0'][0]}.npz", allow_pickle=True))
        k0 = t["tick"] == 0
        o = np.argsort(t["win"][k0], kind="stable")
        ctl["K7_tick0_P_vs_p10_eval_pred6_descriptive"] = {"shape_P": list(t["P"][k0].shape), "shape_p10": list(p10.shape),
                                                         "max_abs_m": float(np.abs(t["P"][k0][o] - p10).max()) if t["P"][k0].shape == p10.shape else None}
    gate = {k: v["PASS"] for k, v in ctl.items() if isinstance(v, dict) and "PASS" in v}
    res["controls_gate_all_pass"] = bool(all(gate.values()))
    res["controls_gate"] = gate
    # READER_FIXES.md fix 6 (reporting only): the card's rule "a control that fails REFUSES the outcome" written into the JSON, so the outcome words below cannot be quoted without it
    res["outcome_status"] = "QUOTABLE (all controls pass)" if res["controls_gate_all_pass"] else "REFUSED by the card: controls failed " + ", ".join(sorted(k for k, v in gate.items() if not v)) + " -- the outcome words below are descriptive, not quotable"
    print("controls:", json.dumps(gate))
    # ---- the rows
    import torch
    sys.path.insert(0, f"{TREE}/taniteval")
    os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", f"{TREE}/stack".replace("/", os.sep))
    from taniteval import four_families as FF
    for arm in ("CTRL", "SOUP"):
        per_seed, rows_out = {}, {}
        for s in (0, 1):
            xk = f"{arm}_s{s}"
            zx, zy = Z[xk], Y[(arm, s)]
            # READER_FIXES.md fix 2: b is Y's P10 array on the 16 non-S-POS rows (the banked floor is passed to the extractor); on S-POS@k b is the CTRV of the run's OWN ticks, so it differs by construction
            assert all(np.array_equal(zx[q], zy[q]) for q in ("eid", "v0", "g", "ws", "clip_index")) and all(np.array_equal(zx[f"b_{i}"], zy[f"b_{i}"], equal_nan=True) for i, l in enumerate(labels) if not l.startswith("S-POS"))
            cls, masks = GD.classify(zy["v0"], zy["g"])
            tabs = {"XP": {}, "XY": {}, "YP": {}}
            for i, lab in enumerate(labels):
                pool = GD.POOL[lab]
                for nm, (u, v) in (("XP", (zx[f"a_{i}"], zx[f"b_{i}"])), ("XY", (zx[f"a_{i}"], zy[f"a_{i}"])), ("YP", (zy[f"a_{i}"], zy[f"b_{i}"]))):
                    r = GD.decompose(u, v, zy["eid"], cls, pool, a.n_boot)
                    tabs[nm][lab] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"], "n": f"{r['n_windows']}/{r['n_episodes_pool']}", "family": FAMILY[lab],
                                     "direction": "higher" if lab in GD.HIGHER_BETTER else "lower", "classes": {c: {"n": r["classes"][c]["n"], "delta": r["classes"][c]["delta"], "contribution": r["classes"][c]["contribution"]} for c in GD.CLASSES}}
            per_seed[s] = {"A": read_a(GD, tabs["XP"]), "XY": read_xy(GD, tabs["XY"])}
            rows_out[f"s{s}"] = tabs
            # descriptive curvature-jitter block
            tag = RUNS[xk][0]
            nx, dx = load_dump(f"{a.out_dir}/{tag}/dump_{tag}")
            tx = dict(np.load(f"{a.out_dir}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True))
            eb = man["arms"][arm][str(s)]["t1"]
            nb, dbk = load_dump(eb["dump_dir"])
            tbk = dict(np.load(eb["ticks"], allow_pickle=True))
            res["descriptive"][f"{arm}_s{s}"] = {"X_smoothed": curv_block(GD, dx, tx["plan"], tx["plan_raw"], tx["v0"], torch, FF), "Y_banked": curv_block(GD, dbk, tbk["plan"], None, tbk["v0"], torch, FF)}
        res["arms"][arm] = {"name": ARM_NAME[arm], "per_seed": {f"s{s}": per_seed[s] for s in (0, 1)}, "rows": rows_out, **arm_outcome(per_seed)}
        print(arm, res["arms"][arm]["outcome"], {f"s{s}": {"D1": per_seed[s]["A"]["D1"], "D2": per_seed[s]["A"]["D2"], "D3": per_seed[s]["A"]["D3"], "MOVED": per_seed[s]["XY"]["MOVED_seed"], "HARMS": per_seed[s]["XY"]["HARMS_seed"]} for s in (0, 1)})
    res["wall_s"] = round(time.time() - t_start, 1)
    out = f"{a.raw}/SMOOTH_T1.json"
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("READ OK ->", out, "sha256", sha256_file(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
