#!/usr/bin/env python3
"""ORC - ORC-0 per seed on the LONGITUDINAL / LATERAL / TACTICAL(plan-derived) families and DISTANCE KEEPING, at T0 and T1-kin.

Driver only: every statistic comes from the programme's own instruments at commit 07dcb513 (a `git archive` under E:/Projects/TanitAD-artifacts/ro/tree):
  * `taniteval.v8f_beta_eval.family_windows` + `paired_row`  (per-window family components; paired episode-cluster bootstrap, as v8f_kscore / oacc use them)
  * `v8f_dk_read.read_dk`                                     (distance keeping: headway / time gap / TTC / LEAD-VIOL; its identity control runs inside)
The TACTICAL heads + STRATEGIC cells at T0 come from the unchanged CLI `stack/scripts/v8f_paired_families.py` (run separately; this file does not touch them).

Tracks: T0 = `pred` of box 2's standard eval (ages cycle), the R-T0 source. T1-kin = `cl` of the dev-box `--limit causal` dump (the R-TLAT / R-ADE source).
Controls written to the output: pipeline identity (ORC vs ORC = exactly 0), reproduction of the reader's R-T0 / R-ADE point values, same-grid checks.
CUDA is never used.
"""
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

TREE = "E:/Projects/TanitAD-artifacts/ro/tree"
PKG = TREE + "/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
ART = "E:/Projects/TanitAD-artifacts"
OUT = os.environ.get("ORCF_OUT", "C:/Users/Admin/results_orc/ORC_FAMILIES.json")
NB = int(os.environ.get("ORCF_NBOOT", "2000"))
for p in (TREE + "/stack", TREE + "/taniteval", TREE + "/stack/scripts"):
    sys.path.insert(0, p)
import torch  # noqa: E402

torch.set_num_threads(8)
from taniteval import v8f_beta_eval as BE  # noqa: E402
from taniteval import ci as CI  # noqa: E402

spec = importlib.util.spec_from_file_location("v8f_dk_read", PKG + "/code/v8f_dk_read.py")
DK = importlib.util.module_from_spec(spec)
sys.modules["v8f_dk_read"] = DK
spec.loader.exec_module(DK)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


READ = json.load(open(ART + "/v8f-orecover-t1/read/ORC_READ.json", encoding="utf-8"))

# ---- ground truth / ids, exactly `v8f_dk_read.ground_truth` -------------------------------------------------------------------------------------------
gt_a = SimpleNamespace(gt_npz=None, eval_grid=PKG + "/raw/eval_grid_windows.json", eval_kin=PKG + "/raw/eval141_kin.npz", eval_join=PKG + "/raw/join_eval141.npz")
G = DK.ground_truth(gt_a)
GT, EID, CLIP, FRAME, V0 = G["gt"], np.asarray(G["eid"]), G["clip"], G["frame"], G["v0"]
N = GT.shape[0]
print("grid", N, len(np.unique(EID)), flush=True)
BLOCK = DK.load_block(PKG + "/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz")

SEEDS = (0, 1)
T0P = {("ORC", s): f"{ART}/v8f-orecover/M2b_s{s}_orc/eval_M2b-orc_s{s}.npz" for s in SEEDS}
T0P.update({("ORC0", s): f"{ART}/v8f-orecover/M2b_s{s}_orc0/eval_M2b-orc0_s{s}.npz" for s in SEEDS})
T1P = {(a, s): f"{ART}/v8f-orecover-t1/dump_{a}_s{s}_m10_causal_heads" for a in ("ORC", "ORC0") for s in SEEDS}

controls = {}


def load_t0(arm, s):
    z = np.load(T0P[(arm, s)])
    assert np.array_equal(z["eid"], EID), f"T0 {arm} s{s}: eid differs from the grid"
    return {"pred": z["pred"].astype(np.float64), "ade6": z["ade6"].astype(np.float64), "path": T0P[(arm, s)], "sha256": sha256(T0P[(arm, s)])}


def load_t1(arm, s):
    fs = sorted(glob.glob(T1P[(arm, s)] + "/ep*.npz"))
    parts = {k: [] for k in ("g", "cl", "v0", "eid", "clip_index", "ws")}
    for f in fs:
        with np.load(f) as z:
            for k in parts:
                parts[k].append(z[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["n_files"] = len(fs)
    return d


def fam(pred, gt):
    return BE.family_windows(np.asarray(pred, np.float64), np.asarray(gt, np.float64), 0.2)


FAM_ROWS = [  # (family, key in family_windows, label, unit, lower_is_better)
    ("LONGITUDINAL", "lon_speed_mae", "speed MAE", "m/s", True),
    ("LONGITUDINAL", "lon_accel_mae", "accel MAE", "m/s^2", True),
    ("LONGITUDINAL", "lon_along_mae", "along-track MAE", "m", True),
    ("LATERAL", "lat_heading_mae_deg", "heading MAE", "deg", True),
    ("LATERAL", "lat_curvature_mae_1pm", "curvature MAE", "1/m", True),
    ("LATERAL", "lat_yaw_rate_mae_degps", "yaw-rate MAE", "deg/s", True),
    ("LATERAL", "lat_cross_mae", "cross-track MAE", "m", True),
    ("TACTICAL (plan-derived)", "tac_lat_acc", "lateral decision agreement", "frac", False),
    ("TACTICAL (plan-derived)", "tac_lon_acc", "longitudinal decision agreement", "frac", False),
]


def paired_fam(fa, fb, eid):
    out = {}
    for famly, key, label, unit, lib in FAM_ROWS:
        r = BE.paired_row(fa[key], fb[key], eid, n_boot=NB, lower_is_better=lib)
        a, b = np.asarray(fa[key]), np.asarray(fb[key])
        ok = np.isfinite(a) & np.isfinite(b)
        r.update({"family": famly, "label": label, "unit": unit,
                  "mean_a_on_pairs": float(a[ok].mean()) if ok.any() else None, "mean_b_on_pairs": float(b[ok].mean()) if ok.any() else None,
                  "n_a_finite": int(np.isfinite(a).sum()), "n_b_finite": int(np.isfinite(b).sum())})
        out[key] = r
    return out


def ade6_pair(pa, pb, gt, eid):
    a = np.linalg.norm(np.asarray(pa, np.float64) - gt, axis=-1).mean(1)
    b = np.linalg.norm(np.asarray(pb, np.float64) - gt, axis=-1).mean(1)
    r = CI.paired_episode_cluster_bootstrap(a, b, eid, n_boot=NB, seed=0)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "n_windows": int(r["n_windows"]),
            "n_episodes": int(r["n_episodes"]), "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def dk_compact(rd, a, b):
    """The slice of read_dk's output the RESULT quotes (it computes more: by-speed strata, TTC stride sensitivity, ...)."""
    pr = rd["pairs"][f"{a}__minus__{b}"]
    arms = {nm: {"metrics": rd["arms"][nm]["metrics"], "lead_viol": rd["arms"][nm]["lead_viol"], "n_lead_windows": rd["arms"][nm]["n_lead_windows"]} for nm in (a, b)}
    return {"pair": pr, "arms": arms, "join": rd["join"], "identity_control": rd["identity_control"],
            "human_reference_metrics": rd["human_reference"]["metrics"], "lead_block": rd["lead_block"]}


res = {"schema": "orc-families-v1", "commit_tools": "07dcb513", "n_boot": NB, "n_windows": int(N), "n_episodes": int(len(np.unique(EID))),
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap (via v8f_beta_eval.paired_row / lead_metrics), B=%d, seed 0, over episodes; ORC - ORC-0 per seed on the same windows" % NB,
       "inputs": {"lead_block_sha256": sha256(PKG + "/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz"), "grid_sha256": sha256(gt_a.eval_grid), "kin_sha256": sha256(gt_a.eval_kin),
                  "join_sha256": sha256(gt_a.eval_join), "read_json_sha256": sha256(ART + "/v8f-orecover-t1/read/ORC_READ.json")},
       "T0": {}, "T1": {}}
t_start = time.time()

for tier in ("T0", "T1"):
    tr = {}
    for s in SEEDS:
        if tier == "T0":
            A, B = load_t0("ORC", s), load_t0("ORC0", s)
            pa, pb = A["pred"], B["pred"]
            ids = {"ORC": {"path": A["path"], "sha256": A["sha256"]}, "ORC0": {"path": B["path"], "sha256": B["sha256"]}}
            stored = {f"ORC_s{s}": A["ade6"], f"ORC0_s{s}": B["ade6"]}
            v0 = V0
        else:
            A, B = load_t1("ORC", s), load_t1("ORC0", s)
            chk = {}
            for nm, d in (("ORC", A), ("ORC0", B)):
                chk[nm] = {"n_files": d["n_files"], "n_windows": int(d["g"].shape[0]), "eid_equal_grid": bool(np.array_equal(d["eid"], EID)),
                           "g_max_abs_diff_vs_grid_gt_m": float(np.abs(d["g"].astype(np.float64) - GT).max()),
                           "v0_max_abs_diff_vs_grid": float(np.abs(d["v0"].astype(np.float64) - V0).max()),
                           "ws_times2_equals_grid_frame": bool(np.array_equal(2 * d["ws"].astype(np.int64), FRAME))}
            chk["ORC_vs_ORC0_same_windows"] = bool(np.array_equal(A["ws"], B["ws"]) and np.array_equal(A["clip_index"], B["clip_index"]) and np.array_equal(A["eid"], B["eid"]))
            controls[f"T1_grid_s{s}"] = chk
            pa, pb = A["cl"].astype(np.float64), B["cl"].astype(np.float64)
            ids = {"ORC": {"path": T1P[("ORC", s)]}, "ORC0": {"path": T1P[("ORC0", s)]}}
            stored = None
            v0 = A["v0"].astype(np.float64)                      # the dump's own v0, as the T1 families reader does; its max diff vs the grid's is in controls
        t0 = time.time()
        fa, fb = fam(pa, GT), fam(pb, GT)
        rows = paired_fam(fa, fb, EID)
        ident = paired_fam(fa, fa, EID)
        ident_zero = {k: bool(r.get("mean") == 0.0 and r.get("lo") == 0.0 and r.get("hi") == 0.0) for k, r in ident.items()}
        ade = ade6_pair(pa, pb, GT, EID)
        rd = DK.read_dk(preds={f"ORC_s{s}": pa, f"ORC0_s{s}": pb}, gt=GT, eid=EID, clip=CLIP, frame=FRAME, v0=v0, block=BLOCK, k=30, n_boot=NB, seed=0,
                        saved_ade6=stored, pairs=[(f"ORC_s{s}", f"ORC0_s{s}")])
        dkc = dk_compact(rd["read"], f"ORC_s{s}", f"ORC0_s{s}")
        tr[f"s{s}"] = {"ids": ids, "families_paired_ORC_minus_ORC0": rows, "identity_all_exactly_zero": ident_zero, "ade6_paired_ORC_minus_ORC0": ade,
                       "distance_keeping": dkc, "seconds": round(time.time() - t0, 1)}
        print(f"[{tier} s{s}] {time.time() - t0:.0f}s ade6 {ade['delta']:+.4f}", flush=True)
    res[tier] = tr

res["controls"] = controls
rep = {}
for s in SEEDS:
    rd0 = READ["rows"][f"s{s}"]
    out_s = {}
    for tier, key in (("T0", "R-T0"), ("T1", "R-ADE")):
        h, r = res[tier][f"s{s}"]["ade6_paired_ORC_minus_ORC0"], rd0[key]
        # the reader's `delta/lo/hi` are the estimator's own 4-dp values (same estimator, same seed 0, same windows): they must be equal here
        out_s[tier] = {"here": {k: h[k] for k in ("delta", "lo", "hi", "separated")}, "reader": {k: r[k] for k in ("delta", "lo", "hi", "separated")},
                       "round4_of_reader_point_equals_here_delta": bool(round(r["point"], 4) == h["delta"]),
                       "PASS": bool(all(h[k] == r[k] for k in ("delta", "lo", "hi", "separated")) and round(r["point"], 4) == h["delta"])}
    out_s["PASS"] = bool(out_s["T0"]["PASS"] and out_s["T1"]["PASS"])
    rep[f"s{s}"] = out_s
res["controls"]["reproduces_reader_ade6"] = rep
res["wall_s"] = round(time.time() - t_start, 1)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(clean(res), f, indent=1)
print("wrote", OUT, "sha256", sha256(OUT), flush=True)
