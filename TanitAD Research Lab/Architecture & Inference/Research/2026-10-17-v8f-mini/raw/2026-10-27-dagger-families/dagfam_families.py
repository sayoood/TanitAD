#!/usr/bin/env python3
"""DAG - DAG-0 (and DAG - M2b at T0) on the LONGITUDINAL / LATERAL / TACTICAL(plan-derived) families and DISTANCE KEEPING, at T0 and T1-kin.

This is `orc_families.py` (O-RECOVER, raw/2026-10-26-orecover-result/) RE-PARAMETRISED, nothing else: ORC -> DAG, ORC0 -> DAG0, ORC_READ.json -> DAG_READ.json,
and one added pair (DAG - M2b at T0). Every statistic comes from the programme's own instruments, unchanged:
  * `taniteval.v8f_beta_eval.family_windows` + `paired_row`  (per-window family components; paired episode-cluster bootstrap)
  * `v8f_dk_read.read_dk`                                     (distance keeping: headway / time gap / TTC / LEAD-VIOL; its identity control runs inside)
Tool tree: E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a (the DAG chain's own tree; instruments byte-equal to O-RECOVER's 07dcb513 tree modulo CRLF, checked).
Tracks: T0 = `pred` of box 2's standard eval (ages cycle), the R-T0 source. T1-kin = `cl` of the dev-box `--limit causal` dump (the R-TLAT / R-ADE source).
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

ART = "E:/Projects/TanitAD-artifacts"
TREE = ART + "/dagt1_tree_b43c7a9a"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
PKG = TREE + "/" + PK
LEAD = ART + "/ro/tree/" + PK + "/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz"   # only in O-RECOVER's tree; sha256 asserted below
M2B_DIR = ART + "/dagfam/ref/m2b"                                                    # raw/full/eval/eval_M2b_s{0,1}.npz exported from mirror tip 7d29172b
OUT = os.environ.get("DAGF_OUT", ART + "/dagfam/DAG_FAMILIES.json")
NB = int(os.environ.get("DAGF_NBOOT", "2000"))
os.environ["TANITEVAL_STACK_OVERRIDE"] = TREE + "/stack"
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


READ_PATH = ART + "/v8f-dagger-t1/read/DAG_READ.json"
READ = json.load(open(READ_PATH, encoding="utf-8"))
RUNS = json.load(open(ART + "/v8f-dagger-t1/runs.json", encoding="utf-8"))

gt_a = SimpleNamespace(gt_npz=None, eval_grid=PKG + "/raw/eval_grid_windows.json", eval_kin=PKG + "/raw/eval141_kin.npz", eval_join=PKG + "/raw/join_eval141.npz")
G = DK.ground_truth(gt_a)
GT, EID, CLIP, FRAME, V0 = G["gt"], np.asarray(G["eid"]), G["clip"], G["frame"], G["v0"]
N = GT.shape[0]
print("grid", N, len(np.unique(EID)), flush=True)
LEAD_SHA = sha256(LEAD)
assert LEAD_SHA.startswith("7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"[:20]), LEAD_SHA
BLOCK = DK.load_block(LEAD)

SEEDS = (0, 1)
ARM_KEY = {"DAG": "DAG", "DAG0": "DAG0"}
T0P = {(a, s): RUNS["arms"][ARM_KEY[a]][str(s)]["t0"]["npz"] for a in ARM_KEY for s in SEEDS}
T0P.update({("M2b", s): f"{M2B_DIR}/eval_M2b_s{s}.npz" for s in SEEDS})
T1P = {(a, s): RUNS["arms"][ARM_KEY[a]][str(s)]["t1"]["dump_dir"] for a in ARM_KEY for s in SEEDS}

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
    pr = rd["pairs"][f"{a}__minus__{b}"]
    arms = {nm: {"metrics": rd["arms"][nm]["metrics"], "lead_viol": rd["arms"][nm]["lead_viol"], "n_lead_windows": rd["arms"][nm]["n_lead_windows"]} for nm in (a, b)}
    return {"pair": pr, "arms": arms, "join": rd["join"], "identity_control": rd["identity_control"],
            "human_reference_metrics": rd["human_reference"]["metrics"], "lead_block": rd["lead_block"]}


def block(pa, pb, v0, la, lb, stored):
    fa, fb = fam(pa, GT), fam(pb, GT)
    rows = paired_fam(fa, fb, EID)
    ident = paired_fam(fa, fa, EID)
    ident_zero = {k: bool(r.get("mean") == 0.0 and r.get("lo") == 0.0 and r.get("hi") == 0.0) for k, r in ident.items()}
    ade = ade6_pair(pa, pb, GT, EID)
    rd = DK.read_dk(preds={la: pa, lb: pb}, gt=GT, eid=EID, clip=CLIP, frame=FRAME, v0=v0, block=BLOCK, k=30, n_boot=NB, seed=0, saved_ade6=stored, pairs=[(la, lb)])
    return {"families_paired_A_minus_B": rows, "identity_all_exactly_zero": ident_zero, "ade6_paired_A_minus_B": ade, "distance_keeping": dk_compact(rd["read"], la, lb)}


res = {"schema": "dag-families-v1", "tool_tree": TREE, "n_boot": NB, "n_windows": int(N), "n_episodes": int(len(np.unique(EID))),
       "estimator": "taniteval.ci.paired_episode_cluster_bootstrap (via v8f_beta_eval.paired_row / lead_metrics), B=%d, seed 0, over episodes; A - B per seed on the same windows" % NB,
       "inputs": {"lead_block_sha256": LEAD_SHA, "grid_sha256": sha256(gt_a.eval_grid), "kin_sha256": sha256(gt_a.eval_kin), "join_sha256": sha256(gt_a.eval_join),
                  "read_json_sha256": sha256(READ_PATH), "runs_json_sha256": sha256(ART + "/v8f-dagger-t1/runs.json")},
       "T0": {}, "T1": {}, "T0_vs_M2b": {}}
t_start = time.time()

# ---- DAG - DAG0, both tiers (exactly O-RECOVER's recipe)
for tier in ("T0", "T1"):
    tr = {}
    for s in SEEDS:
        t0 = time.time()
        if tier == "T0":
            A, B = load_t0("DAG", s), load_t0("DAG0", s)
            pa, pb, v0 = A["pred"], B["pred"], V0
            ids = {"DAG": {"path": A["path"], "sha256": A["sha256"]}, "DAG0": {"path": B["path"], "sha256": B["sha256"]}}
            stored = {f"DAG_s{s}": A["ade6"], f"DAG0_s{s}": B["ade6"]}
        else:
            A, B = load_t1("DAG", s), load_t1("DAG0", s)
            chk = {}
            for nm, d in (("DAG", A), ("DAG0", B)):
                chk[nm] = {"n_files": d["n_files"], "n_windows": int(d["g"].shape[0]), "eid_equal_grid": bool(np.array_equal(d["eid"], EID)),
                           "g_max_abs_diff_vs_grid_gt_m": float(np.abs(d["g"].astype(np.float64) - GT).max()),
                           "v0_max_abs_diff_vs_grid": float(np.abs(d["v0"].astype(np.float64) - V0).max()),
                           "ws_times2_equals_grid_frame": bool(np.array_equal(2 * d["ws"].astype(np.int64), FRAME))}
            chk["DAG_vs_DAG0_same_windows"] = bool(np.array_equal(A["ws"], B["ws"]) and np.array_equal(A["clip_index"], B["clip_index"]) and np.array_equal(A["eid"], B["eid"]))
            controls[f"T1_grid_s{s}"] = chk
            pa, pb = A["cl"].astype(np.float64), B["cl"].astype(np.float64)
            ids = {"DAG": {"path": T1P[("DAG", s)]}, "DAG0": {"path": T1P[("DAG0", s)]}}
            stored = None
            v0 = A["v0"].astype(np.float64)
        b = block(pa, pb, v0, f"DAG_s{s}", f"DAG0_s{s}", stored)
        b["ids"], b["seconds"] = ids, round(time.time() - t0, 1)
        tr[f"s{s}"] = b
        print(f"[{tier} s{s}] DAG-DAG0 {time.time() - t0:.0f}s ade6 {b['ade6_paired_A_minus_B']['delta']:+.4f}", flush=True)
    res[tier] = tr

# ---- DAG - M2b at T0 (M2b's banked box-2 eval; same standard eval, ages cycle)
for s in SEEDS:
    t0 = time.time()
    A, B = load_t0("DAG", s), load_t0("M2b", s)
    stored = {f"DAG_s{s}": A["ade6"], f"M2b_s{s}": B["ade6"]}
    b = block(A["pred"], B["pred"], V0, f"DAG_s{s}", f"M2b_s{s}", stored)
    b["ids"] = {"DAG": {"path": A["path"], "sha256": A["sha256"]}, "M2b": {"path": B["path"], "sha256": B["sha256"]}}
    b["seconds"] = round(time.time() - t0, 1)
    b["m2b_ade6_mean_vs_registry_s17"] = {"here": float(B["ade6"].mean()), "registry": {0: 2.1753, 1: 2.0546}[s], "equal_at_4dp": bool(round(float(B["ade6"].mean()), 4) == {0: 2.1753, 1: 2.0546}[s])}
    res["T0_vs_M2b"][f"s{s}"] = b
    print(f"[T0 s{s}] DAG-M2b {time.time() - t0:.0f}s ade6 {b['ade6_paired_A_minus_B']['delta']:+.4f}", flush=True)

res["controls"] = controls
# ---- reproduction of the reader's R-T0 / R-ADE (DAG - DAG0): same estimator, seed, windows => the 4-dp values must be equal
rep = {}
for s in SEEDS:
    rd0 = READ["rows"][f"s{s}"]
    out_s = {}
    for tier, key in (("T0", "R-T0"), ("T1", "R-ADE")):
        h, r = res[tier][f"s{s}"]["ade6_paired_A_minus_B"], rd0[key]
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
