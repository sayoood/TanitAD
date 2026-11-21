#!/usr/bin/env python3
"""t1kf_read.py -- the read of PREREG_V8F_T1_FAMILIES.md (commit 635487e5), applied verbatim. Zero training, CPU only.

Rows (arm - P10, per seed; the arm's executed track `cl`, the P10 floor's executed track `fl`, GT `g`; window-origin ego frame, 30 x 0.2 s):
  TAC-LAT  per-window indicator lat_label(executed) == lat_label(GT)   (worse if delta < 0)
  TAC-LON  the same, longitudinal 3-way                                  (worse if delta < 0)
  DK-VIOL  lead_metrics.lead_viol / paired_lead_viol, AT-RISK denominator (worse if delta > 0)
  DK-HW    |headway(executed) - headway(human)| on LEAD windows, both finite   (worse if delta > 0)
  DK-TG    |time-gap(executed) - time-gap(human)|, the same                    (worse if delta > 0)
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2000, seed 0, on the windows where both sides are finite.
Instruments are IMPORTED, never re-implemented: four_families.maneuver_kinematics / tactical_from_trajectory (labeller =
tanitad.refs.refc_tactical.factor_from_kinematics), lead_metrics.*, ci.*, and v8f_dk_read's join guards / contract control / arm_block / pair_block;
`comps` and `paired` are AST-extracted from t1k_read.py (the same helpers every T1 read uses).

Controls C1..C6 (PREREG) are read BEFORE any outcome; a failed one => REFUSED, no outcome word.

  python t1kf_read.py [--tree DIR] [--out T1KF_READ.json] [--n-boot 2000] [--seed 0] [... every input path is an argument ...]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
DT = 0.2
K = 30
LEAD_BLOCK_SHA = "7a879baa2b6f4c140c2b64c9c098a38f84ecb04926da61fa8e96cfffc33073d9"
BANKED_ADE6 = {0: -0.7680, 1: -0.7336}                 # PREREG C5 (HACC - P10, the causal-bucket headline)
BESIDE_BANKED_ADE6 = {"logged": {0: -0.8291, 1: -0.7945}, "strict": {0: -0.7278, 1: -0.6413}}   # t1kin-hacc/read4, t1kin-strict/read (informational)
C5_TOL = 1e-3
TIER = ("T1 (T1-kin: a self-action OPEN loop; ego state = own executed motion; cameras logged) -- never 'closed loop'. "
        "The lead's logged future does not react to the ego: a track that closes on the lead is scored against where the lead WAS. "
        "That is the instrument's contract and it is stated with every number.")
ESTIMATOR = ("taniteval.ci.paired_episode_cluster_bootstrap, B=%d, seed %d, resampling EPISODES on the windows where both sides are finite. It answers ONLY "
             "'would another draw of EPISODES say this?' -- blind to training-run variance (H-ESTIM-SEED-1); the seed floor F=|mean_s0-mean_s1| of the arm rides "
             "beside every delta and no outcome reads it.")
TAC_ROWS = ("TAC-LAT", "TAC-LON")
DK_ROWS = ("DK-VIOL", "DK-HW", "DK-TG")
ALL_ROWS = TAC_ROWS + DK_ROWS
WORSE_IF = {"TAC-LAT": "delta < 0", "TAC-LON": "delta < 0", "DK-VIOL": "delta > 0", "DK-HW": "delta > 0", "DK-TG": "delta > 0"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default=os.path.join(HERE, "tree"), help="export of commit 635487e5 (has stack/, taniteval/ and the v8f-mini package)")
    ap.add_argument("--pkg", help="the v8f-mini package dir (default <tree>/" + PKG_REL + ")")
    ap.add_argument("--stack", help="default <tree>/stack")
    ap.add_argument("--taniteval", help="default <tree>/taniteval")
    ap.add_argument("--dk-read", help="v8f_dk_read.py (default <pkg>/code/v8f_dk_read.py)")
    ap.add_argument("--t1k-read", default=os.path.join(HERE, "ref", "t1k_read_repo.py"),
                    help="t1k_read.py (byte-identical to the committed raw/2026-10-26-t1kin-hacc/t1k_read.py); `comps` and `paired` are AST-extracted")
    ap.add_argument("--lead-block", help="default <pkg>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz")
    ap.add_argument("--lead-block-sha256", default=LEAD_BLOCK_SHA)
    ap.add_argument("--join", help="join_eval141.npz (default <pkg>/raw/join_eval141.npz): clip_index -> clip id")
    ap.add_argument("--grid", help="eval_grid_windows.json (default <pkg>/raw/eval_grid_windows.json)")
    for cfg, rel, suf in (("decisive", "raw/2026-10-26-t1kin-causalbucket/run/dumps", "_causal"), ("logged", "raw/2026-10-26-t1kin-hacc/run/dumps", ""),
                          ("strict", "raw/2026-10-26-t1kin-strict/run/dumps", "_strict")):
        for s in (0, 1):
            ap.add_argument(f"--dump-{cfg}-s{s}", help=f"default <pkg>/{rel}/dumps_HACC_s{s}_m10{suf}.npz")
    ap.add_argument("--out", default=os.path.join(HERE, "T1KF_READ.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.tree = os.path.abspath(a.tree)
    a.pkg = a.pkg or os.path.join(a.tree, PKG_REL)
    a.stack = a.stack or os.path.join(a.tree, "stack")
    a.taniteval = a.taniteval or os.path.join(a.tree, "taniteval")
    a.dk_read = a.dk_read or os.path.join(a.pkg, "code", "v8f_dk_read.py")
    a.lead_block = a.lead_block or os.path.join(a.pkg, "raw", "2026-10-22-dk", "b1_eval_lead_block_k30.npz")
    a.join = a.join or os.path.join(a.pkg, "raw", "join_eval141.npz")
    a.grid = a.grid or os.path.join(a.pkg, "raw", "eval_grid_windows.json")
    for cfg, rel, suf in (("decisive", "raw/2026-10-26-t1kin-causalbucket/run/dumps", "_causal"), ("logged", "raw/2026-10-26-t1kin-hacc/run/dumps", ""),
                          ("strict", "raw/2026-10-26-t1kin-strict/run/dumps", "_strict")):
        for s in (0, 1):
            k = f"dump_{cfg}_s{s}"
            if getattr(a, k) is None:
                setattr(a, k, os.path.join(a.pkg, *rel.split("/"), f"dumps_HACC_s{s}_m10{suf}.npz"))
    return a


A = parse_args()
sys.path.insert(0, A.stack)
sys.path.insert(0, A.taniteval)
import torch  # noqa: E402
from taniteval import four_families as FF  # noqa: E402
from taniteval import lead_metrics as LM  # noqa: E402
from taniteval import ci as CI  # noqa: E402
from tanitad.refs.refc_tactical import factor_from_kinematics, LAT_CLASSES, LON_CLASSES  # noqa: E402
import tanitad  # noqa: E402
import taniteval as TE  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DK = _load_module("v8f_dk_read", A.dk_read)

# ---- `comps` / `paired` / ROWS from t1k_read.py, by AST (t1k_read.py runs a CLI at import; t1klc_read.py extracts them the same way) -------------------
_ns = {"np": np, "torch": torch, "FF": FF, "PB": CI.paired_episode_cluster_bootstrap, "math": math, "DT": DT}
for _node in ast.parse(open(A.t1k_read, encoding="utf-8").read()).body:
    if (isinstance(_node, ast.FunctionDef) and _node.name in ("comps", "paired")) or \
       (isinstance(_node, ast.Assign) and any(getattr(t, "id", "") == "ROWS" for t in _node.targets)):
        exec(compile(ast.Module([_node], []), A.t1k_read, "exec"), _ns)
comps = _ns["comps"]
_paired_t1k = _ns["paired"]
NB = A.n_boot
SEED = A.seed
if NB != 2000 or SEED != 0:                  # SMOKE/DEBUG ONLY: t1k_read.paired hard-codes B=2000 seed=0; the real read never takes this branch
    _ns["PB"] = lambda a, b, e, n_boot=NB, seed=SEED: CI.paired_episode_cluster_bootstrap(a, b, e, n_boot=NB, seed=SEED)


def paired(a, b, eid):
    """t1k_read.paired (B 2000, seed 0 are its constants) on the windows where both sides are finite; B/seed asserted equal to this run's."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    r = {"n_a_finite": int(np.isfinite(a).sum()), "n_b_finite": int(np.isfinite(b).sum()), "n_a_only": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
         "n_b_only": int((np.isfinite(b) & ~np.isfinite(a)).sum())}
    if not m.any():
        return {**r, "status": "NOT-APPLICABLE", "reason": "no window where both sides are finite", "n_windows": 0}
    return {**r, "status": "OK", **_paired_t1k(a, b, np.asarray(eid))}


# ---------------------------------------------------------------------------------------------------------------- inputs
def load_dump(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    need = ("g", "cl", "fl", "ol", "v0", "eid", "clip_index", "ws")
    miss = [k for k in need if k not in d]
    if miss:
        raise SystemExit(f"dump {path} is missing {miss}")
    for k in ("g", "cl", "fl", "ol", "v0", "ha0"):
        if k in d:
            d[k] = d[k].astype(np.float64)
    d["_path"], d["_sha256"] = path, sha256(path)
    return d


def labels(X):
    """-> (lat, lon) int arrays: maneuver_kinematics -> factor_from_kinematics, exactly the chain of four_families.tactical_from_trajectory."""
    t = torch.as_tensor(np.asarray(X, np.float64))
    dyaw, dv, v0, v1, _ = FF.maneuver_kinematics(t, DT)
    lat, lon = factor_from_kinematics(dyaw, dv, v0, v1)
    return lat.numpy().astype(np.int64), lon.numpy().astype(np.int64)


def tac_blocks(X, G, eid):
    return FF.tactical_from_trajectory(torch.as_tensor(np.asarray(X, np.float64)), torch.as_tensor(np.asarray(G, np.float64)), DT, eid=eid, n_boot=NB,
                                       seed=SEED, tier="T1")


def tac_compact(tb):
    out = {"status": tb.get("status"), "n": tb.get("n")}
    for k in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        b = tb[k]
        out[k] = {kk: b.get(kk) for kk in ("n", "classes", "accuracy", "kappa", "kappa_undefined_reason", "per_class", "confusion_gt_rows_pred_cols",
                                          "class_order", "never_predicted", "ci")}
    gs = tb["goal_setting"]
    out["goal_setting"] = {kk: gs.get(kk) for kk in ("goal_point_error_m", "goal_bearing_mae_deg", "goal_bearing_bias_deg", "n_bearing", "goal_range_ratio",
                                                     "goal_long_bias_m", "goal_lat_bias_m", "ci")}
    return out


# ---------------------------------------------------------------------------------------------------------------- per-track preparation
class Ctx:
    """Everything a track is scored against: GT, human distance keeping, the joined lead data, episodes."""

    def __init__(self, G, eid, Jd, hdk):
        self.G, self.eid, self.Jd, self.hdk = G, eid, Jd, hdk
        self.glat, self.glon = labels(G)


def prep(name, X, cx: Ctx):
    X = np.asarray(X, np.float64)
    DK.check_pred(name, X, cx.G, cx.Jd["speeds"], K)
    lat, lon = labels(X)
    dk = LM.distance_keeping(X[:, :K], cx.Jd["leads"], cx.Jd["lead_lens"], cx.Jd["speeds"], DT)
    ab, lvd = DK.arm_block(name, dk, cx.hdk, cx.Jd, cx.G[:, :K], X[:, :K], cx.eid, NB, SEED)
    return {"name": name, "X": X, "lat": lat, "lon": lon, "ind": {"TAC-LAT": (lat == cx.glat).astype(np.float64), "TAC-LON": (lon == cx.glon).astype(np.float64)},
            "dk": dk, "lv": lvd, "arm_block": ab,
            "abs": {"DK-HW": np.abs(np.asarray(dk["headway_min_m"], np.float64) - np.asarray(cx.hdk["headway_min_m"], np.float64)),
                    "DK-TG": np.abs(np.asarray(dk["time_gap_min_s"], np.float64) - np.asarray(cx.hdk["time_gap_min_s"], np.float64))}}


def keep(r):
    return {k: r.get(k) for k in ("status", "delta", "lo", "hi", "separated", "n_windows", "n_episodes", "n_a_only", "n_b_only", "n_a_finite", "n_b_finite",
                                  "p_delta_gt0", "reason") if k in r}


def rows(P, Q, eid):
    """arm P minus arm Q on the five PREREG rows. `separated_worse` follows the table's 'worse if' column."""
    out = {}
    for r in TAC_ROWS:
        out[r] = keep(paired(P["ind"][r], Q["ind"][r], eid))
    pl = LM.paired_lead_viol(P["lv"], Q["lv"], eid, names=(P["name"], Q["name"]), n_boot=NB, seed=SEED)
    at = pl.get("at_risk") or {}
    out["DK-VIOL"] = {**keep(at), "n_at_risk": at.get("n_at_risk"), "n_viol_a": at.get("n_viol_a"), "n_viol_b": at.get("n_viol_b"),
                      "n_lead_windows": pl.get("n_lead_windows"), "denominator": at.get("denominator"),
                      "all_lead_windows_beside": {k: pl.get(k) for k in ("delta", "lo", "hi", "separated", "n_viol_a", "n_viol_b", "n_lead_windows")}}
    for r in ("DK-HW", "DK-TG"):
        out[r] = keep(paired(P["abs"][r], Q["abs"][r], eid))
    for r in ALL_ROWS:
        v = out[r]
        if v.get("status") == "OK":
            v["worse_if"] = WORSE_IF[r]
            v["separated_worse"] = bool(v["separated"] and ((v["delta"] < 0) if r in TAC_ROWS else (v["delta"] > 0)))
        else:
            v["separated_worse"] = None
    return out


def is_zero(v):
    return v.get("status") == "OK" and v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0


def row_floor(Ps0, Ps1):
    """F = |mean_s0 - mean_s1| of the ARM's row statistic (own finite windows), per row."""
    F = {}
    for r in TAC_ROWS:
        a, b = float(Ps0["ind"][r].mean()), float(Ps1["ind"][r].mean())
        F[r] = {"F": abs(a - b), "mean_s0": a, "mean_s1": b, "statistic": "accuracy of the executed label vs GT, all windows"}
    a, b = (Q["lv"].get("share_of_at_risk") for Q in (Ps0, Ps1))
    F["DK-VIOL"] = {"F": (abs(a - b) if a is not None and b is not None else None), "mean_s0": a, "mean_s1": b, "statistic": "lead_viol share_of_at_risk"}
    for r in ("DK-HW", "DK-TG"):
        va, vb = Ps0["abs"][r], Ps1["abs"][r]
        a, b = float(np.nanmean(va)), float(np.nanmean(vb))
        F[r] = {"F": abs(a - b), "mean_s0": a, "mean_s1": b, "n_s0": int(np.isfinite(va).sum()), "n_s1": int(np.isfinite(vb).sum()),
                "statistic": "mean |arm - human| over the arm's own finite windows"}
    return F


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    t_start = time.time()
    prov = {"tanitad": os.path.abspath(tanitad.__file__), "taniteval": os.path.abspath(TE.__file__), "v8f_dk_read": os.path.abspath(DK.__file__),
            "t1k_read": os.path.abspath(A.t1k_read), "t1k_read_sha256": sha256(A.t1k_read), "v8f_dk_read_sha256": sha256(A.dk_read),
            "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0], "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    for k in ("tanitad", "taniteval", "v8f_dk_read"):
        if prov[k].replace("\\", "/").upper().startswith("G:/"):
            raise SystemExit(f"{k} was imported from the dead G: tree: {prov[k]}")
    DK.code_provenance()
    prov["code"] = DK.code_provenance()
    controls, refused = {}, []
    report = {"schema": "t1kf-read-v1", "prereg": "Project Steering/PREREG_V8F_T1_FAMILIES.md @ 635487e5 (claim row H-V8F-T1-FAMILIES-1)", "tier": TIER,
              "estimator": ESTIMATOR % (NB, SEED), "controls": controls, "OUTCOME": None, "provenance": prov}

    def done(status_msg, code):
        report["OUTCOME"] = {"TACTICAL": "REFUSED", "DISTANCE_KEEPING": "REFUSED", "refused_controls": refused, "reason": status_msg}
        with open(A.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(DK._clean(report), f, indent=1, default=str)
        print("OUTCOME TACTICAL REFUSED  DISTANCE_KEEPING REFUSED :", status_msg)
        return code

    # ---- inputs ----
    dumps = {(c, s): load_dump(getattr(A, f"dump_{c}_s{s}")) for c in ("decisive", "logged", "strict") for s in (0, 1)}
    d0 = dumps[("decisive", 0)]
    n = d0["g"].shape[0]
    report["inputs"] = {f"dump_{c}_s{s}": {"path": d["_path"], "sha256": d["_sha256"]} for (c, s), d in dumps.items()}
    report["inputs"].update({"join": {"path": A.join, "sha256": sha256(A.join)}, "grid": {"path": A.grid, "sha256": sha256(A.grid)},
                             "lead_block": {"path": A.lead_block, "sha256": sha256(A.lead_block)}})
    # ---- X3: the six dumps are on one window grid, one GT ----
    same = {}
    for (c, s), d in dumps.items():
        same[f"{c}_s{s}"] = {"clip_index": bool(np.array_equal(d["clip_index"], d0["clip_index"])), "ws": bool(np.array_equal(d["ws"], d0["ws"])),
                             "eid": bool(np.array_equal(d["eid"], d0["eid"])), "g_max_abs_diff_m": float(np.abs(d["g"] - d0["g"]).max()),
                             "v0_max_abs_diff": float(np.abs(d["v0"] - d0["v0"]).max())}
    x3 = {"PASS": all(v["clip_index"] and v["ws"] and v["eid"] and v["g_max_abs_diff_m"] == 0.0 and v["v0_max_abs_diff"] == 0.0 for v in same.values()),
          "per_dump": same, "n_windows": int(n)}
    controls["X3_one_grid_one_gt"] = x3

    # ---- C3 (first: nothing else means anything on a wrong join): the clip_id mapping + the three v8f_dk_read join guards ----
    c3 = {"what": "dump v0 == block speed at every joined row to 1e-3 m/s; block ts_rel_s == 0.2*(1..30) to 1e-6; no duplicate row, no missing grid window",
          "mapping": {"clip_id": "join_eval141.npz['clip'][clip_index]   (v8f_t1_eval.make_states: clip = str(J['clip'][ci]); wins = D.eval_grid_windows(grid, J) -> (ci, t0, 0, 0); "
                                "write_dumps: clip_index = ci, ws = t0)",
                      "frame": "2 * ws   (RAW 10 Hz frame = 2 x cache step; v8f_dk_read: 'cache step t <-> frame 2t')"}, "per_dump": {}}
    try:
        Jtab = np.load(A.join, allow_pickle=False)
        clip_of = Jtab["clip"].astype(str)
        gridj = json.load(open(A.grid, encoding="utf-8"))
        expect = [(c, int(t)) for c in gridj["order"] for t in gridj["windows"][c]]
        clip = clip_of[d0["clip_index"]]
        frame = 2 * d0["ws"].astype(np.int64)
        got = list(zip(clip.tolist(), d0["ws"].astype(int).tolist()))
        c3["grid"] = {"n_grid_windows": len(expect), "n_dump_windows": int(n), "dump_order_equals_grid_enumeration": got == expect,
                      "dump_pairs_unique": len(set(zip(d0["clip_index"].tolist(), d0["ws"].tolist()))) == n, "n_episodes": int(np.unique(d0["clip_index"]).size),
                      "join_clip_order_equals_grid_order": clip_of.tolist() == list(gridj["order"])}
        if got != expect or not c3["grid"]["dump_pairs_unique"] or len(expect) != n:
            raise DK.Refusal("the dump's (clip_index, ws) sequence is not the eval grid's enumeration: the clip mapping is wrong")
        blk = DK.load_block(A.lead_block)                                    # refuses a duplicate (clip, frame) row
        c3["lead_block"] = {"sha256": report["inputs"]["lead_block"]["sha256"], "expected_sha256": A.lead_block_sha256,
                            "sha256_equal": report["inputs"]["lead_block"]["sha256"] == A.lead_block_sha256, "n_rows": len(blk["_idx"]), "block_k": blk["_k"],
                            "built_utc": blk["_meta"].get("built_utc")}
        if not c3["lead_block"]["sha256_equal"]:
            raise DK.Refusal("the lead block's sha256 is not the registered 7a879baa...3d9")
        ts = np.asarray(blk["ts_rel_s"], np.float64).reshape(-1)
        c3["ts_rel_s_max_abs_dev_from_0p2x1to30_s"] = float(np.max(np.abs(ts[:K] - DT * np.arange(1, K + 1))))
        Jds = {}
        for (c, s), d in dumps.items():
            Jd = DK.join_block(blk, clip, frame, d["v0"], k=K, dt=DT)          # ts guard, speed proof, missing-window guard
            Jds[(c, s)] = Jd
            c3["per_dump"][f"{c}_s{s}"] = {"speed_proof_max_abs_mps": Jd["speed_proof_max_mps"], "tol_mps": DK.LEAD_SPEED_TOL_MPS, "counts": Jd["counts"],
                                           "n_rows_joined": int((Jd["rows"] >= 0).sum()), "n_rows_unique": int(np.unique(Jd["rows"]).size)}
        Jd = Jds[("decisive", 0)]
        c3["window_states"] = dict(Jd["counts"])
        c3["window_states_reasons"] = {"LEAD": "a lead agent inside the straight corridor within 80 m: the ONLY windows any lead row scores",
                                       "NO_LEAD": "road clear within 80 m: counted, never scored as free flow",
                                       "NOT_STRAIGHT": "the ego's own future leaves the straight corridor (a bend hides the lead): counted, never scored",
                                       "NO_LABEL": "no obstacle.offline for the clip or the horizon leaves the labelled span: counted, never scored"}
        c3["window_states_sum"] = int(sum(Jd["counts"].values()))
        c3["PASS"] = bool(all(v["speed_proof_max_abs_mps"] <= DK.LEAD_SPEED_TOL_MPS and v["n_rows_joined"] == n and v["n_rows_unique"] == n
                              for v in c3["per_dump"].values()) and c3["window_states_sum"] == n and c3["ts_rel_s_max_abs_dev_from_0p2x1to30_s"] <= DK.LEAD_TS_TOL_S)
    except DK.Refusal as e:
        c3["PASS"], c3["refusal"] = False, str(e)
    controls["C3_join"] = c3
    if not c3["PASS"]:
        refused.append("C3")
        return done("C3 (the join) failed: " + str(c3.get("refusal", "see controls.C3_join")), 2)

    # ---- the human reference and C4 (the kinematic contract) ----
    G = d0["g"]
    hdk = LM.distance_keeping(G[:, :K], Jd["leads"], Jd["lead_lens"], Jd["speeds"], DT)
    try:
        ct = DK.contract_check(hdk, Jd)
        c4 = {**ct, "tol_median_m": DK.CONTRACT_MEDIAN_TOL_M, "PASS": ct.get("status") == "OK" and ct["median_abs_headway_diff_m"] <= DK.CONTRACT_MEDIAN_TOL_M}
    except DK.Refusal as e:
        c4 = {"PASS": False, "refusal": str(e)}
    controls["C4_kinematic_contract"] = c4
    if not c4["PASS"]:
        refused.append("C4")

    eid = d0["eid"]
    cx = Ctx(G, eid, Jd, hdk)

    # ---- C2: the labeller on GT ----
    tg = tac_blocks(G, G, eid)
    c2 = {"lateral_accuracy_GT_vs_GT": tg["lateral_decision"]["accuracy"], "longitudinal_accuracy_GT_vs_GT": tg["longitudinal_decision"]["accuracy"],
          "per_window_indicator_min_lat": float((labels(G)[0] == cx.glat).min()), "per_window_indicator_min_lon": float((labels(G)[1] == cx.glon).min())}
    c2["PASS"] = bool(c2["lateral_accuracy_GT_vs_GT"] == 1.0 and c2["longitudinal_accuracy_GT_vs_GT"] == 1.0 and c2["per_window_indicator_min_lat"] == 1.0
                      and c2["per_window_indicator_min_lon"] == 1.0)
    c2["GT_class_counts"] = {"lat": {nm: int((cx.glat == i).sum()) for i, nm in enumerate(LAT_CLASSES)},
                             "lon": {nm: int((cx.glon == i).sum()) for i, nm in enumerate(LON_CLASSES)}}
    controls["C2_labeller_on_GT"] = c2
    if not c2["PASS"]:
        refused.append("C2")

    # ---- per config x seed: prepare tracks, rows ----
    cfgs = {}
    prepared = {}
    for c in ("decisive", "logged", "strict"):
        for s in (0, 1):
            d = dumps[(c, s)]
            t0 = time.time()
            prepared[(c, s)] = {"arm": prep(f"HACC_s{s}", d["cl"], cx), "p10": prep("P10", d["fl"], cx)}
            print(f"[prep] {c} s{s} {time.time() - t0:.0f}s", flush=True)
    # ---- C5: the dumps reproduce the banked ADE6 delta ----
    c5 = {"what": "ADE6 (HACC - P10) read from the dumps, via t1k_read.comps/paired, vs the banked causal-bucket numbers", "tol": C5_TOL, "per_seed": {}}
    for s in (0, 1):
        d = dumps[("decisive", s)]
        r = _paired_t1k(comps(d["cl"], d["g"])["ade6_m"], comps(d["fl"], d["g"])["ade6_m"], d["eid"])
        c5["per_seed"][f"s{s}"] = {"read": keep(r), "banked_delta": BANKED_ADE6[s], "abs_diff": abs(r["delta"] - BANKED_ADE6[s]),
                                   "PASS": bool(abs(r["delta"] - BANKED_ADE6[s]) <= C5_TOL)}
    c5["PASS"] = all(v["PASS"] for v in c5["per_seed"].values())
    c5["beside_informational_other_configs"] = {}
    for c in ("logged", "strict"):
        for s in (0, 1):
            d = dumps[(c, s)]
            r = _paired_t1k(comps(d["cl"], d["g"])["ade6_m"], comps(d["fl"], d["g"])["ade6_m"], d["eid"])
            c5["beside_informational_other_configs"][f"{c}_s{s}"] = {"read_delta": r["delta"], "banked_delta": BESIDE_BANKED_ADE6[c][s],
                                                                      "abs_diff": abs(r["delta"] - BESIDE_BANKED_ADE6[c][s])}
    controls["C5_dumps_ade6"] = c5
    if not c5["PASS"]:
        refused.append("C5")

    # ---- the rows ----
    for c in ("decisive", "logged", "strict"):
        cfgs[c] = {"role": "decisive" if c == "decisive" else "beside (decides nothing)", "seeds": {}}
        for s in (0, 1):
            Pm, Pf = prepared[(c, s)]["arm"], prepared[(c, s)]["p10"]
            t0 = time.time()
            rw = rows(Pm, Pf, eid)
            cfgs[c]["seeds"][f"s{s}"] = {"rows_HACC_minus_P10": rw}
            print(f"[rows] {c} s{s} {time.time() - t0:.0f}s", flush=True)
        F = row_floor(prepared[(c, 0)]["arm"], prepared[(c, 1)]["arm"])
        cfgs[c]["seed_floor_F"] = F
        for s in (0, 1):
            for r, v in cfgs[c]["seeds"][f"s{s}"]["rows_HACC_minus_P10"].items():
                Fv = F[r]["F"]
                v["seed_floor_F"] = Fv
                v["abs_delta_over_F"] = (abs(v["delta"]) / Fv if v.get("status") == "OK" and Fv else None)

    # ---- C1: identity ----
    c1 = {"what": "arm vs itself: every row's paired delta exactly +0 [0, 0]", "per_config_seed": {}}
    ok1 = True
    for (c, s), pr in prepared.items():
        ident = rows(pr["arm"], pr["arm"], eid)
        z = {r: is_zero(v) for r, v in ident.items()}
        c1["per_config_seed"][f"{c}_s{s}"] = {"rows": {r: {k: v.get(k) for k in ("delta", "lo", "hi", "n_windows")} for r, v in ident.items()}, "all_exactly_zero": all(z.values())}
        ok1 &= all(z.values())
    c1["PASS"] = bool(ok1)
    controls["C1_identity"] = c1
    if not c1["PASS"]:
        refused.append("C1")

    # ---- C6: deliberate regression: decisive HACC tracks rolled by one window (window i scored with track i+1) vs P10 ----
    c6 = {"what": "np.roll(cl, -1, axis=0): window i scored with track i+1 (the last with track 0), vs P10; TAC-LAT, TAC-LON, DK-HW must each be separated-worse at s0",
          "per_seed": {}}
    for s in (0, 1):
        d = dumps[("decisive", s)]
        Pr = prep(f"HACC_s{s}_rolled", np.roll(d["cl"], -1, axis=0), cx)
        rw = rows(Pr, prepared[("decisive", s)]["p10"], eid)
        Pu, Pp = prepared[("decisive", s)]["arm"], prepared[("decisive", s)]["p10"]
        means = {r: {"HACC": float(Pu["ind"][r].mean()), "HACC_rolled": float(Pr["ind"][r].mean()), "P10": float(Pp["ind"][r].mean())} for r in TAC_ROWS}
        means.update({r: {"HACC": float(np.nanmean(Pu["abs"][r])), "HACC_rolled": float(np.nanmean(Pr["abs"][r])), "P10": float(np.nanmean(Pp["abs"][r]))}
                      for r in ("DK-HW", "DK-TG")})
        means["DK-VIOL"] = {"HACC": Pu["lv"].get("share_of_at_risk"), "HACC_rolled": Pr["lv"].get("share_of_at_risk"), "P10": Pp["lv"].get("share_of_at_risk")}
        c6["per_seed"][f"s{s}"] = {"rows_rolled_minus_P10": rw, "gating": s == 0,
                                   "must_be_separated_worse": {r: rw[r]["separated_worse"] for r in ("TAC-LAT", "TAC-LON", "DK-HW")},
                                   "DIAGNOSTIC_NOT_GATING_arm_means": means,
                                   "DIAGNOSTIC_NOT_GATING_rolled_minus_unrolled_HACC": rows(Pr, Pu, eid)}
    c6["PASS"] = bool(all(c6["per_seed"]["s0"]["must_be_separated_worse"].values()))
    controls["C6_deliberate_regression"] = c6
    if not c6["PASS"]:
        refused.append("C6")

    # ---- X1 / X2: internal consistency of this implementation (not PREREG controls; a failure is an implementation bug and refuses the read) ----
    x1 = {"what": "the per-window indicators' mean reproduces tactical_from_trajectory's accuracy (4 dp) for every arm, P10 and config", "per": {}}
    tacs = {}
    ok_x1 = True
    for (c, s), pr in prepared.items():
        for who, key in (("arm", "arm"), ("p10", "p10")):
            P = pr[key]
            tb = tac_blocks(P["X"], G, eid)
            tacs[(c, s, who)] = tb
            la, lo = tb["lateral_decision"]["accuracy"], tb["longitudinal_decision"]["accuracy"]
            ok = abs(round(float(P["ind"]["TAC-LAT"].mean()), 4) - la) < 1e-9 and abs(round(float(P["ind"]["TAC-LON"].mean()), 4) - lo) < 1e-9
            x1["per"][f"{c}_s{s}_{who}"] = {"indicator_mean_lat": float(P["ind"]["TAC-LAT"].mean()), "instrument_lat": la,
                                            "indicator_mean_lon": float(P["ind"]["TAC-LON"].mean()), "instrument_lon": lo, "ok": bool(ok)}
            ok_x1 &= ok
    x1["PASS"] = bool(ok_x1)
    controls["X1_indicator_vs_instrument_accuracy"] = x1
    x2 = {"what": "DK-HW / DK-TG deltas equal v8f_dk_read.pair_block's abs_error_vs_human (same instrument, same windows)", "per": {}}
    ok_x2 = True
    pairblocks = {}
    for (c, s), pr in prepared.items():
        Pm, Pf = pr["arm"], pr["p10"]
        dkS = {Pm["name"]: Pm["dk"], "P10": Pf["dk"]}
        lvS = {Pm["name"]: Pm["lv"], "P10": Pf["lv"]}
        absE = {Pm["name"]: {kk: np.abs(np.asarray(Pm["dk"][kk], np.float64) - np.asarray(hdk[kk], np.float64)) for kk in DK.DK_KEYS},
                "P10": {kk: np.abs(np.asarray(Pf["dk"][kk], np.float64) - np.asarray(hdk[kk], np.float64)) for kk in DK.DK_KEYS}}
        pb = DK.pair_block(Pm["name"], "P10", dkS, lvS, absE, Jd, eid, {}, NB, SEED)
        pairblocks[(c, s)] = pb
        mine = cfgs[c]["seeds"][f"s{s}"]["rows_HACC_minus_P10"]
        e = {}
        for r, kk in (("DK-HW", "headway_min_m"), ("DK-TG", "time_gap_min_s")):
            theirs = pb["abs_error_vs_human"][kk]
            e[r] = {"mine": [mine[r].get("delta"), mine[r].get("lo"), mine[r].get("hi"), mine[r].get("n_windows")],
                    "pair_block": [theirs.get("delta"), theirs.get("lo"), theirs.get("hi"), theirs.get("n_used")]}
            ok_x2 &= e[r]["mine"] == e[r]["pair_block"]
        vt = pb["lead_viol"]["at_risk"] or {}
        e["DK-VIOL"] = {"mine": [mine["DK-VIOL"].get("delta"), mine["DK-VIOL"].get("lo"), mine["DK-VIOL"].get("hi")], "pair_block": [vt.get("delta"), vt.get("lo"), vt.get("hi")]}
        ok_x2 &= e["DK-VIOL"]["mine"] == e["DK-VIOL"]["pair_block"]
        x2["per"][f"{c}_s{s}"] = e
    x2["PASS"] = bool(ok_x2)
    controls["X2_dk_rows_vs_pair_block"] = x2
    if not (x1["PASS"] and x2["PASS"] and x3["PASS"]):
        refused.append("X-internal:" + ",".join(k for k, v in (("X1", x1), ("X2", x2), ("X3", x3)) if not v["PASS"]))

    # ---- beside blocks ----
    beside = {"_note": "decides nothing (PREREG 'Beside, never decisive'). by_speed strata of v8f_dk_read.arm_block are omitted from this JSON for size."}
    for (c, s), pr in prepared.items():
        b = {}
        for who in ("arm", "p10"):
            P = pr[who]
            ab = dict(P["arm_block"])
            ab.pop("by_speed", None)
            b[who] = {"tactical": tac_compact(tacs[(c, s, who)]), "distance_keeping": ab}
        pb = pairblocks[(c, s)]
        b["paired_arm_minus_p10"] = {"dk_paired_headway_timegap_min_ttc": pb["dk"], "abs_error_vs_human": pb["abs_error_vs_human"], "lead_viol": pb["lead_viol"]}
        # the 5-way/lat/lon confusion of the EXECUTED label against GT is in b[who].tactical.*.confusion_gt_rows_pred_cols
        beside[f"{c}_s{s}"] = b
    # label dtype sensitivity (decisive s0): float32 vs float64 labelling of the stored tracks
    sens = {}
    for nm, X in (("cl", dumps[("decisive", 0)]["cl"]), ("fl", dumps[("decisive", 0)]["fl"]), ("g", G)):
        l64 = labels(X)
        t32 = torch.as_tensor(np.asarray(X, np.float32))
        dyaw, dv, v0_, v1_, _ = FF.maneuver_kinematics(t32, DT)
        l32 = tuple(a.numpy() for a in factor_from_kinematics(dyaw, dv, v0_, v1_))
        sens[nm] = {"lat_flips": int((l64[0] != l32[0]).sum()), "lon_flips": int((l64[1] != l32[1]).sum())}
    beside["label_dtype_sensitivity_decisive_s0_float32_vs_float64"] = sens
    beside["P10_floor_identical_across_seeds"] = {c: float(np.abs(dumps[(c, 0)]["fl"] - dumps[(c, 1)]["fl"]).max()) for c in ("decisive", "logged", "strict")}
    report["beside"] = beside

    # ---- outcome (decisive config only; controls first) ----
    if refused:
        report["OUTCOME"] = {"TACTICAL": "REFUSED", "DISTANCE_KEEPING": "REFUSED", "refused_controls": refused,
                             "reason": "a control failed (PREREG: 'Each must pass, or the read is REFUSED ... The family is then reported REFUSED, never as a pass')"}
    else:
        dec = cfgs["decisive"]["seeds"]
        tac_w = {f"s{s}": [r for r in TAC_ROWS if dec[f"s{s}"]["rows_HACC_minus_P10"][r]["separated_worse"]] for s in (0, 1)}
        dk_w = {f"s{s}": [r for r in DK_ROWS if dec[f"s{s}"]["rows_HACC_minus_P10"][r]["separated_worse"]] for s in (0, 1)}
        report["OUTCOME"] = {"TACTICAL": "DEFICIT" if any(tac_w.values()) else "NO WORSE", "DISTANCE_KEEPING": "DEFICIT" if any(dk_w.values()) else "NO WORSE",
                             "STRATEGIC": "NOT COMPUTED: the dumps carry no per-tick S checkpoints (PREREG 'Not covered here')",
                             "TACTICAL_T_CODE": "NOT COMPUTED: the dumps carry no per-tick T logits (PREREG 'Not covered here')",
                             "separated_worse_rows": {"TACTICAL": tac_w, "DISTANCE_KEEPING": dk_w}, "refused_controls": [],
                             "rule": "TACTICAL DEFICIT if either TAC row is separated-worse at >= 1 seed; DISTANCE KEEPING DEFICIT if any of DK-VIOL/HW/TG is; else NO WORSE. "
                                     "Decisive config = HACC causal speed bucket; read per family, never pooled."}
    report["configs"] = {"_status": ("OUTCOME-BEARING (all controls passed)" if not refused else
                                     "COMPUTED BUT THE READ IS REFUSED (control(s) failed: %s): these rows are data, they carry NO outcome word" % refused), **cfgs}
    report["runtime_s"] = round(time.time() - t_start, 1)
    # order: controls and outcome first
    ordered = {k: report[k] for k in ("schema", "prereg", "tier", "estimator", "controls", "OUTCOME", "configs", "beside", "inputs", "provenance", "runtime_s")}
    with open(A.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DK._clean(ordered), f, indent=1, default=str)

    # ---- summary ----
    print("\n=== CONTROLS ===")
    for k, v in controls.items():
        print(f"  {k:42s} PASS={v['PASS']}")
    print("\n=== DECISIVE (HACC causal) rows, arm - P10 ===")
    for s in (0, 1):
        print(f" seed {s}  (F per row: " + ", ".join(f"{r}={cfgs['decisive']['seed_floor_F'][r]['F']:.4g}" for r in ALL_ROWS if cfgs['decisive']['seed_floor_F'][r]['F'] is not None) + ")")
        for r in ALL_ROWS:
            v = cfgs["decisive"]["seeds"][f"s{s}"]["rows_HACC_minus_P10"][r]
            print(f"   {r:8s} delta {v.get('delta')} [{v.get('lo')}, {v.get('hi')}] sep={v.get('separated')} worse={v.get('separated_worse')} n={v.get('n_windows')}"
                  + (f" n_at_risk={v.get('n_at_risk')}" if r == "DK-VIOL" else ""))
    print("\nwindow states:", Jd["counts"])
    print("OUTCOME:", json.dumps(report["OUTCOME"]["TACTICAL"]), "(TACTICAL)", json.dumps(report["OUTCOME"]["DISTANCE_KEEPING"]), "(DISTANCE KEEPING)", "refused:", refused)
    print(f"runtime {report['runtime_s']} s -> {A.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
