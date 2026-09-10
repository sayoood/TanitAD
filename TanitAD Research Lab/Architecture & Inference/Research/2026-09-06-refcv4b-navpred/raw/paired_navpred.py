"""os_navpred: the paired margins + every pre-registered control (SPEC.md sec.4).

Usage:
  python paired_navpred.py <new_dump_dir> <banked_landing_dump_dir|-> <out.json>

Reads the ep*.npz arm trajectories written by refcv3_arm.py, computes per-window
ADE over the 2 s grid, and runs the PAIRED episode-cluster bootstrap
(taniteval.ci) for every margin the SPEC committed to. Every control that must
read a known value is evaluated and reported PASS/FAIL, before any margin.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TANITEVAL_ROOT", "."))
from taniteval import ci as _ci                                   # noqa: E402

NEW, BANKED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N_BOOT, SEED = 2000, 0
ARMS = ["os", "ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "os_navpred"]


def load(dump):
    """-> (per-arm [N,K,2], gt [N,K,2], episode ids [N], decisions dict)."""
    eps = sorted(glob.glob(os.path.join(dump, "ep*.npz")))
    if not eps:
        raise SystemExit("no ep*.npz in %s" % dump)
    A, G, E = {}, [], []
    for i, f in enumerate(eps):
        z = np.load(f)
        G.append(z["g"])
        E += [i] * len(z["g"])
        for a in ARMS:
            if a in z.files:
                A.setdefault(a, []).append(z[a])
    A = {k: np.concatenate(v) for k, v in A.items()}
    dec = {}
    dd = os.path.join(dump, "decisions")
    if os.path.isdir(dd):
        keys = None
        for f in sorted(glob.glob(os.path.join(dd, "*.npz"))):
            z = np.load(f)
            keys = z.files if keys is None else keys
            for k in keys:
                if k in z.files and z[k].ndim >= 1 and len(z[k]) == len(z["ws"]):
                    dec.setdefault(k, []).append(z[k])
        dec = {k: np.concatenate(v) for k, v in dec.items()}
    return A, np.concatenate(G), np.array(E), dec


def ade(pred, gt):
    """Per-window ADE over the K grid instants (mean L2)."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


A, G, E, DEC = load(NEW)
N = len(G)
res = {
    "_is": "os_navpred paired margins + the SPEC's pre-registered controls",
    "evidence_class": "MEASURED (ours)", "tier": "T1 for every arm here",
    "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                 "n_boot=%d, seed=%d; NEVER overlapping_holdout_se" % (N_BOOT, SEED),
    "n_windows": int(N), "n_episodes": int(E.max() + 1), "dump": NEW,
    "controls": {}, "ade": {}, "paired": {},
}

ADE = {a: ade(A[a], G) for a in ARMS if a in A}
for a in ARMS:
    if a in ADE:
        b = _ci.episode_cluster_bootstrap(ADE[a], list(E), n_boot=N_BOOT, seed=SEED)
        res["ade"][a] = {"mean": round(float(ADE[a].mean()), 4),
                         "ci95": [round(float(b["ci95"][0]), 4),
                                  round(float(b["ci95"][1]), 4)]}

# ------------------------------------------------------------------ CONTROLS
BANKED_ADE = {"os": 0.2975, "ha": 0.2996, "ha0": 0.6723, "ha0_ext": 0.2874,
              "os_navshuf": 0.3013, "os_navzero": 0.3928}
C = res["controls"]

C["C2_grid"] = {"want": "4823 windows / 141 episodes", "got": "%d / %d" % (N, E.max() + 1),
                "verdict": "PASS" if (N == 4823 and E.max() + 1 == 141) else "FAIL"}

mf = {a: (round(float(ADE[a].mean()), 4), BANKED_ADE[a]) for a in ("ha", "ha0") if a in ADE}
C["C1_model_free_known_value"] = {
    "_is": "ha and ha0 consume no model output; on the same windows they MUST "
           "reproduce the pod's banked values or the two surfaces differ",
    "measured_vs_banked": {k: {"measured": v[0], "banked": v[1],
                               "abs_diff": round(abs(v[0] - v[1]), 6)}
                           for k, v in mf.items()},
    "verdict": "PASS" if all(abs(v[0] - v[1]) < 5e-4 for v in mf.values()) else "FAIL"}

d_os = abs(float(ADE["os"].mean()) - BANKED_ADE["os"]) if "os" in ADE else float("nan")
C["C3_os_reproduction"] = {
    "measured": round(float(ADE["os"].mean()), 4), "banked": BANKED_ADE["os"],
    "abs_diff": round(d_os, 6), "tolerance_m": 0.001,
    "_why_a_tolerance": "`os` moves from row 0 of a 2-row batched call to row 0 "
                        "of a 3-row one; the documented cross-call float32 "
                        "batching floor on this model is ~6e-7 m",
    "verdict": "PASS" if d_os < 1e-3 else "FAIL"}

if "nav_cmd_pred" in DEC and "nav_cmd" in DEC:
    npred, ncmd = DEC["nav_cmd_pred"].astype(int), DEC["nav_cmd"].astype(int)
    n_diff = int((npred != ncmd).sum())
    dist = [int((npred == i).sum()) for i in range(4)]
    C["C6_arm_is_not_degenerate"] = {
        "n_navpred_differs_from_navcmd": n_diff, "PREREGISTERED": 1917,
        "n_equal": int((npred == ncmd).sum()), "PREREGISTERED_equal": 2906,
        "navpred_distribution_follow_left_right_straight": dist,
        "PREREGISTERED_distribution": [3465, 614, 744, 0],
        "verdict": "PASS" if (n_diff == 1917 and dist == [3465, 614, 744, 0]) else "FAIL"}
    if "os_navpred" in A and "os" in A:
        eq = npred == ncmd
        dmax = float(np.abs(A["os_navpred"][eq] - A["os"][eq]).max()) if eq.any() else float("nan")
        dmax_ne = (float(np.abs(A["os_navpred"][~eq] - A["os"][~eq]).max())
                   if (~eq).any() else float("nan"))
        C["C5_coverage_identity"] = {
            "n_same_token": int(eq.sum()),
            "max_abs_path_diff_on_same_token_m": dmax, "tolerance_m": 1e-4,
            "CONTROL_max_abs_path_diff_on_DIFFERENT_token_m": dmax_ne,
            "_is": "same token -> same path (to the batching floor). The second "
                   "row is the same-breath control that must read LARGE, or the "
                   "first row's small number proves nothing.",
            "verdict": "PASS" if (dmax < 1e-4 and dmax_ne > 1e-3) else "FAIL"}

if "route_pred_nav_predicted" in DEC and "route_pred_nav_zero" in DEC:
    rp, rz = DEC["route_pred_nav_predicted"].astype(int), DEC["route_pred_nav_zero"].astype(int)
    same = int((rp == rz).sum())
    C["C4_route_head_non_circularity"] = {
        "route_pred(nav_predicted) == route_pred(nav_zero)": "%d/%d" % (same, N),
        "banked_identity_rate_nav_true_vs_nav_zero": "4822/4823",
        "_is": "feeding the model its own predicted route must not move the "
               "route head, or the construction is circular",
        "verdict": "PASS" if same >= N - 2 else "FAIL"}

# ------------------------------------------------------------------- MARGINS
def paired(name, a, b):
    if a not in ADE or b not in ADE:
        return
    r = _ci.paired_episode_cluster_bootstrap(ADE[a], ADE[b], list(E),
                                             n_boot=N_BOOT, seed=SEED)
    lo, hi = float(r["ci95"][0]), float(r["ci95"][1])
    res["paired"][name] = {
        "delta_m": round(float(ADE[a].mean() - ADE[b].mean()), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "separated": bool(lo > 0 or hi < 0),
        "n_windows_differing": int((np.abs(A[a] - A[b]).max(axis=(1, 2)) > 1e-9).sum()),
    }


for x in ("ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero", "os"):
    paired("os_navpred_minus_%s" % x, "os_navpred", x)
for x in ("ha", "ha0", "ha0_ext", "os_navshuf", "os_navzero"):
    paired("os_minus_%s" % x, "os", x)
paired("os_navzero_minus_ha0_ext", "os_navzero", "ha0_ext")

# ------------------------------------------------- the headline arithmetic
if "os_navpred" in ADE:
    a_np, a_z, a_o = (float(ADE["os_navpred"].mean()), float(ADE["os_navzero"].mean()),
                      float(ADE["os"].mean()))
    denom = a_z - a_o
    res["RECOVERY"] = {
        "oracle_nav_worth_m": round(denom, 4),
        "recovered_m": round(a_z - a_np, 4),
        "recovery_fraction_R": round((a_z - a_np) / denom, 4) if abs(denom) > 1e-9 else None,
        "_R_is": "(ADE(os_navzero) - ADE(os_navpred)) / (ADE(os_navzero) - ADE(os))",
        "residual_gap_to_ha0_ext_m": round(a_np - float(ADE["ha0_ext"].mean()), 4),
        "landing_deployment_gap_m": 0.1054,
    }

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps({"controls": {k: v.get("verdict") for k, v in C.items()},
                  "ade": {k: v["mean"] for k, v in res["ade"].items()},
                  "RECOVERY": res.get("RECOVERY")}, indent=2))
print("[out]", OUT)
