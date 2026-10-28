"""rederive_sub.py -- an independent re-derivation of the SPEEDSUB read from the per-window dumps, NOT through the instrument's `subread`.

What it does (CPU, numpy only, one process):
  1. loads the five per-mode eval dumps per seed (own, identity, gt, ego, egoshuf) and the banked SPEEDREAD `own` / `gt` / identity-cells dumps;
  2. recomputes the controls it can recompute from arrays (S1, S2, S3, S4, S5, S6, S7, S7b, S8, S9, S10, S11, S12, S13);
  3. recomputes the three paired contrasts (ego, egoshuf, gt, each against own) with taniteval.ci.paired_episode_cluster_bootstrap, B = 2000, seed 0
     (the library function from the pinned run tree, called here directly, not through v8f_mini_eval_speedread.py);
  4. applies PREREG_V8F_SPEEDSUB section 5's table, written out below from the PREREG's text, to those rows;
  5. beside, post-hoc and not registered: the plan rule-violation rates (dump keys viol_*) as paired contrasts ego - own and gt - own, same estimator.
It compares every recomputed row with SUBREAD.json to 4 dp and stops on a mismatch.
run: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -X utf8 rederive_sub.py
"""
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

TREE = "F:/TanitAD-scratch/speedsub_tmp/tree"
W = "F:/TanitAD-scratch/speedsub/"
SR = "F:/TanitAD-scratch/speedread/"
OUT = "E:/Projects/TanitAD-artifacts/dagland/speedsub_result/"
sys.path.insert(0, TREE + "/taniteval")
from taniteval.ci import paired_episode_cluster_bootstrap as PB    # noqa: E402

RO_SHA = "a205609ab47325a0ad9c0141be8ae3a0fec653d59c6db17ec1cf0d460bf2316d"
F_M = 0.1207                  # PREREG section 1: F, INHERITED from RESULT_V8F_GRID.md
F_PASS = F_M / 2.0            # 0.06035
PERM_SEED, N_WIN = 20261029, 2399
MODES = ("own", "identity", "gt", "ego", "egoshuf")
VIOL = ("viol_any", "viol_SPEED", "viol_STOP", "viol_TURN", "viol_LEAD")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SUB = json.load(open(W + "subread/SUBREAD.json", encoding="utf-8"))


def ld(s, m):
    return np.load(f"{W}eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.npz")


def cells(s, m):
    return np.load(f"{W}eval_s{s}/speedread_cells_M2b-mixo_s{s}-speedsub-{m}.npz")


def bs(a, b, eid):
    o = PB(np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(eid), n_boot=2000, seed=0)
    return {"delta": float(o["delta"]), "lo": float(o["lo"]), "hi": float(o["hi"]), "separated": bool(o["separated"]),
            "n_windows": int(o["n_windows"]), "n_episodes": int(o["n_episodes"]), "unrounded_point": float(np.mean(a) - np.mean(b))}


sep_better = lambda r: r["hi"] < 0
sep_worse = lambda r: r["lo"] > 0

R = {"schema": "speedsub-rederive-v1", "script": "rederive_sub.py", "taniteval_ci_sha256": sha(TREE + "/taniteval/taniteval/ci.py"),
     "F_M": F_M, "F_PASS": F_PASS, "seeds": {}}
ctl = {}
rows = {}
EJ = {}
for s in (0, 1):
    D = {m: ld(s, m) for m in MODES}
    C = {m: cells(s, m) for m in ("identity", "ego", "egoshuf", "gt")}
    eid = D["own"]["eid"]
    c = {}
    c["S4_eid_aligned"] = all(np.array_equal(D[m]["eid"], eid) for m in MODES)
    c["S1_identity_eq_own"] = bool(np.max(np.abs(D["identity"]["ade6"] - D["own"]["ade6"])) == 0.0 and np.array_equal(D["identity"]["pred"], D["own"]["pred"]))
    bo = np.load(f"{SR}eval_s{s}/eval_M2b-mixo_s{s}-speedread-own.npz")["ade6"].astype(np.float64)
    bg = np.load(f"{SR}eval_s{s}/eval_M2b-mixo_s{s}-speedread-gt.npz")["ade6"].astype(np.float64)
    c["S5_max_abs_dev_own_vs_speedread_own"] = float(np.max(np.abs(D["own"]["ade6"] - bo)))
    c["S6_max_abs_dev_gt_vs_speedread_gt"] = float(np.max(np.abs(D["gt"]["ade6"] - bg)))
    bid = np.load(f"{SR}eval_s{s}/speedread_cells_M2b-mixo_s{s}-speedread-identity.npz")["pred_mlp_ego"].astype(np.float64)
    pid = C["identity"]["pred_mlp_ego"]
    c["S7b_max_abs_dev_identity_pred_vs_speedread_banked"] = float(np.max(np.abs(pid - bid)))
    mu, sd = C["identity"]["cons_mu"].astype(np.float64), C["identity"]["cons_sd"].astype(np.float64)
    want = ((pid - mu) / sd).astype(np.float32)
    c["S7_max_abs_dev_ego_used_vs_identity_pred_std"] = float(np.max(np.abs(C["ego"]["used_std"] - want)))
    c["S10_frac_ego_cells_moved_vs_own"] = float(np.mean(np.any(C["ego"]["used_std"] != C["ego"]["own_std"], axis=1)))
    c["S10_frac_egoshuf_differs_from_ego"] = float(np.mean(np.any(C["egoshuf"]["used_std"] != C["ego"]["used_std"], axis=1)))
    perm = np.random.default_rng(PERM_SEED).permutation(N_WIN)
    c["S9_perm_sha256"] = hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()
    c["S9_alignment_max_abs_dev_egoshuf_used_vs_ego_used_permuted"] = float(np.max(np.abs(C["egoshuf"]["used_std"] - C["ego"]["used_std"][perm])))
    c["S9_n_fixed_points"] = int(np.sum(perm == np.arange(N_WIN)))
    c["S2_own_std_equal_across_modes"] = all(np.array_equal(C[m]["own_std"], C["identity"]["own_std"]) for m in ("ego", "egoshuf", "gt"))
    # the dumps' own readout pin and hook stats
    ej = {m: json.load(open(f"{W}eval_s{s}/eval_M2b-mixo_s{s}-speedsub-{m}.json", encoding="utf-8")) for m in MODES}
    EJ[s] = ej
    c["S8_readout_sha_in_dumps"] = all(ej[m]["speedread"]["readout_sha256"] == RO_SHA for m in MODES)
    c["S2_nonspeed_max_abs_dev_in_hooked_dumps"] = max(ej[m]["speedread"]["hook"]["nonspeed_max_abs_dev"] for m in ("identity", "gt", "ego", "egoshuf"))
    c["S3_tick_mirror_ok_in_hooked_dumps"] = all(ej[m]["speedread"]["hook"]["tick_mirror_ok"] for m in ("identity", "gt", "ego", "egoshuf"))
    c["S9_perm_sha256_in_dump"] = ej["egoshuf"]["speedread"]["hook"]["shuffle"]["perm_sha256"]
    # per-cell eval RMSE of T_own and mlp_ego against the GT, where the GT cell is present (cross-check of SUBREAD.json cells)
    gt = C["identity"]["gt_phys"].astype(np.float64)
    own_phys = C["identity"]["own_std"].astype(np.float64) * sd + mu
    rm = {"T_own": [], "mlp_ego": [], "n_present": []}
    for j in range(5):
        ok = np.isfinite(gt[:, j])
        rm["n_present"].append(int(ok.sum()))
        rm["T_own"].append(float(np.sqrt(np.mean((own_phys[ok, j] - gt[ok, j]) ** 2))))
        rm["mlp_ego"].append(float(np.sqrt(np.mean((pid[ok, j] - gt[ok, j]) ** 2))))
    c["cells_rmse"] = rm
    ctl[s] = c
    rows[s] = {"ego": bs(D["ego"]["ade6"], D["own"]["ade6"], eid), "egoshuf": bs(D["egoshuf"]["ade6"], D["own"]["ade6"], eid), "gt": bs(D["gt"]["ade6"], D["own"]["ade6"], eid)}
    rows[s]["ade6_means"] = {m: float(np.mean(D[m]["ade6"])) for m in MODES}
    rows[s]["ade2_ego_minus_own"] = bs(D["ego"]["ade2"], D["own"]["ade2"], eid)
    # beside, post-hoc, not registered: plan rule-violation rates against the GT-derived constraints (dump keys viol_*)
    vh = {}
    for v in VIOL:
        vh[v] = {"ego_minus_own": bs(D["ego"][v].astype(np.float64), D["own"][v].astype(np.float64), eid),
                 "gt_minus_own": bs(D["gt"][v].astype(np.float64), D["own"][v].astype(np.float64), eid),
                 "rate_own": float(np.mean(D["own"][v])), "rate_ego": float(np.mean(D["ego"][v])), "rate_gt": float(np.mean(D["gt"][v]))}
    rows[s]["viol_post_hoc"] = vh
    R["seeds"][str(s)] = {"controls_recomputed": c, "rows": rows[s], "n_windows": int(len(eid)), "n_episodes": int(len(np.unique(eid)))}

# S11 / S12 / S13 from the banked fit.json (the read-out was fitted before any eval batch; this run does not re-fit)
fit = json.load(open(SR + "fit/fit.json", encoding="utf-8"))
mean_rmse = {k: float(np.mean(fit[k]["holdout_rmse_sd"])) for k in ("lin", "mlp", "lin_ego", "mlp_ego")}
R["S13_mean_fold0_holdout_rmse_sd"] = mean_rmse
R["S13_argmin"] = min(mean_rmse, key=mean_rmse.get)
R["S11_train_parity_ok_fit_json"] = bool(fit["train_parity_ok"])
R["S12_eval_train_disjoint_fit_json"] = bool(fit["eval_train_disjoint"])
R["fit_json_readout_sha256"] = fit["readout_sha256"]
R["readout_pt_sha256_file"] = sha(SR + "fit/readout.pt")
R["fit_json_sha256_file"] = sha(SR + "fit/fit.json")

# PREREG_V8F_SPEEDSUB section 4 tolerances, section 5 table. Written out here from the PREREG's text.
tol_ok = {}
for s in (0, 1):
    c = ctl[s]
    tol_ok[s] = {
        "S1": c["S1_identity_eq_own"], "S2": c["S2_nonspeed_max_abs_dev_in_hooked_dumps"] == 0.0 and c["S2_own_std_equal_across_modes"], "S3": c["S3_tick_mirror_ok_in_hooked_dumps"],
        "S4": c["S4_eid_aligned"], "S5": c["S5_max_abs_dev_own_vs_speedread_own"] <= 1e-5, "S6": c["S6_max_abs_dev_gt_vs_speedread_gt"] <= 1e-4,
        "S7": c["S7_max_abs_dev_ego_used_vs_identity_pred_std"] <= 1e-5, "S7b": c["S7b_max_abs_dev_identity_pred_vs_speedread_banked"] <= 1e-5,
        "S8": c["S8_readout_sha_in_dumps"] and R["readout_pt_sha256_file"] == RO_SHA == R["fit_json_readout_sha256"],
        "S9": c["S9_perm_sha256"] == c["S9_perm_sha256_in_dump"] and c["S9_alignment_max_abs_dev_egoshuf_used_vs_ego_used_permuted"] <= 1e-6
        and c["S9_n_fixed_points"] <= max(3, 0.01 * N_WIN),
        "S10": c["S10_frac_ego_cells_moved_vs_own"] >= 0.99 and c["S10_frac_egoshuf_differs_from_ego"] >= 0.95,
    }
ctrl_all = all(all(v.values()) for v in tol_ok.values()) and R["S13_argmin"] == "mlp_ego" and R["S11_train_parity_ok_fit_json"] and R["S12_eval_train_disjoint_fit_json"]
R["controls_tolerance_checks"] = {str(s): tol_ok[s] for s in tol_ok}
R["controls_all_pass_recomputed"] = bool(ctrl_all)


def decide(rws, controls_ok):
    if not controls_ok or sorted(rws) != [0, 1]:
        return "REFUSED"
    if not all(sep_worse(rws[s]["egoshuf"]) for s in (0, 1)):
        return "REFUSED"
    if any(sep_worse(rws[s]["ego"]) for s in (0, 1)):
        return "HARMS"
    if all(sep_better(rws[s]["ego"]) and rws[s]["ego"]["delta"] <= -F_PASS for s in (0, 1)):
        return "PASS"
    return "NULL"


R["decision_rounded_delta"] = decide(rows, ctrl_all)
R["decision_unrounded_point"] = decide({s: {**rows[s], "ego": {**rows[s]["ego"], "delta": rows[s]["ego"]["unrounded_point"]}} for s in rows}, ctrl_all)

# every recomputed row equals SUBREAD.json to 4 dp
mism = []
for s in (0, 1):
    for k in ("ego", "egoshuf", "gt"):
        for q in ("delta", "lo", "hi"):
            if round(rows[s][k][q], 4) != round(SUB["rows"][str(s)][k][q], 4):
                mism.append((s, k, q, rows[s][k][q], SUB["rows"][str(s)][k][q]))
    for j in range(5):
        for who in ("T_own", "mlp_ego"):
            if abs(ctl[s]["cells_rmse"][who][j] - SUB["cells"][str(s)][who]["rmse"][j]) > 1e-4:
                mism.append((s, "cells", who, j, ctl[s]["cells_rmse"][who][j], SUB["cells"][str(s)][who]["rmse"][j]))
        if ctl[s]["cells_rmse"]["n_present"][j] != SUB["cells"][str(s)]["n_present"][j]:
            mism.append((s, "n_present", j))
R["mismatches_vs_SUBREAD_json"] = mism
R["equals_SUBREAD_json_outcome"] = (R["decision_rounded_delta"] == SUB["outcome"])
json.dump(R, open(OUT + "REDERIVE.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("controls_all_pass_recomputed", ctrl_all)
for s in (0, 1):
    print("seed", s, {k: (rows[s][k]["delta"], rows[s][k]["lo"], rows[s][k]["hi"]) for k in ("ego", "egoshuf", "gt")}, "ade6 means", {m: round(v, 4) for m, v in rows[s]["ade6_means"].items()})
print("decision (rounded delta):", R["decision_rounded_delta"], "| (unrounded point):", R["decision_unrounded_point"], "| SUBREAD.json:", SUB["outcome"])
print("mismatches vs SUBREAD.json:", len(mism), mism[:5])
assert not mism and R["equals_SUBREAD_json_outcome"] and ctrl_all and R["decision_unrounded_point"] == R["decision_rounded_delta"]
print("REDERIVE OK")
