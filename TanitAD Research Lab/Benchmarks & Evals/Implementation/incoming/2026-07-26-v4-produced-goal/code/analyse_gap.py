"""ORACLE vs PRODUCED vs NEUTRAL — bit-identity proof + the paired gap.

Estimator: `taniteval.ci.paired_episode_cluster_bootstrap` over the 40 val
episodes, named on every interval. `overlapping_holdout_se` is NEVER used.
Decomposed lateral (cross-track) / longitudinal (along-track) via
`taniteval.lateral.decompose`.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")

from taniteval import ci as _ci          # noqa: E402
from taniteval import lateral as _lat    # noqa: E402

RES = "/root/v4eval/results_goalmode"
BASE = "/root/v4eval/results/windows_flagship-v4-fromscratch-15k.pt"
N_BOOT = 2000
DT_SPARSE = 0.5          # 4-waypoint surface: knot j is j*0.5 s, NOT j*0.1 s


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def tensor_md5(t):
    return hashlib.md5(np.ascontiguousarray(
        t.detach().cpu().numpy()).tobytes()).hexdigest()


def per_window_ade(pred, gt):
    """[N,4,2] -> [N] mean over the 4 waypoints of the L2 error. This is exactly
    `ade_0_2s`'s per-window component (taniteval.driving tier-0)."""
    return (pred - gt).norm(dim=-1).mean(dim=-1).numpy()


def axis_components(pred, gt, mode="ego"):
    """-> (|along| [N,4], |cross| [N,4]) via taniteval.lateral.decompose."""
    a, c = _lat.decompose(_lat._as3(pred, "pred"), _lat._as3(gt, "gt"), mode)
    return a.abs(), c.abs()


out = {
    "title": "flagship-v4-fromscratch @ 15000 — goal-oracle vs produced-goal",
    "evidence_class": "MEASURED (ours; artifacts in results_goalmode/)",
    "estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py), "
                 f"B={N_BOOT}, 40 val episodes, percentile CI, alpha=0.05. "
                 "overlapping_holdout_se is NOT used anywhere in this file.",
    "surface_note": ("the persisted v4 windows are the 4-WAYPOINT surface "
                     "(steps 5/10/15/20 = 0.5/1.0/1.5/2.0 s). Knot j is "
                     "j*0.5 s. taniteval.lateral.paired_cross_track stamps "
                     "horizon_s = step*0.1 unconditionally, which mislabels "
                     "this surface (knot 4 -> '0.4 s', truly 2.0 s); the "
                     "horizons below are RE-STAMPED at 0.5 s spacing."),
}

W = {m: load(f"{RES}/windows_v4-15k-goal-{m}.pt")
     for m in ("oracle", "produced", "neutral")}
base = load(BASE)

# ---------------------------------------------------------------- 1. BIT-ID --
o = W["oracle"]
bi = {}
for k in ("pred", "gt", "cv", "speed", "head_deg"):
    a, b = o[k], base[k]
    bi[k] = {"shape_a": list(a.shape), "shape_b": list(b.shape),
             "bit_identical": bool(a.shape == b.shape and torch.equal(a, b)),
             "md5_new": tensor_md5(a), "md5_baseline": tensor_md5(b),
             "max_abs_diff": (float((a.float() - b.float()).abs().max())
                              if a.shape == b.shape else None)}
bi["eid"] = {"identical": bool(list(o["eid"]) == list(base["eid"])),
             "n": len(o["eid"])}
bi["wp_steps"] = {"new": list(o["wp_steps"]), "baseline": list(base["wp_steps"])}
bi["_baseline"] = BASE
bi["_baseline_note"] = ("written 2026-07-25 22:07Z by the PRE-SYNC, "
                        "PRE---goal-mode eval_flagship_v4.py. `pred` "
                        "bit-identity proves the --goal-mode edit did not move "
                        "the evaluated forward pass.")
bi["ALL_BIT_IDENTICAL"] = bool(
    all(v["bit_identical"] for k, v in bi.items()
        if isinstance(v, dict) and "bit_identical" in v)
    and bi["eid"]["identical"])
out["bit_identity_oracle_vs_prepublished_baseline"] = bi

# --------------------------------------------------------- 2. point + CI ----
eid = [str(x) for x in o["eid"]]
gt = o["gt"]
ade = {m: per_window_ade(W[m]["pred"], W[m]["gt"]) for m in W}
# same windows, same gt -> assert it
for m in W:
    assert torch.equal(W[m]["gt"], gt), f"gt differs for {m}"
    assert [str(x) for x in W[m]["eid"]] == eid, f"eid differs for {m}"

out["ade_0_2s_by_goal_mode"] = {
    m: _ci.episode_cluster_bootstrap(ade[m], eid, n_boot=N_BOOT, seed=0)
    for m in ("oracle", "produced", "neutral")}
out["cv_baseline_ade_0_2s"] = _ci.episode_cluster_bootstrap(
    per_window_ade(o["cv"], gt), eid, n_boot=N_BOOT, seed=0)

# ---------------------------------------------------------- 3. paired gaps --
def paired(a, b, reduce="mean"):
    return _ci.paired_episode_cluster_bootstrap(ade[a], ade[b], eid,
                                                n_boot=N_BOOT, seed=0,
                                                reduce=reduce)

out["paired_gaps_ade_0_2s"] = {
    "produced_minus_oracle": {
        **paired("produced", "oracle"),
        "_read": "THE SIZE OF THE GOAL PRIVILEGE on our own arm. Positive = "
                 "the produced-goal (deployable) arm is WORSE than the "
                 "goal-oracle arm, i.e. the oracle was worth this many metres."},
    "neutral_minus_oracle": {
        **paired("neutral", "oracle"),
        "_read": "the gap if the goal is withheld entirely — the ceiling on "
                 "what any produced goal can cost."},
    "produced_minus_neutral": {
        **paired("produced", "neutral"),
        "_read": "is the produced goal worth more than NO goal? NEGATIVE = "
                 "yes, the model's own goal helps."},
    "produced_minus_oracle_p90": {
        **paired("produced", "oracle", reduce="p90"),
        "_read": "the same gap in the tail (p90 of per-window ADE)."},
}

# ------------------------------------------------- 4. lateral / longitudinal -
dec = {}
for m in W:
    al, cr = axis_components(W[m]["pred"], gt)
    dec[m] = {"along": al, "cross": cr}

K = gt.shape[1]
axis_out = {}
for axis in ("along", "cross"):
    per_h = {}
    for j in range(K):
        d = _ci.paired_episode_cluster_bootstrap(
            dec["produced"][axis][:, j].numpy(),
            dec["oracle"][axis][:, j].numpy(), eid, n_boot=N_BOOT, seed=0)
        d["horizon_s"] = round((j + 1) * DT_SPARSE, 2)     # RE-STAMPED
        d.pop("estimator", None)
        per_h[f"{(j + 1) * DT_SPARSE:g}s"] = d
    # horizon-mean of the axis magnitude (the axis analogue of ade_0_2s)
    mean_p = dec["produced"][axis].mean(dim=-1).numpy()
    mean_o = dec["oracle"][axis].mean(dim=-1).numpy()
    axis_out[axis] = {
        "paired_produced_minus_oracle_meanhorizon":
            _ci.paired_episode_cluster_bootstrap(mean_p, mean_o, eid,
                                                 n_boot=N_BOOT, seed=0),
        "oracle_meanhorizon":
            _ci.episode_cluster_bootstrap(mean_o, eid, n_boot=N_BOOT, seed=0),
        "produced_meanhorizon":
            _ci.episode_cluster_bootstrap(mean_p, eid, n_boot=N_BOOT, seed=0),
        "by_horizon_paired": per_h,
    }
# energy share: which axis carries the squared error, per mode
for m in W:
    a2 = float((dec[m]["along"] ** 2).sum())
    c2 = float((dec[m]["cross"] ** 2).sum())
    axis_out.setdefault("energy_share_longitudinal", {})[m] = round(
        a2 / (a2 + c2), 4)
axis_out["_read"] = ("along = longitudinal (forward), cross = lateral. "
                     "taniteval.lateral.decompose(mode='ego'). The paired "
                     "deltas are produced - oracle, so POSITIVE = the produced "
                     "arm is worse on that axis.")
out["lateral_longitudinal_decomposition"] = axis_out

# ------------------------------------------- 5. the lateral BLOCK now runs ---
lb = {}
for m in ("oracle", "produced"):
    try:
        b = _lat.from_sparse_windows(W[m], n_boot=500)
        lb[m] = {"surface": b.get("surface"), "dt_s": b.get("dt_s"),
                 "horizon_s": b.get("horizon_s"),
                 "energy_share": b.get("energy_share"),
                 "growth": b.get("growth"),
                 "cross_peak_mean": b["dense_aggregate"]["cross_peak"]["mean"],
                 "cross_peak_p90": b["dense_aggregate"]["cross_peak_p90"]["mean"],
                 "axis_check": b.get("axis_check"),
                 "verdict": b.get("verdict")}
    except Exception as e:
        lb[m] = {"ERROR": f"{type(e).__name__}: {e}"}
# and prove the DENSE block's own refusal is a clean, informative skip
try:
    d = _lat.block(W["oracle"], n_boot=100)
    lb["dense_block_on_v4_windows"] = {
        "skipped": d.get("skipped"), "ran": not d.get("skipped")}
except Exception as e:
    lb["dense_block_on_v4_windows"] = {"ERROR": f"{type(e).__name__}: {e}"}
out["lateral_block"] = lb

# ------------------------------------------------ 6. goal-quality readout ----
for m in ("produced",):
    try:
        j = json.load(open(f"{RES}/v4-15k-goal-{m}.json"))
        out["produced_goal_quality"] = \
            j["goal_provenance"].get("goal_agreement_vs_oracle")
    except Exception as e:
        out["produced_goal_quality"] = f"{type(e).__name__}: {e}"

# --- 6b. produced route vs the MAJORITY baseline (the `nonav_route_beats_
# --- majority` KILL secondary, which eval_flagship_v4 reports as null "NOT
# --- REACHABLE"). The produced route CAN now be scored against it.
try:
    conf = np.array(out["produced_goal_quality"]
                    ["route_confusion_oracle_rows_x_produced_cols"])[:4, :4]
    n_all = int(conf.sum())
    # the produced head can only emit 0/1/2 -> every ROUTE_UNKNOWN(3) oracle
    # window is unscoreable, not merely wrong. Score on judgeable windows only.
    judge = conf[:3, :]
    n_j = int(judge.sum())
    corr_j = int(np.trace(judge[:, :3]))
    maj_j = int(judge.sum(axis=1).max())
    out["produced_route_vs_majority"] = {
        "_read": ("the KILL secondary `nonav_route_beats_majority`, which the "
                  "harness reports as null/'NOT REACHABLE'. It IS reachable "
                  "against the produced route. Scored on windows where the "
                  "ORACLE route is a real judgement (class 0/1/2): the produced "
                  "head cannot emit ROUTE_UNKNOWN, so oracle-UNKNOWN windows "
                  "are unscoreable, not wrong."),
        "n_windows_all": n_all,
        "n_windows_oracle_judgeable": n_j,
        "produced_accuracy_judgeable": round(corr_j / n_j, 4) if n_j else None,
        "majority_class_accuracy_judgeable": round(maj_j / n_j, 4) if n_j else None,
        "beats_majority_by_pp": round(100 * (corr_j - maj_j) / n_j, 2) if n_j else None,
        "produced_predicted_class_share": {
            str(c): round(float(conf[:, c].sum()) / n_all, 4) for c in range(4)},
        "oracle_class_share": {
            str(c): round(float(conf[c, :].sum()) / n_all, 4) for c in range(4)},
        "beats_majority": bool(n_j and corr_j > maj_j),
    }
except Exception as e:
    out["produced_route_vs_majority"] = f"{type(e).__name__}: {e}"

p = f"{RES}/GOAL_MODE_GAP.json"
json.dump(out, open(p, "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
print("\n->", p)
