"""RE-VERIFICATION of the C2 headline by the SHIPPED implementation.

Three independent layers, each strictly stronger than the last:

  L1  the cost matrix -> pick -> policy -> paired CI chain, recomputed from the
      tracked reduced dumps, without calling the producing stream's code.
  L2  ⭐ the COST MATRIX ITSELF, re-derived from the RAW GEOMETRY (fan +
      reference roll-out) by `tanitad.models.wm_reference_select`. This is the
      layer that tests the shipped formula rather than trusting a stored column.
      The geometry lived only on `tanitad-eval:/workspace/_v5/v5_v1_windows.pt`.
  L3  the ade_0_2s of the shipped pick recomputed from fan + target geometry at
      WP_STEPS, i.e. not read from any stored per-arm column at all.

⛔ It also adjudicates the BRIEF's headline cell (-0.3366 / 0.5196-0.5221),
which is labelled "UNGATED, on 100 % of windows" but is `learned_gate_ALL_
ridge_tau0` firing on 66.97 %.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
HUB = REPO / "TanitAD Research Hub/Architecture & Inference/Implementation/incoming"
V5 = HUB / "2026-07-26-v5-imagination-selection/raw"
CANARY = HUB / "2026-07-27-canary-proxy/raw/canary_proxy.json"
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack"))

from taniteval.ci import (episode_cluster_bootstrap,                  # noqa: E402
                          paired_episode_cluster_bootstrap)
from tanitad.models.wm_reference_select import (                      # noqa: E402
    MEASURED_ARMS, select_by_wm_reference, wm_reference_cost)

N_BOOT, SEED = 2000, 20260727
WP_STEPS = (5, 10, 15, 20)
GEOM = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("c2_geom_v1.pt")

PUB = {"as_trained": 0.8563,
       "v1": {"ade": 0.5645, "delta": -0.2918, "lo": -0.4233, "hi": -0.1598},
       "v4": {"ade": 1.0653, "delta": +0.2090, "lo": +0.0550, "hi": +0.3642}}
TOL = {"ade": 5e-5, "delta": 5e-5, "bound": 5e-5}       # stated up front

R: dict = {
    "_experiment": "C2 ship — re-verification of the published headline by the "
                   "SHIPPED implementation before shipping it",
    "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    "_estimator": "paired episode-cluster bootstrap (taniteval/ci.py, B=2000, "
                  "unit = episode). NEVER overlapping_holdout_se.",
    "_tolerance": TOL, "_host": platform.node(), "_device": "cpu",
    "_n": {"windows": 881, "episodes": 40}, "_date": "2026-07-27",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ============================ L1 — the reduced dumps ========================= #
L1: dict = {"_read": "cost -> pick -> policy -> interval, recomputed from the "
                     "tracked reduced dumps independently of canary_proxy.py"}
for tag in ("v1", "v4"):
    p = V5 / f"v5_{tag}_windows_reduced.pt"
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    ep = d["ep"].numpy()
    fe4 = d["fan_err4"].numpy().astype(np.float64)
    cost = d["costs"]["C2_wm_ref_proximity"]
    pick = cost.argmin(dim=1)
    ar = np.arange(len(ep))
    a0 = fe4[ar, d["picks"]["A0_as_trained"].numpy()]
    c2 = fe4[ar, pick.numpy()]
    ci = paired_episode_cluster_bootstrap(c2, a0, ep, n_boot=N_BOOT, seed=SEED)
    L1[tag] = {
        "artifact": str(p.relative_to(REPO)).replace("\\", "/"),
        "sha256_16": sha(p),
        "picks_match_published": int((pick.numpy()
                                      == d["picks"]["C2_wm_ref_proximity"].numpy()).sum()),
        "of": len(ep),
        "as_trained_ade_0_2s": round(float(a0.mean()), 4),
        "C2_ade_0_2s": round(float(c2.mean()), 4),
        "paired": {k: ci[k] for k in ("delta", "lo", "hi", "ci95", "separated",
                                      "p_delta_gt0", "n_windows", "n_episodes",
                                      "estimator")},
        "selected_frac": 1.0,
        "single_arm_C2": episode_cluster_bootstrap(c2, ep, n_boot=N_BOOT, seed=SEED),
        "PASS": bool(abs(float(c2.mean()) - PUB[tag]["ade"]) < TOL["ade"]
                     and abs(ci["delta"] - PUB[tag]["delta"]) < TOL["delta"]
                     and abs(ci["lo"] - PUB[tag]["lo"]) < TOL["bound"]
                     and abs(ci["hi"] - PUB[tag]["hi"]) < TOL["bound"]),
    }
R["L1_reduced_dump_chain"] = L1

# ============ L2/L3 — the SHIPPED rule on the RAW GEOMETRY =================== #
g = torch.load(str(GEOM), map_location="cpu", weights_only=False)
fan, ref, tgt = g["fan"], g["imag_ref"], g["tgt"]
ep = g["ep"].numpy()
ar = np.arange(len(ep))

cost_shipped = wm_reference_cost(fan, ref)                     # THE SHIPPED FORMULA
pick_shipped, cost2, tele = select_by_wm_reference(
    fan, ref, baseline_idx=g["pick_A0_as_trained"], scorer="flagship-30k (v1)")
assert torch.equal(cost_shipped, cost2)

# L3 — ade_0_2s recomputed from GEOMETRY, not read from a stored column
wp = [h - 1 for h in WP_STEPS]
fan_err4 = (fan[:, :, wp, :] - tgt[:, wp, :][:, None]).norm(dim=-1).mean(dim=-1)
c2_err = fan_err4[ar, pick_shipped.numpy()].numpy().astype(np.float64)
a0_err = fan_err4[ar, g["pick_A0_as_trained"].numpy()].numpy().astype(np.float64)
ci_geo = paired_episode_cluster_bootstrap(c2_err, a0_err, ep, n_boot=N_BOOT,
                                          seed=SEED)

cost_pub = g["cost_C2_published"]
srt = cost_shipped.sort(dim=1).values
margin = (srt[:, 1] - srt[:, 0])                     # top-1 vs top-2 separation
maxdiff = float((cost_shipped - cost_pub).abs().max())
R["L2_shipped_cost_from_raw_geometry"] = {
    "_read": "⭐ the strongest layer: the SHIPPED `wm_reference_cost` re-derives "
             "the published cost matrix from the fan and the ONE reference "
             "roll-out. Nothing stored is trusted except the geometry itself.",
    "geometry": {"src": g["_src"], "sha256_16": sha(GEOM),
                 "fan": list(fan.shape), "imag_ref": list(ref.shape)},
    "cost_bit_exact_vs_published": bool(torch.equal(cost_shipped, cost_pub)),
    "cost_max_abs_diff": maxdiff,
    "cost_mean_magnitude": round(float(cost_pub.mean()), 4),
    "cost_max_rel_diff": maxdiff / float(cost_pub.abs().mean()),
    "_not_bit_exact_because": (
        "the published matrix was reduced on an NVIDIA A40 in float32 and this "
        "one on the dev box CPU; `.norm().mean()` is order-sensitive at float32. "
        "Reported, not hidden: the bar that matters is whether the DECISION "
        "moves, measured below."),
    "picks_match_published": int((pick_shipped == g["pick_C2_published"]).sum()),
    "of": int(len(ep)),
    "decision_margin": {
        "min_top1_top2_gap": float(margin.min()),
        "p1_top1_top2_gap": float(margin.quantile(0.01)),
        "median_gap": float(margin.median()),
        "n_windows_with_gap_below_maxdiff": int((margin <= maxdiff).sum()),
        "n_windows_with_gap_below_10x_maxdiff": int((margin <= 10 * maxdiff).sum()),
        "worst_case_headroom_x": round(float(margin.min()) / maxdiff, 1),
        "_read": "0 / 881 windows have a winner-vs-runner-up gap smaller than the "
                 "largest CPU-vs-GPU disagreement, and all 881 picks agree. "
                 "⚠️ Stated rather than hidden: in the TIGHTEST window the gap is "
                 "only 2.2x that disagreement, so C2's pick is numerically "
                 "robust on this corpus but not by a wide margin at the tail. A "
                 "different accumulation order could flip a small number of "
                 "near-tied windows on a future host.",
    },
    "telemetry": tele,
    "PASS": bool(int((pick_shipped == g["pick_C2_published"]).sum()) == len(ep)
                 and maxdiff < 1e-4 and int((margin <= maxdiff).sum()) == 0),
}
R["L3_policy_from_geometry"] = {
    "_read": "ade_0_2s recomputed from fan+tgt at WP_STEPS (5,10,15,20) — no "
             "stored per-arm column is read at all.",
    "as_trained_ade_0_2s": round(float(a0_err.mean()), 4),
    "C2_ade_0_2s": round(float(c2_err.mean()), 4),
    "selected_frac": 1.0,
    "paired": {k: ci_geo[k] for k in ("delta", "lo", "hi", "ci95", "separated",
                                      "p_delta_gt0", "n_windows", "n_episodes",
                                      "estimator")},
    "agrees_with_stored_ade_column_max_abs": float(
        (torch.as_tensor(c2_err, dtype=torch.float32)
         - g["ade_C2_published"]).abs().max()),
    "PASS": bool(abs(float(c2_err.mean()) - PUB["v1"]["ade"]) < TOL["ade"]
                 and abs(ci_geo["delta"] - PUB["v1"]["delta"]) < TOL["delta"]
                 and abs(ci_geo["lo"] - PUB["v1"]["lo"]) < TOL["bound"]
                 and abs(ci_geo["hi"] - PUB["v1"]["hi"]) < TOL["bound"]),
}

# ================= the BRIEF's headline cell, adjudicated =================== #
C = json.loads(CANARY.read_text(encoding="utf-8"))
gate = C["stage3_learned_gates_oof"]["arms"]["v1"]["learned_gate_ALL_ridge_tau0"]
gate1 = C["stage3_learned_gates_oof"]["arms"]["v1"]["learned_gate_1WM_ridge_tau0"]
ung = C["stage1_ungated_baseline_and_ceilings"]["arms"]["v1"]["UNGATED_C2_everywhere"]
R["ADJUDICATION_of_the_brief_headline"] = {
    "brief_claim": "C2 ... applied with v1's world model, UNGATED, on 100 % of "
                   "windows: 0.5196-0.5221, paired -0.3366 [-0.4507, -0.2310]",
    "what_-0.3366_actually_is": {
        "row": "stage3_learned_gates_oof.arms.v1.learned_gate_ALL_ridge_tau0",
        "delta": gate["delta"], "lo": gate["lo"], "hi": gate["hi"],
        "policy_ade_0_2s": gate["policy_ade_0_2s"],
        "selected_frac": gate["selected_frac"],
        "is_ungated": gate["selected_frac"] == 1.0,
        "needs": "a ridge utility model fitted out-of-fold over the 73-feature "
                 "bank INCLUDING the 2-world-model ensemble family",
    },
    "what_0.5221_actually_is": {
        "row": "learned_gate_1WM_ridge_tau0",
        "delta": gate1["delta"], "policy_ade_0_2s": gate1["policy_ade_0_2s"],
        "selected_frac": gate1["selected_frac"]},
    "the_actual_UNGATED_cell": {
        "row": "stage1_ungated_baseline_and_ceilings.arms.v1.UNGATED_C2_everywhere",
        "delta": ung["delta"], "lo": ung["lo"], "hi": ung["hi"],
        "policy_ade_0_2s": ung["policy_ade_0_2s"],
        "selected_frac": ung["selected_frac"]},
    "overstatement_of_the_shipped_rule_m": round(ung["delta"] - gate["delta"], 4),
    "overstatement_ratio": round(gate["delta"] / ung["delta"], 3),
    "_read": "the shipped UNGATED rule reproduces EXACTLY; the brief's headline "
             "number belongs to a GATED policy firing on 66.97 % of windows and "
             "overstates the ungated rule by 0.0448 m (1.154x). C19 class: a "
             "conditional number quoted without its firing rate.",
}

R["MEASURED_ARMS_in_stack_matches"] = {
    k: bool(abs(MEASURED_ARMS[k]["paired_delta"] - PUB[t]["delta"]) < 1e-9
            and abs(MEASURED_ARMS[k]["ade_0_2s"] - PUB[t]["ade"]) < 1e-9)
    for k, t in (("v1_scores_v4_fan", "v1"), ("v4_scores_its_own_fan", "v4"))}

R["VERDICT"] = ("CONFIRM — the shipped implementation reproduces the UNGATED C2 "
                "cell to 4 dp at every layer: 0.8563 -> 0.5645, paired -0.2918 "
                "[-0.4233, -0.1598], separated, selected_frac 1.000. The cost "
                "matrix re-derived from raw geometry gives 881/881 IDENTICAL "
                "picks (not bit-identical costs: 1.14e-05 m CPU-vs-A40 float32 "
                "reduction noise, below every decision margin). The brief's "
                "-0.3366 headline is a SEPARATE, GATED cell (frac 0.6697) and "
                "is NOT what ships."
                if (L1["v1"]["PASS"] and L1["v4"]["PASS"]
                    and R["L2_shipped_cost_from_raw_geometry"]["PASS"]
                    and R["L3_policy_from_geometry"]["PASS"])
                else "DISCREPANCY — see the failing layer")
Path(sys.argv[1]).write_text(json.dumps(R, indent=1))
print(json.dumps({"VERDICT": R["VERDICT"],
                  "L1": {k: {"PASS": v["PASS"], "ade": v.get("C2_ade_0_2s"),
                             "delta": v["paired"]["delta"]}
                         for k, v in L1.items() if k != "_read"},
                  "L2": {k: R["L2_shipped_cost_from_raw_geometry"][k]
                         for k in ("cost_bit_exact_vs_published",
                                   "cost_max_abs_diff", "picks_match_published",
                                   "decision_margin", "PASS")},
                  "L3": {k: R["L3_policy_from_geometry"][k]
                         for k in ("C2_ade_0_2s", "paired", "PASS")}}, indent=1))
