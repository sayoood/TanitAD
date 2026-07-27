"""C2 — the shipped rule against the PUBLISHED 881-window numbers, both directions.

Primary artifacts, tracked in this repo, read in place and never modified:
  `…/incoming/2026-07-26-v5-imagination-selection/raw/v5_{v1,v4}_windows_reduced.pt`
  `…/incoming/2026-07-27-canary-proxy/raw/canary_proxy.json`

881 canonical val windows / 40 episode clusters. Estimator: paired episode-cluster
bootstrap (`taniteval.ci`, B=2000, unit = episode). NEVER `overlapping_holdout_se`.

⚠️ **BOTH DIRECTIONS ARE PINNED HERE ON PURPOSE.** The identical rule is
separated-BETTER under v1's world model (-0.2918) and separated-WORSE when v4
scores its own fan (+0.2090). A file that pinned only the win would be a guard
that cannot fail, and would let the default be flipped without contradiction.

⛔ **AND THE THIRD TEST IS THE ONE THAT MATTERS FOR RELAY.** The figure
`-0.3366 / 0.5196` has travelled in briefs labelled "UNGATED, on 100 % of
windows". It is not: it is `learned_gate_ALL_ridge_tau0`, a FITTED gate firing on
**66.97 %** of windows over a 73-feature bank that includes 2-world-model
ensemble features. `test_the_published_headline_is_a_GATE_not_this_rule` asserts
that from the primary JSON so the conflation cannot travel again.
"""
import json
import os

import numpy as np
import torch

from taniteval.ci import paired_episode_cluster_bootstrap
from tanitad.models import wm_reference_select as W

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HUB = os.path.join(_REPO, "TanitAD Research Hub", "Architecture & Inference",
                    "Implementation", "incoming")
V5 = os.path.join(_HUB, "2026-07-26-v5-imagination-selection", "raw")
CANARY = os.path.join(_HUB, "2026-07-27-canary-proxy", "raw", "canary_proxy.json")

N_BOOT, SEED = 2000, 20260727

#: The published cells. Changing a digit here without changing the artifact makes
#: this file red — which is the point.
PUBLISHED = {
    "as_trained_ade_0_2s": 0.8563,
    "v1": {"ade_0_2s": 0.5645, "delta": -0.2918, "lo": -0.4233, "hi": -0.1598},
    "v4": {"ade_0_2s": 1.0653, "delta": +0.2090, "lo": +0.0550, "hi": +0.3642},
    "gate_headline": {"row": "learned_gate_ALL_ridge_tau0", "delta": -0.3366,
                      "policy_ade_0_2s": 0.5196, "selected_frac": 0.6697},
}


def _load(tag):
    p = os.path.join(V5, f"v5_{tag}_windows_reduced.pt")
    assert os.path.exists(p), (
        f"primary artifact missing: {p}. It is git-tracked; without it the "
        "shipped C2 numbers have no source and must not be quoted.")
    return torch.load(p, map_location="cpu", weights_only=False)


def _arm(tag):
    d = _load(tag)
    cost = d["costs"]["C2_wm_ref_proximity"]                      # [881, 256]
    ep = d["ep"].numpy()
    fan_err4 = d["fan_err4"].numpy().astype(np.float64)            # ORACLE readout
    ar = np.arange(len(ep))
    # Route the PUBLISHED cost matrix through the SHIPPED selector rather than a
    # bare `argmin`, so the candidate-axis guard, the tie/degeneracy telemetry and
    # the mask path are all exercised on real data. The reduced dump carries no
    # fan and no reference roll, so the geometry is reconstructed to be exactly
    # cost-equivalent: with ref at the origin and the same offset at TWO lead
    # times, `mean_k ||fan - ref||` = |fan_x| = cost (costs are >= 0).
    # ⚠️ TWO steps, not one: at a single lead time `mean` and `sum` coincide, so
    # the reconstruction could not detect a changed reducer. The red/green sweep
    # caught exactly that — this line is the fix, and it is why the sweep exists.
    assert bool((cost >= 0).all())
    fan = torch.zeros(cost.shape[0], cost.shape[1], 2, 2)
    fan[:, :, 0, 0] = cost
    fan[:, :, 1, 0] = cost
    ref = torch.zeros(cost.shape[0], 2, 2)
    pick, cost_rt, tele = W.select_by_wm_reference(
        fan, ref, baseline_idx=d["picks"]["A0_as_trained"], scorer=tag)
    assert torch.allclose(cost_rt, cost, atol=0, rtol=0)
    return {
        "ep": ep, "cost": cost, "tele": tele,
        "pick": pick.numpy(), "published_pick": d["picks"]["C2_wm_ref_proximity"].numpy(),
        "a0_err": fan_err4[ar, d["picks"]["A0_as_trained"].numpy()],
        "c2_err": fan_err4[ar, pick.numpy()],
    }


def test_shipped_argmin_reproduces_the_published_picks_on_both_arms():
    for tag in ("v1", "v4"):
        a = _arm(tag)
        n = int((a["pick"] == a["published_pick"]).sum())
        assert n == len(a["pick"]) == 881, f"{tag}: {n}/881 picks matched"


def test_shipped_rule_reproduces_the_published_policy_values():
    for tag in ("v1", "v4"):
        a = _arm(tag)
        assert round(float(a["a0_err"].mean()), 4) == PUBLISHED["as_trained_ade_0_2s"]
        assert round(float(a["c2_err"].mean()), 4) == PUBLISHED[tag]["ade_0_2s"]


def test_paired_interval_reproduces_BOTH_directions():
    for tag in ("v1", "v4"):
        a = _arm(tag)
        ci = paired_episode_cluster_bootstrap(a["c2_err"], a["a0_err"], a["ep"],
                                              n_boot=N_BOOT, seed=SEED)
        want = PUBLISHED[tag]
        assert ci["estimator"] == "paired_episode_cluster_bootstrap"
        assert ci["n_episodes"] == 40 and ci["n_windows"] == 881
        assert round(ci["delta"], 4) == want["delta"], (tag, ci)
        assert round(ci["lo"], 4) == want["lo"] and round(ci["hi"], 4) == want["hi"]
        assert ci["separated"] is True
    # ...and the two directions really are opposite signs, not a copy of one row
    assert PUBLISHED["v1"]["delta"] < 0 < PUBLISHED["v4"]["delta"]


def test_the_published_headline_is_a_GATE_not_this_rule():
    """-0.3366 is NOT the ungated rule this module ships. Asserted from the JSON."""
    with open(CANARY, encoding="utf-8") as f:
        R = json.load(f)
    v1 = R["stage3_learned_gates_oof"]["arms"]["v1"]
    row = v1[PUBLISHED["gate_headline"]["row"]]
    g = PUBLISHED["gate_headline"]
    assert round(row["delta"], 4) == g["delta"]
    assert round(row["policy_ade_0_2s"], 4) == g["policy_ade_0_2s"]
    assert round(row["selected_frac"], 4) == g["selected_frac"]
    assert g["selected_frac"] < 1.0, "a gate that fires on 100 % is not a gate"
    # the ungated reference in the same JSON is the cell this module ships
    ung = R["stage1_ungated_baseline_and_ceilings"]["arms"]["v1"]["UNGATED_C2_everywhere"]
    assert ung["selected_frac"] == 1.0
    assert round(ung["delta"], 4) == PUBLISHED["v1"]["delta"]
    assert row["delta"] < ung["delta"], (
        "the gated row is better than ungated — so quoting it as the ungated "
        "number OVERSTATES the shipped rule, which is the C19 relay failure")


def test_MEASURED_ARMS_in_code_matches_the_primary_json():
    """Cross-file guard: the constants shipped in `stack/` must equal the JSON."""
    with open(CANARY, encoding="utf-8") as f:
        R = json.load(f)
    for tag, key in (("v1", "v1_scores_v4_fan"), ("v4", "v4_scores_its_own_fan")):
        row = R["stage1_ungated_baseline_and_ceilings"]["arms"][tag]["UNGATED_C2_everywhere"]
        m = W.MEASURED_ARMS[key]
        assert round(row["delta"], 4) == m["paired_delta"], key
        assert round(row["lo"], 4) == m["lo"] and round(row["hi"], 4) == m["hi"]
        assert row["policy_ade_0_2s"] == m["ade_0_2s"]
        assert row["selected_frac"] == m["selected_frac"] == 1.0
        assert row["separated"] is m["separated"]
        assert m["better"] == (m["paired_delta"] < 0)
    a0 = R["stage1_ungated_baseline_and_ceilings"]["A0_as_trained_ade_0_2s"]
    assert a0 == PUBLISHED["as_trained_ade_0_2s"]
    for m in W.MEASURED_ARMS.values():
        assert m["as_trained_ade_0_2s"] == a0


def test_the_rule_is_not_degenerate_on_the_real_cost_matrices():
    """The check whose absence would have written up two pure-noise gates."""
    for tag in ("v1", "v4"):
        t = _arm(tag)["tele"]
        assert t["n_tied_argmin"] == 0, (
            f"{tag}: tied argmins exist, so the pick depends on row order — the "
            "stable-argsort-over-ties class")
        assert t["n_constant_cost_rows"] == 0
        assert t["n_candidates"] == 256 and t["n_windows"] == 881
        assert t["selected_frac"] == 1.0
        assert t["cost_span_mean"] > 0.0
        # the rule must actually move the pick, or "it helps" means nothing
        assert t["frac_pick_equals_baseline"] < 0.2
        assert t["n_distinct_picks"] > 1
