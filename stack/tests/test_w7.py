"""W7 ``w7_roll_rerank`` pure-part smoke — CPU-only, no checkpoint, no
corpus, no GPU. The full path (frozen v5f trunk + W4 emission + WM roll on
the v2 corpus) is pod-side and NOT exercised here.

What is pinned (the brief's five groups):
  * COST ARITHMETIC hand-cases: ``w7_cost`` term-by-term against hand
    numbers, the imported ``kinematic_cost`` (mean|a| + 0.5*mean|jerk|) on a
    constant / known-jerk case, the progress sign, and shape rejection;
  * the ARGMAX-INCLUSION GUARANTEE: ``force_include`` inserts the frozen
    pick into the weakest slot only when absent, and with an oracle cost the
    W7 pick is never worse than the frozen pick (the exact form of the
    guarantee);
  * ROLL-CONSISTENCY on synthetic rolls: identical roll -> exactly 0, a
    known constant offset -> that offset;
  * the GATE JSON on every branch: pass / fail / threshold boundary
    (0.4505 inclusive, and its consistency with the 50 % definition), the
    tougher read vs the 0.4815 arm, the oracle-debug VERDICT REFUSAL, the
    verbatim reference numbers, tier stamp and estimator note;
  * PRUNING logic: frozen-rule top-K by scores, oracle-debug top-K by true
    error (lowest first), K clipping, missing-input rejection — plus the
    candidate-actions builder (steer = atan(2.9*kappa) encoding, history
    untouched, horizon guard) and the Spearman/cluster-bootstrap
    calibration helpers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from w7_roll_rerank import (GATE_FRAC, P7_GATE_RHO,  # noqa: E402
                            REF_FROZEN_ARGMAX, REF_ORACLE,
                            REF_RESCORER_TOP8_KINCOST, W7_GATE_THRESHOLD,
                            StateTap, build_w7_gate,
                            candidate_roll_actions,
                            cluster_bootstrap_spearman, force_include,
                            progress_arc_length, roll_consistency,
                            select_from_cost, shortlist_indices, spearman,
                            w7_cost)
from stage_a_probes import WHEELBASE  # noqa: E402
from tanitad.models.v58f import kinematic_cost  # noqa: E402

B, N, P, K = 3, 16, 5, 20


# ---------------------------------------------------------------------------
# cost arithmetic hand-cases
# ---------------------------------------------------------------------------
def test_kinematic_cost_hand_cases():
    """The imported v58f cost on hand numbers: constant a=2 -> mean|a|=2,
    jerk 0; a=[0,1] over dt=0.1 -> mean|a|=0.5, jerk=10 -> +0.5*10=5."""
    const = torch.full((1, 1, 4), 2.0)
    assert float(kinematic_cost(const)) == pytest.approx(2.0)
    ramp = torch.tensor([[[0.0, 1.0]]])
    assert float(kinematic_cost(ramp)) == pytest.approx(0.5 + 0.5 * 10.0)


def test_w7_cost_hand_case_and_weights():
    r = torch.tensor([[1.0, 2.0]])
    k = torch.tensor([[3.0, 1.0]])
    arc = torch.tensor([[10.0, 20.0]])
    c = w7_cost(r, k, arc, w_roll=1.0, w_kin=0.5, w_prog=0.1)
    assert torch.allclose(c, torch.tensor([[1.0 + 1.5 - 1.0,
                                            2.0 + 0.5 - 2.0]]))
    # defaults: progress OFF (w_prog=0) — arc must not move the cost
    c0 = w7_cost(r, k, arc)
    c1 = w7_cost(r, k, arc * 100)
    assert torch.allclose(c0, c1)
    # progress ON: longer arc -> lower cost (the negative-arc-length term)
    cp = w7_cost(r, k, arc, w_prog=0.1)
    cp_long = w7_cost(r, k, arc + 5.0, w_prog=0.1)
    assert (cp_long < cp).all()
    with pytest.raises(ValueError):
        w7_cost(r, k[:, :1], arc)                 # term shape mismatch


def test_progress_arc_length_hand_case():
    """Straight line 1 m/step for 4 steps (origin prepended) -> 4 m."""
    fan = torch.zeros(1, 1, 4, 2)
    fan[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert float(progress_arc_length(fan)) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        progress_arc_length(torch.zeros(4, 3))    # last dim != 2


# ---------------------------------------------------------------------------
# the argmax-inclusion guarantee
# ---------------------------------------------------------------------------
def test_force_include_inserts_only_when_absent():
    short = torch.tensor([[7, 3, 9], [1, 2, 3]])
    frozen = torch.tensor([3, 5])
    out = force_include(short, frozen)
    assert torch.equal(out[0], short[0])          # already present: untouched
    assert torch.equal(out[1], torch.tensor([1, 2, 5]))   # weakest slot swap
    assert (out == frozen[:, None]).any(dim=1).all()
    assert torch.equal(short, torch.tensor([[7, 3, 9], [1, 2, 3]]))  # no mut
    with pytest.raises(ValueError):
        force_include(short, frozen[:1])          # batch mismatch


def test_guarantee_oracle_cost_never_loses_to_frozen():
    """The exact form of the guarantee: with cost == true error on the
    shortlist, the W7 pick's error <= the frozen pick's error on EVERY
    window (the frozen pick is always available via force_include)."""
    torch.manual_seed(0)
    err = torch.rand(64, N)
    scores = torch.randn(64, N)
    frozen = torch.randint(0, N, (64,))
    short = force_include(shortlist_indices("frozen", P, scores=scores),
                          frozen)
    cost = err.gather(1, short)                   # oracle cost
    sel = select_from_cost(short, cost)
    ar = torch.arange(64)
    assert (err[ar, sel] <= err[ar, frozen] + 1e-9).all()
    # and the pick is exactly the shortlist oracle
    assert torch.allclose(err[ar, sel], err.gather(1, short).min(dim=1).values)


# ---------------------------------------------------------------------------
# roll-consistency
# ---------------------------------------------------------------------------
def test_roll_consistency_identical_is_zero():
    wp = torch.randn(B, P, 10, 2)
    out = roll_consistency(wp, wp.clone())
    assert out.shape == (B, P)
    assert torch.allclose(out, torch.zeros(B, P))


def test_roll_consistency_known_offset():
    wp = torch.randn(B, P, 10, 2)
    off = wp.clone()
    off[..., 1] += 0.5                            # constant +0.5 m lateral
    assert torch.allclose(roll_consistency(wp, off),
                          torch.full((B, P), 0.5), atol=1e-6)
    with pytest.raises(ValueError):
        roll_consistency(wp, wp[:, :, :-1])       # k mismatch


# ---------------------------------------------------------------------------
# gate JSON: every branch
# ---------------------------------------------------------------------------
def _mini(selected: float) -> dict:
    return {"n_windows": 881, "n_candidates": 256, "shortlist_k": 8,
            "grid": {"episodes": 40, "stride": 8, "batch": 16,
                     "expected_n": 881, "matches_banked_grid": True},
            "selected_ade": selected, "frozen_selected_ade_in_run": 0.7933,
            "oracle_ade_in_run": 0.1077, "shortlist_oracle_ade": 0.30,
            "sel_gap": round(selected - 0.1077, 6),
            "winner_hit_frac": 0.1, "sel_rank_pct_mean": 0.05,
            "frozen_in_shortlist_frac": 1.0, "sel_matches_frozen_frac": 0.2,
            "winner_in_shortlist_frac": 0.4,
            "w7_pick_le_frozen_pick_frac": 0.9, "families": {},
            "selgap_ci": "stub", "wallclock_s": 1.0}


_PRUNE = {"rule": "frozen", "topk": 8, "diagnostic_only": False,
          "surface": "stub", "argmax_inclusion": "stub"}
_COST = {"terms": {}, "selection": "stub", "_weights_note": "stub"}
_CALIB = {"across_windows_cost_vs_realised_error": {"spearman_rho": 0.5},
          "p7_reference": {}}
_ROLL = {"k": 10, "imagination_closed": True}


def _gate(selected, prune=None):
    return build_w7_gate(_mini(selected), prune=prune or _PRUNE,
                         cost_cfg=_COST, calib=_CALIB, roll=_ROLL)


def test_gate_pass_branch_and_fractions():
    g = _gate(0.40)
    gg = g["gate_W7_selgap_closed"]
    assert gg["pass"] is True
    assert gg["threshold_m"] == W7_GATE_THRESHOLD == 0.4505
    assert gg["frac_selgap_closed_vs_frozen_argmax"] == pytest.approx(
        (0.7933 - 0.40) / (0.7933 - 0.1077), abs=1e-6)
    t = g["tougher_read_vs_rescorer_top8_kincost"]
    assert t["frac_selgap_closed_vs_0.4815"] == pytest.approx(
        (0.4815 - 0.40) / (0.4815 - 0.1077), abs=1e-6)
    assert t["w7_beats_rescorer_arm"] is True


def test_gate_fail_branch():
    g = _gate(0.60)
    assert g["gate_W7_selgap_closed"]["pass"] is False
    t = g["tougher_read_vs_rescorer_top8_kincost"]
    assert t["w7_beats_rescorer_arm"] is False
    assert t["frac_selgap_closed_vs_0.4815"] < 0     # worse than the arm


def test_gate_threshold_boundary_inclusive_and_consistent():
    """0.4505 passes (<= threshold), and the threshold IS the 50 % point of
    the frozen gap: 0.7933 - 0.5*(0.7933-0.1077) == 0.4505 exactly."""
    assert REF_FROZEN_ARGMAX - GATE_FRAC * (REF_FROZEN_ARGMAX - REF_ORACLE) \
        == pytest.approx(W7_GATE_THRESHOLD, abs=1e-12)
    g = _gate(W7_GATE_THRESHOLD)
    assert g["gate_W7_selgap_closed"]["pass"] is True
    g2 = _gate(W7_GATE_THRESHOLD + 1e-4)
    assert g2["gate_W7_selgap_closed"]["pass"] is False


def test_gate_oracle_debug_refuses_verdict():
    prune = dict(_PRUNE, rule="oracle-debug", diagnostic_only=True)
    g = _gate(0.20, prune=prune)                  # would pass by number...
    gg = g["gate_W7_selgap_closed"]
    assert gg["pass"] is None                     # ...but verdict REFUSED
    assert "diagnostic only" in gg["verdict_refused_reason"]
    assert "UPPER BOUND" in gg["verdict_refused_reason"]


def test_gate_carries_references_tier_estimator_and_guarantee():
    g = _gate(0.40)
    r = g["reference"]
    assert r["frozen_argmax_selected_ade"] == REF_FROZEN_ARGMAX == 0.7933
    assert r["rescorer_top8_kincost_selected_ade"] == \
        REF_RESCORER_TOP8_KINCOST == 0.4815
    assert r["w4_oracle_ade"] == REF_ORACLE == 0.1077
    assert g["tier"] == "T0"
    assert "imagination-closed" in g["_tier_note"]
    assert "not GT teacher-forcing" in g["_tier_note"]
    assert "T1" in g["gate_W7_selgap_closed"]["rule"]   # prereg verbatim tier
    assert "EPISODE-CLUSTER BOOTSTRAP" in g["_estimator_note"]
    assert g["_evidence_class"].startswith("MEASURED")
    a = g["argmax_inclusion_guarantee"]
    assert "cannot lose by exclusion" in a["statement"]
    assert a["frozen_pick_in_shortlist_frac"] == 1.0
    import json
    json.dumps(g)                                 # JSON-serialisable as-is


# ---------------------------------------------------------------------------
# pruning logic
# ---------------------------------------------------------------------------
def test_shortlist_frozen_rule_topk_by_scores():
    scores = torch.arange(N, dtype=torch.float32)[None].expand(B, N)
    short = shortlist_indices("frozen", 4, scores=scores)
    assert short.shape == (B, 4)
    assert torch.equal(short[0], torch.tensor([N - 1, N - 2, N - 3, N - 4]))


def test_shortlist_oracle_debug_lowest_error_first():
    err = torch.arange(N, dtype=torch.float32)[None].expand(B, N)
    short = shortlist_indices("oracle-debug", 3, err=err)
    assert torch.equal(short[0], torch.tensor([0, 1, 2]))


def test_shortlist_clips_and_rejects():
    scores = torch.randn(B, N)
    assert shortlist_indices("frozen", N + 50, scores=scores).shape == (B, N)
    with pytest.raises(ValueError):
        shortlist_indices("frozen", 4)                  # missing scores
    with pytest.raises(ValueError):
        shortlist_indices("oracle-debug", 4)            # missing err
    with pytest.raises(ValueError):
        shortlist_indices("nonsense", 4, scores=scores)
    with pytest.raises(ValueError):
        shortlist_indices("frozen", 0, scores=scores)   # k < 1
    with pytest.raises(ValueError):
        shortlist_indices("frozen", 4, scores=scores[0])  # rank 1


def test_select_from_cost_maps_back_to_fan_indices():
    short = torch.tensor([[10, 4, 7], [2, 9, 0]])
    cost = torch.tensor([[3.0, 1.0, 2.0], [0.5, 0.4, 0.6]])
    assert torch.equal(select_from_cost(short, cost), torch.tensor([4, 9]))
    with pytest.raises(ValueError):
        select_from_cost(short, cost[:, :2])


# ---------------------------------------------------------------------------
# candidate roll actions (the stage_a placement + steer encoding)
# ---------------------------------------------------------------------------
def test_candidate_roll_actions_encoding_and_history():
    torch.manual_seed(0)
    b, p, kh, w, roll_k = 2, 3, 12, 8, 10
    aw2 = torch.randn(b, w, 2)
    a_c = torch.randn(b, p, kh)
    kap = torch.rand(b, p, kh) * 0.2 - 0.1
    aw_cf, fa_cf = candidate_roll_actions(aw2, a_c, kap, roll_k)
    assert aw_cf.shape == (b * p, w, 2)
    assert fa_cf.shape == (b * p, roll_k - 1, 2)
    aw_r = aw_cf.reshape(b, p, w, 2)
    fa_r = fa_cf.reshape(b, p, roll_k - 1, 2)
    # history (all window slots but the last) untouched, for every candidate
    assert torch.allclose(aw_r[:, :, :-1],
                          aw2[:, None, :-1].expand(b, p, w - 1, 2))
    # last window slot = the candidate's step-0 control, steer THROUGH the
    # corpus encoding steer = atan(2.9 * kappa)
    assert torch.allclose(aw_r[:, :, -1, 0],
                          torch.atan(WHEELBASE * kap[..., 0]))
    assert torch.allclose(aw_r[:, :, -1, 1], a_c[..., 0])
    # futures = steps 1..roll_k-1
    assert torch.allclose(fa_r[..., 0],
                          torch.atan(WHEELBASE * kap[..., 1:roll_k]))
    assert torch.allclose(fa_r[..., 1], a_c[..., 1:roll_k])
    # B-major/P-minor flattening (matches states.repeat_interleave(P, 0))
    assert torch.allclose(aw_cf[1], aw_r[0, 1])


def test_candidate_roll_actions_guards():
    aw2 = torch.randn(2, 8, 2)
    a_c = torch.randn(2, 3, 5)
    kap = torch.randn(2, 3, 5)
    with pytest.raises(ValueError):
        candidate_roll_actions(aw2, a_c, kap, 10)       # Kh=5 < roll_k=10
    with pytest.raises(ValueError):
        candidate_roll_actions(aw2, a_c, kap[:, :, :-1], 4)  # a/kappa mismatch
    with pytest.raises(ValueError):
        candidate_roll_actions(aw2[:, :, :1], a_c, kap, 4)   # aw2 not 2-ch
    with pytest.raises(ValueError):
        candidate_roll_actions(aw2[:1], a_c, kap, 4)         # batch mismatch
    # roll_k == 1 edge: empty futures, valid
    aw_cf, fa_cf = candidate_roll_actions(aw2, a_c, kap, 1)
    assert fa_cf.shape == (6, 0, 2)


# ---------------------------------------------------------------------------
# calibration helpers (P7-style)
# ---------------------------------------------------------------------------
def test_spearman_monotone_and_anti():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x ** 3) == pytest.approx(1.0)       # monotone, nonlin
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones_like(x)))          # constant -> nan
    with pytest.raises(ValueError):
        spearman(x, x[:3])


def test_cluster_bootstrap_spearman_monotone_ci():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = x + 0.1 * rng.normal(size=n)                       # strongly monotone
    eids = np.repeat(np.arange(10), n // 10)
    out = cluster_bootstrap_spearman(x, y, eids, n_boot=200, seed=0)
    assert out["spearman_rho"] > 0.9
    lo, hi = out["rho_ci_cluster"]
    assert lo > 0.5 and hi <= 1.0                          # CI excludes 0
    assert out["estimator"] == "episode_cluster_bootstrap"
    assert out["n_episodes"] == 10
    assert out["spearman_rho"] >= P7_GATE_RHO              # the P7 gate read


def test_cluster_bootstrap_single_episode_refuses_ci():
    x = np.arange(20.0)
    out = cluster_bootstrap_spearman(x, x, np.zeros(20), n_boot=50)
    assert out["rho_ci_cluster"] is None
    assert "not computable" in out["ci_note"]


# ---------------------------------------------------------------------------
# StateTap contract (mock readout — no v5f checkpoint on this box)
# ---------------------------------------------------------------------------
def test_state_tap_capture_and_strictness():
    readout = torch.nn.Linear(4, 6)
    tap = StateTap(readout)
    with pytest.raises(RuntimeError):
        tap.states(1, 1)                                   # nothing captured
    b, w = 2, 4
    tap.clear()
    full = readout(torch.randn(b * w, 4))                  # one encode pass
    st = tap.states(b, w)
    assert st.shape == (b, w, 6)
    assert not st.requires_grad
    assert torch.allclose(st, full.reshape(b, w, 6))
    tap.clear()
    readout(torch.randn(b * w, 4))
    readout(torch.randn(b * w, 4))                         # a SECOND pass
    with pytest.raises(RuntimeError):
        tap.states(b, w)                                   # must fail loudly
    tap.clear()
    readout(torch.randn(6, 4))
    with pytest.raises(ValueError):
        tap.states(2, 2)                                   # B*W mismatch
    tap.remove()
    readout(torch.randn(4, 4))                             # after remove
    assert tap.n_calls() == 1                              # unchanged buffer
