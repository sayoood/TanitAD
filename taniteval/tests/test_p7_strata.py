#!/usr/bin/env python3
"""Tests for ``taniteval/p7_strata.py`` -- the per-stratum P7 read the T3 gate row
needs and pooled P7 cannot give.

Every test here pins a claim that, if it broke, would produce a *plausible* wrong
answer rather than a crash -- which is the failure mode this programme keeps
paying for:

  * the statistic must be the SAME statistic P7 already uses (parity against
    ``w7_roll_rerank``), or "per-stratum P7" would silently be a different probe;
  * a stratum too small for a rho must be REFUSED, not printed, because a rho on
    9 windows from 2 episodes reads exactly like a rho on 300;
  * ``NO_LABEL`` must never be folded into free-flow, and must never enter an
    interaction verdict -- that is the bias `lead_source`'s three states exist for;
  * an ego-derived stratifier must be REFUSED by default, because cutting on ego
    state is cutting on the situation label's own source;
  * the pooled number must be structurally unable to answer the gate.

pytest is not installed everywhere, so this also runs standalone:
  python taniteval/tests/test_p7_strata.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                       # <repo>/taniteval
_REPO = os.path.dirname(_PKG_PARENT)
for _p in (_PKG_PARENT, os.path.join(_REPO, "stack"),
           os.path.join(_REPO, "stack", "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval.p7_strata import (  # noqa: E402
    DEFAULT_GAP_EDGES_M, MIN_N_EPISODES, MIN_N_WINDOWS, P7_GATE_RHO,
    STRATIFIER_KIND_EGO, STRATIFIER_KIND_LABEL, STRATIFIER_KIND_MODEL,
    arm_controls, assert_stratifier_admissible, cluster_bootstrap_spearman,
    lead_state_strata, p7_per_stratum, permutation_null, spearman)

LABEL_DECL = {
    "name": "obstacle.offline lead state + proximity",
    "kind": STRATIFIER_KIND_LABEL,
    "derived_from": "obstacle.offline 3D agent cuboids (dataset annotation)",
    "why_admissible": ("an external annotation of other traffic: not computed "
                       "from ego dynamics (the situation labels' source) and not "
                       "computed from any model output"),
}


# --------------------------------------------------------------------------- #
# synthetic fixture with a KNOWN answer                                        #
# --------------------------------------------------------------------------- #
def _fixture(seed=0, n_ep=20, per_ep=20, rho_lead=0.9, rho_free=0.0):
    """Two interaction bands that are calibrated and one free-flow band that is
    not, so a per-stratum read has something to separate that pooling hides."""
    rng = np.random.default_rng(seed)
    eid, state, gap, spread, err = [], [], [], [], []
    for e in range(n_ep):
        for j in range(per_ep):
            eid.append(f"ep_{e:05d}")
            if j % 4 == 0:
                state.append("LEAD"); gap.append(float(rng.uniform(2, 19)))
                r = rho_lead
            elif j % 4 == 1:
                state.append("LEAD"); gap.append(float(rng.uniform(20, 39)))
                r = rho_lead
            elif j % 4 == 2:
                state.append("NO_LEAD"); gap.append(np.nan)
                r = rho_free
            else:
                state.append("NO_LABEL"); gap.append(np.nan)
                r = rho_free
            z = rng.normal()
            spread.append(z)
            # +10 keeps the "error" positive like a real ADE; Spearman is
            # rank-based so the shift does not touch the known rho.
            err.append(10.0 + r * z
                       + np.sqrt(max(1 - r * r, 1e-9)) * rng.normal())
    return (np.asarray(spread), np.asarray(err), np.asarray(eid),
            np.asarray(state), np.asarray(gap))


# --------------------------------------------------------------------------- #
# 1. it must be the SAME statistic P7 already uses                             #
# --------------------------------------------------------------------------- #
def test_spearman_is_bit_identical_to_the_w7_p7_implementation():
    """⛔ 'per-stratum P7' is only P7 if the statistic is P7's. Two copies of a
    rank convention drift; this pins them (the pattern `v6.py` uses for
    ``P7_GATE_RHO`` itself)."""
    import w7_roll_rerank as w7
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(3, 200))
        a, b = rng.normal(size=n), rng.normal(size=n)
        assert spearman(a, b) == w7.spearman(a, b)
    assert np.isnan(spearman(np.ones(10), np.arange(10.0)))
    assert np.isnan(w7.spearman(np.ones(10), np.arange(10.0)))


def test_gate_constant_matches_the_pre_registered_p7_gate():
    import w7_roll_rerank as w7
    assert P7_GATE_RHO == w7.P7_GATE_RHO == 0.3


# --------------------------------------------------------------------------- #
# 2. the estimator                                                             #
# --------------------------------------------------------------------------- #
def test_cluster_bootstrap_recovers_a_known_rho_and_excludes_zero():
    rng = np.random.default_rng(1)
    eid = np.repeat([f"ep_{i:03d}" for i in range(30)], 20)
    x = rng.normal(size=eid.size)
    y = x + 0.3 * rng.normal(size=eid.size)
    out = cluster_bootstrap_spearman(x, y, eid, n_boot=400, seed=0)
    assert out["spearman_rho"] > 0.8
    assert out["rho_ci_cluster"][0] > 0.0
    assert out["estimator"] == "episode_cluster_bootstrap"
    assert out["bracket_kind"] == "episode_cluster_bootstrap_percentile_95"
    assert out["n_episodes"] == 30


def test_cluster_bootstrap_straddles_zero_when_there_is_no_signal():
    rng = np.random.default_rng(2)
    eid = np.repeat([f"ep_{i:03d}" for i in range(30)], 20)
    x, y = rng.normal(size=eid.size), rng.normal(size=eid.size)
    out = cluster_bootstrap_spearman(x, y, eid, n_boot=400, seed=0)
    lo, hi = out["rho_ci_cluster"]
    assert lo < 0.0 < hi


def test_one_episode_yields_no_interval_and_says_so():
    """⛔ A fake interval is worse than none: it reads like a passing gate."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    out = cluster_bootstrap_spearman(x, x + 0.1 * rng.normal(size=40),
                                     ["ep_0"] * 40, n_boot=50)
    assert out["rho_ci_cluster"] is None
    assert "NOT computable" in out["ci_note"]


# --------------------------------------------------------------------------- #
# 3. the strata                                                                #
# --------------------------------------------------------------------------- #
def test_lead_state_strata_bands_and_three_states():
    state = np.array(["LEAD", "LEAD", "LEAD", "NO_LEAD", "NO_LABEL"])
    gap = np.array([5.0, 25.0, 55.0, np.nan, np.nan])
    lab, spec = lead_state_strata(state, gap, edges=DEFAULT_GAP_EDGES_M)
    assert lab.tolist() == ["LEAD_le20m", "LEAD_20_40m", "LEAD_ge40m",
                            "NO_LEAD", "NO_LABEL"]
    assert spec["interaction_rich"]["NO_LABEL"] is None
    assert spec["interaction_rich"]["NO_LEAD"] is False
    assert all(spec["interaction_rich"][s] for s in
               ("LEAD_le20m", "LEAD_20_40m", "LEAD_ge40m"))
    for s in spec["stratum_order"]:
        assert spec["inclusion_rule"][s]           # C110: rule beside the count


def test_no_label_is_never_folded_into_free_flow():
    """⚠️ `obstacle.offline` spans ~20 s while `egomotion` runs 20-140 s, so most
    of a long clip is unlabelled. Counting that as 'no lead agent' manufactures
    free-flow and flatters every arm."""
    state = np.array(["NO_LABEL"] * 5 + ["NO_LEAD"] * 5)
    lab, _ = lead_state_strata(state, np.full(10, np.nan))
    assert (lab[:5] == "NO_LABEL").all() and (lab[5:] == "NO_LEAD").all()


def test_a_lead_window_with_no_gap_raises_rather_than_landing_in_a_band():
    try:
        lead_state_strata(np.array(["LEAD"]), np.array([np.nan]))
    except ValueError as e:
        assert "non-finite gap" in str(e)
    else:
        raise AssertionError("a LEAD window with no gap must raise")


# --------------------------------------------------------------------------- #
# 4. admissibility -- the ruling this instrument has to respect                #
# --------------------------------------------------------------------------- #
def test_an_ego_derived_stratifier_is_refused_by_default():
    """⛔ Cutting on ego state is cutting on the situation label's own source
    (situations.py). The refusal must name that, not just say 'invalid'."""
    spec = dict(LABEL_DECL, name="ego speed bands", kind=STRATIFIER_KIND_EGO,
                derived_from="poses[:, 3] (ego speed at t0)")
    try:
        assert_stratifier_admissible(spec)
    except ValueError as e:
        assert "situations.py" in str(e)
    else:
        raise AssertionError("an ego-derived stratifier must be refused")
    rec = assert_stratifier_admissible(spec, override_reason="pre-registered "
                                       "sensitivity check, reported as such")
    assert rec["admissible"] is False and rec["override_reason"]


def test_a_model_derived_stratifier_is_refused_by_default():
    spec = dict(LABEL_DECL, name="P8 predicted occupancy entropy",
                kind=STRATIFIER_KIND_MODEL,
                derived_from="the arm's own occupancy readout")
    try:
        assert_stratifier_admissible(spec)
    except ValueError as e:
        assert "grade it" in str(e)
    else:
        raise AssertionError("a model-derived stratifier must be refused")


def test_a_stratifier_with_no_declaration_is_refused():
    try:
        assert_stratifier_admissible({"name": "mystery"})
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("an undeclared stratifier must be refused")


# --------------------------------------------------------------------------- #
# 5. the read                                                                  #
# --------------------------------------------------------------------------- #
def test_per_stratum_separates_what_pooling_hides():
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=300)
    assert out["strata"]["LEAD_le20m"]["gate_pass"] is True
    assert out["strata"]["LEAD_20_40m"]["gate_pass"] is True
    assert out["strata"]["NO_LEAD"]["gate_pass"] is False
    assert out["verdict"] == "PASS"
    assert out["tier"] == "T0"


def test_the_pooled_number_is_structurally_unable_to_answer_the_gate():
    """⛔ The whole point of the row is 'not just pooled'. Pooled is reported and
    stamped, never used for the verdict."""
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=200)
    assert out["pooled"]["is_gate_read"] is False
    assert "not just pooled" in out["pooled"]["note"].lower()
    # a pooled PASS must not carry a stratum FAIL with it
    bad = _fixture(seed=5, rho_lead=0.0, rho_free=0.95)
    lab2, spec2 = lead_state_strata(bad[3], bad[4])
    out2 = p7_per_stratum(bad[0], bad[1], bad[2], lab2, stratifier=LABEL_DECL,
                          stratum_spec=spec2, n_boot=200)
    assert out2["verdict"] == "FAIL"


def test_a_stratum_below_the_floor_is_refused_not_reported():
    """⛔ A rho on 9 windows prints identically to a rho on 300."""
    spread, err, eid, state, gap = _fixture(n_ep=4, per_ep=8)
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=100)
    for s, row in out["strata"].items():
        assert row["status"] == "REFUSED_MIN_N", s
        assert "spearman_rho" not in row, s
        assert row["n"] < MIN_N_WINDOWS or row["n_episodes"] < MIN_N_EPISODES
    assert out["verdict"] == "NOT_COMPUTABLE"


def test_every_stratum_row_carries_its_n_and_its_inclusion_rule():
    """⚠️ C110 -- a count is a claim about the filter."""
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=100)
    for s, row in out["strata"].items():
        assert isinstance(row["n"], int) and row["inclusion_rule"], s


def test_every_bracket_is_labelled():
    """⚠️ C109 -- a dispersion is not a confidence interval."""
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=100)
    for row in list(out["strata"].values()) + [out["pooled"]]:
        if "rho_ci_cluster" in row:
            assert row["bracket_kind"] == \
                "episode_cluster_bootstrap_percentile_95"


def test_no_label_never_enters_the_verdict():
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    out = p7_per_stratum(spread, err, eid, lab, stratifier=LABEL_DECL,
                         stratum_spec=spec, n_boot=100)
    assert out["strata"]["NO_LABEL"]["interaction_rich"] is None
    assert "gate_pass" not in out["strata"]["NO_LABEL"] or \
        out["strata"]["NO_LABEL"].get("interaction_rich") is not True


# --------------------------------------------------------------------------- #
# 6. controls -- per arm                                                       #
# --------------------------------------------------------------------------- #
def test_positive_control_is_detected_and_constant_control_is_not():
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    c = arm_controls(spread, err, eid, lab, stratifier=LABEL_DECL,
                     stratum_spec=spec, n_boot=200)
    for s in ("LEAD_le20m", "LEAD_20_40m"):
        assert c["positive"]["result"]["strata"][s]["gate_pass"] is True, s
        row = c["constant"]["result"]["strata"][s]
        assert row["gate_pass"] is False and np.isnan(row["spearman_rho"]), s


def test_permutation_null_is_centred_on_zero_and_labels_its_bracket():
    """⚠️ C109 -- the null's percentiles are a DISPERSION, not a CI, and must say
    so. This control replaced a single-shuffle-plus-interval control that produced
    a false 'significant' on real windows (see ``permutation_null``'s docstring)."""
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    c = arm_controls(spread, err, eid, lab, stratifier=LABEL_DECL,
                     stratum_spec=spec, n_boot=200, n_perm=300)
    null = c["permutation_null"]
    assert null["bracket_kind"] == "permutation_null_dispersion_not_a_ci"
    for s in ("LEAD_le20m", "LEAD_20_40m", "NO_LEAD"):
        row = null["strata"][s]
        assert row["bracket_kind"] == "permutation_null_dispersion_not_a_ci"
        assert row["null_centred_on_zero"] is True, (s, row["null_median"])
        lo, hi = row["null_p2p5_p97p5"]
        assert lo < 0.0 < hi, (s, lo, hi)
    # a calibrated stratum's observed rho sits in the null's tail; an
    # uncalibrated one does not.
    assert null["strata"]["LEAD_le20m"]["p_two_sided"] < 0.05
    assert null["strata"]["NO_LEAD"]["p_two_sided"] > 0.05


def test_a_single_shuffle_plus_an_interval_is_not_a_null():
    """⛔ Pins the retracted control's failure mode so it cannot come back: an
    interval around ONE shuffle's rho can exclude zero purely by chance, because
    it brackets that draw rather than the null."""
    rng = np.random.default_rng(11)
    eid = np.repeat([f"ep_{i:03d}" for i in range(14)], 6)
    x, y = rng.normal(size=eid.size), rng.normal(size=eid.size)
    strata = np.array(["S"] * eid.size)
    null = permutation_null(x, y, eid, strata, min_n=8, min_eps=4, n_perm=400,
                            seed=0)["strata"]["S"]
    lo, hi = null["null_p2p5_p97p5"]
    # the null is wide enough that single draws land well outside +-0.05 --
    # exactly the draws a one-shuffle control would call "significant".
    assert hi > 0.2 and lo < -0.2, (lo, hi)
    assert null["null_centred_on_zero"] is True


def test_trivial_proxy_control_is_reported_when_supplied():
    spread, err, eid, state, gap = _fixture()
    lab, spec = lead_state_strata(state, gap)
    c = arm_controls(spread, err, eid, lab, stratifier=LABEL_DECL,
                     stratum_spec=spec, trivial_proxy=err * 0 + np.arange(err.size),
                     trivial_proxy_name="ego speed at t0", n_boot=100)
    assert "trivial_proxy" in c
    assert c["trivial_proxy"]["what"] == "ego speed at t0"
    assert "strata" in c["trivial_proxy"]["result"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:                              # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)
