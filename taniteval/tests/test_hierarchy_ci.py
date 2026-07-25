"""Pins for the 2026-07-25 hierarchy-panel estimator migration (HPP-1 P3).

WHY THIS FILE EXISTS
--------------------
``hierarchy.py`` produced every hierarchy interval the program has ever quoted,
and until 2026-07-25 it produced them with ``overlapping_holdout_se`` — the
estimator CLAUDE.md records as **1.28-2.06x too narrow**. That is not one panel
among many: the Hierarchy Proof Program states PC1's exit criterion and every
discriminating prediction HP-1..HP-6 as *"CI-separated"*
(``01_EXECUTION_PLAN.md`` Part A), so while this panel was on the deprecated
estimator **there was no admissible interval for any hierarchy claim at all**.

The tests below pin the four properties that make the migration trustworthy:

1. **The bootstrap supplies the interval, it never moves the point estimate.**
   Every migrated delta reproduces the arithmetic mean of the per-window
   difference EXACTLY. A migration that silently shifted a point estimate would
   invalidate every historical comparison, and a sibling agent MEASURED exactly
   that failure mode elsewhere (1.5-5.9 % point-estimate bias).
2. **The load-bearing verdict keeps its one-sided meaning.** ``_jack``'s
   ``separated`` was ``mean - ci95 > 0``; the program's standard ``separated``
   is two-sided. Reading the two-sided flag would promote a **harmful** seam to
   LOAD-BEARING. ``separated_positive`` must be what decides it.
3. **The deprecated numbers survive under their true name**, so no published
   hierarchy interval becomes unverifiable — and the width ratio is re-measured
   in every artifact rather than cited from a doc.
4. **The panel is behind the estimator guard**, which is the specific omission
   the HPP-0 confound audit logged as §7 #6.
"""
from __future__ import annotations

import numpy as np
import pytest

from taniteval import driving as drv
from taniteval import hierarchy as H


# --------------------------------------------------------------------------- #
# A synthetic panel record with the canonical val SHAPE (881 windows / 40 eps)  #
# --------------------------------------------------------------------------- #
N_WIN, N_EP = 880, 40


def _rec(seed=0, ctx_effect=0.05, h18_effect=2.7, valid_every=4):
    """A ``hierarchy.run`` record dict with a KNOWN, planted effect per seam.

    Episode ids are ints numbered 0..39 exactly as ``taniteval.data`` numbers
    them, because the quarantined legacy estimator splits on
    ``sorted(set(int(e)))`` and would otherwise not be evaluable."""
    rng = np.random.default_rng(seed)
    eids = [i * N_EP // N_WIN for i in range(N_WIN)]
    route_tgt = rng.integers(0, 3, N_WIN)
    man_tgt = rng.integers(0, 5, N_WIN)
    rec = {
        "eid": eids,
        "valid": [bool(i % valid_every == 0) for i in range(N_WIN)],
        "route_tgt": route_tgt.tolist(),
        "route_nav": route_tgt.tolist(),                 # the perfect echo
        "route_follow": [1] * N_WIN,                     # always "straight"
        "route_zero": rng.integers(0, 3, N_WIN).tolist(),
        "man_tgt": man_tgt.tolist(),
        "man_pred": man_tgt.tolist(),
        "gt_dir": rng.integers(0, 3, N_WIN).tolist(),
        "traj_dir": rng.integers(0, 3, N_WIN).tolist(),
        "cond_ae_norm": rng.random(N_WIN).tolist(),
        "cond_intent_norm": rng.random(N_WIN).tolist(),
    }
    base_man = rng.random(N_WIN) < 0.55
    rec["man_corr_real"] = base_man.astype(float).tolist()
    # A planted ctx->tactical effect of |ctx_effect| in expectation.
    #   ctx_effect > 0  ->  mean-ctx is WORSE  (the seam HELPS, delta > 0)
    #   ctx_effect < 0  ->  mean-ctx is BETTER (the seam HARMS, delta < 0)
    flip = rng.random(N_WIN) < abs(ctx_effect)
    rec["man_corr_meanctx"] = ((base_man & ~flip) if ctx_effect >= 0
                               else (base_man | flip)).astype(float).tolist()
    rec["man_corr_zeroctx"] = (base_man & ~(rng.random(N_WIN) < 0.08)).astype(
        float).tolist()
    op = rng.random(N_WIN) * 0.4 + 0.3                   # grounded rollout ADE
    rec["ade_op_none"] = op.tolist()
    rec["wp_ade_head_real"] = (op + h18_effect).tolist()  # H18: head is worse
    for tag in ("meanctx", "zeroctx"):
        rec[f"wp_ade_head_{tag}"] = (op + h18_effect + 0.1).tolist()
        rec[f"goal_cos_{tag}"] = (rng.random(N_WIN) * 0.2).tolist()
    rec["goal_cos_real"] = (rng.random(N_WIN) * 0.2 + 0.03).tolist()
    for tag in ("real", "mean", "zero", "none"):
        rec[f"lat_cos_{tag}"] = (rng.random(N_WIN) * 0.2 + 0.5).tolist()
        rec[f"lat_rel_{tag}"] = (rng.random(N_WIN) * 0.5 + 0.5).tolist()
        if tag != "none":
            rec[f"ade_op_{tag}"] = (op + 0.2).tolist()
    return rec


@pytest.fixture(scope="module")
def panel():
    return H._assemble(_rec(), n_boot=400, seed=0)


# --------------------------------------------------------------------------- #
# 1. The estimator is the decision-grade one, everywhere                        #
# --------------------------------------------------------------------------- #
def test_every_seam_names_the_decision_grade_estimator(panel):
    for block in ("seam_nav_to_strategic", "seam_ctx_to_tactical",
                  "seam_intent_to_operative", "consistency",
                  "h18_grounded_vs_ungrounded"):
        drv.assert_no_deprecated_estimator(panel[block], _path=block)


def test_panel_is_behind_the_guard(panel):
    """HPP-0 audit §7 #6: ``driving.py``'s guard did not cover this block."""
    assert H.assert_no_deprecated_estimator(panel) is True


def test_guard_rejects_a_smuggled_deprecated_interval(panel):
    bad = dict(panel)
    bad["seam_ctx_to_tactical"] = {
        "delta": {"mean": 0.04, "ci95": 0.03, "lo": 0.01, "hi": 0.07,
                  "estimator": H.DEPRECATED_ESTIMATOR}}
    with pytest.raises(ValueError, match="refusing to emit"):
        H.assert_no_deprecated_estimator(bad)


def test_guard_rejects_an_unlabelled_interval(panel):
    bad = dict(panel)
    bad["seam_ctx_to_tactical"] = {"delta": {"mean": 0.04, "lo": 0.01, "hi": 0.07}}
    with pytest.raises(ValueError, match="without a named estimator"):
        H.assert_no_deprecated_estimator(bad)


def test_deltas_are_paired_not_two_independent_intervals(panel):
    """The two conditionings score the SAME windows; quadrature is invalid."""
    for d in (panel["seam_nav_to_strategic"]["delta_nav_vs_follow"],
              panel["seam_ctx_to_tactical"]["maneuver_acc"]["delta_real_vs_mean"],
              panel["h18_grounded_vs_ungrounded"]["delta_ungrounded_minus_grounded"]):
        assert d["estimator"] == "paired_episode_cluster_bootstrap"
        assert d["n_episodes"] == N_EP


def test_rates_use_the_unpaired_interval(panel):
    ag = panel["consistency"]["maneuver_vs_trajectory"]["agreement"]
    assert ag["estimator"] == "episode_cluster_bootstrap"


# --------------------------------------------------------------------------- #
# 2. The bootstrap supplies the interval — it must NEVER move the mean          #
# --------------------------------------------------------------------------- #
def test_point_estimates_are_the_full_set_means():
    rec = _rec()
    out = H._assemble(rec, n_boot=200, seed=0)
    A = {k: np.asarray(v) for k, v in rec.items()}

    d = out["h18_grounded_vs_ungrounded"]["delta_ungrounded_minus_grounded"]
    exact = float(np.mean(A["wp_ade_head_real"] - A["ade_op_none"]))
    assert d["delta"] == pytest.approx(round(exact, 4), abs=1e-9)
    assert d["mean"] == d["delta"], "`mean` must stay an alias of `delta`"

    d = out["seam_ctx_to_tactical"]["maneuver_acc"]["delta_real_vs_mean"]
    exact = float(np.mean(A["man_corr_real"] - A["man_corr_meanctx"]))
    assert d["delta"] == pytest.approx(round(exact, 4), abs=1e-9)

    # lower-is-better orientation must still be helps-positive (mean - real)
    d = out["seam_ctx_to_tactical"]["wp_ade_2s"]["delta_real_vs_mean"]
    exact = float(np.mean(A["wp_ade_head_meanctx"] - A["wp_ade_head_real"]))
    assert d["delta"] == pytest.approx(round(exact, 4), abs=1e-9)


def test_point_estimate_is_invariant_to_n_boot_and_seed():
    """The interval may move with B/seed; the mean may not."""
    rec = _rec()
    a = H._assemble(rec, n_boot=200, seed=0)
    b = H._assemble(rec, n_boot=800, seed=7)
    for get in (lambda o: o["h18_grounded_vs_ungrounded"]
                ["delta_ungrounded_minus_grounded"]["delta"],
                lambda o: o["seam_ctx_to_tactical"]["maneuver_acc"]
                ["delta_real_vs_mean"]["delta"],
                lambda o: o["seam_nav_to_strategic"]["route_skill"]):
        assert get(a) == get(b)


def test_route_skill_delta_equals_acc_minus_majority():
    """PC1's quantity, computed two ways, must agree exactly."""
    out = H._assemble(_rec(), n_boot=200, seed=0)
    nav = out["seam_nav_to_strategic"]
    assert nav["route_skill"] == pytest.approx(
        round(nav["route_acc_follow"] - nav["majority_straight_rate"], 4),
        abs=1e-9)
    assert nav["route_skill_vs_majority"]["delta"] == pytest.approx(
        nav["route_skill"], abs=2e-4)


# --------------------------------------------------------------------------- #
# 3. The load-bearing predicate keeps its ONE-SIDED meaning                     #
# --------------------------------------------------------------------------- #
def test_meaningful_requires_positive_separation():
    helps = {"mean": 0.10, "separated": True, "separated_positive": True}
    harms = {"mean": -0.10, "separated": True, "separated_positive": False}
    tiny = {"mean": 0.001, "separated": True, "separated_positive": True}
    assert H._meaningful(helps, H.MIN_ACC) is True
    assert H._meaningful(harms, H.MIN_ACC) is False, \
        "a CI-separated HARMFUL seam must never read as load-bearing"
    assert H._meaningful(tiny, H.MIN_ACC) is False


def test_a_harmful_seam_is_not_load_bearing_end_to_end():
    """Plant a ctx seam where the real signal HURTS; the verdict must not flip."""
    rec = _rec(ctx_effect=-0.20)         # mean-ctx is BETTER than real
    out = H._assemble(rec, n_boot=400, seed=0)
    d = out["seam_ctx_to_tactical"]["maneuver_acc"]["delta_real_vs_mean"]
    assert d["delta"] < 0
    assert d["separated"] is True, "the two-sided flag SHOULD fire"
    assert d["separated_positive"] is False
    assert H._meaningful(d, H.MIN_ACC) is False


def test_both_separation_flags_are_emitted(panel):
    for d in (panel["seam_nav_to_strategic"]["delta_nav_vs_follow"],
              panel["h18_grounded_vs_ungrounded"]["delta_ungrounded_minus_grounded"]):
        assert set(("separated", "separated_positive")) <= set(d)
        if d["separated_positive"]:
            assert d["separated"], "positive separation implies separation"


def test_harmful_if_engaged_uses_the_real_upper_bound(panel):
    """`_jack` had to assume a symmetric CI (`mean + ci95 < 0`); the bootstrap
    emits a real, possibly asymmetric `hi`."""
    dcn = panel["seam_intent_to_operative"]["delta_cos_real_vs_none"]
    expected = bool(dcn["hi"] is not None and dcn["hi"] < 0
                    and abs(dcn["mean"]) >= H.MIN_COS)
    assert panel["seam_intent_to_operative"]["harmful_if_engaged"] is expected


# --------------------------------------------------------------------------- #
# 4. The deprecated numbers are preserved, quarantined and self-labelling       #
# --------------------------------------------------------------------------- #
def test_legacy_block_is_present_and_labelled(panel):
    leg = panel[H.LEGACY_BLOCK]
    assert leg["_estimator"] == H.DEPRECATED_ESTIMATOR
    assert leg["seam_nav_to_strategic"]["delta_nav_vs_follow"]["deprecated"] is True
    assert leg["h18_grounded_vs_ungrounded"]["estimator"] == H.DEPRECATED_ESTIMATOR


def test_legacy_block_is_the_only_thing_the_guard_exempts(panel):
    """It must fail the guard on its own — that is what makes it a quarantine."""
    with pytest.raises(ValueError, match="refusing to emit"):
        drv.assert_no_deprecated_estimator(panel[H.LEGACY_BLOCK], _path="legacy")


def test_legacy_jack_reproduces_the_pre_migration_arithmetic():
    """`_jack` must stay byte-for-byte what it was, or history stops reproducing."""
    rng = np.random.default_rng(3)
    vals = rng.normal(0.05, 0.3, N_WIN)
    eids = [i * N_EP // N_WIN for i in range(N_WIN)]
    got = H._jack(vals, eids)
    # the pre-migration expression, inlined
    from tanitad.eval.gates import split_by_episode
    sm = [float(np.nanmean(vals[va]))
          for s in range(H.N_SPLITS)
          for _tr, va in [split_by_episode(eids, 0.2, s)] if va]
    sm = np.asarray(sm)
    assert got["mean"] == pytest.approx(round(float(np.mean(sm)), 4), abs=1e-9)
    assert got["ci95"] == pytest.approx(
        round(float(1.96 * np.std(sm) / len(sm) ** 0.5), 4), abs=1e-9)


def test_width_ratio_is_measured_per_artifact(panel):
    wr = panel[H.LEGACY_BLOCK]["ci_width_ratio_new_over_legacy"]
    ratios = [v for k, v in wr.items() if not k.startswith("_") and v is not None]
    assert ratios, "the width ratio must be emitted, not left to a doc citation"
    # The honest estimator must not be NARROWER than the deprecated one on the
    # seams; that would mean the migration went the wrong way.
    assert min(ratios) > 0.9, f"a migrated interval got narrower: {wr}"


def test_verdict_flips_are_recorded(panel):
    flips = panel[H.LEGACY_BLOCK]["verdict_flips_vs_legacy"]
    for k, v in flips.items():
        if k.startswith("_"):
            continue
        assert set(("legacy_load_bearing", "migrated_load_bearing",
                    "flipped")) <= set(v)
        assert v["flipped"] == (v["legacy_load_bearing"]
                                != v["migrated_load_bearing"])


def test_legacy_block_declines_non_int_episode_ids():
    rec = _rec()
    rec["eid"] = [f"ep_{e:02d}" for e in rec["eid"]]
    out = H._assemble(rec, n_boot=200, seed=0)
    assert out[H.LEGACY_BLOCK]["not_evaluable"] is True
    # ...and the decision-grade path is unaffected by the id type
    assert out["h18_grounded_vs_ungrounded"][
        "delta_ungrounded_minus_grounded"]["n_episodes"] == N_EP


# --------------------------------------------------------------------------- #
# 5. Degenerate subsets must refuse to invent an interval                       #
# --------------------------------------------------------------------------- #
def test_single_episode_subset_emits_no_interval():
    B = H._Boot([0] * 20)
    out = H._interval(B, np.linspace(0, 1, 20))
    assert out["insufficient_episodes"] is True
    assert out["lo"] is None and out["hi"] is None
    assert out["separated"] is False and out["separated_positive"] is False
    assert out["estimator"] == "episode_cluster_bootstrap"
    drv.assert_no_deprecated_estimator(out, _path="degenerate")


def test_empty_mask_does_not_crash():
    B = H._Boot([i // 5 for i in range(20)])
    out = H._paired(B, np.zeros(20), np.zeros(20), np.zeros(20, dtype=bool))
    assert out["n"] == 0 and out["separated"] is False


def test_back_compat_keys_survive_for_published_readers(panel):
    """`report.py`, `runner.py` and `gate_emitters.py` read these by name."""
    nav = panel["seam_nav_to_strategic"]
    assert isinstance(nav["vision_route_beats_majority"], bool)
    assert nav["delta_nav_vs_follow"]["n"] == nav["n_valid"]
    assert "load_bearing" in panel["seam_ctx_to_tactical"]
    assert isinstance(panel["h18_grounded_vs_ungrounded"]["grounded_wins"], bool)
    th = panel["thesis_read"]["A_conditioning_helps_conditioned_layer"]
    assert "n_of_3_seams_beneficial" in th
    assert panel["consistency"]["maneuver_vs_trajectory"]["agreement"]["mean"] \
        is not None


def test_pc1_read_is_emitted_with_its_interval(panel):
    pc1 = panel["thesis_read"]["PC1_route_input_works"]
    assert pc1["route_skill_ci"]["estimator"] == "paired_episode_cluster_bootstrap"
    assert pc1["verdict"].startswith("NOT MET"), \
        "the fixture's follow-head predicts straight always -> route_skill 0"
