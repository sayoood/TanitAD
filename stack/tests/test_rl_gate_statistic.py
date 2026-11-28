"""⭐⭐ PIN THE GATE-STATISTIC FINDINGS BY MUTATION, NOT BY INSPECTION.

`.../2026-09-06-refcv4b-rl-repair/PREREG_STATISTIC_RANKING.md` retired `G-REWARD`'s RATE
on INSTRUMENT grounds and bound every replacement to a mutation test. This file makes the
two load-bearing facts machine-checkable, and every proof below **reintroduces the defect**
rather than reading the code for it -- this programme has an AST census that read 0
suspects on both the fixed and the broken trainer.

⛔ `G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE). Nothing here re-scores
it, and no assertion below is a gate verdict.
"""
import importlib.util
import os

import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # stack/
RANK_PATH = os.path.join(HERE, "scripts", "rl_statistic_mutation_rank.py")


def _rank_mod():
    if not os.path.exists(RANK_PATH):
        pytest.skip("rl_statistic_mutation_rank.py not present in this tree")
    spec = importlib.util.spec_from_file_location("_rl_statrank_for_test", RANK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. the STATISTIC's blindness -- the property that retired the rate            #
# --------------------------------------------------------------------------- #
def test_the_rate_is_bit_identical_under_a_magnitude_change_and_the_wmr_is_not():
    """⛔⛔ THE INSTRUMENT ARGUMENT, AS AN EXECUTABLE FACT.

    `G-REWARD`'s RATE is `P(hold_v0 >= human)` -- a SIGN statistic. Scale every paired
    gap by 10x and NOT ONE SIGN CHANGES, so the rate is **bit-identical** while the
    reward's preference has changed by an order of magnitude. The win-magnitude ratio is
    likewise scale-free BY CONSTRUCTION (it is a ratio), so the discriminating mutation
    is a change in the BALANCE of the mass, which the rate cannot see at all.
    """
    rk = _rank_mod()
    np = rk.np
    # 8 windows: 3 small wins for hold_v0, 5 larger wins for the human.
    g = np.array([+0.01, +0.01, +0.01, -0.02, -0.02, -0.02, -0.02, -0.02])
    rate0, wmr0 = rk.S1_rate(g), rk.S6_wmr(g)

    # MUTATION A -- multiply every gap by 10. No sign moves.
    gA = g * 10.0
    assert rk.S1_rate(gA) == rate0, "the rate must be bit-identical under pure scaling"
    assert rk.S6_wmr(gA) == pytest.approx(wmr0), "wmr is a ratio: also scale-free"

    # MUTATION B -- move MASS from the human's side to hold_v0's without flipping a sign.
    # ⭐ This is the change a distance-keeping statistic MUST see.
    gB = g.copy()
    gB[0] = +0.50                      # one window pays hold_v0 far more; still positive
    assert rk.S1_rate(gB) == rate0, (
        "⛔ the RATE is bit-identical while the reward now pays the trivial path 50x "
        "more on that window -- this is why it was retired")
    assert rk.S6_wmr(gB) > wmr0 + 0.10, (
        "the win-magnitude ratio must MOVE, and substantially")

    # ⭐ THE DISCRIMINATING CONTROL: a mutation that flips a sign MUST move the rate,
    # or the assertions above would merely prove the rate is constant.
    gC = g.copy()
    gC[3] = +0.02
    assert rk.S1_rate(gC) != rate0, (
        "control: the rate is not a constant -- a SIGN change does move it")


def test_the_family_ordering_is_a_property_of_the_statistics_not_of_our_data():
    """SIGN and ORDER statistics discard magnitude; MASS statistics do not.

    Constructed, data-free: shifting mass inside the positive tail leaves every
    sign-based and order-based candidate bit-identical, and moves every mass-based one.
    """
    rk = _rank_mod()
    np = rk.np
    g = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, +0.01, +0.02, +0.03, +0.04, +0.05])
    base = {k: fn(g) for k, (fn, _f) in rk.CANDIDATES.items()}
    g2 = g.copy()
    g2[-1] = +0.60                      # fatten the extreme positive tail only
    moved = {k: fn(g2) != base[k] for k, (fn, _f) in rk.CANDIDATES.items()}

    for cid in ("S1_rate", "S2_cliffs"):
        assert not moved[cid], f"{cid} is a SIGN statistic and must be blind here"
    for cid in ("S3_median",):
        assert not moved[cid], f"{cid} is an ORDER statistic and must be blind here"
    for cid in ("S5_mean", "S6_wmr", "S8_logmassratio"):
        assert moved[cid], f"{cid} is a MASS statistic and must move here"


def test_the_split_rule_is_deterministic_and_content_addressed():
    """The SELECT/SCORE split must not depend on enumeration order."""
    rk = _rank_mod()
    ids = ["clip-%04d" % i for i in range(400)]
    a = [rk.split_of(c) for c in ids]
    b = [rk.split_of(c) for c in reversed(ids)][::-1]
    assert a == b
    assert set(a) == {"select", "score"}, "both splits must be populated"
    n_sel = a.count("select")
    assert 150 < n_sel < 250, f"the split is grossly unbalanced: {n_sel}/400"


# --------------------------------------------------------------------------- #
# 2. the REWARD fact this turn PRICED -- the second cap in `achievable`         #
# --------------------------------------------------------------------------- #
def _ctx(lead_x, cap, v0=10.0):
    lead = torch.tensor([[[[[float(x), 0.0] for x in lead_x]]]], dtype=torch.float32)
    return {"dt": 0.5, "v0": torch.tensor([[[v0]]]), "lead_len_m": 4.5,
            "lead_path": lead, "progress_lead_cap": cap,
            "target_time_gap_s": 2.0, "progress_min_ref_m": 5.0}


def test_achievable_caps_a_FASTER_THAN_v0_candidate_and_leaves_hold_v0_UNTOUCHED():
    """⛔⛔ THE MECHANISM BEHIND THE +0.153679 SEPARATED COST, AS AN EXECUTABLE FACT.

    `PREREG.md` said the lead cap is *"inert without a lead"*, but the pre-registered
    `achievable` form is `clamp(min(ref_free, ref_lead), min_ref)` -- so it caps at the
    FREE reference `max(v0*H, min_ref)` as well. ⭐ That second cap truncates exactly the
    candidates that travel FURTHER than `v0*H`, and leaves a candidate sitting at exactly
    `v0` completely untouched. ⛔ **The human accelerates; `hold_v0` by construction does
    not.** So the cap removes the HUMAN's headroom and none of the trivial path's.

    MEASURED on the held-out half of the RL-fit windows, 2,417 windows / 33 episodes,
    paired episode-cluster bootstrap (`n_boot` 4,000): switching the reward from `lead` to
    `achievable` moves the win-magnitude ratio **+0.153679 [+0.084053, +0.254735],
    SEPARATED** -- the shipped default pays the trivial path ~15 pp MORE of the total gap
    mass. Attribution is unambiguous: `progress`'s weighted mean gap goes **-0.009769 ->
    +0.000890** while `headway` and `collision` are identical to the digit; and `progress`
    differs from the uncapped term on **73.28 %** of `human` windows against **38.18 %**
    of `hold_v0` windows -- the asymmetry this test pins.
    ⛔ This test does not decide the default; it makes the mechanism impossible to lose.
    """
    from tanitad.rl import rewards as R

    v0 = 10.0                       # => ref_free = max(v0 * 2.0 s, 5) = 20 m
    far = [400.0, 405.0, 410.0, 415.0, 420.0]     # a lead far beyond any reach

    def prog(xs, cap):
        traj = torch.tensor([[[[[float(x), 0.0] for x in xs]]]], dtype=torch.float32)
        return float(R._progress(traj, _ctx(far, cap, v0)).reshape(-1)[0])

    faster = [0.0, 7.5, 15.0, 22.5, 30.0]         # 30 m in 2 s -- FASTER than v0
    at_v0 = [0.0, 5.0, 10.0, 15.0, 20.0]          # exactly v0 -- this is `hold_v0`

    f_off, f_lead, f_ach = (prog(faster, m) for m in ("off", "lead", "achievable"))
    h_off, h_lead, h_ach = (prog(at_v0, m) for m in ("off", "lead", "achievable"))

    assert f_lead == f_off, (
        "with the lead 400 m away the LEAD cap must be inert -- nothing about the lead "
        "binds here")
    assert f_ach < f_off, (
        "⛔ `achievable` truncates a faster-than-v0 candidate even with no lead in "
        "reach: that is the SECOND cap at ref_free")
    assert h_ach == h_off, (
        "⛔⛔ AND IT LEAVES `hold_v0` UNTOUCHED -- the cap removes the human's headroom "
        "and none of the trivial path's. That asymmetry IS the +0.153679 cost")

    # ⭐ DISCRIMINATING CONTROL: bring the lead CLOSE and the `lead` cap MUST bite, or
    # the first assertion would merely prove `lead` is always inert.
    near = [30.0, 31.0, 32.0, 33.0, 34.0]

    def prog_near(xs, cap):
        traj = torch.tensor([[[[[float(x), 0.0] for x in xs]]]], dtype=torch.float32)
        return float(R._progress(traj, _ctx(near, cap, v0)).reshape(-1)[0])

    assert prog_near(faster, "lead") < prog_near(faster, "off"), (
        "control: with a near lead the `lead` cap must bind, so its inertness above is "
        "a property of the DISTANCE and not of the mode")


def test_capping_progress_raises_how_often_a_DO_NOTHING_plan_beats_the_human():
    """⚠️ A CAUTION THAT ATTACHES TO BOTH CAP VARIANTS, not only to `achievable`.

    MEASURED on the same held-out half: `frozen >= human` reads **0.003724** with the cap
    OFF and **0.112536** with it ON -- the SAME value for `achievable` and for `lead`, so
    this is a property of capping `progress` at all, not of the second cap. A reward
    under which a plan that never moves beats the human on 11 % of windows is worth
    knowing about before an arm trains on it.

    Pinned here on a fixture: a FROZEN candidate loses less ground to the human under a
    cap, because the cap removes the human's headroom rather than the frozen plan's.
    """
    from tanitad.rl import rewards as R

    human = torch.tensor([[[[[0.0, 0.0], [6.0, 0.0], [12.0, 0.0],
                             [18.0, 0.0], [24.0, 0.0]]]]], dtype=torch.float32)
    frozen = torch.zeros(1, 1, 1, 5, 2)
    near = [30.0, 31.0, 32.0, 33.0, 34.0]

    def gap(cap):
        h = float(R._progress(human, _ctx(near, cap)).reshape(-1)[0])
        f = float(R._progress(frozen, _ctx(near, cap)).reshape(-1)[0])
        return f - h

    off, lead, ach = gap("off"), gap("lead"), gap("achievable")
    assert lead > off, (
        "⚠️ under the LEAD cap the do-nothing plan closes ground on the human")
    assert ach > off, (
        "⚠️ and under `achievable` too -- the caution attaches to both variants")
    # ⭐ CONTROL: the frozen plan must still LOSE to the human under every mode, or the
    # comparison above would be measuring a reward that is simply broken.
    for cap in ("off", "lead", "achievable"):
        assert gap(cap) < 0.0, f"frozen must still score below the human under {cap!r}"
