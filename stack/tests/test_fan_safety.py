"""Pin `taniteval/tools/fan_safety.py` — the PRIMARY endpoint of the REF-C RL re-scope.

⛔ WHY THIS FILE EXISTS AT ALL. `fan_safety.py`'s own docstring promises that
"`test_fan_safety.py` asserts equality against it [`refc_select.reachability_mask`]
on a [B, N, S, 2] fan so this stays ONE definition, not a second implementation".
When it was banked, that test did not exist — a docstring claiming a test is
exactly the "true but wrong for the reader" class: a reader takes the promise as
evidence and stops checking. The promise is now kept.

WHAT IS PINNED, and why each one is a failure that has actually happened here:

  * `reach_ok` == `refc_select.reachability_mask`. Two implementations of one
    geometry is how two "independent" checks come to agree on a wrong answer
    (`flyability.py`'s own docstring makes the same argument about the
    integrator). The 72.08 % survival number was measured with THAT function.
  * the ZERO-PATH control reads kamm 0 / envelope 0 EXACTLY. A control that
    must read a known value is the only thing that catches a probe tuned on its
    own data (CLAUDE.md 2026-08-22: four distinct failures in one afternoon,
    every one caught only by a control reading the same value as the thing being
    measured).
  * contact is TIME-ALIGNED: a competent follower is NOT flagged, and a path
    that actually drives into the lead's position at the matching step IS. The
    static-lead defect flags every human window with a time gap shorter than the
    horizon — the H-RL-THRESH-1 class, a safety term firing on the demonstration.
  * the probability mass is the softmax mass on the flagged candidates, and a
    `-inf` (selector-excluded) row carries EXACTLY zero mass. A row that keeps
    small mass makes an unsafe candidate the selector cannot pick look like risk
    the model is carrying.
  * lead-only flags read False, and never True, when there is no lead. A rate
    over a population where the event cannot occur is a denominator artifact.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FS_PATH = os.path.join(_REPO, "taniteval", "tools", "fan_safety.py")

if not os.path.exists(_FS_PATH):
    pytest.skip(f"fan_safety.py not present at {_FS_PATH}", allow_module_level=True)

_TE = os.path.join(_REPO, "taniteval")
for _p in (os.path.join(_REPO, "stack"), _TE, os.path.dirname(_TE)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("fan_safety_under_test", _FS_PATH)
FS = importlib.util.module_from_spec(_spec)
sys.modules["fan_safety_under_test"] = FS
_spec.loader.exec_module(FS)

from tanitad.refs.refc_select import reachability_mask                # noqa: E402
from tanitad.rl import rewards as RW                                  # noqa: E402


def _straight(v, n=1, b=1):
    """A constant-velocity straight fan [b, n, 5, 2]. ``v`` is a scalar or a [b] vector."""
    ts = torch.tensor(FS.GRID_S)                                  # [5]
    vv = torch.as_tensor(v, dtype=torch.float32).reshape(-1)      # [B']
    x = (vv[:, None] * ts[None, :])[:, None, :]                   # [B', 1, 5]
    x = x.expand(vv.shape[0], n, len(FS.GRID_S))
    if vv.shape[0] == 1 and b > 1:
        x = x.expand(b, n, len(FS.GRID_S))
    assert x.shape[0] == b, f"_straight got {vv.shape[0]} speeds for b={b}"
    return torch.stack([x, torch.zeros_like(x)], dim=-1).contiguous()


# --------------------------------------------------------------------------- #
# 1. ONE definition of the reach band                                           #
# --------------------------------------------------------------------------- #
def test_reach_ok_equals_refc_select_reachability_mask_on_a_fan():
    """The docstring's promise. [B, N, S, 2] fan, several speeds, both signs."""
    torch.manual_seed(0)
    b, n = 4, 16
    v0 = torch.tensor([0.0, 2.33, 10.09, 18.0])
    # a deliberately WIDE fan so both sides of the band are exercised
    fan = torch.randn(b, n, len(FS.GRID_S), 2) * 4.0
    fan = fan + _straight(v0, n=n, b=b)

    ours = FS.reach_ok(fan, v0)
    theirs = reachability_mask(fan, v0, accel_max=FS.A_MAX_REACH_MPS2,
                               horizon_s=FS.HORIZON_REACH_S)
    assert ours.shape == theirs.shape == (b, n)
    assert torch.equal(ours, theirs.bool()), (
        "fan_safety.reach_ok has drifted from refc_select.reachability_mask; "
        f"disagree on {int((ours != theirs.bool()).sum())}/{b * n} candidates")
    # the test must be able to FAIL: both classes present, or it proves nothing
    assert 0 < int(ours.sum()) < ours.numel(), (
        f"degenerate fixture: {int(ours.sum())}/{ours.numel()} in band — "
        "a mask that is all-True or all-False cannot detect a drift")


# --------------------------------------------------------------------------- #
# 2. the zero-path control must read the no-information value EXACTLY           #
# --------------------------------------------------------------------------- #
def test_frozen_zero_path_reads_exactly_zero_load():
    """The `frozen` control arm. Not 'small' — exactly 0.0, and `== 0.0` is the
    assertion because a stationary path has no acceleration by construction."""
    paths = torch.zeros(1, 3, len(FS.GRID_S), 2)
    v0 = torch.zeros(1)
    sc = FS.score_paths(paths, v0, None)
    assert float(sc["peak_g"].max()) == 0.0, f"peak_g {float(sc['peak_g'].max())!r} != 0.0"
    assert not bool(sc["kamm_over"].any())
    assert not bool(sc["envelope"].any())
    assert not bool(sc["contact"].any())
    # v0 = 0 and a zero path: mean speed 0 is inside [0, 0 + 2.5*2] -> in band
    assert not bool(sc["off_reach"].any())
    assert not bool(sc["flagged"].any()), "the frozen control flagged something"


# --------------------------------------------------------------------------- #
# 3. contact is TIME-ALIGNED, not a static snapshot                             #
# --------------------------------------------------------------------------- #
def test_contact_does_not_fire_on_a_competent_follower():
    """⛔ THE STATIC-LEAD DEFECT. Ego at 10 m/s, lead 15 m ahead ALSO at 10 m/s.
    The gap is constant, so no competent follower ever touches it — but a lead
    held STATIC at its t0 position is reached after 1.5 s and would be flagged."""
    ego = _straight(10.0)                                     # [1, 1, 5, 2]
    lead = ego + torch.tensor([15.0, 0.0])                    # same speed, +15 m
    sc = FS.score_paths(ego, torch.tensor([10.0]), lead)
    assert not bool(sc["contact"].any()), "time-aligned contact fired on a constant 15 m gap"

    # the static reading of the SAME scene, via the reward's static branch:
    # the lead frozen at its first sample. This is what the defect looks like.
    static = lead[..., :1, :].expand_as(lead)
    sc_static = FS.score_paths(ego, torch.tensor([10.0]), static)
    assert bool(sc_static["contact"].any()), (
        "the fixture no longer demonstrates the static-lead defect — the "
        "time-aligned assertion above would then be vacuous")


def test_contact_fires_when_the_path_really_reaches_the_lead():
    """The other half: a rate that never fires is not safety, it is a dead probe."""
    lead = _straight(0.0) + torch.tensor([6.0, 0.0])          # lead parked at x = 6 m
    ego = _straight(10.0)                                     # ego runs it down
    sc = FS.score_paths(ego, torch.tensor([10.0]), lead)
    assert bool(sc["contact"].all()), "contact never fired on a path through the lead"


# --------------------------------------------------------------------------- #
# 4. no lead -> the lead-only flags are False and OUT of the population         #
# --------------------------------------------------------------------------- #
def test_no_lead_reads_false_for_every_lead_only_flag():
    sc = FS.score_paths(_straight(12.0), torch.tensor([12.0]), None)
    for f in ("contact", "ttc_below", "ttc_veto"):
        assert not bool(sc[f].any()), f"{f} fired with no lead present"
    assert float(sc["min_ttc_s"].min()) == float("inf")


# --------------------------------------------------------------------------- #
# 5. TTC agrees with the reward's own veto geometry                             #
# --------------------------------------------------------------------------- #
def test_min_ttc_agrees_with_the_reward_ttc_violation():
    """`min_ttc_s` and `rewards.ttc_violation` must not drift: the threshold sweep
    and the veto have to be reading the same geometry."""
    torch.manual_seed(1)
    ego = _straight(12.0, n=8) + torch.randn(1, 8, len(FS.GRID_S), 2) * 0.7
    lead = _straight(6.0, n=8) + torch.tensor([14.0, 0.0])
    ttc = FS.min_ttc_s(ego, lead, lead_len_m=FS.LEAD_LEN_DEFAULT_M)
    for thr in (1.5, 2.93, 5.0):
        ours = ttc < thr
        theirs = RW.ttc_violation(ego, {"dt": FS.DT_S, "lead_path": lead,
                                        "lead_len_m": FS.LEAD_LEN_DEFAULT_M,
                                        "ttc_min_s": thr})
        assert torch.equal(ours, theirs.bool()), f"TTC disagreement at {thr} s"


# --------------------------------------------------------------------------- #
# 6. the probability mass, including the -inf rows                              #
# --------------------------------------------------------------------------- #
def test_summarise_fan_mass_matches_softmax_and_excluded_rows_carry_zero():
    n = 6
    sc = {f: torch.zeros(1, n, dtype=torch.bool) for f in FS.FLAGS}
    sc["contact"][0, 0] = True          # ONE flagged candidate: index 0
    sc["flagged"][0, 0] = True
    for extra in ("peak_g", "min_ttc_s", "min_gap_m", "v_mean_2s"):
        sc[extra] = torch.zeros(1, n)

    rank = torch.zeros(1, n)            # uniform over the kept rows
    rank[0, 5] = float("-inf")          # ...except one the selector excluded
    conf = torch.zeros(1, n)
    out = FS.summarise_fan(sc, rank, conf, torch.tensor([0]))

    # 5 finite rows, uniform -> the flagged one holds exactly 1/5
    assert float(out["mass_rank_contact"][0]) == pytest.approx(0.2, abs=1e-6)
    # conf has no -inf: 6 uniform rows -> 1/6
    assert float(out["mass_conf_contact"][0]) == pytest.approx(1.0 / 6.0, abs=1e-6)
    # the selected candidate IS the flagged one
    assert float(out["sel_contact"][0]) == 1.0
    # fraction over the whole fan
    assert float(out["fan_contact"][0]) == pytest.approx(1.0 / n, abs=1e-6)

    excluded = torch.softmax(rank.float(), dim=1)[0, 5]
    assert float(excluded) == 0.0, "a -inf selector row did not carry zero mass"


def test_summarise_fan_topk_uses_the_rank_order():
    """top-k must follow `rank`, not candidate order — otherwise the 'emitted fan'
    numbers describe an arbitrary slice rather than what the selector prefers."""
    n = 8
    sc = {f: torch.zeros(1, n, dtype=torch.bool) for f in FS.FLAGS}
    for extra in ("peak_g", "min_ttc_s", "min_gap_m", "v_mean_2s"):
        sc[extra] = torch.zeros(1, n)
    # flag ONLY the two worst-ranked candidates
    sc["flagged"][0, 6] = sc["flagged"][0, 7] = True
    rank = torch.arange(n, dtype=torch.float32).flip(0).reshape(1, n)  # 0 is best
    out = FS.summarise_fan(sc, rank, rank.clone(), torch.tensor([0]), topk=(4,))
    assert float(out["top4_flagged"][0]) == 0.0, "top-4 picked up a bottom-ranked flag"
    assert float(out["fan_flagged"][0]) == pytest.approx(2.0 / n, abs=1e-6)
