"""`eval_flagship_v4 --select-rule` — the C2 wiring, and the default that must not move.

⛔ The load-bearing test in this file is :func:`test_default_select_rule_is_as_trained`.
A `vision_rank` default moved once and made every committed v4 number
unreproducible; this rule changes WHICH candidate of the fan is deployed, so a
default drift here would silently invalidate every published `ade_0_2s`.

Both directions: the wiring is proved to re-select AND proved to leave the pick
alone when the reference points at the incumbent. A re-selector that always
moves the pick, or never does, is untested either way.
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_flagship_v4 as E                                        # noqa: E402


def _fan_out(fan, sel):
    return {"anchor_traj": fan, "sel_idx": sel,
            "traj": fan[torch.arange(len(sel)), sel],
            "wp_seq": fan[torch.arange(len(sel)), sel],
            "waypoints": {}}


# --------------------------------------------------------------------------- #
# the default                                                                  #
# --------------------------------------------------------------------------- #
def test_default_select_rule_is_as_trained():
    p = inspect.signature(E.collect_planner).parameters
    assert p["select_rule"].default == "as-trained"
    assert p["c2"].default is None


def test_cli_default_is_as_trained_and_needs_no_scorer():
    ap_defaults = {}
    for line in inspect.getsource(E.main).splitlines():
        if "--select-rule" in line or "--c2-scorer" in line:
            ap_defaults[line.strip()[:40]] = line
    assert ap_defaults, "the flags disappeared from the CLI"
    src = inspect.getsource(E.main)
    assert 'choices=("as-trained", "c2-wm-ref")' in src
    assert 'default="as-trained"' in src


def test_c2_rule_without_a_scorer_is_refused_by_collect_planner():
    with pytest.raises(SystemExit, match="needs --c2-scorer"):
        E.collect_planner(None, None, None, None, "cpu", None, 1, 1, 1,
                          select_rule="c2-wm-ref", c2=None)


def test_an_unknown_rule_is_refused():
    with pytest.raises(SystemExit, match="unknown --select-rule"):
        E.collect_planner(None, None, None, None, "cpu", None, 1, 1, 1,
                          select_rule="c2-but-better")


def test_self_scoring_cannot_be_reached_by_omission():
    with pytest.raises(ValueError, match="no scoring world model was named"):
        E.build_c2_scorer(None, None, None, "cpu")


# --------------------------------------------------------------------------- #
# the re-selection contract — both directions                                  #
# --------------------------------------------------------------------------- #
def test_apply_c2_selection_moves_the_pick_and_keeps_every_derived_key_in_step():
    horizons = (1, 2, 3)
    fan = torch.zeros(2, 4, 3, 2)
    for c in range(4):
        fan[:, c] = float(c)                     # candidate c sits at (c, c)
    out = _fan_out(fan, torch.zeros(2, dtype=torch.long))
    before = out["anchor_traj"].clone()

    ref = torch.full((2, 3, 2), 3.0)             # closest to candidate 3
    tele = E.apply_c2_selection(out, horizons, ref, tag="unit")

    assert out["sel_idx"].tolist() == [3, 3]
    ar = torch.arange(2)
    assert torch.equal(out["traj"], out["anchor_traj"][ar, out["sel_idx"]])
    assert torch.equal(out["wp_seq"], out["traj"])
    assert set(out["waypoints"]) == set(horizons)
    for i, k in enumerate(horizons):
        assert torch.equal(out["waypoints"][k], out["traj"][:, i])
    # the FAN is untouched -> oracle_ade / coverage are invariant by construction
    assert torch.equal(out["anchor_traj"], before)
    assert tele["selected_frac"] == 1.0
    assert tele["frac_pick_equals_baseline"] == 0.0


def test_apply_c2_selection_leaves_the_pick_alone_when_it_already_agrees():
    horizons = (1, 2, 3)
    fan = torch.zeros(2, 4, 3, 2)
    for c in range(4):
        fan[:, c] = float(c)
    out = _fan_out(fan, torch.ones(2, dtype=torch.long))
    ref = torch.ones(2, 3, 2)                    # closest to candidate 1
    tele = E.apply_c2_selection(out, horizons, ref, tag="unit")
    assert out["sel_idx"].tolist() == [1, 1]
    assert tele["frac_pick_equals_baseline"] == 1.0


def test_a_sparse_head_reads_the_reference_at_its_own_lead_times():
    """A 4-waypoint head against a dense 20-step roll must not be broadcast."""
    horizons = (5, 10, 15, 20)
    fan = torch.zeros(1, 3, 4, 2)
    fan[:, 2] = 7.0
    out = _fan_out(fan, torch.zeros(1, dtype=torch.long))
    ref = torch.zeros(1, 20, 2)
    ref[:, [4, 9, 14, 19]] = 7.0                 # the roll matches cand 2 at 5/10/15/20
    E.apply_c2_selection(out, horizons, ref, tag="unit")
    assert out["sel_idx"].tolist() == [2]
    with pytest.raises(ValueError):              # a 3-step roll cannot serve h=20
        E.apply_c2_selection(out, horizons, torch.zeros(1, 3, 2), tag="unit")
