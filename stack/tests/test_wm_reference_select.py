"""C2 — the world-model-reference selection rule, and the guards on shipping it.

Self-contained (synthetic tensors only), so this file runs on any host with the
`stack/` tree and nothing else. The fidelity of the RULE to the published 881-window
numbers is a separate, artifact-backed file: `taniteval/tests/test_c2_published_policy.py`.

⚠️ Both directions everywhere. A selection rule that can only be shown to help is
an instrument that cannot report a loss — and this rule DOES lose on one measured
arm (`MEASURED_ARMS["v4_scores_its_own_fan"]`, +0.2090 m separated-WORSE), which is
exactly why the default is OFF.
"""
import inspect

import pytest
import torch

from tanitad.models import wm_reference_select as W
from tanitad.models.flagship_v15 import NoCandidateAxis


# --------------------------------------------------------------------------- #
# 1. the default, pinned to the EVIDENCE and not to an opinion                 #
# --------------------------------------------------------------------------- #
def test_default_is_off():
    assert W.WM_REFERENCE_SELECT_DEFAULT is False


def test_default_is_off_because_a_measured_arm_is_separated_worse():
    """The guard that can actually fail: flip the default and this goes red.

    A rule may only default ON if EVERY measured arm is better. One arm is
    separated-WORSE, so the default must be OFF. If a future measurement removes
    that arm, this test stops constraining the default — which is the correct
    coupling: the default follows the evidence table, in code.
    """
    worse = [k for k, v in W.MEASURED_ARMS.items()
             if v["separated"] and not v["better"]]
    assert worse, ("MEASURED_ARMS lost its failing arm; the default is now "
                   "unconstrained by evidence — re-measure before trusting it")
    assert W.WM_REFERENCE_SELECT_DEFAULT is False, (
        f"default is ON while {worse} are separated-WORSE")


def test_every_measured_row_carries_its_firing_rate():
    """C19: a conditional win quoted without `selected_frac` overstates itself."""
    for k, v in W.MEASURED_ARMS.items():
        assert "selected_frac" in v, k
        assert v["selected_frac"] == 1.0, (
            f"{k} is not the unconditional rule; a gated row may not live in "
            "MEASURED_ARMS without its firing rate being read")
        assert v["lo"] < v["paired_delta"] < v["hi"]
        assert v["separated"] == (v["lo"] * v["hi"] > 0)


def test_the_module_ships_a_rule_and_cannot_fit_one():
    """No public entry point may consume ground truth. Same rule as C8."""
    banned = ("target", "tgt", "gt", "ground", "future_pose", "label", "canary",
              "fan_err", "y_true", "err")
    for name in W.__all__:
        obj = getattr(W, name)
        if not callable(obj):
            continue
        for p in inspect.signature(obj).parameters:
            assert not any(b in p.lower() for b in banned), (
                f"{name}({p}) looks like a ground-truth input — this module "
                "ships a rule and must contain no fitter")


# --------------------------------------------------------------------------- #
# 2. the cost IS the published formula                                         #
# --------------------------------------------------------------------------- #
def test_cost_is_hand_computable():
    # one window, two candidates, two horizon steps
    fan = torch.tensor([[[[0.0, 0.0], [3.0, 4.0]],       # dists 0, 5 -> 2.5
                         [[1.0, 0.0], [0.0, 0.0]]]])     # dists 1, 0 -> 0.5
    ref = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    cost = W.wm_reference_cost(fan, ref)
    assert torch.allclose(cost, torch.tensor([[2.5, 0.5]]))
    idx, _, tele = W.select_by_wm_reference(fan, ref, scorer="unit")
    assert int(idx[0]) == 1
    assert tele["selected_frac"] == 1.0


def test_cost_matches_the_stream_reference_implementation():
    """Bit-for-bit against `v5_imagination_select.cost_C2_ref`, re-typed here."""
    def cost_C2_ref(fan, imag_ref):                      # the published line
        return (fan - imag_ref[:, None]).norm(dim=-1).mean(dim=-1)

    g = torch.Generator().manual_seed(20260728)
    fan = torch.randn(7, 13, 20, 2, generator=g)
    ref = torch.randn(7, 20, 2, generator=g)
    assert torch.equal(W.wm_reference_cost(fan, ref), cost_C2_ref(fan, ref))


def test_sparse_fan_reads_the_reference_at_its_own_lead_times():
    g = torch.Generator().manual_seed(1)
    ref = torch.randn(3, 20, 2, generator=g)
    fan = torch.randn(3, 5, 4, 2, generator=g)
    horizons = (5, 10, 15, 20)
    got = W.wm_reference_cost(fan, ref, horizons=horizons)
    want = (fan - ref[:, [4, 9, 14, 19]][:, None]).norm(dim=-1).mean(dim=-1)
    assert torch.equal(got, want)


def test_shape_mismatch_raises_instead_of_broadcasting():
    fan = torch.randn(2, 5, 4, 2)
    ref = torch.randn(2, 20, 2)
    with pytest.raises(ValueError, match="different points in time"):
        W.wm_reference_cost(fan, ref)                 # no horizons -> refuse
    with pytest.raises(ValueError, match="reference steps"):
        W.wm_reference_cost(fan, torch.randn(2, 8, 2), horizons=(5, 10, 15, 20))
    with pytest.raises(ValueError, match="batch mismatch"):
        W.wm_reference_cost(torch.randn(2, 5, 20, 2), torch.randn(3, 20, 2))


# --------------------------------------------------------------------------- #
# 3. the reference roll: ONE roll, zero-order hold, no future actions          #
# --------------------------------------------------------------------------- #
class _RecordingPredictor:
    """Records every action window it is shown. Returns a deterministic latent."""

    def __init__(self):
        self.seen = []

    def __call__(self, states, actions):
        self.seen.append(actions.clone())
        return None, states[:, -1] * 0.5


def _step_readout(z_prev, z_hat):
    b = z_prev.shape[0]
    return torch.stack([torch.ones(b), torch.zeros(b), torch.zeros(b)], dim=1)


def test_reference_roll_is_one_roll_zero_order_hold_and_sees_no_future():
    b, win, s, a, k = 2, 4, 6, 3, 5
    states = torch.randn(b, win, s)
    actions = torch.randn(b, win, a)
    p = _RecordingPredictor()
    wp = W.wm_reference_rollout(p, states, actions, _step_readout, k)
    assert wp.shape == (b, k, 2)
    assert len(p.seen) == k, "one roll of k steps, not k rolls"
    last = actions[:, -1]
    # every action appended after step 0 is the HELD observed action
    for j in range(1, k):
        appended = p.seen[j][:, -1]
        assert torch.allclose(appended, last), (
            f"step {j} was driven by something other than the held observed "
            "action — that would make this a plan-conditioned or "
            "expert-future roll, not the C2 reference")


def test_reference_roll_never_receives_a_future_action_argument():
    """`wm_fidelity_ade_2s` hands the world model the TRUE future actions
    (taniteval/rollout.py, actions_source='expert_future'). C2 must not, or the
    'deployable' claim is false. The signature is the guard."""
    sig = inspect.signature(W.wm_reference_rollout).parameters
    assert "future_actions" not in sig and "fut_act" not in sig
    src = inspect.getsource(W.wm_reference_rollout)
    assert "rollout_decode(predictor, states, actions, None," in src


# --------------------------------------------------------------------------- #
# 4. degeneracy — the checks that caught two pure-noise gates upstream         #
# --------------------------------------------------------------------------- #
def test_a_cost_constant_across_candidates_is_refused():
    fan = torch.zeros(3, 8, 20, 2)          # every candidate identical
    ref = torch.randn(3, 20, 2)
    with pytest.raises(NoCandidateAxis, match="CONSTANT"):
        W.select_by_wm_reference(fan, ref, scorer="unit")


def test_telemetry_reports_ties_and_span_and_baseline_agreement():
    fan = torch.zeros(2, 4, 3, 2)
    fan[:, 1] = 1.0                          # candidates 0,2,3 tie exactly
    ref = torch.zeros(2, 3, 2)
    idx, cost, tele = W.select_by_wm_reference(
        fan, ref, baseline_idx=torch.zeros(2, dtype=torch.long), scorer="unit")
    assert tele["n_tied_argmin"] == 2        # both windows have a tied minimum
    assert tele["n_constant_cost_rows"] == 0
    assert tele["cost_span_mean"] == pytest.approx(2.0 ** 0.5, rel=1e-6)
    assert tele["frac_pick_equals_baseline"] == 1.0
    assert tele["selected_frac"] == 1.0
    assert tele["keep_applied"] is False


def test_keep_mask_restricts_and_an_empty_row_keeps_its_whole_fan():
    fan = torch.zeros(2, 3, 2, 2)
    fan[:, 0] = 0.0
    fan[:, 1] = 1.0
    fan[:, 2] = 2.0
    ref = torch.zeros(2, 2, 2)
    keep = torch.tensor([[False, True, True], [False, False, False]])
    idx, _, tele = W.select_by_wm_reference(fan, ref, keep=keep, scorer="unit")
    assert int(idx[0]) == 1, "masked candidate 0 must not be selectable"
    assert int(idx[1]) == 0, "an empty survivor set keeps the whole fan"
    assert tele["keep_applied"] is True and "_unmeasured" in tele


# --------------------------------------------------------------------------- #
# 5. the rule CAN return a failing value — proved, not asserted                #
# --------------------------------------------------------------------------- #
def _policy_err(fan, ref, err):
    idx, _, _ = W.select_by_wm_reference(fan, ref, scorer="unit")
    return err[torch.arange(len(idx)), idx].mean()


def test_the_rule_can_come_out_worse_than_the_baseline():
    """Both directions. Same code, two reference rolls: one that points at the
    good candidate and one that points at the bad one. If only the winning
    direction were constructible, the instrument could not report a loss."""
    fan = torch.zeros(4, 2, 3, 2)
    fan[:, 0] = 0.0                    # candidate 0: the GOOD plan
    fan[:, 1] = 10.0                   # candidate 1: the BAD plan
    err = torch.tensor([[0.10, 3.00]]).repeat(4, 1)
    baseline = err[:, 0].mean()        # a baseline that always picks candidate 0

    good_ref = torch.zeros(4, 3, 2)                       # -> picks 0
    bad_ref = torch.full((4, 3, 2), 10.0)                 # -> picks 1
    assert _policy_err(fan, good_ref, err) == pytest.approx(float(baseline))
    assert _policy_err(fan, bad_ref, err) > baseline, (
        "the rule could not be made to lose — a selection instrument that "
        "cannot report a loss is the C13 class")


# --------------------------------------------------------------------------- #
# 6. you cannot self-score by omission                                         #
# --------------------------------------------------------------------------- #
def test_self_scoring_requires_an_explicit_request():
    with pytest.raises(ValueError, match="no scoring world model was named"):
        W.resolve_scorer_tag(None)
    assert W.resolve_scorer_tag(W.SELF_SCORING) == "self"
    assert W.resolve_scorer_tag("/root/models/flagship-30k/ckpt.pt").endswith("ckpt.pt")


def test_the_refusal_quotes_both_measured_directions():
    """The error message is the only place most callers will read the evidence."""
    with pytest.raises(ValueError) as e:
        W.resolve_scorer_tag(None)
    msg = str(e.value)
    assert "+0.2090" in msg and "-0.2918" in msg
