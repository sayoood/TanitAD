"""Certification for the LATENT ABLATIONS in :mod:`taniteval.blindimag` (2026-07-27).

These arms exist to answer one question: *is the blind-driving horizon the world
model's imagined semantics, or the integration of the action channel?* They
replace ONLY the latent appended to the predictor's window and leave the action
tensor — including the CONSTANT true ``v0`` channel the integrator hypothesis
names — untouched.

Per M3 every guard here is shown capable of FAILING:

* the PLUMBING SELF-TEST runs in **both** directions — an identity permutation
  must reduce each new source to an arm that already exists (bit-identical), and
  the ANTI-NO-OP assertion must reject a source that does not move the path;
* the fixed-point probe is shown to return **both** of its readings;
* the whole extension is pinned inert on every call site that predates it.
"""
import pytest
import torch

from taniteval import blindimag as bi
from tanitad.models.metric_dynamics import StepDisplacementReadout

#: Deliberately SELF-CONTAINED — it does not import ``test_blindimag``'s fixture.
#: A cross-test-module import made this file uncollectable against an older copy
#: of the sibling module on pod2, and a certification that cannot run where the
#: code runs is not a certification. Same shapes and same stub contract.
S, A, WIN, B = 24, 3, 8, 5

_ABL = ("frozen_other", "shuffled", "shuffled_obs", "mean_latent",
        "zero_latent")


class _StubPredictor(torch.nn.Module):
    """1-step predictor with the WorldModel predictor's contract. Depends on
    BOTH the window and the actions, so a test that swaps either can fail."""

    def __init__(self, s=S, a=A):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(s * WIN + a * WIN, s)

    def forward(self, states, actions):
        x = torch.cat([states.flatten(1), actions.flatten(1)], dim=-1)
        return {1: torch.tanh(self.lin(x))}


def _kin_fixture(seed=0, k=12):
    torch.manual_seed(seed)
    states = torch.randn(B, WIN, S)
    actions = torch.randn(B, WIN, A) * 0.1
    obs = torch.randn(B, k, S)
    pred = _StubPredictor().eval()
    ro = StepDisplacementReadout(S).eval()
    v0 = torch.rand(B) * 8 + 2
    return states, actions, obs, pred, ro, v0, k


def _abl(state_source, *, seed=None, action="own_kinematic|blend=0.25", k=12):
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, k)
    kw = {} if seed is None else {"latent_perm_seed": seed}
    return bi.blind_rollout(pred, states, actions, ro, k,
                            state_source=state_source, action_source=action,
                            obs_states=obs, v_last=v0, **kw)["waypoints"]


# --------------------------------------------------------------------------- #
# the spec parser                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec,expect", [
    ("shuffled", ("shuffled", {})),
    ("shuffled|seed=7", ("shuffled", {"seed": 7})),
    ("frozen_other|seed=-1", ("frozen_other", {"seed": -1})),
    ("imagination", ("imagination", {})),
])
def test_parse_state_source(spec, expect):
    assert bi.parse_state_source(spec) == expect


@pytest.mark.parametrize("bad", ["shuffled|blend=0.5", "imagination|seed=3",
                                 "frozen_last|seed=1", "shuffled|seed=x"])
def test_parse_state_source_rejects(bad):
    """A silently-dropped knob is how a sweep produces a flat, wrong table."""
    with pytest.raises(ValueError):
        bi.parse_state_source(bad)


# --------------------------------------------------------------------------- #
# PLUMBING SELF-TEST, direction 1 — the identity permutation must REDUCE       #
# --------------------------------------------------------------------------- #
def test_identity_permutation_reduces_shuffled_to_imagination():
    assert torch.equal(_abl("shuffled|seed=-1"), _abl("imagination"))


def test_identity_permutation_reduces_shuffled_obs_to_full_obs():
    assert torch.equal(_abl("shuffled_obs|seed=-1"), _abl("full_obs"))


def test_identity_permutation_reduces_frozen_other_to_frozen_last():
    assert torch.equal(_abl("frozen_other|seed=-1"), _abl("frozen_last"))


# --------------------------------------------------------------------------- #
# PLUMBING SELF-TEST, direction 2 — ANTI-NO-OP                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", _ABL)
def test_every_latent_ablation_actually_moves_the_path(src):
    """An ablation identical to the intact arm makes the whole table vacuous.
    This is the assertion that fails if a source is silently a no-op."""
    assert not torch.equal(_abl(src, seed=3), _abl("imagination"))


@pytest.mark.parametrize("src", _ABL)
def test_latent_ablation_leaves_the_first_transition_identical(src):
    """Every state source decodes step 1 from the SAME real window, so the first
    transition is bit-identical — the substitution only enters at step 2. This
    is amendment A4's premise (why T_blind's contiguity starts at N=2),
    re-pinned for the new sources."""
    assert torch.equal(_abl(src, seed=3)[:, 0], _abl("imagination")[:, 0])


def test_the_latent_ablations_are_pairwise_distinct():
    got = {s: _abl(s, seed=3) for s in _ABL}
    got["frozen_last"] = _abl("frozen_last")
    got["imagination"] = _abl("imagination")
    names = sorted(got)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not torch.equal(got[x], got[y]), f"{x} == {y}"


def test_permuted_ablations_are_deterministic_in_their_seed():
    assert torch.equal(_abl("shuffled", seed=5), _abl("shuffled", seed=5))
    assert not torch.equal(_abl("shuffled", seed=5), _abl("shuffled", seed=6))


def test_derangement_has_no_fixed_point_and_preserves_the_multiset():
    for b in (2, 3, 5, 32):
        for j in range(6):
            p = bi._derangement(b, 11, j, "cpu")
            assert (p != torch.arange(b)).all(), "a fixed point is a no-op row"
            assert sorted(p.tolist()) == list(range(b))
    assert torch.equal(bi._derangement(5, bi.IDENTITY_PERM_SEED, 0, "cpu"),
                       torch.arange(5))


def test_permuted_ablation_refuses_a_batch_of_one():
    """b=1 has no derangement, so the ablation would silently be a no-op."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 6)
    with pytest.raises(ValueError):
        bi.blind_rollout(pred, states[:1], actions[:1], ro, k,
                         state_source="shuffled", action_source="hold_last",
                         obs_states=obs[:1], v_last=v0[:1])


# --------------------------------------------------------------------------- #
# the substitution really reaches the predictor's window                       #
# --------------------------------------------------------------------------- #
def _spy_windows(state_source, **kw):
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 12)
    seen, orig = [], pred.forward

    def spy(s, a):
        seen.append(s.clone())
        return orig(s, a)
    pred.forward = spy
    try:
        bi.blind_rollout(pred, states, actions, ro, k,
                         state_source=state_source, action_source="hold_last",
                         obs_states=obs, v_last=v0, **kw)
    finally:
        pred.forward = orig
    return states, seen


def test_zero_latent_really_zeroes_the_window():
    _states, seen = _spy_windows("zero_latent")
    assert (seen[-1] == 0).all()          # WIN=8 < k=12 => the whole window


def test_frozen_other_window_holds_a_DIFFERENT_windows_percept():
    states, seen = _spy_windows("frozen_other", latent_perm_seed=3)
    p = bi._derangement(B, 3, 0, "cpu")
    assert torch.equal(seen[-1][:, -1], states[p][:, -1])
    assert not torch.equal(seen[-1][:, -1], states[:, -1])


def test_mean_latent_window_is_the_batch_mean():
    states, seen = _spy_windows("mean_latent")
    assert torch.allclose(seen[-1][:, -1],
                          states[:, -1].mean(dim=0, keepdim=True)
                          .expand(B, -1), atol=1e-6)


# --------------------------------------------------------------------------- #
# ⭐ the ablations do NOT touch the action channel                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", _ABL)
def test_latent_ablation_leaves_the_constant_v0_channel_untouched(src):
    """The integrator hypothesis names the `v0` channel. These arms must leave
    it fully intact, or they are not a test of it."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 12)
    fed = bi.blind_rollout(pred, states, actions, ro, k, state_source=src,
                           action_source="own_kinematic|blend=0.25",
                           obs_states=obs, v_last=v0,
                           latent_perm_seed=3)["fed_actions"]
    ref = actions[:, -1, 2]
    assert torch.equal(fed[..., 2], ref[:, None].expand_as(fed[..., 2]))


@pytest.mark.parametrize("src", _ABL + ("frozen_last",))
def test_at_alpha_one_the_fed_action_is_identical_across_state_sources(src):
    """⭐ THE UNCONFOUNDED ROW. At blend=1.0 the fed steer/accel are the last
    OBSERVED action for every arm, so they cannot vary with the latent — any
    difference in the decoded path is the latent and nothing else."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 12)

    def fed(s):
        return bi.blind_rollout(pred, states, actions, ro, k, state_source=s,
                                action_source="own_kinematic|blend=1.0",
                                obs_states=obs, v_last=v0,
                                latent_perm_seed=3)["fed_actions"]
    assert torch.equal(fed(src), fed("imagination"))


def test_at_alpha_zero_the_fed_action_DOES_depend_on_the_latent():
    """The failing direction of the test above: at alpha=0 the action IS the
    kinematic inverse of the decoded Delta-pose, so a latent ablation moves it.
    That is the confound alpha=1 exists to remove, and it is real."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 12)

    def fed(s):
        return bi.blind_rollout(pred, states, actions, ro, k, state_source=s,
                                action_source="own_kinematic",
                                obs_states=obs, v_last=v0,
                                latent_perm_seed=3)["fed_actions"]
    assert not torch.equal(fed("zero_latent"), fed("imagination"))


# --------------------------------------------------------------------------- #
# the FIXED-POINT probe                                                        #
# --------------------------------------------------------------------------- #
def test_latent_stats_are_off_by_default_and_inert_when_on():
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 12)
    kw = dict(state_source="imagination", action_source="own_kinematic",
              obs_states=obs, v_last=v0)
    a = bi.blind_rollout(pred, states, actions, ro, k, **kw)
    b = bi.blind_rollout(pred, states, actions, ro, k, latent_stats=True, **kw)
    assert "lat_dz" not in a and "lat_dz" in b
    assert torch.equal(a["waypoints"], b["waypoints"])


def test_latent_stats_are_the_quantities_they_name():
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 6)
    r = bi.blind_rollout(pred, states, actions, ro, k,
                         state_source="imagination", action_source="hold_last",
                         obs_states=obs, v_last=v0, latent_stats=True)
    for key in ("lat_dz", "lat_d0", "lat_cos0", "lat_norm"):
        assert r[key].shape == (B, k)
    # step 1 is measured against the last REAL percept, so d0 == dz there
    assert torch.allclose(r["lat_d0"][:, 0], r["lat_dz"][:, 0], atol=1e-6)
    assert (r["lat_cos0"].abs() <= 1.0 + 1e-5).all()
    assert (r["lat_norm"] > 0).all()


def test_fixed_point_probe_can_report_BOTH_readings():
    """C13: a probe whose reading is structural is worthless. A frozen context
    drives the predictor to a repeating latent (step size -> 0 exactly); a live
    one does not. Both readings are attainable, so the criterion discriminates."""
    states, actions, obs, pred, ro, v0, k = _kin_fixture(0, 20)
    live = bi.blind_rollout(pred, states, actions, ro, k,
                            state_source="imagination",
                            action_source="own_kinematic", obs_states=obs,
                            v_last=v0, latent_stats=True)["lat_dz"]
    froz = bi.blind_rollout(pred, states, actions, ro, k,
                            state_source="frozen_last",
                            action_source="hold_last", obs_states=obs,
                            v_last=v0, latent_stats=True)["lat_dz"]
    assert float(froz[:, -1].mean()) < 1e-6          # FIXED POINT reading
    assert float(live[:, -1].mean()) > float(froz[:, -1].mean())


# --------------------------------------------------------------------------- #
# inertness on everything that predates the extension                          #
# --------------------------------------------------------------------------- #
def test_no_ablation_leaves_every_pre_existing_path_bit_identical():
    states, actions, obs, pred, ro, v0, k = _kin_fixture()
    fut = torch.randn(B, k, A) * 0.1
    for ss in ("imagination", "frozen_last", "full_obs", "observed_pair"):
        for spec in ("own_kinematic", "hold_last", "true_future",
                     "own_kinematic|blend=0.25"):
            a = bi.blind_rollout(pred, states, actions, ro, k, state_source=ss,
                                 action_source=spec, obs_states=obs,
                                 v_last=v0, future_actions=fut)["waypoints"]
            b = bi.blind_rollout(pred, states, actions, ro, k, state_source=ss,
                                 action_source=spec, obs_states=obs,
                                 v_last=v0, future_actions=fut,
                                 latent_perm_seed=99,
                                 latent_stats=False)["waypoints"]
            assert torch.equal(a, b)


def test_step_readout_contract_unchanged():
    """Sanity: the readout the ablations share is still the grounded one."""
    assert issubclass(StepDisplacementReadout, torch.nn.Module)
