"""Pins ``tanitad.rl.ddv2_il`` — lever **L1 / D9**, the two declared changes to DDv2's RL stage.

⭐ THE CENTRAL TEST OF THIS FILE IS A **MUTATION PROOF**, not an inspection.
``test_MUTATION_a_synthetic_fan_survives_the_matched_term_and_does_NOT_survive_the_release``
builds a wide synthetic fan, optimises it under BOTH imitation terms, and requires:

* the RELEASE's all-modes term to **collapse** it (the defect is re-introduced and the guard's
  failure branch is shown to be reachable — an assertion that only ever sees the fixed code
  proves nothing: MEMORY "Guards need mutation, not inspection", where an AST census read 0
  suspects on BOTH the fixed and the broken trainer);
* the MATCHED-ANCHOR term to **preserve** it.

The other evidence classes, in order of strength:

1. **The release form it replaces, executed side by side** — the all-modes term is kept as a
   named function and every departure is measured against it, never against a description.
2. **Analytic literals** — L1 values written by hand on tensors small enough to add up on paper.
3. **Deliberate regressions** — anchor-major gather, a winner-take-all over the predictions
   instead of the bank, matching against a fixed vocabulary instead of the decoded bank, and a
   clip applied per rollout step instead of once before the optimiser step.

⛔ NOTHING HERE TOUCHES THE GPU. The lever was implemented, tested and pre-registered on CPU;
the four GPU arms are the Master Mind's to schedule.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

from tanitad.rl import ddv2_il as L
from tanitad.rl import ddv2_rl as D

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNNER = REPO / "stack" / "scripts" / "ddv2_rl_refcv5.py"
CHECKER = (REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research"
           / "2026-09-16-refcv6-rl-l1" / "code" / "check_arm_l1.py")


def _load(path, name):
    if not path.exists():
        pytest.skip(f"{name} not in this (sparse) checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner():
    return _load(RUNNER, "ddv2_rl_refcv5_for_test")


def _checker():
    return _load(CHECKER, "check_arm_l1_for_test")


class _TrainParser:
    """Drives ``runner.main`` up to the point where it has parsed, and no further.

    ⛔ Reads the defaults from the REAL CLI rather than re-declaring them here: a test that
    re-states the default it is checking cannot catch a default that drifted.
    """

    def __init__(self, runner):
        self.runner = runner

    def parse_args(self, argv):
        captured = {}
        orig = self.runner.cmd_train
        self.runner.cmd_train = lambda a: captured.__setitem__("a", a)
        try:
            # a non-default --ckpt so `main`'s cold-start md5 guard (a 1.3 GB read) is skipped
            self.runner.main(["train", "--ckpt", str(RUNNER), *argv])
        finally:
            self.runner.cmd_train = orig
        return captured["a"]


def _train_parser(runner):
    return _TrainParser(runner)


# --------------------------------------------------------------------------- #
# 0. the settings object cannot be ambiguous about which recipe ran            #
# --------------------------------------------------------------------------- #
def test_default_settings_name_the_lever_and_is_release_is_honest():
    assert L.IlSettings().form == L.RELEASE_IL_FORM
    assert L.IlSettings().grad_clip == L.GRAD_CLIP
    # the release is the all-modes form at lambda 1.0 with NO clipping — all three
    assert L.IlSettings(grad_clip=None).is_release
    assert L.IlSettings(grad_clip=0.0).is_release
    assert not L.IlSettings().is_release                       # clipping is a departure
    assert not L.IlSettings(form="matched_anchor", grad_clip=None).is_release


def test_AMENDMENT_A1_the_default_clip_is_a_spike_guard_not_an_every_step_rescale():
    """⭐ AMENDMENT A-1, 2026-09-16, made BEFORE any arm ran.

    The pre-registered max-norm was 1.0. MEASURED on the banked logs and verified independently
    by the coordinator: ``> 1.0`` binds on **600/600** steps of all three 2026-09-15 arms
    (minimum norm 1.87 / 2.22 / 3.45, medians 16.67 / 17.93 / 61.96), so 1.0 is a 17-62×
    rescale on EVERY step; ``> 100`` binds on **1/600** for each stable arm and **276/600** for
    the diverged seed. The default is therefore 100, and this test pins the amended value so it
    cannot drift back silently.
    """
    assert L.GRAD_CLIP == 100.0
    assert L.IlSettings().grad_clip == 100.0
    # 1.0 and the release's 0 stay available as declared arms
    assert L.IlSettings(grad_clip=1.0).grad_clip == 1.0
    assert L.IlSettings(grad_clip=0.0).is_release
    # the amendment is a THRESHOLD change only: it is still one call, still before opt.step()
    ps = _params_with_grad(15712.0)
    out = L.clip_gradients(ps, L.IlSettings())
    assert out["grad_clip"] == 100.0 and out["grad_clipped"]
    assert abs(math.sqrt(sum(float((p.grad ** 2).sum()) for p in ps)) - 100.0) < 1e-2
    # and the median banked step (norm ~17) is now UNtouched, where 1.0 would have rescaled it
    ps = _params_with_grad(16.67)
    assert not L.clip_gradients(ps, L.IlSettings())["grad_clipped"]
    ps = _params_with_grad(16.67)
    assert L.clip_gradients(ps, L.IlSettings(grad_clip=1.0))["grad_clipped"]


def test_the_lambda_arm_is_lambda_0p01_on_the_row_class_that_binds():
    """D6 MEASURED that essentially every cold-start row carries a positive advantage, so the
    0.1 row weight is the one that binds: scale 0.1 makes it 0.01."""
    s = L.IlSettings(form="release_all_modes_lambda", lambda_scale=0.1)
    eff = s.effective_lambda()
    assert math.isclose(eff["rows_with_positive_advantage"], 0.01, rel_tol=1e-12)
    assert math.isclose(eff["rows_with_no_positive_advantage"], 0.1, rel_tol=1e-12)


@pytest.mark.parametrize("kw, frag", [
    ({"form": "nearest"}, "not in"),
    ({"lambda_scale": 0.0}, "DELETES the term"),
    ({"lambda_scale": -1.0}, "must be > 0"),
    ({"grad_clip": -1.0}, "must be >= 0"),
    ({"form": "release_all_modes_lambda", "lambda_scale": 1.0}, "IS the release"),
])
def test_settings_refuse_the_records_that_would_be_ambiguous(kw, frag):
    with pytest.raises(D.Ddv2ConfigError) as e:
        L.IlSettings(**kw)
    assert frag in str(e.value)


# --------------------------------------------------------------------------- #
# 1. the matched-anchor index IS the trainer's arithmetic                      #
# --------------------------------------------------------------------------- #
def test_matched_anchor_index_is_the_trainers_own_expression():
    """`refc_v3_train.py:2383-2386`, re-typed here from the trainer and not imported, so the
    two are independently authored and a drift in either is visible."""
    torch.manual_seed(0)
    b, n, s = 3, 7, 8
    bank = torch.randn(b, n, s, 2)
    gt = torch.randn(b, s, 2)
    sv = torch.ones(b, s)
    # the trainer's line, verbatim
    dist = (((gt[:, None] - bank) ** 2).sum(-1) * sv[:, None]).sum(-1)
    a_star_trainer = dist.argmin(dim=1)
    assert torch.equal(L.matched_anchor_index(bank, gt), a_star_trainer)
    assert torch.equal(L.matched_anchor_index(bank, gt, sv.bool()), a_star_trainer)


def test_matched_anchor_index_literal_by_hand():
    """Two waypoints, three anchors, distances computable on paper."""
    gt = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])                    # straight ahead
    bank = torch.tensor([[
        [[0.0, 0.0], [10.0, 6.0]],       # sq dist 36
        [[0.0, 0.0], [10.0, 1.0]],       # sq dist  1   <- nearest
        [[0.0, 0.0], [10.0, -3.0]],      # sq dist  9
    ]])
    assert int(L.matched_anchor_index(bank, gt)[0]) == 1


def test_REGRESSION_a_masked_out_waypoint_changes_which_anchor_wins():
    """The `valid` mask is not decoration: it must be able to change `a_star`."""
    gt = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
    bank = torch.tensor([[
        [[0.0, 9.0], [10.0, 0.0]],       # bad at t0, perfect at t1
        [[0.0, 0.0], [10.0, 5.0]],       # perfect at t0, bad at t1
    ]])
    assert int(L.matched_anchor_index(bank, gt)[0]) == 1                       # 25 < 81
    mask = torch.tensor([[False, True]])
    assert int(L.matched_anchor_index(bank, gt, mask)[0]) == 0                 # only t1 counts


# --------------------------------------------------------------------------- #
# 2. the two L1 forms, by hand                                                 #
# --------------------------------------------------------------------------- #
def _fan(b=2, g=4, n=5, s=3, spread=8.0):
    """A deterministic wide fan: anchor `k`'s endpoint sits at lateral offset `k*spread`."""
    lat = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * spread        # [N]
    t = torch.linspace(0.0, 1.0, s)                                           # [S]
    path = torch.zeros(b, g, n, s, 2)
    path[..., 0] = torch.arange(1, s + 1, dtype=torch.float32) * 5.0          # along-track
    path[..., 1] = lat.view(1, 1, n, 1) * t.view(1, 1, 1, s)                  # fans out
    return path.reshape(b, g * n, s, 2).clone()


def test_all_modes_il_is_exactly_the_expression_the_release_ran():
    torch.manual_seed(1)
    path, gt = torch.randn(2, 12, 4, 2), torch.randn(2, 4, 2)
    assert torch.equal(L.all_modes_il(path, gt), (path - gt[:, None]).abs().mean())


def test_matched_anchor_il_literal_and_only_the_matched_chains_enter_it():
    b, g, n, s = 1, 2, 3, 2
    path = torch.zeros(b, g * n, s, 2)
    # group-major m = g*N + k: anchor 1's two chains are m = 1 and m = 4
    path[0, 1] = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    path[0, 4] = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    path[0, 0] = path[0, 2] = path[0, 3] = path[0, 5] = torch.full((s, 2), 999.0)
    gt = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    # |1|+|0|+|3|+|0| = 4 ; |1|+|2|+|3|+|0| = 6 ; mean over G*S*2 = 8 entries -> 10/8
    got = L.matched_anchor_il(path, gt, torch.tensor([1]), n)
    assert abs(float(got) - 1.25) < 1e-6, "the 999.0 chains must not enter the term at all"


def test_with_one_anchor_the_matched_term_degenerates_to_the_release_term():
    torch.manual_seed(2)
    path, gt = torch.randn(3, 4, 5, 2), torch.randn(3, 5, 2)
    a = torch.zeros(3, dtype=torch.long)
    assert torch.allclose(L.matched_anchor_il(path, gt, a, 1), L.all_modes_il(path, gt),
                          atol=0, rtol=0)


def test_group_major_gather_picks_the_right_chains():
    """⛔ THE DEFECT THIS CATCHES IS INVISIBLE BY SHAPE. An anchor-major reshape produces a
    tensor of the SAME shape [B, G, S, 2] holding DIFFERENT chains."""
    b, g, n, s = 1, 3, 4, 2
    path = torch.arange(b * g * n * s * 2, dtype=torch.float32).reshape(b, g * n, s, 2)
    a_star = torch.tensor([2])
    gt = torch.zeros(b, s, 2)
    want_rows = [0 * n + 2, 1 * n + 2, 2 * n + 2]                # group-major m = g*N + n
    want = (path[0, want_rows] - gt[0]).abs().mean()
    assert torch.allclose(L.matched_anchor_il(path, gt, a_star, n), want)
    # the anchor-major mutation: m = n*G + g would gather rows 6, 7, 8 instead
    wrong_rows = [2 * g + j for j in range(g)]
    wrong = (path[0, wrong_rows] - gt[0]).abs().mean()
    assert not torch.allclose(want, wrong), "fixture too weak to separate the two layouts"


@pytest.mark.parametrize("bad, frag", [
    (dict(n_anchors=4), "not a whole number of groups"),
    (dict(a_star=torch.tensor([9])), "out of range"),
    (dict(a_star=torch.tensor([1.0])), "integer index"),
    (dict(a_star=torch.tensor([0, 0])), "must be [B]"),
])
def test_matched_anchor_il_refuses_the_silently_wrong_call(bad, frag):
    path, gt = torch.zeros(1, 6, 2, 2), torch.zeros(1, 2, 2)
    kw = {"a_star": torch.tensor([1]), "n_anchors": 3}
    kw.update(bad)
    with pytest.raises(D.Ddv2ConfigError) as e:
        L.matched_anchor_il(path, gt, kw["a_star"], kw["n_anchors"])
    assert frag in str(e.value)


def test_imitation_term_dispatches_and_stays_a_zero_dim_scalar():
    """``per_step_loss`` refuses a non-scalar IL; every form must satisfy that contract."""
    b, g, n, s = 2, 2, 3, 4
    path, gt = _fan(b, g, n, s), torch.randn(b, s, 2)
    bank = _fan(1, 1, n, s)[0][None].expand(b, n, s, 2).contiguous()
    for st in (L.IlSettings(), L.IlSettings(form="release_all_modes_lambda", lambda_scale=0.1)):
        v = L.imitation_term(path, gt, st)
        assert v.dim() == 0 and torch.equal(v, L.all_modes_il(path, gt))
    v = L.imitation_term(path, gt, L.IlSettings(form="matched_anchor"), bank=bank)
    assert v.dim() == 0
    assert torch.equal(v, L.matched_anchor_il(path, gt, L.matched_anchor_index(bank, gt), n))
    with pytest.raises(D.Ddv2ConfigError, match="needs `bank`"):
        L.imitation_term(path, gt, L.IlSettings(form="matched_anchor"))


# --------------------------------------------------------------------------- #
# 3. ⭐ THE MUTATION PROOF — a synthetic fan whose spread must survive          #
# --------------------------------------------------------------------------- #
def _spread(path, n):
    return L.fan_endpoint_spread(path, n)


def _optimise_fan(form, steps=400, lr=0.2, b=2, g=4, n=9, s=6, spread=8.0):
    """Optimise a free fan under ONE imitation term alone and report its endpoint spread.

    The 'model' is the fan itself — the cleanest possible isolation of the question "does this
    objective collapse the modes?", with no decoder, no reward and no policy gradient to share
    the blame. The GT is anchor 0's own path, so a term that preserves modes has a mode to
    preserve and a term that does not has somewhere to drag everything.

    ⚠️ Adam, not SGD, and that is the point of a FAIR comparison rather than a convenience.
    An L1 gradient is ``sign(err) / (B·M·S·2)``, so plain SGD's step size is inversely
    proportional to M — the all-modes term would move 117× slower than the matched one purely
    through its denominator and would 'preserve' the fan by being too slow to collapse it, which
    is exactly the false PASS this test exists to avoid. Adam normalises the per-element step to
    ≈ ``lr`` for BOTH forms, so what separates them is WHICH elements get a gradient at all —
    the mechanism under test. It is also the family the real runs use (AdamW 2e-4).
    """
    torch.manual_seed(0)
    theta = _fan(b, g, n, s, spread).clone().requires_grad_(True)
    bank = _fan(1, 1, n, s, spread)[0][None].expand(b, n, s, 2).contiguous()
    gt = bank[:, 0].clone()                                    # anchor 0 is the logged plan
    a_star = L.matched_anchor_index(bank, gt)
    assert torch.equal(a_star, torch.zeros(b, dtype=torch.long)), "fixture: GT must match a0"
    opt = torch.optim.Adam([theta], lr=lr)
    before = _spread(theta, n)
    for _ in range(steps):
        opt.zero_grad()
        (L.all_modes_il(theta, gt) if form == "all_modes"
         else L.matched_anchor_il(theta, gt, a_star, n)).backward()
        opt.step()
    return before, _spread(theta, n), theta.detach()


def test_MUTATION_a_synthetic_fan_survives_the_matched_term_and_does_NOT_survive_the_release():
    """⭐⭐ THE PROOF THE LEVER IS WORTH RUNNING, and the defect it removes is REACHABLE.

    Held-out MEASURED 2026-09-15: the release's all-modes IL term took refcv5-v2's fan endpoint
    spread from 37.5 m to 2.45 m (−93 %). This test reproduces the mechanism in isolation and
    requires the replacement to NOT do it.
    """
    before_a, after_a, _ = _optimise_fan("all_modes")
    before_m, after_m, theta_m = _optimise_fan("matched_anchor")
    assert before_a == before_m > 5.0, "fixture: the fan must start wide"

    # (a) the DEFECT, re-introduced, must be reachable: the release's term collapses the fan
    assert after_a < 0.10 * before_a, (
        f"the release's all-modes term did NOT collapse this fan ({before_a:.3f} -> "
        f"{after_a:.3f} m). The mutation is not reachable, so a PASS on (b) means nothing.")

    # (b) the REPLACEMENT must preserve it
    assert after_m > 0.95 * before_m, (
        f"the matched-anchor term collapsed the fan ({before_m:.3f} -> {after_m:.3f} m); it is "
        "not mode preserving and the lever is refuted before it reaches a GPU.")

    # (c) and the two must be separated by more than measurement noise
    assert after_m > 8.0 * after_a


def test_MUTATION_only_the_matched_anchors_chains_moved_at_all():
    """The mechanism, not just the summary statistic: every UNmatched chain is bit-unchanged."""
    b, g, n, s = 2, 4, 9, 6
    _, _, theta = _optimise_fan("matched_anchor", b=b, g=g, n=n, s=s)
    start = _fan(b, g, n, s)
    moved = (theta - start).reshape(b, g, n, s, 2).abs().amax(dim=(0, 1, 3, 4))   # [N]
    assert float(moved[0]) >= 0.0                       # anchor 0 is the match (may already fit)
    assert float(moved[1:].max()) == 0.0, (
        f"an UNMATCHED anchor moved by {float(moved[1:].max()):.3e} m — the term is leaking "
        "gradient onto modes it must not touch")
    # the same run under the release's term moves every one of them
    _, _, theta_a = _optimise_fan("all_modes", b=b, g=g, n=n, s=s)
    moved_a = (theta_a - start).reshape(b, g, n, s, 2).abs().amax(dim=(0, 1, 3, 4))
    assert float(moved_a[1:].min()) > 0.0, "fixture too weak: the release's term moved nothing"


def test_REGRESSION_a_winner_take_all_over_the_PREDICTIONS_is_a_different_objective():
    """⛔ Matching on `path` instead of on the fixed bank would let the objective re-choose its
    mode every step — no fixed mode is supervised, and the match is not attributable to an
    anchor. It must not be silently equal to the bank match."""
    b, g, n, s = 1, 2, 5, 4
    path = _fan(b, g, n, s)
    bank = _fan(1, 1, n, s)[0][None].expand(b, n, s, 2).contiguous()
    gt = bank[:, 0].clone()
    a_bank = L.matched_anchor_index(bank, gt)          # the BANK match: anchor 0, always
    # a training step has already moved the predictions off their anchors: anchor 0's chains
    # drifted away and anchor 3's chain happens to sit on the GT right now
    path[0, 0 * n + 0] = path[0, 1 * n + 0] = gt[0] + 50.0
    path[0, 0 * n + 3] = gt[0]
    a_pred = ((path - gt[:, None]).pow(2).sum(-1).sum(-1).argmin(dim=1) % n)
    assert int(a_bank[0]) == 0 and int(a_pred[0]) == 3
    assert not torch.allclose(L.matched_anchor_il(path, gt, a_bank, n),
                              L.matched_anchor_il(path, gt, a_pred, n))


def test_REGRESSION_matching_against_a_fixed_vocabulary_is_not_matching_the_decoded_bank():
    """The trainer's own refusal (`refc_v3_train.py:2376-2382`): on a v0-conditioned build the
    decoded bank is NOT `decoder.anchors`, and matching the wrong one reads plausibly."""
    b, n, s = 2, 6, 4
    ref_bank = _fan(1, 1, n, s, spread=8.0)[0][None].expand(b, n, s, 2).contiguous()
    rolled = ref_bank * torch.tensor([2.0, 0.25])          # this window's v0-rolled geometry
    gt = rolled[:, 4].clone()
    assert int(L.matched_anchor_index(rolled, gt)[0]) == 4
    assert int(L.matched_anchor_index(ref_bank, gt)[0]) != 4


# --------------------------------------------------------------------------- #
# 4. the lambda arm scales the IL coefficient and NOTHING else                 #
# --------------------------------------------------------------------------- #
def _weights(b=3, m=8, t=10):
    torch.manual_seed(3)
    adv = torch.randn(b, m)
    adv[0] = 0.0                                      # a row with no positive -> weight 1.0
    return D.step_loss_weights(adv, t)


def test_lambda_scale_1_returns_the_same_dict_object_so_off_is_off_by_construction():
    w = _weights()
    assert L.apply_lambda_scale(w, L.IlSettings()) is w


def test_lambda_scale_multiplies_only_the_il_coefficient_and_the_row_weights():
    w = _weights()
    s = L.IlSettings(form="release_all_modes_lambda", lambda_scale=0.1)
    out = L.apply_lambda_scale(w, s)
    assert torch.allclose(out["il_coef"], w["il_coef"] * 0.1)
    assert torch.allclose(out["il_weight_b"], w["il_weight_b"] * 0.1)
    assert out["coef_rl"] is w["coef_rl"], "a lambda arm must not touch the policy gradient"
    for k in ("n_nonzero_b", "adv_discounted"):
        assert out[k] is w[k]
    # the row structure the release derives from the advantage survives the scale
    assert float(w["il_weight_b"][0]) == 1.0
    assert float(out["il_weight_b"][0]) == pytest.approx(0.1, rel=1e-6)


# --------------------------------------------------------------------------- #
# 5. gradient clipping                                                         #
# --------------------------------------------------------------------------- #
def _params_with_grad(norm: float, n=3):
    ps = [torch.zeros(4, requires_grad=True) for _ in range(n)]
    torch.manual_seed(5)
    g = [torch.randn(4) for _ in ps]
    cur = math.sqrt(sum(float((x ** 2).sum()) for x in g))
    for p, x in zip(ps, g):
        p.grad = x * (norm / cur)
    return ps


@pytest.mark.parametrize("clip", [100.0, 1.0])
def test_clip_reports_the_PRE_clip_norm_and_binds_at_the_measured_divergence(clip):
    """RL-s1 diverged at grad norm 15,712.3 (RESULT §7.1); RL-s0's last-50 mean was 14.28.

    Parametrised over the amended default (100) and the superseded value (1.0), which stays an
    available arm — so neither is tested only through the default.
    """
    ps = _params_with_grad(15712.0)
    before = [p.grad.clone() for p in ps]
    out = L.clip_gradients(ps, L.IlSettings(grad_clip=clip))
    assert abs(out["grad_norm"] - 15712.0) < 1.0, "the logged norm must stay the PRE-clip one"
    assert out["grad_clipped"] and out["grad_clip"] == clip
    post = math.sqrt(sum(float((p.grad ** 2).sum()) for p in ps))
    assert abs(post - clip) < 1e-2 and abs(out["grad_norm_clipped"] - clip) < 1e-6
    # direction preserved: the update is smaller, not a different update
    for p, b0 in zip(ps, before):
        assert torch.allclose(p.grad, b0 * (clip / 15712.0), atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("clip, norm", [(100.0, 16.67), (100.0, 0.25), (1.0, 0.25)])
def test_clip_does_not_bind_below_the_max_norm(clip, norm):
    ps = _params_with_grad(norm)
    before = [p.grad.clone() for p in ps]
    out = L.clip_gradients(ps, L.IlSettings(grad_clip=clip))
    assert not out["grad_clipped"] and abs(out["grad_norm"] - norm) < 1e-4
    assert abs(out["grad_norm_clipped"] - norm) < 1e-4
    for p, b0 in zip(ps, before):
        assert torch.equal(p.grad, b0)


def test_REGRESSION_with_the_lever_off_the_release_gradient_is_bit_unchanged():
    for st in (L.IlSettings(grad_clip=None), L.IlSettings(grad_clip=0.0)):
        ps = _params_with_grad(15712.0)
        before = [p.grad.clone() for p in ps]
        out = L.clip_gradients(ps, st)
        assert out["grad_clip"] is None and not out["grad_clipped"]
        assert out["grad_norm"] == out["grad_norm_clipped"]
        for p, b0 in zip(ps, before):
            assert torch.equal(p.grad, b0), "the release path must not write any .grad"


def test_clip_refuses_to_report_a_norm_when_nothing_carries_a_gradient():
    with pytest.raises(D.Ddv2ConfigError, match="no parameter carries a gradient"):
        L.clip_gradients([torch.zeros(4, requires_grad=True)], L.IlSettings())


@pytest.mark.parametrize("clip", [100.0, 1.0])
def test_REGRESSION_clipping_per_rollout_step_is_not_clipping_the_update(clip):
    """The grad pass is decomposed over T=10 steps. Clipping inside that loop clips ten partial
    gradients to the max-norm EACH and can leave a TOTAL far above it — the defect this call
    site's placement (once, before ``opt.step()``) avoids. Parametrised over the amended
    default and the superseded 1.0, so the placement is proven independent of the threshold."""
    t = 10
    st = L.IlSettings(grad_clip=clip)
    p_once = torch.zeros(4, requires_grad=True)
    p_each = torch.zeros(4, requires_grad=True)
    torch.manual_seed(7)
    parts = [torch.randn(4) * (500.0 * clip) for _ in range(t)]
    p_once.grad = sum(parts)
    L.clip_gradients([p_once], st)
    p_each.grad = torch.zeros(4)
    for part in parts:                              # the mutation: clip every partial
        tmp = torch.zeros(4, requires_grad=True)
        tmp.grad = part.clone()
        L.clip_gradients([tmp], st)
        p_each.grad = p_each.grad + tmp.grad
    assert abs(float(p_once.grad.norm()) - clip) < 1e-3 * clip
    assert float(p_each.grad.norm()) > 1.5 * clip, (
        "fixture too weak: per-step clipping happened to stay under the max-norm")


# --------------------------------------------------------------------------- #
# 6. telemetry                                                                 #
# --------------------------------------------------------------------------- #
def test_fan_endpoint_spread_is_the_heldout_statistic_on_one_group():
    b, g, n, s = 2, 3, 5, 4
    path = _fan(b, g, n, s)
    ends = path.reshape(b, g, n, s, 2)[:, 0, :, -1]
    assert abs(L.fan_endpoint_spread(path, n) - float(torch.cdist(ends, ends).mean())) < 1e-6
    # a collapsed fan reads ~0
    flat = path.clone()
    flat.reshape(b, g, n, s, 2)[...] = path.reshape(b, g, n, s, 2)[:, :, :1]
    assert L.fan_endpoint_spread(flat, n) < 1e-6


def test_anchor_match_diagnostics_flag_a_degenerate_match():
    d = L.anchor_match_diagnostics(torch.tensor([3, 3, 3, 3]), 117)
    assert d == {"n_distinct_anchors_matched": 1, "modal_anchor": 3, "modal_frac": 1.0}
    d2 = L.anchor_match_diagnostics(torch.tensor([0, 1, 2, 2]), 117)
    assert d2["n_distinct_anchors_matched"] == 3 and d2["modal_frac"] == 0.5


# --------------------------------------------------------------------------- #
# 7. every OTHER release element is untouched                                  #
# --------------------------------------------------------------------------- #
def test_the_release_path_is_bit_identical_when_the_lever_is_off():
    """``il_form=release`` + ``grad_clip=0`` must reproduce the 2026-09-15 arithmetic exactly:
    the same IL scalar, the same weights dict object, and no write to any ``.grad``."""
    torch.manual_seed(11)
    path, gt = torch.randn(2, 12, 4, 2), torch.randn(2, 4, 2)
    off = L.IlSettings(grad_clip=0.0)
    assert off.is_release
    assert torch.equal(L.imitation_term(path, gt, off), (path - gt[:, None]).abs().mean())
    w = _weights()
    assert L.apply_lambda_scale(w, off) is w


def test_the_runner_wires_each_flag_to_the_settings_it_names():
    """⛔ The most likely silent failure of a lever is not the arithmetic — it is a flag that
    never reaches the loss. This pins flag -> IlSettings -> the argparse defaults."""
    runner = _runner()
    assert runner.IL_FORM_FLAGS == {"release": ("release_all_modes", 1.0),
                                    "matched": ("matched_anchor", 1.0),
                                    "lambda": ("release_all_modes_lambda", 0.1)}
    # the default ChainArm is the RELEASE — a call site that does not name the lever is not given it
    from tanitad.rl import ddv2_refc_chain as C
    assert runner.ChainArm("rl", C.ChainSettings()).il.is_release
    # and the train CLI's defaults ARE the lever, with `release` + `0` reproducing 2026-09-15
    ap = _train_parser(runner)
    d = ap.parse_args(["--arm", "rl", "--steps", "1", "--out-dir", "x"])
    assert d.il_form == "matched" and d.grad_clip == L.GRAD_CLIP == 100.0
    r = ap.parse_args(["--arm", "rl", "--steps", "1", "--out-dir", "x",
                       "--il-form", "release", "--grad-clip", "0"])
    form, lam = runner.IL_FORM_FLAGS[r.il_form]
    assert L.IlSettings(form=form, lambda_scale=lam,
                        grad_clip=None if r.grad_clip == 0 else r.grad_clip).is_release


def test_the_checker_and_the_runner_agree_on_what_each_flag_means():
    assert _checker().IL_FORM_FLAGS == _runner().IL_FORM_FLAGS


# --------------------------------------------------------------------------- #
# 7b. every L1 integrity check must be able to FAIL                            #
# --------------------------------------------------------------------------- #
def _rows(n=4, form="matched_anchor", scale=1.0, clip=1.0):
    out = []
    for i in range(n):
        out.append({"step": i, "loss": 1.0, "rl_part": 0.1, "il_mean_m": 1.5,
                    "grad_norm": 0.5 + i, "param_delta_norm": 0.1 * (i + 1), "reward_mean": 0.8,
                    "rl_coef_abs_sum": 1.0, "frac_positive_after_bar": 0.05,
                    "human_nc_eq_1_frac": 1.0, "il_form": form, "il_lambda_scale": scale,
                    "il_all_modes_m": 6.4, "il_matched_anchor_m": 1.5,
                    "grad_norm_clipped": min(0.5 + i, clip) if clip else 0.5 + i,
                    "grad_clipped": bool(clip) and (0.5 + i) > clip, "grad_clip": clip,
                    "chain_endpoint_spread_m": 30.0,
                    "match_modal_frac": 0.5, "match_n_distinct_anchors_matched": 3,
                    "match_modal_anchor": 7})
    return out


def _run(form="matched_anchor", scale=1.0, clip=1.0):
    return {"il": {"form": form, "lambda_scale": scale}, "grad_clip": clip}


def test_MUTATION_each_L1_integrity_check_fires_on_its_own_defect():
    """⛔ A checker that cannot fail has never passed anything. Each defect below is the one its
    criterion exists to catch, re-introduced, and the criterion is REQUIRED to name it."""
    ck = _checker()
    base = dict(steps=4, arm="rl", il_form="matched", grad_clip=1.0)
    assert ck.check(_rows(), _run(), **base) == [], "the clean fixture must pass"

    def fires(rows, run, tag, **over):
        kw = dict(base); kw.update(over)
        got = ck.check(rows, run, **kw)
        assert any(p.startswith(tag) for p in got), f"{tag} did not fire; got {got}"

    # I1 — a non-finite step, and parameters that never moved
    r = _rows(); r[2]["loss"] = float("nan"); fires(r, _run(), "I1")
    r = _rows(); [x.__setitem__("param_delta_norm", 0.0) for x in r]; fires(r, _run(), "I1")
    # I3 — no positive advantage
    r = _rows(); [x.__setitem__("frac_positive_after_bar", 0.0) for x in r]; fires(r, _run(), "I3")
    # I7 — the arm SAYS matched and RAN the release (the failure this package exists to avoid)
    fires(_rows(form="release_all_modes"), _run(form="release_all_modes"), "I7")
    fires(_rows(), _run(clip=None), "I7")                       # record says unclipped
    r = _rows(); r[1]["il_form"] = "release_all_modes"; fires(r, _run(), "I7")
    # I8 — il_mean_m is the all-modes number while the arm claims matched
    r = _rows(); [x.__setitem__("il_mean_m", x["il_all_modes_m"]) for x in r]
    fires(r, _run(), "I8")
    r = _rows(); [x.__setitem__("il_all_modes_m", x["il_matched_anchor_m"]) for x in r]
    fires(r, _run(), "I8")                                      # the gather selected everything
    # and the mirror check on a RELEASE arm
    r = _rows(form="release_all_modes")                          # il_mean_m still the matched one
    fires(r, _run(form="release_all_modes"), "I8", il_form="release")
    # I9 — every window matched one anchor
    r = _rows(); [x.__setitem__("match_modal_frac", 1.0) for x in r]; fires(r, _run(), "I9")
    # I10 — a norm above the max-norm survived, and a dead clip flag
    r = _rows(); r[3]["grad_norm_clipped"] = 4.0; fires(r, _run(), "I10")
    r = _rows(); [x.__setitem__("grad_clipped", False) for x in r]; fires(r, _run(), "I10")
    r = _rows(clip=None); [x.__setitem__("grad_clipped", True) for x in r]
    fires(r, _run(clip=None), "I10", il_form="matched", grad_clip=0.0)


def test_the_lever_changes_nothing_in_the_ported_release_constants():
    """L1 is two changes. DDv2's own constants are not among them."""
    c = D.DDV2
    assert (c.group_size, c.rollout_steps, c.label_span, c.trunc_t) == (4, 10, 20, 8)
    assert (c.explore_std_floor, c.likelihood_std_floor, c.gamma, c.eta) == (0.04, 0.10, 0.8, 1.0)
    assert (c.il_weight_with_positive, c.il_weight_no_positive) == (0.1, 1.0)
    assert (c.lr, c.weight_decay, c.adv_std_eps, c.veto_value) == (2e-4, 1e-4, 1e-4, -1.0)
    assert (c.clip_sample, c.clip_sample_range, c.bar_eps) == (True, 1.0, 1e-6)
