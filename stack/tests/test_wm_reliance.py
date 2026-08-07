"""The instrument that decides whether the decoder uses the world model.

⛔ THE TESTS THAT MATTER ARE THE TWO SYNTHETIC EXTREMES. A metric like this is only
trustworthy if it returns ~0 for a head that provably ignores the latents and ~1 for one
that provably depends on them. Everything else it reports is uninterpretable without
those two anchors, so they are built from heads whose behaviour is known by construction.
"""
import torch
from torch import nn

from tanitad.models.wm_reliance import wm_reliance, wm_reliance_gate


def _setup(b=32, w=4, s=8, k=20, dt=0.1, seed=0):
    torch.manual_seed(seed)
    states = torch.randn(b, w, s, dtype=torch.float64)
    actions = torch.zeros(b, w, 3, dtype=torch.float64)
    v0 = torch.rand(b, dtype=torch.float64) * 12.0 + 3.0
    # a GT that CURVES — so constant velocity is genuinely wrong and there is something
    # for the head to add. A straight GT would make CV optimal and the ratio undefined.
    curv = torch.randn(b, dtype=torch.float64) * 0.05
    yaw = torch.cumsum(v0[:, None] * curv[:, None] * dt * torch.ones(b, k), 1)
    step = torch.stack([v0[:, None] * dt * torch.ones(b, k),
                        torch.zeros(b, k), yaw.diff(dim=1, prepend=torch.zeros(b, 1))], -1)
    from tanitad.models.metric_dynamics import accumulate_se2
    gt = accumulate_se2(step)
    return states, actions, gt, v0, curv, k, dt


def test_a_head_that_ignores_latents_scores_near_zero_reliance():
    """⭐ ANCHOR 1. A head whose output depends ONLY on v0 — the pure driving-dynamics
    predictor Sayed is worried about. It can beat constant velocity (it is fitted to do
    so) and must still be reported as ~0 reliance."""
    states, actions, gt, v0, curv, k, dt = _setup()
    from tanitad.models.metric_dynamics import accumulate_se2

    def rollout(st, ac, fa, v):
        # a fixed curvature per sample derived from v0 alone: no latents anywhere
        kap = 0.0 * v                      # deliberately mediocre but latent-free
        yawr = v * kap
        step = torch.stack([v[:, None] * dt * torch.ones_like(gt[:, :, 0]),
                            torch.zeros_like(gt[:, :, 0]),
                            yawr[:, None] * dt * torch.ones_like(gt[:, :, 0])], -1)
        return accumulate_se2(step)

    rel = wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    for arm in ("real", "mean", "shuffled", "frozen"):
        assert rel[arm]["ade_m"] == rel["real"]["ade_m"], arm     # latents do nothing
    # identical to CV here, so the ratio is undefined and must be reported as such
    assert rel["verdict"]["wm_reliance"] is None
    assert wm_reliance_gate(rel)["status"] == "UNAVAILABLE"


def test_a_head_that_depends_on_latents_scores_high_reliance():
    """⭐ ANCHOR 2. A head that reads the ANSWER out of the latents. Reliance must be
    ~1.0, or the instrument cannot detect the good case either."""
    states, actions, gt, v0, curv, k, dt = _setup()
    from tanitad.models.metric_dynamics import accumulate_se2
    # plant the true curvature in latent channel 0 of the last window step
    states = states.clone()
    states[:, -1, 0] = curv * 100.0

    def rollout(st, ac, fa, v):
        kap = st[:, -1, 0] / 100.0
        yawr = v * kap
        ones = torch.ones_like(gt[:, :, 0])
        step = torch.stack([v[:, None] * dt * ones, torch.zeros_like(ones),
                            yawr[:, None] * dt * ones], -1)
        return accumulate_se2(step)

    rel = wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    assert rel["real"]["ade_m"] < 1e-9                       # exact by construction
    assert rel["mean"]["ade_m"] > rel["real"]["ade_m"]
    assert rel["verdict"]["wm_reliance"] > 0.95, rel["verdict"]
    assert wm_reliance_gate(rel)["status"] == "PASS"


def test_a_half_and_half_head_lands_in_between():
    """The instrument must be MONOTONE, not just bimodal — otherwise it cannot rank two
    real candidate heads.

    ⛔ THE MIX MUST BE AGAINST A SAMPLE-VARYING LATENT-FREE SIGNAL, not a constant. A
    first version blended the latent channel with `curv.mean()`, which is invariant under
    the `mean` ablation — so BOTH mixtures produced an identical `mean` arm and the ratio
    could not move. The shortcut a real head would take is v0-shaped, not constant, so
    the surrogate has to be too."""
    states, actions, gt, v0, curv, k, dt = _setup()
    from tanitad.models.metric_dynamics import accumulate_se2
    states = states.clone()
    states[:, -1, 0] = curv * 100.0
    # a latent-free but SAMPLE-VARYING surrogate: exactly the shortcut we fear
    proxy = 0.05 * (v0 - v0.mean()) / v0.std()

    def make(alpha):
        def rollout(st, ac, fa, v):
            kap = alpha * st[:, -1, 0] / 100.0 + (1 - alpha) * proxy
            yawr = v * kap
            ones = torch.ones_like(gt[:, :, 0])
            step = torch.stack([v[:, None] * dt * ones, torch.zeros_like(ones),
                                yawr[:, None] * dt * ones], -1)
            return accumulate_se2(step)
        return rollout

    r_lo = wm_reliance(make(0.2), states, actions, None, gt, v0, k=k, dt=dt)
    r_hi = wm_reliance(make(0.95), states, actions, None, gt, v0, k=k, dt=dt)
    assert r_lo["verdict"]["wm_reliance"] < r_hi["verdict"]["wm_reliance"], \
        (r_lo["verdict"], r_hi["verdict"])


def test_reliance_above_one_is_reported_not_clipped():
    """⚠️ `wm_reliance > 1` happens when the latent-free arm is WORSE than constant
    velocity — the shortcut pathway actively hurts without the latents to steer it. That
    is STRONGER evidence of reliance, not a bug, and clipping it to 1.0 would erase the
    distinction between 'fully reliant' and 'cannot function at all without the WM'."""
    states, actions, gt, v0, curv, k, dt = _setup()
    from tanitad.models.metric_dynamics import accumulate_se2
    states = states.clone()
    states[:, -1, 0] = curv * 100.0

    def rollout(st, ac, fa, v):
        # exact with real latents; with the batch-mean latent it OVERSHOOTS every turn
        kap = st[:, -1, 0] / 100.0 * 3.0 - 2.0 * curv
        yawr = v * kap
        ones = torch.ones_like(gt[:, :, 0])
        return accumulate_se2(torch.stack(
            [v[:, None] * dt * ones, torch.zeros_like(ones), yawr[:, None] * dt * ones], -1))

    rel = wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    assert rel["mean"]["ade_m"] > rel["cv"]["ade_m"], rel
    assert rel["verdict"]["wm_reliance"] > 1.0, rel["verdict"]
    assert wm_reliance_gate(rel)["status"] == "PASS"


def test_shuffled_preserves_the_marginal_exactly():
    """⛔ `shuffled` must destroy the PAIRING while keeping the marginal EXACTLY — if it
    changed the distribution too, a drop could be blamed on distribution shift instead
    of on lost content, and the arm would prove nothing."""
    states, actions, gt, v0, curv, k, dt = _setup()
    seen = {}

    def rollout(st, ac, fa, v):
        seen[len(seen)] = st
        return torch.zeros_like(gt)

    wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    shuffled = seen[2]
    assert torch.allclose(shuffled.sort(dim=0).values, states.sort(dim=0).values)
    assert not torch.allclose(shuffled, states)


def test_mean_arm_removes_per_window_content_but_keeps_the_scale():
    states, actions, gt, v0, curv, k, dt = _setup()
    seen = []

    def rollout(st, ac, fa, v):
        seen.append(st)
        return torch.zeros_like(gt)

    wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    mean_arm = seen[1]
    assert torch.allclose(mean_arm[0], mean_arm[1])          # identical across the batch
    assert torch.allclose(mean_arm.mean(0), states.mean(0))  # same first moment


def test_gate_reports_unavailable_rather_than_a_fake_number():
    """⛔ A head that does not beat CV has NOTHING to attribute; the ratio is undefined.
    Reporting 0.0 would read as 'bypassed' when the truth is 'not computable'."""
    states, actions, gt, v0, curv, k, dt = _setup()

    def rollout(st, ac, fa, v):
        from tanitad.models.metric_dynamics import accumulate_se2
        ones = torch.ones_like(gt[:, :, 0])
        return accumulate_se2(torch.stack(
            [v[:, None] * dt * ones, torch.zeros_like(ones), torch.zeros_like(ones)], -1))

    rel = wm_reliance(rollout, states, actions, None, gt, v0, k=k, dt=dt)
    assert rel["verdict"]["wm_reliance"] is None
    g = wm_reliance_gate(rel)
    assert g["status"] == "UNAVAILABLE" and "nothing to attribute" in g["reason"]


def test_cv_arm_matches_a_zero_initialised_unicycle_head():
    """The floor must be EXACTLY what the untrained head emits, or 'gain over CV' is
    measuring something the head never actually starts from."""
    from tanitad.models.metric_dynamics import (UnicycleStepReadout, accumulate_se2,
                                                rollout_decode_unicycle)
    states, actions, gt, v0, curv, k, dt = _setup(b=4, s=8)

    class P(nn.Module):
        def forward(self, ws, wa):
            return None, ws[:, -1]

    uni = UnicycleStepReadout(8, hidden=16).double()
    wp_head, _ = rollout_decode_unicycle(P(), states, actions, None, uni, k, v0, dt)
    from tanitad.models.wm_reliance import _cv
    assert torch.allclose(wp_head, _cv(v0, k, dt), atol=1e-12)
