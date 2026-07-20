"""REF-C v1.2 learned re-scorer — contract tests.

The two that matter most:
  * ``test_instrumented_forward_matches_refc`` — ``refc_forward_fan`` must
    reproduce ``RefCModel.forward`` BIT-EXACTLY. It re-derives the decoder
    orchestration in order to expose the discarded per-anchor embeddings; if
    refc.py ever changes, this fails loud instead of silently caching a
    different decode than the one TanitEval scores.
  * ``test_identity_at_init`` — an untrained re-scorer must select EXACTLY what
    the frozen decoder selects, so any measured movement from REF-C's 0.458 is
    attributable to training and nothing else.
"""

import pytest
import torch

from tanitad.models.refc_rescorer import (N_GEOM, RefCRescorer, RescorerConfig,
                                          fan_ade_axes, fan_ade_from,
                                          geom_features, param_breakdown,
                                          q_width, rank_metrics,
                                          refc_forward_fan, rescorer_loss,
                                          select_q, topk_view)
from tanitad.refs.refc import RefCModel, refc_smoke_config


def test_select_q_and_width():
    d = {"q": torch.ones(2, 3, 4), "q0": torch.zeros(2, 3, 4)}
    assert torch.equal(select_q(d, "final"), d["q"])
    assert torch.equal(select_q(d, "t0"), d["q0"])
    assert select_q(d, "both").shape == (2, 3, 8)
    assert (q_width(512, "final"), q_width(512, "t0"),
            q_width(512, "both")) == (512, 512, 1024)
    with pytest.raises(ValueError):
        select_q(d, "nope")


def _smoke_model(seed=0):
    torch.manual_seed(seed)
    cfg = refc_smoke_config()
    cfg.graft_imagination = True          # exercise the gated H15 path too
    m = RefCModel(cfg).eval()
    return m, cfg


def _inputs(cfg, b=3):
    torch.manual_seed(1)
    frames = torch.rand(b, cfg.window, cfg.encoder.in_channels,
                        cfg.encoder.image_size, cfg.encoder.image_size)
    v0 = torch.rand(b) * 20.0
    return frames, v0


def test_instrumented_forward_matches_refc():
    m, cfg = _smoke_model()
    frames, v0 = _inputs(cfg)
    with torch.no_grad():
        ref = m(frames, nav_cmd=None, v0=v0,
                steps=cfg.decoder.diffusion_steps)
        got = refc_forward_fan(m, frames, nav_cmd=None, v0=v0)
    for k in ("anchor_logits", "anchor_traj", "traj"):
        assert torch.equal(ref[k], got[k]), f"{k} diverged from refc.py"
    assert torch.equal(ref["sel_idx"], got["sel_idx"])
    n = m.decoder.anchors.shape[0]
    assert got["q"].shape == (3, n, cfg.decoder.d)
    assert got["q0"].shape == (3, n, cfg.decoder.d)
    assert got["refined_conf"].shape == (3, n)
    assert got["cond"].shape == (3, cfg.decoder.d)
    # the two embeddings must be DIFFERENT objects: q0 is the t=0 pass whose
    # linear readout is the frozen selection score, q is the final pass
    assert not torch.equal(got["q"], got["q0"])
    assert torch.allclose(m.decoder.conf_head(got["q0"]).squeeze(-1)
                          + m.decoder.maneuver_to_anchor(
                              torch.log_softmax(got["maneuver_logits"], -1)),
                          got["anchor_logits"], atol=1e-5)


def test_instrumented_forward_classifier_mode():
    """steps=0 must also match (the 0-step floor path)."""
    m, cfg = _smoke_model()
    frames, v0 = _inputs(cfg)
    with torch.no_grad():
        ref = m(frames, nav_cmd=None, v0=v0, steps=0)
        got = refc_forward_fan(m, frames, nav_cmd=None, v0=v0, steps=0)
    assert torch.equal(ref["anchor_traj"], got["anchor_traj"])
    assert torch.equal(ref["traj"], got["traj"])
    # with no denoise pass the refined confidence IS the classifier confidence
    assert torch.equal(got["refined_conf"], got["anchor_logits"])
    assert torch.equal(got["q"], got["q0"])


def test_instrumented_forward_requires_eval_mode():
    m, cfg = _smoke_model()
    frames, v0 = _inputs(cfg)
    m.train()
    with pytest.raises(AssertionError):
        refc_forward_fan(m, frames, nav_cmd=None, v0=v0)


def _head_inputs(b=4, n=16, s=4, d_q=32, d_pool=24, d_cond=32):
    torch.manual_seed(2)
    return dict(q=torch.randn(b, n, d_q), base_logit=torch.randn(b, n),
                fan=torch.randn(b, n, s, 2) * 5.0,
                pooled=torch.randn(b, d_pool), cond=torch.randn(b, d_cond),
                v0=torch.rand(b) * 20.0)


def _head_cfg(**kw):
    base = dict(n_steps=4, d_q=32, d_pooled=24, d_cond=32, d=32, n_heads=4,
                layers=2, ff_mult=2, topk=0)
    base.update(kw)
    return RescorerConfig(**base)


def test_identity_at_init():
    """Zero-init residual head -> the frozen ranking, unchanged.

    With ``normalize_base`` the score is an increasing affine map of the frozen
    logits per window, so the PICK (and the whole ordering) is preserved even
    though the values are rescaled. Must hold at EVERY K, because the top-K
    gather is itself driven by the frozen confidence.
    """
    x = _head_inputs()
    for k in (0, 4, 8):
        for norm in (True, False):
            head = RefCRescorer(_head_cfg(topk=k, normalize_base=norm)).eval()
            with torch.no_grad():
                out = head(**x)
            assert torch.equal(out["sel_idx"], x["base_logit"].argmax(1)), \
                (k, norm)
    head = RefCRescorer(_head_cfg(normalize_base=False)).eval()
    with torch.no_grad():
        o = head(**x)
    assert torch.allclose(o["score"], x["base_logit"], atol=1e-6)
    assert torch.equal(o["score"].argsort(1), x["base_logit"].argsort(1))


def test_topk_view_selects_the_best_k_and_maps_back():
    base = torch.tensor([[0.0, 5.0, 1.0, 4.0]])
    other = torch.arange(4).float().reshape(1, 4, 1).expand(1, 4, 3)
    idx, (o_k,) = topk_view(base, 2, other)
    assert idx.tolist() == [[1, 3]]
    assert o_k[0, :, 0].tolist() == [1.0, 3.0]
    # k >= N (and k <= 0) must be the identity, so call sites stay K-agnostic
    idx_all, (o_all,) = topk_view(base, 0, other)
    assert idx_all.tolist() == [[0, 1, 2, 3]]
    assert torch.equal(o_all, other)


def test_topk_restricts_the_reachable_set():
    """With K=1 the head cannot move the pick at all — the top-K gather is a
    real constraint, not decoration."""
    x = _head_inputs()
    head = RefCRescorer(_head_cfg(topk=1))
    with torch.no_grad():
        head.score_head.weight.normal_(std=5.0)      # a LOUD residual
        out = head(**x)
    assert out["score"].shape[1] == 1
    assert torch.equal(out["sel_idx"], x["base_logit"].argmax(1))


def test_trained_residual_can_reorder_within_topk():
    head = RefCRescorer(_head_cfg(topk=4))
    x = _head_inputs()
    torch.manual_seed(11)
    with torch.no_grad():
        head.score_head.weight.normal_(std=5.0)
        head.score_head.bias.normal_(std=5.0)
        out = head(**x)
    moved = (out["sel_idx"] != x["base_logit"].argmax(1)).any()
    assert bool(moved), "a loud residual never re-ordered inside the top-K"
    # every pick must still come from the top-K set
    assert bool((out["topk_idx"] == out["sel_idx"][:, None]).any(1).all())


def test_geom_features_width_and_finiteness():
    x = _head_inputs()
    g = geom_features(x["fan"], x["v0"])
    assert g.shape == (4, 16, N_GEOM)
    assert torch.isfinite(g).all()


def test_soft_target_limits_to_hard():
    """tau -> 0 must reproduce the v1.5 hard-argmin CE (the sweep's endpoint)."""
    torch.manual_seed(3)
    out = {"score": torch.randn(5, 12)}
    ade = torch.rand(5, 12) + 0.1
    hard = rescorer_loss(out, ade, target="hard")
    soft = rescorer_loss(out, ade, target="soft", tau=1e-3)
    assert torch.allclose(hard, soft, atol=1e-4), (hard, soft)


def test_soft_target_stops_pushing_on_near_ties_hard_does_not():
    """THE degeneration mechanism, isolated.

    Two essentially identical plans (0.300 m vs 0.301 m). At the soft target's
    own optimum — ``score = -ADE/tau``, a FINITE gap of |dADE|/tau = 0.002 —
    the soft gradient is exactly zero: the objective is satisfied. The hard
    argmin CE at the SAME point still pushes with gradient ~0.5 toward an
    unbounded separation, which is how flagship v1.5's fan sharpened itself into
    ``frac_sel_2x_worse`` 0.099 -> 0.40.
    """
    ade = torch.tensor([[0.300, 0.301]])
    tau = 0.5
    s_soft = (-ade / tau).clone().requires_grad_(True)
    rescorer_loss({"score": s_soft}, ade, target="soft", tau=tau).backward()
    s_hard = (-ade / tau).clone().requires_grad_(True)
    rescorer_loss({"score": s_hard}, ade, target="hard").backward()
    assert float(s_soft.grad.abs().max()) < 1e-5
    assert float(s_hard.grad.abs().max()) > 0.4


def test_soft_target_still_pushes_on_gross_mis_rankings():
    """The same warm temperature must NOT go soft on a real error."""
    ade = torch.tensor([[0.30, 3.00]])
    score = torch.tensor([[0.0, 0.0]], requires_grad=True)
    rescorer_loss({"score": score}, ade, target="soft", tau=0.5).backward()
    assert float(score.grad.abs().max()) > 0.4


def test_pair_loss_zero_when_perfectly_ranked():
    ade = torch.tensor([[0.1, 0.5, 2.0]])
    score = torch.tensor([[10.0, 0.0, -10.0]])     # gaps exceed every margin
    assert float(rescorer_loss({"score": score}, ade, target="pair",
                               margin_scale=1.0)) == pytest.approx(0.0)


def test_pair_loss_weights_by_distance():
    """With tied scores the hinge IS the distance weight: a near-tie demands
    ~nothing, a gross gap dominates."""
    tied = torch.tensor([[0.0, 0.0]])
    tie = rescorer_loss({"score": tied}, torch.tensor([[0.30, 0.31]]),
                        target="pair", margin_scale=1.0)
    gross = rescorer_loss({"score": tied}, torch.tensor([[0.30, 3.00]]),
                          target="pair", margin_scale=1.0)
    assert float(gross) > 100 * float(tie)


def test_pair_loss_forgives_a_correctly_ordered_near_tie():
    """A small correct gap already satisfies a near-tie's margin, while the
    same gap leaves a gross mis-ordering fully unpaid."""
    score = torch.tensor([[0.05, 0.0]])            # correct order, tiny gap
    tie = rescorer_loss({"score": score}, torch.tensor([[0.30, 0.31]]),
                        target="pair", margin_scale=1.0)
    gross = rescorer_loss({"score": score}, torch.tensor([[0.30, 3.00]]),
                          target="pair", margin_scale=1.0)
    assert float(tie) == pytest.approx(0.0)
    assert float(gross) > 2.0


def test_regress_target_selects_argmin_predicted():
    head = RefCRescorer(_head_cfg()).eval()
    x = _head_inputs()
    with torch.no_grad():
        out = head(**x, target="regress")
    assert "ade_hat" in out
    assert torch.equal(out["score"].argmax(1), out["ade_hat"].argmin(1))
    loss = rescorer_loss(out, torch.rand(4, 16) + 0.1, target="regress")
    assert torch.isfinite(loss)


def test_loss_gathers_the_topk_slice_of_the_full_fan_ade():
    """The loss takes the FULL [B, N] ADE and must restrict it to the head's
    own top-K view — a silent mismatch here would train on the wrong rows."""
    head = RefCRescorer(_head_cfg(topk=4)).eval()
    x = _head_inputs()
    with torch.no_grad():
        out = head(**x)
    ade_full = torch.rand(4, 16) + 0.1
    ade_k = ade_full.gather(1, out["topk_idx"])
    assert torch.allclose(rescorer_loss(out, ade_full, target="soft", tau=0.3),
                          rescorer_loss(out, ade_k, target="soft", tau=0.3))


def test_rank_metrics_read():
    ade = torch.tensor([[0.2, 1.0, 3.0], [0.5, 0.1, 4.0]])
    base = torch.tensor([[0.0, 1.0, 2.0], [9.0, 0.0, 0.0]])   # picks 2 then 0
    out = {"sel_idx": torch.tensor([0, 1]),
           "score": torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]),
           "topk_idx": torch.tensor([[0, 1, 2], [0, 1, 2]])}
    m = rank_metrics(out, base, ade)
    assert m["sel_ade"] == pytest.approx((0.2 + 0.1) / 2)
    assert m["base_ade"] == pytest.approx((3.0 + 0.5) / 2)
    assert m["oracle_ade"] == pytest.approx((0.2 + 0.1) / 2)
    assert m["oracle_k_ade"] == pytest.approx((0.2 + 0.1) / 2)
    assert m["sel_gap"] == pytest.approx(0.0)
    assert m["rank_acc"] == pytest.approx(1.0)
    assert m["base_2x"] == pytest.approx(1.0)
    assert m["sel_2x"] == pytest.approx(0.0)
    # the whole gap was recovered: base 1.75 -> sel 0.15, oracle 0.15
    assert m["gap_recovered"] == pytest.approx(1.0)


def test_gap_recovered_is_zero_for_the_frozen_pick():
    """v1.0's headline result must be expressible: picking exactly what the
    frozen score picks recovers 0.0 % of the gap."""
    ade = torch.tensor([[0.2, 1.0, 3.0]])
    base = torch.tensor([[0.0, 1.0, 2.0]])
    out = {"sel_idx": torch.tensor([2]), "score": torch.tensor([[0., 1., 2.]]),
           "topk_idx": torch.tensor([[0, 1, 2]])}
    assert rank_metrics(out, base, ade)["gap_recovered"] == pytest.approx(0.0)


def test_fan_ade_matches_manual():
    fan = torch.zeros(1, 2, 4, 2)
    fan[0, 0, :, 0] = 1.0
    tgt = torch.zeros(1, 4, 2)
    ade = fan_ade_from(fan, tgt)
    assert ade[0, 0] == pytest.approx(1.0)
    assert ade[0, 1] == pytest.approx(0.0)


def test_fan_ade_axes_decomposition():
    """along = ego-x error, cross = ego-y error; the joint ADE is what the
    objective uses, the split is report-only."""
    fan = torch.zeros(1, 2, 4, 2)
    fan[0, 0, :, 0] = 3.0                     # pure along-track
    fan[0, 1, :, 1] = -4.0                    # pure cross-track
    tgt = torch.zeros(1, 4, 2)
    along, cross = fan_ade_axes(fan, tgt)
    assert along[0].tolist() == pytest.approx([3.0, 0.0])
    assert cross[0].tolist() == pytest.approx([0.0, 4.0])
    assert fan_ade_from(fan, tgt)[0].tolist() == pytest.approx([3.0, 4.0])


def test_gradient_reaches_the_head_through_the_score():
    """argmax has no gradient — the loss must move the score itself."""
    head = RefCRescorer(_head_cfg())
    x = _head_inputs()
    out = head(**x)
    loss = rescorer_loss(out, torch.rand(4, 16) + 0.1, target="soft", tau=0.2)
    loss.backward()
    g = head.score_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_ablations_change_param_count():
    full = param_breakdown(RefCRescorer(_head_cfg()))["total"]
    no_q = param_breakdown(RefCRescorer(_head_cfg(use_q=False)))["total"]
    no_geom = param_breakdown(RefCRescorer(_head_cfg(use_geom=False)))["total"]
    assert no_q < full and no_geom < full


def test_paired_delta_is_episode_clustered_and_signed():
    """The paired statistic must (a) sign improvement positive, (b) refuse
    significance when the effect is zero, (c) resample EPISODES not windows."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from refc_v12_eval import paired_delta

    eid = ["a"] * 50 + ["b"] * 50
    base = torch.full((100,), 0.50)
    better = torch.full((100,), 0.40)
    r = paired_delta(base, better, eid, n_boot=200)
    assert r["mean_delta_m"] == pytest.approx(0.10)
    assert r["significant"] and r["frac_windows_improved"] == 1.0
    assert r["n_episodes"] == 2

    same = paired_delta(base, base.clone(), eid, n_boot=200)
    assert same["mean_delta_m"] == pytest.approx(0.0)
    assert not same["significant"]
    assert same["frac_windows_unchanged"] == 1.0

    # a single episode carrying the whole effect must NOT read as significant
    # once episodes are the resampling unit
    torch.manual_seed(0)
    one_ep = torch.cat([base[:50] - 1.0, base[50:]])
    r2 = paired_delta(base, one_ep, ["a"] * 50 + ["b"] * 50, n_boot=500)
    assert not r2["significant"], r2


def test_end_to_end_frozen_decoder_plus_head():
    """The full v1.2 decode path on the smoke model: fan -> score -> pick."""
    m, cfg = _smoke_model()
    frames, v0 = _inputs(cfg, b=2)
    with torch.no_grad():
        o = refc_forward_fan(m, frames, nav_cmd=None, v0=v0)
    n = m.decoder.anchors.shape[0]
    head = RefCRescorer(RescorerConfig(
        n_steps=len(cfg.trajectory.horizons), d_q=cfg.decoder.d,
        d_pooled=m.encoder.feat_dim, d_cond=cfg.decoder.d, d=32, n_heads=4,
        layers=1, ff_mult=2, topk=8)).eval()
    with torch.no_grad():
        ho = head(o["q"], o["anchor_logits"], o["anchor_traj"], o["pooled"],
                  o["cond"], v0)
    assert ho["score"].shape == (2, 8)
    assert ho["topk_idx"].max() < n
    # untrained -> identical selection to the frozen decoder
    assert torch.equal(ho["sel_idx"], o["sel_idx"])
