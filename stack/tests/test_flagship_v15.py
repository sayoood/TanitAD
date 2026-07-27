"""flagship v1.5 tests (tanitad/models/flagship_v15.py + tanitad/lake/vtarget.py).

Pins the v1.5 spec — a REF-C anchored-diffusion head on the FROZEN v1 trunk:

(a) the head builds in the 25-55 M band with a param_breakdown that sums exactly,
    and every conditioning arm (a / ab / abc) is constructible;
(b) THE SCORING FIX. REF-C refines the whole anchor fan but selects with the
    t=0 classifier score, discarding the denoise passes' confidences — measured
    corpus-wide at oracle-in-fan 0.1640 m with 45.4 % of picks >2x worse than
    the fan's best. v1.5 keeps the last denoise
    pass's confidence and selects on it. Pinned: with steps>0 the refined logits
    DIFFER from the anchor logits; with steps=0 they are the same tensor (exact
    parent parity); the loss supervises the SCORE that argmax consumes, so the
    longitudinal gate actually receives a gradient (argmax has none);
(c) the goal seams obey the anti-shortcut discipline: goal-dropout is
    training-gated and per-sample, DROPPED is a distinct embedding row from the
    v2.1 ROUTE_UNKNOWN sentinel, and the H26 norm-parity telemetry is emitted;
(d) conditioning actually conditions: changing a goal token changes the output;
(e) the state re-expansion is layout-faithful to SpatialGridReadout;
(f) the imagination roll matches metric_dynamics' rollout mechanism and holds the
    v1 speed channel at the observed v0 (leakage-safe);
(g) VTARGET minting: vtarget_raw reproduces planner_p2's defect (VT_LOOK_LO
    unenforced, silent hold-speed fallback) and vtarget_v2 fixes it (smoothed
    track, enforced lookahead floor, explicit invalid mask).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tanitad.lake.vtarget import (VT_LOOK_HI, VT_MIN_STEPS, savgol,
                                  vtarget_raw, vtarget_v2)
from tanitad.models.flagship_v15 import (N_ROUTE_CLASSES, N_VTARGET_BANDS,
                                         ROUTE_DROPPED, ROUTE_UNKNOWN,
                                         SPEED_SCALE, VT_DROPPED,
                                         FlagshipV15Head, V15Config,
                                         build_probe_vocabulary,
                                         imagine_probes, param_breakdown,
                                         v15_ablation_config, v15_config,
                                         v15_losses)


def _small() -> V15Config:
    """A CPU-sized v1.5 (same structure, shrunk widths) for the shape tests."""
    from tanitad.refs.refc import DecoderConfig
    cfg = V15Config()
    cfg.state_dim = 64
    cfg.readout_grid = 4
    cfg.d_cell = 4
    cfg.window = 4
    cfg.n_anchors = 12
    cfg.d_token = 16
    cfg.d_meas = 8
    cfg.n_probes = 2
    cfg.decoder = DecoderConfig(d=16, n_heads=2, layers=2, ff_mult=2,
                                aux_hidden=16, diffusion_steps=2, noise_std=0.1)
    return cfg


def _batch(cfg: V15Config, b: int = 3):
    g = torch.Generator().manual_seed(0)
    return {
        "states": torch.randn(b, cfg.window, cfg.state_dim, generator=g),
        "v0": torch.rand(b, generator=g) * 20 + 3,
        "imagined": torch.randn(b, cfg.n_probes * len(cfg.imag_read),
                                cfg.state_dim, generator=g),
        "vt_band": torch.randint(0, N_VTARGET_BANDS, (b,), generator=g),
        "route": torch.randint(0, N_ROUTE_CLASSES, (b,), generator=g),
        "route_graded": torch.randn(b, generator=g),
        "traj_tgt": torch.randn(b, len(cfg.horizons), 2, generator=g),
    }


# --------------------------------------------------------------- (a) build --
def test_param_budget_and_breakdown():
    head = FlagshipV15Head(v15_config())
    pb = param_breakdown(head)
    assert pb["total"] == sum(p.numel() for p in head.parameters())
    assert 25e6 <= pb["total"] <= 55e6, pb["total"]
    # the brief's budget: the head, not the trunk
    for k in ("decoder", "state_proj", "imag_proj", "vtarget", "route"):
        assert pb[k] > 0, k


@pytest.mark.parametrize("arm", [(True, False, False), (True, True, False),
                                 (True, True, True)])
def test_every_ablation_arm_builds_and_runs(arm):
    states, imag, goal = arm
    cfg = _small()
    cfg.cond_states, cfg.cond_imagination = states, imag
    cfg.cond_vtarget = cfg.cond_route = goal
    head = FlagshipV15Head(cfg)
    b = _batch(cfg)
    out = head(b["states"], b["v0"],
               imagined=b["imagined"] if imag else None,
               vt_band=b["vt_band"] if goal else None,
               route=b["route"] if goal else None,
               route_graded=b["route_graded"] if goal else None,
               vt_speed=b["v0"] if goal else None)
    assert out["traj"].shape == (3, len(cfg.horizons), 2)
    assert out["anchor_traj"].shape == (3, cfg.n_anchors, len(cfg.horizons), 2)
    assert torch.isfinite(out["traj"]).all()


def test_needs_at_least_one_kv_source():
    cfg = _small()
    cfg.cond_states = cfg.cond_imagination = False
    with pytest.raises(ValueError, match="KV source"):
        cfg.__post_init__()


# ------------------------------------------------- (b) THE SCORING FIX ------
def test_refined_logits_differ_from_anchor_logits_when_denoising():
    """REF-C discards the denoise-pass confidences and ranks the REFINED fan by
    the UNREFINED anchor's score. v1.5 must not."""
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    assert not torch.equal(out["refined_logits"], out["anchor_logits"])


def test_steps0_reproduces_the_parent_exactly():
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"], steps=0)
    assert out["refined_logits"] is out["anchor_logits"]


def test_selection_uses_the_returned_score():
    """With the reachability clamp OFF, `sel_idx` is the bare argmax of the
    returned score — the pre-2026-07-27 contract, kept reachable by the flag."""
    cfg = _small()
    cfg.sel_reach_clamp = False
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    assert torch.equal(out["sel_idx"], out["sel_score"].argmax(dim=1))
    picked = out["anchor_traj"][torch.arange(3), out["sel_idx"]]
    assert torch.equal(out["traj"], picked)


# ------------------------------------------- (b2) the reachability clamp ----
# FIX 2. The vocabulary is blameless — `furthest_point_sample` returns
# `pool[chosen]`, so all 256 anchors are bitwise identical to real human
# windows. The UNBOUNDED OFFSET HEAD is what emits 171.5 km/h candidates (p99
# 159.6 against a val GT max of 132.4). MEASURED on 881 canonical val windows:
# the head's OWN clamp applied to the candidates removes 72.08 % of the fan, the
# ADE-oracle survives in 100 % of windows, and the pick moves in ZERO windows
# (paired Δ exactly 0.0000). The real-data proof is
# `taniteval/tests/test_reach_clamp_committed_windows.py`; these are the unit
# properties that hold on any fan.

def test_reachability_mask_is_the_band_it_claims_to_be():
    from tanitad.models.flagship_v15 import (candidate_mean_speed,
                                             reachability_mask)
    v0 = torch.tensor([10.0, 10.0])
    # terminal waypoints at 2 s implying 5 / 10 / 15 / 30 m/s mean speed
    fan = torch.zeros(2, 4, 4, 2)
    for j, v in enumerate((5.0, 10.0, 15.0, 30.0)):
        fan[:, j, -1, 0] = v * 2.0
    assert torch.allclose(candidate_mean_speed(fan, 2.0)[0],
                          torch.tensor([5.0, 10.0, 15.0, 30.0]))
    keep = reachability_mask(fan, v0, accel_max=2.5, horizon_s=2.0)
    # band = 10 +- 5 => 5 and 15 are ON the edge and admitted, 30 is not
    assert keep[0].tolist() == [True, True, True, False]

    # v0 = 1 m/s: the lower edge clamps at 0, never negative
    keep2 = reachability_mask(fan, torch.tensor([1.0, 1.0]),
                              accel_max=2.5, horizon_s=2.0)
    assert keep2[0].tolist() == [True, False, False, False]


def test_the_clamp_MOVES_the_pick_when_the_top_candidate_is_unflyable():
    """DESIGNED TO FAIL: the highest-scoring candidate implies 47 m/s.

    A clamp that cannot change any outcome is the C13 class — a guard that
    cannot fire. This is the input that makes it fire."""
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    n = cfg.n_anchors
    fan = torch.zeros(1, n, len(cfg.horizons), 2)
    fan[0, :, -1, 0] = 20.0                      # 10 m/s: inside the band
    fan[0, 0, -1, 0] = 94.0                      # 47 m/s: far outside it
    logits = torch.zeros(1, n)
    logits[0, 0] = 10.0                          # ...and it wins the argmax
    v0 = torch.tensor([10.0])

    head.cfg.sel_reach_clamp = False
    _, idx_off, _, _ = head.select(fan, logits, None, None, v0)
    assert int(idx_off) == 0, "without the clamp the unflyable plan is chosen"

    head.cfg.sel_reach_clamp = True
    _, idx_on, score, tele = head.select(fan, logits, None, None, v0)
    assert int(idx_on) != 0, "the clamp must delete the unflyable plan"
    assert tele["reach_frac_candidates_clipped"] == pytest.approx(1.0 / n)
    assert tele["reach_frac_windows_empty"] == 0.0


def test_the_clamp_never_touches_the_SUPERVISED_score():
    """The mask is applied to the argmax only. `sel_score` — the tensor
    `v15_losses` cross-entropies — must be bit-identical with the clamp on and
    off, or turning the default on would perturb a live training run."""
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    kw = dict(imagined=b["imagined"], vt_band=b["vt_band"], route=b["route"],
              route_graded=b["route_graded"], vt_speed=b["v0"])
    head.cfg.sel_reach_clamp = True
    on = head(b["states"], b["v0"], **kw)
    head.cfg.sel_reach_clamp = False
    off = head(b["states"], b["v0"], **kw)
    assert torch.equal(on["sel_score"], off["sel_score"])
    assert torch.isfinite(on["sel_score"]).all(), "no -inf may reach the loss"
    la = v15_losses(on, head.decoder.anchors, b["traj_tgt"])
    lb = v15_losses(off, head.decoder.anchors, b["traj_tgt"])
    for k in ("cls", "cls_refined", "traj"):
        assert torch.equal(la[k], lb[k]), k


def test_a_window_with_no_reachable_candidate_keeps_its_whole_fan():
    """DESIGNED TO FAIL: every candidate is out of band.

    An unreachable-everywhere window is a measurement failure, not a licence to
    return no plan. It must fall back, not crash and not emit -inf."""
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    n = cfg.n_anchors
    fan = torch.zeros(1, n, len(cfg.horizons), 2)
    fan[0, :, -1, 0] = 200.0                     # 100 m/s everywhere
    logits = torch.randn(1, n)
    _, idx, _, tele = head.select(fan, logits, None, None, torch.tensor([10.0]))
    assert int(idx) == int(logits.argmax(dim=1)), "fall back to the full fan"
    assert tele["reach_frac_windows_empty"] == 1.0
    assert torch.isfinite(fan[0, int(idx)]).all()


def test_the_clamp_reuses_the_heads_OWN_constant_and_is_not_tuned():
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    lo, hi = head.reach_band(torch.tensor([10.0, 1.0]))
    reach = cfg.sel_accel_max * cfg.horizons[-1] * 0.1
    assert reach == pytest.approx(5.0)
    assert lo.tolist() == [5.0, 0.0] and hi.tolist() == [15.0, 6.0]
    assert V15Config().sel_reach_clamp is True, "default ON"


def test_longitudinal_gate_receives_a_gradient():
    """argmax has no gradient, so the ranking loss must be applied to the SCORE.
    Without this the gate sits at its init forever, silently inert.

    goal_dropout is pinned to 0 here: the term is CORRECTLY masked off wherever
    the goal was withheld, so with dropout live a small batch can legitimately
    produce a zero gradient. That masking is asserted separately.
    """
    cfg = _small()
    cfg.goal_dropout = 0.0
    head = FlagshipV15Head(cfg).train()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    v15_losses(out, head.decoder.anchors, b["traj_tgt"])["loss"].backward()
    assert head.sel_gate.grad is not None
    assert torch.isfinite(head.sel_gate.grad)
    assert float(head.sel_gate.grad) != 0.0, "the gate must actually learn"


def test_oracle_gap_diagnostic_is_reported_and_consistent():
    """oracle-in-fan vs selected: separates 'cannot propose it' from
    'cannot rank it' (the fleet's standing metric request)."""
    cfg = _small()
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    L = v15_losses(out, head.decoder.anchors, b["traj_tgt"])
    for k in ("oracle_ade", "sel_gap", "rank_acc",
              "frac_sel_2x_worse_than_oracle"):
        assert k in L and torch.isfinite(L[k])
    assert float(L["oracle_ade"]) <= float(L["ade"]) + 1e-5
    assert float(L["sel_gap"]) >= -1e-5


# The per-anchor confidence head's BIAS is structurally unidentifiable: it is a
# single scalar added to every anchor logit, so it cancels in the softmax over
# anchors and its gradient is analytically zero (numerically it flickers between
# 0 and ~1e-9). Inherited from REF-C's AnchoredDiffusionDecoder, not a v1.5
# defect — excluded by name and asserted separately so the exclusion cannot
# quietly grow to hide a real dead parameter.
_UNIDENTIFIABLE = {"decoder.conf_head.bias"}


def test_loss_reaches_every_head_parameter():
    torch.manual_seed(0)            # deterministic ego-dropout mask (order-independent):
    cfg = _small()                  # else an all-dropped batch starves measurement.0.weight
    cfg.goal_dropout = 0.0          # deterministic: every goal seam is live
    head = FlagshipV15Head(cfg).train()
    b = _batch(cfg)
    out = head(b["states"], b["v0"], imagined=b["imagined"],
               vt_band=b["vt_band"], route=b["route"],
               route_graded=b["route_graded"], vt_speed=b["v0"])
    v15_losses(out, head.decoder.anchors, b["traj_tgt"])["loss"].backward()
    dead = [n for n, p in head.named_parameters()
            if p.requires_grad and (p.grad is None or not p.grad.abs().any())
            and n not in _UNIDENTIFIABLE]
    assert dead == [], f"parameters with no gradient: {dead}"


def test_anchor_softmax_makes_the_conf_bias_unidentifiable():
    """Pins WHY conf_head.bias is excluded above: a constant shift across all
    anchor logits leaves the anchor-classification CE exactly invariant."""
    logits = torch.randn(4, 12, requires_grad=False)
    tgt = torch.randint(0, 12, (4,))
    base = torch.nn.functional.cross_entropy(logits, tgt)
    shifted = torch.nn.functional.cross_entropy(logits + 3.7, tgt)
    assert torch.allclose(base, shifted, atol=1e-5)


# ------------------------------------- (c) anti-shortcut / goal discipline --
def test_goal_dropout_is_training_gated_and_uses_distinct_rows():
    cfg = _small()
    cfg.goal_dropout = 1.0
    head = FlagshipV15Head(cfg)
    b = _batch(cfg)
    head.train()
    m_tr, tele_tr, keep_tr = head.condition(b["v0"], b["vt_band"], b["route"],
                                            b["route_graded"])
    assert not keep_tr.any(), "goal_dropout=1.0 must drop every VTARGET"
    head.eval()
    _m_ev, _t, keep_ev = head.condition(b["v0"], b["vt_band"], b["route"],
                                        b["route_graded"])
    assert keep_ev.all(), "dropout must be OFF in eval"
    # UNKNOWN (the labeler could not judge) and DROPPED (we withheld it) are
    # different states and must not share an embedding row.
    assert ROUTE_UNKNOWN != ROUTE_DROPPED
    assert head.route_emb.num_embeddings == N_ROUTE_CLASSES + 1
    assert head.vtarget_emb.num_embeddings == N_VTARGET_BANDS + 1
    assert VT_DROPPED == N_VTARGET_BANDS
    for k in ("m_norm", "vt_norm", "vt_over_m", "rt_norm", "rt_over_m"):
        assert k in tele_tr, f"H26 norm-parity telemetry missing {k}"


def test_untrustworthy_vtarget_is_masked_out_of_the_selection_term():
    """A DROPPED band must not sneak back into the ranking."""
    cfg = _small()
    cfg.goal_dropout = 0.0
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    dropped = torch.full_like(b["vt_band"], VT_DROPPED)
    _m, _t, keep = head.condition(b["v0"], dropped, b["route"],
                                  b["route_graded"])
    assert not keep.any()


def test_ego_dropout_is_training_gated():
    cfg = _small()
    cfg.ego_dropout = 1.0
    cfg.goal_dropout = 0.0
    head = FlagshipV15Head(cfg)
    b = _batch(cfg)
    head.train()
    m_a, _, _ = head.condition(b["v0"], b["vt_band"], b["route"],
                               b["route_graded"])
    m_b, _, _ = head.condition(b["v0"] * 3.0, b["vt_band"], b["route"],
                               b["route_graded"])
    assert torch.allclose(m_a, m_b), "ego-dropout=1.0 must erase v0"


# ------------------------------------------- (d) conditioning conditions ----
def test_goal_tokens_change_the_output():
    """Causality: swap a goal token, the conditioned output must change."""
    cfg = _small()
    cfg.goal_dropout = 0.0
    torch.manual_seed(0)
    head = FlagshipV15Head(cfg).eval()
    # a trained gate is what makes the seam live; init 0.1 is small but non-zero
    with torch.no_grad():
        head.vt_gate.fill_(1.0)
        head.rt_gate.fill_(1.0)
    b = _batch(cfg)
    o1 = head(b["states"], b["v0"], imagined=b["imagined"],
              vt_band=torch.zeros_like(b["vt_band"]), route=b["route"],
              route_graded=b["route_graded"], vt_speed=b["v0"])
    o2 = head(b["states"], b["v0"], imagined=b["imagined"],
              vt_band=torch.full_like(b["vt_band"], N_VTARGET_BANDS - 1),
              route=b["route"], route_graded=b["route_graded"],
              vt_speed=b["v0"])
    assert not torch.allclose(o1["anchor_logits"], o2["anchor_logits"])


def test_imagination_tokens_change_the_output():
    cfg = _small()
    torch.manual_seed(0)
    head = FlagshipV15Head(cfg).eval()
    b = _batch(cfg)
    o1 = head(b["states"], b["v0"], imagined=b["imagined"],
              vt_band=b["vt_band"], route=b["route"],
              route_graded=b["route_graded"], vt_speed=b["v0"])
    o2 = head(b["states"], b["v0"], imagined=b["imagined"] * -3.0,
              vt_band=b["vt_band"], route=b["route"],
              route_graded=b["route_graded"], vt_speed=b["v0"])
    assert not torch.allclose(o1["anchor_logits"], o2["anchor_logits"])


# ----------------------------------------- (e) state re-expansion layout ----
def test_state_reexpansion_is_layout_faithful():
    """SpatialGridReadout emits [B, G*G, d_r] flattened; the head must
    re-expand on exactly that layout or the spatial grid is scrambled."""
    from tanitad.models.readout import SpatialGridReadout
    ro = SpatialGridReadout(n_tokens=256, d_model=32, grid=4, d_readout=4)
    tokens = torch.randn(2, 256, 32)
    flat = ro(tokens)                                   # [2, 64]
    cells = flat.reshape(2, 4 * 4, 4)                   # the head's re-expand
    ref = ro.proj(ro.pool(tokens.transpose(1, 2).reshape(2, 32, 16, 16))
                  .flatten(2).transpose(1, 2))          # [2, 16, 4]
    assert torch.allclose(cells, ref)


def test_config_rejects_an_inconsistent_readout_geometry():
    cfg = V15Config()
    cfg.d_cell = 127
    with pytest.raises(ValueError, match="scramble"):
        cfg.__post_init__()


# --------------------------------------------- (f) imagination mechanism ----
class _StubPredictor:
    """Records the actions it is rolled under; returns a deterministic latent."""

    def __init__(self):
        self.seen = []

    def __call__(self, states, actions):
        self.seen.append(actions[:, -1].clone())
        return {1: states[:, -1] + 1.0}


def test_imagine_probes_shape_and_speed_channel_is_held():
    b, w, s, m, k = 2, 4, 8, 3, 6
    pred = _StubPredictor()
    states = torch.zeros(b, w, s)
    actions = torch.zeros(b, w, 3)
    probes = torch.randn(m, k, 2)
    v0n = torch.tensor([1.5, 2.5])
    read = (2, 4, 6)
    out = imagine_probes(pred, states, actions, probes, read, v0n)
    assert out.shape == (b, m * len(read), s)
    # every appended action carries the OBSERVED v0 in channel 3 — never a
    # future speed (leakage-safe; the SPEED_SCALE contract with the v1 trunk)
    for seen in pred.seen[1:]:
        assert torch.allclose(seen[:, 2].reshape(b, m)[:, 0], v0n)
    assert SPEED_SCALE == 10.0


# ------------------------------- FIX 1: the missing candidate axis, made LOUD
# MEASURED 2026-07-27 (`…/2026-07-27-percandidate-labels/raw/
# t4_imagination_conditioning.json`): `imagine_probes` returns 32 tokens at
# n_anchors 64 AND 256, identical for every candidate. The imagination path
# structurally cannot rank candidates, so E-V5-1's imagination-scoring negative
# is OVER-DETERMINED. A function that silently returns the same thing for every
# candidate is the same class as a guard that cannot fail.

def test_imagination_token_count_is_INVARIANT_to_n_anchors():
    """The measured fact, pinned as an executable invariant."""
    from tanitad.models.flagship_v15 import (IMAGINATION_HAS_CANDIDATE_AXIS,
                                             imagination_token_count)
    counts = set()
    for n_anchors in (64, 256):
        cfg = V15Config()
        cfg.n_anchors = n_anchors
        counts.add(imagination_token_count(cfg))
    assert counts == {32}, counts
    assert IMAGINATION_HAS_CANDIDATE_AXIS is False


def test_imagine_probes_output_is_IDENTICAL_for_every_candidate():
    """The tokens serve the whole fan. Shown, not asserted in prose."""
    from tanitad.models.flagship_v15 import (NoCandidateAxis,
                                             assert_candidate_axis)
    b, w, s, m, k = 3, 4, 8, 8, 8
    out = imagine_probes(_StubPredictor(), torch.zeros(b, w, s),
                         torch.zeros(b, w, 3), torch.randn(m, k, 2),
                         (2, 4, 6, 8), torch.tensor([1.5, 2.5, 3.5]))
    assert out.shape == (b, 32, s)      # 8 probes x 4 read steps, as shipped
    # broadcast it across a 256-candidate fan the way the decoder does, and the
    # guard must refuse it: constant along the candidate axis.
    broadcast = out[:, None].expand(b, 256, 32, s)
    with pytest.raises(NoCandidateAxis, match="CONSTANT along it"):
        assert_candidate_axis(broadcast, 256, name="imagination tokens")


def test_asking_imagine_probes_for_a_candidate_axis_RAISES():
    """DESIGNED TO FAIL: the request E-V5-1 implicitly made."""
    from tanitad.models.flagship_v15 import NoCandidateAxis
    with pytest.raises(NoCandidateAxis) as e:
        imagine_probes(_StubPredictor(), torch.zeros(2, 4, 8),
                       torch.zeros(2, 4, 3), torch.randn(8, 8, 2), (2, 4, 6, 8),
                       torch.tensor([1.0, 2.0]), n_candidates=256)
    assert "imagine_candidates" in str(e.value)
    assert "invariant to n_anchors" in str(e.value)
    # ...and the ordinary call still works, or the guard is just a wall
    assert imagine_probes(_StubPredictor(), torch.zeros(2, 4, 8),
                          torch.zeros(2, 4, 3), torch.randn(8, 8, 2),
                          (2, 4, 6, 8), torch.tensor([1.0, 2.0])
                          ).shape == (2, 32, 8)


def test_assert_candidate_axis_fires_on_BOTH_degenerate_shapes():
    from tanitad.models.flagship_v15 import (NoCandidateAxis,
                                             assert_candidate_axis)
    with pytest.raises(NoCandidateAxis, match="no candidate axis"):
        assert_candidate_axis(torch.randn(2, 32, 8), 256, name="x")
    with pytest.raises(NoCandidateAxis, match="CONSTANT"):
        assert_candidate_axis(torch.zeros(2, 4, 8), 4, name="x")
    ok = torch.randn(2, 4, 8)
    assert assert_candidate_axis(ok, 4, name="x") is ok


def test_imagine_candidates_HAS_a_real_candidate_axis():
    """The repair: one rollout per candidate, and the output actually varies."""
    from tanitad.models.flagship_v15 import (assert_candidate_axis,
                                             imagine_candidates)
    b, w, s, n, k = 2, 4, 8, 5, 6
    read = (2, 4, 6)

    class _P:            # latent depends on the ACTION, so candidates differ
        def __call__(self, states, actions):
            return {1: states[:, -1] + actions[:, -1, :1]}

    out = imagine_candidates(_P(), torch.zeros(b, w, s), torch.zeros(b, w, 3),
                             torch.randn(b, n, k, 2), read,
                             torch.tensor([1.0, 2.0]))
    assert out.shape == (b, n, len(read), s)
    assert_candidate_axis(out, n, name="imagine_candidates")     # must NOT raise


def test_imagine_candidates_rolls_only_the_REACHABLE_candidates():
    """FIX 2 paying for FIX 1: 72.08 % of the fan is unflyable, so a masked
    roll costs 3.58x less. Masked rows must come back zero, not stale."""
    from tanitad.models.flagship_v15 import imagine_candidates
    b, w, s, n, k = 2, 4, 8, 6, 6
    read = (2, 4)
    calls = []

    class _P:
        def __call__(self, states, actions):
            calls.append(states.shape[0])
            return {1: states[:, -1] + 1.0}

    keep = torch.zeros(b, n, dtype=torch.bool)
    keep[:, :2] = True                                  # 4 of 12 survive
    out = imagine_candidates(_P(), torch.zeros(b, w, s), torch.zeros(b, w, 3),
                             torch.randn(b, n, k, 2), read,
                             torch.tensor([1.0, 2.0]), keep=keep)
    assert out.shape == (b, n, len(read), s)
    assert calls[0] == 4, "only the surviving candidates may be rolled"
    assert torch.count_nonzero(out[~keep]) == 0, "masked rows must stay zero"
    assert torch.count_nonzero(out[keep]) > 0

    # an empty mask must not crash and must not fabricate latents
    out0 = imagine_candidates(_P(), torch.zeros(b, w, s), torch.zeros(b, w, 3),
                              torch.randn(b, n, k, 2), read,
                              torch.tensor([1.0, 2.0]),
                              keep=torch.zeros(b, n, dtype=torch.bool))
    assert torch.count_nonzero(out0) == 0


def test_probe_vocabulary_is_fps_and_deterministic():
    pool = torch.randn(200, 6, 2)
    p1 = build_probe_vocabulary(pool, 5, seed=0)
    p2 = build_probe_vocabulary(pool, 5, seed=0)
    assert p1.shape == (5, 6, 2)
    assert torch.equal(p1, p2)


# -------------------------------------------------- (g) VTARGET minting -----
def _track(t_len=199, v=20.0):
    return np.full(t_len, v, dtype=np.float64)


def test_vtarget_raw_reproduces_the_unenforced_lookahead_defect():
    """planner_p2 computes fut = v[L+1 : L+VT_LOOK_HI] and only checks
    >= VT_MIN_STEPS. VT_LOOK_LO (10 s) is defined and never used, so a window
    3 s from the clip end is still 'valid'."""
    v = _track()
    near_end = np.array([len(v) - 1 - VT_MIN_STEPS - 1])
    vt, valid = vtarget_raw(v, near_end)
    assert bool(valid[0]), "the raw mint accepts a ~3 s lookahead as valid"
    assert VT_LOOK_HI == 200 and VT_MIN_STEPS == 30


def test_vtarget_v2_enforces_a_lookahead_floor_and_marks_invalid():
    v = _track()
    near_end = np.array([len(v) - 1 - VT_MIN_STEPS - 1])
    _vt, valid, look, _vs = vtarget_v2(v, near_end, min_lookahead=50)
    assert not bool(valid[0]), "v2 must refuse a lookahead below the floor"
    assert int(look[0]) < 50
    early = np.array([10])
    _vt2, valid2, look2, _ = vtarget_v2(v, early, min_lookahead=50)
    assert bool(valid2[0]) and int(look2[0]) >= 50


def test_vtarget_v2_smooths_before_the_free_flow_gate():
    """The gate keeps a step only if the step INTO it decelerated less than
    1.5 m/s^2. Differentiating a jittery track at dt=0.1 amplifies jitter 10x,
    so the gate must see the SMOOTHED track."""
    rng = np.random.default_rng(0)
    v = 30.0 + rng.normal(0, 0.5, 199)
    vs = savgol(v)
    assert vs.std() < v.std()
    assert abs(vs.mean() - v.mean()) < 0.1          # zero-phase: no drift
    raw_hard = (np.diff(v) / 0.1 < -1.5).mean()
    sm_hard = (np.diff(vs) / 0.1 < -1.5).mean()
    assert sm_hard < raw_hard
    _vt, valid, _l, _ = vtarget_v2(v, np.array([10]), min_lookahead=50)
    assert bool(valid[0])


def test_savgol_is_length_preserving_and_exact_on_a_ramp_in_the_interior():
    """A real accel ramp must survive the smoother, or every speed label is
    biased. Order-2 SavGol reproduces a line EXACTLY in the interior; the
    even-mirror edge padding is not linear, so the first/last `half` samples
    carry a bounded artefact. Pinned rather than hidden — and harmless in the
    mint: the leading samples sit inside the window-8 warm-up (labels start at
    L=7) and the trailing ones only occur in windows whose lookahead is already
    below the v2 floor, i.e. already invalid.
    """
    v = np.linspace(0.0, 10.0, 199)
    out = savgol(v)
    assert out.shape == v.shape
    half = 11 // 2
    assert np.abs(out - v)[half:-half].max() < 1e-9, "interior must be exact"
    step = v[1] - v[0]
    assert np.abs(out - v).max() < step, "edge artefact under one sample step"
