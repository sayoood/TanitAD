"""The three remaining diagram cells — diffusion proposals · MPC top-K
refinement · the context-brain fallback trigger — gated DEFAULT-OFF, built
per the measurement constraints that bind them.

⛔ WHAT THIS FILE PROTECTS AND PROVES:

1. **Byte-identity of the default build** (per tensor, ``torch.equal``,
   against a CONTENT-anchored pre-change revision of ``v6.py`` — C75, never
   HEAD) plus the RNG stream. The live resumes are **87,893,449 / 405**
   (default) and **336,542,025 / 573** (config E) and a broken strict resume
   kills them.
2. **The measurement constraints as code**: the MPC path is INERT without the
   ``"goal"`` selector (the pure roll-consistency argmin is REFUTED — winner's
   curse, +5.9787 m, error-rank RISING with N; W7-PROG requires a
   goal-conditioned, candidate-independent component); the post-refinement
   re-score is GOAL DISTANCE ONLY (E-S1-0: off-distribution rescoring
   2.8–2.95x worse); the fallback uses roll-cost VARIANCE as an uncertainty
   signal (P7's validated use, rho 0.7164) and is PERMUTATION-INVARIANT over
   candidates, so it structurally CANNOT select.
3. **Temporally-correlated noise is MEASURED, not asserted**: the OU draw's
   lag-1 autocorrelation is computed on the actual sample and travels with
   every fan.
4. **Every guard is proven able to FAIL** — each refusal in the config, the
   calibration loader, and the trainer preflight is exercised in its firing
   direction.
5. **X3 = {0, 0, 0}** with each cell ON alone and all three combined, and the
   planner-surface totality probe extended to the diffusion arm.

Every literal here is MEASURED (2026-08-16, this box, torch CPU at seed 0) —
recompute on drift, never inherit.
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.models.v6 import (  # noqa: E402
    P7_GATE_RHO, DiffusionProposalGenerator, FallbackTrigger, MpcRefiner,
    V6Config, V6Stack)

#: the counts the DEFAULT build must not move (the live S-W resume's geometry
#: class; same literals as tests/test_v6_gstr_port.py).
HEAD_FULL_PARAMS, HEAD_FULL_KEYS = 87_893_449, 405
CONFIG_E_PARAMS, CONFIG_E_KEYS = 336_542_025, 573
#: MEASURED 2026-08-16 at the PRODUCTION geometry (d_plan_feat 256,
#: d_goal_embed 128, diffusion_hidden 256): the diffusion arm's cost.
DIFFUSION_DELTA_PROD = 437_954
#: the fallback trigger's state_dict keys (buffers only, 0 params).
FALLBACK_KEYS = 8


def tiny_cfg(**kw) -> V6Config:
    """test_v6_staged.py's tiny geometry — same wiring, ~1000x fewer params;
    plan_steps stays 60 because §4b is part of what is under test."""
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8, diffusion_hidden=32)
    base.update(kw)
    return V6Config(**base)


def build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


GOOD_CALIB = {"spearman_rho": 0.7164, "rho_ci": [0.5847, 0.7696],
              "slope": 1.0, "intercept": 0.0, "threshold": 0.5,
              "w_spread": 1.0, "w_rollvar": 1.0,
              "provenance": "SYNTHETIC test calibration; the rho/CI are P7's "
                            "repaired-trunk reference values"}


# =========================================================================== #
# 1. ⛔ BYTE-IDENTITY OF THE DEFAULT BUILD — the live resumes are untouchable
# =========================================================================== #

def test_all_three_default_off_and_absent():
    c = V6Config()
    assert c.proposals == "query"
    assert c.mpc_refine is False
    assert c.fallback_trigger is False
    s = build(tiny_cfg())
    assert s.prop_diffusion is None and s.mpc is None and s.fallback is None
    sd = s.state_dict()
    assert not any(k.startswith(("prop_diffusion.", "fallback.")) for k in sd)


#: The marker that separates the pre-change architecture from this one. Any
#: revision of ``v6.py`` WITHOUT it is one the live checkpoints could have
#: been built from.
_PRE_CHANGE_MARKER = "prop_diffusion"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_change_module():
    """``tanitad.models.v6`` as it was BEFORE these cells existed — CONTENT-
    anchored, never HEAD (C75): walk ``v6.py``'s git history for the NEWEST
    revision that does not carry the marker. Stable however many commits land
    afterwards; None when git cannot answer (caller skips — a skipped test is
    honest, a self-comparison dressed as a real one is not)."""
    root = _STACK.parent
    try:
        log = subprocess.run(["git", "log", "--format=%H", "--", _V6_REL],
                             cwd=root, capture_output=True, timeout=180)
        if log.returncode != 0:
            return None
        for sha in log.stdout.decode().split():
            r = subprocess.run(["git", "show", f"{sha}:{_V6_REL}"], cwd=root,
                               capture_output=True, timeout=120)
            if r.returncode != 0 or not r.stdout:
                continue
            if _PRE_CHANGE_MARKER.encode() in r.stdout:
                continue
            src = r.stdout
            break
        else:
            return None
    except Exception:
        return None
    sp = str(_STACK / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_dmf.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_dmf", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_dmf"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_is_byte_identical_to_the_PRE_CHANGE_architecture():
    """⛔ THE ONE THAT PROTECTS THE LIVE RUN. Per tensor, ``torch.equal`` —
    never a digest of a ``torch.save`` container (C72) — plus the RNG STREAM,
    because a default path that consumed one extra draw would leave the
    state_dict identical and still desynchronise everything initialised after
    the model."""
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")
    kw = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=3, aux_hidden=16, sigreg_slices=8)
    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config(**kw))
    rng_old = torch.random.get_rng_state()
    torch.manual_seed(0)
    new = build(tiny_cfg())
    rng_new = torch.random.get_rng_state()

    so, sn = old.state_dict(), new.state_dict()
    assert sorted(so) == sorted(sn), "key sets differ"
    diff = [k for k in so if not torch.equal(so[k], sn[k])]
    assert not diff, f"tensors differ at: {diff[:8]}"
    assert torch.equal(rng_old, rng_new), \
        "the default path consumed a different number of RNG draws"


@pytest.mark.slow
def test_default_FULL_config_counts_are_the_live_resume_counts():
    f = build(V6Config())
    assert (_n(f), len(f.state_dict())) == (HEAD_FULL_PARAMS, HEAD_FULL_KEYS)


@pytest.mark.slow
def test_config_E_default_build_is_unchanged():
    e = build(V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=256, image_width=640,
                              patch_size=16, d_model=768, depth=12,
                              n_heads=12),
        readout=ReadoutConfig(grid=4, d_readout=128),
        predictor=PredictorConfig(d_model=1024, depth=12, n_heads=16,
                                  window=6, horizons=(1, 2, 4), action_dim=3,
                                  residual=True, modern=True),
        d_tac=768, d_str=512, d_goal_embed=128, adapter_hidden=512,
        f_hidden_tac=1024, f_hidden_str=1024, f_blocks=6,
        vit5_encoder=True, n_registers=4, param_budget=350_000_000))
    assert (_n(e), len(e.state_dict())) == (CONFIG_E_PARAMS, CONFIG_E_KEYS)


# =========================================================================== #
# 2. DIFFUSION PROPOSALS (F-15)
# =========================================================================== #

def test_config_refuses_an_unknown_proposals_arm():
    with pytest.raises(ValueError, match="query|diffusion"):
        tiny_cfg(proposals="cem")


@pytest.mark.parametrize("kw", [dict(diffusion_steps=0),
                                dict(diffusion_noise_rho=1.0),
                                dict(diffusion_noise_rho=-0.1),
                                dict(diffusion_sigma_a=0.0),
                                dict(diffusion_sigma_k=-1.0),
                                dict(diffusion_hidden=0)])
def test_config_refuses_bad_diffusion_knobs(kw):
    with pytest.raises(ValueError):
        tiny_cfg(proposals="diffusion", **kw)


def test_diffusion_builds_only_under_flag_and_perturbs_no_shared_tensor():
    """The arm's cost is NEW parameters only — every pre-existing tensor is
    bit-identical, which is what makes 'query' vs 'diffusion' an ARM
    comparison rather than two different models."""
    d = build(tiny_cfg())
    s = build(tiny_cfg(proposals="diffusion"))
    assert s.prop_diffusion is not None
    delta = _n(s) - _n(d)
    assert delta > 0
    sd_d, sd_s = d.state_dict(), s.state_dict()
    new = {k for k in sd_s if k not in sd_d}
    assert new and all(k.startswith("prop_diffusion.") for k in new)
    for k in sd_d:
        assert torch.equal(sd_d[k], sd_s[k]), f"shared tensor moved: {k}"


@pytest.mark.slow
def test_diffusion_param_delta_at_production_geometry_is_the_measured_one():
    d = build(V6Config())
    s = build(V6Config(proposals="diffusion"))
    assert _n(s) - _n(d) == DIFFUSION_DELTA_PROD


def test_ou_noise_lag1_autocorr_is_MEASURED_on_the_draw():
    """The binding constraint: the autocorrelation is measured on the actual
    sample, and the measurement can DETECT ABSENCE (rho=0 reads ~0) — a
    measurement that cannot fail is an assertion wearing a measurement's name.

    ⚠️ THE BAND IS CENTRED ON THE ESTIMATOR'S KNOWN SMALL-SAMPLE BIAS, NOT ON
    RHO. Per-series mean subtraction gives the classic Kendall bias
    ``E[r1] ~ rho − (1 + 3 rho)/T``; at rho=0.9, T=60 that is **0.8383**, and
    the seed-0 draw MEASURES **0.8331** (this box, 2026-08-16). The estimator
    is verified to CONVERGE on rho when T grows: at T=600 the same code
    measures **0.8952** against a Kendall expectation of 0.8938. Centring the
    band on 0.9 would fail on the bias and 'fixing' that by widening the band
    would blunt the absence check — so the band is the bias-corrected one."""
    g = torch.Generator().manual_seed(0)
    p9 = DiffusionProposalGenerator(8, 8, 60, rho=0.9, hidden=8)
    e9 = p9.sample_ou_noise(4, g)
    r9 = p9.measured_lag_autocorr(e9)
    assert 0.79 <= r9 <= 0.89, f"rho=0.9 draw measured {r9} " \
        f"(Kendall-expected 0.8383 at T=60)"
    p0 = DiffusionProposalGenerator(8, 8, 60, rho=0.0, hidden=8)
    e0 = p0.sample_ou_noise(4, g)
    r0 = p0.measured_lag_autocorr(e0)
    assert abs(r0) <= 0.06, f"rho=0 (white) draw measured {r0}"
    # lag-2 of an AR(1) is rho^2 = 0.81 with roughly doubled bias; the seed-0
    # draw measures 0.6786.
    r9_2 = p9.measured_lag_autocorr(e9, lag=2)
    assert 0.62 <= r9_2 <= 0.80, f"lag-2 at rho=0.9 measured {r9_2}"
    # convergence-to-rho check at long T: the bias is the ESTIMATOR's, not the
    # process's (T=600 => Kendall 0.8938; measured 0.8952 at seed 1).
    pl = DiffusionProposalGenerator(8, 64, 600, rho=0.9, hidden=8)
    rl = pl.measured_lag_autocorr(
        pl.sample_ou_noise(8, torch.Generator().manual_seed(1)))
    assert 0.87 <= rl <= 0.92, f"T=600 draw measured {rl}"


def test_diffusion_fan_is_feasible_by_construction_and_reports_the_measure():
    s = build(tiny_cfg(proposals="diffusion"))
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    assert p["prop_mechanism"] == "diffusion"
    assert p["a"].shape == (2, 3, 60) and p["kappa"].shape == (2, 3, 60)
    assert p["waypoints"].shape == (2, 3, 60, 2)
    assert float(p["a"].detach().abs().max()) < s.cfg.a_max + 1e-6
    assert float(p["kappa"].detach().abs().max()) < s.cfg.kappa_max + 1e-6
    # the MEASURED autocorrelation travels with the fan
    assert isinstance(p["prop_noise_lag1_autocorr"], float)
    assert 0.75 <= p["prop_noise_lag1_autocorr"] <= 1.0
    # the paired query/CV reference fan is emitted beside it
    assert p["qfan_a"].shape == (2, 3, 60)
    assert p["qfan_waypoints"].shape == (2, 3, 60, 2)


def test_diffusion_fan_is_DIVERSE_at_init_while_the_query_fan_is_degenerate():
    """The zero-init discipline's two faces, measured on one window: the
    query fan at the CV warm start is DEGENERATE (every candidate identical),
    while the diffusion fan's candidates differ by construction (the fan IS
    the squashed correlated-noise prior at init)."""
    s = build(tiny_cfg(proposals="diffusion"))
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    q_spread = p["qfan_waypoints"].detach()[:, :, -1, :].std(dim=1).norm(dim=-1)
    d_spread = p["waypoints"].detach()[:, :, -1, :].std(dim=1).norm(dim=-1)
    assert float(q_spread.max()) < 1e-6, "query fan should be CV-degenerate"
    assert float(d_spread.min()) > 0.1, "diffusion fan should be diverse"


def test_denoiser_output_layer_is_zero_init_and_departure_is_learned():
    g = DiffusionProposalGenerator(8, 4, 60, hidden=16, n_steps=2)
    assert float(g.conv_out.weight.abs().max()) == 0.0
    assert float(g.conv_out.bias.abs().max()) == 0.0
    gen = torch.Generator().manual_seed(3)
    pf, ge = torch.randn(2, 4), torch.randn(2, 4)
    v0 = torch.rand(2) * 10 + 1
    a0 = g(pf, ge, v0, generator=torch.Generator().manual_seed(7))["a"]
    with torch.no_grad():
        g.conv_out.weight.normal_(0.0, 0.5, generator=gen)
    a1 = g(pf, ge, v0, generator=torch.Generator().manual_seed(7))["a"]
    assert not torch.allclose(a0, a1), \
        "waking the zero-init output layer must move the fan"


def test_diffusion_is_deterministic_under_a_generator_and_stochastic_without():
    s = build(tiny_cfg(proposals="diffusion"))
    z = torch.randn(2, s.cfg.d_op)
    g = torch.randn(2, s.cfg.d_goal_embed)
    v0 = torch.rand(2) * 10 + 1
    p1 = s.emit(z, g, v0, generator=torch.Generator().manual_seed(11))
    p2 = s.emit(z, g, v0, generator=torch.Generator().manual_seed(11))
    p3 = s.emit(z, g, v0, generator=torch.Generator().manual_seed(12))
    assert torch.equal(p1["a"], p2["a"])
    assert not torch.equal(p1["a"], p3["a"])


def test_X3_is_zero_with_diffusion_on():
    s = build(tiny_cfg(proposals="diffusion"))
    rep = s.assert_isolation()
    assert rep["pass"] is True
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}


def test_planner_surface_is_total_with_the_diffusion_arm():
    """Every planner-group parameter — including the denoiser's AND the
    now-reference-only emission/cand_queries — must stay REACHABLE from the
    declared planner_side surface. Probed off the zero inits, per the
    established test-local perturbation pattern (reachability is an
    architecture property; the shipped init is untouched)."""
    s = build(tiny_cfg(proposals="diffusion"))
    with torch.no_grad():
        s.emission.net[-1].weight.normal_(0.0, 0.1)
        s.emission.net[-1].bias.normal_(0.0, 0.1)
        for blk in s.predictor_op.blocks:
            blk.film.to_scale_shift.weight.normal_(0.0, 0.1)
        s.prop_diffusion.conv_out.weight.normal_(0.0, 0.1)
        s.prop_diffusion.conv_out.bias.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    planner = list(s.group_parameters("planner"))
    assert any(n.startswith("prop_diffusion.") for n, _ in planner)
    live = V6Stack._live_edges(V6Stack._probe_scalar(out["planner_side"]),
                               planner)
    missing = {n for n, _ in planner} - set(live)
    assert not missing, f"planner params invisible to the probe: {missing}"


# =========================================================================== #
# 3. MPC TOP-K REFINEMENT (the selection cell, per D-1)
# =========================================================================== #

def test_mpc_refuses_to_exist_without_the_goal_selector():
    """⛔ INERT UNLESS A SELECTOR IS ADMISSIBLE — the structural half. 'none'
    has no selector; 'mlp' (the capacity control) emits no goal point, and
    descending on its score would be candidate-DEPENDENT — the refuted
    roll-cost family."""
    with pytest.raises(ValueError, match="selector='goal'"):
        tiny_cfg(mpc_refine=True)
    with pytest.raises(ValueError, match="candidate-DEPENDENT"):
        tiny_cfg(mpc_refine=True, selector="mlp")


@pytest.mark.parametrize("kw,msg", [
    (dict(mpc_w_goal=0.0), "primary"),
    (dict(mpc_w_consist=0.5), "silently never computes"),
    (dict(mpc_topk=0), "mpc_topk"),
    (dict(mpc_topk=99), "mpc_topk"),
    (dict(mpc_steps=0), "mpc_steps"),
    (dict(mpc_lr=0.0), "mpc_lr"),
    (dict(mpc_w_kin=-1.0), "mpc_w"),
])
def test_mpc_config_guards_fire(kw, msg):
    with pytest.raises(ValueError, match=msg):
        tiny_cfg(selector="goal", mpc_refine=True, **kw)


def test_mpc_module_itself_refuses_a_zeroed_primary_term():
    with pytest.raises(ValueError, match="REFUTED"):
        MpcRefiner(w_goal=0.0)


def _mpc_stack(**kw):
    base = dict(selector="goal", mpc_refine=True, mpc_topk=2, mpc_steps=3)
    base.update(kw)
    return build(tiny_cfg(**base))


def test_mpc_outputs_shapes_bounds_and_detachment():
    s = _mpc_stack()
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    assert p["mpc_controls"].shape == (2, 2, 60, 2)
    assert p["mpc_waypoints"].shape == (2, 2, 60, 2)
    assert p["mpc_topk_idx"].shape == (2, 2)
    # feasible at every iterate by construction (squash re-bounding)
    assert float(p["mpc_controls"][..., 0].abs().max()) <= s.cfg.a_max + 1e-6
    assert float(p["mpc_controls"][..., 1].abs().max()) \
        <= s.cfg.kappa_max + 1e-6
    # ⛔ the refinement trains NOTHING and nothing trains through it
    for k in ("mpc_controls", "mpc_waypoints", "mpc_cost_pre",
              "mpc_cost_post", "mpc_goal_dist_post"):
        assert p[k].grad_fn is None and not p[k].requires_grad, k


def test_mpc_cost_descent_actually_descends():
    s = _mpc_stack(mpc_steps=5, mpc_lr=0.05)
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    assert float(p["mpc_cost_post"].mean()) < float(p["mpc_cost_pre"].mean())


def test_mpc_rescore_is_goal_distance_only():
    """⛔ D-1 / E-S1-0: the selected refined candidate is the argmin of the
    CANDIDATE-INDEPENDENT goal distance — never roll-consistency, never a
    learned score off-distribution. The emitted audit string says so."""
    s = _mpc_stack()
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    want = p["mpc_goal_dist_post"].argmin(dim=-1)
    assert torch.equal(p["mpc_selected_local"], want)
    ar = torch.arange(2)
    assert torch.equal(p["mpc_selected"],
                       p["mpc_topk_idx"][ar, want])
    assert "goal_distance_only" in p["mpc_rescore"]
    assert "REFUTED" in p["mpc_rescore"]


def test_mpc_consistency_regularizer_rolls_and_leaves_params_untouched():
    s = _mpc_stack(mpc_w_consist=0.5, mpc_roll_k=4)
    for prm in s.parameters():
        assert prm.grad is None
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    assert "mpc_consist_pre" in p and "mpc_consist_post" in p
    assert p["mpc_consist_post"].shape == (2, 2)
    # the inner descent used autograd.grad on delta only — no parameter
    # accumulated a gradient
    for prm in s.parameters():
        assert prm.grad is None


def test_X3_is_zero_with_mpc_on():
    s = _mpc_stack(mpc_w_consist=0.2, mpc_roll_k=3)
    rep = s.assert_isolation()
    assert rep["pass"] is True
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}


def test_roll_consistency_shapes_and_gradient_reaches_controls_not_trunk():
    """The shared W7-quantity instrument: [B, N] cost; gradient flows to the
    CONTROLS (the MPC descent's requirement) and NOT to the passed trunk
    tensors (detached inside — the zh_op_seam discipline)."""
    s = build(tiny_cfg())
    b = s.synthetic_batch(2)
    states = s.encode_window(b["frames"]).detach().requires_grad_(True)
    actions = b["actions"].requires_grad_(True)
    a_ctl = (torch.randn(2, 3, 60) * 0.5).requires_grad_(True)
    kappa = (torch.randn(2, 3, 60) * 0.05).requires_grad_(True)
    cost = s.roll_consistency(states, actions, a_ctl, kappa, b["v0"], k=4)
    assert cost.shape == (2, 3)
    ga, gs = torch.autograd.grad(cost.sum(), [a_ctl, states],
                                 allow_unused=True)
    assert ga is not None and float(ga.abs().max()) > 0.0
    assert gs is None, "trunk states must be detached inside the instrument"


# =========================================================================== #
# 4. THE FALLBACK TRIGGER (context brain, F-17)
# =========================================================================== #

def test_fallback_uncalibrated_refuses_to_fire_and_says_why():
    s = build(tiny_cfg(fallback_trigger=True, fallback_roll_k=0))
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    assert p["fb_fired"] is None and p["fb_pred_err"] is None
    assert "UNCALIBRATED" in p["fb_status"]
    assert p["fb_signal_includes_rollvar"] is False  # roll_k=0: spread-only
    assert p["fb_spread"].shape == (2,)


def test_fallback_calibration_gate_refuses_an_uncalibrated_signal():
    """⛔ P7's pre-registered gate as a loader refusal — proven able to FAIL
    in every direction it guards."""
    t = FallbackTrigger()
    with pytest.raises(ValueError, match="P7"):
        t.load_calibration({**GOOD_CALIB, "spearman_rho": 0.2})   # rho < gate
    with pytest.raises(ValueError, match="P7"):
        t.load_calibration({**GOOD_CALIB, "rho_ci": [-0.05, 0.4]})  # CI incl 0
    with pytest.raises(ValueError, match="interval-free"):
        t.load_calibration({**GOOD_CALIB, "rho_ci": None})
    with pytest.raises(ValueError, match="disabled planner"):
        t.load_calibration({**GOOD_CALIB, "threshold": 0.0})
    with pytest.raises(ValueError, match="w_spread"):
        t.load_calibration({**GOOD_CALIB, "w_spread": 0.0, "w_rollvar": 0.0})
    with pytest.raises(ValueError, match="missing"):
        t.load_calibration({"spearman_rho": 0.7})
    assert bool(t.calib_ready) is False  # nothing partial was installed
    prov = t.load_calibration(GOOD_CALIB)                      # the good path
    assert bool(t.calib_ready) is True
    assert prov["gate"].endswith("PASSED")


def test_fallback_fires_both_ways_when_calibrated():
    """The comparator can say YES and can say NO on the same calibration —
    a trigger that only ever says one of them is a constant, not a guard."""
    t = FallbackTrigger(plan_steps=60)
    t.load_calibration(GOOD_CALIB)
    v0 = torch.full((2,), 5.0)
    tight = torch.zeros(2, 4, 60, 2)                # zero spread
    wide = torch.zeros(2, 4, 60, 2)
    wide[:, :, -1, 0] = torch.tensor([[0., 2., 4., 6.]] * 2)  # spread >> thr
    o_t = t(tight, v0)
    o_w = t(wide, v0)
    assert o_t["fired"].tolist() == [False, False]
    assert o_w["fired"].tolist() == [True, True]


def test_fallback_is_permutation_invariant_hence_not_a_selector():
    """⛔ NEVER A SELECTOR — structural, not rhetorical: every statistic is
    computed over the candidate axis as a SET (std/var), so permuting the
    fan changes NOTHING and no per-candidate output exists for an argmax to
    read. A module that cannot distinguish candidates cannot select one."""
    t = FallbackTrigger(plan_steps=60)
    t.load_calibration(GOOD_CALIB)
    g = torch.Generator().manual_seed(5)
    wp = torch.randn(2, 6, 60, 2, generator=g)
    rc = torch.randn(2, 6, generator=g).abs()
    perm = torch.randperm(6, generator=g)
    o1 = t(wp, torch.full((2,), 5.0), roll_cost=rc)
    o2 = t(wp[:, perm], torch.full((2,), 5.0), roll_cost=rc[:, perm])
    for k in ("spread", "roll_cost_var", "signal", "pred_err"):
        assert torch.allclose(o1[k], o2[k], atol=1e-6), k
    assert torch.equal(o1["fired"], o2["fired"])
    per_candidate = [k for k, v in o1.items()
                     if torch.is_tensor(v) and v.ndim >= 2
                     and v.shape[:2] == (2, 6)]
    assert not per_candidate, \
        f"per-candidate outputs would let a consumer select: {per_candidate}"


def test_fallback_action_is_hold_v0_straight_and_feasible():
    """The DEFINED fallback: zero accel, zero curvature, integrated by the
    same unicycle — x advances at v0, y stays 0."""
    t = FallbackTrigger(plan_steps=60)
    v0 = torch.tensor([4.0, 10.0])
    o = t(torch.zeros(2, 3, 60, 2), v0)
    assert float(o["controls"].abs().max()) == 0.0
    wp = o["waypoints"]
    assert wp.shape == (2, 60, 2)
    assert torch.allclose(wp[:, -1, 0], v0 * 6.0, atol=1e-4)  # 60 x 0.1 s
    assert float(wp[..., 1].abs().max()) < 1e-6
    assert "hold_v0" in o["action"]


def test_fallback_outputs_are_monitor_grade_no_grad():
    """Monitored, never optimised: nothing the trigger emits carries a graph,
    so no loss can learn to blind it."""
    s = build(tiny_cfg(fallback_trigger=True, fallback_roll_k=3))
    o = s.forward(**s.synthetic_batch(2))
    p = o["plan"]
    for k in ("fb_spread", "fb_roll_cost_var", "fb_signal", "fb_waypoints"):
        assert p[k].grad_fn is None and not p[k].requires_grad, k
    assert p["fb_signal_includes_rollvar"] is True


def test_fallback_keys_are_buffers_only_and_X3_zero():
    d = build(tiny_cfg())
    s = build(tiny_cfg(fallback_trigger=True))
    assert _n(s) - _n(d) == 0, "the trigger must hold no trainable parameter"
    new = {k for k in s.state_dict() if k not in d.state_dict()}
    assert len(new) == FALLBACK_KEYS
    assert all(k.startswith("fallback.") for k in new)
    rep = s.assert_isolation()
    assert rep["pass"] is True


def test_p7_gate_constant_matches_the_instrument():
    """v6 duplicates P7_GATE_RHO as data (a model module must not import an
    instrument script's dependency chain); this is the drift pin."""
    import w7_roll_rerank
    assert P7_GATE_RHO == w7_roll_rerank.P7_GATE_RHO


# =========================================================================== #
# 5. ALL THREE COMBINED + the dry S-T step end-to-end
# =========================================================================== #

def _all_on_cfg(**kw):
    base = dict(proposals="diffusion", selector="goal", mpc_refine=True,
                mpc_topk=2, mpc_steps=2, mpc_w_consist=0.2, mpc_roll_k=3,
                fallback_trigger=True, fallback_roll_k=3)
    base.update(kw)
    return tiny_cfg(**base)


def test_X3_is_zero_with_all_three_on_combined():
    s = build(_all_on_cfg())
    rep = s.assert_isolation()
    assert rep["pass"] is True
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}


def test_dry_ST_step_end_to_end_with_all_three_on():
    """⛔ THE EXERCISE CLAUSE: a full S-T loss step — forward, every S-T
    loss in force (t1 + seam + plan WTA + softade selection), backward —
    with diffusion + MPC + fallback all ON. The diffusion generator must
    RECEIVE gradient through the plan loss (it is the fan generator, trained
    by the same objective as the query fan), while the MPC/fallback paths
    stay out of the graph."""
    from train_v6_staged import (V6LossWeights, synthetic_train_batch,
                                 v6_loss_step)
    s = build(_all_on_cfg())
    batch = synthetic_train_batch(s, batch=2, k=12)
    w = V6LossWeights(lambda_plan=1.0, w_select=0.1)
    out = v6_loss_step(s, batch, stage="S-T", weights=w)
    assert torch.isfinite(out["loss"])
    log = out["log"]
    for k in ("t1_latent", "seam_op", "plan_wta", "sel_softade", "sel_ade",
              "sel_norm_err_rank", "sel_gap", "fan_oracle_ade"):
        assert k in log, k
    plan = out["out"]["plan"]
    assert plan["prop_mechanism"] == "diffusion"
    assert "mpc_controls" in plan and "fb_status" in plan
    out["loss"].backward()
    dn = [p for n, p in s.named_parameters()
          if n.startswith("prop_diffusion.") and p.grad is not None
          and float(p.grad.abs().max()) > 0.0]
    assert dn, "the diffusion generator must train through the plan loss"
    # and the S-T objective reaches NO encoder/readout parameter even before
    # apply_stage_freeze — every S-T term reads a cut or detached view (the
    # X3 probe's claim, re-measured here on the real loss graph).
    enc = [n for n, p in s.group_parameters("encoder", "readout")
           if p.grad is not None and float(p.grad.abs().max()) > 0.0]
    assert not enc, f"S-T loss reached the trunk: {enc[:4]}"
    s.zero_grad(set_to_none=True)


def test_emit_without_roll_ctx_skips_the_roll_halves_and_says_so():
    s = build(_all_on_cfg())
    z = torch.randn(2, s.cfg.d_op)
    g = torch.randn(2, s.cfg.d_goal_embed)
    v0 = torch.rand(2) * 10 + 1
    p = s.emit(z, g, v0)                      # no roll_ctx
    assert p["fb_signal_includes_rollvar"] is False
    assert "mpc_consist_skipped" in p
    assert "mpc_consist_post" not in p


# =========================================================================== #
# 6. TRAINER SURFACE — flags, preflight refusals, stage introduction
# =========================================================================== #

def _args(tmp_path, stage="S-T", **over):
    from train_v6_staged import build_parser
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack")
    base = ["--stage", stage, "--out", str(tmp_path / "out")]
    argv = base + [x for kv in over.items()
                   for x in ([kv[0]] if kv[1] is True else [kv[0], str(kv[1])])]
    return ap.parse_args(argv)


def test_parser_defaults_are_the_inert_ones(tmp_path):
    a = _args(tmp_path)
    assert a.proposals == "query"
    assert a.mpc_refine is False and a.fallback_trigger is False
    assert a.fallback_calibration is None
    assert a.diffusion_noise_rho == 0.9 and a.diffusion_steps == 4
    assert a.mpc_w_goal == 1.0 and a.mpc_w_consist == 0.0


def test_preflight_refuses_diffusion_and_fallback_in_S_W(tmp_path):
    from train_v6_staged import preflight
    p = preflight(_args(tmp_path, stage="S-W", **{"--proposals": "diffusion"}))
    assert any("prop_diffusion" in x for x in p)
    p = preflight(_args(tmp_path, stage="S-W", **{"--fallback-trigger": True}))
    assert any("fallback.*" in x for x in p)


def test_preflight_refuses_mpc_without_the_goal_selector(tmp_path):
    from train_v6_staged import preflight
    p = preflight(_args(tmp_path, **{"--mpc-refine": True}))
    assert any("candidate-DEPENDENT" in x for x in p)


def test_preflight_refuses_an_orphan_or_missing_calibration(tmp_path):
    from train_v6_staged import preflight
    calib = tmp_path / "cal.json"
    calib.write_text("{}")
    p = preflight(_args(tmp_path, **{"--fallback-calibration": str(calib)}))
    assert any("reaches no comparator" in x for x in p)
    p = preflight(_args(tmp_path, **{"--fallback-trigger": True,
                                     "--fallback-calibration":
                                         str(tmp_path / "missing.json")}))
    assert any("does not" in x and "exist" in x for x in p)


def test_stage_may_introduce_admits_the_new_modules_at_S_T_only():
    from train_v6_staged import STAGE_MAY_INTRODUCE
    assert "prop_diffusion." in STAGE_MAY_INTRODUCE["S-T"]
    assert "fallback." in STAGE_MAY_INTRODUCE["S-T"]
    for st in ("S-W", "S-S", "S-J"):
        assert "prop_diffusion." not in STAGE_MAY_INTRODUCE[st]
        assert "fallback." not in STAGE_MAY_INTRODUCE[st]


def test_load_stage_init_introduces_the_new_keys_over_a_bare_ckpt(tmp_path):
    """The introduction path, EXECUTED: an S-W-shaped checkpoint (no new
    modules) strict-loads into an S-T stack built with diffusion + fallback
    ON; the fresh keys are adjudicated INTRODUCED, not fatal — and the same
    load at a stage with no allowance is REFUSED (the guard can fail)."""
    from train_v6_staged import load_stage_init
    donor = build(tiny_cfg())
    ck = tmp_path / "ckpt.pt"
    torch.save({"stack": donor.state_dict(), "step": 7,
                "config": {"stage": "S-W"}}, ck)
    s = build(tiny_cfg(proposals="diffusion", fallback_trigger=True))
    rep = load_stage_init(s, ck, stage="S-T")
    intro = set(rep["introduced_keys"])
    assert intro and all(k.startswith(("prop_diffusion.", "fallback."))
                         for k in intro)
    assert not rep["missing_keys"] and not rep["unexpected_keys"]
    s2 = build(tiny_cfg(proposals="diffusion", fallback_trigger=True))
    with pytest.raises(SystemExit, match="not a valid predecessor"):
        load_stage_init(s2, ck, stage="S-S")


def test_build_stack_from_args_builds_all_three_and_loads_a_calibration(
        tmp_path):
    import json as _json
    from train_v6_staged import build_stack_from_args
    calib = tmp_path / "cal.json"
    calib.write_text(_json.dumps(GOOD_CALIB))
    a = _args(tmp_path, **{
        "--in-channels": 3, "--frame-h": 32, "--frame-w": 32,
        "--enc-dim": 32, "--enc-depth": 1, "--enc-heads": 2,
        "--readout-grid": 4, "--readout-dim": 8,
        "--pred-dim": 32, "--pred-depth": 1, "--pred-heads": 2,
        "--window": 4, "--d-tac": 32, "--d-str": 16, "--d-goal-embed": 16,
        "--adapter-hidden": 32, "--f-hidden-tac": 32, "--f-hidden-str": 32,
        "--n-candidates": 3, "--diffusion-hidden": 32,
        "--proposals": "diffusion", "--selector": "goal",
        "--w-select": "0.1", "--mpc-refine": True,
        "--fallback-trigger": True, "--fallback-calibration": str(calib)})
    s = build_stack_from_args(a)
    assert s.prop_diffusion is not None and s.mpc is not None
    assert s.fallback is not None and bool(s.fallback.calib_ready)
    assert float(s.fallback.calib_rho) == pytest.approx(0.7164)
