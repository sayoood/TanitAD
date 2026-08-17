"""The g_str -> P_T conditioning port (F-1) — gated, zero-init, proved inert
when off and ALIVE when on.

⛔ WHY THIS FILE EXISTS. DIAGRAM_CONFORMANCE.md (2026-08-16) audited the binding
v6 diagram element by element and its TOP 🟥 finding was F-1: the diagram,
`HIERARCHY_VOCABULARY` §5 (*"z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)"*),
`V6_TRAINER_DESIGN` §1.2's ASCII, **and `V6Stack`'s own class docstring** all
spec the strategic goal conditioning the tactical DYNAMICS — and the code did
not build it. `FTac`'s one conditioning input was fully consumed by the LAT×LON
action embeddings; `e_g_str` reached only `goal_head_tac`. An S-T launched
as-is would never train the strategic→tactical dynamics downlink in its own
stage — the `intent_proj` defect one level up (a conditioning path present in
the diagram, absent from the optimisation), and precisely the fake-hierarchy
failure the PI's remarks guard against.

THE FIX UNDER TEST: `cfg.tac_goal_cond` (default OFF) builds
`V6Stack.cond_tac_dyn`, a ZERO-INIT `Linear(d_goal_embed -> 2*d_goal_embed)`
whose output is ADDED to `e_a_tac` before `predictor_tac` consumes it — the
same additive idiom the accepted g_tac→P_O seam uses one level down
(`cond = act_emb(actions) + intent_proj(intent)`), and a NEW MODULE rather
than a widened `FTac.in_proj` because `STAGE_MAY_INTRODUCE` adjudicates KEYS
and a shape change bypasses it entirely (`load_state_dict(strict=False)` still
RAISES on shapes — measured, quoted in `trainer_argv`'s --n-candidates note).

⛔ AND THE CONSTRAINT THAT OUTRANKS ALL OF IT: v6F S-W is resuming on Thor from
a checkpoint of the incumbent architecture — **87,893,449 params / 405 keys**
(config E: **336,542,025 / 573** at the PI-raised 350 M budget). The DEFAULT
build's ``state_dict`` must stay byte-identical, proved per tensor with
``torch.equal`` against a CONTENT-anchored pre-change revision of ``v6.py`` —
never against HEAD (C75: while the factored-goal file was being written HEAD
moved and swept the in-progress module into a commit, after which a HEAD
comparison is a module compared with itself).

Every number in this file's literals is MEASURED (2026-08-16, this box, torch
CPU build at seed 0) — recompute on drift, never inherit.
"""
from __future__ import annotations

import copy
import importlib.util
import json
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
    STAGE_GROUPS, V6Config, V6Stack, apply_stage_freeze)

#: MEASURED before this turn's edits (the factored-goal file's own literals,
#: re-verified on the edited module 2026-08-16): the counts the DEFAULT build
#: must not move. FULL is the geometry class of the live S-W resume.
HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS = 611_293, 223
HEAD_FULL_PARAMS, HEAD_FULL_KEYS = 87_893_449, 405
#: MEASURED 2026-08-16 by rebuilding config E from the banked live-run argv
#: (`…/2026-08-15-v6-thor-resume/code/RESTART_v6F_SW.sh`, generated from
#: /proc of the RUNNING trainer): enc 768x12x12 ViT-5 + registers, pred-modern
#: 1024x12x16, d_tac 768 / d_str 512, f_hidden 1024 x 6 blocks, budget 350 M.
#: 336,542,025 matches the registry/handover number exactly, which is the
#: cross-check that this reconstruction IS config E.
CONFIG_E_PARAMS, CONFIG_E_KEYS = 336_542_025, 573
#: the port's cost at the production d_goal_embed=128:
#: Linear(128 -> 256) = 128*256 + 256. MEASURED at both geometries below.
PORT_DELTA = 33_024
PORT_KEYS = ("cond_tac_dyn.weight", "cond_tac_dyn.bias")

_SMALL_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1,
    op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0,
    hz_str=0.5, d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
    n_candidates=8)


def _sub_cfgs() -> dict:
    return dict(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  horizons=(1,), action_dim=3, residual=True))


def _small(**kw) -> V6Config:
    return V6Config(**{**_sub_cfgs(), **_SMALL_KW, **kw})


def _config_e(**kw) -> V6Config:
    return V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=256, image_width=640,
                              patch_size=16, d_model=768, depth=12,
                              n_heads=12),
        readout=ReadoutConfig(grid=4, d_readout=128),
        predictor=PredictorConfig(d_model=1024, depth=12, n_heads=16, window=6,
                                  horizons=(1, 2, 4), action_dim=3,
                                  residual=True, modern=True),
        d_tac=768, d_str=512, d_goal_embed=128, adapter_hidden=512,
        f_hidden_tac=1024, f_hidden_str=1024, f_blocks=6,
        vit5_encoder=True, n_registers=4, param_budget=350_000_000, **kw)


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def _n(m) -> int:
    return sum(p.numel() for p in m.parameters())


def _wake_port(s: V6Stack, std: float = 0.5, seed: int = 7) -> None:
    """Take the port OFF its zero init — the 'trained' surrogate every
    aliveness check needs (at zero weights the port is a no-op BY DESIGN)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        s.cond_tac_dyn.weight.copy_(
            torch.randn(s.cond_tac_dyn.weight.shape, generator=g) * std)


def _perturb_g_str(s: V6Stack, scale: float = 3.0, seed: int = 11) -> None:
    """Move the strategic goal path (the vocabulary the conditioner reads), so
    `e_g_str` provably changes between two forwards on the SAME input."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        s.vocab_str.table.weight.add_(
            torch.randn(s.vocab_str.table.weight.shape, generator=g) * scale)


# =========================================================================== #
# 1. ⛔ BYTE-IDENTITY OF THE DEFAULT BUILD — the live resume is untouchable
# =========================================================================== #

def test_the_port_defaults_off_and_is_absent():
    c = V6Config()
    assert c.tac_goal_cond is False
    s = _build(_small())
    assert s.cond_tac_dyn is None
    assert not any(k.startswith("cond_tac_dyn.") for k in s.state_dict())


def test_default_counts_are_the_MEASURED_head_counts():
    s = _build(_small())
    assert (_n(s), len(s.state_dict())) == (HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS)


@pytest.mark.slow
def test_default_FULL_config_counts_are_the_live_resume_counts():
    """87,893,449 / 405 — the numbers a broken strict resume would kill."""
    f = _build(V6Config())
    assert (_n(f), len(f.state_dict())) == (HEAD_FULL_PARAMS, HEAD_FULL_KEYS)


@pytest.mark.slow
def test_config_E_default_build_is_unchanged_and_within_its_budget():
    """The REAL live geometry (PI-raised 350 M budget). The literal matching
    the registry/handover 336,542,025 is itself the check that this
    reconstruction from the banked /proc argv IS config E."""
    e = _build(_config_e())
    assert (_n(e), len(e.state_dict())) == (CONFIG_E_PARAMS, CONFIG_E_KEYS)
    rep = e.assert_param_budget()
    assert rep["within_budget"] and rep["budget"] == 350_000_000


#: The marker that separates the pre-F-1 architecture from this one. Any
#: revision of ``v6.py`` WITHOUT it is one the live checkpoints could have
#: been built from.
_PRE_CHANGE_MARKER = "tac_goal_cond"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_change_module():
    """``tanitad.models.v6`` as it was BEFORE the port existed — CONTENT-
    anchored, never HEAD (C75): we walk ``v6.py``'s own git history for the
    NEWEST revision that does not yet carry ``tac_goal_cond``. That reference
    is stable however many commits land afterwards, and it is the semantically
    right one: it is the architecture the live checkpoints stand on.

    Returns None when git cannot answer; the caller skips — a skipped test is
    honest, a self-comparison dressed as a real one is not.
    """
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
                continue                      # already carries the port
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    sp = str(_STACK / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_gstr_port.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_gstr_port", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_gstr_port"] = mod
    spec.loader.exec_module(mod)
    mod._ref = ref
    return mod


def test_default_is_byte_identical_to_the_PRE_CHANGE_architecture():
    """⛔ THE ONE THAT PROTECTS THE LIVE RUN. Per tensor, ``torch.equal`` —
    never a digest of a ``torch.save`` container (C72: those bytes are not
    canonical) — plus the RNG STREAM, because a default path that consumed one
    extra draw would leave the state_dict identical and still desynchronise
    everything initialised after the model."""
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")

    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config(**{**_sub_cfgs(), **_SMALL_KW}))
    rng_old = torch.random.get_rng_state()
    torch.manual_seed(0)
    new = _build(_small())
    rng_new = torch.random.get_rng_state()

    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn), (
        f"state_dict KEYS moved against {head._ref}: "
        f"only-old={set(so) - set(sn)}, only-new={set(sn) - set(so)}")
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} MOVED against {head._ref}"
    assert torch.equal(rng_old, rng_new), \
        "the default build consumed a different number of random draws"


@pytest.mark.slow
def test_default_FULL_build_is_byte_identical_to_the_PRE_CHANGE_architecture():
    """The same proof at the FULL default geometry — the class the live S-W
    resume belongs to. Slow: two 87.9 M-parameter CPU builds."""
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")
    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config())
    torch.manual_seed(0)
    new = _build(V6Config())
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} MOVED against {head._ref}"


def test_default_forward_is_bit_identical_and_emits_no_new_key():
    """Weights identical is necessary, not sufficient: the default FORWARD must
    also be untouched. With the flag off, `g_cond_tac` is the SAME tensor
    object as `e_a_tac`, so `zhat_tac` cannot move — asserted against the
    pre-change module's forward on the same batch."""
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")
    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config(**{**_sub_cfgs(), **_SMALL_KW}))
    new = _build(_small())
    b = new.synthetic_batch(2)
    o_old = old.forward(**b)
    o_new = new.forward(**b)
    assert set(o_old) == set(o_new), "the default forward grew an output key"
    assert torch.equal(o_old["zhat_tac"], o_new["zhat_tac"])
    assert torch.equal(o_old["e_g_str"], o_new["e_g_str"])


# =========================================================================== #
# 2. ⭐ TURNING THE PORT ON — additive, attributable, grouped with its dynamics
# =========================================================================== #

def test_turning_the_port_on_perturbs_NO_pre_existing_tensor():
    """The capacity-control discipline every gated lever here obeys: the ON
    arm is the OFF arm plus tensors, never the OFF arm re-initialised — which
    only holds because the port is constructed at the END of ``__init__``."""
    off, on = _build(_small()), _build(_small(tac_goal_cond=True))
    so, sn = off.state_dict(), on.state_dict()
    assert set(sn) - set(so) == set(PORT_KEYS)
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} moved when the port flipped on"
    assert _n(on) - _n(off) == PORT_DELTA


@pytest.mark.slow
def test_port_delta_on_the_REAL_config_is_the_recorded_literal():
    """MEASURE, do not arithmetic (the selector's +41,089 estimate was never
    realised; the implementation cost +33,801)."""
    off = _build(V6Config())
    on = _build(V6Config(tac_goal_cond=True))
    assert _n(on) - _n(off) == PORT_DELTA
    assert len(on.state_dict()) - len(off.state_dict()) == len(PORT_KEYS)


def test_the_port_is_zero_init_and_reads_exactly_d_goal_embed():
    """Zero-init is the loss-continuity-at-introduction discipline
    (GoalDistanceScorer.goal_point / MLPCandidateScorer.fc2 / the emission
    head); the input width pins WHAT the port reads — the strategic goal
    embedding, nothing wider (no ego, no situation channel exists to widen it
    with — X1)."""
    s = _build(_small(tac_goal_cond=True))
    assert float(s.cond_tac_dyn.weight.detach().abs().sum()) == 0.0
    assert float(s.cond_tac_dyn.bias.detach().abs().sum()) == 0.0
    assert s.cond_tac_dyn.in_features == s.cfg.d_goal_embed
    assert s.cond_tac_dyn.out_features == 2 * s.cfg.d_goal_embed
    assert s.cond_tac_dyn.out_features == s.predictor_tac.d_goal


def test_the_port_is_grouped_layer_tac_and_freezes_with_its_dynamics():
    """`layer_tac`, NOT `planner` — the port conditions predictor_tac, which
    already trains in S-T (the stage whose t1 loss flows through the
    conditioned prediction), so it trains exactly when the conditioning is
    live: the property whose absence was the intent_proj defect. A planner
    grouping would also break the planner-surface totality test, because the
    port's output feeds `zh_tac` (uplink-side), not the plan."""
    s = _build(_small(tac_goal_cond=True))
    assert s.group_of("cond_tac_dyn.weight") == "layer_tac"
    assert s.group_of("cond_tac_dyn.bias") == "layer_tac"
    for stage, trains in (("S-W", False), ("S-T", True),
                          ("S-S", False), ("S-J", True)):
        apply_stage_freeze(s, stage)
        assert s.cond_tac_dyn.weight.requires_grad is trains, stage
        assert ("layer_tac" in STAGE_GROUPS[stage]) is trains


# =========================================================================== #
# 3. ⭐ ZERO-INIT NO-OP · ALIVENESS · THE NEGATIVE CONTROL
# =========================================================================== #
# The detector used by all three: hold the INPUT fixed, move the strategic
# goal path (vocab_str's table, which cond_tac encodes), and ask whether
# `zhat_tac` moved. e_g_str provably changes in every arm; whether that change
# REACHES P_T is exactly the F-1 question.

def test_the_port_is_an_exact_noop_at_zero_init():
    """The zero-init discipline, measured: at init, varying g_str moves
    `e_g_str` and must NOT move `zhat_tac` — bit-for-bit, not approximately.
    (This is also the negative half of the aliveness pair: the detector reads
    'not alive' on a build where the port is provably present but untrained.)"""
    s = _build(_small(tac_goal_cond=True))
    b = s.synthetic_batch(2)
    o0 = s.forward(**b)
    _perturb_g_str(s)
    o1 = s.forward(**b)
    assert not torch.equal(o0["e_g_str"], o1["e_g_str"])      # the lever moved
    assert torch.equal(o0["zhat_tac"], o1["zhat_tac"])        # the port did not


def test_t1_is_continuous_at_introduction():
    """The introduction contract: an S-T stack that just grew the zero-init
    port predicts EXACTLY what the portless stack predicts on the same batch —
    so the t1 loss curve cannot jump when `STAGE_MAY_INTRODUCE` admits the new
    keys over an S-W checkpoint."""
    off = _build(_small(), seed=0)
    on = _build(_small(tac_goal_cond=True), seed=0)
    b = off.synthetic_batch(2)
    assert torch.equal(off.forward(**b)["zhat_tac"],
                       on.forward(**b)["zhat_tac"])


def test_the_port_is_ALIVE_once_off_zero():
    """⭐ THE POINT OF F-1: with the port awake, varying g_str MOVES the
    tactical dynamics prediction — P_T is conditioned on the strategic goal,
    through the REAL `V6Stack.forward` path, not a unit harness."""
    s = _build(_small(tac_goal_cond=True))
    _wake_port(s)
    b = s.synthetic_batch(2)
    o0 = s.forward(**b)
    _perturb_g_str(s)
    o1 = s.forward(**b)
    assert not torch.equal(o0["e_g_str"], o1["e_g_str"])
    assert not torch.equal(o0["zhat_tac"], o1["zhat_tac"])
    # and the movement is THROUGH the port, not through some other seam the
    # perturbation opened: the operative/strategic predictions are driven by
    # their own conditioning, so zh_str (own action tokens, no g_str input)
    # must be unmoved by construction.
    assert torch.equal(o0["zhat_str"], o1["zhat_str"])


def test_NEGATIVE_CONTROL_the_default_build_cannot_see_g_str():
    """⛔ The aliveness detector must be able to FAIL — and its failure case is
    the F-1 defect itself, executed: on the DEFAULT build the same perturbation
    moves `e_g_str` and `zhat_tac` stays bit-identical, because no path from
    the strategic goal into `predictor_tac` exists. (If this test ever fails,
    either the default build grew a hidden g_str→P_T path — a byte-identity
    emergency — or the detector is broken; both need a human.)"""
    s = _build(_small())
    assert s.cond_tac_dyn is None
    b = s.synthetic_batch(2)
    o0 = s.forward(**b)
    _perturb_g_str(s)
    o1 = s.forward(**b)
    assert not torch.equal(o0["e_g_str"], o1["e_g_str"])      # lever moved...
    assert torch.equal(o0["zhat_tac"], o1["zhat_tac"])        # ...port absent
    # the perturbation DOES reach the places g_str legitimately flows —
    # the goal head one level down is conditioned on it — so the control is
    # not vacuous:
    assert not torch.equal(o0["g_tac"]["logits"], o1["g_tac"]["logits"])


def test_forward_emits_no_new_output_key_with_the_port_on():
    off, on = _build(_small()), _build(_small(tac_goal_cond=True))
    b = off.synthetic_batch(2)
    assert set(off.forward(**b)) == set(on.forward(**b))


# =========================================================================== #
# 4. ⛔ THE DETACH DISCIPLINE + X3 — goals flow down, gradient does not go up
# =========================================================================== #

def test_t1_gradient_reaches_the_port_and_nothing_above_or_below_it():
    """The downward-port rule, measured on a real graph: a t1-style loss on
    `zhat_tac` must reach the port's OWN parameters (even at zero init — the
    input is what is detached, not the weights) and must NOT reach
    (a) `layer_str` — `goal_head_str`/`vocab_str`, the goal's source, which the
        `self._cut(e_g_str, cut)` detach protects (the intent-port rule one
        level up: the WM loss must not train the goal path backwards through
        the conditioning seam), nor
    (b) any encoder/readout/predictor_op parameter (X3's uplink cut)."""
    s = _build(_small(tac_goal_cond=True), seed=1)
    for p in s.parameters():
        p.requires_grad_(True)
    out = s.forward(**s.synthetic_batch(2))
    loss = out["zhat_tac"].float().pow(2).mean()
    names = [nm for nm, _ in s.named_parameters()]
    grads = torch.autograd.grad(loss, [p for _, p in s.named_parameters()],
                                allow_unused=True)
    live = {nm for nm, gr in zip(names, grads)
            if gr is not None and float(gr.abs().max()) > 0}
    assert {nm for nm in live if nm.startswith("cond_tac_dyn.")} == \
        set(PORT_KEYS), "the port must train under t1 — at zero init"
    assert not {nm for nm in live if s.group_of(nm) == "layer_str"}, \
        "t1 leaked into layer_str through the port — the detach is gone"
    assert not {nm for nm in live
                if s.group_of(nm) in ("encoder", "readout", "predictor_op")}
    assert {s.group_of(nm) for nm in live} == {"layer_tac"}


@pytest.mark.parametrize("flags", [
    dict(tac_goal_cond=True),
    dict(tac_goal_cond=True, goal_factored=True),
    dict(tac_goal_cond=True, selector="goal"),
    dict(tac_goal_cond=True, selector="mlp"),
    dict(tac_goal_cond=True, goal_factored=True, goal_cat_args=True,
         selector="goal"),
])
def test_X3_isolation_holds_with_the_port_on(flags):
    """``{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}``
    — measured on a real autograd graph. `strategic_to_below` is exactly the
    edge the port adds a LEGAL, declared version of: goals flow down BY
    DESIGN, and the probe certifies the gradient does not flow back."""
    s = _build(_small(**flags))
    iso = s.assert_isolation(batch_size=2, strict=True)
    assert iso["pass"], iso
    assert iso["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0}
    assert all(v > 0 for v in iso["n_probed"].values()), iso["n_probed"]


def test_the_declared_planner_surface_stays_TOTAL_with_the_port_on():
    """The port must not orphan any planner parameter from the declared
    surface (it is layer_tac, so the planner set is unchanged — asserted, not
    assumed)."""
    s = _build(_small(tac_goal_cond=True, selector="goal"))
    with torch.no_grad():                       # off the zero inits
        s.emission.net[-1].weight.normal_(0.0, 0.1)
        s.emission.net[-1].bias.normal_(0.0, 0.1)
        for blk in s.predictor_op.blocks:
            blk.film.to_scale_shift.weight.normal_(0.0, 0.1)
    out = s.forward(**s.synthetic_batch(2))
    planner = list(s.group_parameters("planner"))
    live = V6Stack._live_edges(V6Stack._probe_scalar(out["planner_side"]),
                               planner)
    assert not {n for n, _ in planner} - set(live)
    assert not any(n.startswith("cond_tac_dyn.") for n, _ in planner)


def test_the_port_survives_a_strict_state_dict_round_trip():
    s = _build(_small(tac_goal_cond=True))
    _wake_port(s)
    t = _build(_small(tac_goal_cond=True), seed=9)
    t.load_state_dict(s.state_dict())           # strict
    assert torch.equal(t.cond_tac_dyn.weight, s.cond_tac_dyn.weight)


# =========================================================================== #
# 5. ⛔ THE LADDER SEAM — S-T introduces the port; S-S/S-J carry it
# =========================================================================== #

def _ckpt(tmp_path, stack, stage, name="ckpt.pt", step=30000):
    p = tmp_path / name
    torch.save({"stack": stack.state_dict(), "step": step,
                "config": {"stage": stage}}, p)
    return p


def test_S_T_introduces_the_port_over_an_S_W_ckpt(tmp_path):
    """The designed transition: an S-W checkpoint never carries the port; an
    S-T stack built with it strict-loads through the `STAGE_MAY_INTRODUCE`
    allowance, and the report NAMES the fresh keys."""
    from train_v6_staged import load_stage_init
    ck = _ckpt(tmp_path, _build(_small()), "S-W")
    rep = load_stage_init(_build(_small(tac_goal_cond=True)), ck, stage="S-T")
    assert rep["missing_keys"] == [] and rep["unexpected_keys"] == []
    assert sorted(rep["introduced_keys"]) == sorted(PORT_KEYS)
    assert "cond_tac_dyn." in rep["introduced_allowance"]


def test_S_T_introduces_port_AND_selector_together(tmp_path):
    from train_v6_staged import load_stage_init
    ck = _ckpt(tmp_path, _build(_small()), "S-W")
    rep = load_stage_init(
        _build(_small(tac_goal_cond=True, selector="goal")), ck, stage="S-T")
    assert rep["missing_keys"] == []
    intro = set(rep["introduced_keys"])
    assert set(PORT_KEYS) <= intro
    assert any(k.startswith("cand_score.") for k in intro)


def test_S_S_carries_the_port_cleanly_and_refuses_dropping_it(tmp_path):
    """Once S-T trained the port, it is GEOMETRY: an S-S stack with the flag
    loads clean (introduced=[] — S-S may introduce nothing and needs to); one
    without it dies on unexpected cond_tac_dyn.* keys — the same carry rule as
    the selector, measured here rather than asserted."""
    from train_v6_staged import load_stage_init
    st = _build(_small(tac_goal_cond=True))
    _wake_port(st)                        # a TRAINED port, not a zero block
    ck = _ckpt(tmp_path, st, "S-T")
    rep = load_stage_init(_build(_small(tac_goal_cond=True), seed=5), ck,
                          stage="S-S")
    assert rep["missing_keys"] == rep["unexpected_keys"] == []
    assert rep["introduced_keys"] == []
    with pytest.raises(SystemExit, match="cond_tac_dyn"):
        load_stage_init(_build(_small(), seed=6), ck, stage="S-S")


def test_a_PARTIALLY_present_port_is_fatal_not_an_introduction(tmp_path):
    """An introduction must be WHOLE (the selector's rule, inherited): a ckpt
    carrying HALF the port is a geometry mismatch wearing an allowance's
    clothes."""
    from train_v6_staged import load_stage_init
    sd = _build(_small(tac_goal_cond=True)).state_dict()
    sd.pop("cond_tac_dyn.weight")             # keep the bias
    p = tmp_path / "half.pt"
    torch.save({"stack": sd, "step": 1, "config": {"stage": "S-W"}}, p)
    with pytest.raises(SystemExit, match="cond_tac_dyn"):
        load_stage_init(_build(_small(tac_goal_cond=True)), p, stage="S-T")


# =========================================================================== #
# 6. the trainer's launch surface — preflight refusals + config plumbing
# =========================================================================== #

def _args(*argv):
    from train_v6_staged import build_parser
    import argparse
    ap = build_parser()
    ap.add_argument("--i-know-this-is-the-control-arm", action="store_true",
                    dest="control_arm_ack", help=argparse.SUPPRESS)
    return ap.parse_args(list(argv))


def test_preflight_refuses_the_port_in_S_W(tmp_path):
    """S-W freezes layer_tac AND its resume is the live run: dead weight plus
    a broken strict resume. Same family as --selector/--anchor-goal in S-W."""
    from train_v6_staged import preflight
    a = _args("--stage", "S-W", "--out", str(tmp_path), "--tac-goal-cond",
              "--dry-run")
    assert any("--tac-goal-cond" in p and "S-W" in p for p in preflight(a))


def test_preflight_refuses_an_inert_port_without_the_ack(tmp_path):
    """--tac-goal-cond with --w-t1 0 in a stage that TRAINS layer_tac: t1 is
    the port's ONLY gradient source, so this builds a port that never trains —
    the intent_proj dead-weight defect, recreated by launch line."""
    from train_v6_staged import preflight
    base = ("--out", str(tmp_path), "--tac-goal-cond", "--w-t1", "0",
            "--init-from", "x", "--dry-run")
    for stage in ("S-T", "S-J"):
        probs = preflight(_args("--stage", stage, *base))
        assert any("inert-port" in p or "never receive a gradient" in p
                   for p in probs), stage
    a = _args("--stage", "S-T", *base, "--i-know-this-is-the-control-arm")
    assert not any("--tac-goal-cond" in p for p in preflight(a))


def test_preflight_accepts_the_port_where_it_belongs(tmp_path):
    from train_v6_staged import preflight
    # S-T with t1 live: the production shape — no port-related problem.
    a = _args("--stage", "S-T", "--out", str(tmp_path), "--tac-goal-cond",
              "--init-from", "x", "--dry-run")
    assert not any("tac-goal-cond" in p or "inert-port" in p
                   for p in preflight(a))
    # S-S carries the GEOMETRY with layer_tac frozen: not inert, not refused.
    a = _args("--stage", "S-S", "--out", str(tmp_path), "--tac-goal-cond",
              "--init-from", "x", "--dry-run")
    assert not any("tac-goal-cond" in p or "inert-port" in p
                   for p in preflight(a))


def test_build_stack_from_args_wires_the_flag():
    from train_v6_staged import build_stack_from_args
    a = _args("--stage", "S-T", "--out", "unused", "--dry-run",
              "--tac-goal-cond",
              "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
              "--enc-dim", "32", "--enc-depth", "1", "--enc-heads", "2",
              "--readout-grid", "4", "--readout-dim", "8",
              "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
              "--window", "4", "--horizons", "1", "2",
              "--d-tac", "32", "--d-str", "16", "--d-goal-embed", "16",
              "--adapter-hidden", "32", "--sigreg-slices", "8")
    s = build_stack_from_args(a)
    assert s.cfg.tac_goal_cond is True and s.cond_tac_dyn is not None
    assert s.cond_tac_dyn.in_features == 16


# =========================================================================== #
# 7. the chain — S-T launches WITH the port; the geometry carries
# =========================================================================== #

def _chain():
    import v6_chain as C
    return C


def _ccfg(root, **kw):
    C = _chain()
    c = C.ChainConfig(root=str(root).replace("\\", "/"), dry=True, tiny=True,
                      dry_steps=1)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _seed(plan, c):
    """Each step's own `config.json`, as the trainer writes it at startup.

    ⚠️ REQUIRED SINCE E1 (2026-08-17): `trainer_argv` now CARRIES the ancestor's
    model geometry out of that file and REFUSES rather than emit a
    geometry-free line — MEASURED, the old line built 87.93 M against a
    336.54 M checkpoint. A fixture with no ancestor record is asking for an
    argv the chain will not write.
    """
    C = _chain()
    for s in plan:
        Path(s.out).mkdir(parents=True, exist_ok=True)
        args = C.parse_argv_geometry(
            list(C.TINY_GEOMETRY) + ["--n-candidates", str(c.n_candidates)]
        ) | {"tac_goal_cond": bool(s.tac_goal_cond), "selector": s.selector}
        (Path(s.out) / "config.json").write_text(json.dumps({"args": args}))


def test_the_default_ladder_carries_the_port_on_every_stage_after_S_W(
        tmp_path):
    """F-1's operational half: S-T must not launch without the port, and once
    S-T has it, S-S/S-J must carry the geometry — all from the ONE place a v6
    launch line is constructed."""
    C = _chain()
    c = _ccfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    for s in plan:
        av = C.trainer_argv(s, c, plan)
        if s.key == "S-W":
            assert "--tac-goal-cond" not in av    # the live run: untouchable
        else:
            assert "--tac-goal-cond" in av, s.key


def test_the_port_can_be_dropped_only_as_a_DECLARED_decision(tmp_path):
    C = _chain()
    c = _ccfg(tmp_path, tac_goal_cond=False)
    plan = C.build_plan(c)
    _seed(plan, c)
    for s in plan:
        assert "--tac-goal-cond" not in C.trainer_argv(s, c, plan), s.key
    # and the CLI spelling reaches the config
    a = C.build_parser().parse_args(["plan", "--no-tac-goal-cond"])
    assert C._cfg_from_args(a).tac_goal_cond is False


def test_the_S_T_argv_parses_through_the_trainer_parser(tmp_path):
    """The chain and the trainer must agree on the flag's spelling — parsed,
    not eyeballed (the dry ladder's write_dry_predecessor does exactly this
    round-trip)."""
    C = _chain()
    c = _ccfg(tmp_path)
    plan = C.build_plan(c)
    _seed(plan, c)
    st = C.step_by_key(plan, "S-T")
    a = _args(*C.trainer_argv(st, c, plan))
    assert a.tac_goal_cond is True and a.stage == "S-T"


def test_geometry_carry_treats_S_T_as_the_ports_introduction(tmp_path):
    C = _chain()
    c = _ccfg(tmp_path)
    plan = C.build_plan(c)
    sw = C.step_by_key(plan, "S-W")
    Path(sw.out).mkdir(parents=True, exist_ok=True)
    (Path(sw.out) / "config.json").write_text(
        json.dumps({"args": {"selector": "none", "tac_goal_cond": False}}))
    r = C.assert_geometry_carry(C.step_by_key(plan, "S-T"), plan)
    assert r["ok"] and r["introduces_port"] == "cond_tac_dyn."


def test_geometry_carry_REFUSES_dropping_the_port_above_S_T(tmp_path):
    """S-S launched without the flag against an S-T lineage that trained the
    port: the checkpoint's cond_tac_dyn.* keys would be unexpected and
    --init-from fatal — refused from a JSON read, before anything is built."""
    from dataclasses import replace
    C = _chain()
    c = _ccfg(tmp_path)
    plan = list(C.build_plan(c))
    st = C.step_by_key(tuple(plan), "S-T")
    Path(st.out).mkdir(parents=True, exist_ok=True)
    (Path(st.out) / "config.json").write_text(
        json.dumps({"args": {"selector": "none", "tac_goal_cond": True}}))
    ss = C.step_by_key(tuple(plan), "S-S")
    plan[plan.index(ss)] = replace(ss, tac_goal_cond=False)
    with pytest.raises(SystemExit, match="tac-goal-cond|cond_tac_dyn"):
        C.assert_geometry_carry(plan[[p.key for p in plan].index("S-S")],
                                tuple(plan))


def test_geometry_carry_REFUSES_introducing_the_port_above_S_T(tmp_path):
    """A pre-F-1 S-T lineage cannot grow the port at S-S: S-S may introduce
    nothing, and arriving above S-T without the port means the downlink never
    trained in its stage — a mis-ordered ladder, not an introduction."""
    C = _chain()
    c = _ccfg(tmp_path, tac_goal_cond=False)
    plan = list(C.build_plan(c))
    st = C.step_by_key(tuple(plan), "S-T")
    Path(st.out).mkdir(parents=True, exist_ok=True)
    (Path(st.out) / "config.json").write_text(
        json.dumps({"args": {"selector": "none", "tac_goal_cond": False}}))
    from dataclasses import replace
    ss = C.step_by_key(tuple(plan), "S-S")
    plan[plan.index(ss)] = replace(ss, tac_goal_cond=True)
    with pytest.raises(SystemExit, match="introduced at S-T|may introduce"):
        C.assert_geometry_carry(plan[[p.key for p in plan].index("S-S")],
                                tuple(plan))
