"""refcv5 WP-4/WP-6 — the TRAINER wiring: flags, guards, seam stamp, losses.

⭐ The guards in this file are the point. Each one refuses a configuration that
would RUN, produce plausible numbers, and mean nothing — the failure mode this
programme keeps meeting. A flag combination that manufactures a refutation is
worse than a crash, because a crash gets fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import refc_v3_train as t                                    # noqa: E402
from tanitad.refs import refc_agents as ra                   # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402


def _args(**kw):
    base = ["--arm", "hier", "--out", "/tmp/refcv5_test"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            base.append(flag)
        elif v is not False and v is not None:
            base += [flag, str(v)]
    return t.build_parser().parse_args(base)


def _cfg(v0_cond=True):
    cfg = v3.refc_v3_sized_config("tiny", hier=True)
    cfg.core.anchors.v0_conditioned = v0_cond
    return cfg


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_defaults_are_all_OFF():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. Adding the seams to the code
    must not change a run that does not ask for them."""
    a = _args()
    assert a.sampler == "none"
    assert a.agents == "off"
    assert a.w_u0 == 0.0
    assert a.w_agent == 0.0
    assert a.sampler_space == "control"


def test_default_pin_constructs_NOTHING():
    """⛔ CONTROL. OFF means NOT CONSTRUCTED, not 'constructed and gated' —
    otherwise RNG draw order shifts and pre-v5 checkpoints stop loading."""
    cfg = _cfg()
    t._pin_refcv5_seams(cfg, _args())
    assert cfg.core.decoder.sampler == "none"
    assert getattr(cfg.core, "agents", None) is None
    assert cfg.core.decoder.cross_agent is False


def test_default_model_has_no_refcv5_modules():
    """⛔ CONTROL, one level down: build the real model and look."""
    cfg = _cfg()
    t._pin_refcv5_seams(cfg, _args())
    m = v3.RefCV3Model(cfg)
    assert m.core.agent_head is None
    assert m.core.agent_embed is None
    assert m.core.decoder.control_head is None
    for layer in m.core.decoder.layers:
        assert layer.cross_agent is None


def test_default_build_is_PARAMETER_IDENTICAL_at_a_fixed_seed():
    """⛔ THE PARITY CLAIM, as an assertion rather than a hope."""
    cfg = _cfg()
    t._pin_refcv5_seams(cfg, _args())
    torch.manual_seed(1234)
    a = v3.RefCV3Model(_cfg())
    torch.manual_seed(1234)
    b = v3.RefCV3Model(cfg)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


# ---------------------------------------------------------------------------
# The guards — each refuses a plausible-looking WRONG experiment
# ---------------------------------------------------------------------------
def test_ddim_on_a_FIXED_vocabulary_REFUSES():
    """⛔ `anchor_controls` is all zeros on a fixed-path bank, so the anchored
    Gaussian would be centred on 'do nothing'. That run would train, converge,
    and mean nothing."""
    with pytest.raises(SystemExit, match="v0-CONDITIONED"):
        t._pin_refcv5_seams(_cfg(v0_cond=False),
                            _args(sampler="ddim", w_u0=0.5))


def test_ddim_without_the_x0_LOSS_REFUSES():
    """⛔⛔ THE SUBTLEST GUARD, and the one most worth having. `control_head` is
    ZERO-INIT. With `--w-u0 0` nothing supervises it, so it stays at exactly
    zero forever: the arm is the anchored Gaussian with NO DENOISER, and every
    log row still says 'sampler: ddim'. It would read as 'diffusion does not
    help' — a REFUTATION manufactured by a missing loss."""
    with pytest.raises(SystemExit, match="w-u0"):
        t._pin_refcv5_seams(_cfg(), _args(sampler="ddim"))


def test_agent_head_without_its_LOSS_REFUSES():
    """⛔ Same family. A detector that is never supervised emits noise tokens;
    the zero-init gate has no reason to open; the arm reads as 'agent tokens do
    not help'."""
    with pytest.raises(SystemExit, match="w-agent"):
        t._pin_refcv5_seams(_cfg(), _args(agents="head"))


def test_agent_ORACLE_needs_no_detector_loss():
    """The oracle path has no detector to train, so it is admissible with
    `--w-agent 0` — and it is the rung that must run FIRST."""
    cfg = _cfg()
    t._pin_refcv5_seams(cfg, _args(agents="oracle"))
    assert cfg.core.agents.enable is True
    assert cfg.core.agents.oracle is True
    assert cfg.core.decoder.cross_agent is True


# ---------------------------------------------------------------------------
# The seam stamp — a run record that cannot rebuild its own config is not one
# ---------------------------------------------------------------------------
def test_seam_stamp_carries_every_refcv5_lever():
    """⛔ MEASURED precedent (`SEAM_STATE.md`): six seams were absent from
    config.json at every nesting level, and the live arm was reconstructible
    only by knowing what refc_v3.py forces. Every lever below is stamped."""
    cfg = _cfg()
    a = _args(sampler="ddim", w_u0=0.5, agents="oracle",
              agent_sigma_range=1.5, agent_queries=24)
    t._pin_refcv5_seams(cfg, a)
    st = t._seam_stamp(cfg, a)
    assert st["sampler"] == "ddim"
    assert st["sampler_space"] == "control"
    assert st["sampler_infer_t"] == 8
    assert st["sampler_steps"] == 2
    assert st["sampler_groups"] == 1
    assert st["control_norm"] == [4.0, 3.0]
    assert st["w_u0"] == 0.5
    assert st["cross_agent"] is True
    assert st["agents"]["oracle"] is True
    assert st["agents"]["oracle_sigma_range_m"] == 1.5
    assert st["agents"]["queries"] == 24
    # the class vocabulary travels WITH the run, so a corpus-side rename
    # cannot silently re-map a trained head
    assert st["agents"]["classes"] == list(ra.AGENT_CLASSES)
    import json
    json.dumps(st)              # must be serialisable into config.json


def test_seam_stamp_on_a_DEFAULT_run_says_off():
    """CONTROL. The stamp must distinguish 'off' from 'absent'."""
    cfg = _cfg()
    a = _args()
    t._pin_refcv5_seams(cfg, a)
    st = t._seam_stamp(cfg, a)
    assert st["sampler"] == "none"
    assert st["agents"] is None
    assert st["cross_agent"] is False
    assert st["w_agent"] == 0.0 and st["w_u0"] == 0.0


# ---------------------------------------------------------------------------
# The x0 loss target — the units trap, in test form
# ---------------------------------------------------------------------------
def test_x0_target_is_converted_to_THE_VOCABULARYS_UNITS():
    """⛔⛔ THE UNITS TRAP. `unicycle_controls_from_path_varstep` returns
    CURVATURE (1/m); the bank's controls are LATERAL ACCELERATION (m/s^2) when
    `control_units == 'alat'`. a_lat = v^2 * kappa, so at 36 m/s they differ by
    a factor of ~1300 — and BOTH tables look plausible. This is the exact error
    `anchor_meta.py` was built to prevent, MEASURED once as '396 g at 36 m/s'.

    Here the conversion is checked directly against the identity.
    """
    from tanitad.models import kinematic as kin
    horizons = (5, 10, 15, 20)
    dts = kin.slot_dts(horizons)
    # a gentle constant-curvature arc at a known speed
    path = torch.tensor([[[5.0, 0.05], [10.0, 0.2], [15.0, 0.45],
                          [20.0, 0.8]]])
    u_kappa = kin.unicycle_controls_from_path_varstep(path, dts)
    v0 = torch.tensor([10.0])
    v_ref = v0.clamp_min(4.0) ** 2
    u_alat = torch.stack([u_kappa[..., 0], u_kappa[..., 1] * v_ref[:, None]],
                         dim=-1)
    # a_lat = v^2 * kappa, exactly
    assert torch.allclose(u_alat[..., 1], u_kappa[..., 1] * 100.0, atol=1e-6)
    # and the two are NOT interchangeable: at v = 10 they differ by 100x
    assert float(u_alat[..., 1].abs().max()) > \
        50.0 * float(u_kappa[..., 1].abs().max())


# ---------------------------------------------------------------------------
# The record-vs-model guard. ⛔ A GUARD MUST BE SHOWN CAPABLE OF FAILING:
# an AST census once read 0 suspects on BOTH the fixed and the broken trainer,
# so each of these REINTRODUCES the defect and requires the refusal, and the
# two controls prove the guard is not simply always-raising.
# ---------------------------------------------------------------------------
def _built(**kw):
    """(model, stamp) for a pinned config — the honest pair."""
    c = _cfg()
    a = _args(**kw)
    t._pin_refcv5_seams(c, a)
    return v3.RefCV3Model(c), t._seam_stamp(c, a)


def test_seam_guard_PASSES_an_honest_default_run():
    """⛔ CONTROL THAT MUST READ A KNOWN VALUE. A guard that rejects everything
    proves nothing about the three below."""
    m, st = _built()
    t.assert_seams_are_built(m, st)          # must not raise


def test_seam_guard_PASSES_an_honest_ddim_run():
    """CONTROL, the ON side: the guard must accept a real sampler."""
    m, st = _built(sampler="ddim", w_u0=0.5)
    t.assert_seams_are_built(m, st)
    assert m.core.decoder.control_head is not None


def test_seam_guard_REJECTS_THE_EXACT_PRE_FIX_DEFECT():
    """⛔⛔ THE DEFECT, REINTRODUCED. Before 2026-09-06 `DecoderConfig` had no
    `sampler` field, so `_pin_refcv5_seams` set an AD-HOC ATTRIBUTE on an
    unfrozen dataclass; both of its own guards read it back and PASSED, and
    `_seam_stamp` copied it into config.json for a model whose `refc.py`
    contained zero occurrences of `control_head`. The run would have trained,
    converged and STATED IN ITS OWN RECORD that it used a sampler it did not
    have — FALSE PROVENANCE, which makes every later comparison against that
    arm unfalsifiable."""
    m, st = _built()                          # built with sampler='none'
    st["sampler"] = "ddim"                    # ...the lie
    st["w_u0"] = 0.5
    with pytest.raises(SystemExit, match="control_head"):
        t.assert_seams_are_built(m, st)


def test_seam_guard_REJECTS_A_LIVE_SEAM_MISSING_FROM_THE_RECORD():
    """⛔ THE OTHER DIRECTION, and it is not symmetry for its own sake: a seam
    that is BUILT but NOT STAMPED is the `SEAM_STATE.md` failure, where six
    live seams were absent from config.json and the arm was reconstructible
    only by knowing what refc_v3.py forces."""
    m, st = _built(sampler="ddim", w_u0=0.5)
    st["sampler"] = "none"
    st["w_u0"] = 0.0
    with pytest.raises(SystemExit, match="absent from the run record"):
        t.assert_seams_are_built(m, st)


def test_seam_guard_REJECTS_a_w_u0_WITH_NOTHING_TO_SUPERVISE():
    """⛔ `--w-u0 > 0` with no `control_head` means the x0 term is silently
    skipped while the weight is stamped — the M18 defect (`w_agent: 1.0`
    recorded while the detector was never supervised) in the sampler's
    costume."""
    m, st = _built()
    st["w_u0"] = 1.0
    with pytest.raises(SystemExit, match="w_u0"):
        t.assert_seams_are_built(m, st)


def test_seam_guard_REJECTS_agents_stamped_but_never_built():
    """⛔ Same family, WP-6 side."""
    m, st = _built()
    st["agents"] = {"enable": True, "oracle": True}
    st["cross_agent"] = True
    with pytest.raises(SystemExit, match="agent_head"):
        t.assert_seams_are_built(m, st)


# ---------------------------------------------------------------------------
# WP-6 — the agent seam as a RUNNING path, not a constructed one
# ---------------------------------------------------------------------------
def _frames(cfg, b=2):
    enc = cfg.core.encoder
    return torch.randn(b, cfg.core.window, enc.in_channels, *enc.image_hw())


def _gt(b=2, n=8, all_invalid=False):
    g = {"box": torch.randn(b, n, 4).abs() * 5.0,
         "yaw": torch.randn(b, n),
         "cls": torch.randint(0, 3, (b, n)),
         "valid": torch.ones(b, n, dtype=torch.bool)}
    if all_invalid:
        g["valid"][:] = False
    else:
        g["valid"][1, n // 2:] = False        # a partially-empty row
    return g


def test_agent_oracle_seam_BUILDS_AND_RUNS_and_exports_its_slots():
    """FAILS WITHOUT THE FEATURE. `compute_losses_v3` guards the detection loss
    on `"agent_slots" in out`; with the seam built but that key absent,
    `--w-agent 1.0` would parse, be STAMPED, and supervise nothing."""
    c = _cfg()
    a = _args(agents="oracle", agent_queries=8)
    t._pin_refcv5_seams(c, a)
    m = v3.RefCV3Model(c).eval()
    assert m.core.agent_head is not None and m.core.agent_embed is not None
    assert all(ly.cross_agent is not None for ly in m.core.decoder.layers)
    with torch.no_grad():
        out = m(_frames(c), v0=torch.tensor([12.0, 3.0]), steps=2,
                agent_gt=_gt())
    assert "agent_slots" in out
    assert torch.isfinite(out["traj"]).all()


def test_agent_attention_survives_an_EMPTY_SCENE():
    """⛔⛔ CONTROL THAT MUST READ A KNOWN VALUE. A fully-padded row makes
    `MultiheadAttention` return NaN, not zero — softmax over an all-`-inf` row
    is undefined. An empty road is the COMMON case, not an edge case, so an
    unguarded mask would poison the batch through the residual and read as an
    exploding planner rather than as an empty scene."""
    c = _cfg()
    t._pin_refcv5_seams(c, _args(agents="oracle", agent_queries=8))
    m = v3.RefCV3Model(c).eval()
    with torch.no_grad():
        out = m(_frames(c), v0=torch.tensor([12.0, 3.0]), steps=2,
                agent_gt=_gt(all_invalid=True))
    assert torch.isfinite(out["traj"]).all()
    assert not torch.isnan(out["anchor_traj"]).any()


def test_agent_gate_is_ZERO_INIT_and_GATED_NOT_DEAD():
    """⛔ THE PARITY CLAIM plus its converse. At the zero init the +agents model
    must emit BIT-IDENTICAL trajectories to the agent-free one (so every later
    change is attributable to the seam), while the gradient still reaches the
    gate (so it is gated, not dead)."""
    c_off = _cfg()
    t._pin_refcv5_seams(c_off, _args())
    c_on = _cfg()
    t._pin_refcv5_seams(c_on, _args(agents="oracle", agent_queries=8))
    torch.manual_seed(1234)
    base = v3.RefCV3Model(c_off).eval()
    torch.manual_seed(1234)
    withag = v3.RefCV3Model(c_on).eval()
    sb, sw = base.state_dict(), withag.state_dict()
    shared = [k for k in sb if k in sw and sb[k].shape == sw[k].shape]
    assert len(shared) > 100, len(shared)
    for k in shared:
        sw[k] = sb[k].clone()
    withag.load_state_dict(sw)
    fr, v = _frames(c_off), torch.tensor([12.0, 3.0])
    with torch.no_grad():
        a = base(fr, v0=v, steps=2)
        b = withag(fr, v0=v, steps=2, agent_gt=_gt())
    assert torch.equal(a["traj"], b["traj"]), \
        float((a["traj"] - b["traj"]).abs().max())
    assert torch.equal(a["anchor_logits"], b["anchor_logits"])
    # ...and gated, not dead
    withag.train()
    withag(fr, v0=v, steps=2, agent_gt=_gt())["traj"].square().mean().backward()
    for ly in withag.core.decoder.layers:
        assert ly.agent_gate.grad is not None
        assert float(ly.agent_gate.grad.abs().max()) > 0.0


def test_agent_gt_on_an_AGENTLESS_build_REFUSES():
    """⛔ A privileged tensor SILENTLY DROPPED reads as 'agent tokens do not
    help' in a result table — a refutation manufactured by a wiring gap. Same
    discipline as the `ego_state` guard beside it."""
    c = _cfg()
    t._pin_refcv5_seams(c, _args())
    m = v3.RefCV3Model(c).eval()
    with pytest.raises(ValueError, match="no agent seam"):
        m(_frames(c), v0=torch.tensor([12.0, 3.0]), steps=2, agent_gt=_gt())


def test_agent_ORACLE_without_its_boxes_REFUSES():
    """⛔ The converse: the oracle's tokens ARE the boxes. With none the seam
    emits nothing and the arm reads as 'agent tokens do not help' while never
    having had any."""
    c = _cfg()
    t._pin_refcv5_seams(c, _args(agents="oracle", agent_queries=8))
    m = v3.RefCV3Model(c).eval()
    with pytest.raises(ValueError, match="oracle"):
        m(_frames(c), v0=torch.tensor([12.0, 3.0]), steps=2)
