"""F-18 — the PERCEPTION agent-slot decoder: inert when off, isolated when on,
and joined to the label path that already exists.

⛔ WHY THIS FILE EXISTS. ``DIAGRAM_CONFORMANCE.md`` (2026-08-16) audited the
binding v6 diagram element by element. §4.2's first interpretation-head row —
*perception agent slots (bbox cx,cy,yaw,l,w · state v,yaw-rate,occluded · class
& size)* — read ⬜ **NOT BUILT**, with the status *"NEW — design here;
DETR-style slot decoder ~2–4 M params on spatial tokens"*, the note that the
LABEL side already exists (``scripts/build_obstacle_join.py``), and fix F-18.
It was the LAST unbuilt PERCEPTION cell.

WHAT THIS FILE PINS, and why each pin is load-bearing:

1. ⛔ **INERTNESS.** v6F S-W is resuming on Thor from a checkpoint of the
   incumbent architecture — **87,893,449 params / 405 keys** (config E:
   **336,542,025 / 573** at the PI-raised 350 M budget). The DEFAULT build's
   ``state_dict``, its RNG stream, AND its forward's OUTPUT KEY SET must be
   untouched, proved per tensor with ``torch.equal`` against a CONTENT-anchored
   pre-change revision of ``v6.py`` — never against HEAD (C75: while a module
   is being written HEAD moves, after which a HEAD comparison is a module
   compared with itself).
   ⚠️ The key-set half is not hypothetical: the first implementation returned
   ``interp_side``/``agent_slots`` unconditionally and
   ``test_v6_gstr_port.py::test_default_forward_is_bit_identical_and_emits_no_new_key``
   FAILED. The guard caught it; the keys are now conditional.

2. ⛔ **X3 UNDER A NEW EDGE.** This head is not planner-side — it is a
   PERCEPTION head, and its supervision is a PERCEPTION LABEL, which the
   diagram's header row forbids in ANY trunk loss. So it gets its own
   ``interp`` group, its own ``ISOLATION_MATRIX`` row ``("interp",)``, and its
   own probed edge ``perception_to_trunk`` — present ONLY when the head is
   built, because a probe over an absent module reports zero violations and has
   established nothing. And the edge must be able to FAIL: the mis-wired arm
   ``isolate_interp_from_encoder=False`` is tested to raise, which is what
   separates a check from a comment (the C13 "guard that cannot fail" family).

3. ⛔ **VISION-ONLY AT INFERENCE** (PI 2026-08-03). Tested two ways: the
   decoder's ``forward`` signature admits exactly one tensor and no keyword, so
   there is no door for a privileged input; and — the detector that could
   actually fail — moving ``v0`` and the actions leaves the slot output
   bit-identical while moving the FRAMES changes it.

4. ⛔ **ONE LABEL PATH.** Targets are built from the arrays
   ``build_obstacle_join.py`` writes and ``train_p8_occupancy.JoinFileReader``
   reads, through a real round-trip of a real join file — not from a
   re-implemented reader.

Every number in this file's literals is MEASURED (2026-08-16, this box, torch
CPU build at seed 0) — recompute on drift, never inherit.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import (  # noqa: E402
    EncoderConfig, PredictorConfig, ReadoutConfig)
from tanitad.data.bev_raster import ALL_CLASSES, agents_to_array  # noqa: E402
from tanitad.models.agent_slots import (  # noqa: E402
    AGENT_CLASSES, N_QUERIES_DEFAULT, PARAM_BAND, SLOT_FIELDS, SLOT_SLICES,
    SLOT_WIDTH, AgentSlotDecoder, hungarian, match_slots, slot_set_loss,
    targets_from_join, track_rates_from_join)
from tanitad.models.v6 import (  # noqa: E402
    ISOLATION_MATRIX, LADDER_UNTRAINED_GROUPS, MODULE_GROUPS, STAGE_GROUPS,
    STAGES, IsolationViolation, V6Config, V6Stack, apply_stage_freeze)
from train_v6_staged import STAGE_MAY_INTRODUCE, load_stage_init  # noqa: E402

#: MEASURED before this turn's edits (carried from `test_v6_gstr_port.py`'s
#: literals and re-verified on the edited module 2026-08-16): the counts the
#: DEFAULT build must not move. FULL is the geometry class of the live S-W
#: resume; CONFIG_E is the live run itself.
HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS = 611_293, 223
HEAD_FULL_PARAMS, HEAD_FULL_KEYS = 87_893_449, 405
CONFIG_E_PARAMS, CONFIG_E_KEYS = 336_542_025, 573

#: MEASURED 2026-08-16 at the §6 PRODUCTION geometry: 16 queries x d_model 256
#: x depth 3 x 8 heads, over the 16 readout cells of width 128. Inside the
#: pre-registered 2–4 M band, which is the point of the number.
PROD_SLOT_PARAMS = 3_207_445
PROD_SLOT_KW = dict(n_queries=16, d_model=256, depth=3, n_heads=8)

_SMALL_KW = dict(
    d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32, f_hidden_str=32,
    f_blocks=1, aux_hidden=16, sigreg_slices=8, plan_steps=6, dt=0.1,
    op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6), hz_op=10.0, hz_tac=2.0,
    hz_str=0.5, d_plan_feat=16, emission_hidden=16, d_goal_embed=128,
    n_candidates=8)
#: the slot geometry small enough for a toy stack (the §6 band is asserted
#: separately, at the production geometry).
_SLOT_SMALL = dict(agent_slots=True, slot_hidden=32, slot_heads=4,
                   slot_depth=1, n_slot_queries=6)


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


# =========================================================================== #
# 1. ⛔ INERTNESS — the live resume is untouchable
# =========================================================================== #

def test_the_head_defaults_off_and_is_absent():
    c = V6Config()
    assert c.agent_slots is False
    assert c.slot_src == "cells"
    assert c.isolate_interp_from_encoder is True
    s = _build(_small())
    assert s.agent_slots is None
    assert not any(k.startswith("agent_slots.") for k in s.state_dict())


def test_default_counts_are_the_MEASURED_head_counts():
    s = _build(_small())
    assert (_n(s), len(s.state_dict())) == (HEAD_SMALL_PARAMS, HEAD_SMALL_KEYS)


def test_the_interp_group_exists_and_is_EMPTY_by_default():
    """A new MODULE_GROUPS entry must partition to zero when its module is not
    built — otherwise adding the group would itself have moved the model."""
    s = _build(_small())
    rep = s.param_report()
    assert "interp" in MODULE_GROUPS
    assert rep["per_group"]["interp"] == 0
    assert sum(rep["per_group"].values()) == rep["total"]


@pytest.mark.slow
def test_default_FULL_config_counts_are_the_live_resume_counts():
    """87,893,449 / 405 — the numbers a broken strict resume would kill."""
    f = _build(V6Config())
    assert (_n(f), len(f.state_dict())) == (HEAD_FULL_PARAMS, HEAD_FULL_KEYS)


@pytest.mark.slow
def test_config_E_default_build_is_unchanged_and_within_its_budget():
    e = _build(_config_e())
    assert (_n(e), len(e.state_dict())) == (CONFIG_E_PARAMS, CONFIG_E_KEYS)
    rep = e.assert_param_budget()
    assert rep["within_budget"] and rep["budget"] == 350_000_000


#: The marker separating the pre-F-18 architecture from this one.
#: ⚠️ NOT ``"agent_slots"``: ``n_agent_slots`` (the categorical goal-arg
#: cardinality) has existed since the factored-goal build, so that string
#: matches EVERY revision and the history walk would find no pre-change one —
#: a marker that always matches turns this proof into a skip. Measured the
#: difference and used a string that exists only after F-18.
_PRE_CHANGE_MARKER = "isolate_interp_from_encoder"
_V6_REL = "stack/tanitad/models/v6.py"


def _pre_change_module():
    """``tanitad.models.v6`` as it was BEFORE the slot decoder existed —
    CONTENT-anchored, never HEAD (C75). Returns None when git cannot answer;
    the caller skips, because a skipped test is honest and a self-comparison
    dressed as a real one is not."""
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
                continue                      # already carries the head
            src, ref = r.stdout, sha
            break
        else:
            return None
    except Exception:
        return None
    sp = str(_STACK / "scripts")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    tmp = Path(tempfile.mkdtemp()) / "v6_pre_agent_slots.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("v6_pre_agent_slots", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v6_pre_agent_slots"] = mod
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


def test_default_forward_is_bit_identical_and_grows_no_output_key():
    """Weights identical is necessary, not sufficient. ⚠️ THE KEY-SET HALF IS
    THE ONE THAT ACTUALLY FAILED during this build: returning ``interp_side``
    and ``agent_slots`` unconditionally (even as ``None``) tripped
    ``test_v6_gstr_port.py``'s twin of this assertion. The keys are now added
    only when the head exists."""
    head = _pre_change_module()
    if head is None:
        pytest.skip("git could not produce a pre-change revision of v6.py")
    torch.manual_seed(0)
    old = head.V6Stack(head.V6Config(**{**_sub_cfgs(), **_SMALL_KW}))
    new = _build(_small())
    b = new.synthetic_batch(2)
    o_old, o_new = old.forward(**b), new.forward(**b)
    assert set(o_old) == set(o_new), "the default forward grew an output key"
    assert "interp_side" not in o_new and "agent_slots" not in o_new
    assert torch.equal(o_old["z_op_win"], o_new["z_op_win"])
    assert torch.equal(o_old["zhat_tac"], o_new["zhat_tac"])
    assert torch.equal(o_old["plan"]["waypoints"], o_new["plan"]["waypoints"])


def test_encode_window_default_path_is_unchanged_by_the_token_option():
    """``return_tokens`` was added to ``encode_window`` for the ``"tokens"``
    arm. The DEFAULT call must return exactly what it returned — same tensor,
    not merely the same shape."""
    s = _build(_small())
    f = s.synthetic_batch(2)["frames"]
    z = s.encode_window(f)
    z2, tok = s.encode_window(f, return_tokens=True)
    assert torch.equal(z, z2)
    assert tok.shape[:2] == f.shape[:2]
    assert tok.shape[2] == s.encoder.n_tokens
    assert tok.shape[3] == s.cfg.encoder.d_model


# =========================================================================== #
# 2. ⭐ TURNING IT ON — additive, grouped, introducible
# =========================================================================== #

@pytest.mark.parametrize("src", ["cells", "tokens"])
def test_turning_it_on_perturbs_NO_pre_existing_tensor(src):
    """The capacity-control discipline every gated lever here obeys: the ON
    arm is the OFF arm plus tensors, never the OFF arm re-initialised — which
    only holds because the head is constructed at the END of ``__init__``."""
    off = _build(_small())
    on = _build(_small(**_SLOT_SMALL, slot_src=src))
    so, sn = off.state_dict(), on.state_dict()
    new = set(sn) - set(so)
    assert new and all(k.startswith("agent_slots.") for k in new)
    for k in so:
        assert torch.equal(so[k], sn[k]), f"{k} moved when the head flipped on"
    assert _n(on) - _n(off) == on.agent_slots.n_params


def test_the_head_is_grouped_interp_and_the_matrix_row_says_so():
    """`interp`, NOT `aux` — and the distinction IS the X3 argument. `aux` MAY
    backprop into the encoder (O3/O6 are label-free trunk losses and that is
    their job); this head's supervision is a PERCEPTION LABEL, which the
    diagram's header row forbids in any trunk loss. Filed under `aux` the
    matrix would PERMIT the edge and the probe would pass over a violation."""
    s = _build(_small(**_SLOT_SMALL))
    assert ISOLATION_MATRIX["interp"] == ("interp",)
    assert ISOLATION_MATRIX["aux"] == ("encoder", "readout", "aux")
    for n, _ in s.named_parameters():
        if n.startswith("agent_slots."):
            assert s.group_of(n) == "interp"
    assert s.param_report()["per_group"]["interp"] == s.agent_slots.n_params


def test_no_ladder_stage_TRAINS_it_and_that_is_deliberate():
    """⛔ The v6 batch carries frames/actions/poses/future_* and NO agent
    labels (the episode contract), so a stage that listed `interp` as trainable
    would report a module as training while it receives exactly zero gradient —
    the lie ``V6LossWeights.for_stage`` zeroes its planner terms to avoid.

    ⛔ **CORRECTED 2026-08-16 — S-J WAS NOT AN EXCEPTION, IT WAS THE DEFECT.**
    This test used to assert ``"interp" in STAGE_GROUPS["S-J"]`` and
    ``requires_grad is True`` at S-J, excused as "the sole exception BY
    DEFINITION (it is ``MODULE_GROUPS``)". That excuse only held while the group
    was EMPTY. With ``agent_slots=True`` — the build this very file exercises —
    ``apply_stage_freeze(s, "S-J")`` unfroze **62 tensors / 3,207,445 parameters
    at the production geometry** that the S-J loss reaches **exactly 0** of
    (MEASURED, ``…/2026-08-16-evidence-and-flake/``). "By definition" was a
    restatement of the alias, not a reason. v6.py now declares
    ``LADDER_UNTRAINED_GROUPS`` and derives S-J as ``MODULE_GROUPS`` minus it,
    so the answer is **False at all four stages** — which is what the file title
    ("no ladder stage TRAINS it") always claimed.
    """
    for stage in STAGES:
        assert "interp" not in STAGE_GROUPS[stage], stage
    assert "interp" in LADDER_UNTRAINED_GROUPS
    assert "interp" in MODULE_GROUPS, \
        "it must still be a GROUP — apply_stage_freeze partitions over it"
    s = _build(_small(**_SLOT_SMALL))
    for stage in STAGES:
        rep = apply_stage_freeze(s, stage)
        assert s.agent_slots.head.weight.requires_grad is False, stage
        assert rep["per_group"]["interp"]["trainable"] == 0, stage
        # non-vacuity: the head IS built here, so "0 trainable" is a statement
        # about real parameters and not about an empty group.
        assert rep["per_group"]["interp"]["frozen"] == s.agent_slots.n_params


def test_S_T_may_INTRODUCE_it_over_an_S_W_checkpoint(tmp_path):
    """The transition that must work: an S-W checkpoint never carried the head,
    and S-T's allowlist admits exactly its keys and nothing else."""
    assert "agent_slots." in STAGE_MAY_INTRODUCE["S-T"]
    p = tmp_path / "sw.pt"
    torch.save({"stack": _build(_small()).state_dict(), "step": 30000,
                "config": {"stage": "S-W"}}, p)
    rep = load_stage_init(_build(_small(**_SLOT_SMALL)), p, stage="S-T")
    assert rep["missing_keys"] == [] and rep["unexpected_keys"] == []
    assert rep["introduced_keys"]
    assert all(k.startswith("agent_slots.")
               for k in rep["introduced_keys"])


def test_a_PARTIALLY_present_head_is_fatal_not_an_introduction(tmp_path):
    """⛔ An introduction must be WHOLE. Half a head is a geometry mismatch
    wearing an allowance's clothes."""
    sd = _build(_small(**_SLOT_SMALL)).state_dict()
    sd.pop("agent_slots.queries")
    p = tmp_path / "half.pt"
    torch.save({"stack": sd, "step": 1, "config": {"stage": "S-W"}}, p)
    with pytest.raises(SystemExit, match="agent_slots"):
        load_stage_init(_build(_small(**_SLOT_SMALL)), p, stage="S-T")


# =========================================================================== #
# 3. ⛔ X3 — the new edge, and its ability to FAIL
# =========================================================================== #

@pytest.mark.parametrize("src", ["cells", "tokens"])
def test_X3_holds_with_the_head_on_and_the_new_edge_is_NON_VACUOUS(src):
    s = _build(_small(**_SLOT_SMALL, slot_src=src))
    rep = s.assert_isolation(batch_size=2, strict=True)
    assert rep["pass"] and rep["violations"] == {}
    assert rep["n_violations"] == {"planner_to_encoder": 0,
                                   "tactical_to_below": 0,
                                   "strategic_to_below": 0,
                                   "perception_to_trunk": 0}
    # ...and the probe actually looked at something. A probe over an empty
    # parameter set reports zero violations and has established nothing.
    assert all(v > 0 for v in rep["n_probed"].values()), rep["n_probed"]
    assert rep["config"]["agent_slots"] is True
    assert rep["config"]["slot_src"] == src


def test_the_new_edge_is_ABSENT_when_the_head_is():
    """A probe key that is structurally always-zero is the vacuous pass this
    module's comments spend most of their length preventing — and the exact
    three-key dict is what `test_v6_ladder_edges` pins."""
    rep = _build(_small()).assert_isolation(batch_size=2, strict=True)
    assert set(rep["n_violations"]) == {"planner_to_encoder",
                                        "tactical_to_below",
                                        "strategic_to_below"}
    assert "perception_to_trunk" not in rep["n_probed"]


def test_the_MIS_WIRED_arm_makes_the_edge_FAIL():
    """⛔ THE CHECK THAT MAKES IT A CHECK. ``isolate_interp_from_encoder=False``
    lets the PERCEPTION-LABEL gradient reach the encoder — forbidden by the
    diagram's header row — and the probe must SAY SO, naming encoder/readout
    parameters. Without this, ``perception_to_trunk`` would be a guard that
    cannot fail (C13)."""
    bad = _build(_small(**_SLOT_SMALL, isolate_interp_from_encoder=False))
    with pytest.raises(IsolationViolation, match="X3 gradient-isolation"):
        bad.assert_isolation(batch_size=2)
    rep = bad.assert_isolation(batch_size=2, strict=False)
    assert rep["pass"] is False
    assert rep["n_violations"]["perception_to_trunk"] > 0
    named = rep["violations"]["perception_to_trunk"]
    assert any(n.startswith(("encoder.", "readout.")) for n in named), named
    # the OTHER three edges must stay clean — one mis-wire, one lever.
    for k in ("planner_to_encoder", "tactical_to_below",
              "strategic_to_below"):
        assert rep["n_violations"][k] == 0


def test_the_declared_interp_surface_is_TOTAL():
    """Every `interp` parameter must be reachable from the DECLARED
    ``interp_side``. A field added to the head without appending to the
    declaration escapes the isolation probe — the `intent_proj` defect moved
    into the audit."""
    s = _build(_small(**_SLOT_SMALL))
    out = s.forward(**s.synthetic_batch(2))
    interp = list(s.group_parameters("interp"))
    assert interp
    live = V6Stack._live_edges(V6Stack._probe_scalar(out["interp_side"]),
                               interp)
    unreached = {n for n, _ in interp} - set(live)
    assert not unreached, f"unreachable from interp_side: {sorted(unreached)}"


# =========================================================================== #
# 4. ⛔ VISION-ONLY AT INFERENCE (PI 2026-08-03)
# =========================================================================== #

def test_the_decoder_signature_admits_exactly_one_tensor():
    """The structural half: there is no keyword through which ego state, a
    goal, or a situation-classifier output could arrive. The signature IS the
    audit — the same discipline as ``GoalHead``'s refusal of an undeclared
    conditioning path (no ``**kwargs``)."""
    sig = inspect.signature(AgentSlotDecoder.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert [p.name for p in params] == ["memory"]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    assert not any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                   for p in sig.parameters.values())


def test_the_slots_move_with_the_FRAMES_and_not_with_v0_or_the_ACTIONS():
    """⭐ THE DETECTOR THAT COULD FAIL. Vision-only is not a comment: hold the
    frames and move every privileged channel the forward accepts — the slot
    output must be bit-identical. Then move the FRAMES and it must change, or
    the detector is reading a constant."""
    s = _build(_small(**_SLOT_SMALL))
    b = s.synthetic_batch(2, seed=3)
    base = s.forward(**b)["agent_slots"]

    priv = dict(b)
    priv["v0"] = b["v0"] * 3.0 + 7.0
    priv["actions"] = b["actions"] + 5.0
    moved = s.forward(**priv)["agent_slots"]
    for k in ("box", "presence_logit", "cls_logits", "rates", "occ_logit"):
        assert torch.equal(base[k], moved[k]), \
            f"{k} moved with a PRIVILEGED input — vision-only is violated"
    # the privileged move DID reach the parts it legitimately drives, so the
    # comparison above is not vacuous:
    assert not torch.equal(s.forward(**b)["plan"]["waypoints"],
                           s.forward(**priv)["plan"]["waypoints"])

    seen = dict(b)
    seen["frames"] = b["frames"] + 0.5
    assert not torch.equal(base["box"], s.forward(**seen)["agent_slots"]["box"])


# =========================================================================== #
# 5. ⛔ ONE LABEL PATH — the join that already exists
# =========================================================================== #

def test_the_class_vocabulary_IS_the_label_side_s(#
        ):
    """Imported, never re-listed. 10 classes, all dynamic agents — which is
    also why `TRAFFIC_LIGHT_REACT`'s agent slot can never come from
    `obstacle.offline`."""
    assert AGENT_CLASSES == tuple(ALL_CLASSES)
    assert len(AGENT_CLASSES) == 10
    assert SLOT_SLICES["cls"].stop - SLOT_SLICES["cls"].start == 10


def test_the_emitted_fields_are_the_DIAGRAM_CELL():
    """§4.2's cell, channel by channel: bbox cx,cy,yaw,l,w · state v,yaw-rate,
    occluded · class. Plus DETR's ∅ logit, which the cell cannot mention
    because it describes an agent and not the set."""
    names = [n for n, _ in SLOT_FIELDS]
    assert names == ["presence", "cls", "cx", "cy", "l", "w", "yaw_sin",
                     "yaw_cos", "v_rel_x", "v_rel_y", "yaw_rate_rel",
                     "occluded"]
    assert SLOT_WIDTH == 21
    assert list(SLOT_SLICES) == names


def _join_records(n_frames: int = 3) -> list[dict]:
    """Join LINES in exactly ``build_obstacle_join.py``'s schema."""
    recs = []
    for f in range(n_frames):
        recs.append({
            "clip_id": "clip_abc", "frame_idx": f, "t_s": round(f * 0.1007, 6),
            "agents": [
                {"cx": 12.0 + 0.5 * f, "cy": 1.25, "yaw": 0.05, "l": 4.6,
                 "w": 1.9, "occ": 0, "track_id": "t1", "cls": "automobile"},
                {"cx": 30.0 - 1.0 * f, "cy": -3.5, "yaw": 3.10, "l": 11.0,
                 "w": 2.5, "occ": 1, "track_id": "t2", "cls": "bus"}]})
    return recs


def test_targets_round_trip_through_the_REAL_JoinFileReader(tmp_path):
    """⛔ NOT a re-implemented reader: the file is written in the join schema,
    read back with ``train_p8_occupancy.JoinFileReader`` (the consumer
    ``build_obstacle_join.py`` self-verifies against), and the target tensors
    are built from what THAT returns."""
    from train_p8_occupancy import (JoinFileReader,  # noqa: PLC0415
                                    episode_uid_of_clip)
    p = tmp_path / "join.jsonl"
    recs = _join_records()
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    rd = JoinFileReader(p)
    assert rd.has_classes and rd.has_occlusion_flags
    uid = episode_uid_of_clip("clip_abc")
    ag = rd.lookup(uid, 1)
    assert ag is not None and ag.shape == (2, 6)
    cls = rd.lookup_classes(uid, 1)

    rates, mask = track_rates_from_join(recs[0], recs[1], recs[2])
    t = targets_from_join(ag, cls, rates, mask, n_pad=4)
    assert t["valid"][0].tolist() == [True, True, False, False]
    # (cx, cy, l, w) — the join's own column order, via agents_to_array
    assert t["box"][0, 0].tolist() == pytest.approx([12.5, 1.25, 4.6, 1.9])
    assert t["yaw"][0, 1].item() == pytest.approx(3.10, abs=1e-5)
    assert t["cls"][0].tolist() == [AGENT_CLASSES.index("automobile"),
                                    AGENT_CLASSES.index("bus"), -1, -1]
    assert t["occ"][0].tolist() == [0.0, 1.0, -1.0, -1.0]
    # v_rel_x = central difference over 2 x 0.1007 s
    assert t["rates"][0, 0, 0].item() == pytest.approx(1.0 / 0.2014, rel=1e-4)
    assert t["rates"][0, 1, 0].item() == pytest.approx(-2.0 / 0.2014, rel=1e-4)
    assert t["rates_mask"][0].tolist() == [True, True, False, False]


def test_a_missing_frame_is_NO_LABEL_and_an_empty_list_is_CLEAR():
    """⛔ Two different states, and conflating them is how "road clear" gets
    invented (obstacle-join doc §4). ``None`` must RAISE; ``[0, 6]`` must be a
    valid, empty, labelled set."""
    with pytest.raises(ValueError, match="NO_LABEL"):
        targets_from_join(None)
    t = targets_from_join(agents_to_array([]), n_pad=3)
    assert t["valid"].sum().item() == 0
    assert t["box"].shape == (1, 3, 4)


def test_an_unseen_track_has_its_rate_MASKED_not_zeroed():
    """⛔ Zero is a legitimate rate (a stationary car). Filling an unobserved
    one with zero would teach the head that unseen means still."""
    recs = _join_records()
    recs[0]["agents"] = [recs[0]["agents"][0]]      # t2 not in the past...
    recs[2]["agents"] = [recs[2]["agents"][0]]      # ...nor in the future
    rates, mask = track_rates_from_join(recs[0], recs[1], recs[2])
    assert mask.tolist() == [True, False]
    assert rates[1].tolist() == [0.0, 0.0, 0.0]     # present but MASKED


def test_the_yaw_rate_is_WRAP_SAFE():
    """A car driving straight through ±π must not produce a ~63 rad/s spike."""
    mk = lambda t, y: {"t_s": t, "agents": [                    # noqa: E731
        {"cx": 5.0, "cy": 0.0, "yaw": y, "l": 4.0, "w": 2.0, "track_id": "x"}]}
    rates, mask = track_rates_from_join(mk(0.0, 3.13), mk(0.1, -3.13),
                                        mk(0.2, -3.10))
    assert mask.tolist() == [True]
    assert abs(float(rates[0, 2])) < 1.0        # naive wrap gives ~ -62.6


def test_a_join_without_classes_yields_minus_one_and_says_so():
    """An older join file (no ``cls`` column) must not silently become class
    0 — ``JoinFileReader.has_classes`` reports the absence and the target
    carries -1, which the loss then counts as zero class items."""
    t = targets_from_join(agents_to_array(
        [{"cx": 1.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0}]),
        classes=None)
    assert t["cls"].tolist() == [[-1]]
    assert t["occ"].tolist() == [[-1.0]]        # no occ key -> "no information"


# =========================================================================== #
# 6. THE MATCHER — exact, and pinned against the reference implementation
# =========================================================================== #

def test_hungarian_equals_scipy_on_random_and_tied_matrices():
    """⚠️ The duplication (scipy is NOT a core dependency) is admissible only
    WITH this equivalence proof — the same contract
    ``bev_raster.yaw_from_quaternion`` carries against
    ``physicalai.quaternion_yaw``. Compared on the OPTIMAL COST, not on the
    index tuple: ties have several optimal assignments and asserting one of
    them would pin an implementation detail rather than the answer."""
    lsa = pytest.importorskip("scipy.optimize").linear_sum_assignment
    rng = np.random.default_rng(0)
    for trial in range(200):
        n, m = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        c = rng.normal(size=(n, m))
        if trial % 5 == 0:
            c = np.round(c)                     # force ties
        r1, k1 = hungarian(c)
        r2, k2 = lsa(c)
        assert len(r1) == min(n, m)
        assert sorted(r1.tolist()) == sorted(set(r1.tolist()))
        assert sorted(k1.tolist()) == sorted(set(k1.tolist()))
        assert c[r1, k1].sum() == pytest.approx(c[r2, k2].sum(), abs=1e-9)


def test_hungarian_refuses_a_non_finite_cost():
    with pytest.raises(ValueError, match="non-finite"):
        hungarian(np.array([[1.0, np.nan]]))


def test_an_OVER_FULL_frame_drops_the_FARTHEST_and_COUNTS_it():
    """⛔ Declared policy, not incidental: the near field is the one the plan
    acts on, and a silent drop would flatter the head exactly on the crowded
    frames where driving is hardest."""
    d = AgentSlotDecoder(8, 4, n_queries=2, d_model=32, depth=1, n_heads=4,
                         enforce_band=False)
    pred = d(torch.randn(1, 4, 8))
    ag = agents_to_array([
        {"cx": 5.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0},
        {"cx": 50.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0},   # farthest
        {"cx": 9.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    t = targets_from_join(ag)
    m = match_slots(pred, t)
    assert m["n_target"] == [3] and m["n_dropped"] == [1]
    assert sorted(m["cols"][0].tolist()) == [0, 2]      # the 50 m one dropped
    assert slot_set_loss(pred, t, match=m)["n"]["dropped"] == 1


# =========================================================================== #
# 7. THE SET LOSS — per term, per unit, with its counts
# =========================================================================== #

def _toy_pred_and_targets():
    d = AgentSlotDecoder(8, 4, n_queries=6, d_model=32, depth=1, n_heads=4,
                         enforce_band=False)
    torch.manual_seed(1)
    pred = d(torch.randn(1, 4, 8))
    recs = _join_records()
    rates, mask = track_rates_from_join(recs[0], recs[1], recs[2])
    t = targets_from_join(agents_to_array(recs[1]["agents"]),
                          [a["cls"] for a in recs[1]["agents"]], rates, mask,
                          n_pad=3)
    return d, pred, t


def test_every_term_is_returned_SEPARATELY_with_its_count():
    """⛔ A pooled score hides exactly the trade-off one wants to see — the
    ADE-only failure in a detector's costume. Units differ per term BY
    CONSTRUCTION (metres · nats · m/s · rad/s), so they are never summed
    without their declared weights, and a term computed over ZERO items is
    reported as 0.0 WITH its count rather than dropped."""
    _, pred, t = _toy_pred_and_targets()
    out = slot_set_loss(pred, t)
    for k in ("presence", "cls", "centre", "size", "yaw", "rates", "occ"):
        assert f"loss_{k}" in out and torch.isfinite(out[f"loss_{k}"])
    assert out["n"]["matched"] == 2 and out["n"]["target"] == 2
    assert out["n"]["cls"] == 2 and out["n"]["rates"] == 2
    assert torch.isfinite(out["total"])
    assert set(out["_weights"]) == {"presence", "cls", "centre", "size",
                                    "yaw", "rates", "occ"}


def test_an_empty_frame_still_scores_presence_and_nothing_else():
    """A labelled-clear frame is a real training signal (every slot must say
    ∅) and every other term must report a ZERO count, not a zero loss dressed
    as a measurement."""
    d = AgentSlotDecoder(8, 4, n_queries=6, d_model=32, depth=1, n_heads=4,
                         enforce_band=False)
    pred = d(torch.randn(1, 4, 8))
    out = slot_set_loss(pred, targets_from_join(agents_to_array([]), n_pad=2))
    assert out["n"]["matched"] == 0 and out["n"]["target"] == 0
    assert float(out["loss_presence"]) > 0.0
    for k in ("cls", "centre", "size", "yaw", "rates", "occ"):
        assert out["n"][k] == 0 and float(out[f"loss_{k}"]) == 0.0


def test_the_yaw_term_is_a_metric_ON_THE_CIRCLE():
    """+π and −π are the SAME heading. A scalar yaw regression would score a
    2π error there; the unit (sin, cos) pair scores zero, which is the whole
    reason the field is a pair."""
    d = AgentSlotDecoder(8, 4, n_queries=1, d_model=32, depth=1, n_heads=4,
                         enforce_band=False)
    pred = d(torch.randn(1, 4, 8))
    with torch.no_grad():                        # point the slot at -pi
        pred["yaw_vec"][:] = torch.tensor([[[np.sin(-np.pi), np.cos(-np.pi)]]],
                                          dtype=pred["yaw_vec"].dtype)
    t = targets_from_join(agents_to_array(
        [{"cx": 5.0, "cy": 0.0, "yaw": float(np.pi), "l": 4.0, "w": 2.0}]))
    assert float(slot_set_loss(pred, t)["loss_yaw"]) == pytest.approx(0.0,
                                                                      abs=1e-6)


def test_the_loss_gradient_reaches_ONLY_interp_parameters():
    """The X3 claim, executed end to end through the REAL loss rather than the
    probe's synthetic reduction: backprop the set loss on a full stack and read
    which parameters received gradient."""
    s = _build(_small(**_SLOT_SMALL))
    for p in s.parameters():
        p.requires_grad_(True)
    out = s.forward(**s.synthetic_batch(2))
    t = targets_from_join(agents_to_array(
        [{"cx": 8.0, "cy": 1.0, "yaw": 0.2, "l": 4.4, "w": 1.9, "occ": 0}]),
        ["automobile"])
    t = {k: (v.expand(2, *v.shape[1:]).contiguous() if v.shape[0] == 1 else v)
         for k, v in t.items()}
    slot_set_loss(out["agent_slots"], t)["total"].backward()
    touched = {n for n, p in s.named_parameters() if p.grad is not None
               and float(p.grad.abs().max()) > 0.0}
    assert touched, "the loss reached nothing at all"
    outside = {n for n in touched if s.group_of(n) != "interp"}
    assert not outside, f"a PERCEPTION LABEL reached the trunk: {sorted(outside)}"


# =========================================================================== #
# 8. THE §6 PARAMETER BAND, and the refusals
# =========================================================================== #

def test_production_geometry_is_inside_the_preregistered_band():
    """§6: *"DETR-style slot decoder ~2–4 M params on spatial tokens"*. The
    literal is MEASURED, not arithmetic (the selector's +41,089 estimate was
    never realised; the implementation cost +33,801)."""
    d = AgentSlotDecoder(d_memory=128, n_memory=16, **PROD_SLOT_KW)
    assert d.n_params == PROD_SLOT_PARAMS
    assert PARAM_BAND[0] <= d.n_params <= PARAM_BAND[1]


def test_the_band_check_actually_FIRES():
    """A band that never refuses is decoration. V6Stack passes
    ``enforce_band=False`` because it is instantiated at toy geometries by
    dozens of tests — so the check's aliveness is proved HERE."""
    with pytest.raises(ValueError, match="outside the §6"):
        AgentSlotDecoder(8, 4, n_queries=4, d_model=32, depth=1, n_heads=4)


def test_the_two_agent_slot_counts_must_agree_when_both_levers_are_on():
    """⛔ `n_agent_slots` is the CARDINALITY of the categorical `agent_slot`
    arg that four g_tac tokens index; this decoder is what would POPULATE that
    set. Two cardinalities make an emitted index that refers to nothing — the
    type error the categorical channel exists to remove."""
    with pytest.raises(ValueError, match="n_agent_slots"):
        _small(**_SLOT_SMALL, goal_cat_args=True, n_agent_slots=8)
    # ...and they are allowed to agree
    _small(**{**_SLOT_SMALL, "n_slot_queries": 8}, goal_cat_args=True,
           n_agent_slots=8)


def test_slot_src_is_a_declared_arm_with_a_refusal():
    with pytest.raises(ValueError, match="slot_src must be cells|tokens"):
        _small(slot_src="bev")


def test_the_decoder_refuses_a_memory_of_the_wrong_length():
    """The positional table is per-token, so a different memory length is a
    geometry mismatch and not a resize — the ``SpatialGridReadout`` firewall
    argument applied one module along."""
    d = AgentSlotDecoder(8, 4, n_queries=4, d_model=32, depth=1, n_heads=4,
                         enforce_band=False)
    with pytest.raises(ValueError, match="geometry mismatch"):
        d(torch.randn(1, 5, 8))


def test_the_defaults_are_the_declared_placeholders():
    """⚠️ ``n_slot_queries`` is a DECLARED PLACEHOLDER, not a fitted value: the
    right number is the join's measured per-frame agent-count distribution,
    which is UNMEASURED (no join file lives in the repo). This test exists so
    the placeholder cannot quietly become a claim."""
    assert V6Config().n_slot_queries == N_QUERIES_DEFAULT == 16
