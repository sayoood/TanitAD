"""`--init-from` MUST let S-T introduce its selector, and refuse everything else.

THE DEFECT THIS PINS — MEASURED 2026-08-16, and it would have killed the S-T
launch on its first command::

    S-T selector="goal" -> RuntimeError: Missing key(s) in state_dict:
                           "cand_score.cand_bias", "cand_score.log_tau", ...
    S-T selector="mlp"  -> RuntimeError: Missing key(s) in state_dict:
                           "cand_score.fc1.weight", ...

`load_stage_init` loaded with torch's ``strict=True``, so a stage that
LEGITIMATELY ADDS a module was refused. The guard was right in spirit and blind
in practice: it could not distinguish

  * "the TRUNK is missing"           -> fatal. The stage would train on a random
    encoder while its log looked healthy — exactly what its docstring warns of;
  * "the PLANNER'S NEW HEAD is missing" -> expected. S-T is *where the selector
    is built*, and it is supposed to start from a fresh initialisation.

The fix is an EXPLICIT per-stage allowlist (`STAGE_MAY_INTRODUCE`), not a
relaxed ``strict`` — because ``strict=False`` would equally wave through a
missing ``emission.*`` and silently random-init the whole emission head.

A second, quieter defect went with it: under torch's ``strict=True`` a mismatch
RAISES, so the ``missing_keys`` / ``unexpected_keys`` this function returned
could only ever be empty. The report was structurally incapable of describing
the thing it was named for — the C13 family (a guard that cannot fail).
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from train_v6_staged import (  # noqa: E402
    STAGE_MAY_INTRODUCE, load_stage_init)


def _stack(selector="none"):
    torch.manual_seed(0)
    return V6Stack(V6Config(selector=selector))


@pytest.fixture(scope="module")
def sw_ckpt(tmp_path_factory):
    """A real S-W checkpoint: no selector, because S-W never builds one."""
    p = tmp_path_factory.mktemp("sw") / "ckpt.pt"
    torch.save({"stack": _stack("none").state_dict(), "step": 30000,
                "config": {"stage": "S-W"}}, p)
    return p


# ----------------------------------------------------------------- the data --

def test_only_s_t_may_introduce_and_only_its_declared_modules():
    """S-T introduces the selector (by design), since F-1
    (DIAGRAM_CONFORMANCE.md 2026-08-16) the zero-init g_str->P_T port
    `cond_tac_dyn.`, since the diffusion/MPC/fallback build (same day) the
    diffusion proposal generator `prop_diffusion.` (F-15, +437,954 params
    MEASURED at production geometry) and the fallback trigger's calibration
    buffers `fallback.` (F-17, 8 keys, 0 params), and since F-18 (same day)
    the PERCEPTION AGENT-SLOT DECODER `agent_slots.` (+3,207,445 params
    MEASURED at the §6 production geometry, inside its pre-registered 2-4 M
    band). The MPC refiner needs no entry — it holds no parameter and no
    buffer. Every other stage still introduces nothing.

    ⚠️ THE F-18 ENTRY SHARPENS WHAT AN ENTRY MEANS, so the reasoning is
    recorded here rather than rediscovered: an allowance is about KEYS
    ARRIVING, **not** about the stage optimising the module. No ladder stage
    trains `agent_slots.` — the v6 batch carries frames/actions/poses/future_*
    and no agent labels, so `interp` appears in NO `STAGE_GROUPS` entry AT ALL
    (it is in `LADDER_UNTRAINED_GROUPS`, and `stage_trainable_groups` raises if
    a stage declares it), and a frozen-trunk probe in the P8 idiom is what
    optimises it.
    ⛔ CORRECTED 2026-08-16: this read "`interp` appears in no `STAGE_GROUPS`
    entry but S-J's (which is MODULE_GROUPS by definition)". S-J was NOT an
    exception — it was a DEFECT, and "by definition" was a restatement of the
    `STAGE_GROUPS["S-J"] is MODULE_GROUPS` alias rather than a reason. With
    `agent_slots=True` it marked 3,207,445 parameters trainable under a loss
    that reaches 0 of them. `fallback.` (0 trainable params) was
    already this shape; F-18 makes it the second case, which is when a pattern
    should be written down.

    ⚠️ This pin is EXACT on purpose: growing the allowance must fail here
    first and be extended consciously, which is what happened for each entry
    after `cand_score.`."""
    assert STAGE_MAY_INTRODUCE["S-T"] == (
        "cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.",
        "agent_slots.")
    assert STAGE_MAY_INTRODUCE["S-W"] == ()
    assert STAGE_MAY_INTRODUCE["S-S"] == ()
    assert STAGE_MAY_INTRODUCE["S-J"] == ()


# -------------------------------------------------------- the transition it --
# ------------------------------------------------------- used to make fatal --

@pytest.mark.parametrize("selector", ["goal", "mlp"])
def test_s_t_can_now_init_from_an_s_w_ckpt_with_a_selector(sw_ckpt, selector):
    """THE ACTUAL DEFECT: both of these raised RuntimeError before the fix."""
    rep = load_stage_init(_stack(selector), sw_ckpt, stage="S-T")
    assert rep["missing_keys"] == []          # nothing fatal
    assert rep["unexpected_keys"] == []
    assert rep["introduced_keys"]             # ...and the head is NAMED
    assert all(k.startswith("cand_score.") for k in rep["introduced_keys"])
    assert rep["introduced_allowance"] == [
        "cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.",
        "agent_slots."]
    assert rep["init_step"] == 30000


def test_the_plain_transition_still_works(sw_ckpt):
    rep = load_stage_init(_stack("none"), sw_ckpt, stage="S-T")
    assert rep["missing_keys"] == rep["unexpected_keys"] == []
    assert rep["introduced_keys"] == []


# ---------------------------------------------------------- what stays fatal --

def test_a_stage_with_no_allowance_still_refuses_the_selector(sw_ckpt):
    """S-S may introduce nothing. Arriving there with an unbuilt selector means
    S-T never built one — a mis-ordered ladder, not an introduction."""
    with pytest.raises(SystemExit, match="not a valid predecessor"):
        load_stage_init(_stack("goal"), sw_ckpt, stage="S-S")


def test_a_missing_TRUNK_is_still_fatal(tmp_path):
    """The case the original guard existed for, and it must keep failing:
    training on a randomly-initialised encoder while the log looks healthy."""
    sd = _stack("none").state_dict()
    for k in [k for k in sd if k.startswith("encoder.")][:4]:
        sd.pop(k)
    p = tmp_path / "holed.pt"
    torch.save({"stack": sd, "step": 1, "config": {"stage": "S-W"}}, p)
    with pytest.raises(SystemExit, match="encoder"):
        load_stage_init(_stack("goal"), p, stage="S-T")


def test_unexpected_keys_are_fatal(tmp_path):
    """The ckpt carrying something this stack does not build IS a geometry
    mismatch, and no allowlist covers it."""
    sd = _stack("goal").state_dict()          # has cand_score.*
    p = tmp_path / "extra.pt"
    torch.save({"stack": sd, "step": 1, "config": {"stage": "S-W"}}, p)
    with pytest.raises(SystemExit, match="unexpected"):
        load_stage_init(_stack("none"), p, stage="S-T")


def test_a_PARTIALLY_present_selector_is_fatal_not_an_introduction(tmp_path):
    """⛔ An introduction must be WHOLE. A checkpoint carrying HALF a selector
    is a geometry mismatch wearing an allowance's clothes — waving the rest
    through would random-init part of a head and report success."""
    sd = _stack("goal").state_dict()
    sd.pop("cand_score.goal_point.weight")    # keep the rest of cand_score.*
    p = tmp_path / "half.pt"
    torch.save({"stack": sd, "step": 1, "config": {"stage": "S-W"}}, p)
    with pytest.raises(SystemExit, match="cand_score"):
        load_stage_init(_stack("goal"), p, stage="S-T")


def test_no_stage_means_no_allowance(sw_ckpt):
    """Callers that do not name the stage get the old, strictest behaviour —
    an allowance must be asked for, never inherited by default."""
    with pytest.raises(SystemExit, match="not a valid predecessor"):
        load_stage_init(_stack("goal"), sw_ckpt)


def test_the_trunk_md5_is_per_tensor_not_a_container_hash(sw_ckpt):
    """⚠️ RETRACTION_LOG C68: `torch.save` writes a zip whose bytes are NOT
    canonical, so a hash of the FILE proves the container, not the tensors.
    This md5 walks named_parameters and hashes tensor bytes — and it must be
    reproducible across two independent loads of the same checkpoint."""
    a = load_stage_init(_stack("goal"), sw_ckpt, stage="S-T")
    b = load_stage_init(_stack("mlp"), sw_ckpt, stage="S-T")
    # different selectors, IDENTICAL trunk -> identical trunk md5
    assert a["trunk_md5_after_load"] == b["trunk_md5_after_load"]
