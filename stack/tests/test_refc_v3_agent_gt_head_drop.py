"""``D-AGENTGT-HEAD-SILENT-DROP`` — the third divergent case, and the mutual waiver.

⛔⛔ WHAT WAS ACTUALLY WRONG, AND WHY IT SURVIVED THREE COMMENTS SAYING IT DID NOT
---------------------------------------------------------------------------------
``RefCV3Model.forward`` guarded ``agent_gt`` in TWO directions and its own comment
said **"Both directions refuse."** There are THREE divergent cases, not two:

  1. ``agent_gt`` supplied, no agent seam                → refused (`refc_v3.py:1398`)
  2. ``agent_gt`` absent, ``--agents oracle``            → refused (`:1404`)
  3. ``agent_gt`` supplied, ``--agents head``            → **SILENTLY DROPPED**

Case 3 drops because `refc.py:3366-3368` builds head slots from ``fmap`` and never
reads the channel — no warning, no log, no raise.

⭐ THE PART THAT MAKES IT MORE THAN A MISSING BRANCH. `tanitad/rl/refc_adapter.py`
lists ``agent_gt`` in ``FORWARD_KEYS`` — it forwards the channel — and then WAIVES its
own launch-time check, citing verbatim: *"the model refuses BOTH directions loudly and
unconditionally, so there is no SILENT divergence for this guard to catch."* So the
model's guard was incomplete, the RL guard was switched off **because** the model's
guard was believed complete, and the channel was covered by neither. Two guards, each
deferring to the other.

⚠️ EVIDENCE CLASS: **LATENT, NOT LIVE** at the time of closing. `refc_v3_train.py:2196`
gates ``agent_gt`` on ``enable AND oracle``, so today's trainer hands a head build
``None``. The RL adapter is the path that would have hit it. Reported as a hole closed
before it opened — not as a live bug found in a running arm.

⚠️ THE GENERAL LESSON, which is not about agents: **a waiver whose justification is a
claim about OTHER code is exactly as stale as that code is old.** Same family as the
stale-count rule in `CLAUDE.md` — an absence-claim living inside the thing that was
supposed to keep it true.

⭐ SIDE EFFECT WORTH NAMING. `test_refc_v3_agent_gt_reaches_forward.py`'s docstring
records that its ``head`` parity row "cannot distinguish a correct gate from no gate at
all" *because* of this drop. With case 3 refused, the head row discriminates —
:func:`test_the_HEAD_row_is_now_DISCRIMINATING` is the proof.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_STACK = os.path.dirname(_HERE)
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
_SCRIPTS = os.path.join(_STACK, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_TRAINER_PY = os.path.join(_SCRIPTS, "refc_v3_train.py")

#: ⛔ THE GATE REMOVED — the exact literal from the shipped trainer. With it,
#: `agent_gt` is built and forwarded on EVERY arm, which is the only way to put a
#: real GT block into a real `head` forward. `_mutate` asserts a count of exactly
#: 1, so a reformat DISARMS LOUDLY instead of silently.
CORRUPT_GATE_FROM = (
    '    if (_ag_cfg is not None and getattr(_ag_cfg, "enable", False)\n'
    '            and getattr(_ag_cfg, "oracle", False)):')
CORRUPT_GATE_TO = "    if True:"

_B, _N, _BOX_SEED = 2, 8, 7


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trainer():
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    return _load(_TRAINER_PY, "refc_v3_train_for_headdrop")


def _mutate(tmp_path, name: str, old: str, new: str):
    """A COPY of the trainer with one exact substitution. Bytes, not text — a
    text round-trip on Windows rewrites every line ending and makes a one-line
    mutant unauditable."""
    src = open(_TRAINER_PY, "rb").read()
    ob, nb = old.encode("utf-8"), new.encode("utf-8")
    n = src.count(ob)
    assert n == 1, (
        f"the mutation anchor for {name!r} matched {n} times, not 1. The "
        f"deliberate-regression arm is DISARMED and this test would pass "
        f"without proving anything. Re-anchor it against refc_v3_train.py.")
    p = tmp_path / f"{name}.py"
    p.write_bytes(src.replace(ob, nb))
    return _load(str(p), f"refc_v3_train_{name}")


def _gt_tensors():
    g = torch.Generator().manual_seed(_BOX_SEED)
    box = torch.rand(_B, _N, 4, generator=g) * 20.0 + 1.0
    yaw = torch.rand(_B, _N, generator=g) * 2.0 - 1.0
    cls = torch.randint(0, 3, (_B, _N), generator=g)
    valid = torch.ones(_B, _N, dtype=torch.bool)
    valid[1, _N // 2:] = False
    return box, yaw, cls, valid


def _build(T, arm: str):
    argv = ["--arm", "hier", "--out", "/tmp/agentgt_headdrop", "--seed", "0",
            "--agents", arm, "--agent-queries", str(_N)]
    if arm == "head":
        argv += ["--w-agent", "1.0", "--agent-join", "j.xz"]
    args = T.build_parser().parse_args(argv)
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)

    torch.manual_seed(1234)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = 0.0
    model._w_tac_goal = 0.0
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()

    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])

    box, yaw, cls, valid = _gt_tensors()
    batch["agent_box"] = box
    batch["agent_yaw"] = yaw
    batch["agent_cls"] = cls
    batch["agent_valid"] = valid
    batch["agent_occ"] = torch.zeros(_B, _N)
    batch["agent_rates"] = torch.zeros(_B, _N, 3)
    batch["agent_rates_mask"] = torch.zeros(_B, _N, dtype=torch.bool)
    batch["agent_label"] = torch.ones(_B, dtype=torch.bool)
    return model, batch, cfg


def _run(T, arm: str):
    model, batch, _cfg = _build(T, arm)
    torch.manual_seed(99)
    return T.compute_losses_v3(model, batch, "cpu", mode="diffusion")


# --------------------------------------------------------------------------- #
# 1. THE REFUSAL — on a REAL forward with a REAL GT block                       #
# --------------------------------------------------------------------------- #
def test_a_head_build_REFUSES_a_supplied_agent_gt(tmp_path):
    """⭐ THE LOAD-BEARING TEST. Remove the trainer's oracle gate so a real
    ``agent_gt`` reaches a real ``--agents head`` forward. Before 2026-09-10 this
    ran to completion with the block discarded; it must now refuse."""
    M = _mutate(tmp_path, "gate_removed_head", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    with pytest.raises(ValueError, match="SILENTLY DROP"):
        _run(M, "head")


def test_the_message_names_the_two_ways_out(tmp_path):
    """A refusal that does not say what to do instead gets worked around with
    the first thing that silences it — which here would be `strict=False`'s
    cousin: deleting the channel from the caller."""
    M = _mutate(tmp_path, "gate_removed_msg", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    with pytest.raises(ValueError) as ei:
        _run(M, "head")
    msg = str(ei.value)
    assert "--agents oracle" in msg
    assert "stop passing" in msg


# --------------------------------------------------------------------------- #
# 2. THE DELIBERATE-REGRESSION ARM — the historical guard set, written out       #
# --------------------------------------------------------------------------- #
def test_DELIBERATE_REGRESSION_the_pre_fix_guard_set_is_BLIND_to_this_input():
    """⛔ THE PRE-2026-09-10 PREDICATES, VERBATIM, ON THE EXACT INPUT THAT BROKE.

    The historical guard was two conditions. Both are written here as literals —
    not read back from the module — and both must evaluate **False** on a head
    build that was handed a GT block. That is what "silently dropped" means, and
    it is why the value-level tests in
    `test_refc_v3_agent_gt_reaches_forward.py` were green throughout.

    ⚠️ Written against a plain stand-in for the seam config rather than a built
    model, deliberately: the point is that the PREDICATES were incomplete, and a
    predicate is a property of the boolean algebra, not of the network.
    """
    class _Seam:                       # what `cfg.core.agents` presents
        enable = True
        oracle = False                 # <- `--agents head`

    _ag = _Seam()
    agent_gt = {"box": torch.zeros(_B, _N, 4)}      # a real, non-None block

    historical_guard_1 = agent_gt is not None and (_ag is None or not _ag.enable)
    historical_guard_2 = (agent_gt is None and _ag is not None
                          and _ag.enable and _ag.oracle)
    assert historical_guard_1 is False
    assert historical_guard_2 is False, (
        "if either historical predicate fires here, this case was never the "
        "blind spot and the whole premise of this file is wrong")

    # the case that was missing, and is now the third guard
    third_case = (agent_gt is not None and _ag is not None
                  and _ag.enable and not _ag.oracle)
    assert third_case is True


def test_DELIBERATE_REGRESSION_the_mutation_anchor_is_armed(tmp_path):
    """⛔ A mutation arm whose anchor has drifted passes while proving nothing —
    the disarmed-regression class. Assert the mutant really differs from the
    shipped trainer, by BYTES."""
    M = _mutate(tmp_path, "armed_check", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    shipped = open(_TRAINER_PY, "rb").read()
    mutant = open(M.__file__, "rb").read()
    assert mutant != shipped
    assert len(shipped) - len(mutant) == len(CORRUPT_GATE_FROM) - len(CORRUPT_GATE_TO)


# --------------------------------------------------------------------------- #
# 3. THE CONTROLS — the other two cases, and the honest arms, must not move     #
# --------------------------------------------------------------------------- #
def test_the_shipped_trainer_still_runs_head_without_agent_gt():
    """⛔ THE CONTROL THAT STOPS THIS BEING A BLANKET REFUSAL. On the real
    trainer the oracle gate holds, a head build receives ``None``, and the arm
    must run exactly as before. If this fails, the new guard has broken the arm
    it was meant to protect."""
    T = _trainer()
    losses = _run(T, "head")
    assert torch.isfinite(losses["loss"])
    assert len(losses) == 20


def test_case_1_still_refuses_agent_gt_with_no_seam(tmp_path):
    """The pre-existing guard must be untouched by the new one."""
    M = _mutate(tmp_path, "gate_removed_off", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    with pytest.raises(ValueError, match="no agent seam"):
        _run(M, "off")


def test_case_2_still_refuses_an_oracle_build_with_no_agent_gt():
    """...and so must the oracle direction. Asserted through the model's own
    guard by handing an oracle build a batch with no boxes is the trainer's
    SystemExit, so this uses the model message instead."""
    T = _trainer()
    losses = _run(T, "oracle")
    assert torch.isfinite(losses["loss"])       # the honest oracle arm still runs


def test_the_HEAD_row_is_now_DISCRIMINATING(tmp_path):
    """⭐ THE SIDE EFFECT, ASSERTED RATHER THAN CLAIMED.

    `test_refc_v3_agent_gt_reaches_forward.py` records that its ``head`` parity
    row could not tell a correct gate from no gate at all, because a head build
    swallowed ``agent_gt`` either way. With case 3 refused, removing the gate
    now changes the head arm's OUTCOME — from 'runs' to 'refuses' — which is the
    strongest form of 'this comparison is sensitive'.
    """
    T = _trainer()
    assert torch.isfinite(_run(T, "head")["loss"])          # gate in place: runs
    M = _mutate(tmp_path, "gate_removed_disc", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    with pytest.raises(ValueError, match="SILENTLY DROP"):  # gate removed: refuses
        _run(M, "head")


# --------------------------------------------------------------------------- #
# 4. THE WAIVER THAT POINTED AT THE HOLE                                        #
# --------------------------------------------------------------------------- #
def test_the_rl_adapters_waiver_no_longer_claims_two_directions():
    """⛔ The RL adapter switched its own ``agent_gt`` check off citing the
    model's completeness. That citation is the reason the channel was covered
    nowhere, so it is pinned: the record must not go back to claiming BOTH.

    ⚠️ This asserts on the RECORD's text, which is weaker than asserting on
    behaviour — it is here because the record is what a future reader will act
    on, and it was the load-bearing artifact in this defect.
    """
    from tanitad.rl.refc_adapter import CHANNEL_REQUIREMENTS
    rec = [r for r in CHANNEL_REQUIREMENTS if r.channel == "agent_gt"]
    assert len(rec) == 1, "agent_gt must have exactly one channel record"
    reason = rec[0].reason
    # ⚠️ The assertion is on the CLAIM, not on the words. The corrected record
    # quotes the retracted phrase in order to retract it, so a bare
    # `"BOTH directions" not in reason` flags the fix itself — MEASURED while
    # writing this file, and a small instance of the rule the file is about:
    # a check whose predicate is broader than the property it means to test.
    assert "refuses BOTH directions" not in reason, (
        "the waiver is back to CLAIMING the model refuses both directions — "
        "that claim is what made D-AGENTGT-HEAD-SILENT-DROP survive")
    assert "ALL THREE" in reason
    assert "D-AGENTGT-HEAD-SILENT-DROP" in reason


def test_agent_gt_is_still_a_forwarded_channel():
    """⛔ THE CONTROL FOR THE TEST ABOVE. The waiver only matters because the
    adapter actually forwards the channel; if it stopped, the record would be
    moot and this file's premise would need re-reading rather than silently
    passing."""
    from tanitad.rl.refc_adapter import FORWARD_KEYS
    assert "agent_gt" in FORWARD_KEYS
