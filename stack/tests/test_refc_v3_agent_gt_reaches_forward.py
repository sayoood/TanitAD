"""⛔⛔ ``--agents oracle`` MUST DELIVER ``agent_gt`` TO THE FORWARD — and this
file is the regression test the banked measurement deliberately was not.

⭐⭐ WHY THIS FILE EXISTS. On 2026-09-09 the WP-C oracle gate's 20-step smoke
reached ``RefCV3Model.forward`` with ``agent_gt=None`` under ``--agents
oracle``. The token ``agent_gt`` occurred **zero times** in
``refc_v3_train.py``: the trainer plumbed the agent join into the detection
**loss** only, which is sufficient for ``--agents head`` (whose tokens come
from the image) and leaves the ORACLE rung with **no supplier**, because on
that path the same tensors are the TOKENS rather than the targets.
``refc.py:3346-3347`` states that design in its own words.

⛔ **The model's own guard caught it, and that is the whole point.** Without
``refc_v3.py``'s ``ValueError`` the ~40 GPU-hour gate would have produced a
clean, separated, entirely FALSE *"the waypoint index does not help"* — a
manufactured negative, not an error. A guard is not a test, though: it fires at
run time, on a pod, after the config is pinned. This file asks the same
question at CI time, on a real forward.

⛔ **THE DISCRIMINATOR IS A VALUE, NOT A NON-``None``.** Asserting only that a
kwarg arrived would pass on a branch that built the dict from the wrong
tensors. ``degrade_boxes`` returns *"the inputs unchanged objects"* at
``sigma_range_m = 0`` and ``miss_rate = 0`` (``refc_agents.py:404-409``), so
the ORACLE arm's emitted ``agent_slots["box"]`` must be **BITWISE** the boxes
this test put in the batch. That is the assertion that has content.

⛔ **EVERY ARM BELOW GETS THE SAME BATCH, GT BOXES INCLUDED.** Injecting the
boxes only on the oracle arm would confound *"the branch did not fire"* with
*"the batch was different"*, and the ``off``/``head`` parity tests would pass
for the wrong reason.

⚠️ **A guard must be shown capable of failing** (the repo's AST-census lesson:
a probe read 0 suspects on BOTH the fixed and the broken trainer). So the
regression arm here does not describe the defect — it **REINTRODUCES** it, by
reverting the patch's one-line change to the forward call in a *copy* of the
trainer source, and requires the model's refusal. Two further mutants prove
each parity comparison is sensitive rather than vacuous.

⛔ Every expectation is a LITERAL. Nothing below is an expression over the code
under test.

MEASURED 2026-09-10, dev box, smoke width, CPU:
  * pre-patch  : ``ValueError: this build is `--agents oracle` but no agent_gt
    reached the forward``  · ``agent_gt`` token count **0**
  * post-patch : ``agent_gt`` reaches the forward with keys
    ``[box, cls, rates, valid, yaw]``; ``agent_slots["box"]`` bitwise equal to
    the batch's ``agent_box``; total loss 111.02839660644531, finite, backward
    OK · ``agent_gt`` token count **7**
  * ``--agents off``  : 193 params / 20 losses / 153 grads BITWISE IDENTICAL
  * ``--agents head`` : 275 params / 20 losses / 235 grads BITWISE IDENTICAL
Record: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-09-09-wpc-oracle-gate/raw/APPLICATION_RECORD.md``
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)
_SCRIPTS = os.path.join(_STACK, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_TRAINER_PY = os.path.join(_SCRIPTS, "refc_v3_train.py")

#: the banked application record; every failure message names it.
RECORD_DOC = ("TanitAD Research Lab/Architecture & Inference/Research/"
              "2026-09-09-wpc-oracle-gate/raw/APPLICATION_RECORD.md")

# --------------------------------------------------------------------------- #
# THE THREE MUTATIONS. Each is an EXACT LITERAL from the shipped source.        #
# `_mutate` asserts a count of exactly 1 and names the anchor when it drifts,   #
# so a reformat FAILS LOUD instead of silently disarming the arm.               #
# --------------------------------------------------------------------------- #

#: ⛔ THE PATCH, REVERTED — the pre-2026-09-10 forward call, verbatim. The dict
#: is still built; it simply never reaches the model, which is EXACTLY the
#: defect: a supplier that exists and is not connected.
REVERT_FROM = ("                v_max_ms=v_max_ms, v_max_valid=v_max_valid,\n"
               "                agent_gt=agent_gt)")
REVERT_TO = "                v_max_ms=v_max_ms, v_max_valid=v_max_valid)"

#: ⛔ IN-BRANCH VALUE CORRUPTION. The gate is untouched, so `off`/`head` cannot
#: see it; only the oracle's value-flow assertion can. Proves the ON comparison
#: is not vacuous.
CORRUPT_VALUE_FROM = '        agent_gt = {"box": batch["agent_box"].to(device),'
CORRUPT_VALUE_TO = \
    '        agent_gt = {"box": batch["agent_box"].to(device) * 2.0,'

#: ⛔ THE GATE REMOVED. The branch fires on every arm. Proves the OFF
#: comparison is not vacuous.
CORRUPT_GATE_FROM = (
    '    if (_ag_cfg is not None and getattr(_ag_cfg, "enable", False)\n'
    '            and getattr(_ag_cfg, "oracle", False)):')
CORRUPT_GATE_TO = "    if True:"


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trainer():
    """The LIVE ``refc_v3_train.py``, by path — the convention this repo uses
    for the trainer (``test_built_heads_receive_gradient.py``, ``refcv3_arm``).
    """
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    return _load(_TRAINER_PY, "refc_v3_train_for_agentgt")


def _mutate(tmp_path, name: str, old: str, new: str):
    r"""A COPY of the trainer with one exact substitution, imported as its own
    module.

    ⛔ Read and written as BYTES: a text-mode round-trip on Windows rewrites
    every line ending, which turns a one-line mutant into a 5,640-line diff and
    makes the arm impossible to audit (MEASURED 2026-09-10 — and ``grep -c``
    for a carriage return reported 0 while the file SIZE said otherwise, so the
    size was the artifact that settled it).
    """
    src = open(_TRAINER_PY, "rb").read()
    ob, nb = old.encode("utf-8"), new.encode("utf-8")
    n = src.count(ob)
    assert n == 1, (
        f"the mutation anchor for {name!r} matched {n} times, not 1. The "
        f"deliberate-regression arm is DISARMED, so this test would pass "
        f"without proving anything. Re-anchor it against the current "
        f"`refc_v3_train.py`; see {RECORD_DOC}.\nanchor:\n{old}")
    p = tmp_path / f"{name}.py"
    p.write_bytes(src.replace(ob, nb))
    return _load(str(p), f"refc_v3_train_{name}")


# --------------------------------------------------------------------------- #
# THE FIXTURE BATCH — one batch, every arm, GT boxes always present.            #
# --------------------------------------------------------------------------- #
#: LITERALS. B, N and the seed are fixed here so every shape assertion below is
#: a written-down number rather than a read-back of the code's own choice.
_B, _N, _BOX_SEED = 2, 8, 7


def _gt_tensors():
    g = torch.Generator().manual_seed(_BOX_SEED)
    box = torch.rand(_B, _N, 4, generator=g) * 20.0 + 1.0
    yaw = torch.rand(_B, _N, generator=g) * 2.0 - 1.0
    cls = torch.randint(0, 3, (_B, _N), generator=g)
    valid = torch.ones(_B, _N, dtype=torch.bool)
    valid[1, _N // 2:] = False              # a partially-empty row
    return box, yaw, cls, valid


def _build(T, arm: str):
    """(model, batch) for one ``--agents`` arm at smoke width, on CPU."""
    argv = ["--arm", "hier", "--out", "/tmp/agentgt_test", "--seed", "0",
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
    return model, batch


def _run(T, arm: str, spy: bool = False):
    """One ``compute_losses_v3`` on one arm. With ``spy``, records what the
    forward ACTUALLY received — a real forward, not a grep."""
    model, batch = _build(T, arm)
    seen: dict = {}
    real = T.v3.RefCV3Model.forward
    if spy:
        def _spy(self, *a, **kw):
            seen["n_calls"] = seen.get("n_calls", 0) + 1
            seen["agent_gt"] = kw.get("agent_gt")
            out = real(self, *a, **kw)
            seen["out"] = out
            return out
        T.v3.RefCV3Model.forward = _spy
    try:
        torch.manual_seed(99)
        losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    finally:
        if spy:
            T.v3.RefCV3Model.forward = real
    return model, losses, seen


# --------------------------------------------------------------------------- #
# 1. THE POSITIVE ASSERTION — on a REAL forward                                 #
# --------------------------------------------------------------------------- #
def test_agent_gt_REACHES_THE_FORWARD_under_agents_oracle():
    """⛔ THE CLAIM, as an assertion. Not *"the token appears in the file"* —
    the forward is run and asked what it was handed."""
    T = _trainer()
    _model, losses, seen = _run(T, "oracle", spy=True)
    assert seen["n_calls"] == 1
    ag = seen["agent_gt"]
    assert ag is not None, (
        "`--agents oracle` reached RefCV3Model.forward with agent_gt=None. "
        "The oracle's tokens ARE the ground-truth boxes; the arm would read "
        f"as 'agent tokens do not help' while never having had any. "
        f"{RECORD_DOC}")
    assert sorted(ag.keys()) == ["box", "cls", "rates", "valid", "yaw"]
    assert tuple(ag["box"].shape) == (2, 8, 4)
    assert tuple(ag["yaw"].shape) == (2, 8)
    assert tuple(ag["cls"].shape) == (2, 8)
    assert tuple(ag["valid"].shape) == (2, 8)
    assert tuple(ag["rates"].shape) == (2, 8, 3)
    assert ag["valid"].dtype is torch.bool
    assert ag["cls"].dtype is torch.int64
    # ...and it is THIS test's tensors, not some other batch's
    box, yaw, cls, valid = _gt_tensors()
    assert torch.equal(ag["box"], box)
    assert torch.equal(ag["yaw"], yaw)
    assert torch.equal(ag["cls"], cls)
    assert torch.equal(ag["valid"], valid)
    assert bool(torch.isfinite(losses["loss"]))


def test_the_ORACLES_EMITTED_TOKENS_ARE_THE_BATCHS_BOXES_BITWISE():
    """⛔⛔ THE ASSERTION WITH CONTENT — value flow THROUGH the seam, one level
    past the kwarg. ``degrade_boxes`` returns its inputs unchanged at
    ``sigma = 0 / miss = 0``, so a byte-for-byte match proves the batch's boxes
    are what the decoder cross-attends. A dict built from the wrong tensors
    would satisfy the test above and fail this one."""
    T = _trainer()
    _model, _losses, seen = _run(T, "oracle", spy=True)
    out = seen["out"]
    assert "agent_slots" in out
    slots = out["agent_slots"]
    assert tuple(slots["box"].shape) == (2, 8, 4)
    box, _yaw, _cls, valid = _gt_tensors()
    assert torch.equal(slots["box"].detach(), box)
    assert torch.equal(slots["valid"], valid)


def test_the_oracle_arm_TRAINS__loss_finite_and_backward_reaches_the_gate():
    """⛔ A forward that runs is not a run that learns. The trainer's own total
    must build a graph and put gradient on the agent gate — otherwise the gate
    stays at its zero init and the arm is agent-free while stamped +agents."""
    T = _trainer()
    model, losses, _seen = _run(T, "oracle")
    total = losses["loss"]
    assert total.requires_grad
    assert bool(torch.isfinite(total))
    total.backward()
    gates = [ly.agent_gate for ly in model.core.decoder.layers]
    assert len(gates) > 0
    assert all(g.grad is not None for g in gates)
    assert max(float(g.grad.abs().max()) for g in gates) > 0.0


# --------------------------------------------------------------------------- #
# 2. THE DELIBERATE-REGRESSION ARM — the defect, reintroduced                    #
# --------------------------------------------------------------------------- #
def test_REVERTING_THE_PATCH_REPRODUCES_THE_DEFECT(tmp_path):
    """⛔⛔ THE ARM THAT GIVES EVERY TEST ABOVE ITS MEANING. Revert the one-line
    change to the forward call — the dict is still built, it simply never
    arrives — and the model's own guard must refuse.

    This is the EXACT state the WP-C oracle gate hit on 2026-09-09, and it is
    the reason a ~40 GPU-hour negative would have been manufactured rather than
    measured had the guard not existed."""
    R = _mutate(tmp_path, "reverted", REVERT_FROM, REVERT_TO)
    with pytest.raises(ValueError, match="no agent_gt reached the forward"):
        _run(R, "oracle")


def test_the_reverted_trainer_still_runs_agents_off(tmp_path):
    """⛔ SAME-BREATH CONTROL. A mutant that is broken for every arm would make
    the test above pass for the wrong reason — the refusal must be about the
    ORACLE path, not about the mutation having wrecked the trainer."""
    R = _mutate(tmp_path, "reverted_ctl", REVERT_FROM, REVERT_TO)
    _model, losses, _seen = _run(R, "off")
    assert bool(torch.isfinite(losses["loss"]))


# --------------------------------------------------------------------------- #
# 3. PARITY — `off` and `head` are BITWISE unchanged by the new branch           #
# --------------------------------------------------------------------------- #
def _bitwise_diff(a: dict, b: dict) -> list[str]:
    assert set(a.keys()) == set(b.keys())
    bad = []
    for k in a:
        x, y = a[k], b[k]
        x = x if torch.is_tensor(x) else torch.tensor(float(x))
        y = y if torch.is_tensor(y) else torch.tensor(float(y))
        if x.shape != y.shape or not torch.equal(x.detach(), y.detach()):
            bad.append(k)
    return bad


@pytest.mark.parametrize("arm", ["off", "head"])
def test_agents_off_and_head_are_BITWISE_UNCHANGED(tmp_path, arm):
    """⛔ THE PARITY CLAIM, against the trainer WITHOUT the branch. The batch
    carries GT boxes on both sides, so a gate that leaked would change the
    answer.

    ⚠️ **Scope it honestly: only the ``off`` arm's parity is load-bearing.**
    MEASURED 2026-09-10 — with the gate deliberately removed, ``off`` REFUSES
    (``refc_v3.py:1398``) but ``head`` stays bitwise identical, because
    ``refc.py:3366-3367`` builds head slots from ``fmap`` and never reads
    ``agent_gt``, while the reverse guard only fires when ``core.agents`` is
    absent or disabled. So on a head build ``agent_gt`` is SILENTLY DROPPED,
    and this row cannot distinguish a correct gate from no gate at all. It is
    kept because it still pins that the head arm did not MOVE; the sensitivity
    proof is :func:`test_the_OFF_parity_comparison_CAN_FAIL`."""
    T = _trainer()
    R = _mutate(tmp_path, f"nobranch_{arm}", REVERT_FROM, REVERT_TO)
    _m0, l0, _ = _run(R, arm)
    _m1, l1, _ = _run(T, arm)
    bad = _bitwise_diff(l0, l1)
    assert bad == [], bad
    assert len(l0) == 20


def test_the_OFF_parity_comparison_CAN_FAIL(tmp_path):
    """⛔⛔ THE DISCRIMINATING CONTROL. A parity test that cannot fail proves
    nothing — WP-B's removability proof was green for a reason unrelated to
    WP-B because a zero-init gate multiplied its branch away.

    Remove the gate so the branch fires on EVERY arm and ``--agents off``
    must stop being identical. It does not merely differ: the model's reverse
    guard refuses outright, which is the strongest possible form of 'the
    comparison is sensitive'."""
    M = _mutate(tmp_path, "gate_removed", CORRUPT_GATE_FROM, CORRUPT_GATE_TO)
    with pytest.raises(ValueError, match="no agent seam"):
        _run(M, "off")


def test_the_ORACLE_value_assertion_CAN_FAIL(tmp_path):
    """⛔ The other half of the same discipline, on the ON side. Corrupt the
    VALUE inside the branch and leave the gate alone: the oracle's bitwise
    assertion must break while nothing else can see the change."""
    M = _mutate(tmp_path, "value_corrupt",
                CORRUPT_VALUE_FROM, CORRUPT_VALUE_TO)
    _model, _losses, seen = _run(M, "oracle", spy=True)
    box, _yaw, _cls, _valid = _gt_tensors()
    assert not torch.equal(seen["agent_gt"]["box"], box)
    assert not torch.equal(seen["out"]["agent_slots"]["box"].detach(), box)


def test_the_VALUE_corruption_leaves_agents_off_UNTOUCHED(tmp_path):
    """⛔ ...and its converse, which is what makes the pair a discriminator
    rather than two separate observations: the in-branch corruption must be
    INVISIBLE to ``--agents off``. If this failed, the corruption would be
    leaking outside the gate and the test above would prove nothing about
    where the sensitivity lives."""
    T = _trainer()
    M = _mutate(tmp_path, "value_corrupt_off",
                CORRUPT_VALUE_FROM, CORRUPT_VALUE_TO)
    _m0, l0, _ = _run(M, "off")
    _m1, l1, _ = _run(T, "off")
    assert _bitwise_diff(l0, l1) == []


# --------------------------------------------------------------------------- #
# 4. THE REFUSAL THE BRANCH ADDS — an oracle arm with no boxes                   #
# --------------------------------------------------------------------------- #
def test_agents_oracle_WITHOUT_agent_box_IN_THE_BATCH_REFUSES():
    """⛔ A refusal, never a silent ``None``. An oracle arm whose batch carries
    no boxes would read as 'agent tokens do not help' while never having had
    any — and the trainer's refusal names the CAUSE (``--agent-join``) where
    the model's can only name the symptom."""
    T = _trainer()
    model, batch = _build(T, "oracle")
    for k in ("agent_box", "agent_yaw", "agent_cls", "agent_valid"):
        batch.pop(k)
    with pytest.raises(SystemExit, match="no `agent_box`"):
        T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
