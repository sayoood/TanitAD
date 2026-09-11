"""D-TACGOAL-2 / PI queue item 10 — the ``--grad-probe-modules`` gradient probe.

⛔ WHY THIS FILE EXISTS. ``tac_goal_tok_head`` is 11,286 parameters that took
``grad_abs_sum`` **EXACTLY 0.00000 for all 40,284 steps** of refcv5-v2, because
``--w-tac-goal`` is declared with ``default=0.0`` and the arm never passed it.
The head was built, stamped, forward-run and rollable — and untrained.
⭐ *"Rollable and trained are different claims."* Nothing in that run's record
said which one was true, and the only instrument that can tell them apart is the
gradient the head actually receives. This file guards that instrument.

⛔ THE THREE FAILURES THIS FILE IS AIMED AT, each written as a test that must go
RED if the defect returns:

1. **The zero-weight defect itself.** With the term absent from the graph every
   parameter's ``.grad`` is ``None``. The probe must then read ``grad_abs_sum``
   **exactly 0.0** with ``n_grad_none == n_tensors`` — the no-information value,
   exactly. A probe that cannot read the defect certifies nothing.
2. **The rounding hole.** The trainer's log row rounds every scalar to 5 dp. A
   genuinely non-zero gradient of ``1e-8`` rounds to ``0.0`` — i.e. the log would
   manufacture the very defect being measured. The probe's values are therefore
   merged into the row AFTER the rounding comprehension, and that ORDER is pinned
   here with a mutation arm that reintroduces the wrong order and goes RED.
3. **A silent absence.** A mistyped module path must read ``found = 0.0``, never
   look like a module that simply received no gradient. An unreadable thing and
   an unreached thing must never be the same reading — the same rule that makes a
   bare ``grep -c 0`` inadmissible on a flaky mount.

⛔ EVERY EXPECTATION HERE IS A LITERAL. Not one is an expression over the code
under test: ``0.0``, ``2``, ``11286``, ``1e-8``. Re-deriving the producer's own
arithmetic and finding agreement measures determinism, not correctness.

⚠️ CPU ONLY, DELIBERATELY. The dev box's RTX 4060 is shared; nothing here
allocates on CUDA, and the probe's ``gp_cuda_max_mem_gb`` key is asserted ABSENT
on a CPU run rather than asserted present.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_STACK = os.path.dirname(_HERE)                             # <repo>/stack
if _STACK not in sys.path:
    sys.path.insert(0, _STACK)

_TRAINER_PY = os.path.join(_STACK, "scripts", "refc_v3_train.py")


# ==========================================================================
# rig
# ==========================================================================
def _trainer():
    """``refc_v3_train.py`` by PATH — it is a script, and ``build_parser`` plus
    ``_grad_probe_row`` are only reachable that way. Same convention as
    ``test_max_speed_wiring.py`` and ``test_built_heads_receive_gradient.py``."""
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_gradprobe", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refc_v3_train_for_gradprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


def _source() -> str:
    return io.open(_TRAINER_PY, encoding="utf-8").read()


class _TwoHeads(torch.nn.Module):
    """A trunk feeding two heads. ``reached`` is in the loss; ``unreached`` is
    the stand-in for ``tac_goal_tok_head`` under ``--w-tac-goal 0.0``: built,
    forward-runnable, and ABSENT from the graph."""

    def __init__(self, d: int = 4):
        super().__init__()
        self.trunk = torch.nn.Linear(d, d)
        self.reached = torch.nn.Linear(d, 3)
        self.unreached = torch.nn.Linear(d, 22)

    def forward(self, x):
        h = self.trunk(x)
        return self.reached(h), self.unreached(h)


def _rig(seed: int = 0):
    torch.manual_seed(seed)
    m = _TwoHeads()
    x = torch.ones(2, 4)
    return m, x


# ==========================================================================
# 1 — the flag's contract, as literals
# ==========================================================================
def test_flag_defaults_are_the_literals_the_defect_depended_on():
    """⛔ ``--w-tac-goal`` defaulting to ``0.0`` IS the mechanism of the 40,284
    zero-gradient steps. It is pinned here as a literal so a future change to it
    is a deliberate act with a failing test attached, not a silent one. The probe
    flag defaults to the empty string so a recipe that does not pass it is
    identical to the pre-probe trainer."""
    tr = _trainer()
    a = tr.build_parser().parse_args(
        ["--arm", "hier", "--size", "small", "--out", "/x", "--v2-cache", "/c"])
    assert a.grad_probe_modules == ""            # literal
    assert a.w_tac_goal == 0.0                   # literal — the defect's cause
    assert a.tac_goal_tok_head is False          # literal


def test_probe_returns_nothing_when_the_flag_is_absent():
    """The OFF path computes nothing at all, so the metrics.jsonl schema of a
    run that does not pass the flag is unchanged."""
    tr = _trainer()
    m, x = _rig()
    m(x)[0].sum().backward()
    assert tr._grad_probe_row(m, [], log_every_hit=True) == {}
    assert tr._grad_probe_row(m, ["reached"], log_every_hit=False) == {}


# ==========================================================================
# 2 — THE DELIBERATE-REGRESSION ARM: the zero-weight defect, reintroduced
# ==========================================================================
def test_unreached_head_reads_exactly_zero_and_reached_head_does_not():
    """⛔ THE REGRESSION ARM. ``unreached`` is excluded from the loss exactly as
    ``tac_goal_tok_head`` was excluded by ``--w-tac-goal 0.0`` — the term is
    ABSENT from the graph, not multiplied by zero. The probe must read the
    no-information value EXACTLY.

    ⭐ The discriminating control is the other half: ``reached`` must read
    non-zero in the SAME backward. Without it, "reads 0.0" would be satisfied by
    a probe that always reads 0.0."""
    tr = _trainer()
    m, x = _rig()
    reached, _unreached = m(x)
    reached.sum().backward()                     # `unreached` is not in the graph
    row = tr._grad_probe_row(m, ["unreached", "reached"])

    # the defect, read exactly
    assert row["gp_unreached_grad_abs_sum"] == 0.0        # literal
    assert row["gp_unreached_n_grad_none"] == 2.0         # literal: weight+bias
    assert row["gp_unreached_n_tensors"] == 2.0           # literal
    assert row["gp_unreached_found"] == 1.0               # built, not missing

    # the control that makes the zero mean something
    assert row["gp_reached_grad_abs_sum"] > 0.0
    assert row["gp_reached_n_grad_none"] == 0.0           # literal


def test_including_the_term_flips_the_same_head_to_a_real_gradient():
    """The other side of the regression arm: the ONLY change is that the second
    head's output enters the loss. Same module, same seed, same rig."""
    tr = _trainer()
    m, x = _rig()
    reached, unreached = m(x)
    (reached.sum() + 0.05 * unreached.sum()).backward()
    row = tr._grad_probe_row(m, ["unreached"])
    assert row["gp_unreached_n_grad_none"] == 0.0         # literal
    assert row["gp_unreached_grad_abs_sum"] > 0.0


def test_probe_sums_ABSOLUTE_GRADIENT_against_an_analytic_target():
    """⛔ THIS TEST EXISTS BECAUSE THE TWO TESTS ABOVE WERE INERT AGAINST A REAL
    MUTATION. Replacing ``p.grad.abs().sum()`` with ``p.numel()`` — a probe that
    counts PARAMETERS and can therefore never read a gradient at all — left both
    of them GREEN: the unreached head still summed to 0.0 (its grads are
    ``None``, so the mutated line never ran) and the reached head still summed to
    something positive. *A check that shares the defect it checks for is green
    forever*, and "> 0.0" is exactly that kind of check.

    ⭐ The discriminator is an ANALYTIC TARGET, written as a literal. ``reached``
    is ``Linear(4, 3)``: 12 weights + 3 biases = **15** parameters in **2**
    tensors. Every gradient element is set to exactly ``-2.0``, so

        sum(|grad|) = 15 x 2.0 = 30.0        exactly

    while a ``numel``-counting probe reads **15.0** and an ``abs``-less probe
    reads **-30.0**. One expectation separates all three."""
    tr = _trainer()
    m, x = _rig()
    m(x)[0].sum().backward()
    with torch.no_grad():
        for p in m.reached.parameters():
            p.grad = torch.full_like(p, -2.0)
    row = tr._grad_probe_row(m, ["reached"])

    assert row["gp_reached_n_params"] == 15.0             # literal
    assert row["gp_reached_n_tensors"] == 2.0             # literal
    assert row["gp_reached_grad_abs_sum"] == 30.0         # literal, analytic


# ==========================================================================
# 3 — THE ROUNDING HOLE, and the mutation that reintroduces it
# ==========================================================================
def _round_row(d):
    """The trainer's own log-row rounding, transcribed. 5 dp, as at
    ``refc_v3_train.py``'s ``row = {k: round(float(v), 5) ...}``."""
    return {k: round(float(v), 5) for k, v in d.items()}


def test_a_gradient_below_the_log_rounding_survives_the_probe():
    """⛔ THE HAZARD IS REAL, asserted first as a literal: ``round(1e-8, 5)`` IS
    ``0.0``. A probe value that went through the row's rounding would therefore
    be indistinguishable from the 40,284-step defect."""
    assert round(1e-8, 5) == 0.0                          # the hazard, literal

    tr = _trainer()
    m, x = _rig()
    reached, _ = m(x)
    reached.sum().backward()
    with torch.no_grad():                     # a real but tiny gradient
        for p in m.unreached.parameters():
            p.grad = torch.zeros_like(p)
        m.unreached.bias.grad[0] = 1e-8
    row = tr._grad_probe_row(m, ["unreached"])

    got = row["gp_unreached_grad_abs_sum"]
    assert got > 0.0
    assert got == pytest.approx(1e-8, rel=1e-6)

    # CORRECT ORDER (probe merged AFTER rounding) — the value survives
    correct = _round_row({"loss": 1.234567891}) | {k: v for k, v in row.items()}
    assert correct["gp_unreached_grad_abs_sum"] > 0.0

    # ⛔ MUTATION ARM: merge BEFORE rounding, which is the order the trainer
    # must never use. The 1e-8 is destroyed and the row reads as the defect.
    mutated = _round_row({"loss": 1.234567891, **row})
    assert mutated["gp_unreached_grad_abs_sum"] == 0.0    # RED if order flips
    assert mutated["gp_unreached_grad_abs_sum"] != correct[
        "gp_unreached_grad_abs_sum"]


def test_trainer_merges_the_probe_after_its_rounding_comprehension():
    """The source-order assertion the mutation above stands for. Positive
    controls first, so an unreadable file cannot pass as a satisfied test."""
    s = _source()
    assert s.count("def train(args) -> dict:") == 1       # control: file served

    # ⛔ THE EXACT BLOCK, as a literal. A weaker form of this test — searching
    # for the substring ``row.update(_gp_row)`` alone — was MEASURED INERT
    # against a mutation that merely DISABLED the merge (``if _gp_row and
    # False:``): the string is still present, the order is still right, and the
    # probe never reaches metrics.jsonl. The guard condition is pinned too.
    block = ("            if _gp_row:" + chr(10) +
             "                row.update(_gp_row)" + chr(10))
    assert s.count(block) == 1

    i_round = s.find("row = {k: (round(float(v.detach()), 5)")
    i_merge = s.find(block)
    i_write = s.find('log.write(json.dumps(row) + "' + chr(92) + 'n")')
    assert i_round > 0 and i_merge > 0 and i_write > 0
    assert i_round < i_merge < i_write


def test_probe_is_read_after_backward_and_before_the_global_clip():
    """⛔ ``clip_grad_norm_`` rescales every gradient by a GLOBAL factor. A
    post-clip reading would confound this head's own gradient with how large
    everything else's was that step, so the probe site is pinned to sit between
    ``backward()`` and the clip."""
    s = _source()
    i_bwd = s.find('losses["loss"].backward()')
    i_probe = s.find("_gp_row = _grad_probe_row(")
    i_clip = s.find("torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)")
    assert i_bwd > 0 and i_probe > 0 and i_clip > 0
    assert i_bwd < i_probe < i_clip


# ==========================================================================
# 4 — a missing module must not look like an unreached one
# ==========================================================================
def test_a_mistyped_module_reads_found_zero_rather_than_a_silent_absence():
    """⚠️ Same rule that makes a bare ``grep -c`` of 0 inadmissible on a flaky
    mount: a thing that could not be READ and a thing that is genuinely absent
    must not produce the same reading."""
    tr = _trainer()
    m, x = _rig()
    m(x)[0].sum().backward()
    row = tr._grad_probe_row(m, ["tac_goal_tok_haed"])     # deliberate typo
    assert row["gp_tac_goal_tok_haed_found"] == 0.0        # literal
    assert "gp_tac_goal_tok_haed_grad_abs_sum" not in row


def test_cpu_run_emits_no_cuda_key():
    """The memory key is emitted only when CUDA is present. On Thor it is the
    ONLY admissible device-memory probe (``mem_get_info``, ``free``/``tegrastats``
    and ``VmRSS`` all lie there, in both directions); on a CPU test box it must
    simply be absent rather than reading a fabricated 0.0."""
    tr = _trainer()
    m, x = _rig()
    m(x)[0].sum().backward()
    row = tr._grad_probe_row(m, ["reached"])
    if not torch.cuda.is_available():
        assert "gp_cuda_max_mem_gb" not in row


# ==========================================================================
# 5 — the row is JSON-serialisable, because metrics.jsonl is the artifact
# ==========================================================================
def test_row_is_json_serialisable_and_keeps_full_precision():
    tr = _trainer()
    m, x = _rig()
    m(x)[0].sum().backward()
    row = tr._grad_probe_row(m, ["reached", "unreached"])
    back = json.loads(json.dumps(row))
    assert back["gp_unreached_grad_abs_sum"] == 0.0        # literal
    assert back["gp_reached_grad_abs_sum"] == row["gp_reached_grad_abs_sum"]
