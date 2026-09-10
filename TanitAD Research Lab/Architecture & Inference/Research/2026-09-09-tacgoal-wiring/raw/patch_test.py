"""Update `test_built_heads_receive_gradient.py` for the closed seam.

⛔ THE LIST SHRINKS TO EMPTY, and it is REPLACED BY A STRONGER OBLIGATION --
not by a wider exemption. `KNOWN_UNWIRED` meant *"no loss exists"*. That is no
longer true of `tac_goal_tok_head`, so leaving it there would be dishonest in
the direction that matters: it would keep describing a seam as OPEN after it
was closed.

The replacement, `OFF_BY_WEIGHT`, carries an obligation `KNOWN_UNWIRED` never
did: for every entry, a test TURNS THE NAMED FLAG ON and proves the verdict
flips to GRADIENT_REACHES. An entry that cannot be flipped fails.
"""
from __future__ import annotations

import ast
import io
import sys

SRC, DST = sys.argv[1], sys.argv[2]
# ⚠️ MEASURED: this file is CRLF (260/260) while `refc_v3_train.py` is LF
# (0/5380). Matching LF anchors against un-translated CRLF text silently finds
# ZERO occurrences -- which reads exactly like "the anchor moved". Normalise for
# matching, then restore the file's OWN convention on write, or the patch shows
# up as a whole-file rewrite.
with io.open(SRC, "rb") as fh:
    _raw = fh.read()
CRLF = _raw.count(b"\r\n")
LF = _raw.count(b"\n")
NEWLINE = "\r\n" if CRLF * 2 > LF else "\n"
print(f"source newline: CRLF={CRLF} LF={LF} -> writing {NEWLINE!r}")
s = _raw.decode("utf-8").replace("\r\n", "\n")

EDITS: list[tuple[str, str, str]] = []

# --------------------------------------------------------------------------- #
A1 = '''KNOWN_UNWIRED: dict[str, str] = {
    "tac_goal_tok_head":
        "D-TACGOAL-TRAINER-SEAM-OPEN (measured 2026-09-07): the head is built, "
        "stamped and forward-run (refc_v3.py:1320 -> cache['tac_goal_logits']) "
        "and NO trainer calls `tac_goal_loss` or `TacGoalEmitter`, so its "
        "11,286 parameters take no gradient. Closing it needs the two additive "
        "edits inside refc_v3_train.py AND the MANEUVER_WEIGHT budget decision "
        "(an owner/PI call). When it lands, DELETE this entry -- the test will "
        "tell you by failing the `must shrink` assertion below.",
}'''
B1 = '''KNOWN_UNWIRED: dict[str, str] = {
    # ⭐⭐ 2026-09-09 -- `tac_goal_tok_head` LEFT THIS LIST. The trainer now
    # calls `tac_goal_head.tac_goal_loss` on `out["tac_goal_logits"]` behind
    # `--w-tac-goal`, and a real backward puts grad_abs_sum 3.602122873067856
    # on its two tensors (MEASURED, smoke width; RESULT.md in
    # Research/2026-09-09-tacgoal-wiring/). It did NOT move to a wider
    # exemption: it moved to OFF_BY_WEIGHT below, which is STRICTER, because
    # every entry there must be provably flippable by its own flag.
    #
    # ⛔ THE LIST IS EMPTY AND MUST STAY THAT WAY. A new entry is a NEW
    # UNWIRED HEAD -- a finding, never a way to make this file green.
}

#: ⛔⛔ WIRED, AND SWITCHED OFF BY AN EXPLICITLY-ZERO WEIGHT. ``{module: flag}``.
#:
#: This is NOT `KNOWN_UNWIRED` with a friendlier name, and the difference is the
#: whole point. A `KNOWN_UNWIRED` entry says *"no loss exists"* and is discharged
#: only by writing one. An `OFF_BY_WEIGHT` entry says *"a loss exists and this
#: arm did not buy it"*, and it carries an obligation the other never had:
#: :func:`test_off_by_weight_heads_are_wired_when_their_flag_is_on` TURNS THE
#: FLAG ON and requires the verdict to flip to ``GRADIENT_REACHES``. An entry
#: that cannot be flipped FAILS -- so this map cannot be used to park a broken
#: head the way an allow-list can.
#:
#: ⚠️ WHY THE DEFAULT ARM STILL READS ``NOT_WIRED`` HERE, AND WHY THAT IS NOW
#: HONEST. ``LIVE_ARGV`` is the RECORDED argv of refcv5-v2 and carries no
#: ``--w-tac-goal``, so on that arm the term is genuinely absent from the graph
#: -- deliberately, because that absence is what makes the OFF path
#: bit-identical to the pre-wiring trainer (MEASURED: 0 of 167 parameter
#: tensors differ after one optimizer step, and the total loss is bitwise equal
#: at 19.020389556884766). ``tac_goal_head.py`` warns that a guarded term makes
#: ``p.grad is None`` and so reads identically to "never wired". That ambiguity
#: is resolved HERE by an independent record rather than by the gradient:
#: the head now has a WEIGHT, so ``effective_weights_stamp_v3`` carries a row
#: for it (``status: OFF_BY_DEFAULT``, ``builds_graph: false``) where before it
#: produced NO ROW AT ALL. config.json can now distinguish the two states; in
#: September 2026 it could not.
OFF_BY_WEIGHT: dict[str, str] = {
    "tac_goal_tok_head": "--w-tac-goal",
}'''
EDITS.append(("T1 KNOWN_UNWIRED -> OFF_BY_WEIGHT", A1, B1))

# --------------------------------------------------------------------------- #
A2 = '''    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")'''
B2 = '''    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
    batch["nav_valid"] = torch.tensor([True, True])
    # ⭐ The 22-token goal-SET target, injected exactly the way `lat_v7` /
    # `lon_v7` are: the smoke dataset carries no v7.2 join. Row 0 supervises
    # THREE cells and row 1 supervises none, so `tac_goal_n_supervised` has a
    # LITERAL expectation (3.0) rather than one derived from the code.
    # ⛔ Emitted unconditionally, including on the w=0 arm: the loss reads it
    # only when the weight is positive, so its presence cannot make the OFF
    # arm differ -- and injecting it only on the ON arm would confound the
    # weight with the target in every comparison below.
    _K = len(v7l.TAC_GOAL_TOKENS)
    _tg_y = torch.zeros(2, _K)
    _tg_w = torch.zeros(2, _K)
    _tg_y[0, 0] = 1.0
    _tg_w[0, 0] = 1.0
    _tg_w[0, 1] = 1.0
    _tg_w[0, 2] = 1.0
    batch["tac_goal_y"] = _tg_y
    batch["tac_goal_w"] = _tg_w
    if corrupt is not None:
        corrupt(T)

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")'''
EDITS.append(("T2 inject the goal target", A2, B2))

# --------------------------------------------------------------------------- #
A3 = '''def _build_and_backward(extra_argv: list[str] | None = None):'''
B3 = '''def _build_and_backward(extra_argv: list[str] | None = None, *, corrupt=None):'''
EDITS.append(("T3 corrupt hook signature", A3, B3))

# --------------------------------------------------------------------------- #
A4 = '''    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model.train()'''
B4 = '''    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()'''
EDITS.append(("T4 carry the tac-goal weight", A4, B4))

# --------------------------------------------------------------------------- #
A5 = '''    unexpected = {k: v for k, v in unwired.items() if k not in KNOWN_UNWIRED}
    assert not unexpected, ('''
B5 = '''    allowed = set(KNOWN_UNWIRED) | set(OFF_BY_WEIGHT)
    unexpected = {k: v for k, v in unwired.items() if k not in allowed}
    assert not unexpected, ('''
EDITS.append(("T5 allow OFF_BY_WEIGHT on the default arm", A5, B5))

# --------------------------------------------------------------------------- #
A6 = '''                  if v["verdict"] == "NOT_WIRED" and k not in KNOWN_UNWIRED}
    assert unexpected == {"MUTANT_unwired_head"}, ('''
B6 = '''                  if v["verdict"] == "NOT_WIRED"
                  and k not in (set(KNOWN_UNWIRED) | set(OFF_BY_WEIGHT))}
    assert unexpected == {"MUTANT_unwired_head"}, ('''
EDITS.append(("T6 mutation test respects OFF_BY_WEIGHT", A6, B6))

# --------------------------------------------------------------------------- #
# THE NEW TESTS -- appended.                                                   #
# --------------------------------------------------------------------------- #
NEW = '''

# --------------------------------------------------------------------------- #
# D-TACGOAL: THE SEAM THAT CLOSED, AND THE PROOF IT CANNOT SILENTLY RE-OPEN     #
# --------------------------------------------------------------------------- #
def test_off_by_weight_heads_are_wired_when_their_flag_is_on() -> None:
    """⛔ THE OBLIGATION THAT `KNOWN_UNWIRED` NEVER CARRIED.

    For every ``OFF_BY_WEIGHT`` entry: turn the named flag on and the module
    MUST read ``GRADIENT_REACHES``. An entry that cannot be flipped is a broken
    head parked behind a weight, which is the very thing this file exists to
    make impossible.
    """
    for name, flag in OFF_BY_WEIGHT.items():
        _model, census, losses = _build_and_backward([flag, "1.0"])
        assert name in census, f"{name!r} is not a built child of the model"
        row = census[name]
        assert row["verdict"] == "GRADIENT_REACHES", (
            f"\\n  ⛔ {name!r} reads {row['verdict']} with {flag} 1.0 -- the "
            f"weight is declared and STILL no gradient lands.\\n"
            f"  Census row: {row}\\n"
            f"  This is D-TACGOAL-TRAINER-SEAM-OPEN re-opened: a head that is "
            f"built, stamped, given a weight, and supervised by nothing.\\n")
        assert row["grad_abs_sum"] > 0.0, (
            f"{name!r} has gradient TENSORS but their absolute sum is exactly "
            f"0.0 -- the term is in the graph and contributes nothing. Row: "
            f"{row}")
        assert row["n_grad_none"] == 0, (
            f"{name!r}: {row['n_grad_none']} of {row['n_tensors']} tensors "
            f"still took a None gradient. A PARTIALLY wired head is not wired.")


def test_the_tac_goal_term_is_ABSENT_at_weight_zero_not_multiplied_by_zero(
) -> None:
    """⛔ THE BIT-IDENTITY PRECONDITION, asserted in-tree.

    `--w-tac-goal` defaults to 0.0 and the term must then NOT ENTER THE GRAPH.
    A term multiplied by zero would still appear in the loss dict and would
    still put a zeros-gradient on the head -- and it would make the OFF arm's
    autograd graph differ from the pre-wiring trainer's, which is exactly what
    may not happen while a live recipe is mid-flight.
    """
    _model, census, losses = _build_and_backward()
    assert "tac_goal" not in losses, (
        "the tac_goal term is present in the loss dict on an arm that did not "
        "ask for it -- it is being multiplied by zero rather than skipped, so "
        "the OFF path is NOT bit-identical to the pre-wiring trainer")
    assert census["tac_goal_tok_head"]["verdict"] == "NOT_WIRED", (
        "at --w-tac-goal 0 the head must be OUT of the graph entirely; it "
        f"reads {census['tac_goal_tok_head']['verdict']}")


def test_the_supervised_cell_count_is_reported_with_the_term() -> None:
    """⛔ A bare 0.0 in a metrics row reads as 'supervised, and perfect'.

    The literal is 3: the injected target supervises cells (0, 0), (0, 1) and
    (0, 2) and nothing else. ⭐ Written as a LITERAL, never as an expression
    over the target tensor -- re-deriving the producer's own arithmetic would
    measure determinism, not correctness.
    """
    _model, _census, losses = _build_and_backward(["--w-tac-goal", "1.0"])
    assert "tac_goal_n_supervised" in losses, (
        "the tac_goal term shipped without its `n`. `tac_goal_loss` returns "
        "n_supervised precisely so a zero can be read as 'nothing was in "
        "band' rather than 'the head is broken'.")
    assert float(losses["tac_goal_n_supervised"]) == 3.0, (
        f"expected exactly 3 supervised cells, got "
        f"{float(losses['tac_goal_n_supervised'])}")


def test_the_detector_fires_when_the_trainer_call_is_REMOVED() -> None:
    """⭐⭐ THE MUTATION THAT REINTRODUCES THE REAL HISTORICAL DEFECT.

    D-TACGOAL-TRAINER-SEAM-OPEN was exactly this: the head is BUILT, its logits
    ARE produced, and nothing consumes them. Reproduce it by replacing
    ``tac_goal_loss`` with one that ignores its ``logits`` argument -- the
    output is then detached from the head in precisely the way "no trainer
    call" detaches it -- and require the guard to go RED.

    ⛔ A guard that cannot be made to fail proves nothing. WP-B's removability
    proof was green with the head deliberately corrupted, because a zero-init
    gate had multiplied its whole branch away.
    """
    def _sever(T):
        tgh = T._tac_goal_head

        def _ignores_its_logits(logits, y, w, **kw):
            # a constant with a graph of its own -- NOT connected to `logits`
            return (y.sum() * 0.0).requires_grad_(False) + y.sum() * 0.0, 0

        tgh.tac_goal_loss = _ignores_its_logits

    T = _trainer()
    orig = T._tac_goal_head.tac_goal_loss
    try:
        _model, census, _losses = _build_and_backward(
            ["--w-tac-goal", "1.0"], corrupt=_sever)
        assert census["tac_goal_tok_head"]["verdict"] == "NOT_WIRED", (
            "⛔ THE GUARD IS DEAD. With the head's logits severed from the "
            "loss -- the D-TACGOAL defect verbatim -- the census still reads "
            f"{census['tac_goal_tok_head']['verdict']}. Every clean verdict "
            "from this detector is therefore uninformative.")
    finally:
        T._tac_goal_head.tac_goal_loss = orig


def test_the_head_now_has_a_row_in_the_effective_weights_stamp() -> None:
    """⭐⭐ THE OTHER HALF OF THE FIX, AND THE REASON IT IS NOT PACKAGING.

    The 2026-09-07 census established that ``effective_weights_stamp_v3``
    enumerates DECLARED LOSS WEIGHTS, and that this head produced **no row at
    all** because it had none. An instrument that enumerates weights is
    structurally blind to a head without one. Giving the head a weight is what
    makes it visible -- so the run record can finally distinguish *"switched
    off"* from *"never wired"*, which is the ambiguity `tac_goal_head.py`
    warns about.
    """
    T = _trainer()
    argv = LIVE_ARGV + ["--w-tac-goal", "1.0"]
    args = T.build_parser().parse_args(argv)
    args._ew_parser = T.build_parser()
    args._ew_argv = list(argv)
    stamp = T.effective_weights_stamp_v3(args, echo=False)
    rows = [r for r in stamp["terms"] if r["flag"] == "--w-tac-goal"]
    assert len(rows) == 1, (
        f"expected exactly one --w-tac-goal row in the effective-weights "
        f"stamp, found {len(rows)}. Flags present: "
        f"{[r['flag'] for r in stamp['terms']]}")
    row = rows[0]
    assert row["effective"] == 1.0, row
    assert row["builds_graph"] is True, row
    assert row["status"] == "TRAINS", row


def test_a_weight_without_its_head_is_REFUSED_at_launch() -> None:
    """⛔ The `w_agent` defect verbatim: a weight STAMPED in config.json while
    its loss term is silently skipped. Refuse in the preflight, in
    milliseconds, not after the corpus mounts."""
    T = _trainer()
    argv = [a for a in LIVE_ARGV if a != "--tac-goal-tok-head"]
    argv += ["--w-tac-goal", "1.0"]
    args = T.build_parser().parse_args(argv)
    args._ew_parser = T.build_parser()
    args._ew_argv = list(argv)
    with pytest.raises(SystemExit) as ei:
        T.check_effective_weights(args)
    assert "--w-tac-goal" in str(ei.value), str(ei.value)


def test_every_tac_goal_knob_the_parser_accepts_is_actually_CONSUMED() -> None:
    """⭐ DERIVED FROM ARGPARSE, NEVER FROM A HAND-WRITTEN LIST.

    ``--wp-index`` shipped with 3 of 6 knobs parsed, stamped and INERT, and it
    was caught only because a test enumerated the knobs from the parser rather
    than from a list someone maintained by hand. Same shape here: every
    ``--*tac-goal*`` option this parser accepts must be READ somewhere in the
    trainer source, by its argparse dest.
    """
    import re
    T = _trainer()
    dests = set()
    for act in T.build_parser()._actions:
        for opt in act.option_strings:
            if "tac-goal" in opt:
                dests.add(act.dest)
    assert dests, "the parser exposes no tac-goal knob at all"
    with open(_TRAINER_PY, encoding="utf-8") as fh:
        src = fh.read()
    inert = []
    for d in sorted(dests):
        # a READ of the dest, not merely the add_argument that declares it
        n = len(re.findall(r'(?<!add_argument\\()["\\']' + re.escape(d)
                           + r'["\\']|\\bargs\\.' + re.escape(d) + r'\\b', src))
        reads = len(re.findall(r'getattr\\(\\s*args\\s*,\\s*["\\']'
                               + re.escape(d) + r'["\\']', src))
        reads += len(re.findall(r'\\bargs\\.' + re.escape(d) + r'\\b', src))
        if reads == 0:
            inert.append(d)
    assert not inert, (
        f"\\n  ⛔ PARSED, STAMPED AND INERT: {inert}\\n"
        f"  These tac-goal knobs are accepted by the parser and never read by "
        f"the trainer -- the `--wp-index` defect (3 of 6 knobs inert).\\n")
'''
# --------------------------------------------------------------------------- #
for tag, a, b in EDITS:
    n = s.count(a)
    if n != 1:
        raise SystemExit(f"⛔ ANCHOR NOT UNIQUE for {tag}: {n} occurrences")
    s = s.replace(a, b, 1)
    print(f"applied {tag}")

s = s.rstrip("\n") + "\n" + NEW
ast.parse(s)
assert s.count("OFF_BY_WEIGHT") >= 5, "the new map is not used"
assert '"tac_goal_tok_head":\n        "D-TACGOAL-TRAINER-SEAM-OPEN' not in s

with io.open(DST, "wb") as fh:
    fh.write(s.replace("\n", NEWLINE).encode("utf-8"))
print(f"WROTE {DST} lines={s.count(chr(10))}")
