"""⛔ EVERY BUILT MODULE MUST RECEIVE A GRADIENT — or be on a NAMED list of
known-unwired seams that MUST SHRINK.

⭐⭐ WHY THIS FILE EXISTS. The v7 vocabulary reach census (2026-09-07) measured
``tac_goal_tok_head`` at ``p.grad is None`` on **both** its tensors under the
LIVE refcv5-v2 argv — 11,286 parameters that are built, stamped, forward-run and
cannot learn. **Both existing guards were green on it, and neither is wrong:**

* ``assert_seams_are_built`` asks *"is the head BUILT?"* — it is;
* ``effective_weights_stamp_v3`` enumerates **declared loss weights** and asks
  which build a graph. Its own preamble cites the right measurement (*42/138
  optimizer tensors took no gradient — 52.2 % of a declared trainable budget*)
  and its ``_discriminator`` is exactly ``p.grad is None``. But
  ``tac_goal_tok_head`` **has no weight flag at all**, so it produces no row.
  An instrument that enumerates *weights* cannot see a head with *no weight*.

⇒ the missing question is neither *"is it built?"* nor *"is its weight zero?"*
but **"does a gradient actually land on it?"**, asked of EVERY built module.
That is what this file asks, once, structurally.

⭐ THE PRECEDENT IS ALREADY IN THIS TRAINER, ONE FLAG OVER.
``_check_goal_point_args`` refuses ``--goal-point-inject`` with
``--goal-point-w <= 0`` in exactly these words: *"a head that is built, stamped,
and **supervised by nothing** … a failure that is not about the lever is worse
than no run."* ``--tac-goal-tok-head`` is that same configuration and has no such
refusal, because it has no weight to key on.

⛔ THE DISCRIMINATOR IS ``p.grad is None``, NEVER THE GRADIENT'S VALUE. A module
whose gradient is present but ZERO is *wired and switched off* (a zero-init gate,
an all-ignored batch, a mask) — legitimate, and common at step 0. A module whose
gradient is ``None`` on **every** tensor was never in the graph. Only the second
is a defect, and conflating them is how the *"guarded term makes p.grad None"*
trap gets re-derived.

⛔ THE ALLOW-LIST IS A LITERAL AND IT MUST SHRINK. It is written as an explicit
``{module: reason}`` map, never as an expression over the model — the same shape
as ``vocab_v7.NOT_YET_EXTRACTABLE``, and for the same reason: an honest gap that
is NAMED cannot be silent, and a NEW unwired head fails immediately.
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

#: the census this guard came out of; every failure message names it.
CENSUS_DOC = ("TanitAD Research Lab/Architecture & Inference/Research/"
              "2026-09-07-v7-vocab-reach-census/RESULT.md")


def _trainer():
    """``refc_v3_train.py`` by PATH — the same convention as
    ``test_tac_goal_trainer_flag.py`` and ``refcv3_arm``."""
    if not os.path.exists(_TRAINER_PY):
        pytest.skip(f"trainer not present at {_TRAINER_PY}")
    spec = importlib.util.spec_from_file_location(
        "refc_v3_train_for_gradreach", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: ⛔ THE RECORDED argv OF THE LIVE refcv5-v2 ARM, copied from its own
#: ``config.json`` (banked at ``Research/2026-09-07-refcv5-v2-compose/raw/
#: refcv5_v2_launch_config_20260906.json``). Data paths are pod paths and are
#: never opened here — the smoke config supplies synthetic episodes.
LIVE_ARGV = [
    "--arm", "hier", "--size", "base",
    "--v2-cache", "/root/data/train",
    "--v7-labels", "/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz",
    "--image-hw", "256", "640", "--steps", "40284", "--batch", "20",
    "--lr", "1e-4", "--warmup", "2000", "--seed", "0", "--u8-batches",
    "--out", "/workspace/experiments/refcv5-v2-noagents-b1-v72-40k",
    "--nav-from-v7", "--ego-state-inject", "--ego-dropout", "0.5",
    "--n-anchors", "117", "--anchor-v0-conditioned",
    "--anchor-control-units", "alat", "--sel-accel-max", "2.0",
    "--sampler", "ddim", "--w-u0", "0.5", "--sel-refined",
    "--sel-score-emitted", "--goal-str", "--tac-goal-tok-head",
    "--agents", "off",
]

#: ⛔⛔ KNOWN-UNWIRED SEAMS. Every entry is a FINDING, not an exemption, and this
#: map MUST SHRINK. ⛔ Literals only — never derived from the model.
KNOWN_UNWIRED: dict[str, str] = {
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
}


def _module_grad_census(model) -> dict[str, dict]:
    """Per named child: how many parameter tensors took ``None`` gradient.

    ⛔ MODULE level, not tensor level. A module with SOME ``None`` gradients is
    partially masked, which is ordinary; a module where EVERY tensor is ``None``
    was never in the graph.
    """
    out: dict[str, dict] = {}
    for name, mod in model.named_children():
        ps = [p for p in mod.parameters() if p.requires_grad]
        if not ps:
            continue
        n_none = sum(1 for p in ps if p.grad is None)
        gsum = sum(float(p.grad.abs().sum()) for p in ps if p.grad is not None)
        out[name] = {
            "n_tensors": len(ps),
            "n_params": sum(p.numel() for p in ps),
            "n_grad_none": n_none,
            "grad_abs_sum": gsum,
            "verdict": ("NOT_WIRED" if n_none == len(ps)
                        else "ZERO_GRAD" if gsum == 0.0
                        else "GRADIENT_REACHES"),
        }
    return out


def _build_and_backward(extra_argv: list[str] | None = None, *, corrupt=None):
    """Build the arm the LIVE argv describes (smoke width), run the trainer's
    OWN loss, one backward. Returns ``(model, census, losses)``.

    ⛔ Backwards ``losses["loss"]`` — the trainer's own total. Summing every
    tensor that happens to carry grad would ADD gradient paths the run does not
    have and turn an unwired head into a reached one.
    """
    T = _trainer()
    args = T.build_parser().parse_args(LIVE_ARGV + list(extra_argv or []))
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    model._w_goal_point = float(getattr(args, "goal_point_w", 0.0) or 0.0)
    model._w_tac_goal = float(getattr(args, "w_tac_goal", 0.0) or 0.0)
    model._tac_goal_pos_weight = None
    model._tac_goal_class_mask = None
    model.train()

    eps = T._synth_episodes(2, cfg.core, seed=0)
    ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                     channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    v7l = T.v7l
    n_lon = len(v7l.HEADS["tac_lon"])
    batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
    batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID], dtype=torch.long)
    batch["nav_cmd"] = torch.tensor([1, 2], dtype=torch.long)
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

    losses = T.compute_losses_v3(model, batch, "cpu", mode="diffusion")
    total = losses["loss"]
    assert total.requires_grad, (
        "the trainer's own total loss does not require grad -- the probe is "
        "UNPOWERED, not clean")
    total.backward()
    return model, _module_grad_census(model), losses


# --------------------------------------------------------------------------- #
# THE GUARD                                                                     #
# --------------------------------------------------------------------------- #
def test_no_built_module_is_unwired_except_the_named_list() -> None:
    """⛔ THE CHECK THAT WAS MISSING. Any built module whose EVERY parameter
    takes a ``None`` gradient is unreachable from the loss."""
    _model, census, _losses = _build_and_backward()
    unwired = {k: v for k, v in census.items() if v["verdict"] == "NOT_WIRED"}
    allowed = set(KNOWN_UNWIRED) | set(OFF_BY_WEIGHT)
    unexpected = {k: v for k, v in unwired.items() if k not in allowed}
    assert not unexpected, (
        f"\n  ⛔ BUILT BUT UNWIRED: {sorted(unexpected)}\n"
        f"     {unexpected}\n"
        f"  Every parameter of these modules took `p.grad is None` after the\n"
        f"  trainer's own loss and one backward -- they are built, stamped and\n"
        f"  cannot learn. This is the D-TACGOAL-1 shape.\n"
        f"  ⇒ EITHER wire the loss, OR add the module to KNOWN_UNWIRED with a\n"
        f"     reason and register the gap. Do NOT widen the check.\n"
        f"  Census: {CENSUS_DOC}\n")


def test_the_known_unwired_list_must_shrink_and_never_go_stale() -> None:
    """⛔ Two-sided. An entry that is no longer unwired is GOOD NEWS and still
    fails, because the register and the docs describe the gap."""
    _model, census, _losses = _build_and_backward()
    for name, reason in KNOWN_UNWIRED.items():
        assert name in census, (
            f"\n  KNOWN_UNWIRED names {name!r} but the model has no such built\n"
            f"  module. If it was removed, delete the entry. Reason on file:\n"
            f"  {reason}\n")
        assert census[name]["verdict"] == "NOT_WIRED", (
            f"\n  ⭐ GOOD NEWS: {name!r} now reads "
            f"{census[name]['verdict']} -- it is WIRED.\n"
            f"  ⇒ DELETE it from KNOWN_UNWIRED, and update in the SAME commit:\n"
            f"      - Project Steering/GOALS_AND_CLAIMS.md "
            f"(D-TACGOAL-TRAINER-SEAM-OPEN)\n"
            f"      - {CENSUS_DOC}\n"
            f"      - stack/tests/test_tactical_label_reach.py\n"
            f"      - stack/tests/test_v7_vocab_reach_census.py "
            f"(EXPECTED_CONSUMER_STATUS)\n")


def test_the_probe_is_powered_not_merely_quiet() -> None:
    """⛔ C109: a probe that cannot fire proves nothing. Modules that SHOULD be
    reached must actually read GRADIENT_REACHES in the same run."""
    _model, census, _losses = _build_and_backward()
    for name in ("core", "lat_head_tac", "lon_head_tac", "phi_tac"):
        assert name in census, f"{name!r} is not a built child of the model"
        assert census[name]["verdict"] == "GRADIENT_REACHES", (
            f"{name!r} reads {census[name]['verdict']} -- the probe is "
            f"UNPOWERED, so its NOT_WIRED verdicts elsewhere are "
            f"uninformative. Census row: {census[name]}")


def test_the_detector_actually_fires_on_a_deliberately_unwired_head() -> None:
    """⭐ MUTATION. Attach a brand-new head that no loss touches and prove the
    detector names it."""
    import torch.nn as nn
    model, _census, _losses = _build_and_backward()
    model.add_module("MUTANT_unwired_head", nn.Linear(4, 3))
    census = _module_grad_census(model)
    assert census["MUTANT_unwired_head"]["verdict"] == "NOT_WIRED", (
        "the detector did NOT fire on a head that plainly received no "
        "gradient -- it is dead, and every clean verdict from it is "
        "uninformative")
    unexpected = {k for k, v in census.items()
                  if v["verdict"] == "NOT_WIRED"
                  and k not in (set(KNOWN_UNWIRED) | set(OFF_BY_WEIGHT))}
    assert unexpected == {"MUTANT_unwired_head"}, (
        f"the mutation changed the verdict of unrelated modules "
        f"({unexpected}) -- the detector is not specific")


def test_a_zero_gradient_is_NOT_read_as_unwired() -> None:
    """⛔ THE DISCRIMINATOR. ``p.grad`` present-but-zero means wired and
    switched off (a zero-init gate, a mask, an all-ignored batch). Reading it
    as 'unwired' would re-derive the guarded-term trap in the opposite
    direction and make this guard useless at step 0."""
    import torch.nn as nn
    model, _census, _losses = _build_and_backward()
    lin = nn.Linear(4, 3)
    (lin(torch.zeros(1, 4)).sum() * 0.0).backward()      # real grad, all zeros
    assert lin.weight.grad is not None
    assert float(lin.weight.grad.abs().sum()) == 0.0
    model.add_module("MUTANT_zero_grad_head", lin)
    census = _module_grad_census(model)
    assert census["MUTANT_zero_grad_head"]["verdict"] == "ZERO_GRAD", (
        "a module with a real, all-zero gradient was classified as NOT_WIRED "
        "-- the guard would fire on every zero-init gate at step 0 and would "
        "be turned off within a day")


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
            f"\n  ⛔ {name!r} reads {row['verdict']} with {flag} 1.0 -- the "
            f"weight is declared and STILL no gradient lands.\n"
            f"  Census row: {row}\n"
            f"  This is D-TACGOAL-TRAINER-SEAM-OPEN re-opened: a head that is "
            f"built, stamped, given a weight, and supervised by nothing.\n")
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
        n = len(re.findall(r'(?<!add_argument\()["\']' + re.escape(d)
                           + r'["\']|\bargs\.' + re.escape(d) + r'\b', src))
        reads = len(re.findall(r'getattr\(\s*args\s*,\s*["\']'
                               + re.escape(d) + r'["\']', src))
        reads += len(re.findall(r'\bargs\.' + re.escape(d) + r'\b', src))
        if reads == 0:
            inert.append(d)
    assert not inert, (
        f"\n  ⛔ PARSED, STAMPED AND INERT: {inert}\n"
        f"  These tac-goal knobs are accepted by the parser and never read by "
        f"the trainer -- the `--wp-index` defect (3 of 6 knobs inert).\n")
