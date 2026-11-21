"""⛔⛔ THE ENCODER'S x0.5 MUST SURVIVE EVERY STEP, NOT JUST THE BUILD.

⭐⭐ THE DEFECT, MEASURED 2026-09-22 (`2026-09-22-refcv6-review/raw/recipe_probe.json`).
`build_optimizer` was CORRECT: it returns an encoder group at `5.0e-05` against the heads'
`1.0e-04`, **ratio 0.5000 at build**. The training loop then wrote

    for g in opt.param_groups:
        g["lr"] = args.lr * sched(step)          # <- args.lr, not the group's own

into **every** group, so the multiplier was overwritten on the very first step. The measured
ratio read **1.0000 at steps 0 / 1 / 1999 / 2000 / 15000 / 29999** — all six. ⇒ **21.28 M
encoder parameters, 99.3 % of the trainable weight, trained at the HEAD rate for the whole
run**, while `config.json` stamped `encoder_lr_mult: 0.5`.

SPEC §2's *"encoder lr ×0.5"* is DiffusionDrive's own recipe (`rl_config.py:124-128`), and it
had never been applied by any run. `initial_lr` — the standard latch — occurred **0 times** in
the trainer.

⭐ WHY A BUILD-TIME TEST COULD NOT HAVE CAUGHT IT, and this is the transferable part: the
optimiser was right at construction and wrong one step later. A guard that inspects the object
it just built measures the CONSTRUCTOR. The question is what the object holds **after the loop
body has run**, so this file runs the loop body.

⛔ THE RATIO IS THE ASSERTION, NOT THE ABSOLUTE LR. The ratio is schedule-independent by
construction — both groups take the same scalar `sched(step)` — so a test on the ratio cannot
be broken by a change to the warm-up or cosine shape, and cannot silently pass because the
schedule happens to be flat somewhere.

⛔ CPU only.
"""
from __future__ import annotations

import importlib.util
import math
import os

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_PY = os.path.join(_HERE, "..", "scripts", "refc_v3_train.py")

#: ⛔ LITERAL, from DiffusionDrive `rl_config.py:124` via SPEC §2. Not read from the code.
EXPECT_RATIO = 0.5
#: the exact steps the defect was measured at, so a regression is compared like with like
STEPS = (0, 1, 1999, 2000, 15000, 29999)


def _trainer():
    if not os.path.exists(_TRAINER_PY):
        pytest.skip("trainer not present")
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_lr", _TRAINER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _optimizer_with_two_groups():
    T = _trainer()
    args = T.build_parser().parse_args(
        ["--arm", "hier", "--out", "zz", "--opt", "dd", "--lr", "1e-4",
         "--trunk", "timm", "--steps", "30000", "--warmup", "2000"])
    cfg = T._pin_trainer_cfg(T.v3.refc_v3_smoke_config(True), args)
    # the timm trunk needs whole RGB frames; the smoke config ships 1 channel
    cfg.core.encoder.in_channels = 3
    torch.manual_seed(0)
    model = T.v3.RefCV3Model(cfg)
    opt = T.build_optimizer(model, args)
    return T, args, opt


def _named(opt):
    g = {gg.get("name", "?"): float(gg["lr"]) for gg in opt.param_groups}
    enc = [v for k, v in g.items() if "enc" in k.lower()]
    head = [v for k, v in g.items() if "enc" not in k.lower()]
    return enc, head


def _sched(args):
    """Warm-up then cosine — the SHAPE only. ⭐ Irrelevant to the assertion by design: both
    groups receive the same scalar, so the RATIO is invariant to whatever shape runs."""
    w, tot = float(args.warmup), float(args.steps)
    return lambda s: ((s + 1) / w if s < w
                      else 0.5 * (1 + math.cos(math.pi * (s - w) / max(tot - w, 1))))


def _apply_loop_body(opt, args, sched, step):
    """⛔⛔ CALLS THE TRAINER'S OWN FUNCTION. It does NOT transcribe it.

    ⭐ THE FIRST VERSION OF THIS FILE TRANSCRIBED THE LOOP BODY, AND THE MUTATION PROOF
    CAUGHT IT: restoring the real defect in `refc_v3_train.py` (`g["lr"] = args.lr *
    sched(step)`) left the behavioural tests GREEN — only the source-text test went red —
    and removing the `initial_lr` latch was caught by NOTHING. **1 of 3 arms.** That is the
    advisory's class-F shape 3, *"the test STUBS the method under test"*, in its purest
    form: a copy of the code cannot regress when the original does.

    ⇒ the trainer now exposes `apply_lr_schedule` and this calls it. The copy is gone."""
    _trainer().apply_lr_schedule(opt, args, sched, step)


def test_the_optimizer_is_BUILT_with_the_multiplier() -> None:
    """The half that was already true — kept as the control. If this fails, the defect is in
    `build_optimizer` and the schedule is innocent."""
    _T, _args, opt = _optimizer_with_two_groups()
    enc, head = _named(opt)
    assert enc and head, "the `dd` optimiser did not produce an encoder and a head group"
    assert enc[0] / head[0] == pytest.approx(EXPECT_RATIO, abs=1e-12)


def test_the_multiplier_SURVIVES_the_schedule_at_every_step() -> None:
    """⛔ THE DEFECT ITSELF. Before the fix this read 1.0000 at all six steps."""
    _T, args, opt = _optimizer_with_two_groups()
    sched = _sched(args)
    seen = {}
    for st in STEPS:
        _apply_loop_body(opt, args, sched, st)
        enc, head = _named(opt)
        seen[st] = enc[0] / head[0]
    bad = {k: v for k, v in seen.items() if abs(v - EXPECT_RATIO) > 1e-12}
    assert not bad, (
        f"the encoder/head lr ratio left {EXPECT_RATIO} at steps {sorted(bad)}: {seen}. "
        f"The schedule is scaling from `args.lr` instead of each group's own "
        f"`initial_lr`, so the encoder trains at the head rate.")


def test_the_schedule_is_ACTUALLY_MOVING_so_the_ratio_is_not_trivially_constant() -> None:
    """⭐ THE CONTROL THAT MAKES THE TEST ABOVE MEAN SOMETHING.

    A ratio held at 0.5 by a schedule that never changes the learning rate proves nothing —
    the absolute rates must move a lot across these steps for the invariance to have been
    tested at all."""
    _T, args, opt = _optimizer_with_two_groups()
    sched = _sched(args)
    encs = []
    for st in STEPS:
        _apply_loop_body(opt, args, sched, st)
        enc, _head = _named(opt)
        encs.append(enc[0])
    # ⛔ POST-WARM-UP ONLY. The warm-up ramp alone spans several orders of magnitude, so a
    # max/min over ALL steps stays huge even when the post-warm-up schedule is FLAT — the
    # mutation arm `L3` proved that control vacuous. The decay is what must be present.
    post = [e for st, e in zip(STEPS, encs) if st >= int(args.warmup)]
    assert len(post) >= 2, "no post-warm-up steps sampled"
    assert max(post) / max(min(post), 1e-30) > 100.0, (
        f"the encoder lr barely moved AFTER warm-up ({post}) -- a flat schedule makes the "
        f"ratio test vacuous, because a ratio held constant by a constant proves nothing")


def test_the_trainer_ITSELF_carries_the_initial_lr_latch() -> None:
    """⛔ THE TEST ABOVE RUNS A TRANSCRIPTION OF THE LOOP BODY, so it would stay green if the
    trainer regressed and this file did not. This asserts the real source, positively, with a
    same-breath control that the search can find things in that file."""
    src = open(_TRAINER_PY, encoding="utf-8").read()
    assert 'g["initial_lr"] = float(g.get("lr", args.lr))' in src, (
        "the trainer no longer latches each group's own `initial_lr`; the encoder "
        "multiplier will be overwritten on the first step")
    assert 'g["lr"] = g["initial_lr"] * sched(step)' in src, (
        "the trainer is not scaling from `initial_lr`")
    # the control: a string that must be present regardless, so a 0-hit above is about the
    # latch and not about an unreadable file
    assert "def build_optimizer" in src
