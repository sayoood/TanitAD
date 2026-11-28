"""refcv8 P9 (MM ruling Q1; SPEC_REFCV8 sec. 3.1 I-2; MM item 3, 2026-10-04) -- `--r8-alloc-emit-start S_emit`.

With a warm start, the extra candidates (allocated / prior-free) are generated and trained from step 0 but may WIN the
argmax only from step S_emit, so step 0 emits refcv7's plan. Identity is checked with emission OFF on the I-2 literals
(sel_idx 100 %, max |dtraj| <= 1e-3 m, max |d base score| <= 1e-4); an argv emitting at step 0 FAILS the identity row.

Pinned:
1. the schedule literal (step S-1 off, S on) and the decoder switch: emission OFF keeps every pick on the base fan,
   ON lets an extra win;
2. I-2 on the rig: the refcv8 build (allocation + prior-free, emission scheduled) at step 0 against refcv7 reads the
   literals -- bit-identical in fact, the split decode -- and PASSES; the SAME build emitting from step 0 FAILS the row;
3. the row's literal cases (each literal broken once);
4. the argv: >= 1, needs an emit flag, a warm start emitting from step 0 is refused, the tiny arms may emit from 0;
5. the loader's step: a refcv7 checkpoint is refcv8 step 0, a refcv8 checkpoint its own step;
6. G-DVB.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

S_EMIT = 2000


def test_the_schedule_literal():
    assert [RT.emit_live(s, S_EMIT) for s in (0, 1, 1999, 2000, 2001)] == [False, False, False, True, True]
    assert RT.emit_live(0, 0) is True                                          # no schedule: from the first step
    assert RT.I2_LITERALS == {"sel_idx_frac": 1.0, "max_dtraj_m": 1e-3, "max_dbase_score": 1e-4}


@pytest.fixture(scope="module")
def rigs():
    pytest.importorskip("timm")
    T = R.trainer()
    cfg7, m7 = R.build(T, False)
    kw = dict(n_alloc=8, alloc_emit=True, prior_free_group=True, prior_free_emit=True)
    cfg8, m8 = R.build(T, True, emit_start=S_EMIT, **kw)
    R.copy_into(m7, m8)
    cfg0, m0 = R.build(T, True, emit_start=0, **kw)                           # the same build, emitting from 0
    R.copy_into(m7, m0)
    return T, (cfg7, m7), (cfg8, m8), (cfg0, m0)


def _rows(cfg7, m7, cfg8, m8, seeds=(1, 2, 3)):
    rows = []
    for sd in seeds:
        bt = R.batch(cfg7, b=3, seed=sd)
        o7 = R.forward(cfg7, m7, bt, seed=10 + sd)
        o8 = R.forward(cfg8, m8, bt, seed=10 + sd)
        nb = int(o8.get("r8_n_base", o7["anchor_traj"].shape[1]))
        for i in range(o7["traj"].shape[0]):
            rows.append({"sel_idx_equal": bool(o7["sel_idx"][i] == o8["sel_idx"][i]),
                         "max_abs_dtraj_m": float((o7["traj"][i] - o8["traj"][i]).abs().max()),
                         "base_score_max_abs_diff": float((o7["sel_score_v3"][i]
                                                           - o8["sel_score_v3"][i, :nb]).abs().max()),
                         "picked_extra": bool(int(o8["sel_idx"][i]) >= nb)})
    return rows


def test_I2_emission_off_at_step0_PASSES_and_the_step0_emitting_argv_FAILS(rigs):
    T, (cfg7, m7), (cfg8, m8), (cfg0, m0) = rigs
    assert RT.apply_emit_schedule(m8, 0) is False and m8.core.decoder.r8_emit_live is False
    assert RT.apply_emit_schedule(m0, 0) is True
    rows8 = _rows(cfg7, m7, cfg8, m8)
    row8 = RT.i2_identity_row(rows8, emit_at_step0=RT.emits_at_step0(cfg8.refcv8))
    assert row8["PASS"], row8["reasons"]
    assert row8["max_abs_dtraj_m"] == 0.0 and row8["max_base_score_abs_diff"] == 0.0     # bit-identical in fact
    rows0 = _rows(cfg7, m7, cfg0, m0)
    row0 = RT.i2_identity_row(rows0, emit_at_step0=RT.emits_at_step0(cfg0.refcv8))
    assert not row0["PASS"] and row0["emit_at_step0"] is True                  # FAILS by the ruling, whatever it reads


def test_the_switch_keeps_every_pick_on_the_base_fan_while_off(rigs):
    T, (cfg7, m7), (cfg8, m8), (cfg0, m0) = rigs
    RT.apply_emit_schedule(m8, 0)
    off = _rows(cfg7, m7, cfg8, m8)
    assert not any(r["picked_extra"] for r in off)
    RT.apply_emit_schedule(m8, S_EMIT)
    bt = R.batch(cfg8, b=3, seed=1)
    o = R.forward(cfg8, m8, bt, seed=11)
    ek = o["r8_emit_keep"]
    nb = int(o["r8_n_base"])
    assert bool(ek[:, nb:].all())                                              # every extra may now be emitted
    RT.apply_emit_schedule(m8, 0)
    o = R.forward(cfg8, m8, bt, seed=11)
    assert not bool(o["r8_emit_keep"][:, nb:].any()) and bool(o["r8_emit_keep"][:, :nb].all())


@pytest.mark.parametrize("mut,needle", [
    (dict(sel_idx_equal=False), "sel_idx"),
    (dict(max_abs_dtraj_m=2e-3), "dtraj"),
    (dict(base_score_max_abs_diff=2e-4), "base score"),
])
def test_each_I2_literal_fails_its_violation(mut, needle):
    ok = {"sel_idx_equal": True, "max_abs_dtraj_m": 9e-4, "base_score_max_abs_diff": 9e-5}
    assert RT.i2_identity_row([ok, ok], emit_at_step0=False)["PASS"]
    bad = RT.i2_identity_row([ok, dict(ok, **mut)], emit_at_step0=False)
    assert not bad["PASS"] and any(needle in r for r in bad["reasons"])
    assert not RT.i2_identity_row([], emit_at_step0=False)["PASS"]


def _pin(argv):
    tr = R.trainer()
    args = tr.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=C.R8Config())
    tr._pin_refcv8(cfg, args)
    return cfg.refcv8


ON = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05",
      "--r8-n-alloc", "8", "--w-r8-alloc-l1", "1.0"]


def test_the_argv_rules():
    assert _pin(ON + ["--r8-alloc-emit", "--r8-alloc-emit-start", "2000", "--init-from", "c.pt"]).emit_start == 2000
    assert _pin(ON + ["--r8-alloc-emit"]).emit_start == 0                     # a tiny arm: no warm start, emit from 0
    for argv, needle in ((ON + ["--r8-alloc-emit", "--init-from", "c.pt"], "WARM-STARTED"),
                         (ON + ["--r8-alloc-emit", "--r8-alloc-emit-start", "0"], ">= 1"),
                         (ON + ["--r8-alloc-emit-start", "2000"], "never switched on"),
                         (["--arm", "hier", "--out", "X", "--r8-alloc-emit-start", "2000"], "without --refcv8")):
        with pytest.raises(SystemExit) as e:
            _pin(argv)
        assert needle in str(e.value), str(e.value)


def test_the_loader_step_of_a_checkpoint():
    assert RT.checkpoint_r8_step(["core.decoder.anchor_controls", "core.encoder.w"], 50400) == 0   # refcv7 = warm start
    assert RT.checkpoint_r8_step(["core.decoder.r8_mod.w", "core.encoder.w"], 3000) == 3000
    assert RT.checkpoint_r8_step(["core.decoder.r8_mod.w"], None) == 0


def test_G_DVB_reads_the_start_off_the_built_config():
    from tanitad.train import declared_vs_built as dvb
    assert dvb.REGISTRY["r8_alloc_emit_start"].kind == "built"
