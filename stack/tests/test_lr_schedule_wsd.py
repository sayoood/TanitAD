"""refcv9 C6: the WSD learning-rate schedule and the decay branch (``tanitad/train/lr_schedule.py``).

REFCV9_DESIGN sec. 2.6 "Schedule (R6)" + sec. 4.4 E2 / E3 -- the design is PINNED, this file pins the implementation.

WHAT IS PROVEN (every RED arm must FAIL, i.e. the test of it passes only when the broken variant is refused / caught):

(A) THE FACTOR, pure.
    * ``cosine`` (the default) is BIT-IDENTICAL to the lambda it replaced: ``==`` at EVERY step of a 4,000-step run (three
      ``--warmup`` values), against a VERBATIM copy of the old expression.
    * ``wsd`` at the named steps (0, warmup-1, warmup, mid-stable, decay start, last step, steps) for the design's full run
      (``--warmup 1000 --steps 73280 --wsd-decay-steps 12458``) and E2 (``--steps 6000 --wsd-decay-steps 1000``).
    * RED: deleting the stable branch of the factor makes the factor table FAIL; changing the cosine makes the bit-identity FAIL.
(B) THE REFUSALS fire before any data or GPU work: wsd with decay 0, cosine with a decay, warm-up running into the decay,
    unpaired / cosine-with-branch flags, a ``--save-at`` that can never be written.
(C) THE BUILT ``sched`` IS CHECKED AGAINST ARGV (``assert_sched_matches_args``, the G-DVB ``elsewhere`` guard): a sched with a
    wrong decay start is refused.
(D) ``train()`` END TO END on the synthetic rig (CPU): the optimiser's lr at every logged step IS ``initial_lr * factor(step-1)``
    (flag -> namespace -> sched -> ``apply_lr_schedule`` -> the param group); config.json stamps the schedule; a STOPPED WSD run,
    relaunched with the same argv, reproduces the uninterrupted run's rows bit-for-bit and ends on identical tensors.
(E) THE DECAY BRANCH: a 40-step WSD run with a full-state checkpoint at 20, a branch decaying over 10 -> its step-20 factor equals
    the source's, step 30 = 0, the staged checkpoint is the source's bytes, config.json stamps (path, md5, step), the first
    branch row is the source's row bit-for-bit; REFUSED: a warm-up-phase checkpoint, a decay-phase checkpoint, an argv that
    differs in ``--lr`` (the key is NAMED), ``--steps != s + M``, a model-only checkpoint, a cosine source, a ``--out`` that
    already holds another run's ``ckpt.pt``.
(F) REACHABILITY from the caller: the trainer builds its ``sched`` with ``make_sched``, runs the refusals, stages the branch before
    its resume, stamps config.json and writes ``--save-at``; RED: the same pin on the trainer with each line deleted FAILS.
CPU only, synthetic data; nothing touches a pod.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.train import lr_schedule as L                   # noqa: E402

TRAINER = ROOT / "scripts" / "refc_v3_train.py"
LRS_SRC = Path(L.__file__)
BASE_LR = 1e-4            # --lr default; --opt adam => ONE param group, so row["lr"] == BASE_LR * factor


def OLD_SCHED(args):
    """⛔ VERBATIM the ``sched`` lambda of ``refc_v3_train.train`` at the integration tree (tip 1ee6954 + C8 + C3, lines
    10906-10908), the expression C6 replaced. Do not edit: it is the reference the bit-identity is measured against."""
    return lambda s: (s + 1) / max(1, args.warmup) if s < args.warmup else \
        0.5 * (1.0 + math.cos(math.pi * (s - args.warmup)
                              / max(1, args.steps - args.warmup)))   # noqa: E731


def NS(**kw):
    d = dict(lr_schedule="cosine", warmup=2000, steps=30000, wsd_decay_steps=0, wsd_branch_from=None, wsd_branch_decay=0,
             save_at=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


# =========================================================================== #
# (A) the factor                                                              #
# =========================================================================== #
@pytest.mark.parametrize("warmup", [2000, 100, 0])
def test_A1_cosine_is_BIT_IDENTICAL_to_the_old_lambda_at_every_step_of_a_4000_step_run(warmup):
    args = NS(steps=4000, warmup=warmup)
    new, old = L.make_sched(args), OLD_SCHED(args)
    for s in range(0, 4001):
        assert new(s) == old(s), (warmup, s, new(s), old(s))


def test_A1b_the_parser_default_is_cosine_and_reads_the_run_args_at_call_time():
    a = T.build_parser().parse_args(["--arm", "hier", "--out", "x"])
    assert (a.lr_schedule, a.wsd_decay_steps, a.wsd_branch_from, a.wsd_branch_decay, a.save_at) == ("cosine", 0, None, 0, None)
    sched = L.make_sched(a)
    a.steps, a.warmup = 4000, 100                       # mutated AFTER make_sched: the old lambda read args at call time
    assert all(sched(s) == OLD_SCHED(a)(s) for s in range(0, 4001))
    assert L.lr_factor(7, types.SimpleNamespace(warmup=10, steps=100)) == OLD_SCHED(NS(warmup=10, steps=100))(7)   # an old namespace


def check_wsd_factor_table(mod):
    """The factor table of the design's runs, against CLOSED FORMS (shared code with nothing in ``mod``)."""
    full = NS(lr_schedule="wsd", warmup=1000, steps=73280, wsd_decay_steps=12458)
    f = mod.make_sched(full)
    start = 73280 - 12458                               # 60822: the decay start
    rows = [(0, 1 / 1000), (999, 1.0), (1000, 1.0), (30000, 1.0), (start - 1, 1.0), (start, 1.0),
            (start + 1, (12458 - 1) / 12458), (73279, 1 / 12458), (73280, 0.0)]
    for s, want in rows:
        assert f(s) == pytest.approx(want, abs=1e-15, rel=0), (s, f(s), want)
    assert f(start) == 1.0 and f(73280) == 0.0 and f(73279) > 0.0           # EXACT at the two junctions
    assert f(start + 6229) == pytest.approx(0.5, abs=1e-12)                 # linear: the midpoint of the decay
    e2 = mod.make_sched(NS(lr_schedule="wsd", warmup=1000, steps=6000, wsd_decay_steps=1000))
    for s, want in [(0, 1e-3), (999, 1.0), (1000, 1.0), (3000, 1.0), (4999, 1.0), (5000, 1.0), (5001, 0.999), (5999, 0.001),
                    (6000, 0.0)]:
        assert e2(s) == pytest.approx(want, abs=1e-15, rel=0), (s, e2(s), want)
    vals = [f(s) for s in range(1000, 73281)]                              # stable, then strictly down, never up, never < 0
    assert all(b <= a for a, b in zip(vals, vals[1:])) and min(vals) == 0.0
    assert [f(s) for s in range(0, 1000)] == [(s + 1) / 1000 for s in range(0, 1000)]       # the SAME linear warm-up as the cosine


def test_A2_the_wsd_factor_table():
    check_wsd_factor_table(L)


def test_A3_the_cosine_factor_table_and_the_phases():
    f = L.make_sched(NS(warmup=1000, steps=73280))
    assert f(0) == 1 / 1000 and f(999) == 1.0 and f(1000) == 1.0 and f(73280) == 0.0
    assert 0 < f(73279) < 1e-6 and f(1000 + 36140) == pytest.approx(0.5, abs=1e-12)
    for s, want in [(0, "warmup"), (999, "warmup"), (1000, "stable"), (60821, "stable"), (60822, "decay"), (73280, "decay")]:
        assert L.phase_of(s, 1000, 73280, 12458) == want, s


def _load_mutated(old, new):
    """lr_schedule.py with ``old`` replaced by ``new`` (must occur exactly once), as a fresh module."""
    text = LRS_SRC.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    mod = types.ModuleType("lr_schedule_mutated")
    mod.__file__ = str(LRS_SRC)
    exec(compile(text.replace(old, new), "lr_schedule_mutated", "exec"), mod.__dict__)
    return mod


def test_A4_RED_deleting_the_stable_branch_of_the_factor_FAILS_the_factor_table():
    check_wsd_factor_table(L)                                                    # the control: the real module passes
    broken = _load_mutated("    if step < steps - decay:\n        return 1.0\n", "")
    with pytest.raises(AssertionError):
        check_wsd_factor_table(broken)


def test_A5_RED_a_changed_cosine_FAILS_the_bit_identity():
    broken = _load_mutated("0.5 * (1.0 + math.cos(", "0.5 * (1.0000001 + math.cos(")
    args = NS(steps=4000, warmup=100)
    bad = [s for s in range(0, 4001) if broken.make_sched(args)(s) != OLD_SCHED(args)(s)]
    assert bad, "the bit-identity test cannot see a perturbed cosine"
    assert [s for s in range(0, 4001) if L.make_sched(args)(s) != OLD_SCHED(args)(s)] == []     # control


# =========================================================================== #
# (B) refusals                                                                #
# =========================================================================== #
@pytest.mark.parametrize("kw,needle", [
    (dict(lr_schedule="wsd", wsd_decay_steps=0), "wsd-decay-steps N > 0"),                          # RED: decay 0 with wsd
    (dict(lr_schedule="wsd", wsd_decay_steps=-5), "wsd-decay-steps N > 0"),
    (dict(lr_schedule="cosine", wsd_decay_steps=100), "change nothing"),                            # the dead flag
    (dict(lr_schedule="wsd", wsd_decay_steps=40000), "> --steps"),
    (dict(lr_schedule="wsd", wsd_decay_steps=29000, warmup=2000), "no stable phase"),               # warm-up into the decay
    (dict(lr_schedule="cosine", wsd_branch_from="x.pt", wsd_branch_decay=10), "need --lr-schedule wsd"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, wsd_branch_from="x.pt", wsd_branch_decay=0), "go together"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, wsd_branch_decay=5), "go together"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, save_at="0"), "--save-at"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, save_at="30001"), "--save-at"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, save_at="5,5"), "duplicate"),
    (dict(lr_schedule="wsd", wsd_decay_steps=100, save_at="a"), "comma list"),
])
def test_B1_RED_each_bad_argv_is_refused(kw, needle):
    with pytest.raises(SystemExit) as e:
        L.check_args(NS(**kw))
    assert needle in str(e.value), str(e.value)


def test_B2_the_good_argvs_pass():
    L.check_args(NS())                                                                           # the defaults
    L.check_args(NS(lr_schedule="wsd", warmup=1000, steps=73280, wsd_decay_steps=12458, save_at="30000,60000"))
    L.check_args(NS(lr_schedule="wsd", warmup=1000, steps=6000, wsd_decay_steps=1000))           # E2
    L.check_args(NS(lr_schedule="wsd", warmup=1000, steps=36000, wsd_decay_steps=12458,          # a D1 branch: N is the SOURCE's
                    wsd_branch_from="x.pt", wsd_branch_decay=6000))


# =========================================================================== #
# (C) the built sched is held against argv                                    #
# =========================================================================== #
def test_C1_assert_sched_matches_args_passes_the_real_sched_and_REFUSES_a_wrong_one():
    for a in (NS(), NS(lr_schedule="wsd", warmup=1000, steps=73280, wsd_decay_steps=12458),
              NS(lr_schedule="wsd", warmup=3, steps=12, wsd_decay_steps=4), NS(steps=1)):      # the last: a CI smoke
        probe = L.assert_sched_matches_args(L.make_sched(a), a)
        assert probe and all(isinstance(k, str) for k in probe)
    a = NS(lr_schedule="wsd", warmup=1000, steps=73280, wsd_decay_steps=12458)
    for wrong in (lambda s: L.wsd_factor(s, 1000, 73280, 12000),          # a decay that is not the declared one
                  lambda s: L.cosine_factor(s, 1000, 73280),               # the cosine under a wsd declaration
                  lambda s: 1.0):                                          # a schedule that never decays
        with pytest.raises(SystemExit) as e:
            L.assert_sched_matches_args(wrong, a)
        assert "NOT THE ONE" in str(e.value)


# =========================================================================== #
# (D) train() end to end                                                      #
# =========================================================================== #
def _argv(out, steps, *extra, warmup=3, sched=("--lr-schedule", "wsd", "--wsd-decay-steps", "4")):
    return ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2", "--steps", str(steps), "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "100", "--warmup", str(warmup), *sched, *extra]


def _run(argv):
    args = T.build_parser().parse_args(argv)
    args._ew_argv = list(argv)                            # what main() does
    return T.train(args)


def _rows(out):
    return [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and '"lr"' in l]


def _cfg(out):
    return json.loads((Path(out) / "config.json").read_text(encoding="utf-8"))


def test_D1_the_optimiser_lr_IS_initial_lr_times_the_wsd_factor_at_every_step(tmp_path):
    """flag -> namespace -> sched -> apply_lr_schedule -> the param group. Row `step=k` carries the lr of the update that
    produced it, i.e. factor(k-1)."""
    _run(_argv(tmp_path / "w", 12))
    a = NS(lr_schedule="wsd", warmup=3, steps=12, wsd_decay_steps=4)
    rows = _rows(tmp_path / "w")
    assert [r["step"] for r in rows] == list(range(1, 13))
    assert [r["lr"] for r in rows] == [BASE_LR * L.wsd_factor(k - 1, 3, 12, 4) for k in range(1, 13)]
    assert [round(r["lr"] / BASE_LR, 9) for r in rows] == [round(x, 9) for x in
                                                           [1 / 3, 2 / 3, 1, 1, 1, 1, 1, 1, 1, .75, .5, .25]]
    c = _cfg(tmp_path / "w")
    assert (c["lr_schedule"], c["wsd_decay_steps"], c["wsd_branch"]) == ("wsd", 4, None)
    assert c["lr_factor_probe"]["11"] == 0.25 and c["lr_factor_probe"]["12"] == 0.0 and c["argv_effective"][0] == "--arm"
    assert c["declared_vs_built"]["mismatches"] == 0


def test_D2_the_DEFAULT_run_is_the_cosine_and_its_rows_equal_the_old_expression(tmp_path):
    _run(_argv(tmp_path / "c", 6, sched=()))
    rows = _rows(tmp_path / "c")
    old = OLD_SCHED(NS(warmup=3, steps=6))
    assert [r["lr"] for r in rows] == [BASE_LR * old(k - 1) for k in range(1, 7)]
    c = _cfg(tmp_path / "c")
    assert (c["lr_schedule"], c["wsd_decay_steps"], c["wsd_branch"]) == ("cosine", 0, None)


def _patch_batches(monkeypatch, stop_after=None, reseed=True):
    """Wrap the loop's batch fetch. ``reseed``: after each fetch, seed the GLOBAL torch RNG from the data position (a pure
    function of it, and the position is what a checkpoint resumes) -- see D3. ``stop_after=n``: the (n+1)-th fetch raises
    ``KeyboardInterrupt`` (a stopped run). Returns the undo."""
    real, calls = T.next_train_batch, {"n": 0}

    def _wrapped(it, dl, s, dpos):
        calls["n"] += 1
        if stop_after is not None and calls["n"] > stop_after:
            raise KeyboardInterrupt("simulated stop")
        it, batch = real(it, dl, s, dpos)
        if reseed:
            torch.manual_seed(7919 * (int(dpos["epoch"]) + 1) + int(dpos["batch"]))
        return it, batch
    monkeypatch.setattr(T, "next_train_batch", _wrapped)
    return lambda: monkeypatch.setattr(T, "next_train_batch", real)


def _clean(r):
    """A row without the two keys that are NOT numerical state: ``elapsed_s`` (wall clock) and ``goal_gate_grad`` (a diagnostic
    that reads the PREVIOUS step's ``.grad``, so it is absent at the first step of every launch -- MEASURED: it is the only
    key the stopped-and-relaunched run lacks at step 5)."""
    return {k: v for k, v in r.items() if k not in ("elapsed_s", "goal_gate_grad")}


def _stopped_and_uninterrupted(tmp_path, monkeypatch, reseed):
    argv_u, argv_r = _argv(tmp_path / "u", 8, "--save-every", "4"), _argv(tmp_path / "r", 8, "--save-every", "4")
    undo = _patch_batches(monkeypatch, reseed=reseed)
    _run(argv_u)
    undo()
    undo = _patch_batches(monkeypatch, stop_after=4, reseed=reseed)      # the 5th fetch dies: steps 1-4 done, ckpt.pt at step 4
    with pytest.raises(KeyboardInterrupt):
        _run(argv_r)
    undo()
    assert torch.load(tmp_path / "r" / "ckpt.pt", weights_only=False)["step"] == 4
    undo = _patch_batches(monkeypatch, reseed=reseed)
    _run(argv_r)                                                          # relaunch: the SAME argv resumes at 4
    undo()
    ru, rr = _rows(tmp_path / "u"), _rows(tmp_path / "r")
    assert [r["step"] for r in rr] == list(range(1, 9)) == [r["step"] for r in ru]
    return ru, rr


def test_D3_a_STOPPED_wsd_run_relaunched_with_the_same_argv_reproduces_the_uninterrupted_run_bit_for_bit(tmp_path, monkeypatch):
    """The state a resume carries (model, optimiser, data position) plus the PURE factor reproduce the trajectory: every row
    of steps 5-8 (every loss scalar, lr, data position) and every final tensor are equal. The global RNG is pinned per step
    (``_patch_batches(reseed=True)``): written when the trainer did NOT restore it (see D3b, which now pins that it does) -- kept, it pins the schedule independently of the RNG."""
    ru, rr = _stopped_and_uninterrupted(tmp_path, monkeypatch, reseed=True)
    assert [_clean(r) for r in rr] == [_clean(r) for r in ru], "the resumed rows are not the uninterrupted rows"
    assert [r["lr"] for r in rr] == [BASE_LR * L.wsd_factor(k - 1, 3, 8, 4) for k in range(1, 9)]
    cu, cr = (torch.load(tmp_path / x / "ckpt.pt", weights_only=False) for x in ("u", "r"))
    assert cu["step"] == cr["step"] == 8
    assert all(torch.equal(cu["model"][k], cr["model"][k]) for k in cu["model"])
    assert _cfg(tmp_path / "r")["data_order"]["resumed_at_step"] == 4


def test_D3b_the_RNG_resumes_too__a_relaunch_equals_the_uninterrupted_run_WITHOUT_the_per_step_reseed(tmp_path, monkeypatch):
    """WAS the control that PINNED THE DEFECT (MEASURED 2026-11-03, CPU smoke rig, WSD, stop at step 4: without the per-step
    reseed the resumed run's losses differed from step 5 on, 92.78 vs 148.98, because ``train()`` re-seeded the global RNG from
    ``--seed`` at every launch and ``ckpt.pt`` carried no RNG state). RNG-RESUME (2026-11-03, ``tests/test_resume_rng.py``) puts
    the streams in the checkpoint and restores them last, so this control FLIPS -- as its own docstring said it would: with NO
    reseed the resumed rows (lr, data position, every loss scalar) and the final tensors are the uninterrupted run's. The reseed
    in D3 is now redundant (kept: it pins the schedule independently of the RNG)."""
    ru, rr = _stopped_and_uninterrupted(tmp_path, monkeypatch, reseed=False)
    assert [r["lr"] for r in rr] == [r["lr"] for r in ru]
    assert [_clean(r) for r in rr[:4]] == [_clean(r) for r in ru[:4]]
    assert [_clean(r) for r in rr[4:]] == [_clean(r) for r in ru[4:]]
    cu, cr = (torch.load(tmp_path / x / "ckpt.pt", weights_only=False) for x in ("u", "r"))
    assert all(torch.equal(cu["model"][k], cr["model"][k]) for k in cu["model"])


# =========================================================================== #
# (E) the decay branch                                                        #
# =========================================================================== #
@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """ONE 40-step WSD source run: warm-up 5, stable 5..29, decay 30..39 (N=10); full-state checkpoints at 3 (warm-up),
    20 (STABLE) and 35 (decay); a model-only snapshot at 20; an EMA so the branch has one to carry."""
    out = tmp_path_factory.mktemp("c6src") / "src"
    sched = ("--lr-schedule", "wsd", "--wsd-decay-steps", "10")
    argv = _argv(out, 40, "--save-at", "3,20,35", "--snapshot-steps", "20", "--ema-decay", "0.9", warmup=5, sched=sched)
    with pytest.MonkeyPatch.context() as mp:
        _patch_batches(mp, reseed=True)             # the RNG is pinned per step so E2 can compare rows bit-for-bit (see D3b)
        _run(argv)
    return types.SimpleNamespace(out=out, argv=argv, sched=sched, ck20=out / "ckpt_full_20.pt")


def _branch_argv(src, out, steps=30, m=10, ck=None, extra=()):
    argv = _argv(out, steps, "--save-at", "3,20,35", "--snapshot-steps", "20", "--ema-decay", "0.9", warmup=5, sched=src.sched)
    i = argv.index("--save-at")
    del argv[i:i + 2]                                    # --save-at is the one cadence flag a branch may differ in
    return argv + ["--wsd-branch-from", str(ck or src.ck20), "--wsd-branch-decay", str(m), *extra]


def test_E1_the_source_run_wrote_full_state_checkpoints_at_the_named_steps(src):
    ck = torch.load(src.ck20, weights_only=False)
    live = torch.load(src.out / "ckpt.pt", weights_only=False)
    assert ck["step"] == 20 and live["step"] == 40
    assert set(ck) == set(live) >= {"model", "opt", "step", "data_pos", "param_names", "ema"}      # the resumable payload, key for key
    assert ck["ema"]["num_updates"] == 20
    for s in (3, 35):
        assert (src.out / f"ckpt_full_{s}.pt").exists()
    assert set(torch.load(src.out / "ckpt_20.pt", weights_only=False)) == {"model", "step"}        # the MODEL-ONLY snapshot: cannot seed a branch
    assert not (src.out / "ckpt_full_21.pt").exists()


def test_E2_the_branch_continues_the_source_then_decays_to_zero(src, tmp_path, monkeypatch):
    br = tmp_path / "br"
    argv = _branch_argv(src, br)
    _patch_batches(monkeypatch, reseed=True)
    _run(argv)
    rs, rb = _rows(src.out), _rows(br)
    assert [r["step"] for r in rb] == list(range(21, 31))
    # the factor: step-20 equals the source's, step 30 = 0, linear between
    a_src = NS(lr_schedule="wsd", warmup=5, steps=40, wsd_decay_steps=10)
    a_br = NS(lr_schedule="wsd", warmup=5, steps=30, wsd_decay_steps=10, wsd_branch_from="x", wsd_branch_decay=10)
    assert L.make_sched(a_br)(20) == L.make_sched(a_src)(20) == 1.0
    assert L.make_sched(a_br)(30) == 0.0 and L.make_sched(a_br)(29) == pytest.approx(0.1)
    # the optimiser's lr: step 21 is the source's step 21 BIT FOR BIT, then it decays (the source stays at the peak until 30)
    assert rb[0]["lr"] == rs[20]["lr"] == BASE_LR
    assert [r["lr"] for r in rb] == [BASE_LR * ((30 - (k - 1)) / 10) for k in range(21, 31)]
    assert rb[-1]["lr"] == pytest.approx(BASE_LR * 0.1) and rb[-1]["lr"] < rs[29]["lr"] == BASE_LR
    # the FULL state resumed: the first row (same state, same data position, same lr) IS the source's row
    assert _clean(rb[0]) == _clean(rs[20]), "the branch did not resume the source's state at step 20"
    # the stamps
    c = _cfg(br)
    wb = c["wsd_branch"]
    assert (wb["step"], wb["branch_decay"], wb["steps"], wb["path"], wb["staged"]) == (20, 10, 30, str(src.ck20), True)
    assert wb["md5"] == L.file_md5(src.ck20) and wb["source_run"]["phase_of_checkpoint"] == "stable"
    assert (c["lr_schedule"], c["wsd_decay_steps"]) == ("wsd", 10)
    assert c["data_order"]["resumed_at_step"] == 20
    fin = torch.load(br / "ckpt.pt", weights_only=False)
    assert fin["step"] == 30 and fin["ema"]["num_updates"] == 30                  # the EMA CONTINUED (20 -> 30), not restarted
    assert json.loads((br / L.BRANCH_RECORD).read_text(encoding="utf-8"))["md5"] == wb["md5"]
    assert L.file_md5(src.ck20) == wb["md5"]                                      # the source checkpoint is untouched


def test_E3_the_staged_checkpoint_is_the_sources_bytes_and_a_relaunch_of_the_branch_does_not_restage(src, tmp_path):
    br = tmp_path / "br"
    parser = T.build_parser()
    args = parser.parse_args(_branch_argv(src, br))
    pre = L.branch_preflight(args, parser)
    rec = L.stage_branch(args, br, pre)
    assert rec["staged"] is True and L.file_md5(br / "ckpt.pt") == L.file_md5(src.ck20) == rec["md5"]
    again = L.stage_branch(args, br, pre)                                          # the out dir now holds THIS branch's ckpt.pt
    assert again["staged"] is False and again["md5"] == rec["md5"]


def _refused(argv, needle):
    parser = T.build_parser()
    with pytest.raises(SystemExit) as e:
        args = parser.parse_args(argv)
        L.check_args(args)
        pre = L.branch_preflight(args, parser)
        L.stage_branch(args, Path(args.out), pre)
    assert needle in str(e.value), str(e.value)
    return str(e.value)


def test_E4_RED_a_branch_from_a_WARMUP_checkpoint_is_refused(src, tmp_path):
    msg = _refused(_branch_argv(src, tmp_path / "b", steps=13, ck=src.out / "ckpt_full_3.pt"), "WARMUP phase")
    assert "step 3" in msg
    assert not (tmp_path / "b" / "ckpt.pt").exists()                               # nothing staged


def test_E5_RED_a_branch_from_a_DECAY_checkpoint_is_refused(src, tmp_path):
    _refused(_branch_argv(src, tmp_path / "b", steps=45, ck=src.out / "ckpt_full_35.pt"), "DECAY phase")
    _refused(_branch_argv(src, tmp_path / "b2", steps=50, ck=src.out / "ckpt.pt"), "DECAY phase")      # the finished run's own


def test_E6_RED_an_argv_that_differs_in_lr_is_refused_and_the_key_is_NAMED(src, tmp_path):
    msg = _refused(_branch_argv(src, tmp_path / "b", extra=("--lr", "0.0002")), "--lr")
    assert "0.0002" in msg and "0.0001" in msg
    msg = _refused(_branch_argv(src, tmp_path / "b2", extra=("--seed", "7", "--batch", "4")), "--seed")
    assert "--batch" in msg                                                         # EVERY differing key is named
    _refused(_branch_argv(src, tmp_path / "b3") + ["--wsd-decay-steps", "9"], "--wsd-decay-steps")   # the SOURCE's N is part of the argv


def test_E7_RED_steps_must_be_the_source_step_plus_M(src, tmp_path):
    msg = _refused(_branch_argv(src, tmp_path / "b", steps=31), "--steps 31")
    assert "pass --steps 30" in msg


def test_E8_RED_a_model_only_checkpoint_cannot_seed_a_branch(src, tmp_path):
    _refused(_branch_argv(src, tmp_path / "b", ck=src.out / "ckpt_20.pt"), "model-only")


def test_E9_RED_a_cosine_source_has_no_stable_phase(tmp_path, src):
    cos = tmp_path / "cos"
    _run(_argv(cos, 6, "--save-at", "3", sched=()))
    argv = _argv(tmp_path / "b", 9, sched=()) + ["--lr-schedule", "wsd", "--wsd-decay-steps", "3",
                                                 "--wsd-branch-from", str(cos / "ckpt_full_3.pt"), "--wsd-branch-decay", "6"]
    _refused(argv, "--lr-schedule")                                                 # argv differs (cosine vs wsd) -> named first
    # defence in depth (a hand-edited config.json): the checkpoint-side check refuses a cosine source on its own
    cfg_ns = T.build_parser().parse_args(_argv(cos, 6, sched=()))
    with pytest.raises(SystemExit) as e:
        L.check_branch_checkpoint(NS(wsd_branch_decay=6, steps=9), cfg_ns, 3)
    assert "no stable phase" in str(e.value)


def test_E10_RED_an_out_dir_holding_another_runs_ckpt_is_refused(src, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "ckpt.pt").write_bytes((src.ck20).read_bytes())                        # a ckpt.pt, but NO branch record
    _refused(_branch_argv(src, other), "NOT this branch's")


def test_E11_RED_a_save_at_at_or_before_the_branch_start_is_refused(src, tmp_path):
    argv = _branch_argv(src, tmp_path / "b") + ["--save-at", "20"]
    _refused(argv, "never written")


def test_E12_end_to_end_a_refused_branch_stops_train_before_any_step(src, tmp_path):
    """The refusal is reached from train() itself (not only from the module's functions)."""
    with pytest.raises(SystemExit) as e:
        _run(_branch_argv(src, tmp_path / "b", extra=("--lr", "0.0002")))
    assert "--lr" in str(e.value) and not (tmp_path / "b" / "metrics.jsonl").exists()
    with pytest.raises(SystemExit) as e:
        _run(_branch_argv(src, tmp_path / "b2", steps=13, ck=src.out / "ckpt_full_3.pt"))
    assert "INSIDE THE WARM-UP" in str(e.value) and not (tmp_path / "b2" / "metrics.jsonl").exists()
    with pytest.raises(SystemExit) as e:                                            # RED: wsd with decay 0, from train()
        _run(_argv(tmp_path / "z", 5, sched=("--lr-schedule", "wsd")))
    assert "wsd-decay-steps N > 0" in str(e.value) and not (tmp_path / "z").exists()


# =========================================================================== #
# (F) reachability from the caller                                            #
# =========================================================================== #
def pins(text: str) -> dict:
    """Each line of the trainer that makes the schedule REACHABLE, counted in ``text``."""
    i_stage, i_resume = text.find("_lrs.stage_branch(args, out_dir, _wsd_src)"), text.find("    if ck.exists():\n        state = torch.load(ck")
    return {
        "import": text.count("from tanitad.train import lr_schedule as _lrs"),
        "G-DVB register": text.count("_lrs.register_dvb_levers(_dvb.register)"),
        "check_args": text.count("    _lrs.check_args(args)"),
        "preflight": text.count("_wsd_src = _lrs.branch_preflight(args, build_parser())"),
        "make_sched": text.count("    sched = _lrs.make_sched(args)"),
        "sched guard": text.count("_lrs.assert_sched_matches_args(sched, args)"),
        "stage before resume": int(0 < i_stage < i_resume),
        "config stamp": text.count("**_lrs.config_stamp(args, _wsd_branch)"),
        "save-at parsed": text.count("_save_at = _lrs.parse_save_at("),
        "save-at written": text.count("if step in _save_at:"),
        "flags": sum(text.count(f) for f in ('"--lr-schedule"', '"--wsd-decay-steps"', '"--wsd-branch-from"',
                                              '"--wsd-branch-decay"', '"--save-at"')),
        "old lambda gone": int("sched = lambda s:" not in text),
    }


def _all_pinned(p):
    return all(v >= 1 for v in p.values()) and p["import"] == p["make_sched"] == p["check_args"] == 1 and p["flags"] == 5


def test_F1_the_trainer_reaches_every_part_of_the_schedule():
    p = pins(TRAINER.read_text(encoding="utf-8"))
    assert _all_pinned(p), p


@pytest.mark.parametrize("line", [
    "    sched = _lrs.make_sched(args)          # refcv9 C6: cosine (bit-identical to the lambda this replaced) | wsd\n",
    "    _lrs.check_args(args)                  # refcv9 C6: the schedule / branch / save-at refusals (defaults: nothing)\n",
    "    _wsd_branch = _lrs.stage_branch(args, out_dir, _wsd_src)",
    "        if step in _save_at:",
    "        **_lrs.config_stamp(args, _wsd_branch)",
    "_lrs.register_dvb_levers(_dvb.register)",
])
def test_F2_RED_deleting_any_one_line_from_the_trainer_FAILS_the_reachability_pin(line):
    text = TRAINER.read_text(encoding="utf-8")
    assert text.count(line) == 1, line
    assert _all_pinned(pins(text))                                                  # the control
    assert not _all_pinned(pins(text.replace(line, "    pass  # deleted\n" if line.endswith("\n") else "pass  # deleted")))


def test_F3_the_G_DVB_registry_holds_the_five_levers_and_two_way_pin_against_the_parser_still_holds():
    from tanitad.train import declared_vs_built as dvb
    parser_dests = {a.dest for a in T.build_parser()._actions}
    for d in ("lr_schedule", "wsd_decay_steps", "wsd_branch_from", "wsd_branch_decay", "save_at"):
        assert d in dvb.REGISTRY and d in parser_dests, d
