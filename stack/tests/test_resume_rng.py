"""RNG-RESUME: a relaunched ``refc_v3_train.py`` run must EQUAL the uninterrupted run (MM 207ebd, 2026-11-03).

THE DEFECT (MEASURED by C6, ``tests/test_lr_schedule_wsd.py::test_D3b``): ``ckpt.pt`` carried no RNG state and ``train()``
re-seeds from ``--seed`` at launch, so the losses of a resumed run differed from the uninterrupted run's from the first step
after the resume (92.78 vs 148.98 at step 5, same lr). The data ORDER already resumed (``data_pos``); the stochastic draws
(diffusion noise, dropout, the DataLoader iterator's base seed, any model-owned generator) did not. Every arm under
``supervise_run.sh`` (MAX_RELAUNCH=4) silently became a different run after a relaunch -- training variance the paired
episode bootstrap cannot see (H-ESTIM-SEED-1).

THE FIX (``scripts/refc_v3_train.py``): every resumable checkpoint (``ckpt.pt``, ``ckpt_full_<s>.pt``) carries ``ckpt["rng"]``
(torch CPU + CUDA, numpy, python, every model-owned dedicated generator); ``train()`` restores it LAST, after every launch-time
seeding / consumption and right before the first step; the aggregate in-training eval (which draws from the global stream and
runs AFTER the checkpoint is written) is stream-neutral (``EVAL_STREAM_NEUTRAL``); an OLD checkpoint resumes with a WARNING.

What is pinned here (CPU tiny rig; ``--smoke --synth-episodes 2 --batch 2``, 40 windows = 20 batches per epoch):
 (A) the capture / restore pair round-trips every stream, is READ-ONLY, is ``weights_only``-loadable, and NAMES what it could
     not restore;
 (B) CONTROL: two uninterrupted runs with the same seed are equal;
 (C) THE HEADLINE: a run stopped after a checkpoint and relaunched with the same argv reproduces the uninterrupted run --
     every training row, every final parameter and optimiser tensor and the final RNG state -- mid-epoch, at an exact epoch
     boundary, and through an epoch rollover;
 (D) the same with a model-owned dedicated generator (the refcv8 ``R8Generator``), and across a REAL process boundary;
 (E) RED ARMS (each must FAIL the equality): the restore dropped; the restore placed BEFORE the launch-time model build; the
     restore placed right after the launch-time seeding; the dedicated-generator restore dropped; an OLD checkpoint (resumes,
     names the missing key, and is NOT equal);
 (F) the eval boundary: a resume from a checkpoint that COINCIDES with an eval is exact (including a death INSIDE the eval),
     and ``EVAL_STREAM_NEUTRAL=False`` (the tip's behaviour) is measured not to be;
 (G) the checkpoint contract: ``ckpt.pt`` and ``ckpt_full_<s>.pt`` gain the SAME one key (G-CKPT's key-set equality holds,
     ``launch_gate._save_at_check`` reports nothing), the ``rng`` key is the only change to the key set, and the capture is
     observation-only (a run with it patched out is bit-identical);
 (H) DataLoader workers: this trainer's dataset draws NO RNG in ``__getitem__`` (measured on the real class), the iterator's
     base seed is drawn from the global stream at ``iter(dl)`` (why the restore must come after it), and a persistent-worker
     loader seeds its workers ONCE (a dataset that did draw would not be resumable mid-epoch at ``--workers > 0``).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data import v2_dataset as V2                    # noqa: E402
from tanitad.refs import refcv8_conditioning as r8c          # noqa: E402

HERE = Path(__file__).resolve()


# ============================================================================ #
# rig                                                                          #
# ============================================================================ #
def _argv(out, steps, save_every, *extra):
    return ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2", "--steps", str(steps), "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", str(save_every), *extra]


def _run(argv):
    args = T.build_parser().parse_args(argv)
    args._ew_argv = list(argv)                              # what main() does
    return T.train(args)


def _rows(out):
    return [json.loads(l) for l in (Path(out) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and '"lr"' in l]


def _clean(r):
    """A row without ``elapsed_s`` (wall clock) and ``goal_gate_grad`` (reads the PREVIOUS step's ``.grad``, so it is absent at
    the first step of every launch -- the one key a relaunched run lacks at its first step; see test_lr_schedule_wsd._clean)."""
    return {k: v for k, v in r.items() if k not in ("elapsed_s", "goal_gate_grad")}


def _ck(out, name="ckpt.pt"):
    return torch.load(Path(out) / name, map_location="cpu", weights_only=False)


def _tree_diffs(a, b, path="", out=None):
    """Every path at which two checkpoint trees differ (tensors by dtype / shape / ``torch.equal``, leaves by ``==``)."""
    out = [] if out is None else out
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
            out.append(path)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append(path + "{keys}")
        else:
            for k in a:
                _tree_diffs(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            out.append(path + "[len]")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _tree_diffs(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)
    return out


def _stop_after(monkeypatch, n):
    """The (n+1)-th training batch fetch raises ``KeyboardInterrupt`` -- a stopped run (the test_lr_schedule_wsd D3 device).
    Returns the undo."""
    real, calls = T.next_train_batch, {"n": 0}

    def _wrapped(it, dl, s, dpos):
        calls["n"] += 1
        if calls["n"] > n:
            raise KeyboardInterrupt("simulated stop")
        return real(it, dl, s, dpos)
    monkeypatch.setattr(T, "next_train_batch", _wrapped)
    return lambda: monkeypatch.setattr(T, "next_train_batch", real)


def _fresh_process_streams():
    """A relaunched process starts with whatever the OS gives numpy / python (``train()`` only seeds torch). Emulate it
    in-process so a restore that does not reach numpy / python is caught too."""
    torch.manual_seed(987654321)
    np.random.seed(123456789)
    random.seed(424242)


def _uninterrupted(tmp, name, steps, save_every, *extra):
    out = tmp / name
    _fresh_process_streams()
    _run(_argv(out, steps, save_every, *extra))
    return out


def _stopped_and_relaunched(tmp, monkeypatch, name, steps, save_every, stop_after, *extra, between=None):
    """Run to ``stop_after`` training steps, die (the checkpoint at ``save_every`` is on disk), scramble every global stream as
    a new process would, optionally let ``between(out)`` tamper with the checkpoint, then relaunch with the SAME argv."""
    out = tmp / name
    argv = _argv(out, steps, save_every, *extra)
    undo = _stop_after(monkeypatch, stop_after)
    _fresh_process_streams()
    with pytest.raises(KeyboardInterrupt):
        _run(argv)
    undo()
    gc.collect()
    assert _ck(out)["step"] == (stop_after // save_every) * save_every
    if between is not None:
        between(out)
    _fresh_process_streams()
    _run(argv)
    return out


def _final_rows(out):
    """The row of each step as the run that FINISHED it wrote it (a relaunch re-runs the steps after its checkpoint, so
    ``metrics.jsonl`` holds those steps twice: the dead launch's and the relaunch's)."""
    last = {}
    for x in _rows(out):
        last[x["step"]] = x
    return [last[k] for k in sorted(last)]


def _assert_same_run(u, r, steps, rng_streams=None):
    """``rng_streams``: restrict the final-RNG comparison to these ``ckpt["rng"]`` keys (across SEPARATE processes numpy's and
    python's global streams start from OS entropy in each, so only the streams training consumes are comparable)."""
    ru, rr = _final_rows(u), _final_rows(r)
    assert [x["step"] for x in rr] == list(range(1, steps + 1)) == [x["step"] for x in ru]
    assert [_clean(x) for x in rr] == [_clean(x) for x in ru], "the resumed rows are not the uninterrupted rows"
    by_step = {x["step"]: _clean(x) for x in ru}
    for x in _rows(r):                       # the DEAD launch's rows for the replayed steps were the uninterrupted run's too
        assert _clean(x) == by_step[x["step"]], x["step"]
    cu, cr = _ck(u), _ck(r)
    assert cu["step"] == cr["step"] == steps
    assert _tree_diffs(cu["model"], cr["model"]) == []
    assert _tree_diffs(cu["opt"], cr["opt"]) == []
    keys = list(cu["rng"]) if rng_streams is None else list(rng_streams)
    assert _tree_diffs({k: cu["rng"][k] for k in keys}, {k: cr["rng"][k] for k in keys}) == [],         "the final RNG state of the resumed run is not the uninterrupted run's"


def _first_differing_step(u, r):
    ru, rr = _final_rows(u), _final_rows(r)
    for a, b in zip(ru, rr):
        if _clean(a) != _clean(b):
            return a["step"]
    return None


# ============================================================================ #
# (A) the capture / restore pair                                               #
# ============================================================================ #
class _Owner(torch.nn.Module):
    """A module holding a refcv8 dedicated generator the way ``refc.py`` does (``self.r8_gen = R8Generator(seed)``)."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.r8_gen = r8c.R8Generator(4242)


def _draws(m=None):
    out = {"torch": torch.rand(3).tolist(), "numpy": np.random.rand(3).tolist(), "python": [random.random() for _ in range(3)]}
    if m is not None:
        out["ded"] = m.r8_gen.rand((3,), "cpu").tolist()
    return out


def test_A1_capture_then_restore_replays_every_stream_and_capture_consumes_nothing():
    m = _Owner()
    torch.manual_seed(1)
    np.random.seed(2)
    random.seed(3)
    m.r8_gen.rand((5,), "cpu")                                  # advance the dedicated stream so it is not at its seed
    snap = T.rng_state_capture(m)
    again = T.rng_state_capture(m)
    assert _tree_diffs(snap, again) == []                       # READ-ONLY: a second capture sees the same streams
    first = _draws(m)
    assert first != _draws(m)                                   # the streams really moved
    rep = T.rng_state_restore(snap, m)
    assert rep["skipped"] == {} and set(rep["restored"]) == {"dedicated::r8_gen", "numpy", "python", "torch_cpu"}
    assert _draws(m) == first


def test_A2_a_generator_created_after_the_capture_is_dropped_and_recreated_fresh():
    m = _Owner()
    snap = T.rng_state_capture(m)                               # the dedicated generator has not been used: `_g` is EMPTY
    assert snap["dedicated"] == {":r8_gen": {}}
    ref = _Owner().r8_gen.rand((4,), "cpu")                     # what a fresh seed-4242 generator gives first
    m.r8_gen.rand((9,), "cpu")                                  # created and advanced AFTER the capture
    assert "cpu" in m.r8_gen._g
    T.rng_state_restore(snap, m)
    assert m.r8_gen._g == {}                                    # dropped ...
    assert torch.equal(m.r8_gen.rand((4,), "cpu"), ref)         # ... and re-created at its seed on first use, as it was


def test_A3_the_state_is_plain_data_that_a_weights_only_load_reads(tmp_path):
    m = _Owner()
    m.r8_gen.rand((2,), "cpu")
    snap = T.rng_state_capture(m)
    torch.save({"rng": snap}, tmp_path / "s.pt")
    back = torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=True)["rng"]
    assert _tree_diffs(snap, back) == []
    first = _draws(m)
    _draws(m)
    assert T.rng_state_restore(back, m)["skipped"] == {}
    assert _draws(m) == first


def test_A4_what_cannot_be_restored_is_NAMED_and_never_aborts_the_rest():
    m = _Owner()
    snap = T.rng_state_capture(m)
    bad = dict(snap, schema="trainer-rng/0")
    rep = T.rng_state_restore(bad, m)
    assert rep["restored"] == [] and "trainer-rng/0" in rep["skipped"]["*"]
    cuda = dict(snap, torch_cuda=[snap["torch_cpu"], snap["torch_cpu"]])        # a 2-GPU checkpoint on a CPU box
    rep = T.rng_state_restore(cuda, m)
    assert "torch_cuda" in rep["skipped"] and "torch_cpu" in rep["restored"] and "numpy" in rep["restored"]
    ghost = dict(snap, dedicated={"core:gone": {"cpu": snap["torch_cpu"]}})     # an owner this build does not have
    rep = T.rng_state_restore(ghost, m)
    assert "dedicated:core:gone" in rep["skipped"] and "numpy" in rep["restored"]


def test_A5_a_state_loaded_with_map_location_on_another_device_is_still_a_cpu_byte_state():
    snap = T.rng_state_capture(None)
    moved = dict(snap, torch_cpu=snap["torch_cpu"].to(torch.int64).to(torch.uint8))     # dtype round-trip stands in for a device move
    assert T.rng_state_restore(moved, None)["skipped"] == {}


# ============================================================================ #
# (B)(C) the headline                                                          #
# ============================================================================ #
def test_B_CONTROL_two_uninterrupted_runs_with_the_same_seed_are_equal(tmp_path):
    a = _uninterrupted(tmp_path, "a", 6, 3)
    b = _uninterrupted(tmp_path, "b", 6, 3)
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == 6 and [_clean(x) for x in ra] == [_clean(x) for x in rb]
    assert _tree_diffs(_ck(a)["model"], _ck(b)["model"]) == []
    # the control has teeth: the rows genuinely depend on the RNG (a different seed gives different rows)
    c = _uninterrupted(tmp_path, "c", 6, 3, "--seed", "1")
    assert [_clean(x)["loss"] for x in _rows(c)] != [_clean(x)["loss"] for x in ra]


def test_C1_a_run_stopped_mid_epoch_and_relaunched_equals_the_uninterrupted_run(tmp_path, monkeypatch, capsys):
    u = _uninterrupted(tmp_path, "u", 8, 4)
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5)         # dies on the 6th fetch: ckpt at 4, step 5 was done
    out = capsys.readouterr().out
    assert "[v3] resumed hier at step 4" in out and "RNG-RESUME: restored" in out and "WITHOUT RNG STATE" not in out
    _assert_same_run(u, r, 8)
    do = json.loads((r / "config.json").read_text(encoding="utf-8"))["data_order"]
    assert do["resumed_at_step"] == 4 and do["rng"]["in_ckpt"] is True and do["rng"]["skipped"] == {}
    assert {"torch_cpu", "numpy", "python"} <= set(do["rng"]["restored"])


def test_C2_at_an_exact_epoch_boundary_and_through_the_rollover(tmp_path, monkeypatch):
    """40 windows / batch 2 = 20 batches per epoch. The checkpoint at step 20 is written when the epoch's LAST batch was
    consumed (``data_pos`` = epoch 0, batch 20): the resumed iterator is empty, the next fetch rolls the epoch over and
    ``iter(dl)`` DRAWS a base seed from the restored stream -- the same draw the uninterrupted run made there."""
    u = _uninterrupted(tmp_path, "u", 26, 20)
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 26, 20, 22)
    assert _ck(r)["data_pos"]["epoch"] == 1
    _assert_same_run(u, r, 26)
    assert _final_rows(r)[-1]["data_epoch"] == 1


def test_C3_a_second_relaunch_after_a_relaunch_is_still_the_uninterrupted_run(tmp_path, monkeypatch):
    """MAX_RELAUNCH=4: the checkpoint a RESUMED run writes carries the stream it continued with, so a second death and
    relaunch composes."""
    u = _uninterrupted(tmp_path, "u", 9, 3)
    out = tmp_path / "r"
    argv = _argv(out, 9, 3)
    for n in (4, 4):                      # fetches allowed per launch: 4 (steps 1-4, ckpt 3), then 4 (resumes at 3: steps 4-7, ckpt 6)
        undo = _stop_after(monkeypatch, n)
        _fresh_process_streams()
        with pytest.raises(KeyboardInterrupt):
            _run(argv)
        undo()
        gc.collect()
    assert _ck(out)["step"] == 6
    _fresh_process_streams()
    _run(argv)
    _assert_same_run(u, out, 9)


# ============================================================================ #
# (D) a model-owned dedicated generator, and a real process boundary           #
# ============================================================================ #
def _install_dedicated_generator(monkeypatch):
    """What ``enable_refcv8`` does at model build (``core.r8_gen = R8Generator(seed)``), plus a loss that CONSUMES it every
    training step and scales the gradient with the draw -- so the parameters, not only the log, depend on its stream."""
    real_opt, real_loss = T.build_optimizer, T.compute_losses_v3

    def _opt(model, args, *a, **k):
        if not hasattr(model.core, "rng_probe"):
            model.core.rng_probe = r8c.R8Generator(777)
        return real_opt(model, args, *a, **k)

    def _loss(model, batch, device, **kw):
        out = real_loss(model, batch, device, **kw)
        if model.training:
            out["loss"] = out["loss"] * (1.0 + 0.5 * model.core.rng_probe.rand((1,), "cpu").sum())
        return out
    monkeypatch.setattr(T, "build_optimizer", _opt)
    monkeypatch.setattr(T, "compute_losses_v3", _loss)


def test_D1_a_dedicated_generator_consumed_every_step_resumes_exactly(tmp_path, monkeypatch):
    _install_dedicated_generator(monkeypatch)
    u = _uninterrupted(tmp_path, "u", 8, 4)
    assert "core:rng_probe" in _ck(u)["rng"]["dedicated"] and _ck(u)["rng"]["dedicated"]["core:rng_probe"]["cpu"].numel() > 0
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5)
    _assert_same_run(u, r, 8)
    assert "dedicated:core:rng_probe" in json.loads((r / "config.json").read_text(encoding="utf-8"))["data_order"]["rng"]["restored"]


def test_D2_across_a_real_process_boundary(tmp_path):
    """Two OS processes: one runs 8 steps uninterrupted, another runs 5 and exits hard (``os._exit``) after the checkpoint at 4,
    a third relaunches. numpy / python start from OS entropy in each, exactly as under ``supervise_run.sh``."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONHASHSEED="0")
    ru, rr = tmp_path / "u", tmp_path / "r"

    def _child(out, stop):
        cmd = [sys.executable, str(HERE), "--child", str(out), "8", "4", str(stop)]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        return p

    pu = _child(ru, 0)
    assert pu.returncode == 0, pu.stderr[-2000:]
    p1 = _child(rr, 5)
    assert p1.returncode == 17, (p1.returncode, p1.stderr[-2000:])          # the hard exit
    killed = _ck(rr)
    assert killed["step"] == 4
    p2 = _child(rr, 0)
    assert p2.returncode == 0, p2.stderr[-2000:]
    assert "RNG-RESUME: restored" in p2.stdout
    _assert_same_run(ru, rr, 8, rng_streams=("torch_cpu", "torch_cuda", "dedicated"))
    # numpy / python are not consumed by training, so the relaunch's final state IS the killed process's: they crossed the boundary
    final = _ck(rr)["rng"]
    assert _tree_diffs(killed["rng"]["numpy"], final["numpy"]) == [] and _tree_diffs(killed["rng"]["python"], final["python"]) == []
    assert _tree_diffs(_ck(ru)["rng"]["numpy"], final["numpy"]) != []      # ... and they are NOT the uninterrupted process's


def test_D3_with_two_DataLoader_workers_the_resume_is_exact_too_and_equals_the_zero_worker_run(tmp_path, monkeypatch):
    """MEASURED: this trainer's dataset draws no RNG in ``__getitem__`` (H1), so worker-side streams are inert and the
    workers add nothing to restore. ``iter(dl)`` draws the persistent workers' base seed once, at launch, BEFORE the restore
    (the restore then puts the main stream back), and the rows are the zero-worker run's as well."""
    u0 = _uninterrupted(tmp_path, "u0", 8, 4)
    u2 = _uninterrupted(tmp_path, "u2", 8, 4, "--workers", "2")
    r2 = _stopped_and_relaunched(tmp_path, monkeypatch, "r2", 8, 4, 5, "--workers", "2")
    _assert_same_run(u2, r2, 8)
    assert [_clean(x) for x in _final_rows(u2)] == [_clean(x) for x in _final_rows(u0)]


# ============================================================================ #
# (E) RED ARMS -- each must FAIL the equality                                  #
# ============================================================================ #
def _red_u(tmp_path):
    return _uninterrupted(tmp_path, "u", 8, 4)


def test_E1_RED_the_restore_dropped_the_resumed_run_DIFFERS(tmp_path, monkeypatch):
    u = _red_u(tmp_path)
    monkeypatch.setattr(T, "rng_state_restore", lambda *a, **k: {"restored": [], "skipped": {}})
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5)
    assert _first_differing_step(u, r) == 5                 # the first step after the resume is already a different step
    assert [x["loss"] for x in _final_rows(r)][4:] != [x["loss"] for x in _final_rows(u)][4:]
    with pytest.raises(AssertionError):
        _assert_same_run(u, r, 8)


def test_E2_RED_the_restore_placed_BEFORE_the_launch_time_model_build_DIFFERS(tmp_path, monkeypatch):
    """The state is put back, then ``iter(dl)`` (which DRAWS the worker base seed) and the rest of the launch consume it."""
    u = _red_u(tmp_path)
    real_opt = T.build_optimizer

    def _early(model, args, *a, **k):                        # build_optimizer runs before `iter(dl)`, after the checkpoint is readable
        ck = Path(args.out) / "ckpt.pt"
        if ck.exists():
            T.rng_state_restore(torch.load(ck, map_location="cpu", weights_only=False)[T.RNG_CKPT_KEY], model)
        return real_opt(model, args, *a, **k)
    monkeypatch.setattr(T, "build_optimizer", _early)
    monkeypatch.setattr(T, "rng_state_restore", lambda *a, **k: {"restored": [], "skipped": {}})   # the late (correct) restore is OFF
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5)
    assert _first_differing_step(u, r) == 5
    with pytest.raises(AssertionError):
        _assert_same_run(u, r, 8)


def test_E3_RED_the_restore_placed_right_after_the_launch_time_seeding_DIFFERS(tmp_path, monkeypatch):
    u = _red_u(tmp_path)
    real_seed, state = torch.manual_seed, {"fired": False}

    def _seed_then_restore(s):
        g = real_seed(s)
        ck = Path(state.get("out", "")) / "ckpt.pt"
        if state.get("out") and not state["fired"] and ck.exists():
            state["fired"] = True
            T.rng_state_restore(torch.load(ck, map_location="cpu", weights_only=False)[T.RNG_CKPT_KEY], None)
        return g
    real_train = T.train

    def _train(args):
        state["out"] = str(args.out)
        return real_train(args)
    monkeypatch.setattr(T, "train", _train)
    monkeypatch.setattr(torch, "manual_seed", _seed_then_restore)
    r_restore = T.rng_state_restore
    monkeypatch.setattr(T, "rng_state_restore", lambda s, m=None: r_restore(s, m) if state.get("on") else {"restored": [], "skipped": {}})
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5)
    assert state["fired"] is True
    assert _first_differing_step(u, r) == 5
    with pytest.raises(AssertionError):
        _assert_same_run(u, r, 8)


def test_E4_RED_the_dedicated_generator_restore_dropped_DIFFERS(tmp_path, monkeypatch):
    _install_dedicated_generator(monkeypatch)
    u = _red_u(tmp_path)
    real_owners = T._dedicated_generator_owners
    flag = {"resume": False}
    monkeypatch.setattr(T, "_dedicated_generator_owners", lambda m: {} if flag["resume"] else real_owners(m))

    def _between(out):
        flag["resume"] = True
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5, between=_between)
    assert _first_differing_step(u, r) == 5                 # torch / numpy / python were restored; the dedicated stream was not
    with pytest.raises(AssertionError):
        _assert_same_run(u, r, 8)


def test_E5_an_OLD_checkpoint_resumes_names_the_missing_key_and_is_NOT_equal(tmp_path, monkeypatch, capsys):
    u = _red_u(tmp_path)

    def _strip(out):                                          # a checkpoint written by the tip: no `rng`
        st = _ck(out)
        st.pop("rng")
        torch.save(st, Path(out) / "ckpt.pt")
        assert "rng" not in _ck(out)
    capsys.readouterr()
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5, between=_strip)
    out = capsys.readouterr().out
    assert "RESUMED FROM A ckpt.pt WITHOUT RNG STATE" in out and "missing key: 'rng'" in out and "from step 4 on" in out
    assert "RNG-RESUME: restored" not in out
    assert _final_rows(r)[-1]["step"] == 8                    # ... and it still RESUMES: the old checkpoint is not refused
    do = json.loads((r / "config.json").read_text(encoding="utf-8"))["data_order"]
    assert do["rng"]["in_ckpt"] is False and do["resumed_at_step"] == 4
    assert _first_differing_step(u, r) == 5                   # the tip's behaviour, named instead of silent
    # data order and the model state at the resume point are still exact (what the old checkpoint DID carry)
    assert [x["data_batch"] for x in _final_rows(r)] == [x["data_batch"] for x in _final_rows(u)]


# ============================================================================ #
# (F) the eval boundary                                                        #
# ============================================================================ #
EVAL_TRAIN_IDS, EVAL_IDS, EVAL_CACHE = (101, 102), (201, 202), "synthetic-held-out"


def _eval_rig(monkeypatch):
    """The rig of test_refc_v3_save_before_eval: integer episode ids on both sides and a fake held-out provider."""
    real_synth, seen = T._synth_episodes, {}

    def _train_eps(n, cfg, seed=0, min_frames=40):
        seen["core"] = cfg
        eps = real_synth(n, cfg, seed=seed, min_frames=min_frames)
        for ep, sid in zip(eps, EVAL_TRAIN_IDS):
            ep.episode_id = sid
        return eps

    def _eval_providers(paths, lru_size=6, **kw):
        eps = real_synth(len(EVAL_IDS), seen["core"], seed=77)
        for ep, sid in zip(eps, EVAL_IDS):
            ep.episode_id = sid
        return eps
    monkeypatch.setattr(T, "_synth_episodes", _train_eps)
    monkeypatch.setattr(V2, "build_v2_providers", _eval_providers)


_EVAL_FLAGS = ("--eval-cache", EVAL_CACHE, "--eval-every", "4", "--eval-batches", "2")


def test_F1_a_resume_from_a_checkpoint_that_COINCIDES_with_an_eval_is_exact(tmp_path, monkeypatch):
    """--save-every 4 = --eval-every 4: the checkpoint is written BEFORE the eval (load-bearing) and the eval draws from the
    global stream. The step-4 eval is in the uninterrupted run's past; the relaunch must not see it as its future."""
    _eval_rig(monkeypatch)
    u = _uninterrupted(tmp_path, "u", 8, 4, *_EVAL_FLAGS)
    assert any("eval_loss" in l for l in (u / "metrics.jsonl").read_text(encoding="utf-8").splitlines())     # the eval RAN
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5, *_EVAL_FLAGS)
    _assert_same_run(u, r, 8)


def test_F2_a_death_INSIDE_the_eval_resumes_exactly(tmp_path, monkeypatch):
    """The historic death (C-REFCV3-EVAL-DEATH): the process is killed mid-eval, after ``ckpt.pt`` was written. A
    ``BaseException`` out of the eval forward is not caught by the trainer's ``except Exception`` -- it ends the run there."""
    _eval_rig(monkeypatch)
    u = _uninterrupted(tmp_path, "u", 8, 4, *_EVAL_FLAGS)
    real, n = T.compute_losses_v3, {"eval": 0}

    def _dies_in_eval(model, batch, device, **kw):
        if not model.training:
            n["eval"] += 1
            if n["eval"] == 2:                                # after the first eval batch has drawn from the stream
                raise KeyboardInterrupt("killed inside the eval")
        return real(model, batch, device, **kw)
    r = tmp_path / "r"
    argv = _argv(r, 8, 4, *_EVAL_FLAGS)
    monkeypatch.setattr(T, "compute_losses_v3", _dies_in_eval)
    _fresh_process_streams()
    with pytest.raises(KeyboardInterrupt):
        _run(argv)
    monkeypatch.setattr(T, "compute_losses_v3", real)
    gc.collect()
    assert _ck(r)["step"] == 4 and n["eval"] == 2
    _fresh_process_streams()
    _run(argv)
    ru, rr = _final_rows(u), _final_rows(r)
    assert [_clean(x) for x in rr if x["step"] > 4] == [_clean(x) for x in ru if x["step"] > 4]
    assert _tree_diffs(_ck(u)["model"], _ck(r)["model"]) == [] and _tree_diffs(_ck(u)["rng"], _ck(r)["rng"]) == []


def test_F3_MEASURED_without_the_stream_neutral_eval_a_resume_at_an_eval_boundary_DIVERGES(tmp_path, monkeypatch):
    """The tip's behaviour (``EVAL_STREAM_NEUTRAL = False``): the eval's draws stay in the uninterrupted run's stream, the
    checkpoint was written before them, so the relaunch cannot reproduce it -- the RNG capture alone is not enough here."""
    _eval_rig(monkeypatch)
    monkeypatch.setattr(T, "EVAL_STREAM_NEUTRAL", False)
    u = _uninterrupted(tmp_path, "u", 8, 4, *_EVAL_FLAGS)
    r = _stopped_and_relaunched(tmp_path, monkeypatch, "r", 8, 4, 5, *_EVAL_FLAGS)
    assert _first_differing_step(u, r) == 5
    with pytest.raises(AssertionError):
        _assert_same_run(u, r, 8)


def test_F4_MEASURED_the_neutral_eval_changes_an_eval_ON_trajectory_and_only_that(tmp_path, monkeypatch):
    """THE DECISION THIS FLAG RECORDS: the neutral eval removes the eval's draws from the training stream, so an eval-ON
    uninterrupted run is no longer bit-equal to the tip's from the first step after its first eval. Its EVAL rows (what the
    eval itself computed) are unchanged, and an eval-OFF run is untouched (F5)."""
    _eval_rig(monkeypatch)
    a = _uninterrupted(tmp_path, "neutral", 8, 4, *_EVAL_FLAGS)
    monkeypatch.setattr(T, "EVAL_STREAM_NEUTRAL", False)
    b = _uninterrupted(tmp_path, "tip", 8, 4, *_EVAL_FLAGS)
    ra, rb = _rows(a), _rows(b)
    assert [_clean(x) for x in ra if x["step"] <= 4] == [_clean(x) for x in rb if x["step"] <= 4]    # identical up to the first eval
    assert [x["loss"] for x in ra if x["step"] > 4] != [x["loss"] for x in rb if x["step"] > 4]       # then the trajectories part
    ev = lambda out: [json.loads(l) for l in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if '"eval_loss"' in l]
    ea, eb = ev(a), ev(b)
    assert ea[0]["step"] == eb[0]["step"] == 4 and ea[0]["eval_loss"] == eb[0]["eval_loss"]           # the step-4 eval row is unchanged


def test_F5_with_the_eval_OFF_the_run_is_untouched_by_the_neutral_switch(tmp_path, monkeypatch):
    a = _uninterrupted(tmp_path, "on", 6, 3)
    monkeypatch.setattr(T, "EVAL_STREAM_NEUTRAL", False)
    b = _uninterrupted(tmp_path, "off", 6, 3)
    assert [_clean(x) for x in _rows(a)] == [_clean(x) for x in _rows(b)]


def test_F6_with_the_neutral_eval_an_eval_ON_run_trains_exactly_like_an_eval_OFF_run(tmp_path, monkeypatch):
    """The property the switch buys beyond the resume: the in-run eval cadence no longer changes the training trajectory. (The tip's
    eval-ON run parted from the eval-OFF run at the first step after its first eval: F4.)"""
    _eval_rig(monkeypatch)
    on = _uninterrupted(tmp_path, "on", 8, 4, *_EVAL_FLAGS)
    off = _uninterrupted(tmp_path, "off", 8, 4)
    assert [_clean(x) for x in _final_rows(on)] == [_clean(x) for x in _final_rows(off)]
    assert _tree_diffs(_ck(on)["model"], _ck(off)["model"]) == []
    monkeypatch.setattr(T, "EVAL_STREAM_NEUTRAL", False)
    tip = _uninterrupted(tmp_path, "tip", 8, 4, *_EVAL_FLAGS)
    assert [_clean(x) for x in _final_rows(tip)] != [_clean(x) for x in _final_rows(off)]


# ============================================================================ #
# (G) the checkpoint contract                                                  #
# ============================================================================ #
def test_G1_ckpt_and_ckpt_full_gain_the_SAME_one_key_and_the_gate_agrees(tmp_path):
    sched = ("--lr-schedule", "wsd", "--wsd-decay-steps", "4", "--warmup", "3")
    out = tmp_path / "w"
    _fresh_process_streams()
    _run(_argv(out, 8, 4, *sched, "--save-at", "4"))
    live, full = _ck(out), _ck(out, "ckpt_full_4.pt")
    assert set(live) == set(full) == {"model", "opt", "step", "data_pos", "param_names", "rng"}      # the ONE added key: rng
    assert {"model", "opt", "step", "data_pos"} <= set(live)                                          # G-CKPT's `judge_ckpt` floor
    assert full["step"] == 4 and live["step"] == 8
    assert set(_ck(out, "ckpt_0.pt") if (out / "ckpt_0.pt").exists() else {"model", "step"}) == {"model", "step"}
    import launch_gate as LG                                                                           # noqa: E402
    reasons: list = []
    seen = LG._save_at_check(out, [4], reasons)
    assert reasons == [] and seen["4"] == sorted(live)
    # the stream a FULL-STATE checkpoint carries is the one at ITS step, not the final one
    assert _tree_diffs(full["rng"], live["rng"]) != []


def test_G2_the_capture_is_observation_only__a_run_without_it_is_bit_identical(tmp_path, monkeypatch):
    a = _uninterrupted(tmp_path, "with", 6, 3)
    monkeypatch.setattr(T, "_rng_ckpt_extra", lambda model: {})
    b = _uninterrupted(tmp_path, "without", 6, 3)
    assert "rng" not in _ck(b) and "rng" in _ck(a)
    assert [_clean(x) for x in _rows(a)] == [_clean(x) for x in _rows(b)]
    assert _tree_diffs(_ck(a)["model"], _ck(b)["model"]) == [] and _tree_diffs(_ck(a)["opt"], _ck(b)["opt"]) == []


def test_G3_the_checkpoint_of_an_uninterrupted_run_loads_weights_only(tmp_path):
    out = _uninterrupted(tmp_path, "u", 4, 2)
    st = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=True)
    assert set(st) >= {"model", "opt", "step", "data_pos", "param_names", "rng"}


# ============================================================================ #
# (H) DataLoader workers                                                       #
# ============================================================================ #
def _fingerprint():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes() + repr(np.random.get_state()[2]).encode()
                          + repr(random.getstate()[1][-1]).encode()).hexdigest()[:12]


def test_H1_MEASURED_the_real_dataset_draws_no_global_rng_in_getitem(tmp_path, monkeypatch):
    """So worker-side streams are inert: at ``--workers > 0`` each worker is seeded once and nothing consumes the seed."""
    seen = {"calls": 0, "moved": 0}
    real = T.V3Dataset.__getitem__

    def _spy(self, i):
        before = _fingerprint()
        out = real(self, i)
        seen["calls"] += 1
        seen["moved"] += int(_fingerprint() != before)
        return out
    monkeypatch.setattr(T.V3Dataset, "__getitem__", _spy)
    _run(_argv(tmp_path / "d", 4, 2))
    assert seen["calls"] >= 8 and seen["moved"] == 0


def test_H2_MEASURED_iter_of_the_loader_DRAWS_the_base_seed_from_the_global_stream():
    """Why the restore comes AFTER ``it = iter(dl)``: creating the iterator moves the global stream even at 0 workers."""
    dl = torch.utils.data.DataLoader(list(range(8)), batch_size=2, num_workers=0)
    torch.manual_seed(5)
    before = torch.get_rng_state().clone()
    it = iter(dl)
    after = torch.get_rng_state().clone()
    assert not torch.equal(before, after)
    torch.manual_seed(5)
    assert torch.equal(torch.get_rng_state(), before)
    it2 = iter(dl)                                             # the same draw again from the same state: deterministic
    assert torch.equal(torch.get_rng_state(), after)
    assert it is not it2


class _DrawDS(torch.utils.data.Dataset):
    """A dataset that DOES draw in ``__getitem__`` (this trainer's does not): one worker-stream value per item."""

    def __len__(self):
        return 16

    def __getitem__(self, i):
        return torch.rand(1)


def test_H3_MEASURED_persistent_workers_seed_once__a_dataset_that_drew_would_not_resume_mid_epoch():
    """The limit, MEASURED so it is not assumed: with ``persistent_workers=True`` the base seed is drawn at the FIRST iterator and
    the workers keep their streams across epochs; a re-created loader (what a relaunch builds) starts every worker stream over.
    Hence a worker-drawing dataset cannot be resumed bit-exactly at ``--workers > 0`` -- which is why H1 matters."""
    def _epoch(dl):
        return torch.cat([b for b in dl]).flatten().tolist()
    torch.manual_seed(11)
    dl = torch.utils.data.DataLoader(_DrawDS(), batch_size=2, num_workers=2, persistent_workers=True)
    g0 = torch.get_rng_state().clone()
    e1 = _epoch(dl)
    g1 = torch.get_rng_state().clone()
    e2 = _epoch(dl)
    g2 = torch.get_rng_state().clone()
    assert not torch.equal(g0, g1) and torch.equal(g1, g2)     # the base seed is drawn ONCE (first iterator), not per epoch
    assert e1 != e2                                            # the persistent workers' streams ran on into epoch 2
    torch.manual_seed(11)                                      # a relaunch: same stream position, a NEW loader
    dl2 = torch.utils.data.DataLoader(_DrawDS(), batch_size=2, num_workers=2, persistent_workers=True)
    assert _epoch(dl2) == e1                                   # its workers start over: epoch-1 values again, not epoch 2's
    del dl, dl2


# ============================================================================ #
# (I) REACHABILITY: built, tested -- and CALLED. (CLAUDE.md: "rollable and trained are different claims")                #
# ============================================================================ #
TRAINER = ROOT / "scripts" / "refc_v3_train.py"


def _pins(text):
    i_iter = text.index("    it = iter(dl)\n    _bev_parity_checked")
    i_rest = text.find("        _rng_rep = rng_state_restore(_rng_resume, model)")
    i_catch = text.index("    if _b1 is not None and _ema_resume is not None and not _b1[\"decided\"]")
    i_while = text.index("    while step < args.steps:")
    i_ckpt = text.find('"opt": opt.state_dict(), "step": step,')
    i_eval = text.index("if eval_dl is not None and (step % args.eval_every")
    return {
        "inline save carries rng": text.count("**_p10_ckpt_extra(model), **_rng_ckpt_extra(model)}, ck)"),
        "_ckpt_payload (ckpt_full, B1 re-save) carries rng": text.count("**_p10_ckpt_extra(model), **_rng_ckpt_extra(model)}\n"),
        "the resumed ckpt's rng is read": text.count("        _rng_resume = state.get(RNG_CKPT_KEY)"),
        "restore called once": text.count("rng_state_restore(_rng_resume, model)"),
        "restore AFTER iter(dl), BEFORE the B1 catch-up and the first step":
            int(i_iter < i_rest < i_catch < i_while) if i_rest > 0 else 0,
        "old checkpoint warns": text.count("RESUMED FROM A ckpt.pt WITHOUT RNG STATE"),
        "B1 catch-up eval is neutral": text.count("_gc_snap = rng_state_capture(model) if EVAL_STREAM_NEUTRAL else None"),
        "aggregate eval snapshot": text.count("_eval_stream = rng_state_capture(model) if EVAL_STREAM_NEUTRAL else None"),
        "aggregate eval restored (before the B1 re-save, and at the block's end)": text.count("rng_state_restore(_eval_stream, model)"),
        "ckpt is still written BEFORE the eval": int(0 < i_ckpt < i_eval),
        "the switch is on": text.count("EVAL_STREAM_NEUTRAL = True"),
    }


def _pinned(p):
    return (p["inline save carries rng"] == 1 and p["_ckpt_payload (ckpt_full, B1 re-save) carries rng"] == 1
            and p["the resumed ckpt's rng is read"] == 1 and p["restore called once"] == 1
            and p["restore AFTER iter(dl), BEFORE the B1 catch-up and the first step"] == 1 and p["old checkpoint warns"] == 1
            and p["B1 catch-up eval is neutral"] == 1 and p["aggregate eval snapshot"] == 1
            and p["aggregate eval restored (before the B1 re-save, and at the block's end)"] == 2
            and p["ckpt is still written BEFORE the eval"] == 1 and p["the switch is on"] == 1)


def test_I1_the_trainer_reaches_every_part_of_the_rng_machinery():
    p = _pins(TRAINER.read_text(encoding="utf-8"))
    assert _pinned(p), p


@pytest.mark.parametrize("old,new", [
    ("**_p10_ckpt_extra(model), **_rng_ckpt_extra(model)}, ck)", "**_p10_ckpt_extra(model)}, ck)"),
    ("**_p10_ckpt_extra(model), **_rng_ckpt_extra(model)}\n", "**_p10_ckpt_extra(model)}\n"),
    ("        _rng_resume = state.get(RNG_CKPT_KEY)", "        _rng_resume = None"),
    ("        _rng_rep = rng_state_restore(_rng_resume, model)", "        _rng_rep = {'restored': [], 'skipped': {}}"),
    ("    it = iter(dl)\n    _bev_parity_checked", "    _bev_parity_checked"),   # (restore now BEFORE the iterator exists: order pin)
    ("            _eval_stream = rng_state_capture(model) if EVAL_STREAM_NEUTRAL else None", "            _eval_stream = None"),
    ("        _gc_snap = rng_state_capture(model) if EVAL_STREAM_NEUTRAL else None", "        _gc_snap = None"),
    ("EVAL_STREAM_NEUTRAL = True", "EVAL_STREAM_NEUTRAL = False"),
])
def test_I2_RED_each_edit_that_unwires_the_mechanism_FAILS_the_reachability_pin(old, new):
    text = TRAINER.read_text(encoding="utf-8")
    assert text.count(old) >= 1, old
    assert _pinned(_pins(text))                                                    # the control
    try:
        broken = _pins(text.replace(old, new, 1))
    except ValueError:                                                             # an anchor the pin needs is gone: also RED
        return
    assert not _pinned(broken)


# ============================================================================ #
# child entry (D2)                                                             #
# ============================================================================ #
def _child_main(argv):
    out, steps, save_every, stop = argv[0], int(argv[1]), int(argv[2]), int(argv[3])
    if stop:
        real = T.next_train_batch
        calls = {"n": 0}

        def _wrapped(it, dl, s, dpos):
            calls["n"] += 1
            if calls["n"] > stop:
                sys.stdout.flush()
                os._exit(17)                                   # SIGKILL stand-in: no cleanup, no atexit
            return real(it, dl, s, dpos)
        T.next_train_batch = _wrapped
    _run(_argv(out, steps, save_every))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    _child_main(sys.argv[2:])
