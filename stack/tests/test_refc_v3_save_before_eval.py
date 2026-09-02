"""Checkpoint BEFORE the in-training eval, and an eval that fails LOUD but
never takes the run down -- ``scripts/refc_v3_train.py``'s step loop.

C-REFCV3-EVAL-DEATH (MEASURED 2026-09-02): refcv3 died SILENTLY at
``--eval-every 500`` boundaries (steps 2,000 / 4,500 / 6,000 / 10,500 /
17,000; empty stderr) -- the pod's 50 GB memory cgroup SIGKILLs the trainer
while the in-training eval decodes its batches in the main process. The step
loop ran the eval BEFORE the checkpoint, so each death lost everything since
the previous boundary (the 17,000 death left ``ckpt.pt`` at 16,500).

Pinned here, on the synthetic 1-step ``train()`` rig of
``test_refc_v3_nav_from_v7.py`` (integer episode ids on both sides, a fake
held-out provider, ``MILESTONES`` at step 1):
(a) ORDER -- with the eval forward made to RAISE (the eval call is told apart
    from the training call by ``model.training``), ``ckpt.pt`` AND the
    milestone ``ckpt_1.pt`` ALREADY EXIST at step 1 when the eval runs,
    ``train()`` returns normally, ``summary.json`` is written, and the model
    is back in train mode. A kernel SIGKILL cannot be simulated in-process;
    what this pins is the ORDER, which is the whole protection for that case.
(b) FAIL LOUD -- the ``eval_error`` row lands in ``metrics.jsonl`` AFTER the
    step-1 training row and names the exception; the ``[v3:eval] FAILED``
    line is printed after the ckpt line; the traceback reaches stderr; no
    ``eval_loss`` row is fabricated.
(c) SAME CHECKPOINT -- with a NON-raising eval that really runs, the
    checkpoint saved BEFORE the eval is ``torch.equal`` on every parameter,
    every optimizer tensor and every non-tensor leaf to the state AFTER the
    eval, with the counts asserted. ⚠️ MEASURED 2026-09-02 while writing this
    pin: EXACTLY TWO of 192 model tensors DO move -- the buffers
    ``core.lat_log_prior`` / ``core.lon_log_prior`` -- because
    ``compute_losses_v3`` calls ``model.core.update_tactical_prior()``
    UNCONDITIONALLY and the in-training eval reuses ``compute_losses_v3``, so
    the HELD-OUT split's labels EMA into a buffer that alters decodes
    (``refc.py`` ``logit_adjust``). ``refc.py``'s own docstring forbids
    exactly this. That is a pre-existing leak in EITHER block order (the eval
    runs at the same point either way); the order only decides whether the
    checkpoint carries the pre- or post-eval buffers. (c) therefore excludes
    those two names and asserts the differing set is a SUBSET of them, so it
    keeps passing when the leak is fixed;
(d) THE LEAK, PINNED -- a strict xfail asserts the two buffers are untouched
    by the eval. It fails today; the day the one-line fix lands (gate the
    call on ``model.training``) it XPASSes, strict turns that into a failure,
    and the exclusion in (c) gets removed CONSCIOUSLY.
(e) SOURCE ORDER -- a textual pin that the save precedes the eval block.
Control (run once, 2026-09-02): against the PRE-patch trainer (a)/(b) fail
with the RuntimeError propagating out of ``train()`` and no ``ckpt.pt`` on
disk, (c) on the missing ``ckpt step 1`` line, (e) on the order -- the pins
have teeth. CPU-only, synthetic data; nothing touches a pod.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from tanitad.data import v2_dataset as V2                    # noqa: E402

TRAIN_IDS = (101, 102)
EVAL_IDS = (201, 202)
EVAL_CACHE = "synthetic-held-out"      # never opened: the provider is faked
#: the two state_dict entries the in-training eval is KNOWN to move (see the
#: module docstring, (c)/(d)) -- buffers, not parameters
PRIOR_BUFFERS = ("core.lat_log_prior", "core.lon_log_prior")


# ---------------------------------------------------------------- the rig
def _argv(out):
    return ["--arm", "hier", "--out", str(out), "--smoke",
            "--synth-episodes", "2", "--steps", "1", "--batch", "2",
            "--device", "cpu", "--log-every", "1", "--save-every", "1",
            "--eval-cache", EVAL_CACHE, "--eval-every", "1",
            "--eval-batches", "2"]


def _rig(monkeypatch):
    """Integer episode ids on BOTH sides (the train/eval overlap guard does
    ``int(episode_id)``), a fake held-out provider (the trainer imports
    ``build_v2_providers`` INSIDE ``train()``, so the module attribute is what
    it calls; the same seam ``test_v2_parity.py`` uses), ``MILESTONES`` at
    step 1, and a recording Adam so the optimizer state is reachable after
    ``train()`` returns. Returns the list the Adam instances land in."""
    real_synth = T._synth_episodes
    seen_cfg = {}

    def _train_eps(n, cfg, seed=0, min_frames=40):
        seen_cfg["core"] = cfg
        eps = real_synth(n, cfg, seed=seed, min_frames=min_frames)
        assert len(eps) == len(TRAIN_IDS)
        for ep, sid in zip(eps, TRAIN_IDS):
            ep.episode_id = sid
        return eps

    def _eval_providers(paths, lru_size=6, **kw):
        assert list(paths) == [EVAL_CACHE]
        # the SAME geometry the trainer built the model at (train eps first)
        eps = real_synth(len(EVAL_IDS), seen_cfg["core"], seed=77)
        for ep, sid in zip(eps, EVAL_IDS):
            ep.episode_id = sid
        return eps

    opts = []

    class _RecordingAdam(torch.optim.Adam):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            opts.append(self)

    monkeypatch.setattr(T, "_synth_episodes", _train_eps)
    monkeypatch.setattr(V2, "build_v2_providers", _eval_providers)
    monkeypatch.setattr(T, "MILESTONES", (1,))
    monkeypatch.setattr(torch.optim, "Adam", _RecordingAdam)
    return opts


def _spy_losses(monkeypatch, *, raise_in_eval, on_eval=None):
    """Wrap ``compute_losses_v3``: the TRAINING call (``model.training``) runs
    the real loss; the EVAL call (``model.eval()`` was called) runs
    ``on_eval`` first and then either raises or runs the real loss."""
    real = T.compute_losses_v3
    seen = {"train": 0, "eval": 0, "model": None}

    def _wrapped(model, batch, device, **kw):
        seen["model"] = model
        if model.training:
            seen["train"] += 1
            return real(model, batch, device, **kw)
        seen["eval"] += 1
        if on_eval is not None:
            on_eval()
        if raise_in_eval:
            raise RuntimeError(
                "synthetic eval death (C-REFCV3-EVAL-DEATH rig)")
        return real(model, batch, device, **kw)

    monkeypatch.setattr(T, "compute_losses_v3", _wrapped)
    return seen


def _rows(out):
    lines = (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _diff_tree(a, b, path, counted, diffs):
    """Structural comparison: tensors by dtype/shape/``torch.equal``, leaves
    by ``==``, containers key-for-key (a key-set or length mismatch is an
    immediate failure). Appends every compared tensor's path to ``counted``
    and every differing leaf's path to ``diffs`` -- ALL of them, so a failure
    names the whole set rather than the first hit."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b), path
        counted.append(path)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            diffs.append(path)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _diff_tree(a[k], b[k], f"{path}/{k}", counted, diffs)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_tree(x, y, f"{path}[{i}]", counted, diffs)
    elif a != b:
        diffs.append(path)


# --------------------------------------------- (a)+(b) order + fail loud
def test_ckpt_on_disk_before_the_eval_and_an_eval_death_does_not_end_the_run(
        tmp_path, monkeypatch, capsys):
    _rig(monkeypatch)
    out = tmp_path / "run"
    at_eval = {}

    def _probe():        # runs INSIDE the eval forward, before it raises
        ck = out / "ckpt.pt"
        at_eval["ckpt_exists"] = ck.exists()
        at_eval["ckpt_step"] = (torch.load(ck, map_location="cpu",
                                           weights_only=False)["step"]
                                if ck.exists() else None)
        at_eval["milestone_exists"] = (out / "ckpt_1.pt").exists()
        at_eval["rows"] = [(r["step"], "eval_error" in r, "loss" in r)
                           for r in _rows(out)]

    seen = _spy_losses(monkeypatch, raise_in_eval=True, on_eval=_probe)
    args = T.build_parser().parse_args(_argv(out))
    assert T.train(args) == {"step": 1}            # returned NORMALLY
    assert seen["train"] == 1 and seen["eval"] == 1
    # (a) the checkpoint -- ck AND the milestone -- was ALREADY on disk at
    #     step 1 when the eval ran, and only the training row was logged yet
    assert at_eval == {"ckpt_exists": True, "ckpt_step": 1,
                       "milestone_exists": True, "rows": [(1, False, True)]}
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    assert set(ck) == {"model", "opt", "step"} and ck["step"] == 1
    ms = torch.load(out / "ckpt_1.pt", map_location="cpu", weights_only=False)
    assert set(ms) == {"model", "step"} and ms["step"] == 1
    assert json.loads((out / "summary.json").read_text())["done"] is True
    assert seen["model"].training is True          # back in train mode
    # (b) fail loud: the eval_error row AFTER the training row, nothing faked
    rows = _rows(out)
    assert [r["step"] for r in rows] == [1, 1]
    assert "loss" in rows[0] and "eval_error" not in rows[0]
    assert rows[1]["eval_error"].startswith(
        "RuntimeError: synthetic eval death")
    assert rows[1]["eval_batches_done"] == 0
    assert not any(k.startswith("eval_loss") for r in rows for k in r)
    cap = capsys.readouterr()
    failed = cap.out.index("[v3:eval] FAILED at step 1: RuntimeError: "
                           "synthetic eval death")
    assert cap.out.index("[v3:hier] ckpt step 1 -> ckpt.pt") < failed
    assert "[v3:eval] step 1 " not in cap.out       # no success line
    assert "Traceback" in cap.err and "synthetic eval death" in cap.err
    assert "DONE at 1" in cap.out


# ------------------------------------------------------- (c) same ckpt
def _run_non_raising_eval(tmp_path, monkeypatch):
    """One 1-step run whose eval really runs (2 batches). Returns the ckpt
    saved BEFORE the eval, the live model/opt state AFTER it, the spy and
    the run dir."""
    opts = _rig(monkeypatch)
    out = tmp_path / "run"
    seen = _spy_losses(monkeypatch, raise_in_eval=False)
    args = T.build_parser().parse_args(_argv(out))
    assert T.train(args) == {"step": 1}
    assert seen["train"] == 1 and seen["eval"] == 2    # the eval RAN, fully
    assert len(opts) == 1
    model, opt = seen["model"], opts[0]
    assert model.training is True
    before = torch.load(out / "ckpt.pt", map_location="cpu",
                        weights_only=False)          # saved BEFORE the eval
    assert before["step"] == 1
    after = {"model": model.state_dict(), "opt": opt.state_dict()}
    return before, after, seen, out


def test_checkpoint_saved_before_the_eval_equals_the_state_after_it(
        tmp_path, monkeypatch, capsys):
    before, after, seen, out = _run_non_raising_eval(tmp_path, monkeypatch)
    counted, diffs = [], []
    _diff_tree(before["model"], after["model"], "model", counted, diffs)
    n_model = len(counted)
    _diff_tree(before["opt"], after["opt"], "opt", counted, diffs)
    n_opt = len(counted) - n_model
    assert n_model >= 100 and n_opt >= 100, (n_model, n_opt)   # teeth
    # every parameter, every optimizer tensor, every leaf: identical. The
    # ONLY admissible differences are the two known-leak buffers (module
    # docstring (c)/(d)); a SUBSET so the leak's fix does not break this pin.
    allowed = {f"model/{k}" for k in PRIOR_BUFFERS}
    assert set(diffs) <= allowed, sorted(set(diffs) - allowed)
    params = {n for n, _ in seen["model"].named_parameters()}
    assert not (set(PRIOR_BUFFERS) & params)         # buffers, not params
    # the milestone file carries the same model tensors as ckpt.pt
    ms = torch.load(out / "ckpt_1.pt", map_location="cpu", weights_only=False)
    ms_diffs = []
    _diff_tree(ms["model"], before["model"], "milestone", [], ms_diffs)
    assert ms_diffs == []
    # the eval row exists, follows the training row, and is a real eval
    rows = _rows(out)
    assert [r["step"] for r in rows] == [1, 1]
    assert "loss" in rows[0] and "eval_loss" in rows[1]
    assert rows[1]["eval_batches"] == 2 and rows[1]["eval_windows"] == 4
    assert "eval_error" not in rows[1]
    cap = capsys.readouterr()
    assert cap.out.index("[v3:hier] ckpt step 1 -> ckpt.pt") \
        < cap.out.index("[v3:eval] step 1 ")
    assert "FAILED" not in cap.out


# ------------------------------------------------ (d) the leak, pinned
# (was a strict xfail pinning C-REFCV3-EVAL-PRIOR-LEAK; the gate landed 2026-09-02, so the
#  leak-free behaviour is now asserted positively)
def test_eval_leaves_the_tactical_prior_buffers_untouched(tmp_path,
                                                          monkeypatch):
    before, after, _, _ = _run_non_raising_eval(tmp_path, monkeypatch)
    for k in PRIOR_BUFFERS:
        assert torch.equal(before["model"][k], after["model"][k]), k


# ------------------------------------------- (e) the order is in the source
def test_step_loop_saves_before_it_evals_in_the_source():
    """Cheap textual pin so a future edit that swaps the blocks back fails
    loudly even if the rig above is ever skipped."""
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    loop = src[src.index("    while step < args.steps:"):
               src.index("    # ⛔ the done-marker")]
    i_save = loop.index('"opt": opt.state_dict(), "step": step}, ck)')
    i_eval = loop.index("if eval_dl is not None and (step % args.eval_every")
    assert i_save < i_eval, "ckpt.pt must be written BEFORE the eval block"
    assert loop.index("if step in MILESTONES:") < i_eval
    assert "except Exception as exc:" in loop and '"eval_error"' in loop
    assert loop.count("model.eval()") == 1 and loop.count("model.train()") == 1
