"""`--skip-nonfinite` for `refa_v1_train.py` (D-REFAV1-SKIP-NONFINITE, 2026-09-03).

MEASURED on the clean epoch (Thor, `--precision bf16 --tf32`): two gradient-
overflow events in 3,750 steps — row 900 a pre-clip norm of 2.5e18, row 2750
`inf` with the batch loss at 1.28 and the next row at 2.26, ~150 steps to
recover — where the fp32 incumbent's 900 rows peaked at 4.3. On an `inf` total
norm `clip_grad_norm_` scales every gradient by 0 (finite elements → 0, an
`inf` element → NaN) and the optimizer steps anyway.

What is pinned:
(a) FLAG ON: a step whose pre-clip norm is non-finite leaves the weights
    EXACTLY where they were (the checkpoint after the poisoned step 2 equals
    the checkpoint after step 1, tensor by tensor), the row counts it
    (`skipped_steps` 1 from that row on), and the run continues finite.
(b) FLAG OFF (the default): the same poisoned step is APPLIED — the next row's
    loss is non-finite. That is the hazard the flag exists for, documented
    rather than assumed; the default path is otherwise unchanged (the
    identity constants in test_refa_v1_precision.py still hold).
(c) With no event the flag changes nothing: a 2-step run with and without it
    logs identical rows (`skipped_steps` 0) and identical checkpoints.

Injection: `torch.nn.utils.clip_grad_norm_` is wrapped for one run; on the
chosen step one gradient element is set to `inf` BEFORE the real clip runs, so
the trainer sees exactly what Thor saw (a genuine `inf` total norm from the
real clip, real scaling), not a mocked return value.
"""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

COMMON = ["--smoke", "--bs", "2", "--log-every", "1", "--seed", "0",
          "--device", "cpu"]


def _trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "refa_v1_train.py"
    spec = importlib.util.spec_from_file_location(
        "refa_v1_train_skip_nonfinite_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tr, out: Path, *extra: str, steps: int):
    argv = COMMON + ["--steps", str(steps), "--save-every", str(steps),
                     "--out", str(out)] + list(extra)
    assert tr.main(argv) == 0
    rows = [json.loads(l) for l in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert [r["step"] for r in rows] == list(range(1, steps + 1))
    return rows


def _weights(out: Path) -> dict:
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
    return {k: v.clone() for k, v in ck["model"].items()}


class _Poison:
    """Wrap clip_grad_norm_ so that on `at_step`-th call one grad element is
    inf before the REAL clip runs."""

    def __init__(self, at_call: int):
        self.at_call = at_call
        self.calls = 0
        self.real = torch.nn.utils.clip_grad_norm_

    def __call__(self, params, max_norm, *args, **kw):
        params = list(params)
        self.calls += 1
        if self.calls == self.at_call:
            for p in params:
                if p.grad is not None:
                    p.grad.view(-1)[0] = float("inf")
                    break
        return self.real(params, max_norm, *args, **kw)


@pytest.fixture
def poison(monkeypatch):
    def arm(at_call: int):
        p = _Poison(at_call)
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", p)
        return p
    yield arm


def test_flag_on_skips_the_poisoned_step_and_keeps_weights(tmp_path, poison):
    tr = _trainer()
    _run(tr, tmp_path / "ref", steps=1)                 # weights after step 1
    ref = _weights(tmp_path / "ref")
    p = poison(at_call=2)
    rows = _run(tr, tmp_path / "skip", "--skip-nonfinite", steps=2)
    assert p.calls == 2
    got = _weights(tmp_path / "skip")
    assert not math.isfinite(rows[1]["grad_norm"]), rows[1]["grad_norm"]
    assert rows[0]["skipped_steps"] == 0 and rows[1]["skipped_steps"] == 1
    assert ref.keys() == got.keys()
    for k in ref:
        assert torch.equal(ref[k], got[k]), f"{k} moved on a skipped step"


def test_flag_on_run_continues_finite_after_the_event(tmp_path, poison):
    tr = _trainer()
    poison(at_call=2)
    rows = _run(tr, tmp_path / "cont", "--skip-nonfinite", steps=3)
    assert not math.isfinite(rows[1]["grad_norm"])
    assert math.isfinite(rows[2]["loss"]) and math.isfinite(rows[2]["grad_norm"])
    assert [r["skipped_steps"] for r in rows] == [0, 1, 1]


def test_flag_off_applies_the_poisoned_step(tmp_path, poison):
    """The documented hazard: without the flag the NaN'd gradient is APPLIED,
    the next row's loss is NaN, and the trainer's own non-finite refusal
    (SystemExit) ends the run — on Thor that is a supervisor relaunch from
    the last checkpoint, up to 1,000 steps lost. MEASURED here, not assumed."""
    tr = _trainer()
    poison(at_call=2)
    out = tmp_path / "off"
    argv = COMMON + ["--steps", "3", "--save-every", "3", "--out", str(out)]
    with pytest.raises(SystemExit):
        tr.main(argv)
    rows = [json.loads(l) for l in
            (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    assert [r["step"] for r in rows] == [1, 2, 3]
    assert not math.isfinite(rows[1]["grad_norm"])
    assert not math.isfinite(rows[2]["loss"]), rows[2]["loss"]
    assert [r["skipped_steps"] for r in rows] == [0, 0, 0]


def test_flag_is_inert_without_an_event(tmp_path):
    tr = _trainer()
    a = _run(tr, tmp_path / "a", steps=2)
    b = _run(tr, tmp_path / "b", "--skip-nonfinite", steps=2)
    keys = ("loss", "loss_feat_op", "grad_norm", "participation", "skipped_steps")
    for ra, rb in zip(a, b):
        for k in keys:
            assert ra[k] == rb[k], (k, ra[k], rb[k])
    wa, wb = _weights(tmp_path / "a"), _weights(tmp_path / "b")
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


def test_config_stamps_the_flag(tmp_path):
    tr = _trainer()
    _run(tr, tmp_path / "c", "--skip-nonfinite", steps=1)
    cfg = json.loads((tmp_path / "c" / "config.json").read_text(encoding="utf-8"))
    text = json.dumps(cfg)
    assert "skip_nonfinite" in text or "skip-nonfinite" in text, \
        "the launch record must carry the guard"
