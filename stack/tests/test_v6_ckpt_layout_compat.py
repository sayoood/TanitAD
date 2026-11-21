"""The eval/probe loaders must understand the v6 checkpoint layout.

⛔ WHY. The P-battery died on a real v6 checkpoint with ``KeyError: 'model'``
(MEASURED 2026-08-14 on pod4, run `v6F-pbattery2`). The gate scripts and the
replay arms predate v6 and assumed the v5 wrapper key. The silent half of the
bug is worse than the loud half: the fallback path returns the ENTIRE
checkpoint dict as a state_dict, so ``opt``/``step``/``config`` masquerade as
parameter entries rather than failing.

These tests pin both loaders against all three layouts.
"""
import torch

from tanitad.eval.ckpt_compat import state_dict_of
from tanitad.replay.arms import load_checkpoint_state

SD = {"enc.w": torch.zeros(2, 2), "pred.b": torch.ones(3)}


def _same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_state_dict_of_v6_stack_wrapper():
    assert _same(state_dict_of({"stack": SD, "opt": {}, "step": 2000,
                                "config": {}}), SD)


def test_state_dict_of_v5_model_wrapper():
    assert _same(state_dict_of({"model": SD, "step": 7}), SD)


def test_state_dict_of_bare():
    assert _same(state_dict_of(dict(SD)), SD)


def test_state_dict_of_prefers_model_when_both_present():
    """`model` wins — a v5 ckpt that also carries an unrelated `stack` entry
    must not be re-routed."""
    other = {"z": torch.full((1,), 9.0)}
    assert _same(state_dict_of({"model": SD, "stack": other}), SD)


def test_load_checkpoint_state_reads_v6(tmp_path):
    p = tmp_path / "ckpt.pt"
    torch.save({"stack": SD, "opt": {}, "step": 2000, "config": {}}, p)
    sd, step = load_checkpoint_state(p)
    assert _same(sd, SD)
    assert step == 2000


def test_load_checkpoint_state_reads_v5(tmp_path):
    p = tmp_path / "ckpt.pt"
    torch.save({"model": SD, "step": 30000}, p)
    sd, step = load_checkpoint_state(p)
    assert _same(sd, SD)
    assert step == 30000


def test_load_checkpoint_state_bare_reports_unknown_step(tmp_path):
    p = tmp_path / "ckpt.pt"
    torch.save(SD, p)
    sd, step = load_checkpoint_state(p)
    assert _same(sd, SD)
    assert step == -1


def test_v6_wrapper_keys_never_leak_into_the_state_dict(tmp_path):
    """The regression that motivated this file: wrapper metadata must not be
    returned as if it were parameters."""
    p = tmp_path / "ckpt.pt"
    torch.save({"stack": SD, "opt": {}, "step": 2000, "config": {}}, p)
    sd, _ = load_checkpoint_state(p)
    assert not ({"opt", "step", "config", "stack"} & set(sd))
