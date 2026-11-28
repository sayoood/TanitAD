"""refcv8 seams consume NO global RNG (WP-B, 2026-10-05).

MEASURED on the dev-box CPU smoke of the ladder's V0 and V-R8 (same seed, same first batch: the fed-speed CRC was
identical): the agent terms were identical on 36 of 36 keys. The 10 cm map terms differed on 184 of 354, and box3d
on 18 of 53. Mechanism: `enable_refcv8` drew from the global stream inside the model, and the trainer builds the
perception and map branches AFTER the model, so those heads got a different init in V-R8. That is a second variable
in L1, and it failed the registered I-0 arm-level identity. The deliberate-regression arm re-opens the global stream
(a fork that does not restore) and must go RED.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402


def _after(refcv8: bool, seed: int = 0, **kw):
    T = R.trainer()
    cfg, m = R.build(T, refcv8, seed=seed, **kw)
    return m, torch.rand(16)            # what the trainer's NEXT module would draw (the map / box branch)


def test_the_global_stream_leaves_the_seam_build_exactly_as_it_entered():
    _m0, a = _after(False)
    _m1, b = _after(True)
    _m2, c = _after(True, critic_drivable=True, v9_cons=True)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_every_shared_parameter_is_identical_with_and_without_the_seams():
    m0, _ = _after(False)
    m1, _ = _after(True)
    s0, s1 = m0.state_dict(), m1.state_dict()
    shared = [k for k in s0 if k in s1]
    assert len(shared) == len(s0)                                  # refcv8 only ADDS keys
    assert all(torch.equal(s0[k], s1[k]) for k in shared)


def test_the_seams_init_from_their_own_stream():
    """The seam parameters do not depend on the global seed: a dedicated stream, refcv8 seed + R8_INIT_SEED_OFFSET."""
    m1, _ = _after(True, seed=0)
    m2, _ = _after(True, seed=123)
    s1, s2 = m1.state_dict(), m2.state_dict()
    m0, _ = _after(False)
    seam = [k for k in s1 if k not in m0.state_dict()]
    assert seam and all(torch.equal(s1[k], s2[k]) for k in seam)


def test_RED_ARM_a_seam_build_on_the_global_stream_shifts_the_next_init(monkeypatch):
    @contextlib.contextmanager
    def no_fork(*_a, **_k):              # the pre-fix behaviour: nothing is restored
        yield
    _m0, a = _after(False)
    monkeypatch.setattr(torch.random, "fork_rng", no_fork)
    _m1, b = _after(True)
    assert not torch.equal(a, b)
