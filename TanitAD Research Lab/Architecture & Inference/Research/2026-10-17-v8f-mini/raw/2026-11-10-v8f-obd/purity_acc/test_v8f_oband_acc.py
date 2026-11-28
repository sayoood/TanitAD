"""PROPOSED test (NOT applied anywhere): the launch purity probe `oband_probe` must read the acceleration term (`--w-acc`) as an O term.
Lives beside stack/tests/test_v8f_otband.py (imports its helpers). See theory/obd/PURITY_ACC_FINDING.md."""
from __future__ import annotations

import pytest
import torch

from test_v8f_otband import BAND, M, TR, build, make_batch, world  # noqa: F401  (world is the module fixture)

W_ACC = 49.6969


def _by_term(world, **kw):
    b, J = make_batch(world)
    m = build(J, wake=False, detach_upper=True, w_acc=W_ACC, **kw)
    return TR.oband_probe(m, b, m.cfg)["purity"]["by_term"], b, J


def test_the_probe_reads_the_acceleration_term_on_the_control_arm(world):
    by, _, _ = _by_term(world)
    assert set(by) == {"ade", "fde", "acc"} and all(v["upper_grad_abs_sum"] == 0.0 and v["head_o"] > 0.0 for v in by.values())


def test_the_probe_reads_the_acceleration_term_on_the_band_arm(world):
    by, _, _ = _by_term(world, **BAND)
    assert set(by) == {"op", "tac", "acc"} and all(v["upper_grad_abs_sum"] == 0.0 and v["head_o"] > 0.0 for v in by.values())


def test_without_w_acc_the_probe_reads_what_it_read_before(world):
    b, J = make_batch(world)
    m = build(J, wake=False, detach_upper=True)
    assert set(TR.oband_probe(m, b, m.cfg)["purity"]["by_term"]) == {"ade", "fde"}


def test_deliberate_regression_a_leak_that_only_the_acceleration_term_carries_goes_red(world, monkeypatch):
    """The acceleration term alone is wired to S's head (1e-6 x a parameter sum): ade / fde stay pure, so only the new key can see it."""
    b, J = make_batch(world)
    m = build(J, wake=False, detach_upper=True, w_acc=W_ACC)
    orig = M.V8FMini.loss

    def leaky(self, *a, **k):                                                     # the copy the probe makes is the model `self` here
        out = orig(self, *a, **k)
        out["terms"]["acc"] = out["terms"]["acc"] + 1e-6 * sum(p.sum() for p in self.head_s.parameters())
        return out
    monkeypatch.setattr(M.V8FMini, "loss", leaky)
    with pytest.raises(AssertionError, match="O's acc term puts gradient"):
        TR.oband_probe(m, b, m.cfg)


def test_deliberate_regression_the_detach_removed_goes_red_with_the_acceleration_term_on(world):
    b, J = make_batch(world)
    plain = build(J, wake=False, w_acc=W_ACC)
    import dataclasses
    with pytest.raises(AssertionError, match="a detach is missing"):
        TR.oband_probe(plain, b, dataclasses.replace(plain.cfg, detach_upper=True))
