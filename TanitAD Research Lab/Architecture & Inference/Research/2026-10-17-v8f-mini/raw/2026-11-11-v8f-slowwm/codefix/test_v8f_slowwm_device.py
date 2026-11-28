"""PREREG_V8F_SLOWWM, AMEND_SLOWWM_3 (code fix): ``WMAux.built_basis`` (the G-DVB reader of ``--wm-basis``) planted its probe ``u`` on the CPU, so on a module that lives on the GPU the
launch gate crashed in ``layer_norm`` (``weight is on cuda:0, different from other tensors on cpu``, box 2, 2026-11-12 00:52, ``bad_wm_basis`` -> ``built_basis``). Every earlier test ran CPU-only.

These tests need no GPU: the ``meta`` device is the one non-CPU device torch offers on a CPU-only box, and a meta module meeting a CPU tensor is the same device mismatch.
* ``test_built_basis_reads_back_on_a_non_cpu_module`` is RED at f7e790ab (RuntimeError, device mismatch) and GREEN with the fix.
* ``test_the_planted_basis_values_are_the_cpu_values_bit_for_bit`` pins that the fix did not change the planted values: on the CPU the probe is exactly ``torch.randn(1, 4, d)`` of the generator seeded 0.
* ``test_level_head_vision_probes_draw_their_noise_on_the_feats_device`` pins the two sibling CPU-drawn tensors (the vision-only probes of the level / lead heads) at the source level.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts", REPO / "stack" / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.models import v8f_mini as M  # noqa: E402
from test_v8f_slowwm import model_of  # noqa: E402

torch.set_num_threads(1)
EXPECT = {"mode": "slow", "in_dim": 256, "mean_path_refused": True, "token_path_refused": True, "basis_path_refused": False, "basis_path_is_proj_tail": True}


@pytest.fixture
def record_u(monkeypatch):
    """Records every ``u`` handed to ``_z1_post`` and lets ``torch.equal`` compare meta tensors (it has no meta kernel: shape and dtype stand in for the values there)."""
    seen = []
    real_post, real_equal = M.WMAux._z1_post, torch.equal

    def post(self, am):
        seen.append(am.detach().clone())
        return real_post(self, am)

    def equal(a, b):
        if a.device.type == "meta" or b.device.type == "meta":
            return a.device == b.device and a.shape == b.shape and a.dtype == b.dtype
        return real_equal(a, b)
    monkeypatch.setattr(M.WMAux, "_z1_post", post)
    monkeypatch.setattr(torch, "equal", equal)
    return seen


def test_built_basis_reads_back_on_a_non_cpu_module(record_u):
    _, _, m = model_of(["--wm-basis", "slow"])
    wm = m.wm.to("meta")
    assert next(wm.parameters()).device.type == "meta"
    assert wm.built_basis() == EXPECT                                           # f7e790ab: RuntimeError (the planted u is on the CPU, the weights are not)
    assert record_u and all(u.device.type == "meta" and tuple(u.shape) == (1, 4, 256) for u in record_u)


def test_the_planted_basis_values_are_the_cpu_values_bit_for_bit(record_u):
    _, _, m = model_of(["--wm-basis", "slow"])
    assert m.wm.built_basis() == EXPECT
    want = torch.randn(1, 4, 256, generator=torch.Generator().manual_seed(0))
    assert record_u and all(u.device.type == "cpu" and u.dtype == torch.float32 and torch.equal(u, want) for u in record_u)


@pytest.mark.parametrize("fn", [M.assert_level_head_vision_only, M._vision_only_run])
def test_level_head_vision_probes_draw_their_noise_on_the_feats_device(fn):
    src = inspect.getsource(fn)
    assert 'generator=torch.Generator().manual_seed(3)).to(device=batch["feats"].device, dtype=batch["feats"].dtype)' in src
