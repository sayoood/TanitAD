"""Standalone tests for the operative-predictor fail-fast validator.

Run: pytest "TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-07-09-models-predictor-failfast/tests" -q

The validator is self-contained (imports only torch), so these tests do not
need the tanitad package. The export-safety test is skipped if onnx is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate import validate_operative_inputs  # noqa: E402

W, S, A = 8, 2048, 2


def _ok():
    return torch.randn(1, W, S), torch.randn(1, W, A)


def test_valid_input_is_noop():
    s, a = _ok()
    assert validate_operative_inputs(s, a, W, S, A) is None


def test_wrong_window_raises_valueerror():
    # Current stack code raises AssertionError here (stripped under -O).
    s, a = torch.randn(1, W - 1, S), torch.randn(1, W - 1, A)
    with pytest.raises(ValueError, match="window mismatch"):
        validate_operative_inputs(s, a, W, S, A)


def test_wrong_state_dim_raises_clear_error():
    # Current stack code raises a cryptic matmul RuntimeError here.
    s, a = torch.randn(1, W, S - 1), torch.randn(1, W, A)
    with pytest.raises(ValueError, match="state_dim mismatch"):
        validate_operative_inputs(s, a, W, S, A)


def test_wrong_action_dim_raises_clear_error():
    s, a = torch.randn(1, W, S), torch.randn(1, W, A + 1)
    with pytest.raises(ValueError, match="action_dim mismatch"):
        validate_operative_inputs(s, a, W, S, A)


def test_states_actions_window_disagree():
    s, a = torch.randn(1, W, S), torch.randn(1, W - 1, A)
    with pytest.raises(ValueError, match="batch.window mismatch"):
        validate_operative_inputs(s, a, W, S, A)


def test_non_3d_raises():
    with pytest.raises(ValueError, match="3-D"):
        validate_operative_inputs(torch.randn(W, S), torch.randn(W, A), W, S, A)


def test_survives_python_O_semantics():
    """The guard must NOT rely on `assert` (which -O strips). Parse the module
    and assert there are zero real `assert` STATEMENTS (docstring mentions of
    the word don't count) — a regression tripwire."""
    import ast
    src = (Path(__file__).resolve().parents[1] / "validate.py").read_text()
    asserts = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assert)]
    assert asserts == []


def test_export_safe():
    """A tiny predictor whose forward calls the validator still exports to ONNX
    at opset 17 — the shape checks constant-fold on static shapes (same as the
    assert they replace, which exported clean per the 2026-07-08 ONNX note)."""
    onnx = pytest.importorskip("onnx")  # noqa: F841

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(S, 16)
            self.head = nn.Linear(16 + A, S)

        def forward(self, states: Tensor, actions: Tensor) -> Tensor:
            validate_operative_inputs(states, actions, W, S, A)
            h = self.proj(states[:, -1])
            return self.head(torch.cat([h, actions[:, -1]], dim=-1))

    m = Tiny().eval()
    s, a = _ok()
    out_path = Path(__file__).with_name("_tiny_export.onnx")
    try:
        torch.onnx.export(m, (s, a), str(out_path),
                          input_names=["states", "actions"], output_names=["z"],
                          opset_version=17, dynamo=False)
        assert out_path.exists() and out_path.stat().st_size > 0
    finally:
        if out_path.exists():
            out_path.unlink()
