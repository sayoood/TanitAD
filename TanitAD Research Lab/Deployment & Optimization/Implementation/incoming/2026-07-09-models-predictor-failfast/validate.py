"""Fail-fast input validation for the operative predictor (models review #2).

Production & Optimization compliance review of `stack/tanitad/models/`. The
operative predictor runs on the batch-1 streaming hot path every decision tick.
Its only input guard today is

    assert w == self.cfg.window, f"window mismatch: {w} != {self.cfg.window}"

which has two production failure modes (both measured, see INTAKE.md):

1. `assert` is stripped under `python -O` / `-OO`. With the assert gone, a
   caller that passes a shorter window (e.g. states/actions of window W-1 at
   stream start before the causal buffer is full) runs SILENTLY: pos-embedding
   slice, causal mask and FiLM cond all re-align to the shorter length, so the
   predictor happily produces an output for a window it was never configured
   for — the same silent-wrong-data class as the epcache / cosmos bugs.
2. A wrong state_dim or action_dim raises a cryptic `RuntimeError: mat1 and
   mat2 shapes cannot be multiplied (8x2047 and 2048x768)` from deep inside a
   Linear — no hint that the *contract* was violated at the module boundary.

`validate_operative_inputs` replaces the assert with explicit, always-on
(`-O`-proof) `ValueError`s that name the offending axis. The checks are pure
Python-int comparisons on `tensor.shape[i]`; on a static-shape ONNX export they
constant-fold exactly like the assert they replace (which already exported
clean at opset 17/18, see the 2026-07-08 ONNX note), so export is unaffected —
proven by `tests/test_predictor_failfast.py::test_export_safe`.
"""

from __future__ import annotations

from torch import Tensor


def validate_operative_inputs(states: Tensor, actions: Tensor, window: int,
                              state_dim: int, action_dim: int) -> None:
    """Raise ValueError if (states, actions) violate the predictor contract.

    states  : [B, window, state_dim]
    actions : [B, window, action_dim]

    Named-axis, `-O`-proof fail-fast for the operative decision tick. No-op on
    valid input; constant-folds to nothing on static-shape export.
    """
    if states.ndim != 3 or actions.ndim != 3:
        raise ValueError(
            "operative predictor expects 3-D (states, actions) "
            f"[B, W, D]; got states.ndim={states.ndim}, actions.ndim={actions.ndim}")
    b, w, s = states.shape
    ba, wa, a = actions.shape
    if b != ba or w != wa:
        raise ValueError(
            "states/actions batch|window mismatch: "
            f"states={tuple(states.shape)} vs actions={tuple(actions.shape)}")
    if w != window:
        raise ValueError(
            f"window mismatch: got {w}, predictor configured for {window} "
            "(buffer the causal window to full length before the decision tick)")
    if s != state_dim:
        raise ValueError(f"state_dim mismatch: got {s}, expected {state_dim}")
    if a != action_dim:
        raise ValueError(f"action_dim mismatch: got {a}, expected {action_dim}")
