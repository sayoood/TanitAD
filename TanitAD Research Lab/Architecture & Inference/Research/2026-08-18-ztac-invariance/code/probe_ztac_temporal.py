"""INDEPENDENT re-verification of C115: does ``z_tac`` integrate the window?

⛔ THIS FILE DELIBERATELY DOES NOT IMPORT ``stack/tests/test_v6_t2_contrastive.py``.
C115 rests on one stream and one night. The brief's rule is "re-verify the finding
yourself before building on it, two probes". Re-running the original author's
assertions would re-measure their instrument, not the architecture. So the two
probes here use MECHANISMS THE ORIGINAL DID NOT USE:

  * P1 — AUTOGRAD REACHABILITY. Backprop a scalar from ``z_tac`` to the INPUT
    PIXELS and read ``frames.grad`` per time index. Independence shows up as an
    EXACT structural zero, not as an ``allclose`` tolerance. (The original probe
    froze inputs and compared outputs; this one asks the graph directly.)
  * P2 — CROSS-SAMPLE SUBSTITUTION. Splice a DIFFERENT batch element's history
    onto the same last frame. If ``z_tac`` integrates a window, two windows that
    share only their last frame must differ.

⭐ EVERY PROBE CARRIES BOTH CONTROLS THE PROGRAMME REQUIRES:

  * POSITIVE CONTROL — the SAME probe, same graph, same tensors, applied to
    ``zh_op`` (the operative predictor's output). ``OperativePredictor.forward``
    runs CAUSAL SELF-ATTENTION over the window (``predictor.py:167-170``), so it
    MUST show window dependence. A probe that reports "no dependence" for BOTH
    heads is inert (C109: a positive control that cannot fire proves nothing).
  * TRIVIAL-PROXY CONTROL — is the network merely DEAD? A constant network is
    invariant to everything, which would make P1/P2 pass for the wrong reason.
    So we assert ``z_tac`` genuinely VARIES across batch elements, and that the
    per-frame operative states genuinely VARY across time.

Run:  PYTHONUTF8=1 python probe_ztac_temporal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_STACK = Path(__file__).resolve()
while _STACK.name != "TanitAD":
    _STACK = _STACK.parent
_STACK = _STACK / "stack"
sys.path.insert(0, str(_STACK))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402
from tanitad.models.tactical import PhiTac, TacticalStage0  # noqa: E402

B, W, C, H, Wd = 4, 4, 9, 64, 64
ATOL = 0.0  # bit-identity, not tolerance

R: dict = {"probes": {}, "controls": {}, "meta": {}}


def cfg(**kw) -> V6Config:
    """A small but STRUCTURALLY FAITHFUL build: the flatten in
    ``encode_window`` and the causal mask in ``OperativePredictor`` are the
    same code at any geometry, so the mechanism is geometry-independent."""
    return V6Config(**{**dict(
        encoder=EncoderConfig(in_channels=C, image_size=H, image_width=Wd,
                              patch_size=16, d_model=32, depth=2, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=2, n_heads=2, window=W,
                                  horizons=(1,), action_dim=3, residual=True),
        d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, f_blocks=1, aux_hidden=16, sigreg_slices=8,
        plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6),
        hz_op=10.0, hz_tac=2.0, hz_str=0.5, d_plan_feat=16,
        emission_hidden=16, d_goal_embed=128, n_candidates=8), **kw})


def build(seed: int = 0, **kw) -> V6Stack:
    torch.manual_seed(seed)
    m = V6Stack(cfg(**kw))
    m.eval()
    return m


def inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randn(B, W, C, H, Wd, generator=g)
    actions = torch.randn(B, W, 3, generator=g)
    v0 = torch.rand(B, generator=g) * 15.0
    return frames, actions, v0


def fwd(m: V6Stack, frames, actions, v0, seed: int = 7):
    torch.manual_seed(seed)          # kill any RNG path as an explanation
    return m(frames, actions, v0)


def scalarise(z: torch.Tensor, seed: int = 4242) -> torch.Tensor:
    """⛔ NEVER ``z.sum()`` FOR A LAYERNORM-TERMINATED MODULE.

    MEASURED while building this probe, and it is a DEFECT I NEARLY REPORTED AS
    A FINDING. ``adapter_tac`` / ``adapter_str`` both end in ``nn.LayerNorm``
    (v6.py:3739, 3753). LayerNorm inits ``gamma=1, beta=0``, and its output is
    zero-mean across the normalised axis BY CONSTRUCTION — so ``z.sum()`` is
    the CONSTANT 0 for every input, and its gradient wrt the pixels is EXACTLY
    ZERO no matter what the network does. The first run of this probe returned
    all-zero gradients for ``z_tac`` INCLUDING THE LAST FRAME, which would have
    "confirmed" C115 far too strongly (and falsely: P2 proves the last frame
    DOES matter).

    ⇒ This is C109's class — a probe inert BY CONSTRUCTION at the setting it is
    run at — occurring inside the instrument built to check somebody else's
    instrument. A FIXED RANDOM PROJECTION has no such null space.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(z.shape[-1], generator=g)
    return (z * w).sum()


# --------------------------------------------------------------------------
# P1 — AUTOGRAD REACHABILITY (+ positive control on the SAME graph)
# --------------------------------------------------------------------------
def p1_autograd_reachability():
    # ⚠️ CONFOUND FOUND WHILE BUILDING THIS PROBE, AND IT MATTERS.
    # Under the DEFAULT config ``isolate_uplink=True`` (v6.py:2989), so
    # ``uplink_tac`` does ``src = self._cut(z_op, cfg.isolate_uplink)`` ->
    # ``z_op.detach()`` (v6.py:4114). The gradient from ``z_tac`` to ``frames``
    # is then None at EVERY time index INCLUDING THE LAST — which superficially
    # reads as "z_tac ignores all frames" but is the X3 stop-grad, a COMPLETELY
    # DIFFERENT MECHANISM. An autograd probe run only at the default would have
    # "confirmed" the finding for the wrong reason. So P1 is run at BOTH
    # settings and both are reported.
    m_def = build()                              # isolate_uplink=True (default)
    f_def = inputs()[0].clone().requires_grad_(True)
    _fr, actions_d, v0_d = inputs()
    out_def = fwd(m_def, f_def, actions_d, v0_d)
    g_def = torch.autograd.grad(scalarise(out_def["z_tac"]), f_def,
                                retain_graph=False, allow_unused=True)[0]
    R["probes"]["P1_default_config_is_UNINFORMATIVE"] = {
        "grad_is_None_because_of_stopgrad": g_def is None,
        "isolate_uplink": True,
        "why": "v6.py:4114 `src = self._cut(z_op, cfg.isolate_uplink)`; the X3 "
               "stop-grad severs the graph, so autograd cannot speak about "
               "temporal structure at the default setting. NOT evidence.",
    }

    # The informative run: open the uplink so the graph reaches the pixels.
    m = build(isolate_uplink=False)
    frames, actions, v0 = inputs()
    frames = frames.clone().requires_grad_(True)
    out = fwd(m, frames, actions, v0)

    z_tac = out["z_tac"]
    g_tac = torch.autograd.grad(scalarise(z_tac), frames, retain_graph=True,
                                allow_unused=True)[0]
    per_t_tac = [float(g_tac[:, t].abs().max()) for t in range(W)]

    # POSITIVE CONTROL: same input tensor, same graph, a head that MUST see the
    # window (causal attention over all W positions).
    zh_op = out["zhat_op"][1]
    g_op = torch.autograd.grad(scalarise(zh_op), frames, retain_graph=True,
                               allow_unused=True)[0]
    per_t_op = [float(g_op[:, t].abs().max()) for t in range(W)]

    # z_str is DOWNSTREAM of z_tac -> inherits the property. Reported, not assumed.
    g_str = torch.autograd.grad(scalarise(out["z_str"]), frames,
                                retain_graph=True, allow_unused=True)[0]
    per_t_str = [float(g_str[:, t].abs().max()) for t in range(W)]

    R["probes"]["P1_autograd"] = {
        "grad_absmax_per_frame_z_tac": per_t_tac,
        "grad_absmax_per_frame_z_str": per_t_str,
        "history_frames_are_EXACT_zero_z_tac": all(v == 0.0 for v in per_t_tac[:-1]),
        "last_frame_is_nonzero_z_tac": per_t_tac[-1] > 0.0,
        "history_frames_are_EXACT_zero_z_str": all(v == 0.0 for v in per_t_str[:-1]),
    }
    # ⭐ PIN THE DEFECT so it cannot come back: the naive .sum() scalar must be
    # demonstrably inert on this module. This is the control on the CONTROL.
    g_naive = torch.autograd.grad(out["z_tac"].sum(), frames,
                                  retain_graph=True, allow_unused=True)[0]
    R["controls"]["P1_layernorm_sum_null_is_REAL"] = {
        "naive_sum_grad_absmax_last_frame": (
            0.0 if g_naive is None else float(g_naive[:, -1].abs().max())),
        "naive_sum_scalar_value": float(out["z_tac"].sum()),
        "inert_as_predicted": (g_naive is None
                               or float(g_naive.abs().max()) == 0.0),
        "why": "adapter_tac ends in nn.LayerNorm (v6.py:3739); at gamma=1/beta=0 "
               "the sum over the normalised axis is identically 0, so .sum() "
               "has an exactly-zero gradient for ANY input. Kept as a pinned "
               "negative control: it is what a careless probe would report.",
    }
    R["controls"]["P1_positive_zh_op"] = {
        "grad_absmax_per_frame_zh_op": per_t_op,
        "ALL_frames_nonzero": all(v > 0.0 for v in per_t_op),
        "fires": all(v > 0.0 for v in per_t_op),
        "note": "operative predictor consumes z_op_win with a causal mask; if "
                "this were also zero the probe would be inert (C109).",
    }


# --------------------------------------------------------------------------
# P2 — CROSS-SAMPLE HISTORY SUBSTITUTION (+ positive + trivial-proxy controls)
# --------------------------------------------------------------------------
def p2_history_substitution():
    m = build()
    frames, actions, v0 = inputs(seed=1)
    other, _, _ = inputs(seed=99)            # a genuinely different history

    a = frames.clone()
    b = frames.clone()
    b[:, :-1] = other[:, :-1]                # SAME last frame, DIFFERENT history

    za = fwd(m, a, actions, v0)
    zb = fwd(m, b, actions, v0)

    bitwise = bool(torch.equal(za["z_tac"], zb["z_tac"]))
    maxdiff = float((za["z_tac"] - zb["z_tac"]).abs().max())

    # POSITIVE CONTROL: change the LAST frame instead. z_tac MUST move.
    c = frames.clone()
    c[:, -1] = other[:, -1]
    zc = fwd(m, c, actions, v0)
    last_diff = float((za["z_tac"] - zc["z_tac"]).abs().max())
    scale = float(za["z_tac"].abs().max())

    # POSITIVE CONTROL 2: the operative predictor under the SAME substitution.
    op_a = za["zhat_op"][1]
    op_b = zb["zhat_op"][1]
    op_hist_diff = float((op_a - op_b).abs().max())

    # TRIVIAL-PROXY CONTROL: is the net simply constant/dead?
    z_tac_batch_std = float(za["z_tac"].std(dim=0).mean())
    zopwin = za["z_op_win"]
    per_frame_spread = float((zopwin - zopwin.mean(dim=1, keepdim=True))
                             .abs().mean())

    R["probes"]["P2_history_substitution"] = {
        "z_tac_BIT_IDENTICAL_under_different_history": bitwise,
        "z_tac_maxabsdiff_history_swapped": maxdiff,
        "z_tac_maxabsdiff_last_frame_swapped": last_diff,
        "z_tac_absmax_scale": scale,
    }
    R["controls"]["P2_positive_last_frame"] = {
        "z_tac_moves_when_last_frame_changes": last_diff > 0.0,
        "relative_move": last_diff / max(scale, 1e-12),
        "fires": last_diff > 0.0,
    }
    R["controls"]["P2_positive_zh_op_sees_history"] = {
        "zh_op_maxabsdiff_history_swapped": op_hist_diff,
        "fires": op_hist_diff > 0.0,
    }
    R["controls"]["P2_trivial_proxy_network_is_not_dead"] = {
        "z_tac_std_across_batch": z_tac_batch_std,
        "z_op_win_mean_abs_spread_across_time": per_frame_spread,
        "not_dead": z_tac_batch_std > 1e-6 and per_frame_spread > 1e-6,
    }


# --------------------------------------------------------------------------
# P3 — TRAIN MODE. Rules out a norm/regulariser that only couples in train().
# --------------------------------------------------------------------------
def p3_train_mode():
    m = build()
    m.train()
    frames, actions, v0 = inputs(seed=1)
    other, _, _ = inputs(seed=99)
    b = frames.clone()
    b[:, :-1] = other[:, :-1]
    za = fwd(m, frames.clone(), actions, v0)
    zb = fwd(m, b, actions, v0)
    R["probes"]["P3_train_mode"] = {
        "z_tac_BIT_IDENTICAL_in_train_mode": bool(
            torch.equal(za["z_tac"], zb["z_tac"])),
        "maxabsdiff": float((za["z_tac"] - zb["z_tac"]).abs().max()),
        "note": "BatchNorm is banned in the inference path (encoder.py:5); this "
                "probe is what makes that a MEASURED fact here rather than an "
                "INHERITED one.",
    }


# --------------------------------------------------------------------------
# P4 — SCOPE. The programme has TWO tactical implementations. Measure BOTH,
# because "the tactical layer is fake" is the over-correction to guard against.
# --------------------------------------------------------------------------
def p4_scope_tacticalstage0():
    """``TacticalStage0`` (tactical.py:448) is the SPEC-CONFORMANT tactical
    layer: ``z_tac = self.phi_tac(z_op)`` (tactical.py:505) with ``PhiTac`` a
    CAUSAL TCN over the window (tactical.py:99). It is not shelf-ware — it was
    TRAINED (MODEL_REGISTRY.md §1.13b, E4.4, 2026-08-10).

    If its z_tac ALSO ignored history, the defect would be programme-wide. It
    does not — and that is the boundary of the finding."""
    torch.manual_seed(0)
    d_op, W4 = 32, 4
    m = TacticalStage0(d_op=d_op, d_tac=16, window=W4, phi_hidden=16)
    m.eval()
    g = torch.Generator().manual_seed(5)
    z = torch.randn(B, W4, d_op, generator=g)
    other = torch.randn(B, W4, d_op, generator=g)

    a = z.clone()
    b = z.clone()
    b[:, :-1] = other[:, :-1]          # same last slot, different history
    za = m(a)["z_tac"]
    zb = m(b)["z_tac"]
    hist_diff = float((za - zb).abs().max())

    # POSITIVE CONTROL for this probe: change the LAST slot instead.
    c = z.clone()
    c[:, -1] = other[:, -1]
    last_diff = float((za - m(c)["z_tac"]).abs().max())

    # gradient view: history slots must be NONZERO here (the mirror image of P1)
    zg = z.clone().requires_grad_(True)
    gz = torch.autograd.grad(scalarise(m(zg)["z_tac"]), zg)[0]
    per_slot = [float(gz[:, t].abs().max()) for t in range(W4)]

    R["probes"]["P4_TacticalStage0_DOES_integrate"] = {
        "z_tac_maxabsdiff_history_swapped": hist_diff,
        "z_tac_maxabsdiff_last_slot_swapped": last_diff,
        "grad_absmax_per_slot": per_slot,
        "ALL_history_slots_nonzero": all(v > 0.0 for v in per_slot),
        "INTEGRATES_THE_WINDOW": hist_diff > 0.0 and all(v > 0.0 for v in per_slot),
        "note": "SAME probe, SAME scalarise(), opposite result -> the v6 result "
                "is a fact about V6Stack, NOT about 'the tactical layer'.",
    }


def main() -> int:
    # ⛔ C114: v6.py is being edited by sibling streams WHILE this runs (it grew
    # 4914 -> 5154 lines mid-session). A measurement on a moving tree must name
    # the exact bytes it measured, or its line citations are unreproducible.
    import hashlib
    src = _STACK / "tanitad" / "models" / "v6.py"
    tac = _STACK / "tanitad" / "models" / "tactical.py"
    R["meta"] = {"torch": torch.__version__, "B": B, "W": W,
                 "shape": [C, H, Wd], "device": "cpu",
                 "v6_py_sha256": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
                 "v6_py_lines": len(src.read_bytes().splitlines()),
                 "tactical_py_sha256": hashlib.sha256(tac.read_bytes()).hexdigest()[:16]}
    p1_autograd_reachability()
    p2_history_substitution()
    p3_train_mode()
    p4_scope_tacticalstage0()

    p1 = R["probes"]["P1_autograd"]
    p2 = R["probes"]["P2_history_substitution"]
    p3 = R["probes"]["P3_train_mode"]
    c1 = R["controls"]["P1_positive_zh_op"]
    c2 = R["controls"]["P2_positive_last_frame"]
    c3 = R["controls"]["P2_trivial_proxy_network_is_not_dead"]

    controls_ok = c1["fires"] and c2["fires"] and c3["not_dead"]
    finding = (p1["history_frames_are_EXACT_zero_z_tac"]
               and p1["last_frame_is_nonzero_z_tac"]
               and p2["z_tac_BIT_IDENTICAL_under_different_history"]
               and p3["z_tac_BIT_IDENTICAL_in_train_mode"])

    R["verdict"] = {
        "controls_all_fired": controls_ok,
        "C115_REPRODUCED_independently": bool(finding and controls_ok),
        "admissible": bool(controls_ok),
    }
    print(json.dumps(R, indent=2))
    return 0 if controls_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
