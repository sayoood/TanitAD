"""⛔ refcv6 F3: the per-stage (cascade) loss must REACH the trainer on the REAL `train()`.

MEASURED 2026-09-26 on the live run `refcv6-r101-s0` (A16 audit,
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv6-frozen-trunk-audit/`):
**0 of 668** training rows and **0 of 66** in-run eval rows carried `cascade`, and the stage-0..2
cascade heads and AdaLN modulations were bit-identical at steps 1,000 / 5,000 / 30,000.
`AnchoredDiffusionDecoder` exported `layer_u0_hat` / `layer_logits` (`refc.py`, "refcv6 F3: the
per-stage predictions the cascade loss needs"), but `RefCModel.forward` copied decoder outputs
through a WHITELIST that omitted them, so `compute_losses_v3`'s F3 block was skipped every step.

Why nothing caught it: every F3 test (`test_refcv6_diffusion.py`) calls the DECODER directly, and
`test_built_heads_receive_gradient.py` classifies TOP-LEVEL children, where `core` reads
GRADIENT_REACHES whatever its sub-modules do. This file asserts on the CONSUMER -- the rows the
real `train()` writes and the checkpoint it saves -- at LEAF level.

Expected values are LITERALS. The deliberate-regression arm re-introduces the historical defect
exactly (the two keys absent from the production forward's output) and must go RED.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.refs import anchor_meta as am  # noqa: E402
from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_sampler as rs  # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS  # noqa: E402

#: the live run's diffusion flags (its config.json `argv`), F7-F9 excepted: F7/F8 were off there
#: and F9 asserts the 117-anchor production vocabulary, which a CPU rig does not carry.
F1_F6 = ["--f1-random-t", "--f2-dd-step", "--f3-per-layer", "--f4-adaln", "--f5-focal",
         "--f5-emitting-conf", "--f6-w-u0-zero"]
STEPS = 3
#: stage 0 of the SMOKE decoder (2 layers; stage 1 is the emitted fan): it is reached by the
#: per-stage loss and by NOTHING else, and it is ZERO-INIT (`test_refcv6_diffusion.py::
#: test_F3_cascade_heads_are_ZERO_INIT`) -- so a non-zero value after training is a non-zero
#: gradient, with no reference run needed.
STAGE0_CONTROL = "core.decoder.cascade.control_heads.0.weight"


def _anchor_file(tmp: Path) -> Path:
    """A 20-anchor v0-conditioned alat vocabulary on the V3 horizons, INCLUDING straight-ahead."""
    a_lon = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    a_lat = torch.tensor([-1.5, -0.5, 0.0, 0.5])
    ctrl = torch.cartesian_prod(a_lon, a_lat)                         # [20, 2]
    n, s = ctrl.shape[0], len(V3_HORIZONS)
    u = ctrl[None, :, None, :].expand(1, n, s, 2).contiguous()
    anchors = rs.roll_controls(u, torch.tensor([10.0]), tuple(V3_HORIZONS),
                               control_units="alat")[0]
    art = am.build_anchor_artifact(anchors, ctrl, control_units="alat", horizons=V3_HORIZONS,
                                   dt=0.1, ref_speed_ms=10.0, kappa_cap=0.12, alat_v_floor=4.0,
                                   builder=None)
    p = tmp / "anchors_v0cond_alat_20.pt"
    torch.save(art, p)
    return p


def _train(tmp: Path) -> tuple[list, dict]:
    out = tmp / "run"
    argv = ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2",
            "--steps", str(STEPS), "--batch", "2", "--device", "cpu", "--save-every", "100",
            "--log-every", "1", "--sampler", "ddim", "--anchors", str(_anchor_file(tmp)),
            "--anchor-v0-conditioned", "--n-anchors", "20"] + F1_F6
    T.train(T.build_parser().parse_args(argv))
    rows = [json.loads(x) for x in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    ck = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["model"]
    return [r for r in rows if "loss" in r], ck


def _assert_f3_live(tmp: Path) -> None:
    rows, ck = _train(tmp)
    assert len(rows) == STEPS, f"expected {STEPS} logged training rows, got {len(rows)}"
    missing = [r["step"] for r in rows if "cascade" not in r]
    assert not missing, (f"F3 is built but training rows {missing} carry no `cascade`: the "
                         f"per-stage loss was skipped (the refcv6-r101-s0 defect)")
    for r in rows:
        assert math.isfinite(r["cascade"]) and r["cascade"] > 0.0, r
    w0 = float(ck[STAGE0_CONTROL].abs().max())
    assert w0 > 0.0, (f"{STAGE0_CONTROL} is still exactly 0.0 after {STEPS} steps: stage 0 "
                      f"received NO gradient (zero-init, so any gradient would have moved it)")


def test_F3_cascade_is_LOGGED_and_stage0_RECEIVES_gradient(tmp_path) -> None:
    _assert_f3_live(tmp_path)


def test_DELIBERATE_REGRESSION_the_as_shipped_whitelist_goes_RED(tmp_path, monkeypatch) -> None:
    """The historical defect, re-introduced exactly: the production forward's output carries
    neither per-stage key. With the fix the trainer REFUSES (SystemExit); without the refusal the
    row / gradient assertions fail. Either way the guard above must not pass."""
    orig = refc.RefCModel.forward

    def as_shipped(self, *a, **k):
        o = orig(self, *a, **k)
        o.pop("layer_u0_hat", None)
        o.pop("layer_logits", None)
        return o
    monkeypatch.setattr(refc.RefCModel, "forward", as_shipped)
    with pytest.raises((SystemExit, AssertionError)):
        _assert_f3_live(tmp_path)
