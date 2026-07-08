"""End-to-end smoke of the checkpoint evaluator: a real (untrained) WorldModel
+ toy episodes must produce a well-formed instruments-first gate report."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_checkpoint import build_eval_tensors, evaluate  # noqa: E402

from tanitad.config import smoke_config
from tanitad.data.toy_driving import generate_episode
from tanitad.models.fourbrain import WorldModel


def test_evaluator_end_to_end_wellformed():
    torch.manual_seed(0)
    world = WorldModel(smoke_config()).eval()
    eps = [generate_episode(i, steps=60, size=64) for i in range(4)]
    report = evaluate(world, eps, "cpu", exp_id="smoke-eval", git_hash="test")
    assert set(report["summary"]) == {"D1", "D2", "D3"}
    for g in report["gates"]:
        assert g["instruments"][0]["row"] == "I1"        # instruments FIRST
        assert g["gate"] in {"D1", "D2", "D3"}
        # untrained model: statuses must be honest, never a spurious PASS of D2/D3
        assert (g["passed"] is False) or g["gate"] == "D1" or g["admissible"]
    assert report["d3_horizon_s"] > 0
    assert "spectral" in report and report["n_eval_windows"] > 0
    # I2 must genuinely pass on the batch-free-norm encoder
    d1_i2 = [r for r in report["gates"][0]["instruments"] if r["row"] == "I2"][0]
    assert d1_i2["pass"] is True


def test_build_eval_tensors_shapes():
    world = WorldModel(smoke_config()).eval()
    eps = [generate_episode(9, steps=60, size=64)]
    t = build_eval_tensors(world, eps, "cpu",
                           window=world.predictor.cfg.window,
                           k_max=max(world.predictor.cfg.horizons))
    n = t["states"].shape[0]
    assert n > 0
    assert t["waypoints"].shape == (n, 4, 2)
    assert t["disp1"].shape == (n, 2)
    assert t["actions"].shape == (n, 2)
    assert t["prev_state"].shape == (n, 2)
    assert len(t["eps"]) == n
