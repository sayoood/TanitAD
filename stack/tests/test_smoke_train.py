"""End-to-end smoke: tiny world-model training must run and emit instrument
rows. Marked slow-ish but still < ~2 min on CPU."""

import json
from pathlib import Path

from tanitad.config import smoke_config
from tanitad.train.train_worldmodel import train


def test_smoke_training(tmp_path: Path):
    cfg = smoke_config()
    cfg.train.steps = 5
    cfg.train.out_dir = str(tmp_path / "run")
    metrics = train(cfg, n_episodes=4)
    assert "instruments" in metrics
    inst = metrics["instruments"]
    assert inst["I2_batch_consistency_pass"], "I2 failed in smoke train"
    assert inst["I3_split"] == "episode-level"
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "config.json").exists()
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["n_params"] > 0
