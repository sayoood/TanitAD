"""Control-flow test of `g0_refcv7.main()` with the A6 additions, on CPU with a FAKE trainer / loader.

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_g0_main_a6_flow.py

Why it exists: the step-50,400 G0 is launched unattended by `waiter_final.sh`; a crash in the new code
paths (seed-0 capture, the fp32_s0 arm, the A6 verdict, the RAM/disk batch lever) would stop the final
battery after its GPU work. This drives the REAL `main()` end to end -- argument parsing, batching,
24 seeds, the A6 arm, the verdicts, the JSON -- with the model, dataset and loss replaced by small fakes
that still call the REAL `refcv6_tactical.tactical_behaviour_losses` through the trainer-module seam the
capture hooks. Assertions are on the written JSON (the artifact), never on a return code.
"""
import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class _FakeTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.memory_levers = {"bf16": True, "channels_last": True, "chunk_ckpt": 8}
        self.cfg = types.SimpleNamespace(equalize_bottom_rows=43)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _FakeTrunk()
        self.head = torch.nn.Linear(3, 22)


def _fake_trainer(model_holder):
    from tanitad.refs import refcv6_tactical as v6

    tr = types.SimpleNamespace()
    tr.LAW_AHEAD = 2
    tr.V3Dataset = type("V3Dataset", (), {})
    tr.v6tac = v6
    tr._det_metrics = types.SimpleNamespace(HEADS=())

    def compute_losses_v3(model, eb, device, mode=None, ablate_frames=False):
        x = eb["x"].float()                                  # [B, 3]
        trunk = model.trunk
        # a NUMERICS lever: the fp32 arm (bf16 lever off) moves the logits by a hair, like the real one
        eps = 0.0 if trunk.memory_levers.get("bf16") else 3e-3
        logits = model.head(x) + eps
        conf = model.head(x * 0.5)
        B = x.shape[0]
        y = (torch.arange(22).repeat(B, 1) % 3 == 0).float()
        w = torch.ones(B, 22)
        _t, tele = tr.v6tac.tactical_behaviour_losses(
            {"goal_logits": logits, "goal_conf": conf, "lat_logits": torch.zeros(B, 8),
             "lon_logits": torch.zeros(B, 8)},
            goal_y=y, goal_w=w, lat_target=torch.full((B,), -100), lon_target=torch.full((B,), -100),
            goal_pos_weight=None, goal_class_mask=torch.ones(22), ignore_index=-100)
        return {"tacv6_goal_conf_bce": tele["tac_goal_conf_bce"], "tacv6_goal_bce": tele["tac_goal_bce"],
                "lat": (x.sum() * 0.01 + eps * 0.1).abs() + 0.5, "traj": torch.tensor(0.5),
                "tacv6_n_supervised_goal_cells": float(B * 22)}
    tr.compute_losses_v3 = compute_losses_v3

    def _eval_row_from_acc(acc, nb, model):
        return {f"eval_{k}": round(v / nb, 5) for k, v in acc.items()}
    tr._eval_row_from_acc = _eval_row_from_acc
    tr.frames_to_device = lambda x, device: x
    return tr


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import g0_refcv7 as G
    from tanitad.models import timm_trunk
    monkeypatch.setattr(timm_trunk, "TimmResNetTrunk", _FakeTrunk)
    model = _FakeModel()
    tr = _fake_trainer(model)
    monkeypatch.setattr(G.L, "trainer", lambda: tr)
    monkeypatch.setattr(G.L, "load_config", lambda p: {})
    mrec = {"state_dict": {"missing": [], "unexpected": [], "step": 100},
            "param_breakdown": {"equal": True}, "build_s": 0.1, "anchor_file_vs_ckpt_buffers": {},
            "declared_vs_built": {"mismatches": []}}
    args = argparse.Namespace(batch=4, eval_batches=2, mode="diffusion", ablate_frames=False)
    monkeypatch.setattr(G.L, "build_model", lambda c, ck, dev: (model, None, args, mrec))
    g = torch.Generator().manual_seed(3)
    items = [{"x": torch.randn(3, generator=g), "future_frames": torch.zeros(2, 4, dtype=torch.uint8)}
             for _ in range(12)]
    monkeypatch.setattr(G.L, "build_eval_dataset",
                        lambda *a, **k: (items, None, {"n_episodes": 3, "n_windows": len(items)}))
    for name, fn in (("get_device_name", lambda i=0: "FAKE"), ("reset_peak_memory_stats", lambda: None),
                     ("max_memory_allocated", lambda: 0), ("manual_seed_all", lambda s: None)):
        monkeypatch.setattr(torch.cuda, name, fn)
    ck = tmp_path / "ckpt.pt"
    ck.write_bytes(b"x")
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    # the in-run row: what the fake replay gives under the bf16 lever, +0.4 % on the conf term
    G.patch_frames_to_device(tr)
    perm = G.L.inrun_eval_perm(items, 2, 4)
    acc = {}
    for b in range(2):
        eb = G.collate(items, perm[b * 4:(b + 1) * 4], 1)
        with torch.no_grad():
            el = tr.compute_losses_v3(model, eb, "cpu")
        for k, v in el.items():
            acc[k] = acc.get(k, 0.0) + float(v)
    inrun = {f"eval_{k}": round(v / 2, 5) for k, v in acc.items()}
    inrun["eval_tacv6_goal_conf_bce"] = round(inrun["eval_tacv6_goal_conf_bce"] * 1.004, 5)
    met = tmp_path / "metrics.jsonl"
    met.write_text(json.dumps({"step": 100, "eval_loss": 1.0, **inrun}) + "\n", encoding="utf-8")
    monkeypatch.setattr(G, "A6_REGISTRATION", tmp_path / "SPEC_SHA256_AMENDMENT_A6.txt")
    return G, tmp_path, ck, cfg, met


def _run(G, tmp_path, ck, cfg, met, out, extra=()):
    argv = ["g0_refcv7.py", "--ckpt", str(ck), "--config", str(cfg), "--metrics", str(met),
            "--seeds", ",".join(str(i) for i in range(24)), "--mutations", "", "--diagnostic-arms", "",
            "--skip-wrapper-control", "--micro", "1,3", "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        G.main()
    finally:
        sys.argv = old
    return json.load(open(out, encoding="utf-8"))


def test_main_runs_the_a6_arm_and_reports_a6_as_DRAFT(harness, monkeypatch):
    G, tmp_path, ck, cfg, met = harness
    monkeypatch.delenv("REFCV7_G0_BATCH_CACHE", raising=False)
    rec = _run(G, tmp_path, ck, cfg, met, tmp_path / "g0.json")
    assert len(rec["a6"]["cells"]["s0"]) == 2 and len(rec["a6"]["cells"]["fp32_s0"]) == 2
    assert rec["a6"]["capture_errors_s0"] == [] and rec["a6"]["capture_errors"] == []
    assert rec["a6"]["fp32_s0"]["row"]["eval_lat"] != rec["by_seed"]["0"]["row"]["eval_lat"]
    v6 = rec["verdict_A6"]
    assert v6["amendment"] == "A6" and v6["registration"]["registered"] is False
    t = v6["terms"]["eval_tacv6_goal_conf_bce"]
    assert t["cls"] == "THRESHOLD_TARGET" and t.get("a6_lo") is not None
    # A6 is a DRAFT here: the gate is A5, unchanged
    assert rec["verdict"]["amendment"] == "A5" and rec["verdict_A5"]["amendment"] == "A5"
    # the disk batch cache was used and removed afterwards
    assert rec["batch_cache"] != "RAM" and not Path(rec["batch_cache"]).exists()


def test_main_gates_on_A6_when_registered_before_start_and_ram_lever(harness, monkeypatch):
    G, tmp_path, ck, cfg, met = harness
    reg = tmp_path / "SPEC_SHA256_AMENDMENT_A6.txt"
    reg.write_text("sha256 ...", encoding="utf-8")
    past = time.time() - 3600
    os.utime(reg, (past, past))
    monkeypatch.setenv("REFCV7_G0_BATCH_CACHE", "ram")
    rec = _run(G, tmp_path, ck, cfg, met, tmp_path / "g0b.json")
    assert rec["batch_cache"] == "RAM"
    assert rec["verdict"]["amendment"] == "A6" and rec["verdict"]["registration"]["registered"] is True
    assert rec["verdict_A5"]["amendment"] == "A5"


def test_main_no_a6_is_the_previous_flow(harness, monkeypatch):
    G, tmp_path, ck, cfg, met = harness
    monkeypatch.delenv("REFCV7_G0_BATCH_CACHE", raising=False)
    rec = _run(G, tmp_path, ck, cfg, met, tmp_path / "g0c.json", extra=("--no-a6",))
    assert "verdict_A6" not in rec and rec["verdict"]["amendment"] == "A5"
    assert rec["a6"] == {"status": "skipped (--no-a6)"}
