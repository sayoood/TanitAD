"""KL — this package's refcv6 loader vs the BATTERY stream's trainer-exact loader, on the NavSim
forward (integration; needs the kit checkpoint AND the battery stream's `refcv6_loader.py`).

The battery stream (`C:/Users/Admin/ev6_battery/code/refcv6_loader.py`) replays `train()`'s model
construction line for line (lift bank WITH `equalize_bottom_rows`, the model-level attributes the
trainer sets, strict load) and reproduces the run's own in-run eval. This package loads through
`taniteval/tools/refcv3_arm.load_model` and supplies the lift geometry itself. The claim tested: on a
NavSim scene, with the same inputs, lift geometry, precision and inference seed, the two models emit
the SAME plan — i.e. nothing `train()` sets on the model and `refcv3_arm` omits reaches the forward
this package runs. Recorded to ``raw/controls/KL_loader_crosscheck.json``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import refcv6_bridge as R6  # noqa: E402
import rig6  # noqa: E402
import torch  # noqa: E402

BATTERY = os.environ.get("R6_BATTERY_LOADER", "C:/Users/Admin/ev6_battery/code/refcv6_loader.py")
CKPT = os.environ.get("R6_TEST_CKPT", "D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt")
CONFIG = os.environ.get("R6_TEST_CONFIG", "D:/refcv6_eval_kit/ckpt/config.json")
BANK = "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage"
INPUTS = os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge", "raw",
                      "navsim_agent_inputs.json")
SPEED = os.path.join(PKG, "raw", "inputs", "speed_limits_warmup_two_stage.json")
ROAD = os.path.join(PKG, "raw", "inputs", "road_plane_navhard_warmup_logs.json")


@pytest.mark.skipif(not (os.path.isfile(BATTERY) and os.path.isfile(CKPT)),
                    reason="NO_TREE: the battery loader or the kit checkpoint is absent")
def test_KL_same_plan_as_the_trainer_exact_loader():
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location("battery_refcv6_loader", BATTERY)
    bl = importlib.util.module_from_spec(spec)
    sys.modules["battery_refcv6_loader"] = bl
    spec.loader.exec_module(bl)
    cfgj = bl.load_config(CONFIG)
    mb, _cb, _ab, rec = bl.build_model(cfgj, CKPT, device="cpu")
    mm, _cfg, _targs, prov, ra, meta = R6.load_refcv6(CKPT, CONFIG, "cpu", "fp32")
    for m in (mb, mm):                       # the same arithmetic on both: fp32 backbone
        m.core.encoder.memory_levers["bf16"] = False
        m.core.encoder.memory_levers["channels_last"] = False
        R6.exact_dedup(m.core.encoder)
        m.eval()
    doc = json.load(open(INPUTS, encoding="utf-8"))
    speed = json.load(open(SPEED, encoding="utf-8"))
    road = json.load(open(ROAD, encoding="utf-8"))["summary"]["median"]
    bank = R6.Bank416(BANK)
    toks = sorted(t for t, r in doc["tokens"].items() if r["stage"] == 2)[:3]
    out = {"n": len(toks), "battery_record_departures": rec.get("departures"), "rows": []}
    for tok in toks:
        r = doc["tokens"][tok]
        times = [(int(t) - int(r["timestamps_us"][-1])) / 1e6 for t in r["timestamps_us"]]
        decl = R6.declare6(r["ego_statuses"], "R6_A1")
        nav = R6.nav_input(decl, "R6_A1")
        vmax = R6.max_speed_input(speed["tokens"].get(tok), "R6_A1")
        hist = R6.ego_history_poses(decl, times)
        fr, rk, _ = bank.load(r["scene_token"])
        rows = R6.pack_rows(fr, R6.slot_sources6(times, "ST")).numpy()
        g, v, _ = rig6.lift_geometry(bank.rigs[rk], road, stride=16)
        seed = R6.scene_seed(0, tok)
        a = R6.run_model6(mb, ra, rows, decl, "R6_A1", nav, vmax, hist, g, v, 2, seed, "cpu")
        b = R6.run_model6(mm, ra, rows, decl, "R6_A1", nav, vmax, hist, g, v, 2, seed, "cpu")
        out["rows"].append({"token": tok, "max_abs_traj_diff_m": float(np.abs(a["traj"] - b["traj"]).max()),
                            "sel_battery": a["diag"]["sel_idx"], "sel_mine": b["diag"]["sel_idx"]})
    os.makedirs(os.path.join(PKG, "raw", "controls"), exist_ok=True)
    json.dump(out, open(os.path.join(PKG, "raw", "controls", "KL_loader_crosscheck.json"), "w"),
              indent=1, default=str)
    assert all(x["sel_battery"] == x["sel_mine"] for x in out["rows"]), out
    assert max(x["max_abs_traj_diff_m"] for x in out["rows"]) < 1e-5, out
