"""refcv8's forward on a REAL checkpoint, fed through the bridge (TANITAD VENV; CPU by default).

    R8_TEST_CKPT=E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt R8_TEST_DEVICE=cpu pytest -q tests/test_model_seam8.py

Skipped when the checkpoint is absent.  The frames are a SYNTHETIC noise stack (these controls test determinism and feed wiring,
not driving): ``run_navsim_refcv8.py`` re-runs ``-k "K0 or KD"`` on CUDA before any CUDA split.
  K0    same seed -> bit-identical plan (the seed replicate and common random numbers rest on it)
  KNAV  r8_nav = None and r8_nav = zeros[1,6] give the SAME plan (the model turns None into zeros, refc_v3.py:1968-1973)
  KR8   a KNOWN r8_nav row changes the plan: the input REACHES the model (so feeding one to the LEGAL row is a real defect)
  KTOK  the nav token reaches the plan (TURN_L vs FOLLOW differ)
  KD    the eval-only exact-dedup path equals the native path (selection identical, max |d| <= 1 mm); LAST: it patches the encoder
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
CKPT = os.environ.get("R8_TEST_CKPT", "E:/refcv8_eval_kit/ckpt/ckpt_ema_40000.pt")
CONFIG = os.environ.get("R8_TEST_CONFIG", "E:/refcv8_eval_kit/ckpt/config.json")
DEVICE = os.environ.get("R8_TEST_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(CONFIG)), reason="no refcv8 checkpoint on this box")
if DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import refcv8_bridge as B  # noqa: E402
import torch  # noqa: E402


def _level_rig(cam_z_ego=1.5, cam_x=1.7):
    fwd = np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    down = np.cross(fwd, right)
    return {"rig_key": "synthetic", "f0_sensor2lidar_translation": [cam_x, 0.0, cam_z_ego],
            "virtual_R_cam_to_lidar": np.stack([right, down, fwd], axis=1).tolist()}


@pytest.fixture(scope="module")
def world():
    model, cfg, args, rec, meta = B.load_refcv8(CKPT, CONFIG, DEVICE, "auto")
    g, v, _ = B.hires_lift_for_rig(_level_rig(), -0.35, model._lift_bank_hires)
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 256, size=(4, 416, 1024, 3), dtype=np.uint8)
    times = [-1.5, -1.0, -0.5, 0.0]
    rows = B.pack_rows(fr, B.slot_sources8(times, "ST"), None).numpy()
    st = [{"ego_pose": [10.0 * t, 0.0, 0.0], "ego_velocity": [10.0, 0.0], "ego_acceleration": [0.0, 0.0],
           "driving_command": [0, 1, 0, 0]} for t in times]
    decl = B.declare8(st, "R8_LEGAL")
    return {"model": model, "meta": meta, "rec": rec, "g": g, "v": v, "rows": rows, "decl": decl,
            "hist": B.ego_history_poses(decl, times)}


def _go(w, token=0, r8_nav=None, seed=0):
    return B.run_model8(w["model"], w["rows"], w["decl"], {"nav_index": token}, B.speed_input8(None, "R8_LEGAL"), w["hist"],
                        w["g"], w["v"], int(w["rec"]["decoder_steps"]), seed, DEVICE, r8_nav=r8_nav)


def test_premises_all_hold(world):
    bad = [k for k, v in world["meta"]["seam_checks"].items() if not v]
    assert not bad, bad
    assert world["meta"]["r8_legal_row_stamped"] and world["model"]._r8_legal_row is True


def test_K0_same_seed_is_bit_identical(world):
    a, b = _go(world, 0, seed=7), _go(world, 0, seed=7)
    assert np.array_equal(a["traj"], b["traj"]) and a["diag"]["sel_idx"] == b["diag"]["sel_idx"]
    assert np.isfinite(a["traj"]).all() and a["traj"].shape == (8, 2)


def test_KNAV_none_equals_the_zero_row(world):
    """The explicit zero row (the trainer's path) and the model-level ``None`` give the SAME plan."""
    a = _go(world, 0, r8_nav=B.r8_nav_row(), seed=3)
    L = B.boot8.loader()
    fr = L.trainer().frames_to_device(torch.from_numpy(world["rows"])[None], DEVICE)
    kw = B.forward_kwargs8(fr, {"nav_index": 0}, 10.0, B.speed_input8(None, "R8_LEGAL"), world["g"], world["v"], 2, DEVICE)
    assert torch.equal(kw["r8_nav"], torch.zeros(1, 6))
    kw["r8_nav"] = None
    world["model"].core.set_ego_window(torch.from_numpy(world["hist"])[None].to(DEVICE), 8, actions=None)
    torch.manual_seed(3)
    with torch.no_grad():
        out = world["model"](fr, **kw)
    assert np.array_equal(out["traj"].float()[0].cpu().numpy().astype(np.float64), a["traj"])


def test_KR8_a_known_nav_row_reaches_the_model(world):
    from tanitad.train import refcv8_train as tr
    known = tr.scale_nav(torch.tensor([[40.0, 120.0, 90.0, 1.0, 150.0]]), torch.ones(1, dtype=torch.bool))
    a, b = _go(world, 1, seed=3), _go(world, 1, r8_nav=known, seed=3)
    assert float(np.abs(a["traj"] - b["traj"]).max()) > 1e-6


def test_KTOK_the_nav_token_reaches_the_plan(world):
    a, b = _go(world, 0, seed=3), _go(world, 1, seed=3)
    assert float(np.abs(a["traj"] - b["traj"]).max()) > 1e-6


def test_KD_exact_dedup_matches_the_native_path(world):
    a = _go(world, 0, seed=5)
    B.exact_dedup(world["model"].core.encoder)
    b = _go(world, 0, seed=5)
    assert a["diag"]["sel_idx"] == b["diag"]["sel_idx"]
    assert float(np.abs(a["traj"] - b["traj"]).max()) <= 1e-3, "exact dedup moved the plan by more than the registered 1 mm"
