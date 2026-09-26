"""refcv6 model-side controls on REAL warmup scenes (integration; needs the kit checkpoint).
TANITAD VENV, CPU:  CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_model_seam6.py

* K0  determinism: the same scene + the same per-scene seed -> BIT-identical plan;
* KD  the eval-only exact-duplicate dedup is the SAME function as the trunk's native path on a
      static-history window (MEASURED difference asserted small, selections identical);
* KI  every declared input REACHES the model: frames, nav, v0 and the ego history each move the
      plan when mutated (a mutation that moves nothing would mean the channel is dead-wired) —
      and the max-speed one-hot is shown to reach the tactical decoder's logits;
* KT  bank tamper: one flipped byte in a bank file is REFUSED by the sha check.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import refcv6_bridge as R6  # noqa: E402
import rig6  # noqa: E402
import torch  # noqa: E402

CKPT = os.environ.get("R6_TEST_CKPT", "D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt")
CONFIG = os.environ.get("R6_TEST_CONFIG", "D:/refcv6_eval_kit/ckpt/config.json")
BANK = os.environ.get("R6_TEST_BANK", "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage")
INPUTS = os.path.join(PKG, "..", "..", "2026-09-19-navsim-refcv4b-bridge", "raw",
                      "navsim_agent_inputs.json")
SPEED = os.path.join(PKG, "raw", "inputs", "speed_limits_warmup_two_stage.json")
ROAD = os.path.join(PKG, "raw", "inputs", "road_plane_navhard_warmup_logs.json")
N_SCENES = 4
#: the device the controls run on (the runner re-runs K0/KD on CUDA before a GPU milestone)
DEV = os.environ.get("R6_TEST_DEVICE", "cpu")

need = pytest.mark.skipif(not (os.path.isfile(CKPT) and os.path.isdir(BANK) and os.path.isfile(INPUTS)),
                          reason="NO_TREE: kit checkpoint / 416 warmup bank / E2 export absent")


@pytest.fixture(scope="module")
def rig():
    torch.set_num_threads(8)
    model, cfg, targs, prov, ra, meta = R6.load_refcv6(CKPT, CONFIG, DEV,
                                                       "fp32" if DEV == "cpu" else "as_trained")
    doc = json.load(open(INPUTS, encoding="utf-8"))
    speed = json.load(open(SPEED, encoding="utf-8"))
    road = json.load(open(ROAD, encoding="utf-8"))["summary"]["median"]
    bank = R6.Bank416(BANK)
    toks = sorted(t for t, r in doc["tokens"].items() if r["stage"] == 2)[:N_SCENES]
    return {"model": model, "prov": prov, "ra": ra, "doc": doc, "speed": speed, "road": road,
            "bank": bank, "toks": toks}


def _scene(rg, tok, arm="R6_A1", mutate=None):
    r = rg["doc"]["tokens"][tok]
    times = [(int(t) - int(r["timestamps_us"][-1])) / 1e6 for t in r["timestamps_us"]]
    st = json.loads(json.dumps(r["ego_statuses"]))
    decl = R6.declare6(st, arm)
    nav = R6.nav_input(decl, arm)
    vmax = R6.max_speed_input(rg["speed"]["tokens"].get(tok), arm)
    hist = R6.ego_history_poses(decl, times)
    fr, rk, _ = rg["bank"].load(r["scene_token"])
    grey = None
    if mutate == "frames":
        grey = int(round(rg["bank"].mean_px))
    rows = R6.pack_rows(fr, R6.slot_sources6(times, "ST"), grey).numpy()
    g, v, _ = rig6.lift_geometry(rg["bank"].rigs[rk], rg["road"], stride=16)
    if mutate == "hist":
        hist = hist.copy()
        hist[:, 3] = hist[-1, 3]                       # a flat speed history, same v0
        hist[:, 2] = 0.0
        hist[:5, 3] += 3.0                             # ... that was 3 m/s faster 0.3 s ago
    if mutate == "nav":
        nav = {"nav_index": 1 if nav["nav_index"] != 1 else 2, "name": "x", "navsim_argmax": -1}
    if mutate == "v0":
        decl = dict(decl)
        decl["ego_velocity[3]"] = [decl["ego_velocity[3]"][0] + 4.0, decl["ego_velocity[3]"][1]]
    seed = R6.scene_seed(0, tok)
    return R6.run_model6(rg["model"], rg["ra"], rows, decl, arm, nav, vmax, hist, g, v,
                         int(rg["prov"]["decoder_steps"]), seed, DEV)


@need
def test_K0_same_seed_is_bit_identical(rig):
    tok = rig["toks"][0]
    a, b = _scene(rig, tok), _scene(rig, tok)
    assert np.array_equal(a["traj"], b["traj"]) and a["diag"]["sel_idx"] == b["diag"]["sel_idx"]


@need
def test_KD_exact_dedup_equals_native(rig):
    enc = rig["model"].core.encoder
    native = [_scene(rig, t) for t in rig["toks"]]
    assert all(r["diag"]["dedup"] == [24, 10] for r in native)       # the trunk's own path
    saved = enc._backbone_dedup
    R6.exact_dedup(enc)
    try:
        exact = [_scene(rig, t) for t in rig["toks"]]
    finally:
        enc._backbone_dedup = saved
        enc.memory_levers.pop("eval_exact_dedup", None)
    assert all(r["diag"]["dedup"] == [24, 1] for r in exact)
    d = max(float(np.abs(x["traj"] - y["traj"]).max()) for x, y in zip(native, exact))
    sel_same = sum(x["diag"]["sel_idx"] == y["diag"]["sel_idx"] for x, y in zip(native, exact))
    out = {"n": len(native), "max_abs_traj_diff_m": d, "sel_identical": sel_same, "device": DEV,
           "ckpt": CKPT}
    os.makedirs(os.path.join(PKG, "raw", "controls"), exist_ok=True)
    json.dump(out, open(os.path.join(PKG, "raw", "controls", f"KD_exact_dedup_{DEV}.json"), "w"),
              indent=1)
    assert sel_same == len(native)
    assert d < 1e-3, out


@need
@pytest.mark.parametrize("what", ["frames", "nav", "v0", "hist"])
def test_KI_every_declared_input_reaches_the_plan(rig, what):
    moved = 0
    for t in rig["toks"]:
        a, b = _scene(rig, t), _scene(rig, t, mutate=what)
        moved += int(float(np.abs(a["traj"] - b["traj"]).max()) > 1e-4)
    assert moved >= 1, f"mutating {what} moved no plan on {len(rig['toks'])} scenes"


@need
def test_KI_max_speed_reaches_the_tactical_decoder(rig):
    """The one-hot must change the v6 tactical decoder's input (its logits); whether it also
    moves the PLAN is a property of the checkpoint, measured by the R6_VMAXOFF arm, not assumed."""
    m = rig["model"]
    tok = [t for t in rig["toks"] if rig["speed"]["tokens"][t]["status"] == "limit"]
    if not tok:
        pytest.skip("no warmup scene in the first N with a map limit")
    tok = tok[0]
    r = rig["doc"]["tokens"][tok]
    times = [(int(x) - int(r["timestamps_us"][-1])) / 1e6 for x in r["timestamps_us"]]
    decl = R6.declare6(r["ego_statuses"], "R6_A1")
    fr, rk, _ = rig["bank"].load(r["scene_token"])
    rows = R6.pack_rows(fr, R6.slot_sources6(times, "ST")).numpy()
    g, v, _ = rig6.lift_geometry(rig["bank"].rigs[rk], rig["road"], stride=16)
    frt = rig["ra"].trainer().frames_to_device(torch.from_numpy(rows)[None], DEV)
    outs = []
    for vm, ok in ((11.17568, 1.0), (0.0, 0.0)):
        m.core.set_ego_window(torch.from_numpy(R6.ego_history_poses(decl, times))[None].to(DEV), 8)
        torch.manual_seed(5)
        with torch.no_grad():
            outs.append(m(frt, nav_cmd=torch.tensor([0], device=DEV),
                          v0=torch.tensor([R6.v0_of(decl)], device=DEV), steps=2,
                          v_max_ms=torch.tensor([vm], device=DEV),
                          v_max_valid=torch.tensor([ok], device=DEV),
                          perception_grid=g[None].to(DEV), perception_valid=v[None].to(DEV)))
    dl = float((outs[0]["tacv6_lon_logits"] - outs[1]["tacv6_lon_logits"]).abs().max())
    assert dl > 1e-4


def test_KT_bank_tamper_is_refused(tmp_path):
    if not os.path.isdir(BANK):
        pytest.skip("NO_TREE: warmup 416 bank absent")
    import pandas as pd
    p = pd.read_parquet(os.path.join(BANK, "frames_provenance.parquet"))
    key = p.scene_token.iloc[0]
    (tmp_path / "frames").mkdir()
    for f in ("frames_provenance.parquet", "BUILD_REPORT.json"):
        shutil.copy(os.path.join(BANK, f), tmp_path / f)
    a = np.load(os.path.join(BANK, "frames", f"{key}.npy"))
    np.save(tmp_path / "frames" / f"{key}.npy", a)
    b = R6.Bank416(str(tmp_path))
    b.load(key)                                          # intact: loads
    a2 = a.copy()
    a2[0, 200, 500, 1] ^= 1                               # one flipped bit
    np.save(tmp_path / "frames" / f"{key}.npy", a2)
    with pytest.raises(R6.RefusedInput):
        b.load(key)


#: the REAL navtest 416 bank (``code/build_navtest416.py``); any bank with >= 1 DONE shard serves.
NAVTEST_BANK = os.environ.get("R6_TEST_NAVTEST_BANK",
                              "D:/Archive/devbox-C/navsim/exp/refcv6_navtest416/frame_bank")


def test_KT_navtest_bank_sha_and_tamper(tmp_path):
    """The W3-layout 416 bank: every token's stacked [4,416,1024,3] re-hashes to its index sha;
    one flipped bit is REFUSED. The tamper arm runs on a ONE-TOKEN copy (its 4 rows, ~5 MB) so the
    test never copies a multi-GB shard file."""
    if not os.path.isfile(os.path.join(NAVTEST_BANK, "index.json")):
        pytest.skip("NO_TREE: navtest 416 bank absent")
    b = R6.BankNavtest416(NAVTEST_BANK)
    toks = sorted(b.prov)
    assert len(toks) >= 1
    for t in toks[:5]:
        a, rk, sha = b.load(t)
        assert a.shape == (4, 416, 1024, 3) and sha == b.prov[t]["sha256"]
    t0 = toks[0]
    a, _rk, _sha = b.load(t0)
    idx = dict(b.index)
    idx["tokens"] = {t0: dict(b.prov[t0], file="mini.npy", rows=[0, 1, 2, 3])}
    with open(tmp_path / "index.json", "w", encoding="utf-8") as fh:
        json.dump(idx, fh)
    np.save(tmp_path / "mini.npy", np.ascontiguousarray(a))
    assert R6.BankNavtest416(str(tmp_path)).load(t0)[2] == b.prov[t0]["sha256"]   # intact copy loads
    a2 = np.array(a)
    a2[-1, 100, 100, 0] ^= 1                                     # one flipped bit in the t0 frame
    np.save(tmp_path / "mini.npy", a2)
    with pytest.raises(R6.RefusedInput):
        R6.BankNavtest416(str(tmp_path)).load(t0)
