"""refcv7 model-side controls on REAL scenes (integration; needs a refcv7 checkpoint).
TANITAD VENV, CPU:  CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_model_seam7.py

* KT7 THE FEED IS THE TRAINER'S: one real PhysicalAI eval window through the trainer's OWN
      ``compute_losses_v3`` (its ``model(...)`` call captured) and the same window through this
      bridge's ``run_model7`` -> the SAME keywords, the SAME tensors, a BIT-identical plan;
* K0  determinism (same scene + seed -> bit-identical plan);
* KD  exact-duplicate dedup == the trunk's native path;
* KI  frames, nav, v0 and the ego window each move >= 1 of 4 plans; max speed reaches the logits;
* KPR the model's emitted prior == the bridge's model-free prior (<= 1e-4 m);
* KF  the emitted pick is replicated; the filter never reaches it (SPEC A1); CEILDECL obeys it;
* RED the model REFUSES a window that does not end at v0, and the refcv6 stride-16 geometry.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "code"))
import refcv7_bridge as R7  # noqa: E402
import torch  # noqa: E402

CKPT = os.environ.get("R7_TEST_CKPT", "D:/refcv7_eval_kit/ckpt/ckpt_1500.pt")
CONFIG = os.environ.get("R7_TEST_CONFIG", "D:/refcv7_eval_kit/ckpt/config.json")
BANK = os.environ.get("R7_TEST_BANK",
                      "C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923/warmup_two_stage")
INC = os.path.join(R7.boot7.TREE, "FlyWheels", "TanitAD_EvalFlyWheel", "incoming")
INPUTS = os.path.join(INC, "2026-09-19-navsim-refcv4b-bridge", "raw", "navsim_agent_inputs.json")
SPEED = os.path.join(R7.boot7.R6PKG, "raw", "inputs", "speed_limits_warmup_two_stage.json")
ROAD = os.path.join(R7.boot7.R6PKG, "raw", "inputs", "road_plane_navhard_warmup_logs.json")
N_SCENES = 4
DEV = os.environ.get("R7_TEST_DEVICE", "cpu")

need = pytest.mark.skipif(not (os.path.isfile(CKPT) and os.path.isdir(BANK)
                               and os.path.isfile(INPUTS)),
                          reason="NO_TREE: refcv7 checkpoint / 416 warmup bank / E2 export absent")


@pytest.fixture(scope="module")
def rig():
    torch.set_num_threads(8)
    model, cfg, args, rec, meta = R7.load_refcv7(CKPT, CONFIG, DEV,
                                                 "fp32" if DEV == "cpu" else "as_trained")
    doc = json.load(open(INPUTS, encoding="utf-8"))
    speed = json.load(open(SPEED, encoding="utf-8"))
    road = json.load(open(ROAD, encoding="utf-8"))["summary"]["median"]
    bank = R7.Bank416(BANK)
    s2 = sorted(t for t, r in doc["tokens"].items() if r["stage"] == 2)
    lim = [t for t in s2 if (speed["tokens"].get(t) or {}).get("status") == "limit"]
    toks = (lim[:2] + [t for t in s2 if t not in lim][:2])[:N_SCENES]
    return {"model": model, "cfg": cfg, "args": args, "rec": rec, "doc": doc, "speed": speed,
            "road": road, "bank": bank, "toks": toks, "steps": int(rec["decoder_steps"])}


def _inputs(rg, tok, arm="R7_A1", mutate=None):
    r = rg["doc"]["tokens"][tok]
    times = [(int(t) - int(r["timestamps_us"][-1])) / 1e6 for t in r["timestamps_us"]]
    decl = R7.declare7(json.loads(json.dumps(r["ego_statuses"])), arm)
    nav = R7.nav_input(decl, arm)
    vmax = R7.max_speed_input(rg["speed"]["tokens"].get(tok), arm)
    hist = R7.ego_history_poses(decl, times)
    fr, rk, _ = rg["bank"].load(r["scene_token"])
    grey = int(round(rg["bank"].mean_px)) if mutate == "frames" else None
    rows = R7.pack_rows(fr, R7.slot_sources7(times, "ST"), grey).numpy()
    g, v, _ = R7.hires_lift_for_rig(rg["bank"].rigs[rk], rg["road"], rg["model"]._lift_bank_hires)
    if mutate == "hist":
        hist = hist.copy()
        hist[:, 2] = np.linspace(0.3, 0.0, 8)               # a turning past, same t0 state
        hist[:6, 3] += 2.0                                  # ... and decelerating into t0
    if mutate == "nav":
        nav = {"nav_index": 1 if nav["nav_index"] != 1 else 2, "name": "x", "navsim_argmax": -1}
    if mutate == "v0":
        decl = dict(decl)
        decl["ego_velocity[3]"] = [decl["ego_velocity[3]"][0] + 4.0, decl["ego_velocity[3]"][1]]
        hist = hist.copy()
        hist[-1, 3] = R7.v0_of(decl)                        # the window must END at v0
    return rows, decl, nav, vmax, hist, g, v


def _run(rg, tok, arm="R7_A1", mutate=None, filter_on=True, seed_base=0):
    rows, decl, nav, vmax, hist, g, v = _inputs(rg, tok, arm, mutate)
    return R7.run_model7(rg["model"], rows, decl, nav, vmax, hist, g, v, rg["steps"],
                         R7.scene_seed(seed_base, tok), DEV, filter_on=filter_on)


# --------------------------------------------------------------------------- #
# KT7 — the feed is the trainer's (compute_losses_v3), on a REAL eval window      #
# --------------------------------------------------------------------------- #
class _Captured(Exception):
    pass


@need
def test_KT7_bridge_feed_equals_compute_losses_v3(rig):
    L = R7.boot7.loader()
    tr = L.trainer()
    model = rig["model"]
    cfgd = L.load_config(CONFIG)
    e_ds, e_eps, drec = L.build_eval_dataset(model, rig["cfg"], rig["args"], cfgd,
                                             with_perception_targets=False)
    # THREE windows: the first, one mid-split, and the first TURN-commanded window moving > 3 m/s
    # found by scanning each episode's first window (a follow-only check could miss a nav wiring)
    cand = [0, len(e_ds) // 2]
    starts = sorted({min(i for i, (e, _t) in enumerate(e_ds.index) if e == e_i)
                     for e_i in {e for e, _t in e_ds.index}})
    for i in starts:
        it = e_ds[i + 40] if i + 40 < len(e_ds) else e_ds[i]
        if int(it["nav_cmd"]) != 0 and float(it["pose_last"][3]) > 3.0:
            cand.append(i + 40 if i + 40 < len(e_ds) else i)
            break
    recs = []
    for idx in cand:
        recs.append(_kt7_one(rig, L, tr, model, e_ds, idx))
    os.makedirs(os.path.join(PKG, "raw", "controls"), exist_ok=True)
    json.dump({"control": "KT7 -- real PhysicalAI eval windows: compute_losses_v3's model(...) call "
                          "vs the bridge's forward_kwargs7 + set_ego_window",
               "n_windows": len(e_ds), "windows": recs, "ckpt": CKPT, "device": DEV},
              open(os.path.join(PKG, "raw", "controls", "KT7_trainer_feed.json"), "w",
                   encoding="utf-8"), indent=1)
    assert len(recs) >= 2 and all(r["traj_bit_identical"] and r["kwargs_equal"] for r in recs)


def _kt7_one(rig, L, tr, model, e_ds, idx) -> dict:
    batch = torch.utils.data.default_collate([e_ds[idx]])
    # with_perception_targets=False skips the map target, so `map_ep` (the window's episode id,
    # which the trainer's dataset emits beside the 10 cm target, refc_v3_train.py:3944) is set
    # here to the SAME value: the episode of this window
    e_i, _t = e_ds.index[idx]
    batch["map_ep"] = torch.tensor([int(e_ds.episodes[e_i].episode_id)], dtype=torch.long)
    cap = {}
    orig = model.forward

    def spy(frames, *a, **kw):
        cap["frames"] = frames.detach().clone()
        cap["kw"] = kw
        cap["ego_window"] = [x.detach().clone() if torch.is_tensor(x) else x
                             for x in (model.core._ego_window or ())]
        torch.manual_seed(1234)
        cap["out"] = orig(frames, *a, **kw)
        raise _Captured()

    model.forward = spy
    try:
        with torch.no_grad():
            try:
                tr.compute_losses_v3(model, batch, DEV, mode=str(rig["rec"]["mode"]))
            except _Captured:
                pass
    finally:
        model.forward = orig
    assert "out" in cap, "compute_losses_v3 never called model(...)"
    assert tuple(cap["kw"]) == R7.FORWARD_KWARGS
    # the SAME window through the bridge's functions
    rows = batch["frames"][0].numpy()
    v0 = float(batch["pose_last"][0, 3])
    decl = {"ego_velocity[3]": [v0, 0.0]}
    nav = {"nav_index": int(batch["nav_cmd"][0])}
    vmax = {"v_max_ms": float(batch["v_max_ms"][0]), "v_max_valid": float(batch["v_max_valid"][0])}
    hist = batch["pose_hist"][0].numpy()
    g, v = model._lift_bank_hires.for_episodes(batch["map_ep"], device=DEV)
    fr = tr.frames_to_device(torch.from_numpy(rows)[None], DEV)
    kw = R7.forward_kwargs7(fr, nav, R7.v0_of(decl), vmax, g[0], v[0], rig["steps"], DEV)
    assert torch.equal(fr, cap["frames"])
    for k in R7.FORWARD_KWARGS:
        a_, b_ = kw[k], cap["kw"][k]
        if torch.is_tensor(b_):
            assert torch.is_tensor(a_) and a_.dtype == b_.dtype and torch.equal(a_, b_), k
        else:
            assert a_ == b_, k
    model.core.set_ego_window(torch.from_numpy(np.asarray(hist, np.float32))[None].to(DEV), 8,
                              actions=None)
    torch.manual_seed(1234)
    with torch.no_grad():
        out = model(fr, **kw)
    assert torch.equal(out["traj"], cap["out"]["traj"])
    assert torch.equal(out["residual_prior_path"], cap["out"]["residual_prior_path"])
    return {"window_index": int(idx), "nav_cmd": int(batch["nav_cmd"][0]),
            "v0_mps": round(float(batch["pose_last"][0, 3]), 3),
            "v_max_valid": float(batch["v_max_valid"][0]), "kwargs_equal": True,
            "traj_bit_identical": True, "pose_hist_dtype": str(batch["pose_hist"].dtype)}


@need
def test_KL_lift_on_real_clips_of_the_models_bank(rig):
    """SPEC §7 KL-lift on REAL PhysicalAI clips: for 3 clips of ``model._lift_bank_hires``, the
    bridge's geometry for the clip's own camera == the bank's ``geometry(ep)``, bit-exact."""
    b = rig["model"]._lift_bank_hires
    eps = sorted(b._extr)[:: max(1, len(b._extr) // 3)][:3]
    for ep in eps:
        g0, v0 = b.geometry(ep)
        g1, v1 = R7.lift_geometry_for_camera(b._extr[ep], b)
        assert torch.equal(g0, g1) and torch.equal(v0, v1), ep


# --------------------------------------------------------------------------- #
# K0 / KD / KI / KPR / KF                                                        #
# --------------------------------------------------------------------------- #
@need
def test_K0_same_seed_is_bit_identical(rig):
    t = rig["toks"][0]
    a, b = _run(rig, t), _run(rig, t)
    assert np.array_equal(a["traj"], b["traj"]) and a["diag"]["sel_idx"] == b["diag"]["sel_idx"]


@need
def test_KD_exact_dedup_equals_native(rig):
    enc = rig["model"].core.encoder
    saved = enc._backbone_dedup
    native = [_run(rig, t) for t in rig["toks"]]
    assert all(r["diag"]["dedup"] == [24, 10] for r in native), [r["diag"]["dedup"] for r in native]
    R7.exact_dedup(enc)
    try:
        exact = [_run(rig, t) for t in rig["toks"]]
    finally:
        enc._backbone_dedup = saved
        enc.memory_levers.pop("eval_exact_dedup", None)
    assert all(r["diag"]["dedup"] == [24, 1] for r in exact)
    d = max(float(np.abs(x["traj"] - y["traj"]).max()) for x, y in zip(native, exact))
    sel = sum(x["diag"]["sel_idx"] == y["diag"]["sel_idx"] for x, y in zip(native, exact))
    out = {"n": len(native), "max_abs_traj_diff_m": d, "sel_identical": sel, "device": DEV,
           "ckpt": CKPT}
    os.makedirs(os.path.join(PKG, "raw", "controls"), exist_ok=True)
    json.dump(out, open(os.path.join(PKG, "raw", "controls", f"KD_exact_dedup_{DEV}.json"), "w",
                        encoding="utf-8"), indent=1)
    assert sel == len(native) and d < 1e-3, out


@need
@pytest.mark.parametrize("what", ["frames", "nav", "v0", "hist"])
def test_KI_every_declared_input_reaches_the_plan(rig, what):
    moved = 0
    for t in rig["toks"]:
        a, b = _run(rig, t), _run(rig, t, mutate=what)
        moved += int(float(np.abs(a["traj"] - b["traj"]).max()) > 1e-4)
    assert moved >= 1, f"mutating {what} moved no plan on {len(rig['toks'])} scenes"


@need
def test_KI_max_speed_reaches_the_tactical_logits(rig):
    t = next(t for t in rig["toks"] if rig["speed"]["tokens"][t]["status"] == "limit")
    rows, decl, nav, _, hist, g, v = _inputs(rig, t)
    m = rig["model"]
    frt = R7.boot7.loader().trainer().frames_to_device(torch.from_numpy(rows)[None], DEV)
    outs = []
    for vm in ({"v_max_ms": 11.17568, "v_max_valid": 1.0}, {"v_max_ms": 0.0, "v_max_valid": 0.0}):
        kw = R7.forward_kwargs7(frt, nav, R7.v0_of(decl), vm, g, v, rig["steps"], DEV)
        m.core.set_ego_window(torch.from_numpy(hist)[None].to(DEV), 8, actions=None)
        torch.manual_seed(5)
        with torch.no_grad():
            outs.append(m(frt, **kw))
    assert float((outs[0]["tacv6_lon_logits"] - outs[1]["tacv6_lon_logits"]).abs().max()) > 1e-5


@need
def test_KPR_emitted_prior_equals_the_model_free_prior(rig):
    worst = 0.0
    for t in rig["toks"]:
        r = _run(rig, t)
        rows, decl, nav, vmax, hist, g, v = _inputs(rig, t)
        mf = R7.prior_model_free(hist, R7.v0_of(decl))
        worst = max(worst, float(np.abs(r["prior_emitted_path"] - mf["path"]).max()))
        assert r["prior_emitted_ctrl"][0] == pytest.approx(mf["a0"], abs=1e-5)
        assert r["prior_emitted_ctrl"][1] == pytest.approx(mf["kappa0"], abs=1e-6)
    assert worst <= 1e-4, worst


@need
def test_KF_emitted_pick_is_replicated_and_the_filter_never_reaches_it(rig):
    """SPEC amendment A1. (a) the bridge's re-derivation argmax(sel_score_v3 | reach_keep) IS the
    model's sel_idx, filter ON and OFF; (b) the EMITTED plan with the filter OFF (a real second
    forward) is bit-identical to the filter-ON plan -- the E9 re-selection (refc_v3.py:2285-2298)
    never applies the ceiling; (c) the ceiling AS DECLARED (R7_CEILDECL_d) keeps the pick inside
    the ceiling whenever a reach-kept candidate is."""
    import torch as _t
    from tanitad.refs import refcv6_selection as v6
    rec = []
    for t in rig["toks"]:
        on = _run(rig, t, filter_on=True)
        off = _run(rig, t, filter_on=False)
        assert on["diag"]["derived_ok"] and off["diag"]["derived_ok"]
        same = bool(np.array_equal(on["traj"], off["traj"]))
        c = on["diag"]["ceiling_read_ms"]
        vm_decl = float(v6.planned_max_speed(_t.tensor(on["traj_ceiling_declared"],
                                                       dtype=_t.float32)[None, None],
                                             horizons=R7.HORIZONS, tick_s=0.1)[0, 0])
        rec.append({"identical_on_off": same, "ceiling": c, "decl_max_speed": vm_decl,
                    "decl_row_empty": on["diag"]["decl_row_empty"],
                    "emitted_over_ceiling": on["diag"]["emitted_over_ceiling"]})
        assert same, "the filter moved the EMITTED plan -- amendment A1's premise no longer holds"
        if c is not None and c != float("inf") and not on["diag"]["decl_row_empty"]:
            assert vm_decl <= c + 1e-5
    json.dump({"control": "KF (SPEC amendment A1)", "rows": rec, "ckpt": CKPT, "device": DEV},
              open(os.path.join(PKG, "raw", "controls", f"KF_filter_reach_{DEV}.json"), "w",
                   encoding="utf-8"), indent=1)


# --------------------------------------------------------------------------- #
# RED — the model refuses mis-wired inputs                                        #
# --------------------------------------------------------------------------- #
@need
def test_REGRESSION_window_not_ending_at_v0_is_refused(rig):
    t = rig["toks"][0]
    rows, decl, nav, vmax, hist, g, v = _inputs(rig, t)
    hist = hist.copy()
    hist[-1, 3] += 1.0                                      # the window ends 1 m/s off v0
    with pytest.raises(Exception):
        R7.run_model7(rig["model"], rows, decl, nav, vmax, hist, g, v, rig["steps"], 0, DEV)


@need
def test_REGRESSION_stride16_geometry_is_refused(rig):
    t = rig["toks"][0]
    r = rig["doc"]["tokens"][t]
    rows, decl, nav, vmax, hist, _, _ = _inputs(rig, t)
    _, rk, _ = rig["bank"].load(r["scene_token"])
    g16, v16, _ = R7.RIG6.lift_geometry(rig["bank"].rigs[rk], rig["road"], stride=16)
    with pytest.raises(Exception):
        R7.run_model7(rig["model"], rows, decl, nav, vmax, hist, g16, v16, rig["steps"], 0, DEV)
