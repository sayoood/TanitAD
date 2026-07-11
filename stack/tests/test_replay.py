"""Replay app tests (tanitad/replay/* + scripts/replay_app.py).

CPU-only, synthetic tiny episodes (unicycle kinematics, the test_refb.py
fixture pattern). Pins:
(a) engine windowing: record count, `_ego` ground-truth waypoints on a
    hand-computable straight-line episode, fail-loud on short episodes,
(b) RefBArm + RefAArm(pool, smoke dims, toy tokenizer) + MainArm(smoke
    config) each run windows end-to-end with sane, finite outputs,
(c) stats aggregation matches hand-computed values,
(d) regression compare flags injected degradation and passes within
    tolerance (both metric directions),
(e) rr_log writes a non-empty .rrd (skipped gracefully without rerun-sdk),
(f) replay_app end-to-end: --mode test on synthetic episodes produces
    stats.json; the --baseline gate returns 0 on self-compare and 1 on an
    injected regression.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data._contract import assemble_episode  # noqa: E402
from tanitad.data.mixing import save_episode  # noqa: E402
from tanitad.replay import stats as replay_stats  # noqa: E402
from tanitad.replay.engine import (WAYPOINT_STEPS, ArmOutput,  # noqa: E402
                                   ReplayEngine, TimestepRecord,
                                   load_corpora, split_fit_replay)

# ---------- synthetic kinematics (test_refb.py fixture pattern) --------------

T_EP = 64
SIZE = 64
V0 = 8.0
DT = 0.1


def _poses(T: int, dt: float = DT, v0: float = V0, yaw_rate: float = 0.0,
           accel: float = 0.0) -> torch.Tensor:
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T: int, eid: int, yaw_rate: float = 0.0, accel: float = 0.0):
    g = torch.Generator().manual_seed(1000 + eid)
    frames = [torch.rand(1, SIZE, SIZE, generator=g) for _ in range(T)]
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    return assemble_episode(frames, [p.numpy() for p in poses],
                            [yaw_rate] * T, DT, eid)


def _make_cache(tmp_path: Path, n: int = 4, T: int = T_EP,
                name: str = "toy-val") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    specs = [(0.0, 0.0), (0.05, 0.0), (0.0, -0.8), (0.02, 0.4)]
    for i in range(n):
        yr, ac = specs[i % len(specs)]
        save_episode(_episode(T, eid=i, yaw_rate=yr, accel=ac),
                     str(d / f"ep_{i:05d}.pt"))
    return d


class DummyArm:
    """Minimal ArmAdapter: constant outputs, isolates the engine."""
    name = "dummy"
    window = 4
    needs_ahead = 20
    requires_fit = False

    def prepare(self, engine, fit_reps):
        pass

    def run_batch(self, batch):
        return [ArmOutput(latency_ms=1.0,
                          waypoints=np.zeros((len(WAYPOINT_STEPS), 2),
                                             dtype=np.float32),
                          action=np.zeros(2, dtype=np.float32))
                for _ in range(len(batch))]


# ---------- (a) engine windowing + ground truth -------------------------------

def test_engine_emits_records_with_exact_gt(tmp_path):
    cache = _make_cache(tmp_path, n=2)
    reps = load_corpora(cache)
    assert [r.corpus for r in reps] == ["toy-val", "toy-val"]

    engine = ReplayEngine([DummyArm()], batch_size=3, stride=8)
    records = list(engine.run(reps))

    # Window count: starts = range(0, T - W - need, stride) per episode.
    per_ep = len(range(0, T_EP - 4 - 20, 8))
    assert len(records) == 2 * per_ep
    assert [r.step for r in records] == list(range(len(records)))

    # Episode 0 is a straight line at constant v: waypoint at k steps is
    # exactly (v*k*dt, 0) in the ego frame; action = (steer 0, accel 0).
    r0 = records[0]
    assert r0.corpus == "toy-val" and r0.episode_id == 0 and r0.t == 3
    expect_x = np.array([V0 * k * DT for k in WAYPOINT_STEPS])
    assert np.allclose(r0.gt_waypoints[:, 0], expect_x, atol=1e-4)
    assert np.allclose(r0.gt_waypoints[:, 1], 0.0, atol=1e-4)
    assert np.allclose(r0.gt_action, 0.0, atol=1e-6)
    assert r0.speed == pytest.approx(V0)
    assert r0.yaw_rate == pytest.approx(0.0)
    assert r0.frame is None                       # emit_frames off by default
    assert set(r0.arms) == {"dummy"}


def test_engine_fails_loud_on_short_episode(tmp_path):
    cache = _make_cache(tmp_path, n=1, T=20)      # < window + need_ahead
    reps = load_corpora(cache)
    engine = ReplayEngine([DummyArm()])
    with pytest.raises(ValueError, match="too short"):
        list(engine.run(reps))


def test_split_fit_replay_is_episode_level(tmp_path):
    cache = _make_cache(tmp_path, n=4)
    reps = load_corpora(cache)
    fit, replay = split_fit_replay(reps, fit_frac=0.5)
    assert len(fit) == 2 and len(replay) == 2
    fit_ids = {r.episode.episode_id for r in fit}
    rep_ids = {r.episode.episode_id for r in replay}
    assert fit_ids.isdisjoint(rep_ids)
    with pytest.raises(ValueError, match="no replay"):
        split_fit_replay(reps[:1], fit_frac=0.5)


# ---------- (b) the three arms -------------------------------------------------

def _save_ckpt(model, path: Path) -> Path:
    torch.save({"model": model.state_dict(), "step": 7}, path)
    return path


def test_refb_arm_runs_and_emits_heads(tmp_path):
    from tanitad.refs.refb import (MANEUVER_CLASSES, RefBModel,
                                   refb_smoke_config)
    from tanitad.replay.arms import RefBArm

    torch.manual_seed(0)
    cfg = refb_smoke_config()
    ckpt = _save_ckpt(RefBModel(cfg), tmp_path / "refb.pt")
    cache = _make_cache(tmp_path, n=2)
    reps = load_corpora(cache)

    arm = RefBArm(ckpt, cfg=cfg, device="cpu")
    assert arm.step == 7 and not arm.requires_fit
    engine = ReplayEngine([arm], batch_size=4, stride=10)
    engine.prepare([])                            # nothing to fit — must pass
    records = list(engine.run(reps))
    assert records
    out = records[0].arms["refb"]
    assert out.waypoints.shape == (len(WAYPOINT_STEPS), 2)
    assert out.action.shape == (2,)
    assert out.action_seq.shape == (cfg.operative.action_seq, 2)
    assert out.maneuver_probs.shape == (len(MANEUVER_CLASSES),)
    assert out.maneuver_probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert 0 <= out.maneuver_gt < len(MANEUVER_CLASSES)
    assert out.nav_cmd in (0, 1, 2)
    assert out.conf is not None and math.isfinite(out.conf)
    assert out.ood is not None                    # 0.0 pre-warmup (count < 2)
    assert out.latency_ms > 0
    assert np.isfinite(out.waypoints).all()


def _smoke_pred_cfg():
    import refa_train
    return refa_train.smoke_pred_config()


def test_refa_arm_pool_smoke(tmp_path):
    from tanitad.refs.refa import RefAModel
    from tanitad.replay.arms import RefAArm, ToyTokenizer

    torch.manual_seed(0)
    tok = ToyTokenizer(n_tokens=16, d=32, seed=1)
    model = RefAModel(_smoke_pred_cfg(), d_dino=32, state_dim=64,
                      sigreg_slices=16, adapter_kind="pool")
    feats = [tok(torch.rand(6, 1, SIZE, SIZE)) for _ in range(3)]
    model.standardizer.fit(iter(feats))
    ckpt = _save_ckpt(model, tmp_path / "refa.pt")

    cache = _make_cache(tmp_path, n=4)
    fit, replay = split_fit_replay(load_corpora(cache), fit_frac=0.5)
    arm = RefAArm(ckpt, adapter="pool", pred_cfg=_smoke_pred_cfg(),
                  tokenizer=tok, d_dino=32, state_dim=64, sigreg_slices=16)
    engine = ReplayEngine([arm], batch_size=4, stride=6, fit_stride=2)
    engine.prepare(fit)
    assert arm.fit_report["fit_windows"] >= engine.min_fit_windows
    assert "probe_r2_wp" in arm.fit_report

    records = list(engine.run(replay))
    out = records[0].arms["refa"]
    assert out.waypoints.shape == (len(WAYPOINT_STEPS), 2)
    assert np.isfinite(out.waypoints).all()
    assert out.action.shape == (2,)
    assert set(out.imag_rel) == set(arm.horizons)
    assert all(v >= 0 and math.isfinite(v) for v in out.imag_rel.values())
    assert set(out.imag_traj) == set(arm.horizons)
    assert out.maneuver_probs is None             # structural: no such head
    assert out.sigma is None


def test_refa_arm_refuses_unfitted_standardizer(tmp_path):
    from tanitad.refs.refa import RefAModel
    from tanitad.replay.arms import RefAArm

    model = RefAModel(_smoke_pred_cfg(), d_dino=32, state_dim=64,
                      sigreg_slices=16, adapter_kind="pool")
    ckpt = _save_ckpt(model, tmp_path / "refa_unfit.pt")
    with pytest.raises(RuntimeError, match="UNFITTED standardizer"):
        RefAArm(ckpt, adapter="pool", pred_cfg=_smoke_pred_cfg(),
                d_dino=32, state_dim=64, sigreg_slices=16)


def test_main_arm_smoke(tmp_path):
    from tanitad.config import smoke_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.replay.arms import MainArm

    torch.manual_seed(0)
    cfg = smoke_config()
    ckpt = _save_ckpt(WorldModel(cfg), tmp_path / "main.pt")
    cache = _make_cache(tmp_path, n=4)
    fit, replay = split_fit_replay(load_corpora(cache), fit_frac=0.5)

    arm = MainArm(ckpt, cfg=cfg, device="cpu")
    assert arm.requires_fit and arm.window == cfg.predictor.window
    engine = ReplayEngine([arm], batch_size=4, stride=8, fit_stride=2)

    # Fail-loud: replay before probe fitting must raise, not emit garbage.
    with pytest.raises(RuntimeError, match="no fitted probes"):
        list(engine.run(replay))

    engine.prepare(fit)
    records = list(engine.run(replay))
    out = records[0].arms["main"]
    assert out.waypoints.shape == (len(WAYPOINT_STEPS), 2)
    assert np.isfinite(out.waypoints).all()
    assert out.sigma is not None and out.sigma > 0          # H15 belief sigma
    assert set(out.imag_rel) == set(cfg.predictor.horizons)
    assert set(out.imag_traj) == set(cfg.predictor.horizons)
    assert out.latency_ms > 0


# ---------- (c) stats aggregation on hand-computed values ---------------------

def _record(step, ade_wp, action, probs=None, man_gt=None, latency=10.0,
            t=3):
    gt_wp = np.zeros((len(WAYPOINT_STEPS), 2), dtype=np.float64)
    out = ArmOutput(latency_ms=latency,
                    waypoints=np.full((len(WAYPOINT_STEPS), 2), ade_wp,
                                      dtype=np.float64),
                    action=np.asarray(action, dtype=np.float64),
                    maneuver_probs=(np.asarray(probs) if probs is not None
                                    else None),
                    maneuver_gt=man_gt)
    return TimestepRecord(step=step, corpus="toy", episode_id=0, ep_index=0,
                          t=t, gt_waypoints=gt_wp,
                          gt_action=np.zeros(2), speed=8.0, yaw_rate=0.0,
                          arms={"a": out})


def test_aggregate_matches_hand_computation():
    recs = [
        _record(0, 1.0, [0.5, -0.5], probs=[0.1, 0.7, 0.2, 0.0, 0.0],
                man_gt=1, latency=10.0, t=3),
        _record(1, 0.0, [0.0, 0.0], probs=[0.9, 0.1, 0.0, 0.0, 0.0],
                man_gt=1, latency=20.0, t=11),
    ]
    s = replay_stats.aggregate(recs, meta={"who": "test"}, worst_k=1)
    m = s["arms"]["a"]
    rt2 = math.sqrt(2.0)
    assert m["n_windows"] == 2
    assert m["ade"] == pytest.approx(rt2 / 2, abs=1e-4)      # (sqrt2 + 0)/2
    for k in WAYPOINT_STEPS:
        assert m[f"ade@{k}"] == pytest.approx(rt2 / 2, abs=1e-4)
    assert m["fde@20"] == m["ade@20"]
    assert m["steer_mae"] == pytest.approx(0.25)
    assert m["accel_mae"] == pytest.approx(0.25)
    assert m["maneuver_acc"] == pytest.approx(0.5)           # hit then miss
    assert m["latency_p50_ms"] == pytest.approx(15.0)
    assert m["per_episode_ade"]["toy/ep0"] == pytest.approx(rt2 / 2, abs=1e-4)
    assert len(m["worst_windows"]) == 1                      # worst_k=1
    assert m["worst_windows"][0]["t"] == 3                   # the bad window
    assert m["worst_windows"][0]["ade"] == pytest.approx(rt2, abs=1e-3)
    assert s["meta"]["who"] == "test" and s["meta"]["n_records"] == 2


def test_aggregate_fails_loud_on_empty_and_wiring_bugs():
    with pytest.raises(ValueError, match="zero records"):
        replay_stats.aggregate([])
    rec = _record(0, 1.0, [0, 0])
    rec.arms["a"].waypoints = None                # arm that never decodes
    with pytest.raises(ValueError, match="no waypoint outputs"):
        replay_stats.aggregate([rec])


# ---------- (d) regression compare ---------------------------------------------

def _stats(ade, acc, lat):
    return {"arms": {"m": {"ade": ade, "maneuver_acc": acc,
                           "latency_p50_ms": lat, "n_windows": 100}}}


def test_compare_flags_degradation_and_passes_within_tolerance():
    base = _stats(ade=1.0, acc=0.90, lat=10.0)

    ok = _stats(ade=1.02, acc=0.895, lat=12.0)     # all inside tolerance
    regs, rows = replay_stats.compare(ok, base)
    assert regs == []
    assert {r["metric"]: r["status"] for r in rows}["m.n_windows"] == "OK"

    bad = _stats(ade=1.20, acc=0.90, lat=10.0)     # ADE +20 % > 5 % tol
    regs, _ = replay_stats.compare(bad, base)
    assert [r["metric"] for r in regs] == ["m.ade"]

    drop = _stats(ade=1.0, acc=0.50, lat=10.0)     # higher-better direction
    regs, _ = replay_stats.compare(drop, base)
    assert [r["metric"] for r in regs] == ["m.maneuver_acc"]

    slow = _stats(ade=1.0, acc=0.90, lat=16.0)     # +60 % > 50 % latency tol
    regs, _ = replay_stats.compare(slow, base)
    assert [r["metric"] for r in regs] == ["m.latency_p50_ms"]

    # user tolerance override rescues the ADE regression
    regs, _ = replay_stats.compare(bad, base, tolerances={"ade": 0.5})
    assert regs == []

    # improvements are reported BETTER, never a regression
    better = _stats(ade=0.5, acc=0.99, lat=5.0)
    regs, rows = replay_stats.compare(better, base)
    assert regs == []
    assert all(r["status"] in ("BETTER", "OK", "INFO") for r in rows)
    assert "m.ade" in replay_stats.format_table(rows)


def test_compare_reports_one_sided_metrics_as_info():
    base = _stats(1.0, 0.9, 10.0)
    cur = json.loads(json.dumps(base))
    cur["arms"]["m"]["sigma_mean"] = 0.3           # new head, no baseline
    regs, rows = replay_stats.compare(cur, base)
    assert regs == []
    info = [r for r in rows if r["metric"] == "m.sigma_mean"]
    assert info and info[0]["status"] == "INFO"


# ---------- (e) rr_log smoke ----------------------------------------------------

def test_rr_log_writes_rrd(tmp_path):
    pytest.importorskip("rerun", reason="rerun-sdk not installed")
    from tanitad.replay.rr_log import RerunLogger

    rrd = tmp_path / "smoke.rrd"
    logger = RerunLogger(rrd=str(rrd))
    for i in range(3):
        rec = _record(i, ade_wp=0.5 * i, action=[0.1, -0.2],
                      probs=[0.2, 0.2, 0.2, 0.2, 0.2], man_gt=2, t=3 + i)
        rec.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        rec.arms["a"].imag_traj = {1: np.array([0.8, 0.0])}
        rec.arms["a"].imag_rel = {1: 0.9}
        rec.arms["a"].conf = 0.4
        logger.log_record(rec)
    logger.close()
    assert rrd.exists() and rrd.stat().st_size > 0


def test_rr_logger_requires_a_sink():
    pytest.importorskip("rerun", reason="rerun-sdk not installed")
    from tanitad.replay.rr_log import RerunLogger
    with pytest.raises(ValueError, match="sink"):
        RerunLogger()


# ---------- (f) replay_app end-to-end -------------------------------------------

def test_replay_app_test_mode_and_regression_gate(tmp_path):
    import replay_app
    from tanitad.config import smoke_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.refs.refb import RefBModel, refb_smoke_config

    torch.manual_seed(0)
    data_root = tmp_path / "episodes"
    _make_cache(data_root, n=4, name="toy-val")
    main_ck = _save_ckpt(WorldModel(smoke_config()), tmp_path / "main.pt")
    refb_ck = _save_ckpt(RefBModel(refb_smoke_config()), tmp_path / "refb.pt")
    out = tmp_path / "out"

    argv = ["--mode", "test", "--smoke",
            "--arms", f"main:{main_ck}", f"refb:{refb_ck}",
            "--data-root", str(data_root), "--stride", "8",
            "--batch", "4", "--out", str(out), "--device", "cpu"]
    assert replay_app.main(argv) == 0
    stats = json.loads((out / "stats.json").read_text())
    assert set(stats["arms"]) == {"main", "refb"}
    assert stats["arms"]["main"]["n_windows"] > 0
    assert stats["meta"]["arms"]["main"]["fit"]["fit_windows"] > 0

    # Self-compare passes the gate...
    assert replay_app.main(argv + ["--baseline",
                                   str(out / "stats.json")]) == 0
    # ...and an injected degradation (baseline claims better ADE) fails it.
    degraded = json.loads((out / "stats.json").read_text())
    for arm in degraded["arms"].values():
        arm["ade"] = arm["ade"] * 0.5
    baseline2 = tmp_path / "baseline2.json"
    baseline2.write_text(json.dumps(degraded))
    assert replay_app.main(argv + ["--baseline", str(baseline2)]) == 1
    assert (out / "regression.json").exists()


def test_parse_arm_spec_windows_paths(tmp_path):
    import replay_app
    ck = tmp_path / "c.pt"
    ck.write_bytes(b"x")
    name, path, opt = replay_app.parse_arm_spec(f"refa:{ck}:grid")
    assert name == "refa" and Path(path) == ck and opt == "grid"
    name, path, opt = replay_app.parse_arm_spec(f"main:{ck}")
    assert name == "main" and Path(path) == ck and opt is None
    with pytest.raises(SystemExit):
        replay_app.parse_arm_spec(f"bogus:{ck}")
    with pytest.raises(SystemExit):
        replay_app.parse_arm_spec(f"refb:{ck}:grid")   # opt only for refa
