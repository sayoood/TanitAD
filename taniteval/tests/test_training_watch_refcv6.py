"""The refcv6 Training Watch builds from a run's own artifacts -- and its health verdicts can FAIL.

`taniteval/tools/training_watch/build_watch_refcv6.py` turns metrics.jsonl + config.json + the
supervisor log + the live-pid state into the published page. The builder runs here with no
network (`build()` only, never `pull()`) on a synthetic run directory:

* a healthy run renders every section and chart, stamps T0, and reports its numbers;
* an UNPLANNED relaunch (an extra elapsed_s reset beyond the one planned switch) flips the
  summary's `unplanned` to 1 and the chip to "unplanned deaths";
* conflict readings that stopped arriving flip `readings_ok` to False;
* a dead trainer pid flips `train_alive` and the chip to "NOT RUNNING".
The expected values are literals, not expressions over the builder.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "taniteval" / "tools" / "training_watch" / "build_watch_refcv6.py"


def _load(monkeypatch, run_dir):
    monkeypatch.setenv("REFCV6_WATCH_DIR", str(run_dir))
    spec = importlib.util.spec_from_file_location("build_watch_refcv6_under_test", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_dir(tmp_path, *, relaunch=False, stale_readings=False, trainer_alive=True):
    d = tmp_path / "watch"
    d.mkdir()
    rows = [{"step": 1, "conflict_controls": {"ok": True}}]
    el = 0.0
    for step in range(50, 1001, 50):
        if step == 550 or (relaunch and step == 800):
            el = 0.0                                     # a process restart resets elapsed_s
        el += 320.0
        rows.append({"step": step, "loss": 80.0 - step / 20, "traj": 2.0 - step / 2000,
                     "elapsed_s": el, "lr": step * 5e-8, "cuda_max_mem_gb": 21.2,
                     "goal2s_err_m": 7.0 - step / 500, "anchor_acc": step / 5000,
                     "tacv6_lat_ce": 1.0, "tacv6_lon_ce": 1.7, "tacv6_goal_bce": 0.9,
                     "tacv6_goal_conf_bce": 0.3, "map": 1.0, "map_iou_drivable": 0.5 + step / 10000,
                     "box3d": 12.0, "agent_cls": 2.0, "agent_centre": 0.5, "nav_injected": 1.0,
                     "trunk_frame_slots": 21600, "trunk_frames_computed": 10400})
        if step in (500, 1000):
            rows.append({"step": step, "eval_loss": 40.0 - step / 100, "eval_traj": 1.3 - step / 5000,
                         "eval_goal2s_err_m": 7.5 - step / 1000, "eval_anchor_acc": 0.4,
                         "eval_tacv6_lat_ce": 0.96, "eval_tacv6_lon_ce": 1.68, "eval_tacv6_goal_bce": 0.8,
                         "eval_map": 1.05, "eval_map_iou_drivable": 0.56, "eval_box3d": 11.1,
                         "eval_agent_cls": 2.03, "eval_agent_centre": 0.4, "eval_windows": 128})
    last_cd = 700 if stale_readings else 991
    for s in range(501, last_cd + 1, 10):
        rows.append({"step": s, "cd_cos": -0.05, "cd_stage_layer4_cos": 0.1,
                     "cd_fuse_cos": 0.02, "cd_stem_cos": -0.1})
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "config.json").write_text(json.dumps({
        "seams": {"ego_history": {"enable": True, "steps": 8, "channels": 3}},
        "trunk_memory_levers": {"chunk_ckpt": 8, "bn_folded": 104}}), encoding="utf-8")
    n_launch = 3 if relaunch else 2
    (d / "sup.log").write_text("".join(
        f"[sup] lock acquired (fd 200); supervisor pid {100 + i}\n" for i in range(n_launch))
        + "ZZrefcv6-r101-s0-1000-50400-0-1ZZ\n", encoding="utf-8")
    (d / "remote_state.json").write_text(json.dumps({
        "sup_pid": "101", "train_pid": "202", "sup_alive": "1",
        "train_alive": "1" if trainer_alive else "0", "stderr_bytes": "0",
        "ckpt": "1170622369 1790000000", "done": "0", "now": "1790000600"}), encoding="utf-8")
    import shutil
    shutil.copy(BUILDER.parent / "watch.css", d / "watch.css")
    return d


def test_a_healthy_run_renders_every_section_and_stamps_T0(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path))
    page, summary = mod.build()
    assert "<title>refcv6 Training Watch</title>" in page
    for cid in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c7b", "c8", "c9", "c10", "c11"):
        assert f'data-chart="{cid}"' in page, cid
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert f"<td>{fam}</td>" in page, fam
    assert "<b>T0</b>" in page and "never a driving-performance claim" in page
    assert summary["step"] == 1000 and summary["segments"] == 2 and summary["unplanned"] == 0
    assert summary["readings_ok"] is True and summary["train_alive"] is True
    assert summary["eval_last"]["step"] == 1000
    assert "0 unplanned deaths · 1 planned switch" in page


def test_an_UNPLANNED_relaunch_is_reported_as_a_death(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path, relaunch=True))
    page, summary = mod.build()
    assert summary["segments"] == 3 and summary["unplanned"] == 1
    assert "1 unplanned deaths" in page and 'class="chip crit"' in page


def test_conflict_readings_that_STOPPED_are_reported_missing(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path, stale_readings=True))
    page, summary = mod.build()
    assert summary["readings_ok"] is False
    assert "conflict readings MISSING" in page


def test_a_dead_trainer_is_reported_NOT_RUNNING(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path, trainer_alive=False))
    page, summary = mod.build()
    assert summary["train_alive"] is False
    assert "NOT RUNNING" in page
