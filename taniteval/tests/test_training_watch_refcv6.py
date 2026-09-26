"""The refcv6 Training Watch builds from a run's own artifacts -- and its health verdicts can FAIL.

`taniteval/tools/training_watch/build_watch_refcv6.py` turns metrics.jsonl + config.json + the
supervisor log + the live-pid state into the published page. The builder runs here with no
network (`build()` only, never `pull()`) on a synthetic run directory:

* a healthy run renders every section and chart, stamps T0, and reports its numbers;
* an UNPLANNED relaunch (an extra elapsed_s reset beyond the one planned switch) flips the
  summary's `unplanned` to 1 and the chip to "unplanned deaths";
* conflict readings that stopped arriving flip `readings_ok` to False;
* a dead trainer pid flips `train_alive` and the chip to "NOT RUNNING";
* a supervisor token SPLIT over two lines (its `grep -c ... || echo 0` printing 0 twice, MEASURED
  2026-09-26 on the live run from step 6,571) reads as 0 tracebacks, never as the step count;
* stderr is judged by its CONTENT: the one diagnosed line passes, an undiagnosed line or a
  traceback fails, a second instance of the diagnosed warning is undiagnosed again, and a stderr
  that has bytes on Thor but was not read fails.
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


def _load(monkeypatch, run_dir, eval_pkg=None, eval_live=None):
    monkeypatch.setenv("REFCV6_WATCH_DIR", str(run_dir))
    # hermetic by default: an EMPTY eval package, so no test reads the real NavSim / battery results
    # and no verdict chip on the page can satisfy an assertion about a health chip
    monkeypatch.setenv("REFCV6_EVAL_PKG", str(eval_pkg or run_dir / "no_eval_pkg"))
    monkeypatch.setenv("REFCV6_EVAL_LIVE", str(eval_live or run_dir / "no_eval_live"))
    spec = importlib.util.spec_from_file_location("build_watch_refcv6_under_test", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # the synthetic run has exactly one planned switch; pin it here so the live run's own segment
    # list (FACTS) can grow without changing what these tests expect
    mod.FACTS["segments"] = [("a", "launch", "t0"), ("b", "planned switch", "t1")]
    return mod


# the live run's one stderr line, byte for byte (112 B with its newline)
DIAGNOSED = ("W0924 09:03:45.814000 3346338 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use "
             "max_autotune_gemm mode")


def _run_dir(tmp_path, *, relaunch=False, stale_readings=False, trainer_alive=True,
             split_token=False, stderr=None, stderr_bytes=None):
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
        + ("ZZrefcv6-r101-s0-1000-50400-0\n0-1ZZ\n" if split_token
           else "ZZrefcv6-r101-s0-1000-50400-0-1ZZ\n"), encoding="utf-8")
    if stderr is not None:
        (d / "stderr_tail.log").write_bytes(stderr.encode("utf-8"))
    sb = stderr_bytes if stderr_bytes is not None else len((stderr or "").encode("utf-8"))
    (d / "remote_state.json").write_text(json.dumps({
        "sup_pid": "101", "train_pid": "202", "sup_alive": "1",
        "train_alive": "1" if trainer_alive else "0", "stderr_bytes": str(sb),
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


def test_a_SPLIT_supervisor_token_reads_zero_tracebacks_not_the_step_count(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path, split_token=True))
    page, summary = mod.build()
    assert summary["n_err"] == 0 and summary["token_split"] is True and summary["token_ok"] is True
    assert "0 unplanned deaths · 1 planned switch" in page


def test_the_DIAGNOSED_stderr_line_passes(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path, stderr=DIAGNOSED + "\n"))
    page, summary = mod.build()
    assert summary["stderr_bytes"] == 112 and summary["stderr_lines"] == 1
    assert summary["stderr_undiagnosed"] == 0 and summary["n_err_client"] == 0
    assert summary["stderr_read_ok"] is True
    assert "stderr: 1 line(s), all diagnosed" in page


def test_an_UNDIAGNOSED_line_fails_and_a_traceback_is_counted_from_the_content(tmp_path, monkeypatch):
    err = DIAGNOSED + "\nTrace" "back (most recent call last):\n  File \"x.py\", line 1\nRuntimeError: boom\n"
    mod = _load(monkeypatch, _run_dir(tmp_path, stderr=err))
    page, summary = mod.build()
    assert summary["stderr_lines"] == 4 and summary["stderr_undiagnosed"] == 3
    assert summary["n_err_client"] == 1
    assert "stderr: 3 UNDIAGNOSED line(s)" in page and "<b>UNDIAGNOSED</b>" in page


def test_a_SECOND_instance_of_the_diagnosed_warning_is_undiagnosed_again(tmp_path, monkeypatch):
    again = ("W0926 11:00:00.000000 3346338 torch/_inductor/utils.py:1953] [0/3] Not enough SMs to use "
             "max_autotune_gemm mode")
    mod = _load(monkeypatch, _run_dir(tmp_path, stderr=DIAGNOSED + "\n" + again + "\n"))
    page, summary = mod.build()
    assert summary["stderr_lines"] == 2 and summary["stderr_undiagnosed"] == 1
    assert "stderr: 1 UNDIAGNOSED line(s)" in page


def test_a_stderr_that_was_NOT_READ_fails(tmp_path, monkeypatch):
    # 112 B on Thor, nothing pulled: never read that as a clean stderr
    mod = _load(monkeypatch, _run_dir(tmp_path, stderr="", stderr_bytes=112))
    page, summary = mod.build()
    assert summary["stderr_read_ok"] is False and "stderr NOT READ" in page


def test_an_UNREAD_supervisor_count_fails(tmp_path, monkeypatch):
    # the fixed supervisor prints U when grep could not read its log (exit 2)
    d = _run_dir(tmp_path)
    (d / "sup.log").write_text((d / "sup.log").read_text(encoding="utf-8").replace(
        "ZZrefcv6-r101-s0-1000-50400-0-1ZZ", "ZZrefcv6-r101-s0-1000-50400-U-1ZZ"), encoding="utf-8")
    mod = _load(monkeypatch, d)
    page, summary = mod.build()
    assert summary["n_err"] is None and summary["token_ok"] is False
    assert "tracebacks NOT READ" in page


# ---------------------------------------------------------------- NavSim + battery (2026-09-26) ----
def _eval_pkg(tmp_path):
    """A banked step 5000 (all three splits) and a step 30000 with only warmup banked; the live lane
    has started step 30000's navtest (a bridge dir) but not its navhard."""
    pkg, live = tmp_path / "evalpkg", tmp_path / "evallive"
    m5 = pkg / "navsim" / "raw" / "milestones" / "step5000"
    m30 = pkg / "navsim" / "raw" / "milestones" / "step30000"
    m5.mkdir(parents=True)
    m30.mkdir(parents=True)
    iv = {"lo": 0.443, "hi": 0.4836}
    arms_nt = {"R6_A1": {"NC": 75.5146, "DAC": 73.2505, "TTC": 61.1806, "EP": 47.2631, "C": 99.9835, "DDC": 89.1734,
                         "PDMS": 46.4846, "interval": iv},
               "STOP": {"PDMS": 61.8202}, "CV": {"PDMS": 20.6517}, "HUMAN": {"PDMS": 94.5514}}
    (m5 / "summary_navtest.json").write_text(json.dumps({"n_tokens": 12146, "arms": arms_nt, "pairs": {
        "R6_A1__minus__STOP": {"interval": {"delta": -0.1534, "lo": -0.1829, "hi": -0.1277}}}}), encoding="utf-8")
    (m5 / "summary_navhard.json").write_text(json.dumps({"arms": {
        "R6_A1": {"official_two_stage_EPDMS": 0.1512, "n_stage2": 5462}, "STOP_zero": {"official_two_stage_EPDMS": 0.2985},
        "CV_official": {"official_two_stage_EPDMS": 0.1148}, "ECHO_ha0_ext": {"official_two_stage_EPDMS": 0.1429}}}),
        encoding="utf-8")
    for d in (m5, m30):
        (d / "summary_warmup.json").write_text(json.dumps({"arms": {"R6_A1": {"n_stage2": 204}}}), encoding="utf-8")
    warm = lambda v, m: {"verdict": "FAIL", "values": {"R6_A1": v, "CV_official": 0.3971, "STOP_zero": 0.5212,
                                                      "ECHO_ha0_ext": 0.4287}, "margin": m, "seed_floor": 0.0157,
                         "interval": "UNAVAILABLE (7 logs < 8)"}
    (m5 / "BARS.json").write_text(json.dumps({"bars": {
        "warmup": warm(0.3966, -0.1247),
        "navhard": {"verdict": "FAIL", "paired_vs_STOP": {"delta": -0.1473, "lo": -0.1794, "hi": -0.1137}},
        "navtest": {"verdict": "FAIL", "stretch_published": {"RefPlanner": "77.7 PDMS (a banked reference)"}}}}),
        encoding="utf-8")
    (m30 / "BARS.json").write_text(json.dumps({"bars": {
        "warmup": warm(0.4753, -0.046),
        "navhard": {"status": "UNAVAILABLE"}, "navtest": {"status": "UNAVAILABLE"}}}), encoding="utf-8")
    (live / "navsim" / "raw" / "milestones" / "step30000" / "bridge_navtest").mkdir(parents=True)
    bat = pkg / "battery" / "raw" / "step5000"
    bat.mkdir(parents=True)
    (bat / "battery_summary.json").write_text(json.dumps({"bars": [
        {"id": "BAR-R6-1", "statement": "refcv6 beats the ECHO control", "verdict": "FAIL",
         "per_inference_seed": {"0": {"delta": 0.0885, "lo": 0.0695, "hi": 0.1094, "separated": True,
                                      "n_windows": 4754, "n_episodes": 139}}}]}), encoding="utf-8")
    return pkg, live


def test_navsim_kpis_are_read_from_the_banked_milestones_and_a_started_split_reads_running(tmp_path, monkeypatch):
    pkg, live = _eval_pkg(tmp_path)
    mod = _load(monkeypatch, _run_dir(tmp_path), eval_pkg=pkg, eval_live=live)
    page, summary = mod.build()
    assert summary["navsim"] == {"5000": {"navtest": 46.4846, "navhard": 0.1512, "warmup": 0.3966},
                                 "30000": {"navtest": "running", "navhard": "not run", "warmup": 0.4753}}
    assert summary["battery_steps"] == [5000]
    assert "<b>46.48</b> [44.30, 48.36]" in page                  # PDMS x100 with its log-cluster interval
    assert "vs STOP -15.34 [-18.29, -12.77]" in page             # the paired read against the STOP plan
    assert "<b>0.1512</b>" in page and "<b>0.4753</b>" in page
    assert "pre-switch: F3 detach-only" in page                   # both checkpoints predate the A16 fixes
    assert "RefPlanner: 77.7 PDMS (a banked reference)" in page   # the published line is READ, never typed
    assert "No checkpoint beats the STOP plan yet on any split" in page
    assert "BAR-R6-1" in page and 'class="chip crit verdict"' in page


def test_a_checkpoint_that_BEATS_STOP_does_not_get_the_no_checkpoint_sentence(tmp_path, monkeypatch):
    pkg, live = _eval_pkg(tmp_path)
    m30 = pkg / "navsim" / "raw" / "milestones" / "step30000" / "BARS.json"
    bars = json.loads(m30.read_text(encoding="utf-8"))
    bars["bars"]["warmup"]["values"]["R6_A1"] = 0.6001          # above STOP 0.5212
    bars["bars"]["warmup"]["verdict"] = "PASS"
    m30.write_text(json.dumps(bars), encoding="utf-8")
    mod = _load(monkeypatch, _run_dir(tmp_path), eval_pkg=pkg, eval_live=live)
    page, summary = mod.build()
    assert summary["navsim"]["30000"]["warmup"] == 0.6001
    assert "No checkpoint beats the STOP plan yet" not in page
    assert 'class="chip good verdict"' in page


def test_an_unreadable_eval_package_shows_nothing_rather_than_a_guess(tmp_path, monkeypatch):
    mod = _load(monkeypatch, _run_dir(tmp_path))                  # the default: an empty eval package
    page, summary = mod.build()
    assert "No banked NavSim milestone was readable" in page
    assert summary["navsim"] == {} and summary["battery_steps"] == []
    assert 'class="chip crit verdict"' not in page and 'class="chip good verdict"' not in page
