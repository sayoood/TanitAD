"""The refcv7 Training Watch builds from a run's own artifacts -- and every one of its alarms can FIRE.

`taniteval/tools/training_watch/build_watch_refcv7.py` turns metrics.jsonl + config.json + the
supervisor log + the live-pid state into the published page. The builder runs here with no network
(`build()` / `main(["--metrics", ...])`, never `pull()`) on synthetic run directories in a temp dir:

* the 10 cm map's 40-key contract (SPEC_REFCV7 A8): all present -> no alarm; ONE key missing ->
  UNAVAILABLE + alarm, never drawn as 0; the thin-class alarm is the REGISTERED rule only
  (LOGGING_SPEC_MAP10 5.3, the Master Mind's ruling 2026-09-27): any class at or below 0.05 at 0-20 m from
  step 5,000 -> RED and latched; 20-60 m -> amber, informative; beyond 60 m -> values only;
* the box heads (A10): confident / VIS-1 positives 3.4 -> ALARM, 1.0 -> none, the band edges inclusive,
  armed from step 5,000 (grey "warming up" before);
* the declared gradient reach (D3): a declared `ga_mh_refine` absent or exactly 0 -> ALARM;
* the NavSim count guard, the gate's metrics-only form, the refcv6 health checks, the pace line.

The expected values are LITERALS, never expressions over the builder: the class and band names are
written out here from SPEC_REFCV7 A8 item 3, not imported from the module under test.
"""
from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "taniteval" / "tools" / "training_watch" / "build_watch_refcv7.py"
ARM = "refcv7-r101-s0"

# SPEC_REFCV7 A8 item 3, written out (a literal contract, not the module's own tuple)
CLASSES = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")
BANDS = ("0_20", "20_40", "40_60", "60_80", "80_100")
THIN = ("lane", "crosswalk", "arrow", "edge", "hatched")
DECLARED_GA = sorted(f"ga_{p}{s}" for p in ("trunk", "planner", "tac_decoder", "box_memory", "box_decoder",
                                            "mh_lift", "mh_encoder", "mh_refine", "mh_trunk_s8_stage")
                     for s in ("", "_n"))


NO_PKG = str(Path(__file__).resolve().parent / "no-refcv7-eval-package-here")   # never created


def _load(monkeypatch, *, eval_pkg=None, eval_live=None, navsim_arm=None):
    for k in ("REFCV7_ARM", "REFCV7_RUN", "REFCV7_WATCH_DIR"):
        monkeypatch.delenv(k, raising=False)
    # hermetic: the builder's DEFAULT package is a real D: path, so every test names its own -- a path
    # that does not exist unless the test builds one
    assert not Path(NO_PKG).exists()
    monkeypatch.setenv("REFCV7_EVAL_PKG", str(eval_pkg) if eval_pkg is not None else NO_PKG)
    if eval_live is None:
        monkeypatch.delenv("REFCV7_EVAL_LIVE", raising=False)
    else:
        monkeypatch.setenv("REFCV7_EVAL_LIVE", str(eval_live))
    if navsim_arm is None:
        monkeypatch.delenv("REFCV7_NAVSIM_ARM", raising=False)
    spec = importlib.util.spec_from_file_location("build_watch_refcv7_under_test", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the synthetic run ----
def _iou(c, b):
    if c in THIN:
        return 0.2 if b in ("0_20", "20_40", "40_60") else 0.01
    return 0.8 if b == "0_20" else 0.5


def _map_eval(overrides=None):
    """40 x (iou, iouraw, lshare, n, inter, union) of an eval row. `overrides[(cls, band)]` =
    a float, None (undefined: no GT cells, no prediction), "DROP" (the iou key absent) or "ZERO_NOGT"
    (IoU 0.0 with NO ground-truth cells: the class was predicted where it does not exist)."""
    row = {"eval_map_hires": 1.2, "eval_n_map_hires_cells": 1.0e6}
    for c in CLASSES:
        for b in BANDS:
            v = (overrides or {}).get((c, b), _iou(c, b))
            k = f"eval_map_hires_iou_{c}_{b}"
            if v == "ZERO_NOGT":
                row.update({k: 0.0, f"eval_map_hires_n_{c}_{b}": 0.0, f"eval_map_hires_inter_{c}_{b}": 0.0,
                            f"eval_map_hires_union_{c}_{b}": 50.0, f"eval_map_hires_iouraw_{c}_{b}": 0.0,
                            f"eval_map_hires_lshare_{c}_{b}": 0.0})
                continue
            if v is None:
                row.update({k: None, f"eval_map_hires_n_{c}_{b}": 0.0, f"eval_map_hires_inter_{c}_{b}": 0.0,
                            f"eval_map_hires_union_{c}_{b}": 0.0})
            else:
                vv = 0.3 if v == "DROP" else v
                row.update({f"eval_map_hires_n_{c}_{b}": 125.0, f"eval_map_hires_union_{c}_{b}": 200.0,
                            f"eval_map_hires_inter_{c}_{b}": vv * 200.0})
                if v != "DROP":
                    row[k] = v
            row[f"eval_map_hires_iouraw_{c}_{b}"] = None if v is None else 0.1
            row[f"eval_map_hires_lshare_{c}_{b}"] = 0.025
    return row


def _box_eval(ratios=None):
    row = {}
    for h in ("box3d", "agent"):
        r = (ratios or {}).get(h, 1.0)
        n_pos, n_conf = 100.0, round(100.0 * r, 5)
        tp = 60.0
        row.update({f"eval_{h}_n_pos": n_pos, f"eval_{h}_n_conf": n_conf, f"eval_{h}_tp@gate": tp,
                    f"eval_{h}_conf_ratio": r, f"eval_{h}_conf_ratio_alarm": 0.0 if 0.5 <= r <= 1.5 else 1.0,
                    f"eval_{h}_prec@gate": round(tp / n_conf, 5), f"eval_{h}_rec@gate": 0.6, f"eval_{h}_f1@gate": 0.5,
                    f"eval_{h}_ap2m": 0.4, f"eval_{h}_auroc_matched": 0.9, f"eval_{h}_auroc_objectness": 0.91,
                    f"eval_{h}_n_ignore": 30.0, f"eval_{h}_n_windows": 128.0, f"eval_{h}_centre_err_p50": 0.8,
                    f"eval_{h}_cls_acc_tp": 0.7, f"eval_{h}_cls_acc_tp_priorcorr": 0.72,
                    f"eval_{h}_calib_pr_gate": 0.61, f"eval_{h}_calib_prec": 0.5, f"eval_{h}_calib_rec": 0.5,
                    f"eval_{h}_calib_n_pos": 900.0, f"eval_{h}_calib_n_windows": 256.0})
        for t in ("0p5", "1", "2", "4"):
            for b in ("all", "0_20", "20_40", "40_60"):
                row[f"eval_{h}_det_map{t}_{b}"] = 0.3
    return row


def _run_dir(tmp_path, *, last_step=5500, s_per_step=6.5, relaunch=False, trainer_alive=True,
             split_token=False, stderr=None, stderr_bytes=None, map_over=None, map_over_steps=None,
             drop_map_all=(), ratios=None, ga_last=None, config_extra=None, signal_last=None, name="watch"):
    d = tmp_path / name
    d.mkdir()
    rows, el = [], 0.0
    for step in range(50, last_step + 1, 50):
        if relaunch and step == 3050:
            el = 0.0                                     # a process restart resets elapsed_s
        el += 50 * s_per_step
        r = {"step": step, "loss": 80.0 - step / 200, "traj": 2.0 - step / 20000, "elapsed_s": el,
             "lr": 1e-4, "cuda_max_mem_gb": 30.1, "goal2s_err_m": 7.0 - step / 5000, "anchor_acc": 0.2,
             "tacv6_lat_ce": 1.0, "tacv6_lon_ce": 1.7, "tacv6_goal_bce": 0.9, "tacv6_goal_conf_bce": 0.3,
             "map_hires": 1.3, "box3d": 12.0, "agent_cls": 2.0, "agent_centre": 0.5, "nav_injected": 1.0,
             "cascade": 3.0, "trunk_frame_slots": 21600, "trunk_frames_computed": 10400}
        r.update({k: (100.0 if k.endswith("_n") else 1.5) for k in DECLARED_GA})
        for c in CLASSES:                                 # LOGGING_SPEC_MAP10 sec. 2, the train-row signal
            for b in BANDS:
                r.update({f"map_hires_lshare_{c}_{b}": 0.025, f"map_hires_n_{c}_{b}": 1000.0,
                          f"map_hires_gno_{c}_{b}": 1e-3})
        rows.append(r)
        if step % 500 == 0:
            e = {"step": step, "eval_loss": 40.0 - step / 1000, "eval_traj": 1.3 - step / 50000,
                 "eval_goal2s_err_m": 7.5 - step / 10000, "eval_anchor_acc": 0.4, "eval_tacv6_lat_ce": 0.96,
                 "eval_tacv6_lon_ce": 1.68, "eval_tacv6_goal_bce": 0.8, "eval_box3d": 11.1, "eval_agent_cls": 2.03,
                 "eval_agent_centre": 0.4, "eval_windows": 128, "eval_batches": 8}
            use = map_over if (map_over_steps is None or step in map_over_steps) else None
            e.update(_map_eval(use))
            for c, b in drop_map_all:
                e.pop(f"eval_map_hires_iou_{c}_{b}", None)
            e.update(_box_eval(ratios))
            rows.append(e)
        if step % 50 == 0:
            rows.append({"step": step + 1, "cd_cos": -0.05, "cd_stage_layer4_cos": 0.1, "cd_fuse_cos": 0.02,
                         "cd_stem_cos": -0.1})
    tr_last = [r for r in rows if "loss" in r][-1]
    for k, v in (ga_last or {}).items():
        if v == "DROP":
            tr_last.pop(k, None)
        else:
            tr_last[k] = v
    for c, how in (signal_last or {}).items():         # "dead": cells, no pull; "nocells": neither
        for b in BANDS:
            tr_last[f"map_hires_gno_{c}_{b}"] = 0.0
            if how == "nocells":
                tr_last[f"map_hires_n_{c}_{b}"] = 0.0
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg = {"argv": ["--trunk-name", "resnet101.a1_in1k", "--image-hw", "416", "1024", "--residual-prior", "ha0_ext_pose",
                    "--map-hires", "on", "--map-hires-x-max-m", "100", "--map-hires-y-half-m", "30",
                    "--map-hires-decision-rule", "prior_corrected", "--bev-source", "map_hires_pool",
                    "--graft-tac8-prior", "--graft-nav-compliance", "--speed-ceiling-filter",
                    "--eval-every", "500", "--save-every", "500", "--log-every", "50", "--batch", "16",
                    "--steps", "50400", "--out", "/home/nvidia/refcv7_run/runs/refcv7-r101-s0"],
           "seams": {"ego_history": {"enable": True, "steps": 8, "channels": 3},
                     "residual_prior": {"residual_prior": "ha0_ext_pose", "definition": "constant a0 and kappa0",
                                        "reads": ["pose window"], "vocabulary_space": "residual on P",
                                        "equals_battery_echo": False}},
           "trunk_memory_levers": {"chunk_ckpt": 8, "bn_folded": 104},
           "map_hires": {"decision_rule": "prior_corrected"},
           "grad_reach_logging": {"declared": True, "keys": DECLARED_GA, "log_every": 50,
                                  "source": "refcv6_perception_branch.grad_reach_report, read off the BUILT model"},
           "declared_vs_built": {"grad_unreachable": {
               "core.decoder.control_head": "F3: the cascade's per-layer heads emit the fan",
               "core.decoder.offset_head": "a sampler build: _sample REPLACES the classifier-pass fan bank + offset",
               "scorer.goal_point": "E9 always passes the structured goal"}}}
    cfg.update(config_extra or {})
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    n_launch = 2 if relaunch else 1
    sup = "".join(f"[sup:{ARM}] 2026-09-27T1{i}:00:00Z lock acquired on /x/{ARM}.lock (fd 200); supervisor pid {100 + i}\n"
                  f"ZZGATEOK-{ARM}-{i + 1}ZZ\n[sup:{ARM}] 2026-09-27T1{i}:00:05Z launch #{i + 1} of 4: python gate exec\n"
                  for i in range(n_launch))
    sup += (f"ZZ{ARM}-{last_step}-50400-0\n0-1ZZ\n" if split_token else f"ZZ{ARM}-{last_step}-50400-0-1ZZ\n")
    (d / "sup.log").write_text(sup, encoding="utf-8")
    if stderr is not None:
        (d / "stderr_tail.log").write_bytes(stderr.encode("utf-8"))
    sb = stderr_bytes if stderr_bytes is not None else len((stderr or "").encode("utf-8"))
    (d / "remote_state.json").write_text(json.dumps({
        "sup_pid": "101", "train_pid": "202", "sup_alive": "1", "train_alive": "1" if trainer_alive else "0",
        "stderr_bytes": str(sb), "ckpt": "1170622369 1790000000", "done": "0", "now": "1790000600",
        "run": "/home/nvidia/refcv7_run/runs/refcv7-r101-s0", "arm": ARM}), encoding="utf-8")
    return d


def _cell_html(page, cls, band):
    m = re.search(r'<td class="hm[^"]*" data-cell="%s %s"[^>]*>(.*?)</td>' % (cls, band), page)
    assert m, (cls, band)
    return m.group(0)


# ------------------------------------------------------------------ the healthy run ----
def test_a_healthy_refcv7_run_renders_every_section_and_stamps_T0(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert "<title>refcv7 Training Watch</title>" in page
    for cid in ("c1", "c2", "c3", "c4", "c5", "c6", "cmh", "c8", "c9", "c10", "c11", "c12",
                "b_box3d_ratio", "b_box3d_q", "b_agent_ratio", "b_agent_q") + tuple(f"m_{c}" for c in CLASSES):
        assert f'data-chart="{cid}"' in page, cid
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
        assert f"<td>{fam}</td>" in page, fam
    assert "<b>T0</b>" in page and "never a driving-performance claim" in page
    assert s["step"] == 5500 and s["total"] == 50400 and s["segments"] == 1 and s["unplanned"] == 0
    assert "0 unplanned deaths · 0 planned switch" in page
    assert s["map"]["state"] == "clear" and s["map"]["alarm"] is False and s["map"]["present_latest"] == 40
    assert s["map"]["breaches"] == [] and s["map"]["iou_count_mismatch"] == [] and s["map"]["amber_latest"] == []
    assert s["map"]["first_red_step"] is None and s["map"]["now_min_0_20"] == ["lane", 0.2]
    assert s["map"]["iou_count_checked"] == 40 and "agrees on all 40 keys that carry their counts" in page
    assert s["box"]["alarm"] is False and s["box"]["box3d"]["ratio"] == 1.0 and s["box"]["box3d"]["checks"] == []
    assert s["box"]["box3d"]["armed"] is True and s["box"]["box3d"]["state"] == "in band"
    assert "the battery reads it at milestones, never this page" in page            # A8 is not a Watch alarm
    assert s["ga"]["alarm"] is False and s["ga"]["absent"] == [] and s["ga"]["zero"] == []
    assert s["prior"]["mode"] == "ha0_ext_pose" and s["prior"]["keys"] == []
    assert s["gate"] == {"match": 1, "refused": []}
    # every one of the 40 contract keys is carried by the page, literally
    for c in CLASSES:
        for b in BANDS:
            assert f"eval_map_hires_iou_{c}_{b}" in page, (c, b)
    assert 'id="thin-class-alarm"' in page and "Thin-class alarm" in page


def test_the_Berlin_clock_follows_the_EU_rule_without_a_tz_database(monkeypatch):
    mod = _load(monkeypatch)
    utc = timezone.utc
    assert f"{mod.berlin(datetime(2026, 9, 27, 10, 0, tzinfo=utc)):%H:%M %Z}" == "12:00 CEST"
    assert f"{mod.berlin(datetime(2026, 10, 25, 0, 59, tzinfo=utc)):%H:%M %Z}" == "02:59 CEST"
    assert f"{mod.berlin(datetime(2026, 10, 25, 1, 0, tzinfo=utc)):%H:%M %Z}" == "02:00 CET"
    assert f"{mod.berlin(datetime(2027, 3, 28, 1, 0, tzinfo=utc)):%H:%M %Z}" == "03:00 CEST"


# ------------------------------------------------------------------ the 40-key map contract ----
def test_ONE_missing_map_key_renders_UNAVAILABLE_and_raises_the_alarm(tmp_path, monkeypatch):
    """RED arm: the latest eval row lacks `eval_map_hires_iou_lane_0_20` (earlier rows carry it)."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("lane", "0_20"): "DROP"}, map_over_steps={5500})
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["missing_latest"] == ["eval_map_hires_iou_lane_0_20"]
    assert s["map"]["present_latest"] == 39 and s["map"]["state"] == "UNAVAILABLE" and s["map"]["alarm"] is True
    assert s["map"]["rows_missing"] == [[5500, 1]]
    cell = _cell_html(page, "lane", "0_20")
    assert ">UNAVAILABLE</td>" in cell and "0.000" not in cell and "data-key" not in cell   # never drawn as 0
    assert "map 10 cm: 1 keys UNAVAILABLE" in page and 'class="alarm crit" id="thin-class-alarm"' in page


def test_a_key_missing_from_EVERY_row_leaves_no_literal_on_the_page(tmp_path, monkeypatch):
    """The launch gate reads the key literals as 'this series is carried'. A key the log never had must
    therefore never be printed -- not in a heatmap cell, not in the alarm tile, not in a chart."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, drop_map_all=[("crosswalk", "40_60")])
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert "eval_map_hires_iou_crosswalk_40_60" not in page
    assert "eval_map_hires_iou_crosswalk_20_40" in page                     # its neighbours still are
    assert s["map"]["alarm"] is True and len(s["map"]["rows_missing"]) == 11 and s["map"]["series_keys"] == 39


def test_all_40_keys_present_and_every_class_above_its_bar_is_clear(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["map"]["alarm"] is False and s["map"]["rows_missing"] == []
    assert 'class="alarm good" id="thin-class-alarm"' in page
    assert "map 10 cm: 40/40 keys · 0–20 m above 0.05" in page


def test_a_0_20_reading_at_or_below_0p05_from_step_5000_is_RED(tmp_path, monkeypatch):
    """The registered rule (LOGGING_SPEC_MAP10 5.3), and the current value shown beside the latched state."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("lane", "0_20"): 0.03}, map_over_steps={5500})
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["alarm"] is True and s["map"]["state"] == "ALARM" and s["map"]["first_red_step"] == 5500
    assert s["map"]["breaches"] == [[5500, "lane", "0_20", 0.03, 0.05]]
    assert "THIN-CLASS ALARM · first red at step 5,500" in page
    assert "RED (latched) · first red at step 5,500" in page and "RED now: lane · 0–20 m: IoU 0.030 ≤ 0.05" in page
    assert "now (step 5,500): lowest 0–20 m IoU lane 0.030 — at or below 0.05" in page
    assert 'class="alarm crit" id="thin-class-alarm"' in page
    assert "breach" in _cell_html(page, "lane", "0_20")


def test_the_registered_rule_starts_at_step_5000(tmp_path, monkeypatch):
    """Control: the same 0.03 at step 4,500 is before LOGGING_SPEC_MAP10's start step -- no alarm."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("lane", "0_20"): 0.03}, map_over_steps={4500})
    _, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["breaches"] == [] and s["map"]["alarm"] is False


def test_an_EARLIER_breach_stays_latched_and_shows_first_red(tmp_path, monkeypatch):
    """LOGGING_SPEC_MAP10 5.3: 'at ANY eval at or after step 5,000' -- a recovered class is still RED, and the
    page shows the current value beside the latched state."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("edge", "0_20"): 0.05}, map_over_steps={5000})
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["breaches"] == [[5000, "edge", "0_20", 0.05, 0.05]] and s["map"]["breaches_latest"] == []
    assert s["map"]["alarm"] is True and s["map"]["first_red_step"] == 5000
    assert "first red at step 5,000: edge 0–20 m IoU 0.050 ≤ 0.05" in page
    assert "now (step 5,500): lowest 0–20 m IoU lane 0.200 — above 0.05" in page


def test_ANY_class_at_0_20_is_RED_while_20_60_m_is_AMBER_only(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("drivable", "0_20"): 0.04, ("sidewalk", "20_40"): 0.01},
                 map_over_steps={5500})
    _, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["breaches"] == [[5500, "drivable", "0_20", 0.04, 0.05]]
    assert s["map"]["amber_latest"] == [[5500, "sidewalk", "20_40", 0.01, 0.05]]


def test_20_60_m_at_or_below_0p05_is_AMBER_informative_and_not_red(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("edge", "20_40"): 0.03, ("lane", "40_60"): 0.05}, map_over_steps={5500})
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["alarm"] is False and s["map"]["breaches"] == [] and s["map"]["state"] == "AMBER"
    assert s["map"]["amber_latest"] == [[5500, "lane", "40_60", 0.05, 0.05], [5500, "edge", "20_40", 0.03, 0.05]]
    assert 'class="alarm warn" id="thin-class-alarm"' in page
    assert '<span class="chip warn"><i></i>map 10 cm: 20–60 m ≤ 0.05 on 2 cell(s), informative</span>' in page
    cell = _cell_html(page, "edge", "20_40")
    assert 'class="hm hb0 amber"' in cell and "~ " in cell


def test_beyond_60_m_is_VALUES_ONLY(tmp_path, monkeypatch):
    """The Master Mind's ruling: refcv6 predicted nothing there and A7 expects far bands low -- no status, no fill,
    whatever the value. An undefined far cell is plain too; it is still never drawn as 0."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("arrow", "80_100"): 0.0, ("hatched", "60_80"): None,
                                     ("crosswalk", "60_80"): 0.004}, map_over_steps={5500})
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["alarm"] is False and s["map"]["breaches"] == [] and s["map"]["amber_latest"] == []
    cell = _cell_html(page, "arrow", "80_100")
    assert cell.startswith('<td class="hm plain"') and ">0.000<" in cell
    assert "breach" not in cell and "amber" not in cell
    cell = _cell_html(page, "hatched", "60_80")
    assert cell.startswith('<td class="hm plain"') and "undefined" in cell and "0.000" not in cell


def test_the_registered_rule_reads_the_IoU_as_logged_with_no_ground_truth_condition(tmp_path, monkeypatch):
    """A class predicted at 0-20 m where it has no ground truth logs IoU 0.0: the registered rule is literal."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, map_over={("crosswalk", "0_20"): "ZERO_NOGT"}, map_over_steps={5500})
    _, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["breaches"] == [[5500, "crosswalk", "0_20", 0.0, 0.05]] and s["map"]["alarm"] is True


def test_an_IoU_that_does_not_reproduce_from_its_own_counts_is_flagged(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    last_eval = [r for r in rows if "eval_loss" in r][-1]
    last_eval["eval_map_hires_inter_lane_0_20"] = 10.0          # 10 / 200 = 0.05, the row says 0.2
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["iou_count_mismatch"] == [["lane", "0_20", 0.2, 0.05]]
    assert "DISAGREES on 1 key(s): lane 0–20 m" in page


def test_a_class_with_cells_and_NO_pull_raises_the_signal_alarm(tmp_path, monkeypatch):
    """LOGGING_SPEC_MAP10 5 item 2: gno = 0 on a class's own labelled cells = the class gets no signal."""
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, signal_last={"arrow": "dead"})), arm=ARM)
    assert s["map"]["signal"] == {"state": "ALARM", "alarm": True, "dead_latest": ["arrow"], "missing_gno": []}
    assert "MAP SIGNAL: no pull on arrow" in page and "no signal</span>" in page


def test_a_class_with_NO_cells_and_no_pull_is_no_evidence(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, signal_last={"arrow": "nocells"})), arm=ARM)
    assert s["map"]["signal"]["alarm"] is False and s["map"]["signal"]["dead_latest"] == []
    assert "no cells in the last row" in page
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, name="w2")), arm=ARM)
    assert s["map"]["signal"] == {"state": "clear", "alarm": False, "dead_latest": [], "missing_gno": []}


# ------------------------------------------------------------------ the box heads (A10) ----
def test_a_box_ratio_of_3p4_raises_the_A10_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, ratios={"box3d": 3.4})), arm=ARM)
    assert s["box"]["box3d"]["ratio"] == 3.4 and s["box"]["box3d"]["alarm"] is True
    assert s["box"]["box3d"]["state"] == "ALARM" and s["box"]["alarm"] is True
    assert s["box"]["agent"]["alarm"] is False                             # the other head is in band
    assert "box3d RATIO ALARM 3.40" in page


def test_the_box_ratio_is_WARMING_UP_before_step_5000_and_armed_after(tmp_path, monkeypatch):
    """The Master Mind's ruling: armed from step 5,000; before it the tile is grey with the value shown. The
    trainer flags the band at every eval, and its flag agrees with this page's BAND reading (no false check)."""
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, last_step=4500, ratios={"box3d": 0.02})), arm=ARM)
    b = s["box"]["box3d"]
    assert b["state"] == "warming up (presence prior 0.01)" and b["alarm"] is False and b["armed"] is False
    assert b["ratio"] == 0.02 and b["checks"] == [] and s["box"]["alarm"] is False
    assert '<span class="chip info"><i></i>box3d warming up · ratio 0.02</span>' in page
    assert "0.02 · warming up (presence prior 0.01)" in page
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, last_step=5500, ratios={"box3d": 0.02}, name="w2")), arm=ARM)
    assert s["box"]["box3d"]["alarm"] is True and s["box"]["box3d"]["armed"] is True


def test_a_box_ratio_of_1p0_raises_nothing_and_the_band_edges_are_inclusive(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, ratios={"box3d": 1.0, "agent": 1.5})), arm=ARM)
    assert s["box"]["alarm"] is False and s["box"]["agent"]["ratio"] == 1.5
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, ratios={"box3d": 0.5, "agent": 0.49}, name="w2")), arm=ARM)
    assert s["box"]["box3d"]["alarm"] is False and s["box"]["agent"]["alarm"] is True


def test_a_ratio_that_does_not_reproduce_from_the_counts_is_flagged(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    last_eval = [r for r in rows if "eval_loss" in r][-1]
    last_eval["eval_agent_conf_ratio"] = 3.4                    # n_conf / n_pos is still 1.0
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["box"]["agent"]["alarm"] is True
    assert any("does not reproduce from n_conf / n_pos = 1.00000" in c for c in s["box"]["agent"]["checks"])
    assert any("they disagree" in c for c in s["box"]["agent"]["checks"])   # the trainer's flag said 0


def test_absent_P0_keys_are_UNAVAILABLE_and_an_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [{k: v for k, v in r.items() if not k.startswith(("eval_box3d_", "eval_agent_"))} for r in rows]
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["box"]["box3d"]["state"] == "UNAVAILABLE" and s["box"]["box3d"]["alarm"] is True
    assert "box3d ratio UNAVAILABLE" in page and "no eval row carries the P0 ratio" in page


# ------------------------------------------------------------------ the declared gradient reach (D3) ----
def test_a_declared_ga_mh_refine_that_is_ABSENT_raises_the_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, ga_last={"ga_mh_refine": "DROP"})), arm=ARM)
    assert s["ga"]["alarm"] is True and s["ga"]["absent"] == ["ga_mh_refine"] and s["ga"]["zero"] == []
    assert "GRAD REACH ALARM: 1 dead or absent" in page


def test_a_declared_ga_mh_refine_that_is_exactly_ZERO_raises_the_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, ga_last={"ga_mh_refine": 0.0, "ga_mh_refine_n": 0})),
                     arm=ARM)
    assert s["ga"]["alarm"] is True and s["ga"]["zero"] == ["ga_mh_refine", "ga_mh_refine_n"]


def test_an_UNDECLARED_zero_is_information_not_an_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, ga_last={"ga_extra_head": 0.0})), arm=ARM)
    assert s["ga"]["alarm"] is False and s["ga"]["undeclared"] == ["ga_extra_head"]


def test_the_grad_unreachable_declaration_is_shown_as_information(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["ga"]["grad_unreachable"] == ["core.decoder.control_head", "core.decoder.offset_head",
                                           "scorer.goal_point"]
    assert "Declared grad-unreachable (information, not an alarm)" in page and s["ga"]["alarm"] is False


def test_a_config_without_the_D3_declaration_is_UNAVAILABLE_and_an_alarm(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    del cfg["grad_reach_logging"]
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["ga"]["declared"] is None and s["ga"]["alarm"] is True
    assert "grad reach declaration UNAVAILABLE" in page


# ------------------------------------------------------------------ pace ----
def test_pace_above_the_PI_line_is_shown_and_not_decided(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, s_per_step=8.5)), arm=ARM)
    assert s["pace_s"] == 8.5 and s["pace_above_pi_line"] is True and s["pace_warm_median_s"] == 8.5
    assert ('<span class="chip warn"><i></i>pace 8.50 s/step (marginal, incl. eval + ckpt) &gt; 8.0 · warm median '
            '8.50</span>') in page
    assert "the PI rule itself is applied at the G-LIVE smoke" in page
    mod = _load(monkeypatch)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, s_per_step=6.5, name="w2")), arm=ARM)
    assert s["pace_s"] == 6.5 and s["pace_above_pi_line"] is False


# ------------------------------------------------------------------ the launch gate's form ----
def _gate_reads(html):
    """launch_gate.map_watch_reasons, written out: the 24 keys of the gate's profile (3 bands) as text,
    and the alarm-tile regex."""
    want = [f"eval_map_hires_iou_{c}_{b}" for c in CLASSES for b in ("0_20", "20_40", "40_60")]
    return [k for k in want if k not in html], re.search(r"thin[-_ ]?class[-_ ]?alarm", html, re.I) is not None


def test_the_gate_form_builds_from_ONE_metrics_file_and_carries_the_contract(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    for f in ("sup.log", "remote_state.json"):
        (d / f).unlink()
    out = tmp_path / "gate_out" / "watch_refcv7.html"
    out.parent.mkdir()
    assert mod.main(["--metrics", str(d / "metrics.jsonl"), "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    missing, alarm = _gate_reads(html)
    assert missing == [] and alarm is True
    assert "no run state pulled" in html and "supervisor log UNAVAILABLE" in html
    s = json.loads((tmp_path / "gate_out" / "watch_refcv7.summary.json").read_text(encoding="utf-8"))
    assert s["state_read"] is False and s["map"]["present_latest"] == 40


def test_RED_the_gate_form_on_a_drivable_only_log_FAILS_the_gates_reading(tmp_path, monkeypatch):
    """The refcv6 pattern (LOGGING_SPEC_MAP10 RL1): only drivable logged. The page must NOT print the 21
    absent keys anywhere, so the gate's text check fails as it must -- the alarm tile is still there."""
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path, drop_map_all=[(c, b) for c in CLASSES if c != "drivable" for b in BANDS])
    out = tmp_path / "w.html"
    mod.main(["--metrics", str(d / "metrics.jsonl"), "--out", str(out)])
    missing, alarm = _gate_reads(out.read_text(encoding="utf-8"))
    assert len(missing) == 21 and alarm is True


# ------------------------------------------------------------------ back-compat: a refcv6-shaped log ----
def test_a_refcv6_shaped_log_renders_its_shared_keys_and_the_refcv7_sections_UNAVAILABLE(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    rows = [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    keep = []
    for r in rows:
        r = {k: v for k, v in r.items() if not (k.startswith(("eval_map_hires", "map_hires", "ga_", "eval_box3d_",
                                                                 "eval_agent_")))}
        if "loss" in r and "eval_loss" not in r:
            r.update({"map": 1.0, "map_iou_drivable": 0.55})
        keep.append(r)
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    for k in ("grad_reach_logging", "map_hires", "declared_vs_built"):
        cfg.pop(k)
    cfg["seams"].pop("residual_prior")
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["map"]["state"] == "UNAVAILABLE" and s["map"]["series_keys"] == 0
    assert s["box"]["box3d"]["state"] == "UNAVAILABLE" and s["ga"]["declared"] is None
    assert s["prior"]["mode"] is None
    assert "eval_map_hires_iou_" not in page                                  # no contract key is claimed
    assert "NOT the refcv7 map" in page                                       # the 0.5 m keys are named, not drawn
    # a cross-check that read nothing says so -- it never reads as agreement
    assert s["map"]["iou_count_checked"] == 0
    assert "the row's own inter / union counts could not run" in page and "agrees on" not in page
    assert s["map"]["signal"]["state"] == "UNAVAILABLE" and s["map"]["signal"]["alarm"] is True
    for cid in ("c1", "c2", "c3", "c4", "c5", "c6", "c8", "c9", "c10", "c11", "c12"):
        m = re.search(r'<figure class="chart"><figcaption><b>[^<]*</b><span>[^<]*</span></figcaption>'
                      r'<span class="ylab">[^<]*</span><svg class="plot" viewBox="0 0 560 260" data-chart="%s"' % cid,
                      page)
        assert m, cid                                                          # a drawn chart, not "No data yet"


# ------------------------------------------------------------------ the run dir, read not guessed ----
def test_the_run_dir_is_read_from_the_gated_argv_and_env_wins(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    tree = tmp_path / "tree"
    (tree / "stack" / "ops" / "runs.d").mkdir(parents=True)
    (tree / "stack" / "ops" / "runs.d" / f"{ARM}.argv.json").write_text(json.dumps(
        {"arm": ARM, "argv": ["--steps", "50400", "--out", "/home/nvidia/refcv7_run/runs/refcv7-r101-s0"]}),
        encoding="utf-8")
    monkeypatch.setattr(mod, "TREE", str(tree))
    assert mod.resolve_run(ARM) == ("/home/nvidia/refcv7_run/runs/refcv7-r101-s0",
                                    f"--out of stack/ops/runs.d/{ARM}.argv.json")
    monkeypatch.setenv("REFCV7_RUN", "/elsewhere")
    assert mod.resolve_run(ARM) == ("/elsewhere", "$REFCV7_RUN")
    assert mod.resolve_run(ARM, "/cli") == ("/cli", "--run")
    monkeypatch.delenv("REFCV7_RUN")
    monkeypatch.setattr(mod, "TREE", str(tmp_path / "empty"))
    run, why = mod.resolve_run(ARM)
    assert run is None and "no --run, no $REFCV7_RUN" in why


def test_an_unsafe_run_dir_is_refused_before_any_ssh(monkeypatch):
    mod = _load(monkeypatch)
    with pytest.raises(SystemExit, match="unsafe run dir"):
        mod.pull("/home/nvidia/x; rm -rf /", ARM, "unused")


# ------------------------------------------------------------------ the refcv6 health checks, kept ----
DIAGNOSED = ("W0927 09:03:45.814000 3346338 torch/_inductor/utils.py:1953] [0/2] Not enough SMs to use "
             "max_autotune_gemm mode")


def test_an_UNPLANNED_relaunch_is_reported_as_a_death(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, relaunch=True)), arm=ARM)
    assert s["segments"] == 2 and s["unplanned"] == 1
    assert "1 unplanned deaths" in page and "UNPLANNED relaunch" in page


def test_a_dead_trainer_is_reported_NOT_RUNNING(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, trainer_alive=False)), arm=ARM)
    assert s["train_alive"] is False and "NOT RUNNING" in page


def test_a_SPLIT_supervisor_token_reads_zero_tracebacks_not_the_step_count(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, split_token=True)), arm=ARM)
    assert s["n_err"] == 0 and s["token_split"] is True and s["token_ok"] is True


def test_an_UNDIAGNOSED_stderr_line_fails_and_a_traceback_is_counted_from_the_content(tmp_path, monkeypatch):
    err = DIAGNOSED + "\nTrace" "back (most recent call last):\n  File \"x.py\", line 1\nRuntimeError: boom\n"
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, stderr=err)), arm=ARM)
    assert s["stderr_lines"] == 4 and s["stderr_undiagnosed"] == 4 and s["n_err_client"] == 1
    assert "stderr: 4 UNDIAGNOSED line(s)" in page
    mod = _load(monkeypatch)
    mod.FACTS["stderr_diagnosed"] = {DIAGNOSED: "benign (test)"}
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path, stderr=DIAGNOSED + "\n", name="w2")), arm=ARM)
    assert s["stderr_undiagnosed"] == 0 and s["stderr_read_ok"] is True


def test_a_stderr_that_was_NOT_READ_fails(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path, stderr="", stderr_bytes=112)), arm=ARM)
    assert s["stderr_read_ok"] is False and "stderr NOT READ" in page


def test_a_REFUSED_gate_is_reported(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    d = _run_dir(tmp_path)
    (d / "sup.log").write_text((d / "sup.log").read_text(encoding="utf-8") + f"ZZGATEREFUSED-{ARM}-2ZZ\n",
                               encoding="utf-8")
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["gate"] == {"match": 1, "refused": [2]} and "gate REFUSED launch #2" in page


def test_a_run_STOPPED_by_the_PI_reads_stopped_not_NOT_RUNNING(tmp_path, monkeypatch):
    d = _run_dir(tmp_path, trainer_alive=False)
    st = json.loads((d / "remote_state.json").read_text(encoding="utf-8"))
    st["sup_alive"], st["stopped"] = "0", "1"
    (d / "remote_state.json").write_text(json.dumps(st), encoding="utf-8")
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(d), arm=ARM)
    assert s["stopped"] is True and "stopped by the PI" in page and "NOT RUNNING" not in page
    assert "finish ≈" not in page


# ------------------------------------------------------------------ NavSim count guard (refcv6's, kept) ----
def _eval_pkg(tmp_path):
    pkg, live = tmp_path / "evalpkg", tmp_path / "evallive"
    m5 = pkg / "navsim" / "raw" / "milestones" / "step5000"
    m5.mkdir(parents=True)
    arms_nt = {"R7_A1": {"NC": 80.0, "DAC": 75.0, "TTC": 65.0, "EP": 50.0, "C": 99.9, "DDC": 90.0, "PDMS": 52.25,
                         "interval": {"lo": 0.5, "hi": 0.545}},
               "STOP": {"PDMS": 61.8202}, "CV": {"PDMS": 20.6517}, "HUMAN": {"PDMS": 94.5514}}
    (m5 / "summary_navtest.json").write_text(json.dumps({"n_tokens": 12146, "arms": arms_nt, "pairs": {
        "R7_A1__minus__STOP": {"interval": {"delta": -0.0957, "lo": -0.12, "hi": -0.07}}}}), encoding="utf-8")
    (m5 / "summary_navhard.json").write_text(json.dumps({"arms": {
        "R7_A1": {"official_two_stage_EPDMS": 0.2012, "n_stage1": 450, "n_stage2": 5462},
        "STOP_zero": {"official_two_stage_EPDMS": 0.2985}}}), encoding="utf-8")
    (m5 / "summary_warmup.json").write_text(json.dumps({"arms": {"R7_A1": {"n_stage1": 16, "n_stage2": 204}}}),
                                            encoding="utf-8")
    (m5 / "BARS.json").write_text(json.dumps({"bars": {
        "navtest": {"verdict": "FAIL"}, "navhard": {"verdict": "FAIL", "paired_vs_STOP": {"delta": -0.1, "lo": -0.12, "hi": -0.08}},
        "warmup": {"verdict": "FAIL", "values": {"R7_A1": 0.41, "STOP_zero": 0.5212}, "margin": -0.11,
                   "seed_floor": 0.0157, "interval": "UNAVAILABLE (7 logs < 8)"}}}), encoding="utf-8")
    live.mkdir()
    return pkg, live, m5


def test_navsim_kpis_are_read_from_the_banked_milestone(tmp_path, monkeypatch):
    pkg, live, _ = _eval_pkg(tmp_path)
    mod = _load(monkeypatch, eval_pkg=pkg, eval_live=live)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"] == {"5000": {"navtest": 52.25, "navhard": 0.2012, "warmup": 0.41}}
    assert "<b>52.25</b> [50.00, 54.50]" in page and "vs STOP -9.57 [-12.00, -7.00]" in page


def test_a_PARTIAL_full_split_navtest_is_REFUSED_not_shown(tmp_path, monkeypatch):
    pkg, live, m5 = _eval_pkg(tmp_path)
    s_ = json.loads((m5 / "summary_navtest.json").read_text(encoding="utf-8"))
    s_["n_tokens"] = 1464
    (m5 / "summary_navtest.json").write_text(json.dumps(s_), encoding="utf-8")
    mod = _load(monkeypatch, eval_pkg=pkg, eval_live=live)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"]["5000"]["navtest"] == "UNAVAILABLE — count guard FAIL (1,464 of 12,146)"
    assert "52.25" not in page


def test_navhard_and_warmup_with_the_wrong_stage_counts_are_REFUSED(tmp_path, monkeypatch):
    pkg, live, m5 = _eval_pkg(tmp_path)
    h = json.loads((m5 / "summary_navhard.json").read_text(encoding="utf-8"))
    h["arms"]["R7_A1"]["n_stage2"] = 2517
    (m5 / "summary_navhard.json").write_text(json.dumps(h), encoding="utf-8")
    (m5 / "summary_warmup.json").write_text(json.dumps({"arms": {"R7_A1": {"n_stage2": 204}}}), encoding="utf-8")
    mod = _load(monkeypatch, eval_pkg=pkg, eval_live=live)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"]["5000"]["navhard"] == "UNAVAILABLE — count guard FAIL (n_stage2 2,517 of 5,462)"
    assert s["navsim"]["5000"]["warmup"] == "UNAVAILABLE — count guard FAIL (n_stage1 missing of 16)"
    assert "<b>0.2012</b>" not in page


def test_a_WRONG_arm_id_is_refused_by_the_count_guard_never_read_as_a_number(tmp_path, monkeypatch):
    """If the banked arm is not the one this page reads, the stage counts read missing and the split is
    REFUSED -- loud, never a wrong number."""
    pkg, live, _ = _eval_pkg(tmp_path)
    monkeypatch.setenv("REFCV7_NAVSIM_ARM", "R7_B9")
    mod = _load(monkeypatch, eval_pkg=pkg, eval_live=live, navsim_arm="R7_B9")
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"]["5000"]["navhard"] == ("UNAVAILABLE — count guard FAIL (n_stage1 missing of 450; "
                                              "n_stage2 missing of 5,462)")


def test_no_eval_package_shows_nothing_rather_than_a_guess(tmp_path, monkeypatch):
    mod = _load(monkeypatch)
    page, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"] == {} and s["battery_steps"] == []
    assert "No banked NavSim milestone was readable under" in page and "nothing is shown rather than a guess" in page


def test_an_unbanked_split_reads_NOT_BANKED_without_a_live_lane(tmp_path, monkeypatch):
    """Without $REFCV7_EVAL_LIVE the page cannot tell running from not run, and says so."""
    pkg, _live, m5 = _eval_pkg(tmp_path)
    bars = json.loads((m5 / "BARS.json").read_text(encoding="utf-8"))
    bars["bars"]["navhard"] = {"status": "UNAVAILABLE"}
    (m5 / "BARS.json").write_text(json.dumps(bars), encoding="utf-8")
    mod = _load(monkeypatch, eval_pkg=pkg)
    _, s = mod.build(watch_dir=str(_run_dir(tmp_path)), arm=ARM)
    assert s["navsim"]["5000"]["navhard"] == "not banked"


def test_the_EvalFlyWheel_defaults_are_the_Master_Minds_ruling(monkeypatch):
    for k in ("REFCV7_EVAL_PKG", "REFCV7_EVAL_LIVE", "REFCV7_NAVSIM_ARM"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("build_watch_refcv7_defaults", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.EVAL_PKG == (r"D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming"
                            r"\2026-09-27-refcv7-standard-tests")
    assert mod.NAVSIM_ARM == "R7_A1" and mod.EVAL_LIVE is None
