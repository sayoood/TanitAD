"""test_refcv8_det_shipped_pin.py -- P0: pins the SHIPPED refcv8 box post-processing files (EMA-48,852) with LITERAL values.

The two files are the TRAIN-fitted outputs of ``stack/scripts/refit_perception_thresholds.py`` on the refcv8 EMA-48,852 checkpoint (perc8, 2026-10-21).
They are OPT-IN readout configs: ``refc_v3_train.py --det-presence-gates / --det-nms`` (in-run eval census) and the post-hoc box-read scripts. They NEVER reach the
planner or a loss (pinned for the machinery by ``test_refcv8_det_nms.py`` / ``test_refcv7_det_presence_gates.py``). A refcv8 checkpoint must not be read at refcv7's
files and vice versa: the checkpoint md5 each file is bound to is pinned here, and a deliberate-regression arm shows the pin goes RED on the wrong file.
Expected values are LITERALS taken from the artifacts (perc8 ``raw/T_nosync/*.json``), never an expression over the code under test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tanitad.eval import detection_metrics as D
from tanitad.eval import detection_nms as N

CFG = Path(N.__file__).resolve().parent.parent / "configs"
NMS8, GATES8 = CFG / "refcv8_det_nms_train.json", CFG / "refcv8_det_presence_gates_train.json"
NMS7, GATES7 = CFG / "refcv7_det_nms_train.json", CFG / "refcv7_det_presence_gates_train.json"
CKPT8_MD5 = "b4364029b68f29a33eeb4aebbbbd002b"          # E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt
NMS8_MD5, GATES8_MD5 = "375a0014bf145a49cb87a01bb274738b", "c7df00ba6439a944ab0d80a651c1c5ba"
NMS7_MD5, GATES7_MD5 = "fa76f8ac9503bcafbbec3b2aba466aaf", "30a5cfe9108c0adcf84f7dbc4d1605e4"   # the refcv7 files stay byte-identical


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def test_P0_files_are_the_perc8_outputs_byte_for_byte():
    assert _md5(NMS8) == NMS8_MD5 and _md5(GATES8) == GATES8_MD5


def test_P0_the_refcv7_files_were_not_touched():
    assert _md5(NMS7) == NMS7_MD5 and _md5(GATES7) == GATES7_MD5


def test_P0_nms_file_values_and_provenance():
    cfg, stamp = N.load_head_nms(NMS8)                  # the CONSUMER's own loader accepts it
    assert cfg["box3d"]["radius_m"] == 2.5 and cfg["agent"]["radius_m"] == 3.0
    assert round(cfg["box3d"]["gate"], 4) == 0.2165 and round(cfg["agent"]["gate"], 4) == 0.1844
    assert all(v["p_floor"] == 0.05 for v in cfg.values())
    d = json.loads(NMS8.read_text(encoding="utf-8"))
    pv = d["provenance"]
    assert d["schema"] == N.NMS_SCHEMA == "tanitad.det_nms/1"
    assert pv["fit_split"] == "TRAIN" and pv["checkpoint_step"] == 48852 and pv["run"] == "refcv8-r101-s0-ema48852"
    assert pv["checkpoint_md5"] == CKPT8_MD5 and pv["n_windows"] == 1074
    assert stamp["sha256"] == hashlib.sha256(NMS8.read_bytes()).hexdigest()
    g = pv["train_grid"]["box3d"]                       # the choice is auditable: 2.5 m is the TRAIN-AP@2m argmax
    assert round(g["2.5"]["train_ap2m"], 4) == 0.4722 and round(g["none"]["train_ap2m"], 4) == 0.3462
    assert g["2.5"]["train_ap2m"] > max(g[r]["train_ap2m"] for r in g if r != "2.5")


def test_P0_gates_file_values_and_provenance():
    gates, stamp = D.load_head_gates(GATES8)
    assert round(gates["box3d"], 4) == 0.2660 and round(gates["agent"], 4) == 0.2394
    d = json.loads(GATES8.read_text(encoding="utf-8"))
    pv = d["provenance"]
    assert d["schema"] == D.GATES_SCHEMA == "tanitad.det_presence_gates/1"
    assert pv["fit_split"] == "TRAIN" and pv["checkpoint_step"] == 48852 and pv["checkpoint_md5"] == CKPT8_MD5
    assert pv["run"] == "refcv8-r101-s0-ema48852" and pv["train_n_pos"]["box3d"] == 3164
    assert stamp["sha256"] == hashlib.sha256(GATES8.read_bytes()).hexdigest()


def test_P0_DELIBERATE_REGRESSION_the_refcv7_files_are_not_refcv8s():
    """If someone ships refcv7's file under refcv8's name, the checkpoint pin above must go RED: show the discriminating fact."""
    assert json.loads(NMS7.read_text(encoding="utf-8"))["provenance"]["checkpoint_md5"] != CKPT8_MD5
    assert json.loads(GATES7.read_text(encoding="utf-8"))["provenance"]["checkpoint_md5"] != CKPT8_MD5
    assert _md5(NMS7) != NMS8_MD5 and _md5(GATES7) != GATES8_MD5


def test_P0_files_are_not_imported_by_anything_planner_side():
    root = Path(N.__file__).resolve().parent.parent
    for sub in ("models", "refs"):
        for f in (root / sub).rglob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="ignore")
            assert "refcv8_det_nms_train" not in txt and "refcv8_det_presence_gates_train" not in txt, f
