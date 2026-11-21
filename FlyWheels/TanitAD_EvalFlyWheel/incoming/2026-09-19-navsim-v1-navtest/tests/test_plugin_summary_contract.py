"""W3's ``navsim_v1`` plugin against W1's run-dir CONTRACT, without a 12,146-token run.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_plugin_summary_contract.py

``assemble_summary`` is a pure function of the scored CSVs, so the 20-token smoke run is enough
to prove the summary this plugin writes VALIDATES — schema + the cross-field rules
(``taniteval.bench.contract.validate_summary``: every floor is an arm, every arm is paired
against every floor, the headline column is never ``pdm_score``, STOP and CV mandatory).
Each guard carries a deliberate-regression arm that must go RED.
"""
import json
import os
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[3]
sys.path[:0] = [str(REPO / "taniteval")]

from taniteval.bench import contract                                     # noqa: E402
from taniteval.bench.plugins import navsim_v1 as P                       # noqa: E402

ARMS = ("CV", "HUMAN", "STOP")
CLUSTERS = json.load(open(REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                          "2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json",
                          encoding="utf-8"))["token_to_log_name"]


def _smoke():
    tok, avg, counts = {}, {}, {}
    for a in ARMS:
        lab = f"{a}_smoke20"
        csv = PKG / "raw" / lab / f"{lab}.csv"
        if not csv.exists():
            pytest.skip(f"{csv} absent — run the 20-token smoke first")
        t, v = P._rows(csv)
        tok[a], avg[a] = t, v
        counts[a] = json.load(open(PKG / "raw" / lab / f"{lab}.counts.json", encoding="utf-8"))
    return tok, avg, counts


def _summary(**over):
    tok, avg, counts = _smoke()
    kw = dict(run_id="20260920T000000Z-navsim_v1-none-abcdef",
              stamps={"tier": "T1-family", "loop": {"single_stage": "OPEN"},
                      "evidence_class": "MEASURED", "closed_loop": False},
              known=list(ARMS), model_arms=[], tok_rows=tok, avg_rows=avg, counts=counts,
              clusters=CLUSTERS, cache=str(P.EXP / "metric_cache_smoke20"))
    kw.update(over)
    return P.assemble_summary(**kw)


def test_summary_validates_against_the_contract():
    errs = contract.validate_summary(_summary())
    assert errs == [], errs


def test_headline_is_the_score_column_and_floors_are_present():
    s = _summary()
    assert s["headline_metric"]["column"] == "score"
    assert sorted(s["floors"]) == ["CV", "STOP"]
    for arm in ARMS:
        assert s["arms"][arm]["headline"]["column"] == "score"
        assert set(s["arms"][arm]["paired"]) == {"CV", "STOP"}
        assert s["arms"][arm]["paired"].get(arm, {}).get("status") == "SELF" or arm == "HUMAN"
    assert s["protocol"] == "PDMS_v1_navtest"


def test_pdms_headline_equals_the_devkit_average_row():
    s = _summary()
    for arm in ARMS:
        blk = s["arms"][arm]
        assert abs(blk["headline"]["value"] - blk["controls"]["devkit_average_row_score"]) < 1e-9


def test_interval_is_refused_on_the_smoke_with_a_reason():
    """4 logs < the RG-14 floor of 8 ⇒ UNAVAILABLE with a reason — never a scene-token fallback."""
    s = _summary()
    iv = s["arms"]["CV"]["interval"]
    assert iv["status"] == "UNAVAILABLE" and iv.get("reason")
    assert "n_clusters" not in iv or iv["n_clusters"] < 8


def test_regression_dropping_the_stop_floor_goes_RED():
    s = _summary()
    del s["arms"]["STOP"]
    s["floors"] = ["CV"]
    errs = contract.validate_summary(s)
    assert any("STOP" in e for e in errs), errs


def test_regression_pdm_score_column_goes_RED():
    s = _summary()
    s["arms"]["CV"]["headline"]["column"] = "pdm_score"
    errs = contract.validate_summary(s)
    assert errs, "a pdm_score headline must be refused"
