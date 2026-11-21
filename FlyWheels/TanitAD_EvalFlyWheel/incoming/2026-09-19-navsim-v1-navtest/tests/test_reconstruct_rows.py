"""The aggregation-crash recovery: the streamed rows rebuild the devkit's own CSV, bit for bit.

    PYTHONPATH=<v1.1 tree> C:/Users/Admin/navsim-crun/venv/Scripts/python.exe -m pytest -q tests/test_reconstruct_rows.py

MEASURED on navhard 2026-09-19: the official v2 runner scored all 5,912 scenarios and then DIED
in aggregation — the CSV is written only after it, so the whole compute was lost. W3 streams every
per-token row to JSONL as ``pdm_score`` returns it. This test proves the recovery is the SAME
NUMBERS as the devkit's CSV on a real run (the 20-token smoke), and that a recovery from nothing
is refused rather than producing an empty "success".
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("w3_run_v1_rec", PKG / "code" / "run_v1.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["w3_run_v1_rec"] = R
_spec.loader.exec_module(R)

LABEL = "CV_smoke20"
CSV = PKG / "raw" / LABEL / f"{LABEL}.csv"
JSONL = PKG / "raw" / LABEL / f"{LABEL}_rows.jsonl"


@pytest.mark.skipif(not (CSV.exists() and JSONL.exists()), reason="the 20-token smoke is absent")
def test_reconstruction_equals_the_devkit_csv(tmp_path):
    dst = tmp_path / "rec.csv"
    n = R.reconstruct_csv(str(JSONL), str(dst))
    assert n == 20
    want = {r["token"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if r["token"] != "average"}
    got = {r["token"]: r for r in csv.DictReader(open(dst, encoding="utf-8"))
           if r["token"] != "average"}
    assert set(got) == set(want) and len(got) == 20
    for t in want:
        for c in R.TERMS:
            assert float(got[t][c]) == float(want[t][c]), f"{t}.{c}"
    # the rebuilt average row is the mean of the rebuilt rows, and matches the devkit's
    avg_w = next(r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["token"] == "average")
    avg_g = next(r for r in csv.DictReader(open(dst, encoding="utf-8")) if r["token"] == "average")
    for c in R.TERMS:
        assert abs(float(avg_g[c]) - float(avg_w[c])) < 1e-12, c


def test_reconstruction_of_nothing_is_refused(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert R.reconstruct_csv(str(empty), str(tmp_path / "x.csv")) == 0
    assert not (tmp_path / "x.csv").exists()


def test_hook_error_lines_are_not_rows(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"hook_error": "boom"}\n'
                 '{"token": "abc", "row": {"no_at_fault_collisions": 1.0, '
                 '"drivable_area_compliance": 1.0, "ego_progress": 0.5, '
                 '"time_to_collision_within_bound": 1.0, "comfort": 0.0, '
                 '"driving_direction_compliance": 1.0, "score": 0.625}}\n', encoding="utf-8")
    dst = tmp_path / "rec.csv"
    assert R.reconstruct_csv(str(p), str(dst)) == 1
    rows = [r for r in csv.DictReader(open(dst, encoding="utf-8")) if r["token"] != "average"]
    assert len(rows) == 1 and rows[0]["token"] == "abc"
    assert R.formula(rows[0]) == 0.625 == float(rows[0]["score"])
