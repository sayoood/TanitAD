"""tests of make_spread_gapfile.py (card PREREG_V8F_PROOF_ADD134). The worlds of the test of card ADD132 are reused. Run one file at a time.
GAPSPREAD_KIT names the folder of the kit of cards ADD132 and ADD133 (default: ../../gapspread/kit, or the pod path)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.environ.get("GAPSPREAD_KIT"), os.path.join(os.path.dirname(os.path.dirname(HERE)), "gapspread", "kit"), "/workspace/v8proof/gapspread/kit"):
    if cand and os.path.isfile(os.path.join(cand, "read_gapspreadb_t0.py")):
        KIT = cand
        break
else:
    raise RuntimeError("the kit of cards ADD132 and ADD133 was not found")
for p in (KIT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_read_gapspread_t0 as TA  # noqa: E402
import make_spread_gapfile as MS  # noqa: E402
from test_read_gapspread_t0 import fast_render, small_floor  # noqa: E402,F401

GS, MJ = MS.GS, MS.MJ


def world(tmp_path, capsys, monkeypatch, tag="g"):
    info, x, a, W = TA.marked_world(tmp_path, capsys, monkeypatch)
    x.out_dir = str(tmp_path / ("gapfiles_" + tag))
    k = GS.solve_k(W.s[W.has])
    g = GS.sp_gap(W.m, W.s, k)
    exp = {"k": float(k), "rows": int(len(W.rows)), "long": float(GS.long_part(g[W.rows], W.lab).mean()), "short": float(GS.short_part(g[W.rows], W.lab).mean())}
    return info, x, a, W, exp


def test_clean_world_passes_and_the_file_differs_in_the_gap_alone(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W, exp = world(tmp_path, capsys, monkeypatch)
    rec = MS.build(x, a, expect=exp)
    assert MS.print_record(rec) and capsys.readouterr().out.strip().splitlines()[-1] == "ZZ SPREAD-GAPFILE ALL-PASS"
    B, F = MJ.load(x.base), MJ.load(os.path.join(x.out_dir, "VGAPSP90.npz"))
    assert sorted(B) == sorted(F) and all(np.array_equal(B[k], F[k]) for k in B if k != "gap")
    has = B["has_out"].astype(bool)
    assert np.array_equal(np.isfinite(F["gap"]), has) and (F["gap"][has] < B["gap"][has]).any() and (F["gap"][has] <= B["gap"][has] + 1e-12).all()
    assert (F["gap"][has] >= GS.FLOOR * B["gap"][has] - 1e-12).all() and abs((F["gap"][has] / B["gap"][has]).mean() - GS.DOSE) < 1e-6
    rj = json.load(open(os.path.join(x.out_dir, "GAPSPREAD10K_GAPFILE_RECORD.json"), encoding="utf-8"))
    assert rj["card"] == "PREREG_V8F_PROOF_ADD134" and rj["pass_p0"] is True and rj["files"]["VGAPSP90"]["sha256"] == MS._sha(os.path.join(x.out_dir, "VGAPSP90.npz"))
    with pytest.raises(SystemExit):
        MS.build(x, a, expect=exp)


def test_p0_fails_on_a_planted_shift(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W, exp = world(tmp_path, capsys, monkeypatch)
    rec = MS.build(x, a, expect=exp, fault="shift")
    assert rec["pass_p0"] is False and "second routine" in rec["failures"][0]
    assert not MS.print_record(rec) and "ctl p0 FAIL" in capsys.readouterr().out


def test_p0_fails_against_the_values_of_the_card_in_a_small_world(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W, exp = world(tmp_path, capsys, monkeypatch)
    rec = MS.build(x, a)
    assert rec["pass_p0"] is False and any("row 201" in f for f in rec["failures"])


def test_p0_fails_for_another_hash_in_the_record_of_card_add106(tmp_path, capsys, fast_render, small_floor, monkeypatch):
    info, x, a, W, exp = world(tmp_path, capsys, monkeypatch)
    rj = json.load(open(x.rec106, encoding="utf-8"))
    rj["files"]["VGAPLE"]["file_sha256"] = "0" * 64
    MS.BS.wt(x.rec106, json.dumps(rj))
    rec = MS.build(x, a, expect=exp)
    assert rec["pass_p0"] is False and rec["failures"][0].startswith("files: ")
