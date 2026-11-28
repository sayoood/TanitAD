"""``--reuse-floors`` — the wiring and the identity primitives, pinned.

⭐ The end-to-end identity check is exercised against the REAL banked floors run by
``FlyWheels/…/2026-09-20-navhard-refcv4b/code/probe_floor_reuse.py`` (both floors adopt byte-
identically; 6/6 mutations REFUSED: wrong devkit sha, a dropped token, a flipped stage, a removed
patch, another export, a STOP pose perturbed by 1 mm). This file pins what CI can check without
the 5,912-token artifacts: the flag reaches the benchmark, and the two primitives the check rests on
behave — token stages are read from WHICH columns the devkit populated, and a patch set compares by
name AND blob.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "taniteval"))
FR = pytest.importorskip("taniteval.bench.navsim.floor_reuse")


def test_flag_reaches_the_benchmark_and_the_benchmark_calls_the_check():
    """⛔ 'built, tested and UNREACHABLE FROM ITS CALLER' — both hops, with a same-breath control."""
    from taniteval.bench import cli
    a = cli.build_parser().parse_args(
        ["navsim_v2", "--ckpt", "none", "--split", "navhard_two_stage", "--reuse-floors", "R"])
    assert a.reuse_floors == "R"
    src = (REPO / "taniteval" / "taniteval" / "bench" / "navsim" / "benchmark.py").read_text(encoding="utf-8")
    # two adoption sites: floors (CV/STOP/ECHO) and scored MODEL arms (re-derivation, kind="model")
    assert src.count("FR.check_and_adopt(") == 2
    assert src.count('log=ctx.log, kind="model")') == 1     # the CALL argument, not a comment's mention
    assert src.count('getattr(a, "reuse_floors", None)') == 1
    assert src.count('getattr(a, "reuse_echo", None)') == 1
    assert src.count('getattr(a, "reuse_model_scores", None)') == 1
    c = cli.build_parser().parse_args(
        ["navsim_v2", "--ckpt", "none", "--split", "navhard_two_stage", "--reuse-model-scores", "M"])
    assert c.reuse_model_scores == "M"
    assert src.count("SC.score_arm(") == 1                 # control: the scoring path still exists
    b = cli.build_parser().parse_args(
        ["navsim_v2", "--ckpt", "none", "--split", "navhard_two_stage", "--reuse-echo", "E"])
    assert b.reuse_echo == "E"


def test_ECHO_is_a_seam_floor_and_must_pass_the_seam_equality_check():
    """ECHO is reusable ONLY because its plan is deterministic and checkpoint-independent — so it
    must carry the same byte-equal seam check as STOP, never the weaker CV path."""
    assert "ECHO" in FR.REUSABLE_FLOORS and "ECHO" in FR.SEAM_FLOORS
    assert "STOP" in FR.SEAM_FLOORS and "CV" not in FR.SEAM_FLOORS


def test_a_model_arm_is_never_reusable_as_a_floor(tmp_path):
    with pytest.raises(FR.FloorReuseRefused):
        FR.check_and_adopt(arm="A1", src_run=tmp_path, run=None, prof=None, split_yaml={},
                           devkit_sha="x", patches=[], preflight={}, export_sha256=None, new_seam=None)


def test_token_stage_is_read_from_the_POPULATED_columns(tmp_path):
    """A stage-1 row fills the *_stage_one columns and leaves *_stage_two empty, and vice versa; the
    stage is inferred from that, never from a label that could be wrong."""
    p = tmp_path / "s.csv"
    p.write_text(",token,valid,ego_progress_stage_one,ego_progress_stage_two,score\n"
                 "0,aaa,True,0.5,,0.4\n"
                 "1,bbb,True,,0.7,0.6\n"
                 "2,extended_pdm_score_combined,True,0.5,0.7,0.1\n", encoding="utf-8")
    assert FR._csv_token_stages(p) == {"aaa": 1, "bbb": 2}      # summary rows excluded


def test_seam_byte_equality_is_BYTES_not_values_so_NaN_rows_compare_equal(tmp_path):
    """⚠️ REGRESSION, MEASURED 2026-09-21. A model arm's CV-stand-in rows are all-NaN; `np.array_equal`
    says NaN != NaN, so three files with the SAME sha256 were refused as "the arm itself changed". The
    check claims BYTE equality and must implement it. Control: a one-ulp change is still refused."""
    import numpy as np
    FR_ = FR
    poses = np.zeros((5, 8, 3), np.float32)
    poses[:2] = np.nan                                         # the stand-in rows
    tok = np.asarray([f"t{i}" for i in range(5)])
    a, b, c = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    np.savez(a, token=tok, poses=poses)
    np.savez(b, token=tok, poses=poses.copy())
    p2 = poses.copy()
    p2[3, 0, 0] = np.nextafter(np.float32(0), np.float32(1))   # ONE ulp
    np.savez(c, token=tok, poses=p2)
    za, zb, zc = (np.load(x, allow_pickle=False) for x in (a, b, c))
    eq = lambda x, y: all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape  # noqa: E731
                          and x[k].tobytes() == y[k].tobytes() for k in ("token", "poses"))
    assert not np.array_equal(za["poses"], zb["poses"])        # the trap: value-equality says DIFFERENT
    assert eq(za, zb)                                          # bytes say SAME — and they are
    assert not eq(za, zc)                                      # and one ulp is still caught
    src = (REPO / "taniteval" / "taniteval" / "bench" / "navsim" / "floor_reuse.py").read_text(encoding="utf-8")
    assert "a[k].tobytes() == b[k].tobytes()" in src and "np.array_equal(a[\"poses\"]" not in src


def test_patch_sets_compare_by_name_AND_blob():
    a = [{"name": "idm", "raw_bytes_git_blob_now": "b95b"}, {"name": "setup.py"}]
    same = [{"name": "setup.py"}, {"name": "idm", "raw_bytes_git_blob_now": "b95b"}]    # order-free
    other_blob = [{"name": "idm", "raw_bytes_git_blob_now": "1924"}, {"name": "setup.py"}]
    missing = [{"name": "setup.py"}]
    assert FR._patch_key(a) == FR._patch_key(same)
    assert FR._patch_key(a) != FR._patch_key(other_blob)   # the IDM patch pre vs post: 166 tokens
    assert FR._patch_key(a) != FR._patch_key(missing)
