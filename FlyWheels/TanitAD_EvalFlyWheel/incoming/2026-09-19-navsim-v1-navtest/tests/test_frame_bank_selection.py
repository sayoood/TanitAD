"""The frame bank's KB3 selection — which tokens a shard can build, and what is CARRIED OVER.

    PYTHONPATH=D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval \
      C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q tests/test_frame_bank_selection.py

This is the part of the 32-shard unattended build that cannot be checked by looking at the
output: a log that SPANS two shards leaves tokens with some of their 12 jpgs, and dropping them
(or building them from 11 jpgs) is a silent hole in a 12,146-token bank. Expectations are
literals.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
_spec = importlib.util.spec_from_file_location("w3_bank", CODE / "build_navtest_frames.py")
BK = importlib.util.module_from_spec(_spec)
sys.modules["w3_bank"] = BK
_spec.loader.exec_module(BK)


def _tok(log: str, frames):
    """A token record shaped like the export's: 4 frames x cam_l0/f0/r0."""
    return {"cams": {c: [f"{log}/{c.upper()}/{f}.jpg" for f in frames] for c in BK.CAMS}}


TOKS = {
    "t1": _tok("logA", ["a", "b", "c", "d"]),          # entirely inside shard 0
    "t2": _tok("logA", ["c", "d", "e", "f"]),          # e, f only arrive in shard 1
    "t3": _tok("logB", ["g", "h", "i", "j"]),          # entirely inside shard 1
}
NEED = {}
for _t, _r in TOKS.items():
    for _p in BK.token_paths(_r):
        NEED.setdefault(_p, set()).add(_t)


def test_token_paths_is_12_paths_in_cam_order():
    ps = BK.token_paths(TOKS["t1"])
    assert len(ps) == 12 and len(set(ps)) == 12
    assert ps[0] == "logA/CAM_L0/a.jpg" and ps[4] == "logA/CAM_F0/a.jpg" and ps[8] == "logA/CAM_R0/a.jpg"


def test_shard0_builds_only_the_complete_token_and_carries_the_split_one():
    shard0 = {p for p in NEED if p.split("/")[-1][:-4] in ("a", "b", "c", "d")}
    complete, partial = BK.complete_partial(TOKS, NEED, shard0)
    assert complete == ["t1"]                    # t1 has all 12
    assert partial == ["t2"]                     # t2 has 6 of 12 — carried, never built
    assert "t3" not in complete + partial        # not touched by this shard at all


def test_shard1_completes_the_carried_token():
    shard0 = {p for p in NEED if p.split("/")[-1][:-4] in ("a", "b", "c", "d")}
    shard1 = {p for p in NEED if p.split("/")[-1][:-4] in ("e", "f", "g", "h", "i", "j")}
    carried = {p for p in shard0 if p in BK.token_paths(TOKS["t2"])}
    # t1 was BUILT in shard 0, so it is skipped: the carried jpgs c,d belong to t2 as well, and
    # without `skip` an already-banked token would reappear as partial and be carried forever.
    complete, partial = BK.complete_partial(TOKS, NEED, shard1 | carried, skip={"t1"})
    assert complete == ["t2", "t3"] and partial == []
    # without the skip the built token IS a candidate again — the behaviour `skip` exists to stop
    c2, p2 = BK.complete_partial(TOKS, NEED, shard1 | carried)
    assert c2 == ["t2", "t3"] and p2 == ["t1"]


def test_one_missing_jpg_is_never_built():
    almost = set(BK.token_paths(TOKS["t1"])[:-1])     # 11 of 12
    complete, partial = BK.complete_partial(TOKS, NEED, almost)
    # t2 shares frames c,d with t1, so it is a candidate too — and equally incomplete
    assert complete == [] and partial == ["t1", "t2"]


def test_empty_shard_builds_nothing():
    assert BK.complete_partial(TOKS, NEED, set()) == ([], [])


def test_verified_shards_requires_sha_equal_to_the_etag(tmp_path, monkeypatch):
    import json
    rec = {"files": {
        "openscene_sensor_test_camera_0.tgz": {"status": "COMPLETE_VERIFIED", "sha256": "aa",
                                               "x_linked_etag": "aa"},
        "openscene_sensor_test_camera_1.tgz": {"status": "COMPLETE_VERIFIED", "sha256": "aa",
                                               "x_linked_etag": "bb"},      # must be REFUSED
        "openscene_sensor_test_camera_2.tgz": {"status": "FAILED", "sha256": "cc",
                                               "x_linked_etag": "cc"}}}
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    monkeypatch.setattr(BK, "RECEIPT", str(p))
    assert sorted(BK.verified_shards()) == [0]
