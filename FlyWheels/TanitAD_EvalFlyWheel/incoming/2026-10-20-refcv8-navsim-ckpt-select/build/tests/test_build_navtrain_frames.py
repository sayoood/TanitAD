"""LITERAL tests for ``code/build_navtrain_frames.py`` -- the navtrain port of W3's frame-bank builder (KB3 + B7 hard refusal).

Synthetic world (no navtrain data exists): one log ``a_log`` of 20 frames, sample tokens at positions 3, 4, 5 -> tokens ``03a 04a 05a``;
their 4-frame windows are frames 0..3, 1..4, 2..5 -> 6 UNIQUE frames (rows 0..5).  The fake stitcher paints a frame with a value derived from
its CAM_F0 file (``shards.expected_value``) so the content of every bank row says WHICH file it came from.

    expected rows:  03a -> [0,1,2,3]   04a -> [1,2,3,4]   05a -> [2,3,4,5]

The REAL stitch and the REAL ``BankNavtest416`` reader are exercised in the ``stitch`` / ``bridge`` tests at the bottom.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

import build_navtrain_frames as B
import navtrain_common as N
import shards as SH

LOG = "a_log"
POSITIONS = [3, 4, 5]
TOKENS = ["03a", "04a", "05a"]


@pytest.fixture(autouse=True)
def small_frames(monkeypatch):
    monkeypatch.setattr(B, "H", 8)
    monkeypatch.setattr(B, "W", 16)


class World:
    def __init__(self, tmp, *, drop=(), black=(), dup_diff=(), current=range(3, 6), history=range(0, 5), n_frames=20):
        self.tmp = str(tmp).replace("\\", "/")
        self.frames = {LOG: SH.make_frames(LOG, n_frames)}
        self.export = SH.make_export(self.frames, {LOG: POSITIONS})
        self.sample = SH.make_sample_doc(TOKENS)
        self.logs_root = f"{self.tmp}/trainval"
        SH.write_pickles(self.logs_root, self.frames)

        def content(rel):
            return b"BLACK" + rel.encode() if rel in black else SH.jpg_content(rel)

        cams = SH.CAMS
        cur = [SH.frame_name(LOG, i, c) for i in current for c in cams if SH.frame_name(LOG, i, c) not in drop]
        his = [SH.frame_name(LOG, i, c) for i in history for c in cams if SH.frame_name(LOG, i, c) not in drop]
        os.makedirs(f"{self.tmp}/shards", exist_ok=True)
        SH.write_tgz(f"{self.tmp}/shards/navtrain_current_1.tgz", "navtrain_current_1", cur, content=content)
        SH.write_tgz(f"{self.tmp}/shards/navtrain_history_1.tgz", "navtrain_history_1", his,
                     content=lambda rel: (b"DIFFERENT" + rel.encode()) if rel in dup_diff else content(rel))
        self.cur, self.his = cur, his
        self.sources = N.discover_shards(f"{self.tmp}/shards")
        self.bank = f"{self.tmp}/bank"
        self.scratch = f"{self.tmp}/scratch"

    def build(self, e2=None, **kw):
        return B.build_bank(self.export, self.sample, self.sources, self.logs_root, self.bank, self.scratch, e2 or SH.FakeE2(),
                            progress=lambda *_: None, **kw)


def bank_arr(w):
    return np.load(f"{w.bank}/frames_s00.npy", mmap_mode="r")


# ---------------------------------------------------------------------------------------------------------------------------
def test_unique_frames_are_stored_once_and_rows_are_the_history_windows(tmp_path):
    w = World(tmp_path)
    rep = w.build()
    idx = json.load(open(f"{w.bank}/index.json", encoding="utf-8"))
    assert rep["n_tokens_built"] == 3 and rep["n_frames"] == 6 and rep["n_needed_jpgs"] == 6 * 3
    assert idx["tokens"]["03a"]["rows"] == [0, 1, 2, 3]
    assert idx["tokens"]["04a"]["rows"] == [1, 2, 3, 4]
    assert idx["tokens"]["05a"]["rows"] == [2, 3, 4, 5]
    assert bank_arr(w).shape == (6, 8, 16, 3)


def test_every_bank_row_is_the_frame_it_claims_to_be(tmp_path):
    """The fake stitcher paints each frame with a value derived from ITS OWN F0 file: row k must carry the value of frame k."""
    w = World(tmp_path)
    w.build()
    a = bank_arr(w)
    for k in range(6):
        rel = SH.frame_name(LOG, k, "CAM_F0")
        assert int(a[k].min()) == int(a[k].max()) == SH.expected_value(rel), k


def test_a_token_reads_its_own_four_frames_in_history_order(tmp_path):
    w = World(tmp_path)
    w.build()
    idx = json.load(open(f"{w.bank}/index.json", encoding="utf-8"))
    a = bank_arr(w)
    for tok, first in (("03a", 0), ("04a", 1), ("05a", 2)):
        stack = np.ascontiguousarray(a[idx["tokens"][tok]["rows"]])
        got = [int(stack[i].min()) for i in range(4)]
        assert got == [SH.expected_value(SH.frame_name(LOG, first + i, "CAM_F0")) for i in range(4)], tok


def test_the_index_schema_is_the_one_BankNavtest416_reads(tmp_path):
    w = World(tmp_path)
    w.build()
    idx = json.load(open(f"{w.bank}/index.json", encoding="utf-8"))
    assert set(idx) == {"frame", "tokens", "shards", "rigs"}
    assert idx["frame"] == {"h": 8, "w": 16, "f_ref": 488.92398517830253, "projection": "cylindrical"}
    t = idx["tokens"]["04a"]
    assert set(t) == {"shard", "file", "rows", "log_name", "rig_key", "observed_frac", "mean_px", "sha256"}
    assert t["shard"] == 0 and t["file"] == "frames_s00.npy" and t["log_name"] == "a_log" and t["rig_key"] == "rigA"
    a = bank_arr(w)
    stack = np.ascontiguousarray(a[t["rows"]])
    assert t["sha256"] == hashlib.sha256(stack.tobytes()).hexdigest()[:16] and len(t["sha256"]) == 16
    assert stack.shape == (4, 8, 16, 3)
    assert os.path.exists(f"{w.bank}/rigs/rigA.src.npy")


def test_done_marker_report_and_scratch(tmp_path):
    w = World(tmp_path)
    rep = w.build()
    d = json.load(open(f"{w.bank}/DONE.json", encoding="utf-8"))
    assert d["done"] is True and d["n_tokens"] == 3 and d["n_frames"] == 6 and len(d["index_sha256"]) == 64
    assert json.load(open(f"{w.bank}/BUILD_REPORT.json", encoding="utf-8"))["sample_sha256_sorted_tokens"] == w.sample["sha256_sorted_tokens"]
    assert rep["scratch_deleted"] is True and not os.path.exists(w.scratch)


def test_the_stitcher_is_called_once_per_unique_frame(tmp_path):
    w = World(tmp_path)
    e2 = SH.FakeE2()
    w.build(e2)
    assert e2.calls == 6                                       # 3 tokens x 4 frames = 12 slots, 6 unique frames


def test_a_directory_source_gives_the_same_bank_as_the_archives(tmp_path):
    w = World(tmp_path / "t")
    w.build()
    w2 = World(tmp_path / "d")
    SH.write_dir(f"{w2.tmp}/dirs/navtrain_current_1", w2.cur, content=SH.jpg_content)
    SH.write_dir(f"{w2.tmp}/dirs/navtrain_history_1", w2.his, content=SH.jpg_content)
    w2.sources = [f"{w2.tmp}/dirs/navtrain_current_1", f"{w2.tmp}/dirs/navtrain_history_1"]
    w2.build()
    assert hashlib.sha256(bank_arr(w).tobytes()).hexdigest() == hashlib.sha256(bank_arr(w2).tobytes()).hexdigest()
    assert json.load(open(f"{w.bank}/index.json"))["tokens"] == json.load(open(f"{w2.bank}/index.json"))["tokens"]


# ---- KB3 / B7: a missing frame is a HARD REFUSAL -----------------------------------------------------------------------------
def _refused(w, why):
    with pytest.raises(B.Refused) as e:
        w.build()
    assert e.value.why == why, (e.value.why, e.value.detail)
    assert not os.path.exists(f"{w.bank}/index.json") and not os.path.exists(f"{w.bank}/DONE.json")
    return e.value.detail


def test_one_missing_history_jpg_refuses_and_names_every_affected_token(tmp_path):
    # frame 2 of CAM_R0 is a history frame of tokens 03a, 04a, 05a (windows 0..3, 1..4, 2..5)
    gone = SH.frame_name(LOG, 2, "CAM_R0")
    w = World(tmp_path, drop=(gone,))
    d = _refused(w, "missing_frames")
    assert d["n_missing_jpgs"] == 1 and d["missing_first50"] == [gone] and d["n_tokens_affected"] == 3
    assert d["tokens_first50"] == ["03a", "04a", "05a"] and "KB3" in d["rule"]


def test_a_missing_t0_jpg_of_one_camera_refuses(tmp_path):
    gone = SH.frame_name(LOG, 5, "CAM_L0")                  # frame 5 is in the window of 05a only
    w = World(tmp_path, drop=(gone,))
    d = _refused(w, "missing_frames")
    assert d["tokens_first50"] == ["05a"] and d["n_missing_jpgs"] == 1


def test_a_missing_FRONT_only_history_is_not_enough_the_L0_R0_cameras_are_required(tmp_path):
    drops = tuple(SH.frame_name(LOG, i, c) for i in range(0, 6) for c in ("CAM_L0", "CAM_R0"))
    w = World(tmp_path, drop=drops)
    d = _refused(w, "missing_frames")
    assert d["n_missing_jpgs"] == 12 and d["n_tokens_affected"] == 3


def test_an_archive_that_is_absent_refuses(tmp_path):
    w = World(tmp_path)
    w.sources = [s for s in w.sources if "history" not in s]      # only the `current` archive was downloaded
    d = _refused(w, "missing_frames")
    assert d["n_tokens_affected"] == 3


def test_an_all_black_frame_refuses(tmp_path):
    black = SH.frame_name(LOG, 3, "CAM_F0")
    w = World(tmp_path, black=(black,))
    d = _refused(w, "frame_failed_to_stitch")
    assert d["n_failed"] == 1 and "KB2" in d["first"][0]["reason"]


def test_a_frame_that_fails_to_stitch_refuses(tmp_path):
    w = World(tmp_path)
    bad = SH.FakeE2(fail_if=lambda paths: paths[1].name.startswith("04a"))
    with pytest.raises(B.Refused) as e:
        w.build(bad)
    assert e.value.why == "frame_failed_to_stitch" and e.value.detail["n_failed"] == 1
    assert not os.path.exists(f"{w.bank}/index.json")


def test_two_shard_members_with_the_same_path_but_different_bytes_refuse(tmp_path):
    dup = SH.frame_name(LOG, 3, "CAM_F0")                    # frame 3 sits in BOTH archives (current 3..5, history 0..4)
    w = World(tmp_path, dup_diff=(dup,))
    d = _refused(w, "duplicate_member_content_differs")
    assert d["rel"] == dup


def test_identical_duplicate_members_are_fine(tmp_path):
    w = World(tmp_path)                                       # frames 3 and 4 are in both archives with the same bytes
    assert w.build()["n_tokens_built"] == 3


def test_an_inputs_export_that_does_not_cover_the_sample_refuses(tmp_path):
    w = World(tmp_path)
    del w.export["tokens"]["05a"]
    d = _refused(w, "inputs_do_not_cover_the_sample")
    assert d["n_absent"] == 1 and d["first"] == ["05a"]


def test_a_navtest_export_is_refused(tmp_path):
    w = World(tmp_path)
    w.export["split"] = "navtest"
    _refused(w, "inputs_split_is_not_navtrain")


def test_a_sample_that_is_not_fixed_refuses(tmp_path):
    w = World(tmp_path)
    w.sample = SH.make_sample_doc(TOKENS, status="UNDERPOWERED")
    _refused(w, "sample_not_fixed")


def test_a_sample_whose_sha256_does_not_rederive_refuses(tmp_path):
    w = World(tmp_path)
    w.sample["sha256_sorted_tokens"] = "0" * 64
    _refused(w, "sample_sha256_does_not_rederive")


def test_a_finished_bank_is_never_rebuilt_in_place(tmp_path):
    w = World(tmp_path)
    w.build()
    with pytest.raises(B.Refused) as e:
        w.build()
    assert e.value.why == "bank_already_done"


def test_verify_bank_catches_a_flipped_byte(tmp_path):
    w = World(tmp_path)
    w.build()
    assert B.verify_bank(w.bank, TOKENS) == []
    a = np.load(f"{w.bank}/frames_s00.npy", mmap_mode="r+")
    a[2, 0, 0, 0] = np.uint8((int(a[2, 0, 0, 0]) + 1) % 256)
    a.flush()
    del a
    p = B.verify_bank(w.bank, TOKENS)
    assert p == ["03a: sha256 does not re-derive", "04a: sha256 does not re-derive", "05a: sha256 does not re-derive"]


def test_the_mutation_list_is_the_registered_one():
    assert B.MUTATIONS == ("missing_frame_standin", "kb3_t0_only", "zero_frame_ok", "dup_content_ignored", "partial_on_stitch_failure",
                           "rows_shifted")


# ---------------------------------------------------------------------------------------------------------------------------
# the REAL BankNavtest416 reads a bank built here (needs the launch tree + torch; 416 x 1024 frames)
# ---------------------------------------------------------------------------------------------------------------------------
R6CODE = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/code"
R7CODE = "C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code"


def _import_bridge():
    for p in (R6CODE, R7CODE, "C:/lgs/Lfb47a58376/stack", "C:/lgs/Lfb47a58376/taniteval"):
        if p not in sys.path and os.path.isdir(p):
            sys.path.append(p)
    os.environ.setdefault("TANITAD_REPO", "C:/lgs/Lfb47a58376")
    sys.dont_write_bytecode = True
    import refcv6_bridge                                                      # noqa: WPS433
    return refcv6_bridge


@pytest.mark.bridge
def test_the_real_BankNavtest416_loads_the_bank(tmp_path, monkeypatch):
    try:
        R6 = _import_bridge()
    except Exception as e:                                                    # noqa: BLE001
        pytest.skip(f"refcv6_bridge not importable here: {type(e).__name__}: {e}")
    monkeypatch.setattr(B, "H", 416)
    monkeypatch.setattr(B, "W", 1024)
    w = World(tmp_path)
    w.build(SH.FakeE2(h=416, w=1024))
    bank = R6.BankNavtest416(w.bank)
    assert sorted(bank.prov) == TOKENS and bank.keep == 4
    fr, rk, sha = bank.load("04a")
    assert fr.shape == (4, 416, 1024, 3) and fr.dtype == np.uint8 and rk == "rigA" and len(sha) == 16
    for i, k in enumerate([1, 2, 3, 4]):
        assert int(fr[i].min()) == int(fr[i].max()) == SH.expected_value(SH.frame_name(LOG, k, "CAM_F0"))
    a = np.load(f"{w.bank}/frames_s00.npy", mmap_mode="r+")
    a[3, 5, 5, 1] = np.uint8((int(a[3, 5, 5, 1]) + 3) % 256)      # a corrupted byte inside token 04a's rows must be REFUSED on load
    a.flush()
    del a
    fresh = R6.BankNavtest416(w.bank)
    with pytest.raises(Exception) as e:
        fresh.load("04a")
    assert "bank sha" in str(e.value)


# ---------------------------------------------------------------------------------------------------------------------------
# the REAL E2 stitch (torch + tanitad stack), REAL navtest rig calibrations, synthetic pixels
# ---------------------------------------------------------------------------------------------------------------------------
NAVTEST_LOGS = "D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test"
EXPORT1 = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs_shard0log1.json.gz"
FRAMES416 = f"{R6CODE}/frames416.py"


@pytest.mark.stitch
def test_real_stitch_pipeline_on_real_rigs_with_synthetic_pixels_and_the_real_bank_reader(tmp_path, monkeypatch, capsys):
    import gzip
    for p in (NAVTEST_LOGS, EXPORT1, FRAMES416):
        if not os.path.exists(p):
            pytest.skip(f"fixture absent: {p}")
    try:
        R6 = _import_bridge()
        from PIL import Image
        import torch  # noqa: F401
    except Exception as e:                                                    # noqa: BLE001
        pytest.skip(f"stack not importable: {type(e).__name__}: {e}")
    monkeypatch.setattr(B, "H", 416)
    monkeypatch.setattr(B, "W", 1024)
    doc = json.load(gzip.open(EXPORT1, "rt", encoding="utf-8"))
    toks = sorted(doc["tokens"])[:2]
    # consecutive-ish tokens share frames: take the two with the largest window overlap
    by_log = {}
    for t in sorted(doc["tokens"]):
        by_log.setdefault(doc["tokens"][t]["log_name"], []).append(t)
    log = next(iter(by_log))
    frames, _ = N.load_log_pickle(f"{NAVTEST_LOGS}/{log}.pkl")
    pos = {f["token"]: i for i, f in enumerate(frames)}
    cand = sorted(by_log[log], key=lambda t: pos[t])
    pair = None
    for a, b in zip(cand, cand[1:]):
        if pos[b] - pos[a] == 1:
            pair = [a, b]
            break
    toks = pair or cand[:2]
    doc["tokens"] = {t: doc["tokens"][t] for t in toks}
    doc["split"] = "navtrain"                                                 # fixture relabel: the pipeline under test is the navtrain one
    sample = SH.make_sample_doc(toks)
    # synthetic 1920x1080 jpgs at the real data_paths, in a navtrain-shaped archive
    rng = np.random.default_rng(0)
    rels = sorted({p for t in toks for c in doc["tokens"][t]["cams"].values() for p in c})
    base = (np.linspace(40, 200, 1920)[None, :, None] * np.ones((1080, 1, 3))).astype(np.uint8)

    def jpg(rel):
        import io
        arr = np.clip(base.astype(np.int16) + rng.integers(-20, 20, base.shape), 1, 254).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=60)
        return buf.getvalue()

    os.makedirs(tmp_path / "shards")
    SH.write_tgz(str(tmp_path / "shards" / "navtrain_current_1.tgz"), "navtrain_current_1", rels, content=jpg)
    e2 = B.load_stitcher(FRAMES416)
    bank, scratch = str(tmp_path / "bank"), str(tmp_path / "scratch")
    rep = B.build_bank(doc, sample, N.discover_shards(str(tmp_path / "shards")), NAVTEST_LOGS, bank, scratch, e2,
                       keep_scratch=True, progress=lambda *_: None)
    assert rep["n_tokens_built"] == 2 and rep["frame"]["h"] == 416 and rep["frame"]["w"] == 1024 and rep["frame"]["projection"] == "cylindrical"
    bk = R6.BankNavtest416(bank)
    fr, rk, sha = bk.load(toks[0])
    assert fr.shape == (4, 416, 1024, 3) and float(fr.mean()) >= 1.0
    # the builder must not alter a pixel: re-stitch one frame with E2's build_scene directly and compare bit for bit
    ft = doc["tokens"][toks[0]]["frame_tokens"][-1]
    idx = {f["token"]: f for f in frames}
    one, src, rk2, obs = e2.build_scene([idx[ft]["cams"]], __import__("pathlib").Path(scratch), {})
    assert np.array_equal(one[0], fr[-1]) and rk2 == rk
    # the same through the CLI at the process boundary: a missing jpg must produce the REFUSAL marker, a complete set the DONE marker
    sample_p, inputs_p = str(tmp_path / "sample.json"), str(tmp_path / "inputs.json.gz")
    json.dump(sample, open(sample_p, "w"))
    with gzip.open(inputs_p, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    monkeypatch.undo()
    rc = B.main(["--sample", sample_p, "--inputs", inputs_p, "--shards-dir", str(tmp_path / "shards"), "--logs-root", NAVTEST_LOGS,
                 "--bank", str(tmp_path / "bank_cli"), "--frames416", FRAMES416, "--threads", "2"])
    assert rc == 0 and "ZZBANK_DONE tokens=2" in capsys.readouterr().out
    SH.write_tgz(str(tmp_path / "shards2.tgz"), "navtrain_current_1", [r for r in rels if r != rels[7]], content=jpg)
    rc = B.main(["--sample", sample_p, "--inputs", inputs_p, "--shard", str(tmp_path / "shards2.tgz"), "--logs-root", NAVTEST_LOGS,
                 "--bank", str(tmp_path / "bank_cli2"), "--frames416", FRAMES416, "--threads", "2"])
    out = capsys.readouterr().out
    assert rc == 3 and "ZZBANK_REFUSED_missing_frames" in out
    assert json.load(open(tmp_path / "bank_cli2" / "REFUSED.json"))["detail"]["missing_first50"] == [rels[7]]
