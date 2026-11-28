"""Local tests of the TELEGAP data-side kit (card ADD122). Synthetic data only: no network, no GPU, no real video.

Run the whole file (one pytest at a time):
  cd /f/TanitAD-scratch/v8proof && PYTHONDONTWRITEBYTECODE=1 /c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest telegap/enc/test_tele_enc.py \
     -q -p no:cacheprovider --basetemp C:/Users/Admin/AppData/Local/Temp/tg_enc_bt
Needs the stack tree (TANITAD_STACK, default E:/Projects/TanitAD/stack) for the geometry tests; they skip when it is absent.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "cropread", "enc"))      # the data kit of card ADD99, used unchanged
import enc_common as E   # noqa: E402
import tele_stage as TS   # noqa: E402
import tele_encode as TE   # noqa: E402
import ctl_t2 as T2   # noqa: E402

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
HAVE_STACK = os.path.isdir(os.path.join(E.STACK, "tanitad"))
needs_stack = pytest.mark.skipif(not HAVE_STACK, reason="stack tree not found")
TELE = TS.TELE


# ------------------------------------------------------------------------------------------------ synthetic sources
def parquet_bytes(df):
    b = io.BytesIO()
    df.to_parquet(b)
    return b.getvalue()


def ts_df(n=605, t0=-147491, step=33333):
    return pd.DataFrame({"timestamp": (t0 + np.arange(n) * step).astype(np.int64)}, index=pd.RangeIndex(n, name="frame_index"))


def make_zip(path, clips, corrupt=None):
    """A small stored archive standing in for a chunk: per clip the three members; the payloads are marker bytes (no real video)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for c in clips:
            z.writestr(f"{TELE}/{c}.{TELE}.mp4", b"MP4DATA-" + c.encode() * 40)
            z.writestr(f"{TELE}/{c}.{TELE}.timestamps.parquet", parquet_bytes(ts_df(40)))
            z.writestr(f"{TELE}/{c}.{TELE}.blurred_boxes.parquet", b"BOXES")
    if corrupt:
        raw = bytearray(open(path, "rb").read())
        k = raw.find(b"MP4DATA-" + corrupt.encode())
        raw[k + 12] ^= 0xFF
        open(path, "wb").write(bytes(raw))


def calib_tables(clips_by_chunk):
    ia, ea = [], []
    for chunk, clips in clips_by_chunk.items():
        for c in clips:
            for cam in (TELE, "camera_front_wide_120fov"):
                row = {"clip_id": c, "camera_name": cam, "width": 1920, "height": 1080, "cx": 958.5 if cam == TELE else 960.0, "cy": 540.0}
                row.update({f"bw_poly_{i}": 0.1 * i for i in range(5)})
                row.update({f"fw_poly_{i}": 0.0 if i == 0 else (1900.0 if i == 1 else 0.5 * i) for i in range(5)})
                ia.append(row)
                ea.append({"clip_id": c, "sensor_name": cam, "qx": 0.5, "qy": -0.5, "qz": 0.5, "qw": -0.5, "x": 2.0, "y": 0.0, "z": 1.4})
    return pd.DataFrame(ia).set_index(["clip_id", "camera_name"]), pd.DataFrame(ea).set_index(["clip_id", "sensor_name"])


@pytest.fixture()
def patched_source(tmp_path, monkeypatch):
    """Two chunks (5: a1, a2 ; 7: b1), a fake index, fake downloads, a fake probe, no network."""
    chunks = {5: ["aaaa1111-0000", "aaaa2222-0000"], 7: ["bbbb1111-0000"]}
    src = tmp_path / "src"
    src.mkdir()
    for ch, cl in chunks.items():
        make_zip(str(src / f"c{ch}.zip"), cl)
    idx = pd.DataFrame({"chunk": [5, 5, 7, 9], "clip_is_valid": True}, index=pd.Index(["aaaa1111-0000", "aaaa2222-0000", "bbbb1111-0000", "zzzz-nochunk"], name="clip_id"))
    ia, ea = calib_tables(chunks)
    for ch in chunks:
        ia.loc[chunks[ch]].to_parquet(src / f"intr{ch}.parquet")
        ea.loc[chunks[ch]].to_parquet(src / f"extr{ch}.parquet")
    opened = []

    @contextlib.contextmanager
    def fake_open(chunk):
        opened.append(int(chunk))
        with zipfile.ZipFile(str(src / f"c{int(chunk)}.zip")) as z:
            yield z

    def fake_dl(path, local_dir):
        for ch in chunks:
            if path == TS.INTR_PQ.format(ch):
                return str(src / f"intr{ch}.parquet")
            if path == TS.EXTR_PQ.format(ch):
                return str(src / f"extr{ch}.parquet")
        raise FileNotFoundError(path)
    monkeypatch.setattr(TS, "open_chunk_zip", fake_open)
    monkeypatch.setattr(TS, "hf_dl", fake_dl)
    monkeypatch.setattr(TS, "chunk_index", lambda hf_dir: idx)
    monkeypatch.setattr(TS, "probe_mp4", lambda p: (1920, 1080))
    monkeypatch.setattr(TS, "_TOK", "tok-secret-xyz-123")
    monkeypatch.setenv("TG_NO_SLEEP", "1")
    return {"chunks": chunks, "src": src, "opened": opened, "idx": idx, "tmp": tmp_path}


# ------------------------------------------------------------------------------------------------ staging
def test_group_by_chunk_keeps_order_and_reports_clips_without_an_index_row(patched_source):
    clips = ["bbbb1111-0000", "aaaa1111-0000", "nope", "aaaa2222-0000"]
    groups, missing = TS.group_by_chunk(clips, patched_source["idx"])
    assert groups == {7: ["bbbb1111-0000"], 5: ["aaaa1111-0000", "aaaa2222-0000"]} and missing == ["nope"]


def test_member_selection_takes_exactly_one_mp4_and_one_timestamps_member():
    names = [f"d/{c}.{TELE}.{k}" for c in ("c1", "c2") for k in ("mp4", "timestamps.parquet", "blurred_boxes.parquet")]
    m = TS.select_members(names, "c2")
    assert m == {"mp4": f"d/c2.{TELE}.mp4", "ts": f"d/c2.{TELE}.timestamps.parquet"}
    with pytest.raises(ValueError):
        TS.select_members(names, "c3")                                     # no member
    with pytest.raises(ValueError):
        TS.select_members(names + [f"e/c2.{TELE}.mp4"], "c2")                # ambiguous


def test_stage_run_writes_files_calibration_ledger_and_record_then_resumes(patched_source, capsys):
    root = str(patched_source["tmp"] / "tele_stage")
    clips = ["aaaa1111-0000", "aaaa2222-0000", "bbbb1111-0000", "zzzz-nochunk", "unknown-clip"]
    rc = TS.run(clips, root, str(patched_source["tmp"] / "hf"), workers=2, min_free_gb=0.0)
    assert rc == 0 and patched_source["opened"].count(5) == 1 and patched_source["opened"].count(7) == 1     # one open archive per chunk
    L = E.Layout(root)
    for c in clips[:3]:
        assert os.path.getsize(L.mp4(c)) > 0 and os.path.exists(L.ts(c))
    intr = pd.read_csv(L.intr_csv, dtype={"clip_id": str})
    assert sorted(intr["clip_id"]) == sorted(clips[:3]) and list(intr.columns) == TS.INTR_COLS
    assert set(intr["cx"]) == {958.5}                                     # the TELE rows, not the wide rows
    ext = pd.read_csv(f"{root}/calibration/front_tele_extrinsics.csv", dtype={"clip_id": str})
    assert list(ext.columns) == TS.EXTR_COLS and len(ext) == 3
    recs = [json.loads(x) for x in open(L.ledger, encoding="utf-8")]
    assert sum(r["status"] == "ok" for r in recs) == 3
    rec_files = os.listdir(f"{root}/records")
    assert len(rec_files) == 1
    rec = json.load(open(f"{root}/records/{rec_files[0]}", encoding="utf-8"))
    assert rec["n_ok"] == 3 and rec["n_failed"] == 2
    row = next(r for r in rec["clips"] if r["clip"] == "aaaa1111-0000")
    assert row["chunk"] == 5 and row["width"] == 1920 and row["height"] == 1080 and row["n_frames"] == 40 and row["crc_ok"] is True and row["bytes"] > 0
    assert any(r["status"].startswith("fail") for r in rec["clips"] if r["clip"] == "zzzz-nochunk")
    out = capsys.readouterr().out
    assert "tok-secret" not in out and "aaaa1111-0000" not in out and "NOT STAGED zzzz-noc" in out    # ids print as 8 characters
    # resume: nothing is opened again, the complete clips are skipped
    patched_source["opened"].clear()
    rc2 = TS.run(clips[:3], root, str(patched_source["tmp"] / "hf"), workers=2, min_free_gb=0.0)
    assert rc2 == 0 and patched_source["opened"] == []
    rec2 = [json.load(open(f"{root}/records/{f}", encoding="utf-8")) for f in sorted(os.listdir(f"{root}/records"))][-1]
    assert rec2["n_skipped_complete"] == 3 and rec2["n_ok"] == 0


def test_a_member_with_a_bad_crc_is_recorded_as_crc_not_ok_and_not_staged(patched_source):
    make_zip(str(patched_source["src"] / "c5.zip"), patched_source["chunks"][5], corrupt="aaaa2222-0000")
    root = str(patched_source["tmp"] / "tele_stage2")
    TS.run(patched_source["chunks"][5], root, str(patched_source["tmp"] / "hf2"), workers=1, min_free_gb=0.0)
    rec = json.load(open(f"{root}/records/" + os.listdir(f"{root}/records")[0], encoding="utf-8"))
    bad = next(r for r in rec["clips"] if r["clip"] == "aaaa2222-0000")
    good = next(r for r in rec["clips"] if r["clip"] == "aaaa1111-0000")
    assert bad["crc_ok"] is False and bad["status"].startswith("fail") and good["crc_ok"] is True and good["status"] == "ok"
    assert not os.path.exists(E.Layout(root).mp4("aaaa2222-0000"))


# ------------------------------------------------------------------------------------------------ the time match
def test_match_rule_17ms_edge_and_missing_frame():
    t_tele = np.array([0.0, 100000.0, 200000.0])
    t_wide = np.array([17000.0, 17001.0, 100000.0 - 17000.0, 100000.0 - 17001.0, 200000.0 + 17000.0, 200000.0 + 17001.0, 50000.0])
    idx, dt, mask = TE.match_frames(t_wide, t_tele)
    assert mask.tolist() == [True, False, True, False, True, False, False]
    assert idx.tolist() == [0, -1, 1, -1, 2, -1, -1]
    assert dt[0] == -17000.0 and dt[2] == 17000.0 and np.isnan(dt[1])
    idx2, _, m2 = TE.match_frames(np.array([1.0, 2.0]), np.array([]))
    assert idx2.tolist() == [-1, -1] and not m2.any()
    i3, d3, m3 = TE.match_frames(np.array([50.0]), np.array([0.0, 100.0]), tol_us=60.0)
    assert i3.tolist() == [0] and d3.tolist() == [-50.0]                   # a tie takes the earlier frame
    with pytest.raises(ValueError):
        TE.match_frames(np.array([1.0]), np.array([5.0, 3.0]))


def test_pilot_times_are_the_decoded_wide_frames(tmp_path):
    p = tmp_path / "w.timestamps.parquet"
    ts_df(180, t0=1_000_000).to_parquet(p)
    plan = E.frame_plan(str(p))
    t = TE.pilot_times(plan)
    assert t.shape[0] == plan["n_cache"] and np.array_equal(t, plan["t_frames"][plan["cache_idx"]])


# ------------------------------------------------------------------------------------------------ windows and tokens
def test_window_constants_and_order():
    assert TE.TELE_WINDOWS == ((8.0, 0.0, -2.0), (4.0, 0.0, -1.0))
    assert TE.TELE_SIZES_DEG == (8.0, 4.0) and TE.TOL_US == 17000 and TE.TOK_DIR == "tok_T"
    assert [w[0] for w in TE.TELE_WINDOWS] == list(TE.TELE_SIZES_DEG)
    assert (T2.T8_AZ, T2.T8_EL) == (0.0, -2.0) and T2.THRESHOLD == 0.9 and T2.OFFSET_DEG == 2.0


def synth_intr_tele():
    from tanitad.data.calib import FThetaIntrinsics
    return FThetaIntrinsics(poly=(0.0, 3700.0, 20.0, -50.0, 10.0), cx=958.0, cy=540.0, width=1920, height=1080, per_clip=True)


@needs_stack
def test_windows_from_video_builds_t8_then_t4_with_the_unchanged_routines():
    intr = synth_intr_tele()
    g = torch.Generator().manual_seed(2)
    vid = (torch.rand(2, 3, 108, 192, generator=g) * 255).to(torch.uint8)
    frames = [E.window_frame(s) for s in TE.TELE_SIZES_DEG]
    out = TE.windows_from_video(vid, intr, frames)
    assert out.shape == (2, 2, 3, 256, 256) and out.dtype == np.uint8
    for i, (size, az, el) in enumerate(TE.TELE_WINDOWS):
        gr, m = E.window_grid(intr, 108, 192, E.window_frame(size), az, el)
        ref = E.window_sample(vid.float(), gr.expand(2, -1, -1, -1), m.expand(2, -1, -1)).numpy()
        assert np.array_equal(out[:, i], ref)
    assert not np.array_equal(out[:, 0], out[:, 1])


class StubEnc:
    def window_tokens(self, wins):
        assert wins.dtype == torch.uint8 and wins.shape[1:] == (3, 256, 256)
        return torch.full((wins.shape[0], 65, 1024), 3.0)


def build_roots(tmp_path, clips, gap=(60, 100), no_mp4=()):
    wide, tele = str(tmp_path / "wide"), str(tmp_path / "tele")
    Lw, Lt = E.Layout(wide), E.Layout(tele)
    os.makedirs(f"{tele}/calibration", exist_ok=True)
    rows = []
    for c in clips:
        os.makedirs(os.path.dirname(Lw.ts(c)), exist_ok=True)
        os.makedirs(os.path.dirname(Lt.ts(c)), exist_ok=True)
        os.makedirs(os.path.dirname(Lt.mp4(c)), exist_ok=True)
        ts_df(180, t0=1_000_000).to_parquet(Lw.ts(c))                                  # wide: 6 s at 30 Hz
        t = ts_df(180, t0=1_005_000)                                                   # tele: same clock, 5 ms later, with a hole
        t = t.drop(index=range(gap[0], gap[1])).reset_index(drop=True)
        t.to_parquet(Lt.ts(c))
        if c not in no_mp4:
            open(Lt.mp4(c), "wb").write(b"x")
        row = {"clip_id": c, "width": 1920, "height": 1080, "cx": 958.0, "cy": 540.0}
        row.update({f"bw_poly_{i}": 0.0 for i in range(5)})
        row.update({f"fw_poly_{i}": v for i, v in enumerate((0.0, 3700.0, 20.0, -50.0, 10.0))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(Lt.intr_csv, index=False)
    # note N1 of card ADD122: the encode carries the window directions through the extrinsics of both cameras (equal mounts here)
    ext = pd.DataFrame([{"clip_id": c, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 0.0, "y": 0.0, "z": 0.0} for c in clips])
    os.makedirs(f"{wide}/calibration", exist_ok=True)
    ext.to_csv(f"{wide}/calibration/front_wide_extrinsics.csv", index=False)
    ext.to_csv(f"{tele}/calibration/front_tele_extrinsics.csv", index=False)
    return Lw, Lt


@needs_stack
def test_encode_run_writes_tokens_masks_ledger_and_resumes(tmp_path, monkeypatch):
    clips = ["clipA-0001", "clipB-0002"]
    Lw, Lt = build_roots(tmp_path, clips)
    calls = []

    def fake_decode(mp4, need, threads=None):
        calls.append(list(need))
        g = torch.Generator().manual_seed(1)
        for i in need:
            yield int(i), (torch.rand(3, 54, 96, generator=g) * 255).to(torch.uint8)
    monkeypatch.setattr(TE, "decode_selected", fake_decode)
    out = str(tmp_path / "out")
    n_done, fails, no_tele, rc = TE.run_clips(clips + ["clipC-0003"], Lw, Lt, out, workers=2, min_free_gb=0.0, enc=StubEnc(), keep_windows=True)
    # clipC has wide timestamps? no: refused as a whole run
    assert rc == 4 and n_done == 0
    n_done, fails, no_tele, rc = TE.run_clips(clips, Lw, Lt, out, workers=2, min_free_gb=0.0, enc=StubEnc(), keep_windows=True)
    assert rc == 0 and n_done == 2 and not fails
    plan = E.frame_plan(Lw.ts(clips[0]))
    n = plan["n_cache"]
    tok = np.load(TE.tok_path(out, clips[0]))
    assert tok.shape == (n, 2, 65, 1024) and tok.dtype == np.float16
    z = np.load(TE.match_path(out, clips[0]))
    mask = z["mask"]
    assert mask.shape == (n,) and 0 < mask.sum() < n and z["tele_idx"].dtype == np.int32
    assert np.all(z["tele_idx"][~mask] == -1) and np.all(np.abs(z["dt_us"][mask]) <= 17000)
    assert np.all(tok[mask] == 3.0) and np.all(tok[~mask] == 0.0)           # zero token rows exactly where the mask is false
    needed = sorted({i for c in calls for i in c})
    gap_len = 40
    assert max(needed) < 180 - gap_len                                      # only frames of the shortened tele table are asked for
    man = json.load(open(os.path.join(out, "tok_T", "MANIFEST.json"), encoding="utf-8"))
    assert man["n_clips"] == 2 and man["clips"][clips[0]]["n_matched"] == int(mask.sum()) and man["windows"] == [[8.0, 0.0, -2.0], [4.0, 0.0, -1.0]]
    assert os.path.exists(os.path.join(out, "win", f"{clips[0]}.npy"))
    calls.clear()
    n2, f2, nt2, rc2 = TE.run_clips(clips, Lw, Lt, out, workers=2, min_free_gb=0.0, enc=StubEnc())
    assert rc2 == 0 and n2 == 0 and calls == []                              # resume: nothing is encoded again


@needs_stack
def test_encode_lists_a_clip_without_tele_and_refuses_a_pilot_frame_count_mismatch(tmp_path, monkeypatch):
    clips = ["clipA-0001", "clipB-0002"]
    Lw, Lt = build_roots(tmp_path, clips, no_mp4=(clips[1],))                # a clip with no staged tele video
    monkeypatch.setattr(TE, "decode_selected", lambda mp4, need, threads=None: ((int(i), torch.zeros(3, 54, 96, dtype=torch.uint8)) for i in need))
    out = str(tmp_path / "out")
    wtok = tmp_path / "wtok"
    wtok.mkdir()
    np.save(wtok / f"{clips[0]}.npy", np.zeros((7, 2, 65, 4), np.float16))   # PLANTED: the pilot file has another frame count
    n_done, fails, no_tele, rc = TE.run_clips(clips, Lw, Lt, out, workers=1, min_free_gb=0.0, enc=StubEnc(), wide_tok_dir=str(wtok))
    assert [c for c, _ in no_tele] == [clips[1]] and n_done == 0 and rc == 5 and "pilot token file has 7 frames" in fails[0][1]


@needs_stack
def test_limit_option_takes_the_first_n_clips(tmp_path, monkeypatch):
    clips = ["clipA-0001", "clipB-0002", "clipC-0003"]
    Lw, Lt = build_roots(tmp_path, clips)
    monkeypatch.setattr(TE, "decode_selected", lambda mp4, need, threads=None: ((int(i), torch.zeros(3, 54, 96, dtype=torch.uint8)) for i in need))
    out = str(tmp_path / "out")
    n_done, fails, no_tele, rc = TE.run_clips(clips, Lw, Lt, out, workers=1, min_free_gb=0.0, enc=StubEnc(), limit=1)
    assert n_done == 1 and rc == 0 and os.path.exists(TE.tok_path(out, clips[0])) and not os.path.exists(TE.tok_path(out, clips[1]))


# ------------------------------------------------------------------------------------------------ control t2
def smooth_image(seed, h=540, w=960, k=9):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, h, w, generator=g)
    x = torch.nn.functional.avg_pool2d(x, k, stride=1, padding=k // 2)
    x = (x - x.amin()) / (x.amax() - x.amin())
    return (x[0] * 255).to(torch.uint8)


@needs_stack
def test_t2_same_scene_passes_and_a_shifted_copy_loses():
    intr = synth_intr_tele()
    same = []
    shifted = []
    for seed in range(6):
        img = smooth_image(seed)
        b, s = T2.frame_corrs(img, img, intr, intr)
        same.append(T2.frame_passes(b, s))
        moved = torch.roll(img, shifts=32, dims=2)                          # about 2 degrees of azimuth at this scale (32 px)
        b2, s2 = T2.frame_corrs(img, moved, intr, intr)
        shifted.append(T2.frame_passes(b2, s2))
    ok, share, n = T2.decide(same)
    assert ok and share == 1.0 and n == 6
    ok2, share2, n2 = T2.decide(shifted)
    assert not ok2 and share2 < 0.9


def test_t2_decision_rule_threshold_and_helpers():
    assert T2.decide([True] * 9 + [False]) == (True, 0.9, 10)
    assert T2.decide([True] * 8 + [False] * 2)[0] is False
    assert T2.decide([])[0] is False
    assert T2.frame_passes(0.9, [0.1, 0.2, 0.3, 0.4]) and not T2.frame_passes(0.5, [0.1, 0.2, 0.5, 0.4])      # a tie is not higher
    assert not T2.frame_passes(float("nan"), [0.1] * 4)
    a = np.random.default_rng(0).random((8, 8))
    assert abs(T2.ncc(a, 3 * a + 7) - 1.0) < 1e-12 and abs(T2.ncc(a, -a) + 1.0) < 1e-12 and np.isnan(T2.ncc(a, np.ones((8, 8))))
    assert T2.pick_frames([False, True, True, True, True, True, True]) == [3, 5]
    assert T2.pick_frames([True] + [False] * 5) == []
    ids = [f"e{i}" for i in range(10)]
    got = T2.select_clips(ids, 3, lambda c: c != "e0")
    assert len(got) == 3 and "e0" not in got


@needs_stack
def test_encode_and_control_use_one_routine_for_the_carried_direction():
    rng = np.random.default_rng(5)
    for _ in range(20):
        q1, _r1 = np.linalg.qr(rng.normal(size=(3, 3)))
        q2, _r2 = np.linalg.qr(rng.normal(size=(3, 3)))
        a, e = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
        assert np.allclose(TE.point_in_other_cam(q1, q2, a, e), T2.point_in_other_cam(q1, q2, a, e))
    R = np.eye(3)
    assert np.allclose(TE.point_in_other_cam(R, R, 0.0, -1.0), (0.0, -1.0))


def test_windows_from_video_takes_the_carried_directions():
    import inspect
    assert list(inspect.signature(TE.windows_from_video).parameters) == ["vid", "intr", "frames", "dirs"]
    src = inspect.getsource(TE.make_tele_windows)
    assert "dirs = tele_dirs(clip, Lw, Lt)" in src and "windows_from_video(vid, intr, frames, dirs)" in src
    assert "TE.tele_dirs(clip, Lw, Lt)[0]" in inspect.getsource(T2.t2_clip)


def test_tele_dirs_equal_mounts_keep_the_numbers_and_a_missing_row_raises(tmp_path):
    Lw, Lt = build_roots(tmp_path, ["clipA-00"])
    d = TE.tele_dirs("clipA-00", Lw, Lt)
    assert np.allclose(d, [(0.0, -2.0), (0.0, -1.0)])
    with pytest.raises(Exception):
        TE.tele_dirs("clipZ-99", Lw, Lt)


def test_point_in_other_cam_is_identity_for_equal_mounts_and_turns_with_the_mount():
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    a, e = T2.point_in_other_cam(R, R, 0.0, -2.0)
    assert abs(a) < 1e-9 and abs(e + 2.0) < 1e-9
    th = np.radians(1.0)                                                    # the destination camera is pitched 1 degree further down
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    a2, e2 = T2.point_in_other_cam(R, R @ Rx, 0.0, -2.0)
    assert abs(abs(e2 - (-2.0)) - 1.0) < 1e-6 and abs(a2) < 1e-6


# ------------------------------------------------------------------------------------------------ the queue
BASH = shutil.which("bash")


def _bash_ok():
    if not BASH:
        return False
    try:
        r = subprocess.run([BASH, "-c", "echo ok"], capture_output=True, encoding="utf-8", timeout=60)
        return r.stdout.strip() == "ok"
    except Exception:                                                        # noqa: BLE001
        return False


needs_bash = pytest.mark.skipif(not _bash_ok(), reason="no working bash")
QUEUE = os.path.join(HERE, "pod_telegap_enc_queue.sh").replace("\\", "/")


def run_queue(tmp_path, extra_env, root_name="troot"):
    root = (tmp_path / root_name).as_posix()
    env = dict(os.environ)
    env.update({"TG_ROOT": root, "TG_FREE_PATH": tmp_path.as_posix(), "TG_MIN_FREE_GB": "0"})
    env.update(extra_env)
    r = subprocess.run([BASH, QUEUE], capture_output=True, encoding="utf-8", env=env, timeout=120)
    return r, root


@needs_bash
def test_queue_dry_run_prints_the_steps_in_order_and_creates_nothing(tmp_path):
    r, root = run_queue(tmp_path, {"DRY": "1"})
    assert r.returncode == 0, r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("CMD:")]
    joined = "\n".join(cmds)
    pos = [joined.index("tele_stage.py"), joined.index("--limit 3"), joined.index("ctl_t2.py"), joined.rindex("tele_encode.py --list")]
    assert pos == sorted(pos) and joined.count("tele_encode.py") == 2
    last_enc = [c for c in cmds if "tele_encode.py" in c][-1]
    assert "--limit" not in last_enc and "--keep-windows" not in last_enc and "--out " + root + "/enc_out" in last_enc
    assert "--keep-windows" in [c for c in cmds if "--limit 3" in c][0]
    assert "ZZ TELEGAP_ENC DONE" in cmds[-1]
    assert not os.path.exists(root)


@needs_bash
def test_queue_refuses_low_disk_and_an_existing_start_marker(tmp_path):
    r, root = run_queue(tmp_path, {"DRY": "1", "TG_MIN_FREE_GB": "999999999"})
    assert r.returncode == 2 and "refused" in r.stdout and not os.path.exists(root)
    os.makedirs(root := (tmp_path / "started").as_posix())
    open(root + "/TELEGAP_ENC_START.txt", "w").write("1 2026-11-22T00:00:00Z pid 1\n")
    r2, _ = run_queue(tmp_path, {"DRY": "0"}, root_name="started")
    assert r2.returncode == 3 and "refused" in r2.stdout
    assert sorted(os.listdir(root)) == ["TELEGAP_ENC_START.txt"]            # nothing else was created


def test_queue_text_has_the_marker_names_limits_and_no_delete_or_move():
    txt = open(os.path.join(HERE, "pod_telegap_enc_queue.sh"), encoding="utf-8").read()
    for k in ("TELEGAP_ENC_START.txt", "TELEGAP_ENC_DONE.txt", "TELEGAP_ENC_STOP.txt", "TG_STOP_S:-14400", "TG_MAX_GB:-40", "memory.max", "memory.stat",
              "set -C", "all steps done", "ZZ TELEGAP_ENC DONE", "TG_ROOT:-/workspace/v8proof/telegap"):
        assert k in txt, k
    code = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("#")]
    for ln in code:
        toks = ln.replace(";", " ").replace("(", " ").replace(")", " ").replace("|", " ").split()
        assert "rm" not in toks and "mv" not in toks, ln


# ------------------------------------------------------------------------------------------------ hygiene
def test_banned_words_do_not_appear_in_the_kit_files():
    words = ["vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no lon" + "ger", "closed " + "loop"]
    for f in sorted(os.listdir(HERE)):
        if f.endswith((".py", ".sh", ".md")):
            txt = open(os.path.join(HERE, f), encoding="utf-8").read().lower()
            for w in words:
                assert w not in txt, (f, w)
    for f in sorted(os.listdir(HERE)):
        if f.endswith((".py", ".sh", ".md")):
            assert len(os.path.join(HERE, f)) <= 200
