"""Tests for ``tanitad.data.semantic_map_gt`` -- the SAM3 map GT reader (refcv6-grounded).

(a) every refusal on synthetic fixtures, and on a REAL published GT file, including
    the mutation the brief names: a GT file renamed to another clip id MUST be refused;
(e) real-data alignment: the reader's 1 ms time check on the episode frame timestamps
    reconstructed from the camera timestamps, and the pose-content check against the
    v2ep cache payload, on every local clip that has BOTH a GT file and a v2ep episode.

Real data is found through environment variables (dev-box defaults below) and the
real tests SKIP, naming what is missing, when it is absent. Clip ids are derived at
runtime from file names and never written into this file or its messages (sha12 only).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tanitad.data import semantic_map_gt as G

SAM3_GT_ROOT = Path(os.environ.get("TANITAD_SAM3_GT_ROOT",
                                   "C:/Users/Admin/tanitad-caches/refcv6g-20260915"))
V2EP_DIR = Path(os.environ.get("TANITAD_V2EP_EVAL_DIR",
                               "D:/Projects/TanitAD-artifacts/refcv5cmp/data/eval"))
CAM_TS_DIR = Path(os.environ.get(
    "TANITAD_CAM_TS_DIR",
    "C:/Users/Admin/tanitad-data/physicalai-b1/r0/camera_front_wide_120fov"))


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #
def _meta(clip_id: str, **over) -> dict:
    m = {"schema": G.SCHEMA, "frame": "rig",
         "cartesian": {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5,
                       "shape": [120, 64], "row0": "x in [0, cell_m)"},
         "channels": list(G.CHANNELS), "fraction_scale": 255,
         "source": {"clip_sha12": G.sha12(clip_id), "split": "train"}}
    m.update(over)
    return m


def _write_gt(root: Path, clip_id: str, T: int = 6, meta: dict | None = None,
              arrays: dict | None = None, name: str | None = None) -> Path:
    """A schema-/2-shaped file. Frames on a 10 Hz grid over a jittered 30 Hz camera."""
    t_cam = np.arange(3 * T + 3, dtype=np.float64) * 33_333.0 + 1_000.0
    t_cam[1::2] += 250.0
    t_query = np.linspace(t_cam[0], t_cam[0] + (T - 1) * 100_000.0, T)
    idx = np.searchsorted(t_cam, t_query)
    cart = np.zeros((T, 9, 120, 64), np.uint8)
    cart[:, 1] = 255                                   # fully seen drivable, except row 0:
    cart[:, [1, 8], 0, 0] = (128, 127)                 # not seen 127 -> seen share 0.502
    cart[:, [1, 8], 0, 1] = (127, 128)                 # not seen 128 -> seen share 0.498
    cart[:, [1, 8], 0, 2] = (0, 255)                   # never seen
    cart[:, [0, 1], 0, 3] = (127, 128)                 # seen, half without a class
    for t in range(T):
        cart[t, [1, 2], 5, 7] = (255 - t, t)           # a per-frame marker
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, 0, 3] = 10.0 * (t_cam[idx] - t_cam[0]) / 1e6   # 10 m/s along +x
    arr = {"meta_json": np.array(json.dumps(meta or _meta(clip_id))),
           "t_query_us": t_query, "t_img_us": t_cam[idx].astype(np.int64),
           "cam_frame_idx": idx.astype(np.int32), "T_world_rig": pose,
           "cart_frac": cart, "polar_frac": np.zeros((T, 9, 24, 20), np.uint8),
           "polar48_frac": np.zeros((T, 9, 48, 40), np.uint8),
           "fine_codes": np.zeros((T, 600, 320), np.uint8)}
    arr.update(arrays or {})
    arr = {k: v for k, v in arr.items() if v is not None}
    d = root.joinpath(*G.GT_SUBDIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name or clip_id}{G.GT_SUFFIX}"
    np.savez_compressed(p, **arr)
    return p


CLIP_A, CLIP_B = "synthetic-clip-a", "synthetic-clip-b"


def test_reads_fractions_and_seen_mask(tmp_path):
    _write_gt(tmp_path, CLIP_A)
    c = G.open_clip(tmp_path, CLIP_A)
    assert c.n_frames == 6 and c.clip_sha12 == G.sha12(CLIP_A)
    m = c.read([4, 1])
    assert m.cart.shape == (2, 9, 120, 64) and m.cart.dtype == np.float32
    assert m.seen.shape == (2, 120, 64) and m.seen.dtype == np.bool_
    assert m.cart[0, 2, 5, 7] == np.float32(4 / 255) and m.cart[1, 2, 5, 7] == np.float32(1 / 255)
    assert float(m.cart.min()) >= 0.0 and float(m.cart.max()) <= 1.0
    # the >= 0.5 seen boundary, exactly: not_seen 127 -> seen, 128 -> not seen
    assert m.seen[:, 0, 0].all() and not m.seen[:, 0, 1].any()
    assert not m.seen[:, 0, 2].any() and m.seen[:, 0, 3].all() and m.seen[:, 5, 5].all()
    np.testing.assert_array_equal(m.frame_idx, [4, 1])
    np.testing.assert_array_equal(m.t_img_us, c.t_img_us[[4, 1]])
    assert c.read(3).cart.shape == (1, 9, 120, 64)     # a scalar is one frame


@pytest.mark.parametrize("bad", [
    {"schema": "tanitad.sam3_map_gt/1"},
    {"frame": "world"},
    {"channels": [G.CHANNELS[1], G.CHANNELS[0]] + list(G.CHANNELS[2:])},
    {"fraction_scale": 100},
    {"cartesian": {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.25, "shape": [240, 128]}},
])
def test_refuses_wrong_schema(tmp_path, bad):
    _write_gt(tmp_path, CLIP_A, meta=_meta(CLIP_A, **bad))
    with pytest.raises(G.SchemaMismatch):
        G.open_clip(tmp_path, CLIP_A)


@pytest.mark.parametrize("arrays", [
    {"cart_frac": np.zeros((6, 9, 64, 120), np.uint8)},        # transposed grid
    {"cart_frac": np.zeros((6, 9, 120, 64), np.float32)},      # not uint8
    {"cart_frac": None},                                       # missing
    {"t_img_us": np.zeros(5, np.int64)},                        # frame count differs
    {"meta_json": None},
])
def test_refuses_wrong_arrays(tmp_path, arrays):
    _write_gt(tmp_path, CLIP_A, arrays=arrays)
    with pytest.raises(G.SchemaMismatch):
        G.open_clip(tmp_path, CLIP_A)


def test_refuses_non_monotone_time_axis(tmp_path):
    t = np.array([0, 100_000, 50_000, 300_000, 400_000, 500_000], np.int64)
    _write_gt(tmp_path, CLIP_A, arrays={"t_img_us": t})
    with pytest.raises(G.SchemaMismatch):
        G.open_clip(tmp_path, CLIP_A)


def test_refuses_corrupt_archive(tmp_path):
    p = _write_gt(tmp_path, CLIP_A)
    p.write_bytes(b"not a zip archive")
    with pytest.raises(G.SchemaMismatch):
        G.open_clip(tmp_path, CLIP_A)


def test_MUTATION_file_renamed_to_another_clip_is_refused(tmp_path):
    """The brief's mutation: A's labels under B's name must never read as B's."""
    p = _write_gt(tmp_path, CLIP_A)
    G.open_clip(tmp_path, CLIP_A)                              # control: A reads
    p.rename(G.gt_path(tmp_path, CLIP_B))
    with pytest.raises(G.ClipIdentityMismatch) as ei:
        G.open_clip(tmp_path, CLIP_B)
    assert G.sha12(CLIP_A) in str(ei.value) and CLIP_B not in str(ei.value)
    with pytest.raises(FileNotFoundError):
        G.open_clip(tmp_path, CLIP_A)


@pytest.mark.parametrize("idx", [6, -1, [0, 6], 2.0, [True, False], [[0, 1]]])
def test_refuses_out_of_range_or_malformed_indices(tmp_path, idx):
    _write_gt(tmp_path, CLIP_A)
    c = G.open_clip(tmp_path, CLIP_A)
    with pytest.raises(G.FrameIndexError):
        c.read(idx)


def test_refuses_time_misalignment(tmp_path):
    _write_gt(tmp_path, CLIP_A)
    c = G.open_clip(tmp_path, CLIP_A)
    idx = np.array([0, 2, 5])
    t = c.t_img_us[idx].astype(np.float64)
    assert c.read(idx, t).cart.shape[0] == 3                    # exact
    assert c.read(idx, t + 999.0).cart.shape[0] == 3            # inside 1 ms
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, t + 1001.0)                                 # outside 1 ms
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, c.t_img_us[idx + [1, 1, 0]])                # one frame off
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, t[:2])                                      # wrong length
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, [t[0], math.nan, t[2]])


def test_pose_alignment_check(tmp_path):
    _write_gt(tmp_path, CLIP_A, T=12)
    c = G.open_clip(tmp_path, CLIP_A)
    xy = c.T_world_rig[:, :2, 3]
    # v2ep poses at t_query: back the GT pose up by v * (t_img - t_query)
    dt = (c.t_img_us - c.t_query_us) / 1e6
    poses = np.stack([xy[:, 0] - 10.0 * dt, xy[:, 1], np.zeros(12), np.full(12, 10.0)], 1)
    rep = c.check_pose_alignment(poses)
    assert rep["verdict"] == "ALIGNED" and rep["aligned_max_m"] < 1e-6
    with pytest.raises(G.TimeMisalignment):
        c.check_pose_alignment(np.roll(poses, 1, axis=0))       # frame axis shifted
    with pytest.raises(G.TimeMisalignment):
        c.check_pose_alignment(poses[:-1])                      # different episode grid
    still = poses.copy()
    still[:, 0], still[:, 3] = 0.0, 0.0
    c.T_world_rig[:, 0, 3] = 0.0
    assert c.check_pose_alignment(still)["verdict"] == "INCONCLUSIVE"


def test_coverage_three_states(tmp_path):
    _write_gt(tmp_path, CLIP_A)                                       # readable
    _write_gt(tmp_path, "synthetic-clip-c").write_bytes(b"garbage")    # corrupt
    _write_gt(tmp_path, CLIP_A, name="synthetic-clip-d")              # another clip's file
    ids = [CLIP_A, CLIP_B, "synthetic-clip-c", "synthetic-clip-d", CLIP_A]
    cov = G.coverage(ids, tmp_path)
    st = {k: v["state"] for k, v in cov["states"].items()}
    assert st == {G.sha12(CLIP_A): "readable", G.sha12(CLIP_B): "absent",
                  G.sha12("synthetic-clip-c"): "inconclusive",
                  G.sha12("synthetic-clip-d"): "inconclusive"}
    assert (cov["n"], cov["n_duplicates_ignored"], cov["n_readable"], cov["n_absent"],
            cov["n_inconclusive"]) == (4, 1, 1, 1, 2)
    assert cov["frac_readable"] == 0.25 and cov["frac_readable_upper"] == 0.75
    assert cov["verdict"] == "INCONCLUSIVE"
    assert not any(c in json.dumps(cov) for c in ids)                  # sha12 only
    # an unreadable / wrong root is never "absent"
    for bad_root in (tmp_path / "no-such-root", G.gt_path(tmp_path, CLIP_A)):
        cov = G.coverage([CLIP_A, CLIP_B], bad_root)
        assert cov["n_absent"] == 0 and cov["n_inconclusive"] == 2
    cov = G.coverage([CLIP_A], tmp_path)
    assert cov["verdict"] == "COMPLETE" and cov["frac_readable"] == 1.0


def test_episode_grid_restatement_and_stacked_rows():
    t = np.arange(61, dtype=np.float64) * 33_366.0 - 80_260.0          # 2.0 s at ~30 Hz
    g = G.episode_frame_times_us(t)
    assert len(g["t_img_us"]) == 20                                    # int(2.00196 * 10)
    np.testing.assert_array_equal(g["cam_frame_idx"][:4], [0, 4, 7, 10])
    assert np.all(g["t_img_us"] >= g["t_query_us"]) and np.all(np.diff(g["t_img_us"]) > 0)
    assert np.all(g["t_img_us"] - g["t_query_us"] < 33_366.0)
    with pytest.raises(ValueError):
        G.episode_frame_times_us(t / 1000.0)                           # milliseconds
    np.testing.assert_array_equal(G.raw_frame_index([0, 5]), [2, 7])
    np.testing.assert_array_equal(G.raw_frame_index([0, 5], n_stack=1), [0, 5])


# --------------------------------------------------------------------------- #
# real published GT files (a) and real-data alignment (e)                      #
# --------------------------------------------------------------------------- #
def _real_clip_ids() -> list[str]:
    d = SAM3_GT_ROOT.joinpath(*G.GT_SUBDIR)
    if not d.is_dir():
        return []
    return sorted(p.name[: -len(G.GT_SUFFIX)] for p in d.glob(f"*{G.GT_SUFFIX}"))


REAL_IDS = _real_clip_ids()
needs_real = pytest.mark.skipif(len(REAL_IDS) < 2, reason=(
    f"needs >= 2 real GT files under $TANITAD_SAM3_GT_ROOT/semantic_maps/gt "
    f"(found {len(REAL_IDS)})"))


@needs_real
def test_real_file_reads_and_seen_mask_matches_channel_8():
    for cid in REAL_IDS:
        c = G.open_clip(SAM3_GT_ROOT, cid)
        m = c.read(np.arange(c.n_frames))
        assert m.cart.shape == (c.n_frames, 9, 120, 64)
        np.testing.assert_array_equal(m.seen, (1.0 - m.cart[:, 8]) >= 0.5)
        s = m.cart.sum(1)          # 9 channels each rounded to 1/255: |sum - 1| <= 4/255
        assert float(np.abs(s - 1.0).max()) <= 4.0 / 255 + 1e-6
        assert 0.5 < float(m.seen.mean()) < 1.0


@needs_real
def test_real_MUTATION_renamed_and_reschema_refused(tmp_path):
    src = G.gt_path(SAM3_GT_ROOT, REAL_IDS[0])
    other = REAL_IDS[1]
    dst = G.gt_path(tmp_path, other)
    dst.parent.mkdir(parents=True)
    shutil.copyfile(src, dst)                               # clip 0's labels, clip 1's name
    with pytest.raises(G.ClipIdentityMismatch):
        G.open_clip(tmp_path, other)
    assert G.coverage([other], tmp_path)["states"][G.sha12(other)]["state"] == "inconclusive"
    # same real arrays, schema string changed -> refused
    with np.load(src, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr["meta_json"]))
    meta["schema"] = "tanitad.sam3_map_gt/3"
    arr["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(G.gt_path(tmp_path, REAL_IDS[0]), **arr)
    with pytest.raises(G.SchemaMismatch):
        G.open_clip(tmp_path, REAL_IDS[0])
    c = G.open_clip(SAM3_GT_ROOT, REAL_IDS[0])
    with pytest.raises(G.FrameIndexError):
        c.read([c.n_frames])


def _aligned_real_ids() -> list[str]:
    return [c for c in REAL_IDS
            if (V2EP_DIR / f"{c}.v2ep.pt").is_file()
            and (CAM_TS_DIR / f"{c}.timestamps.parquet").is_file()]


@needs_real
def test_real_alignment_time_and_pose_on_v2ep_episodes():
    """(e): every local clip with a GT file AND a v2ep episode AND camera timestamps."""
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    ids = _aligned_real_ids()
    if len(ids) < 2:
        pytest.skip(f"needs >= 2 clips with GT + v2ep ($TANITAD_V2EP_EVAL_DIR) + camera "
                    f"timestamps ($TANITAD_CAM_TS_DIR); found {len(ids)}")
    for cid in ids:
        c = G.open_clip(SAM3_GT_ROOT, cid)
        payload = torch.load(V2EP_DIR / f"{cid}.v2ep.pt", map_location="cpu",
                             weights_only=False, mmap=True)
        T = len(payload["jpeg_len"])
        ts = pd.read_parquet(CAM_TS_DIR / f"{cid}.timestamps.parquet")
        tcol = next(k for k in ts.columns if "time" in k.lower())   # as _resampled does
        grid = G.episode_frame_times_us(ts[tcol].to_numpy())
        assert T == c.n_frames == len(grid["t_img_us"]), c.clip_sha12
        idx = np.arange(T)
        assert c.check_times(idx, grid["t_img_us"]) == 0.0, c.clip_sha12
        # a training window: stacked rows 10..19 read raw frames 12..21
        raw = G.raw_frame_index(np.arange(10, 20), int(payload["n_stack"]))
        assert c.read(raw, grid["t_img_us"][raw]).cart.shape[0] == 10
        with pytest.raises(G.TimeMisalignment):                 # the off-by-one row
            c.read(raw - 1, grid["t_img_us"][raw])
        rep = c.check_pose_alignment(payload["poses"].numpy())
        assert rep["verdict"] == "ALIGNED", (c.clip_sha12, rep)
        with pytest.raises(G.TimeMisalignment):
            c.check_pose_alignment(np.roll(payload["poses"].numpy(), 1, axis=0))
