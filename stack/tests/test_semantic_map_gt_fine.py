"""refcv7 NEW-2 -- the 10 cm SAM3 map reader (``tanitad.data.semantic_map_gt_fine``).

What is pinned, each with a literal expectation:

1. **A refusal per guard** -- schema, clip identity, a missing / mis-typed /
   mis-shaped ``fine_codes``, a different ``fine`` spec (every key), the 0.5 m <->
   0.1 m relation, the world-map resolution, out-of-range / malformed frame
   indices, the 1 ms time check, and an out-of-legend code.
2. **The analytic control on REAL eval GT files** -- ``cart_frac`` equals the
   5 x 5 block fraction of ``fine_codes`` (``rint(count * 255 / 25)``) on every
   compared cell, and the SAME control reads NON-zero on a left-right mirrored
   10 cm map (so it can see an orientation error; a control that can only pass
   certifies nothing).
3. **The deliberate-regression arm** -- the spec check removed, a file with a
   different 10 cm spec is read as if it were ours: the refusal test must go RED.

Real data is found through environment variables (dev-box defaults below) and the
real tests SKIP, naming what is missing. Clip ids are derived at runtime from the
v2ep file names and never written here or printed (sha12 only).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tanitad.data import semantic_map_gt as G
from tanitad.data import semantic_map_gt_fine as F

FINE_DIR = Path(os.environ.get("TANITAD_SAM3_FINE_DIR",
                               "D:/refcv6_eval_kit/data/sam3_gt_eval_thor137"))
V2EP_DIR = Path(os.environ.get("TANITAD_V2EP_FINE_DIR",
                               "D:/refcv6_eval_kit/data/refcv6-b1-416x1024-eval139"))
CLIP_A, CLIP_B = "synthetic-fine-a", "synthetic-fine-b"


# --------------------------------------------------------------------------- #
# synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #
def _meta(clip_id: str, fine=None, **over) -> dict:
    m = {"schema": G.SCHEMA, "frame": "rig",
         "cartesian": {"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5,
                       "shape": [120, 64], "row0": "x in [0, cell_m)"},
         "fine": dict(F.FINE_SPEC) if fine is None else fine,
         "sub_samples_per_axis": 5,
         "channels": list(G.CHANNELS), "fraction_scale": 255,
         "source": {"clip_sha12": G.sha12(clip_id), "split": "train",
                    "world_map_res_m": 0.1}}
    m.update(over)
    return m


def _codes(T: int) -> np.ndarray:
    """A legal 10 cm map: drivable, a 2-cell lane line at cols 150-151, a
    not-seen block near the ego, and a per-frame marker cell."""
    c = np.full((T, 600, 320), 1, np.uint8)
    c[:, :, 150:152] = 2
    c[:, :10, :40] = 255
    c[:, 300:320, 200:260] = 7
    for t in range(T):
        c[t, 500, 5 + t] = 3
    return c


def _write(root: Path, clip_id: str, T: int = 6, meta: dict | None = None,
           arrays: dict | None = None, name: str | None = None) -> Path:
    t_cam = np.arange(3 * T + 3, dtype=np.float64) * 33_333.0 + 1_000.0
    t_query = np.linspace(t_cam[0], t_cam[0] + (T - 1) * 100_000.0, T)
    idx = np.searchsorted(t_cam, t_query)
    fine = _codes(T)
    pose = np.tile(np.eye(4), (T, 1, 1))
    pose[:, 0, 3] = 10.0 * (t_cam[idx] - t_cam[0]) / 1e6
    arr = {"meta_json": np.array(json.dumps(meta or _meta(clip_id))),
           "t_query_us": t_query, "t_img_us": t_cam[idx].astype(np.int64),
           "cam_frame_idx": idx.astype(np.int32), "T_world_rig": pose,
           "cart_frac": F.block_fraction_u8(fine),
           "polar_frac": np.zeros((T, 9, 24, 20), np.uint8),
           "polar48_frac": np.zeros((T, 9, 48, 40), np.uint8),
           "fine_codes": fine}
    arr.update(arrays or {})
    arr = {k: v for k, v in arr.items() if v is not None}
    d = root.joinpath(*G.GT_SUBDIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name or clip_id}{G.GT_SUFFIX}"
    np.savez_compressed(p, **arr)
    return p


# --------------------------------------------------------------------------- #
# 0. it reads                                                                  #
# --------------------------------------------------------------------------- #
def test_reads_codes_and_seen(tmp_path):
    _write(tmp_path, CLIP_A)
    c = F.open_clip_fine(tmp_path, CLIP_A)
    assert c.n_frames == 6 and c.clip_sha12 == G.sha12(CLIP_A)
    m = c.read([4, 1])
    assert m.codes.shape == (2, 600, 320) and m.codes.dtype == np.uint8
    assert m.seen.shape == (2, 600, 320) and m.seen.dtype == np.bool_
    assert int(m.codes[0, 500, 9]) == 3 and int(m.codes[1, 500, 6]) == 3
    assert int(m.codes[0, 0, 0]) == 255 and not m.seen[0, 0, 0]
    assert int(m.codes[0, 100, 150]) == 2 and m.seen[0, 100, 150]
    np.testing.assert_array_equal(m.frame_idx, [4, 1])
    np.testing.assert_array_equal(m.t_img_us, c.t_img_us[[4, 1]])
    assert c.read(3).codes.shape == (1, 600, 320)


def test_block_fraction_rule_is_rint_of_count_times_10_2():
    """Literal values of the exporter's rule for every count 0..25."""
    want = [0, 10, 20, 31, 41, 51, 61, 71, 82, 92, 102, 112, 122, 133, 143,
            153, 163, 173, 184, 194, 204, 214, 224, 235, 245, 255]
    got = []
    for k in range(26):
        c = np.full((1, 600, 320), 1, np.uint8)
        blk = c[0, :5, :5].reshape(-1)
        blk[:k] = 2
        c[0, :5, :5] = blk.reshape(5, 5)
        got.append(int(F.block_fraction_u8(c)[0, 2, 0, 0]))
    assert got == want


# --------------------------------------------------------------------------- #
# 1. a refusal per guard                                                       #
# --------------------------------------------------------------------------- #
def test_refuses_wrong_schema(tmp_path):
    _write(tmp_path, CLIP_A, meta=_meta(CLIP_A, schema="tanitad.sam3_map_gt/1"))
    with pytest.raises(G.SchemaMismatch):
        F.open_clip_fine(tmp_path, CLIP_A)


def test_refuses_file_of_another_clip(tmp_path):
    p = _write(tmp_path, CLIP_A)
    F.open_clip_fine(tmp_path, CLIP_A)                       # control: A reads
    p.rename(G.gt_path(tmp_path, CLIP_B))
    with pytest.raises(G.ClipIdentityMismatch) as ei:
        F.open_clip_fine(tmp_path, CLIP_B)
    assert G.sha12(CLIP_A) in str(ei.value) and CLIP_B not in str(ei.value)


@pytest.mark.parametrize("arrays", [
    {"fine_codes": None},                                        # absent
    {"fine_codes": np.zeros((6, 600, 320), np.uint16)},          # dtype
    {"fine_codes": np.zeros((6, 320, 600), np.uint8)},           # transposed
    {"fine_codes": np.zeros((5, 600, 320), np.uint8)},           # frame count
    {"fine_codes": np.zeros((6, 300, 160), np.uint8)},           # 0.2 m grid
])
def test_refuses_bad_fine_array(tmp_path, arrays):
    _write(tmp_path, CLIP_A, arrays=arrays)
    G.open_clip(tmp_path, CLIP_A)                 # control: the 0.5 m reader is fine
    with pytest.raises(F.FineSpecMismatch):
        F.open_clip_fine(tmp_path, CLIP_A)


BAD_SPECS = [("cell_m", 0.05), ("cell_m", 0.2), ("x_max_m", 50.0),
             ("y_half_m", 20.0), ("shape", [320, 600]),
             ("codes", "0-8 class, 255 not seen")]


def _assert_refuses_every_wrong_spec(tmp_path) -> None:
    for i, (key, val) in enumerate(BAD_SPECS):
        root = tmp_path / f"spec{i}"
        fine = dict(F.FINE_SPEC)
        fine[key] = val
        _write(root, CLIP_A, meta=_meta(CLIP_A, fine=fine))
        with pytest.raises(F.FineSpecMismatch):
            F.open_clip_fine(root, CLIP_A)
    root = tmp_path / "nospec"
    m = _meta(CLIP_A)
    m.pop("fine")
    _write(root, CLIP_A, meta=m)
    with pytest.raises(F.FineSpecMismatch):
        F.open_clip_fine(root, CLIP_A)


def test_refuses_a_different_fine_spec(tmp_path):
    _assert_refuses_every_wrong_spec(tmp_path)


def test_DELIBERATE_REGRESSION_spec_check_removed_goes_RED(tmp_path, monkeypatch):
    """The reader with its spec check removed reads a 0.05 m file as ours: the
    refusal test above must FAIL (pytest's DID-NOT-RAISE), not pass."""
    monkeypatch.setattr(F, "_check_fine_meta", lambda *a, **k: None)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        _assert_refuses_every_wrong_spec(tmp_path)


@pytest.mark.parametrize("over", [
    {"sub_samples_per_axis": 4},
    {"sub_samples_per_axis": None},
])
def test_refuses_a_broken_coarse_fine_relation(tmp_path, over):
    m = _meta(CLIP_A)
    m.update(over)
    _write(tmp_path, CLIP_A, meta=m)
    with pytest.raises(F.FineSpecMismatch):
        F.open_clip_fine(tmp_path, CLIP_A)


@pytest.mark.parametrize("wres", [0.2, 0.5, None])
def test_refuses_a_coarser_world_map(tmp_path, wres):
    m = _meta(CLIP_A)
    m["source"] = dict(m["source"], world_map_res_m=wres)
    _write(tmp_path, CLIP_A, meta=m)
    with pytest.raises(F.FineSpecMismatch):
        F.open_clip_fine(tmp_path, CLIP_A)


@pytest.mark.parametrize("idx", [6, -1, [0, 6], 2.0, [True, False], [[0, 1]]])
def test_refuses_out_of_range_or_malformed_indices(tmp_path, idx):
    _write(tmp_path, CLIP_A)
    c = F.open_clip_fine(tmp_path, CLIP_A)
    with pytest.raises(G.FrameIndexError):
        c.read(idx)


def test_refuses_time_misalignment(tmp_path):
    _write(tmp_path, CLIP_A)
    c = F.open_clip_fine(tmp_path, CLIP_A)
    idx = np.array([0, 2, 5])
    t = c.t_img_us[idx].astype(np.float64)
    assert c.read(idx, t).codes.shape[0] == 3                 # exact
    assert c.read(idx, t + 999.0).codes.shape[0] == 3         # inside 1 ms
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, t + 1001.0)                               # outside 1 ms
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, c.t_img_us[idx + [1, 1, 0]])              # one frame off
    with pytest.raises(G.TimeMisalignment):
        c.read(idx, t[:2])                                    # wrong length


@pytest.mark.parametrize("bad_code", [8, 9, 128, 254])
def test_refuses_a_code_outside_the_legend(tmp_path, bad_code):
    fine = _codes(6)
    fine[3, 200, 100] = bad_code
    _write(tmp_path, CLIP_A, arrays={"fine_codes": fine,
                                     "cart_frac": F.block_fraction_u8(_codes(6))})
    c = F.open_clip_fine(tmp_path, CLIP_A)
    assert c.read([0, 1, 2]).codes.shape[0] == 3     # clean frames read
    with pytest.raises(F.FineCodeError):
        c.read([2, 3])


def test_store_resolves_the_same_three_layouts(tmp_path):
    from tanitad.data.perception_targets import MapGTStore
    flat = tmp_path / "flat"
    flat.mkdir()
    p = _write(tmp_path / "canon", CLIP_A)
    shutil.copyfile(p, flat / f"{G.sha12(CLIP_A)}{G.GT_SUFFIX}")
    for root, layout in ((tmp_path / "canon", "canonical"), (flat, "flat_sha12")):
        st = F.FineMapGTStore(root, max_open=1)
        m = st.frames_for_windows(CLIP_A, [0, 3], n_stack=3)   # raw 2, 5
        np.testing.assert_array_equal(m.frame_idx, [2, 5])
        assert st.layout_of[G.sha12(CLIP_A)] == layout
        assert MapGTStore(root).resolve(CLIP_A) == st.resolve(CLIP_A)
    with pytest.raises(FileNotFoundError):
        F.FineMapGTStore(flat).open(CLIP_B)


def test_band_rows_are_the_spec_bands():
    assert F.BAND_ROWS == ((0, 200), (200, 400), (400, 600))
    np.testing.assert_array_equal(F.band_of_row([0, 199, 200, 399, 400, 599]),
                                  [0, 0, 1, 1, 2, 2])
    with pytest.raises(ValueError):
        F.band_of_row([600])


# --------------------------------------------------------------------------- #
# 2. real eval GT files: identity, frame axis, and the analytic control        #
# --------------------------------------------------------------------------- #
def _real_pairs() -> list[tuple[str, Path]]:
    """(clip_id, gt path) for eval GT files whose clip id is known from a v2ep
    file name. Never printed."""
    if not FINE_DIR.is_dir() or not V2EP_DIR.is_dir():
        return []
    by12 = {G.sha12(p.name[:-len(".v2ep.pt")]): p.name[:-len(".v2ep.pt")]
            for p in V2EP_DIR.glob("*.v2ep.pt")}
    out = []
    for p in sorted(FINE_DIR.glob(f"*{G.GT_SUFFIX}")):
        s12 = p.name[:-len(G.GT_SUFFIX)]
        if s12 in by12:
            out.append((by12[s12], p))
    return out


REAL = _real_pairs()
needs_real = pytest.mark.skipif(len(REAL) < 2, reason=(
    f"needs >= 2 eval GT files ($TANITAD_SAM3_FINE_DIR={FINE_DIR}) with a v2ep "
    f"payload ($TANITAD_V2EP_FINE_DIR={V2EP_DIR}); found {len(REAL)}"))


@needs_real
def test_real_ANALYTIC_CONTROL_cart_frac_is_the_5x5_block_fraction():
    """Every eval file, every 50th frame: 0 differing cells, all 9 channels. And
    the control DISCRIMINATES: mirrored left-right it reads > 0 differing cells
    on every file (a control that only ever reads 0 certifies nothing)."""
    n_cells = n_bad = 0
    mirrored_bad_min = None
    for cid, p in REAL:
        c = F.open_path_fine(p, cid)
        idx = np.arange(0, c.n_frames, 50)
        fine = c.read(idx).codes
        cart = c.base.cart_u8()[idx]
        got = F.block_fraction_u8(fine)
        n_bad += int((got != cart).sum())
        n_cells += int(cart.size)
        mb = int((F.block_fraction_u8(fine[:, :, ::-1]) != cart).sum())
        mirrored_bad_min = mb if mirrored_bad_min is None else min(mirrored_bad_min, mb)
    assert n_cells > 1_000_000
    assert n_bad == 0, f"{n_bad} of {n_cells} cells differ"
    assert mirrored_bad_min > 0


@needs_real
def test_real_codes_are_in_the_legend_and_identity_is_content():
    for cid, p in REAL[:12]:
        c = F.open_path_fine(p, cid)
        m = c.read(np.arange(c.n_frames))
        u = set(np.unique(m.codes).tolist())
        assert u <= set(range(8)) | {255}, (c.clip_sha12, sorted(u))
        assert 0.3 < float(m.seen.mean()) < 1.0
    (cid0, p0), (cid1, _p1) = REAL[0], REAL[1]
    with pytest.raises(G.ClipIdentityMismatch):
        F.open_path_fine(p0, cid1)                 # clip 0's file asked as clip 1


@needs_real
def test_real_frame_axis_follows_the_v2ep_poses():
    torch = pytest.importorskip("torch")
    for cid, p in REAL[:6]:
        c = F.open_path_fine(p, cid)
        payload = torch.load(V2EP_DIR / f"{cid}.v2ep.pt", map_location="cpu",
                             weights_only=False, mmap=True)
        poses = payload["poses"].numpy()
        assert len(poses) == c.n_frames, c.clip_sha12
        rep = c.check_pose_alignment(poses)
        assert rep["verdict"] in ("ALIGNED", "INCONCLUSIVE"), (c.clip_sha12, rep)
        if rep["verdict"] == "ALIGNED":
            with pytest.raises(G.TimeMisalignment):
                c.check_pose_alignment(np.roll(poses, 1, axis=0))


@needs_real
def test_real_wrong_spec_copy_is_refused(tmp_path):
    cid, p = REAL[0]
    with np.load(p, allow_pickle=False) as z:
        arr = {k: z[k] for k in z.files}
    meta = json.loads(str(arr["meta_json"]))
    meta["fine"]["cell_m"] = 0.05
    arr["meta_json"] = np.array(json.dumps(meta))
    dst = tmp_path / p.name
    np.savez_compressed(dst, **arr)
    G.open_path(dst, cid)                       # the 0.5 m reader still accepts it
    with pytest.raises(F.FineSpecMismatch):
        F.open_path_fine(dst, cid)


# --------------------------------------------------------------------------- #
# 3. SPEC_REFCV7 §11.2 / §12 (A6/A7): the EXTENT, the /3 schema, anchoring    #
# --------------------------------------------------------------------------- #
A7 = F.EXTENT_REFCV7


def test_extent_literals_A7():
    assert (A7.x_max_m, A7.y_half_m) == (100.0, 30.0)
    assert A7.fine_shape == (1000, 600) and A7.grid(0.25).shape == (400, 240)
    assert A7.grid(0.5).shape == (200, 120)
    assert A7.band_keys == ("0_20", "20_40", "40_60", "60_80", "80_100")
    assert [b[3:] for b in A7.bands] == [(0, 200), (200, 400), (400, 600), (600, 800),
                                         (800, 1000)]
    assert A7.anchor_offset(0.1) == 140 and A7.anchor_offset(0.5) == 28
    assert F.EXTENT_V2.window_in(A7) == (0, 600, 140, 460)
    assert F.MapExtent(70.0, 20.0).band_keys[-1] == "60_70"
    for bad in ((55.0, 16.0), (60.0, 15.5), (60.25, 16.0), (float("nan"), 16.0)):
        with pytest.raises(ValueError):
            F.MapExtent(*bad)
    with pytest.raises(ValueError):
        A7.window_in(F.EXTENT_V2)                   # the big one does not fit the small


def test_anchored_lateral_centres_are_the_v2_floats_and_the_naive_ones_are_not():
    """SPEC_REFCV7 §12 item 3: ``y = -16 + (j_rel + 0.5) * cell``. Inside the old
    window the anchored float IS the /2 exporter's float, bit for bit, on all 320
    columns; the naive ``-30 + (j + 0.5) * 0.1`` differs on 127 of them (MEASURED here)
    -- the reason a floor() at a world-map edge could flip a code without anchoring."""
    j = np.arange(320)
    v2 = -16.0 + (j + 0.5) * 0.1
    anch = A7.lateral_centres(0.1)[140:460]
    assert np.array_equal(anch, v2)
    naive = -30.0 + (np.arange(140, 460) + 0.5) * 0.1
    assert int((naive != v2).sum()) == 127
    np.testing.assert_allclose(A7.lateral_centres(0.1),
                               -30.0 + (np.arange(600) + 0.5) * 0.1, atol=1e-12)


def _block_frac(codes: np.ndarray) -> np.ndarray:
    """cart_frac's rule on ANY extent (the exporter's `fractions`, 5 x 5)."""
    n, h, w = codes.shape
    blk = codes.reshape(n, h // 5, 5, w // 5, 5)
    out = np.empty((n, 9, h // 5, w // 5), np.uint8)
    for k in range(8):
        out[:, k] = np.rint((blk == k).sum(axis=(2, 4)) * (255 / 25)).astype(np.uint8)
    out[:, 8] = np.rint((blk == 255).sum(axis=(2, 4)) * (255 / 25)).astype(np.uint8)
    return out


def _v3_codes(T: int, shift: int = 0) -> np.ndarray:
    """A /3 map at the A7 extent whose old window IS `_codes(T)` (anchored), with a
    far-range edge row at 90 m and a side strip past +-16 m. ``shift`` != 0 builds the
    UN-anchored defect: the old window displaced by that many columns."""
    c = np.full((T, 1000, 600), 7, np.uint8)
    c[:, 900, :] = 5                                   # an edge at x = 90 m
    c[:, :, :20] = 255                                 # never seen at y < -28 m
    c[:, :600, 140 + shift:460 + shift] = _codes(T)
    return c


def _meta3(clip_id: str, ext=A7, **over) -> dict:
    m = {"schema": F.SCHEMA_V3, "frame": "rig", "channels": list(G.CHANNELS),
         "fine": {"x_max_m": float(ext.x_max_m), "y_half_m": float(ext.y_half_m),
                  "cell_m": 0.1, "shape": list(ext.fine_shape),
                  "codes": "0-7 class, 255 not seen", "y_anchor_m": -16.0},
         "cartesian": {"x_max_m": float(ext.x_max_m),
                       "y_half_m": float(ext.y_half_m), "cell_m": 0.5,
                       "shape": list(ext.grid(0.5).shape)},
         "fraction_scale": 255,
         "source": {"clip_sha12": G.sha12(clip_id), "world_map_res_m": 0.1}}
    m.update(over)
    return m


def _write3(root: Path, clip_id: str, T: int = 6, meta=None, arrays=None,
            shift: int = 0) -> Path:
    t_cam = np.arange(3 * T + 3, dtype=np.float64) * 33_333.0 + 1_000.0
    t_query = np.linspace(t_cam[0], t_cam[0] + (T - 1) * 100_000.0, T)
    idx = np.searchsorted(t_cam, t_query)
    fine = _v3_codes(T, shift)
    arr = {"meta_json": np.array(json.dumps(meta or _meta3(clip_id))),
           "t_query_us": t_query, "t_img_us": t_cam[idx].astype(np.int64),
           "cam_frame_idx": idx.astype(np.int32),
           "T_world_rig": np.tile(np.eye(4), (T, 1, 1)),
           "cart_frac": _block_frac(fine), "fine_codes": fine}
    arr.update(arrays or {})
    arr = {k: v for k, v in arr.items() if v is not None}
    d = root.joinpath(*G.GT_SUBDIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{clip_id}{G.GT_SUFFIX}"
    np.savez_compressed(p, **arr)
    return p


def test_v3_reads_at_the_declared_extent_and_its_window_is_the_v2_map(tmp_path):
    _write3(tmp_path, CLIP_A)
    c = F.open_clip_fine(tmp_path, CLIP_A, A7)
    assert c.schema == F.SCHEMA_V3 and c.extent == A7
    m = c.read([4, 1])
    assert m.codes.shape == (2, 1000, 600)
    assert int(m.codes[0, 900, 300]) == 5 and int(m.codes[0, 950, 10]) == 255
    np.testing.assert_array_equal(m.codes[:, :600, 140:460], _codes(6)[[4, 1]])
    st = F.FineMapGTStore(tmp_path, extent=A7)
    assert st.frames_for_windows(CLIP_A, [2], n_stack=3).codes.shape == (1, 1000, 600)


def test_v3_REFUSES_another_extent_never_crops_or_pads(tmp_path):
    _write3(tmp_path, CLIP_A)
    with pytest.raises(F.FineSpecMismatch, match="DECLARED extent"):
        F.open_clip_fine(tmp_path, CLIP_A, F.EXTENT_V2)
    with pytest.raises(F.FineSpecMismatch, match="DECLARED extent"):
        F.open_clip_fine(tmp_path, CLIP_A, F.MapExtent(80.0, 30.0))
    # a /2 file under the A7 extent: refused with the re-export instruction
    _write(tmp_path / "v2", CLIP_B)
    with pytest.raises(F.FineSpecMismatch, match="Re-export"):
        F.open_clip_fine(tmp_path / "v2", CLIP_B, A7)
    # the store census would count both as not covered: same refusal, through it
    with pytest.raises(F.FineSpecMismatch):
        F.FineMapGTStore(tmp_path / "v2", extent=A7).open(CLIP_B)


@pytest.mark.parametrize("meta_over,arrays,exc,needle", [
    ({"fine": {"x_max_m": 100.0, "y_half_m": 30.0, "cell_m": 0.1,
               "shape": [1000, 600]}, "extent": {"x_max_m": 80.0, "y_half_m": 30.0}},
     None, F.FineSpecMismatch, "contradicts"),
    ({"fine": {"cell_m": 0.1, "shape": [1000, 600]}}, None, F.FineSpecMismatch,
     "declares no extent"),
    ({"fine": {"x_max_m": 100.0, "y_half_m": 30.0, "cell_m": 0.05,
               "shape": [1000, 600]}}, None, F.FineSpecMismatch, "cell_m"),
    ({"source": {"clip_sha12": "0" * 12, "world_map_res_m": 0.1}}, None,
     G.ClipIdentityMismatch, "another clip"),
    ({"source": {"clip_sha12": G.sha12(CLIP_A), "world_map_res_m": 0.2}}, None,
     F.FineSpecMismatch, "world_map_res_m"),
    ({"channels": list(G.CHANNELS)[::-1]}, None, G.SchemaMismatch, "channel"),
    ({"frame": "world"}, None, G.SchemaMismatch, "frame"),
    ({"schema": "tanitad.sam3_map_gt/4"}, None, G.SchemaMismatch, "not in"),
    ({}, {"t_img_us": None}, G.SchemaMismatch, "t_img_us"),
    ({}, {"fine_codes": np.zeros((6, 600, 320), np.uint8)}, F.FineSpecMismatch,
     "fine_codes shape"),
    ({}, {"fine_codes": None}, F.FineSpecMismatch, "no fine_codes"),
])
def test_v3_every_guard_refuses(tmp_path, meta_over, arrays, exc, needle):
    _write3(tmp_path, CLIP_A, meta=_meta3(CLIP_A, **meta_over), arrays=arrays)
    with pytest.raises(exc, match=needle):
        F.open_clip_fine(tmp_path, CLIP_A, A7)


def test_v3_time_axis_and_code_legend_are_the_base_readers(tmp_path):
    _write3(tmp_path, CLIP_A)
    c = F.open_clip_fine(tmp_path, CLIP_A, A7)
    c.read([1], frame_t_us=[float(c.t_img_us[1])])
    with pytest.raises(G.TimeMisalignment):
        c.read([1], frame_t_us=[float(c.t_img_us[1]) + 2_000.0])
    bad = _v3_codes(6)
    bad[3, 950, 300] = 9
    _write3(tmp_path / "b", CLIP_A, arrays={"fine_codes": bad})
    with pytest.raises(F.FineCodeError):
        F.open_clip_fine(tmp_path / "b", CLIP_A, A7).read([3])


def _control(root_v3, root_v2, clip_id) -> dict:
    v3 = F.open_clip_fine(root_v3, clip_id, A7)
    v2 = F.open_clip_fine(root_v2, clip_id)
    idx = np.arange(v3.n_frames)
    with np.load(v3.path, allow_pickle=False) as z:
        cart3 = z["cart_frac"]
    return F.compare_v2_window(v3.read(idx).codes, A7, v2.read(idx).codes,
                               cart3, v2.base.cart_u8())


def test_the_v2_window_control_passes_anchored_and_catches_a_shift(tmp_path):
    """SPEC_REFCV7 §12 item 3's control, on synthetic files: 0 differing cells in the
    old window (fine AND cart) for an anchored export; a 5-column (0.5 m) shift --
    the un-anchored defect -- differs in both."""
    _write(tmp_path / "v2", CLIP_A)
    _write3(tmp_path / "v3", CLIP_A)
    r = _control(tmp_path / "v3", tmp_path / "v2", CLIP_A)
    assert (r["n_differ"], r["n_cart_differ"]) == (0, 0)
    assert r["n_cells"] == 6 * 600 * 320 and r["window"] == [0, 600, 140, 460]
    assert r["cart_window"] == [0, 120, 28, 92]
    _write3(tmp_path / "v3s", CLIP_A, shift=5)
    s = _control(tmp_path / "v3s", tmp_path / "v2", CLIP_A)
    assert s["n_differ"] > 0 and s["n_cart_differ"] > 0


V3_DIR = Path(os.environ.get("TANITAD_SAM3_V3_DIR",
                             "D:/refcv6_eval_kit/data/sam3_gt_v3_eval"))


def _real_v3_pairs() -> list[tuple[str, Path, Path]]:
    """(clip_id, /3 path, /2 path) for eval clips with BOTH exports and a v2ep file."""
    if not V3_DIR.is_dir():
        return []
    from tanitad.data.perception_targets import MapGTStore
    st3 = MapGTStore(V3_DIR)
    out = []
    for cid, p2 in REAL:
        p3 = st3.resolve(cid)
        if p3 is not None:
            out.append((cid, p3, p2))
    return out


REAL_V3 = _real_v3_pairs()


@pytest.mark.skipif(not REAL_V3, reason=(
    f"no /3 export with /2 twins and v2ep files on this box ($TANITAD_SAM3_V3_DIR={V3_DIR}, "
    f"with $TANITAD_SAM3_FINE_DIR / $TANITAD_V2EP_FINE_DIR); the full-set control is the NEW-2 "
    f"package's raw/v3_reader_control_eval.json (SPEC_REFCV7 §12 item 3)"))
def test_real_v3_opens_at_A7_and_its_old_window_is_byte_identical_to_v2():
    """The export's own control, re-run by the READER on real files: every frame, the
    /3 fine_codes and cart_frac inside the anchored old window equal the /2 arrays."""
    n_files = n_cells = 0
    for cid, p3, p2 in REAL_V3[:8]:
        v3 = F.open_path_fine(p3, cid, A7)
        v2 = F.open_path_fine(p2, cid)
        assert v3.n_frames == v2.n_frames, v3.clip_sha12
        idx = np.arange(v3.n_frames)
        with np.load(p3, allow_pickle=False) as z:
            cart3 = z["cart_frac"] if "cart_frac" in z.files else None
        r = F.compare_v2_window(v3.read(idx).codes, A7, v2.read(idx).codes,
                                cart3, None if cart3 is None else v2.base.cart_u8())
        assert r["n_differ"] == 0, (v3.clip_sha12, r)
        assert r.get("n_cart_differ", 0) == 0, (v3.clip_sha12, r)
        n_files += 1
        n_cells += r["n_cells"]
        with pytest.raises(F.FineSpecMismatch):
            F.open_path_fine(p3, cid, F.EXTENT_V2)          # the extent is refused
    assert n_files >= 1 and n_cells > 1_000_000
