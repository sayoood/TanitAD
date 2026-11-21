"""refcv8 WP-C I3 (the ego-footprint row strip) x refcv7 VIS-1 -- the row re-key (WP-B, 2026-10-05).

MEASURED on Thor (the L1 full-size smoke, `refcv8-wpb-smoke-L1`, 2026-10-05 00:58): a DataLoader worker died on
"an in-scope row with a 3-D label is ABSENT from the sidecar" at a frame that `refcv8_join_label_defects.json` lists
as an ego-footprint frame. The strip removes the ego row at READ time, every later row moves up one position, and the
VIS-1 sidecar is keyed by the join's ORIGINAL positions -- so every refcv8 arm carrying `--join-defect-masks` dies the
first time a batch touches one of the 693 listed frames. The fix re-keys the rows of a LISTED frame by track id.
Expectations are literals; the deliberate-regression arm is the pre-fix path and must reproduce the Thor refusal.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
from tanitad.data import vis1 as V  # noqa: E402

S12 = "abcdef012345"


def _sidecar(tmp_path):
    """The join's frame 7 held THREE rows: 0 = track 7 at (10, 0.5); 1 = the EGO box (track 3, at (1.0, 0.0),
    in no store scope's interest -- not stored); 2 = track 9 at (30, -1). The sidecar stores rows 0 and 2."""
    a = {"row": np.array([0, 2], np.int16), "track": np.array([7, 9], np.int64),
         "n_full": np.array([1000, 500], np.int32), "n_img": np.array([1000, 500], np.int32),
         "n_vis": np.array([800, 40], np.int32), "vis_rows": np.array([30, 5], np.int32),
         "cx": np.array([10.0, 30.0], np.float32), "cy": np.array([0.5, -1.0], np.float32),
         "frame_clip": np.array([0], np.int32), "frame_f": np.array([7], np.int32),
         "frame_ptr": np.array([0, 2], np.int64), "frame_n_zh": np.array([2], np.int32),
         "frame_n_valid": np.array([3], np.int32),
         "clip_sha12": np.array([S12], dtype="<U12"), "clip_frame_ptr": np.array([0, 1], np.int64),
         "clip_mask_ftheta": np.array([True]), "clip_split": np.array(["train"], dtype="<U5")}
    meta = {"store_scope": dict(V.STORE_SCOPE), "n_stack": 3, "window": 8, "splits": {"train": {}}}
    V.write_sidecar(tmp_path / "s.npz", a, meta)
    return V.VIS1Sidecar(tmp_path / "s.npz")


#: the STRIPPED frame as the dataset sees it: the ego row is gone, track 9 moved from row 2 to row 1
BOX = np.array([[10.0, 0.5, 4, 2], [30.0, -1.0, 4, 2]], np.float32)
TIDS = ["7", "9"]


def test_rekey_literals(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    assert T.vis1_rows_by_track(sc, S12, 7, TIDS).tolist() == [0, 2]
    assert T.vis1_rows_by_track(sc, S12, 7, ["7", "5", ""]).tolist() == [0, -2, -3]     # unknown -> unique negative
    assert T.vis1_rows_by_track(sc, S12, 99, TIDS).tolist() == [-1, -2]                  # frame absent


def test_RED_ARM_the_position_key_reproduces_the_Thor_refusal(tmp_path):
    sc = _sidecar(tmp_path)
    with pytest.raises(V.VIS1SidecarError, match="row 1: an in-scope row with a 3-D label is ABSENT"):
        V.vis1_block_for_rows(sc, S12, 7, box=BOX, track_ids=TIDS, zh_mask=[True, True], pad=4)


def test_the_track_key_reads_the_stored_values(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    nf, nv, kn = V.vis1_block_for_rows(sc, S12, 7, box=BOX, track_ids=TIDS, zh_mask=[True, True], pad=4,
                                       order=T.vis1_rows_by_track(sc, S12, 7, TIDS))
    assert nf.tolist() == [1000, 500, 0, 0] and nv.tolist() == [800, 40, 0, 0] and kn.tolist() == [True, True,
                                                                                                    False, False]


def test_the_rekey_never_launders_a_different_box(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    with pytest.raises(V.VIS1SidecarError, match="centre"):                 # same track, moved centre -> refused
        V.vis1_block_for_rows(sc, S12, 7, box=BOX + np.float32(0.25), track_ids=TIDS, zh_mask=[True, True], pad=4,
                              order=T.vis1_rows_by_track(sc, S12, 7, TIDS))
    with pytest.raises(V.VIS1SidecarError, match="ABSENT"):                 # an unstored in-scope track -> refused
        V.vis1_block_for_rows(sc, S12, 7, box=BOX, track_ids=["7", "5"], zh_mask=[True, True], pad=4,
                              order=T.vis1_rows_by_track(sc, S12, 7, ["7", "5"]))


def _item_self(sc, listed: bool):
    dm = types.SimpleNamespace(ego_frames=({S12: {7}} if listed else {}))
    aj = types.SimpleNamespace(defect_masks=dm, lookup_track_ids=lambda eid, f: list(TIDS))
    return types.SimpleNamespace(agent_join=aj, vis1_sidecar=sc, _vis1_sha12={5: S12})


def _item():
    box = torch.zeros(4, 4)
    box[:2] = torch.from_numpy(BOX)
    return {"agent_valid": torch.tensor([True, True, False, False]), "agent_label": torch.tensor(True),
            "agent_box": box, "agent_zh_mask": torch.tensor([True, True, False, False])}


def test_the_dataset_hook_rekeys_a_listed_frame_and_only_a_listed_frame(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    got = T.V3Dataset._vis1_item(_item_self(sc, listed=True), 5, 7, _item(), 2, 4, None)
    assert got["agent_vis_full"].tolist() == [1000, 500, 0, 0] and got["agent_vis_known"].tolist()[:2] == [True, True]
    # deliberate regression: the same STRIPPED rows on an UNLISTED frame take the position key -> the Thor refusal
    with pytest.raises(V.VIS1SidecarError, match="ABSENT"):
        T.V3Dataset._vis1_item(_item_self(sc, listed=False), 5, 7, _item(), 2, 4, None)


def test_the_rekey_composes_with_the_truncation_order(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    it = _item()
    it["agent_box"][:2] = torch.from_numpy(BOX[::-1].copy())             # rows as `_agent_item` emits them after
    got = T.V3Dataset._vis1_item(_item_self(sc, listed=True), 5, 7, it, 2, 4, np.array([1, 0]))   # a permutation
    assert got["agent_vis_full"].tolist() == [500, 1000, 0, 0]


def test_a_sidecar_with_a_duplicated_track_refuses_the_rekey(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    sc.a = dict(sc.a)
    sc.a["track"] = np.array([7, 7], np.int64)
    with pytest.raises(SystemExit, match="twice"):
        T.vis1_rows_by_track(sc, S12, 7, TIDS)


# ------------------------------------------------------------------ the full-size read (MM 2026-10-05) ----------- #
def test_the_window_carries_its_rekey_flag(tmp_path):
    T = R.trainer()
    sc = _sidecar(tmp_path)
    assert int(T.V3Dataset._vis1_item(_item_self(sc, listed=True), 5, 7, _item(), 2, 4, None)["vis1_rekeyed"]) == 1
    it = _item()
    it["agent_label"] = torch.tensor(False)                                  # NO_LABEL: the key is still emitted
    it["agent_valid"][:] = False
    assert int(T.V3Dataset._vis1_item(_item_self(sc, listed=True), 5, 7, it, 0, 4, None)["vis1_rekeyed"]) == 0


def _seed_ds(ego):
    eps = [types.SimpleNamespace(episode_id=5)]
    aj = types.SimpleNamespace(defect_masks=types.SimpleNamespace(ego_frames=ego))
    return types.SimpleNamespace(index=[(0, 0), (0, 1), (0, 2)], episodes=eps, window=8, agent_join=aj,
                                 _vis1_sha12={5: S12}, vis1_sidecar=object())


def test_smoke_seed_picks_the_windows_whose_NOW_is_a_listed_frame():
    T = R.trainer()
    assert T.smoke_seed_windows(_seed_ds({S12: {8}}), 4) == [1]                 # NOW = t + 8 - 1
    assert T.smoke_seed_windows(_seed_ds({S12: {7, 8, 9}}), 2) == [0, 1]
    with pytest.raises(SystemExit, match="no window"):
        T.smoke_seed_windows(_seed_ds({S12: {50}}), 4)
    ds = _seed_ds({S12: {8}})
    ds.agent_join.defect_masks = None
    with pytest.raises(SystemExit, match="needs --join-defect-masks"):
        T.smoke_seed_windows(ds, 4)


def test_the_smoke_seed_flag_is_refused_on_a_real_run():
    from tanitad.refs import refcv8_conditioning as r8c
    T = R.trainer()
    base = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--smoke-seed-ego-frames"]
    cfg = types.SimpleNamespace(refcv8=r8c.R8Config())
    T._pin_refcv8(cfg, T.build_parser().parse_args(base + ["--steps", "30"]))           # a smoke: allowed
    with pytest.raises(SystemExit, match="SMOKE flag"):
        T._pin_refcv8(cfg, T.build_parser().parse_args(base + ["--steps", "101"]))
