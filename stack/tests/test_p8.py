"""P8 CPU tests — BEV rasteriser exactness, occupancy head, gate JSON, join reader.

Covers the off-pod-runnable part of the P8 battery (WM_PHYSICS_PROOF.md P8):
  * `tanitad.data.bev_raster`: synthetic agents at known poses -> raster cells
    assert EXACTLY (against an independent analytic loop, not the function under
    test); the world->ego rotation pinned against `refb_labels.ego_frame`
    (scripts/refb_labels.py:86-90) itself; staggered-track time lookup.
  * `train_p8_occupancy.BEVOccupancyHead`: shapes, the ~1M param band asserted
    at the flagship state_dim 2048, band enforcement.
  * `p8_gate_dict`: PASS / FAIL / not-computable branches + JSON round-trip.
  * `JoinFileReader`: jsonl roundtrip, NO_LABEL vs labelled-clear distinction,
    legacy-id ambiguity refusal, occlusion flags, duplicate/bad-record refusal.

The full trainer path (GPU + checkpoint + v2 corpora + pod-built join) is
POD-SIDE and is NOT exercised here.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data.bev_raster import (BEVGrid, GRID_DEFAULT,  # noqa: E402
                                     agents_at_time, agents_to_array,
                                     ego_frame_agents, rasterize,
                                     yaw_from_quaternion)
from train_p8_occupancy import (PARAM_BAND, BEVOccupancyHead,  # noqa: E402
                                EpisodeCarriedSource, JoinFileReader,
                                cell_recall, covered_indices,
                                episode_uid_of_clip, iou_at_05,
                                legacy_episode_id_of_clip,
                                make_covered_sampler, p8_gate_dict,
                                window_frame)

NX, NY = GRID_DEFAULT.shape                                  # (120, 64)


def expected_axis_aligned(cx, cy, half_l, half_w):
    """Independent analytic footprint: closed-boundary cell-center test, written
    as explicit loops (NOT via the rotation math under test)."""
    exp = np.zeros((NX, NY), dtype=np.float32)
    for i in range(NX):
        xc = (i + 0.5) * 0.5
        for j in range(NY):
            yc = -16.0 + (j + 0.5) * 0.5
            if abs(xc - cx) <= half_l and abs(yc - cy) <= half_w:
                exp[i, j] = 1.0
    return exp


# ============================================================================
# rasteriser exactness
# ============================================================================
def test_grid_shape_dtype_values_and_empty():
    assert GRID_DEFAULT.shape == (120, 64)                   # 60 m fwd / +-16 m
    r = rasterize([])
    assert r.shape == (120, 64) and r.dtype == np.float32
    assert float(r.sum()) == 0.0
    r2 = rasterize([{"cx": 10.25, "cy": 0.25, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    assert set(np.unique(r2).tolist()) <= {0.0, 1.0}


def test_axis_aligned_box_exact_cells():
    """4.0 x 2.0 box at (10.25, 0.25), yaw 0 — every value exactly representable
    in binary, so the occupied set must match the analytic loop BIT-EXACTLY:
    9 rows x 5 cols = 45 cells (closed boundary hit on both ends)."""
    r = rasterize([{"cx": 10.25, "cy": 0.25, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    exp = expected_axis_aligned(10.25, 0.25, 2.0, 1.0)
    assert exp.sum() == 45
    np.testing.assert_array_equal(r, exp)


def test_rotated_90deg_box_swaps_footprint():
    """At yaw = pi/2 the footprint is analytically the axis-aligned box with
    l <-> w swapped. Dims chosen with 0.05 m boundary margin from every cell
    center so float wobble in cos(pi/2) cannot flip a cell."""
    r = rasterize([{"cx": 10.25, "cy": 0.25, "yaw": math.pi / 2,
                    "l": 3.9, "w": 1.9}])
    exp = expected_axis_aligned(10.25, 0.25, 0.95, 1.95)     # swapped halves
    assert exp.sum() > 0
    np.testing.assert_array_equal(r, exp)


def test_forward_left_axis_convention():
    """+x fwd = rows, +y LEFT = HIGH cols (col 0 is 16 m to the RIGHT)."""
    left = rasterize([{"cx": 30.0, "cy": 5.0, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    right = rasterize([{"cx": 30.0, "cy": -5.0, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    far = rasterize([{"cx": 50.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0}])
    assert np.argwhere(left)[:, 1].min() > NY // 2           # left -> cols > 32
    assert np.argwhere(right)[:, 1].max() < NY // 2
    assert np.argwhere(far)[:, 0].mean() > np.argwhere(left)[:, 0].mean()


def test_out_of_grid_and_behind_ego_empty():
    for ag in ({"cx": -10.0, "cy": 0.0, "yaw": 0.0, "l": 4.0, "w": 2.0},
               {"cx": 100.0, "cy": 0.0, "yaw": 0.3, "l": 4.0, "w": 2.0},
               {"cx": 30.0, "cy": 30.0, "yaw": 0.0, "l": 4.0, "w": 2.0}):
        assert float(rasterize([ag]).sum()) == 0.0


def test_world_to_ego_matches_refb_ego_frame():
    """The world->ego rotation is EXACTLY refb_labels.ego_frame
    (scripts/refb_labels.py:86-90) — pinned against the torch original."""
    import refb_labels
    rng = np.random.default_rng(0)
    for _ in range(20):
        ex, ey, eyaw = rng.uniform(-50, 50), rng.uniform(-50, 50), \
            rng.uniform(-math.pi, math.pi)
        ax, ay = rng.uniform(-80, 80), rng.uniform(-80, 80)
        ego = ego_frame_agents(
            np.array([[ax, ay, 0.3, 4.0, 2.0]]), (ex, ey, eyaw))
        ref = refb_labels.ego_frame(
            torch.tensor([ax - ex, ay - ey], dtype=torch.float64),
            torch.tensor(eyaw, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(ego[0, :2], ref, atol=1e-9)
    # heading maps into the ego frame and wraps
    out = ego_frame_agents(np.array([[0.0, 0.0, 3.0, 4.0, 2.0]]),
                           (0.0, 0.0, -3.0))
    assert abs(out[0, 2] - (6.0 - 2 * math.pi)) < 1e-9


def test_rotated_ego_places_agent_correctly():
    """Ego at (5, 3) facing +y world (yaw pi/2); agent 10 m ahead, 2 m LEFT in
    world coords -> ego-frame (10, 2) -> same raster as the direct build."""
    world_ag = np.array([[5.0 - 2.0, 3.0 + 10.0, math.pi / 2, 3.9, 1.9]])
    ego_ag = ego_frame_agents(world_ag, (5.0, 3.0, math.pi / 2))
    np.testing.assert_allclose(ego_ag[0, :2], [10.0, 2.0], atol=1e-9)
    assert abs(ego_ag[0, 2]) < 1e-9                          # heading aligned
    r = rasterize(ego_ag)
    direct = rasterize([{"cx": 10.0, "cy": 2.0, "yaw": 0.0, "l": 3.9, "w": 1.9}])
    np.testing.assert_array_equal(r, direct)


def test_agents_at_time_staggered_tracks():
    """Per-track nearest-within-tolerance lookup over STAGGERED samples (the
    corpus fact: 1.00-1.005 rows per unique timestamp — a frame index would be
    wrong by construction). Stale tracks drop; class filter applies; yaw comes
    from the quaternion."""
    q90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    obs = {
        "timestamp_us": np.array([1_000_000, 1_100_000, 900_000, 1_050_000]),
        "track_id": np.array(["A", "A", "B", "C"]),
        "center_x": np.array([12.0, 13.0, 5.0, 20.0]),
        "center_y": np.array([0.5, 0.6, -1.0, 2.0]),
        "size_x": np.array([4.0, 4.0, 4.5, 0.6]),
        "size_y": np.array([1.8, 1.8, 1.9, 0.6]),
        "orientation_x": np.array([0.0, 0.0, 0.0, q90[0]]),
        "orientation_y": np.array([0.0, 0.0, 0.0, q90[1]]),
        "orientation_z": np.array([0.0, 0.0, 0.0, q90[2]]),
        "orientation_w": np.array([1.0, 1.0, 1.0, q90[3]]),
        "label_class": np.array(["automobile", "automobile", "automobile",
                                 "person"]),
    }
    ag = agents_at_time(obs, 1.03)                            # tol 0.06 default
    got = {round(a[0], 2): a for a in ag}
    assert set(got) == {12.0, 20.0}          # A@1.00 (dt .03), C@1.05 (dt .02);
    assert 5.0 not in got                    # B@0.90 is 0.13 s stale -> dropped
    assert abs(got[20.0][2] - math.pi / 2) < 1e-9            # quaternion yaw
    veh = agents_at_time(obs, 1.03, classes=("automobile",))
    assert {round(a[0], 2) for a in veh} == {12.0}           # person filtered
    assert agents_at_time({"timestamp_us": np.array([])}, 1.0).shape == (0, 6)


def test_agents_to_array_forms():
    a = agents_to_array([{"cx": 1, "cy": 2, "yaw": 0.1, "l": 4, "w": 2,
                          "occ": 1}])
    assert a.shape == (1, 6) and a[0, 5] == 1.0
    b = agents_to_array(np.zeros((2, 5)))
    assert b.shape == (2, 6) and (b[:, 5] == -1.0).all()     # no-flag sentinel
    assert agents_to_array([]).shape == (0, 6)
    with pytest.raises(ValueError):
        agents_to_array(np.zeros((2, 4)))


# ============================================================================
# decoder head
# ============================================================================
def test_head_shapes_and_grad_small():
    h = BEVOccupancyHead(64, enforce_band=False)
    z = torch.randn(3, 64, requires_grad=True)
    out = h(z)
    assert out.shape == (3, 120, 64)
    out.sum().backward()
    assert z.grad is not None and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        h(torch.randn(2, 65))                                # wrong latent dim


def test_head_param_band_at_flagship_width():
    """state_dim 2048 (the flagship contract: 16x16 grid, state_dim 2048 —
    tanitad/config.py:404) with default widths must land inside the
    pre-registered ~1M band, enforced by the default ctor."""
    h = BEVOccupancyHead(2048)                               # enforce_band=True
    assert PARAM_BAND[0] <= h.n_params <= PARAM_BAND[1]
    assert h.n_params == sum(p.numel() for p in h.parameters())


def test_head_band_enforced_and_grid_divisibility():
    with pytest.raises(ValueError, match="band"):
        BEVOccupancyHead(64)                                 # ~0.03M -> refuse
    with pytest.raises(ValueError, match="divisible"):
        BEVOccupancyHead(64, grid=BEVGrid(3.0, 1.0, 0.5),    # (6, 4) shape
                         enforce_band=False)


# ============================================================================
# metrics
# ============================================================================
def test_iou_at_05_hand_case_and_empty_union():
    logits = torch.full((2, 120, 64), -5.0)
    target = torch.zeros(2, 120, 64)
    logits[0, :10, :4] = 5.0                                 # 40 predicted
    target[0, 5:15, :4] = 1.0                                # 40 true, 20 inter
    v = iou_at_05(logits, target)
    assert abs(float(v[0]) - 20.0 / 60.0) < 1e-6
    assert torch.isnan(v[1])                                 # both empty -> NaN


def test_cell_recall_hand_case():
    logits = torch.full((1, 120, 64), -5.0)
    logits[0, :10, :10] = 5.0
    sub = torch.zeros(1, 120, 64)
    sub[0, 8:12, 0] = 1.0                                    # 4 cells, 2 hit
    assert abs(float(cell_recall(logits, sub)[0]) - 0.5) < 1e-6
    assert torch.isnan(cell_recall(logits, torch.zeros(1, 120, 64))[0])


# ============================================================================
# gate JSON — both branches
# ============================================================================
def _row(enc, pred, n=100):
    return {"iou_enc": enc, "n_enc": n, "iou_pred": pred, "n_pred": n}


def test_gate_pass_branch_and_json_roundtrip(tmp_path):
    g = p8_gate_dict({10: _row(0.5, 0.45)})                  # ratio 0.9 >= 0.8
    assert g["PASS"] is True and g["gate_a"]["pass"] is True
    assert abs(g["gate_a"]["ratio"] - 0.9) < 1e-6
    p = tmp_path / "p8_gate.json"
    p.write_text(json.dumps(g))
    back = json.loads(p.read_text())
    assert back["PASS"] is True and back["gate_a"]["computable"] is True


def test_gate_fail_branch():
    g = p8_gate_dict({10: _row(0.5, 0.3)})                   # ratio 0.6 < 0.8
    assert g["PASS"] is False and g["gate_a"]["pass"] is False
    assert g["gate_a"]["computable"] is True


def test_gate_not_computable_branches():
    assert p8_gate_dict({5: _row(0.5, 0.5)})["PASS"] is None  # k=10 missing
    g0 = p8_gate_dict({10: {"iou_enc": None, "n_enc": 0,
                            "iou_pred": 0.4, "n_pred": 9}})
    assert g0["PASS"] is None and "n_enc=0" in g0["gate_a"]["reason"]
    gz = p8_gate_dict({10: _row(0.0, 0.0)})                  # readout failed
    assert gz["PASS"] is None and "undefined" in gz["gate_a"]["reason"]


def test_gate_reads_string_keys():
    """per_k arrives with string keys after a JSON round-trip — still gated."""
    g = p8_gate_dict({"10": _row(0.4, 0.36)})
    assert g["PASS"] is True


# ============================================================================
# join-file reader
# ============================================================================
def _write_join(tmp_path, records, name="agents.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return p


CAR = {"cx": 12.0, "cy": 0.5, "yaw": 0.0, "l": 4.0, "w": 2.0}


def test_join_roundtrip(tmp_path):
    p = _write_join(tmp_path, [
        {"clip_id": "clipA-uuid-0001", "frame_idx": 7, "agents": [CAR]},
        {"clip_id": "clipA-uuid-0001", "frame_idx": 8, "agents": []},
        {"clip_id": "zzzz-uuid-0002", "frame_idx": 7,
         "agents": [dict(CAR, cx=30.0)]},
    ])
    r = JoinFileReader(p)
    assert r.n_records == 3 and r.n_clips == 2
    uid = episode_uid_of_clip("clipA-uuid-0001")
    assert r.covers_episode(uid)
    ag = r.lookup(uid, 7)
    assert ag.shape == (1, 6) and ag[0, 0] == 12.0
    # labelled-and-CLEAR is an empty array + all-zero raster, NOT None
    assert r.lookup(uid, 8).shape == (0, 6)
    assert float(r.raster(uid, 8).sum()) == 0.0
    # absent frame / absent episode = NO_LABEL = None (never "empty road")
    assert r.lookup(uid, 99) is None
    assert r.raster(uid, 99) is None
    assert not r.covers_episode(episode_uid_of_clip("never-seen"))
    # the raster equals a direct rasterize of the same agents
    np.testing.assert_array_equal(r.raster(uid, 7), rasterize([CAR]))


def test_join_legacy_id_resolution_and_ambiguity(tmp_path):
    p = _write_join(tmp_path, [
        {"clip_id": "abcd1111", "frame_idx": 0, "agents": [CAR]},
        {"clip_id": "abcd2222", "frame_idx": 0, "agents": []},   # same 4 bytes
        {"clip_id": "wxyz0000", "frame_idx": 0, "agents": [CAR]},
    ])
    r = JoinFileReader(p)
    # unique legacy prefix resolves
    assert r.lookup(legacy_episode_id_of_clip("wxyz0000"), 0).shape == (1, 6)
    # colliding legacy prefix REFUSES rather than guessing a clip
    with pytest.raises(RuntimeError, match="LEGACY"):
        r.lookup(legacy_episode_id_of_clip("abcd1111"), 0)
    # the collision-free uid still resolves both clips
    assert r.lookup(episode_uid_of_clip("abcd1111"), 0).shape == (1, 6)
    assert r.lookup(episode_uid_of_clip("abcd2222"), 0).shape == (0, 6)


def test_join_occlusion_flags_and_subsets(tmp_path):
    vis = dict(CAR, occ=0)
    hid = dict(CAR, cx=30.0, occ=1)
    p = _write_join(tmp_path, [
        {"clip_id": "clipF", "frame_idx": 3, "agents": [vis, hid]}])
    r = JoinFileReader(p)
    assert r.has_occlusion_flags
    uid = episode_uid_of_clip("clipF")
    full = r.raster(uid, 3)
    rv = r.raster(uid, 3, subset="visible")
    ro = r.raster(uid, 3, subset="occluded")
    np.testing.assert_array_equal(rv, rasterize([vis]))
    np.testing.assert_array_equal(ro, rasterize([hid]))
    np.testing.assert_array_equal(np.maximum(rv, ro), full)
    # a flag-free file refuses subsets instead of returning empties
    p2 = _write_join(tmp_path,
                     [{"clip_id": "clipG", "frame_idx": 0, "agents": [CAR]}],
                     name="noflags.jsonl")
    r2 = JoinFileReader(p2)
    assert not r2.has_occlusion_flags
    with pytest.raises(RuntimeError, match="occlusion"):
        r2.raster(episode_uid_of_clip("clipG"), 0, subset="visible")


def test_join_refuses_duplicates_bad_records_empty(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        JoinFileReader(_write_join(tmp_path, [
            {"clip_id": "c", "frame_idx": 1, "agents": []},
            {"clip_id": "c", "frame_idx": 1, "agents": [CAR]}]))
    with pytest.raises(ValueError, match="bad join record"):
        JoinFileReader(_write_join(tmp_path, [{"clip_id": "c"}]))
    bad = tmp_path / "empty.jsonl"
    bad.write_text("")
    with pytest.raises(ValueError, match="empty"):
        JoinFileReader(bad)


def test_episode_uid_matches_canonical_formula():
    """The fallback formula IS the canonical one (v2_dataset.py:95-96):
    blake2b(clip_id, digest_size=8) >> 1 — checked against a hand computation
    (always runnable), legacy id against physicalai.py:740's rule."""
    import hashlib
    cid = "abcdef-uuid-1234"
    hand = int.from_bytes(
        hashlib.blake2b(cid.encode("utf-8"), digest_size=8).digest(),
        "big") >> 1
    assert episode_uid_of_clip(cid) == hand
    # legacy: first 4 BYTES (physicalai.py:740), null-padded
    assert legacy_episode_id_of_clip("ab") == int.from_bytes(b"ab\0\0", "big")


def test_episode_uid_matches_stable_episode_id():
    """Parity with the real v2_dataset.stable_episode_id — skips only where
    torchvision (v2_dataset's module-level import) is absent; runs pod-side."""
    v2 = pytest.importorskip("tanitad.data.v2_dataset")
    for cid in ("abcdef-uuid-1234", "x", "0123456789abcdef"):
        assert v2.stable_episode_id(cid) == episode_uid_of_clip(cid)


# ============================================================================
# episode-carried source + window/coverage plumbing (stub datasets)
# ============================================================================
class _StubEp:
    def __init__(self, eid, agents=None):
        self.episode_id = eid
        if agents is not None:
            self.agents = agents


class _StubDS:
    """index/episodes/window surface of FlagshipWindowDataset (window 8)."""

    def __init__(self, eps, ts):
        self.episodes = eps
        self.window = 8
        self.index = ts


def test_episode_carried_source_refuses_today():
    """No corpus carries `.agents` today — construction must refuse with the
    pod-side alternative, not pretend (the honest-data-path contract)."""
    with pytest.raises(SystemExit, match="join-file"):
        EpisodeCarriedSource([_StubEp(1), _StubEp(2)])


def test_episode_carried_source_future_path():
    ep = _StubEp(7, agents=[[CAR], None, []])
    s = EpisodeCarriedSource([ep])
    assert s.covers_episode(7)
    assert s.lookup(7, 0).shape == (1, 6)
    assert s.lookup(7, 1) is None                            # NO_LABEL frame
    assert s.lookup(7, 2).shape == (0, 6)                    # labelled clear
    assert s.lookup(7, 99) is None
    with pytest.raises(RuntimeError):
        s.raster(7, 0, subset="visible")


def test_window_frame_and_covered_indices(tmp_path):
    """present frame = t + window - 1 (the canonical 881-grid origin rule) and
    NO_LABEL windows are excluded from coverage, never zero-filled."""
    uid = episode_uid_of_clip("clipZ")
    p = _write_join(tmp_path, [
        {"clip_id": "clipZ", "frame_idx": 7, "agents": [CAR]},   # t=0 present
        {"clip_id": "clipZ", "frame_idx": 9, "agents": []},      # t=2 present
    ])
    r = JoinFileReader(p)
    ds = _StubDS([_StubEp(uid)], [(0, 0), (0, 1), (0, 2)])
    assert window_frame(ds, 0) == (uid, 7)
    assert window_frame(ds, 2) == (uid, 9)
    assert covered_indices(ds, r, k=0) == [0, 2]             # t=1 -> frame 8: NO_LABEL
    rng = __import__("random").Random(0)
    sample = make_covered_sampler(ds, [0, 2], eps_per_batch=2, rng=rng)
    got = sample(16)
    assert len(got) == 16 and set(got) <= {0, 2}
