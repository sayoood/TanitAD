"""I1a CPU tests — obstacle.offline -> episode join (scripts/build_obstacle_join.py).

Synthetic clips only (no corpus, no pod):
  * rig->world pinned as the EXACT inverse of `bev_raster.ego_frame_agents`
    (itself pinned against refb_labels.ego_frame in tests/test_p8.py), plus a
    hand-computed case.
  * ego-compensation correctness on a synthetic clip: a WORLD-STATIC parked car
    read through staggered rig-frame samples lands at its true per-frame gap
    (the raw uncompensated rig read is shown to be off by ~v*dt — the
    bev_raster.py:69-77 mis-registration the join exists to remove).
  * P4 visibility flags: straight ahead vs 90-deg lateral at close range vs
    behind, the +-60-deg edge, and a track whose flag flips 0 -> 1 while the
    track continues (P4's selection case).
  * NO_LABEL discipline: frames outside the label span produce NO record.
  * jsonl / jsonl.xz roundtrip against the REAL consumer
    (`train_p8_occupancy.JoinFileReader` + `bev_raster.agents_to_array` /
    `rasterize`) — schema compatibility proven by import, not prose.
  * registration failure (stationary clip) is a loud RegistrationError skip.
"""
import json
import lzma
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.data.bev_raster import (agents_to_array,  # noqa: E402
                                     ego_frame_agents, rasterize)
from build_obstacle_join import (EgoTrack, HFOV_DEG_DEFAULT,  # noqa: E402
                                 clip_tracks, join_clip, open_out,
                                 rig_to_world, verify_with_reader,
                                 visibility_occ, world_agents_at,
                                 write_records)
from train_p8_occupancy import (JoinFileReader,  # noqa: E402
                                episode_uid_of_clip)


# ============================================================================
# rig -> world: pinned inverse of the committed world -> ego
# ============================================================================
def test_rig_to_world_inverts_ego_frame_agents():
    """ego_frame_agents(rig_to_world(ag, pose), pose) == ag for random agents
    and poses — pins the composition against the refb ego_frame convention
    WITHOUT re-deriving it (test_p8 pins ego_frame_agents to the torch original)."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        pose = (rng.uniform(-50, 50), rng.uniform(-50, 50),
                rng.uniform(-math.pi, math.pi))
        ag = np.column_stack([rng.uniform(-40, 40, 4), rng.uniform(-40, 40, 4),
                              rng.uniform(-math.pi, math.pi, 4),
                              rng.uniform(2, 5, 4), rng.uniform(1, 2.5, 4),
                              np.full(4, -1.0)])
        back = ego_frame_agents(rig_to_world(ag, np.asarray(pose)), pose)
        np.testing.assert_allclose(back[:, :2], ag[:, :2], atol=1e-9)
        # headings equal modulo 2*pi (ego_frame_agents wraps to (-pi, pi])
        dyaw = back[:, 2] - ag[:, 2]
        np.testing.assert_allclose(np.cos(dyaw), 1.0, atol=1e-12)
        np.testing.assert_array_equal(back[:, 3:5], ag[:, 3:5])


def test_rig_to_world_hand_case():
    """Ego at world (5, 3) facing +y (yaw pi/2); cuboid 10 m ahead, 2 m LEFT in
    the rig frame -> world (5-2, 3+10) = (3, 13); heading pi/2 + 0.3."""
    w = rig_to_world(np.array([[10.0, 2.0, 0.3, 4.0, 2.0, -1.0]]),
                     np.array([5.0, 3.0, math.pi / 2]))
    np.testing.assert_allclose(w[0, :2], [3.0, 13.0], atol=1e-12)
    assert abs(w[0, 2] - (math.pi / 2 + 0.3)) < 1e-12
    # per-row poses: two samples of one track carry two different ego poses
    w2 = rig_to_world(
        np.array([[10.0, 0.0, 0.0, 4.0, 2.0, -1.0],
                  [10.0, 0.0, 0.0, 4.0, 2.0, -1.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(w2[:, 0], [10.0, 11.0], atol=1e-12)


# ============================================================================
# P4 visibility: 120-deg front frustum on the ego-frame center azimuth
# ============================================================================
def test_visibility_ahead_lateral_behind():
    ag = np.array([
        [10.0, 0.0, 0.0, 4.0, 2.0, -1.0],     # straight ahead        -> visible
        [0.0, 5.0, 0.0, 4.0, 2.0, -1.0],      # 90 deg LEFT, close    -> occluded
        [0.0, -5.0, 0.0, 4.0, 2.0, -1.0],     # 90 deg RIGHT, close   -> occluded
        [-10.0, 0.0, 0.0, 4.0, 2.0, -1.0],    # behind                -> occluded
    ])
    occ = visibility_occ(ag, hfov_deg=120.0)
    np.testing.assert_array_equal(occ, [0, 1, 1, 1])
    assert visibility_occ(np.zeros((0, 6))).shape == (0,)


def test_visibility_edge_of_frustum():
    """+-60 deg is the 120-deg half-angle: 59 deg in, 61 deg out; the exact
    edge is CLOSED (<=), matching the raster's closed-boundary convention."""
    def at(deg):
        r = math.radians(deg)
        return [10.0 * math.cos(r), 10.0 * math.sin(r), 0.0, 4.0, 2.0, -1.0]
    occ = visibility_occ(np.array([at(59.0), at(61.0), at(-59.0), at(-61.0)]),
                         hfov_deg=120.0)
    np.testing.assert_array_equal(occ, [0, 1, 0, 1])
    assert visibility_occ(np.array([at(60.0)]), hfov_deg=120.0)[0] == 0
    assert HFOV_DEG_DEFAULT == 120.0


# ============================================================================
# synthetic clip: the ego-compensation + span + registration path end to end
# ============================================================================
A_GRID, B_GRID = 0.3, 0.1007          # episode grid: t_i = a + b*i (NOT 0.1 —
T_EP = 60                             # the measured ~0.1007 s spacing)
V_EGO = 10.0                          # straight +x at 10 m/s, yaw 0


def synth_ego(t_end=25.0, hz=100.0):
    """Egomotion dict (schema physicalai.py:9-10): 100 Hz, x = V*t, yaw 0."""
    t = np.arange(0.0, t_end, 1.0 / hz)
    n = t.size
    return {"timestamp": t * 1e6, "x": V_EGO * t, "y": np.zeros(n),
            "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n),
            "qw": np.ones(n)}


def synth_poses():
    """Episode poses [T, 4] on the affine grid — what the v2 manifest carries."""
    t = A_GRID + B_GRID * np.arange(T_EP)
    return np.column_stack([V_EGO * t, np.zeros(T_EP), np.zeros(T_EP),
                            np.full(T_EP, V_EGO)]), t


def synth_obs():
    """Two WORLD-STATIC agents as staggered rig-frame samples at their OWN
    timestamps (the corpus fact, bev_raster.py:27-33):
      A: parked car at world (30, 2), heading +y (world yaw pi/2), samples at
         0.033 + 0.1 k (10 Hz, offset from the frame grid on purpose);
      B: object at world (10, 5) — crosses the +-60-deg frustum edge as the
         ego drives past, then falls behind (the P4 flag-flip case).
    rig@t: cx = wx - V*t (ego yaw 0), cy = wy; rig yaw = world yaw - 0."""
    rows = []
    q90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    qid = (0.0, 0.0, 0.0, 1.0)
    for k in range(30):                                   # A: t = 0.033..2.933
        t = 0.033 + 0.1 * k
        rows.append((t, "A", 30.0 - V_EGO * t, 2.0, q90, 4.0, 2.0,
                     "automobile"))
    for k in range(30):                                   # B: same span
        t = 0.033 + 0.1 * k
        rows.append((t, "B", 10.0 - V_EGO * t, 5.0, qid, 0.8, 0.8, "person"))
    return {
        "timestamp_us": np.array([r[0] for r in rows]) * 1e6,
        "track_id": np.array([r[1] for r in rows]),
        "center_x": np.array([r[2] for r in rows]),
        "center_y": np.array([r[3] for r in rows]),
        "size_x": np.array([r[5] for r in rows]),
        "size_y": np.array([r[6] for r in rows]),
        "orientation_x": np.array([r[4][0] for r in rows]),
        "orientation_y": np.array([r[4][1] for r in rows]),
        "orientation_z": np.array([r[4][2] for r in rows]),
        "orientation_w": np.array([r[4][3] for r in rows]),
        "label_class": np.array([r[7] for r in rows]),
    }


@pytest.fixture()
def synth_join():
    poses, t_i = synth_poses()
    recs, stats = join_clip("synthclip-uuid-0001", poses,
                            EgoTrack(synth_ego()), synth_obs())
    return recs, stats, t_i


def test_ego_compensation_beats_raw_rig_read(synth_join):
    """Compensated cx of the parked car == its TRUE gap at the FRAME time
    (30 - 10*t_i) to < 5 cm on every frame; the raw rig read (nearest sample,
    no compensation) errs up to ~|v|*tol ~ 0.5 m — the bev_raster.py:69-77
    mis-registration. Registration must also recover the ~0.1007 s grid."""
    recs, stats, t_i = synth_join
    assert stats["n_labelled"] == len(recs) > 20
    assert abs(stats["registration"]["b"] - B_GRID) < 1e-3
    max_err, max_raw = 0.0, 0.0
    for r in recs:
        a = next(x for x in r["agents"] if x["track_id"] == "A")
        t = t_i[r["frame_idx"]]
        max_err = max(max_err, abs(a["cx"] - (30.0 - V_EGO * t)))
        assert abs(a["cy"] - 2.0) < 0.05
        assert abs(a["yaw"] - math.pi / 2) < 1e-4        # heading composed
        # (tolerance sits above the writer's 5-decimal yaw rounding, 5e-6)
        assert a["l"] == 4.0 and a["w"] == 2.0
        # the UNCOMPENSATED read: nearest sample's raw rig cx
        k = round((t - 0.033) / 0.1)
        t_s = 0.033 + 0.1 * min(max(k, 0), 29)
        max_raw = max(max_raw, abs((30.0 - V_EGO * t_s) - (30.0 - V_EGO * t)))
    assert max_err < 0.05, f"compensated error {max_err:.3f} m"
    assert max_raw > 0.3, "synthetic stagger too small to discriminate"


def test_no_label_outside_span_never_empty_agents(synth_join):
    """Label span ends at 2.933 s but the episode runs to ~6.2 s: frames past
    the span emit NO line (NO_LABEL), never `agents: []` (labelled clear) —
    the join-doc §4 bias, enforced at the builder."""
    recs, stats, t_i = synth_join
    emitted = {r["frame_idx"] for r in recs}
    span_lo, span_hi = stats["label_span_s"]
    for i in range(T_EP):
        inside = span_lo - 0.06 <= t_i[i] <= span_hi + 0.06
        assert (i in emitted) == inside, f"frame {i} (t={t_i[i]:.3f})"
    assert max(emitted) < T_EP - 1                       # tail is NO_LABEL
    # every emitted frame here has both tracks live -> agents non-empty
    assert all(len(r["agents"]) == 2 for r in recs)


def test_visibility_flip_while_track_continues(synth_join):
    """Track B: azimuth crosses +60 deg as the ego passes -> occ flips 0 -> 1
    while the track keeps being emitted (P4's 'visible at t, occluded at t+k,
    track continues' selection, WM_PHYSICS_PROOF.md P4). Flip time: ego-frame
    cx = 10 - 10t equals cy/tan(60 deg) = 5/sqrt(3) at t ~= 0.711 s."""
    recs, stats, t_i = synth_join
    occ_b = {r["frame_idx"]: next(x for x in r["agents"]
                                  if x["track_id"] == "B")["occ"]
             for r in recs}
    t_flip = (10.0 - 5.0 / math.tan(math.radians(60.0))) / V_EGO
    for i, o in occ_b.items():
        want = 0 if t_i[i] < t_flip - 0.02 else (1 if t_i[i] > t_flip + 0.02
                                                 else o)
        assert o == want, f"frame {i} t={t_i[i]:.3f} occ={o}"
    assert 0 in occ_b.values() and 1 in occ_b.values()
    # class + track id survive into the record (P4 needs track identity)
    any_b = next(x for r in recs for x in r["agents"] if x["track_id"] == "B")
    assert any_b["cls"] == "person"


def test_registration_failure_is_loud():
    """A stationary clip cannot be registered by position — RegistrationError
    (lead_source's loud refusal), which main() reports as a per-episode skip."""
    from taniteval.lead_source import RegistrationError
    ego = synth_ego()
    ego["x"] = np.zeros_like(ego["x"])                   # parked ego
    poses = np.zeros((T_EP, 4))
    with pytest.raises(RegistrationError):
        join_clip("stationary-clip", poses, EgoTrack(ego), synth_obs())


def test_non_rig_reference_frame_refused():
    obs = synth_obs()
    obs["reference_frame"] = np.array(["rig"] * 59 + ["map"])
    with pytest.raises(ValueError, match="reference_frame"):
        clip_tracks(obs)
    obs["reference_frame"] = np.array(["rig"] * 60)      # all-rig passes
    assert len(clip_tracks(obs)) == 2


def test_world_agents_at_nearest_and_stale():
    """Per-track NEAREST-within-tol lookup over staggered samples; a track with
    no sample within tol contributes nothing at that time."""
    tracks = clip_tracks(synth_obs())
    ego = EgoTrack(synth_ego())
    ag, tids, _cls = world_agents_at(tracks, 1.03, ego, tol_s=0.06)
    assert set(tids) == {"A", "B"}
    a = ag[tids.index("A")]
    np.testing.assert_allclose(a[:2], [30.0, 2.0], atol=1e-9)  # world-static
    ag2, tids2, _ = world_agents_at(tracks, 3.4, ego, tol_s=0.06)
    assert tids2 == [] and ag2.shape == (0, 6)           # 0.467 s past span


# ============================================================================
# the roundtrip proof: writer output parses with the REAL P8 consumer
# ============================================================================
def _write(tmp_path, recs, name):
    p = tmp_path / name
    fh = open_out(p)
    try:
        write_records(fh, recs)
    finally:
        fh.close()
    return p


def test_roundtrip_with_p8_reader(tmp_path, synth_join):
    recs, stats, _t = synth_join
    p = _write(tmp_path, recs, "agents.jsonl")
    r = JoinFileReader(p)                                # the EXACT consumer
    assert r.n_records == len(recs) and r.n_clips == 1
    assert r.has_occlusion_flags                         # P4 flags visible to P8
    uid = episode_uid_of_clip("synthclip-uuid-0001")
    assert r.covers_episode(uid)
    fi = recs[0]["frame_idx"]
    ag = r.lookup(uid, fi)
    assert ag is not None and ag.shape == (2, 6)
    assert set(np.unique(ag[:, 5])) <= {0.0, 1.0}        # occ carried through
    assert r.lookup(uid, T_EP - 1) is None               # NO_LABEL frame
    # reader raster == direct rasterize of the same record's agents
    np.testing.assert_array_equal(r.raster(uid, fi),
                                  rasterize(recs[0]["agents"]))
    # visible/occluded subsets split on the occ flag we emitted
    frame_flip = next(x["frame_idx"] for x in recs
                      if any(a["occ"] == 1 for a in x["agents"]))
    rec = next(x for x in recs if x["frame_idx"] == frame_flip)
    vis = [a for a in rec["agents"] if a["occ"] == 0]
    hid = [a for a in rec["agents"] if a["occ"] == 1]
    np.testing.assert_array_equal(r.raster(uid, frame_flip, subset="visible"),
                                  rasterize(vis))
    np.testing.assert_array_equal(r.raster(uid, frame_flip, subset="occluded"),
                                  rasterize(hid))


def test_roundtrip_xz_via_verify_helper(tmp_path, synth_join):
    """.xz output (the pod-relay form) verifies through the script's own
    decompress-then-JoinFileReader helper; bytes equal the plain jsonl."""
    recs, _s, _t = synth_join
    p_xz = _write(tmp_path, recs, "agents.jsonl.xz")
    p_plain = _write(tmp_path, recs, "agents.jsonl")
    with lzma.open(p_xz, "rt", encoding="utf-8") as fh:
        assert fh.read() == p_plain.read_text(encoding="utf-8")
    ver = verify_with_reader(p_xz)
    assert ver["n_records"] == len(recs) and ver["n_clips"] == 1
    assert ver["has_occlusion_flags"] is True
    assert not (tmp_path / "agents.jsonl.xz.verify.tmp").exists()  # cleaned up


def test_agent_extras_ignored_by_agents_to_array(synth_join):
    """track_id/cls/t_s are AUDIT extras: agents_to_array reads only its known
    keys, so the P8 path is untouched by them — proven, not asserted in prose."""
    recs, _s, _t = synth_join
    rec = recs[0]
    assert {"clip_id", "frame_idx", "t_s", "agents"} <= set(rec)
    a = agents_to_array(rec["agents"])
    assert a.shape == (len(rec["agents"]), 6)
    assert a[0, 0] == rec["agents"][0]["cx"]
    assert set(np.unique(a[:, 5])) <= {0.0, 1.0}
    # a json round-trip of the record leaves the agents parseable too
    back = json.loads(json.dumps(rec, separators=(",", ":")))
    np.testing.assert_array_equal(agents_to_array(back["agents"]), a)
