"""CPU tests — Stage 1 SCHEMA EXTENSION + the enrichment compose
(``lake.enrich``): VLM-pending stubs, the per-episode sidecar + flat enrichment
catalog, and the end-to-end non-VLM pipeline. Proves the core tensors stay
byte-identical (the pass writes only sidecars + an enrichment catalog).
"""

import math

import torch

from tanitad.lake import enrich as E
from tanitad.lake import vocab as V


def _roll(yaw_rate, speed, T, dt=0.1):
    rows, x, y, yaw = [], 0.0, 0.0, 0.0
    for t in range(T):
        v = float(speed[t])
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += float(yaw_rate[t]) * dt
    return torch.tensor(rows, dtype=torch.float32)


def _frames(seed, T=120, S=48):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, (T, 9, S, S), generator=g, dtype=torch.uint8)


def _episode(i, source="comma2k19", v=25.0, T=200, **kw):
    return E.enrich_episode(
        episode_id=i, source=source, frames=_frames(i, T), poses=_straight(v, T),
        split_unit_id=f"route{i}", license_name="MIT", cy=437.0, hz=10.0, **kw)


def _straight(v, T):
    return _roll(torch.zeros(T), torch.full((T,), float(v)), T)


# --------------------------------------------------------------------------- #
# VLM-pending stubs — the ONLY deferred step                                  #
# --------------------------------------------------------------------------- #
def test_vlm_stubs_are_shape_fixed_and_pending():
    st = E.vlm_pending_scene_tags()
    assert st["_pending"] and st["weather"]["prov"] == "vlm"
    assert st["notable_events"] == []
    ls = E.vlm_pending_lead_state()
    assert ls["_pending"] and ls["ttc_s"] is None and ls["gap_m"] is None
    sr = E.vlm_pending_sign_reads()
    assert sr[0]["_pending"]
    lang = E.vlm_pending_language()
    assert lang["_pending"] and lang["coc_trace"]["physics_flag"] is None


def test_kinematic_label_stamp_marks_vlm_pending():
    st = E.kinematic_label_stamp("MIT")
    assert st["kinematic"]["provenance"] == "kinematic"
    assert st["kinematic"]["source_license"] == "MIT"
    assert st["vlm"]["_pending"] is True


# --------------------------------------------------------------------------- #
# per-episode enrich                                                          #
# --------------------------------------------------------------------------- #
def test_enrich_episode_fills_tier_rig_quality_goal_and_stubs():
    e = _episode(1)
    assert e.tier == "ship" and e.rig_id == "comma2k19:mono"
    assert e.crop_center_cy == 437.0
    assert e.quality["corrupt"] is None and e.quality["egomotion_sane"]
    # kinematic goal minted; VLM slots stay unknown
    assert e.goal["LONMODE"]["prov"] == "kinematic"
    assert e.goal["HEADWAY"] == {"token": "unknown", "prov": "unknown"}
    assert e.scene_tags["_pending"] and e.lead_state["_pending"]
    V.validate_goal(e.goal)


def test_enrich_episode_tiers_track_license():
    ship = _episode(2, license_class="owned-safe", share_alike=False,
                    commercial_ok=True)
    sa = _episode(3, license_class="owned-safe", share_alike=True,
                  commercial_ok=False)
    nc = _episode(4, license_class="nc-research", share_alike=False,
                  commercial_ok=False)
    assert (ship.tier, sa.tier, nc.tier) == ("ship", "ship-sa", "nc")


def test_sidecar_and_catalog_row_shapes():
    e = _episode(5)
    E.enrich_corpus([e])                               # fill dedup + curation
    sc = e.to_sidecar_dict()
    assert set(sc) >= {"tier", "rig_id", "quality", "dedup", "scene_tags",
                       "lead_state", "sign_reads", "goal", "curation",
                       "label_stamp", "goal_provenance"}
    row = e.to_catalog_row()
    assert row["tier"] == "ship" and row["goal_lonmode"] == "free_cruise"
    assert row["goal_kinematic_slots"] >= 5 and isinstance(row["phash"], int)


# --------------------------------------------------------------------------- #
# CORE TENSORS STAY BYTE-IDENTICAL (the acceptance invariant)                  #
# --------------------------------------------------------------------------- #
def test_enrichment_does_not_touch_core_tensors():
    frames = _frames(11, T=120)
    poses = _straight(22.0, 120)
    f0, p0 = frames.clone(), poses.clone()
    e = E.enrich_episode(episode_id=11, source="comma2k19", frames=frames,
                         poses=poses, split_unit_id="r11", license_name="MIT",
                         cy=437.0)
    E.enrich_corpus([e])
    assert torch.equal(frames, f0) and torch.equal(poses, p0)   # untouched


# --------------------------------------------------------------------------- #
# CORPUS pass — dedup + curation                                              #
# --------------------------------------------------------------------------- #
def test_enrich_corpus_dedup_collapses_a_duplicate():
    a = _episode(20)
    b = E.enrich_episode(episode_id=21, source="comma2k19",
                         frames=_frames(20, 200),          # SAME pixels as a
                         poses=_straight(25.0, 200), split_unit_id="route21",
                         license_name="MIT", cy=437.0)
    E.enrich_corpus([a, b])
    assert a.dedup["dedup_cluster_id"] == b.dedup["dedup_cluster_id"]
    assert a.dedup["is_exemplar"] and not b.dedup["is_exemplar"]
    assert a.curation and "weight" in a.curation


# --------------------------------------------------------------------------- #
# END-TO-END driver — sidecars + queryable enrichment catalog                 #
# --------------------------------------------------------------------------- #
def test_run_enrichment_writes_sidecars_and_queryable_catalog(tmp_path):
    eps = [_episode(30 + i, v=20.0 + i * 5) for i in range(5)]
    summary = E.run_enrichment(tmp_path, eps, run_id="0")
    assert summary["episodes"] == 5 and summary["catalog_rows"] == 5
    assert summary["tiers"] == {"ship": 5}

    # sidecars written + round-trip
    sc = sorted((tmp_path / "sidecars").glob("*.goal.json"))
    assert len(sc) == 5
    d = E.read_sidecar(sc[0])
    assert d["goal"]["VTARGET"]["prov"] == "kinematic"
    assert d["scene_tags"]["_pending"] is True         # VLM step still stubbed

    # enrichment catalog is a queryable, tier-partitioned parquet
    import pyarrow.dataset as pads
    ds = pads.dataset(tmp_path / "enrichment", format="parquet",
                      partitioning="hive")
    t = ds.to_table()
    assert t.num_rows == 5 and "curation_weight" in t.column_names
    ship = ds.to_table(filter=(pads.field("tier") == "ship"))
    assert ship.num_rows == 5                           # partition prune works
    # a training-run-as-a-query: non-holdout, sane, exemplar clips
    train = ds.to_table(filter=(pads.field("is_eval_holdout") == False) &  # noqa: E712
                        (pads.field("egomotion_sane") == True) &            # noqa: E712
                        (pads.field("is_exemplar") == True))                # noqa: E712
    assert train.num_rows >= 1


def test_enrich_lake_over_ingested_lake(tmp_path):
    """Integration: build a tiny synthetic lake with the REAL ingestor, then run
    the whole non-VLM pipeline over it via ``enrich_lake`` (reads the catalog +
    shards, no decode). Proves the stages connect to the live lake read path."""
    import hashlib

    import numpy as np
    from tanitad.data.comma2k19 import discover_segments  # noqa: F401
    from tanitad.lake.ingest import Comma2k19Ingestor, ingest_source

    # --- fabricate a couple of comma2k19 segments (test_lake.py's fixture) ---
    def make_seg(route, seg):
        d = tmp_path / "comma" / "Chunk_1" / route / seg
        (d / "global_pose").mkdir(parents=True)
        ft = np.arange(120) / 20.0
        ref = np.array([4278000.0, 635000.0, 4672000.0])
        east = np.array([-0.147, 0.989, 0.0])
        pos = ref[None] + 10.0 * ft[:, None] * east[None]
        np.save(d / "global_pose" / "frame_times", ft)
        np.save(d / "global_pose" / "frame_positions", pos)
        np.save(d / "global_pose" / "frame_velocities", np.tile(10.0 * east, (120, 1)))
        for name, val in [("speed", 10.0), ("steering_angle", 15.3)]:
            c = d / "processed_log" / "CAN" / name
            c.mkdir(parents=True)
            np.save(c / "t", ft)
            np.save(c / "value", np.full(120, val))
        for p in d.rglob("*.npy"):
            p.rename(p.with_suffix(""))
        (d / "video.hevc").write_bytes(b"")

    def det_decode(seg, stride, size, max_frames):
        n = min(max_frames or 40, 40)
        s = int(hashlib.sha1(str(seg).encode()).hexdigest()[:8], 16)
        return torch.randint(0, 255, (n, 3, size, size),
                             generator=torch.Generator().manual_seed(s),
                             dtype=torch.uint8)

    for r in range(2):
        for s in range(2):
            make_seg(f"d{r}_2018-01-0{r + 1}--10-00-00", str(s))
    lake = tmp_path / "lake"
    ing = Comma2k19Ingestor(size=64, stride=2, max_steps=30, n_stack=3,
                            decode_fn=det_decode)
    ingest_source(ing, tmp_path / "comma", lake, seed=0, verbose=False)

    summary = E.enrich_lake(lake, run_id="0")
    assert summary["episodes"] == 4 and summary["catalog_rows"] == 4
    assert summary["tiers"] == {"ship": 4}
    assert len(list((lake / "sidecars").glob("*.goal.json"))) == 4

    import pyarrow.dataset as pads
    ds = pads.dataset(lake / "enrichment", format="parquet", partitioning="hive")
    t = ds.to_table()
    assert t.num_rows == 4
    # the CAN blinker channel is absent on comma -> SIGNAL never faked; every row
    # carries a minted kinematic goal + a rig id derived from the real intrinsics
    assert all(r.startswith("comma2k19:") for r in t.column("rig_id").to_pylist())
    assert all(s >= 1 for s in t.column("goal_kinematic_slots").to_pylist())


def test_run_enrichment_mixed_tiers_partition(tmp_path):
    eps = [
        _episode(40, license_class="owned-safe", share_alike=False,
                 commercial_ok=True),
        _episode(41, source="zod", license_class="owned-safe", share_alike=True,
                 commercial_ok=False),
        _episode(42, source="nuscenes", license_class="nc-research",
                 share_alike=False, commercial_ok=False),
    ]
    E.run_enrichment(tmp_path, eps, run_id="0")
    import pyarrow.dataset as pads
    ds = pads.dataset(tmp_path / "enrichment", format="parquet",
                      partitioning="hive")
    tiers = set(ds.to_table().column("tier").to_pylist())
    assert tiers == {"ship", "ship-sa", "nc"}          # segregated by tier
