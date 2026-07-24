"""TanitAD Data Lake (Phase A) — byte-equivalence + license-firewall tests.

All on synthetic comma2k19 segments with an injected ``decode_fn`` (no real
video, no network) — the same CI fixture pattern as ``test_comma2k19``. The
headline test proves the acceptance gate: ``LakeWindowDataset`` windows are
BIT-IDENTICAL to ``EpisodeWindowDataset`` over the same episodes, through the
real ``build_episode``-wrapping ingestor.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.data._contract import EpisodeWindowDataset
from tanitad.data.comma2k19 import discover_segments
from tanitad.data.toy_driving import generate_episode
from tanitad.lake.ingest import Comma2k19Ingestor, ingest_source
from tanitad.lake.license_guard import LicenseScopeError, verify_license_scope
from tanitad.lake.proof import assert_datasets_bit_identical
from tanitad.lake.schema import assemble_lake_record, validate_superset
from tanitad.lake.shards import ShardWriter, iter_shard_samples
from tanitad.lake.view import LakeView, LakeWindowDataset

T_FRAMES = 120


def _make_fake_segment(root: Path, route: str, seg: str) -> Path:
    d = root / "Chunk_1" / route / seg
    (d / "global_pose").mkdir(parents=True)
    ft = np.arange(T_FRAMES) / 20.0
    ref = np.array([4278000.0, 635000.0, 4672000.0])
    east = np.array([-0.147, 0.989, 0.0])
    pos = ref[None] + 10.0 * ft[:, None] * east[None]
    vel = np.tile(10.0 * east, (T_FRAMES, 1))
    np.save(d / "global_pose" / "frame_times", ft)
    np.save(d / "global_pose" / "frame_positions", pos)
    np.save(d / "global_pose" / "frame_velocities", vel)
    for name, val in [("speed", 10.0), ("steering_angle", 15.3)]:
        c = d / "processed_log" / "CAN" / name
        c.mkdir(parents=True)
        np.save(c / "t", ft)
        np.save(c / "value", np.full(T_FRAMES, val))
    for p in d.rglob("*.npy"):
        p.rename(p.with_suffix(""))
    (d / "video.hevc").write_bytes(b"")
    return d


def _det_decode(seg, stride, size, max_frames):
    """Deterministic 'decode' seeded by the segment path — reproducible frames so
    the reference build and the ingest build see IDENTICAL pixels (byte-proof)."""
    n = min(max_frames or 40, 40)
    seed = int(hashlib.sha1(str(seg).encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 255, (n, 3, size, size), generator=g,
                         dtype=torch.uint8)


def _corpus(root: Path, n_routes=3, n_seg=2):
    for r in range(n_routes):
        for s in range(n_seg):
            _make_fake_segment(root, f"d{r}_2018-01-0{r + 1}--10-00-00", str(s))


# --------------------------------------------------------------------------- #
# THE ACCEPTANCE GATE                                                          #
# --------------------------------------------------------------------------- #
def test_lake_window_byte_identical_to_episode_window(tmp_path):
    """LakeWindowDataset == EpisodeWindowDataset, bit-for-bit, via the ingestor
    that wraps comma2k19.build_episode. This is the Phase-A acceptance gate."""
    src = tmp_path / "comma"
    lake = tmp_path / "lake"
    _corpus(src)

    ing = Comma2k19Ingestor(size=64, stride=2, max_steps=30, n_stack=3,
                            decode_fn=_det_decode)

    # reference = exactly what the trainer builds today (same build_core path)
    segs = discover_segments(src)
    ref_eps = sorted((ing.build_core(s) for s in segs),
                     key=lambda e: int(e.episode_id))
    ds_ref = EpisodeWindowDataset(ref_eps, window=4, max_horizon=2)

    # lake: ingest the same segments, then read back through the catalog + shards
    summary = ingest_source(ing, src, lake, seed=0, verbose=False)
    assert summary["catalog_rows"] == len(segs)

    import pyarrow.dataset as pads
    view = LakeView(lake, name="proof",
                    filter_expr=(pads.field("source") == "comma2k19"))
    ds_lake = LakeWindowDataset(view, window=4, max_horizon=2,
                                cache_dir=tmp_path / "cache")

    result = assert_datasets_bit_identical(ds_ref, ds_lake)
    assert result["bit_identical"] and result["windows"] > 0
    assert result["channels"] == 9 and result["image_size"] == 64


def test_hydrate_is_idempotent_and_reuses(tmp_path):
    src, lake = tmp_path / "c", tmp_path / "l"
    _corpus(src, n_routes=2)
    ing = Comma2k19Ingestor(size=48, stride=2, max_steps=25, decode_fn=_det_decode)
    ingest_source(ing, src, lake, verbose=False)
    import pyarrow.dataset as pads
    view = LakeView(lake, filter_expr=(pads.field("source") == "comma2k19"))
    d1 = LakeWindowDataset(view, window=4, max_horizon=2,
                           cache_dir=tmp_path / "ca")
    d2 = LakeWindowDataset(view, window=4, max_horizon=2,
                           cache_dir=tmp_path / "ca")   # reuse cache
    assert len(d1) == len(d2) > 0


# --------------------------------------------------------------------------- #
# LICENSE FIREWALL (structural)                                               #
# --------------------------------------------------------------------------- #
def test_gated_confidential_source_refused():
    ep = generate_episode(1, steps=20, size=32)
    with pytest.raises(PermissionError, match="gated-confidential"):
        assemble_lake_record(ep, source="physicalai_av", split="train",
                             build_params_hash="x")


def test_ingest_refuses_gated(tmp_path):
    ing = Comma2k19Ingestor(size=32, decode_fn=_det_decode)
    ing.source = "physicalai_av"     # force the firewall
    with pytest.raises(PermissionError):
        ingest_source(ing, tmp_path, tmp_path / "l", verbose=False)


def test_refuse_class_source_refused():
    """`refuse` (Waymax / Waymo Open) raises on assembly exactly like
    gated-confidential — its terms follow the trained WEIGHTS, so no tier can
    contain it (TANITDATASET_TIER_INTEGRATION §2)."""
    from tanitad.lake.schema import (LICENSE_CLASSES, SOURCE_REGISTRY,
                                     PERMISSIVE_SOURCES)
    assert "refuse" in LICENSE_CLASSES
    for src in ("waymo", "waymax"):
        assert SOURCE_REGISTRY[src].license_class == "refuse"
        assert not SOURCE_REGISTRY[src].commercial_ok
        assert src not in PERMISSIVE_SOURCES        # cannot physically enter
        ep = generate_episode(1, steps=20, size=32)
        with pytest.raises(PermissionError, match="refuse"):
            assemble_lake_record(ep, source=src, split="train",
                                 build_params_hash="x")


def test_ingest_refuses_refuse_class(tmp_path):
    ing = Comma2k19Ingestor(size=32, decode_fn=_det_decode)
    ing.source = "waymo"             # force the refuse firewall
    with pytest.raises(PermissionError, match="refuse"):
        ingest_source(ing, tmp_path, tmp_path / "l", verbose=False)


def test_refuse_cannot_be_export_scope():
    with pytest.raises(LicenseScopeError, match="NEVER be in an export"):
        verify_license_scope([], allowed_classes={"refuse"})


def test_refuse_has_no_tier():
    from tanitad.lake.filtering import tier_of
    with pytest.raises(ValueError, match="no tier"):
        tier_of("refuse", False, False)


def test_license_scope_guard_blocks_out_of_scope():
    rows = [{"episode_id": 1, "license_class": "owned-safe",
             "commercial_ok": True, "share_alike": False, "source": "comma2k19"},
            {"episode_id": 2, "license_class": "nc-research",
             "commercial_ok": False, "share_alike": False, "source": "drivelm"}]
    with pytest.raises(LicenseScopeError, match="outside scope"):
        verify_license_scope(rows, allowed_classes={"owned-safe"})
    # commercial gate rejects share-alike even inside owned-safe
    sa = [{"episode_id": 3, "license_class": "owned-safe", "commercial_ok": False,
           "share_alike": True, "source": "zod"}]
    with pytest.raises(LicenseScopeError):
        verify_license_scope(sa, allowed_classes={"owned-safe"},
                             require_commercial_ok=True)


def test_scope_cannot_include_gated():
    with pytest.raises(LicenseScopeError, match="NEVER be in an export"):
        verify_license_scope([], allowed_classes={"gated-confidential"})


# --------------------------------------------------------------------------- #
# SHARD INTEGRITY + SCHEMA                                                     #
# --------------------------------------------------------------------------- #
def test_shard_sha256_roundtrip_and_corruption(tmp_path):
    ep = generate_episode(7, steps=20, size=32)
    rec = assemble_lake_record(ep, source="comma2k19", split="val",
                               build_params_hash="p", action_source="can")
    with ShardWriter(tmp_path, "owned-safe", "comma2k19", "val",
                     share_alike=False) as w:
        w.write(rec)
    shard = tmp_path / rec.shard_key
    got = list(iter_shard_samples(shard, verify_sha256=True))
    assert len(got) == 1
    assert torch.equal(got[0]["frames"], ep.frames)          # byte-identical
    # corrupt the meta sha -> reader must refuse
    import io
    import json
    import tarfile
    data = {}
    with tarfile.open(shard, "r") as t:
        for m in t.getmembers():
            data[m.name] = t.extractfile(m).read()
    with tarfile.open(shard, "w") as t:
        for name, payload in data.items():
            if name.endswith(".meta.json"):
                meta = json.loads(payload)
                meta["sha256"] = "0" * 64
                payload = json.dumps(meta).encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            t.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        list(iter_shard_samples(shard, verify_sha256=True))


def test_hf_exporter_scaffold_guards_and_stages(tmp_path):
    """The owned-safe exporter stages a bundle (guard passes) but NEVER pushes."""
    from tanitad.lake.hf_export import export_hf
    src, lake = tmp_path / "c", tmp_path / "l"
    _corpus(src, n_routes=2)
    ing = Comma2k19Ingestor(size=32, stride=2, max_steps=20, decode_fn=_det_decode)
    ingest_source(ing, src, lake, verbose=False)

    out = tmp_path / "hf_stage"
    summary = export_hf(lake, "tanitad-own-core", out)
    assert summary["pushed"] is False
    assert summary["episodes"] > 0
    assert (out / "DATA_CARD.md").exists() and (out / "MANIFEST.json").exists()
    # shards staged, mirroring the lake partition layout (no basename collision:
    # train + val both number from shard-00000, so a flat copy would drop shards)
    staged = list((out / "shards").rglob("*.tar"))
    assert staged
    assert len(staged) == len(summary["shards"])         # nothing dropped/collided
    # an explicit push attempt must be refused in Phase A
    with pytest.raises((PermissionError, NotImplementedError)):
        export_hf(lake, "tanitad-own-core", tmp_path / "x", push=True,
                  confirm="PUSH tanitad-own-core")


def test_modality_flags_consistency():
    ep = generate_episode(3, steps=15, size=24)
    rec = assemble_lake_record(ep, source="comma2k19", split="train",
                               build_params_hash="p")
    assert rec.modality_flags["has_actions"] and rec.modality_flags["has_poses"]
    assert rec.commercial_ok and rec.license_class == "owned-safe"
    # tamper: flag says no poses but poses present -> validate must raise
    rec.modality_flags["has_poses"] = False
    with pytest.raises(ValueError, match="has_poses"):
        validate_superset(rec)
