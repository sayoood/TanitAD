"""Standalone tests for the R1 selection tool (intake gate-E).

No network, no HF token: synthetic egomotion zips + a minimal clip_index /
data_collection mirror the ACTUAL PhysicalAI-AV schema the tool consumes.

Run: pytest TanitAD Research Hub/.../2026-07-09-physicalai-r1-selection/tests -q
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import physicalai_r1 as r1  # noqa: E402


def _ego_df(kind: str, n: int = 200) -> pd.DataFrame:
    """Synthetic egomotion in the real schema: timestamp, vx, vy, curvature."""
    t = np.arange(n) * 0.1
    if kind == "urban":          # ~8 m/s, some turning, a brief stop
        v = np.full(n, 8.0); v[:20] = 0.1
        curv = np.full(n, 0.02)
    elif kind == "highway":      # ~30 m/s, straight -> fails speed gate
        v = np.full(n, 30.0); curv = np.zeros(n)
    else:                        # parked -> fails speed gate
        v = np.full(n, 0.05); curv = np.zeros(n)
    return pd.DataFrame({"timestamp": t, "vx": v, "vy": np.zeros(n),
                         "curvature": curv})


def _make_chunk(ego_dir: Path, chunk: int, spec: dict[str, str]) -> list[str]:
    ego_dir.mkdir(parents=True, exist_ok=True)
    zp = ego_dir / f"egomotion.chunk_{chunk:04d}.zip"
    ids = []
    with zipfile.ZipFile(zp, "w") as z:
        for clip_id, kind in spec.items():
            buf = _ego_df(kind).to_parquet()
            z.writestr(f"{clip_id}.egomotion.parquet", buf)
            ids.append(clip_id)
    return ids


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    ego = tmp_path / "labels" / "egomotion"
    ids0 = _make_chunk(ego, 10, {f"c10_{i:03d}": ("urban" if i < 30 else "highway")
                                 for i in range(50)})
    ids1 = _make_chunk(ego, 11, {f"c11_{i:03d}": ("urban" if i < 40 else "parked")
                                 for i in range(50)})
    all_ids = ids0 + ids1
    # minimal clip_index + data_collection (index = clip_id)
    idx = pd.DataFrame(index=pd.Index(all_ids, name="clip_id"))
    idx.to_parquet(tmp_path / "clip_index.parquet")
    (tmp_path / "metadata").mkdir()
    countries = ["Germany", "France", "Italy"]
    dc = pd.DataFrame({"country": [countries[i % 3] for i in range(len(all_ids))],
                       "hour_of_day": [i % 24 for i in range(len(all_ids))]},
                      index=pd.Index(all_ids, name="clip_id"))
    dc.to_parquet(tmp_path / "metadata" / "data_collection.parquet")
    return tmp_path


def test_scoring_and_gates(root: Path):
    scored = r1.score_cached_egomotion(root / "labels" / "egomotion")
    assert len(scored) == 100                     # all members scored
    passing = scored[scored["urban_score"] > 0]
    # only the 30 + 40 urban clips clear the driving gate
    assert len(passing) == 70
    # highway (30 m/s) and parked (0.05 m/s) are gate-zeroed
    assert (scored[scored["mean_speed"] > 14]["urban_score"] == 0).all()


def test_selection_schema_loader_compatible(root: Path):
    scored = r1.attach_meta(r1.score_cached_egomotion(root / "labels" / "egomotion"),
                            root)
    sel = r1.select(scored, target=50)
    assert len(sel) == 50
    # loader (physicalai.discover_r0_clips) needs clip_id (str) + chunk (int)
    assert {"clip_id", "chunk"}.issubset(sel.columns)
    assert sel["chunk"].dtype.kind in "iu"
    assert sel["urban_score"].min() > 0            # never select a gate-failed clip


def test_report_reachability_and_cost(root: Path):
    scored = r1.attach_meta(r1.score_cached_egomotion(root / "labels" / "egomotion"),
                            root)
    sel = r1.select(scored, target=50)
    rep = r1.build_report(scored, sel, target=50)
    assert rep["clips_passing_gates"] == 70
    assert rep["r1_reachable_from_cache"] is True   # 70 >= 50
    assert rep["n_camera_chunks"] == 2
    assert rep["camera_download_gb_est"] == pytest.approx(4.0)
    # unreachable case: ask for more than the pool
    rep2 = r1.build_report(scored, r1.select(scored, 999), target=999)
    assert rep2["r1_reachable_from_cache"] is False
