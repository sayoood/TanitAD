"""PhysicalAI-AV Stage R1 selection (backlog P0.0 / DATASET_LANDSCAPE rank #1).

Proposed target: `stack/scripts/physicalai_r1.py` (sibling of `physicalai_r0.py`).

Sayed directive 2026-07-08: leverage all high-quality data — scale R0 (500 urban
clips) to R1 (2,000). This tool answers, with MEASURED numbers, whether R1 is
reachable from the egomotion ALREADY CACHED for R0 and, if not, how many more
chunks are needed; it writes a loader-compatible `r1_selection.parquet` and a
camera-fetch cost plan.

Design:
- Reuses the EXACT R0 urban scorer (`_urban_score_from_egomotion`) so R1 scores are
  directly comparable to R0. No re-implementation of the gate logic (single source
  of truth).
- Scores every clip in the cached egomotion zips (offline; no HF token, no network).
- Selection: top urban scores, round-robin across countries to keep geo-diversity,
  with a widened per-country cap so the target is actually reachable from the pool.
- Camera-fetch cost: the front_wide_120fov camera chunk is ~2 GB; cost is per-CHUNK
  (the whole zip is downloaded, then only selected members extracted), so fetching
  ALL gate-passing clips of an already-downloaded chunk is near-zero marginal cost.

CLI:
  python physicalai_r1.py --root <physicalai_root> --target 2000 --out-subdir r1
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CAM_CHUNK_GB = 2.0  # measured: front_wide_120fov camera chunk ~2 GB (R0 data card)


def _import_r0_scorer():
    """The single source of truth for the urban gate lives in physicalai_r0.py.
    When integrated into stack/scripts this is a plain sibling import; standalone
    (intake) we walk up to the repo `stack/` and shim sys.path."""
    try:
        from scripts.physicalai_r0 import _urban_score_from_egomotion  # type: ignore
        return _urban_score_from_egomotion
    except Exception:
        here = Path(__file__).resolve()
        for up in here.parents:
            cand = up / "stack"
            if (cand / "scripts" / "physicalai_r0.py").exists():
                sys.path.insert(0, str(cand))
                from scripts.physicalai_r0 import _urban_score_from_egomotion  # type: ignore
                return _urban_score_from_egomotion
        raise


def score_cached_egomotion(ego_dir: Path, scorer=None) -> pd.DataFrame:
    """Score every clip in the cached egomotion zips under `ego_dir`."""
    scorer = scorer or _import_r0_scorer()
    rows, t0 = [], time.time()
    zips = sorted(Path(ego_dir).glob("egomotion.chunk_*.zip"))
    for zp in zips:
        chunk = int(zp.stem.split("_")[-1])
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith(".egomotion.parquet"):
                    continue
                clip_id = name.split("/")[-1].split(".")[0]
                try:
                    df = pd.read_parquet(io.BytesIO(z.read(name)))
                    s = scorer(df)
                except Exception as e:  # loud, but keep going (data-card audit)
                    print(f"[r1] skip {clip_id}: {type(e).__name__}: {e}")
                    continue
                rows.append({"clip_id": clip_id, "chunk": chunk, **s})
    print(f"[r1] scored {len(rows)} clips from {len(zips)} cached chunks "
          f"in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


def attach_meta(scored: pd.DataFrame, root: Path) -> pd.DataFrame:
    idx = pd.read_parquet(Path(root) / "clip_index.parquet")
    dc = pd.read_parquet(Path(root) / "metadata" / "data_collection.parquet")
    cat = idx.join(dc).reset_index().rename(columns={"index": "clip_id"})
    cat["clip_id"] = cat["clip_id"].astype(str)
    keep = [c for c in ("clip_id", "country", "hour_of_day") if c in cat.columns]
    return scored.merge(cat[keep], on="clip_id", how="left")


def select(scored: pd.DataFrame, target: int) -> pd.DataFrame:
    """Top urban scores, round-robin over countries. Cap widened vs R0 so the
    target is reachable from the gate-pass pool while keeping geo spread."""
    passing = scored[scored["urban_score"] > 0].sort_values(
        "urban_score", ascending=False)
    if "country" not in passing or passing["country"].isna().all():
        return passing.head(target).reset_index(drop=True)
    n_c = max(1, passing["country"].nunique())
    cap = max(20, int(np.ceil(target / n_c)) * 3)
    sel, per_c = [], {}
    for _, r in passing.iterrows():
        c = str(r["country"])
        if per_c.get(c, 0) >= cap:
            continue
        sel.append(r)
        per_c[c] = per_c.get(c, 0) + 1
        if len(sel) >= target:
            break
    return pd.DataFrame(sel).reset_index(drop=True)


def build_report(scored: pd.DataFrame, sel: pd.DataFrame, target: int) -> dict:
    passing = scored[scored["urban_score"] > 0]
    qs = [50, 75, 90, 95, 99]
    rep = {
        "target": target,
        "cached_chunks": int(scored["chunk"].nunique()),
        "clips_scored_total": int(len(scored)),
        "clips_passing_gates": int(len(passing)),
        "gate_pass_rate": round(len(passing) / max(1, len(scored)), 4),
        "selected": int(len(sel)),
        "r1_reachable_from_cache": bool(len(passing) >= target),
        "urban_score_percentiles_passing": {
            f"p{p}": round(float(np.percentile(passing["urban_score"], p)), 4)
            for p in qs} if len(passing) else {},
        "mean_urban_score_selected": round(float(sel["urban_score"].mean()), 4),
        "mean_speed_selected": round(float(sel["mean_speed"].mean()), 4),
        "n_camera_chunks": int(sel["chunk"].nunique()),
        "camera_download_gb_est": round(sel["chunk"].nunique() * CAM_CHUNK_GB, 1),
        "camera_chunks_needed": sorted(sel["chunk"].unique().tolist()),
    }
    if "country" in sel:
        rep["by_country_selected"] = sel["country"].value_counts().to_dict()
    if "hour_of_day" in sel:
        rep["by_hour_selected"] = sel["hour_of_day"].value_counts().sort_index().to_dict()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\Admin\tanitad-data\physicalai")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--out-subdir", default="r1")
    a = ap.parse_args()
    root = Path(a.root)
    scored = attach_meta(score_cached_egomotion(root / "labels" / "egomotion"), root)
    sel = select(scored, a.target)
    out = root / a.out_subdir
    out.mkdir(exist_ok=True)
    sel.to_parquet(out / "r1_selection.parquet")
    rep = build_report(scored, sel, a.target)
    (out / "R1_REPORT.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
