"""PhysicalAI-AV Stage R0 builder (DATA_STRATEGY.md §2, D-012 — usage tagged `data:physicalai`).

Strategy: egomotion labels are ~40 MB/chunk while camera chunks are ~2 GB, so
selection runs on egomotion FIRST, then only selected clips' camera chunks are
fetched. Urban-ness has no explicit tag in the dataset; we use a measured
motion proxy: low mean speed + stop fraction + yaw activity = urban
interaction. The achieved (country x hour x score) distribution is written
next to the selection for the data card (semantic-coverage audit).

Stages:
  select        download N egomotion chunks (stratified by country), score
                clips, write r0_selection.parquet + R0_REPORT.md
  fetch-camera  download camera chunks containing selected clips, extract only
                selected members, delete the zips (disk discipline)

Usage:
  python scripts/physicalai_r0.py select --chunks 30 --target 500
  python scripts/physicalai_r0.py fetch-camera --max-chunks 15
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import os
ROOT = Path(os.environ.get("TANITAD_PHYSICALAI_ROOT",
                           r"C:\Users\Admin\tanitad-data\physicalai"))
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
EGO_TMPL = "labels/egomotion/egomotion.chunk_{chunk_id:04d}.zip"
CAM_TMPL = ("camera/camera_front_wide_120fov/"
            "camera_front_wide_120fov.chunk_{chunk_id:04d}.zip")


def _hf():
    try:
        # dev machine: proxy TLS + Keys.txt. On pods, export HF_TOKEN instead.
        from tanitad.keys import enable_tls, load_keys
        enable_tls()
        load_keys()
    except Exception as e:
        print(f"[r0] keys helper unavailable ({e}) — relying on env HF_TOKEN")
    from huggingface_hub import hf_hub_download
    return hf_hub_download


def _load_catalog() -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "clip_index.parquet")
    dc = pd.read_parquet(ROOT / "metadata" / "data_collection.parquet")
    cat = idx.join(dc)
    cat = cat[cat["clip_is_valid"] & (cat["split"].astype(str) == "train")]
    return cat


def _pick_chunks(cat: pd.DataFrame, n_chunks: int) -> list[int]:
    """Spread chunks across countries (round-robin over country-majority)."""
    by_country = (cat.groupby(["chunk", "country"]).size().reset_index(name="n")
                  .sort_values("n", ascending=False)
                  .drop_duplicates("chunk"))
    picked, seen = [], {}
    for _, row in by_country.sample(frac=1.0, random_state=0).iterrows():
        c = row["country"]
        if seen.get(c, 0) >= max(2, n_chunks // 12):
            continue
        picked.append(int(row["chunk"]))
        seen[c] = seen.get(c, 0) + 1
        if len(picked) >= n_chunks:
            break
    return picked


def _urban_score_from_egomotion(df: pd.DataFrame) -> dict | None:
    """Motion-statistics urban proxy against the ACTUAL egomotion schema:
    timestamp, q{x,y,z,w}, x/y/z [m], v{x,y,z} [m/s], a{x,y,z}, curvature.

    First version of this scorer matched 'vel' substrings (vx/vy don't match),
    fell through to quaternion columns and selected 500 PARKED clips with a
    uniform score — caught by the R0 report audit (mean speed 6e-9). Lesson
    baked in here: hard gates that require actual driving, plus explicit
    column names with a loud failure when absent.
    """
    need = {"vx", "vy", "curvature"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"unexpected egomotion schema: {list(df.columns)}")
    v = np.linalg.norm(df[["vx", "vy"]].to_numpy(np.float64), axis=1)
    curv = df["curvature"].to_numpy(np.float64)
    mean_v = float(np.nanmean(v))
    stop_frac = float(np.mean(v < 0.5))
    dur = 20.0                                    # clips are 20 s
    distance = float(np.nanmean(v) * dur)
    yaw_rate = np.abs(curv) * v                   # |kappa| * v = yaw rate
    yaw_act = float(np.nanmean(yaw_rate))
    # HARD GATES — the clip must contain actual driving:
    if not (2.0 <= mean_v <= 14.0) or stop_frac > 0.8 or distance < 40.0:
        score = 0.0
    else:
        # urban interaction: moderate speed band + some (not permanent)
        # stopping + turning activity
        speed_band = 1.0 - abs(mean_v - 8.0) / 8.0          # peak at 8 m/s
        stop_term = min(stop_frac / 0.3, 1.0)               # saturate at 30 %
        turn_term = min(yaw_act / 0.08, 1.0)                # ~4.6 deg/s mean
        score = max(0.0, speed_band) + stop_term + turn_term
    return {"mean_speed": mean_v, "stop_frac": stop_frac, "distance": distance,
            "yaw_activity": yaw_act, "urban_score": score}


def stage_select(n_chunks: int, target: int) -> None:
    dl = _hf()
    cat = _load_catalog()
    chunks = _pick_chunks(cat, n_chunks)
    print(f"[r0] scoring {n_chunks} egomotion chunks: {chunks}")
    rows = []
    for ch in chunks:
        try:
            zp = dl(REPO, EGO_TMPL.format(chunk_id=ch), repo_type="dataset",
                    local_dir=str(ROOT))
        except Exception as e:
            print(f"[r0] chunk {ch} download failed: {e}")
            continue
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.endswith(".egomotion.parquet"):
                    continue
                clip_id = name.split("/")[-1].split(".")[0]
                try:
                    df = pd.read_parquet(io.BytesIO(z.read(name)))
                except Exception:
                    continue
                s = _urban_score_from_egomotion(df)
                if s is None:
                    if not rows:      # log schema once for the data card
                        print("[r0] unscored schema:", list(df.columns)[:20])
                    continue
                rows.append({"clip_id": clip_id, "chunk": ch, **s})
        print(f"[r0] chunk {ch}: cumulative scored clips = {len(rows)}")
    scored = pd.DataFrame(rows)
    if scored.empty:
        raise SystemExit("[r0] nothing scored — inspect egomotion schema")
    meta = _load_catalog().reset_index().rename(columns={"index": "clip_id"})
    meta["clip_id"] = meta["clip_id"].astype(str)
    scored = scored.merge(meta[["clip_id", "country", "hour_of_day"]],
                          on="clip_id", how="left")
    # stratified pick: top urban scores round-robin over countries
    scored = scored.sort_values("urban_score", ascending=False)
    sel, per_c = [], {}
    cap = max(10, target // max(1, scored["country"].nunique()) * 2)
    for _, r in scored.iterrows():
        c = str(r["country"])
        if per_c.get(c, 0) >= cap:
            continue
        sel.append(r)
        per_c[c] = per_c.get(c, 0) + 1
        if len(sel) >= target:
            break
    sel_df = pd.DataFrame(sel)
    ROOT.joinpath("r0").mkdir(exist_ok=True)
    sel_df.to_parquet(ROOT / "r0" / "r0_selection.parquet")
    report = {
        "target": target, "selected": len(sel_df),
        "chunks_scored": chunks,
        "mean_urban_score": float(sel_df["urban_score"].mean()),
        "mean_speed_selected": float(sel_df["mean_speed"].mean()),
        "by_country": sel_df["country"].value_counts().to_dict(),
        "camera_chunks_needed": sorted(sel_df["chunk"].unique().tolist()),
    }
    (ROOT / "r0" / "R0_REPORT.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def stage_fetch_camera(max_chunks: int) -> None:
    dl = _hf()
    sel = pd.read_parquet(ROOT / "r0" / "r0_selection.parquet")
    out = ROOT / "r0" / "camera_front_wide"
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(sel["clip_id"].astype(str))
    for ch in sorted(sel["chunk"].unique())[:max_chunks]:
        zp = Path(dl(REPO, CAM_TMPL.format(chunk_id=int(ch)),
                     repo_type="dataset", local_dir=str(ROOT)))
        n = 0
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                clip_id = name.split("/")[-1].split(".")[0]
                if clip_id in wanted:
                    z.extract(name, out)
                    n += 1
        zp.unlink()                       # 2 GB each — never keep the zips
        print(f"[r0] chunk {ch}: extracted {n} files, zip deleted")
    print("[r0] camera fetch done ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "fetch-camera"])
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--max-chunks", type=int, default=15)
    a = ap.parse_args()
    if a.stage == "select":
        stage_select(a.chunks, a.target)
    else:
        stage_fetch_camera(a.max_chunks)
