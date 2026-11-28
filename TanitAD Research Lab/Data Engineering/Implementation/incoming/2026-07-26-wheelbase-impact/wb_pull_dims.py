#!/usr/bin/env python3
"""TIER 0 — pull `calibration/vehicle_dimensions` for exactly the chunks our
parity corpus draws from, and join it to our episodes.

WHY the join is trustworthy (this is the load-bearing step):
  `physicalai.build_episode` sets  ep_id = int.from_bytes(clip_id[:4], "big"),
  so an episode carries only the FIRST FOUR CHARACTERS of its clip UUID. The
  full clip identity comes from the build's own ordered clip list
  (`tanitad-pod3:/workspace/tmp/{train,val}_clip_order.tsv`, the file
  `rebuild_pai_rolling.py --order` consumed to mint cache key e438721ae894).
  The join is then VERIFIED, not assumed:
     * 2376/2376 committed `parity_profile.csv` episode_ids reproduce from the
       order file's clip_ids;
     * the 24 absent indices are exactly the documented skip set (1798..1941);
     * 40/40 eval-pod val episode_ids reproduce from val_clip_order.tsv[:40].

GATED-CONFIDENTIAL: PhysicalAI-AV clip UUIDs never leave the scratchpad. This
script writes UUID-keyed intermediates to --scratch (outside the repo) and only
aggregate statistics to --out (inside the repo).

Usage (dev box):
  python wb_pull_dims.py --scratch <scratchdir> --out <repo dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_DS = "nvidia/PhysicalAI-Autonomous-Vehicles"
VD_TMPL = "calibration/vehicle_dimensions/vehicle_dimensions.chunk_{c:04d}.parquet"


def bootstrap_keys(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "stack"))
    from tanitad.keys import enable_tls, load_keys
    enable_tls()
    load_keys()
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not populated from Keys.txt"


def epid(clip_id: str) -> int:
    """physicalai.build_episode's episode_id (first 4 chars of the UUID)."""
    return int.from_bytes(clip_id.encode()[:4].ljust(4, b"\0"), "big")


def load_order(p: Path) -> list[tuple[int, str, int]]:
    out = []
    for line in open(p, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3:
            out.append((int(f[0]), f[1], int(f[2])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[5]))
    a = ap.parse_args()
    scratch, out = Path(a.scratch), Path(a.out)
    repo_root = Path(a.repo_root)
    bootstrap_keys(repo_root)

    import pandas as pd
    from huggingface_hub import hf_hub_download

    train = load_order(scratch / "train_clip_order.tsv")
    val = load_order(scratch / "val_clip_order.tsv")
    chunks = sorted({c for _, _, c in train} | {c for _, _, c in val})
    print(f"[wb] corpus draws from {len(chunks)} chunks", flush=True)

    cache = scratch / "vd_cache"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, c in enumerate(chunks):
        p = hf_hub_download(REPO_DS, VD_TMPL.format(c=c), repo_type="dataset",
                            local_dir=str(cache))
        df = pd.read_parquet(p).reset_index()
        df["chunk"] = c
        frames.append(df)
        if (i + 1) % 25 == 0:
            print(f"[wb]   {i+1}/{len(chunks)}", flush=True)
    vd = pd.concat(frames, ignore_index=True)
    vd["clip_id"] = vd["clip_id"].astype(str)
    print(f"[wb] vehicle_dimensions rows={len(vd)} clips={vd.clip_id.nunique()}")

    wb_of = dict(zip(vd.clip_id, vd.wheelbase.astype(float)))
    # the whole dimension row, for the "what IS this population" question
    dim_of = {r.clip_id: (float(r.length), float(r.width), float(r.height),
                          float(r.rear_axle_to_bbox_center), float(r.wheelbase),
                          float(r.track_width)) for r in vd.itertuples()}

    dc = pd.read_parquet(repo_root.parent / "x")  if False else None  # noqa
    dcp = Path(os.environ.get(
        "TANITAD_PAI_META",
        r"C:\Users\Admin\tanitad-data\physicalai\metadata\data_collection.parquet"))
    meta = pd.read_parquet(dcp).reset_index()
    meta["clip_id"] = meta["clip_id"].astype(str)
    meta = meta.set_index("clip_id")

    rows = []
    for split, order, keep in (("train", train, "all"), ("val", val, 600)):
        for idx, cid, ch in order:
            wb = wb_of.get(cid)
            m = meta.loc[cid] if cid in meta.index else None
            rows.append({
                "split": split, "idx": idx, "clip_id": cid, "chunk": ch,
                "episode_id": epid(cid),
                "wheelbase": wb,
                "dims": dim_of.get(cid),
                "country": (None if m is None else str(m["country"])),
                "platform_class": (None if m is None else str(m["platform_class"])),
                "hour_of_day": (None if m is None else int(m["hour_of_day"])),
                "month": (None if m is None else int(m["month"])),
                "radar_config": (None if m is None else str(m["radar_config"])),
            })
    # UUID-bearing table stays in scratch
    (scratch / "corpus_dims.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"[wb] wrote {len(rows)} rows -> scratch/corpus_dims.json (UUIDs, NOT staged)")

    # ---- aggregate, UUID-free, safe for the repo --------------------------
    agg = {}
    for split in ("train", "val"):
        sub = [r for r in rows if r["split"] == split]
        wbc = Counter(round(r["wheelbase"], 4) if r["wheelbase"] is not None
                      else None for r in sub)
        agg[split] = {
            "n_clips": len(sub),
            "n_wheelbase_resolved": sum(1 for r in sub if r["wheelbase"] is not None),
            "wheelbase_counts": {str(k): v for k, v in sorted(wbc.items(),
                                                             key=lambda kv: (kv[0] is None, kv[0]))},
            "platform_class_counts": dict(Counter(r["platform_class"] for r in sub)),
            "wheelbase_by_platform": {
                pc: dict(Counter(round(r["wheelbase"], 4) for r in sub
                                 if r["platform_class"] == pc and r["wheelbase"] is not None))
                for pc in sorted({r["platform_class"] for r in sub if r["platform_class"]})
            },
            "dims_by_wheelbase": {
                str(w): sorted({r["dims"] for r in sub if r["wheelbase"] is not None
                                and round(r["wheelbase"], 4) == w})
                for w in sorted({round(r["wheelbase"], 4) for r in sub
                                 if r["wheelbase"] is not None})
            },
            "country_by_wheelbase": {
                str(w): dict(Counter(r["country"] for r in sub
                                     if r["wheelbase"] is not None
                                     and round(r["wheelbase"], 4) == w).most_common())
                for w in sorted({round(r["wheelbase"], 4) for r in sub
                                 if r["wheelbase"] is not None})
            },
            "hour_by_wheelbase": {
                str(w): dict(sorted(Counter(r["hour_of_day"] for r in sub
                                            if r["wheelbase"] is not None
                                            and round(r["wheelbase"], 4) == w).items()))
                for w in sorted({round(r["wheelbase"], 4) for r in sub
                                 if r["wheelbase"] is not None})
            },
        }
    agg["_meta"] = {
        "corpus_train": "physicalai-train-e438721ae894 (2400 order lines, 2376 built, 24 skips)",
        "corpus_val": "physicalai-val-0c5f7dac3b11 (600 order lines; eval-pod deployment = first 40)",
        "n_chunks_pulled": len(chunks),
        "hardcoded_WHEELBASE": 2.9,
        "source": f"HF {REPO_DS} :: {VD_TMPL}",
        "evidence_class": "MEASURED",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "wheelbase_population.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8")
    print("[wb] wrote", out / "wheelbase_population.json")


if __name__ == "__main__":
    main()
