#!/usr/bin/env python3
"""⛔ MANDATORY LEAKAGE CONTROL: are the NuRec T1 scenes in the arms' TRAINING corpus?

NuRec scenes are reconstructions of **PhysicalAI-Autonomous-Vehicles clips and carry that
clip's UUID as their scene id**. flagship-v1 and REF-C were trained on
``physicalai-train-e438721ae894`` — 2 376 episodes drawn from the SAME corpus. If a T1
scene is a train clip, then the strategic score on it is partly memorisation and the
number is not a generalisation claim.

The check is a set intersection against the corpus's own catalogue
(``clip_index.parquet`` + ``metadata/data_collection.parquet``, the exact pair
``scripts/physicalai_r0.py:53-58`` uses to define the train split), so it is decided by
the dataset, not by our cache.

⚠️ **Two-sided reporting.** A clip in ``split == "train"`` is only a *candidate* leak: our
build took 2 376 of the split's clips, not all of them. So this reports BOTH
``in_train_split`` (the upper bound on leakage) and ``in_val_split`` / ``absent``, and the
family is re-scored on the leak-free subset rather than the whole set being discarded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"


def fetch(rel: str, dest: Path) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    h = {"User-Agent": "tanitad-leakcheck/1"}
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        h["Authorization"] = "Bearer " + tok
    with urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                timeout=600) as r:
        dest.write_bytes(r.read())
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=str(HERE / "results" / "junction_turn_scenes.tsv"))
    ap.add_argument("--labels", default=str(HERE / "results" / "strategic_gt_t1"))
    ap.add_argument("--cache", default="/home/nvidia/pai_catalog")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    c = Path(a.cache)
    idx = pd.read_parquet(fetch("clip_index.parquet", c / "clip_index.parquet"))
    try:
        dc = pd.read_parquet(fetch("metadata/data_collection.parquet",
                                   c / "data_collection.parquet"))
        cat = idx.join(dc)
    except Exception as e:                                       # noqa: BLE001
        cat, dc_err = idx, repr(e)[:200]
    else:
        dc_err = None

    cat = cat.reset_index()
    id_col = next((k for k in ("clip_id", "index", "id") if k in cat.columns),
                  cat.columns[0])
    cat[id_col] = cat[id_col].astype(str)
    split = cat["split"].astype(str) if "split" in cat.columns else None
    valid = cat["clip_is_valid"] if "clip_is_valid" in cat.columns else None

    train = set(cat.loc[(split == "train") & (valid if valid is not None else True),
                        id_col]) if split is not None else set()
    val = set(cat.loc[(split == "val") | (split == "validation"), id_col]) \
        if split is not None else set()
    known = set(cat[id_col])

    t1 = []
    for line in Path(a.tsv).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 4 and f[0] == "T1":
            t1.append(f[2])
    scored = sorted(p.stem.replace("strategic_gt_", "")
                    for p in Path(a.labels).glob("strategic_gt_*.json"))
    with_branch = []
    for s in scored:
        d = json.loads((Path(a.labels) / f"strategic_gt_{s}.json").read_text())
        if d.get("ADMISSIBLE") and d.get("n_SCOREABLE_events", 0) > 0:
            with_branch.append(s)

    def part(ids):
        return {"n": len(ids),
                "in_train_split": sorted(x for x in ids if x in train),
                "in_val_split": sorted(x for x in ids if x in val),
                "not_in_catalogue": sorted(x for x in ids if x not in known)}

    out = {
        "tool": "leakage_check_t1.py", "evidence_class": "MEASURED",
        "catalogue": {"repo": REPO, "n_clips": int(len(cat)),
                      "n_train_valid": len(train), "n_val": len(val),
                      "columns": list(cat.columns)[:24],
                      "data_collection_join_error": dc_err},
        "T1_all": part(t1),
        "T1_labelled": part(scored),
        "T1_with_a_scoreable_branch": part(with_branch),
    }
    for k in ("T1_all", "T1_labelled", "T1_with_a_scoreable_branch"):
        b = out[k]
        b["n_in_train_split"] = len(b["in_train_split"])
        b["n_in_val_split"] = len(b["in_val_split"])
        b["n_not_in_catalogue"] = len(b["not_in_catalogue"])
        b["leak_upper_bound_frac"] = round(b["n_in_train_split"] / max(b["n"], 1), 4)
    out["VERDICT"] = (
        "CLEAN — no scored T1 scene is a clip of the training split"
        if out["T1_with_a_scoreable_branch"]["n_in_train_split"] == 0 else
        f"{out['T1_with_a_scoreable_branch']['n_in_train_split']} of "
        f"{out['T1_with_a_scoreable_branch']['n']} scored scenes are TRAIN-SPLIT clips — "
        "an UPPER BOUND on leakage (our build took 2376 of the split). Re-score on the "
        "complement before quoting a generalisation number.")
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if not isinstance(vv, list)})
                      for k, v in out.items()}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
