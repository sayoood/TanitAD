#!/usr/bin/env python3
"""Write the metric cache's metadata CSV with the DEVKIT'S OWN writer, from the pickles on disk.

    PYTHONPATH=<v1.1 tree> <navsim venv python> code/write_cache_metadata.py --cache-name navtest

WHY. `run_metric_caching`'s final full pass exists only to emit this one CSV — the file
`MetricCacheLoader` reads — but to emit it the devkit rebuilds a `Scene` and a `NavSimScenario`
for all 12,146 tokens behind a SceneLoader that holds every log: **2.0 GB RSS for ~53 minutes**.
MEASURED 2026-09-20: all 12,146 pickles were cached successfully (34/34 groups, 0 failures) and
that final pass was then killed by the RAM guard at **1,507 MB available** — leaving a 215-row
CSV and a step that correctly failed its own count guard. Repeating a 53-minute 2 GB window to
re-derive a list of paths that are already on disk is the expensive way to be exactly as right.

⛔ THE FILE IS STILL WRITTEN BY THE DEVKIT. `nuplan…cache_metadata_entry.save_cache_metadata` is
called with `CacheMetadataEntry` objects; only the ENTRY LIST comes from a directory scan instead
of from rebuilding every scene. Nothing about the format, the column name or the path rendering is
re-implemented here.

⛔ AND IT IS VERIFIED ON CONTENT, not on exit code: every row's file must exist, every row's parent
directory must be a navtest token, the row set must EQUAL the split's 12,146 tokens, and the
PATCHED `MetricCacheLoader` must then map 12,146 tokens to 12,146 distinct existing paths. Any
failure refuses (exit 1) and the devkit's own pass remains the fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_v1  # noqa: E402  (navtest_tokens: the yaml parse, pinned against yaml.safe_load)

EXP = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-name", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cache = Path(f"{EXP}/metric_cache_{a.cache_name}")
    want = set(run_v1.navtest_tokens()[1])
    paths = sorted(str(p) for p in cache.glob("*/*/*/metric_cache.pkl"))
    toks = [re.split(r"[\\/]", p)[-2] for p in paths]
    rep = {"cache": str(cache), "n_pickles_found": len(paths), "n_expected": len(want),
           "n_distinct_tokens": len(set(toks)),
           "n_missing": len(want - set(toks)), "n_unexpected": len(set(toks) - want),
           "missing_examples": sorted(want - set(toks))[:10],
           "unexpected_examples": sorted(set(toks) - want)[:10]}
    rep["ok_to_write"] = bool(len(paths) == len(want) == len(set(toks))
                              and not rep["n_missing"] and not rep["n_unexpected"])
    if not rep["ok_to_write"]:
        print(json.dumps(rep, indent=1))
        print("⛔ refusing to write a metadata CSV that does not name exactly the split's tokens")
        return 1
    if a.dry_run:
        print(json.dumps(rep, indent=1))
        return 0

    from nuplan.planning.training.experiments.cache_metadata_entry import (  # noqa: E402
        CacheMetadataEntry, save_cache_metadata)
    save_cache_metadata([CacheMetadataEntry(Path(p)) for p in paths], cache, 0)

    # ---- verify on CONTENT ---------------------------------------------------------- #
    csvs = sorted((cache / "metadata").glob("*.csv"))
    rep["metadata_csvs"] = [str(c) for c in csvs]
    rows = []
    if len(csvs) == 1:
        rows = [r for r in csvs[0].read_text(encoding="utf-8").splitlines()[1:] if r.strip()]
    rep["n_rows"] = len(rows)
    rep["n_rows_exist_on_disk"] = sum(1 for r in rows if os.path.exists(r.strip()))
    row_toks = [re.split(r"[\\/]", r.strip())[-2] for r in rows]
    rep["n_row_tokens_distinct"] = len(set(row_toks))
    rep["row_tokens_equal_split"] = set(row_toks) == want
    rep["header"] = csvs[0].read_text(encoding="utf-8").splitlines()[0] if csvs else None

    sys.path.insert(0, os.environ.get("W3_V11_TREE", run_v1.V11_TREE))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "e1w", "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
               "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
    e1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e1)
    mapped = e1._load_metric_cache_paths_patched(None, cache)
    rep["loader_n_tokens"] = len(mapped)
    rep["loader_tokens_equal_split"] = set(mapped) == want
    rep["loader_n_files_exist"] = sum(1 for v in mapped.values() if os.path.exists(v))
    rep["PASS"] = bool(rep["n_rows"] == len(want) == rep["n_rows_exist_on_disk"]
                       == rep["n_row_tokens_distinct"] == rep["loader_n_tokens"]
                       == rep["loader_n_files_exist"]
                       and rep["row_tokens_equal_split"] and rep["loader_tokens_equal_split"])
    out = os.path.join(os.path.dirname(HERE), "raw", f"metadata_csv_{a.cache_name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith("_examples")}, indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
