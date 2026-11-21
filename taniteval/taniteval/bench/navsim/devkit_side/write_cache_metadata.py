#!/usr/bin/env python3
"""(NAVSIM VENV) Write a v2 METRIC CACHE's metadata CSV with the DEVKIT'S OWN writer, from the pickles
on disk — verified on CONTENT — then ``CACHE_MANIFEST.json`` (per-file sha256) and ``CACHE_DONE.json``.

    <navsim venv python> write_cache_metadata.py --cache <dir> --expected-tokens <json> --split navtest \
        --logs-dir <dir> --out-json <json> [--load all|sample|none]

W8 (EvalFlyWheel, 2026-09-26). PROMOTED from W3 ``2026-09-19-navsim-v1-navtest/code/write_cache_metadata.py``
(v1.1): same mechanism, v2 devkit, and the verification extended to a per-file manifest + an entry load.

WHY (W3, MEASURED 2026-09-20): the devkit's own final pass exists only to emit ONE CSV — the file
``MetricCacheLoader`` reads — and to emit it rebuilds every scene behind a SceneLoader that holds every log
(2 GB RSS for ~53 min on navtest); a per-log-group build overwrites that CSV on every group
(``save_cache_metadata`` always writes ``<cache.name>_metadata_node_0.csv``). So the CSV is written ONCE,
at the end, from the pickles on disk.

⛔ THE FILE IS STILL WRITTEN BY THE DEVKIT: nuPlan's ``save_cache_metadata`` with ``CacheMetadataEntry``
objects; only the entry LIST comes from a directory scan. Nothing about format, column or path rendering is
re-implemented here.

⛔ AND IT IS VERIFIED ON CONTENT, never on exit code: the pickle token set must EQUAL the expected set (no
missing, no extra, no duplicate across log/type dirs); after writing, every CSV row must exist and the
row tokens must equal the expected set; the PATCHED ``MetricCacheLoader`` path parse (the suite's
``navsim_win._load_metric_cache_paths_patched``) must map exactly the expected tokens to existing files;
and loaded entries must carry their own directory's log/token, ``scene_type == ORIGINAL`` and a human
trajectory. Any failure exits 1 and writes NO ``CACHE_DONE.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--expected-tokens", required=True, type=Path)
    ap.add_argument("--split", required=True, help="the devkit train_test_split (navtest)")
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--load", choices=("all", "sample", "none"), default="all")
    ap.add_argument("--sample-every", type=int, default=25)
    a = ap.parse_args(argv)
    t0 = time.time()
    cache = a.cache
    exp_doc = json.loads(a.expected_tokens.read_text(encoding="utf-8"))
    want = set(exp_doc["tokens"])
    rep = {"schema": "w8-write-cache-metadata/1", "cache": str(cache).replace(os.sep, "/"),
           "n_expected": len(want), "split": a.split}

    # ---- the pickles on disk --------------------------------------------------- #
    paths = sorted(cache.glob("*/*/*/metric_cache.pkl"))
    toks = [p.parts[-2] for p in paths]
    rep.update({"n_pickles_found": len(paths), "n_distinct_tokens": len(set(toks)),
                "n_missing": len(want - set(toks)), "n_unexpected": len(set(toks) - want),
                "missing_examples": sorted(want - set(toks))[:10],
                "unexpected_examples": sorted(set(toks) - want)[:10]})
    rep["ok_to_write"] = bool(len(paths) == len(want) == len(set(toks))
                              and not rep["n_missing"] and not rep["n_unexpected"])
    if not rep["ok_to_write"]:
        rep["PASS"] = False
        a.out_json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))
        print("⛔ refusing to write a metadata CSV that does not name exactly the expected tokens")
        return 1

    # ---- the devkit's own writer ---------------------------------------------- #
    from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata
    save_cache_metadata([CacheMetadataEntry(p) for p in paths], cache, 0)

    # ---- verify on CONTENT ------------------------------------------------------ #
    csvs = sorted((cache / "metadata").glob("*.csv"))
    rep["metadata_csvs"] = [c.name for c in csvs]
    rows = []
    if len(csvs) == 1:
        rows = [r for r in csvs[0].read_text(encoding="utf-8").splitlines()[1:] if r.strip()]
    rep["n_rows"] = len(rows)
    rep["n_rows_exist_on_disk"] = sum(1 for r in rows if os.path.exists(r.strip()))
    row_toks = [re.split(r"[\\/]", r.strip())[-2] for r in rows]
    rep["n_row_tokens_distinct"] = len(set(row_toks))
    rep["row_tokens_equal_expected"] = set(row_toks) == want
    rep["header"] = csvs[0].read_text(encoding="utf-8").splitlines()[0] if csvs else None
    rep["metadata_csv_sha256"] = hashlib.sha256(csvs[0].read_bytes()).hexdigest() if len(csvs) == 1 else None

    import navsim_win                                        # the suite's promoted E1 wrapper (patched parse)
    mapped = navsim_win._load_metric_cache_paths_patched(None, cache)
    rep["loader_n_tokens"] = len(mapped)
    rep["loader_tokens_equal_expected"] = set(mapped) == want
    rep["loader_n_files_exist"] = sum(1 for v in mapped.values() if os.path.exists(v.strip()))
    # the RUNTIME devkit's own parse (the C: copy carries E1's separator patch at dataloader.py:316-317 of the
    # COPY; the pinned source @0a380a9 splits on '/' only, dataloader.py:316, and raises on this cache)
    try:
        from navsim.common import dataloader as _dl
        _dl.MetricCacheLoader._load_metric_cache_paths(None, cache)
        rep["runtime_devkit_loader"] = {"status": "READS", "module": str(_dl.__file__).replace(os.sep, "/"),
                                        "note": "the C: runtime copy is PATCHED (E1 apply_dataloader_patch.py)"}
    except Exception as e:                                   # noqa: BLE001
        rep["runtime_devkit_loader"] = {"status": "RAISES", "exception": f"{type(e).__name__}: {e}"[:200]}

    # ---- the per-file manifest + entry load ------------------------------------ #
    from navsim.common.enums import SceneFrameType
    manifest, load_fail, n_loaded, n_orig, n_human, load_s = {}, [], 0, 0, 0, 0.0
    step = 1 if a.load == "all" else (a.sample_every if a.load == "sample" else 0)
    n_lzma_ok, lzma_fail = 0, []
    for i, p in enumerate(paths):
        b = p.read_bytes()
        rel = str(p.relative_to(cache)).replace(os.sep, "/")
        manifest[rel] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
        # ⛔ EVERY entry is decompressed (integrity): a RAM-guard yield is an os._exit that can land inside
        # save_buffer, leaving a TRUNCATED pickle whose file still EXISTS — presence is not validity.
        try:
            raw_b = lzma.decompress(b)
            n_lzma_ok += 1
        except Exception as e:                               # noqa: BLE001
            lzma_fail.append({"file": rel, "exception": f"{type(e).__name__}: {e}"[:200]})
            continue
        if step and i % step == 0:
            t1 = time.time()
            try:
                mc = pickle.loads(raw_b)
                n_loaded += 1
                ok_tok = Path(str(mc.file_path)).parent.name == p.parts[-2] or re.split(r"[\\/]", str(mc.file_path))[-2] == p.parts[-2]
                ok_log = str(mc.log_name) == p.parts[-4]
                is_orig = mc.scene_type == SceneFrameType.ORIGINAL
                has_h = mc.human_trajectory is not None
                n_orig += int(is_orig)
                n_human += int(has_h)
                if not (ok_tok and ok_log and is_orig and has_h):
                    load_fail.append({"file": rel, "token_ok": ok_tok, "log_ok": ok_log, "scene_type": str(mc.scene_type),
                                      "human_trajectory": has_h})
            except Exception as e:                           # noqa: BLE001
                load_fail.append({"file": rel, "exception": f"{type(e).__name__}: {e}"[:200]})
            load_s += time.time() - t1
    man = {"schema": "w8-cache-manifest/1", "cache": rep["cache"], "n_files": len(manifest), "files": manifest}
    man_bytes = json.dumps(man, indent=0, sort_keys=True).encode("utf-8")
    (cache / "CACHE_MANIFEST.json").write_bytes(man_bytes)
    rep["manifest_sha256"] = hashlib.sha256(man_bytes).hexdigest()
    rep["manifest_total_bytes"] = sum(v["bytes"] for v in manifest.values())
    rep["integrity"] = {"n_lzma_decompress_ok": n_lzma_ok, "n_failures": len(lzma_fail), "failures": lzma_fail[:20],
                        "what": "every entry lzma-decompressed (a truncated pickle raises)"}
    rep["load"] = {"mode": a.load, "every": step, "n_loaded": n_loaded, "n_scene_type_ORIGINAL": n_orig,
                   "n_with_human_trajectory": n_human, "n_failures": len(load_fail), "failures": load_fail[:20],
                   "load_s": round(load_s, 1)}

    # ---- is this the FULL split? (the devkit yaml, read here) ------------------ #
    import yaml
    dk = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "C:/Users/Admin/navsim-crun/devkit"))
    sf = yaml.safe_load(open(dk / "navsim/planning/script/config/common/train_test_split/scene_filter"
                             / f"{a.split}.yaml", encoding="utf-8"))
    yaml_tokens, yaml_logs = set(sf["tokens"]), set(sf["log_names"])
    full = want == yaml_tokens
    tokens_sha = hashlib.sha256("\n".join(sorted(want)).encode("utf-8")).hexdigest()
    rep["PASS"] = bool(rep["n_rows"] == len(want) == rep["n_rows_exist_on_disk"] == rep["n_row_tokens_distinct"]
                       == rep["loader_n_tokens"] == rep["loader_n_files_exist"]
                       and rep["row_tokens_equal_expected"] and rep["loader_tokens_equal_expected"]
                       and not load_fail and not lzma_fail and n_lzma_ok == len(paths)
                       and (a.load == "none" or n_loaded > 0))
    rep["wall_s"] = round(time.time() - t0, 1)
    if rep["PASS"]:
        done = {"marker": "CACHE_DONE", "schema": "w8-cache-done/1",
                "written_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "written_by": "EvalFlyWheel W8 (2026-09-26-navtest-single-stage) via "
                              "taniteval/taniteval/bench/navsim/devkit_side/write_cache_metadata.py",
                "split": a.split, "cache_path": rep["cache"], "logs_dir": str(a.logs_dir).replace(os.sep, "/"),
                "token_set": "FULL_SPLIT" if full else "SUBSET",
                "n_cached": {"total": len(want), "stage_one_original": len(want), "stage_two_synthetic": 0},
                "n_expected_from_yaml": {"stage_one": len(yaml_tokens), "stage_two": 0, "log_groups": len(yaml_logs)},
                "token_sets_equal_yaml": bool(full), "token_sets_equal_expected": True,
                "tokens_sha256": tokens_sha,
                "scene_types_match_yaml": bool(n_orig == n_loaded and n_loaded > 0),
                "scene_types_evidence": (f"{n_orig}/{n_loaded} LOADED entries (load mode {a.load}, every {step}) are "
                                         "SceneFrameType.ORIGINAL with their own directory's log/token and a human "
                                         "trajectory; the build never loads synthetic scenes (include_synthetic_scenes "
                                         "false in the devkit's navtest scene filter)"),
                "metadata_csv": rep["metadata_csvs"][0], "metadata_csv_sha256": rep["metadata_csv_sha256"],
                "manifest": "CACHE_MANIFEST.json", "manifest_sha256": rep["manifest_sha256"],
                "manifest_total_bytes": rep["manifest_total_bytes"],
                "integrity_every_entry_lzma_ok": rep["integrity"]["n_lzma_decompress_ok"],
                "reading_on_windows": rep["runtime_devkit_loader"],
                "verified_by": "W8 write_cache_metadata.py (content: rows, files, patched-loader parse, entry load)"}
        (cache / "CACHE_DONE.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
        rep["cache_done"] = done
    a.out_json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("missing_examples", "unexpected_examples", "cache_done")},
                     indent=1))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
