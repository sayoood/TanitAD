#!/usr/bin/env python3
"""Verify a NavSim metric cache by CONTENT, then (and only then) write CACHE_DONE.json.

Controls (SPEC §4):
  C3  counts: the token set on disk == the token set in the metadata CSV == the yaml sets,
      16 ORIGINAL + 204 SYNTHETIC for warmup; zero anywhere is a FAIL.
  C2  scene-type: every entry's pickled `scene_type` must match its yaml membership
      (`tokens` -> ORIGINAL, `reactive_synthetic_initial_tokens` -> SYNTHETIC). The cache
      assigns the type by `len(token) == 17` (metric_cache_processor.py:317); the yaml
      membership is an independent derivation.
  +   the Windows loader defect, MEASURED: the UNPATCHED MetricCacheLoader on this CSV,
      and the patched one, side by side.

⛔ CACHE_DONE.json is written ONLY if every check passes, and atomically (tmp + rename),
so a poller can never read a partial or premature marker.

Usage: python verify_cache.py --split warmup_two_stage --cache <dir> --manifest <wrapper manifest>
       --hooks <wrapper hooks json> --out <raw json> [--write-marker]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import lzma
import os
import pickle
import sys
import time
from pathlib import Path

import yaml

TTS = "C:/Users/Admin/navsim-crun/devkit/navsim/planning/script/config/common/train_test_split/"


def yaml_sets(split: str):
    sf = yaml.safe_load(open(TTS + f"scene_filter/{split}.yaml", encoding="utf-8"))
    return set(sf["tokens"]), set(sf["reactive_synthetic_initial_tokens"]), len(sf["log_names"])


def load_entry(p: Path):
    t = time.perf_counter()
    with lzma.open(p, "rb") as f:
        mc = pickle.load(f)
    return {"token": p.parent.name, "scene_type": str(mc.scene_type), "log_name": mc.log_name,
            "map_root": mc.map_parameters.map_root, "map_name": mc.map_parameters.map_name,
            "human_traj_is_none": mc.human_trajectory is None, "bytes": p.stat().st_size,
            "load_s": round(time.perf_counter() - t, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--hooks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--write-marker", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    s1_yaml, s2_yaml, n_logs = yaml_sets(a.split)
    res = {"split": a.split, "cache": str(a.cache), "checks": {}, "fail": []}

    # --- two independent listings -------------------------------------------------
    on_disk = sorted(a.cache.rglob("metric_cache.pkl"))
    disk_tokens = {p.parent.name for p in on_disk}
    csvs = sorted(p for p in (a.cache / "metadata").iterdir() if ".csv" in str(p))
    res["metadata_csvs"] = [str(p) for p in csvs]
    csv_rows = open(csvs[0], "r").read().splitlines()[1:] if csvs else []
    res["csv_header"] = open(csvs[0], "r").read().splitlines()[0] if csvs else None
    res["csv_rows_contain_backslash"] = sum("\\" in r for r in csv_rows)
    res["csv_rows_contain_forwardslash"] = sum("/" in r for r in csv_rows)

    # --- the Windows loader defect, measured ------------------------------------------
    from navsim.common.dataloader import MetricCacheLoader
    try:
        n_unpatched = len(MetricCacheLoader(a.cache).tokens)
        res["unpatched_loader"] = {"status": "OK", "n_tokens": n_unpatched}
    except Exception as e:                                                # noqa: BLE001
        res["unpatched_loader"] = {"status": "RAISES", "exception": f"{type(e).__name__}: {e}"}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import navsim_win
    MetricCacheLoader._load_metric_cache_paths = navsim_win._load_metric_cache_paths_patched
    patched = MetricCacheLoader(a.cache)
    csv_tokens = set(patched.tokens)
    res["patched_loader"] = {"status": "OK", "n_tokens": len(csv_tokens)}

    # --- C3 counts -----------------------------------------------------------------------
    exp_all = s1_yaml | s2_yaml
    c3 = {"n_on_disk": len(on_disk), "n_disk_tokens": len(disk_tokens), "n_csv_rows": len(csv_rows),
          "n_csv_tokens": len(csv_tokens), "n_expected": len(exp_all),
          "n_expected_stage_one": len(s1_yaml), "n_expected_stage_two": len(s2_yaml),
          "disk_eq_csv": disk_tokens == csv_tokens, "disk_eq_yaml": disk_tokens == exp_all,
          "missing_vs_yaml": sorted(exp_all - disk_tokens), "unused_vs_yaml": sorted(disk_tokens - exp_all),
          "n_log_groups_in_scene_filter": n_logs}
    c3["pass"] = bool(len(on_disk) > 0 and c3["disk_eq_csv"] and c3["disk_eq_yaml"]
                      and len(on_disk) == len(csv_rows) == len(exp_all))
    res["checks"]["C3_counts"] = c3
    if not c3["pass"]:
        res["fail"].append("C3_counts")

    # --- C2 scene types (load every entry) ------------------------------------------------
    with cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        entries = list(ex.map(load_entry, on_disk))
    wrong = [e["token"] for e in entries
             if (e["token"] in s1_yaml and e["scene_type"] != "SceneFrameType.ORIGINAL")
             or (e["token"] in s2_yaml and e["scene_type"] != "SceneFrameType.SYNTHETIC")]
    types = {}
    for e in entries:
        types[e["scene_type"]] = types.get(e["scene_type"], 0) + 1
    hum_none_ok = all(e["human_traj_is_none"] == (e["token"] in s2_yaml) for e in entries)
    c2 = {"n_loaded": len(entries), "scene_type_counts": types, "n_mismatch_vs_yaml": len(wrong),
          "mismatch": wrong, "human_trajectory_none_iff_synthetic": hum_none_ok}
    c2["pass"] = bool(entries and not wrong and hum_none_ok and len(entries) == len(exp_all))
    res["checks"]["C2_scene_type"] = c2
    if not c2["pass"]:
        res["fail"].append("C2_scene_type")
    res["map_roots_recorded"] = sorted({e["map_root"] for e in entries})
    res["map_names"] = {m: sum(e["map_name"] == m for e in entries) for m in sorted({e["map_name"] for e in entries})}
    res["cache_bytes_total"] = sum(e["bytes"] for e in entries)
    res["entry_load_s_mean"] = round(sum(e["load_s"] for e in entries) / max(len(entries), 1), 3)

    # --- build cost (from the wrapper's hooks + manifest) ------------------------------------
    man = json.load(open(a.manifest, encoding="utf-8"))
    hooks = json.load(open(a.hooks, encoding="utf-8"))["cache_scenes"]
    s1_s = [h["s"] for h in hooks if not h["synthetic_by_len17"]]
    s2_s = [h["s"] for h in hooks if h["synthetic_by_len17"]]
    res["build"] = {"wrapper_status": man.get("status"), "wall_s": man["resources"]["wall_s"],
                    "peak_rss_mb": man["resources"]["peak_rss_mb"],
                    "min_system_available_mb": man["resources"]["min_system_available_mb"],
                    "n_hook_scenes": len(hooks), "n_hook_ok": sum(h["ok"] for h in hooks),
                    "stage_one_scene_s": {"n": len(s1_s), "sum": round(sum(s1_s), 2),
                                          "mean": round(sum(s1_s) / max(len(s1_s), 1), 3),
                                          "max": max(s1_s) if s1_s else None},
                    "stage_two_scene_s": {"n": len(s2_s), "sum": round(sum(s2_s), 2),
                                          "mean": round(sum(s2_s) / max(len(s2_s), 1), 3),
                                          "max": max(s2_s) if s2_s else None},
                    "first_scene_s_includes_map_load": hooks[0]["s"] if hooks else None,
                    "command": man.get("argv"), "overrides": man.get("overrides"),
                    "imported_from": man.get("imported_from"), "devkit_sha": man.get("navsim_devkit_sha")}
    if man.get("status") != "OK":
        res["fail"].append("wrapper_status")
    res["verify_wall_s"] = round(time.time() - t0, 1)
    res["pass"] = not res["fail"]
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pass", "fail")}, indent=None),
          "| C3", {k: c3[k] for k in ("n_on_disk", "n_csv_tokens", "n_expected", "disk_eq_yaml")},
          "| C2", {k: c2[k] for k in ("scene_type_counts", "n_mismatch_vs_yaml")},
          "| unpatched", res["unpatched_loader"])

    if a.write_marker:
        if not res["pass"]:
            print("⛔ NOT writing CACHE_DONE.json — checks failed:", res["fail"])
            return 1
        marker = {
            "marker": "CACHE_DONE", "schema": "e1-cache-done/1", "written_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "written_by": "EvalFlyWheel stream E1 (2026-09-19-navsim-warmup-reference-epdms)",
            "split": a.split, "cache_path": str(a.cache).replace("\\", "/"),
            "n_cached": {"stage_one_original": types.get("SceneFrameType.ORIGINAL", 0),
                         "stage_two_synthetic": types.get("SceneFrameType.SYNTHETIC", 0), "total": len(entries)},
            "n_expected_from_yaml": {"stage_one": len(s1_yaml), "stage_two": len(s2_yaml), "log_groups": n_logs},
            "token_sets_equal_yaml": c3["disk_eq_yaml"], "scene_types_match_yaml": c2["pass"],
            "wall_clock_s": res["build"]["wall_s"], "peak_rss_mb": res["build"]["peak_rss_mb"],
            "per_scene_s_mean": {"stage_one": res["build"]["stage_one_scene_s"]["mean"],
                                 "stage_two": res["build"]["stage_two_scene_s"]["mean"]},
            "devkit_sha": man.get("navsim_devkit_sha"), "nuplan_devkit_sha": man.get("nuplan_devkit_sha"),
            "harness_modifications": [
                "PRE-EXISTING navsim/common/dataclasses.py PosixPath unpickler (worktree blob 596cb7d)",
                "PRE-EXISTING venv fcntl.py flock shim (sha256 75184a48...)",
                "PRE-EXISTING nuplan-devkit setup.py packaging-only change (blob 3ad18c6)"],
            "exact_command": " ".join(man.get("argv", [])),
            "runtime": ("C:/Users/Admin/navsim-crun (verified mirror: venv reinstalled offline from uv cache, "
                        "devkit copies blob-verified, data sha256-verified) - see E1 raw/devkit_copy_verify.json"),
            "map_root_recorded_in_entries": res["map_roots_recorded"],
            "map_root_note": ("the IDM reactive traffic policy RE-OPENS the map at scoring time from this recorded "
                              "root (navsim_IDM_traffic_agents.py:88). It is a byte-identical copy of "
                              "C:/Users/Admin/navsim/data/maps. To use another root pass "
                              "traffic_agents_policy.reactive.map_root_override=<root>."),
            "reading_on_windows": ("⛔ the UNPATCHED devkit MetricCacheLoader cannot read this cache on Windows: "
                                   + json.dumps(res["unpatched_loader"]) + ". Apply the separator patch "
                                   "(E1 code/navsim_win.py --patch-loader, _load_metric_cache_paths_patched)."),
            "verified_by": "E1 code/verify_cache.py (C2 scene types + C3 counts, every entry loaded)",
        }
        tmp = a.cache / "CACHE_DONE.json.tmp"
        tmp.write_text(json.dumps(marker, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, a.cache / "CACHE_DONE.json")
        (a.out.parent / "CACHE_DONE.json").write_text(json.dumps(marker, indent=1, ensure_ascii=False), encoding="utf-8")
        print("CACHE_DONE.json written:", a.cache / "CACHE_DONE.json")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
