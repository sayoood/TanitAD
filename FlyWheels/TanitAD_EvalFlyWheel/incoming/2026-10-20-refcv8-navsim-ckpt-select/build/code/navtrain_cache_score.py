#!/usr/bin/env python3
"""navtrain_cache_score.py -- the METRIC-CACHE and SCORER step of the D-R8-CKPT-RULE navtrain selection (driver: any python >= 3.9; the
devkit steps run in the NAVSIM venv exactly as W3's do).

PORT of the navtest path, by REUSE and not by copy: W3's ``run_v1.py`` (``cmd_cache`` / ``cmd_score``, count guards C1/C2/C3/C9) is loaded
BY PATH and called UNCHANGED, the way the Thor backend's ``score_navtest_thor.py`` does it.  What this driver changes is only what names
the SPLIT and the PATHS, each one a module global of W3's that it re-points and records:

* ``split_overrides``: ``train_test_split=navtrain`` (``data_split: trainval``) + the sample's log_names and tokens.  W3's version
  hard-codes ``navtest``; a navtest scorer pointed at navtrain tokens would score NOTHING and say so late.
* the roots: ``OPENSCENE_DATA_ROOT`` (must hold ``navsim_logs/trainval/<log>.pkl``), ``NUPLAN_MAPS_ROOT``, ``NAVSIM_EXP_ROOT``.
* ``navtest_tokens`` / ``_tokens_per_log`` (W3 per-log cache mode) now answer for the SAMPLE.

Then, per scored arm, a ``<label>.binding.json`` is written ONLY after the scorer's own counts say PASS and an independent re-read of the
CSV finds exactly the sample's token set.  The binding carries the sample's sorted-token sha256, the seam's sha256, the CSV's sha256 and
``split: navtrain`` -- the four things ``select_ckpt_r8.load_arm`` refuses without.  A ``--split navtest`` control writes
``control: true, split: navtest``, which the decision script REJECTS (tested).

ASSETS (MEASURED from source, 2026-10-20; ``tests/test_navtrain_cache_score.py`` pins each one):

* metric cache build AND scoring: ``NUPLAN_MAPS_ROOT`` with ``nuplan-maps-v1.0.json`` and ALL FOUR cities' ``map.gpkg``.  The nuPlan
  ``GPKGMapsDB`` constructor loads the ``lane_connectors`` layer of every one of ``MAP_LOCATIONS`` (``gpkg_mapsdb.py:97-101``), so a map
  directory holding only the sample's cities -- or Thor's Las-Vegas-only ``navsim_maps`` -- cannot build a map API at all.
  Sizes on the dev box: sg-one-north 317,575,168 B; us-ma-boston 254,087,168; us-nv-las-vegas-strip 523,767,808; us-pa-pittsburgh-hazelwood
  186,761,216 (plus the json).  The v1.1 devkit reads maps as ``nuplan-maps-v1.0`` (``caching.py``: ``map_version="nuplan-maps-v1.0"``).
* metadata: ``<OPENSCENE_DATA_ROOT>/navsim_logs/trainval/<log>.pkl`` for every log of the sample.  Caching reads it through a SceneLoader
  with ``sensor_blobs_path=None``; scoring through one with ``sensor_blobs_path=<root>/sensor_blobs/trainval`` and the agent's own sensor
  config -- the seam agent reads no sensors, so that directory is never opened and need not exist.
* NO jpg, lidar or sensor blob is read by either step.

COMMAND-LINE LIMIT.  The devkit takes the sample as a Hydra override ``...scene_filter.tokens=[...]`` (16 hex + 3 chars per token).  Windows
``CreateProcess`` caps the whole line at 32,767 characters, i.e. ~1,500 tokens; Linux caps ONE argument at 131,072 bytes, i.e. ~6,900.
``check_cmdline`` refuses BEFORE launching.  On Windows the cache build is chunked by whole logs (``--chunk-tokens``) and its metadata CSV is
re-emitted by the DEVKIT'S OWN ``save_cache_metadata`` (W3's ``write_cache_metadata.py`` pattern); scoring is not chunked -- it belongs on Thor.

    python navtrain_cache_score.py preflight --profile thor --sample NAVTRAIN_SAMPLE_TOKENS.json
    python navtrain_cache_score.py cache     --profile thor --sample S --out OUT --label r8sel_cache
    python navtrain_cache_score.py score     --profile thor --sample S --out OUT --label r8sel_ema40000_s0 --seam seam_R8_RT.npz
    python navtrain_cache_score.py score     --profile thor --sample S --out OUT --label r8sel_ref_STOP --official STOP
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types

MUTATION = os.environ.get("R8CS_MUTATION", "")
MUTATIONS = ("split_navtest", "tokens_not_restricted", "log_names_not_restricted", "no_cmdline_guard", "maps_sample_cities_only",
             "pickles_not_checked", "binding_without_sample_sha", "binding_on_failed_counts", "csv_token_set_unchecked",
             "rescore_overwrites", "label_prefix_unchecked", "chunk_splits_a_log", "cache_verified_by_count_only",
             "control_binding_says_navtrain")

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_PREFIX = "r8sel_"
MIN_TOKENS = 1000
MAP_CITIES = ("sg-one-north", "us-ma-boston", "us-nv-las-vegas-strip", "us-pa-pittsburgh-hazelwood")
MAP_VERSION = "nuplan-maps-v1.0"
SPLIT_DIR = {"navtrain": "trainval", "navtest": "test"}
WIN_CMDLINE_LIMIT = 30000          # CreateProcess: 32,767 characters for the WHOLE line; the wrapper's own words take the rest
ARG_LIMIT = 120000                 # Linux MAX_ARG_STRLEN is 131,072 bytes for ONE argument

W3_REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code"
PROFILES = {
    "thor": {"nv_py": "/home/nvidia/venvs/navsim-cpu/bin/python", "v11_tree": "/home/nvidia/navsim-thor/src/navsim-v11",
             "exp": "/dev/shm/navsim/exp/r8sel_navtrain", "data": "/dev/shm/navsim/data/openscene", "maps": "/dev/shm/navsim/data/maps",
             "wrap": "/home/nvidia/navsim-thor/code/thor/navsim_v1_thor.py", "w3_code": "/home/nvidia/navsim-thor/code/w3",
             "stubs": "/home/nvidia/navsim-thor/code/stubs", "cuda_visible": "", "patch_loader": True},
    "devbox": {"nv_py": "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe",
               "v11_tree": "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896",
               "exp": "D:/Archive/devbox-C/navsim/exp/r8sel_navtrain", "data": "D:/Archive/devbox-C/navsim/data/openscene",
               "maps": "D:/Archive/devbox-C/navsim/data/maps", "wrap": f"C:/lgs/Lfb47a58376/{W3_REL}/navsim_v1_win.py",
               "w3_code": f"C:/lgs/Lfb47a58376/{W3_REL}", "stubs": "", "cuda_visible": "-1", "patch_loader": True},
}


# ------------------------------------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------------------------------------
def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_tokens(tokens) -> str:
    return hashlib.sha256("\n".join(sorted(tokens)).encode()).hexdigest()


def _norm(p: str) -> str:
    return p.replace("\\", "/")


# ------------------------------------------------------------------------------------------------------------------------------
# the sample
# ------------------------------------------------------------------------------------------------------------------------------
def load_sample(path: str, split: str = "navtrain") -> dict:
    """Returns the sample dict; raises SystemExit naming the first defect.  A navtrain run needs a FIXED sample of >= 1,000 distinct tokens
    whose sorted-token sha256 re-derives and whose ``token_log`` covers every token.  A ``navtest`` control is not held to FIXED."""
    s = json.load(open(path, encoding="utf-8"))
    toks = list(s.get("tokens") or [])
    tl = s.get("token_log") or {}
    if any(t not in tl for t in toks) or not toks:
        sys.exit("[score] the sample carries no complete token_log (token -> log_name)")
    if split == "navtrain":
        if s.get("status") != "FIXED":
            sys.exit(f"[score] sample status {s.get('status')!r} is not FIXED -- no cache and no score exists for an unfixed sample")
        if len(set(toks)) != len(toks) or len(toks) != s.get("count") or len(toks) < MIN_TOKENS:
            sys.exit(f"[score] sample count {len(toks)} (distinct {len(set(toks))}, recorded {s.get('count')}) is not a fixed >= {MIN_TOKENS} sample")
        if sha256_tokens(toks) != s.get("sha256_sorted_tokens"):
            sys.exit("[score] the sample's sorted-token sha256 does not re-derive")
    s["_sha256_sorted_tokens"] = sha256_tokens(toks)
    return s


def sample_logs(sample: dict) -> list:
    return sorted({sample["token_log"][t] for t in sample["tokens"]})


# ------------------------------------------------------------------------------------------------------------------------------
# the devkit overrides
# ------------------------------------------------------------------------------------------------------------------------------
def navtrain_split_overrides(tok_file, split: str = "navtrain"):
    """Replacement for W3's ``split_overrides``: ``(overrides, tokens)``.  Same shape as W3's, so ``cmd_cache`` / ``cmd_score`` use it unchanged."""
    name = "navtest" if MUTATION == "split_navtest" else split
    ov = [f"train_test_split={name}"]
    if not tok_file:
        raise SystemExit("[score] a navtrain run needs the sample file: without a token restriction the devkit would take ALL 103,288 tokens")
    doc = json.load(open(tok_file, encoding="utf-8"))
    toks = list(doc["tokens"])
    logs = sorted({doc["token_log"][t] for t in toks})
    if MUTATION != "log_names_not_restricted":
        ov.append("train_test_split.scene_filter.log_names=[" + ",".join(f"'{x}'" for x in logs) + "]")
    if MUTATION != "tokens_not_restricted":
        ov.append("train_test_split.scene_filter.tokens=[" + ",".join(f"'{x}'" for x in toks) + "]")
    return ov, toks


def check_cmdline(overrides, os_name: str = None) -> None:
    """Refuse a devkit launch whose command line the OS would reject (or truncate).  ``os_name``: 'nt' or 'posix' (tests pass it)."""
    if MUTATION == "no_cmdline_guard":
        return
    os_name = os_name or os.name
    longest = max((len(o) for o in overrides), default=0)
    total = sum(len(o) + 3 for o in overrides)
    if os_name == "nt" and total > WIN_CMDLINE_LIMIT:
        sys.exit(f"[score] the Hydra overrides are {total} characters; Windows CreateProcess allows 32,767 for the whole line. "
                 f"Chunk the cache build (--chunk-tokens) or run on Thor (Linux); scoring belongs on Thor")
    if os_name != "nt" and longest > ARG_LIMIT:
        sys.exit(f"[score] one Hydra override is {longest} characters; Linux caps a single argument at 131,072 bytes")


def chunk_logs(sample: dict, max_tokens: int) -> list:
    """Whole logs, greedily in sorted log order, each chunk at most ``max_tokens`` tokens (a log larger than that gets a chunk to itself).
    Disjoint and exhaustive by construction.  ``[{logs, tokens}]``."""
    by_log: dict = {}
    for t in sample["tokens"]:
        by_log.setdefault(sample["token_log"][t], []).append(t)
    chunks, cur_logs, cur = [], [], []
    for ln in sorted(by_log):
        mine = sorted(by_log[ln])
        if MUTATION == "chunk_splits_a_log":
            for t in mine:
                if len(cur) >= max_tokens:
                    chunks.append({"logs": sorted(set(sample["token_log"][x] for x in cur)), "tokens": cur})
                    cur_logs, cur = [], []
                cur.append(t)
            continue
        if cur and len(cur) + len(mine) > max_tokens:
            chunks.append({"logs": cur_logs, "tokens": cur})
            cur_logs, cur = [], []
        cur_logs.append(ln)
        cur += mine
    if cur:
        chunks.append({"logs": sorted(set(sample["token_log"][x] for x in cur)) if MUTATION == "chunk_splits_a_log" else cur_logs, "tokens": cur})
    return chunks


# ------------------------------------------------------------------------------------------------------------------------------
# assets
# ------------------------------------------------------------------------------------------------------------------------------
def check_maps(maps_root: str) -> list:
    """Problems with a nuPlan maps root for the v1.1 devkit.  ALL FOUR cities are required (GPKGMapsDB loads every one at construction)."""
    probs = []
    jp = os.path.join(maps_root, f"{MAP_VERSION}.json")
    try:
        meta = json.load(open(jp, encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"MAPS_JSON_UNREADABLE:{jp}:{type(e).__name__}"]
    for city in MAP_CITIES:
        if city not in meta:
            probs.append(f"MAPS_JSON_LACKS:{city}")
            continue
        g = os.path.join(maps_root, city, str(meta[city].get("version")), "map.gpkg")
        if not (os.path.isfile(g) and os.path.getsize(g) > 0):
            probs.append(f"MAP_GPKG_MISSING:{city}:{g}")
    return probs


def check_maps_for_sample(maps_root: str, sample_cities=None) -> list:
    """The entry point preflight uses.  (``sample_cities`` exists only for the deliberate-regression arm: the defect is believing the
    sample's own cities are enough.)"""
    if MUTATION == "maps_sample_cities_only" and sample_cities:
        meta = json.load(open(os.path.join(maps_root, f"{MAP_VERSION}.json"), encoding="utf-8"))
        return [f"MAP_GPKG_MISSING:{c}" for c in sample_cities
                if not os.path.isfile(os.path.join(maps_root, c, str(meta[c]["version"]), "map.gpkg"))]
    return check_maps(maps_root)


def check_logs(data_root: str, sample: dict, split: str) -> list:
    if MUTATION == "pickles_not_checked":
        return []
    d = os.path.join(data_root, "navsim_logs", SPLIT_DIR[split])
    return [f"LOG_PICKLE_MISSING:{ln}" for ln in sample_logs(sample)
            if not (os.path.isfile(os.path.join(d, f"{ln}.pkl")) and os.path.getsize(os.path.join(d, f"{ln}.pkl")) > 0)]


def preflight(prof: dict, sample: dict, split: str, cities=None) -> dict:
    rep = {"profile_paths": {k: prof[k] for k in ("nv_py", "v11_tree", "exp", "data", "maps", "wrap", "w3_code")}, "problems": []}
    for k in ("nv_py", "wrap", "w3_code", "v11_tree"):
        if not os.path.exists(prof[k]):
            rep["problems"].append(f"PATH_MISSING:{k}:{prof[k]}")
    rep["problems"] += check_maps_for_sample(prof["maps"], cities)
    rep["problems"] += check_logs(prof["data"], sample, split)
    rep["n_tokens"], rep["n_logs"] = len(sample["tokens"]), len(sample_logs(sample))
    rep["sample_sha256_sorted_tokens"] = sample["_sha256_sorted_tokens"]
    rep["ok"] = not rep["problems"]
    return rep


# ------------------------------------------------------------------------------------------------------------------------------
# the metric cache on disk
# ------------------------------------------------------------------------------------------------------------------------------
def scan_pickles(cache_dir: str) -> list:
    return sorted(glob.glob(os.path.join(cache_dir, "*", "*", "*", "metric_cache.pkl")))


def _row_token(row: str) -> str:
    return re.split(r"[\\/]", row.strip())[-2]


def _row_rel(row: str) -> str:
    return "/".join(re.split(r"[\\/]", row.strip())[-4:])


def verify_cache(cache_dir: str, tokens) -> dict:
    """Independent check of a built cache against the sample: exactly ONE metadata CSV, its rows name exactly the sample's tokens (distinct),
    each row's file exists (re-rooted at ``cache_dir`` so a cache built on another OS reads the same), and the pickles on disk are the same set."""
    want = set(tokens)
    rep = {"cache": cache_dir, "n_expected": len(want), "problems": []}
    csvs = sorted(glob.glob(os.path.join(cache_dir, "metadata", "*.csv")))
    rep["metadata_csvs"] = [os.path.basename(c) for c in csvs]
    if len(csvs) != 1:
        rep["problems"].append(f"METADATA_CSV_COUNT_{len(csvs)}")
        rep["ok"] = False
        return rep
    rows = [r for r in open(csvs[0], encoding="utf-8").read().splitlines()[1:] if r.strip()]
    toks = [_row_token(r) for r in rows]
    pk = scan_pickles(cache_dir)
    pk_toks = [os.path.basename(os.path.dirname(p)) for p in pk]
    rep.update(n_rows=len(rows), n_distinct=len(set(toks)), n_pickles=len(pk), metadata_csv_sha256=sha256_file(csvs[0]),
               n_rows_file_exists=sum(1 for r in rows if os.path.isfile(os.path.join(cache_dir, *_row_rel(r).split("/")))))
    if MUTATION == "cache_verified_by_count_only":
        if len(rows) != len(want):
            rep["problems"].append("ROW_COUNT")
    else:
        if set(toks) != want:
            rep["problems"].append(f"ROW_TOKENS_DIFFER:missing={len(want - set(toks))}:unexpected={len(set(toks) - want)}")
        if len(set(toks)) != len(toks):
            rep["problems"].append("ROW_TOKENS_DUPLICATED")
        if set(pk_toks) != want:
            rep["problems"].append(f"PICKLE_TOKENS_DIFFER:missing={len(want - set(pk_toks))}:unexpected={len(set(pk_toks) - want)}")
        if rep["n_rows_file_exists"] != len(rows):
            rep["problems"].append("ROW_FILE_ABSENT")
    rep["ok"] = not rep["problems"]
    return rep


def finalize_metadata(cache_dir: str, tokens) -> dict:
    """(NAVSIM VENV.) Re-emit the metadata CSV with the DEVKIT'S OWN ``save_cache_metadata`` from the pickles on disk -- after a chunked build each
    chunk's run wrote only its own rows.  Refuses unless the pickles are exactly the sample (W3's ``write_cache_metadata.py`` rule)."""
    want = set(tokens)
    pk = scan_pickles(cache_dir)
    got = [os.path.basename(os.path.dirname(p)) for p in pk]
    rep = {"cache": cache_dir, "n_pickles": len(pk), "n_expected": len(want), "n_missing": len(want - set(got)),
           "n_unexpected": len(set(got) - want)}
    if len(pk) != len(want) or set(got) != want or len(set(got)) != len(got):
        rep["status"] = "REFUSED_PICKLES_ARE_NOT_THE_SAMPLE"
        return rep
    from pathlib import Path

    from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry, save_cache_metadata
    save_cache_metadata([CacheMetadataEntry(Path(p)) for p in pk], Path(cache_dir), 0)
    rep["status"] = "WRITTEN"
    return rep


# ------------------------------------------------------------------------------------------------------------------------------
# the scorer output
# ------------------------------------------------------------------------------------------------------------------------------
def read_csv_tokens(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    tok_rows = [r for r in rows if r.get("token") != "average"]
    valid = [r for r in tok_rows if r.get("valid") in ("True", "true", "1")]
    return [r["token"] for r in tok_rows], len(valid), sum(1 for r in rows if r.get("token") == "average")


def banked_decision(binding_path: str, seam_sha, official, sample_sha: str) -> str:
    """'RUN' (nothing banked under this label, or the operator forces), 'SKIP' (a binding for the SAME seam / agent / sample is banked), or
    'REFUSE' (the label is already bound to DIFFERENT input -- rescoring would silently replace one arm's result with another's)."""
    if not os.path.exists(binding_path) or os.environ.get("R8CS_FORCE_RESCORE") == "1" or MUTATION == "rescore_overwrites":
        return "RUN"
    old = json.load(open(binding_path, encoding="utf-8"))
    same = (old.get("seam_sha256") == seam_sha and old.get("official_agent") == official
            and old.get("sample_sha256_sorted_tokens") == sample_sha)
    return "SKIP" if same else "REFUSE"


def build_binding(*, label, split, sample, sample_path, seam, seam_sha, official, csv_path, counts_path, cache_dir, cache_ver, prof_name,
                  prov) -> dict:
    """The ``binding.json`` ``select_ckpt_r8.load_arm`` requires.  Raises ``ValueError`` (the caller refuses) on any inconsistency."""
    counts = json.load(open(counts_path, encoding="utf-8"))
    if counts.get("status") != "PASS" and MUTATION != "binding_on_failed_counts":
        raise ValueError(f"counts status {counts.get('status')!r}")
    toks, n_valid, n_avg = read_csv_tokens(csv_path)
    if MUTATION != "csv_token_set_unchecked":
        if set(toks) != set(sample["tokens"]) or len(toks) != len(set(toks)):
            raise ValueError(f"CSV token set differs from the sample: rows {len(toks)}, distinct {len(set(toks))}, "
                             f"missing {len(set(sample['tokens']) - set(toks))}, unexpected {len(set(toks) - set(sample['tokens']))}")
        if n_valid != len(sample["tokens"]) or n_avg != 1:
            raise ValueError(f"CSV valid rows {n_valid} / average rows {n_avg}")
    control = split != "navtrain"
    b = {"schema": "r8sel-binding-v1", "label": label,
         "split": "navtrain" if MUTATION == "control_binding_says_navtrain" else split, "control": control,
         "sample_path": _norm(os.path.abspath(sample_path)), "n_tokens": len(sample["tokens"]),
         "seam": _norm(os.path.abspath(seam)) if seam else None, "seam_sha256": seam_sha, "official_agent": official or None,
         "csv": os.path.basename(csv_path), "csv_sha256": sha256_file(csv_path),
         "counts": os.path.basename(counts_path), "counts_sha256": sha256_file(counts_path), "counts_status": counts.get("status"),
         "cache_dir": _norm(cache_dir), "cache_metadata_csv_sha256": (cache_ver or {}).get("metadata_csv_sha256"),
         "cache_n_rows": (cache_ver or {}).get("n_rows"), "backend_profile": prof_name, "provenance": prov}
    if MUTATION != "binding_without_sample_sha":
        b["sample_sha256_sorted_tokens"] = sample["_sha256_sorted_tokens"]
    return b


# ------------------------------------------------------------------------------------------------------------------------------
# W3 reuse
# ------------------------------------------------------------------------------------------------------------------------------
def resolve_profile(args) -> dict:
    prof = dict(PROFILES[args.profile])
    for k in ("nv_py", "v11_tree", "exp", "data", "maps", "wrap", "w3_code", "stubs"):
        v = getattr(args, k, None)
        if v:
            prof[k] = v
    return prof


def load_w3(w3_code: str):
    spec = importlib.util.spec_from_file_location("w3_run_v1_navtrain", os.path.join(w3_code, "run_v1.py"))
    w3 = importlib.util.module_from_spec(spec)
    sys.modules["w3_run_v1_navtrain"] = w3
    spec.loader.exec_module(w3)
    return w3


def configure_w3(w3, prof: dict, sample: dict, sample_path: str, raw_dir: str, split: str):
    """Re-point W3's module globals (each recorded in the binding's provenance).  Returns the provenance dict."""
    glob_ = {"NV_PY": prof["nv_py"], "V11_TREE": prof["v11_tree"], "EXP": prof["exp"], "DATA": prof["data"], "WRAP": prof["wrap"],
             "HERE": prof["w3_code"]}
    for k, v in glob_.items():
        setattr(w3, k, v)
    os.makedirs(prof["exp"], exist_ok=True)
    w3.ENV = dict(w3.ENV, NUPLAN_MAPS_ROOT=prof["maps"], NAVSIM_EXP_ROOT=prof["exp"], NAVSIM_DEVKIT_ROOT=prof["v11_tree"],
                  OPENSCENE_DATA_ROOT=prof["data"], CUDA_VISIBLE_DEVICES=prof["cuda_visible"])
    stubs = prof.get("stubs")
    _orig_env_for = w3.env_for

    def env_for(hashseed: str = "1") -> dict:
        e = _orig_env_for(hashseed)
        if stubs:
            e["PYTHONPATH"] = e["PYTHONPATH"] + os.pathsep + stubs
        return e

    w3.env_for = env_for
    w3.split_overrides = lambda tok_file: navtrain_split_overrides(tok_file, split)
    by_log: dict = {}
    for t in sample["tokens"]:
        by_log[sample["token_log"][t]] = by_log.get(sample["token_log"][t], 0) + 1
    w3.navtest_tokens = lambda: (sorted(by_log), list(sample["tokens"]))
    w3._tokens_per_log = lambda: dict(by_log)
    _orig_run = w3.run_wrapped

    def run_wrapped(script, label, overrides, out_dir, **kw):
        check_cmdline(overrides)
        return _orig_run(script, label, overrides, out_dir, **kw)

    w3.run_wrapped = run_wrapped
    os.makedirs(raw_dir, exist_ok=True)
    w3.RAW = os.path.abspath(raw_dir)
    w3.PKG = os.path.abspath(raw_dir)
    run_v1 = os.path.join(prof["w3_code"], "run_v1.py")
    return {"w3_globals": glob_, "w3_run_v1_sha256": sha256_file(run_v1),
            "w3_agents_v1_sha256": sha256_file(os.path.join(prof["w3_code"], "w3_agents_v1.py")),
            "wrapper_sha256": sha256_file(prof["wrap"]), "driver": _norm(os.path.abspath(__file__)),
            "driver_sha256": sha256_file(__file__), "split": split, "maps_root": prof["maps"], "devkit_tree": prof["v11_tree"],
            "sample_path": _norm(os.path.abspath(sample_path))}


# ------------------------------------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------------------------------------
def cmd_preflight(a) -> int:
    prof = resolve_profile(a)
    sample = load_sample(a.sample, a.split)
    rep = preflight(prof, sample, a.split)
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), indent=1)
    print(f"ZZPREFLIGHT_{'OK' if rep['ok'] else 'REFUSED'} problems={len(rep['problems'])} tokens={rep['n_tokens']} logs={rep['n_logs']}")
    for p in rep["problems"][:40]:
        print("  " + p)
    return 0 if rep["ok"] else 1


def cmd_cache(a) -> int:
    prof = resolve_profile(a)
    sample = load_sample(a.sample, a.split)
    if not a.label.startswith(LABEL_PREFIX) and MUTATION != "label_prefix_unchecked":
        sys.exit(f"[score] labels must start with {LABEL_PREFIX!r} so the devkit scratch never collides with another suite's")
    pf = preflight(prof, sample, a.split)
    if not pf["ok"]:
        print(f"ZZCACHE_REFUSED_preflight problems={len(pf['problems'])}")
        for p in pf["problems"][:20]:
            print("  " + p)
        return 2
    cache_dir = f"{prof['exp']}/metric_cache_{a.cache_name}"
    raw = os.path.join(a.out, "cache")
    w3 = load_w3(prof["w3_code"])
    prov = configure_w3(w3, prof, sample, a.sample, raw, a.split)
    t0 = time.time()
    chunks = chunk_logs(sample, a.chunk_tokens) if a.chunk_tokens else [{"logs": sample_logs(sample), "tokens": list(sample["tokens"])}]
    rc = 0
    if len(chunks) == 1:
        ns = types.SimpleNamespace(label=a.label, tokens=a.sample, cache_name=a.cache_name, worker="sequential", per_log=False)
        rc = w3.cmd_cache(ns)
    else:
        chunk_dir = os.path.join(raw, f"{a.label}_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        reps = []
        for i, ch in enumerate(chunks):
            tf = os.path.join(chunk_dir, f"chunk{i:03d}_tokens.json")
            json.dump({"tokens": ch["tokens"], "token_log": {t: sample["token_log"][t] for t in ch["tokens"]}}, open(tf, "w"), indent=0)
            ov, _ = navtrain_split_overrides(tf, a.split)
            ov += [f"cache.cache_path={cache_dir}", "worker=sequential"]
            rc_i, lp, wall = w3.run_wrapped("metric_caching", f"{a.label}_c{i:03d}", ov, chunk_dir, patch=False)
            reps.append({"chunk": i, "n_tokens": len(ch["tokens"]), "n_logs": len(ch["logs"]), "rc": rc_i, "wall_s": round(wall, 1)})
            print(json.dumps(reps[-1]), flush=True)
            if rc_i != 0:
                rc = rc_i
        json.dump(reps, open(os.path.join(chunk_dir, "chunks.json"), "w"), indent=1)
        if rc == 0:
            fin = subprocess.run([prof["nv_py"], os.path.abspath(__file__), "finalize-metadata", "--cache-dir", cache_dir, "--sample", a.sample,
                                  "--split", a.split], capture_output=True, text=True)
            print(fin.stdout[-1500:])
            rc = fin.returncode
    ver = verify_cache(cache_dir, sample["tokens"])
    ver["wall_s"] = round(time.time() - t0, 1)
    ver["provenance"] = prov
    ver["sample_sha256_sorted_tokens"] = sample["_sha256_sorted_tokens"]
    os.makedirs(raw, exist_ok=True)
    json.dump(ver, open(os.path.join(raw, f"{a.label}.verify.json"), "w", encoding="utf-8"), indent=1)
    ok = rc == 0 and ver["ok"]
    print(f"ZZCACHE_{'DONE' if ok else 'FAILED'} rows={ver.get('n_rows')} expected={ver['n_expected']} problems={len(ver['problems'])} rc={rc}")
    return 0 if ok else 1


def cmd_finalize(a) -> int:
    sample = load_sample(a.sample, a.split)
    rep = finalize_metadata(a.cache_dir, sample["tokens"])
    print(json.dumps(rep))
    return 0 if rep["status"] == "WRITTEN" else 1


def cmd_score(a) -> int:
    prof = resolve_profile(a)
    sample = load_sample(a.sample, a.split)
    if not a.label.startswith(LABEL_PREFIX) and MUTATION != "label_prefix_unchecked":
        sys.exit(f"[score] labels must start with {LABEL_PREFIX!r} so the devkit scratch never collides with another suite's")
    if bool(a.seam) == bool(a.official):
        sys.exit("[score] exactly one of --seam / --official")
    pf = preflight(prof, sample, a.split)
    cache_dir = f"{prof['exp']}/metric_cache_{a.cache_name}"
    ver = verify_cache(cache_dir, sample["tokens"])
    if not pf["ok"] or not ver["ok"]:
        print(f"ZZSCORE_REFUSED_preconditions preflight={len(pf['problems'])} cache={len(ver['problems'])}")
        for p in (pf["problems"] + ver["problems"])[:20]:
            print("  " + p)
        return 2
    seam_sha = sha256_file(a.seam) if a.seam else None
    odir = os.path.join(a.out, "scores", a.label)
    bpath = os.path.join(odir, f"{a.label}.binding.json")
    dec = banked_decision(bpath, seam_sha, a.official or None, sample["_sha256_sorted_tokens"])
    if dec != "RUN":
        print(f"ZZSCORE_{'SKIPPED_banked' if dec == 'SKIP' else 'REFUSED_label_bound_to_a_different_input'} label={a.label}")
        return 0 if dec == "SKIP" else 3
    w3 = load_w3(prof["w3_code"])
    prov = configure_w3(w3, prof, sample, a.sample, os.path.join(a.out, "scores"), a.split)
    arm = a.official if a.official else f"SEAM:{os.path.abspath(a.seam)}"
    ns = types.SimpleNamespace(label=a.label, arm=arm, tokens=a.sample, cache_name=a.cache_name, worker="sequential", hashseed="1",
                               record_poses=True, patch_loader=prof["patch_loader"])
    os.makedirs(odir, exist_ok=True)
    rc = w3.cmd_score(ns)
    csv_path, counts_path = os.path.join(odir, f"{a.label}.csv"), os.path.join(odir, f"{a.label}.counts.json")
    try:
        b = build_binding(label=a.label, split=a.split, sample=sample, sample_path=a.sample, seam=a.seam, seam_sha=seam_sha,
                          official=a.official, csv_path=csv_path, counts_path=counts_path, cache_dir=cache_dir, cache_ver=ver,
                          prof_name=a.profile, prov=prov)
    except (OSError, ValueError) as e:
        json.dump({"label": a.label, "rc": rc, "refused": f"{type(e).__name__}: {e}"}, open(os.path.join(odir, f"{a.label}.FAILED.json"), "w"), indent=1)
        print(f"ZZSCORE_FAILED label={a.label} rc={rc} why={type(e).__name__}")
        return 1
    json.dump(b, open(bpath, "w", encoding="utf-8"), indent=1)
    print(f"ZZSCORE_DONE label={a.label} tokens={b['n_tokens']} csv_sha256={b['csv_sha256'][:16]} split={b['split']}")
    return 0 if rc == 0 else 1


def cmd_compose_check(a) -> int:
    """(NAVSIM VENV.) Compose the devkit's own scoring and caching configs with this driver's overrides and print what they resolve to."""
    prof = resolve_profile(a)
    sample = load_sample(a.sample, a.split)
    os.environ.update(OPENSCENE_DATA_ROOT=prof["data"], NAVSIM_EXP_ROOT=prof["exp"], NUPLAN_MAPS_ROOT=prof["maps"],
                      NAVSIM_DEVKIT_ROOT=prof["v11_tree"], NUPLAN_MAP_VERSION=MAP_VERSION)
    sys.path.insert(0, prof["v11_tree"])
    from hydra import compose, initialize_config_dir
    ov, toks = navtrain_split_overrides(a.sample, a.split)
    out = {}
    for sub, name, extra in (("pdm_scoring", "default_run_pdm_score", ["metric_cache_path=/x/mc"]),
                             ("metric_caching", "default_metric_caching", ["cache.cache_path=/x/mc"])):
        with initialize_config_dir(config_dir=f"{prof['v11_tree']}/navsim/planning/script/config/{sub}", version_base=None):
            cfg = compose(config_name=name, overrides=ov + extra)
            sf = cfg.train_test_split.scene_filter
            out[sub] = {"data_split": cfg.train_test_split.data_split, "navsim_log_path": _norm(str(cfg.navsim_log_path)),
                        "n_tokens": len(sf.tokens), "tokens_sha256_sorted": sha256_tokens(list(sf.tokens)),
                        "logs": list(sf.log_names), "num_history_frames": sf.num_history_frames,
                        "num_future_frames": sf.num_future_frames, "has_route": bool(sf.has_route),
                        "frame_interval": sf.frame_interval, "max_scenes": sf.max_scenes}
            if sub == "pdm_scoring":
                out[sub]["sensor_blobs_path"] = _norm(str(cfg.sensor_blobs_path))
    print("ZZCOMPOSE " + json.dumps(out))
    return 0


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, needs_out=True):
        p.add_argument("--profile", choices=sorted(PROFILES), default="thor")
        p.add_argument("--sample", required=True)
        p.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain", help="navtest = model-free CONTROL only")
        for k in ("nv_py", "v11_tree", "exp", "data", "maps", "wrap", "w3_code", "stubs"):
            p.add_argument("--" + k.replace("_", "-"), dest=k, default="")
        if needs_out:
            p.add_argument("--out", required=True)
        p.add_argument("--cache-name", default="navtrain_sample")

    p = sub.add_parser("preflight")
    common(p, False)
    p.add_argument("--json", default="")
    p = sub.add_parser("cache")
    common(p)
    p.add_argument("--label", required=True)
    p.add_argument("--chunk-tokens", type=int, default=0, help="Windows: build in chunks of whole logs of at most this many tokens")
    p = sub.add_parser("score")
    common(p)
    p.add_argument("--label", required=True)
    p.add_argument("--seam", default="")
    p.add_argument("--official", default="", choices=("", "STOP", "CV"))
    p = sub.add_parser("finalize-metadata")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--split", choices=("navtrain", "navtest"), default="navtrain")
    p = sub.add_parser("compose-check")
    common(p, False)
    return ap


def main(argv=None) -> int:
    if MUTATION:
        sys.exit("[score] R8CS_MUTATION is set -- a mutation-test environment never runs the devkit or writes a binding")
    a = build_parser().parse_args(argv)
    return {"preflight": cmd_preflight, "cache": cmd_cache, "score": cmd_score, "finalize-metadata": cmd_finalize,
            "compose-check": cmd_compose_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
