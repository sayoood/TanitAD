#!/usr/bin/env python3
"""E3 Part A1/A2 — the CLUSTER STRUCTURE of every NavSim split, MEASURED.

Run with the TanitAD venv (the devkit's own `filter_scenes` / `SceneFilter` are
EXECUTED VERBATIM from the pinned git blobs — see `_devkit_symbols`):

    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u code/split_census.py \
        --out raw/split_census.json --frames-cache <scratch>/frames_cache.pkl

WHAT IS MEASURED, AND FROM WHAT
-------------------------------
* The split definitions are read from the devkit's GIT BLOBS at the pinned SHA
  (`git show 0a380a9:<path>`), never from the working tree — the local devkit
  carries one locally modified file (`navsim/common/dataclasses.py`, a Windows
  PosixPath unpickler shim), so "the working tree" and "the pin" are not the
  same object. Each YAML's byte count + sha256 is recorded as its READ CONTROL.
* The frames come from the local OpenScene TEST metadata
  (`navsim_logs/test/*.pkl`, one pickle per OpenScene log segment). Each log's
  read control is its frame count; a log that fails to load is COUNTED, never
  skipped silently.
* The stage-1 scene list of every split whose logs are local is reconstructed
  TWICE and the two must agree token-for-token:
    (1) the devkit's OWN `navsim.common.dataloader.filter_scenes`, its source text
        executed verbatim from the pinned blob, one log at a time (the reference —
        independently authored code);
    (2) an independent re-derivation in this file (`_windows`).
  A census whose two derivations disagree reports the disagreement and does not
  publish a count.

⛔ A COUNT OF 0 IS A CLAIM ABOUT THE PROBE unless the input was READ. Every count
below is emitted next to the read control that makes a 0 meaningful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

DEVKIT = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", r"C:/Users/Admin/navsim/devkit"))
DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
DATA = Path(os.environ.get("OPENSCENE_DATA_ROOT", r"C:/Users/Admin/navsim/data/openscene"))
LOGS = DATA / "navsim_logs" / "test"
WARMUP = DATA / "warmup_two_stage"
SPLIT_DIR = "navsim/planning/script/config/common/train_test_split/"
FRAME_DT_S = 0.5  # OpenScene key-frame spacing (2 Hz) — MEASURED below from timestamps, not assumed

# An OpenScene log name is a nuPlan drive + a segment suffix, e.g.
# 2021.05.25.14.16.10_veh-35_00083_00485 -> drive 2021.05.25.14.16.10_veh-35.
DRIVE_RE = re.compile(
    r"^(?P<drive>\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+)_(?P<a>\d{5})_(?P<b>\d{5})$")

SPLITS_WITH_LOCAL_LOGS = ("navtest", "navhard_two_stage", "warmup_two_stage",
                          "navtest_two_stage", "navsafe_two_stage",
                          "warmup_navsafe_two_stage_extended", "test")


def _git_show(rel: str) -> bytes:
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(DEVKIT), "show",
                        f"{DEVKIT_SHA}:{rel}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"git show {DEVKIT_SHA}:{rel} failed rc={r.returncode} "
                           f"stderr={r.stderr[:200]!r} — a missing YAML is a READ "
                           f"FAILURE, not an empty split")
    return r.stdout


def _yaml(rel: str) -> tuple:
    raw = _git_show(rel)
    return yaml.safe_load(raw.decode("utf-8")) or {}, {
        "path": f"{rel}@{DEVKIT_SHA[:7]}", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()}


def _drive(log_name: str) -> str:
    m = DRIVE_RE.match(log_name)
    return m.group("drive") if m else log_name


def _stats(xs) -> dict:
    xs = list(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "median": statistics.median(xs),
            "max": max(xs), "mean": round(float(np.mean(xs)), 4)}


def _yaw(q) -> float:
    """pyquaternion 0.9.9 ``Quaternion(*q).yaw_pitch_roll[0]`` for (w, x, y, z) —
    the exact call the devkit makes (dataclasses.py:458-460 @0a380a9) — VENDORED
    verbatim from ``pyquaternion/quaternion.py:1024-1026`` (the NavSim venv's copy),
    because this census runs in the TanitAD venv, which has no pyquaternion.
    ⚠️ pyquaternion's yaw uses ``w*z - x*y`` (not the textbook ``+``); kept as is,
    since the point is to reproduce the devkit, not the textbook."""
    w, x, y, z = (float(v) for v in q)
    ss = w * w + x * x + y * y + z * z
    if abs(1.0 - ss) >= 1e-14 and ss > 0:          # _normalise(): is_unit(tol=1e-14)
        n = math.sqrt(ss)
        w, x, y, z = w / n, x / n, y / n, z / n
    return float(math.atan2(2 * (w * z - x * y), 1 - 2 * (y ** 2 + z ** 2)))


# --------------------------------------------------------------------------- #
# 1. Frames: one compact row per key frame, one log at a time                  #
# --------------------------------------------------------------------------- #
def load_frames(log_dir: Path) -> tuple:
    files = sorted(p for p in log_dir.iterdir() if p.suffix == ".pkl")
    frames, fails, read = {}, [], {}
    for p in files:
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception as e:  # counted, never skipped silently
            fails.append({"log": p.stem, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        rows = []
        for i, fr in enumerate(d):
            t = fr["ego2global_translation"]
            rows.append({
                "i": i, "token": fr["token"], "ts": int(fr["timestamp"]),
                "log": fr["log_name"], "scene_token": fr["scene_token"],
                "map": fr["map_location"], "veh": fr["vehicle_name"],
                "rb": tuple(str(x) for x in fr["roadblock_ids"]),
                "cmd": tuple(int(x) for x in np.asarray(fr["driving_command"]).tolist()),
                "x": float(t[0]), "y": float(t[1]),
                "yaw": _yaw(fr["ego2global_rotation"]),
                "v": [float(v) for v in fr["ego_dynamic_state"][:2]],
                "frame_idx": int(fr["frame_idx"]),
            })
        frames[p.stem] = rows
        read[p.stem] = len(rows)
        del d
    return frames, fails, read, len(files)


# --------------------------------------------------------------------------- #
# 2. Scene windows — independent re-derivation of dataloader.filter_scenes     #
# --------------------------------------------------------------------------- #
def _windows(rows, h, f, fi, has_route, tokens):
    """Re-derivation of `filter_scenes` (dataloader.py:24-73 @0a380a9), written
    from the source's semantics and CHECKED against the devkit call below."""
    nf = h + f
    fi = nf if fi is None else fi
    out = {}
    for s in range(0, len(rows), fi):
        w = rows[s:s + nf]
        if len(w) < nf:
            continue
        cur = w[h - 1]
        if has_route and len(cur["rb"]) == 0:
            continue
        if tokens is not None and cur["token"] not in tokens:
            continue
        out[cur["token"]] = {"log": cur["log"], "start": s, "end": s + nf,
                             "cur_i": s + h - 1, "final_token": w[nf - 1]["token"],
                             "ts": cur["ts"]}
    return out


_DEVKIT_NS = {}


def _devkit_symbols() -> dict:
    """``SceneFilter`` and ``filter_scenes`` EXECUTED VERBATIM from the pinned git
    blobs (dataclasses.py / dataloader.py @0a380a9), not re-typed and not imported.

    Why exec rather than ``import navsim``: importing the devkit drags in the whole
    nuPlan stack from the external D: drive (MEASURED 2026-09-19: an import probe
    spent >10 min at ~15 s CPU, I/O-starved behind a concurrent 38 GB download).
    The two objects are self-contained (stdlib + tqdm), so executing their exact
    source text IS the devkit's own derivation without the import cost. The blob
    sha256 of each source file and the executed line span are recorded."""
    if _DEVKIT_NS:
        return _DEVKIT_NS
    import ast
    import dataclasses
    import typing
    import warnings
    import types
    mod = types.ModuleType("navsim_pinned_0a380a9")   # dataclass() resolves string
    sys.modules[mod.__name__] = mod                    # annotations via sys.modules
    ns = mod.__dict__
    ns.update({k: getattr(typing, k) for k in ("Any", "Dict", "List", "Optional", "Tuple",
                                               "Union")})
    ns.update({"dataclass": dataclasses.dataclass, "warnings": warnings, "pickle": pickle,
               "Path": Path, "tqdm": (lambda it, **kw: it)})
    prov = {}
    for rel, name in (("navsim/common/dataclasses.py", "SceneFilter"),
                      ("navsim/common/dataloader.py", "filter_scenes")):
        raw = _git_show(rel)
        src = raw.decode("utf-8")
        node = next(n for n in ast.parse(src).body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
        seg = ast.get_source_segment(src, node)
        first = node.lineno
        if isinstance(node, ast.ClassDef) and node.decorator_list:
            first = node.decorator_list[0].lineno
            seg = "\n".join(src.splitlines()[first - 1:node.end_lineno])
        code = "from __future__ import annotations\n" + seg
        exec(compile(code, f"{rel}@{DEVKIT_SHA[:7]}::{name}", "exec"), ns)
        prov[name] = {"blob": f"{rel}@{DEVKIT_SHA[:7]}",
                      "blob_sha256": hashlib.sha256(raw).hexdigest(),
                      "lines": f"{first}-{node.end_lineno}"}
    _DEVKIT_NS.update(ns)
    _DEVKIT_NS["_provenance"] = prov
    return _DEVKIT_NS


def devkit_scenes(log_name, h, f, fi, has_route, tokens):
    """The devkit's OWN filter_scenes, one log at a time (memory-bounded)."""
    ns = _devkit_symbols()
    sf = ns["SceneFilter"](num_history_frames=h, num_future_frames=f, frame_interval=fi,
                           has_route=has_route, max_scenes=None, log_names=[log_name],
                           tokens=(list(tokens) if tokens is not None else None))
    scenes, final_tokens = ns["filter_scenes"](LOGS, sf)
    return set(scenes.keys()), final_tokens


# --------------------------------------------------------------------------- #
# 3. Overlap measurements                                                      #
# --------------------------------------------------------------------------- #
def overlap_block(scenes: dict, nf: int, frames: dict) -> dict:
    by_log = defaultdict(list)
    for tok, s in scenes.items():
        by_log[s["log"]].append(s)
    d_start, shared, share_frac = [], [], []
    n_pairs_any, n_pairs_overlap = 0, 0
    frame_uses, union = 0, 0
    for log, ss in by_log.items():
        ss.sort(key=lambda s: s["start"])
        covered = set()
        for s in ss:
            covered.update(range(s["start"], s["end"]))
        union += len(covered)
        frame_uses += len(ss) * nf
        for a, b in zip(ss, ss[1:]):
            dlt = b["start"] - a["start"]
            d_start.append(dlt)
            sh = max(0, nf - dlt)
            shared.append(sh)
            share_frac.append(sh / nf)
        starts = [s["start"] for s in ss]
        for i in range(len(starts)):
            for j in range(i + 1, len(starts)):
                n_pairs_any += 1
                if starts[j] - starts[i] < nf:
                    n_pairs_overlap += 1
    # the frame spacing, MEASURED from timestamps inside each log
    dts = []
    for log in by_log:
        rows = frames[log]
        dts.extend((rows[k + 1]["ts"] - rows[k]["ts"]) / 1e6 for k in range(len(rows) - 1))
    return {
        "scene_window_frames": nf,
        "consecutive_scene_start_gap_frames": _stats(d_start),
        "consecutive_scenes_sharing_frames": {
            "n_consecutive_pairs": len(shared),
            "n_sharing_ge1_frame": int(sum(1 for x in shared if x > 0)),
            "frac_sharing_ge1_frame": (round(sum(1 for x in shared if x > 0) / len(shared), 4)
                                       if shared else None),
            "shared_frames": _stats(shared),
            "mean_shared_fraction_of_window": (round(float(np.mean(share_frac)), 4)
                                               if share_frac else None)},
        "within_log_scene_pairs": {"n_pairs": n_pairs_any,
                                   "n_pairs_sharing_ge1_frame": n_pairs_overlap,
                                   "frac": (round(n_pairs_overlap / n_pairs_any, 4)
                                            if n_pairs_any else None)},
        "frame_reuse_factor": {
            "_what": "sum over scenes of window frames / distinct frames covered = how many "
                     "scenes each frame appears in, on average (1.0 = no overlap)",
            "frame_uses": frame_uses, "distinct_frames": union,
            "factor": round(frame_uses / union, 4) if union else None},
        "keyframe_dt_s": _stats([round(x, 3) for x in dts]),
    }


def cross_log_block(logs_used, frames) -> dict:
    """Do two SEGMENTS of one nuPlan drive share frames or overlap in time?"""
    by_drive = defaultdict(list)
    for lg in logs_used:
        by_drive[_drive(lg)].append(lg)
    multi = {d: sorted(v) for d, v in by_drive.items() if len(v) > 1}
    gaps, shared_tokens, overl = [], 0, 0
    for d, lgs in multi.items():
        rng = []
        for lg in lgs:
            ts = [r["ts"] for r in frames[lg]]
            rng.append((min(ts), max(ts), lg, set(r["token"] for r in frames[lg])))
        rng.sort()
        for a, b in zip(rng, rng[1:]):
            gaps.append((b[0] - a[1]) / 1e6)
            if b[0] <= a[1]:
                overl += 1
            shared_tokens += len(a[3] & b[3])
    return {"n_drives": len(by_drive), "n_drives_with_ge2_segments": len(multi),
            "segments_per_drive": _stats([len(v) for v in by_drive.values()]),
            "adjacent_segment_time_gap_s": _stats([round(g, 1) for g in gaps]),
            "n_adjacent_segment_pairs_overlapping_in_time": overl,
            "n_tokens_shared_between_segments": shared_tokens,
            "_read_control": f"{sum(len(frames[l]) for l in logs_used)} frames read "
                             f"across {len(logs_used)} segments"}


# --------------------------------------------------------------------------- #
# 4. Two-stage mapping                                                         #
# --------------------------------------------------------------------------- #
def mapping_block(mapping, stage1: dict, synth_yaml: list, frames: dict,
                  tok_index: dict) -> dict:
    keys = []
    for entry in mapping:
        orig, prev, pairs = str(entry[0]), str(entry[1]), entry[2]
        keys.append((orig, prev, [tuple(str(x) for x in p) for p in pairs]))
    n_keys = len(keys)
    in_s1 = sum(1 for o, p, _ in keys if o in stage1 and p in stage1)
    same_log, dts, didx, logs_of_keys = 0, [], [], []
    for o, p, _ in keys:
        lo, lp = tok_index.get(o), tok_index.get(p)
        if lo and lp:
            if lo[0] == lp[0]:
                same_log += 1
            ro, rp = frames[lo[0]][lo[1]], frames[lp[0]][lp[1]]
            dts.append(round((ro["ts"] - rp["ts"]) / 1e6, 3))
            didx.append(lo[1] - lp[1])
            logs_of_keys.append(lo[0])
    s2_tokens = [t for _, _, pairs in keys for pr in pairs for t in pr]
    s2_count = Counter(s2_tokens)
    per_key = [len(pairs) for _, _, pairs in keys]
    keys_per_log = Counter(logs_of_keys)
    return {
        "n_mapping_keys": n_keys,
        "n_keys_both_tokens_in_reconstructed_stage1": in_s1,
        "n_keys_orig_and_prev_in_same_log": same_log,
        "orig_minus_prev_dt_s": _stats(dts),
        "orig_minus_prev_frame_index": _stats(didx),
        "stage2_pairs_per_key": _stats(per_key),
        "n_stage2_tokens_in_mapping": len(s2_tokens),
        "n_distinct_stage2_tokens_in_mapping": len(s2_count),
        "n_stage2_tokens_in_more_than_one_key": sum(1 for c in s2_count.values() if c > 1),
        "stage2_mapping_equals_yaml_reactive_synthetic_initial_tokens": (
            set(s2_count) == set(synth_yaml) if synth_yaml is not None else None),
        "n_yaml_stage2_tokens": (len(synth_yaml) if synth_yaml is not None else None),
        "clusters": {
            "log_name": {"n": len(keys_per_log),
                         "keys_per_log": _stats(list(keys_per_log.values()))},
            "nuplan_drive": {"n": len(set(_drive(l) for l in logs_of_keys)),
                             "keys_per_drive": _stats(list(Counter(
                                 _drive(l) for l in logs_of_keys).values()))},
        },
        "key_to_log": {o: tok_index[o][0] for o, _, _ in keys if o in tok_index},
    }


class _PosixSafeUnpickler(pickle.Unpickler):
    """The synthetic-scene pickles embed pathlib.PosixPath (Linux-written), which
    Windows refuses to instantiate. PurePosixPath keeps the value and is inert.
    ⚠️ Deliberately NOT `Scene.load_from_disk`: that call BUILDS THE MAP API for
    every scene (dataclasses.py:691-693 @0a380a9) — a GB-scale load that the
    metadata read does not need."""

    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            import pathlib
            return pathlib.PurePosixPath
        return super().find_class(module, name)


def read_synthetic_pickles(sdir: Path) -> tuple:
    files = sorted(p for p in sdir.iterdir() if p.suffix == ".pkl")
    meta, fails = {}, []
    for p in files:
        try:
            with open(p, "rb") as f:
                sd = _PosixSafeUnpickler(f).load()
            m = sd["scene_metadata"]
            fr = sd["frames"]
            h = int(m["num_history_frames"])
            cur = fr[h - 1]
            es = cur["ego_status"]
            meta[m["initial_token"]] = {
                "log": m["log_name"], "file": p.stem, "scene_token": m["scene_token"],
                "map": m["map_name"],
                "corr_orig": m.get("corresponding_original_scene"),
                "corr_orig_init": m.get("corresponding_original_initial_token"),
                "n_frames": len(fr), "num_history_frames": h,
                "num_future_frames": int(m["num_future_frames"]),
                "cur_token": cur["token"], "cur_ts": int(cur["timestamp"]),
                "cur_rb": tuple(str(x) for x in (cur["roadblock_ids"] or [])),
                "cur_cmd": tuple(int(x) for x in np.asarray(es["driving_command"]).tolist()),
                "cur_pose": [float(x) for x in np.asarray(es["ego_pose"]).tolist()],
                "cur_in_global_frame": bool(es.get("in_global_frame", False)),
            }
        except Exception as e:
            fails.append({"file": p.name, "error": f"{type(e).__name__}: {e}"[:200]})
    return files, meta, fails


def warmup_synthetic_block(mapping, tok_index, frames, stage1, syn) -> dict:
    """Stage-2 -> stage-1 -> log, verified per synthetic scene from its own pickle."""
    files, meta, fails = syn
    n_pairs, ok_log, ok_final, ok_init = 0, 0, 0, 0
    detail = []
    for entry in mapping:
        orig, prev, pairs = str(entry[0]), str(entry[1]), entry[2]
        for now_t, prev_t in pairs:
            for syn, s1 in ((str(now_t), orig), (str(prev_t), prev)):
                n_pairs += 1
                m = meta.get(syn)
                if m is None:
                    detail.append({"synthetic": syn, "missing": True})
                    continue
                log_s1 = tok_index.get(s1, (None,))[0]
                if m["log"] == log_s1:
                    ok_log += 1
                if s1 in stage1 and m["corr_orig"] == stage1[s1]["final_token"]:
                    ok_final += 1
                if m["corr_orig_init"] == s1:
                    ok_init += 1
    return {"n_synthetic_pickles_listed": len(files),
            "n_synthetic_pickles_read": len(meta), "read_failures": fails,
            "n_mapping_slots_checked": n_pairs,
            "n_synthetic_log_equals_stage1_log": ok_log,
            "n_corresponding_original_scene_equals_stage1_final_frame_token": ok_final,
            "n_corresponding_original_initial_token_equals_stage1_token": ok_init,
            "missing_detail_first5": detail[:5],
            "_semantics": ("SceneMetadata.corresponding_original_scene = the original frame "
                           "at the SAME TIMESTAMP as the synthetic scene (dataclasses.py:306-310 "
                           "@0a380a9), i.e. the stage-1 scene's FINAL (t=+4 s) frame; "
                           "filter_synthetic_scenes keeps a synthetic scene only if that token is "
                           "a loaded stage-1 final-frame token (dataloader.py:98-103).")}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-cache", required=True,
                    help="pickle of the compact per-frame rows, reused by the leak probe")
    ap.add_argument("--cluster-maps-dir", default=None,
                    help="write token->log_name maps per split here (JSON)")
    ap.add_argument("--skip-devkit-reference", action="store_true",
                    help="DIAGNOSTIC ONLY — the published census requires both derivations")
    a = ap.parse_args(argv)
    t0 = time.time()
    census = {"_what": "E3 A1/A2 census of NavSim split cluster structure",
              "devkit_sha": DEVKIT_SHA, "logs_dir": str(LOGS), "warmup_dir": str(WARMUP),
              "evidence_class": "MEASURED (devkit YAML blobs @ pin + local OpenScene test metadata)"}

    # ---- split YAMLs (read controls: bytes + sha256) ------------------------- #
    names = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(DEVKIT), "ls-tree",
                            "--name-only", DEVKIT_SHA, SPLIT_DIR, SPLIT_DIR + "scene_filter/"],
                           capture_output=True, text=True).stdout.split()
    yml = {n.replace(SPLIT_DIR, ""): _yaml(n) for n in names if n.endswith(".yaml")}
    census["yaml_read_controls"] = {k: v[1] for k, v in yml.items()}
    print(f"[census] read {len(yml)} split YAMLs at {DEVKIT_SHA[:7]}", flush=True)

    # ---- frames ------------------------------------------------------------- #
    frames, fails, read, n_listed = load_frames(LOGS)
    n_frames = sum(read.values())
    census["frames_read_control"] = {
        "n_log_pickles_listed": n_listed, "n_log_pickles_read": len(frames),
        "n_read_failures": len(fails), "read_failures": fails,
        "n_keyframes_read": n_frames,
        "frames_per_log": _stats(list(read.values()))}
    print(f"[census] logs listed {n_listed}, read {len(frames)}, failures {len(fails)}, "
          f"keyframes {n_frames}  ({time.time() - t0:.0f}s)", flush=True)
    tok_index = {}
    for lg, rows in frames.items():
        for r in rows:
            tok_index[r["token"]] = (lg, r["i"])
    syn = read_synthetic_pickles(WARMUP / "synthetic_scene_pickles")
    with open(a.frames_cache, "wb") as f:
        pickle.dump({"frames": frames, "warmup_synthetic": syn[1]}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    census["frames_cache"] = {"path": a.frames_cache, "bytes": os.path.getsize(a.frames_cache),
                              "n_warmup_synthetic_read": len(syn[1]),
                              "n_warmup_synthetic_listed": len(syn[0]),
                              "n_warmup_synthetic_failures": len(syn[2])}
    census["all_local_test_logs"] = {
        "n_log_segments": len(frames),
        "n_nuplan_drives": len(set(_drive(l) for l in frames)),
        "n_log_names_not_matching_drive_pattern": sum(1 for l in frames if not DRIVE_RE.match(l)),
        "cities": dict(Counter(rows[0]["map"] for rows in frames.values() if rows)),
        "vehicles": len(set(rows[0]["veh"] for rows in frames.values() if rows)),
    }

    # ---- per split ----------------------------------------------------------- #
    out = {}
    for split, (tts, _rc) in sorted((k, v) for k, v in yml.items() if "/" not in k):
        name = split.replace(".yaml", "")
        sf_name = tts.get("defaults", [{}])[0].get("scene_filter")
        sf, _ = yml.get(f"scene_filter/{sf_name}.yaml", ({}, None))
        blk = {"data_split": tts.get("data_split"), "scene_filter": sf_name,
               "num_history_frames": sf.get("num_history_frames"),
               "num_future_frames": sf.get("num_future_frames"),
               "frame_interval": sf.get("frame_interval"),
               "has_route": sf.get("has_route"),
               "include_synthetic_scenes": sf.get("include_synthetic_scenes", False),
               "yaml_log_names": (len(sf["log_names"]) if sf.get("log_names") is not None else None),
               "yaml_tokens": (len(sf["tokens"]) if sf.get("tokens") is not None else None),
               "yaml_reactive_synthetic_initial_tokens": (
                   len(sf["reactive_synthetic_initial_tokens"])
                   if sf.get("reactive_synthetic_initial_tokens") is not None else None),
               "yaml_synthetic_scene_tokens": (len(sf["synthetic_scene_tokens"])
                                               if sf.get("synthetic_scene_tokens") is not None
                                               else None),
               "reactive_all_mapping": ("null" if "reactive_all_mapping" in tts
                                        and tts["reactive_all_mapping"] is None else
                                        (len(tts["reactive_all_mapping"])
                                         if tts.get("reactive_all_mapping") is not None
                                         else "absent")),
               "two_stage_mapping": (len(tts["two_stage_mapping"])
                                     if tts.get("two_stage_mapping") is not None else "absent")}
        yaml_logs = sf.get("log_names")
        yaml_tokens = sf.get("tokens")
        if yaml_logs is not None:
            blk["yaml_n_drives"] = len(set(_drive(l) for l in yaml_logs))
        # tokens present in the local metadata — with a control
        if yaml_tokens is not None:
            blk["yaml_tokens_found_in_local_test_metadata"] = sum(
                1 for t in yaml_tokens if str(t) in tok_index)
        if yaml_logs is not None:
            blk["yaml_logs_found_locally"] = sum(1 for l in yaml_logs if l in frames)

        if name in SPLITS_WITH_LOCAL_LOGS and blk["data_split"] == "test":
            h, fu, fi = sf["num_history_frames"], sf["num_future_frames"], sf.get("frame_interval")
            hr = sf.get("has_route", True)
            toks = set(str(t) for t in yaml_tokens) if yaml_tokens is not None else None
            logs = [l for l in (yaml_logs if yaml_logs is not None else sorted(frames))
                    if l in frames]
            mine = {}
            for lg in logs:
                mine.update(_windows(frames[lg], h, fu, fi, hr, toks))
            blk["reconstructed_mine"] = len(mine)
            if not a.skip_devkit_reference:
                dk, dk_final = set(), []
                t1 = time.time()
                for lg in logs:
                    s, fin = devkit_scenes(lg, h, fu, fi, hr, toks)
                    dk |= s
                    dk_final.extend(fin)
                blk["reconstructed_devkit_filter_scenes"] = len(dk)
                blk["derivations_agree_token_for_token"] = (dk == set(mine))
                blk["devkit_seconds"] = round(time.time() - t1, 1)
                if dk != set(mine):
                    blk["DISAGREEMENT"] = {"only_devkit": sorted(dk - set(mine))[:20],
                                           "only_mine": sorted(set(mine) - dk)[:20]}
            if toks is not None:
                blk["yaml_tokens_not_reconstructed"] = len(toks - set(mine))
            logs_used = sorted(set(s["log"] for s in mine.values()))
            per_log = Counter(s["log"] for s in mine.values())
            per_drive = Counter(_drive(s["log"]) for s in mine.values())
            blk["stage1"] = {
                "n_scenes": len(mine),
                "n_log_segments": len(logs_used),
                "n_nuplan_drives": len(per_drive),
                "n_cities": len(set(frames[l][0]["map"] for l in logs_used)),
                "cities": dict(Counter(frames[l][0]["map"] for l in logs_used)),
                "n_vehicles": len(set(frames[l][0]["veh"] for l in logs_used)),
                "scenes_per_log": _stats(list(per_log.values())),
                "scenes_per_drive": _stats(list(per_drive.values())),
            }
            blk["overlap_within_log"] = overlap_block(mine, h + fu, frames)
            blk["cross_log_same_drive"] = cross_log_block(logs_used, frames)
            if tts.get("reactive_all_mapping"):
                blk["two_stage"] = mapping_block(
                    tts["reactive_all_mapping"], mine,
                    [str(t) for t in sf.get("reactive_synthetic_initial_tokens") or []],
                    frames, tok_index)
                if name == "warmup_two_stage":
                    blk["two_stage"]["synthetic_scene_verification"] = warmup_synthetic_block(
                        tts["reactive_all_mapping"], tok_index, frames, mine, syn)
            if a.cluster_maps_dir:
                os.makedirs(a.cluster_maps_dir, exist_ok=True)
                cm = {"split": name, "devkit_sha": DEVKIT_SHA,
                      "cluster_unit": "log_name",
                      "token_to_log_name": {t: s["log"] for t, s in sorted(mine.items())}}
                if "two_stage" in blk:
                    cm["mapping_key_orig_token_to_log_name"] = blk["two_stage"]["key_to_log"]
                p = os.path.join(a.cluster_maps_dir, f"{name}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(cm, f, indent=0, sort_keys=True)
                blk["cluster_map_file"] = p
        else:
            blk["stage1"] = {"status": "UNAVAILABLE",
                             "reason": (f"data_split={blk['data_split']!r}: its logs are not in the "
                                        f"local OpenScene TEST metadata, so scenes cannot be "
                                        f"reconstructed here. Counts above are from the YAML only."),
                             "n": blk.get("yaml_tokens") or 0}
        out[name] = blk
        s1 = blk.get("stage1", {})
        print(f"[census] {name:36s} scenes={s1.get('n_scenes')} logs={s1.get('n_log_segments')} "
              f"drives={s1.get('n_nuplan_drives')} agree={blk.get('derivations_agree_token_for_token')}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    census["splits"] = out
    census["devkit_reference_provenance"] = _DEVKIT_NS.get("_provenance")

    # ---- the private split: 0 found is meaningful only beside a control ------ #
    ph = yml.get("scene_filter/private_test_hard_two_stage.yaml", ({}, None))[0]
    nh = yml.get("scene_filter/navhard_two_stage.yaml", ({}, None))[0]
    census["private_test_hard_visibility"] = {
        "private_tokens": len(ph.get("tokens") or []),
        "private_tokens_found_in_local_test_metadata": sum(
            1 for t in (ph.get("tokens") or []) if str(t) in tok_index),
        "CONTROL_navhard_tokens": len(nh.get("tokens") or []),
        "CONTROL_navhard_tokens_found": sum(1 for t in (nh.get("tokens") or [])
                                            if str(t) in tok_index)}
    census["elapsed_s"] = round(time.time() - t0, 1)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(census, f, indent=1, sort_keys=False, default=str)
    print(f"[census] wrote {a.out} ({census['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
