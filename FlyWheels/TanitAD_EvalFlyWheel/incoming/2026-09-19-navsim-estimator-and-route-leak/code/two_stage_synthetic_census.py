#!/usr/bin/env python3
"""E3 Part A1/A2 — stage-2 (synthetic) scenes -> stage-1 keys -> LOGS, verified per
synthetic scene from its OWN pickle. Runs in the TanitAD venv.

    python -u code/two_stage_synthetic_census.py --split warmup_two_stage \
        --frames-cache <scratch>/frames_cache.pkl --out raw/two_stage_warmup_two_stage.json

WHY THIS IS A SEPARATE STEP FROM split_census.py
------------------------------------------------
The mapping YAML (`reactive_all_mapping`) is a CLAIM that stage-2 scene X belongs
to key (orig, prev). The log-cluster bootstrap is exact only if every stage-2 scene
of a key lives in the SAME log as the key — otherwise one resampled unit would
straddle two clusters. The YAML cannot prove that; the synthetic scene's own
`scene_metadata.log_name` can. So each synthetic pickle is read and joined.

⚠️ The pickles were written on Linux and embed `pathlib.PosixPath` and nuPlan
classes (extended detections / traffic lights). This venv has no nuPlan, so
UNKNOWN classes are stubbed INERTLY (`_Stub`): only the plain-dict metadata and the
numpy ego-status arrays are read; a stub is never used as a value.

⛔ Read controls: files listed vs read vs failed are printed; a 0 is published only
next to a non-zero control.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import yaml

DEVKIT = Path(r"C:/Users/Admin/navsim/devkit")
DEVKIT_SHA = "0a380a9063d7162ec93d0f51e9990ebac585f720"
DATA = Path(r"C:/Users/Admin/navsim/data/openscene")
SPLIT_DIR = "navsim/planning/script/config/common/train_test_split/"


class _Stub:
    """Inert placeholder for a class this venv cannot import (nuPlan)."""

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, state):
        self._state = state


class _Unpickler(pickle.Unpickler):
    _stubs: dict = {}

    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return PurePosixPath
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, ImportError):
            key = f"{module}.{name}"
            if key not in self._stubs:
                self._stubs[key] = type(f"Stub_{name}", (_Stub,), {"_origin": key})
            return self._stubs[key]


def _load(p: Path):
    with open(p, "rb") as f:
        return _Unpickler(f).load()


def _yaml(rel: str):
    r = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(DEVKIT), "show",
                        f"{DEVKIT_SHA}:{rel}"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"cannot read {rel}@{DEVKIT_SHA[:7]}")
    return yaml.safe_load(r.stdout.decode("utf-8")), {
        "bytes": len(r.stdout), "sha256": hashlib.sha256(r.stdout).hexdigest()}


def _stats(xs):
    xs = list(xs)
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "median": float(np.median(xs)), "max": max(xs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["warmup_two_stage", "navhard_two_stage"])
    ap.add_argument("--frames-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default=str(DATA),
                    help="openscene root holding <split>/ (the C: working copy "
                         "C:/Users/Admin/navsim-crun/data/openscene is byte-verified by its "
                         "EXTRACT_DONE.json / E1's mirror receipts)")
    ap.add_argument("--synthetic-out", default=None,
                    help="optional pickle of per-synthetic-scene metadata (for the leak probe)")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = Path(a.data_root) / a.split
    done = root / "EXTRACT_DONE.json"
    extract_receipt = None
    if a.split == "navhard_two_stage":
        if not done.exists():
            print(f"REFUSED: {done} absent — a partial extraction would under-count silently",
                  flush=True)
            return 3
        extract_receipt = json.loads(done.read_text(encoding="utf-8"))
        if not extract_receipt.get("ok"):
            print(f"REFUSED: {done} says ok={extract_receipt.get('ok')!r}", flush=True)
            return 3
    tts, tts_rc = _yaml(f"{SPLIT_DIR}{a.split}.yaml")
    sf, sf_rc = _yaml(f"{SPLIT_DIR}scene_filter/{a.split}.yaml")
    mapping = [(str(e[0]), str(e[1]), [(str(p[0]), str(p[1])) for p in e[2]])
               for e in tts["reactive_all_mapping"]]
    with open(a.frames_cache, "rb") as f:
        cache = pickle.load(f)
    frames = cache["frames"]
    tok_index = {r["token"]: (lg, r["i"]) for lg, rows in frames.items() for r in rows}
    h, fu = int(sf["num_history_frames"]), int(sf["num_future_frames"])

    def final_token(tok):
        lg, i = tok_index[tok]
        j = i - (h - 1) + (h + fu) - 1          # the scene window's last frame
        rows = frames[lg]
        return rows[j]["token"] if j < len(rows) else None

    # ---- synthetic scene pickles ---------------------------------------------- #
    sdir = root / "synthetic_scene_pickles"
    files = sorted(p for p in sdir.iterdir() if p.suffix == ".pkl")
    meta, fails = {}, []
    for p in files:
        try:
            sd = _load(p)
            m = sd["scene_metadata"]
            frs = sd["frames"]
            hh = int(m["num_history_frames"])
            cur = frs[hh - 1]
            es = cur["ego_status"]
            meta[str(m["initial_token"])] = {
                "file": p.stem, "log": m["log_name"], "scene_token": m["scene_token"],
                "map": m["map_name"], "corr_orig": m.get("corresponding_original_scene"),
                "corr_orig_init": m.get("corresponding_original_initial_token"),
                "n_frames": len(frs), "num_history_frames": hh,
                "num_future_frames": int(m["num_future_frames"]),
                "cur_token": str(cur["token"]), "cur_ts": int(cur["timestamp"]),
                "cur_rb": tuple(str(x) for x in (cur["roadblock_ids"] or [])),
                "cur_cmd": tuple(int(x) for x in np.asarray(es["driving_command"]).tolist()),
                "cur_pose": [float(x) for x in np.asarray(es["ego_pose"]).tolist()],
                "cur_in_global_frame": bool(es.get("in_global_frame", False)),
                "hist_cmds": [tuple(int(x) for x in np.asarray(fr["ego_status"]["driving_command"]).tolist())
                              for fr in frs[:hh]],
                "hist_poses": [[float(x) for x in np.asarray(fr["ego_status"]["ego_pose"]).tolist()]
                               for fr in frs[:hh]],
            }
        except Exception as e:
            fails.append({"file": p.name, "error": f"{type(e).__name__}: {e}"[:200]})
    print(f"[2stage] {a.split}: synthetic pickles listed {len(files)} read {len(meta)} "
          f"failed {len(fails)} ({time.time() - t0:.0f}s)", flush=True)

    # ---- attributes CSV --------------------------------------------------------- #
    csv_rows = []
    cpath = root / "synthetic_scenes_attributes.csv"
    if cpath.exists():
        with open(cpath, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                csv_rows.append({k: r[k] for k in ("log_name", "synthetic_scene_token",
                                                   "corresponding_original_scene_token",
                                                   "previous_original_scene_initial_token",
                                                   "map_name", "pair_identifier")})

    # ---- join: every mapping slot -> its synthetic scene -> its log ------------- #
    slots = n_meta = ok_log = ok_final = ok_init = ok_map = 0
    bad = []
    keys_per_log = Counter()
    s2_per_log = Counter()
    for orig, prev, pairs in mapping:
        lg_key = tok_index.get(orig, (None,))[0]
        keys_per_log[lg_key] += 1
        for now_t, prev_t in pairs:
            for syn, s1 in ((now_t, orig), (prev_t, prev)):
                slots += 1
                m = meta.get(syn)
                if m is None:
                    bad.append({"synthetic": syn, "why": "no pickle"})
                    continue
                n_meta += 1
                s2_per_log[m["log"]] += 1
                lg_s1 = tok_index.get(s1, (None,))[0]
                ok_log += (m["log"] == lg_s1)
                ok_final += (m["corr_orig"] is not None and m["corr_orig"] == final_token(s1))
                ok_init += (m["corr_orig_init"] == s1)
                if lg_s1 and frames[lg_s1][0]["map"] == m["map"]:
                    ok_map += 1
                if m["log"] != lg_s1:
                    bad.append({"synthetic": syn, "stage1": s1, "syn_log": m["log"],
                                "stage1_log": lg_s1})
    out = {
        "split": a.split, "devkit_sha": DEVKIT_SHA, "data_root": str(root),
        "extract_receipt": extract_receipt,
        "evidence_class": "MEASURED (synthetic scene pickles + OpenScene test metadata + "
                          "split YAML blobs @ pin)",
        "yaml_read_controls": {"train_test_split": tts_rc, "scene_filter": sf_rc},
        "n_mapping_keys": len(mapping),
        "n_mapping_slots": slots,
        "synthetic_pickles": {"listed": len(files), "read": len(meta), "failed": len(fails),
                              "failures_first5": fails[:5],
                              "stubbed_classes": sorted(_Unpickler._stubs)},
        "n_slots_with_a_pickle": n_meta,
        "n_slots_synthetic_log_equals_stage1_log": ok_log,
        "n_slots_corresponding_original_scene_is_stage1_final_frame": ok_final,
        "n_slots_corresponding_original_initial_token_is_stage1_token": ok_init,
        "n_slots_same_map": ok_map,
        "mismatches_first10": bad[:10], "n_mismatches": len(bad),
        "keys_per_log": _stats(list(keys_per_log.values())),
        "n_logs_with_keys": len([k for k in keys_per_log if k]),
        "stage2_scenes_per_log": _stats(list(s2_per_log.values())),
        "synthetic_frames_per_scene": _stats([m["n_frames"] for m in meta.values()]),
        "synthetic_num_future_frames": dict(Counter(m["num_future_frames"] for m in meta.values())),
        "csv": {"path": str(cpath), "rows": len(csv_rows),
                "distinct_logs": len(set(r["log_name"] for r in csv_rows)),
                "rows_whose_prev_original_initial_token_is_a_key_orig":
                    sum(1 for r in csv_rows
                        if r["previous_original_scene_initial_token"] in {m[0] for m in mapping})},
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    if a.synthetic_out:
        with open(a.synthetic_out, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({k: out[k] for k in ("n_mapping_keys", "n_mapping_slots",
                                          "n_slots_with_a_pickle",
                                          "n_slots_synthetic_log_equals_stage1_log",
                                          "n_slots_corresponding_original_scene_is_stage1_final_frame",
                                          "n_logs_with_keys", "n_mismatches")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
