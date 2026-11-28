"""build_train139_windows.py -- P2 of the L7 train-split fan capture: the REGISTERED TRAIN139 clip + window rule.

THE RULE (nothing here is new; it is the RETIME train-fit / refcv6-A6 precedent, REPRODUCED and then WRITTEN DOWN):
  CLIPS   the 139 smallest sha12(clip_id) among train-cache clips that are not in eval139 and that have a train label
          record AND a train max-speed sidecar row (pool 4,369)            [refcv6 A6 select_record.json, 2026-09-26]
  WINDOWS for every selected clip, every V3Dataset index entry (e_i, t) with t % 5 == 0, ws = t + W - 1 (W = 8);
          the dataset's own index is t in range(T - W - max_horizon) [tanitad/data/_contract.py:121] with max_horizon = 20 stacked rows
          [roll_seed_r8_train.py: the eval dataset's index restricted to t % 5 == 0]

THIS SCRIPT (all CPU, no model, no GPU):
  1. re-derives the 139 clips FROM THE RULE (pool list + label ids + sidecar ids + eval139 ids) and asserts the result
     equals (a) the banked train_clips_sha12.json, (b) the file names in the dev-box A6 cache dir          [K-CLIP]
  2. re-derives every clip's window list FROM THE CACHE MANIFEST (T per clip) and asserts it equals the ws list the
     banked train-fit roll actually rolled (trainfit/.../dump_s0, 4,855 windows)                           [K-WIN]
  3. asserts episode-disjointness from eval139 by sha12                                                    [K-DISJOINT]
  4. writes TRAIN139_WINDOWS.json keyed by `clip-sha12-<12 hex>` ONLY (never a clip id, never an 8+ hex id prefix),
     its sha256, and a verification record.

    python build_train139_windows.py --pool-list <refcv6_train_clips.txt> --out-dir <l7cap>
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys

import numpy as np

KIT = "E:/refcv6_eval_kit/data"
A6_CACHE = f"{KIT}/refcv6-b1-416x1024-train-a6"
EVAL_CACHE = f"{KIT}/refcv6-b1-416x1024-eval139"
A6_LABELS = f"{KIT}/a6/s2_labels_v8_train.jsonl.gz"
A6_SIDECAR = f"{KIT}/a6/refcv6_speed_max_v8_train.jsonl"
A6_SELECT_SHA12 = ("C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/"
                   "battery/raw/a6/train_clips_sha12.json")
BANKED_DUMP = "C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf/dump_s0"
EVAL_WINDOWS_SHA12 = "E:/refcv8_eval_kit/windows/s2_windows.json.sha12.json"
STRIDE, W, MAX_H, N_CLIPS = 5, 8, 20, 139
TAG = "clip-sha12-"


def sha12(clip_id: str) -> str:
    return hashlib.sha256(str(clip_id).encode()).hexdigest()[:12]


def ids_of_cache(d: str) -> list:
    return sorted(os.path.basename(f)[:-len(".v2ep.pt")] for f in glob.glob(d + "/*.v2ep.pt"))


def label_ids(path: str) -> set:
    out = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                out.add(str(r.get("clip_id") or r.get("sid") or r.get("episode_id")))
    return out


def sidecar_ids(path: str) -> set:
    out = set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                r = json.loads(ln)
                out.add(str(r.get("clip_id") or r.get("sid") or r.get("episode_id")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-list", required=True, help="the train-cache clip list (Thor data/refcv6_train_clips.txt, 4,369 ids)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    import torch
    pool = [ln.strip() for ln in open(a.pool_list, encoding="utf-8") if ln.strip()]
    eval_ids = ids_of_cache(EVAL_CACHE)
    lab, sc = label_ids(A6_LABELS), sidecar_ids(A6_SIDECAR)
    ev12 = {sha12(c) for c in eval_ids}
    cand = [c for c in pool if sha12(c) not in ev12 and c in lab and c in sc]
    sel = sorted(cand, key=sha12)[:N_CLIPS]
    sel12 = [sha12(c) for c in sel]
    rec = {"n_pool": len(pool), "n_eval": len(eval_ids), "n_label_records": len(lab), "n_sidecar_rows": len(sc),
           "n_candidates": len(cand), "n_selected": len(sel)}
    # K-CLIP
    banked = json.load(open(A6_SELECT_SHA12, encoding="utf-8"))
    a6_files = ids_of_cache(A6_CACHE)
    k_clip = {"equals_banked_train_clips_sha12_json": sorted(sel12) == sorted(banked),
              "equals_a6_cache_dir_file_names": sorted(sel) == sorted(a6_files),
              "n_banked": len(banked), "n_a6_files": len(a6_files),
              "first_last_sha12": [sel12[0], sel12[-1]]}
    # K-DISJOINT
    k_dis = {"n_intersection_with_eval139_by_sha12": len(set(sel12) & ev12), "n_eval": len(ev12)}
    # K-WIN: derive windows from the cache manifest alone, compare with the banked roll
    man = torch.load(f"{A6_CACHE}/_v2manifest.pt", weights_only=False)
    by_id = {str(c): (i, int(man["poses"][i].shape[0])) for i, c in enumerate(man["clip_id"])}
    derived = {}
    for c in sel:
        _, T = by_id[c]
        derived[sha12(c)] = [t + W - 1 for t in range(T - W - MAX_H) if t % STRIDE == 0]
    dm = json.load(open(f"{BANKED_DUMP}/manifest.json", encoding="utf-8"))
    rolled = {}
    for e in dm["episodes"]:
        z = np.load(f"{BANKED_DUMP}/ep{int(e['file_index']):03d}.npz", allow_pickle=True)
        rolled[sha12(e["clip_id"])] = sorted(int(x) for x in np.asarray(z["ws"]).ravel())
    n_diff = [k for k in derived if derived[k] != rolled.get(k)]
    k_win = {"n_clips_derived": len(derived), "n_clips_rolled": len(rolled),
             "n_windows_derived": int(sum(map(len, derived.values()))), "n_windows_rolled": int(sum(map(len, rolled.values()))),
             "n_clips_with_different_window_list": len(n_diff), "same_clip_set": sorted(derived) == sorted(rolled)}
    ok = (k_clip["equals_banked_train_clips_sha12_json"] and k_clip["equals_a6_cache_dir_file_names"]
          and k_dis["n_intersection_with_eval139_by_sha12"] == 0 and not n_diff and k_win["same_clip_set"])
    # eval grid comparison (stride / phase), reported
    ew = json.load(open(EVAL_WINDOWS_SHA12, encoding="utf-8"))
    ephase = {}
    for v in ew.values():
        p = (int(v[0]) - (W - 1)) % STRIDE
        ephase[p] = ephase.get(p, 0) + 1
    eval_stride = sorted({int(b - a_) for v in ew.values() for a_, b in zip(v[:-1], v[1:])})
    grid_cmp = {"train139": {"stride_ticks": STRIDE, "t_phase_mod5": {0: len(derived)}, "n_windows": k_win["n_windows_derived"]},
                "eval139": {"stride_ticks_seen": eval_stride, "first_window_t_phase_mod5_by_clip": {int(k): v for k, v in sorted(ephase.items())},
                            "n_windows": int(sum(map(len, ew.values())))}}
    out = {"schema": "l7cap-train139-windows/1",
           "rule_clips": "139 smallest sha12(clip_id) among train-cache clips not in eval139 with a train label record and a train "
                         "max-speed sidecar row (pool 4,369); refcv6 A6 select_record.json 2026-09-26",
           "rule_windows": "t % 5 == 0 over the V3Dataset index t in range(T-W-20), ws = t + W - 1, W = 8 (roll_seed_r8_train.py)",
           "W": W, "stride_ticks": STRIDE, "max_horizon_rows": MAX_H,
           "n_clips": len(derived), "n_windows": k_win["n_windows_derived"],
           "clips": {TAG + k: {"ws": v} for k, v in sorted(derived.items())}}
    blob = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "TRAIN139_WINDOWS.json")
    with open(p, "wb") as fh:
        fh.write(json.dumps(out, sort_keys=True, indent=1).encode())
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    open(p + ".sha256", "w").write(f"{h}  TRAIN139_WINDOWS.json\n")
    ver = {"ok": bool(ok), "pool": rec, "K_CLIP": k_clip, "K_DISJOINT": k_dis, "K_WIN": k_win, "grid_vs_eval139": grid_cmp,
           "TRAIN139_WINDOWS.json_sha256": h}
    json.dump(ver, open(os.path.join(a.out_dir, "TRAIN139_VERIFY.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(ver, indent=1)[:2500])
    # banked-file hygiene: no raw id, no 8+ hex prefix of any id may appear in what we bank
    txt = open(p, encoding="utf-8").read()
    leak = [c for c in sel if c in txt or c[:8] in txt]
    print("ID_LEAK_CHECK", len(leak), "ZZT139OK" if (ok and not leak) else "ZZT139BADZZ")
    return 0 if (ok and not leak) else 2


if __name__ == "__main__":
    sys.exit(main())
