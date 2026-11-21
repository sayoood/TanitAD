#!/usr/bin/env python3
"""P6-HN step 0 -- loader.  Builds ONE npz joining, per navtrain held-out set (log, token, step, rank):

  proposals   [N,64,20,2]  x,y in the ego frame, 20 poses   <- sets/onpolicy_r*_hw*.jsonl  field `traj`   (the generator's fan, ckpt 10075)
  prop_yaw    [N,64,20]    <- same files, field `yaw`
  teacher     [N,20,3]     x,y,yaw of the banked TEACHER plan <- queue/props_r*_*.jsonl*  field `teacher` (= TargetBank row['traj'])
  nc,dac,ep,ttc,c,ddc,pdms [N,64]  <- the S1 banked PDM labels (label_version 5, 'executed (A7 repair)')
  set_navsim_dac [N,64]    <- the SETS file's own `targets[k]['navsim_dac.violation']` (used ONLY for the order-alignment gate)

No census quantity is computed here.  Memory-light: one line is parsed, converted to float32, dropped.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe hn_load.py --out ../raw/hn_joined.npz
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ART = "D:/Projects/TanitAD-artifacts"
LABELS = ART + "/s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_*.jsonl"
SETS = ART + "/_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl"
QUEUE = ART + "/_tlane_work/ho/refe_heldout/queue/props_r*"
HEADS = ("nc", "dac", "ep", "ttc", "c", "ddc", "pdms")


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def key_of(d: dict) -> tuple:
    k = d["key"] if "key" in d else (d["log_name"], d["token"], d["step"], d["rank"])
    return (str(k[0]), str(k[1]), int(k[2]), int(k[3]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lab = {}
    for p in sorted(glob.glob(LABELS)):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            lab[key_of(d)] = {h: np.asarray(d[h], np.float32) for h in HEADS}
    print("labels", len(lab), flush=True)

    sets = {}
    dup_sets = 0
    for p in sorted(glob.glob(SETS)):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            k = key_of(d)
            if k not in lab:
                continue
            if k in sets:
                dup_sets += 1
                # keep the higher (ckpt_step, label_version), like the trainer's bank
                if (d["ckpt_step"], d["label_version"]) <= sets[k]["rank_key"]:
                    continue
            nav = np.array([t.get("navsim_dac.violation", np.nan) if t.get("navsim_dac.violation") is not None else np.nan
                            for t in d["targets"]], np.float32)
            sets[k] = {"traj": np.asarray(d["traj"], np.float32), "yaw": np.asarray(d["yaw"], np.float32), "nav": nav,
                       "rank_key": (d["ckpt_step"], d["label_version"]), "lv": d["label_version"], "ck": d["ckpt_step"]}
    print("sets matched to labels", len(sets), "dup_sets", dup_sets, flush=True)

    teach = {}
    dup_q = 0
    for p in sorted(glob.glob(QUEUE)):
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            k = key_of(d)
            if k not in lab:
                continue
            t = np.asarray(d["teacher"], np.float32)
            if k in teach:
                dup_q += 1
                if not np.array_equal(teach[k], t):
                    print("WARN teacher differs across duplicate queue rows for", k, flush=True)
                continue
            teach[k] = t
    print("teacher rows matched", len(teach), "dup_q", dup_q, flush=True)

    keys = sorted(k for k in lab if k in sets and k in teach)
    miss_sets = [k for k in lab if k not in sets]
    miss_teach = [k for k in lab if k not in teach]
    print("joined", len(keys), "labels without set", len(miss_sets), "labels without teacher", len(miss_teach), flush=True)

    N = len(keys)
    out = {"keys_log": np.array([k[0] for k in keys]), "keys_token": np.array([k[1] for k in keys]),
           "keys_step": np.array([k[2] for k in keys], np.int32), "keys_rank": np.array([k[3] for k in keys], np.int32),
           "proposals": np.stack([sets[k]["traj"] for k in keys]), "prop_yaw": np.stack([sets[k]["yaw"] for k in keys]),
           "teacher": np.stack([teach[k] for k in keys]), "set_navsim_dac": np.stack([sets[k]["nav"] for k in keys]),
           "set_label_version": np.array([sets[k]["lv"] for k in keys], np.int32),
           "set_ckpt_step": np.array([sets[k]["ck"] for k in keys], np.int32)}
    for h in HEADS:
        out[h] = np.stack([lab[k][h] for k in keys])
    meta = {"n_sets": N, "n_labels": len(lab), "n_label_without_set": len(miss_sets), "n_label_without_teacher": len(miss_teach),
            "dup_sets": dup_sets, "dup_queue": dup_q,
            "shapes": {k: list(v.shape) for k, v in out.items()},
            "inputs_sha256": {os.path.basename(p): sha256(p) for p in sorted(glob.glob(LABELS)) + sorted(glob.glob(SETS))},
            "queue_files": len(glob.glob(QUEUE))}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    json.dump(meta, open(os.path.splitext(a.out)[0] + "_meta.json", "w", encoding="utf-8"), indent=1)
    print("wrote", a.out, N, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
