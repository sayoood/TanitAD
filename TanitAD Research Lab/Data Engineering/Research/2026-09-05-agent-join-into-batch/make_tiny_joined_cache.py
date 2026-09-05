#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a TINY epcache out of REAL, join-covered episodes so the CPU smoke rig
can carry REAL obstacle.offline labels.

WHY THIS AND NOT `--synth-episodes`
-----------------------------------
`refc_v3_smoke_config` wants 1-channel 64 px episodes; the real epcache is
9-channel 256 px.  The refcv5 smoke therefore ran on the SYNTHETIC corpus,
which by construction has no join -- which is exactly why `--agents head`
could not be exercised.  This script keeps everything the join is keyed on
(`episode_id`) and everything the labels are geometrically tied to (`poses`,
frame ORDER and COUNT) and reduces ONLY the pixels.

WHAT IS AND IS NOT CLAIMED
--------------------------
The frames are a genuine reduction of the real frames (latest RGB triplet ->
grey -> 64 px), NOT fabricated content.  But a 64 px grey frame cannot support
real monocular 3D detection, so this rig proves the LOSS IS REAL AND MOVES --
a WIRING claim.  It is not, and must never be quoted as, a detection result.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack\scripts")

SRC = (r"C:\Users\Admin\tanitad-data\physicalai\_epcache"
       r"\physicalai-train-14231cd29c74")
META = (r"C:\Users\Admin\tanitad-data\joins\joins"
        r"\train2400_agents.jsonl.xz.meta.json")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--want", type=int, default=16)
ap.add_argument("--scan", type=int, default=200)
ap.add_argument("--px", type=int, default=64)
a = ap.parse_args()

from train_p8_occupancy import (episode_uid_of_clip,          # noqa: E402
                                legacy_episode_id_of_clip)

meta = json.load(open(META, encoding="utf-8"))
clips = [c["clip_id"] for c in meta["per_clip"]]
uids = {episode_uid_of_clip(c) for c in clips}
legs = {}
for c in clips:
    legs.setdefault(legacy_episode_id_of_clip(c), []).append(c)
# only UNAMBIGUOUS legacy ids -- a colliding key would join another clip's
# agents and no downstream metric could attribute the corruption.
legs_ok = {k for k, v in legs.items() if len(v) == 1}

outdir = os.path.join(a.out, "physicalai-train-tiny64")
os.makedirs(outdir, exist_ok=True)
files = sorted(f for f in os.listdir(SRC) if f.startswith("ep_"))[:a.scan]
kept = []
for f in files:
    if len(kept) >= a.want:
        break
    d = torch.load(os.path.join(SRC, f), map_location="cpu",
                   weights_only=False)
    eid = int(d["episode_id"])
    if eid not in uids and eid not in legs_ok:
        continue
    fr = d["frames_u8"]                                # [T, 9, 256, 256] uint8
    x = fr[:, -3:].float().mean(1, keepdim=True)       # latest RGB -> grey
    x = F.interpolate(x, size=(a.px, a.px), mode="area")
    d2 = dict(d)
    d2["frames_u8"] = x.round().clamp(0, 255).to(torch.uint8)
    torch.save(d2, os.path.join(outdir, "ep_%05d.pt" % len(kept)))
    kept.append((f, eid, int(fr.shape[0])))
    print("  kept %s id=%d T=%d -> %s" % (f, eid, fr.shape[0],
                                          tuple(d2["frames_u8"].shape)))

open(os.path.join(outdir, "DONE"), "w").write("tiny64\n")
print("wrote %d episodes to %s (scanned %d)" % (len(kept), outdir, len(files)))
# CONTROL: a bank whose content was never asserted is the all-zero-floor trap.
d = torch.load(os.path.join(outdir, "ep_00000.pt"), map_location="cpu",
               weights_only=False)
fr = d["frames_u8"]
print("CONTENT CONTROL: frames dtype=%s shape=%s mean=%.2f nonzero_frac=%.4f"
      % (fr.dtype, tuple(fr.shape), fr.float().mean(),
         (fr != 0).float().mean()))
assert fr.float().mean() > 1.0, "all-zero frames -- the poisoned-memmap trap"
print("episode ids:", [e for _, e, _ in kept])
