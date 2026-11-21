#!/usr/bin/env python3
"""Render the T-B mask onto real frames BEFORE the arm is rolled.

⛔ WHY THIS EXISTS. `RETR-2026-09-13-SAM3MAP-A2-GHOST-GROUND`: a display mask built from
the tracked box's own z blanked the ROAD in front of every vehicle, and nothing in the
number said so -- the PI saw it in a picture. So the mask is LOOKED AT, on real pixels,
before a single GPU hour is spent on it.

Writes `<out>/mask_<sha12>_f<raw>.png` (current sub-frame, original | lead | random) and
prints the mask areas. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.environ.get("D3_REPO", r"C:\Users\Admin\refcv5cmp\repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leadmask as LM  # noqa: E402

EPS = r"C:\Users\Admin\refcv5cmp\data\eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clips", type=int, default=4)
    ap.add_argument("--per-clip", type=int, default=1)
    ap.add_argument("--gap-max-m", type=float, default=30.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from tanitad.data.v2_dataset import build_v2_providers, load_or_build_manifest
    man = load_or_build_manifest(EPS, verbose=False)
    clip_ids = [str(c) for c in man["clip_id"]]
    n_stack = int(man["n_stack"][0])
    keep = set(clip_ids)
    agents = LM.load_agents(
        r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz", keep)
    agents.pop("_stats")
    leads = LM.load_leads(r"C:\Users\Admin\refcv5cmp\data\b1_eval_lead_block.npz", keep)
    proj = LM.Projector(r"C:\Users\Admin\refcv5v2_final\extrinsics141.json")
    eps = build_v2_providers([EPS], lru_size=2, verbose=False)

    rep = []
    done = 0
    for ci, cid in enumerate(clip_ids):
        if done >= a.n_clips:
            break
        cand = sorted([f for (c, f), v in leads.items()
                       if c == cid and v[0] and v[1] and v[2] == v[2]
                       and v[2] <= a.gap_max_m])
        if not cand:
            continue
        ep = eps[ci]
        T = int(ep.frames.shape[0])
        picked = 0
        for raw_origin in cand[::max(len(cand) // 8, 1)]:
            if picked >= a.per_clip:
                break
            t = raw_origin - (n_stack - 1) - 7          # window start, W = 8
            if t < 0 or t + 8 > T:
                continue
            fr = ep.frames[t:t + 8]                     # [8, 9, 256, 640] u8
            outs = {}
            for mode in ("lead", "rand"):
                mk = LM.LeadMasker(mode, agents, leads, proj, n_stack=n_stack,
                                   gap_max_m=a.gap_max_m, seed=0)
                outs[mode] = mk.mask_window(cid, int(t), fr)
                outs[mode + "_stats"] = mk.summary()
            if outs["lead_stats"]["subframes_masked"] == 0:
                continue
            cur = lambda x: x[7, 6:9].permute(1, 2, 0).numpy()      # noqa: E731
            strip = np.concatenate([cur(fr), cur(outs["lead"]), cur(outs["rand"])], 0)
            sh = LM.sha12(cid)
            p = os.path.join(a.out, f"mask_{sh}_f{raw_origin}.png")
            try:
                import torchvision.io as tvio
                tvio.write_png(torch.from_numpy(strip).permute(2, 0, 1).contiguous(), p)
            except Exception as e:                                   # pragma: no cover
                print("[png] failed:", e)
            changed_lead = int((fr != outs["lead"]).sum())
            changed_rand = int((fr != outs["rand"]).sum())
            rep.append({"sha12": sh, "raw_origin": int(raw_origin),
                        "gap0_m": round(leads[(cid, raw_origin)][2], 2),
                        "px_changed_lead": changed_lead,
                        "px_changed_rand": changed_rand,
                        "area_lead_mean": outs["lead_stats"]["area_lead_px_mean"],
                        "area_rand_mean": outs["rand_stats"]["area_rand_px_mean"],
                        "subframes_masked": outs["lead_stats"]["subframes_masked"],
                        "png": os.path.basename(p)})
            picked += 1
            done += 1
    print(json.dumps(rep, indent=1))
    with open(os.path.join(a.out, "preview.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
