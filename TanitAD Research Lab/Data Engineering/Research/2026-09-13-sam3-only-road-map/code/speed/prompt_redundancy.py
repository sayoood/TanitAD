"""Screen for the next lever after precision: the NUMBER of prompts. At the approved operating point every grounding stage scales
with the 19 prompts (grounding ~80 % of SAM3 time), and several are synonyms. This screen does not approve anything -- pruning a
prompt changes the map by design and needs its own map-level arm -- it ranks candidates.
Per front frame (all 96 frames of both clips), the approved fast path (fp16 fusion encoder, fp32 decoder / mask head, 19 prompts
batched) returns each prompt's accepted instances exactly as the extractor sees them; per prompt: instances, accepted pixels, and
UNIQUE pixels = accepted pixels no other prompt of the same family covers in that frame (the extractor unions road, non-drivable,
line, crosswalk, gate and symbol prompts within the family before its rules). A prompt whose unique share is ~0 on every frame of
both clips changes the family union by ~nothing -- the candidate list for a pruning arm.
Usage: prompt_redundancy.py <c8> [<c8> ...]"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("HALF", "fp16enc"); os.environ.setdefault("BATCH", "19")
sys.path.insert(0, "/home/nvidia/sam3map"); sys.path.insert(0, "/home/nvidia/sam3map/eval")
import numpy as np
import torch
import sam3map_front_fast as FF

proc, _ = FF.E.S.build(conf=0.25)
fast = FF.FastSAM3(proc, FF.FLAGS)
FAM = {p: fam for fam, table in FF.E.PROMPTS.items() for p in table}
FAM.update({"crosswalk stripe": "stripe", "white stripe on road": "stripe"})
PROMPTS = [p for p, _ in FF.CLS_PT + FF.STRIPE_PT]
tot = {p: {"instances": 0, "px": 0, "unique_px": 0, "frames_with": 0, "frames_unique_gt_1pct": 0} for p in PROMPTS}
fam_px = {}
t0 = time.time(); nfr = 0
for c8 in sys.argv[1:]:
    sd = FF.ROOT / f"seq_{c8}"; toks = sorted(p.name for p in sd.iterdir() if p.is_dir())
    poses = json.loads((sd / "poses.json").read_text())
    for j in range(len(toks)):
        fi = FF.frame_inputs(c8, j, toks, sd, poses)
        st = fast.encode(fi["img"])
        inst = fast.instances(st, FF.CLS_PT + FF.STRIPE_PT)
        H, W = fi["img"].height, fi["img"].width
        unions = {}
        for p in PROMPTS:
            u = np.zeros((H // 2, W // 2), bool)
            for s, m in inst[p]:
                u |= m[::2, ::2]
            unions[p] = u
        for p in PROMPTS:
            others = [unions[q] for q in PROMPTS if q != p and FAM[q] == FAM[p]]
            cov = np.logical_or.reduce(others) if others else np.zeros_like(unions[p])
            px = int(unions[p].sum()); uq = int((unions[p] & ~cov).sum())
            t = tot[p]; t["instances"] += len(inst[p]); t["px"] += px; t["unique_px"] += uq
            t["frames_with"] += px > 0; t["frames_unique_gt_1pct"] += (px > 0 and uq > 0.01 * px)
            fam_px[FAM[p]] = fam_px.get(FAM[p], 0)
        for fam in set(FAM.values()):
            fam_px[fam] += int(np.logical_or.reduce([unions[q] for q in PROMPTS if FAM[q] == fam]).sum())
        nfr += 1
        if nfr % 48 == 0:
            print(f"  {nfr} frames {time.time() - t0:.0f}s", flush=True)
rows = []
print(f"\n{'prompt':28s} {'family':10s} {'inst':>6s} {'px (960x540)':>13s} {'unique':>8s} {'share of family':>15s} {'frames>1% unique':>16s}")
for p in PROMPTS:
    t = tot[p]; fam = FAM[p]
    r = {"prompt": p, "family": fam, **t, "unique_share_of_prompt": t["unique_px"] / max(t["px"], 1), "unique_share_of_family_union": t["unique_px"] / max(fam_px[fam], 1)}
    rows.append(r)
    print(f"{p:28s} {fam:10s} {t['instances']:6d} {t['px']:13d} {r['unique_share_of_prompt']:8.4f} {r['unique_share_of_family_union']:15.4f} {t['frames_unique_gt_1pct']:>9d}/{nfr}")
Path("/home/nvidia/sam3map/prompt_redundancy.json").write_text(json.dumps({"frames": nfr, "family_union_px": fam_px, "rows": rows}, indent=1), encoding="utf-8")
print("ZZREDUND-DONEZZ")
