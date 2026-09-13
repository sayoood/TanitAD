"""Mine marking-rich clips for longer SAM3 map videos: SAM3 on the native front-wide frame at 25 / 50 / 75 % of each
candidate clip (candidates = clips whose Alpamayo2 reasoning mentions crosswalks, arrows, hatching or road paint).
A paint instance counts only when >= 60 % of it lies within 15 px of SAM3's road mask (the extractor's first rule).
Output: /home/nvidia/sam3map/data/mining_counts.json
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/nvidia/sam3vendor"); sys.path.insert(0, "/home/nvidia/sam3paint"); sys.path.insert(0, "/home/nvidia/sam3map")
import numpy as np
import av
import cv2
import sam3_smoke as S
from sam3map_extract import instances

PROMPTS = {"crosswalk": 0.5, "zebra crossing": 0.5, "arrow painted on road": 0.45, "text painted on road": 0.45,
           "hatched road marking": 0.45, "lane marking": 0.4}
D = Path("/home/nvidia/sam3map/data")


def frames_at(path, fracs):
    with av.open(str(path)) as cont:
        st = cont.streams.video[0]
        n = st.frames or 600
        want = sorted({min(n - 1, int(n * f)) for f in fracs})
        out = {}
        for i, fr in enumerate(cont.decode(st)):
            if i in want:
                out[i] = fr.to_image()
            if i >= want[-1]:
                break
    return out


def main():
    cands = json.load(open(D / "mining_candidates.json"))
    proc, _ = S.build(conf=0.25)
    res, t0 = {}, time.time()
    for q, (clip, text_score, chunk, tod, country) in enumerate(cands):
        p = D / "frontwide" / f"{clip}.mp4"
        if not p.exists():
            continue
        counts = {k: 0 for k in PROMPTS}
        try:
            imgs = frames_at(p, (0.25, 0.5, 0.75))
        except Exception as e:                                   # a broken video is a skipped candidate, stated
            res[clip] = {"error": repr(e)[:200]}
            continue
        for img in imgs.values():
            state = proc.set_image(img.convert("RGB"))
            road = np.zeros((img.height, img.width), bool)
            for s, m in instances(proc, state, "road", 0.5):
                road |= m
            zone = cv2.dilate(road.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            for pr, t in PROMPTS.items():
                for s, m in instances(proc, state, pr, t):
                    if (m & zone).sum() >= 0.6 * max(int(m.sum()), 1):
                        counts[pr] += 1
        res[clip] = {"text_score": text_score, "chunk": chunk, "tod": tod, "country": country, "counts": counts}
        if q % 20 == 0:
            print(f"  {q}/{len(cands)} {time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(D / "mining_counts.json", "w"))
    json.dump(res, open(D / "mining_counts.json", "w"))
    print(f"ZZMINE-DONEZZ {len(res)} clips {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
