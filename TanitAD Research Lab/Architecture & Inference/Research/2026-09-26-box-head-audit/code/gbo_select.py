#!/usr/bin/env python3
"""gbo_select.py -- the FROZEN 16-frame TRAIN set for G-BOX-OVERFIT, chosen from LABELS ONLY.

Input: vis_zbuf's per-box rows over the bha_dump TRAIN set (vis_train_boxes.json): 256 windows of 64 TRAIN clips
(the dump's own label-only rule: train manifest sorted by sha12, every 68th clip, 4 labelled windows each). Each row
carries only GT-derived fields (class, range, the trainer-target flag, and the camera z-buffer visibility of the GT
cuboid). NO model output is read.

Rule (literal):
  1. VIS-1 label per row (vis_rules.label): POSITIVE / IGNORE / DROPPED.
  2. Eligible window: 3 <= #POSITIVE <= 15.
  3. Per clip keep ONE eligible window: the most positives, ties by the smallest t.
  4. Class coverage, greedy over clips in sha12 order: until at least 3 chosen frames contain a POSITIVE of each
     group [person], [rider], [large = heavy_truck | bus | trailer | other_vehicle], [automobile] (in that order),
     add the first unchosen clip whose kept window has one; a group with fewer than 3 available is taken whole.
  5. Fill to 16 frames in sha12 order.
Output: gbo_frameset.json (sha12, t, per-frame VIS-1 counts by class) with its own md5 printed.
"""
import collections
import hashlib
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vis_rules import label  # noqa: E402

LARGE = {"heavy_truck", "bus", "trailer", "other_vehicle"}
GROUPS = (("person", {"person"}), ("rider", {"rider"}), ("large", LARGE), ("automobile", {"automobile"}))


def main():
    src, out = sys.argv[1], sys.argv[2]
    rows = [r for r in json.load(open(src)) if r["target"]]
    win = collections.defaultdict(list)
    for r in rows:
        win[(r["sha12"], int(r["w"]))].append(r)
    per_clip = {}
    for (s12, t), rs in win.items():
        pos = [r for r in rs if label(r) == "positive"]
        if not (3 <= len(pos) <= 15):
            continue
        cur = per_clip.get(s12)
        key = (len(pos), -t)
        if cur is None or key > cur[0]:
            per_clip[s12] = (key, t, rs)
    clips = sorted(per_clip)
    chosen = []

    def has(s12, grp):
        return any(label(r) == "positive" and r["cls"] in grp for r in per_clip[s12][2])
    for name, grp in GROUPS:
        avail = [c for c in clips if has(c, grp)]
        need = min(3, len(avail))
        for c in avail:
            if sum(1 for x in chosen if has(x, grp)) >= need:
                break
            if c not in chosen and len(chosen) < 16:
                chosen.append(c)
    for c in clips:
        if len(chosen) >= 16:
            break
        if c not in chosen:
            chosen.append(c)
    frames = []
    for c in sorted(chosen):
        _k, t, rs = per_clip[c]
        lab = collections.Counter(label(r) for r in rs)
        byc = collections.defaultdict(collections.Counter)
        for r in rs:
            byc[r["cls"]][label(r)] += 1
        frames.append({"sha12": c, "t": int(t), "positive": lab["positive"], "ignore": lab["ignore"],
                       "dropped": lab["dropped"], "by_class": {k: dict(v) for k, v in sorted(byc.items())}})
    tot = collections.Counter()
    for f in frames:
        for k, v in f["by_class"].items():
            tot[k] += v.get("positive", 0)
    res = {"rule": __doc__.split("Rule (literal):")[1].split("Output:")[0].strip(),
           "source": src.split("/")[-1], "source_md5": hashlib.md5(open(src, "rb").read()).hexdigest(),
           "split": "TRAIN (bha_dump train set: refcv6-b1-416x1024-train)", "n_eligible_clips": len(clips),
           "n_frames": len(frames), "frames": frames, "positives_by_class_total": dict(tot),
           "positives_total": sum(f["positive"] for f in frames), "ignores_total": sum(f["ignore"] for f in frames),
           "t_convention": "window start t; the NOW frame is t + W - 1 = t + 7 (V3Dataset); the trainer's GT block is "
                           "V3Dataset._agent_item(ep, t + 7)"}
    s = json.dumps(res, indent=1)
    open(out, "w", encoding="utf-8").write(s)
    print(s[:3000])
    print("md5", hashlib.md5(s.encode()).hexdigest())


if __name__ == "__main__":
    main()
