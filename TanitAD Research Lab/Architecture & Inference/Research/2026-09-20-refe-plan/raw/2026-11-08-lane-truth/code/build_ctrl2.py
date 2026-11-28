#!/usr/bin/env python3
"""Control queue v2: rows from the HELD-OUT queue (ho/refe_heldout/queue) in the labeller's own read order (sorted filename, FIRST row per key wins,
as onpolicy_relabel_teacher_lane.main does), restricted to keys that have a banked label. Two arms: REAL teacher, PLACEHOLDER teacher (prop 0)."""
import glob, json, lzma, os
BP = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-04-lane-discipline/heldout_teacher_lane_labels.jsonl.xz"
banked = set()
for ln in lzma.open(BP, "rt"):
    d = json.loads(ln)
    banked.add((tuple(d["key"]), d["ckpt_step"]))
W = "F:/scratch/lane_truth_work/run"
rows, seen = [], set()
for fp in sorted(glob.glob("E:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/queue/props_r0_*")):
    for ln in open(fp, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "teacher" not in r or "props" not in r or int(r.get("rank", 0)) != 0:
            continue
        k = ((r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0))), int(r["ckpt_step"]))
        if k in seen:
            continue
        seen.add(k)
        if k in banked:
            rows.append(r)
print("queue rows first-win", len(seen), "with banked label", len(rows))
# 40 keys spread across logs: every k-th row, so that many logs/maps are covered
sel = rows[:: max(1, len(rows) // 40)][:40]
for nm, ph in (("q_ctrl2_real", False), ("q_ctrl2_ph", True)):
    os.makedirs(f"{W}/{nm}", exist_ok=True)
    with open(f"{W}/{nm}/props_r0_010075_0000000001.jsonl", "w", encoding="utf-8") as f:
        for r in sel:
            r2 = dict(r)
            if ph:
                r2["teacher"] = r2["props"][0]
            f.write(json.dumps(r2) + "\n")
print("control rows", len(sel), "logs", len({r["log_name"] for r in sel}))
