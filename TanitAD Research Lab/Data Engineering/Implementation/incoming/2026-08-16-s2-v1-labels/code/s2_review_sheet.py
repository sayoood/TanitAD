"""S2 v1 review sheet — the per-clip sample the PI judges correctness from.

Deterministic stratified sample over the aug120 labels (the PI's review
loop corpus): every decision class is represented, edge cases included by
name. Writes review/REVIEW_SHEET.md.
"""
import json
import os
from collections import Counter

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PKG, "raw")
LABELS = os.path.join(PKG, "labels")
REV = os.path.join(PKG, "review")
os.makedirs(REV, exist_ok=True)

rows = json.load(open(os.path.join(RAW, "review_rows_aug120.json"),
                      encoding="utf-8"))
by_id = {r["clip_id"]: r for r in rows}
analysis = json.load(open(os.path.join(RAW, "label_analysis.json"),
                          encoding="utf-8"))
excluded = json.load(open(os.path.join(LABELS, "s2_excluded_w120val.json"),
                          encoding="utf-8"))

#: named edge cases the sheet must show (from the gap review)
MUST_SHOW = [
    "8dc5d14d-ab19-43a4-843d-12fdd28173a8",   # turn w/ dyaw=0, dist null
    "079707d3-c6ac-4b41-9b61-1b98b298ecc9",   # the u_turn (was ROUTE_TO)
]


def pick(pred, n, taken):
    out = []
    for r in rows:                                    # rows are cid-sorted
        if len(out) >= n:
            break
        if r["clip_id"] in taken or not pred(r):
            continue
        out.append(r)
        taken.add(r["clip_id"])
    return out


taken = set(MUST_SHOW)
sample = [by_id[c] for c in MUST_SHOW if c in by_id]
sample += pick(lambda r: r["g_str"] in ("TURN_LEFT", "TURN_RIGHT")
               and r["vlm_goal"] in ("turn_left", "turn_right"), 2, taken)
sample += pick(lambda r: r["g_str"] in ("TURN_LEFT", "TURN_RIGHT")
               and r["vlm_goal"] == "follow_main_road", 3, taken)
sample += pick(lambda r: r["remapped_route_to"]
               and r["g_str"] == "TURN_LEFT", 2, taken)
sample += pick(lambda r: r["remapped_route_to"]
               and r["g_str"] == "TURN_RIGHT", 2, taken)
sample += pick(lambda r: r["g_str"] == "NONE_ABSTAIN", 1, taken)
sample += pick(lambda r: r["g_str"] == "STOP_AT" and r["route"] == "follow",
               2, taken)
sample += pick(lambda r: r["g_str"] == "STOP_AT" and r["route"] != "follow",
               2, taken)
sample += pick(lambda r: r["g_str"] == "LANE_TARGET", 4, taken)
sample += pick(lambda r: r["g_str"] == "FOLLOW_MAIN_ROAD"
               and r["route_valid"], 2, taken)
sample += pick(lambda r: r["g_str"] == "FOLLOW_MAIN_ROAD"
               and not r["route_valid"], 1, taken)
sample += pick(lambda r: r["a_str"] == "RESUME_CRUISE", 1, taken)
sample += pick(lambda r: r["a_str"] == "REDUCE_TO", 1, taken)

lt_by_clip = {r["clip"]: r for r in
              analysis["aug120"]["lane_target_audit"]["rows"]}


def fmt_args(d):
    return ", ".join(f"{k}={v}" for k, v in d.items()) or "—"


def check_hint(r):
    g = r["g_str"]
    if g in ("TURN_LEFT", "TURN_RIGHT"):
        side = "left" if g == "TURN_LEFT" else "right"
        return (f"video should show a {side} turn starting ~"
                f"{r['g_args'].get('arg0', '?')} m ahead")
    if g == "STOP_AT":
        return (f"ego should come to a stop ~{r['g_args'].get('arg0', '?')} m "
                "ahead (red light / queue / sign)")
    if g == "LANE_TARGET":
        lt = lt_by_clip.get(r["clip_id"][:8], {})
        return (f"⚠️ VLM did NOT flag this lane change (0/19 corroborated) — "
                f"check a real ~{lt.get('lat_m', '?')} m lateral move vs a "
                "curving road")
    if g == "NONE_ABSTAIN":
        return "abstain: verify neither a turn nor a plain follow fits"
    return "ego should simply follow the corridor"


L: list[str] = []
L.append("# S2 v1 label review sheet — aug120 sample "
         f"(n={len(sample)} of 201)\n")
L.append("Every row: what the geometry measured, what the VLM said, what "
         "the label became, and what the video should show if the label is "
         "right. Labels: `labels/s2_labels_aug120.jsonl`; per-clip Engine A: "
         "`labels/engine_a_aug120.jsonl`; selection is deterministic "
         "(stratified over decision classes, edge cases by name).\n")
L.append("Sample composition: " + ", ".join(
    f"{k} {v}" for k, v in
    Counter(r["g_str"] for r in sample).most_common()) + ".\n")

for i, r in enumerate(sample, 1):
    ea_bits = (f"route `{r['route']}` valid={r['route_valid']} "
               f"dyaw={r['dyaw']} rad dist={None if r['dist_m'] is None else round(r['dist_m'], 1)} m · "
               f"stops={r['stops']} net_dv={r['net_dv']} m/s")
    flags = []
    if r["remapped_route_to"]:
        flags.append("**ROUTE_TO remapped** (G1 gated)")
    if r["g_reason"]:
        flags.append(f"reason: {r['g_reason']}")
    L.append(f"## {i}. `{r['clip_id']}`\n")
    L.append(f"* scene: {r['scenario']}")
    L.append(f"* Engine A (hindsight): {ea_bits}")
    L.append(f"* VLM said: goal `{r['vlm_goal']}` · actions "
             f"{', '.join(r['vlm_actions']) or '—'}")
    L.append(f"* **label: g_str `{r['g_str']}` ({fmt_args(r['g_args'])}) · "
             f"a_str `{r['a_str']}` ({fmt_args(r['a_args'])})**"
             + (" · " + " · ".join(flags) if flags else ""))
    L.append(f"* check: {check_hint(r)}\n")

L.append("---\n\n## The 4 excluded w120val records (item 5)\n")
L.append("Triple-empty (VLM/SAM3/Alpamayo absent, ego_state null); their "
         "fused `NONE_ABSTAIN` was a default-of-absence. Excluded from the "
         "label set with reasons — Engine A geometry is banked for all 4 "
         "and shown here (what a re-fuse would recover):\n")
for e in excluded:
    L.append(f"* `{e['clip_id']}` — geometry route: "
             f"`{e['engine_a_route']}` — {e['reason'][:80]}…")
L.append("")

open(os.path.join(REV, "REVIEW_SHEET.md"), "w", encoding="utf-8") \
    .write("\n".join(L))
print(f"REVIEW_SHEET {len(sample)} clips + {len(excluded)} exclusions")
