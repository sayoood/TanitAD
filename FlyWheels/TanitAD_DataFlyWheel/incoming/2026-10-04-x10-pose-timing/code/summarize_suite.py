"""Compare the affected-test regression run on the OVERLAID tree (tip + fix) with the same run on the PURE tip tree.

    python summarize_suite.py <suite_overlay.log> <suite_base.log> <out.json>

A failure that exists in BOTH trees is pre-existing (it is the tip's, not this change's).  A failure only in the overlay
tree is a regression caused by the change.  The final pytest line gives the counts; the `-rfEs` lines give the ids.
"""
import json
import re
import sys

ov, ba, out = sys.argv[1:4]


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\r", "")
    fails = set(re.findall(r"^(?:FAILED|ERROR) (\S+)", txt, flags=re.M))
    last = [l for l in txt.splitlines() if re.search(r"\d+ (passed|failed)", l)]
    m = re.search(r"(\d+) failed", last[-1]) if last else None
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)", last[-1])} if last else {}
    exit_ = re.search(r"^EXIT=(\d+)", txt, flags=re.M)
    return {"final_line": last[-1] if last else None, "counts": counts, "fail_ids": sorted(fails), "exit": exit_.group(1) if exit_ else None}


o, b = parse(ov), parse(ba)
res = {"overlay_tip_plus_fix": o, "pure_tip": b,
       "failures_only_in_overlay (REGRESSIONS)": sorted(set(o["fail_ids"]) - set(b["fail_ids"])),
       "failures_only_in_pure_tip": sorted(set(b["fail_ids"]) - set(o["fail_ids"])),
       "failures_in_both (pre-existing)": sorted(set(o["fail_ids"]) & set(b["fail_ids"]))}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if k.startswith("failures") is False else v) for k, v in res.items() if k not in ("overlay_tip_plus_fix", "pure_tip")}, indent=1)[:3000])
print("overlay:", o["final_line"]); print("base   :", b["final_line"])
