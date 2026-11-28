"""MM 2026-11-28: exit 0 iff every not-yet-DONE x150 log whose archive is already extracted (fetch_<name>.DONE) has its dump.npz.
Prints 'fetched_pending N dumped M'. Read-only."""
import json, os, sys

X = "F:/safehead_cache/x150"
p = json.load(open(f"{X}/plan/PLAN.json"))
need = []
for a in p["archives"]:
    if os.path.exists(f"{X}/markers/fetch_{a['name']}.DONE"):
        need += [l for l in a["logs"] if not os.path.exists(f"{X}/markers/log_{l}.DONE")]
have = [l for l in need if os.path.exists(f"{X}/dump/{l}.dump.npz")]
print(f"fetched_pending {len(need)} dumped {len(have)}")
sys.exit(0 if len(have) == len(need) else 1)
