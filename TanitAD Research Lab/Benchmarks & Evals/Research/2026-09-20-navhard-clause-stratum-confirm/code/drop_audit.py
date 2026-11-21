"""Were the dropped windows in my LANDED bias numbers legitimate empties, or swallowed errors?

`bias2.py` reported 251 windows from 260 picked (KEPT) and 246 from 260 (`future`). Those drops
passed through a bare `except Exception` that cannot tell "this window has no valid targets" from
"this window failed to load". That is the same class as the 2,731 phantom "unreadable" navhard
scenes, and the numbers it produced are LANDED (f3fdcc9, c6bab8f).

⛔ No swallow here: every exit is counted by REASON, and an exception is recorded WITH ITS TYPE
rather than folded into the empties. Same draw (same seed, same pools) so the accounting applies
to the landed run and not to a different sample.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # guard, not a habit

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
PER_GROUP = 260
SEED = 20260920


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    groups = collections.defaultdict(list)
    for w in range(len(corp.ds)):
        e = corp.eligibility(w)
        groups["KEPT" if e is None else e].append(w)

    rng = np.random.default_rng(SEED)      # identical draw to bias2.py
    out = {}
    for name in ("KEPT", "future", "agents"):
        pool = groups[name]
        pick = rng.choice(len(pool), size=min(PER_GROUP, len(pool)), replace=False)
        why = collections.Counter()
        for i in pick:
            w = int(pool[int(i)])
            try:
                item = corp.ds[w]
            except Exception as e:                       # noqa: BLE001
                why[f"EXCEPTION:{type(e).__name__}"] += 1
                continue
            try:
                val = item["agent_valid"].numpy().astype(bool)
            except Exception as e:                       # noqa: BLE001
                why[f"EXCEPTION:{type(e).__name__}"] += 1
                continue
            if val.sum() == 0:
                why["empty: no valid targets"] += 1
            else:
                why["counted"] += 1
        out[name] = dict(why)
        tot = sum(why.values())
        exc = sum(v for k, v in why.items() if k.startswith("EXCEPTION"))
        print(f"{name:<8} drawn={tot:<4} counted={why['counted']:<4} "
              f"empty={why['empty: no valid targets']:<4} EXCEPTIONS={exc}")
        for k, v in sorted(why.items()):
            if k.startswith("EXCEPTION"):
                print(f"           ⛔ {k}: {v}")

    total_exc = sum(v for g in out.values() for k, v in g.items() if k.startswith("EXCEPTION"))
    print(f"\nTOTAL EXCEPTIONS ACROSS ALL GROUPS: {total_exc}")
    print("=> landed drop counts are LEGITIMATE EMPTIES" if total_exc == 0
          else "=> ⛔ landed numbers rest on swallowed errors; re-read them")
    pathlib.Path("drop_audit.json").write_text(
        json.dumps({"_what": "reason-by-reason accounting for drops in the landed bias numbers",
                    "groups": out, "total_exceptions": total_exc}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
