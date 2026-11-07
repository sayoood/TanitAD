#!/usr/bin/env python3
"""YAXD token subsets for the SMOKE (mechanics only, never decision data): whole A5 logs. smoke = the 3 logs with the most CRUISE tokens among logs of <= 40 tokens; red = the
smallest A5 log with >= 1 CRUISE token. Writes <out>/tokens_smoke.json, tokens_red.json (W3 format: rule, tokens, token_log)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import axd_common as A  # noqa: E402
import ax_analyze as AA  # noqa: E402
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
a5 = A.C.jload(A.C.A5_TOKENS); toks = sorted(a5["tokens"]); tl = a5["token_log"]
cr = AA.cruise_inputs(toks)[-1]
by = defaultdict(list); nc = Counter()
for t, c in zip(toks, cr):
    by[tl[t]].append(t); nc[tl[t]] += int(c)
small = sorted([l for l in by if len(by[l]) <= 40 and nc[l] > 0], key=lambda l: (-nc[l], len(by[l]), l))
smoke = small[:3]; red = sorted([l for l in by if nc[l] > 0], key=lambda l: (len(by[l]), l))[0]
for name, logs in (("smoke", smoke), ("red", [red])):
    ts = sorted(t for l in logs for t in by[l])
    A.C.jdump({"rule": a5.get("rule"), "tokens": ts, "token_log": {t: tl[t] for t in ts}, "what": f"YAXD {name} subset: whole A5 logs {logs}"}, out / f"tokens_{name}.json")
    print(name, len(ts), "tokens", sum(nc[l] for l in logs), "cruise", logs)
