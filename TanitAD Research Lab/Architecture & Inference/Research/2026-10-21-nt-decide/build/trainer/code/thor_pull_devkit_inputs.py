#!/usr/bin/env python3
"""Filter the r8ns devkit export (a1_inputs.json.gz: W3's per-token record from SceneLoader.get_agent_input_from_token,
navsim v1.1 @ 3e8291b) down to a token list, for the K1 / K2 controls' DEVKIT reference (read-only on the export)."""
import gzip, json, sys
src, toks, out = sys.argv[1], sys.argv[2], sys.argv[3]
want = {l.strip() for l in open(toks) if l.strip()}
d = json.load(gzip.open(src, "rt"))
recs = (d.get("tokens") or d.get("records") or d) if isinstance(d, dict) else d
got = {}
if isinstance(recs, dict):
    for t in want:
        if t in recs:
            got[t] = recs[t]
else:
    for r in recs:
        if str(r.get("token")) in want:
            got[str(r["token"])] = r
json.dump({"source": src, "n_want": len(want), "n_got": len(got),
           "top_keys": (sorted(d.keys())[:20] if isinstance(d, dict) else "list"), "records": got},
          open(out, "w"))
print(len(want), len(got))
