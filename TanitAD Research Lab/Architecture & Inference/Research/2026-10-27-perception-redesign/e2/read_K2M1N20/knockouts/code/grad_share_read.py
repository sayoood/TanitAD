"""Thor-side, read-only: K0 vs K2M1N20 trunk gradient conflict (cd_*) and grad-share terms from metrics.jsonl.
Prints, per arm, mean/median over all logged steps and over steps >= 3000 for the cd_* cosines and any *grad_share* / *gshare* keys,
plus the conflict detector's own term names from conflict_provenance."""
import json
import statistics as st
import sys

R = "/home/nvidia/r8ns/r9e2/runs/r9e2-"


def walk(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{p}.{k}" if p else k)
    else:
        yield p, o


for arm in sys.argv[1].split(","):
    rows = [json.loads(l) for l in open(R + arm + "/metrics.jsonl", encoding="utf-8") if l.strip()]
    prov = next((r["conflict_provenance"] for r in rows if "conflict_provenance" in r), {})
    print(f"== {arm}: {len(rows)} rows; conflict terms: " + json.dumps({k: v for k, v in prov.items() if k not in ("groups", "config")})[:400])
    keys = {}
    for r in rows:
        s = r.get("step")
        for k, v in walk(r):
            kl = k.lower()
            if (kl.startswith("cd_") or "grad_share" in kl or "gshare" in kl or kl.startswith("gs_")) and isinstance(v, (int, float)) and v == v:
                keys.setdefault(k, []).append((s, float(v)))
    for k, lst in sorted(keys.items()):
        a = [v for _, v in lst]
        late = [v for s, v in lst if s is not None and s >= 3000]
        print(f"  {k}: n={len(a)} mean {st.mean(a):+.4f} med {st.median(a):+.4f} | step>=3000 n={len(late)} mean {st.mean(late) if late else float('nan'):+.4f}")
