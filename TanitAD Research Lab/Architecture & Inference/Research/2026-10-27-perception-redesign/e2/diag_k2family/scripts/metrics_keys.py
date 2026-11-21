"""Census the keys of a trainer metrics.jsonl: which row kinds exist, which keys mention map / ema / eval. Read-only."""
import json, sys, collections

for p in sys.argv[1:]:
    kinds = collections.Counter()
    keyset = collections.Counter()
    first = {}
    n = 0
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            kinds["<bad>"] += 1
            continue
        n += 1
        kind = r.get("kind") or r.get("type") or r.get("split") or ("eval" if any(k.startswith("eval") for k in r) else "train")
        kinds[str(kind)] += 1
        for k in r:
            keyset[k] += 1
            if k not in first:
                first[k] = (r.get("step"), repr(r[k])[:80])
    print("#####", p, "rows", n, dict(kinds))
    pat = sys.argv[0] and None
    for k, c in sorted(keyset.items()):
        kl = k.lower()
        if any(s in kl for s in ("map", "ema", "eval", "val", "share", "drv", "sdw", "nf_", "swedge", "step", "loss_total", "lr")):
            print("  %-60s n=%5d first@%s %s" % (k, c, first[k][0], first[k][1]))
    print("  total distinct keys:", len(keyset))
