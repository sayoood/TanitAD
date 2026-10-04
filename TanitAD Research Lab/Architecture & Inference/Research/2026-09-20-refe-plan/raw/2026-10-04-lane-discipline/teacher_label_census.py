"""Streaming, CPU-only census of the teacher's per-proposal labels in the held-out on-policy sets (pod).
Reports: target keys, OffRoad category counts, how components() maps them, and lane-related keys."""
import collections, glob, json, os, sys
D = "/workspace/data/refe_heldout/sets"
files = sorted(glob.glob(D + "/*"))
keys = collections.Counter(); cat = collections.Counter(); dacv = collections.Counter(); ddcv = collections.Counter()
navdac = collections.Counter(); lanek = collections.Counter(); sets = 0; props = 0; first = None
set_has = collections.Counter()
for f in files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "onpolicy_set":
                continue
            sets += 1
            if first is None:
                first = sorted(r.keys())
            seen = set()
            for t in r.get("targets", []):
                props += 1
                for k in t:
                    keys[k] += 1
                c = t.get("off_road.OffRoad.info")
                if isinstance(c, (int, float)) and c == c:
                    cat[int(round(c))] += 1; seen.add(("cat", int(round(c))))
                dacv[t.get("dac.violation")] += 1
                ddcv[t.get("ddc.violation")] += 1
                for k, v in t.items():
                    if ("navsim" in k.lower() or "dac" in k.lower()) and k not in ("dac.violation",):
                        navdac[(k, v if not isinstance(v, float) else round(v, 3))] += 1
                    if any(s in k for s in ("CrossLane", "lane_change", "CenterLine", "deviation", "solid")):
                        lanek[k] += 1
            for s in seen:
                set_has[s] += 1
print(json.dumps({"files": len(files), "sets": sets, "proposals": props, "set_keys": first,
                  "offroad_category_counts": dict(sorted(cat.items())),
                  "sets_with_category": {str(k[1]): v for k, v in sorted(set_has.items())},
                  "dac.violation": {str(k): v for k, v in dacv.items()}, "ddc.violation": {str(k): v for k, v in ddcv.items()},
                  "dac_like_keys": {f"{k}={v}": n for (k, v), n in sorted(navdac.items(), key=lambda x: -x[1])[:30]},
                  "lane_keys": dict(lanek), "n_target_keys": len(keys),
                  "target_keys": sorted(keys)}, indent=1))
