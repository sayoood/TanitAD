import json
import sys

n = json.load(open(sys.argv[1]))
o = json.load(open(sys.argv[2]))
for axis in ("lat", "lon"):
    print("==", axis, "| pod2 blocks:", list(n[axis].keys()))
    for tag, d in (("pod3", o), ("pod2", n)):
        blk = d[axis]
        de = blk.get("deltas") or blk.get("paired_deltas") or {}
        for k, v in de.items():
            if isinstance(v, dict) and "mean" in v:
                m, lo, hi = v["mean"], v["lo"], v["hi"]
                sep = v.get("separated", lo * hi > 0)
                print("   %s %-28s %+0.4f [%+0.4f,%+0.4f] sep=%s"
                      % (tag, k, m, lo, hi, sep))
    for key in ("refusal", "verdict", "refusal_verdict", "bars", "skill_bars"):
        if key in n[axis]:
            print("   pod2", key, ":", json.dumps(n[axis][key])[:400])
            print("   pod3", key, ":", json.dumps(o[axis].get(key))[:400])
