"""Compare two parity dumps BITWISE and report per-section."""
import json, sys
import torch
a = torch.load(sys.argv[1], weights_only=False)
b = torch.load(sys.argv[2], weights_only=False)
r = {"arm": a["arm"], "A": a["stack"], "B": b["stack"]}
for sec in ("params", "losses", "grads"):
    da, db = a[sec], b[sec]
    if set(da) != set(db):
        r[sec] = {"VERDICT": "DIFFERENT_KEYS",
                  "only_A": sorted(set(da) - set(db))[:10],
                  "only_B": sorted(set(db) - set(da))[:10]}
        continue
    diff = []
    for k in da:
        x, y = da[k], db[k]
        if x is None and y is None:
            continue
        if (x is None) != (y is None):
            diff.append(k); continue
        if x.shape != y.shape or not torch.equal(x, y):
            diff.append(k)
    r[sec] = {"n": len(da), "n_differing": len(diff),
              "VERDICT": "BITWISE_IDENTICAL" if not diff else "DIFFERS",
              "differing": diff[:12]}
r["loss_A"] = float(a["losses"]["loss"])
r["loss_B"] = float(b["losses"]["loss"])
r["loss_bitwise_equal"] = bool(torch.equal(a["losses"]["loss"],
                                           b["losses"]["loss"]))
r["VERDICT"] = ("BITWISE_IDENTICAL"
                if all(r[s]["VERDICT"] == "BITWISE_IDENTICAL"
                       for s in ("params", "losses", "grads"))
                else "DIFFERS")
print(json.dumps(r, indent=2))
