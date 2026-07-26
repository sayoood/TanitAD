# -*- coding: utf-8 -*-
"""Independent sweep for CONTROL-TYPE nodes across every JSON in the repo.
A control node = one where a NULL is the DESIRED verdict."""
import json, os, re, sys

ROOT = r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"

# keywords that mark a CONTROL (null-is-desired). Deliberately broad; filtered by hand after.
CTRL = [
    "shuf", "shuffle", "permut", "leak", "firewall", "blind", "circular",
    "placebo", "sanity", "no_signal", "nosignal", "negative_control", "neg_control",
    "control", "majority", "random", "dead_", "clock", "scramble", "null_", "_null",
    "degener", "trivial", "chance", "mismatch", "swap", "decoy",
]
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

files = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS]
    for f in fn:
        if f.endswith(".json"):
            files.append(os.path.join(dp, f))
print(f"JSON files found: {len(files)}", file=sys.stderr)

hits = []
parsed = 0
failed = 0

def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def walk(o, path, fpath):
    if isinstance(o, dict):
        # is this node a comparison node with a separated flag?
        if "separated" in o or ("ci95" in o or "lo" in o or "ci_lo" in o):
            lowpath = path.lower()
            if any(k in lowpath for k in CTRL):
                hits.append((fpath, path, o))
        for k, v in o.items():
            walk(v, f"{path}.{k}" if path else k, fpath)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f"{path}[{i}]", fpath)

for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        parsed += 1
    except Exception:
        failed += 1
        continue
    walk(d, "", os.path.relpath(fp, ROOT).replace("\\", "/"))

print(f"parsed={parsed} failed={failed} control-ish nodes={len(hits)}", file=sys.stderr)

def extract(o):
    """return (effect, lo, hi, separated) if extractable"""
    eff = None
    for k in ("delta", "effect", "point", "mean", "value", "estimate", "diff", "d"):
        if k in o and num(o[k]): eff = o[k]; break
    lo = hi = None
    for k in ("ci95", "ci", "ci_95", "interval", "paired_ci"):
        if k in o and isinstance(o[k], list) and len(o[k]) == 2 and all(num(x) for x in o[k]):
            lo, hi = o[k]; break
    if lo is None:
        for kl, kh in (("lo","hi"),("ci_lo","ci_hi"),("low","high"),("lower","upper"),("ci_low","ci_high")):
            if kl in o and kh in o and num(o[kl]) and num(o[kh]):
                lo, hi = o[kl], o[kh]; break
    sep = o.get("separated", None)
    if eff is None and lo is not None and hi is not None:
        eff = (lo + hi) / 2.0
    return eff, lo, hi, sep

rows = []
for fp, path, o in hits:
    eff, lo, hi, sep = extract(o)
    est = None
    for k in ("estimator", "method", "ci_method", "se_method"):
        if k in o and isinstance(o[k], str): est = o[k]; break
    ne = o.get("n_episodes") or o.get("n_clusters") or o.get("n_episode_clusters") or o.get("n_eps")
    nw = o.get("n_windows") or o.get("n") or o.get("n_samples")
    hw = None
    if lo is not None and hi is not None: hw = (hi - lo) / 2.0
    prox = abs(eff) / hw if (eff is not None and hw and hw > 0) else None
    rows.append(dict(file=fp, path=path, effect=eff, lo=lo, hi=hi, hw=hw, sep=sep,
                     est=est, n_ep=ne, n_win=nw, prox=prox, keys=sorted(o.keys())[:14]))

json.dump(rows, open("control_sweep_raw.json","w",encoding="utf-8"), indent=1)
print(f"rows written: {len(rows)}", file=sys.stderr)

# Report only the NULLS (separated false or interval spans zero)
nulls = [r for r in rows if (r["sep"] is False) or (r["lo"] is not None and r["hi"] is not None and r["lo"] <= 0 <= r["hi"])]
print(f"NULL control rows: {len(nulls)}", file=sys.stderr)
nulls.sort(key=lambda r: -(r["prox"] or 0))
for r in nulls:
    print(f"{(r['prox'] or 0):.3f} | n_ep={r['n_ep']} n_win={r['n_win']} | {r['effect']} [{r['lo']}, {r['hi']}] | sep={r['sep']} | {r['file'].split('incoming/')[-1]} :: {r['path']}")
