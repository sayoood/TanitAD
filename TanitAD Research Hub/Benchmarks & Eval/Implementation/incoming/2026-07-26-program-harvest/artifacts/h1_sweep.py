"""H1 sweep: find every `separated: false` node in the program's JSON artifacts,
extract effect / half-width / n, and rank by proximity-to-separation.

CPU only. Read-only over the repo.
"""
import json, os, sys, math

ROOT = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SEARCH = [
    os.path.join(ROOT, "TanitAD Research Hub"),
    os.path.join(ROOT, "taniteval"),
    os.path.join(ROOT, "Project Steering"),
    os.path.join(ROOT, "Reviews"),
]

EFFECT_KEYS = ["delta", "mean", "diff", "effect", "point", "delta_m", "d",
               "paired_delta", "value", "lift", "ratio"]
LO_KEYS = ["lo", "ci_lo", "lower", "ci95_lo", "low"]
HI_KEYS = ["hi", "ci_hi", "upper", "ci95_hi", "high"]


def first(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return k, float(d[k])
    return None, None


rows = []
files_scanned = 0
sep_false_total = 0


def walk(node, path, fpath, ctx):
    global sep_false_total
    if isinstance(node, dict):
        # inherit n / estimator context downward
        nctx = dict(ctx)
        for k in ("n_windows", "n_episodes", "n_boot", "estimator", "n_clusters",
                  "n_episode_clusters", "n", "metric", "horizon_k", "K", "arm",
                  "checkpoint", "ckpt", "step"):
            if k in node and not isinstance(node[k], (dict, list)):
                nctx[k] = node[k]
        if node.get("separated") is False:
            sep_false_total += 1
            ek, ev = first(node, EFFECT_KEYS)
            lk, lv = first(node, LO_KEYS)
            hk, hv = first(node, HI_KEYS)
            hw = None
            if lv is not None and hv is not None:
                hw = (hv - lv) / 2.0
            elif isinstance(node.get("ci95"), (int, float)):
                hw = float(node["ci95"])
            prox = None
            if ev is not None and hw and hw > 0:
                prox = abs(ev) / hw
            rows.append({
                "file": os.path.relpath(fpath, ROOT).replace("\\", "/"),
                "json_path": path,
                "effect_key": ek,
                "effect": ev,
                "lo": lv,
                "hi": hv,
                "half_width": hw,
                "proximity": prox,
                "n_windows": nctx.get("n_windows"),
                "n_episodes": nctx.get("n_episodes") or nctx.get("n_episode_clusters")
                              or nctx.get("n_clusters"),
                "estimator": nctx.get("estimator"),
                "metric": nctx.get("metric"),
                "n_boot": nctx.get("n_boot"),
            })
        for k, v in node.items():
            walk(v, path + "." + str(k) if path else str(k), fpath, nctx)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]", fpath, ctx)


for base in SEARCH:
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "__pycache__")]
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            files_scanned += 1
            walk(d, "", fp, {})

print(f"files_scanned={files_scanned}  separated_false_nodes={sep_false_total}  rows={len(rows)}")

# only keep rows that have an interpretable effect+halfwidth
usable = [r for r in rows if r["proximity"] is not None]
print(f"usable(with effect+hw)={len(usable)}")

usable.sort(key=lambda r: -r["proximity"])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1_raw.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"all": rows, "usable_ranked": usable}, fh, indent=1)
print("wrote", out)

print("\n=== TOP 45 by |effect|/half-width (closest to separation) ===")
for r in usable[:45]:
    print(f"{r['proximity']:.3f}  eff={r['effect']:+.4f} hw={r['half_width']:.4f} "
          f"nw={r['n_windows']} ne={r['n_episodes']} est={str(r['estimator'])[:28]}")
    print(f"       {r['file']}")
    print(f"       @ {r['json_path']}")
