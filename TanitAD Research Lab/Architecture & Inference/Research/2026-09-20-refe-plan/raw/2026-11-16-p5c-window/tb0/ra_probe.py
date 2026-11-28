"""MM 2026-11-16, read-only diagnosis of the TB-0 R-A miss: pod control-pass Tc20s vs the dev-box Tc20s build, per row and per component."""
import glob, json, sys
import numpy as np
CTL = "/workspace/data/p5c_label/control/out"
DEV = "/workspace/data/p5c_ship/dev_labels"


def key(r):
    for a in (("log_name", "token", "rank"), ("log", "token", "rank")):
        if all(x in r for x in a):
            return (r[a[0]], r[a[1]], int(r[a[2]]))
    return None


def load(files):
    out, first = {}, None
    for fp in files:
        for ln in open(fp, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            first = first or r
            k = key(r)
            if k is not None and "tc20s" in r and "error" not in r:
                out[k] = r
    return out, first


pod, f1 = load(sorted(glob.glob(CTL + "/labels_s*of15.jsonl")))
dev, f2 = load(sorted(glob.glob(DEV + "/labels_s*of2.jsonl")))
print("pod rows", len(pod), "keys", sorted(f1)[:14])
print("dev rows", len(dev), "keys", sorted(f2)[:14])
common = sorted(set(pod) & set(dev))
d = []
comp = np.zeros(3)
worst = []
for k in common:
    a = np.asarray(pod[k]["tc20s"], np.float64); b = np.asarray(dev[k]["tc20s"], np.float64)
    if a.shape != (20, 3) or b.shape != (20, 3):
        continue
    e = np.abs(a - b)
    d.append(e.max()); comp = np.maximum(comp, e.max(axis=0))
    worst.append((e.max(), k, int(e.argmax() // 3), int(e.argmax() % 3), float(np.abs(b).max())))
d = np.asarray(d)
print("common", len(common), "compared", len(d))
for t in (0.0, 1e-12, 1e-10, 5e-10, 1e-9, 1.2e-9):
    print("rows with max diff > %g: %d" % (t, int((d > t).sum())))
print("quantiles p50 %.3e p99 %.3e p99.9 %.3e max %.3e" % tuple(np.quantile(d, [0.5, 0.99, 0.999, 1.0])))
print("per component max (x, y, yaw):", comp.tolist())
worst.sort(reverse=True)
for w in worst[:6]:
    print("worst %.3e rank %d step %d comp %d |label|max %.3f" % (w[0], w[1][2], w[2], w[3], w[4]))
print("ZZRA_PROBE_END")
