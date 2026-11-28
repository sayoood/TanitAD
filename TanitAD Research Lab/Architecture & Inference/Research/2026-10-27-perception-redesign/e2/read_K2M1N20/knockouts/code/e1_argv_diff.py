"""Thor-side: argv diff of E1 rig arms (and E2 K0/K2) vs E1 C0, from each run's config.json "argv" list. Read-only."""
import json
import sys

R1 = "/home/nvidia/r8ns/r9e1/runs/r9e1-"
R2 = "/home/nvidia/r8ns/r9e2/runs/r9e2-"


def parse(path):
    a = json.load(open(path, encoding="utf-8"))["argv"]
    out, i = {}, 0
    while i < len(a):
        t = a[i]
        if t.startswith("--"):
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                out[t] = a[i + 1]; i += 2
            else:
                out[t] = True; i += 1
        else:
            out.setdefault("_pos", []).append(t); i += 1
    return out


IGN = {"--out", "--run-dir", "--out-dir", "--seed", "--tag", "--run-name", "--log", "--save-dir"}
base = parse(R1 + "C0/config.json")
arms = [(x, R1 + x + "/config.json") for x in sys.argv[1].split(",")] + [(x, R2 + x + "/config.json") for x in sys.argv[2].split(",")]
for name, p in arms:
    d = parse(p)
    diff = []
    for k in sorted(set(base) | set(d)):
        if k in IGN or k == "_pos":
            continue
        if base.get(k) != d.get(k):
            diff.append(f"{k}: {base.get(k)!r} -> {d.get(k)!r}")
    print(f"== {name} vs C0 ({len(diff)}): " + " | ".join(diff))
