#!/usr/bin/env python3
"""FIXED stability census: distinguish FIELD-ABSENT from genuine NaN/Inf.
Also dump the top gnorm rows for the divergent arms."""
import json, os, math

ROOT = "/home/nvidia/v7tiny"
ARMS30K = ["postrain30k", "postrain30k_freeze", "postrain30k_seed1", "splitp30k",
           "k60clip05p30k", "k8clip05p30k", "k60p30k.DIVERGED-clip1.0.DO-NOT-CENSUS",
           "emao14_30k", "emao14_30k_tauramp", "o14fut30k", "champ30k", "omega30k",
           "k4_30k", "o1ctrl30k", "ok8p30k", "rdw8p30k", "ro128p30k", "o11p30k", "scale1"]

print("%-38s %5s %5s %6s %6s %8s %11s %8s %6s %6s" % (
    "arm", "rows", "gnrw", "absnt", "NaN", "gn_med", "gn_max", "gn@", "sp10x", "splate"))
detail = {}
for name in ARMS30K:
    p = os.path.join(ROOT, name, "train_log.jsonl")
    if not os.path.exists(p):
        print("%-38s MISSING" % name); continue
    total = 0; absent = 0; nan = 0
    gs = []
    for line in open(p, "r", errors="replace"):
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "step" not in r: continue
        total += 1
        if "gnorm" not in r:
            absent += 1; continue
        g = r["gnorm"]
        try: gf = float(g)
        except Exception:
            nan += 1; continue
        if math.isnan(gf) or math.isinf(gf):
            nan += 1; continue
        gs.append((r["step"], gf))
    if not gs:
        print("%-38s NO_GNORM" % name); continue
    gv = sorted(g for _, g in gs)
    med = gv[len(gv) // 2]
    mx = max(gs, key=lambda t: t[1])
    thr = 10.0 * med
    sp = [(s, g) for s, g in gs if g > thr]
    maxstep = max(s for s, _ in gs)
    late = sum(1 for s, g in sp if s >= maxstep / 2.0)
    print("%-38s %5d %5d %6d %6d %8.3f %11.4g %8d %6d %6d" % (
        name, total, len(gs), absent, nan, med, mx[1], mx[0], len(sp), late))
    detail[name] = sorted(sp, key=lambda t: -t[1])[:8]

print()
print("TOP GNORM SPIKES (step, gnorm) for the unstable arms:")
for k in ["k60clip05p30k", "k60p30k.DIVERGED-clip1.0.DO-NOT-CENSUS", "o1ctrl30k", "k8clip05p30k"]:
    if k in detail:
        print("  %-40s %s" % (k, detail[k]))
