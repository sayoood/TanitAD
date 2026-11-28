"""floor_vs_model.py -- descriptive, names nothing (convention in TIMES_NOTE.md before the first run). From the banked dumps of card ADD46 (I10R0, replan interval 10 steps)
and of card ADD52 (I10R0M5, 5 steps), no rule, two model seeds: the model rollout (cl), the rollout of the kinematic prior alone in its own loop (fl) and the model's plan of
the first tick held for the whole window (ol), each against the logged track (g). Per rollout: mean step speed at fixed steps, the share of windows whose last-step speed is
more than 5 m/s above the logged one, the share reaching 59.9 m/s on any step, the mean displacement error over the 30 steps (ADE6, m), the mean speed error over the 30
steps (m/s). Tier T1-kin (kinematic rollout on the model's own actions); never driving performance. Usage: python floor_vs_model.py > FLOOR_VS_MODEL.txt"""
import glob

import numpy as np

W = "/workspace/v8proof/integ10k/t1"
IDX = [0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 25, 29]


def load(arm, k):
    out = {n: [] for n in ("g", "cl", "fl", "ol")}
    files = sorted(glob.glob("%s/%s_s%d/dump_%s_s%d_m10_causal_heads/ep*.npz" % (W, arm, k, arm, k)))
    assert len(files) == 141, (arm, k, len(files))
    for f in files:
        z = np.load(f)
        for n in out:
            out[n].append(np.asarray(z[n], np.float64))
    return {n: np.concatenate(v) for n, v in out.items()}


def speeds(tr):
    p = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1) / 0.2


def line(tag, tr, g):
    s, sg = speeds(tr), speeds(g)
    fin = np.isfinite(tr).all(axis=(1, 2))
    d = s[:, -1] - sg[:, -1]
    ade = np.linalg.norm(tr - g, axis=-1).mean(1)
    print(" %-9s %s | n %d finite %d | last>log+5 %.4f | any>=59.9 %.4f | ADE6 %.3f | speed err %.3f" % (
        tag, " ".join("%5.2f" % np.nanmean(s[:, i]) for i in IDX), len(tr), int(fin.sum()), float((d[fin] > 5).mean()), float((s[fin] >= 59.9).any(1).mean()),
        float(ade[fin].mean()), float(np.abs(s[fin] - sg[fin]).mean())))


print("steps     ", " ".join("%5d" % (i + 1) for i in IDX))
for k in (0, 1):
    a10, a5 = load("I10R0", k), load("I10R0M5", k)
    assert a10["g"].shape == a5["g"].shape == (2399, 30, 2) and np.array_equal(a10["g"], a5["g"])
    print("model seed %d (2,399 windows)" % k)
    line("logged", a10["g"], a10["g"])
    line("ol", a10["ol"], a10["g"])
    line("cl m10", a10["cl"], a10["g"])
    line("fl m10", a10["fl"], a10["g"])
    line("cl m5", a5["cl"], a5["g"])
    line("fl m5", a5["fl"], a5["g"])
    print(" ol equal in both dumps: %s ; fl m10 equal between the seeds is checked below" % np.array_equal(a10["ol"], a5["ol"]))
f0, f1 = load("I10R0", 0)["fl"], load("I10R0", 1)["fl"]
print("fl m10 max abs diff between the two model seeds: %.3e" % float(np.nanmax(np.abs(f0 - f1))))
f0, f1 = load("I10R0M5", 0)["fl"], load("I10R0M5", 1)["fl"]
print("fl m5 max abs diff between the two model seeds: %.3e" % float(np.nanmax(np.abs(f0 - f1))))
print("ZZ FLOOR-VS-MODEL DONE")
