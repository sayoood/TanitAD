"""Param census for a WiSE-style probe: top-level module prefixes of a run's tail weights vs its init checkpoint (CPU only).
usage: python -I param_census.py <tail.pt> <init.pt> <depth>"""
import collections
import sys

import torch


def sd(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "model_state_dict", "ema", "ema_state_dict"):
        if isinstance(o, dict) and k in o and isinstance(o[k], dict):
            return o[k], k
    return o, "<top>"


t, tk = sd(sys.argv[1])
i, ik = sd(sys.argv[2])
dep = int(sys.argv[3])
print("tail key", tk, len(t), "| init key", ik, len(i))
rows = collections.OrderedDict()
for n, v in t.items():
    if not hasattr(v, "shape"):
        continue
    p = ".".join(n.split(".")[:dep])
    r = rows.setdefault(p, [0, 0, 0, 0, 0])
    r[0] += 1
    r[1] += v.numel()
    if n in i and hasattr(i[n], "shape") and tuple(i[n].shape) == tuple(v.shape):
        r[2] += 1
        if torch.equal(i[n].float(), v.float()):
            r[3] += 1
    else:
        r[4] += 1
print("%-48s %6s %11s %8s %8s %6s" % ("prefix", "n", "numel", "in_init", "eq_init", "new"))
for p, r in rows.items():
    print("%-48s %6d %11d %8d %8d %6d" % (p[:48], *r))
