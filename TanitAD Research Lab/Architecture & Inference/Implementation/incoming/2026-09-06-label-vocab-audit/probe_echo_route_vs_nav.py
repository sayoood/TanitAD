"""Audit part 3: the ECHO TEST done on the right index spaces.

nav is a 4-way INPUT vocabulary, route a 3-way TARGET vocabulary; comparing
their raw indices is a category error (it reads 0.3158 and means nothing).
The real question: given the nav token, is the route class DETERMINED?

⚠️ SCOPE: this runs on the SYNTHETIC unicycle corpus, so it measures the
FUNCTIONAL relationship between the two labelers on smooth trajectories. It is
NOT the real corpus's joint distribution.

ASCII only. CPU only.
"""
import importlib.util as iu
import sys
from collections import Counter, defaultdict

import torch

STACK = sys.argv[1]
sys.path.insert(0, STACK)
spec = iu.spec_from_file_location("t", f"{STACK}/scripts/refc_v3_train.py")
T = iu.module_from_spec(spec)
_a, sys.argv = sys.argv, ["t"]
try:
    spec.loader.exec_module(T)
except SystemExit:
    pass
sys.argv = _a

v3 = T.v3
from tanitad.refs import refb

NAV = list(refb.NAV_COMMANDS)        # ['follow','left','right','straight']
ROUTE = list(refb.ROUTE_CLASSES)     # ['route_left','route_straight','route_right']

cfg = v3.refc_v3_smoke_config(hier=True)
eps = T._synth_episodes(40, cfg.core, seed=3)
ds = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                 channels=cfg.core.encoder.in_channels)

joint = defaultdict(Counter)
n = 0
for i in range(len(ds)):
    it = ds[i]
    if not bool(it["route_valid"]) or not bool(it["nav_valid"]):
        continue
    joint[NAV[int(it["nav_cmd"])]][ROUTE[int(it["route_target"])]] += 1
    n += 1

print("windows with BOTH nav_valid and route_valid: %d" % n)
print()
print("P(route | nav) on the SYNTHETIC corpus")
print("%-10s %8s   %s" % ("nav", "n", "route distribution"))
det = 0
tot = 0
for nav in NAV:
    c = joint.get(nav)
    if not c:
        print("%-10s %8d   (absent)" % (nav, 0))
        continue
    s = sum(c.values())
    top = c.most_common(1)[0]
    det += top[1]
    tot += s
    dist = ", ".join("%s %d (%.3f)" % (k, v, v / s)
                     for k, v in c.most_common())
    print("%-10s %8d   %s" % (nav, s, dist))
print()
print("PURITY of the best route class per nav token: %d/%d = %.4f"
      % (det, tot, det / tot if tot else float("nan")))
print("  1.0000 would mean route is an EXACT FUNCTION of nav (a pure echo).")
print("  A 3-class uniform prior would sit near 0.3333.")
