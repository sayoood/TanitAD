#!/usr/bin/env python3
"""Differential check of ``v8f_dagger.match_route`` against the PROBE's prototype (``raw/2026-10-26-dagger-draft/probe/probe_dagger_cost.py::match_route``) on random polylines (stops, creeps, turns, loops).
The two implement the same PREREG section 2 except for ONE line: the probe ties every row with ``|arc - s*| <= min|arc - s*| + ARC_TIE_M`` (a band around the nearest DISTANCE, which can hold two rows 2 m apart
that straddle s*); the PREREG says rows "within 0.05 m of that nearest arc" (``|arc - arc_nearest| <= 0.05``), which is what ``v8f_dagger`` implements.  Usage: diff_vs_probe.py [--n N] [--seed S]"""
import argparse
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 6))
sys.path.insert(0, os.path.join(REPO, "stack"))
import numpy as np

from tanitad.data import v8f_dagger as G

PROBE = os.path.join(os.path.dirname(HERE), "2026-10-26-dagger-draft", "probe", "probe_dagger_cost.py")
spec = importlib.util.spec_from_file_location("probe_dagger_cost", PROBE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
# the SAME function with its one tie-set line replaced by the probe's: it must reproduce the probe on every case, which proves the tie-set line is the ONLY difference
import inspect
src = inspect.getsource(G.match_route)
mine = "    tied = np.nonzero(np.abs(arc - arc[i0]) <= ARC_TIE_M)[0]"
assert src.count(mine) == 1, "the tie-set line moved: update this check"
probe_band = "    da = np.abs(arc - s_star)" + chr(10) + "    tied = np.nonzero(da <= da.min() + ARC_TIE_M)[0]"
src = src.replace(mine, probe_band).replace("def match_route(", "def match_route_probe_band(")
ns = dict(vars(G))
exec(src, ns)
band = ns["match_route_probe_band"]
n_same = n_diff = n_reason_diff = n_band_eq_probe = 0
for _ in range(a.n):
    T = int(rng.integers(50, 130))
    v = np.clip(8 + np.cumsum(rng.normal(0, 1.2, T)), 0.0, 25.0)
    v[rng.random(T) < 0.1] = 0.0                                                 # stops
    if rng.random() < 0.3:
        v[int(rng.integers(0, T - 15)):][:int(rng.integers(3, 15))] = 0.0       # a long wait
    yaw = np.cumsum(rng.normal(0, 0.05, T)) + (rng.random() < 0.2) * np.linspace(0, 6.0, T)
    pg = np.zeros((T, 4))
    pg[:, 0] = np.cumsum(v * np.cos(yaw) * 0.2)
    pg[:, 1] = np.cumsum(v * np.sin(yaw) * 0.2)
    pg[:, 2] = yaw
    t0 = int(rng.integers(3, max(4, T - 35)))
    k = int(rng.integers(t0, T))
    own = (pg[k, 0] + rng.normal(0, 1.5), pg[k, 1] + rng.normal(0, 1.5), pg[k, 2] + rng.normal(0, 0.4))
    j = int(rng.integers(t0, T))
    r1, e1, d1, w1 = P.match_route(pg, t0, own, j)
    why = {"ok": G.OK, "off_lat": G.OFF_LAT, "off_head": G.OFF_HEAD, "end_edge": G.END_EDGE}[w1]
    m, mb = G.match_route(pg, t0, own, j), band(pg, t0, own, j)
    n_band_eq_probe += int((mb.r, mb.reason) == (r1, why) and mb.e_lat == e1)
    if (r1, why) == (m.r, m.reason):
        n_same += 1
    else:
        n_diff += 1
        n_reason_diff += int(why != m.reason)
print(f"cases {a.n}  v8f_dagger == probe on (r', reason): {n_same}   different: {n_diff} (reason differs in {n_reason_diff})   "
      f"v8f_dagger with the probe's tie-band line == probe: {n_band_eq_probe}/{a.n}")
assert n_band_eq_probe == a.n, "the tie-set line is NOT the only difference between the two implementations"
