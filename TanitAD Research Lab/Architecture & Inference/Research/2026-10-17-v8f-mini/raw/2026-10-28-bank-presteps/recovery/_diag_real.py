import sys, collections
import numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "../scripts")
import common as C, labeller as L
tab = C.attach_logged(C.load_table("s0"), C.load_logged()); lg = C.load_logged()
G, V, offs, Ts = lg["G"], lg["V"], lg["offs"], lg["Ts"]
reason = tab["reason"]
cut = np.flatnonzero((reason == C.OFF_LAT) | (reason == C.OFF_HEAD))
sel = cut[np.random.default_rng(1).choice(len(cut), size=3000, replace=False)][:900]
fail = collections.Counter(); first_fail = collections.Counter(); byv = collections.defaultdict(collections.Counter)
for rec in sel:
    c = int(tab["cidx"][rec]); off, T = int(offs[c]), int(Ts[c]); r = int(tab["r"][rec])
    if r + L.H > T - 1: continue
    own = (float(tab["ox"][rec]), float(tab["oy"][rec]), float(tab["oyaw"][rec])); vown = float(tab["v_own"][rec])
    grid, vlog = G[off:off+T], V[off:off+T]
    full = L.recovery_label(grid, vlog, r, own, vown)
    if full["label_own"] is not None: continue
    vb = int(min(vown // 3, 4))
    modes = []
    for kj in L.LADDER:
        res = L.recovery_label(grid, vlog, r, own, vown, check=False, ladder=(kj,))
        if res["label_own"] is None:
            modes.append(res["status"]); fail[(kj, res["status"])] += 1
        else:
            v = L.violations(res["label_own"], vown, res["k_rej"] + 1)
            m = "+".join(k for k in ("accel", "curv", "fric") if v[k] > 0) or "ok"
            modes.append(m); fail[(kj, m)] += 1
    first_fail[modes[0]] += 1; byv[vb][modes[2]] += 1
print("rung-by-rung failure modes (refused records only)")
for kj in L.LADDER: print(kj, {m: n for (k, m), n in sorted(fail.items()) if k == kj})
print("rung5 mode", dict(first_fail)); print("rung15 mode by v_own bin", {b: dict(c) for b, c in sorted(byv.items())})
