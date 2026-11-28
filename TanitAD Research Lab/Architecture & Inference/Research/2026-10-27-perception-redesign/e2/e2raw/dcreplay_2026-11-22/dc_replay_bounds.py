"""dc_replay_bounds.py -- MM 4a3d05, 2026-11-22 ~12:45Z. POST-HOC (NOT pre-committed) follow-up to dc_replay.py: the tip's
dead-beat rule with NARROWER weight bounds, on the same logged probe rows and the same counterfactual (probe i+1's own per-unit
statistics). Exploration only: any pick from here needs its own pre-registration and a real training run to be a result.
"""
import importlib.util, math, sys, statistics as st

spec = importlib.util.spec_from_file_location("mdc", sys.argv[1])
mdc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdc)
TGT, BAND, FLOOR = 0.02, (0.01, 0.03), 3000


def load(p):
    rows = []
    for l in open(p, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        step, wb, we, sr, sp, na, nt, pj = int(float(f[0])), *map(float, f[1:8])
        rows.append(dict(step=step, wb=wb, st=mdc.recover(na, nt, pj, wb)))
    return rows


for name, p in (("K2DC02", sys.argv[2]), ("K2DC02s1", sys.argv[3])):
    rows = load(p)
    for lo, hi in ((1.0, 25.0), (1.5, 6.0), (2.0, 4.0), (2.5, 3.5)):
        w = min(max(rows[0]["wb"], lo), hi); sh = []
        for r in rows:
            s = mdc.share_at(w, r["st"]["a1sq"], r["st"]["a1b"], r["st"]["B2"])
            if r["step"] >= FLOOR:
                sh.append(s)
            w, _ = mdc.solve(TGT, r["st"]["a1sq"], r["st"]["a1b"], r["st"]["B2"], lo, hi)
        n_in = sum(1 for s in sh if BAND[0] <= s <= BAND[1])
        print("%-9s DB[%.1f,%.1f]  inband %2d/%d  mean %.4f  median %.4f  sd %.4f" % (name, lo, hi, n_in, len(sh), st.mean(sh), st.median(sh), st.pstdev(sh)))
