"""dc_replay.py -- MM 4a3d05, 2026-11-22 ~12:30Z. OFFLINE replay of the K2DC map-dose controller on the LOGGED probe rows of
K2DC02 (seed 0) and K2DC02s1 (seed 1): would a smoothed controller have held the share in [0.01, 0.03]?

PRE-COMMITTED before any candidate was run (this header is the record):
  * metric: in-band count of the share READ at probes with step >= 3000 (PREREG_K2DC02 sec. 2 item 2's band and floor);
  * candidates: DB (the tip's dead-beat rule, = control), FIX(w) constants, EMA(beta)+DAMP(alpha) with beta in {0.5, 0.75},
    alpha in {0.3, 0.5, 1.0};
  * selection: the candidate with the most in-band probes on K2DC02 (seed 0, FIT); its K2DC02s1 (seed 1) count is the HELD-OUT read.
COUNTERFACTUAL: the reading at probe i+1 under weight w is s_{i+1}(w) from probe i+1's own recovered per-unit statistics
(map_dose_ctl.recover / share_at, imported from the tip's file). LIMITATION (named): probe i+1's statistics come from the model
trained along the LOGGED weight sequence; a different weight sequence would have changed the model, which this cannot see.
CONTROLS that must hold or the replay is void: (1) s_i(w_before_i) == logged share_read_i (an algebraic identity);
(2) DB replayed == logged dc_w_eff at every probe.
"""
import importlib.util, math, sys, json

spec = importlib.util.spec_from_file_location("mdc", sys.argv[1])
mdc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mdc)
LO, HI, TGT, BAND, FLOOR = 1.0, 25.0, 0.02, (0.01, 0.03), 3000


def load(p):
    rows = []
    for l in open(p, encoding="utf-8"):
        f = l.rstrip("\n").split("\t")
        step, wb, we, sr, sp, na, nt, pj = int(float(f[0])), *map(float, f[1:8])
        st = mdc.recover(na, nt, pj, wb)
        rows.append(dict(step=step, wb=wb, we=we, sr=sr, st=st))
    return rows


def s_at(r, w):
    return mdc.share_at(w, r["st"]["a1sq"], r["st"]["a1b"], r["st"]["B2"])


def run(rows, kind, beta=None, alpha=None, wfix=None):
    """-> list of (step, weight in force at this probe, share read at this probe). Probe 0 runs at the logged initial weight."""
    w = rows[0]["wb"]; ema = None; out = []
    for i, r in enumerate(rows):
        out.append((r["step"], w, s_at(r, w)))
        st = r["st"]
        # the reading is taken at w; recompute the per-unit statistics AT THE WEIGHT IN FORCE (they are weight-free per unit)
        cur = dict(a1sq=st["a1sq"], a1b=st["a1b"], B2=st["B2"])
        if kind == "DB":
            k, _ = mdc.solve(TGT, cur["a1sq"], cur["a1b"], cur["B2"], LO, HI); w = k
        elif kind == "FIX":
            w = wfix
        else:
            ema = cur if ema is None else {q: beta * ema[q] + (1 - beta) * cur[q] for q in cur}
            k, _ = mdc.solve(TGT, ema["a1sq"], ema["a1b"], ema["B2"], LO, HI)
            w = math.exp((1 - alpha) * math.log(w) + alpha * math.log(k))
    return out


def inband(out):
    return sum(1 for s, w, v in out if s >= FLOOR and BAND[0] <= v <= BAND[1]), sum(1 for s, w, v in out if s >= FLOOR)


res = {}
for name, p in (("K2DC02", sys.argv[2]), ("K2DC02s1", sys.argv[3])):
    rows = load(p)
    c1 = max(abs(s_at(r, r["wb"]) - r["sr"]) for r in rows)
    db = run(rows, "DB")
    c2 = max(abs(db[i + 1][1] - rows[i]["we"]) for i in range(len(rows) - 1))
    # the weight that would have read EXACTLY the target at each probe (the per-probe requirement)
    wstar = [mdc.solve(TGT, r["st"]["a1sq"], r["st"]["a1b"], r["st"]["B2"], LO, HI)[0] for r in rows if r["step"] >= FLOOR]
    cands = {"DB": db}
    for wf in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        cands[f"FIX{wf}"] = run(rows, "FIX", wfix=wf)
    for b in (0.5, 0.75):
        for a in (0.3, 0.5, 1.0):
            cands[f"EMA{b}_DAMP{a}"] = run(rows, "EMA", beta=b, alpha=a)
    res[name] = {"n_rows": len(rows), "control1_max_abs": c1, "control2_max_abs_w": c2,
                 "wstar_ge3000": [round(x, 3) for x in wstar],
                 "cands": {k: {"inband": inband(v), "shares_ge3000": [round(x[2], 4) for x in v if x[0] >= FLOOR],
                               "w_ge3000": [round(x[1], 2) for x in v if x[0] >= FLOOR]} for k, v in cands.items()}}
print(json.dumps(res, indent=1))
