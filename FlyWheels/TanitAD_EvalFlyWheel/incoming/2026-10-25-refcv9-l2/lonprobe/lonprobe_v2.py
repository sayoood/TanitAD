"""lonprobe_v2.py -- PREREG_LONPROBE_V2.md: 20-permutation null for v1's cross-fitted probe (seed 0 population; z_tac is seed-invariant)."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import lonprobe as LP  # noqa: E402

def main():
    E = LP.C.load_eval_pf(0)
    idx, Z = LP.load_bank(LP.BANK / "ztac_eval_s0.npz")
    dg = LP.dv_of(E["g"], E["v0"], E["dt"]); gt = LP.cls_of(dg)
    rows = np.flatnonzero(E["compliant"] & np.isin(gt, ["decel", "cruise"]))
    j = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(E["clip"][rows], E["ws"][rows])]); assert (j >= 0).all()
    y = (gt[rows] == "decel").astype(int); eid = E["eid"][rows]; Xz = Z[j]; Xs = E["v0"][rows][:, None]
    of = LP.folds(eid, 5, 0)
    probe = np.empty(len(rows))
    for f in range(5):
        tr, te = np.flatnonzero(of != f), np.flatnonzero(of == f)
        probe[te], _ = LP.fit_pred(Xz, Xs, y, tr, te, eid, True)
    a_probe = LP.auc(-probe[y == 1], -probe[y == 0])
    v1 = json.load(open(Path(__file__).parent / "LONPROBE.json"))["seeds"]["0"]["auc"]["PROBE"]
    nulls = []
    for s in range(100, 120):
        rng = np.random.default_rng(s); sh = np.empty(len(rows))
        for f in range(5):
            tr, te = np.flatnonzero(of != f), np.flatnonzero(of == f)
            ys = y.copy(); ys[tr] = rng.permutation(y[tr])
            sh[te], _ = LP.fit_pred(Xz, Xs, ys, tr, te, eid, True)
        nulls.append(LP.auc(-sh[y == 1], -sh[y == 0])); print("ZZNULL", s, round(nulls[-1], 4), flush=True)
    nm = float(np.mean(nulls))
    N1 = abs(nm - 0.5) <= 0.02; N2 = abs(a_probe - v1) <= 1e-12
    if not (N1 and N2):
        verdict = "VOID"
    else:
        verdict = "HEADROOM-CONFIRMED" if a_probe > max(nulls) else "NOT-CONFIRMED"
    out = {"prereg_sha256": open(Path(__file__).parent / "PREREG_LONPROBE_V2.sha256").read().split()[0],
           "code_sha256": hashlib.sha256(open(Path(__file__).parent / "lonprobe.py", "rb").read()).hexdigest(),
           "probe_auc": a_probe, "v1_probe_auc": v1, "nulls": nulls, "null_mean": nm, "null_sd": float(np.std(nulls)),
           "null_max": float(max(nulls)), "N1_null_mean_ok": N1, "N2_reproduces_v1": N2, "verdict": verdict}
    json.dump(out, open(Path(__file__).parent / "LONPROBE_V2.json", "w"), indent=1)
    print("ZZV2", json.dumps({k: out[k] for k in ("probe_auc", "null_mean", "null_sd", "null_max", "N1_null_mean_ok", "N2_reproduces_v1", "verdict")}))

if __name__ == "__main__":
    main()
