#!/usr/bin/env python3
"""PHEAD open-loop arms (PREREG_PHEAD.md s.4): the adam_sim.py simulation on the gradient bank with the restriction applied.
  g_t = gB(draw t) + lam_t * gE(draw t);  Restricted(group) = clip + AdamW over the group's entries only (kind R), or the EPO gradient masked to the group while BCE moves everything (kind E).
  Both regimes exactly as adam_sim.py:  T = trainer schedule (warm-up lr, EPO ramp), K=250;  P = constant lr and lam from step 0, K=100.
    python phead_open.py --streams 0 1 2 --groups FULL PH_A_EPROW ... --out open_main.pkl [--j-streams 0]
"""
import argparse, os, pickle, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import phead_common as PC
from adam_sim import draws_for                     # noqa: E402  (the H-ADAM instrument's own draw rule)
import adam_common as AC                           # noqa: E402

SNAP = {"P": (25, 50, 100), "T": (50, 100, 250)}
KMAX = {"P": 100, "T": 250}


def run_open(bank, theta0, a, draws, regime, lam, K, sl, kind="R", lr_mult=1.0, snaps=()):
    shapes = bank["shapes"]
    full = [(i, None) for i in range(len(theta0))]
    R = PC.Restricted(theta0, sl if kind == "R" else full, lr=a.lr * lr_mult)
    mask = PC.flat_mask(sl, shapes) if kind == "E" else None
    D, norms = {}, []
    for t in range(K):
        ix = torch.as_tensor(draws[t])
        gb = bank["gB"][ix].sum(0)
        ge = bank["gE"][ix].sum(0)
        if mask is not None:
            ge = ge * mask
        l = AC.lam_t(lam, t, a) if regime == "T" else lam
        lr = (AC.lr_at(t, a) if regime == "T" else a.lr) * lr_mult
        norms.append(R.step(PC.unflat_t(gb + l * ge, shapes), lr))
        if (t + 1) in snaps:
            D[t + 1] = R.theta()
    return D, {"grad_norm_mean": float(np.mean(norms)), "clip_steps": int(sum(n > 1.0 for n in norms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[0])
    ap.add_argument("--j-streams", type=int, nargs="+", default=[0])
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["R"])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    ap.add_argument("--regimes", nargs="+", default=["P", "T"])
    ap.add_argument("--lr-mults", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jP", type=int, nargs="*", default=[50, 100], help="regime-P snapshots at which J is evaluated (j-streams only)")
    ap.add_argument("--jT", type=int, nargs="*", default=[250], help="regime-T snapshots at which J is evaluated (j-streams only)")
    ap.add_argument("--j-lams", type=float, nargs="*", default=None, help="lams at which J is evaluated (default: all)")
    q = ap.parse_args()
    jsn = {"P": set(q.jP), "T": set(q.jT)}
    t0 = time.time()
    SF, a, holder, cfg, tr, ev = PC.setup(q.threads)
    th0 = PC.theta0_of(tr)
    names = tr.names
    shapes = [tuple(p.shape) for p in th0]
    res = {"meta": {"groups": q.groups, "kinds": q.kinds, "lams": q.lams, "regimes": q.regimes, "lr_mults": q.lr_mults, "names": names}, "runs": {}}
    for s in q.streams:
        bank = torch.load(os.path.join(PC.BANK_DIR, f"bank_s{s}.pt"), weights_only=False)
        draws = draws_for(s, 250, bank["gB"].shape[0])
        for regime in q.regimes:
            for g in q.groups:
                sl = PC.group_slices(g, names, shapes)
                for kind in q.kinds:
                    if kind == "E" and g in ("FULL", "EMPTY"):
                        continue
                    for lrm in q.lr_mults:
                        if lrm != 1.0 and kind == "E":
                            continue
                        for lam in q.lams:
                            D, meta = run_open(bank, th0, a, draws, regime, lam, KMAX[regime], sl, kind, lrm, SNAP[regime])
                            for k, th in D.items():
                                key = (s, regime, g, kind, lrm, lam, k)
                                rec = {"meta": meta if k == KMAX[regime] else None, "disp_norm": float(torch.cat([(x - y).reshape(-1) for x, y in zip(th, th0)]).norm()),
                                       "outside": PC.outside_group_displacement(th, th0, sl) if kind == "R" else None}
                                if s in q.j_streams and k in jsn[regime] and (q.j_lams is None or lam in q.j_lams):
                                    rec["eval"] = {pn: e.evaluate(th) for pn, e in ev.items()}
                                res["runs"][key] = rec
                    print(f"stream {s} {regime} {g} done {time.time()-t0:.0f}s", flush=True)
    pickle.dump(res, open(q.out, "wb"))
    print("ZZOPEN_OK", q.out, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
