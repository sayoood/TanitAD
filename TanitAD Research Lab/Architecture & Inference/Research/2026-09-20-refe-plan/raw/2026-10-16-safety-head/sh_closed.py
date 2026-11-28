#!/usr/bin/env python3
"""SAFEHEAD closed-loop arm (PREREG_SAFEHEAD.md s.4/s.7).  One process = one (design, lam, regime, K).  Output: <out-dir>/<name>.pkl with the per-view, per-set arrays at each snapshot.
    python sh_closed.py --design SH_A --lam 1.0 --regime P --K 50 --snaps 25 50 --out-dir out [--threads 6] [--identity-vs-banked]
--identity-vs-banked (regime P only): control (a''): the co-trained arm A (and, for FULL, arm B) vs the banked closed_P_{lam}.pt states D[k] (rel L2 <= 1e-3)."""
import argparse, os, pickle, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sh_common as SH
PC = SH.PC


def rel_vs_banked(theta, th0, D):
    d = [(x - y) - Dk for x, y, Dk in zip(theta, th0, D)]
    dn = float(torch.cat([x.reshape(-1) for x in d]).norm())
    bn = float(torch.cat([x.reshape(-1) for x in D]).norm())
    return {"abs_l2_diff": dn, "banked_l2_norm": bn, "rel_l2_diff": dn / bn, "max_abs_diff": max(float(x.abs().max()) for x in d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=SH.DESIGNS)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--regime", default="P", choices=("P", "T"))
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--snaps", type=int, nargs="+", default=[25, 50])
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--sep", default="on", choices=("on", "off"))
    ap.add_argument("--co-train-a", default="auto", choices=("auto", "yes", "no"))
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--identity-vs-banked", action="store_true")
    ap.add_argument("--tag", default="")
    q = ap.parse_args()
    t0 = time.time()
    SF, a, holder, cfg, tr, ev = SH.setup(q.threads)
    co = None if q.co_train_a == "auto" else (q.co_train_a == "yes")
    name = f"sh_s{q.stream}_{q.regime}_{q.design}_sep{q.sep}_lr{q.lr_mult:g}_lam{q.lam:g}_K{q.K}{q.tag}"
    th0 = [p.detach().clone() for p in tr.base.parameters()]
    keep = q.identity_vs_banked
    log, snaps, thB, thA = SH.run_closed(SF, a, tr, ev, q.design, q.lam, q.regime, q.K, set(q.snaps), q.stream, sep=(q.sep == "on"), co_train_A=co,
                                         lr_mult=q.lr_mult, keep_theta=keep, name=name)
    out = {"name": name, "args": vars(q), "log": log, "closure_hashes": tr.closure_hashes,
           "snaps": {k: {kk: vv for kk, vv in v.items() if kk not in ("thetaB", "thetaA")} for k, v in snaps.items()},
           "logs": {pn: e.logs for pn, e in ev.items()}}
    if keep and q.regime == "P":
        idn = {}
        zA = torch.load(f"{PC.SCR}/closed/closed_P_0.0.pt", weights_only=False)
        zB = torch.load(f"{PC.SCR}/closed/closed_P_{q.lam}.pt", weights_only=False) if q.design == "FULL" and q.sep == "on" else None
        for k, s in snaps.items():
            if s.get("thetaA") is not None and k in zA["D"]:
                idn[f"armA_vs_banked_lam0_D{k}"] = rel_vs_banked(s["thetaA"], th0, zA["D"][k])
            if zB is not None and k in zB["D"]:
                idn[f"armB_vs_banked_lam{q.lam}_D{k}"] = rel_vs_banked(s["thetaB"], th0, zB["D"][k])
        out["identity_vs_banked"] = idn
        print("identity_vs_banked", {k: round(v["rel_l2_diff"], 7) for k, v in idn.items()}, flush=True)
    pickle.dump(out, open(os.path.join(q.out_dir, name + ".pkl"), "wb"))
    print("ZZSH_OK", name, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
