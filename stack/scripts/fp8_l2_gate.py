#!/usr/bin/env python3
"""The fp8 cache gate: does e4m3 quantisation cost the L2 probe targets?

PRE-STATED CRITERION (before any number was computed): the fp8 cache build is
admissible if, on every target, the median per-episode drop in val R2
(fp16 - fp8) is < 0.01 absolute AND no single target drops more than 0.02.
Anything worse and the 330 GiB build waits for a better storage scheme.

Doctrine compliance: episode-disjoint fit/val; lambda selected by
leave-episode-out CV on the FIT split only; constant control must read ~0;
n and d printed (n/d ~ 2.4 -- marginal power, stated).

--extra-fp16 / --extra-ep (2026-09-02): score ONE additional episode (a
rebuilt cache entry, given as its fp16 field + its v2ep) under the SAME probe
fitted on the 19 FIT episodes — it is appended to the VAL split, never to the
fit — and report (a) its own per-target relative MSE increase against the
per-episode bound and (b) the frozen gate re-evaluated over val + extra. The
default run (no flags) is unchanged and must reproduce the banked JSON.
"""
import argparse
import json
from pathlib import Path

import torch

CACHE = Path("C:/Users/Admin/refav1_probe/dinov3cache")
EPS = Path("C:/Users/Admin/refav1_probe/eps")
OUT = Path("C:/Users/Admin/refav1_probe/fp8_l2_gate.json")
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]


def load(cache_pt: Path, ep_pt: Path, fp8: bool):
    x = torch.load(cache_pt, weights_only=True).float()
    if fp8:
        x = x.to(torch.float8_e4m3fn).float()
    feats = x.mean(dim=1)                                  # [T, 1024] pooled
    o = torch.load(ep_pt, map_location="cpu", weights_only=False)
    p = o["poses"]
    T = feats.shape[0]
    idx = torch.arange(T) * 2
    v = p[idx, 3]
    yaw = p[idx, 2]
    nxt = torch.clamp(idx + 2, max=p.shape[0] - 1)
    yr = (p[nxt, 2] - yaw) / 0.2                           # yaw rate rad/s
    ac = (p[nxt, 3] - v) / 0.2                             # accel m/s^2
    return feats, {"speed": v, "yaw_rate": yr, "accel": ac}


def ridge_fit(X, y, lam):
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    A = Xb.T @ Xb + lam * torch.eye(Xb.shape[1])
    return torch.linalg.solve(A, Xb.T @ y)


def r2(X, y, w):
    Xb = torch.cat([X, torch.ones(len(X), 1)], 1)
    resid = ((Xb @ w - y) ** 2).sum()
    tot = ((y - y.mean()) ** 2).sum().clamp_min(1e-9)
    return float(1 - resid / tot)


PCA_D = 128


def run(fp8: bool, cache: Path = CACHE, eps: Path = EPS, extra=None):
    # ⚠️ FIRST RUN OF THIS GATE, KEPT ON THE RECORD: at d=1024 with n=1,918
    # (n/d=1.9) the fp16 probe itself read R2 -13.7/-19.2/-1.2 — the doctrine's
    # failure #4, underpowered BY CONSTRUCTION, and a panel that cannot gate
    # anything. Fix per the doctrine: reduce d with a PCA basis fit on the FIT
    # SPLIT ONLY (a basis fit on all data would leak val structure into the
    # features). d 1024 -> 128 gives n/d = 15.
    names = sorted(p.stem for p in cache.glob("*.pt") if p.stem != "index")
    tr_names, va_names = names[:19], names[19:]
    tr = [load(cache / f"{n}.pt", eps / f"{n}.v2ep.pt", fp8) for n in tr_names]
    va = [load(cache / f"{n}.pt", eps / f"{n}.v2ep.pt", fp8) for n in va_names]
    ex = load(extra[0], extra[1], fp8) if extra is not None else None
    Xall = torch.cat([f for f, _ in tr])
    mu0, sd0 = Xall.mean(0), Xall.std(0).clamp_min(1e-6)
    _, _, V = torch.linalg.svd((Xall - mu0) / sd0, full_matrices=False)
    basis = V[:PCA_D].T                                   # fit-split-only PCA
    tr = [(((f - mu0) / sd0) @ basis, t) for f, t in tr]
    va = [(((f - mu0) / sd0) @ basis, t) for f, t in va]
    if ex is not None:
        ex = (((ex[0] - mu0) / sd0) @ basis, ex[1])
    out = {}
    for tgt in ("speed", "yaw_rate", "accel"):
        Xtr = torch.cat([f for f, _ in tr])
        ytr = torch.cat([t[tgt] for _, t in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        # lambda by leave-episode-out CV on the FIT split ONLY
        best_lam, best = None, -1e9
        for lam in LAMBDAS:
            score = 0.0
            for k in range(len(tr)):
                Xf = torch.cat([f for i, (f, _) in enumerate(tr) if i != k])
                yf = torch.cat([t[tgt] for i, (_, t) in enumerate(tr) if i != k])
                w = ridge_fit((Xf - mu) / sd, yf, lam)
                score += r2((tr[k][0] - mu) / sd, tr[k][1][tgt], w)
            if score > best:
                best, best_lam = score, lam
        w = ridge_fit((Xtr - mu) / sd, ytr, best_lam)
        per_ep = [r2((f - mu) / sd, t[tgt], w) for f, t in va]
        def _mse(X, y):
            Xb = torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1)
            return float(((Xb @ w - y) ** 2).mean())
        per_ep_mse = [_mse(f, t[tgt]) for f, t in va]
        # constant control: the train mean, scored on val
        const = [float(1 - ((t[tgt] - ytr.mean()) ** 2).sum()
                       / ((t[tgt] - t[tgt].mean()) ** 2).sum().clamp_min(1e-9))
                 for _, t in va]
        out[tgt] = {"lam": best_lam, "val_r2_per_ep": per_ep,
                    "val_mse_per_ep": per_ep_mse,
                    "const_r2_per_ep": const,
                    "n": int(sum(f.shape[0] for f, _ in tr)), "d": PCA_D}
        if ex is not None:
            out[tgt]["extra_mse"] = _mse(ex[0], ex[1][tgt])
            out[tgt]["extra_r2"] = r2((ex[0] - mu) / sd, ex[1][tgt], w)
            out[tgt]["extra_n"] = int(ex[0].shape[0])
    return out


# ⚠️ SECOND RUN, ALSO ON THE RECORD: at n/d=15 the fp16 probe still read R2
# -15/-40/-0.9 — NOT a probe-power problem this time but a METRIC defect: the
# per-episode R2 denominator is the WITHIN-episode variance, which on a
# straight-driving episode is ~0, so any absolute error explodes the score
# (the overlapping_holdout_se family: an estimator artifact read as signal).
# ⇒ FINAL criterion, re-stated ONCE and then frozen regardless of outcome:
# per-episode RELATIVE MSE increase (mse_fp8 - mse_fp16)/mse_fp16, gate =
# median < 0.01 AND max < 0.05 across all targets. MSE has no per-episode
# denominator to explode; the R2 numbers stay banked as evidence of the
# metric defect, not of decodability.


def mse_eval(runs, tgt, w_key="w"):
    pass  # (per-episode MSEs are computed inline below)


def _gate(a16, a8, with_extra: bool):
    """The frozen criterion over the val episodes (+ the extra one if asked)."""
    delta, ok = {}, True
    for tgt in a16:
        m16 = list(a16[tgt]["val_mse_per_ep"])
        m8 = list(a8[tgt]["val_mse_per_ep"])
        if with_extra:
            m16.append(a16[tgt]["extra_mse"])
            m8.append(a8[tgt]["extra_mse"])
        rel = [(b - a) / max(a, 1e-12) for a, b in zip(m16, m8)]
        med = sorted(rel)[len(rel) // 2]
        mx = max(rel)
        delta[tgt] = {"rel_mse_per_ep": rel, "median": med, "max": mx,
                      "n_episodes": len(rel)}
        ok &= (med < 0.01 and mx < 0.05)
        if with_extra:
            # ⚠️ `sorted(rel)[n // 2]` is the exact median for the banked ODD
            # n=5 but the UPPER median for an even n; with the extra episode
            # (n=6) both conventions are carried so the reader sees which one
            # the verdict rests on. MEASURED 2026-09-02: accel upper-median
            # +0.0128 (FAIL) vs standard median +0.0034 (PASS), the extra
            # episode's own delta being +0.0237 — within the per-episode bound.
            s = sorted(rel)
            med_std = (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2 if len(s) % 2 == 0 else med
            delta[tgt]["median_standard"] = med_std
            delta[tgt]["median_convention"] = "sorted[n//2] (upper median for even n)"
    return delta, ("PASS" if ok else "FAIL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--eps", type=Path, default=EPS)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default {OUT}; REQUIRED with --extra-* so the "
                         "banked gate JSON is never overwritten")
    ap.add_argument("--extra-fp16", type=Path, default=None,
                    help="fp16 [T,640,1024] field of ONE extra episode to "
                         "score as an additional VAL episode")
    ap.add_argument("--extra-ep", type=Path, default=None,
                    help="that episode's v2ep (poses -> targets)")
    a = ap.parse_args(argv)
    extra = None
    if (a.extra_fp16 is None) != (a.extra_ep is None):
        raise SystemExit("--extra-fp16 and --extra-ep go together")
    if a.extra_fp16 is not None:
        extra = (a.extra_fp16, a.extra_ep)
        if a.out is None:
            raise SystemExit("--out is required with --extra-*")
    out_path = a.out or OUT

    a16, a8 = run(False, a.cache, a.eps, extra), run(True, a.cache, a.eps, extra)
    res = {"criterion_final": "per-ep relative MSE increase: median < 0.01, max < 0.05",
           "criterion_history": ["R2-based (run 1, d=1024): underpowered n/d=1.9",
                                 "R2-based (run 2, d=128): denominator explodes on low-variance episodes"],
           "fp16": a16, "fp8": a8, "delta": {}}
    n_va = len(a16["speed"]["val_mse_per_ep"])
    print(f"n_fit={a16['speed']['n']}  d={a16['speed']['d']}  (n/d={a16['speed']['n']/a16['speed']['d']:.1f})")
    print(f"{'target':>9} {'fp16 MSE':>10} {'fp8 MSE':>10} {'med relD':>9} {'max relD':>9}")
    res["delta"], res["gate"] = _gate(a16, a8, with_extra=False)
    for tgt in a16:
        d = res["delta"][tgt]
        print(f"{tgt:>9} {sum(a16[tgt]['val_mse_per_ep'])/n_va:>10.4f} "
              f"{sum(a8[tgt]['val_mse_per_ep'])/n_va:>10.4f} {d['median']:>+9.4f} {d['max']:>+9.4f}")
    print(f"\nGATE: {res['gate']} (criterion: {res['criterion_final']})")

    if extra is not None:
        ex = {"fp16": str(extra[0]), "ep": str(extra[1]), "per_target": {}}
        print(f"\nEXTRA episode {extra[0].name}: scored as a 6th VAL episode under "
              f"the same 19-episode probe (n={a16['speed']['extra_n']})")
        print(f"{'target':>9} {'fp16 MSE':>10} {'fp8 MSE':>10} {'relD':>9} {'bound':>9}")
        ok = True
        for tgt in a16:
            m16, m8 = a16[tgt]["extra_mse"], a8[tgt]["extra_mse"]
            rel = (m8 - m16) / max(m16, 1e-12)
            ex["per_target"][tgt] = {"mse_fp16": m16, "mse_fp8": m8, "rel_mse_increase": rel,
                                     "r2_fp16": a16[tgt]["extra_r2"], "r2_fp8": a8[tgt]["extra_r2"],
                                     "within_per_episode_bound_0.05": rel < 0.05}
            ok &= rel < 0.05
            print(f"{tgt:>9} {m16:>10.4f} {m8:>10.4f} {rel:>+9.4f} {'<0.05 ok' if rel < 0.05 else 'EXCEEDS':>9}")
        ex["per_episode_bound_verdict"] = "PASS" if ok else "FAIL"
        ex["delta_with_extra"], ex["gate_with_extra"] = _gate(a16, a8, with_extra=True)
        print(f"per-episode bound (max < 0.05 on every target): {ex['per_episode_bound_verdict']}")
        ex["gate_with_extra_standard_median"] = "PASS" if all(
            v["median_standard"] < 0.01 and v["max"] < 0.05
            for v in ex["delta_with_extra"].values()) else "FAIL"
        print(f"GATE re-evaluated over val + extra ({n_va + 1} episodes): "
              f"{ex['gate_with_extra']} under sorted[n//2] (upper median), "
              f"{ex['gate_with_extra_standard_median']} under the standard median\n  "
              + "  ".join(f"{t}: upper-med {v['median']:+.4f} std-med "
                          f"{v['median_standard']:+.4f} max {v['max']:+.4f}"
                          for t, v in ex["delta_with_extra"].items()))
        res["extra"] = ex
    out_path.write_text(json.dumps(res, indent=1))
    print(f"banked -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
