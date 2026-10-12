"""PREREG_VR8_NOLW readout, written BEFORE the arm's data exists (2026-10-12). Applies P1 / P2 and the three outcomes
mechanically. GT-turn windows (turnL | turnR), cross_abs_6s, paired episode-cluster bootstrap (ladder_score), both
sampler seeds.
  P1: <noLW>/base vs <VR8>/base (dev box batch 4): separated BETTER on both seeds, AND R = |d| / floor > 1 on both,
      floor = |mean d(<VR8r> - <VR8f>)| on the same stratum + metric from Thor's batch-8 dumps (same seed).
  P2: <noLW>/oraclebase vs <VR8>/oraclebase: NOT separated worse on both seeds.
  CONFIRMED = P1 and P2; NULL = P1 separated on neither seed; PARTIAL = anything else.
Also prints every metric family on ALL / GT turn / GT straight for the base rows (descriptive).
usage: python nolw_readout.py <noLW cands dir> <VR8 cands dir> <VR8 thor dir> <VR8r thor dir> [--out json]"""
import json
import sys
from pathlib import Path

import numpy as np

import ladder_score as LS

TURN = ("turnL", "turnR")


def pair(dT, rT, dB, rB, s, metric, stratum):
    zt, zb = LS.load_pass(Path(dT) / f"{rT}_s{s}.npz"), LS.load_pass(Path(dB) / f"{rB}_s{s}.npz")
    if zt["_json"]["window_sha12_t_sha256"] != zb["_json"]["window_sha12_t_sha256"]:
        raise SystemExit(f"windows differ: {dT}/{rT} vs {dB}/{rB}")
    wt, wb = LS.window_metrics(zt), LS.window_metrics(zb)
    gtc = np.asarray(LS.gt_class(np.asarray(zb["gt"]), np.asarray(zb["gt_valid"]))[0]).astype(str)
    st = {"ALL": np.ones(len(gtc), bool), "GT turn": np.isin(gtc, TURN), "GT straight": gtc == "straight"}[stratum]
    eps, inv = LS.episodes(zb["win_sha12"])
    W = LS.counts_of(LS.make_draws(len(eps)), len(eps))[:, inv]
    at, mt = wt[metric]
    ab, mb = wb[metric]
    k = st & mt.astype(bool) & mb.astype(bool)
    diff = at - ab
    num, den = (W[:, k] * diff[k]).sum(1), W[:, k].sum(1)
    bs = num[den > 0] / den[den > 0]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(k.sum()), "T": float(np.mean(at[k])), "B": float(np.mean(ab[k])), "d": float(np.mean(diff[k])),
            "lo": float(lo), "hi": float(hi), "better": int(LS.METRICS[metric][1])}


def sep(r):
    """'BETTER' / 'WORSE' / 'ns' for T relative to B, honouring the metric's orientation."""
    if r["lo"] <= 0 <= r["hi"]:
        return "ns"
    good = (r["hi"] < 0) if r["better"] < 0 else (r["lo"] > 0)
    return "BETTER" if good else "WORSE"


def main(a):
    nolw, vr8, vr8_thor, vr8r_thor = a[:4]
    out = a[a.index("--out") + 1] if "--out" in a else None
    rec = {"P1": {}, "P2": {}, "families": {}}
    for s in LS.SEEDS:
        p1 = pair(nolw, "base", vr8, "base", s, "cross_abs_6s", "GT turn")
        fl = pair(vr8r_thor, "base", vr8_thor, "base", s, "cross_abs_6s", "GT turn")
        floor = abs(fl["d"])
        p1.update(sep=sep(p1), floor=floor, floor_src=fl, R=(abs(p1["d"]) / floor if floor > 0 else float("inf")))
        p1["pass"] = bool(p1["sep"] == "BETTER" and p1["R"] > 1.0)
        p2 = pair(nolw, "oraclebase", vr8, "oraclebase", s, "cross_abs_6s", "GT turn")
        p2.update(sep=sep(p2))
        p2["pass"] = bool(p2["sep"] != "WORSE")
        rec["P1"][f"s{s}"], rec["P2"][f"s{s}"] = p1, p2
        fam = {}
        for m, (family, _) in LS.METRICS.items():
            for stratum in ("ALL", "GT turn", "GT straight"):
                try:
                    r = pair(nolw, "base", vr8, "base", s, m, stratum)
                except Exception as e:                       # a family with no valid windows in a stratum is named
                    fam[f"{family}|{m}|{stratum}"] = {"absent": repr(e)[:120]}
                    continue
                if r["n"] >= 5:
                    r["sep"] = sep(r)
                    fam[f"{family}|{m}|{stratum}"] = r
        rec["families"][f"s{s}"] = fam
    p1s = [rec["P1"][f"s{s}"] for s in LS.SEEDS]
    p2s = [rec["P2"][f"s{s}"] for s in LS.SEEDS]
    if all(p["pass"] for p in p1s) and all(p["pass"] for p in p2s):
        verdict = "CONFIRMED"
    elif not any(p["sep"] == "BETTER" for p in p1s):
        verdict = "NULL"
    else:
        verdict = "PARTIAL"
    rec["verdict"] = verdict
    print(f"V-R8-noLW readout: {verdict}")
    for s in LS.SEEDS:
        p1, p2 = rec["P1"][f"s{s}"], rec["P2"][f"s{s}"]
        print(f"  s{s} P1 cross_abs_6s GT turn: noLW {p1['T']:.3f} vs V-R8 {p1['B']:.3f} d={p1['d']:+.3f} "
              f"[{p1['lo']:+.3f},{p1['hi']:+.3f}] {p1['sep']}; floor {p1['floor']:.3f} R={p1['R']:.2f} -> "
              f"{'pass' if p1['pass'] else 'FAIL'}")
        print(f"  s{s} P2 oraclebase cross GT turn: d={p2['d']:+.3f} [{p2['lo']:+.3f},{p2['hi']:+.3f}] {p2['sep']} -> "
              f"{'pass' if p2['pass'] else 'FAIL'}")
    if out:
        Path(out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return verdict


if __name__ == "__main__":
    main(sys.argv[1:])
