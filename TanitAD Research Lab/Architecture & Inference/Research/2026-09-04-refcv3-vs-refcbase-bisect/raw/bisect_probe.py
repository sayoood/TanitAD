#!/usr/bin/env python3
"""0-GPU bisection probes on the BANKED refcv3 step-40,284 open-loop dump.

Answers, from the dump alone (no model, no GPU, no `tanitad` import):

  P0  POSITIVE CONTROL — reproduce the registry §4.5 row exactly. If this
      does not read `os` 0.4419 / `ha` 0.2996 / `os-ha` +0.1423 the probe is
      wrong, not the model.
  P1  Is the E9 goal-distance re-rank doing anything? (`sel_idx` vs
      `sel_idx_base`, both banked per window.)
  P2  v7.2 tactical-label coverage ON THE EVAL SPLIT (`lat_label == -100`).
  P3  Which v7 classes does the 8-wide tactical head actually emit, and how
      much of that mass lands in indices {0,1,2} — the ONLY indices
      `refc_tactical.derive_man5_logprobs` reads when it builds the
      `maneuver_logits` port (its contract is [B,3]x[B,3], the heads are
      [B,8]x[B,8]).
  P4  ALONG vs LATERAL decomposition of the `os - ha` deficit, per horizon.
  P5  The assignment-oracle ceiling (`oracle_sel`) against the same floor —
      does the FAN itself lose, and on which axis?
  P6  Does the deficit differ on the 24 % of windows that carry a tactical
      label? (tests the LOCAL label-coverage hypothesis only.)

Estimator throughout: paired episode-cluster bootstrap over the 141 eval
episodes, B=2000, seed 0 — the same estimator `taniteval/ci.py` uses.
NEVER `overlapping_holdout_se`.

Usage:  python bisect_probe.py <dump_dir>
        (dump_dir = the extracted `refcv3-40284-openloop-dump.tar.gz`)
Needs only numpy.
"""
from __future__ import annotations

import collections
import glob
import os
import sys

import numpy as np

LAT_V7 = ["LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
          "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]
LON_V7 = ["FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD",
          "ADAPT_SPEED_FOR_CURVE", "ACCELERATE"]
#: the three indices derive_man5_logprobs reads out of each head
READ_IDX = (0, 1, 2)
HORIZONS_S = (0.5, 1.0, 1.5, 2.0)


def load(dump: str):
    tf = sorted(glob.glob(os.path.join(dump, "ep*.npz")))
    df = sorted(glob.glob(os.path.join(dump, "decisions", "ep*.npz")))
    if not tf:
        raise SystemExit(f"no ep*.npz under {dump}")
    if len(tf) != len(df):
        raise SystemExit(f"{len(tf)} traj files vs {len(df)} decision files")
    T = collections.defaultdict(list)
    Dc = collections.defaultdict(list)
    ep = []
    for i, (a, b) in enumerate(zip(tf, df)):
        t, d = np.load(a), np.load(b)
        if not np.array_equal(t["ws"], d["ws"]):
            raise SystemExit(f"window index mismatch in {a} / {b}")
        n = len(t["ws"])
        for k in t.files:
            if t[k].shape[:1] == (n,):
                T[k].append(t[k])
        for k in d.files:
            if d[k].ndim == 1 and len(d[k]) == n:
                Dc[k].append(d[k])
        ep.append(np.full(n, i))
    return ({k: np.concatenate(v) for k, v in T.items()},
            {k: np.concatenate(v) for k, v in Dc.items()},
            np.concatenate(ep))


def boot(x, ep, B=2000, seed=0):
    """Episode-cluster bootstrap. `x` is already a per-window PAIRED delta
    when a delta is wanted — never combine two independent intervals."""
    rng = np.random.default_rng(seed)
    eids = np.unique(ep)
    by = {e: np.where(ep == e)[0] for e in eids}
    s = np.empty(B)
    for j in range(B):
        pick = rng.choice(eids, size=len(eids), replace=True)
        s[j] = x[np.concatenate([by[e] for e in pick])].mean()
    return float(x.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def line(name, m, lo, hi, extra=""):
    sep = "SEPARATED" if not (lo < 0 < hi) else "not sep"
    print(f"  {name:34s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sep}{extra}")


def main(dump: str) -> None:
    T, D, ep = load(dump)
    G = T["g"]
    N = len(G)
    print(f"dump={dump}\nn windows = {N}   n episodes = {len(np.unique(ep))}\n")

    ade = lambda k: np.linalg.norm(T[k] - G, axis=-1).mean(1)          # noqa: E731
    alo = lambda k: np.abs(T[k][..., 0] - G[..., 0]).mean(1)           # noqa: E731
    lat = lambda k: np.abs(T[k][..., 1] - G[..., 1]).mean(1)           # noqa: E731

    print("P0  POSITIVE CONTROL — must reproduce MODEL_REGISTRY.md 4.5")
    for k in ("os", "ha", "ha0", "oracle_sel", "os_navzero"):
        m, lo, hi = boot(ade(k), ep)
        print(f"  {k:12s} ADE {m:.4f} [{lo:.4f}, {hi:.4f}]")
    line("os - ha  (registry +0.1423)", *boot(ade("os") - ade("ha"), ep))

    print("\nP1  is the E9 goal-distance re-rank doing anything?")
    chg = D["sel_idx"] != D["sel_idx_base"]
    print(f"  emitted anchor CHANGED by the graft on {chg.sum()} / {N} "
          f"= {chg.mean():.4%} of windows")
    print(f"  goal_gate {np.unique(D['goal_gate'])}  "
          f"goal_score_absmean {D['goal_score_absmean'].mean():.4f}")
    hv = (D["sel_idx"] == D["a_star"]).astype(float)
    hb = (D["sel_idx_base"] == D["a_star"]).astype(float)
    print(f"  assignment-oracle hit: re-ranked {hv.mean():.4f} vs "
          f"pre-graft {hb.mean():.4f}")
    line("re-ranked - pre-graft", *boot(hv - hb, ep))
    if chg.any():
        line("  same, changed windows only", *boot((hv - hb)[chg], ep[chg]),
             extra=f"  n={int(chg.sum())}")

    print("\nP2  v7.2 tactical-label coverage on the EVAL split")
    for k in ("lat_label", "lon_label"):
        ok = D[k] != -100
        print(f"  {k:12s} in-band {ok.sum():5d} / {N} = {ok.mean():.4%}")
    rok = D["route_label"] >= 0
    print(f"  {'route_label':12s} present {rok.sum():5d} / {N} = {rok.mean():.4%}")

    print("\nP3  what the 8-wide tactical head emits, and what the port reads")
    for k, names in (("lat_pred_nav_true", LAT_V7), ("lon_pred_nav_true", LON_V7)):
        c = collections.Counter(D[k].tolist())
        print(f"  {k}:")
        for i, nm in enumerate(names):
            flag = "  <- READ by derive_man5_logprobs" if i in READ_IDX else ""
            print(f"     [{i}] {nm:22s} {c.get(i,0):6d}  {c.get(i,0)/N:7.3%}{flag}")
        print(f"     mass in READ indices {READ_IDX}: "
              f"{np.isin(D[k], READ_IDX).mean():.3%}")
    print("  nav sensitivity (agreement of the argmax):")
    for k in ("lat_pred", "lon_pred", "route_pred"):
        a, z, s = (D[k + "_nav_true"], D[k + "_nav_zero"], D[k + "_nav_shuffled"])
        print(f"     {k:11s} true==zero {np.mean(a == z):.4%}   "
              f"true==shuffled {np.mean(a == s):.4%}")

    print("\nP4  ALONG vs LATERAL decomposition of the os - ha deficit")
    line("ALONG  |dx|  os - ha", *boot(alo("os") - alo("ha"), ep))
    line("LATERAL|dy|  os - ha", *boot(lat("os") - lat("ha"), ep))
    line("ALONG  |dx|  os - ha0", *boot(alo("os") - alo("ha0"), ep))
    line("LATERAL|dy|  os - ha0", *boot(lat("os") - lat("ha0"), ep))
    print("  per horizon (ALONG):")
    for j, h in enumerate(HORIZONS_S):
        a = np.abs(T["os"][:, j, 0] - G[:, j, 0])
        b = np.abs(T["ha"][:, j, 0] - G[:, j, 0])
        c = np.abs(T["oracle_sel"][:, j, 0] - G[:, j, 0])
        m, lo, hi = boot(a - b, ep)
        print(f"     t={h:.1f}s  os {a.mean():.4f}  ha {b.mean():.4f}  "
              f"oracle {c.mean():.4f}  os-ha {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print("  per horizon (LATERAL):")
    for j, h in enumerate(HORIZONS_S):
        a = np.abs(T["os"][:, j, 1] - G[:, j, 1])
        b = np.abs(T["ha"][:, j, 1] - G[:, j, 1])
        c = np.abs(T["oracle_sel"][:, j, 1] - G[:, j, 1])
        m, lo, hi = boot(a - b, ep)
        print(f"     t={h:.1f}s  os {a.mean():.4f}  ha {b.mean():.4f}  "
              f"oracle {c.mean():.4f}  os-ha {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    print("\nP5  the FAN's own assignment-oracle against the trivial floor")
    print("  (oracle_sel = the REFINED trajectory of the raw anchor nearest GT")
    print("   over ALL 8 slots — the trainer's own assignment, NOT the fan's")
    print("   true minimum, so read it as the assignment ceiling.)")
    line("ADE     oracle_sel - ha", *boot(ade("oracle_sel") - ade("ha"), ep))
    line("ALONG   oracle_sel - ha", *boot(alo("oracle_sel") - alo("ha"), ep))
    line("LATERAL oracle_sel - ha", *boot(lat("oracle_sel") - lat("ha"), ep))

    print("\nP6  does the deficit depend on tactical-label coverage? (LOCAL test)")
    inb = D["lat_label"] != -100
    d = ade("os") - ade("ha")
    for nm, m_ in (("in-band", inb), ("no label", ~inb)):
        mm, lo, hi = boot(d[m_], ep[m_])
        print(f"  {nm:9s} n={int(m_.sum()):5d}  os-ha {mm:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    rng = np.random.default_rng(0)
    eids = np.unique(ep)
    by = {e: np.where(ep == e)[0] for e in eids}
    s = []
    for _ in range(2000):
        idx = np.concatenate([by[e] for e in rng.choice(eids, len(eids), True)])
        a, b = d[idx][inb[idx]], d[idx][~inb[idx]]
        if len(a) and len(b):
            s.append(a.mean() - b.mean())
    s = np.asarray(s)
    print(f"  DiD (in-band minus no-label) {d[inb].mean()-d[~inb].mean():+.4f} "
          f"[{np.percentile(s,2.5):+.4f}, {np.percentile(s,97.5):+.4f}]")
    print("  NOTE: this tests only the LOCAL hypothesis (windows near a label")
    print("  score better). A head made globally weaker by 24 % coverage would")
    print("  NOT show up here — that needs a training ablation.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "refcv3_40284_dump")
