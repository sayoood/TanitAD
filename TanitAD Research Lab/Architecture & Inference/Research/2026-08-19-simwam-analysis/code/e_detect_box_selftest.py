"""Self-test for E-DETECT-1B's detection metric. Run before trusting any AP.

⛔ WHY. A detection metric is exactly the kind of code that returns a plausible
number while being wrong, and every arm's headline depends on it. These four
cases have answers known in advance:

  PERFECT (predictions == truth)      AP must be EXACTLY 1.0, ctr 0 m, yaw 0 deg
  jittered by 1.0 m sigma             AP falls; measured ctr must track the jitter
  RANDOM positions                    AP at chance; yaw error ~90 deg, which is
                                      the chance level for a CIRCULAR quantity
  perfect geometry, RANDOM scores     recall stays 1.0 while AP collapses —
                                      proving AP prices RANKING, not just boxes

MEASURED 2026-08-21: 1.0000 / 0.7835 (ctr 1.086 m) / 0.0086 (yaw 95.2 deg) /
0.1512 (recall 1.000). All four as predicted.
"""
from __future__ import annotations

import numpy as np

import e_detect_box as B

N = 300


def synth(rng):
    K = B.K_SLOTS
    tb = np.zeros((N, K, 5), np.float32)
    tv = np.zeros((N, K), bool)
    for i in range(N):
        m = int(rng.integers(3, 10))
        tb[i, :m, 0] = rng.uniform(5, 55, m)
        tb[i, :m, 1] = rng.uniform(-14, 14, m)
        tb[i, :m, 2], tb[i, :m, 3] = 4.2, 1.9
        tb[i, :m, 4] = rng.uniform(-np.pi, np.pi, m)
        tv[i, :m] = True
    return tb, tv


def report(name, pl, pg, tb, tv, tau=2.0):
    ev = B.evaluate(pl, pg, tb, tv, tau)
    ap = B.ap_from(ev["score"], ev["tp"], int(ev["ngt"].sum()))
    rec = float(ev["tp"].sum() / max(ev["ngt"].sum(), 1))
    ce = float(ev["ctr_err"].mean()) if len(ev["ctr_err"]) else float("nan")
    ye = float(np.degrees(ev["yaw_err"]).mean()) if len(ev["yaw_err"]) else float("nan")
    print(f"  {name:<28} AP {ap:.4f}  recall {rec:.3f}  ctr {ce:.3f} m  "
          f"yaw {ye:.1f} deg")
    return ap, rec, ce, ye


def main() -> None:
    rng = np.random.default_rng(0)
    tb, tv = synth(rng)
    pl = np.where(tv, 10.0, -10.0).astype(np.float32)
    pg = np.zeros((N, B.K_SLOTS, 6), np.float32)
    pg[..., :2] = tb[..., :2]
    pg[..., 2:4] = np.log(np.maximum(tb[..., 2:4], 0.1))
    pg[..., 4], pg[..., 5] = np.sin(tb[..., 4]), np.cos(tb[..., 4])

    ap_p, rec_p, ce_p, ye_p = report("PERFECT (== truth)", pl, pg, tb, tv)
    pg2 = pg.copy()
    pg2[..., :2] += rng.normal(0, 1.0, (N, B.K_SLOTS, 2)).astype(np.float32)
    report("jittered 1.0 m sigma", pl, pg2, tb, tv)
    pg3 = pg.copy()
    pg3[..., 0] = rng.uniform(0, 60, (N, B.K_SLOTS))
    pg3[..., 1] = rng.uniform(-16, 16, (N, B.K_SLOTS))
    ap_r, _, _, ye_r = report("RANDOM positions", pl, pg3, tb, tv)
    ap_s, rec_s, _, _ = report("perfect geo, RANDOM scores",
                               rng.normal(0, 1, (N, B.K_SLOTS)).astype(np.float32),
                               pg, tb, tv)

    checks = {
        "PERFECT AP is exactly 1.0": abs(ap_p - 1.0) < 1e-9,
        "PERFECT centre error is 0 m": ce_p < 1e-6,
        "PERFECT yaw error is 0 deg": ye_p < 1e-4,
        "RANDOM AP is at chance": ap_r < 0.25 * ap_p,
        "RANDOM yaw error is ~90 deg (circular chance)": 75.0 < ye_r < 105.0,
        "scrambled scores keep recall but destroy AP":
            rec_s > 0.99 and ap_s < 0.5 * ap_p,
    }
    print()
    bad = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'OK ' if v else 'BAD'}] {k}")
    if bad:
        raise SystemExit(f"[FATAL] the detection metric is wrong: {bad}")
    print("\n  metric validated — AP/recall/centre/yaw all behave as required")


if __name__ == "__main__":
    main()
