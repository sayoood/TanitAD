"""P3 - WHY DOES THE EXTENSION MAKE CURVATURE MAE WORSE?  (four-families duty)

P1 MEASURED, on the 1,297 strict lane-change windows, that the extension
IMPROVES cross-track (1.1617 -> 0.9533 m) while DEGRADING curvature MAE
(0.002784 -> 0.004095 1/m) and heading (2.726 -> 3.082 deg). ⛔ A trade like that
is exactly why the four families are reported separately and never pooled, and it
must be EXPLAINED, not reported as noise.

THE HYPOTHESIS, stated before the numbers: the two-segment candidate flips its
lateral sign INSTANTANEOUSLY, and at t_split = 2.0 s that flip lands exactly on a
SLOT BOUNDARY (slot 3 = 20 ticks), so the 8-slot polyline sees a kink and reads a
curvature spike the human's smooth manoeuvre does not have.

THE DISCRIMINATOR: split the windows by WHETHER THE PICK CHANGED. If the
degradation is concentrated in the windows that repicked onto a new candidate,
the kink is the cause; if it is spread over windows whose pick did not change,
something else is (and the arithmetic is wrong somewhere).
"""
import argparse
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
from tanitad.refs import anchor_twoseg as ts    # noqa: E402
import p1_twoseg_supply as P                    # noqa: E402
import p2_split_sweep as S                      # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sd = torch.load(P.CKPT, map_location="cpu", weights_only=False)
    C0 = sd.get("model", sd)["core.decoder.anchor_controls"].float()
    n_base = C0.shape[0]
    EXT = ts.extend_controls(C0, P.T_S)

    GT, V0, TAG, _ = S.P_load(a)
    n = len(V0)
    LC = TAG[:, 0] == 1
    fb = P.roll(C0, V0, P.SLOTS)
    fe = P.roll(EXT, V0, P.SLOTS)
    ab, ib = P.best(fb, GT)
    ae, ie = P.best(fe, GT)
    pb = fb[np.arange(n), ib]
    pe = fe[np.arange(n), ie]
    repick = ie >= n_base

    res = {"n_scoreable_windows": int(n), "n_lane_change": int(LC.sum()),
           "n_repick_all": int(repick.sum()),
           "n_repick_lane_change": int((repick & LC).sum()), "strata": {}}

    print(f"{'stratum':<38}{'n':>7}{'ADE b':>9}{'ADE e':>9}{'curv b':>10}"
          f"{'curv e':>10}{'cross b':>9}{'cross e':>9}{'head b':>8}"
          f"{'head e':>8}")
    strata = {
        "lane_change, REPICKED": LC & repick,
        "lane_change, pick UNCHANGED": LC & ~repick,
        "all, REPICKED": repick,
        "all, pick UNCHANGED": ~repick,
    }
    for name, m in strata.items():
        if not m.any():
            continue
        fbm = P.families(pb[m], GT[m], P.SLOT_DT)
        fem = P.families(pe[m], GT[m], P.SLOT_DT)
        row = {"n": int(m.sum())}
        for k in ("ade_m", "along_mae_m", "cross_mae_m", "speed_mae_mps",
                  "curv_mae_1pm", "heading_mae_deg"):
            row[k + "_base"] = float(np.nanmean(fbm[k]))
            row[k + "_ext"] = float(np.nanmean(fem[k]))
        res["strata"][name] = row
        print(f"{name:<38}{row['n']:>7d}{row['ade_m_base']:>9.4f}"
              f"{row['ade_m_ext']:>9.4f}{row['curv_mae_1pm_base']:>10.6f}"
              f"{row['curv_mae_1pm_ext']:>10.6f}"
              f"{row['cross_mae_m_base']:>9.4f}{row['cross_mae_m_ext']:>9.4f}"
              f"{row['heading_mae_deg_base']:>8.3f}"
              f"{row['heading_mae_deg_ext']:>8.3f}")

    # ⛔ CONTROL AT A KNOWN VALUE: on windows whose pick did NOT change the two
    # paths are the SAME OBJECT, so every family must read EXACTLY equal.
    same = np.array_equal(pb[~repick], pe[~repick])
    print(f"\n[control] pick-unchanged paths identical: "
          f"{'EXACT' if same else 'DIFFER -- THE ARITHMETIC IS WRONG'}")
    res["control_unchanged_paths_identical"] = bool(same)

    # where the curvature spike sits: per-segment |kappa_pred - kappa_gt|
    m = LC & repick
    Pg = P.path_geom(pe[m], P.SLOT_DT)
    Gg = P.path_geom(GT[m], P.SLOT_DT)
    Bg = P.path_geom(pb[m], P.SLOT_DT)
    both = Pg["kappa_ok"] & Gg["kappa_ok"]
    de = np.abs(Pg["kappa"] - Gg["kappa"]) * both
    db = np.abs(Bg["kappa"] - Gg["kappa"]) * (Bg["kappa_ok"] & Gg["kappa_ok"])
    cnt = both.sum(0)
    print(f"\n[kink] per-segment curvature |err| on the {int(m.sum())} "
          f"repicked lane-change windows (segment j sits between slots j and "
          f"j+1; the t_split=2.0 s flip lands on slot 3):")
    print(f"  {'seg':>4}{'t_lo':>7}{'t_hi':>7}{'n_valid':>9}{'base':>11}"
          f"{'ext':>11}{'delta':>11}")
    segs = []
    for j in range(de.shape[1]):
        nb = max(int(cnt[j]), 1)
        b = float(db[:, j].sum() / max(int((Bg['kappa_ok'] &
                                            Gg['kappa_ok']).sum(0)[j]), 1))
        e = float(de[:, j].sum() / nb)
        segs.append({"seg": j, "t_lo": float(P.SLOT_T[j]),
                     "t_hi": float(P.SLOT_T[j + 1]), "n_valid": int(cnt[j]),
                     "base": b, "ext": e, "delta": e - b})
        print(f"  {j:>4d}{P.SLOT_T[j]:>7.1f}{P.SLOT_T[j+1]:>7.1f}"
              f"{int(cnt[j]):>9d}{b:>11.6f}{e:>11.6f}{e-b:>11.6f}")
    res["kink_per_segment"] = segs
    tot = sum(s["delta"] for s in segs)
    top = max(segs, key=lambda s: s["delta"])
    print(f"\n  the single worst segment is {top['t_lo']:.1f}-{top['t_hi']:.1f} s "
          f"(delta {top['delta']:+.6f}), which is "
          f"{100.0 * top['delta'] / tot if tot else float('nan'):.1f} % of the "
          f"total per-segment degradation")
    res["worst_segment"] = top
    res["total_segment_delta"] = tot

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
