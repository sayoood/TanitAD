"""Zero-GPU: measure the ccos goal-term scale against the curvature penalty,
from the BANKED decisions sidecars. Cost in those arms is the GOAL TERM ALONE
(W_JERK=0, W_KAPPA=0, W_VEND multiplies a term `_cost_chunk` never adds because
`target_speed is None` -- MEASURED: 0 hits for `target_speed` in refav1_arm.py,
controls plan_cfg=4 / cost_weights=12 NON-ZERO)."""
import glob, os, sys
import numpy as np

def rows(dumpdir):
    out = []
    for f in sorted(glob.glob(os.path.join(dumpdir, "decisions", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        c = z["cl_controls"]                      # [w, K, 2] (accel, kappa)
        out.append(dict(
            ep=os.path.basename(f),
            kap=c[..., 1], acc=c[..., 0],
            plan=z["plan_cost_cl"], cv=z["basecost_cv_cl"],
            prop=z["basecost_proposal_cl"], dec=z["basecost_decel_1.5_cl"],
            src=z["plan_source_cl"], lat=z["goal_lat_cl"]))
    return out

def main(dumpdir, label):
    R = rows(dumpdir)
    if not R:
        print("NO DECISIONS in", dumpdir); return
    kap = np.concatenate([r["kap"] for r in R], 0)      # [W, K]
    plan = np.concatenate([r["plan"] for r in R])
    cv = np.concatenate([r["cv"] for r in R])
    prop = np.concatenate([r["prop"] for r in R])
    dec = np.concatenate([r["dec"] for r in R])
    lat = np.concatenate([r["lat"] for r in R])
    W = kap.shape[0]
    mk2 = (kap ** 2).mean(-1)                           # mean(kappa^2) per window
    absk = np.abs(kap).max(-1)
    print("==== %s  (n_windows=%d, K=%d)" % (label, W, kap.shape[1]))
    print("  GOAL TERM (= the whole cost in this arm)")
    print("    winner  plan_cost : median %.6g  mean %.6g  max %.6g" % (np.median(plan), plan.mean(), plan.max()))
    print("    cv      basecost  : median %.6g  mean %.6g  min %.6g" % (np.median(cv), cv.mean(), cv.min()))
    gap = cv - plan
    print("    GAP cv - winner   : median %.6g  mean %.6g  p10 %.6g  p90 %.6g" % (
        np.median(gap), gap.mean(), np.percentile(gap, 10), np.percentile(gap, 90)))
    spread = np.ptp(np.stack([cv, prop, dec]), axis=0)
    print("    spread over the 3 named baselines: median %.6g mean %.6g" % (np.median(spread), spread.mean()))
    print("  REALISED CURVATURE")
    print("    max|kappa| per window: median %.6g  frac at cap(0.2) %.4f  frac >0 %.4f" % (
        np.median(absk), float((absk >= 0.1999).mean()), float((absk > 1e-9).mean())))
    print("    mean(kappa^2)        : median %.6g  mean %.6g  max %.6g" % (
        np.median(mk2), mk2.mean(), mk2.max()))
    print("  PENALTY AT A GIVEN W_KAPPA vs the MEDIAN goal gap %.6g" % np.median(gap))
    for w in (0.05, 0.5, 2.0, 10.0, 32.148575, 64.297150):
        pen_sat = w * 0.04                              # kappa == kappa_max == 0.2
        pen_real = w * float(np.median(mk2))
        pen_corpus = w * (0.01 ** 2)                    # R = 100 m, a real road curve
        print("    W_KAPPA=%-10.4g  charge@cap %.5g  charge@realised %.5g  charge@R100m %.5g"
              % (w, pen_sat, pen_real, pen_corpus))
    print("  LANE_KEEP decode frac (goal_lat==0): %.4f" % float((lat == 0).mean()))
    print("  plan_source: cem %.3f  baseline %.3f" % (
        float((np.concatenate([r['src'] for r in R]) == 0).mean()),
        float((np.concatenate([r['src'] for r in R]) != 0).mean())))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
