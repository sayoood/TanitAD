"""plan_roughness.py -- descriptive, names nothing (convention in TIMES_NOTE.md before the first run). From the banked no-rule dumps of card ADD46 (I10R0, two model seeds):
the first-tick plan of every window (ol, 30 steps of 0.2 s) and the logged track of the same window (g). For a hand-over after m = 5 and after m = 10 steps, per window:
  d_next = s_m+1 - s_m (the plan's next, never executed step change), d_last = s_m - s_m-1 (the last executed step change), with s the step speeds along the plan
  the acceleration the model would read at the first replan under each readout (closed forms of integ10k/check_tick_readout.py and check_tick_readout_rz.py):
    pinned  (0.625 d_next - 0.125 d_last) / 0.1 s      causal  d_last / 0.2 s      window  (s_m - s_m-5) / 1.0 s
  and the speed offset of the pinned readout, d_next / 2.
The same quantities with the logged track in place of the plan (human motion) are printed beside, and the acceleration of the logged track over the 1.0 s after the hand-over
as the reference the model's next plan should follow. Only the first replan can be computed from first-tick plans. Tier: a property of plans at T0 and of the tool's
readout formulas; no rollout number. Usage: python plan_roughness.py > PLAN_ROUGHNESS.txt"""
import glob

import numpy as np

W = "/workspace/v8proof/integ10k/t1"
DT = 0.2


def load(k):
    out = {n: [] for n in ("g", "ol", "v0")}
    files = sorted(glob.glob("%s/I10R0_s%d/dump_I10R0_s%d_m10_causal_heads/ep*.npz" % (W, k, k)))
    assert len(files) == 141
    for f in files:
        z = np.load(f)
        for n in out:
            out[n].append(np.asarray(z[n], np.float64))
    return {n: np.concatenate(v) for n, v in out.items()}


def step_speeds(tr):
    p = np.concatenate([np.zeros((len(tr), 1, 2)), tr], 1)
    return np.linalg.norm(np.diff(p, axis=1), axis=-1) / DT                 # [N, 30]: s_1 .. s_30


def stats(x):
    return "mean %+7.3f median %+7.3f p10 %+7.3f p90 %+7.3f | share above +1: %.3f, below -1: %.3f" % (float(x.mean()), float(np.median(x)), float(np.quantile(x, 0.1)), float(np.quantile(x, 0.9)),
                                                                                                         float((x > 1).mean()), float((x < -1).mean()))


for k in (0, 1):
    d = load(k)
    for src in ("ol", "g"):
        s = step_speeds(d[src])
        print("model seed %d, %s (2,399 windows)" % (k, "first-tick plan of the model" if src == "ol" else "logged track (human motion)"))
        print("   mean |step-to-step speed change| over the 29 changes: %.4f m/s ; mean step speed %.3f m/s" % (float(np.abs(np.diff(s, axis=1)).mean()), float(s.mean())))
        for m in (5, 10):
            d_next, d_last = s[:, m] - s[:, m - 1], s[:, m - 1] - s[:, m - 2]          # s[:, i] = s_{i+1}
            a_pin = (0.625 * d_next - 0.125 * d_last) / 0.1
            a_cau = d_last / DT
            a_win = (s[:, m - 1] - s[:, m - 6]) / 1.0 if m > 5 else (s[:, m - 1] - d["v0"]) / 1.0     # s_0 = the ego speed at the tick (logged)
            print("   hand-over after %2d steps: d_next (m/s)        %s" % (m, stats(d_next)))
            print("                             a0 pinned (m/s2)    %s" % stats(a_pin))
            print("                             a0 causal (m/s2)    %s" % stats(a_cau))
            print("                             a0 window (m/s2)    %s" % stats(a_win))
            if src == "ol":
                sg = step_speeds(d["g"])
                ref = (sg[:, min(m + 4, 29)] - sg[:, m - 1]) / (DT * (min(m + 4, 29) - (m - 1)))
                print("                             logged acceleration over the 1.0 s after it (m/s2) %s" % stats(ref))
                for nm, a in (("pinned", a_pin), ("causal", a_cau), ("window", a_win)):
                    print("                             a0 %s minus that logged acceleration: mean %+.3f, mean abs %.3f, correlation %+.3f" % (
                        nm, float((a - ref).mean()), float(np.abs(a - ref).mean()), float(np.corrcoef(a, ref)[0, 1])))
print("ZZ PLAN-ROUGHNESS DONE")
