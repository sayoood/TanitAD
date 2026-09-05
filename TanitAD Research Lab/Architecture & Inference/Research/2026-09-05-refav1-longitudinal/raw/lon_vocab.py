"""P1a - WHAT THE LONGITUDINAL VOCABULARY CAN COMMAND. Zero GPU, no model.

The lateral analogue (M15 / D-REFAV1-VOCAB-QUANT) asked: what sustained
curvatures can `canonical_controls` express? This asks the same question of the
LON channel, from the SOURCE, never from prose.
"""
import sys, numpy as np, torch
from tanitad.refs import refa_v1 as R

DT, K = 0.2, 10          # plan grid: plan_horizon_s 2.0 / op_dt 0.2
OP = 30                  # canonical_controls' own op_steps (6.0 s manoeuvre)
LON = list(R.TAC_LON_ACTIONS) if hasattr(R, "TAC_LON_ACTIONS") else None
print("# SOURCE: stack/tanitad/refs/refa_v1.py::canonical_controls")
print("# GOAL_REACH_S=%.2f s  GOAL_A_MAX=%.2f m/s2  GOAL_CREEP_MPS=%.2f  "
      "GOAL_CURVE_VMAX_MPS=%.2f" % (R.GOAL_REACH_S, R.GOAL_A_MAX,
                                    R.GOAL_CREEP_MPS, R.GOAL_CURVE_VMAX_MPS))
print("# GOAL_LON_DV_MPS = %r" % (R.GOAL_LON_DV_MPS,))
print()
from tanitad.models import vocab_v7 as V
toks = None
for name in ("LON_ACTIONS", "TACTICAL_LON", "LON_TOKENS", "LON"):
    if hasattr(V, name):
        toks = list(getattr(V, name)); print("# token list from vocab_v7.%s" % name); break
if toks is None:
    toks = ["HOLD", "CREEP", "ADAPT_SPEED_FOR_CURVE", "CRUISE", "FOLLOW",
            "ACCELERATE", "YIELD_MERGE", "BRAKE_TO"]
    print("# token list HARDCODED fallback (vocab_v7 name not found)")
print("# tokens:", toks)
print()

print("== TABLE A. The whole longitudinal vocabulary, at three v0. ==")
print("#  a0      = the FIRST commanded acceleration (m/s2)")
print("#  a_end   = the acceleration at the END of the 2.0 s plan window")
print("#  dv_2s   = speed change realised over the 2.0 s PLAN horizon")
print("#  dv_6s   = speed change over the token's full 6.0 s manoeuvre")
print("#  sust    = |a| still commanded at 2.0 s / |a| at t=0  (0 => decays away)")
hdr = "%-24s" % "token" + "".join("  %9s" % h for h in
      ("a0", "a_end", "dv_2s", "dv_6s", "sust"))
for v0 in (3.0, 10.0, 20.0):
    print("\n-- v0 = %.1f m/s --" % v0)
    print(hdr)
    for lon in toks:
        c = R.canonical_controls("LANE_KEEP", lon, v0, OP, DT)
        a = c[:, 0].numpy()
        dv2 = float(a[:K].sum() * DT); dv6 = float(a.sum() * DT)
        sust = abs(a[K - 1]) / max(abs(a[0]), 1e-12)
        print("%-24s" % lon + "".join("  %9.4f" % x for x in
              (a[0], a[K - 1], dv2, dv6, sust)))
print()
print("== TABLE B. The SUSTAINED-acceleration capability of the vocabulary ==")
print("# The lateral question was 'what sustained curvature?'. Here: after the")
print("# 2 s plan window, what acceleration is the token still commanding?")
print("# EVERY token converges to a==0 because a_i = (v_t - v)/REACH_S and v -> v_t.")
for v0 in (10.0,):
    for lon in toks:
        c = R.canonical_controls("LANE_KEEP", lon, v0, OP, DT)
        a = c[:, 0].numpy()
        print("  %-24s a[0]=%+8.4f a[9]=%+8.4f a[29]=%+8.4f  max|a|=%.4f"
              % (lon, a[0], a[9], a[29], np.abs(a).max()))
print()
print("== TABLE C. The REACHABLE SET of dv over the 2 s plan window ==")
print("# This is the longitudinal analogue of 'kappa in {0, +-0.08}'.")
for v0 in (3.0, 10.0, 20.0):
    vals = sorted({round(float(R.canonical_controls('LANE_KEEP', l, v0, OP, DT)[:K, 0].sum() * DT), 6)
                   for l in toks})
    print("  v0=%5.1f  distinct dv_2s = %s   (n=%d)" % (v0, vals, len(vals)))
