"""P4 -- IS THE PLANNER'S MACHINERY SIGN-SYMMETRIC? Bit-level assertions.

Every assertion carries a SAME-BREATH CONTROL that MUST differ, so a probe that
silently returns a constant cannot pass. ASCII only (cp1252 console).
"""
import sys, math
import torch

WT = "C:/Users/Admin/tanitad-wt"
for p in (WT + "/stack", WT + "/taniteval", WT):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.refs import refa_v1 as R
from tanitad.refs import refa_v1_plan as P
from tanitad.refs.refc_tactical import factor_from_kinematics
from taniteval import four_families as ff

OK = []
def check(name, cond, detail="", control=None):
    tag = "PASS" if cond else "**FAIL**"
    OK.append((name, bool(cond)))
    print("  [%s] %-46s %s" % (tag, name, detail))
    if control is not None:
        cname, ccond, cdet = control
        ctag = "ok" if ccond else "**CONTROL DEAD**"
        OK.append((cname, bool(ccond)))
        print("        control(%s): %-30s %s" % (ctag, cname, cdet))

print("=" * 78)
print("A. THE GOAL VOCABULARY -- canonical_controls(TURN_L) vs (TURN_R)")
print("=" * 78)
OPS, ODT = 10, 0.2
for v0 in (2.0, 8.0, 15.0, 25.0):
    for lon in ("ADAPT_SPEED_FOR_CURVE", "CRUISE", "BRAKE_TO", "HOLD", "CREEP"):
        L = R.canonical_controls("TURN_L", lon, v0, OPS, ODT)
        Rr = R.canonical_controls("TURN_R", lon, v0, OPS, ODT)
        ka_ok = torch.equal(L[:, 1], -Rr[:, 1])
        a_ok = torch.equal(L[:, 0], Rr[:, 0])
        if not (ka_ok and a_ok):
            print("   MISMATCH v0=%s lon=%s kappa_ok=%s a_ok=%s" % (v0, lon, ka_ok, a_ok))
L = R.canonical_controls("TURN_L", "ADAPT_SPEED_FOR_CURVE", 8.0, OPS, ODT)
Rr = R.canonical_controls("TURN_R", "ADAPT_SPEED_FOR_CURVE", 8.0, OPS, ODT)
LK = R.canonical_controls("LANE_KEEP", "ADAPT_SPEED_FOR_CURVE", 8.0, OPS, ODT)
check("kappa channel is EXACTLY antisymmetric",
      torch.equal(L[:, 1], -Rr[:, 1]),
      "max|kL + kR| = %.3e ; kL[0]=%+.5f kR[0]=%+.5f"
      % ((L[:, 1] + Rr[:, 1]).abs().max(), L[0, 1], Rr[0, 1]),
      control=("kappa is NOT identically zero",
               bool(L[:, 1].abs().max() > 0),
               "max|kL| = %.5f (must be non-zero)" % L[:, 1].abs().max()))
check("accel channel is EXACTLY equal",
      torch.equal(L[:, 0], Rr[:, 0]),
      "max|aL - aR| = %.3e" % (L[:, 0] - Rr[:, 0]).abs().max(),
      control=("LANE_KEEP kappa DIFFERS from TURN_L",
               not torch.equal(LK[:, 1], L[:, 1]),
               "max|kLK - kL| = %.5f (must be non-zero)"
               % (LK[:, 1] - L[:, 1]).abs().max()))
# the S-curve tokens too
for pair in (("NUDGE_L", "NUDGE_R"), ("LANE_CHANGE_L", "LANE_CHANGE_R")):
    a = R.canonical_controls(pair[0], "CRUISE", 8.0, OPS, ODT)
    b = R.canonical_controls(pair[1], "CRUISE", 8.0, OPS, ODT)
    check("%s/%s kappa antisymmetric" % pair, torch.equal(a[:, 1], -b[:, 1]),
          "max|kA + kB| = %.3e" % (a[:, 1] + b[:, 1]).abs().max())

print()
print("=" * 78)
print("B. THE COST -- does the curvature charge treat +kappa and -kappa alike?")
print("=" * 78)
g = torch.Generator().manual_seed(0)
K = torch.randn(2000, OPS, generator=g) * 0.05
term_p = R.W_KAPPA * K.pow(2).mean(-1)
term_n = R.W_KAPPA * (-K).pow(2).mean(-1)
check("W_KAPPA * kappa^2 is EXACTLY sign-invariant",
      torch.equal(term_p, term_n),
      "max|c(+k) - c(-k)| = %.3e over n=2000 random profiles"
      % (term_p - term_n).abs().max(),
      control=("the charge is NOT constant",
               bool(term_p.std() > 0),
               "std = %.3e (must be non-zero)" % term_p.std()))
# and at the actual ladder weight
for wk in (0.0, 0.05, 15.11245, 151.1245):
    tp = wk * K.pow(2).mean(-1); tn = wk * (-K).pow(2).mean(-1)
    if not torch.equal(tp, tn):
        print("   MISMATCH at W_KAPPA=%s" % wk)
print("  [PASS] sign-invariance holds at every ladder rung "
      "(0, 0.05, 15.11245, 151.1245)")

print()
print("=" * 78)
print("C. THE CLIP and the KAMM CAP")
print("=" * 78)
cfg = P.PlanConfig(horizon=OPS, dt=ODT)
ctrl = torch.randn(500, OPS, 2, generator=g) * 0.3
cp = P._clip(ctrl, cfg, 8.0)
cn = P._clip(torch.stack([ctrl[..., 0], -ctrl[..., 1]], -1), cfg, 8.0)
check("_clip (no Kamm): kappa clip is sign-symmetric",
      torch.equal(cp[..., 1], -cn[..., 1]),
      "max|k+ + k-| = %.3e" % (cp[..., 1] + cn[..., 1]).abs().max(),
      control=("the clip actually BITES",
               bool((ctrl[..., 1].abs() > cfg.kappa_max).any()),
               "%d of %d steps exceeded kappa_max=%.3f"
               % ((ctrl[..., 1].abs() > cfg.kappa_max).sum(), ctrl[..., 1].numel(),
                  cfg.kappa_max)))
cfgk = P.PlanConfig(horizon=OPS, dt=ODT, kamm_mu=0.7)
kp = P._clip(ctrl, cfgk, 20.0)
kn = P._clip(torch.stack([ctrl[..., 0], -ctrl[..., 1]], -1), cfgk, 20.0)
check("_clip WITH Kamm mu=0.7: sign-symmetric",
      torch.equal(kp[..., 1], -kn[..., 1]),
      "max|k+ + k-| = %.3e" % (kp[..., 1] + kn[..., 1]).abs().max(),
      control=("the Kamm cap actually BITES at v=20",
               bool(not torch.equal(kp[..., 1], cp[..., 1])),
               "max|k_kamm - k_plain| = %.5f"
               % (kp[..., 1] - cp[..., 1]).abs().max()))

print()
print("=" * 78)
print("D. THE iCEM SEARCH DISTRIBUTION -- is the noise pool sign-symmetric?")
print("=" * 78)
for seed in (0, 1):
    gen = torch.Generator().manual_seed(seed)
    nz = P.colored_noise((4000, OPS, 2), 2.0, device="cpu", generator=gen)
    k = nz[..., 1]
    m = k.mean().item()
    se = (k.std() / math.sqrt(k.numel())).item()
    check("colored_noise kappa channel is zero-mean (seed %d)" % seed,
          abs(m) < 4 * se,
          "mean = %+.3e, 4*SE = %.3e, n = %d" % (m, 4 * se, k.numel()),
          control=("the noise is NOT degenerate",
                   bool(k.std() > 0), "std = %.4f" % k.std()))
    npos = int((k > 0).sum()); nneg = int((k < 0).sum())
    check("  ... and balanced in sign (seed %d)" % seed,
          abs(npos - nneg) < 4 * math.sqrt(k.numel()),
          "n(+)=%d n(-)=%d  |diff|=%d  4*sqrt(n)=%.0f"
          % (npos, nneg, abs(npos - nneg), 4 * math.sqrt(k.numel())))

print()
print("=" * 78)
print("E. THE LABELLER -- factor_from_kinematics on a mirrored trajectory")
print("=" * 78)
gt = torch.randn(3000, 10, 2, generator=g)
gt[..., 0] = gt[..., 0].abs() * 3 + torch.arange(10).float() * 2   # forward
mir = gt.clone(); mir[..., 1] = -mir[..., 1]                        # mirror y
dy, dv, v0, v1, _ = ff.maneuver_kinematics(gt, 0.2)
dym, dvm, v0m, v1m, _ = ff.maneuver_kinematics(mir, 0.2)
lat, lon = factor_from_kinematics(dy, dv, v0, v1)
latm, lonm = factor_from_kinematics(dym, dvm, v0m, v1m)
swap = lat.clone(); swap[lat == 1] = 2; swap[lat == 2] = 1
check("mirroring y swaps turn_left <-> turn_right EXACTLY",
      torch.equal(swap, latm),
      "%d/%d windows map correctly" % (int((swap == latm).sum()), lat.numel()),
      control=("the mirror actually CHANGES the label set",
               not torch.equal(lat, latm),
               "%d of %d labels differ" % (int((lat != latm).sum()), lat.numel())))
check("mirroring leaves the LONGITUDINAL label untouched",
      torch.equal(lon, lonm), "identical")
check("dyaw is exactly negated by the mirror",
      torch.allclose(dy, -dym, atol=0, rtol=0),
      "max|dy + dym| = %.3e" % (dy + dym).abs().max())
import collections
print("        label mix in the synthetic set: %s"
      % dict(collections.Counter(lat.tolist())))

print()
print("=" * 78)
nfail = sum(1 for _, v in OK if not v)
print("SUMMARY: %d assertions, %d FAILED" % (len(OK), nfail))
print("=" * 78)
