"""Close panel 1: is the round-trip EXACT when seeded consistently?

Hypothesis from the source: `unicycle_controls_from_path*` recovers speed FROM
THE PATH (`speed = ds/dt`) and its docstring says in writing that ENTRY SPEED IS
NOT AN INPUT. Seeding the rollout with the ego's MEASURED v0 therefore injects
`entry_speed_mismatch` into the reconstruction. Seeding with the path-implied
speed[0] should round-trip exactly.

If that is right: the kernel is exact, AND `unicycle_decode` does not reproduce
its own base anchor whenever v0 differs from the anchor's implied entry speed —
a design fact worth stating, not a bug.
"""
import sys
import torch

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/TanitAD/stack")

import build_anchors6s as B                                   # noqa: E402
from tanitad.models import kinematic as K                     # noqa: E402

TRAIN = "/workspace/TanitAD/data/b1-train-v72"
SLOTS = [5, 10, 15, 20, 30, 40, 50, 60]
DENSE = list(range(1, 61))
LIM = 60

d = K.slot_dts(SLOTS)
gt_s, v0, _ = B.corpus_pool(TRAIN, SLOTS, limit=LIM, tag="slots")
gt_d, v0d, _ = B.corpus_pool(TRAIN, DENSE, limit=LIM, tag="dense")


def implied_v0(path, dts):
    dts = torch.as_tensor(dts, dtype=path.dtype).reshape(1, -1)
    d0 = path[:, :1] - torch.zeros_like(path[:, :1])
    return (d0.pow(2).sum(-1) + 1e-12).sqrt()[:, 0] / dts[0, 0]


def rt(path, dts, v_seed, varstep):
    c = (K.unicycle_controls_from_path_varstep(path, dts) if varstep
         else K.unicycle_controls_from_path(path, dts))
    z = torch.zeros_like(v_seed)
    st0 = torch.stack([z, z, z, v_seed], dim=-1)
    out = (K.rollout_unicycle_varstep(st0, c, dts) if varstep
           else K.rollout_unicycle(st0, c, dts))
    return (out[..., :2] - path).norm(dim=-1)


print("=" * 78)
print("PANEL 1' — round-trip seeded with the PATH-IMPLIED entry speed")
print("=" * 78)
n = gt_s.shape[0]
print(f"  n = {n} windows")

vs = implied_v0(gt_s, d)
e = rt(gt_s, d, vs, True)
print(f"  (a) varstep, non-uniform slots, implied v0 : mean {e.mean():.3e} m  "
      f"max {e.max():.3e} m  {'✅ EXACT' if e.mean() < 1e-4 else '⛔'}")

vd = implied_v0(gt_d, [0.1] * 60)
e2 = rt(gt_d, 0.1, vd, False)
print(f"  (b) scalar,  uniform 10 Hz,     implied v0 : mean {e2.mean():.3e} m  "
      f"max {e2.max():.3e} m  {'✅ EXACT' if e2.mean() < 1e-4 else '⛔'}")

# --- the first-heading limit -------------------------------------------------
# A unicycle seeded at yaw=0 displaces step 0 along +x BY CONSTRUCTION (yaw is
# updated AFTER the displacement). In ego frame a turning vehicle's first chord
# has a non-zero heading, so that heading is INEXPRESSIBLE — and the coarser the
# first step, the larger the resulting error. This is a property of the model,
# not of the inverse map, and it is an argument for emitting at the native tick.
print()
print("  first-chord heading |h0| (rad), by grid:")
for nm, p, dts in (("slot grid (first step 0.5 s)", gt_s, d),
                   ("10 Hz grid (first step 0.1 s)", gt_d, [0.1] * 60)):
    d0 = p[:, 0]
    h0 = torch.atan2(d0[:, 1], d0[:, 0]).abs()
    print(f"    {nm:<32} p50 {h0.quantile(0.5):.5f}  p95 {h0.quantile(0.95):.5f}"
          f"  p99 {h0.quantile(0.99):.5f}")

print()
print("=" * 78)
print("PANEL 5 — what seeding with the MEASURED v0 costs (the decode's real behaviour)")
print("=" * 78)
mism = (v0 - vs).abs()
print(f"  |v0_measured - v0_implied|  n={mism.numel()}  p50 {mism.quantile(0.5):.4f}"
      f"  p95 {mism.quantile(0.95):.4f}  p99 {mism.quantile(0.99):.4f} m/s")
e3 = rt(gt_s, d, v0, True)
print(f"  round-trip seeded with measured v0        : mean {e3.mean():.4f} m  "
      f"p99 {e3.flatten().quantile(0.99):.4f} m")
print("  => unicycle_decode does NOT reproduce its base anchor when v0 differs")
print("     from the anchor's implied entry speed. That is `entry_speed_mismatch`")
print("     being applied, not a defect — but it means the 'model starts exactly")
print("     at the anchor' property of a POSITION-space residual does NOT carry")
print("     over to the control-space decode. State it in the prereg.")

print()
print("=" * 78)
print("PANEL 6 — is the DATA-DRIVEN vocabulary feasible? (control: human GT)")
print("=" * 78)
anch = torch.load("/workspace/refc_anchors_6s_b1train_128.pt",
                  map_location="cpu").float()
print(f"  anchors {tuple(anch.shape)}")
av = implied_v0(anch, d)
ac = K.unicycle_controls_from_path_varstep(anch, d)
gc = K.unicycle_controls_from_path_varstep(gt_s, d)
p99j = 5.3864
p99c = 0.3484
for nm, c_, nn in (("DATA-DRIVEN anchors (128)", ac, anch.shape[0]),
                   ("CONTROL: real human GT   ", gc, gt_s.shape[0])):
    L = K.control_smoothness_losses(c_, d, jerk_limit=p99j, curvature_rate_limit=p99c)
    a = c_[..., 0].abs()
    k = c_[..., 1].abs()
    print(f"  {nm} n={nn:>6}  |accel| p99 {a.flatten().quantile(0.99):7.4f}  "
          f"|kappa| p99 {k.flatten().quantile(0.99):7.4f}  "
          f"barrier jerk {float(L['jerk_lon']):8.4f}  curv {float(L['curv_rate']):7.4f}")
print("  (slot-grid controls straddle the 2 s seam — comparable to each other,")
print("   NOT to the 10 Hz p99 limits above. Same-grid comparison only.)")
print()
print("DONE")
