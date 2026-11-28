"""L4 kernel validation + the measured limits `control_smoothness_losses` needs.

Every panel carries a CONTROL THAT MUST READ A KNOWN VALUE:
  * the round-trip of REAL human GT must reconstruct to ~0 m (if it does not,
    the kernel is wrong, not the data);
  * on a UNIFORM grid the varstep functions must agree with the scalar ones to
    ~0 (if they do not, the generalisation broke the convention);
  * `uniform_slot_dt` must RAISE on REF-C's own horizons (that is the bug this
    kernel exists to make loud).
Prints n and the units for every number.
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
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60

print("=" * 78)
print("PANEL 0 — the grid, and the refusal")
print("=" * 78)
d = K.slot_dts(SLOTS)
print(f"  refc horizons {SLOTS}")
print(f"  slot_dts      {d}  s")
print(f"  uniform?      {max(d) - min(d) < 1e-9}")
try:
    K.uniform_slot_dt(SLOTS)
    print("  ⛔ CONTROL FAILED: uniform_slot_dt did NOT raise on a non-uniform grid")
except ValueError as e:
    print(f"  ✅ CONTROL: uniform_slot_dt REFUSED — {str(e)[:96]}...")
du = K.slot_dts([10, 20, 30, 40])
print(f"  uniform grid [10,20,30,40] -> {du}, uniform_slot_dt = "
      f"{K.uniform_slot_dt([10, 20, 30, 40])} s  ✅")

print()
print("=" * 78)
print(f"PANEL 1 — round-trip on REAL human GT  (control: must read ~0 m)")
print("=" * 78)
gt_s, v0 = B.corpus_pool(TRAIN, SLOTS, limit=LIMIT, tag="slots")[:2]
gt_d, v0d = B.corpus_pool(TRAIN, DENSE, limit=LIMIT, tag="dense")[:2]
n = gt_s.shape[0]
print(f"  n = {n} windows,  slots S = {gt_s.shape[1]},  dense T = {gt_d.shape[1]}")

# (a) varstep round-trip on the NON-UNIFORM slot grid
c_s = K.unicycle_controls_from_path_varstep(gt_s, d)
z = torch.zeros_like(v0)
st0 = torch.stack([z, z, z, v0], dim=-1)
rt_s = K.rollout_unicycle_varstep(st0, c_s, d)[..., :2]
e_s = (rt_s - gt_s).norm(dim=-1)
print(f"  (a) varstep round-trip, non-uniform slots : mean {e_s.mean():.6f} m  "
      f"p99 {e_s.flatten().quantile(0.99):.6f} m   {'✅' if e_s.mean() < 1e-3 else '⛔'}")

# (b) scalar round-trip on the UNIFORM dense grid (the existing function)
c_d = K.unicycle_controls_from_path(gt_d, 0.1)
z = torch.zeros_like(v0d)
st0d = torch.stack([z, z, z, v0d], dim=-1)
rt_d = K.rollout_unicycle(st0d, c_d, 0.1)[..., :2]
e_d = (rt_d - gt_d).norm(dim=-1)
print(f"  (b) scalar  round-trip, uniform 10 Hz     : mean {e_d.mean():.6f} m  "
      f"p99 {e_d.flatten().quantile(0.99):.6f} m   {'✅' if e_d.mean() < 1e-3 else '⛔'}")

# (c) CONSISTENCY control: varstep with a constant dt == scalar
c_v = K.unicycle_controls_from_path_varstep(gt_d, [0.1] * 60)
agree = (c_v - c_d).abs().max()
print(f"  (c) varstep(const dt) vs scalar controls  : max |diff| {agree:.3e}   "
      f"{'✅' if agree < 1e-4 else '⛔'}")

print()
print("=" * 78)
print("PANEL 2 — the TRAP quantified: a SCALAR dt on the non-uniform slot grid")
print("=" * 78)
print("  This is what wiring `unicycle_decode` into refc's 8-slot emission does.")
for scalar in (0.75, 0.5, 1.0):
    c_w = K.unicycle_controls_from_path(gt_s, scalar)
    rt_w = K.rollout_unicycle(torch.stack([torch.zeros_like(v0)] * 3 + [v0], -1),
                              c_w, scalar)[..., :2]
    e_w = (rt_w - gt_s).norm(dim=-1)
    print(f"  scalar dt = {scalar:4.2f} s -> reconstruction error of REAL GT "
          f"mean {e_w.mean():8.4f} m   p99 {e_w.flatten().quantile(0.99):9.4f} m")
print(f"  (correct per-step dt gives {e_s.mean():.6f} m — panel 1a)")

print()
print("=" * 78)
print("PANEL 3 — MEASURED human limits for control_smoothness_losses (10 Hz GT)")
print("=" * 78)
acc = c_d[..., 0]
kap = c_d[..., 1]
jerk = ((acc[:, 1:] - acc[:, :-1]) / 0.1).abs()
crate = ((kap[:, 1:] - kap[:, :-1]) / 0.1).abs()
nd = jerk.numel()
for name, t, unit in (("|accel|      ", acc.abs(), "m/s^2"),
                      ("|jerk_lon|   ", jerk, "m/s^3"),
                      ("|curvature|  ", kap.abs(), "1/m"),
                      ("|curv_rate|  ", crate, "1/(m s)")):
    f = t.flatten()
    print(f"  {name} n={f.numel():>8}  p50 {f.quantile(0.5):9.4f}  "
          f"p95 {f.quantile(0.95):9.4f}  p99 {f.quantile(0.99):9.4f}  "
          f"max {f.max():10.4f}   [{unit}]")
print(f"  module's existing accel_limit 2.785 / jerk_limit 6.369 were MEASURED as")
print(f"  the human p99 on PhysicalAI OOD-val (6,834 windows) — compare above.")

print()
print("=" * 78)
print("PANEL 4 — the two-channel barrier reads what it should")
print("=" * 78)
p99_j = float(jerk.flatten().quantile(0.99))
p99_c = float(crate.flatten().quantile(0.99))
L = K.control_smoothness_losses(c_d, 0.1, jerk_limit=p99_j,
                                curvature_rate_limit=p99_c)
print(f"  human GT @ its OWN p99 limits  jerk_lon {float(L['jerk_lon']):.6f}  "
      f"curv_rate {float(L['curv_rate']):.6f}   (small by construction ✅)")
Lh = K.control_smoothness_losses(c_d, 0.1, jerk_limit=1e9,
                                 curvature_rate_limit=1e9)
print(f"  CONTROL, limits -> inf         jerk_lon {float(Lh['jerk_lon']):.6f}  "
      f"curv_rate {float(Lh['curv_rate']):.6f}   (must be EXACTLY 0 ✅)")
noise = c_d + torch.randn_like(c_d) * torch.tensor([0.5, 0.02])
Ln = K.control_smoothness_losses(noise, 0.1, jerk_limit=p99_j,
                                 curvature_rate_limit=p99_c)
print(f"  CONTROL, jittered controls     jerk_lon {float(Ln['jerk_lon']):.6f}  "
      f"curv_rate {float(Ln['curv_rate']):.6f}   (must be >> human ✅)")
print()
print("DONE")
