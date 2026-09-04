"""WP-6: DOES THE FREE OFFSET HEAD DESTROY THE ANCHORS' KINEMATIC SMOOTHNESS?

refc.py:1400  ``x = anchors[None] + offset``  -- an UNCONSTRAINED per-slot 2-D
offset. The anchors are constant-(yaw_rate, accel) unicycle rollouts (smooth by
construction, sign-flip 0.0000). This measures what the decode does to them, on
the one arm whose DECODED fan is banked in-repo.
"""
import sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from tanitad.refs.refc import default_anchors

fb = torch.load(r"C:/Users/Admin/_wp56/taniteval/results/fan_refc-base-30k.pt",
                map_location="cpu", weights_only=False)
fan = fb["fan"].numpy().astype(np.float64)      # [881, 128, 4, 2] DECODED
A0 = default_anchors((5, 10, 15, 20), 128, 4096, 0).numpy().astype(np.float64)
sel = fb["sel"].numpy(); gt = fb["gt"].numpy().astype(np.float64)
print(f"banked refc-base fan {fan.shape}  ckpt={fb['ckpt']} step={fb['ckpt_step']} "
      f"steps={fb['steps']} wp_steps={fb['wp_steps']}")

off = fan - A0[None]
print(f"\n  |offset| per waypoint: mean {np.linalg.norm(off,axis=-1).mean():.3f} m  "
      f"p90 {np.percentile(np.linalg.norm(off,axis=-1),90):.3f}  "
      f"max {np.linalg.norm(off,axis=-1).max():.2f}")

def stats(P, nm):
    g = S.geom(S.with_origin(P.reshape(-1, P.shape[-2], 2)), S.T4)
    t = np.degrees(np.abs(g["turn"]))
    fl = ((np.sign(g["turn"][:, :-1]) * np.sign(g["turn"][:, 1:])) < 0).mean()
    k = np.abs(g["kappa"])
    Lm = 0.5 * (g["L"][..., :-1] + g["L"][..., 1:])
    dk = np.abs(np.diff(g["kappa"], axis=-1)) / np.maximum(0.5*(Lm[..., :-1]+Lm[..., 1:]), 1e-6)
    print(f"  {nm:<40s} mean|turn| {t.mean():7.3f} deg  p99 {np.percentile(t,99):7.2f}  "
          f"max {t.max():7.2f} | sign-flip {fl:6.4f} | mean|dk/ds| {np.nanmean(dk):.6f} "
          f"| mean|jerk| {np.abs(g['jerk']).mean():7.3f}")
    return fl

print("\n== the SAME 128 candidates, BEFORE and AFTER the decoder's free offset ==")
stats(np.broadcast_to(A0[None], fan.shape).copy(), "ANCHORS (constant yaw-rate rollouts)")
stats(fan, "DECODED FAN = anchors + offset")
stats(fan[np.arange(len(sel)), sel][:, None], "  ...the SELECTED one only")
stats(gt[:, None], "GROUND TRUTH (same 881 windows)")

# ---- CONTROL: is A0 really THIS checkpoint's anchor vocabulary? ------------
# If the run had used an externally built set, `fan - A0` would be meaningless.
mean_fan = fan.mean(0)                                     # [128, 4, 2]
D = np.linalg.norm(mean_fan[:, None] - A0[None], axis=(-1, -2))   # [128, 128]
diag_is_min = (D.argmin(1) == np.arange(128)).mean()
print(f"\n  CONTROL — anchor identity: mean-fan[i] nearest to A0[i] for "
      f"{diag_is_min*100:.1f} % of 128 candidates "
      f"(chance 0.8 %); mean diag dist {np.diag(D).mean():.2f} m vs "
      f"off-diag {D[~np.eye(128,dtype=bool)].mean():.2f} m")
