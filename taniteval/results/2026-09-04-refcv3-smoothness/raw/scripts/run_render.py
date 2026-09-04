"""WP-5: how much of the visible roughness is the RENDER RESOLUTION?

The renderer (render_refcv3_video.py:1025-1027) draws GT from ``_ego_future(poses, t0, 60)``
= 60 DENSE 10 Hz poses, and the model from ``densify(sel)`` = the 8 emitted slots joined
by straight lines. So the two polylines on the same frame have 7.5x different resolution.
This measures the amplification on a path everyone agrees is smooth: GT itself.
"""
import json, sys, numpy as np
sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S

RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
d = json.load(open(RAW + "rollouts_refcv3_openloop.json", encoding="utf-8"))
poses = np.array([[g["x"], g["y"], g["yaw"]] for g in d["gt"]])
steps = {int(s["k"]): s for r in d["rollouts"] for s in r["steps"]}
ks = [k for k in sorted(steps) if k + 60 < len(poses)]
print(f"n windows = {len(ks)}")

T60 = np.arange(0, 61) * 0.1                     # the DENSE 10 Hz GT polyline
gt_dense = np.zeros((len(ks), 60, 2))
gt_8 = np.zeros((len(ks), 8, 2))
mdl_8 = np.array([steps[k]["extra"]["traj_full_6s"] for k in ks])
v0 = np.array([steps[k]["v"] for k in ks])
for i, k in enumerate(ks):
    gt_dense[i], _ = S.ego_future(poses, k, T60)
    gt_8[i], _ = S.ego_future(poses, k, S.T8)

gd = S.geom(S.with_origin(gt_dense), T60)
g8 = S.geom(S.with_origin(gt_8), S.T8)
gm = S.geom(S.with_origin(mdl_8), S.T8)

print("\n== THE RENDER-RESOLUTION AMPLIFICATION (same GT path, two polylines) ==")
for nm, g in (("GT @ 10 Hz (60 pts) - AS DRAWN for GT", gd),
              ("GT @ refcv3's 8 slots  - AS DRAWN for the model", g8),
              ("refcv3 @ its 8 slots", gm)):
    t = np.degrees(np.abs(g["turn"]))
    print(f"  {nm:<44s} mean|turn| {t.mean():7.4f} deg   p90 {np.percentile(t,90):7.4f}"
          f"   max {t.max():6.2f}")
amp = np.degrees(np.abs(g8["turn"])).mean() / np.degrees(np.abs(gd["turn"])).mean()
print(f"  ==> AMPLIFICATION of the SAME smooth path by the 8-slot drawing: {amp:.2f}x")
r_pre = np.degrees(np.abs(gd["turn"])).mean()
print(f"  ==> refcv3's own kink vs the DENSE GT polyline next to it: "
      f"{np.degrees(np.abs(gm['turn'])).mean()/r_pre:.2f}x")

print("\n== SEAM: mean |turn| before (t<=1.5 s) vs after (t>=3 s) the 2 s seam ==")
for nm, g in (("GT @ 8 slots", g8), ("refcv3 @ 8 slots", gm)):
    t = np.degrees(np.abs(g["turn"]))
    pre, post = t[:, :3].mean(), t[:, 4:].mean()
    k = np.abs(g["kappa"]); kpre, kpost = np.nanmean(k[:, :3]), np.nanmean(k[:, 4:])
    print(f"  {nm:<20s} turn pre {pre:6.3f} post {post:6.3f}  ratio {post/pre:5.2f}x"
          f"   | kappa pre {kpre:.5f} post {kpost:.5f} ratio {kpost/kpre:5.2f}x")

print("\n== ZIG-ZAG: fraction of interior vertices where the turn CHANGES SIGN ==")
def zig(g):
    return S.sign_flip(g["turn"])
rows = {"GT @ 10 Hz dense": gd, "GT @ 8 slots": g8, "refcv3 @ 8 slots": gm}
rows["CV control"] = S.geom(S.with_origin(S.cv_path(v0, S.T8)), S.T8)
kap0 = np.array([S.gt_curvature0(poses, k) for k in ks])
rows["ARC control"] = S.geom(S.with_origin(S.arc_path(v0, kap0, S.T8)), S.T8)
for nm, g in rows.items():
    print(f"  {nm:<20s} sign-flip fraction {zig(g):6.3f}")

print("\n== LONGITUDINAL: per-segment implied speed (m/s), and the spacing of the circles ==")
for nm, g, t in (("GT @ 8 slots", g8, S.T8), ("refcv3 @ 8 slots", gm, S.T8)):
    v = g["v"]
    print(f"  {nm:<20s} v per segment mean {np.round(v.mean(0),2)}")
    print(f"  {'':<20s} |v - v0| mean {np.abs(v - v0[:,None]).mean():.3f} m/s"
          f"   |a| mean {np.abs(g['a']).mean():.3f}  |jerk| mean {np.abs(g['jerk']).mean():.3f}")
print(f"  {'v0 (measured)':<20s} mean {v0.mean():.2f} m/s")

json.dump({"n": len(ks), "amplification_8slot_vs_dense": float(amp),
           "gt_dense_mean_turn_deg": float(np.degrees(np.abs(gd['turn'])).mean()),
           "gt_8slot_mean_turn_deg": float(np.degrees(np.abs(g8['turn'])).mean()),
           "refcv3_8slot_mean_turn_deg": float(np.degrees(np.abs(gm['turn'])).mean()),
           "zigzag": {k: zig(v) for k, v in rows.items()}},
          open(r"C:/Users/Admin/_wp56/wp56/render_artifact.json", "w"), indent=1)
