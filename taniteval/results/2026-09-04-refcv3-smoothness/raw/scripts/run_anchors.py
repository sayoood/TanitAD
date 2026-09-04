"""WP-6: the ANCHOR SET and the REACHABILITY GATE — refcv3 vs refc-base.

ZERO GPU, no checkpoint: both vocabularies are DETERMINISTIC
(refc.default_anchors(horizons, 128, 4096, seed=0)); the refcv3-b1 launch script
passes NO --anchors, and `anchors` is a registered BUFFER written only by
`load_anchors`, so the trained model carries exactly this set.
"""
import json, sys, numpy as np, torch
sys.path.insert(0, r"C:/Users/Admin/_wp56"); sys.path.insert(0, r"C:/Users/Admin/_wp56/wp56")
import smooth_lib as S
from tanitad.refs.refc import default_anchors
from tanitad.refs.refc_select import reachability_mask

A3 = default_anchors((5,10,15,20,30,40,50,60), 128, 4096, 0)      # [128, 8, 2]
A0 = default_anchors((5,10,15,20), 128, 4096, 0)                  # [128, 4, 2]
a3, a0 = A3.numpy().astype(np.float64), A0.numpy().astype(np.float64)

print("=" * 96)
print("1. ANCHOR GEOMETRY  (both = FPS over 4096 CONSTANT-(yaw_rate, accel) unicycle rollouts)")
print("=" * 96)
for nm, a, t in (("refcv3  [128, 8, 2]", a3, S.T8), ("refc-base [128, 4, 2]", a0, S.T4)):
    g = S.geom(S.with_origin(a), t)
    r = np.linalg.norm(a[:, -1], axis=-1)
    hd = np.degrees(np.arctan2(a[:, -1, 1], a[:, -1, 0]))
    vimp = r / t[-1]
    print(f"\n {nm}   horizon {t[-1]:g} s")
    print(f"   terminal radius  min {r.min():7.2f}  p50 {np.median(r):7.2f}  max {r.max():7.2f} m")
    print(f"   implied mean speed  min {vimp.min():5.2f}  p50 {np.median(vimp):5.2f}  max {vimp.max():5.2f} m/s")
    print(f"   terminal BEARING |deg|  p50 {np.median(np.abs(hd)):6.2f}  p90 "
          f"{np.percentile(np.abs(hd),90):6.2f}  max {np.abs(hd).max():6.2f}")
    print(f"   n anchors with |bearing| > 10 deg : {(np.abs(hd)>10).sum():3d}/128"
          f"   > 30 deg : {(np.abs(hd)>30).sum():3d}/128"
          f"   > 60 deg : {(np.abs(hd)>60).sum():3d}/128")
    print(f"   mean |turn| per vertex (deg): {np.round(np.degrees(np.abs(g['turn'])).mean(0),3)}")
    print(f"   mean |kappa| per vertex(1/m): {np.round(np.nanmean(np.abs(g['kappa']),0),5)}")
    sflip = ((np.sign(g['turn'][:, :-1]) * np.sign(g['turn'][:, 1:])) < 0).mean()
    print(f"   ZIG-ZAG sign-flip fraction  : {sflip:.4f}   (a constant-yaw-rate rollout MUST read 0)")

print("\n" + "=" * 96)
print("2. THE REACHABILITY GATE  `reach_keep`  — v_mean = ||wp_last||/horizon_s in [v0-a*T, v0+a*T]")
print("   accel_max 2.5;  refcv3 T = 6.0 s -> band +-15.0 m/s ;  refc-base T = 2.0 s -> +-5.0 m/s")
print("=" * 96)
RAW = r"C:/Users/Admin/_wp56/taniteval/results/2026-09-04-refcv3-closedloop/raw/"
d = json.load(open(RAW + "rollouts_refcv3_openloop.json", encoding="utf-8"))
v0_roll = np.array([s["v"] for r in d["rollouts"] for s in r["steps"]])
dumpv0 = []
import glob, os
for f in sorted(glob.glob(r"C:/Users/Admin/_wp56/dump/refcv3_40284_dump/ep*.npz")):
    dumpv0.append(np.load(f)["v0"])
dumpv0 = np.concatenate(dumpv0)
print(f"   v0 sources: render/rollout n={len(v0_roll)} mean {v0_roll.mean():.2f} | "
      f"40284 open-loop dump n={len(dumpv0)} mean {dumpv0.mean():.2f} "
      f"[p5 {np.percentile(dumpv0,5):.2f} p95 {np.percentile(dumpv0,95):.2f}]")

for nm, A, T, v0 in (("refcv3 ANCHORS  @6 s", A3, 6.0, dumpv0),
                     ("refc-base ANCHORS @2 s", A0, 2.0, dumpv0)):
    v = torch.tensor(v0, dtype=torch.float32)
    keep = reachability_mask(A.expand(len(v0), *A.shape), v, accel_max=2.5, horizon_s=T)
    kf = 1.0 - keep.float().mean().item()
    per_win = keep.float().sum(1)
    empty = (per_win == 0).float().mean().item()
    print(f"\n   {nm}:  KILLED {kf*100:6.2f} % of 128 anchors per window"
          f"   survivors mean {per_win.mean():6.1f}  min {int(per_win.min())}  max {int(per_win.max())}")
    print(f"      windows with EMPTY survivor set: {empty*100:.2f} %")
    hd = np.degrees(np.arctan2(A[:, -1, 1].numpy(), A[:, -1, 0].numpy()))
    for thr in (10, 30, 60):
        m = np.abs(hd) > thr
        surv = keep[:, torch.tensor(m)].float().sum(1)
        print(f"      turning anchors |bearing|>{thr:2d} deg: {m.sum():3d} exist, "
              f"survivors/window mean {surv.mean():5.2f}  windows with ZERO "
              f"{(surv==0).float().mean().item()*100:5.1f} %")

print("\n" + "=" * 96)
print("3. THE SAME GATE ON THE **DECODED** FAN  (refc-base, banked: fan_refc-base-30k.pt)")
print("   -- the offsets move the fan, so the anchor-band rate is a PROXY; this is the true rate.")
print("=" * 96)
fb = torch.load(r"C:/Users/Admin/_wp56/taniteval/results/fan_refc-base-30k.pt",
                map_location="cpu", weights_only=False)
fan, v0b = fb["fan"], fb["v0"]
keep_dec = reachability_mask(fan, v0b, accel_max=2.5, horizon_s=2.0)
anc_b = A0.expand(len(v0b), 128, 4, 2)
keep_anc = reachability_mask(anc_b, v0b, accel_max=2.5, horizon_s=2.0)
print(f"   n windows {len(v0b)}  v0 mean {v0b.mean():.2f}")
print(f"   DECODED fan  killed {100*(1-keep_dec.float().mean()):6.2f} %"
      f"   survivors mean {keep_dec.float().sum(1).mean():6.1f}")
print(f"   RAW anchors  killed {100*(1-keep_anc.float().mean()):6.2f} %"
      f"   survivors mean {keep_anc.float().sum(1).mean():6.1f}")
print(f"   => the decode offsets change the kill rate by "
      f"{100*(keep_anc.float().mean()-keep_dec.float().mean()):+.2f} pp")
sel = fb["sel"]
print(f"   selected anchor survives the band: "
      f"{keep_dec.gather(1, sel[:,None]).float().mean().item()*100:.2f} %")
