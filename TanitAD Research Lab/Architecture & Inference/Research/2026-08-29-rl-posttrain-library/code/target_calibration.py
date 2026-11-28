"""⛔ IS THE SAFETY TARGET ITSELF WRONG? The oracle control says yes; this says why.

`anchor_vocab.py` fired its constant-control: on near-miss windows the HUMAN
DRIVER'S OWN FUTURE clears d_safe=5 m in 0.0% of cases, while the model's existing
anchors clear it in 87.5%. A target the ground truth never meets is not a safety
threshold -- it is a target for NOT DRIVING.

`rewards.py:240-241` names the suspected mechanism in its own docstring: obstacles
are "a static snapshot". The ego's 2 s FUTURE is scored against agents FROZEN at
t0, so ordinary car-following -- ego advances, lead car does not -- registers as
closing to zero clearance on every single window.

This measures that directly, using the `track_id` field the join already carries:

  STATIC   clearance of the human's future vs agents frozen at t0   (what we score)
  MOVING   clearance of the human's future vs the SAME tracks at their OWN future
           positions, transformed into the t0 ego frame                (the truth)

If MOVING >> STATIC, the violations are an artifact of the frozen snapshot and the
proximity/collision terms have been penalising competent driving all along.

Tier: T0. NON-PARITY corpus. Evidence class: MEASURED.
"""
import importlib.util, sys, json, collections
import numpy as np

sys.path.insert(0, "stack")
_s = importlib.util.spec_from_file_location("pilot", "stack/scripts/rl_pilot_refc21.py")
P = importlib.util.module_from_spec(_s); _s.loader.exec_module(P)

O = r"C:/Users/Admin/tanitad-data/rl-pilot"
R = 2.0                                            # ego_r + obs_r
src = P.WindowSource(r"C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836",
                     f"{O}/pilot_val_agents.jsonl", seed=0, lru=60)

# raw per-frame agent records, keyed by track, so a track can be followed forward
raw = collections.defaultdict(dict)
import io as _io
for line in _io.open(f"{O}/pilot_val_agents.jsonl", encoding="utf-8"):
    d = json.loads(line)
    raw[d["clip_id"]][d["frame_idx"]] = {a["track_id"]: (a["cx"], a["cy"]) for a in d["agents"]}

rng = np.random.default_rng(1234)
stat, mov, pairs = [], [], 0
for stem in src.stems:
    ep = src._ep(stem)
    poses = ep["poses"].numpy()
    T = int(poses.shape[0])
    lo, hi = P.WINDOW - 1, T - P.HORIZONS[-1] - 1
    if hi <= lo:
        continue
    key = None
    for k in raw:                                   # the join keys by clip stem
        if k in stem or stem in k:
            key = k; break
    if key is None:
        continue
    for t0 in sorted(rng.choice(np.arange(lo, hi), size=min(6, hi - lo), replace=False)):
        t0 = int(t0)
        a0 = raw[key].get(t0)
        if not a0:
            continue
        x0, y0, yaw0 = poses[t0, 0], poses[t0, 1], poses[t0, 2]
        fut = poses[[t0 + h for h in P.HORIZONS]]
        gx, gy = P.ego_frame_np(fut[:, 0], fut[:, 1], x0, y0, yaw0)
        gt = np.stack([gx, gy], -1)                                    # [S,2]

        # tracks visible at t0, ahead and within the join's own gate
        tids = [t for t, (cx, cy) in a0.items() if 0.0 < cx < P.LEAD_MAX_GAP_M]
        if not tids:
            continue
        static = np.array([a0[t] for t in tids], dtype=np.float64)     # [K,2]

        # the SAME tracks at each future step, mapped into the t0 ego frame
        moving = np.full((len(P.HORIZONS), len(tids), 2), np.nan)
        for si, h in enumerate(P.HORIZONS):
            af = raw[key].get(t0 + h)
            if not af:
                continue
            xt, yt, yawt = poses[t0 + h, 0], poses[t0 + h, 1], poses[t0 + h, 2]
            for ki, tid in enumerate(tids):
                if tid not in af:
                    continue
                cx, cy = af[tid]
                wx = xt + cx * np.cos(yawt) - cy * np.sin(yawt)
                wy = yt + cx * np.sin(yawt) + cy * np.cos(yawt)
                ex, ey = P.ego_frame_np(np.array([wx]), np.array([wy]), x0, y0, yaw0)
                moving[si, ki] = (ex[0], ey[0])
        if np.isnan(moving).all():
            continue

        d_s = np.linalg.norm(gt[:, None, :] - static[None, :, :], axis=-1)
        c_s = np.clip(d_s - R, 0, None).min()
        d_m = np.linalg.norm(gt[:, None, :] - moving, axis=-1)          # [S,K]
        c_m = np.nanmin(np.clip(d_m - R, 0, None))
        stat.append(c_s); mov.append(c_m); pairs += 1

stat, mov = np.array(stat), np.array(mov)
print(f"n windows with a followable track: {pairs}")
print(f"\n{'':<10}{'STATIC (scored)':>18}{'MOVING (truth)':>18}")
for q in (5, 25, 50, 75, 95):
    print(f"  p{q:<7} {np.percentile(stat,q):17.3f} {np.percentile(mov,q):17.3f}")
print(f"  mean    {stat.mean():17.3f} {mov.mean():17.3f}")
for th in (2.0, 3.0, 5.0):
    print(f"\n  human clears {th:.0f} m:  STATIC {100*(stat>=th).mean():5.1f}%"
          f"   MOVING {100*(mov>=th).mean():5.1f}%")
json.dump({"n": pairs, "static": {"mean": float(stat.mean()),
           "p50": float(np.median(stat)), "clear_5m": float((stat >= 5).mean())},
           "moving": {"mean": float(mov.mean()), "p50": float(np.median(mov)),
                      "clear_5m": float((mov >= 5).mean())}},
          open(f"{O}/target_calibration.json", "w"), indent=1)
print(f"\n-> {O}/target_calibration.json")
