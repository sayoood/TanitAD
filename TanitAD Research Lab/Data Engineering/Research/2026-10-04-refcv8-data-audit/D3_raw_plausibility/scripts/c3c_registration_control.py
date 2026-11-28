"""D3 check 3c -- K1 registration control done properly.  Box world-frame track speed (ego-motion compensated with the manifest poses) for frames where the ego is
TURNING (|yaw rate| > 0.1 rad/s) and moving (v > 3 m/s), real vs two MUTATIONS that a convention error would cause: (A) yaw sign flipped, (B) ego translation frozen at
the clip's first pose.  Static objects must stay ~still in the real arm; both mutations must push the median track speed far above it.  First N lines of the 3-D join.
NOTE (honest scope): a CONSTANT time lag between the box stream and the pose stream is invisible to this statistic (it shifts every frame alike); this certifies
the rigid transform / frame convention only."""
import json, lzma, math, sys, numpy as np, torch
JOIN = "D:/refcv6_eval_kit/data/join3d/b1_train_plus_eval_agents_3d.jsonl.xz"
mt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
poses = {c: p.numpy().astype(np.float64) for c, p in zip(mt["clip_id"], mt["poses"])}
N = int(sys.argv[2])
sp = {"real": [], "yaw_flipped": [], "translation_frozen": []}
prev = None; last = None; nl = 0
with lzma.open(JOIN, "rt", encoding="utf-8") as f:
    for line in f:
        nl += 1
        if nl > N: break
        r = json.loads(line); cid = r["clip_id"]
        if cid not in poses: continue
        P = poses[cid]; fi = r["frame_idx"]; ag = r["agents"]
        if cid != last: prev = None; last = cid
        if not ag or not (1 <= fi < len(P)):
            prev = None; continue
        w = abs(((P[fi, 2] - P[fi - 1, 2] + math.pi) % (2 * math.pi)) - math.pi) / 0.1007
        turning = (w > 0.1) and (P[fi, 3] > 3.0)
        cx = np.array([a["cx"] for a in ag]); cy = np.array([a["cy"] for a in ag]); tid = [a["track_id"] for a in ag]
        def world(yaw_sign, freeze):
            x0, y0 = (P[0, 0], P[0, 1]) if freeze else (P[fi, 0], P[fi, 1])
            ya = yaw_sign * P[fi, 2]
            return x0 + cx * math.cos(ya) - cy * math.sin(ya), y0 + cx * math.sin(ya) + cy * math.cos(ya)
        cur = {}
        for name, (ys, fr) in {"real": (1, False), "yaw_flipped": (-1, False), "translation_frozen": (1, True)}.items():
            X, Y = world(ys, fr)
            for k in range(len(tid)):
                cur.setdefault(name, {}).setdefault(tid[k], (X[k], Y[k]))
        if prev is not None and prev[0] == fi - 1 and turning:
            for name in sp:
                for t, (x, y) in cur[name].items():
                    q = prev[1][name].get(t)
                    if q is not None: sp[name].append(math.hypot(x - q[0], y - q[1]) / 0.1007)
        prev = (fi, cur)
out = {"lines": nl, "n_steps_turning": len(sp["real"]), **{f"{k}_p25_p50_p75": [round(float(np.quantile(v, q)), 2) for q in (.25, .5, .75)] for k, v in sp.items()}}
json.dump(out, open("raw/c3c_registration_control.json", "w"), indent=1); print(json.dumps(out, indent=1))
