"""Control-target statistics over the TRAIN corpus — widening the head's scale constants.

⛔ WHY THIS RUN EXISTS. The four head design constants (ACCEL_SCALE, YAWRATE_SCALE,
DACCEL_SCALE, DYAWRATE_SCALE) were derived from 39 VAL clips. They are distributional
priors rather than metric tuning, so the leak is mild — but a constant fitted on val and
then used to train an arm evaluated on val is still val information inside the training
recipe, and it is avoidable for the price of one CPU pass.

⭐ It also buys statistical power the val set cannot: 39 clips x 20 steps is 780 samples,
and the quantities that matter most here are TAILS (kurtosis 38.9 on curvature). A p99
estimated from 780 samples is 8 samples deep.

⚠️ Windows are drawn on the SAME grid the trainer will use (`lead_source.window_last_indices`
reproduces `rollout.collect`'s grid), so the statistics describe the distribution the head
actually sees — not an arbitrary resampling of the same episodes.
"""
import argparse, glob, json, math
import numpy as np
import torch


def controls_from_poses(poses, last, K, dt):
    """GT ego-frame future -> (accel, curvature, yaw_rate, speed) per step."""
    x, y, yaw = poses[:, 0], poses[:, 1], poses[:, 2]
    j = np.arange(last, last + K + 1)
    dx, dy = x[j] - x[last], y[j] - y[last]
    c, s = math.cos(yaw[last]), math.sin(yaw[last])
    ex, ey = dx * c + dy * s, -dx * s + dy * c
    p = np.stack([ex, ey], 1)                                # [K+1, 2], p[0] = 0
    d = p[1:] - p[:-1]
    ds = np.linalg.norm(d, axis=-1)
    sp = ds / dt
    acc = np.concatenate([(sp[1:] - sp[:-1]) / dt, [(sp[-1] - sp[-2]) / dt]])
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = (h[1:] - h[:-1] + math.pi) % (2 * math.pi) - math.pi
    moving = ds[:-1] > 0.05
    kap = np.where(moving, dh / np.maximum(ds[:-1], 1e-8), 0.0)
    kap = np.concatenate([kap, kap[-1:]])
    return acc, kap, sp * kap, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.cache}/*.v2ep.pt"))[:a.episodes]
    A, Kp, Y, V = [], [], [], []
    n_win = 0
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        poses = np.asarray(d["poses"], dtype=np.float64)
        T = poses.shape[0]
        # the trainer's own window grid, not an arbitrary resampling
        for st in range(0, max(T - a.window - a.k, 0), a.stride):
            last = st + a.window - 1
            if last + a.k + 1 > T:
                break
            acc, kap, yr, sp = controls_from_poses(poses, last, a.k, a.dt)
            A.append(acc); Kp.append(kap); Y.append(yr); V.append(sp)
            n_win += 1
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(files)}] windows {n_win}", flush=True)

    A, Kp, Y, V = map(lambda z: np.stack(z), (A, Kp, Y, V))   # [n, K]

    def block(name, x):
        d = x[:, 1:] - x[:, :-1]
        lag = float(np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1])
        return {
            "std": round(float(x.std()), 6), "mean": round(float(x.mean()), 6),
            "p99_abs": round(float(np.percentile(np.abs(x), 99)), 6),
            "p999_abs": round(float(np.percentile(np.abs(x), 99.9)), 6),
            "kurtosis": round(float(((x - x.mean()) ** 4).mean() / x.var() ** 2), 3),
            "delta_std": round(float(d.std()), 6),
            "delta_over_abs": round(float(d.std() / max(x.std(), 1e-12)), 4),
            "lag1_autocorr": round(lag, 4),
        }

    slow, fast = V < 3.0, V > 8.0
    res = {
        "_evidence_class": "MEASURED (ours)",
        "_corpus": a.cache, "n_episodes": len(files), "n_windows": n_win,
        "n_samples": int(A.size), "k": a.k, "dt": a.dt,
        "_grid": ("window origins on the trainer's own stride-8 grid, so these describe "
                  "the distribution the head actually sees"),
        "accel": block("accel", A), "curvature": block("curvature", Kp),
        "yaw_rate": block("yaw_rate", Y), "speed": block("speed", V),
        "conditioning": {
            "curvature_p99_slow_v_lt_3": round(float(np.percentile(np.abs(Kp[slow]), 99)), 6),
            "curvature_p99_fast_v_gt_8": round(float(np.percentile(np.abs(Kp[fast]), 99)), 6),
            "yaw_rate_p99_slow": round(float(np.percentile(np.abs(Y[slow]), 99)), 6),
            "yaw_rate_p99_fast": round(float(np.percentile(np.abs(Y[fast]), 99)), 6),
            "n_slow": int(slow.sum()), "n_fast": int(fast.sum()),
            "_read": ("the low/high-speed tail RATIO is the conditioning argument for "
                      "predicting yaw rate instead of curvature"),
        },
    }
    cs = res["conditioning"]
    cs["curvature_tail_ratio"] = round(
        cs["curvature_p99_slow_v_lt_3"] / max(cs["curvature_p99_fast_v_gt_8"], 1e-9), 2)
    cs["yaw_rate_tail_ratio"] = round(
        cs["yaw_rate_p99_slow"] / max(cs["yaw_rate_p99_fast"], 1e-9), 2)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
