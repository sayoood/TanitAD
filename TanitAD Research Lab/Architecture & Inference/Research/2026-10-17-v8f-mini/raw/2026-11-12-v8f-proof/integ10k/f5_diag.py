import glob, numpy as np
W = "/workspace/v8proof/integ10k/t1"; DT = 0.2
def load(arm, k):
    fs = sorted(glob.glob("%s/%s_s%d/dump_%s_s%d_m10_causal_heads/ep*.npz" % (W, arm, k, arm, k)))
    return np.concatenate([np.asarray(np.load(f)["cl"], np.float64) for f in fs])
for arm, a_max, m in (("I10R0A5", 3.0, 5), ("I10R0A10", 3.0, 10), ("I10R0A5L", 1.5, 5), ("I10R0AID", 1e9, 10), ("I10R0", 1e9, 10)):
    for k in ((0, 1) if arm not in ("I10R0AID",) else (0,)):
        cl = load(arm, k)
        p = np.concatenate([np.zeros((len(cl), 1, 2)), cl], 1); d = np.diff(p, axis=1); s = np.linalg.norm(d, axis=-1) / DT
        acc = np.diff(s, axis=1) / DT
        inside = np.array([(i + 1) % m != 0 for i in range(acc.shape[1])])
        ang = np.arctan2(d[..., 1], d[..., 0]); turn = np.abs(np.angle(np.exp(1j * np.diff(ang, axis=1))))   # direction change between consecutive chords
        up, dn = (acc > a_max + 0.25) & inside[None], (acc < -8.25) & inside[None]
        v = up | dn; n_in = inside.sum() * len(cl)
        slow = np.minimum(s[:, :-1], s[:, 1:]) < 1.0
        kink = turn > 0.2
        tick = (np.arange(acc.shape[1]) + 1) // m
        print("%-9s s%d share %.5f (up %.5f, down %.5f) | of the violating steps: slower chord under 1 m/s %.3f, direction change above 0.2 rad %.3f, either %.3f | by block %s | windows with any %.4f | largest up %.2f down %.2f" % (
            arm, k, v.sum() / n_in, up.sum() / n_in, dn.sum() / n_in, (v & slow).sum() / max(v.sum(), 1), (v & kink).sum() / max(v.sum(), 1), (v & (slow | kink)).sum() / max(v.sum(), 1),
            [int(v[:, tick == t].sum()) for t in range(30 // m)], v.any(1).mean(), float(acc[:, inside].max()), float(acc[:, inside].min())))
