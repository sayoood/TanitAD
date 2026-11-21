import glob, numpy as np, os, sys
W = "/workspace/v8proof/integ10k/t1"
def load(run):
    d = os.path.join(W, run, "dump_%s_m10_causal_heads" % run)
    cl, g = [], []
    for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
        z = np.load(f, allow_pickle=True)
        cl.append(np.asarray(z["cl"], float)); g.append(np.asarray(z["g"], float))
    return np.concatenate(cl), np.concatenate(g)
for run in ("I10RLcKF5Hb20_s0", "I10RLcKF5P30Hb20_s0", "I10R0KNlP30Hb20_s0", "I10R0LrCarry_s0"):
    cl, g = load(run)
    s1 = np.linalg.norm(cl[:, 0, :2], axis=1); s2 = np.linalg.norm(cl[:, 1, :2] - cl[:, 0, :2], axis=1); s3 = np.linalg.norm(cl[:, 2, :2] - cl[:, 1, :2], axis=1)
    g1 = np.linalg.norm(g[:, 0, :2], axis=1); g2 = np.linalg.norm(g[:, 1, :2] - g[:, 0, :2], axis=1)
    print(run, "shape", cl.shape, "| first point from origin: median %.3f mean %.3f p90 %.3f | step 2: median %.3f mean %.3f | step 3 median %.3f | logged step 1 median %.3f step 2 median %.3f | share first > 2 x second + 0.5 m: %.3f" % (
        np.median(s1), s1.mean(), np.percentile(s1, 90), np.median(s2), s2.mean(), np.median(s3), np.median(g1), np.median(g2), float(np.mean(s1 > 2 * s2 + 0.5))))
