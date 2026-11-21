import glob, numpy as np, os, sys
sys.path.insert(0, "/workspace/v8proof/podread/root/integ10k")
import read_longroll10k_n4 as N4
R = N4.R
W = "/workspace/v8proof/integ10k/t1"
for run in ("I10R0P30Hb20_s0", "I10R0P30Hb20_s1", "I10R0P60Hb20_s0", "I10R0KP30Hb20_s0"):
    o = N4._orig_load(os.path.join(W, run, "dump_%s_m10_causal_heads" % run))
    z = np.load(os.path.join(W, run, "ticks", "ticks_%s_m10_causal_heads.npz" % run), allow_pickle=True)
    p0, v, miss = N4.p0_rows(z, o["clip_index"], o["ws"])
    cl = np.asarray(o["cl"], float)
    s1 = np.linalg.norm(cl[:, 0] - p0, axis=1); s2 = np.linalg.norm(cl[:, 1] - cl[:, 0], axis=1); s3 = np.linalg.norm(cl[:, 2] - cl[:, 1], axis=1)
    bad = s1 > 0.5 + np.abs(v) * 0.2
    q = lambda x: "[%s]" % ", ".join("%.2f" % t for t in np.percentile(x, [10, 50, 90, 100])) if len(x) else "[]"
    po = np.load(os.path.join(W, run, "ticks", "preroll_offsets_%s.npz" % run), allow_pickle=True)
    m = np.asarray(po["kind"]).astype(str) == "model"
    pos = {(int(c), int(t)): float(d) for c, t, d in zip(np.asarray(po["ci"])[m], np.asarray(po["t0"])[m], np.asarray(po["dpos"], float)[m])}
    dp = np.array([pos.get((int(c), int(t)), np.nan) for c, t in zip(o["clip_index"], o["ws"])])
    print(run, "n", len(cl), "bad", int(bad.sum()), "| bad windows: step1 from p0", q(s1[bad]), "step2", q(s2[bad]), "step3", q(s3[bad]), "v*0.2", q(np.abs(v[bad]) * 0.2),
          "| all: |step1-step2| p50 %.3f p99 %.3f max %.3f; share |step1-step2| <= 0.5: %.4f; share step1 <= 0.5 + max(v*0.2, step2): %.4f | max ||p0|-dpos| %.2e" % (
              np.median(np.abs(s1 - s2)), np.percentile(np.abs(s1 - s2), 99), np.abs(s1 - s2).max(), float(np.mean(np.abs(s1 - s2) <= 0.5)), float(np.mean(s1 <= 0.5 + np.maximum(np.abs(v) * 0.2, s2))), np.nanmax(np.abs(np.linalg.norm(p0, axis=1) - dp))))
