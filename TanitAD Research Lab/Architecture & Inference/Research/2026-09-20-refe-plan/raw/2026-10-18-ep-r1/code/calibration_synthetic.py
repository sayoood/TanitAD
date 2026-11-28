import sys, os, tempfile, shutil
sys.path.insert(0, os.getcwd())
import test_epr1 as T
R = T.R
tmp = tempfile.mkdtemp(prefix="calib_")
cfg = R.Cfg(lr=1e-2, anchor=0.1, epochs=30, batch_sets=16)
for seed in (0, 1, 2):
    w = T.make_world(os.path.join(tmp, f"w{seed}"), seed=seed)
    ws = T.make_world(os.path.join(tmp, f"s{seed}"), seed=seed, shuffle_ep=True)
    out = []
    for world in (w, ws):
        o = T.opts_for(world, os.path.join(tmp, "o", str(len(out)) + str(seed)))
        p = R._setup(o)
        ref = R.p2_reference(p[10], p[8], p[9])["pooled"]
        got = R.oof_run(p[10], cfg, 0, p[8], p[9])["pooled"]
        out.append((round(ref, 4), round(got, 4)))
    print("seed", seed, "real (P2 ref, fit):", out[0], " shuffled (P2 ref, fit):", out[1], flush=True)
shutil.rmtree(tmp, ignore_errors=True)
