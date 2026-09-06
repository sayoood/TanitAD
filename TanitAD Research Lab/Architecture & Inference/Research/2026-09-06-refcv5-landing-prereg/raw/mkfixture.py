"""Synthetic refcv3_arm-shaped dumps, to TEST the landing analysis before it
is trusted with a 44 h run's only output.

PART A of raw/selftest.log runs against these.  PART B runs against the REAL
banked dump at

    TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-arm/
        raw/fixture_dump/

which is what caught two defects a synthetic fixture could not: a real
refcv3_arm manifest carries NO argv (so the original argv-based replicate
audit could never have fired), and the decisions sidecar carries
*_pred_nav_TRUE as well as *_pred_nav_zero (the true one is the deployed
head reading).  Reproduce PART B by copying that directory beside this one
as `real/`, then `cp -r real realB` and perturb realC's manifest.
"""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1]
N_EP, N_W, K = 7, 11, 4
RNG = np.random.default_rng(0)


def base_gt(nw, ep):
    """A gently curving GT path, K slots, ego frame at t0.

    Seeded PER EPISODE so every dump shares the SAME ground truth -- which is
    what the real dumps do, and what the pairing assertion checks.
    """
    r = np.random.default_rng(9000 + ep)
    t = np.arange(1, K + 1) * 0.5
    v = r.uniform(4.0, 12.0, size=(nw, 1))
    kap = r.normal(0.0, 0.004, size=(nw, 1))
    s = v * t[None, :]
    return np.stack([s, 0.5 * kap * s ** 2], axis=-1)        # [nw, K, 2]


def write(dump, os_jitter, os_bias, seed, argv_extra=(), frozen_seed=7):
    os.makedirs(os.path.join(dump, "decisions"), exist_ok=True)
    rng = np.random.default_rng(seed)
    for e in range(N_EP):
        g = base_gt(N_W, e)
        fr = np.random.default_rng(frozen_seed + e)           # model-FREE arms
        ha = g + fr.normal(0, 0.05, g.shape)
        ha0 = g.copy()
        ha0[..., 1] = 0.0                                     # never steers
        ha0_ext = g + fr.normal(0, 0.04, g.shape)
        o = g + rng.normal(0, os_jitter, g.shape) + os_bias
        z = dict(g=g.astype(np.float32), os=o.astype(np.float32),
                 ha=ha.astype(np.float32), ha0=ha0.astype(np.float32),
                 ha0_ext=ha0_ext.astype(np.float32),
                 os_navshuf=(o + 0.01).astype(np.float32),
                 os_navzero=(o + 0.08).astype(np.float32),
                 os_navpred=(o + 0.03).astype(np.float32),
                 v0=np.full(N_W, 8.0, np.float32),
                 ws=np.arange(N_W) * 5, eid=np.array([e]),
                 clip_index=np.array([e]))
        np.savez_compressed(os.path.join(dump, "ep%03d.npz" % e), **z)
        lab = np.random.default_rng(100 + e)
        lat_l = lab.integers(0, 3, N_W)
        lon_l = lab.integers(0, 3, N_W)
        rou_l = lab.integers(0, 3, N_W)
        flip = rng.random(N_W) < 0.15
        np.savez_compressed(
            os.path.join(dump, "decisions", "ep%03d.npz" % e),
            ws=np.arange(N_W) * 5, ep_poses=np.zeros((40, 4), np.float32),
            lat_label=lat_l, lon_label=lon_l, route_label=rou_l,
            lat_pred_nav_zero=np.where(flip, (lat_l + 1) % 3, lat_l),
            lon_pred_nav_zero=np.where(flip, (lon_l + 1) % 3, lon_l),
            route_pred_nav_zero=np.where(flip, (rou_l + 1) % 3, rou_l),
            sel_idx=rng.integers(0, 117, N_W))
    argv = ["--ckpt", "/w/ckpt_40284_FINAL.pt", "--window-stride", "5",
            "--dump-dir", dump, "--out", dump + ".json"] + list(argv_extra)
    with open(os.path.join(dump, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"tool": "taniteval/tools/refcv3_arm.py",
                   "model": {"argv": argv},
                   "grid": {"name": "2s", "dt_s": 0.5, "k": K,
                            "n_windows": N_EP * N_W, "n_episodes": N_EP}},
                  fh, indent=1)
    print("built %s" % dump)


write(os.path.join(OUT, "R0"), 0.030, 0.000, seed=11)
write(os.path.join(OUT, "R1"), 0.030, 0.000, seed=12)
write(os.path.join(OUT, "R2"), 0.030, 0.000, seed=13)
write(os.path.join(OUT, "V4B"), 0.030, 0.060, seed=14)          # "refcv4b"
write(os.path.join(OUT, "DET"), 0.030, 0.000, seed=11)          # == R0 -> VOID
write(os.path.join(OUT, "BADARGV"), 0.030, 0.000, seed=15,
      argv_extra=["--sel-refined"])                             # -> ARGV VOID
