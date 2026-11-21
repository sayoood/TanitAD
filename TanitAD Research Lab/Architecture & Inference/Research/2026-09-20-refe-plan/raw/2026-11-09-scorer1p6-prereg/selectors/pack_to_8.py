#!/usr/bin/env python3
"""pack_to_8.py: fit_pack.npz stores P6's raw proposals, [N,64,20,x] at 5 Hz. The selector runner, like the navtest dump, works on the executed plans [N,64,8,x] at 0.5 s.
This writes that form with navsim_pdm_targets.executed_plans8(props, repair=True): the one function the PDM labels and the navtest proposal dump both went through.
No other array changes. Exit 0 and ZZPACK8_OK on success; a selftest (--selftest) checks the conversion on a known answer and that it has teeth.
usage: pack_to_8.py --refe <dir holding navsim_pdm_targets.py> --pack fit_pack.npz --out fit_pack8.npz"""
import argparse
import hashlib
import os
import sys

import numpy as np


def load_fn(refe):
    sys.path.insert(0, refe)
    import navsim_dac as ND

    def executed8(P20):
        # navsim_pdm_targets.executed_plans8(P20, repair=True), inlined only to avoid that module's navsim imports on this box; selftest compares it against the source text
        P = np.asarray(P20, dtype=np.float64).copy()
        P[..., -1, 2] = P[..., -2, 2]
        return np.stack([ND.to_navsim(p) for p in P])
    return executed8


def check_source(refe):
    """the inlined body must equal the source body, or this tool has drifted from the labeller"""
    s = open(os.path.join(refe, "navsim_pdm_targets.py"), encoding="utf-8").read()
    need = ["P[..., -1, 2] = P[..., -2, 2]", "return np.stack([ND.to_navsim(p) for p in P])", "def executed_plans8(props20: np.ndarray, repair: bool = True)"]
    miss = [n for n in need if n not in s]
    if miss:
        raise SystemExit(f"ZZPACK8_REFUSED source of executed_plans8 changed: {miss}")


def selftest(ex8):
    # constant 10 m/s straight line at 5 Hz: 0.2 s steps of 2 m, so 0.5 s -> 5 m exactly
    P = np.zeros((1, 20, 3))
    P[0, :, 0] = 2.0 * np.arange(1, 21)
    P[0, -1, 2] = 9.0  # the broken last heading the repair must overwrite
    Q = ex8(P)
    assert Q.shape == (1, 8, 3)
    assert np.allclose(Q[0, :, 0], 5.0 * np.arange(1, 9)), Q[0, :, 0]
    assert abs(Q[0, -1, 2]) < 1e-12, "repair did not run"
    # teeth: without the repair the last heading is the broken one, and 0.2 s-index slicing gives other x
    assert abs(P[0, -1, 2] - 9.0) < 1e-12
    assert not np.allclose(P[0, :8, 0], Q[0, :, 0])
    print("ZZPACK8_SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refe", required=True)
    ap.add_argument("--pack")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    check_source(a.refe)
    ex8 = load_fn(a.refe)
    selftest(ex8)
    if a.selftest:
        return 0
    Z = np.load(a.pack, allow_pickle=False)
    xy, yaw = Z["traj_xy"], Z["yaw"]
    if xy.shape[1:] != (64, 20, 2) or yaw.shape[1:] != (64, 20):
        raise SystemExit(f"ZZPACK8_REFUSED shapes {xy.shape} {yaw.shape}")
    n = xy.shape[0]
    X = np.empty((n, 64, 8, 3), np.float32)
    for i in range(n):
        X[i] = ex8(np.concatenate([xy[i], yaw[i][..., None]], -1))
    out = {k: Z[k] for k in Z.files if k not in ("traj_xy", "yaw")}
    out["traj_xy"], out["yaw"] = X[..., :2], X[..., 2]
    out["geometry_note"] = np.array("executed plans on NAVSIM's 8-pose grid: navsim_pdm_targets.executed_plans8(raw 5 Hz proposals, repair=True)")
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, a.out)
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""):
            h.update(b)
    ch = np.linalg.norm(X[:, :, 0, :2], axis=-1) / 0.5
    print(f"ZZPACK8_OK {n} sets; median first-chord speed {float(np.median(ch)):.3f} m/s; out sha256 {h.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
