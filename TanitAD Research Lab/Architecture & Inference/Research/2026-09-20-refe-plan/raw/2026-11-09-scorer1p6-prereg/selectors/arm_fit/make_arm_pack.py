#!/usr/bin/env python3
"""make_arm_pack.py (dev box, CPU): fit_pack8.npz + one arm's fit_logits.npz -> a new pack in which ONLY `logits` is replaced.

The join is by key (log, token, step, rank).  Refuses if the arm file lacks any pack key, has any extra key, holds a duplicate, has the wrong shape or
non-finite values, or if any array other than `logits` would differ from the input pack (checked on the written file, by content).
  make_arm_pack.py --pack fit_pack8.npz --arm-logits OUT/C0/fit_logits.npz --out fit_pack8_C0.npz [--expect-pack-sha256 <sha of the fit_pack.npz the pod used>]
Prints ZZARMPACK_OK <out sha256> on success (exit 0), ZZARMPACK_FAIL <reason> (exit 1) otherwise."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np


class PackRefused(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def keys_of(z, what):
    for k in ("log", "token", "step", "rank"):
        if k not in z:
            raise PackRefused(f"{what} lacks the key array {k!r}")
    ks = list(zip(map(str, z["log"]), map(str, z["token"]), map(int, z["step"]), map(int, z["rank"])))
    if len(set(ks)) != len(ks):
        raise PackRefused(f"{what} has duplicate keys")
    return ks


def make(pack_path, arm_path, out_path, expect_pack_sha=None):
    P = _load(pack_path)
    Z = _load(arm_path)
    if "logits" not in P:
        raise PackRefused("the pack has no logits array")
    if expect_pack_sha is not None:
        got = str(Z["pack_sha256"]) if "pack_sha256" in Z else None
        if got != expect_pack_sha.lower():
            raise PackRefused(f"arm file was made against pack sha256 {got}, expected {expect_pack_sha}")
    pk, ak = keys_of(P, "pack"), keys_of(Z, "arm file")
    row = {k: i for i, k in enumerate(ak)}
    missing = [k for k in pk if k not in row]
    if missing:
        raise PackRefused(f"{len(missing)} pack keys are missing from the arm file, e.g. {missing[0]}")
    extra = set(ak) - set(pk)
    if extra:
        raise PackRefused(f"{len(extra)} arm-file keys are not in the pack, e.g. {sorted(extra)[0]}")
    L = Z["logits"]
    old = P["logits"]
    if L.ndim != 3 or L.shape[0] != len(ak) or L.shape[1:] != old.shape[1:]:
        raise PackRefused(f"arm logits shape {L.shape}, pack logits {old.shape}")
    if not np.isfinite(L).all():
        raise PackRefused("arm logits are not finite")
    new = L[np.array([row[k] for k in pk])].astype(old.dtype)
    out = dict(P)
    out["logits"] = new
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, **out)
    # verify the WRITTEN file by content: every array but logits identical, same set of arrays, logits as joined
    with np.load(tmp, allow_pickle=False) as R:
        if set(R.files) != set(P):
            raise PackRefused(f"array set changed: {sorted(set(R.files) ^ set(P))}")
        for k in P:
            if k == "logits":
                continue
            if R[k].dtype != P[k].dtype or R[k].shape != P[k].shape or not np.array_equal(R[k], P[k]):
                raise PackRefused(f"array {k!r} would change")
        if not np.array_equal(R["logits"], new):
            raise PackRefused("written logits differ from the join")
    os.replace(tmp, out_path)
    return sha256_file(out_path), int((new != old).any(axis=(1, 2)).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--arm-logits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-pack-sha256", default=None)
    a = ap.parse_args(argv)
    try:
        sha, n_changed = make(a.pack, a.arm_logits, a.out, a.expect_pack_sha256)
    except PackRefused as e:
        print(f"ZZARMPACK_FAIL {e}")
        return 1
    print(f"  {n_changed} sets have logits different from the input pack")
    print(f"ZZARMPACK_OK {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
