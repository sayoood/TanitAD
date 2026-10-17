#!/usr/bin/env python3
"""zt5_seam_fix.py -- apply PREREG_ZT5's F_c (heading := path tangent; x, y bit-identical) to a REFe NAVSIM seam.

    python zt5_seam_fix.py --seam <stage2_sft4_B_u400_navtest_full.npz> --out <seam_Fc.npz> [--expect-n 12146]

The seam's poses are the EXECUTED plans (A7 already applied by REFePlanner.executed); F_c replaces every knot heading
where the tangent is defined (zt5_lib.tangents).  Gates (exit 3 on failure): x and y bit-identical on every token,
token / fingerprint / sampling copied unchanged, the token count as expected.  Writes <out>.report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zt5_lib as L  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fix", default="F_c", choices=["F_c"])
    ap.add_argument("--expect-n", type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.seam)
    P = np.asarray(z["poses"], np.float64)
    Q = np.stack([L.fix_heading_tangent(p) for p in P]).astype(np.float32)
    gates = {"xy_bit_identical": bool(np.array_equal(Q[..., :2], z["poses"][..., :2])),
             "n": int(len(Q)), "n_ok": (a.expect_n == 0 or len(Q) == a.expect_n)}
    dpsi = np.abs(L.wrap(Q[..., 2].astype(np.float64) - P[..., 2]))
    rep = {"seam_in": a.seam, "seam_in_sha256": sha(a.seam), "fix": a.fix, "gates": gates,
           "tokens_changed": int((dpsi.max(axis=1) > 0).sum()),
           "abs_heading_change_deg": {"median_of_token_max": float(np.degrees(np.median(dpsi.max(axis=1)))),
                                      "p95_of_token_max": float(np.degrees(np.percentile(dpsi.max(axis=1), 95)))}}
    if not (gates["xy_bit_identical"] and gates["n_ok"]):
        print(json.dumps(rep, indent=1))
        sys.exit(3)
    np.savez(a.out, token=z["token"], fingerprint=z["fingerprint"], poses=Q, sampling=z["sampling"],
             arm=np.array(str(z["arm"]) + "_zt5Fc"))
    rep["seam_out"], rep["seam_out_sha256"] = a.out, sha(a.out)
    json.dump(rep, open(a.out + ".report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
