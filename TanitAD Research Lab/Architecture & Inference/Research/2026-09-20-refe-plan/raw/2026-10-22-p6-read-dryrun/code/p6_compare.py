"""P6-READ-DRYRUN Q3: bit-identity of two plan npz files written by p6_navtrain_plan.py.

  python -I p6_compare.py --a A.npz --b B.npz --label "..." [--out identity.json] [--mutate]

For every array: shape equality, np.array_equal (bitwise for floats: NaN-free here), max |delta|, and the per-token count of tokens with ANY difference.
`--mutate` is the deliberate-regression arm: it adds 1 ulp (np.nextafter) to ONE value of B's raw trajectory and requires the comparison to FAIL, so a green
result is a statement about the comparison tool as well as about the model.
Zero control: A against A must read max |delta| exactly 0.0 and 0 differing tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

KEYS = ("poses", "traj_raw", "score_raw", "pick", "ego_vec", "cmd_seen", "cmd_asked", "tok_hash")


def compare(A, B) -> dict:
    out = {"tokens_equal_and_same_order": bool(np.array_equal(A["token"], B["token"])), "n": int(len(A["token"])), "arrays": {}}
    ok = out["tokens_equal_and_same_order"]
    bad_tok = np.zeros(len(A["token"]), dtype=bool)
    for k in KEYS:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            out["arrays"][k] = {"shape_a": list(a.shape), "shape_b": list(b.shape), "equal": False}
            ok = False
            continue
        eq = bool(np.array_equal(a, b))
        if a.dtype.kind in "fiu":
            md = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        else:
            md = None
        per_tok = np.array([not np.array_equal(a[i], b[i]) for i in range(len(a))])
        bad_tok |= per_tok
        out["arrays"][k] = {"dtype": str(a.dtype), "bit_equal": eq, "max_abs_delta": md, "tokens_differing": int(per_tok.sum()),
                            "bitwise_bytes_equal": bool(np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())}
        ok &= eq and out["arrays"][k]["bitwise_bytes_equal"]
    out["tokens_with_any_difference"] = int(bad_tok.sum())
    out["IDENTICAL"] = bool(ok and not bad_tok.any())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    A = dict(np.load(a.a, allow_pickle=False))
    B = dict(np.load(a.b, allow_pickle=False))
    res = {"label": a.label, "a": a.a, "b": a.b, "sha256_a": hashlib.sha256(open(a.a, "rb").read()).hexdigest(),
           "sha256_b": hashlib.sha256(open(a.b, "rb").read()).hexdigest()}
    res["compare"] = compare(A, B)
    res["zero_control_A_vs_A"] = compare(A, {k: v.copy() for k, v in A.items()})
    assert res["zero_control_A_vs_A"]["IDENTICAL"], "zero control failed: the comparison tool is broken"
    if a.mutate:
        Bm = {k: v.copy() for k, v in B.items()}
        i = len(Bm["traj_raw"]) // 2
        Bm["traj_raw"][i, 3, 5, 0] = np.nextafter(Bm["traj_raw"][i, 3, 5, 0], np.float32(np.inf))
        mm = compare(A, Bm)
        res["deliberate_regression_one_ulp"] = {"must_read_not_identical": True, "IDENTICAL": mm["IDENTICAL"], "traj_raw": mm["arrays"]["traj_raw"],
                                                "caught": (not mm["IDENTICAL"])}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1) if a.out else None
    c = res["compare"]
    print(json.dumps({"label": a.label, "IDENTICAL": c["IDENTICAL"], "n": c["n"], "tokens_with_any_difference": c["tokens_with_any_difference"],
                      "max_abs_delta": {k: v.get("max_abs_delta") for k, v in c["arrays"].items()},
                      **({"mutation_caught": res["deliberate_regression_one_ulp"]["caught"]} if a.mutate else {})}))
    return 0 if c["IDENTICAL"] else 1


if __name__ == "__main__":
    sys.exit(main())
