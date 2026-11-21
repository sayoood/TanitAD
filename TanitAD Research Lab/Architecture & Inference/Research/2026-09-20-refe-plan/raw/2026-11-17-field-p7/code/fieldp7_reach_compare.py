#!/usr/bin/env python3
"""Compare fieldp7_reach.py runs.  F = --w-field w with labels, C = --w-field 0.0 with labels, N = no field flag at all, M = F on a copy of the trainer with the line that adds the term removed.

CHECK(X, C) is GREEN iff all of
  R1  field_loss was called and returned a finite non-zero value in X, and was never called in C
  R2  the gradient contribution (g_X - g_C) is NON-ZERO on traj_head, dec, queries, scene_proj, reg_compress and the backbone LoRA
  R3  the contribution is EXACTLY 0.0 (or no gradient at all) on score_q_mlp, score_dec, score_head
CHECK(C, N) (the zero-weight control) is GREEN iff g_C == g_N bit for bit on every parameter and C never called field_loss.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

PROPOSAL = ("traj_head", "dec", "queries", "scene_proj", "reg_compress", "backbone")
SCORE = ("score_q_mlp", "score_dec", "score_head")


def load(d, tag):
    return json.loads(Path(d, f"{tag}.json").read_text()), torch.load(Path(d, f"{tag}.pt"), weights_only=False)


def delta(gx, gc):
    out = {}
    for n in gx:
        a, b = gx[n], gc[n]
        if a is None and b is None:
            out[n] = None
        else:
            a = torch.zeros_like(b) if a is None else a
            b = torch.zeros_like(a) if b is None else b
            out[n] = (a.double() - b.double())
    return out


def group_stats(dl, g):
    names = [n for n in dl if n.split(".")[0] == g]
    t = [dl[n] for n in names if dl[n] is not None]
    return {"n_with_grad": len(t), "n_nonzero": sum(1 for x in t if bool((x != 0).any())), "abs_sum": float(sum(float(x.abs().sum()) for x in t)), "abs_max": float(max([float(x.abs().max()) for x in t] or [0.0]))}


def check(jx, gx, jc, gc):
    why = []
    calls = jx["field_loss_calls"]
    if not calls or not all(math.isfinite(c["value"]) and c["value"] > 0 for c in calls):
        why.append(f"R1 field_loss not called with a finite positive value in the arm (calls={calls})")
    if jc["field_loss_calls"]:
        why.append("R1 field_loss WAS called in the control")
    dl = delta(gx, gc)
    st = {g: group_stats(dl, g) for g in PROPOSAL + SCORE}
    for g in PROPOSAL:
        if st[g]["n_nonzero"] == 0:
            why.append(f"R2 contribution is zero on {g}")
    for g in SCORE:
        if st[g]["abs_max"] != 0.0:
            why.append(f"R3 contribution is non-zero on {g}")
    return ("GREEN" if not why else "RED"), why, st


def main(d):
    d = Path(d)
    out = {}
    jF, gF = load(d, "F")
    jC, gC = load(d, "C")
    v, why, st = check(jF, gF, jC, gC)
    out["F_vs_C"] = {"verdict": v, "why": why, "groups": st, "F_field_loss_calls": jF["field_loss_calls"]}
    jN, gN = load(d, "N")
    dn = delta(gC, gN)
    mx = max([float(x.abs().max()) for x in dn.values() if x is not None] or [0.0])
    nnone = sum(1 for n in gC if (gC[n] is None) != (gN[n] is None))
    ok = (mx == 0.0) and nnone == 0 and not jC["field_loss_calls"] and jC["grad_sha256_all"] == jN["grad_sha256_all"]
    out["C_vs_N"] = {"verdict": "GREEN" if ok else "RED", "max_abs_diff_all_params": mx, "grad_none_mismatch": nnone, "sha_equal": jC["grad_sha256_all"] == jN["grad_sha256_all"]}
    if (d / "M.json").exists():
        jM, gM = load(d, "M")
        vm, whym, stm = check(jM, gM, jC, gC)
        out["M_vs_C"] = {"verdict": vm, "why": whym, "groups": stm, "M_field_loss_calls": jM["field_loss_calls"]}
    (d / "COMPARE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v_["verdict"] for k, v_ in out.items()}))
    for g in PROPOSAL + SCORE:
        s = out["F_vs_C"]["groups"][g]
        print(f"  F-C {g:13s} nonzero {s['n_nonzero']:4d}/{s['n_with_grad']:4d} abs_sum {s['abs_sum']:.6g} abs_max {s['abs_max']:.6g}")
    print("  C vs N:", out["C_vs_N"])
    if "M_vs_C" in out:
        print("  M vs C:", out["M_vs_C"]["verdict"], out["M_vs_C"]["why"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
