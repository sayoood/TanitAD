#!/usr/bin/env python3
"""H-ADAM section 3.3 (PREREG_H_ADAM.md): REALISED parameter displacement from the banked scorer states.  MEASURED, no substituted input.

  d_p(X,u)  = ||theta_B(X,u) - theta_A(X,u)||   X in {R0, R0b, C0};  C0's B arm is BCE-only too -> its d_p is the numerical floor
  e_p(X,u)  = sqrt(max(d_p(X,u)^2 - d_p(C0,u)^2, 0))
  rho_u     = sqrt(sum_C80 e_p(R0b,u)^2 / sum_C80 e_p(R0,u)^2),   C80 = tensors carrying 80 % of sum_p e_p(R0,u)^2

    python disp_real.py --dl C:/Users/Admin/rls1_adam_dl --out disp_real.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

PREFIX = ("score_q_mlp.", "score_dec.", "score_head.")


def flat_state(st):
    """{module: {k: tensor}} -> {'module.k': float64 numpy}"""
    return {f"{m}.{k}": v.detach().double().numpy() for m, d in st.items() for k, v in d.items()}


def load_base(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {k: v.detach().double().numpy() for k, v in ck["model"].items() if k.startswith(PREFIX)}


def load_states(dl, run, u):
    p = os.path.join(dl, f"rls1-{run}", "scorer_states", f"scorer_states_u{u}.pt")
    ss = torch.load(p, map_location="cpu", weights_only=False)
    assert int(ss["update"]) == u, (p, ss["update"])
    return flat_state(ss["A"]), flat_state(ss["B"])


def nrm(x):
    return float(np.sqrt((x * x).sum()))


def c80(energy: dict, share=0.8):
    tot = sum(energy.values())
    out, acc = [], 0.0
    for k, v in sorted(energy.items(), key=lambda kv: -kv[1]):
        out.append(k)
        acc += v
        if acc >= share * tot:
            break
    return out


def group_of(name):
    if name.startswith("score_head."):
        return "score_head"
    if name.startswith("score_q_mlp."):
        return "score_q_mlp"
    if ".n1." in name or ".n2." in name or ".n3." in name:
        return "score_dec.layernorm"
    if ".self_attn." in name:
        return "score_dec.self_attn"
    if ".cross." in name:
        return "score_dec.cross_attn"
    if ".mlp." in name:
        return "score_dec.mlp"
    return "other"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default="C:/Users/Admin/rls1_adam_dl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = load_base(os.path.join(a.dl, "rls1", "base", "model_sft4_B_u400.pt"))
    res = {"tool": "disp_real/1", "class": "MEASURED (banked parameter tensors; one training seed per arm; RL-val selection tier context)",
           "n_tensors": len(base), "n_params": int(sum(v.size for v in base.values())), "updates": {}}
    S = {(r, u): load_states(a.dl, r, u) for r, us in (("r0", (250, 500, 1000)), ("r0b", (250, 500)), ("c0", (250, 500, 1000))) for u in us}
    names = sorted(base)
    for u in (250, 500):
        d = {}
        for r in ("r0", "r0b", "c0"):
            A, B = S[(r, u)]
            assert sorted(A) == names == sorted(B), "tensor set differs from base"
            d[r] = {k: {"AB": nrm(B[k] - A[k]), "B0": nrm(B[k] - base[k]), "A0": nrm(A[k] - base[k])} for k in names}
        e = {r: {k: float(np.sqrt(max(d[r][k]["AB"] ** 2 - d["c0"][k]["AB"] ** 2, 0.0))) for k in names} for r in ("r0", "r0b")}
        core = c80({k: e["r0"][k] ** 2 for k in names})
        core_raw = c80({k: d["r0"][k]["AB"] ** 2 for k in names})

        def rho(num, den, ks):
            return float(np.sqrt(sum(num[k] ** 2 for k in ks) / sum(den[k] ** 2 for k in ks)))
        rho_corr = rho(e["r0b"], e["r0"], core)
        rho_raw = rho({k: d["r0b"][k]["AB"] for k in names}, {k: d["r0"][k]["AB"] for k in names}, core_raw)
        rho_all = rho(e["r0b"], e["r0"], names)
        A0, B0 = S[("r0", u)]
        A1, B1 = S[("r0b", u)]
        v0 = np.concatenate([(B0[k] - A0[k]).ravel() for k in core])
        v1 = np.concatenate([(B1[k] - A1[k]).ravel() for k in core])
        cos = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        # projection of R0b's EPO displacement on R0's: how much of R0b's movement lies along R0's, in units of R0's norm
        proj = float(v0 @ v1 / (np.linalg.norm(v0) ** 2))
        grp = {}
        for k in names:
            g = group_of(k)
            grp.setdefault(g, {"n_params": 0, "e2_R0": 0.0, "e2_R0b": 0.0, "AB2_C0": 0.0, "AB2_R0": 0.0, "AB2_R0b": 0.0})
            grp[g]["n_params"] += int(base[k].size)
            grp[g]["e2_R0"] += e["r0"][k] ** 2
            grp[g]["e2_R0b"] += e["r0b"][k] ** 2
            grp[g]["AB2_C0"] += d["c0"][k]["AB"] ** 2
            grp[g]["AB2_R0"] += d["r0"][k]["AB"] ** 2
            grp[g]["AB2_R0b"] += d["r0b"][k]["AB"] ** 2
        tot0 = sum(g["e2_R0"] for g in grp.values())
        for g in grp.values():
            g["share_of_R0_EPO_energy"] = g["e2_R0"] / tot0
            g["rho_group"] = float(np.sqrt(g["e2_R0b"] / g["e2_R0"])) if g["e2_R0"] > 0 else None
        res["updates"][str(u)] = {
            "rho_floor_corrected_C80": rho_corr, "rho_uncorrected_C80": rho_raw, "rho_floor_corrected_all_tensors": rho_all,
            "cos_R0b_vs_R0_EPOdisp_C80": cos, "projection_of_R0b_on_R0_norm_units": proj,
            "n_C80_tensors": len(core), "C80_tensors": core,
            "global_norms": {"||dAB|| R0": nrm(np.concatenate([(B0[k] - A0[k]).ravel() for k in names])),
                             "||dAB|| R0b": nrm(np.concatenate([(B1[k] - A1[k]).ravel() for k in names])),
                             "||dAB|| C0 (floor)": nrm(np.concatenate([(S[('c0', u)][1][k] - S[('c0', u)][0][k]).ravel() for k in names])),
                             "||B-base|| R0": nrm(np.concatenate([(B0[k] - base[k]).ravel() for k in names])),
                             "||B-base|| R0b": nrm(np.concatenate([(B1[k] - base[k]).ravel() for k in names])),
                             "||A-base|| R0": nrm(np.concatenate([(A0[k] - base[k]).ravel() for k in names])),
                             "||A-base|| C0": nrm(np.concatenate([(S[('c0', u)][0][k] - base[k]).ravel() for k in names]))},
            "groups": grp,
            "per_tensor": {k: {"d_AB_R0": d["r0"][k]["AB"], "d_AB_R0b": d["r0b"][k]["AB"], "d_AB_C0_floor": d["c0"][k]["AB"],
                               "e_R0": e["r0"][k], "e_R0b": e["r0b"][k],
                               "rho_p": (e["r0b"][k] / e["r0"][k]) if e["r0"][k] > 0 else None,
                               "B0_R0": d["r0"][k]["B0"], "B0_R0b": d["r0b"][k]["B0"], "numel": int(base[k].size)} for k in names}}
    # u1000 (R0 vs C0 only): the late-run EPO displacement size, for scale
    A, B = S[("r0", 1000)]
    Ac, Bc = S[("c0", 1000)]
    res["u1000_R0_vs_C0"] = {"||B-A|| R0": nrm(np.concatenate([(B[k] - A[k]).ravel() for k in names])),
                             "||B-A|| C0 (floor)": nrm(np.concatenate([(Bc[k] - Ac[k]).ravel() for k in names]))}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    for u in ("250", "500"):
        r = res["updates"][u]
        print("u" + u, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("rho_floor_corrected_C80", "rho_uncorrected_C80", "rho_floor_corrected_all_tensors", "cos_R0b_vs_R0_EPOdisp_C80", "projection_of_R0b_on_R0_norm_units", "n_C80_tensors")})
        print("   ", {k: round(v, 5) for k, v in r["global_norms"].items()})
        for g, v in sorted(r["groups"].items(), key=lambda kv: -kv[1]["share_of_R0_EPO_energy"]):
            print("   ", g, "share", round(v["share_of_R0_EPO_energy"], 3), "rho", None if v["rho_group"] is None else round(v["rho_group"], 3), "n", v["n_params"])
    print("u1000", res["u1000_R0_vs_C0"])
    print("ZZDISP_OK", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
