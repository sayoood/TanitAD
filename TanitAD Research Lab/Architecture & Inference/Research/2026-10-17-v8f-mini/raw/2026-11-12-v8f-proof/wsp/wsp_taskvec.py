"""wsp_taskvec.py -- H-V8F-PROOF-2 P2a (PREREG_V8F_PROOF.md): where does the stop lever live in weight space?  Pure tensor math, CPU.
  delta_G2 = theta[BAL-G2_s0] - theta[BAL-NULL_s0] (lever, dose 2);  delta_G = theta[BAL-G_s0] - theta[BAL-NULL_s0] (dose 1);  nu = theta[BAL-NULL_rep] - theta[BAL-NULL_s0] (seed noise).
Per module group (over the trainable tensors only; frozen tensors and buffers are excluded and listed): ||delta_G2||/||nu||, ||delta_G||/||nu||, cos(delta_G, delta_G2), cos(delta_G2, nu),
||delta_G||/||delta_G2||.  Norms are L2 over every element of the group's tensors, concatenated.
Groups (wsp_common.group_of, fixed before any number existed): adapter (adapter.*, pos, time), cond (e_*), S (read_s), T (read_t), O (read_o), head_S, head_O, head_T (trunk_t, h_goal, h_lat, h_lon, h_cons), WM (wm.*).
Card rule P2a: LOCALISED if the T-group ratio exceeds every other trainable group's ratio by a factor >= 1.5, DIFFUSE otherwise.  The card says "largest in T and in T's heads", so the PRIMARY reading
takes T-group = T union head_T (T-block); the card-literal alternative T-group = read_t alone is reported beside, with its own outcome.  Both are in the script before any number was computed.
Writes TASKVEC.json and TASKVEC.md beside this file."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsp_common as C  # noqa: E402
import torch  # noqa: E402

FACTOR = 1.5
OTHERS_PRIMARY = ("adapter", "cond", "S", "O", "head_S", "head_O", "WM")
OTHERS_LITERAL = ("adapter", "cond", "S", "O", "head_S", "head_O", "head_T", "WM")


def main():
    ck = {t: C.load(d + "/ckpt_last.pt") for t, d in C.BANKS.items()}
    sd = {t: ck[t]["model"] for t in ck}
    m = C.build_model(ck["NULL_s0"])
    tn = [k for k in C.trainable_names(m)]
    frozen = [k for k in sd["NULL_s0"] if k not in set(tn)]
    d2 = {k: (sd["G2_s0"][k] - sd["NULL_s0"][k]).double().flatten() for k in tn}
    d1 = {k: (sd["G_s0"][k] - sd["NULL_s0"][k]).double().flatten() for k in tn}
    nu = {k: (sd["NULL_rep"][k] - sd["NULL_s0"][k]).double().flatten() for k in tn}
    groups = {}
    for k in tn:
        groups.setdefault(C.group_of(k), []).append(k)
    groups = {g: v for g, v in groups.items()}
    assert not [g for g in groups if g.startswith("other:")], [g for g in groups if g.startswith("other:")]

    def stats(keys):
        a2, a1, n = (torch.cat([x[k] for k in keys]) for x in (d2, d1, nu))
        n2, n1, nn = float(a2.norm()), float(a1.norm()), float(n.norm())
        cos = lambda x, y: float((x @ y) / (x.norm() * y.norm())) if float(x.norm()) > 0 and float(y.norm()) > 0 else None
        return {"n_tensors": len(keys), "n_elements": int(a2.numel()), "norm_delta_G2": n2, "norm_delta_G": n1, "norm_nu": nn,
                "ratio_G2_over_nu": n2 / nn if nn > 0 else None, "ratio_G_over_nu": n1 / nn if nn > 0 else None,
                "cos_G_G2": cos(a1, a2), "cos_G2_nu": cos(a2, n), "norm_G_over_norm_G2": n1 / n2 if n2 > 0 else None}
    G = {g: stats(v) for g, v in groups.items()}
    G["T_block(T+head_T)"] = stats(groups["T"] + groups["head_T"])
    G["ALL_trainable"] = stats(tn)
    r = {g: G[g]["ratio_G2_over_nu"] for g in G}
    tb = r["T_block(T+head_T)"]
    prim = {"T_group": "T union head_T", "T_ratio": tb, "others": {g: r[g] for g in OTHERS_PRIMARY}}
    prim["max_other"] = max(prim["others"].values())
    prim["factor_over_max_other"] = tb / prim["max_other"]
    prim["outcome"] = "LOCALISED" if tb >= FACTOR * prim["max_other"] else "DIFFUSE"
    lit = {"T_group": "read_t only", "T_ratio": r["T"], "others": {g: r[g] for g in OTHERS_LITERAL}}
    lit["max_other"] = max(lit["others"].values())
    lit["factor_over_max_other"] = r["T"] / lit["max_other"]
    lit["outcome"] = "LOCALISED" if r["T"] >= FACTOR * lit["max_other"] else "DIFFUSE"
    rows = []
    for k in tn:
        nk = float(nu[k].norm())
        if nk > 0:
            rows.append({"tensor": k, "group": C.group_of(k), "numel": int(nu[k].numel()), "norm_delta_G2": float(d2[k].norm()), "norm_delta_G": float(d1[k].norm()), "norm_nu": nk,
                         "ratio_G2_over_nu": float(d2[k].norm()) / nk})
    rows.sort(key=lambda x: -x["ratio_G2_over_nu"])
    zero_nu = [k for k in tn if float(nu[k].norm()) == 0.0]
    zero_d2 = [k for k in tn if float(d2[k].norm()) == 0.0]
    out = {"card": "PREREG_V8F_PROOF.md H-V8F-PROOF-2 P2a (frozen 2026-11-13)", "evidence": "MEASURED (pure tensor arithmetic on the four banked checkpoints, C:/Users/Admin/t2v8f_pulls/stopbal_BAL-{NULL_s0,NULL_rep,G_s0,G2_s0}/ckpt_last.pt)",
           "tier": "weight-space diagnostic; no tier (no model pass)", "estimator": "none: point values of one seed per arm (one replicate gives one nu); no interval",
           "rule": f"LOCALISED if T-group ratio >= {FACTOR} x every other trainable group's ratio, else DIFFUSE", "groups": G, "P2a_primary": prim, "P2a_literal_read_t_only": lit,
           "outcome_primary": prim["outcome"], "frozen_excluded": frozen, "trainable_with_zero_nu": zero_nu, "trainable_with_zero_delta_G2": zero_d2,
           "top15_tensors_by_ratio": rows[:15], "caveat": "nu comes from ONE replicate pair (NULL_s0 vs NULL_rep): the ratio has no spread estimate; delta_G2 and nu share init but differ in data order (nu changes the seed, the lever arms keep seed 0)."}
    json.dump(out, open(C.WSP + "/TASKVEC.json", "w"), indent=1)
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    L = ["# TASKVEC (H-V8F-PROOF-2, P2a), MEASURED, evidence `F:/TanitAD-scratch/v8proof/wsp/TASKVEC.json`", "",
         "delta_G2 = G2_s0 - NULL_s0, delta_G = G_s0 - NULL_s0, nu = NULL_rep - NULL_s0 (trainable tensors only; 252 of 276; frozen 24 excluded). Weight-space diagnostic, one replicate pair, no interval.", "",
         "| group | tensors | ||dG2||/||nu|| | ||dG||/||nu|| | cos(dG,dG2) | cos(dG2,nu) | ||dG||/||dG2|| |", "|---|---|---|---|---|---|---|"]
    for g in list(groups) + ["T_block(T+head_T)", "ALL_trainable"]:
        s = G[g]
        L.append(f"| {g} | {s['n_tensors']} | {f(s['ratio_G2_over_nu'])} | {f(s['ratio_G_over_nu'])} | {f(s['cos_G_G2'])} | {f(s['cos_G2_nu'])} | {f(s['norm_G_over_norm_G2'])} |")
    L += ["", f"P2a primary (T-group = T + head_T): T-block ratio {f(tb)}; largest other group {f(prim['max_other'])}; factor {f(prim['factor_over_max_other'])} (bar {FACTOR}) -> **{prim['outcome']}**.",
          f"P2a card-literal (T-group = read_t only): ratio {f(r['T'])}; largest other {f(lit['max_other'])}; factor {f(lit['factor_over_max_other'])} -> **{lit['outcome']}**.", "",
          "Top 15 tensors by ||dG2||/||nu||: " + "; ".join(f"{x['tensor']} {x['ratio_G2_over_nu']:.2f}" for x in rows[:15]) + ".", "",
          f"Trainable tensors with ||nu|| = 0: {len(zero_nu)}; with ||dG2|| = 0: {len(zero_d2)}. Caveat: nu is one replicate (no spread); lever arms keep seed 0, nu changes the seed."]
    open(C.WSP + "/TASKVEC.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
