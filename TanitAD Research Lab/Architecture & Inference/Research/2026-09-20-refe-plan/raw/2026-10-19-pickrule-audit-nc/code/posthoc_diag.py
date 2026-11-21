"""POST-HOC, DESCRIPTIVE diagnostics after the registered read (PICKRULE-AUDIT-NC).  NOT part of the registration, NO authority over the outcome, NO threshold or exponent is chosen from it.
Written after result.json existed and labelled so.  Same arrays, same draws, same estimator.   python posthoc_diag.py  ->  run/results/posthoc_diag.json

 D1  the NC sigmoid of the EP-led NC-unsafe picks (set A) against P2's pick in the same set;
 D2  where the EP-led unsafe pick sits in the NC ranking of its own top-8 contenders (1 = lowest NC);
 D3  the precision of an NC flag (sigmoid < tau) on the EP-led picks: how many flagged picks are NC-unsafe;
 D4  a soft NC gate, aggregate with p_NC ** k (k = 1 is the planner's rule): the pick-level sweep of the composite, k in {1, 2, 4, 8, 16}, all k reported.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickrule_lib as P                                                                  # noqa: E402

RUN = "C:/Users/Admin/par18/pickrule/run"
KS = (1, 2, 4, 8, 16)


def soft_pick(z6: np.ndarray, k: float) -> np.ndarray:
    p = P.sigmoid64(z6)
    w = P.V1_W
    agg = (p[..., 0] ** k) * p[..., 1] * (p[..., 2] * w[0] + p[..., 3] * w[1] + p[..., 4] * w[2]) / float(sum(w))
    return P.first_argmax(agg)


def main() -> int:
    P.require_prereg()
    z = np.load(f"{RUN}/prepare/arrays.npz", allow_pickle=False)
    arr = {k: z[k] for k in z.files}
    F = P.load_fastneg()
    draws = F.boot_draws(P.N_LOGS, P.BOOT_B, P.BOOT_SEED)
    S = P.Stats(arr["log_idx"], draws, F)
    z6, lab = arr["z6"], arr["lab"]
    pdms = P.pdms_from_labels(lab)
    pk_p = P.rule_pick(z6)
    sig = P.sigmoid64(z6[..., 0])
    N = len(z6)
    rows = np.arange(N)
    pk_p_f = P.pick_flags(lab, pdms, pk_p)
    out = {"label": "POST-HOC, DESCRIPTIVE, no authority; written after result.json existed", "kind": "selection tier, SEEN scenes, P2 in-sample", "by_key": {}}
    for sd in P.SEEDS:
        ep = arr[f"ep_OOF_s{sd}"]
        zE = z6.copy()
        zE[..., P.EP] = ep
        pk_e = P.rule_pick(zE)
        f_e = P.pick_flags(lab, pdms, pk_e)
        A = f_e["nc0"] & ~pk_p_f["nc0"]
        se, sp = sig[rows, pk_e], sig[rows, pk_p]
        qn = lambda x: [float(v) for v in np.quantile(x, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])]          # noqa: E731
        d1 = {"n_A": int(A.sum()), "sigmoid_nc_EP_led_pick_q0_10_25_50_75_90_100": qn(se[A]), "sigmoid_nc_P2_pick_same_sets_q": qn(sp[A]), "delta_EP_led_minus_P2_q": qn(se[A] - sp[A]),
              "share_EP_led_sigmoid_ge_P2_pick": float((se[A] >= sp[A]).mean()), "min_sigmoid_nc_at_all_EP_led_picks": float(se.min()), "min_sigmoid_nc_at_all_P2_picks": float(sp.min())}
        # D2: rank of the EP-led pick's NC sigmoid among its own top-8 contenders (by the composite rule), 1 = lowest
        agg = P.rule_aggregate(zE)
        top8 = np.argsort(-agg, axis=1, kind="stable")[:, :8]
        nc8 = np.take_along_axis(sig, top8, axis=1)
        rank_pick = 1 + (nc8 < se[:, None]).sum(1)
        ranks = rank_pick[A]
        d2 = {"rank_of_nc_among_composite_top8_1_is_lowest": {str(r): int((ranks == r).sum()) for r in range(1, 9)}, "mean_rank": float(ranks.mean()),
              "all_sets_mean_rank": float(rank_pick.mean()), "share_A_at_rank_1": float((ranks == 1).mean())}
        # D3: precision of a flag sigmoid < tau on the EP-led picks (label NC == 0), against the base rate
        d3 = {}
        for t in (0.5, 0.7, 0.9):
            fl = se < t
            d3[str(t)] = {"flagged_sets": int(fl.sum()), "flagged_and_nc_unsafe": int((fl & f_e["nc0"]).sum()), "precision": (float((fl & f_e["nc0"]).sum() / fl.sum()) if fl.sum() else None),
                          "base_rate_nc_unsafe_pick": float(f_e["nc0"].mean())}
        # D3b: what the tau = 0.9 veto does to the flagged NC-unsafe picks (kept because no eligible proposal exists, replaced by another NC-unsafe pick, or replaced by an NC-safe pick)
        pk_v, fb_v = P.veto_pick(zE, 0.9)
        fu = (se < 0.9) & f_e["nc0"]
        repl_nc0 = P.at(lab[..., 0], pk_v) == 0.0
        d3["tau0.9_flagged_unsafe_fate"] = {"flagged_and_nc_unsafe": int(fu.sum()), "kept_fallback_no_eligible": int((fu & fb_v).sum()), "replaced_by_nc_unsafe": int((fu & ~fb_v & repl_nc0).sum()),
                                            "replaced_by_nc_safe_or_half": int((fu & ~fb_v & ~repl_nc0).sum()),
                                            "flagged_but_nc_safe_replaced": int(((se < 0.9) & ~f_e["nc0"] & ~fb_v).sum()), "replacement_made_nc_unsafe_from_safe": int(((se < 0.9) & ~f_e["nc0"] & ~fb_v & repl_nc0).sum())}
        # D4: soft gate p_NC ** k
        d4 = {}
        for k in KS:
            pk_k = soft_pick(zE, k)
            fk, f0 = P.pick_flags(lab, pdms, pk_k), pk_p_f
            d4[str(k)] = {"k": k, "picks_changed_vs_k1": int((pk_k != pk_e).sum()), "nc_unsafe_rate": S.rate(fk["nc0"]), "dac_unsafe_rate": S.rate(fk["dac0"]), "pdms_x100": P.scale_ci(S.mean(fk["pd"]), 100.0),
                          "d_nc_unsafe_rate_vs_P2": S.rate_diff(fk["nc0"], f0["nc0"]), "d_dac_unsafe_rate_vs_P2": S.rate_diff(fk["dac0"], f0["dac0"]), "d_pdms_x100_vs_P2": P.scale_ci(S.mean(fk["pd"] - f0["pd"]), 100.0)}
        # the identity of D4 at k = 1
        assert np.array_equal(soft_pick(zE, 1), pk_e), "soft gate at k = 1 must equal the planner rule"
        out["by_key"][f"OOF_s{sd}"] = {"D1_nc_sigmoid_of_unsafe_picks": d1, "D2_rank": d2, "D3_flag_precision": d3, "D4_soft_gate": d4}
    with open(f"{RUN}/results/posthoc_diag.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(out), indent=1, sort_keys=True))
    for key, v in out["by_key"].items():
        d1, d2, d3, d4 = v["D1_nc_sigmoid_of_unsafe_picks"], v["D2_rank"], v["D3_flag_precision"], v["D4_soft_gate"]
        print(key, "D1 n_A", d1["n_A"], "sigma_E q", [round(x, 3) for x in d1["sigmoid_nc_EP_led_pick_q0_10_25_50_75_90_100"]], "sigma_P q", [round(x, 3) for x in d1["sigmoid_nc_P2_pick_same_sets_q"]],
              "share E>=P", round(d1["share_EP_led_sigmoid_ge_P2_pick"], 3), "min sigma at EP-led picks", round(d1["min_sigmoid_nc_at_all_EP_led_picks"], 4))
        print("   D3b", d3["tau0.9_flagged_unsafe_fate"])
        print("   D2", d2["rank_of_nc_among_composite_top8_1_is_lowest"], "mean rank A", round(d2["mean_rank"], 2), "all", round(d2["all_sets_mean_rank"], 2))
        print("   D3", {t: (x["flagged_sets"], x["flagged_and_nc_unsafe"], None if x["precision"] is None else round(x["precision"], 3)) for t, x in d3.items() if t != "tau0.9_flagged_unsafe_fate"}, "base", round(d3["0.5"]["base_rate_nc_unsafe_pick"], 4))
        for k, x in d4.items():
            print("   D4 k", k, "changed", x["picks_changed_vs_k1"], "NC-unsafe", round(x["nc_unsafe_rate"]["point"], 4), "DAC-unsafe", round(x["dac_unsafe_rate"]["point"], 4), "dPDMS", round(x["d_pdms_x100_vs_P2"]["point"], 3),
                  [round(x["d_pdms_x100_vs_P2"]["lo"], 3), round(x["d_pdms_x100_vs_P2"]["hi"], 3)], "dNC", round(x["d_nc_unsafe_rate_vs_P2"]["point"], 4), [round(x["d_nc_unsafe_rate_vs_P2"]["lo"], 4), round(x["d_nc_unsafe_rate_vs_P2"]["hi"], 4)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
