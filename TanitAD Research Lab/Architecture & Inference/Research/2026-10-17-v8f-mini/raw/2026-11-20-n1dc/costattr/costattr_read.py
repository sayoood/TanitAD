"""costattr_read.py -- card PREREG_V8F_N1DC_COSTATTR (sha256 295db8eb) sections 3-5. P1 = U - N1D, P2 = N1DC - U per seed, 24 cells, floor F2 of the guard,
paired episode-cluster bootstrap B 2000 seed 0 ('another draw of EPISODES' only). Same instrument functions and K-FL-HB loading rule as eval141/n1dc_read141.py. Zero training."""
import json
import sys
import time

import numpy as np

E = "F:/TanitAD-scratch/n1dc/read/eval141"
H = "F:/TanitAD-scratch/n1dc/costattr"
N2R = "F:/TanitAD-scratch/strat1/navrb2/read"
SR = "F:/TanitAD-scratch/strat1/read"
for _p in (E, N2R, SR, SR + "/theory/obd/read", SR + "/theory/otband/read"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SEEDS = (0, 1)
NB = 2000
COST_CELLS = {0: ["TAC-LON"], 1: ["operative_0_2s:ade_m", "operative_0_2s:along_mae_m", "operative_0_2s:dk_hw_abs_err_m"]}


def label(p1, p2):
    """pure: p1 = P1 row of the cell (has 'cost'), p2 = P2 row (has 'separated', 'cost_dir')"""
    if p1.get("cost"):
        return "VIA-WEIGHTS"
    if (not p1.get("separated")) and p2.get("separated") and p2.get("cost_dir"):
        return "VIA-OUTPUT"
    return "UNRESOLVED"


def main():
    t0 = time.time()
    import navrb2_lib as NL
    import navrb2_read as N2
    import strat1_lib as L
    import strat1_read as S
    import obd_read as OR
    import obd_geom as G
    CI = L.load_ci()
    rep = {"tool": "costattr_read.py", "card": "PREREG_V8F_N1DC_COSTATTR.md sha256 295db8eb", "tier": "T1-kin/hb0.2", "evidence_class": "MEASURED (ours; dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap B={NB} seed 0 ({CI.SOURCE}); 'another draw of EPISODES' only", "controls": {}}
    prob = []
    geom = G.build()
    bk = lambda x: f"{SR}/t1/dump_STRAT1-{x}_m10_causal_heads"  # noqa: E731
    hbd = lambda x: f"{E}/dumps141_hb02/dump_N1DC-{x}_m10_causal_heads"  # noqa: E731
    ud = lambda s: f"{H}/dumps141_hb02/dump_N1DC-N1DC_s{s}U_m10_causal_heads"  # noqa: E731
    dg = S.load_dump_n(bk("C0_s0"), 141)
    J = geom.join(dg)
    glat, glon = geom.labels(dg["g"])
    e141 = dg["eid"]
    fl_hb = OR.load_dump(hbd("N1D_s0"))["fl"]
    cells, ksame = {}, {}
    todo = []
    for s in SEEDS:
        todo += [(f"N1DC_s{s}", hbd(f"N1DC_s{s}")), (f"N1D_s{s}", hbd(f"N1D_s{s}")), (f"U_s{s}", ud(s))]
    for name, d in todo:
        x = OR.load_dump(d)
        ksame[name] = {k: bool(np.array_equal(x[k], fl_hb if k == "fl" else dg[k])) for k in OR.SAME}
        cells[name] = OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)
    rep["controls"]["K-SAME_K-FL-HB"] = {"pass": all(all(v.values()) for v in ksame.values()), "per_arm": ksame,
                                         "must_read": "eid, clip_index, ws, g, v0 equal C0_s0; fl equal the hb0.2 floor (N1D_s0 hb0.2), for U and for the comparators"}
    c0rep = json.load(open(NL.C0REP_RESULT, encoding="utf-8"))
    F2 = N2.floor2(c0rep["floors"]["F"], c0rep["floors"]["F_rep"])
    k1_bad = {}
    for s in SEEDS:
        for c, v in S.pair_rows(CI, cells[f"U_s{s}"], cells[f"U_s{s}"], e141, F2, 200).items():
            d = v.get("delta")
            if d is None or (d != 0.0 and d == d):
                k1_bad[f"U_s{s}:{c}"] = d
    rep["controls"]["K1_U_self_pair_exactly_0"] = {"pass": not k1_bad, "nonzero": k1_bad}
    kc = json.load(open(f"{H}/K-COPY.json"))
    rep["controls"]["K-COPY"] = {"pass": bool(kc["pass"]), "source": "K-COPY.json (kcopy.py); pulled sha256 equal the pod's (pod_sha_raw.txt)",
                                 "model_tensors_per_ckpt": kc["0"]["n_model_tensors"], "zero_tensor": kc["0"]["zero_keys"],
                                 "other_model_tensors_equal": kc["0"]["n_model_tensors"] - 1}
    P1, P2, kadd, kguard = {}, {}, {}, {}
    ref = json.load(open(f"{E}/out/RESULT_N1DC_EVAL141.json", encoding="utf-8"))["guard_eval141_hb02_N1DC-N1D"]
    for s in SEEDS:
        U, C, B = cells[f"U_s{s}"], cells[f"N1DC_s{s}"], cells[f"N1D_s{s}"]
        P1[s] = S.pair_rows(CI, U, B, e141, F2, NB)
        P2[s] = S.pair_rows(CI, C, U, e141, F2, NB)
        g0 = S.pair_rows(CI, C, B, e141, F2, NB)
        kadd[s] = float(max(abs((P1[s][c]["delta"] + P2[s][c]["delta"]) - g0[c]["delta"]) for c in OR.CELLS if g0[c].get("status") == "OK"))
        mine = json.loads(json.dumps(S.slim(g0), default=float))
        want = ref[f"N1DC_s{s}-N1D_s{s}"]
        bad = [f"{c}.{k}" for c in want for k, v in want[c].items() if mine.get(c, {}).get(k) != v]
        kguard[s] = {"n_fields": sum(len(v) for v in want.values()), "mismatches": bad[:20]}
    rep["controls"]["K-ADD"] = {"pass": max(kadd.values()) <= 1e-12, "max_abs_by_seed": {str(s): v for s, v in kadd.items()}, "must_read": "<= 1e-12"}
    rep["controls"]["K-GUARD"] = {"pass": all(not v["mismatches"] for v in kguard.values()), "by_seed": {str(s): v for s, v in kguard.items()}, "must_read": "exact on every field"}
    for k, v in rep["controls"].items():
        if not v["pass"]:
            prob.append(k + " failed")

    def row(v, c):
        r = S.slim({c: v})[c]
        sep, dl = r.get("separated"), r.get("delta")
        r["cost_dir"] = bool(sep and dl is not None and ((dl < 0) if c in OR.ACCURACY else (dl > 0)))
        return r
    rep["P1_U_minus_N1D"] = {str(s): {c: row(P1[s][c], c) for c in OR.CELLS} for s in SEEDS}
    rep["P2_N1DC_minus_U"] = {str(s): {c: row(P2[s][c], c) for c in OR.CELLS} for s in SEEDS}
    rep["rule_note"] = "P1 cost = STRAT1 rule with F2 (separated, worse side, above floor). P2 'separated in the cost direction' = separated and worse side (floor not applied, as the card words it); cost_dir."
    for s in SEEDS:
        got = sorted(c for c, v in ref[f"N1DC_s{s}-N1D_s{s}"].items() if v.get("cost"))
        if got != sorted(COST_CELLS[s]):
            prob.append(f"guard cost cells at s{s} are {got}, not the card's {COST_CELLS[s]}")
    labels = {}
    for s in SEEDS:
        for c in COST_CELLS[s]:
            labels[f"s{s}:{c}"] = {"label": label(rep["P1_U_minus_N1D"][str(s)][c], rep["P2_N1DC_minus_U"][str(s)][c]),
                                   "P1": rep["P1_U_minus_N1D"][str(s)][c], "P2": rep["P2_N1DC_minus_U"][str(s)][c],
                                   "other_seed_beside_no_label": {"P1": rep["P1_U_minus_N1D"][str(1 - s)][c], "P2": rep["P2_N1DC_minus_U"][str(1 - s)][c]}}
    if prob:
        rep["status"] = "INVALID (control problems; no label printed)"
        rep["problems"] = prob
    else:
        rep["status"] = "VALID"
        rep["labels"] = labels
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(f"{H}/RESULT_N1DC_COSTATTR.json", "w"), indent=1, default=float)
    lines = [f"COSTATTR read, status {rep['status']}, card sha256 295db8eb"]
    for k, v in rep["controls"].items():
        lines.append(f"  control {k}: pass={v['pass']}")
    if prob:
        lines += ["  " + p for p in prob]
    else:
        def f(r):
            return f"d {r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] sep {r['separated']} floor {r['floor']:.4f} cost {r['cost']} costdir {r['cost_dir']}"
        for k, v in labels.items():
            o = v["other_seed_beside_no_label"]
            lines.append(f"{k}: {v['label']}\n    P1 {f(v['P1'])}\n    P2 {f(v['P2'])}\n    other seed beside: P1 {f(o['P1'])}\n                       P2 {f(o['P2'])}")
    open(f"{H}/READ_COSTATTR.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if prob else 0)


if __name__ == "__main__":
    main()
