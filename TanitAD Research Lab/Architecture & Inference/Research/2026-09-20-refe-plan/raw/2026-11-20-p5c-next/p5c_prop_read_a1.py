#!/usr/bin/env python3
"""P5C-PROP read: controls + selection headroom from p5c_prop_score.py rows (tracked = the benchmark tier, T1 executed via LQR).

AMENDMENT 1 (AMENDMENT_1.md): the selection aggregate is computed in float32, the planner's dtype.
CONTROLS (exit 2 when any fails; with --expect-red, exit 0 IFF CTRL-T FAILED -- the deliberate-regression arm):
  CTRL-T  tracked[pick] == harness CSV (nc, dac, ddc, ep, ttc, c, score) to 1e-9 on every token
  CTRL-P  tp20_pick == P5-C pass1 transparent row of the arm to 1e-9 on every token
  CTRL-S  argmax_k navsim_v1(sigmoid(logits[k])) == the dump's pick on every token (the selection rule is the one we think)
READS (descriptive on a smoke; decision-grade only on the full table, with the log-cluster bootstrap):
  per stratum (pick zeroed / non-zeroed): pick, oracle (best of 64), random (mean of 64) tracked PDMS x100; the share of zeroed tokens
  with >= 1 proposal scoring > 0 (SELECTION-BOUND, the D-P6-GAP-1 next-lever split) vs all 64 zero (GENERATION-BOUND);
  mix-weighted to the full-navtest stratum shares (--mix-zero-share); fixed-grid safety-weighted selection rules (no tuning:
  agg_a = (p_NC p_DAC)^a (5 p_EP + 5 p_TTC + 2 p_C)/12, a in {1, 2, 4, 8}, and a=1 with the TTC weight raised to 10);
  transparent: share of proposals passing transparent comfort, and tokens where a proposal >= the pick's tracked PDMS also passes it.
Usage: p5c_prop_read.py --rows <glob> --arm C1 --csv <harness csv> --pass1 <glob> --proposals <npz> --out OUT.json
       [--mix-zero-share 0.029392] [--boot 2000] [--expect-red]"""
import argparse, csv, glob, json, sys
import numpy as np

SUB = [("nc", "no_at_fault_collisions"), ("dac", "drivable_area_compliance"), ("ddc", "driving_direction_compliance"),
       ("ep", "ego_progress"), ("ttc", "time_to_collision_within_bound"), ("c", "comfort"), ("pdms", "score")]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def agg(L, a=1.0, wttc=5.0):
    p = sig(np.asarray(L, np.float32))
    return (p[..., 0] * p[..., 1]) ** a * (5 * p[..., 2] + wttc * p[..., 3] + 2 * p[..., 4]) / (7 + wttc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix-zero-share", type=float, default=None)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--expect-red", action="store_true")
    a = ap.parse_args()
    R = {}
    for f in sorted(glob.glob(a.rows)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if "error" not in r:
                R[r["token"]] = r
    toks = sorted(R)
    C = {}
    for r in csv.DictReader(open(a.csv, newline="", encoding="utf-8")):
        if r["valid"] == "True" and len(r["token"]) == 16:
            C[r["token"]] = {k: float(r[c]) for k, c in SUB}
    P1 = {}
    for f in sorted(glob.glob(a.pass1)):
        for ln in open(f, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("token") in R and "arms" in r:
                P1[r["token"]] = r["arms"][a.arm]["transparent"]
    z = np.load(a.proposals, allow_pickle=False)
    zi = {str(t): i for i, t in enumerate(z["token"])}
    I = np.array([zi[t] for t in toks])
    LG, PK = z["logits"][I].astype(np.float32), z["pick"][I]
    n = len(toks)
    tr = {k: np.array([R[t]["tr"][k] for t in toks], np.float64) for k in ("nc", "dac", "ddc", "ttc", "c", "ep", "pdms")}
    tp = {k: np.array([R[t]["tp"][k] for t in toks], np.float64) for k in ("nc", "dac", "ddc", "ttc", "c", "ep", "pdms")}
    pick = np.array([R[t]["pick"] for t in toks])
    assert (pick == PK).all()
    ar = np.arange(n)
    dT = {k: float(max(abs(tr[k][i, pick[i]] - C[t][k]) for i, t in enumerate(toks))) for k, _ in SUB}
    nT = int(sum(any(abs(tr[k][i, pick[i]] - C[t][k]) > 1e-9 for k, _ in SUB) for i, t in enumerate(toks)))
    dP = {k: float(max(abs(R[t]["tp20_pick"][k] - P1[t][k]) for t in toks)) for k, _ in SUB if all(t in P1 for t in toks)}
    nP = int(sum(any(abs(R[t]["tp20_pick"][k] - P1[t][k]) > 1e-9 for k, _ in SUB) for t in toks)) if dP else None
    sel = agg(LG).argmax(1)
    ctrl = {"CTRL_T": {"max_abs_diff": dT, "n_tokens_over_1e-9": nT, "pass": nT == 0},
            "CTRL_P": {"max_abs_diff": dP, "n_tokens_over_1e-9": nP, "n_pass1_rows_found": len(P1), "pass": bool(dP) and nP == 0},
            "CTRL_S": {"n_agree": int((sel == pick).sum()), "n": n, "pass": bool((sel == pick).all())}}
    out = {"arm": a.arm, "n_tokens": n, "n_logs": len({R[t]["log"] for t in toks}), "t_s_median": float(np.median([R[t]["t_s"] for t in toks])),
           "controls": ctrl, "tier": "T1 executed (NAVSIM v1.1 LQR + bicycle, the benchmark's tracker); transparent = no tracker",
           "evidence": "MEASURED (this run)"}
    if a.expect_red:
        out["red_expectation"] = "CTRL_T must FAIL"
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps(out["controls"], indent=1))
        return 0 if not ctrl["CTRL_T"]["pass"] else 3
    pp = tr["pdms"] * 100
    pk = pp[ar, pick]
    zero = pk == 0
    rules = {"navsim_v1 (pick)": agg(LG).argmax(1)}
    for al in (2.0, 4.0, 8.0):
        rules[f"safety_power_a{al:g}"] = agg(LG, al).argmax(1)
    rules["ttc_weight_10"] = agg(LG, 1.0, 10.0).argmax(1)
    hc = np.array([R[t]["tp"]["c"] for t in toks], np.float64)

    def stratum(m):
        if not m.any():
            return {"n": 0}
        d = {"n": int(m.sum()), "pick": float(pk[m].mean()), "oracle": float(pp[m].max(1).mean()), "random": float(pp[m].mean(1).mean()),
             "share_props_zero": float((pp[m] == 0).mean() * 100)}
        for nm, s in rules.items():
            d[f"rule::{nm}"] = float(pp[m][np.arange(m.sum()), s[m]].mean())
        return d
    out["strata"] = {"zeroed": stratum(zero), "nonzeroed": stratum(~zero)}
    out["zero_split"] = {"n_zeroed": int(zero.sum()), "selection_bound_any_prop_gt0": int((pp[zero].max(1) > 0).sum()),
                         "generation_bound_all64_zero": int((pp[zero].max(1) == 0).sum()),
                         "selection_bound_pct": float((pp[zero].max(1) > 0).mean() * 100) if zero.any() else None,
                         "zeroed_by_dac": int((tr["dac"][ar, pick][zero] == 0).sum()), "zeroed_by_nc": int((tr["nc"][ar, pick][zero] == 0).sum())}
    ttc0 = (tr["ttc"][ar, pick] == 0) & ~zero
    out["ttc0_nonzeroed"] = {"n": int(ttc0.sum()), "share_with_a_prop_ttc1_and_mult_gt0_pct":
                             float(((tr["ttc"][ttc0] == 1) & (tr["nc"][ttc0] * tr["dac"][ttc0] > 0)).any(1).mean() * 100) if ttc0.any() else None}
    better_comf = (pp >= pk[:, None]) & (hc == 1)
    out["transparent"] = {"share_props_comfort_pass_pct": float((hc == 1).mean() * 100), "pick_comfort_pass_pct": float((hc[ar, pick] == 1).mean() * 100),
                          "tokens_with_prop_ge_pick_tracked_and_transp_comfort_pct": float(better_comf.any(1).mean() * 100),
                          "transparent_oracle_pdms": float((tp["pdms"] * 100).max(1).mean()), "transparent_pick8_pdms": float((tp["pdms"][ar, pick] * 100).mean())}
    if a.mix_zero_share is not None:
        w = a.mix_zero_share
        mix = {}
        for key in ["pick", "oracle", "random"] + [f"rule::{nm}" for nm in rules]:
            zs, ns = out["strata"]["zeroed"], out["strata"]["nonzeroed"]
            if zs.get("n") and ns.get("n"):
                mix[key] = w * zs[key] + (1 - w) * ns[key]
        out["mix_full_navtest_weights"] = {"zero_share": w, **mix}
    if a.boot:
        logs = np.array([R[t]["log"] for t in toks])
        ul = sorted(set(logs))
        ix = {l: np.where(logs == l)[0] for l in ul}
        rng = np.random.default_rng(20261120)
        bs = {k: [] for k in ["oracle-pick"] + [f"{nm}-pick" for nm in rules if nm != "navsim_v1 (pick)"]}
        for _ in range(a.boot):
            s = np.concatenate([ix[ul[i]] for i in rng.integers(0, len(ul), len(ul))])
            bs["oracle-pick"].append((pp[s].max(1) - pk[s]).mean())
            for nm, r in rules.items():
                if nm != "navsim_v1 (pick)":
                    bs[f"{nm}-pick"].append((pp[s, r[s]] - pk[s]).mean())
        out["paired_log_cluster_bootstrap_unweighted"] = {k: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "point":
                                                           float(np.mean(v))} for k, v in bs.items()}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    ok = ctrl["CTRL_T"]["pass"] and ctrl["CTRL_P"]["pass"] and ctrl["CTRL_S"]["pass"]
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
