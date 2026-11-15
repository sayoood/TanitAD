# pool_rb.py : the pooled lead read over seeds of note PREREG_V8F_PROOF_ADD18_N2, section 4 (statistics A, A2, V) with its controls p1 to p4,
# and, reported beside, calibrated forms that a difference of spread between two reads cannot move (A3, A3r, A3c; note PREREG_V8F_PROOF_ADD18_N3).
# Tier T0 (a world-model diagnostic on logged frames, never driving performance). Reads the out-of-fold files OOF_RB_<ARM>_s<k>.npz of rb_trained.py
# (per window: label yk, episode index, probe reads of the rolled-out state p_i, the present state p_ii, the true future state p_iii, the ego floors
# p_iv / p_ivb) and nothing else of the arms.
#   python pool_rb.py --arm RR --seeds 0,1,2,3 --new-seeds 2,3 --base G --out-dir <dir>
# Two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; B 2,000, rng seed 0, percentiles 2.5 and 97.5.
# Calibrated forms: every read is first mapped to the label by a linear fit made out of fold (5 folds by episode), so a read with a smaller or larger
# spread gains or loses nothing by its spread alone:
#   A3  = R2 of (present read and rolled-out read together) minus R2 of the present read alone: what the rollout ADDS to the present state;
#   A3r = R2 of the rolled-out read alone minus R2 of the present read alone;
#   A3c = R2 of (present read and true-future read together) minus R2 of the present read alone: the most the probe could show (headroom);
#   A3e, A3ec = A3 and A3c with the ego-only floor read added on both sides: what the rollout adds beyond the present state AND the ego motion.
import argparse
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rb_change as RC  # noqa: E402  (load_arm, the bank path, resprobe as RC.RP)

RP = RC.RP
TIER = "T0 (a world-model diagnostic on logged frames, never driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL: not measured by this read; STRATEGIC: PAUSED (PI item 25)."
OOF_DIRS = ("C:/Users/Admin/v8proof_c/ctxroll/rb", "F:/TanitAD-scratch/v8proof/spatial/q2/out")
RES_DIR = "F:/TanitAD-scratch/v8proof/spatial/q2/out_m"
PAIRS = {"A": ("i", "ii"), "A2": ("i", "fl"), "A3": ("c_i2", "c_ii"), "A3r": ("c_i", "c_ii"), "A3c": ("c_t2", "c_ii"), "A3e": ("c_iifi", "c_iif"), "A3ec": ("c_iift", "c_iif")}   # statistic = R2 of the first minus R2 of the second
CAL = {"c_ii": ("ii",), "c_i": ("i",), "c_i2": ("ii", "i"), "c_t2": ("ii", "iii"), "c_iif": ("ii", "fl"), "c_iifi": ("ii", "fl", "i"), "c_iift": ("ii", "fl", "iii")}
LABEL = {"A": "A: rollout minus copy", "A2": "A2: rollout minus the ego floor", "V": "V: change skill", "A3": "A3: rollout added to the calibrated copy",
         "A3r": "A3r: calibrated rollout minus calibrated copy", "A3c": "A3c: true future added to the calibrated copy (headroom)", "A3e": "A3e: rollout added to the calibrated copy and ego floor", "A3ec": "A3ec: true future added to the calibrated copy and ego floor (headroom)", "P": "rollout read, arm minus base", "D3": "A3, arm minus base", "D3e": "A3e, arm minus base"}


def find_oof(name):
    for d in OOF_DIRS:
        p = os.path.join(d, "OOF_RB_%s.npz" % name)
        if os.path.exists(p):
            return p
    return None


def find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def calibrate(yk, epi, P, n_fold=5):
    m = np.isfinite(yk) & np.isfinite(P["i"]) & np.isfinite(P["ii"]) & np.isfinite(P["iii"]) & np.isfinite(P["fl"])
    out = {k: np.full(len(yk), np.nan) for k in CAL}
    fold = epi % n_fold
    for f in range(n_fold):
        tr, te = m & (fold != f), m & (fold == f)
        for k, cols in CAL.items():
            X = np.column_stack([np.ones(int(tr.sum()))] + [P[c][tr] for c in cols])
            w = np.linalg.lstsq(X, yk[tr], rcond=None)[0]
            out[k][te] = np.column_stack([np.ones(int(te.sum()))] + [P[c][te] for c in cols]) @ w
    return out


def load_seed(name, gap0_of, perm_rng=None, copy_as_rollout=False):
    path = find_oof(name)
    if path is None:
        raise FileNotFoundError("no OOF_RB file for " + name)
    z = np.load(path, allow_pickle=True)
    yk = z["yk"].astype(np.float64)
    epi = z["w_epi"].astype(np.int64)
    n_ep = len(z["uep"])
    P = {k: z["p_" + k].astype(np.float64) for k in ("i", "ii", "iii", "iv", "ivb")}
    if copy_as_rollout:
        P["i"] = P["ii"].copy()
    if perm_rng is not None:                                   # p3: the labels permuted across the windows that carry one, inside this seed
        m = np.isfinite(yk)
        yk = yk.copy()
        yk[m] = yk[m][perm_rng.permutation(int(m.sum()))]
    r2 = {k: RP.r2_pooled(yk, P[k]) for k in P}
    fl = "ivb" if r2["ivb"] > r2["iv"] else "iv"               # the floor rule of rb_trained.analyse
    P["fl"] = P[fl]; r2["fl"] = r2[fl]
    P.update(calibrate(yk, epi, P))
    for k in CAL:
        r2[k] = RP.r2_pooled(yk, P[k])
    St = {k: RP.ep_stats_r2(epi, yk, P[k], n_ep) for k in ("i", "ii", "fl") + tuple(CAL)}
    A = RC.load_arm(path, gap0_of)
    if copy_as_rollout:
        A["roll"] = np.zeros_like(A["roll"])
    ok = A["ok"]
    d, roll, e2 = A["d"][ok], A["roll"][ok], A["epi"][ok]
    va = np.bincount(e2, weights=(d - roll) ** 2, minlength=n_ep)
    vb = np.bincount(e2, weights=d ** 2, minlength=n_ep)
    mk = np.isfinite(yk)
    rec = {"name": name, "path": path, "n_ep": n_ep, "uep": [str(e) for e in z["uep"]], "key": (z["w_eid"].astype(str), z["w_g0"].astype(np.int64)),
           "mask": mk, "n_lead": int(mk.sum()), "n_change": int(ok.sum()), "floor": fl, "r2": r2, "S": St, "va": va, "vb": vb,
           "std": {"label": float(yk[mk].std()), "rollout": float(P["i"][mk].std()), "copy": float(P["ii"][mk].std()), "true_future": float(P["iii"][mk].std())},
           "V": float(1.0 - va.sum() / vb.sum())}
    for q, (x, y) in PAIRS.items():
        rec[q] = r2[x] - r2[y]
    return rec


def r2_of(t):                                                  # t [..., 4] = n, sum y, sum y^2, SSE
    sst = t[..., 2] - t[..., 1] ** 2 / np.maximum(t[..., 0], 1)
    return 1.0 - t[..., 3] / sst


def pooled(seeds, B, rng_seed=0, base=None):
    """seeds: list of load_seed records (each used once). Point values and the two-level bootstrap draws of every statistic (P: the pair against base)."""
    K, n_ep = len(seeds), seeds[0]["n_ep"]
    rng = np.random.default_rng(rng_seed)
    pick = rng.integers(0, K, size=(B, K))                     # level 1: seeds with replacement
    C = rng.multinomial(n_ep, np.full(n_ep, 1.0 / n_ep), size=(B, K)).astype(np.float64)   # level 2: episodes with replacement inside each drawn seed
    St = {k: np.stack([s["S"][k] for s in seeds]) for k in seeds[0]["S"]}
    va, vb = np.stack([s["va"] for s in seeds]), np.stack([s["vb"] for s in seeds])
    Sb = {x: np.stack([b["S"][x] for b in base]) for x in base[0]["S"]} if base is not None else None
    out = {q: np.zeros(B) for q in list(PAIRS) + ["V"] + (["P", "D3", "D3e"] if base is not None else [])}
    for j in range(K):
        k, c = pick[:, j], C[:, j, :]
        r = {x: r2_of(np.einsum("be,bef->bf", c, St[x][k])) for x in St}
        for q, (x, y) in PAIRS.items():
            out[q] += r[x] - r[y]
        num, den = (c * va[k]).sum(1), (c * vb[k]).sum(1)
        out["V"] += 1.0 - num / np.where(den > 0, den, np.nan)
        if base is not None:                                   # the same draw of seed slot and episodes for the base arm of that seed
            rb = {x: r2_of(np.einsum("be,bef->bf", c, Sb[x][k])) for x in ("i", "c_i2", "c_ii", "c_iifi", "c_iif")}
            out["P"] += r["i"] - rb["i"]
            out["D3"] += (r["c_i2"] - r["c_ii"]) - (rb["c_i2"] - rb["c_ii"])
            out["D3e"] += (r["c_iifi"] - r["c_iif"]) - (rb["c_iifi"] - rb["c_iif"])
    for q in out:
        out[q] = out[q] / K
    pt = {q: float(np.mean([s[q] for s in seeds])) for q in list(PAIRS) + ["V"]}
    if base is not None:
        pt["P"] = float(np.mean([s["r2"]["i"] - b["r2"]["i"] for s, b in zip(seeds, base)]))
        pt["D3"] = float(np.mean([s["A3"] - b["A3"] for s, b in zip(seeds, base)]))
        pt["D3e"] = float(np.mean([s["A3e"] - b["A3e"] for s, b in zip(seeds, base)]))
    return pt, out


def ci(x):
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--new-seeds", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    new = [int(s) for s in a.new_seeds.split(",")] if a.new_seeds else []
    os.makedirs(a.out_dir, exist_ok=True)
    Z = np.load(RC.S.BANK)
    gap0_of = {(e, int(g)): float(v) for e, g, v in zip(Z["beid"].astype(str), Z["bg"].astype(np.int64), Z["lead_gap"].astype(np.float64))}
    L = [load_seed("%s_s%d" % (a.arm, k), gap0_of) for k in seeds]
    QS = list(PAIRS) + ["V"]
    res = {"tier": TIER, "families": FAMILIES, "arm": a.arm, "seeds": seeds, "new_seeds": new, "B": a.n_boot,
           "estimator": "two-level bootstrap: seeds with replacement, then episodes with replacement inside each drawn seed; rng seed 0; percentiles 2.5 and 97.5",
           "per_seed": {}, "controls": {}, "scopes": {}}
    for k, s in zip(seeds, L):
        res["per_seed"][str(k)] = {"file": s["path"], "n_lead_windows": s["n_lead"], "n_change_windows": s["n_change"], "n_episodes": s["n_ep"], "floor_used": s["floor"],
                                   "r2": {x: s["r2"][x] for x in ("i", "ii", "iii", "fl") + tuple(CAL)}, "std_over_lead_windows": s["std"], **{q: s[q] for q in QS}}
    # p4: each seed once, the same windows in every seed
    same = all(np.array_equal(L[0]["key"][0], s["key"][0]) and np.array_equal(L[0]["key"][1], s["key"][1]) and np.array_equal(L[0]["mask"], s["mask"]) and L[0]["uep"] == s["uep"] for s in L)
    res["controls"]["p4_each_seed_once_same_windows"] = {"pass": bool(same and len(set(seeds)) == len(seeds)), "seeds": seeds, "n_lead_windows": [s["n_lead"] for s in L],
                                                         "n_episodes": [s["n_ep"] for s in L], "same_windows_and_labels_mask": bool(same)}
    # p1: one seed alone gives the point value of rb_trained.py for that seed
    p1 = {}
    for k, s in zip(seeds, L):
        rp = os.path.join(RES_DIR, "RESULT_RB_%s_s%d.json" % (a.arm, k))
        if not os.path.exists(rp):
            p1[str(k)] = {"pass": False, "reason": "no RESULT_RB file"}
            continue
        R = json.load(io.open(rp, encoding="utf-8"))
        ref_a, ref_f = find_key(R, "i_minus_ii"), find_key(R, "i_minus_floor")
        pt1, _ = pooled([s], 200)
        da, df = abs(pt1["A"] - ref_a["diff"]), abs(pt1["A2"] - ref_f["diff"])
        p1[str(k)] = {"pass": bool(da <= 1e-9 and df <= 1e-9), "abs_diff_A": da, "abs_diff_A2": df, "rb_trained_A": ref_a["diff"], "rb_trained_A2": ref_f["diff"]}
    res["controls"]["p1_one_seed_equals_rb_trained"] = {"pass": bool(all(v["pass"] for v in p1.values())), "bar": 1e-9, "per_seed": p1}
    # p2: the copy given as the rollout
    Lc = [load_seed("%s_s%d" % (a.arm, k), gap0_of, copy_as_rollout=True) for k in seeds]
    ptc, bc = pooled(Lc, a.n_boot)
    res["controls"]["p2_copy_as_rollout_is_0"] = {"pass": bool(ptc["A"] == 0.0 and float(np.abs(bc["A"]).max()) == 0.0 and ptc["V"] == 0.0), "A": ptc["A"],
                                                  "A_boot_abs_max": float(np.abs(bc["A"]).max()), "V": ptc["V"], "A3_abs": abs(ptc["A3"])}
    # p3: the labels permuted across windows inside each seed (the reads stay as they are)
    prng = np.random.default_rng(7)
    Lp = [load_seed("%s_s%d" % (a.arm, k), gap0_of, perm_rng=prng) for k in seeds]
    ptp, bp = pooled(Lp, a.n_boot)
    cp, cp3 = ci(bp["A"]), ci(bp["A3"])
    res["controls"]["p3_permuted_labels"] = {"pass": bool(not cp[0] > 0.0), "A": ptp["A"], "ci95": cp,
                                             "why": "with labels that carry nothing, A is the difference of the spreads of the two reads over the spread of the label"}
    res["controls"]["ALL_PASS"] = bool(all(v["pass"] for v in res["controls"].values()))
    res["beside_permuted_labels_calibrated"] = {"A3": ptp["A3"], "ci95": cp3, "not_above_0": bool(not cp3[0] > 0.0), "A3r": ptp["A3r"], "A3r_ci95": ci(bp["A3r"]), "A3e": ptp["A3e"], "A3e_ci95": ci(bp["A3e"])}
    # scopes
    base_all = None
    if a.base:
        try:
            base_all = [load_seed("%s_s%d" % (a.base, k), gap0_of) for k in seeds]
            res["base"] = {"arm": a.base, "seeds": seeds, "per_seed": {str(k): {q: b[q] for q in QS} for k, b in zip(seeds, base_all)}}
        except FileNotFoundError as e:
            res["base"] = {"arm": a.base, "absent": str(e)}

    def scope(name, idx):
        S_ = [L[i] for i in idx]
        Bs = [base_all[i] for i in idx] if base_all is not None else None
        pt, bs = pooled(S_, a.n_boot, base=Bs)
        r = {"seeds": [seeds[i] for i in idx]}
        for q in pt:
            c = ci(bs[q])
            r[q] = {"value": pt[q], "ci95": c, "above_0": bool(c[0] > 0.0),
                    "per_seed": [(s["r2"]["i"] - b["r2"]["i"] if q == "P" else s["A3"] - b["A3"] if q == "D3" else s["A3e"] - b["A3e"] if q == "D3e" else s[q]) for s, b in zip(S_, Bs or S_)]}
        res["scopes"][name] = r
        return r
    allr = scope("all", list(range(len(seeds))))
    newr = scope("new", [seeds.index(k) for k in new]) if new and all(k in seeds for k in new) else None
    if base_all is not None:                                   # the base arm read the same way, beside
        ptb, bsb = pooled(base_all, a.n_boot)
        res["base"]["pooled"] = {q: {"value": ptb[q], "ci95": ci(bsb[q])} for q in ptb}
    # names of note N2 (section 4): only with four seeds, two of them new, and every control passing
    names = {"named": False}
    if len(seeds) == 4 and newr is not None and len(new) == 2 and res["controls"]["ALL_PASS"]:
        dA, dV = allr["A"]["per_seed"], allr["V"]["per_seed"]
        im = allr["A"]["above_0"] and allr["A2"]["above_0"] and sum(x > 0 for x in dA) >= 3 and all(x > 0 for x in newr["A"]["per_seed"])
        mv = allr["V"]["above_0"] and sum(x > 0 for x in dV) >= 3 and all(x > 0 for x in newr["V"]["per_seed"])
        names = {"named": True, "imagination": "MOTION-IMAGINES-4" if im else "MOTION-FLAT-4", "change": "MOTION-MOVES-4" if mv else "MOTION-STILL-4"}
    elif not res["controls"]["ALL_PASS"]:
        names["reason"] = "a control failed: nothing is named"
    else:
        names["reason"] = "the names of the note need four seeds with two new ones; this run reads seeds %s" % seeds
    res["names"] = names
    # note PREREG_V8F_PROOF_ADD18_N3: controls n1 to n4 and the names of the calibrated read
    c3, c3e = ci(bp["A3"]), ci(bp["A3e"])
    N = {"n1_p1_p2_p4": {"pass": bool(all(res["controls"][k]["pass"] for k in ("p1_one_seed_equals_rb_trained", "p2_copy_as_rollout_is_0", "p4_each_seed_once_same_windows")))},
         "n2_permuted_labels_calibrated": {"pass": bool(not c3[0] > 0.0 and not c3e[0] > 0.0), "A3": ptp["A3"], "A3_ci95": c3, "A3e": ptp["A3e"], "A3e_ci95": c3e},
         "n3_true_future_as_rollout_above_0": {"pass": bool(allr["A3c"]["above_0"]), "A3c": allr["A3c"]["value"], "ci95": allr["A3c"]["ci95"]},
         "n4_copy_as_rollout_is_0": {"pass": bool(abs(ptc["A3"]) <= 1e-9 and abs(ptc["A3e"]) <= 1e-9), "A3_abs": abs(ptc["A3"]), "A3e_abs": abs(ptc["A3e"]), "bar": 1e-9}}
    N["ALL_PASS"] = bool(all(v["pass"] for v in N.values()))
    res["controls_n3"] = N
    n3 = {"named": False}
    if len(seeds) == 4 and newr is not None and len(new) == 2 and N["ALL_PASS"]:
        ok = lambda q: bool(all(x > 0 for x in newr[q]["per_seed"]) and newr[q]["above_0"] and allr[q]["above_0"])   # noqa: E731
        adds, lead = ok("A3"), ok("A3e")
        n3 = {"named": True, "adds": "MOTION-ADDS-4" if adds else "MOTION-ADDS-NOTHING-4", "lead": ("MOTION-ADDS-LEAD-4" if lead else "MOTION-ADDS-EGO-ONLY-4") if adds else None}
    elif not N["ALL_PASS"]:
        n3["reason"] = "a control of note N3 failed: nothing is named"
    else:
        n3["reason"] = "the names of note N3 need four seeds with two new ones; this run reads seeds %s" % seeds
    res["names_n3"] = n3
    res["headroom_share"] = {nm: {"A3_over_A3c": r["A3"]["value"] / r["A3c"]["value"] if r["A3c"]["value"] else None, "A3e_over_A3ec": r["A3e"]["value"] / r["A3ec"]["value"] if r["A3ec"]["value"] else None} for nm, r in res["scopes"].items()}
    io.open(os.path.join(a.out_dir, "RESULT_POOL_%s.json" % a.arm), "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    f = lambda r: "%+.3f [%+.3f, %+.3f]" % (r["value"], r["ci95"][0], r["ci95"][1])   # noqa: E731
    M = ["# Pooled lead read over seeds, arm %s (note PREREG_V8F_PROOF_ADD18_N2, section 4; calibrated forms beside)" % a.arm, "",
         "MEASURED, tier " + TIER + ". " + FAMILIES, "Estimator: " + res["estimator"] + ", B %d." % a.n_boot, "", "## Controls (first)", ""]
    for k, v in res["controls"].items():
        M.append("* %s: %s" % (k, ("PASS" if v else "FAIL") if isinstance(v, bool) else (("PASS " if v["pass"] else "FAIL ") + json.dumps({x: y for x, y in v.items() if x not in ("pass", "per_seed")})[:330])))
    M += ["* beside, calibrated forms under permuted labels: A3 %+.3f [%+.3f, %+.3f], A3r %+.3f" % (ptp["A3"], cp3[0], cp3[1], ptp["A3r"]), "",
          "## Per seed (each used once)", "", "| seed | lead windows | spread of the reads (label / rollout / copy / true future, m) | R2 rollout / copy / true future | " + " | ".join(LABEL[q] for q in QS) + " |",
          "|---|---|---|---|" + "---|" * len(QS)]
    for k, s in zip(seeds, L):
        M.append("| %d%s | %d | %.2f / %.2f / %.2f / %.2f | %.3f / %.3f / %.3f | " % (k, " (new)" if k in new else "", s["n_lead"], s["std"]["label"], s["std"]["rollout"], s["std"]["copy"], s["std"]["true_future"],
                                                                              s["r2"]["i"], s["r2"]["ii"], s["r2"]["iii"]) + " | ".join("%+.3f" % s[q] for q in QS) + " |")
    M += ["", "## Pooled", ""]
    for nm, r in res["scopes"].items():
        M.append("* scope %s, seeds %s: " % (nm, r["seeds"]) + "; ".join("%s %s" % (LABEL[q], f(r[q])) for q in r if q != "seeds"))
    if base_all is not None:
        M.append("* base arm %s read the same way, seeds %s: " % (a.base, seeds) + "; ".join("%s %s" % (LABEL[q], f(v)) for q, v in res["base"]["pooled"].items()))
    M += ["", "## Controls of note N3 (calibrated read)", ""] + ["* %s: %s" % (k, ("PASS" if v else "FAIL") if isinstance(v, bool) else (("PASS " if v["pass"] else "FAIL ") + json.dumps({x: y for x, y in v.items() if x != "pass"})[:300])) for k, v in N.items()]
    M += ["", "## Names", "", "* note N2: " + (("%s, %s" % (names["imagination"], names["change"])) if names["named"] else ("not named: " + names.get("reason", ""))),
          "* note N3: " + ((n3["adds"] + (", " + n3["lead"] if n3["lead"] else "")) if n3["named"] else ("not named: " + n3.get("reason", ""))),
          "* share of the headroom reached: " + json.dumps({k: {x: (round(y, 3) if y is not None else None) for x, y in v.items()} for k, v in res["headroom_share"].items()}), "",
          "Limits: one training run per seed; probes show what is linearly readable; T0 reads on logged frames; one corpus.", ""]
    io.open(os.path.join(a.out_dir, "VERDICT_POOL_%s.md" % a.arm), "w", encoding="utf-8", newline="\n").write("\n".join(M))
    print("controls", "PASS" if res["controls"]["ALL_PASS"] else "FAIL", "|", {k: (v["pass"] if isinstance(v, dict) else v) for k, v in res["controls"].items()})
    print("permuted labels: A %+.3f [%+.3f, %+.3f] | A3 %+.3f [%+.3f, %+.3f] | A3e %+.3f %s" % (ptp["A"], cp[0], cp[1], ptp["A3"], cp3[0], cp3[1], ptp["A3e"], ci(bp["A3e"])))
    for k, s in zip(seeds, L):
        print("seed %d: R2 i %.3f ii %.3f iii %.3f | " % (k, s["r2"]["i"], s["r2"]["ii"], s["r2"]["iii"]) + " ".join("%s %+.3f" % (q, s[q]) for q in QS))
    for nm, r in res["scopes"].items():
        print("scope %-4s seeds %s | " % (nm, r["seeds"]) + " | ".join("%s %s" % (q, f(r[q])) for q in r if q != "seeds"))
    if base_all is not None:
        print("base %s pooled | " % a.base + " | ".join("%s %s" % (q, f(v)) for q, v in res["base"]["pooled"].items()))
    print("names N2:", names)
    print("controls N3:", {k: (v["pass"] if isinstance(v, dict) else v) for k, v in N.items()}, "| names N3:", n3, "| headroom share:", res["headroom_share"])
    return 0 if res["controls"]["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
