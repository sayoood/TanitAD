"""PREREG_V8F_IMAG1 analysis: compress a draw, thresholds from the R = 199 reference draws, the two-part rule, labels, controls 1 to 3, the committed triggers. Tier T0.
analyse(ctx, real, refs, cals) -> a json-able dict (RESULT_IMAG1.json). Dataset-agnostic (the synthetic test calls it)."""
import numpy as np
import imag1_lib as IL
from imag1_lib import (LEVELS, KS, TARGETS, KIND, FAMILY, PROBES, QTS, FAST, SLOW, DT, STRIDE, Mom, sep_above, sep_below, paired_test, gain_tests)

TAGS = IL.TAGS
FAMS = ("LON", "LAT", "TAC", "SCENE")
LAT_Q = {"lon": "accel", "lat": "yaw_rate"}


def key(*a):
    return "|".join(str(x) for x in a)


def tsi_component(mom, q):
    return 1.0 - (mom.ba_point() if KIND[q] != "cont" else mom.point())


# ------------------------------------------------------------------------------------------------ compress one draw
def compress(c, res):
    """What the reference needs (points) and what a calibration draw needs (rows scored as if real): flags are computed against the cell thresholds later, so rows are kept."""
    cal = res["role"] == "cal"
    out = {"s": res["s"], "role": res["role"], "conv": res["conv"], "ia1": {}, "tsi": {}, "or": {}, "ia0": {}, "pc": {}, "g": {}, "ia1_row": {}, "or_row": {}, "paired": {}, "gain": {}, "ia0_row": {}}
    for tag, t in res["tags"].items():
        for P in PROBES:
            for (l, q), m in t["ia1"][P].items():
                out["ia1"][key(tag, P, l, q)] = m.point()
                out["tsi"][key(tag, P, l, q)] = tsi_component(m, q)
                if cal:
                    out["ia1_row"][key(tag, P, l, q)] = m.row(c.W)
            for (l, k, q, qt), m in t["ia2"][P].items():
                if qt == "oracle":
                    out["or"][key(tag, P, l, k, q)] = m.point()
                    if cal:
                        out["or_row"][key(tag, P, l, k, q)] = m.row(c.W)
                if qt == "pred":
                    md = m.diff(t["ia2"][P][(l, k, q, "copy")])
                    out["pc"][key(tag, P, l, k, q)] = md.point()
                    if cal:
                        pt = paired_test(md, c.W, c.S)
                        out["paired"][key(tag, P, l, k, q)] = [pt["above"], pt["below"], pt["point"], pt["lo"], pt["hi"]]
            for l in LEVELS:
                for k in KS:
                    mm = IL.ia3_moms(t["ia3"][P], l, k)
                    for ch in ("lon", "lat"):
                        out["g"][key(tag, P, l, k, ch)] = mm["gain_" + ch].point()
                        if cal:
                            g = gain_tests(mm["gain_" + ch], c.W, c.S, g0s=(0.0,))
                            out["gain"][key(tag, P, l, k, ch)] = [g["above_0.0"], g["below_0.0"], g["row"]["point"], g["row"]["lo"], g["row"]["hi"]]
        for l in LEVELS:
            m = t["ia0"][l]["mom"]
            out["ia0"][key(tag, l)] = m.point()
            if cal:
                out["ia0_row"][key(tag, l)] = m.row(c.W)
    return out


# ------------------------------------------------------------------------------------------------ helpers
def thr97(refs, kind, k):
    a = np.array([d[kind][k] for d in refs], np.float64)
    a = a[np.isfinite(a)]
    return (float(np.nanpercentile(a, 97.5)) if len(a) else np.nan), a


def rank_p(a, p):
    return float((1 + (a >= p).sum()) / (len(a) + 1))


def nmse_row(row, q):
    if KIND[q] != "cont":
        return None
    return {"point": 1 - row["point"], "lo": 1 - row["hi"], "hi": 1 - row["lo"]}


def agree(a, b):
    return a if a == b else f"PROBE-DEPENDENT(ridge:{a}|mlp:{b})"


def both(a, b):
    return a if a == b else f"SEEDS-DISAGREE({a}|{b})"


def slope_row(c, x, y):
    """OLS slope of y on x (with intercept) pooled over windows, the episode-cluster bootstrap of the moment ratios."""
    ok = np.isfinite(x) & np.isfinite(y)
    f = lambda v: np.bincount(c.qpos[ok], weights=v[ok], minlength=c.n_ep)
    n, sx, sy, sxx, sxy = f(np.ones(len(x))), f(x), f(y), f(x * x), f(x * y)
    sl = lambda W: (W @ n * (W @ sxy) - (W @ sx) * (W @ sy)) / ((W @ n) * (W @ sxx) - (W @ sx) ** 2)
    pt = float(sl(np.ones((1, c.n_ep)))[0])
    v = sl(c.W)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return {"point": pt, "lo": float(lo), "hi": float(hi), "n": int(ok.sum()), "nep": int(len(np.unique(c.qpos[ok])))}


def kin_null(c, tag_unused=None):
    """Beside, no bar: speed from constant acceleration over the last 0.4 s of logged speed; yaw rate held. NMSE with the scoring-fold variance, windows with a finite prior speed."""
    out = {}
    if not hasattr(c, "v_prev"):
        return out
    v0, vp = c.Yw["speed"][:, 0], c.v_prev
    a = (v0 - vp) / 0.4
    for l in LEVELS:
        for k in KS:
            f, h = k * STRIDE[l], k * DT[l]
            for q, pred in (("speed", v0 + a * h), ("yaw_rate", c.Yw["yaw_rate"][:, 0])):
                y = c.Yw[q][:, f]
                valid = np.isfinite(y) & np.isfinite(pred)
                m = IL._moms(c, q, y, np.where(valid, pred, 0.0), valid, c.qpos, c.qfold)
                r = m.row(c.W)
                out[key(l, k, q)] = {"nmse": nmse_row(r, q), "n": r["n"], "nep": r["nep"]}
    return out


def inverse_probe(c, tag, ridge_lam=10.0):
    """ATM inverse probe 2 x 2 (beside): ridge (lambda 10) from [z_t, z_{t+k}] to the mean logged raw action over the horizon, fitted on real pairs (encoded) or predicted pairs, scored on both."""
    out = {}
    for l in LEVELS:
        for k in KS:
            Xs = {"real": np.concatenate([c.Z0[tag][l], c.ORA[tag][(l, k)]], 1).astype(np.float64), "pred": np.concatenate([c.Z0[tag][l], c.PRED[tag][(l, k)]], 1).astype(np.float64)}
            Y = c.actmean[(l, k)]
            for fitk in ("real", "pred"):
                for scorek in ("real", "pred"):
                    pred = np.zeros_like(Y)
                    for j in (0, 1):
                        fit = c.qfold != j
                        mu, sd = Xs[fitk][fit].mean(0), Xs[fitk][fit].std(0) + 1e-8
                        Xf = (Xs[fitk][fit] - mu) / sd
                        beta = np.linalg.solve(Xf.T @ Xf + ridge_lam * np.eye(Xf.shape[1]), Xf.T @ (Y[fit] - Y[fit].mean(0)))
                        pred[c.qfold == j] = ((Xs[scorek][c.qfold == j] - mu) / sd) @ beta + Y[fit].mean(0)
                    cell = {}
                    for ci, nm in enumerate(("accel", "yaw_rate")):
                        m = IL._moms(c, "speed", Y[:, ci], pred[:, ci], np.ones(len(Y), bool), c.qpos, c.qfold)
                        r = m.row(c.W)
                        cell[nm] = {"r2": r["point"], "lo": r["lo"], "hi": r["hi"]}
                    out[key(l, k, fitk, scorek)] = cell
    return out


# ------------------------------------------------------------------------------------------------ the analysis
def analyse(c, real, refs, cals):
    W, S = c.W, c.S
    R = {"dataset": c.name, "n_episodes": int(c.n_ep), "n_bank_frames": int(len(c.beid)), "n_windows": int(len(c.qe)), "R": len(refs), "n_cal": len(cals), "tags": list(c.tags)}
    T = real["tags"]
    # ---- IA-0
    ia0 = {}
    for tag in c.tags:
        for l in LEVELS:
            d = T[tag]["ia0"][l]
            row = d["mom"].row(W)
            thr, a = thr97(refs, "ia0", key(tag, l))
            ia0[key(tag, l)] = {"mrr": d["mrr"], "chance": d["chance"], "excess": row, "thr97": thr, "rank_p": rank_p(a, row["point"]), "separated_two_part": sep_above(row, thr), "rankme": T[tag]["rankme"][l]}
    healthy = {l: all(ia0[key(tag, l)]["separated_two_part"] for tag in c.tags) for l in LEVELS}
    R["ia0"], R["healthy"] = ia0, {str(l): healthy[l] for l in LEVELS}
    # ---- IA-1
    ia1, enc = {}, {}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for q in TARGETS:
                    m = T[tag]["ia1"][P][(l, q)]
                    row = m.row(W)
                    thr, a = thr97(refs, "ia1", key(tag, P, l, q))
                    e = {"score": row, "metric": "1-NMSE" if KIND[q] == "cont" else "BA_minus_majority", "thr97": thr, "rank_p": rank_p(a, row["point"]), "encoded": sep_above(row, thr)}
                    if KIND[q] == "cont":
                        e["nmse"] = nmse_row(row, q)
                    ia1[key(P, tag, l, q)] = e
        for l in LEVELS:
            for q in TARGETS:
                enc[key(P, l, q)] = all(ia1[key(P, tag, l, q)]["encoded"] for tag in c.tags)
    R["ia1"], R["encoded"] = ia1, enc
    if "rand" in real:
        R["ia1_rand"] = {key(rt, P, l, q): {"score": real["rand"][rt][P][(l, q)].row(W)} for rt in real["rand"] for P in PROBES for l in LEVELS for q in TARGETS}
    # ---- IA-2
    ia2, lab_p, lab, lab_perm = {}, {}, {}, {}
    ctl2 = {"cells": [], "offset_beats_oracle_rows": []}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    for q in TARGETS:
                        mo = {qt: T[tag]["ia2"][P][(l, k, q, qt)] for qt in QTS}
                        rows = {qt: mo[qt].row(W) for qt in QTS}
                        thr, a = thr97(refs, "or", key(tag, P, l, k, q))
                        pc = paired_test(mo["pred"].diff(mo["copy"]), W, S)
                        po = paired_test(mo["oracle"].diff(mo["offset"]), W, S)
                        po_b = paired_test(mo["offset"].diff(mo["oracle"]), W, S)
                        pch, _a = thr97(refs, "pc", key(tag, P, l, k, q))
                        pcl = float(np.nanpercentile(_a, 2.5)) if len(_a) else np.nan
                        pc["perm_above"], pc["perm_below"] = sep_above(pc, pch), sep_below(pc, pcl)
                        e = {"rows": rows, "oracle_thr97": thr, "oracle_rank_p": rank_p(a, rows["oracle"]["point"]), "oracle_readable": sep_above(rows["oracle"], thr),
                             "pred_minus_copy": pc, "oracle_minus_offset": po, "n": rows["pred"]["n"], "nep": rows["pred"]["nep"]}
                        if KIND[q] == "cont":
                            e["nmse"] = {qt: nmse_row(rows[qt], q) for qt in QTS}
                        ia2[key(P, tag, l, k, q)] = e
        for l in LEVELS:
            for k in KS:
                for q in TARGETS:
                    es = [ia2[key(P, tag, l, k, q)] for tag in c.tags]
                    if not all(e["oracle_readable"] for e in es):
                        L_ = "UNREADABLE"
                    elif all(e["pred_minus_copy"]["above"] for e in es):
                        L_ = "IMAGINES"
                    elif all(e["pred_minus_copy"]["below"] for e in es):
                        L_ = "WORSE-THAN-COPY"
                    else:
                        L_ = "COPIES"
                    lab_p[key(P, l, k, q)] = L_
                    if L_ != "UNREADABLE":
                        L2_ = "IMAGINES" if all(e["pred_minus_copy"]["perm_above"] for e in es) else ("WORSE-THAN-COPY" if all(e["pred_minus_copy"]["perm_below"] for e in es) else "COPIES")
                    else:
                        L2_ = L_
                    lab_perm[key(P, l, k, q)] = L2_
    for l in LEVELS:
        for k in KS:
            for q in TARGETS:
                a, b = lab_p[key("ridge", l, k, q)], lab_p[key("mlp", l, k, q)]
                lab[key(l, k, q)] = {"ridge": a, "mlp": b, "decision": agree(a, b), "read": bool(healthy[l]),
                                     "beside_perm_referenced": agree(lab_perm[key("ridge", l, k, q)], lab_perm[key("mlp", l, k, q)])}
    R["ia2"], R["ia2_labels"] = ia2, lab
    # ---- control 2 (real-vs-real), pooled per family
    c2_cells, c2_viol = [], []
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    for fam in FAMS:
                        qs = [q for q in TARGETS if FAMILY[q] == fam and ia2[key(P, tag, l, k, q)]["oracle_readable"]]
                        if not qs:
                            continue
                        ms = [T[tag]["ia2"][P][(l, k, q, "oracle")].diff(T[tag]["ia2"][P][(l, k, q, "offset")]) for q in qs]
                        pt = float(np.mean([m.point() for m in ms]))
                        v = np.mean([m.vec(W) for m in ms], 0)
                        fl = np.mean([m.flips(S) for m in ms], 0)
                        lo, hi = np.nanpercentile(v, [2.5, 97.5])
                        thr = float(np.nanpercentile(fl, 97.5))
                        c2_cells.append({"P": P, "tag": tag, "l": l, "k": k, "family": fam, "n_targets": len(qs), "point": pt, "lo": float(lo), "hi": float(hi), "thr97": thr, "separated_above_0": bool(lo > 0 and pt > thr), "separated_below_0": bool(hi < 0 and pt < float(np.nanpercentile(fl, 2.5)))})
    for P in PROBES:
        for l in LEVELS:
            for k in KS:
                for q in TARGETS:
                    es = [ia2[key(P, tag, l, k, q)] for tag in c.tags]
                    if all(e["oracle_readable"] for e in es) and all(paired_test(T[tag]["ia2"][P][(l, k, q, "offset")].diff(T[tag]["ia2"][P][(l, k, q, "oracle")]), W, S)["above"] for tag in c.tags):
                        c2_viol.append(key(P, l, k, q))
    cell_below = {}
    for x in c2_cells:
        cell_below.setdefault(key(x["P"], x["l"], x["k"], x["family"]), []).append(x["separated_below_0"])
    viol_cells = sorted(k_ for k_, v in cell_below.items() if len(v) == len(c.tags) and all(v))
    R["control2"] = {"cells": c2_cells, "n_cells": len(c2_cells), "n_not_separated": int(sum(not x["separated_above_0"] for x in c2_cells)),
                     "offset_beats_oracle_cells": viol_cells, "offset_beats_oracle_rows_beside": c2_viol,
                     "pass": bool(c2_cells and all(x["separated_above_0"] for x in c2_cells) and not viol_cells)}
    # ---- IA-3
    ia3, lab3, lab3p, lab3pp = {}, {}, {}, {}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    mm = IL.ia3_moms(T[tag]["ia3"][P], l, k)
                    h = k * DT[l]
                    Jp = {n: m.point() for n, m in mm["J"].items()}
                    cos_p = float(IL.cosine({n: np.array(v) for n, v in Jp.items()}, h))
                    Jv = IL.jac_boot(mm["J"], W)
                    cos_v = IL.cosine(Jv, h)
                    cos_gain = float(IL.cosine({n: np.array(v) / (h if n.startswith("s_") else 1.0) for n, v in Jp.items()}, 1.0))
                    means = {tq: {cn: T[tag]["ia3"][P][(l, k, "mean", cn, tq)] for cn in list(IL.CONTR) + ["logged"]} for tq in IL.ENC_ONLY}
                    e = {"means": means, "cos": {"point": cos_p, "lo": float(np.nanpercentile(cos_v, 2.5)), "hi": float(np.nanpercentile(cos_v, 97.5))}, "cos_gain_units_beside": cos_gain, "J": Jp, "h": h}
                    for ch in ("lon", "lat"):
                        g = gain_tests(mm["gain_" + ch], W, S)
                        gp = g["row"]["point"]
                        gh, _a = thr97(refs, "g", key(tag, P, l, k, ch))
                        gl = float(np.nanpercentile(_a, 2.5)) if len(_a) else np.nan
                        pa, pb = sep_above(g["row"], gh), sep_below(g["row"], gl)
                        if g["below_0.0"]:
                            L_ = "WRONG-SIGN"
                        elif not g["above_0.0"]:
                            L_ = "INERT"
                        elif g["above_1.5"]:
                            L_ = "OVERSHOOTS"
                        elif gp < 0.5:
                            L_ = "WEAK"
                        elif gp <= 1.5 and cos_p > 0.7:
                            L_ = "ANSWERS"
                        else:
                            L_ = "UNRESOLVED"
                        Lp = "WRONG-SIGN" if pb else ("INERT" if not pa else ("OVERSHOOTS" if g["above_1.5"] else ("WEAK" if gp < 0.5 else ("ANSWERS" if gp <= 1.5 and cos_p > 0.7 else "UNRESOLVED"))))
                        e[ch] = {"gain": g["row"], "above_0": g["above_0.0"], "below_0": g["below_0.0"], "above_1.5": g["above_1.5"], "label": L_, "label_perm_referenced_beside": Lp, "perm_above_0": pa, "perm_below_0": pb}
                    ia3[key(P, tag, l, k)] = e
        for l in LEVELS:
            for k in KS:
                for ch in ("lon", "lat"):
                    lab3p[key(P, l, k, ch)] = both(*[ia3[key(P, tag, l, k)][ch]["label"] for tag in c.tags])
                    lab3pp[key(P, l, k, ch)] = both(*[ia3[key(P, tag, l, k)][ch]["label_perm_referenced_beside"] for tag in c.tags])
    for l in LEVELS:
        for k in KS:
            for ch in ("lon", "lat"):
                a, b = lab3p[key("ridge", l, k, ch)], lab3p[key("mlp", l, k, ch)]
                lab3[key(l, k, ch)] = {"ridge": a, "mlp": b, "decision": agree(a, b), "read": bool(healthy[l]),
                                       "beside_perm_referenced": agree(lab3pp[key("ridge", l, k, ch)], lab3pp[key("mlp", l, k, ch)])}
    R["ia3"], R["ia3_labels"] = ia3, lab3
    # ---- IA-6
    tsi, split = {}, {}
    for P in PROBES:
        for tag in c.tags:
            vec = {}
            for l in LEVELS:
                comp = {}
                for q in FAST + SLOW:
                    m = T[tag]["ia1"][P][(l, q)]
                    comp[q] = 1.0 - (m.ba_vec(W) if KIND[q] != "cont" else m.vec(W))
                vec[l] = np.mean([comp[q] for q in FAST], 0) - np.mean([comp[q] for q in SLOW], 0)
            d = vec[3] - vec[1]
            pt = {l: float(np.mean([tsi_component(T[tag]["ia1"][P][(l, q)], q) for q in FAST]) - np.mean([tsi_component(T[tag]["ia1"][P][(l, q)], q) for q in SLOW])) for l in LEVELS}
            dp = pt[3] - pt[1]
            refd = []
            for dr in refs:
                f = lambda l: np.mean([dr["tsi"][key(tag, P, l, q)] for q in FAST]) - np.mean([dr["tsi"][key(tag, P, l, q)] for q in SLOW])
                refd.append(f(3) - f(1))
            refd = np.array(refd)
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            thr = float(np.nanpercentile(refd, 97.5))
            tsi[key(P, tag)] = {"tsi": pt, "diff_L3_minus_L1": {"point": dp, "lo": float(lo), "hi": float(hi)}, "thr97": thr, "rank_p": rank_p(refd, dp), "separated_two_part": bool(lo > 0 and dp > thr)}
        split[P] = all(tsi[key(P, tag)]["separated_two_part"] for tag in c.tags)
    cons = {}
    for P in PROBES:
        for tag in c.tags:
            dsp = T[tag]["dsp"][P]
            for nm, (xl, yl) in (("L2k1_vs_L1k2_0.4s", ((1, 2), (2, 1))), ("L3k1_vs_L2k2_0.8s", ((2, 2), (3, 1)))):
                r = slope_row(c, dsp[xl], dsp[yl])
                r["consistent"] = bool(r["lo"] <= 1.0 <= r["hi"] and r["lo"] > 0)
                cons[key(P, tag, nm)] = r
    consist = {key(P, nm): all(cons[key(P, tag, nm)]["consistent"] for tag in c.tags) for P in PROBES for nm in ("L2k1_vs_L1k2_0.4s", "L3k1_vs_L2k2_0.8s")}
    R["ia6"] = {"tsi": tsi, "split_by_probe": split, "split_decision": ("SPLIT" if all(split.values()) else ("NO-SPLIT" if not any(split.values()) else "PROBE-DEPENDENT")),
                "consistency": cons, "consistent_by_probe": consist}
    # ---- control 3 (calibration of the two-part rule)
    fam = {f: {"rows": 0, "separated": 0, "bootstrap_only": 0, "by_kind": {}} for f in FAMS + ("HEALTH",)}
    pfam = {f: {"rows": 0, "separated": 0} for f in FAMS}

    def count(f, flag, kind, boot_only=None):
        v = fam[f]
        v["rows"] += 1
        v["separated"] += int(flag)
        if boot_only is not None:
            v["bootstrap_only"] += int(boot_only)
        bk = v["by_kind"].setdefault(kind, [0, 0])
        bk[0] += 1
        bk[1] += int(flag)

    def pcount(f, flag):
        pfam[f]["rows"] += 1
        pfam[f]["separated"] += int(flag)

    def lo_thr(a_):
        return float(np.nanpercentile(a_, 2.5)) if len(a_) else np.nan

    for d in cals:
        for k_, rw in d["ia1_row"].items():
            tag, P, l, q = k_.split("|")
            thr, _ = thr97(refs, "ia1", k_)
            count(FAMILY[q], sep_above(rw, thr), "IA-1", rw["lo"] > 0)
            pcount(FAMILY[q], sep_above(rw, thr))
        for k_, rw in d["or_row"].items():
            tag, P, l, k, q = k_.split("|")
            thr, _ = thr97(refs, "or", k_)
            count(FAMILY[q], sep_above(rw, thr), "IA-2 Oracle", rw["lo"] > 0)
            pcount(FAMILY[q], sep_above(rw, thr))
        for k_, (ab, be, pt_, lo_, hi_) in d["paired"].items():
            q = k_.split("|")[-1]
            count(FAMILY[q], ab, "pred-vs-copy above (sign flips)")
            count(FAMILY[q], be, "pred-vs-copy below (sign flips)")
            rw = {"point": pt_, "lo": lo_, "hi": hi_}
            h_, a_ = thr97(refs, "pc", k_)
            pcount(FAMILY[q], sep_above(rw, h_))
            pcount(FAMILY[q], sep_below(rw, lo_thr(a_)))
        for k_, (ab, be, pt_, lo_, hi_) in d["gain"].items():
            ch = k_.split("|")[-1]
            fm = "LON" if ch == "lon" else "LAT"
            count(fm, ab, "gain above 0 (sign flips)")
            count(fm, be, "gain below 0 (sign flips)")
            rw = {"point": pt_, "lo": lo_, "hi": hi_}
            h_, a_ = thr97(refs, "g", k_)
            pcount(fm, sep_above(rw, h_))
            pcount(fm, sep_below(rw, lo_thr(a_)))
        for k_, rw in d["ia0_row"].items():
            thr, _ = thr97(refs, "ia0", k_)
            count("HEALTH", sep_above(rw, thr), "MRR", rw["lo"] > 0)
    for f, v in fam.items():
        v["share"] = v["separated"] / max(v["rows"], 1)
        v["share_bootstrap_only"] = v["bootstrap_only"] / max(v["rows"], 1) if v["bootstrap_only"] or f in ("HEALTH",) else None
        v["pass"] = bool(v["rows"] > 0 and v["share"] <= 0.05)
    for f, v in pfam.items():
        v["share"] = v["separated"] / max(v["rows"], 1)
    R["control3"] = {"families": fam, "pass": bool(all(v["pass"] for v in fam.values())), "beside_perm_referenced_shares": pfam}
    # ---- control 1 (reported)
    ins = [(k_, v) for tag in c.tags for k_, v in T[tag]["insample"].items()]
    R["control1"] = {"cells": len(ins), "in_sample_not_below_out_of_sample": int(sum(not (v["in"] < v["out"]) for _, v in ins if np.isfinite(v["in"]) and np.isfinite(v["out"]))),
                     "reported_not_a_bar": True}
    # ---- beside, no bar
    R["kinematic_null"] = kin_null(c)
    R["inverse_probe"] = {tag: inverse_probe(c, tag) for tag in c.tags}
    bank = {}
    for tag in c.tags:
        for l in LEVELS:
            for k in KS:
                rk = c.bankrank[tag][(l, k)]
                A = np.bincount(c.qpos, weights=1.0 / rk - c.chance_bank, minlength=c.n_ep)[:, None]
                D = np.bincount(c.qpos, minlength=c.n_ep)[:, None].astype(float)
                m = Mom(A, D, None, len(rk), len(np.unique(c.qpos)))
                r = m.row(W)
                bank[key(tag, l, k)] = {"mrr": float((1.0 / rk).mean()), "chance": c.chance_bank, "excess": r}
    R["action_bank"] = bank
    ok = np.isfinite(c.win_mean_yaw)
    turning = np.abs(c.win_mean_yaw) > 0.1
    R["turning"] = {"windows": int(ok.sum()), "turning": int(turning.sum()), "straight": int((~turning).sum()), "share_turning": float(turning.mean()),
                    "interval": Mom(np.bincount(c.qpos, weights=turning.astype(float), minlength=c.n_ep)[:, None], np.bincount(c.qpos, minlength=c.n_ep)[:, None].astype(float), None, ok.sum(), c.n_ep).row(W)}
    R["conv"] = {"real": real["conv"], "refs_not_converged": int(sum(d["conv"]["not_converged"] for d in refs + cals)), "refs_fits": int(sum(d["conv"]["fits"] for d in refs + cals))}
    R["triggers"] = triggers(R, lab, lab3, healthy)
    return R


# ------------------------------------------------------------------------------------------------ the committed triggers
def triggers(R, lab, lab3, healthy):
    """The card's 'What each outcome triggers', evaluated on the DECISION labels (probes agree) of HEALTHY levels, with the single-probe variants beside."""
    t = {}

    def dec(l, k, q, who="decision"):
        return lab[key(l, k, q)][who]

    for who in ("decision", "ridge", "mlp"):
        # lead gap UNREADABLE at every level -> E3 CAPLADDER (every level, every k, read levels only; no healthy level -> not triggered)
        rl = [dec(l, k, "lead_gap", who) for l in LEVELS if healthy[l] for k in KS]
        t[f"E3_CAPLADDER[{who}]"] = bool(rl) and all(x == "UNREADABLE" for x in rl) and all(healthy.values())
        # lateral IA-3 WEAK / INERT / WRONG-SIGN at every level (every k of a level, both seeds)
        lt = [lab3[key(l, k, "lat")][who] for l in LEVELS for k in KS]
        t[f"E7_DIVRS[{who}]"] = all(x in ("WEAK", "INERT", "WRONG-SIGN") for x in lt) and all(healthy.values())
        # longitudinal L1 OVERSHOOTS (some k, both seeds, a healthy L1)
        t[f"L1_RESIDUAL_ARM[{who}]"] = bool(healthy[1]) and any(lab3[key(1, k, "lon")][who] == "OVERSHOOTS" for k in KS)
        # L2 / L3 IMAGINES on readable LON rows while L1 COPIES or WORSE (same target and k)
        hits = []
        for q in ("speed", "accel", "lead_gap", "lead_closing", "ttc"):
            for k in KS:
                l1 = dec(1, k, q, who)
                if l1 in ("COPIES", "WORSE-THAN-COPY") and any(dec(l, k, q, who) == "IMAGINES" for l in (2, 3) if healthy[l]):
                    hits.append(key(q, k))
        t[f"MRATE_L1_SPACING[{who}]"] = bool(hits)
        t[f"MRATE_L1_SPACING_rows[{who}]"] = hits
    t["NO_SPLIT_MRATE_TSI_AND_LEVEL_FACTOR_SWEEP"] = R["ia6"]["split_decision"] == "NO-SPLIT"
    t["UNHEALTHY_LEVELS_COLLAPSE_DIAGNOSIS"] = [l for l in LEVELS if not healthy[l]]
    t["E9_REL_levels_targets"] = sorted({key(l, k, q) for l in LEVELS for k in KS for q in TARGETS if lab[key(l, k, q)]["decision"] == "IMAGINES"} |
                                        {key("lon" if ch == "lon" else "lat", l, k) for l in LEVELS for k in KS for ch in ("lon", "lat") if lab3[key(l, k, ch)]["decision"] == "ANSWERS"})
    return t
