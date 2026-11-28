"""PREREG_V8F_IMAG1B analysis (card prefix 4,634 bytes 40500d68...; IMPL_CHOICES_IMAG1B.md): compress a draw, the ONE reference for every row (the distribution of that same statistic over the
R = 199 refit-under-permutation draws), the two-part rule against it, labels, controls 1 to 3, the committed triggers. Tier T0.
analyse(ctx, real, refs, cals) -> a json-able dict (RESULT_IMAG1B.json). Dataset-agnostic (the synthetic test calls it). The paired sign-flip reference of IMAG1 is NOT used here."""
import numpy as np
import imag1b_lib as IL
from imag1b_lib import (LEVELS, KS, TARGETS, KIND, FAMILY, PROBES, QTS, OFFD, FAST, SLOW, DT, STRIDE, Mom, sep_above, sep_below)

TAGS = IL.TAGS
FAMS = ("LON", "LAT", "TAC", "SCENE")
LAT_Q = {"lon": "accel", "lat": "yaw_rate"}
PAIRS = (("L2k1_vs_L1k2_0.4s", (1, 2), (2, 1)), ("L3k1_vs_L2k2_0.8s", (2, 2), (3, 1)))     # (name, x = finer level (l, k), y = coarser level (l, k))
BAR_KINDS = ("IA-1", "IA-2 Oracle", "pred-vs-Copy", "gain", "MRR")                          # control 3: the kinds the card bars at 5 % (per family AND per kind)


def key(*a):
    return "|".join(str(x) for x in a)


def tsi_component(mom, q):
    return 1.0 - (mom.ba_point() if KIND[q] != "cont" else mom.point())


def tsi_diff(ia1P, W):
    """TSI(L3) - TSI(L1) of one (probe kind, seed): ia1P[(l, q)] = Mom. Returns (point, bootstrap vector, {l: TSI point})."""
    pt, vec = {}, {}
    for l in LEVELS:
        comp = {q: tsi_component(ia1P[(l, q)], q) for q in FAST + SLOW}
        v = {q: 1.0 - (ia1P[(l, q)].ba_vec(W) if KIND[q] != "cont" else ia1P[(l, q)].vec(W)) for q in FAST + SLOW}
        pt[l] = float(np.mean([comp[q] for q in FAST]) - np.mean([comp[q] for q in SLOW]))
        vec[l] = np.mean([v[q] for q in FAST], 0) - np.mean([v[q] for q in SLOW], 0)
    return pt[3] - pt[1], vec[3] - vec[1], pt


# ------------------------------------------------------------------------------------------------ compress one draw
def compress(c, res):
    """What the reference needs (the point of every statistic) and, for a calibration draw, the full rows (point, bootstrap interval) so the draw can be scored as if real against the reference."""
    cal = res["role"] == "cal"
    out = {"s": res["s"], "role": res["role"], "conv": res["conv"]}
    for kk in ("ia1", "tsi", "or", "on", "pc", "g", "cos", "slope", "ia0", "ia1_row", "or_row", "pc_row", "g_row", "cos_row", "tsi_row", "slope_row", "ia0_row"):
        out[kk] = {}
    for tag, t in res["tags"].items():
        for P in PROBES:
            for (l, q), m in t["ia1"][P].items():
                out["ia1"][key(tag, P, l, q)] = m.point()
                out["tsi"][key(tag, P, l, q)] = tsi_component(m, q)
                if cal:
                    out["ia1_row"][key(tag, P, l, q)] = m.row(c.W)
            if cal:
                dp, dv, _ = tsi_diff(t["ia1"][P], c.W)
                lo, hi = np.nanpercentile(dv, [2.5, 97.5])
                out["tsi_row"][key(P, tag)] = {"point": float(dp), "lo": float(lo), "hi": float(hi)}
            for (l, k, q, qt), m in t["ia2"][P].items():
                kk = key(tag, P, l, k, q)
                if qt == "oracle":
                    out["or"][kk] = m.point()
                    out["on"][kk] = m.diff(t["ia2"][P][(l, k, q, "nf")]).point()          # oracle minus not-future (control 2), the refitted probe on both
                    if cal:
                        out["or_row"][kk] = m.row(c.W)
                if qt == "pred":
                    md = m.diff(t["ia2"][P][(l, k, q, "copy")])                             # predicted minus Copy-Last, the refitted probe on both terms
                    out["pc"][kk] = md.point()
                    if cal:
                        out["pc_row"][kk] = md.row(c.W)
            for l in LEVELS:
                for k in KS:
                    mm = IL.ia3_moms(t["ia3"][P], l, k)
                    h = k * DT[l]
                    for ch in ("lon", "lat"):
                        out["g"][key(tag, P, l, k, ch)] = mm["gain_" + ch].point()
                        if cal:
                            out["g_row"][key(tag, P, l, k, ch)] = mm["gain_" + ch].row(c.W)
                    Jp = {n: m_.point() for n, m_ in mm["J"].items()}
                    cp = float(IL.cosine({n: np.array(v) for n, v in Jp.items()}, h))
                    out["cos"][key(tag, P, l, k)] = cp
                    if cal:
                        cv = IL.cosine(IL.jac_boot(mm["J"], c.W), h)
                        out["cos_row"][key(tag, P, l, k)] = {"point": cp, "lo": float(np.nanpercentile(cv, 2.5)), "hi": float(np.nanpercentile(cv, 97.5))}
            for nm, xl, yl in PAIRS:
                out["slope"][key(tag, P, nm)] = IL.slope_point(c, t["dsp"][P][xl], t["dsp"][P][yl])
                if cal:
                    out["slope_row"][key(tag, P, nm)] = slope_row(c, t["dsp"][P][xl], t["dsp"][P][yl])
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
def pooled(ms, W):
    """Equal-weight mean over targets of a list of paired-difference Moms: point and bootstrap interval of the pooled mean (+ the smallest n)."""
    pt = float(np.mean([m.point() for m in ms]))
    v = np.nanmean([m.vec(W) for m in ms], 0)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return {"point": pt, "lo": float(lo), "hi": float(hi), "n_min": int(min(m.n for m in ms))}


def analyse(c, real, refs, cals):
    W = c.W
    R = {"dataset": c.name, "n_episodes": int(c.n_ep), "n_bank_frames": int(len(c.beid)), "n_windows": int(len(c.qe)), "R": len(refs), "n_cal": len(cals), "tags": list(c.tags),
         "reference": "refit under permutation (labels deranged across episodes inside the fit fold, probe refitted, whole statistic recomputed); the paired sign-flip reference is not used"}
    T = real["tags"]
    # ---- IA-0 (unchanged: the derangement reference involves no probe)
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
    ia2, lab_p, lab = {}, {}, {}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    for q in TARGETS:
                        mo = {qt: T[tag]["ia2"][P][(l, k, q, qt)] for qt in QTS}
                        rows = {qt: mo[qt].row(W) for qt in QTS}
                        kk = key(tag, P, l, k, q)
                        thr, a = thr97(refs, "or", kk)
                        pc = mo["pred"].diff(mo["copy"]).row(W)
                        pch, _a = thr97(refs, "pc", kk)
                        pcl = lo_thr(_a)
                        pc.update({"thr_hi": pch, "thr_lo": pcl, "above": sep_above(pc, pch), "below": sep_below(pc, pcl)})
                        on = mo["oracle"].diff(mo["nf"]).row(W)
                        onh, _ = thr97(refs, "on", kk)
                        on.update({"thr_hi": onh, "above": sep_above(on, onh)})
                        e = {"rows": rows, "oracle_thr97": thr, "oracle_rank_p": rank_p(a, rows["oracle"]["point"]), "oracle_readable": sep_above(rows["oracle"], thr),
                             "pred_minus_copy": pc, "oracle_minus_notfuture": on, "n": rows["pred"]["n"], "nep": rows["pred"]["nep"]}
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
    for l in LEVELS:
        for k in KS:
            for q in TARGETS:
                a, b = lab_p[key("ridge", l, k, q)], lab_p[key("mlp", l, k, q)]
                lab[key(l, k, q)] = {"ridge": a, "mlp": b, "decision": agree(a, b), "read": bool(healthy[l])}
    R["ia2"], R["ia2_labels"] = ia2, lab
    # ---- control 2 (the not-future), pooled per family over the targets whose Oracle is readable (label not UNREADABLE) for that probe kind
    c2_cells, decorr = [], {}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    for fam in FAMS:
                        qs = [q for q in TARGETS if FAMILY[q] == fam and lab_p[key(P, l, k, q)] != "UNREADABLE"]
                        if not qs:
                            continue
                        ms = [T[tag]["ia2"][P][(l, k, q, "oracle")].diff(T[tag]["ia2"][P][(l, k, q, "nf")]) for q in qs]
                        pr = pooled(ms, W)
                        refv = np.array([np.nanmean([d["on"][key(tag, P, l, k, q)] for q in qs]) for d in refs])
                        thr = float(np.nanpercentile(refv[np.isfinite(refv)], 97.5)) if np.isfinite(refv).any() else float("nan")
                        c2_cells.append({"P": P, "tag": tag, "l": l, "k": k, "family": fam, "targets": qs, "n_targets": len(qs), "point": pr["point"], "lo": pr["lo"], "hi": pr["hi"], "thr97": thr,
                                         "separated": bool(np.isfinite(pr["point"]) and pr["lo"] > 0 and pr["point"] > thr)})
                        dc = {"targets": qs, "nf": pr}
                        for d in OFFD:
                            dc[f"off{d}"] = pooled([T[tag]["ia2"][P][(l, k, q, f"orasub{d}")].diff(T[tag]["ia2"][P][(l, k, q, f"off{d}")]) for q in qs], W)
                        decorr[key(P, tag, l, k, fam)] = dc
    R["control2"] = {"cells": c2_cells, "n_cells": len(c2_cells), "n_not_separated": int(sum(not x["separated"] for x in c2_cells)), "pass": bool(c2_cells and all(x["separated"] for x in c2_cells)),
                     "definition": "Oracle two-part separated BELOW the not-future (a window of a different episode, default_rng(0)) in every pooled family cell where the Oracle is readable, at both seeds"}
    R["decorrelation"] = {"offsets_s": {str(d): d * 0.2 for d in OFFD}, "cells": decorr, "note": "beside, no bar: pooled gap score(Oracle) minus score(latent at the offset), same targets as control 2"}
    # ---- IA-3
    ia3, lab3, lab3p = {}, {}, {}
    for P in PROBES:
        for tag in c.tags:
            for l in LEVELS:
                for k in KS:
                    mm = IL.ia3_moms(T[tag]["ia3"][P], l, k)
                    h = k * DT[l]
                    Jp = {n: m.point() for n, m in mm["J"].items()}
                    cos_p = float(IL.cosine({n: np.array(v) for n, v in Jp.items()}, h))
                    cos_v = IL.cosine(IL.jac_boot(mm["J"], W), h)
                    cos_gain = float(IL.cosine({n: np.array(v) / (h if n.startswith("s_") else 1.0) for n, v in Jp.items()}, 1.0))
                    cr = {"point": cos_p, "lo": float(np.nanpercentile(cos_v, 2.5)), "hi": float(np.nanpercentile(cos_v, 97.5))}
                    cth, _ = thr97(refs, "cos", key(tag, P, l, k))
                    cos_sep = bool(np.isfinite(cos_p) and cr["lo"] > 0 and cos_p > cth)
                    cos_ok = bool(cos_p > 0.7 and cos_sep)
                    means = {tq: {cn: T[tag]["ia3"][P][(l, k, "mean", cn, tq)] for cn in list(IL.CONTR) + ["logged"]} for tq in IL.ENC_ONLY}
                    e = {"means": means, "cos": cr, "cos_thr97": cth, "cos_separated_above_0": cos_sep, "cos_gt_0.7_and_separated": cos_ok, "cos_gain_units_beside": cos_gain, "J": Jp, "h": h}
                    for ch in ("lon", "lat"):
                        row = mm["gain_" + ch].row(W)
                        gp = row["point"]
                        gh, _a = thr97(refs, "g", key(tag, P, l, k, ch))
                        gl = lo_thr(_a)
                        fin = bool(np.isfinite(gp))
                        above0, below0 = bool(fin and row["lo"] > 0 and gp > gh), bool(fin and row["hi"] < 0 and gp < gl)
                        above15 = bool(fin and row["lo"] > 1.5 and gp > gh)
                        if below0:
                            L_ = "WRONG-SIGN"
                        elif not above0:
                            L_ = "INERT"
                        elif above15:
                            L_ = "OVERSHOOTS"
                        elif gp < 0.5:
                            L_ = "WEAK"
                        elif gp <= 1.5 and cos_ok:
                            L_ = "ANSWERS"
                        else:
                            L_ = "UNRESOLVED"
                        e[ch] = {"gain": row, "thr97": gh, "thr2.5": gl, "above_0": above0, "below_0": below0, "above_1.5": above15, "label": L_}
                    ia3[key(P, tag, l, k)] = e
        for l in LEVELS:
            for k in KS:
                for ch in ("lon", "lat"):
                    lab3p[key(P, l, k, ch)] = both(*[ia3[key(P, tag, l, k)][ch]["label"] for tag in c.tags])
    for l in LEVELS:
        for k in KS:
            for ch in ("lon", "lat"):
                a, b = lab3p[key("ridge", l, k, ch)], lab3p[key("mlp", l, k, ch)]
                lab3[key(l, k, ch)] = {"ridge": a, "mlp": b, "decision": agree(a, b), "read": bool(healthy[l])}
    R["ia3"], R["ia3_labels"] = ia3, lab3
    # ---- IA-6
    tsi, split = {}, {}
    for P in PROBES:
        for tag in c.tags:
            dp, dv, pt = tsi_diff(T[tag]["ia1"][P], W)
            refd = np.array([np.mean([d["tsi"][key(tag, P, 3, q)] for q in FAST]) - np.mean([d["tsi"][key(tag, P, 3, q)] for q in SLOW]) -
                             (np.mean([d["tsi"][key(tag, P, 1, q)] for q in FAST]) - np.mean([d["tsi"][key(tag, P, 1, q)] for q in SLOW])) for d in refs])
            lo, hi = np.nanpercentile(dv, [2.5, 97.5])
            thr = float(np.nanpercentile(refd, 97.5))
            tsi[key(P, tag)] = {"tsi": pt, "diff_L3_minus_L1": {"point": float(dp), "lo": float(lo), "hi": float(hi)}, "thr97": thr, "rank_p": rank_p(refd, dp), "separated_two_part": bool(lo > 0 and dp > thr)}
        split[P] = all(tsi[key(P, tag)]["separated_two_part"] for tag in c.tags)
    cons = {}
    for P in PROBES:
        for tag in c.tags:
            dsp = T[tag]["dsp"][P]
            for nm, xl, yl in PAIRS:
                r = slope_row(c, dsp[xl], dsp[yl])
                sth, _ = thr97(refs, "slope", key(tag, P, nm))
                r["thr97"] = sth
                r["separated_above_0"] = bool(np.isfinite(r["point"]) and r["lo"] > 0 and r["point"] > sth)
                r["contains_1"] = bool(r["lo"] <= 1.0 <= r["hi"])
                r["consistent"] = bool(r["contains_1"] and r["separated_above_0"])
                cons[key(P, tag, nm)] = r
    consist = {key(P, nm): all(cons[key(P, tag, nm)]["consistent"] for tag in c.tags) for P in PROBES for nm, _x, _y in PAIRS}
    R["ia6"] = {"tsi": tsi, "split_by_probe": split, "split_decision": ("SPLIT" if all(split.values()) else ("NO-SPLIT" if not any(split.values()) else "PROBE-DEPENDENT")),
                "consistency": cons, "consistent_by_probe": consist}
    # ---- control 3: the 20 calibration draws scored as if real against the change-1 references; the bar is per family AND per row kind
    fam = {f: {"rows": 0, "separated": 0, "bootstrap_only": 0, "by_kind": {}, "split": {}} for f in FAMS + ("HEALTH",)}
    beside = {"cosine (above 0)": [0, 0], "TSI(L3)-TSI(L1) (above 0)": [0, 0], "consistency slope (above 0)": [0, 0]}

    def count(f, flag, kind, boot, sub=None):
        v = fam[f]
        v["rows"] += 1
        v["separated"] += int(flag)
        v["bootstrap_only"] += int(boot)
        bk = v["by_kind"].setdefault(kind, [0, 0])
        bk[0] += 1
        bk[1] += int(flag)
        if sub:
            sb = v["split"].setdefault(sub, [0, 0])
            sb[0] += 1
            sb[1] += int(flag)

    for d in cals:
        for k_, rw in d["ia1_row"].items():
            q = k_.split("|")[-1]
            thr, _ = thr97(refs, "ia1", k_)
            count(FAMILY[q], sep_above(rw, thr), "IA-1", rw["lo"] > 0)
        for k_, rw in d["or_row"].items():
            q = k_.split("|")[-1]
            thr, _ = thr97(refs, "or", k_)
            count(FAMILY[q], sep_above(rw, thr), "IA-2 Oracle", rw["lo"] > 0)
        for k_, rw in d["pc_row"].items():
            q = k_.split("|")[-1]
            h_, a_ = thr97(refs, "pc", k_)
            count(FAMILY[q], sep_above(rw, h_), "pred-vs-Copy", rw["lo"] > 0, "IMAGINES side (above)")
            count(FAMILY[q], sep_below(rw, lo_thr(a_)), "pred-vs-Copy", rw["hi"] < 0, "WORSE side (below)")
        for k_, rw in d["g_row"].items():
            fm = "LON" if k_.split("|")[-1] == "lon" else "LAT"
            h_, a_ = thr97(refs, "g", k_)
            count(fm, sep_above(rw, h_), "gain", rw["lo"] > 0, "above 0")
            count(fm, sep_below(rw, lo_thr(a_)), "gain", rw["hi"] < 0, "below 0")
        for k_, rw in d["ia0_row"].items():
            thr, _ = thr97(refs, "ia0", k_)
            count("HEALTH", sep_above(rw, thr), "MRR", rw["lo"] > 0)
        for k_, rw in d["cos_row"].items():
            h_, _ = thr97(refs, "cos", k_)
            beside["cosine (above 0)"][0] += 1
            beside["cosine (above 0)"][1] += int(sep_above(rw, h_))
        for k_, rw in d["tsi_row"].items():
            P_, tag_ = k_.split("|")
            refd = np.array([np.mean([x["tsi"][key(tag_, P_, 3, q)] for q in FAST]) - np.mean([x["tsi"][key(tag_, P_, 3, q)] for q in SLOW]) -
                             (np.mean([x["tsi"][key(tag_, P_, 1, q)] for q in FAST]) - np.mean([x["tsi"][key(tag_, P_, 1, q)] for q in SLOW])) for x in refs])
            beside["TSI(L3)-TSI(L1) (above 0)"][0] += 1
            beside["TSI(L3)-TSI(L1) (above 0)"][1] += int(sep_above(rw, float(np.nanpercentile(refd, 97.5))))
        for k_, rw in d["slope_row"].items():
            h_, _ = thr97(refs, "slope", k_)
            beside["consistency slope (above 0)"][0] += 1
            beside["consistency slope (above 0)"][1] += int(sep_above(rw, h_))
    for f, v in fam.items():
        v["share"] = v["separated"] / max(v["rows"], 1)
        v["share_bootstrap_only"] = v["bootstrap_only"] / max(v["rows"], 1)
        v["kind_shares"] = {kd: (b[1] / max(b[0], 1)) for kd, b in v["by_kind"].items()}
        v["kind_pass"] = {kd: bool(b[0] > 0 and b[1] / b[0] <= 0.05) for kd, b in v["by_kind"].items() if kd in BAR_KINDS}
        v["pass"] = bool(v["rows"] > 0 and v["share"] <= 0.05 and all(v["kind_pass"].values()))
    R["control3"] = {"families": fam, "pass": bool(all(v["pass"] for v in fam.values())), "beside_kinds_no_bar": {k_: {"rows": b[0], "separated": b[1], "share": b[1] / max(b[0], 1)} for k_, b in beside.items()}}
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


def lo_thr(a_):
    return float(np.nanpercentile(a_, 2.5)) if len(a_) else float("nan")


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
