"""IMAG1C library (IMPL_CHOICES_IMAG1C.md): IA-2 rows on frames with CAPLADDER's ridge probe, IMAG1B's two-part rule and labels. T0, CPU, no training.
A Ctx is a SimpleNamespace of arrays: the real one (build_imag1c.py + imag1c_ctx.pkl) or a synthetic one (test_imag1c_controls.py); every function here is shared."""
import hashlib, os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "imag1b"))
import imag1b_lib as IL  # noqa: E402   (Mom, mom_cont, sep_above, sep_below, perm_labels, boot_weights, mrr_mom; unchanged IMAG1B code, read only)

NF = 5
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TAGS, LEVELS, KS = ("M2b_s0", "M2b_s1"), (1, 2, 3), (1, 2, 3)
STRIDE = {1: 1, 2: 2, 3: 4}
READ = ("lead_gap", "lead_closing", "ttc")           # lead_gap decides; the other two are beside, no bar
CTRL = ("speed",)                                     # control-only target (control 2 / 3 machinery on a target known to be encoded)
FIT_TARGETS = READ + CTRL
QTS = ("copy", "pred", "oracle", "nf")
SUBS = ("all", "old")
R_REF, N_CAL = 199, 20
REF_SEEDS, CAL_SEEDS = list(range(1, R_REF + 1)), list(range(1001, 1001 + N_CAL))


def _h(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def outer_fold(eids, nf=NF):
    return np.array([_h(str(e)) % nf for e in eids])


def inner_fold(eids):
    return np.array([_h(str(e) + "|inner") % 5 for e in eids])


# ------------------------------------------------------------------------------------------------ the ridge (CAPLADDER's dual ridge, computed in the primal)
def _decomp(Xc, yc):
    s, V = np.linalg.eigh(Xc.T @ Xc)
    s = np.clip(s, 0.0, None)
    return s, V, V.T @ (Xc.T @ yc)


def fit_ridge(Xs, y, inner, cache=None, ck=None):
    """Xs [n, d] standardised fit rows, y [n], inner [n] inner-fold ids. lambda_c = c tr(K~) / n (tr = sum of the eigenvalues of the centred Gram = of X~'X~),
    c chosen by inner 5-fold episode CV (pooled SSE, first c on ties), refit on all rows. Same predictor as capladder_lib.oof_fold / Split.
    cache (a dict) + ck: the label-free parts (fit-row mean, eigen-decomposition, projected held-out rows) are reused across draws, which change only y."""
    sse = np.zeros(len(C_GRID))
    for j in range(5):
        tr, te = inner != j, inner == j
        if not te.any() or not tr.any():
            continue
        kk = None if cache is None else (ck, j)
        if kk is not None and kk in cache:
            mu, s, V, VX = cache[kk]
            ym = y[tr].mean()
            Z = V.T @ ((Xs[tr] - mu).T @ (y[tr] - ym))
        else:
            mu, ym = Xs[tr].mean(0), y[tr].mean()
            s, V, Z = _decomp(Xs[tr] - mu, y[tr] - ym)
            VX = (Xs[te] - mu) @ V
            if kk is not None:
                cache[kk] = (mu, s, V, VX)
        for ci, c in enumerate(C_GRID):
            lam = c * s.sum() / tr.sum()
            sse[ci] += ((y[te] - (VX @ (Z / (s + lam)) + ym)) ** 2).sum()
    ci = int(np.argmin(sse))
    kk = None if cache is None else (ck, "F")
    if kk is not None and kk in cache:
        mu, s, V = cache[kk]
        ym = y.mean()
        Z = V.T @ ((Xs - mu).T @ (y - ym))
    else:
        mu, ym = Xs.mean(0), y.mean()
        s, V, Z = _decomp(Xs - mu, y - ym)
        if kk is not None:
            cache[kk] = (mu, s, V)
    lam = C_GRID[ci] * s.sum() / len(y)
    return {"mu": mu, "w": V @ (Z / (s + lam)), "ym": ym, "c": ci}


def std_stats(c, tag, l):
    """Per fold: mean and population sd (+1e-8) of the fit-fold rows of set A (finite lead_gap), CAPLADDER section 5."""
    key = (tag, l)
    if key not in c._std:
        X = np.asarray(c.BZ[tag][l], np.float64)
        A = np.isfinite(c.Y["lead_gap"])
        c._std[key] = [(X[A & (c.bfold != k)].mean(0), X[A & (c.bfold != k)].std(0) + 1e-8) for k in range(NF)]
    return c._std[key]


def fit_tag(c, tag, s, targets=FIT_TARGETS):
    """PR[(l, q, k)] = ridge of fold k (fitted on bank rows of the other folds). s = 0 real labels; s > 0 labels deranged across episodes inside the fit fold
    (default_rng(s) created per tag, consumed level -> target -> fold). The lambda selection is refitted on the deranged labels."""
    rng = np.random.default_rng(s) if s > 0 else None
    PR = {}
    for l in LEVELS:
        X = np.asarray(c.BZ[tag][l], np.float64)
        st = std_stats(c, tag, l)
        for q in targets:
            val = np.isfinite(c.Y[q])
            for k in range(NF):
                rows = np.flatnonzero(val & (c.bfold != k))
                y = c.Y[q][rows]
                if rng is not None:
                    y = IL.perm_labels(y, c.beid[rows], rng)
                mu, sd = st[k]
                PR[(l, q, k)] = fit_ridge((X[rows] - mu) / sd, y, c.binner[rows], c._rcache, (tag, l, q, k))
    return PR


def decode(c, PR, tag, l, q, Z, rowfold):
    Z = np.asarray(Z, np.float64)
    out = np.full(len(Z), np.nan)
    st = std_stats(c, tag, l)
    for k in range(NF):
        m = rowfold == k
        if m.any():
            mu, sd = st[k]
            p = PR[(l, q, k)]
            out[m] = ((Z[m] - mu) / sd - p["mu"]) @ p["w"] + p["ym"]
    return out


# ------------------------------------------------------------------------------------------------ scoring (IMAG1B's Mom with 5 scoring folds)
def moms_cont(c, y, pred, valid):
    ym = np.array([y[valid & (c.qfold == k)].mean() if (valid & (c.qfold == k)).any() else 0.0 for k in range(NF)])
    return IL.mom_cont(y, pred, valid, c.qpos, c.n_ep, ym[c.qfold])


def sub_mask(c, sub):
    return np.ones(len(c.qe), bool) if sub == "all" else c.old


def score_windows(c, tag, PR, keep, targets=FIT_TARGETS):
    """Mom (keep) or its score (not keep) for every (l, k, q, qt, sub). Predictions are computed once per (l, q, qt, k)."""
    out = {}
    for l in LEVELS:
        for q in targets:
            dcopy = decode(c, PR, tag, l, q, c.Z0[tag][l], c.qfold)
            for k in KS:
                f = k * STRIDE[l]
                y = c.Yw[q][:, f]
                dec = {"copy": dcopy, "pred": decode(c, PR, tag, l, q, c.PRED[tag][(l, k)], c.qfold),
                       "oracle": decode(c, PR, tag, l, q, c.ORA[tag][(l, k)], c.qfold), "nf": decode(c, PR, tag, l, q, c.ORA[tag][(l, k)][c.nf_idx], c.qfold)}
                for sub in SUBS:
                    valid = np.isfinite(y) & sub_mask(c, sub)
                    for qt in QTS:
                        m = moms_cont(c, y, dec[qt], valid)
                        out[(l, k, q, qt, sub)] = m if keep else m.point()
    return out


def ia0_all(c, tag, rng=None):
    """IA-0 (IMAG1B unchanged): excess MRR of the predicted k = 1 latent retrieving the encoded true future among the windows of its clip. rng None = real; else a derangement of the answer."""
    out = {}
    for l in LEVELS:
        m, mrr, ch = IL.mrr_mom(c, c.PRED[tag][(l, 1)].astype(np.float64), c.ORA[tag][(l, 1)].astype(np.float64), rng=rng)
        out[l] = {"mom": m, "mrr": mrr, "chance": ch}
    return out


def run_draw(c, s, role, keep=None):
    """s = 0: real. s > 0: one reference (role 'ref') or calibration (role 'cal') draw. Arrays kept (Mom) for the real and calibration draws, scores only for references."""
    keep = (role != "ref") if keep is None else keep
    res = {"s": s, "role": role, "tags": {}}
    rngm = np.random.default_rng(10_000 + s)
    for tag in c.tags:
        PR = fit_tag(c, tag, s)
        res["tags"][tag] = {"ia2": score_windows(c, tag, PR, keep), "cs": [int(PR[(l, "lead_gap", k)]["c"]) for l in LEVELS for k in range(NF)],
                            "ia0": ia0_all(c, tag, None if s == 0 else rngm)}
    return res


# ------------------------------------------------------------------------------------------------ the analysis: rows, labels, controls
def thr(refs, tag, key, p):
    a = np.array([d["tags"][tag]["ia2"][key] for d in refs], np.float64)
    a = a[np.isfinite(a)]
    return float(np.nanpercentile(a, p)) if len(a) else np.nan


def thr_diff(refs, tag, k1, k2, p):
    a = np.array([d["tags"][tag]["ia2"][k1] - d["tags"][tag]["ia2"][k2] for d in refs], np.float64)
    a = a[np.isfinite(a)]
    return float(np.nanpercentile(a, p)) if len(a) else np.nan


def row_tests(c, moms, refs, tag, l, k, q, sub, with_refs=True):
    """The IA-2 row of one (tag, l, k, q, sub): scores, NMSE, Oracle READABLE, pred - copy above / below, Oracle - not-future separated. moms {qt: Mom} (real or a calibration draw)."""
    W = c.W
    r = {qt: moms[qt].row(W) for qt in QTS}
    e = {"rows": r, "n": r["pred"]["n"], "nep": r["pred"]["nep"],
         "nmse": {qt: {"point": 1 - r[qt]["point"], "lo": 1 - r[qt]["hi"], "hi": 1 - r[qt]["lo"]} for qt in QTS}}
    pc = moms["pred"].diff(moms["copy"]).row(W)
    on = moms["oracle"].diff(moms["nf"]).row(W)
    po = moms["pred"].diff(moms["oracle"]).row(W)
    e["pred_minus_copy"], e["oracle_minus_nf"], e["pred_minus_oracle"] = pc, on, po
    if with_refs:
        K = lambda qt: (l, k, q, qt, sub)
        o_thr = thr(refs, tag, K("oracle"), 97.5)
        pch, pcl = thr_diff(refs, tag, K("pred"), K("copy"), 97.5), thr_diff(refs, tag, K("pred"), K("copy"), 2.5)
        onh = thr_diff(refs, tag, K("oracle"), K("nf"), 97.5)
        e["oracle_thr97"], e["oracle_readable"] = o_thr, IL.sep_above(r["oracle"], o_thr)
        pc.update({"thr_hi": pch, "thr_lo": pcl, "above": IL.sep_above(pc, pch), "below": IL.sep_below(pc, pcl)})
        on.update({"thr_hi": onh, "above": IL.sep_above(on, onh)})
        e["nmse_ref_p2_5"] = 1 - o_thr
    return e


def label(es):
    """IMAG1B's IA-2 label from the two seeds' row tests."""
    if not all(e["oracle_readable"] for e in es):
        return "UNREADABLE"
    if all(e["pred_minus_copy"]["above"] for e in es):
        return "IMAGINES"
    if all(e["pred_minus_copy"]["below"] for e in es):
        return "WORSE-THAN-COPY"
    return "COPIES"


def analyse(c, real, refs, cals):
    R = {"n_windows": int(len(c.qe)), "n_old": int(c.old.sum()), "n_episodes": int(c.n_ep), "R": len(refs), "n_cal": len(cals), "rows": {}, "labels": {}}
    # IA-0 health
    ia0 = {}
    for tag in c.tags:
        for l in LEVELS:
            d = real["tags"][tag]["ia0"][l]
            row = d["mom"].row(c.W)
            a = np.array([x["tags"][tag]["ia0"][l]["mom"].point() for x in refs], np.float64)
            t = float(np.nanpercentile(a, 97.5))
            ia0[f"{tag}|{l}"] = {"mrr": d["mrr"], "chance": d["chance"], "excess": row, "thr97": t, "separated": IL.sep_above(row, t)}
    R["ia0"] = ia0
    R["healthy"] = {str(l): all(ia0[f"{tag}|{l}"]["separated"] for tag in c.tags) for l in LEVELS}
    for sub in SUBS:
        for q in FIT_TARGETS:
            for l in LEVELS:
                for k in KS:
                    es = []
                    for tag in c.tags:
                        mo = {qt: real["tags"][tag]["ia2"][(l, k, q, qt, sub)] for qt in QTS}
                        e = row_tests(c, mo, refs, tag, l, k, q, sub)
                        R["rows"][f"{sub}|{tag}|{l}|{k}|{q}"] = e
                        es.append(e)
                    R["labels"][f"{sub}|{l}|{k}|{q}"] = {"label": label(es), "healthy": R["healthy"][str(l)], "n": [e["n"] for e in es], "nep": [e["nep"] for e in es]}
    # control 2 (not-future): cells = (tag, l, k, q) of the non-UNREADABLE labels (sub all); speed is the control-only target
    cells, bad = [], []
    for q in FIT_TARGETS:
        for l in LEVELS:
            for k in KS:
                if R["labels"][f"all|{l}|{k}|{q}"]["label"] == "UNREADABLE":
                    continue
                for tag in c.tags:
                    ok_ = R["rows"][f"all|{tag}|{l}|{k}|{q}"]["oracle_minus_nf"]["above"]
                    cells.append(f"{tag}|{l}|{k}|{q}")
                    if not ok_:
                        bad.append(f"{tag}|{l}|{k}|{q}")
    R["control2"] = {"cells": len(cells), "failed": bad, "pass": bool(len(cells) > 0 and not bad)}
    # control 3: the calibration draws scored as if real, rows of the three read targets (and speed printed)
    cnt = {}
    for d in cals:
        for q in FIT_TARGETS:
            for tag in c.tags:
                for l in LEVELS:
                    for k in KS:
                        mo = {qt: d["tags"][tag]["ia2"][(l, k, q, qt, "all")] for qt in QTS}
                        e = row_tests(c, mo, refs, tag, l, k, q, "all")
                        for kind, flag in (("Oracle", e["oracle_readable"]), ("pred-vs-Copy above", e["pred_minus_copy"]["above"]), ("pred-vs-Copy below", e["pred_minus_copy"]["below"])):
                            for grp in (q, "READ3" if q in READ else "CTRL"):
                                x = cnt.setdefault((grp, kind), [0, 0]); x[0] += int(flag); x[1] += 1
    c3 = {}
    for (grp, kind), (a, b) in cnt.items():
        c3[f"{grp}|{kind}"] = {"separated": a, "rows": b, "share": a / b}
    for grp in sorted({g for g, _ in cnt}):
        a = cnt[(grp, "pred-vs-Copy above")]; b = cnt[(grp, "pred-vs-Copy below")]
        c3[f"{grp}|pred-vs-Copy both sides"] = {"separated": a[0] + b[0], "rows": a[1] + b[1], "share": (a[0] + b[0]) / (a[1] + b[1])}
    deciding = [v for k_, v in c3.items() if k_.startswith("lead_gap|") or k_.startswith("READ3|")]
    R["control3"] = {"shares": c3, "pass": bool(all(v["share"] <= 0.05 for k_, v in c3.items() if (k_.startswith("lead_gap|") or k_.startswith("READ3|")) and "above" not in k_ and "below" not in k_))}
    R["banner"] = "VALID" if (R["control3"]["pass"] and R["control2"]["pass"]) else "INVALID"
    return R


# ------------------------------------------------------------------------------------------------ context helpers (real and synthetic)
def finish_ctx(c):
    c.uniq = np.unique(c.beid)
    c.n_ep = len(c.uniq)
    c.bfold, c.binner = outer_fold(c.beid), inner_fold(c.beid)
    c.qfold = outer_fold(c.qe)
    c.qpos = np.searchsorted(c.uniq, c.qe)
    c.groups = [np.flatnonzero(c.qe == e) for e in c.uniq]
    c.W = IL.boot_weights(c.uniq)
    c.nf_idx = nf_indices(c.qe)
    c._std = {}
    c._rcache = {}
    return c


def nf_indices(qe, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.empty(len(qe), dtype=np.int64)
    ue = {e: np.flatnonzero(qe != e) for e in np.unique(qe)}
    for i in range(len(qe)):
        idx[i] = rng.choice(ue[qe[i]])
    return idx
