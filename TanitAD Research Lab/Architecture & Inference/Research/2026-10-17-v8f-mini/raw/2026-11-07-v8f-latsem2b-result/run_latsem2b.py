"""PREREG_V8F_LATSEM2B (card prefix 4,611 bytes bf6b3717...; Amendment 1 prefix 2,778 bytes 120072ff...; IMPL_CHOICES_LATSEM2B.md frozen sha256 a90cd9ca...):
the scene read of LATSEM2 with a permutation-calibrated instrument. Tier T0 (a world-model diagnostic), CPU only, zero training.
Run from theory/latsem2b with WMACT_CODE_ROOT set:  python -u run_latsem2b.py
Deciding set TRAIN (WMACT's 141 TRAIN clips), then eval-141 beside (post hoc). Writes RESULT_LATSEM2B.json (banked after each stage), draws_{train,eval141}.jsonl, fig/*.png.
The statistics are run 1's (theory/latsem2/run_latsem2.py): probes, S1, S2, S3 and the episode-cluster bootstrap are copied; the additions are the permutation
reference, the two-part rule, S3-PARTIAL, the calibration control and the speed control. Permutation draws run in a pool of 4 single-thread worker processes."""
import hashlib, json, os, sys, time, warnings
sys.dont_write_bytecode = True
from types import SimpleNamespace
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

CARD, CARD_N, CARD_SHA = "../PREREG_V8F_LATSEM2B.md", 4611, "bf6b37178818f778cf9220c4abac1a44de11a954d0e3a2b57f39e8e8b1bc48a8"
AMEND, AMEND_N, AMEND_SHA = "../AMENDMENT_LATSEM2B_1.md", 2778, "120072ffb46081bec267fc4f515e5704848272792790eb2577f357c8739ece53"
IMPL, IMPL_SHA = "IMPL_CHOICES_LATSEM2B.md", "a90cd9ca91901b7578ddf4ccbf73bbce461b847753a43ac5f140bdb4ed59dfd0"
TAGS = ("M2b_s0", "M2b_s1")
STRIDE, KMAX = {1: 1, 2: 2, 3: 4}, {1: 10, 2: 3, 3: 3}
KS = {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}
LAM, F = 10.0, 16
ACT3 = {"brake": (-3.0, 0.0), "accel": (1.5, 0.0), "brake_twice": (-3.0, 0.0)}
TARGETS = ("lead_present", "lead_gap", "lead_closing", "agents30", "nearest", "lon7", "turn")
BESIDE = ("lead_closing_uncapped", "agents30_view")
ALLQ = TARGETS + BESIDE
KIND = {"lead_present": "bin", "lead_gap": "cont", "lead_closing": "cont", "agents30": "cont", "nearest": "cont", "lon7": "cls", "turn": "bin",
        "lead_closing_uncapped": "cont", "agents30_view": "cont"}
CLASSES = {"lead_present": [0.0, 1.0], "turn": [0.0, 1.0], "lon7": [float(c) for c in range(7)]}
REF_SEEDS, CAL_SEEDS = list(range(1, 200)), list(range(1001, 1021))
CONV = {"fits": 0, "not_converged": 0}
C = None                                   # the dataset context (set in the main process and in each worker)


def set_ctx(ctx):
    global C
    C = ctx


# ------------------------------------------------------------------------------------------------ statistics (run 1's code; the context C replaces run 1's module globals)
def esum(x, p):
    return np.bincount(p, weights=np.nan_to_num(np.asarray(x, np.float64)), minlength=len(C.uniq))


def pct(b):
    b = np.asarray(b); lo, hi = np.nanpercentile(b, [2.5, 97.5]); return float(lo), float(hi)


def row(num_pos, num_neg, den, n, nep):
    ds = den.sum()
    if ds == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "separated": False, "n": int(n), "n_episodes": int(nep), "undefined": True}
    p = float((num_pos.sum() - num_neg.sum()) / ds)
    b = (C.Wt @ num_pos - C.Wt @ num_neg) / (C.Wt @ den)
    lo, hi = pct(b)
    return {"point": p, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n": int(n), "n_episodes": int(nep)}


def bal(N, TP):
    rec = np.where(N > 0, TP / np.maximum(N, 1e-12), np.nan)
    return np.nanmean(rec, axis=-1)


def bal_row(y, pred, base, val, classes):
    p = C.pos
    N = np.stack([esum((y == c) & val, p) for c in classes], 1)
    TP = np.stack([esum((y == c) & val & (pred == c), p) for c in classes], 1)
    TB = np.stack([esum((y == c) & val & (base == c), p) for c in classes], 1)
    pm, pb = float(bal(N.sum(0), TP.sum(0))), float(bal(N.sum(0), TB.sum(0)))
    d = bal(C.Wt @ N, C.Wt @ TP) - bal(C.Wt @ N, C.Wt @ TB)
    lo, hi = pct(d)
    return {"point": pm - pb, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "bal_model": pm, "bal_base": pb, "n": int(val.sum()),
            "n_episodes": int(len(np.unique(C.beid[val])))}


def fit_probe(kind, X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    if kind == "cont":
        ym = y.mean()
        beta = np.linalg.solve(Xs.T @ Xs + LAM * np.eye(X.shape[1]), Xs.T @ (y - ym))
        return ("cont", mu, sd, beta, ym)
    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000, tol=1e-4)
    with warnings.catch_warnings(record=True) as wr:
        warnings.simplefilter("always")
        clf.fit(Xs, y.astype(np.int64))
    CONV["fits"] += 1
    CONV["not_converged"] += int(any(issubclass(x.category, ConvergenceWarning) for x in wr))
    vals, cnt = np.unique(y, return_counts=True)
    return ("cls", mu, sd, clf, float(vals[cnt.argmax()]))


def fit_probe_majority(y):
    vals, cnt = np.unique(y, return_counts=True)
    return float(vals[cnt.argmax()])


def predict(P, X):
    if P[0] == "cont":
        return ((X - P[1]) / P[2]) @ P[3] + P[4]
    return P[3].predict((X - P[1]) / P[2]).astype(np.float64)


def perm_labels(y, ep, rng):
    """LATSEM2's, unchanged: the fit-fold episodes are put in a random order, each takes the previous episode's label sequence cycled to its length; no episode keeps its own."""
    out = np.empty_like(y); order = np.unique(ep)[rng.permutation(len(np.unique(ep)))]
    for e, e2 in zip(order, np.roll(order, 1)):
        src = y[ep == e2]; n = int((ep == e).sum()); out[ep == e] = src[np.arange(n) % len(src)]
    return out


def residualise(g, v):
    """OLS of the gap on ego speed (with intercept); returns the residual g - (a + b v)."""
    b, a = np.polyfit(v, g, 1)
    return g - (a + b * v)


def s1_row(q, y, pred, base, val):
    if KIND[q] == "cont":
        sst, sse = esum((y - base) ** 2, C.pos), esum((pred - y) ** 2, C.pos)
        r = row(sst, sse, sst, val.sum(), len(np.unique(C.beid[val]))); r["metric"] = "R2"
    else:
        r = bal_row(y, pred, base, val, CLASSES[q]); r["metric"] = "BA_minus_majority"
    return r


def err_of(q, p, ystar, valid):
    e = (p - ystar) ** 2 if KIND[q] == "cont" else (p != ystar).astype(np.float64)
    return np.where(valid, e, 0.0)


def s2_cell(q, ystar, p_start, p_query):
    valid = np.isfinite(ystar); nep = len(np.unique(C.qe[valid]))
    e0 = esum(err_of(q, p_start, ystar, valid), C.qpos)
    e1 = esum(err_of(q, p_query, ystar, valid), C.qpos)
    r = row(e0, e1, e0, valid.sum(), nep)
    r["err_start_mean"], r["err_query_mean"] = float(e0.sum() / max(valid.sum(), 1)), float(e1.sum() / max(valid.sum(), 1))
    return r


def s3_row(ga, gb):
    d = np.where(C.m3, ga - gb, 0.0)
    return row(esum(d, C.qpos), np.zeros(len(C.uniq)), esum(C.m3.astype(np.float64), C.qpos), C.m3.sum(), C.nep3)


def decode(P, l, q, Z):
    out = np.empty(len(Z))
    for k in (0, 1):
        mk = C.qfold == k
        out[mk] = predict(P[(l, q, k)], Z[mk])
    return out


def decode_g(Pg, l, Z):
    out = np.empty(len(Z))
    for k in (0, 1):
        mk = C.qfold == k
        out[mk] = predict(Pg[(l, k)], Z[mk])
    return out


# ------------------------------------------------------------------------------------------------ evaluation of one set of probes (real or permuted: the same functions)
def eval_s1(tag, P):
    out = {}
    for l in (1, 2, 3):
        X = C.BZ[tag][l]
        for q in ALLQ:
            y, val = C.Y[q], C.val[q]
            pred, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for k in (0, 1):
                ev = val & (C.fold == k)
                pred[ev], base[ev] = predict(P[(l, q, k)], X[ev]), C.bases[(q, k)]
            out[f"{tag}/L{l}/{q}"] = s1_row(q, y, pred, base, val)
    return out


def eval_s2(tag, P, extras=None):
    out = {}
    for l, ks in KS.items():
        p0 = {q: decode(P, l, q, C.Z0[tag][l]) for q in ALLQ}
        nm_dec = "shrunk" if l == 1 else "raw"
        for k in ks:
            for q in ALLQ:
                ystar = C.Yw[q][:, k * STRIDE[l]]
                out[f"{tag}/L{l}/k{k}/{q}/{nm_dec}"] = s2_cell(q, ystar, p0[q], decode(P, l, q, C.ZQ[tag][(l, k)]))
                if extras is not None:
                    for nm, Z in extras[(tag, l, k)].items():
                        out[f"{tag}/L{l}/k{k}/{q}/{nm}"] = s2_cell(q, ystar, p0[q], decode(P, l, q, Z))
    return out


def eval_s3(tag, Pg, Pr, with_identical=False, raw_l1=None):
    out = {}
    for l in (1, 2, 3):
        za, zb = C.CF[tag][("accel", l)], C.CF[tag][("brake", l)]
        pa, pb, ra, rb = decode_g(Pg, l, za), decode_g(Pg, l, zb), decode_g(Pr, l, za), decode_g(Pr, l, zb)
        e = {"raw": s3_row(pa, pb), "partial": s3_row(ra, rb)}
        if with_identical:
            zt = C.CF[tag][("brake_twice", l)]
            e["raw_identical"], e["partial_identical"] = s3_row(decode_g(Pg, l, zt), pb), s3_row(decode_g(Pr, l, zt), rb)
            e["decoded_mean"] = {"raw": {"accel": float(pa[C.m3].mean()), "brake": float(pb[C.m3].mean())},
                                 "partial": {"accel": float(ra[C.m3].mean()), "brake": float(rb[C.m3].mean())}}
            if l == 1 and raw_l1 is not None:
                xa, xb = raw_l1[(tag, "accel")], raw_l1[(tag, "brake")]
                e["raw_unshrunk_beside"], e["partial_unshrunk_beside"] = s3_row(decode_g(Pg, l, xa), decode_g(Pg, l, xb)), s3_row(decode_g(Pr, l, xa), decode_g(Pr, l, xb))
        out[f"{tag}/L{l}"] = e
    return out


def fit_real(tag):
    """The real probes: S1 (all levels, targets, folds) and the residualised gap probes of S3-PARTIAL. Returns P, Pg, Pr and the control-4 correlations."""
    P, Pg, Pr, corr = {}, {}, {}, {}
    for l in (1, 2, 3):
        X = C.BZ[tag][l]
        for q in ALLQ:
            for k in (0, 1):
                fit = C.val[q] & (C.fold != k)
                P[(l, q, k)] = fit_probe(KIND[q], X[fit], C.Y[q][fit])
        for k in (0, 1):
            Pg[(l, k)] = P[(l, "lead_gap", k)]
            idx = np.flatnonzero(C.val["lead_gap"] & np.isfinite(C.speed) & (C.fold != k))
            r = residualise(C.Y["lead_gap"][idx], C.speed[idx])
            Pr[(l, k)] = fit_probe("cont", X[idx], r)
            corr[k] = float(np.corrcoef(r, C.speed[idx])[0, 1])
    return P, Pg, Pr, corr


# ------------------------------------------------------------------------------------------------ one permutation draw (worker task)
def compact(d):
    return {k: [v["point"], v["lo"], v["hi"]] for k, v in d.items()}


def draw_task(args):
    s, role = args
    t0 = time.time(); CONV["fits"] = 0; CONV["not_converged"] = 0
    out = {"s": int(s), "role": role, "S1": {}, "S2": {}, "S3raw": {}, "S3part": {}, "dig": {}, "res_corr_max": 0.0}
    for tag in TAGS:
        rng = np.random.default_rng(s)                                         # S1 / S2 probes: consumed level -> target -> fold
        P = {}
        for l in (1, 2, 3):
            X = C.BZ[tag][l]
            for q in ALLQ:
                y, val = C.Y[q], C.val[q]
                for k in (0, 1):
                    fit = val & (C.fold != k)
                    yp = perm_labels(y[fit], C.beid[fit], rng)
                    P[(l, q, k)] = fit_probe(KIND[q], X[fit], yp)
                    if tag == TAGS[0]:
                        out["dig"][f"{l}/{q}/{k}"] = hashlib.sha256(yp.tobytes()).hexdigest()[:16]
        out["S1"].update(compact(eval_s1(tag, P)))
        out["S2"].update(compact(eval_s2(tag, P)))
        rng3 = np.random.default_rng(s)                                        # S3 probes: gaps moved across the fit-fold lead frames, one draw per fold, shared across levels
        idxf = {k: np.flatnonzero(C.val["lead_gap"] & (C.fold != k)) for k in (0, 1)}
        yperm = {k: C.Y["lead_gap"][idxf[k]][rng3.permutation(len(idxf[k]))] for k in (0, 1)}
        Pg, Pr = {}, {}
        for l in (1, 2, 3):
            X = C.BZ[tag][l]
            for k in (0, 1):
                idx = idxf[k]
                Pg[(l, k)] = fit_probe("cont", X[idx], yperm[k])
                sp = C.speed[idx]; keep = np.isfinite(sp)
                r = residualise(yperm[k][keep], sp[keep])
                Pr[(l, k)] = fit_probe("cont", X[idx[keep]], r)
                out["res_corr_max"] = max(out["res_corr_max"], abs(float(np.corrcoef(r, sp[keep])[0, 1])))
        e3 = eval_s3(tag, Pg, Pr)
        out["S3raw"].update({k: [v["raw"]["point"], v["raw"]["lo"], v["raw"]["hi"]] for k, v in e3.items()})
        out["S3part"].update({k: [v["partial"]["point"], v["partial"]["lo"], v["partial"]["hi"]] for k, v in e3.items()})
    out["conv"] = dict(CONV); out["t"] = time.time() - t0
    return out


def init_worker(ctx):
    set_ctx(ctx)


# ------------------------------------------------------------------------------------------------ analysis of reference and calibration draws
def family_of(kind, key):
    if kind == "S1":
        return "S1_classification" if KIND[key.split("/")[2]] != "cont" else "S1_continuous"
    return {"S2": "S2", "S3part": "S3_PARTIAL"}[kind]


def ref_table(draws_ref, kind):
    keys = list(draws_ref[0][kind].keys())
    A = {k: np.array([d[kind][k][0] for d in draws_ref], np.float64) for k in keys}
    return A


def thresholds(a):
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0, "p2.5": float("nan"), "p97.5": float("nan"), "min": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"n": int(len(a)), "p2.5": float(np.percentile(a, 2.5)), "p97.5": float(np.percentile(a, 97.5)), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def two_part(kind, rowv, thr):
    """row [point, lo, hi] or a row dict; claimed side above 0 for S1 and S2, below 0 for S3-PARTIAL."""
    p, lo, hi = (rowv["point"], rowv["lo"], rowv["hi"]) if isinstance(rowv, dict) else rowv
    if not np.isfinite(p):
        return False, False
    if kind == "S3part":
        return bool(hi < 0), bool(hi < 0 and p < thr["p2.5"])
    return bool(lo > 0), bool(lo > 0 and p > thr["p97.5"])


def rank_p(kind, p, ref):
    ref = ref[np.isfinite(ref)]
    n = int((ref <= p).sum()) if kind == "S3part" else int((ref >= p).sum())
    return (1 + n) / (len(ref) + 1)


def analyse(real, draws_ref, draws_cal):
    """Reference thresholds, the two-part rule on the real rows, the calibration control."""
    out = {"R": len(draws_ref), "n_cal": len(draws_cal), "S1": {}, "S2": {}, "S3": {}, "calibration": {}}
    ref = {kd: ref_table(draws_ref, kd) for kd in ("S1", "S2", "S3raw", "S3part")}
    thr = {kd: {k: thresholds(a) for k, a in ref[kd].items()} for kd in ref}
    for k, v in real["S1"].items():
        t = thr["S1"][k]; sb, s2 = two_part("S1", v["main"] if "main" in v else v, t)
        out["S1"][k] = {"main": v, "ref": t, "rank_p": rank_p("S1", v["point"], ref["S1"][k]), "separated_bootstrap_only": sb, "separated_two_part": s2}
    for k, v in real["S2"].items():
        kk = k if k in thr["S2"] else None
        if kk is None:
            continue
        t = thr["S2"][kk]; sb, s2 = two_part("S2", v, t)
        out["S2"][kk] = {"row": v, "ref": t, "rank_p": rank_p("S2", v["point"], ref["S2"][kk]), "separated_bootstrap_only": sb, "separated_two_part": s2}
    for k, v in real["S3"].items():
        tp, tr = thr["S3part"][k], thr["S3raw"][k]
        sbp, s2p = two_part("S3part", v["partial"], tp)
        sbr, s2r = two_part("S3part", v["raw"], tr)
        out["S3"][k] = {"partial": v["partial"], "raw": v["raw"], "ref_partial": tp, "ref_raw": tr,
                        "rank_p_partial": rank_p("S3part", v["partial"]["point"], ref["S3part"][k]), "rank_p_raw": rank_p("S3part", v["raw"]["point"], ref["S3raw"][k]),
                        "partial_separated_bootstrap_only": sbp, "partial_separated_two_part": s2p, "raw_below0_bootstrap_only": sbr, "raw_below0_two_part_beside": s2r}
    # calibration: each calibration draw as if real, against the same cells' reference
    fam = {f: {"rows": 0, "two_part": 0, "bootstrap_only": 0} for f in ("S1_classification", "S1_continuous", "S2", "S3_PARTIAL")}
    for d in draws_cal:
        for kd in ("S1", "S2", "S3part"):
            for k, rv in d[kd].items():
                sb, s2 = two_part(kd, rv, thr[kd][k])
                f = fam[family_of(kd, k)]
                f["rows"] += 1; f["two_part"] += int(s2); f["bootstrap_only"] += int(sb)
    for f, v in fam.items():
        v["share_two_part"] = v["two_part"] / max(v["rows"], 1); v["share_bootstrap_only"] = v["bootstrap_only"] / max(v["rows"], 1)
        v["pass"] = bool(v["rows"] > 0 and v["share_two_part"] <= 0.05)
    out["calibration"] = fam
    out["calibration_pass"] = bool(all(v["pass"] for v in fam.values()))
    # diagnostics
    dig = {}
    for d in draws_ref + draws_cal:
        for k, h in d["dig"].items():
            dig.setdefault(k, set()).add(h)
    out["distinct_label_assignments"] = {"cells": len(dig), "min_distinct": int(min(len(v) for v in dig.values())), "draws": len(draws_ref) + len(draws_cal),
                                         "cells_below_all_distinct": sorted(k for k, v in dig.items() if len(v) < len(draws_ref) + len(draws_cal))[:40]}
    out["speed_control_permuted_max_abs_corr"] = float(max(d["res_corr_max"] for d in draws_ref + draws_cal))
    out["logistic_fits_in_draws"] = {"fits": int(sum(d["conv"]["fits"] for d in draws_ref + draws_cal)), "not_converged": int(sum(d["conv"]["not_converged"] for d in draws_ref + draws_cal))}
    out["nan_points_dropped_from_reference"] = int(sum(int((~np.isfinite(a)).sum()) for kd in ref for a in ref[kd].values()))
    return out


# ------------------------------------------------------------------------------------------------ main (heavy imports live here so workers stay light)
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def build_ctx(ds, W, LB2, LBb, torch):
    """Bank, labels, level latents, S2 / S3 states for one dataset. Returns (ctx, extras, misc)."""
    t0 = time.time()
    if ds == "train":
        B = LBb.build_bank_train(); labs = np.load("labels_latsem2b_train.npz"); speed = labs["speed"]
    else:
        B = LB2.build_bank(); labs = np.load("../latsem2/labels_latsem2.npz"); speed = None
    L, eid, tw, ok, A_raw, wkeep, G, key2b = B["L"], B["eid"], B["tw"], B["ok"], B["A_raw"], B["wkeep"], B["G"], B["key2b"]
    occ_w, occ_f, occ_b, bw, bf, beid, bg = B["occ_w"], B["occ_f"], B["occ_b"], B["bw"], B["bf"], B["beid"], B["bg"]
    assert (labs["beid"] == beid.astype("U36")).all() and (labs["bg"] == bg).all(), "label arrays do not match the bank"
    Y = {q: np.asarray(labs[q], np.float64) for q in ALLQ}
    if speed is None:                                                     # eval-141: native_tracks column 3 at raw row 2g (the expression of latsem2/beside_gap_speed.py)
        TR = {k: np.asarray(v, np.float64) for k, v in torch.load(LB2.TRACKS, weights_only=False)["tracks"].items()}
        speed = np.array([TR[str(e)][2 * int(g), 3] if 2 * int(g) < len(TR[str(e)]) else np.nan for e, g in zip(beid, bg)])
    fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in beid])
    uniq, Wt = W.boot_weights(beid)
    pos = np.searchsorted(uniq, beid)
    val = {q: np.isfinite(Y[q]) for q in ALLQ}
    bases = {}
    for q in ALLQ:
        for k in (0, 1):
            fit = val[q] & (fold != k)
            bases[(q, k)] = float(Y[q][fit].mean()) if KIND[q] == "cont" else fit_probe_majority(Y[q][fit])
    sel = wkeep[ok[wkeep].all(1)]
    n = len(sel); qe = eid[sel]; qpos = np.searchsorted(uniq, qe)
    qfold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in qe])
    bidx = np.array([[key2b[(eid[i], int(G[i, f]))] for f in range(F)] for i in sel])
    Yw = {q: Y[q][bidx] for q in ALLQ}
    m3 = Yw["lead_present"][:, 0] == 1
    SHR = json.load(open("../wmact/RESULT_WMACT_SHRINK.json", encoding="utf-8"))["rows"]
    ctx = SimpleNamespace(name=ds, beid=beid, fold=fold, uniq=uniq, Wt=Wt, pos=pos, Y=Y, val=val, bases=bases, speed=speed, qe=qe, qpos=qpos, qfold=qfold, Yw=Yw, m3=m3,
                          nep3=int(len(np.unique(qe[m3]))), BZ={}, Z0={}, ZQ={}, CF={})
    extras, raw_l1, ctrl = {}, {}, {"copy_maxdiff_z1": 0.0, "level_maxdiff": {}}
    for tag in TAGS:
        Z = L["Z1"][tag]
        ctrl["copy_maxdiff_z1"] = max(ctrl["copy_maxdiff_z1"], float((Z[occ_w, occ_f] - Z[bw[occ_b], bf[occ_b]]).abs().max()))
    ctrl["copy_maxdiff_actions"] = float((A_raw[occ_w, occ_f] - A_raw[bw[occ_b], bf[occ_b]]).abs().max())
    for tag in TAGS:
        m, cfg, ck = W.load_model(W.CKPT[tag]); st = m.wm.stack; wm = m.wm
        with torch.no_grad():
            zs = st.initial_latents(L["Z1"][tag][bw, bf])
        ctx.BZ[tag] = {l + 1: zs[l].double().numpy() for l in range(3)}
        with torch.no_grad():
            enc = st.encode(L["Z1"][tag][sel], wm.scale_actions(A_raw[sel])[:, :-1])
        c2 = 0.0
        for l in (2, 3):
            zl = enc.z[l - 1].double().numpy()
            for i in range(zl.shape[1]):
                b = np.array([key2b[(eid[w], int(G[w, i * STRIDE[l]]))] for w in sel])
                c2 = max(c2, float(np.abs(zl[:, i] - ctx.BZ[tag][l][b]).max()))
        ctrl["level_maxdiff"][tag] = c2
        with torch.no_grad():
            pk = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(A_raw[sel])[:, :-1])
        ctx.Z0[tag] = {l: pk["true"][l][:, 0].double().numpy() for l in (1, 2, 3)}
        for l, ks in KS.items():
            z0 = ctx.Z0[tag][l]
            for k in ks:
                zt, zh = pk["true"][l][:, k].double().numpy(), pk["pred"][(l, k)].double().numpy()
                ex = {"oracle": zt, "start": z0}
                if l == 1:
                    ctx.ZQ.setdefault(tag, {})[(l, k)] = z0 + SHR[f"{tag}/L1/k{k}"]["c_train"] * (zh - z0)
                    ex["raw"] = zh
                else:
                    ctx.ZQ.setdefault(tag, {})[(l, k)] = zh
                extras[(tag, l, k)] = ex
        ctx.ZQ.setdefault(tag, {})
        ctx.CF[tag] = {}
        with torch.no_grad():
            for nm, (a, w_) in ACT3.items():
                Araw = torch.tensor([a, w_], dtype=torch.float32).expand(n, F, 2).contiguous()
                pkc = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(Araw)[:, :-1])
                for l in (1, 2, 3):
                    Zr = pkc["pred"][(l, KMAX[l])].double().numpy(); z0 = ctx.Z0[tag][l]
                    ctx.CF[tag][(nm, l)] = z0 + SHR[f"{tag}/L1/k{KMAX[1]}"]["c_train"] * (Zr - z0) if l == 1 else Zr
                    if l == 1 and nm in ("accel", "brake"):
                        raw_l1[(tag, nm)] = Zr
        print("ctx", ds, tag, round(time.time() - t0, 1), "s", flush=True)
    misc = {"n_bank_frames": int(len(beid)), "n_episodes": int(len(uniq)), "n": {q: int(val[q].sum()) for q in ALLQ}, "n_episodes_by_fold": [int(len(set(beid[fold == k]))) for k in (0, 1)],
            "n_windows": int(n), "n_windows_S3": int(m3.sum()), "n_episodes_S3": ctx.nep3, "gap_speed_pearson": float(np.corrcoef(Y["lead_gap"][val["lead_gap"] & np.isfinite(speed)], speed[val["lead_gap"] & np.isfinite(speed)])[0, 1]),
            "n_gap_speed": int((val["lead_gap"] & np.isfinite(speed)).sum()), "ctx_s": round(time.time() - t0, 1)}
    return ctx, extras, raw_l1, ctrl, misc, Wt


def selftest():
    rng = np.random.default_rng(123)
    n = len(C.beid)
    yy = rng.integers(0, 3, size=n).astype(np.float64); vv = np.ones(n, bool)
    perfect = bal_row(yy, yy.copy(), np.zeros(n), vv, [0.0, 1.0, 2.0])
    const = bal_row(yy, np.zeros(n), np.zeros(n), vv, [0.0, 1.0, 2.0])
    out = {"perfect_bal_model": perfect["bal_model"], "perfect_bal_base": perfect["bal_base"], "same_as_base_point": const["point"], "same_as_base_interval": [const["lo"], const["hi"]]}
    assert out["perfect_bal_model"] == 1.0 and abs(out["perfect_bal_base"] - 1 / 3) < 1e-12 and out["same_as_base_point"] == 0.0 and const["lo"] == 0.0 == const["hi"]
    return out


def pca_picture(ds, ctx, res_ds, plt):
    os.makedirs("fig", exist_ok=True)
    gap = ctx.Y["lead_gap"]; has = np.isfinite(gap)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for c, l in enumerate((1, 2, 3)):
        X = ctx.BZ[TAGS[0]][l]; Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P2 = Xc @ Vt[:2].T
        ev = float((S[:2] ** 2).sum() / (S ** 2).sum())
        axs[c].scatter(P2[~has, 0], P2[~has, 1], c="0.85", s=2)
        sc = axs[c].scatter(P2[has, 0], P2[has, 1], c=gap[has], s=3, cmap="viridis_r", vmin=0, vmax=40)
        axs[c].set_title(f"level {l}: PCA 1-2 ({100 * ev:.0f} % var)", fontsize=9)
        fig.colorbar(sc, ax=axs[c], label="lead gap (m)")
    fig.suptitle(f"{ds}: per-level latent maps of {len(ctx.beid):,} bank frames (M2b s0), coloured by lead gap (grey: no lead within 40 m / unlabelled); no camera frame in this picture", fontsize=9)
    fig.tight_layout(); p = f"fig/pca_levels_gap_{ds}.png"; fig.savefig(p, dpi=80); plt.close(fig)
    return {p: sha256_file(p)}


def main():
    import multiprocessing as mp
    import torch
    sys.path.insert(0, os.path.abspath("../wmact")); sys.path.insert(0, os.path.abspath("../latsem2"))
    import wmact_lib as W
    import labels_latsem2 as LB2
    import labels_latsem2b as LBb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert hashlib.sha256(open(CARD, "rb").read()[:CARD_N]).hexdigest() == CARD_SHA, "card prefix changed"
    assert hashlib.sha256(open(AMEND, "rb").read()[:AMEND_N]).hexdigest() == AMEND_SHA, "amendment prefix changed"
    assert hashlib.sha256(open(IMPL, "rb").read()).hexdigest() == IMPL_SHA, "IMPL_CHOICES changed after its freeze"
    torch.set_num_threads(4)
    t_start = time.time()
    res = {"card": {"file": "PREREG_V8F_LATSEM2B.md", "prefix_bytes": CARD_N, "sha256": CARD_SHA}, "amendment": {"file": "AMENDMENT_LATSEM2B_1.md", "prefix_bytes": AMEND_N, "sha256": AMEND_SHA},
           "impl_choices": {"file": IMPL, "sha256": IMPL_SHA}, "run_script_sha256": sha256_file("run_latsem2b.py"), "tier": "T0", "stage": "start", "datasets": {}, "pictures": {},
           "threads": "main process 4 threads; draws in a pool of 4 workers with 1 BLAS thread each"}
    ref1 = json.load(open("../latsem2/RESULT_LATSEM2.json", encoding="utf-8"))

    def bank(stage):
        res["stage"] = stage; res["wall_s"] = round(time.time() - t_start, 1)
        json.dump(res, open("RESULT_LATSEM2B.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)

    projected_checked = False
    for ds in ("train", "eval141"):
        dsn = "train" if ds == "train" else "eval141"
        ctx, extras, raw_l1, ctrl, misc, Wt = build_ctx(ds, W, LB2, LBb, torch)
        set_ctx(ctx)
        d = res["datasets"][ds] = dict(misc)
        d["role"] = "deciding set (WMACT TRAIN windows)" if ds == "train" else "beside, post hoc (LATSEM2's eval-141 bank)"
        d["controls_raw"] = dict(ctrl); d["selftest"] = selftest()
        t_real = time.time(); CONV["fits"] = 0; CONV["not_converged"] = 0
        real = {"S1": {}, "S2": {}, "S3": {}}
        ctl4 = {}
        for tag in TAGS:
            P, Pg, Pr, corr = fit_real(tag)
            ctl4[tag] = corr
            for kk, v in eval_s1(tag, P).items():
                real["S1"][kk] = v
            real["S2"].update(eval_s2(tag, P, extras))
            real["S3"].update(eval_s3(tag, Pg, Pr, with_identical=True, raw_l1=raw_l1))
            print("real", ds, tag, round(time.time() - t_real, 1), "s", flush=True)
        d["logistic_real"] = dict(CONV)
        t_real = time.time() - t_real
        # controls 1, 2, 4 (real rows)
        st = [v for k, v in real["S2"].items() if k.endswith("/start")]
        d["controls_raw"]["C1_start_rows"] = len(st)
        d["controls_raw"]["C1_start_undefined_rows"] = int(sum(1 for v in st if v.get("undefined")))
        d["controls_raw"]["C1_start_query_skill_max_abs"] = max(abs(v["point"]) for v in st if not v.get("undefined"))
        d["controls_raw"]["C1_start_query_interval_max_abs"] = max(max(abs(v["lo"]), abs(v["hi"])) for v in st if not v.get("undefined"))
        ids = [real["S3"][k][nm] for k in real["S3"] for nm in ("raw_identical", "partial_identical")]
        d["controls_raw"]["C2_identical_max_abs"] = max(max(abs(v["point"]), abs(v["lo"]), abs(v["hi"])) for v in ids)
        d["controls_raw"]["C4_speed_corr"] = {tag: ctl4[tag] for tag in TAGS}
        d["controls_raw"]["C4_speed_corr_max_abs"] = max(abs(c) for tag in TAGS for c in ctl4[tag].values())
        d["real"] = real
        d["real_stage_s"] = round(t_real, 1)
        if ds == "eval141":                                                # reproduction check against run 1 (same probes, same labels)
            dif = {"S1_point": 0.0, "S1_ends": 0.0, "S2_decision_point": 0.0, "S3_raw_point": 0.0}
            for kk, v in real["S1"].items():
                r1 = ref1["S1"][kk]["main"]
                dif["S1_point"] = max(dif["S1_point"], abs(v["point"] - r1["point"])); dif["S1_ends"] = max(dif["S1_ends"], abs(v["lo"] - r1["lo"]), abs(v["hi"] - r1["hi"]))
            for kk, v in real["S2"].items():
                tag, l, k_, q, nm = kk.split("/")
                if nm == ("shrunk" if l == "L1" else "raw"):
                    dif["S2_decision_point"] = max(dif["S2_decision_point"], abs(v["point"] - ref1["S2"][kk]["point"]))
            for kk, v in real["S3"].items():
                dif["S3_raw_point"] = max(dif["S3_raw_point"], abs(v["raw"]["point"] - ref1["S3"][kk]["decision"]["point"]))
            d["reproduction_vs_run1_max_abs_diff"] = dif
        d["pictures"] = pca_picture(ds, ctx, d, plt)
        res["pictures"].update(d["pictures"])
        bank(f"{ds}: real rows and controls 1, 2, 4 banked")
        # permutation draws
        dpath = f"draws_{dsn}.jsonl"
        done = {}
        if os.path.exists(dpath):
            for ln in open(dpath, encoding="utf-8"):
                if ln.strip():
                    r = json.loads(ln); done[(r["role"], r["s"])] = r
        tasks_all = [(s, "ref") for s in REF_SEEDS] + [(s, "cal") for s in CAL_SEEDS]
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        pool = mp.get_context("spawn").Pool(4, initializer=init_worker, initargs=(ctx,))
        try:
            fh = open(dpath, "a", encoding="utf-8", newline="\n")
            if ds == "train" and not projected_checked:
                first = [t for t in tasks_all[:5] if (t[1], t[0]) not in done]
                if first:
                    for r in pool.map(draw_task, first, chunksize=1):
                        done[(r["role"], r["s"])] = r; fh.write(json.dumps(r) + "\n"); fh.flush()
                tm = [done[("ref", s)]["t"] for s in range(1, 6)]
                proj = 2 * t_real + 2 * (len(tasks_all) * float(np.mean(tm)) / 4.0)
                res["timing_gate"] = {"draw_s": tm, "mean_draw_s": float(np.mean(tm)), "real_stage_s_train": t_real, "projected_total_s": proj, "limit_s": 10800}
                print(f"TIMING 5 reference draws on TRAIN: mean {np.mean(tm):.1f} s per draw; projected total {proj / 60:.0f} min ({proj / 3600:.2f} h) for 2 datasets x {len(tasks_all)} draws on 4 workers", flush=True)
                bank("timing gate")
                if proj > 10800:
                    print("PROJECTION EXCEEDS 3 h: stopping", flush=True); pool.terminate(); return
                projected_checked = True
            todo = [t for t in tasks_all if (t[1], t[0]) not in done]
            t_d = time.time()
            for i, r in enumerate(pool.imap_unordered(draw_task, todo, chunksize=1)):
                done[(r["role"], r["s"])] = r; fh.write(json.dumps(r) + "\n"); fh.flush()
                if (i + 1) % 20 == 0:
                    print(f"{ds} draws {len(done)}/{len(tasks_all)} {time.time() - t_d:.0f} s", flush=True)
            fh.close()
        finally:
            pool.close(); pool.join()
        draws_ref = [done[("ref", s)] for s in REF_SEEDS]; draws_cal = [done[("cal", s)] for s in CAL_SEEDS]
        d["analysis"] = analyse(real, draws_ref, draws_cal)
        d["draws_wall_s"] = round(time.time() - t_start, 1)
        bank(f"{ds}: reference and calibration banked")
        print(ds, "calibration", json.dumps({f: round(v["share_two_part"], 4) for f, v in d["analysis"]["calibration"].items()}), "pass", d["analysis"]["calibration_pass"], flush=True)
        del ctx
    finish(res, ref1)
    bank("final")
    print("done", round(time.time() - t_start, 1), "s", flush=True)


def finish(res, ref1):
    """Controls verdict per dataset and the outcomes of the card."""
    D = res["datasets"]
    ctlsum = {}
    for ds, d in D.items():
        c = d["controls_raw"]; a = d["analysis"]
        s = {"copy_levels": bool(c["copy_maxdiff_z1"] <= 1e-5 and c["copy_maxdiff_actions"] == 0.0 and all(v <= 1e-5 for v in c["level_maxdiff"].values())),
             "C1": bool(c["C1_start_query_skill_max_abs"] == 0.0 and c["C1_start_query_interval_max_abs"] == 0.0 and c["C1_start_undefined_rows"] == 0),
             "C2": bool(c["C2_identical_max_abs"] == 0.0), "C3": bool(a["calibration_pass"]), "C4": bool(c["C4_speed_corr_max_abs"] <= 1e-6)}
        ctlsum[ds] = s
    res["controls"] = ctlsum
    res["controls_pass"] = bool(all(all(s.values()) for s in ctlsum.values()))
    res["controls_failed"] = sorted(f"{ds}:{k}" for ds, s in ctlsum.items() for k, v in s.items() if not v)
    tr, ev = D["train"], D["eval141"]
    tab = {"ENCODES": {"train": {}, "eval141": {}}, "IMAGINES": {"train": {}, "eval141": {}}, "ACTION-SEMANTIC-LEAD": {}, "SCENE-AWARE": {}, "SCENE-IMAGINING": {}}
    both = lambda f: bool(all(f(t) for t in TAGS))
    for dsn, d in (("train", tr), ("eval141", ev)):
        for l in (1, 2, 3):
            for q in ALLQ:
                tab["ENCODES"][dsn][f"L{l}/{q}"] = both(lambda t: d["analysis"]["S1"][f"{t}/L{l}/{q}"]["separated_two_part"])
                for k in KS[l]:
                    nm = "shrunk" if l == 1 else "raw"
                    tab["IMAGINES"][dsn][f"L{l}/k{k}/{q}"] = both(lambda t: d["analysis"]["S2"][f"{t}/L{l}/k{k}/{q}/{nm}"]["separated_two_part"])
    for l in (1, 2, 3):
        nm = "shrunk" if l == 1 else "raw"
        aware = {}
        for q in ("lead_gap", "agents30"):
            aware[q] = {"encodes_train": tab["ENCODES"]["train"][f"L{l}/{q}"], "eval141_points_above_0": both(lambda t: ev["analysis"]["S1"][f"{t}/L{l}/{q}"]["main"]["point"] > 0)}
            aware[q]["holds"] = bool(aware[q]["encodes_train"] and aware[q]["eval141_points_above_0"])
        tab["SCENE-AWARE"][f"L{l}"] = {"by": aware, "verdict": bool(any(v["holds"] for v in aware.values()))}
        im = {"imagines_train": tab["IMAGINES"]["train"][f"L{l}/k{KMAX[l]}/lead_gap"],
              "eval141_points_above_0": both(lambda t: ev["analysis"]["S2"][f"{t}/L{l}/k{KMAX[l]}/lead_gap/{nm}"]["row"]["point"] > 0)}
        im["verdict"] = bool(im["imagines_train"] and im["eval141_points_above_0"])
        tab["SCENE-IMAGINING"][f"L{l}"] = im
        asl = {"separated_train": both(lambda t: tr["analysis"]["S3"][f"{t}/L{l}"]["partial_separated_two_part"]),
               "eval141_points_below_0": both(lambda t: ev["analysis"]["S3"][f"{t}/L{l}"]["partial"]["point"] < 0)}
        asl["verdict"] = bool(asl["separated_train"] and asl["eval141_points_below_0"])
        tab["ACTION-SEMANTIC-LEAD"][f"L{l}"] = asl
    res["outcomes"] = tab if res["controls_pass"] else {"all": "INVALID (control %s failed)" % ", ".join(res["controls_failed"])}
    res["outcomes_if_controls_ignored_NOT_DECISIONS"] = tab
    awl = [l for l in (1, 2, 3) if tab["SCENE-AWARE"][f"L{l}"]["verdict"]]
    iml = [l for l in (1, 2, 3) if tab["SCENE-IMAGINING"][f"L{l}"]["verdict"]]
    res["next_lever_per_card"] = ("INVALID: a control failed" if not res["controls_pass"] else
                                  "no level SCENE-AWARE: MRATE W1, then a scene-target auxiliary (own card)" if not awl else
                                  "SCENE-AWARE at levels %s, SCENE-IMAGINING at %s: " % (awl, iml) + ("; ".join(f"L{l}: the reliance arm on that level" for l in awl if l not in iml) or "every SCENE-AWARE level is also SCENE-IMAGINING"))
    res["run1_beside"] = {"note": "LATSEM2 run 1 (INVALID read): eval-141, one-part rule (bootstrap interval only), no permutation reference", "file": "../latsem2/RESULT_LATSEM2.json"}


if __name__ == "__main__":
    main()
