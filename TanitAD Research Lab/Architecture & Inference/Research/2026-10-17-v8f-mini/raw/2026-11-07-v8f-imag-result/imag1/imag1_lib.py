"""PREREG_V8F_IMAG1 library (card prefix 9,212 bytes 14846eef...): statistics, probes' use, one draw (real or permuted-label), all dataset-agnostic (a Ctx of arrays).
Tier T0 (a world-model diagnostic). Nothing here loads a model: build_ctx_imag1.py turns the models into the arrays; the synthetic test builds them by hand."""
import hashlib
import warnings
import numpy as np
import imag1_probes as PB

warnings.filterwarnings("ignore", category=RuntimeWarning)
TAGS = ("M2b_s0", "M2b_s1")
LEVELS, KS = (1, 2, 3), (1, 2, 3)
STRIDE, DT = {1: 1, 2: 2, 3: 4}, {1: 0.2, 2: 0.4, 3: 0.8}
TARGETS = ("speed", "accel", "lead_gap", "lead_closing", "ttc", "yaw_rate", "curvature", "lead_present", "lon7", "agents30")
FAMILY = {"speed": "LON", "accel": "LON", "lead_gap": "LON", "lead_closing": "LON", "ttc": "LON", "yaw_rate": "LAT", "curvature": "LAT",
          "lead_present": "TAC", "lon7": "TAC", "agents30": "SCENE"}
KIND = {t: "cont" for t in TARGETS}
KIND.update({"lead_present": "bin", "lon7": "cls"})
CLASSES = {"lead_present": [0.0, 1.0], "lon7": [float(c) for c in range(7)]}
FAST, SLOW = ("accel", "yaw_rate", "lead_closing"), ("speed", "lead_present", "lon7", "agents30")
PROBES = ("ridge", "mlp")
QTS = ("pred", "copy", "oracle", "offset")
CONTR = ("lonp", "lonm", "latp", "latm")
DELTA = {"lon": 2.0, "lat": 0.5}                       # (+ minus -) size of the contrast: 2 x 1.0 m/s^2, 2 x 0.25 rad/s
B_BOOT, R_REF, N_CAL = 2000, 199, 20
REF_SEEDS, CAL_SEEDS = list(range(1, R_REF + 1)), list(range(1001, 1001 + N_CAL))
ENC_ONLY = ("speed", "yaw_rate")                        # the probes IA-3 / IA-6 decode


# ------------------------------------------------------------------------------------------------ bootstrap, sign flips, moments
def boot_weights(uniq, n_boot=B_BOOT, seed=0):
    """Episode-resampling counts [n_boot, n_ep]: the SAME rng calls as wmact_lib.boot_weights / taniteval.ci._draws (seed 0)."""
    rng = np.random.default_rng(seed)
    n = len(uniq)
    W = np.zeros((n_boot, n))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=n, replace=True)
        W[b] = np.bincount(np.searchsorted(uniq, pick), minlength=n)
    return W


def signflips(n_ep, R=R_REF, seed=777):
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=(R, n_ep))


def _ratio(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _nm(r, axis):
    return np.nanmean(r, axis=axis) if r.shape[axis] > 0 else np.nan


class Mom:
    """Per-episode moments of a score: score = nanmean over columns of (sum A / sum D). Continuous: A = den - num, D = den (score = 1 - NMSE).
    Classes: A = TP - TB (model minus majority baseline), D = N per class, B = TB (so balanced accuracy = score + mean(sum B / sum D))."""
    __slots__ = ("A", "D", "B", "n", "nep")

    def __init__(self, A, D, B=None, n=0, nep=0):
        self.A, self.D, self.B, self.n, self.nep = A, D, B, int(n), int(nep)

    def point(self):
        return float(_nm(_ratio(self.A.sum(0), self.D.sum(0)), 0))

    def vec(self, W):
        return _nm(_ratio(W @ self.A, W @ self.D), 1)

    def ba_point(self):
        return self.point() + float(_nm(_ratio(self.B.sum(0), self.D.sum(0)), 0))

    def ba_vec(self, W):
        return self.vec(W) + _nm(_ratio(W @ self.B, W @ self.D), 1)

    def row(self, W):
        p, v = self.point(), self.vec(W)
        lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
        return {"point": p, "lo": float(lo), "hi": float(hi), "n": self.n, "nep": self.nep}

    def diff(self, o):
        return Mom(self.A - o.A, self.D, None, self.n, self.nep)

    def flips(self, S):
        return _nm(_ratio(S @ self.A, self.D.sum(0)[None, :]), 1)


def sep_above(row, thr):
    return bool(np.isfinite(row["point"]) and row["lo"] > 0 and row["point"] > thr)


def sep_below(row, thr):
    return bool(np.isfinite(row["point"]) and row["hi"] < 0 and row["point"] < thr)


def paired_test(m, W, S):
    """A difference of two scores of the same windows (higher = first better): bootstrap row + the paired sign-flip reference (R = 199 flips of whole episodes' differences)."""
    r = m.row(W)
    fl = m.flips(S)
    hi, lo = float(np.nanpercentile(fl, 97.5)), float(np.nanpercentile(fl, 2.5))
    r.update({"thr_hi": hi, "thr_lo": lo, "above": sep_above(r, hi), "below": sep_below(r, lo)})
    return r


def gain_tests(m, W, S, g0s=(0.0, 1.5)):
    """Gain = sum A / sum D. Two-part against g0: bootstrap bound beyond g0 AND the point beyond g0 + the 97.5th (2.5th) percentile of the sign-flipped residuals (A - g0 D)."""
    r = m.row(W)
    out = {"row": r}
    for g0 in g0s:
        Ar = m.A - g0 * m.D
        fl = _nm(_ratio(S @ Ar, m.D.sum(0)[None, :]), 1)
        hi, lo = float(np.nanpercentile(fl, 97.5)), float(np.nanpercentile(fl, 2.5))
        out[f"above_{g0}"] = bool(np.isfinite(r["point"]) and r["lo"] > g0 and r["point"] > g0 + hi)
        out[f"below_{g0}"] = bool(np.isfinite(r["point"]) and r["hi"] < g0 and r["point"] < g0 + lo)
    return out


def mom_cont(y, pred, valid, pos, n_ep, ymean_row):
    den = np.where(valid, (y - ymean_row) ** 2, 0.0)
    num = np.where(valid, (pred - y) ** 2, 0.0)
    A = np.bincount(pos, weights=den - num, minlength=n_ep)[:, None]
    D = np.bincount(pos, weights=den, minlength=n_ep)[:, None]
    return Mom(A, D, None, valid.sum(), len(np.unique(pos[valid])))


def mom_cls(y, pred, base_row, valid, pos, n_ep, classes):
    f = lambda m: np.bincount(pos, weights=(m & valid).astype(float), minlength=n_ep)
    N = np.stack([f(y == c) for c in classes], 1)
    TP = np.stack([f((y == c) & (pred == c)) for c in classes], 1)
    TB = np.stack([f((y == c) & (base_row == c)) for c in classes], 1)
    return Mom(TP - TB, N, TB, valid.sum(), len(np.unique(pos[valid])))


def perm_labels(y, ep, rng):
    """LATSEM2's, unchanged: the fit-fold episodes are put in a random order, each takes the previous episode's label sequence cycled to its length (no episode keeps its own)."""
    out = np.empty_like(y)
    order = np.unique(ep)[rng.permutation(len(np.unique(ep)))]
    for e, e2 in zip(order, np.roll(order, 1)):
        src = y[ep == e2]
        n = int((ep == e).sum())
        out[ep == e] = src[np.arange(n) % len(src)]
    return out


# ------------------------------------------------------------------------------------------------ context derived fields
def finish_ctx(c):
    """c: SimpleNamespace with beid, fold, Y, qe, Yw, tags, BZ, Z0, PRED, ORA, OFF, CON ... -> adds the derived fields."""
    c.uniq = np.unique(c.beid)
    c.n_ep = len(c.uniq)
    c.pos = np.searchsorted(c.uniq, c.beid)
    c.qpos = np.searchsorted(c.uniq, c.qe)
    c.val = {q: np.isfinite(c.Y[q]) for q in TARGETS}
    c.W = boot_weights(c.uniq)
    c.S = signflips(c.n_ep)
    c.inner = np.array([int(hashlib.sha256((e + "|inner").encode()).hexdigest()[:8], 16) % 5 == 0 for e in c.beid])
    c.qfold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in c.qe])
    c.groups = [np.flatnonzero(c.qe == e) for e in c.uniq]
    c.fbase = {(q, k): (PB.majority(c.Y[q][c.val[q] & (c.fold != k)]) if KIND[q] != "cont" else np.nan) for q in TARGETS for k in (0, 1)}
    return c


# ------------------------------------------------------------------------------------------------ fitting
def fit_tag(c, tag, s):
    """Probes of one model for all (level, fold). s = 0: real labels; s > 0: labels permuted across episodes inside the fit fold (rng = default_rng(s), consumed
    level -> target -> fold, as LATSEM2B). Returns PR[(l, k)] = {'lin': {q: probe}, 'mlp': MLPBank, 'mu', 'sd'}; fold k's probe is fitted on fold != k."""
    rng = np.random.default_rng(s) if s > 0 else None
    PR = {}
    ti = TAGS.index(tag) if tag in TAGS else 0
    for l in LEVELS:
        X = np.asarray(c.BZ[tag][l], np.float64)
        Yf = {0: {}, 1: {}}
        for q in TARGETS:
            for k in (0, 1):
                fit = c.val[q] & (c.fold != k)
                y = c.Y[q][fit]
                if rng is not None:
                    y = perm_labels(y, c.beid[fit], rng)
                Yf[k][q] = (fit, y)
        for k in (0, 1):
            rows = c.fold != k
            lin = {q: PB.fit_lin(KIND[q], X[Yf[k][q][0]], Yf[k][q][1]) for q in TARGETS}
            mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-8
            at = np.flatnonzero(rows)
            ys = {}
            for q in TARGETS:
                full = np.full(len(rows), np.nan)
                full[Yf[k][q][0]] = Yf[k][q][1]
                ys[q] = full[rows]
            mlp = PB.MLPBank(TARGETS, KIND, X.shape[1], seed=s * 1000 + ti * 100 + l * 10 + k).fit((X[rows] - mu) / sd, ys, c.inner[rows])
            PR[(l, k)] = {"lin": lin, "mlp": mlp, "mu": mu, "sd": sd}
    return PR


def decode(PR, P, l, Z, rowfold, only=None):
    """{q: prediction [n]} of probe kind P for rows ``Z`` whose fold is ``rowfold``: fold k rows go through the probe fitted on the other fold."""
    Z = np.asarray(Z, np.float64)
    qs = TARGETS if only is None else only
    out = {q: np.full(len(Z), np.nan) for q in qs}
    for k in (0, 1):
        m = rowfold == k
        if not m.any():
            continue
        if P == "ridge":
            for q in qs:
                out[q][m] = PB.predict_lin(PR[(l, k)]["lin"][q], Z[m])
        else:
            r = PR[(l, k)]
            pr = r["mlp"].predict((Z[m] - r["mu"]) / r["sd"], only=qs)
            for q in qs:
                out[q][m] = pr[q]
    return out


# ------------------------------------------------------------------------------------------------ scoring
def _moms(c, q, y, pred, valid, pos, rowfold):
    n_ep = c.n_ep
    if KIND[q] == "cont":
        ym = np.array([y[valid & (rowfold == k)].mean() if (valid & (rowfold == k)).any() else 0.0 for k in (0, 1)])
        return mom_cont(y, pred, valid, pos, n_ep, ym[rowfold])
    base = np.where(rowfold == 0, c.fbase[(q, 0)], c.fbase[(q, 1)])
    return mom_cls(y, pred, base, valid, pos, n_ep, CLASSES[q])


def score_ia1(c, tag, PR, P, Zbank=None):
    out = {}
    for l in LEVELS:
        Z = c.BZ[tag][l] if Zbank is None else Zbank[l]
        d = decode(PR, P, l, Z, c.fold)
        for q in TARGETS:
            out[(l, q)] = _moms(c, q, c.Y[q], d[q], c.val[q], c.pos, c.fold)
    return out


def score_windows(c, tag, PR, P, with_contrast=True):
    """IA-2 (pred / copy / oracle / offset per level, k, target), IA-3 effect moments and IA-6 decoded speed changes for one probe kind."""
    ia2, ia3, dsp = {}, {}, {}
    nw = len(c.qe)
    for l in LEVELS:
        dec = {"copy": decode(PR, P, l, c.Z0[tag][l], c.qfold)}
        for k in KS:
            dec[("pred", k)] = decode(PR, P, l, c.PRED[tag][(l, k)], c.qfold)
            dec[("oracle", k)] = decode(PR, P, l, c.ORA[tag][(l, k)], c.qfold)
            dec[("offset", k)] = decode(PR, P, l, c.OFF[tag][(l, k)], c.qfold)
        for k in KS:
            f = k * STRIDE[l]
            for q in TARGETS:
                y = c.Yw[q][:, f]
                valid = np.isfinite(y)
                for qt in QTS:
                    p = dec["copy"][q] if qt == "copy" else dec[(qt, k)][q]
                    ia2[(l, k, q, qt)] = _moms(c, q, y, p, valid, c.qpos, c.qfold)
            dsp[(l, k)] = dec[("pred", k)]["speed"] - dec["copy"]["speed"]
        if with_contrast:
            for k in KS:
                cd = {cn: decode(PR, P, l, c.CON[tag][(cn, l, k)], c.qfold, only=ENC_ONLY) for cn in CONTR}
                for tq in ENC_ONLY:
                    for cn in CONTR:
                        ia3[(l, k, "mean", cn, tq)] = float(cd[cn][tq].mean())
                    ia3[(l, k, "mean", "logged", tq)] = float(dec[("pred", k)][tq].mean())
                for tq, tn in (("speed", "s"), ("yaw_rate", "y")):
                    for ch, (a, b) in (("lon", ("lonp", "lonm")), ("lat", ("latp", "latm"))):
                        d = cd[a][tq] - cd[b][tq]
                        A = np.bincount(c.qpos, weights=d, minlength=c.n_ep)[:, None]
                        cnt = np.bincount(c.qpos, minlength=c.n_ep)[:, None].astype(float)
                        ia3[(l, k, tn + ch)] = (A, cnt)
    return ia2, ia3, dsp


def ia3_moms(ia3, l, k):
    """Gain moments (diagonal entries, physical effect in the denominator) and the four Jacobian entries per unit contrast."""
    h = k * DT[l]
    A_sl, n_sl = ia3[(l, k, "slon")]
    A_yt, n_yt = ia3[(l, k, "ylat")]
    return {"gain_lon": Mom(A_sl, n_sl * DELTA["lon"] * h, None, n_sl.sum(), (n_sl > 0).sum()),
            "gain_lat": Mom(A_yt, n_yt * DELTA["lat"] * 1.0, None, n_yt.sum(), (n_yt > 0).sum()),
            "J": {"s_lon": Mom(*[ia3[(l, k, "slon")][0], ia3[(l, k, "slon")][1] * DELTA["lon"]]), "s_lat": Mom(ia3[(l, k, "slat")][0], ia3[(l, k, "slat")][1] * DELTA["lat"]),
                  "y_lon": Mom(ia3[(l, k, "ylon")][0], ia3[(l, k, "ylon")][1] * DELTA["lon"]), "y_lat": Mom(ia3[(l, k, "ylat")][0], ia3[(l, k, "ylat")][1] * DELTA["lat"])}}


def cosine(J, h):
    """cosine between the imagined 2 x 2 Jacobian (rows speed, yaw rate; columns longitudinal, lateral; per unit contrast) and the physical [[h, 0], [0, 1]]."""
    a = np.stack([J["s_lon"], J["s_lat"], J["y_lon"], J["y_lat"]], -1)
    p = np.array([h, 0.0, 0.0, 1.0])
    return (a @ p) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(p) + 1e-300)


def jac_boot(J, W):
    return {k: m.vec(W) for k, m in J.items()}


# ------------------------------------------------------------------------------------------------ IA-0
def rankme(Z, centered=False):
    Z = np.asarray(Z, np.float64)
    if centered:
        Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / s.sum() + 1e-7
    return float(np.exp(-(p * np.log(p)).sum()))


def mrr_mom(c, pred, ora, rng=None):
    """Retrieval of the encoded true future (k = 1) among all windows of the same clip, query = the predicted latent. rng None: the window's own future is the answer;
    else the answer is a random other window of the clip (a derangement), the permutation reference. Returns (Mom of excess MRR, mrr, chance)."""
    rr, ch = np.zeros(len(pred)), np.zeros(len(pred))
    for g in c.groups:
        n = len(g)
        if n < 2:
            rr[g], ch[g] = np.nan, np.nan
            continue
        d = ((pred[g][:, None, :] - ora[g][None, :, :]) ** 2).sum(-1)
        ans = np.arange(n)
        if rng is not None:
            perm = rng.permutation(n)
            ans = np.empty(n, int)
            ans[perm] = np.roll(perm, 1)
        di = d[np.arange(n), ans]
        rank = 1 + (d < di[:, None]).sum(1) + 0.5 * ((d == di[:, None]).sum(1) - 1)
        rr[g] = 1.0 / rank
        ch[g] = (1.0 / np.arange(1, n + 1)).sum() / n
    ok = np.isfinite(rr)
    A = np.bincount(c.qpos[ok], weights=(rr - ch)[ok], minlength=c.n_ep)[:, None]
    D = np.bincount(c.qpos[ok], minlength=c.n_ep)[:, None].astype(float)
    return Mom(A, D, None, ok.sum(), len(np.unique(c.qpos[ok]))), float(rr[ok].mean()), float(ch[ok].mean())


# ------------------------------------------------------------------------------------------------ one draw
def run_draw(c, s, role="ref"):
    """s = 0: the real labels (full output). s > 0: permuted labels, tag by tag; output compressed to what the references and the calibration need."""
    res = {"s": s, "role": role, "tags": {}}
    PB.CONV["fits"] = 0
    PB.CONV["not_converged"] = 0
    rngm = np.random.default_rng(10_000 + s)
    for tag in c.tags:
        PR = fit_tag(c, tag, s)
        t = {"ia1": {}, "ia2": {}, "ia3": {}, "dsp": {}}
        for P in PROBES:
            t["ia1"][P] = score_ia1(c, tag, PR, P)
            t["ia2"][P], t["ia3"][P], t["dsp"][P] = score_windows(c, tag, PR, P, with_contrast=True)
        t["ia0"] = {}
        for l in LEVELS:
            m, mrr, ch = mrr_mom(c, c.PRED[tag][(l, 1)], c.ORA[tag][(l, 1)], rng=None if s == 0 else rngm)
            t["ia0"][l] = {"mom": m, "mrr": mrr, "chance": ch}
        if s == 0:
            t["rankme"] = {l: {"enc": rankme(c.BZ[tag][l]), "enc_c": rankme(c.BZ[tag][l], True), "pred_k1": rankme(c.PRED[tag][(l, 1)]), "pred_k1_c": rankme(c.PRED[tag][(l, 1)], True),
                               "oracle_k1": rankme(c.ORA[tag][(l, 1)]), "oracle_k1_c": rankme(c.ORA[tag][(l, 1)], True)} for l in LEVELS}
            t["insample"] = insample_control(c, tag, PR)
        res["tags"][tag] = t
    if s == 0 and getattr(c, "rand_tags", None):
        res["rand"] = {}
        for rt in c.rand_tags:
            PRr = fit_tag(c, rt, 0)
            res["rand"][rt] = {P: score_ia1(c, rt, PRr, P) for P in PROBES}
    res["conv"] = dict(PB.CONV)
    return res


def insample_control(c, tag, PR):
    """Control 1 (reported): the NMSE of each continuous probe on its own fit-fold encoded latents against the scoring-fold NMSE of the same probe."""
    out = {}
    for P in PROBES:
        for l in LEVELS:
            X = c.BZ[tag][l]
            for k in (0, 1):
                dec_in = decode(PR, P, l, X[c.fold != k], np.full(int((c.fold != k).sum()), k))
                dec_out = decode(PR, P, l, X[c.fold == k], np.full(int((c.fold == k).sum()), k))
                for q in TARGETS:
                    if KIND[q] != "cont":
                        continue
                    r = {}
                    for nm, d, m in (("in", dec_in, c.fold != k), ("out", dec_out, c.fold == k)):
                        v = c.val[q][m]
                        y = c.Y[q][m][v]
                        r[nm] = float(((d[q][v] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if v.sum() > 2 else np.nan
                    out[(P, l, k, q)] = r
    return out
