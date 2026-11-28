"""PREREG_V8F_LATSEM2 (frozen prefix 3,769 bytes, sha256 1325f21c...; IMPL_CHOICES_LATSEM2.md frozen sha256 b17688dc...): does each world-model level encode (S1) and
imagine (S2) the SCENE, and does the imagined lead gap follow the action (S3)? Four controls. Tier T0, CPU only, zero training.
Run from theory/latsem2 with WMACT_CODE_ROOT set. Writes RESULT_LATSEM2.json (banked after S1, after S2 and S3, and at the end) and fig/*.png."""
import hashlib, io, json, os, sys, time, warnings
import numpy as np, torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath("../wmact"))
import wmact_lib as W  # noqa: E402
import labels_latsem2 as LB  # noqa: E402

CARD, CARD_N, CARD_SHA = "../PREREG_V8F_LATSEM2.md", 3769, "1325f21c902967f3c86fdd2f9b56acf3ee28de01648d97496545cf6a130b1751"
IMPL, IMPL_SHA = "IMPL_CHOICES_LATSEM2.md", "b17688dc0fea5c639ab2d44c7088f88cded05ab99a1b75f89272cbb78492e71d"
assert hashlib.sha256(open(CARD, "rb").read()[:CARD_N]).hexdigest() == CARD_SHA, "card prefix changed"
assert hashlib.sha256(open(IMPL, "rb").read()).hexdigest() == IMPL_SHA, "IMPL_CHOICES changed after its freeze"
TAGS = ("M2b_s0", "M2b_s1")
STRIDE, KMAX, HSEC = {1: 1, 2: 2, 3: 4}, {1: 10, 2: 3, 3: 3}, {1: 2.0, 2: 1.2, 3: 2.4}
LAM, F = 10.0, 16
ACT3 = {"brake": (-3.0, 0.0), "accel": (1.5, 0.0)}
ALLQ = LB.TARGETS + LB.BESIDE
CLASSES = {"lead_present": [0.0, 1.0], "turn": [0.0, 1.0], "lon7": [float(c) for c in range(7)]}
QUERIES_BESIDE = ("raw", "oracle", "start")
t0 = time.time()
CONV = {"fits": 0, "not_converged": 0}
res = {"card": {"file": "PREREG_V8F_LATSEM2.md", "prefix_bytes": CARD_N, "sha256": CARD_SHA}, "impl_choices": {"file": IMPL, "sha256": IMPL_SHA},
       "tier": "T0", "controls": {}, "S1": {}, "S2": {}, "S2_permuted": {}, "S3": {}, "uncontrolled_note": "", "stage": "start"}
ctrl = res["controls"]


def bank_json(stage):
    res["stage"] = stage
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open("RESULT_LATSEM2.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)


# ------------------------------------------------------------------------------------------------ data (LATSEM's bank) and labels (labels_latsem2)
B = LB.build_bank()
L, eid, tw, ok, A_raw, wkeep, G, key2b = B["L"], B["eid"], B["tw"], B["ok"], B["A_raw"], B["wkeep"], B["G"], B["key2b"]
occ_w, occ_f, occ_b, bw, bf, beid, bg = B["occ_w"], B["occ_f"], B["occ_b"], B["bw"], B["bf"], B["beid"], B["bg"]
Y, audit = LB.build_labels(beid, bg)
ref = np.load("labels_latsem2.npz")
assert (ref["beid"] == beid.astype("U36")).all() and (ref["bg"] == bg).all()
for q in Y:
    assert np.array_equal(ref[q], Y[q], equal_nan=True), q                       # the labels equal the frozen label arrays
fold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in beid])
res["n_bank_frames"], res["n_episodes"] = int(len(beid)), int(len(set(beid)))
res["n"] = {q: int(np.isfinite(Y[q]).sum()) for q in ALLQ}
uniq, Wt = W.boot_weights(beid)
pos = np.searchsorted(uniq, beid)


def esum(x, p=pos):
    s = np.zeros(len(uniq)); np.add.at(s, p, np.nan_to_num(np.asarray(x, np.float64))); return s


def pct(b):
    b = np.asarray(b); lo, hi = np.nanpercentile(b, [2.5, 97.5]); return float(lo), float(hi)


def row(num_pos, num_neg, den, n, nep):
    ds = den.sum()
    if ds == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "separated": False, "n": int(n), "n_episodes": int(nep), "undefined": True}
    p = float((num_pos.sum() - num_neg.sum()) / ds)
    b = (Wt @ num_pos - Wt @ num_neg) / (Wt @ den)
    lo, hi = pct(b)
    return {"point": p, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n": int(n), "n_episodes": int(nep)}


def bal(N, TP):
    """balanced accuracy over the classes present (N > 0) on the last axis."""
    rec = np.where(N > 0, TP / np.maximum(N, 1e-12), np.nan)
    return np.nanmean(rec, axis=-1)


def bal_row(y, pred, base, val, classes, p=pos, n_ep_data=None):
    """paired difference: balanced accuracy of `pred` minus balanced accuracy of `base`, per-episode sums, bootstrapped."""
    N = np.stack([esum((y == c) & val, p) for c in classes], 1)
    TP = np.stack([esum((y == c) & val & (pred == c), p) for c in classes], 1)
    TB = np.stack([esum((y == c) & val & (base == c), p) for c in classes], 1)
    pm, pb = float(bal(N.sum(0), TP.sum(0))), float(bal(N.sum(0), TB.sum(0)))
    d = bal(Wt @ N, Wt @ TP) - bal(Wt @ N, Wt @ TB)
    lo, hi = pct(d)
    return {"point": pm - pb, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "bal_model": pm, "bal_base": pb, "n": int(val.sum()),
            "n_episodes": int(len(np.unique(beid[val]))) if n_ep_data is None else n_ep_data}


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
    """labels permuted across episodes (LATSEM): each episode takes ANOTHER episode's label sequence, cycled to its length; no episode keeps its own."""
    out = np.empty_like(y); order = np.unique(ep)[rng.permutation(len(np.unique(ep)))]
    for e, e2 in zip(order, np.roll(order, 1)):
        src = y[ep == e2]; n = int((ep == e).sum()); out[ep == e] = src[np.arange(n) % len(src)]
    return out


def s1_row(q, y, pred, base, val):
    if LB.KIND[q] == "cont":
        sst, sse = esum((y - base) ** 2), esum((pred - y) ** 2)
        r = row(sst, sse, sst, val.sum(), len(np.unique(beid[val]))); r["metric"] = "R2"
    else:
        r = bal_row(y, pred, base, val, CLASSES[q]); r["metric"] = "BA_minus_majority"
    return r


# ------------------------------------------------------------------------------------------------ self-test of the new metric code (synthetic; no model, no label)
def selftest():
    rng = np.random.default_rng(123)
    yy = rng.integers(0, 3, size=len(beid)).astype(np.float64); vv = np.ones(len(beid), bool)
    perfect = bal_row(yy, yy.copy(), np.zeros(len(beid)), vv, [0.0, 1.0, 2.0])          # perfect predictor vs constant 0
    const = bal_row(yy, np.zeros(len(beid)), np.zeros(len(beid)), vv, [0.0, 1.0, 2.0])  # the baseline against itself
    out = {"perfect_bal_model": perfect["bal_model"], "perfect_bal_base": perfect["bal_base"], "same_as_base_point": const["point"],
           "same_as_base_interval": [const["lo"], const["hi"]]}
    assert out["perfect_bal_model"] == 1.0 and abs(out["perfect_bal_base"] - 1 / 3) < 1e-12 and out["same_as_base_point"] == 0.0 and const["lo"] == 0.0 == const["hi"]
    return out


res["selftest"] = selftest()

# ------------------------------------------------------------------------------------------------ LATSEM's own copy / per-level controls (re-run, not the card's four)
c1 = 0.0
for tag in TAGS:
    Z = L["Z1"][tag]
    c1 = max(c1, float((Z[occ_w, occ_f] - Z[bw[occ_b], bf[occ_b]]).abs().max()))
ctrl["copy_maxdiff_z1"] = c1
ctrl["copy_maxdiff_actions"] = float((A_raw[occ_w, occ_f] - A_raw[bw[occ_b], bf[occ_b]]).abs().max())

# ------------------------------------------------------------------------------------------------ S1
PROBE, PPERM, BZ, models = {}, {}, {}, {}
res["S1_audit"] = {}
for tag in TAGS:
    m, cfg, ck = W.load_model(W.CKPT[tag]); models[tag] = m
    st = m.wm.stack
    with torch.no_grad():
        zs = st.initial_latents(L["Z1"][tag][bw, bf])
    BZ[tag] = {l + 1: zs[l].double().numpy() for l in range(3)}
    sel_c = wkeep[ok[wkeep].all(1)]
    with torch.no_grad():
        enc = st.encode(L["Z1"][tag][sel_c], m.wm.scale_actions(A_raw[sel_c])[:, :-1])
    c2 = 0.0
    for l in (2, 3):
        zl = enc.z[l - 1].double().numpy()
        for i in range(zl.shape[1]):
            b = np.array([key2b[(eid[w], int(G[w, i * STRIDE[l]]))] for w in sel_c])
            c2 = max(c2, float(np.abs(zl[:, i] - BZ[tag][l][b]).max()))
    ctrl[f"level_maxdiff/{tag}"] = c2
    rng = np.random.default_rng(0)                                               # IMPL section 8.3: created once per seed, consumed level -> target -> fold
    for l in (1, 2, 3):
        X = BZ[tag][l]
        for q in ALLQ:
            y = Y[q]; val = np.isfinite(y); kind = LB.KIND[q]
            pred, base, predp = (np.full(len(y), np.nan) for _ in range(3))
            for k in (0, 1):
                fit, ev = val & (fold != k), val & (fold == k)
                P = fit_probe(kind, X[fit], y[fit]); PROBE[(tag, l, q, k)] = P
                pred[ev], base[ev] = predict(P, X[ev]), P[4]
                Pp = fit_probe(kind, X[fit], perm_labels(y[fit], beid[fit], rng)); PPERM[(tag, l, q, k)] = Pp
                predp[ev] = predict(Pp, X[ev])
            res["S1"][f"{tag}/L{l}/{q}"] = {"main": s1_row(q, y, pred, base, val), "permuted": s1_row(q, y, predp, base, val)}
    print("S1", tag, round(time.time() - t0, 1), "s", flush=True)
ctrl["C3_permuted_probe_any_separated_above_0"] = bool(any(v["permuted"]["lo"] > 0 for v in res["S1"].values()))
ctrl["C3_rows_separated_above_0"] = [k for k, v in res["S1"].items() if v["permuted"]["lo"] > 0]
res["logistic_convergence"] = dict(CONV)
bank_json("S1 banked")
print("C3 any permuted separated above 0:", ctrl["C3_permuted_probe_any_separated_above_0"], ctrl["C3_rows_separated_above_0"], "| logistic fits", CONV, flush=True)

# ------------------------------------------------------------------------------------------------ S2 and S3
sel = wkeep[ok[wkeep].all(1)]
n = len(sel); qe = eid[sel]; qpos = np.searchsorted(uniq, qe)
qfold = np.array([int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 2 for e in qe])
SHR = json.load(open("../wmact/RESULT_WMACT_SHRINK.json", encoding="utf-8"))["rows"]
bidx = np.array([[key2b[(eid[i], int(G[i, f]))] for f in range(F)] for i in sel])      # [n, 16] bank index of every window frame
Yw = {q: Y[q][bidx] for q in ALLQ}                                                     # [n, 16]
m3 = Yw["lead_present"][:, 0] == 1                                                     # S3 windows: a near lead at frame 0
res["n_windows"], res["n_windows_S3"] = int(n), int(m3.sum())
res["n_episodes_S3"] = int(len(np.unique(qe[m3])))
nep3 = res["n_episodes_S3"]


def decode(probes, tag, l, q, Z):
    out = np.empty(len(Z))
    for k in (0, 1):
        mk = qfold == k
        out[mk] = predict(probes[(tag, l, q, k)], Z[mk])
    return out


def err_of(q, p, ystar, valid):
    e = (p - ystar) ** 2 if LB.KIND[q] == "cont" else (p != ystar).astype(np.float64)
    return np.where(valid, e, 0.0)


def s2_cell(q, ystar, p_start, p_query):
    valid = np.isfinite(ystar); nep = len(np.unique(qe[valid]))
    e0 = esum(err_of(q, p_start, ystar, valid), qpos)
    e1 = esum(err_of(q, p_query, ystar, valid), qpos)
    r = row(e0, e1, e0, valid.sum(), nep)
    r["err_start_mean"], r["err_query_mean"] = float(e0.sum() / max(valid.sum(), 1)), float(e1.sum() / max(valid.sum(), 1))
    return r


NNQ = {}
CFZ = {}                                                                                # (tag, action, level) -> decoded-state Z used for S3 (shrunk at L1)
for tag in TAGS:
    wm = models[tag].wm
    with torch.no_grad():
        pk = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(A_raw[sel])[:, :-1])
    for l, ks in W.KS.items():
        z0 = pk["true"][l][:, 0].double().numpy()
        for k in ks:
            zt, zh = pk["true"][l][:, k].double().numpy(), pk["pred"][(l, k)].double().numpy()
            qs = {"raw": zh, "oracle": zt, "start": z0}
            if l == 1:
                qs["shrunk"] = z0 + SHR[f"{tag}/L1/k{k}"]["c_train"] * (zh - z0)
            for q in ALLQ:
                ystar = Yw[q][:, k * STRIDE[l]]
                p0 = decode(PROBE, tag, l, q, z0)
                for nm, Z in qs.items():
                    res["S2"][f"{tag}/L{l}/k{k}/{q}/{nm}"] = s2_cell(q, ystar, p0, decode(PROBE, tag, l, q, Z))
                p0p = decode(PPERM, tag, l, q, z0)
                dec = "shrunk" if l == 1 else "raw"
                res["S2_permuted"][f"{tag}/L{l}/k{k}/{q}/{dec}"] = s2_cell(q, ystar, p0p, decode(PPERM, tag, l, q, qs[dec]))
    # S3: constant raw actions, read at the longest horizon
    with torch.no_grad():
        Zs = {}
        for nm, (a, w) in list(ACT3.items()) + [("brake_twice", ACT3["brake"])]:
            Araw = torch.tensor([a, w], dtype=torch.float32).expand(n, F, 2).contiguous()
            pkc = W.level_pack(wm, L["Z1"][tag][sel], wm.scale_actions(Araw)[:, :-1])
            for l in (1, 2, 3):
                Zs[(nm, l)] = pkc["pred"][(l, KMAX[l])].double().numpy()
    for l in (1, 2, 3):
        z0 = pk["true"][l][:, 0].double().numpy()
        for nm in ("accel", "brake", "brake_twice"):
            Z = Zs[(nm, l)]
            CFZ[(tag, nm, l, "raw")] = Z
            CFZ[(tag, nm, l, "dec")] = z0 + SHR[f"{tag}/L1/k{KMAX[1]}"]["c_train"] * (Z - z0) if l == 1 else Z
    print("S2/S3 rollouts", tag, round(time.time() - t0, 1), "s", flush=True)
bank_json("S1 and S2 banked")


def s3_row(ga, gb):
    d = np.where(m3, ga - gb, 0.0)
    return row(esum(d, qpos), np.zeros(len(uniq)), esum(m3.astype(np.float64), qpos), m3.sum(), nep3)


for tag in TAGS:
    for l in (1, 2, 3):
        pa, pb, pt = (decode(PROBE, tag, l, "lead_gap", CFZ[(tag, nm, l, "dec")]) for nm in ("accel", "brake", "brake_twice"))
        res["S3"][f"{tag}/L{l}"] = {"decision": s3_row(pa, pb), "identical_sequences": s3_row(pt, pb)}
        if l == 1:
            pr = [decode(PROBE, tag, l, "lead_gap", CFZ[(tag, nm, l, "raw")]) for nm in ("accel", "brake")]
            res["S3"][f"{tag}/L{l}"]["raw_beside"] = s3_row(pr[0], pr[1])
        res["S3"][f"{tag}/L{l}"]["decoded_mean"] = {"accel": float(pa[m3].mean()), "brake": float(pb[m3].mean())}

# control 4: lead_gap labels permuted across the fit-fold lead frames before the ridge fit
vg = np.isfinite(Y["lead_gap"])


def shuffled_s3(draw_seed):
    out = {}
    for tag in TAGS:
        rng = np.random.default_rng(draw_seed)
        yp = {}
        for k in (0, 1):                                                          # IMPL 8.4: one draw per (seed, fold), shared across levels
            y_fit = Y["lead_gap"][vg & (fold != k)]
            yp[k] = y_fit[rng.permutation(len(y_fit))]
        probes = {(tag, l, "lead_gap", k): fit_probe("cont", BZ[tag][l][vg & (fold != k)], yp[k]) for l in (1, 2, 3) for k in (0, 1)}
        for l in (1, 2, 3):
            pa, pb = (decode(probes, tag, l, "lead_gap", CFZ[(tag, nm, l, "dec")]) for nm in ("accel", "brake"))
            out[f"{tag}/L{l}"] = s3_row(pa, pb)
    return out


res["S3_shuffled"] = {"primary": shuffled_s3(0)}
ctrl["C4_shuffled_S3_any_separated_below_0"] = bool(any(r["hi"] < 0 for r in res["S3_shuffled"]["primary"].values()))
res["S3_shuffled"]["extra_draws_beside"] = {}
for d in range(1, 21):
    res["S3_shuffled"]["extra_draws_beside"][str(d)] = shuffled_s3(d)
ex = [r for v in res["S3_shuffled"]["extra_draws_beside"].values() for r in v.values()]
res["S3_shuffled"]["extra_summary"] = {"rows": len(ex), "separated_below_0": int(sum(r["hi"] < 0 for r in ex)), "separated_above_0": int(sum(r["lo"] > 0 for r in ex))}

# controls 1 and 2
ctrl["C1_start_query_skill_max_abs"] = max(abs(v["point"]) for k, v in res["S2"].items() if k.endswith("/start"))
ctrl["C1_start_query_interval_max_abs"] = max(max(abs(v["lo"]), abs(v["hi"])) for k, v in res["S2"].items() if k.endswith("/start"))
ctrl["C2_identical_sequences_S3_max_abs"] = max(max(abs(v["identical_sequences"]["point"]), abs(v["identical_sequences"]["lo"]), abs(v["identical_sequences"]["hi"])) for v in res["S3"].values())
ctrl["C3_extra_note"] = "20 extra permutation draws are computed only when control 3 fails"
bank_json("S1, S2, S3 banked")
print("controls", json.dumps({k: v for k, v in ctrl.items() if not isinstance(v, list)}, default=float), flush=True)

# control 3 extra draws (diagnostic of the control's own false-separation rate; deciding nothing, only when the primary control failed)
if ctrl["C3_permuted_probe_any_separated_above_0"]:
    ext = {}
    for tag in TAGS:
        for d in range(1, 21):
            rng = np.random.default_rng(d)
            for l in (1, 2, 3):
                X = BZ[tag][l]
                for q in ALLQ:
                    y = Y[q]; val = np.isfinite(y); kind = LB.KIND[q]
                    base, predp = np.full(len(y), np.nan), np.full(len(y), np.nan)
                    for k in (0, 1):
                        fit, ev = val & (fold != k), val & (fold == k)
                        Pp = fit_probe(kind, X[fit], perm_labels(y[fit], beid[fit], rng))
                        predp[ev] = predict(Pp, X[ev])
                        base[ev] = y[fit].mean() if kind == "cont" else fit_probe_majority(y[fit])      # the baseline is the TRUE fit-fold mean / majority
                    r = s1_row(q, y, predp, base, val)
                    e = ext.setdefault(f"{tag}/L{l}/{q}", {"draws": 0, "above_0": 0, "below_0": 0})
                    e["draws"] += 1; e["above_0"] += int(r["lo"] > 0); e["below_0"] += int(r["hi"] < 0)
        print("C3 extra draws", tag, round(time.time() - t0, 1), "s", flush=True)
    res["C3_extra_draws_beside"] = ext
    bank_json("C3 extra draws banked")

# ------------------------------------------------------------------------------------------------ controls verdict and outcomes
ok_ctrl = bool(ctrl["copy_maxdiff_z1"] <= 1e-5 and ctrl["copy_maxdiff_actions"] == 0.0 and all(v <= 1e-5 for k, v in ctrl.items() if k.startswith("level_maxdiff"))
               and ctrl["C1_start_query_skill_max_abs"] == 0.0 and ctrl["C1_start_query_interval_max_abs"] == 0.0 and ctrl["C2_identical_sequences_S3_max_abs"] == 0.0
               and not ctrl["C3_permuted_probe_any_separated_above_0"] and not ctrl["C4_shuffled_S3_any_separated_below_0"])
res["controls_pass"] = ok_ctrl
failed = [nm for nm, c in (("1", ctrl["C1_start_query_skill_max_abs"] != 0.0 or ctrl["C1_start_query_interval_max_abs"] != 0.0), ("2", ctrl["C2_identical_sequences_S3_max_abs"] != 0.0),
                           ("3", ctrl["C3_permuted_probe_any_separated_above_0"]), ("4", ctrl["C4_shuffled_S3_any_separated_below_0"]),
                           ("copy/level", not (ctrl["copy_maxdiff_z1"] <= 1e-5 and ctrl["copy_maxdiff_actions"] == 0.0 and all(v <= 1e-5 for k, v in ctrl.items() if k.startswith("level_maxdiff")))))
          if c]
res["controls_failed"] = failed
pos_sep = lambda r: bool(r["separated"] and r["point"] > 0)
neg_sep = lambda r: bool(r["separated"] and r["point"] < 0)
both = lambda f: all(f(t) for t in TAGS)
tab = {"ENCODES": {}, "IMAGINES": {}, "ACTION-SEMANTIC-LEAD": {}, "SCENE-AWARE": {}, "SCENE-IMAGINING": {}}
for l in (1, 2, 3):
    for q in ALLQ:
        tab["ENCODES"][f"L{l}/{q}"] = both(lambda t: pos_sep(res["S1"][f"{t}/L{l}/{q}"]["main"]))
        for k in W.KS[l]:
            nm = "shrunk" if l == 1 else "raw"
            tab["IMAGINES"][f"L{l}/k{k}/{q}"] = both(lambda t: pos_sep(res["S2"][f"{t}/L{l}/k{k}/{q}/{nm}"]))
    tab["ACTION-SEMANTIC-LEAD"][f"L{l}"] = both(lambda t: neg_sep(res["S3"][f"{t}/L{l}"]["decision"]))
    tab["SCENE-AWARE"][f"L{l}"] = {"by_lead_gap": tab["ENCODES"][f"L{l}/lead_gap"], "by_agents30": tab["ENCODES"][f"L{l}/agents30"],
                                   "verdict": bool(tab["ENCODES"][f"L{l}/lead_gap"] or tab["ENCODES"][f"L{l}/agents30"])}
    tab["SCENE-IMAGINING"][f"L{l}"] = tab["IMAGINES"][f"L{l}/k{KMAX[l]}/lead_gap"]
res["outcomes"] = tab if ok_ctrl else {"all": "INVALID (control %s failed)" % ", ".join(failed)}
res["outcomes_if_controls_ignored_NOT_DECISIONS"] = tab
aware = [l for l in (1, 2, 3) if tab["SCENE-AWARE"][f"L{l}"]["verdict"]]
imag = [l for l in (1, 2, 3) if tab["SCENE-IMAGINING"][f"L{l}"]]
res["next_lever_per_card"] = ("INVALID: controls failed" if not ok_ctrl else
                              "no level SCENE-AWARE: MRATE W1, then a scene-target auxiliary (own card)" if not aware else
                              "SCENE-AWARE at levels %s, SCENE-IMAGINING at %s: " % (aware, imag) +
                              ("; ".join(f"L{l}: reliance arm on that level" for l in aware if l not in imag) or "every SCENE-AWARE level is also SCENE-IMAGINING"))
bank_json("final numbers")
print("controls_pass", ok_ctrl, "failed", failed)
print(json.dumps({k: v for k, v in tab.items() if k not in ("IMAGINES", "ENCODES")}, indent=0))
print("done numbers", round(time.time() - t0, 1), "s", flush=True)
