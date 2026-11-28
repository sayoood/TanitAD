"""s0p_fit.py -- S0-P step 3: fit the probes on the TRAIN sample, score ONCE on eval-141, read the outcomes (card PREREG_V8F_STOPPERC.md section 4 "S0-P" + section 8; AMENDMENT_1_S0.md).  CPU only.

Stages (each resumable: a stage whose output exists is skipped; every stage prints and banks before the next):
  prep   PCA per grid (v40, v160, mean) fit on the TRAIN sample only (label-free: one fit on the whole sample, dims = the first 16 / 32 / 64 components), projections of train and eval -> <work>/proj_*.npy
  fit    --arm {E,E-limit,M+E,V40+E,V160+E} --model {linear,mlp}: 5-fold GroupKFold by clip over the grid, mean fold AUROC, ties to the smaller model, refit on the whole sample, score eval once -> <work>/fit_<arm>_<model>.{json,npy}
  read   the controls, the bootstrap (taniteval.ci.stat_cluster_bootstrap, B 2,000, seed 0, paired differences on the SAME resample) and the outcomes -> RESULT_S0P.json
  all    prep, every fit, read (use for the smoke; the real run launches the fit stages as separate detached jobs)

Readings fixed here where the card is silent (also in README_S0P.md): PCA is one label-free fit per grid on the whole train sample, shared by all tokens of that grid (the CV selects the dimension, not the PCA);
every feature (PCA outputs and E) is z-scored on the fold's training part inside the CV and on the whole sample in the refit; E / E-limit arms have no PCA dimension; a model's feature vector is [token PCA features | E].
"""
import argparse
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
THEORY = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import s0p_lib as L  # noqa: E402

ARMS = ("E", "E-limit", "M+E", "V40+E", "V160+E")
MODELS = ("linear", "mlp")
DIMS = (16, 32, 64)
LIN_C = (0.001, 0.01, 0.1, 1.0)
MLP_H = (64, 256)
MLP_ALPHA = (1e-4, 1e-2)
NB, BSEED = 2000, 0
RES_MIN, HU_MIN = 0.02, 0.037
STOPDEC_OUT = os.path.join(THEORY, "stopbal", "read", "out", "stopdec")
T_TAGS = ("BAL-NULL_s0", "BAL-NULL_s1")
GRID_OF = {"M+E": "mean", "V40+E": "v40", "V160+E": "v160"}
FEAT_COLS = {"mean": 1, "v40": 40, "v160": 160}


# ============================================================================================ pure logic (unit-tested)
def config_grid(arm, model):
    """The fixed grids, in the order the tie-break prefers (smaller first). -> list of dicts."""
    dims = (None,) if arm in ("E", "E-limit") else DIMS
    if model == "linear":
        return [{"dim": d, "C": c} for d in dims for c in LIN_C]
    return [{"dim": d, "hidden": h, "alpha": a} for d in dims for h in MLP_H for a in sorted(MLP_ALPHA, reverse=True)]


def complexity_key(cfg):
    """Smaller = preferred on a tie: smaller PCA dim, then (linear) smaller C = stronger regularisation, (mlp) smaller hidden, then larger alpha."""
    d = -1 if cfg["dim"] is None else cfg["dim"]
    if "C" in cfg:
        return (d, cfg["C"])
    return (d, cfg["hidden"], -cfg["alpha"])


def select_best(cv_means):
    """cv_means: list of (cfg, mean_fold_auroc). Max mean AUROC; exact ties (|diff| < 1e-12) go to the smaller model."""
    best = max(m for _, m in cv_means)
    tied = [(cfg, m) for cfg, m in cv_means if best - m < 1e-12]
    return min(tied, key=lambda cm: complexity_key(cm[0]))[0]


def episode_permuted_labels(y, clip, seed):
    """Episode-shuffled labels (the card's control): the label SEQUENCE of each clip is replaced by that of another clip (a random derangement-free permutation of clips), cycled / truncated to the clip's own length.
    Keeps every clip's within-clip prevalence structure; destroys any link between a clip's frames and its labels.  Windows are taken in their array order within a clip."""
    y, clip = np.asarray(y), np.asarray(clip)
    rng = np.random.default_rng(seed)
    uc = np.array(sorted(set(clip.tolist())))
    perm = rng.permutation(len(uc))
    if len(uc) > 1:
        while (perm == np.arange(len(uc))).any():                # no clip keeps its own labels
            perm = rng.permutation(len(uc))
    out = np.empty_like(y)
    seqs = {c: y[clip == c] for c in uc}
    for k, c in enumerate(uc):
        src = seqs[uc[perm[k]]]
        n = int((clip == c).sum())
        out[clip == c] = np.resize(src, n)
    return out


def decide_outcomes(s):
    """The card's outcome table for stage 0 (CEILING-FAR is S0-C's).  `s` carries ONLY the paired reads:
      s["res"][model]            = {"point": dAUC(V160+E - V40+E), "lo": CI lower bound}                      model in linear / mlp
      s["hu"][model][t_tag]      = {"diff": V40+E AUROC - T AUROC (paired point), "probe_lo": V40+E AUROC CI lower bound, "t_point": T AUROC point, "diff_lo": paired-diff CI lower bound}   t_tag in BAL-NULL_s0/_s1
    RES-ADDS: paired dAUC >= +0.02 and CI lo > 0, linear OR mlp (within one family).
    HEAD-UNDERUSES (AMENDMENT_2_S0P.md): the PAIRED dAUC (V40+E probe - T, same windows, same resample) >= 0.037 AND its CI lower bound > 0, at BOTH T arms (BAL-NULL_s0 and _s1), linear OR mlp.
    The probe's own CI lower bound vs T's point (`probe_lo`, `t_point`) is reported beside and decides nothing.
    FLAT: neither.  Returns {"RES-ADDS": bool, "HEAD-UNDERUSES": bool, "FLAT": bool, "res_models": [...], "hu_models": [...]}."""
    res_models = [m for m, r in s["res"].items() if r["point"] >= RES_MIN and r["lo"] > 0]
    hu_models = []
    for m, per_t in s["hu"].items():
        if len(per_t) == len(T_TAGS) and all(v["diff"] >= HU_MIN and v["diff_lo"] > 0 for v in per_t.values()):
            hu_models.append(m)
    res, hu = bool(res_models), bool(hu_models)
    return {"RES-ADDS": res, "HEAD-UNDERUSES": hu, "FLAT": not (res or hu), "res_models": sorted(res_models), "hu_models": sorted(hu_models)}


# ============================================================================================ io
class _ShardCat:
    """Read-only lazy concatenation along axis 0 of memory-mapped shard arrays: len, shape, dtype, ndim, int / slice indexing (all the fit code does to a grid array).  Nothing is copied or held in RAM."""

    def __init__(self, parts):
        self.parts = list(parts)
        self.cum = np.concatenate([[0], np.cumsum([len(p) for p in self.parts])]).astype(np.int64)
        self.shape = (int(self.cum[-1]),) + tuple(self.parts[0].shape[1:])
        self.dtype, self.ndim = self.parts[0].dtype, len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            i = int(idx) + (self.shape[0] if idx < 0 else 0)
            k = int(np.searchsorted(self.cum, i, side="right")) - 1
            return self.parts[k][i - int(self.cum[k])]
        assert isinstance(idx, slice) and idx.step in (None, 1), "_ShardCat supports int and unit-step slices only"
        lo, hi, _ = idx.indices(self.shape[0])
        out = [p[max(lo, int(c0)) - int(c0):min(hi, int(c0) + len(p)) - int(c0)] for p, c0 in zip(self.parts, self.cum[:-1]) if max(lo, int(c0)) < min(hi, int(c0) + len(p))]
        return np.concatenate(out) if out else self.parts[0][0:0]


def shard_dirs(path):
    """The shard_<i> sub-directories of a sharded train dir, in numeric order (empty list if it is not one)."""
    subs = [d for d in os.listdir(path) if d.startswith("shard_") and d[6:].isdigit() and os.path.isdir(os.path.join(path, d))]
    return [os.path.join(path, d) for d in sorted(subs, key=lambda d: int(d[6:]))]


def _load_dir(path):
    d = {}
    for k in L.FIELDS:
        d[k] = np.load(os.path.join(path, k + ".npy"), mmap_mode="r" if k in ("v40", "v160", "mean") else None)
    d["meta"] = json.load(open(os.path.join(path, "meta.json"), encoding="utf-8"))
    return d


def load_any(path):
    """A npz file (eval), a directory of .npy + meta.json (train side, written by s0p_extract_train.py), or a directory of shard_<i>/ such directories (--shard i/K): the shards are concatenated in order.
    Directory arrays are memory-mapped (shard grids lazily, never copied)."""
    if os.path.isdir(path):
        sd = shard_dirs(path)
        if not sd:
            return _load_dir(path)
        parts = [_load_dir(p) for p in sd]
        ks = {m["meta"].get("shard", {}).get("k") for m in parts}
        assert len(ks) == 1 and None not in ks and ks.pop() == len(parts), f"{path}: shard dirs {len(parts)} do not match the K in their meta (a shard is missing or extra)"
        assert [m["meta"]["shard"]["i"] for m in parts] == list(range(len(parts))), f"{path}: shard indices are not 0..K-1"
        assert len({m["meta"]["sample_sha256"] for m in parts}) == 1, f"{path}: shards come from different samples"
        d = {}
        for k in L.FIELDS:
            d[k] = _ShardCat([m[k] for m in parts]) if k in ("v40", "v160", "mean") else np.concatenate([m[k] for m in parts])
        meta = dict(parts[0]["meta"])
        c2 = [m["meta"].get("c2_second_route_max_abs", {}) for m in parts]
        meta["c2_second_route_max_abs"] = {k: max(c[k] for c in c2) for k in c2[0]} if c2 and c2[0] else {}
        meta["n_sample"] = int(len(d["y"]))
        meta["causal_limit_ok_false_fraction"] = sum(m["meta"]["causal_limit_ok_false_fraction"] * len(m["y"]) for m in parts) / len(d["y"])
        meta["shards"] = len(parts)
        d["meta"] = meta
        return d
    return L.load_npz(path)


def _decode_chunk(u8):
    return L.decode_fp8(u8) if u8.dtype == np.uint8 else np.asarray(u8, np.float32)


def fit_pca(arr, n_comp=64, chunk=128):
    """Streaming covariance PCA over all tokens of arr [N, T, 1024] (uint8 fp8 bytes) or [N, 1024] (fp16).  Returns mean [1024] float64, comps [1024, n_comp] float64, eigvals."""
    N = len(arr)
    s = np.zeros(1024)
    ss = np.zeros((1024, 1024))
    n = 0
    shift = None
    for i in range(0, N, chunk):
        x = _decode_chunk(np.ascontiguousarray(arr[i:i + chunk]))
        x = x.reshape(-1, 1024)
        if shift is None:
            shift = x.mean(0)
        x = x - shift.astype(np.float32)
        s += x.sum(0, dtype=np.float64)
        ss += (x.T @ x).astype(np.float64)
        n += len(x)
    mu = s / n
    cov = ss / n - np.outer(mu, mu)
    w, v = np.linalg.eigh(cov)
    o = np.argsort(w)[::-1][:n_comp]
    return {"mean": mu + shift.astype(np.float64), "comps": v[:, o], "eig": w[o], "n_tokens": n}


def project(arr, pca, chunk=128):
    """-> float32 [N, T, n_comp] (T = 1 for the mean arrays)."""
    N = len(arr)
    nc = pca["comps"].shape[1]
    mu, W = pca["mean"].astype(np.float32), pca["comps"].astype(np.float32)
    out = None
    for i in range(0, N, chunk):
        x = _decode_chunk(np.ascontiguousarray(arr[i:i + chunk]))
        if x.ndim == 2:
            x = x[:, None, :]
        p = (x - mu) @ W
        if out is None:
            out = np.empty((N, x.shape[1], nc), np.float32)
        out[i:i + len(p)] = p
    return out


def build_features(arm, P, E, dim):
    """P: projected [N, T, 64] for the arm's grid (or None); E [N, 18]. -> float32 [N, F]."""
    Ev = E[:, :L.E_NO_LIMIT] if arm == "E-limit" else E
    if arm in ("E", "E-limit"):
        return np.asarray(Ev, np.float32)
    return np.concatenate([P[:, :, :dim].reshape(len(P), -1), Ev], 1).astype(np.float32)


def _make_model(model, cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "linear":
        clf = LogisticRegression(solver="lbfgs", C=cfg["C"], max_iter=3000)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(cfg["hidden"],), alpha=cfg["alpha"], max_iter=300, random_state=0)
    return make_pipeline(StandardScaler(), clf)


def _score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def cv_select(arm, model, feats_by_dim, y, clip, log=print):
    """feats_by_dim: {dim: X [N, F]}.  Returns (best cfg, cv table list of {cfg, fold_auroc, mean})."""
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups=clip))
    table = []
    for cfg in config_grid(arm, model):
        X = feats_by_dim[cfg["dim"]]
        aucs = []
        for tr, va in folds:
            m = _make_model(model, cfg).fit(X[tr], y[tr])
            aucs.append(L.auroc(_score(m, X[va]), y[va].astype(bool)))
        table.append({"cfg": cfg, "fold_auroc": aucs, "mean": float(np.mean(aucs))})
        log(f"    cv {arm}/{model} {cfg} mean {np.mean(aucs):.4f}")
    best = select_best([(t["cfg"], t["mean"]) for t in table])
    return best, table


def fit_one(arm, model, work, tr, ev, log=print, tag=None):
    """CV on the train sample, refit, score eval once.  Banks <work>/fit_<arm>_<model>.json + .npy (eval scores)."""
    tag = tag or f"{arm}_{model}"
    jp, sp = os.path.join(work, f"fit_{tag}.json"), os.path.join(work, f"fit_{tag}.npy")
    if os.path.exists(jp) and os.path.exists(sp):
        log(f"  skip {tag} (banked)")
        return json.load(open(jp, encoding="utf-8"))
    g = GRID_OF.get(arm)
    Ptr = np.load(os.path.join(work, f"proj_train_{g}.npy"), mmap_mode="r") if g else None
    Pev = np.load(os.path.join(work, f"proj_eval_{g}.npy"), mmap_mode="r") if g else None
    dims = (None,) if g is None else DIMS
    y, clip = np.asarray(tr["y"], np.int64), np.asarray(tr["clip"])
    Xtr = {d: build_features(arm, Ptr, np.asarray(tr["E"]), d) for d in dims}
    Xev = {d: build_features(arm, Pev, np.asarray(ev["E"]), d) for d in dims}
    t0 = time.time()
    best, table = cv_select(arm, model, Xtr, y, clip, log)
    m = _make_model(model, best).fit(Xtr[best["dim"]], y)
    s = _score(m, Xev[best["dim"]])
    auc = L.auroc(s, np.asarray(ev["y"], bool))
    rec = {"arm": arm, "model": model, "best": best, "cv": table, "eval_auroc_point": auc, "n_train": int(len(y)), "n_train_pos": int(y.sum()), "n_eval": int(len(s)), "n_eval_pos": int(np.asarray(ev["y"]).sum()),
           "n_features": int(Xtr[best["dim"]].shape[1]), "elapsed_s": round(time.time() - t0, 1)}
    np.save(sp, s.astype(np.float64))
    json.dump(rec, open(jp, "w", encoding="utf-8"), indent=1)
    log(f"  fit {tag}: best {best}, eval AUROC {auc:.4f} ({rec['elapsed_s']} s)")
    return rec


def prep(work, tr, ev, log=print):
    os.makedirs(work, exist_ok=True)
    for g in ("mean", "v40", "v160"):
        pp = os.path.join(work, f"pca_{g}.npz")
        if not os.path.exists(os.path.join(work, f"proj_eval_{g}.npy")):
            t0 = time.time()
            pca = fit_pca(tr[g])
            np.savez(pp, mean=pca["mean"], comps=pca["comps"], eig=pca["eig"], n_tokens=pca["n_tokens"])
            np.save(os.path.join(work, f"proj_train_{g}.npy"), project(tr[g], pca))
            np.save(os.path.join(work, f"proj_eval_{g}.npy"), project(ev[g], pca))
            log(f"  pca {g}: {pca['n_tokens']} tokens, top-64 variance {float(pca['eig'].sum()):.2f} ({time.time() - t0:.0f} s)")


# ============================================================================================ controls + read
def t_scores(ev):
    """BAL-NULL T's own p(STOP_POINT) on the eval windows, from the STOPDEC npz, aligned by (clip_index, ws) over `has` windows."""
    out = {}
    key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(ev["ci"], ev["ws"]))}
    for tag in T_TAGS:
        z = np.load(os.path.join(STOPDEC_OUT, f"stopdec_{tag}.npz"), allow_pickle=False)
        has = z["has"].astype(bool)
        s = np.full(len(ev["ci"]), np.nan)
        lab = np.zeros(len(ev["ci"]), bool)
        for c, w, p, h in zip(z["clip_index"][has], z["ws"][has], z["goal_prob"][has, L.GOAL_STOP_BIT], z["stop_hit"][has]):
            i = key.get((int(c), int(w)))
            if i is not None:
                s[i], lab[i] = p, bool(h)
        assert np.isfinite(s).all(), f"{tag}: STOPDEC lacks some eval windows"
        assert (lab == np.asarray(ev["y"], bool)).all(), f"{tag}: STOPDEC labels differ from the eval arrays"
        out[tag] = s
    return out


def controls(work, tr, ev, log=print):
    """Label shuffle, episode-shuffled labels, leak, disjointness, the E-equals-dump flags.  Linear, the REAL fit's selected hyperparameters reused (declared; not reselected)."""
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    out = {}
    out["clip_ids_shared_train_eval"] = len(set(map(str, tr["clip"])) & set(map(str, ev["clip"])))
    out["eval_meta_c1_max_abs_diff_E_speed_vs_dump_v0"] = ev["meta"].get("c1_E_speed_vs_dump_v0", {}).get("max_abs_diff")
    out["eval_meta_c2_second_route_max_abs"] = ev["meta"].get("c2_second_route_max_abs")
    out["train_meta_c2_second_route_max_abs"] = tr["meta"].get("c2_second_route_max_abs")
    y, clip = np.asarray(tr["y"], np.int64), np.asarray(tr["clip"])
    yev = np.asarray(ev["y"], bool)
    arms = {}
    for arm in ("E", "V40+E"):
        try:
            rec = json.load(open(os.path.join(work, f"fit_{arm}_linear.json"), encoding="utf-8"))
        except FileNotFoundError:
            continue
        cfg = rec["best"]
        g = GRID_OF.get(arm)
        Ptr = np.load(os.path.join(work, f"proj_train_{g}.npy"), mmap_mode="r") if g else None
        Pev = np.load(os.path.join(work, f"proj_eval_{g}.npy"), mmap_mode="r") if g else None
        Xtr, Xev = build_features(arm, Ptr, np.asarray(tr["E"]), cfg["dim"]), build_features(arm, Pev, np.asarray(ev["E"]), cfg["dim"])
        rng = np.random.default_rng(1)
        arms[arm] = {
            "label_shuffle_scores": _score(_make_model("linear", cfg).fit(Xtr, rng.permutation(y)), Xev),
            "episode_shuffled_scores": _score(_make_model("linear", cfg).fit(Xtr, episode_permuted_labels(y, clip, 2)), Xev)}
        # leak: the GT label appended as a feature on BOTH sides (E arm only)
        if arm == "E":
            Ltr, Lev = np.concatenate([Xtr, y[:, None].astype(np.float32)], 1), np.concatenate([Xev, yev[:, None].astype(np.float32)], 1)
            arms[arm]["leak_scores"] = _score(_make_model("linear", cfg).fit(Ltr, y), Lev)
    return out, arms


def read_all(work, tr, ev, out_json, log=print):
    sys.path.insert(0, os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree"))
    L.setup_tree(os.environ.get("S0P_TREE", "F:/TanitAD-scratch/stopbal_read/tree"))
    from taniteval import ci as CI
    yev = np.asarray(ev["y"], bool)
    eid = np.asarray(ev["eid"], np.int64)
    S = {}
    fits = {}
    for arm in ARMS:
        for model in MODELS:
            p = os.path.join(work, f"fit_{arm}_{model}")
            if os.path.exists(p + ".npy"):
                S[(arm, model)] = np.load(p + ".npy")
                fits[f"{arm}|{model}"] = json.load(open(p + ".json", encoding="utf-8"))
    T = t_scores(ev) if os.path.exists(os.path.join(STOPDEC_OUT, f"stopdec_{T_TAGS[0]}.npz")) else {}
    ctl, ctl_scores = controls(work, tr, ev, log)
    names = {}
    for (arm, model), s in S.items():
        names[f"{arm}|{model}"] = s
    for t, s in T.items():
        names[f"T|{t}"] = s
    for arm, d in ctl_scores.items():
        for k, s in d.items():
            names[f"ctl|{arm}|{k}"] = s
    diffs = {}
    for model in MODELS:
        if ("V160+E", model) in S and ("V40+E", model) in S:
            diffs[f"RES|{model}"] = (f"V160+E|{model}", f"V40+E|{model}")
        for t in T:
            if ("V40+E", model) in S:
                diffs[f"HU|{model}|{t}"] = (f"V40+E|{model}", f"T|{t}")
        for a, b in (("E", "E-limit"), ("M+E", "E"), ("V40+E", "E"), ("V160+E", "E"), ("V40+E", "M+E")):
            if (a, model) in S and (b, model) in S:
                diffs[f"{a} - {b}|{model}"] = (f"{a}|{model}", f"{b}|{model}")

    def stat(sel):
        sel = np.asarray(sel)
        y = yev[sel]
        a = {k: L.auroc(v[sel], y) for k, v in names.items()}
        d = {f"D|{k}": a[x] - a[w] for k, (x, w) in diffs.items()}
        return {**{f"A|{k}": v for k, v in a.items()}, **d}

    log(f"  bootstrap: {len(names)} scores, {len(diffs)} paired differences, B {NB}")
    B = CI.stat_cluster_bootstrap(list(range(len(yev))), eid, stat, n_boot=NB, seed=BSEED)
    # outcome inputs
    s_in = {"res": {}, "hu": {}}
    for model in MODELS:
        k = f"D|RES|{model}"
        if k in B:
            s_in["res"][model] = {"point": B[k]["point"], "lo": B[k]["lo"]}
        per = {}
        for t in T:
            kd, ka, kt = f"D|HU|{model}|{t}", f"A|V40+E|{model}", f"A|T|{t}"
            if kd in B:
                per[t] = {"diff": B[kd]["point"], "diff_lo": B[kd]["lo"], "probe_lo": B[ka]["lo"], "t_point": B[kt]["point"]}
        if per:
            s_in["hu"][model] = per
    decision = decide_outcomes(s_in)
    # control verdicts
    cv = {}
    for arm in ctl_scores:
        for kind in ("label_shuffle_scores", "episode_shuffled_scores"):
            r = B.get(f"A|ctl|{arm}|{kind}")
            if r:
                cv[f"{arm}|{kind}"] = {"auroc": r["point"], "lo": r["lo"], "hi": r["hi"], "ci_covers_0.5": bool(r["lo"] <= 0.5 <= r["hi"])}
        r = B.get(f"A|ctl|{arm}|leak_scores")
        if r:
            cv[f"{arm}|leak"] = {"auroc": r["point"], "ge_0.999": bool(r["point"] >= 0.999)}
    ctl["verdicts"] = cv
    res = {"card": "PREREG_V8F_STOPPERC.md section 4 S0-P / section 8; AMENDMENT_1_S0.md", "tier": "linear / MLP probe on frozen encoder features: a diagnostic of what the inputs carry, never driving performance; eval-141 first-tick windows (STOPCUE's rule)",
           "estimator": f"taniteval.ci.stat_cluster_bootstrap, B {NB}, seed {BSEED}, by episode ('another draw of episodes' only); paired differences on the same resample",
           "n_eval": int(len(yev)), "n_eval_pos": int(yev.sum()), "n_eval_episodes": int(len(np.unique(eid))), "n_train": int(len(tr["y"])), "n_train_pos": int(np.asarray(tr["y"]).sum()),
           "fits": fits, "bootstrap": B, "outcome_inputs": s_in, "outcome": decision, "controls": ctl,
           "ambiguities": ["HEAD-UNDERUSES is the PAIRED rule of AMENDMENT_2_S0P.md: paired dAUC >= 0.037 and its CI lower bound > 0 at both T arms; probe_lo vs t_point reported beside",
                           "HEAD-UNDERUSES model family: linear OR mlp (as RES-ADDS states); hu_models names which",
                           "episode-shuffled labels: each clip's label sequence replaced by another clip's (cycled), on the TRAIN side; label shuffle: window-level permutation on the train side; hyperparameters reused from the real linear fit"]}
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1)
    log(json.dumps({"outcome": decision, "controls": cv, "fits": {k: v["eval_auroc_point"] for k, v in fits.items()}}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "fit", "read", "all"))
    ap.add_argument("--train", required=True, help="the train-side directory (s0p_extract_train.py) or npz")
    ap.add_argument("--eval", required=True, help="s0p_eval.npz")
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--out", default=None, help="read: RESULT_S0P.json path")
    a = ap.parse_args(argv)
    tr, ev = load_any(a.train), load_any(a.eval)
    os.makedirs(a.work, exist_ok=True)
    if a.cmd in ("prep", "all"):
        prep(a.work, tr, ev)
    if a.cmd == "fit":
        assert a.arm and a.model
        fit_one(a.arm, a.model, a.work, tr, ev)
    if a.cmd == "all":
        for arm, model in itertools.product(ARMS, MODELS):
            fit_one(arm, model, a.work, tr, ev)
    if a.cmd in ("read", "all"):
        read_all(a.work, tr, ev, a.out or os.path.join(a.work, "RESULT_S0P.json"))


if __name__ == "__main__":
    main()
