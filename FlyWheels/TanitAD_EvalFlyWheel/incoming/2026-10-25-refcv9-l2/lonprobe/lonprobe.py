"""lonprobe.py -- PREREG_LONPROBE.md (sha256 5cee9672...). CPU only."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/code")
sys.path.insert(0, "C:/Users/Admin/refcv8_pf/refcv9_l2/roc")
import l2_common as C  # noqa: E402
from l2_roc import auc, dv_of, cls_of  # noqa: E402

BANK = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/bank")
KS, CS = (16, 32, 64, 128), (0.01, 0.1, 1.0, 10.0)


def load_bank(p):
    z = np.load(p, allow_pickle=False)
    W = int(json.loads(str(z["meta_json"]))["window_W"])
    ws = z["t"].astype(np.int64) + W - 1
    return {(int(c), int(w)): j for j, (c, w) in enumerate(zip(z["e_i"], ws))}, z["z_tac"].astype(np.float64)


def folds(eid, k, seed):
    eps = np.unique(eid); rng = np.random.default_rng(seed); rng.shuffle(eps)
    f = {e: i % k for i, e in enumerate(eps)}
    return np.array([f[e] for e in eid])


def fit_pred(Xz, Xs, y, tr, te, eid, use_z, seed=0):
    def make(Xz_tr, Xs_tr, k):
        sc = StandardScaler().fit(Xz_tr) if use_z else None
        pca = PCA(n_components=k, random_state=0).fit(sc.transform(Xz_tr)) if use_z else None
        ss = StandardScaler().fit(Xs_tr)
        def tf(Az, As):
            parts = [ss.transform(As)]
            if use_z:
                parts.insert(0, pca.transform(sc.transform(Az)))
            return np.hstack(parts)
        return tf
    best, bk = None, None
    inner = folds(eid[tr], 4, seed + 1)
    for k in (KS if use_z else (0,)):
        for c in CS:
            sc_ = []
            for f in range(4):
                a, b = tr[inner != f], tr[inner == f]
                tf = make(Xz[a], Xs[a], k)
                m = LogisticRegression(C=c, max_iter=2000).fit(tf(Xz[a], Xs[a]), y[a])
                p = m.decision_function(tf(Xz[b], Xs[b]))
                sc_.append(auc(-p[y[b] == 1], -p[y[b] == 0]))
            s = float(np.mean(sc_))
            if best is None or s > best:
                best, bk = s, (k, c)
    k, c = bk
    tf = make(Xz[tr], Xs[tr], k)
    m = LogisticRegression(C=c, max_iter=2000).fit(tf(Xz[tr], Xs[tr]), y[tr])
    return m.decision_function(tf(Xz[te], Xs[te])), bk


def lse(x, idx):
    a = x[:, idx]; mx = a.max(1, keepdims=True)
    return (mx + np.log(np.exp(a - mx).sum(1, keepdims=True)))[:, 0]


def main():
    out = {"prereg_sha256": open(Path(__file__).parent / "PREREG_LONPROBE.sha256").read().split()[0], "seeds": {}}
    for seed in (0, 1):
        E = C.load_eval_pf(seed)
        idx, Z = load_bank(BANK / f"ztac_eval_s{seed}.npz")
        dg = dv_of(E["g"], E["v0"], E["dt"]); gt = cls_of(dg)
        m = E["compliant"] & np.isin(gt, ["decel", "cruise"])
        rows = np.flatnonzero(m)
        j = np.array([idx.get((int(c), int(w)), -1) for c, w in zip(E["clip"][rows], E["ws"][rows])])
        if (j < 0).any():
            raise SystemExit(f"VOID: {(j < 0).sum()} rows without a z_tac bank entry")
        y = (gt[rows] == "decel").astype(int); eid = E["eid"][rows]
        Xz = Z[j]; Xs = E["v0"][rows][:, None]
        L6 = C.lon6(E["logits"][rows]) if hasattr(C, "lon6") else E["logits"][rows]
        head = lse(L6, list(C.G_DECEL)) - lse(L6, list(C.G_CRUISE))
        plan = -dv_of(E["stage_banked"][rows], E["v0"][rows], E["dt"])
        of = folds(eid, 5, 0)
        probe = np.empty(len(rows)); scal = np.empty(len(rows)); shuf = np.empty(len(rows)); picks = []
        rng = np.random.default_rng(0)
        for f in range(5):
            tr, te = np.flatnonzero(of != f), np.flatnonzero(of == f)
            probe[te], bk = fit_pred(Xz, Xs, y, tr, te, eid, True); picks.append(bk)
            scal[te], _ = fit_pred(Xz, Xs, y, tr, te, eid, False)
            ys = y.copy(); ys[tr] = rng.permutation(y[tr])
            shuf[te], _ = fit_pred(Xz, Xs, ys, tr, te, eid, True)
        arms = {"HEAD": head, "PLAN": plan, "PROBE": probe, "SCALAR": scal, "SHUFFLE": shuf}
        A = {k: auc(-v[y == 1], -v[y == 0]) for k, v in arms.items()}   # auc(): lower score = positive; flip
        eps = np.unique(eid); byep = {e: np.flatnonzero(eid == e) for e in eps}
        rb = np.random.default_rng(0); D = {p: [] for p in ("PROBE-HEAD", "PROBE-PLAN", "PROBE-SCALAR")}
        for b in range(2000):
            ii = np.concatenate([byep[e] for e in rb.choice(eps, len(eps), replace=True)])
            yy = y[ii]
            a = {k: auc(-arms[k][ii][yy == 1], -arms[k][ii][yy == 0]) for k in ("HEAD", "PLAN", "PROBE", "SCALAR")}
            D["PROBE-HEAD"].append(a["PROBE"] - a["HEAD"]); D["PROBE-PLAN"].append(a["PROBE"] - a["PLAN"])
            D["PROBE-SCALAR"].append(a["PROBE"] - a["SCALAR"])
        res = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_episodes": int(len(eps)), "auc": A, "picks_k_C": picks,
               "paired": {k: {"delta": float(np.mean([A["PROBE"] - A[k.split('-')[1]]])),
                              "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in D.items()},
               "control_shuffle_ok": abs(A["SHUFFLE"] - 0.5) <= 0.03}
        out["seeds"][str(seed)] = res
        print("ZZSEED", seed, json.dumps({k: res[k] for k in ("n_pos", "n_neg", "n_episodes", "auc", "paired", "control_shuffle_ok")}))
    json.dump(out, open(Path(__file__).parent / "LONPROBE.json", "w"), indent=1)
    print("ZZLONPROBEDONEZZ")


if __name__ == "__main__":
    main()
