"""PREREG_V8F_TFAR zero-training probe (eval-141, beside evidence for ranking levers, never a result).

Does H2's PREDICTED 120 m point (what T would read at inference) carry information about T's longitudinal label
(DECELERATE or STOP) beyond what T already has: its own lon logits, S's predicted 80 m heading, and the speed?
Logistic regression, episode-grouped 2-fold split (fold = episode id parity); the fit never scores an episode it saw.
A negative from a linear probe is not a negative about learnability (TRAPS).

Run: python -I probe_h2_far_info.py --out H2_FAR_PROBE.json
"""
import argparse, hashlib, json
import numpy as np

T0 = "F:/TanitAD-scratch/strat1/read/ckpt120/out/t0/eval_CK120-{}_real{}.npz"
TRUTH = "F:/TanitAD-scratch/strat1/read/ckpt120/inputs/eval141_grid_truth.npz"
JOIN = "F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz"
RC120 = "F:/TanitAD-scratch/strat1/read/ckpt120/inputs/rc120_table.npz"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit_fit(X, y, iters=400, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([np.ones((len(X), 1)), (X - mu) / sd], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    return lambda Xn: 1 / (1 + np.exp(-np.concatenate([np.ones((len(Xn), 1)), (Xn - mu) / sd], 1) @ w))


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv(X, y, fold):
    p = np.empty(len(y))
    for f in (0, 1):
        m = logit_fit(X[fold != f], y[fold != f])
        p[fold == f] = m(X[fold == f])
    ll = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    return {"auc": round(auc(y, p), 4), "logloss": round(ll, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Tr, J, R = np.load(TRUTH), np.load(JOIN), np.load(RC120)
    rows, eid, v, has, on = Tr["rows"], Tr["eid"], Tr["v_now"], Tr["has"], Tr["token_on"]
    lon = J["lon_cls"][rows].astype(int)
    ok = has & R["eval_rc120_ok"][rows].astype(bool) & (lon >= 0) & (lon < 7)
    y_all = np.isin(lon, (2, 4)).astype(float)
    psi120_gt = np.degrees(np.abs(R["eval_rc120"][rows, 2]))
    psi80_gt = np.degrees(np.abs(J["rc"][rows, 2, 2]))
    rep = {"inputs": {}, "note": "post-hoc eval-141 probe, episode-grouped 2-fold; beside evidence for ranking, never a result"}
    for arm in ("H2_s0", "H2_s1"):
        f, ff = T0.format(arm, ""), T0.format(arm, "_far")
        rep["inputs"][f], rep["inputs"][ff] = sha(f), sha(ff)
        Z, Fz = np.load(f), np.load(ff)
        assert (Z["eid"] == eid).all(), "row alignment"
        lo = Z["lon_logits"].astype(np.float64)
        sm = np.exp(lo - lo.max(1, keepdims=True)); sm /= sm.sum(1, keepdims=True)
        psi80p = np.abs(Z["ckpt"][:, 2, 2].astype(np.float64))
        far = Fz["ckpt_far"].astype(np.float64)
        base = np.concatenate([sm, psi80p[:, None], v[:, None]], 1)
        plus = np.concatenate([base, np.abs(far[:, 2:3]), np.abs(far[:, 1:2]) / 100.0], 1)
        out = {}
        for nm, m in (("all_rc120_valid", ok),
                      ("far_bend_only_gt", ok & (psi120_gt >= 25) & (psi80_gt < 25)),
                      ("token_off_scorable", ok & ~on & (Tr["side"] != 0))):
            fold = eid[m] % 2
            y = y_all[m]
            out[nm] = {"n": int(m.sum()), "episodes": int(len(np.unique(eid[m]))), "pos": int(y.sum()),
                       "T_lon_argmax_acc": round(float((np.argmax(lo[m], 1) == lon[m]).mean()), 4),
                       "base": cv(base[m], y, fold), "base_plus_far": cv(plus[m], y, fold)} if m.sum() > 20 and 0 < y.sum() < m.sum() else {"n": int(m.sum())}
        rep[arm] = out
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("H2_s0", "H2_s1")}, indent=1))


if __name__ == "__main__":
    main()
