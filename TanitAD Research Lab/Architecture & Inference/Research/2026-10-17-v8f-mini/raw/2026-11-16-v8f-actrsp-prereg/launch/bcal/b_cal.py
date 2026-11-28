"""B-CAL (PREREG_V8F_ACTRSP section 6, READ_PLAN N4). TRAIN latents only; no scored window.
Cross-fitted (episode-less: grouped by window) ridge probe, R2 per action channel:
  z_t alone  vs  [z_t, z_{t+1}].  Rule: z_t-only R2 > 0.25 * pair R2 on a channel => TRIGGER (amendment needed).
Usage: python b_cal.py <latents_train_R35_sK.pt> <tag> <out.json> [window|eid]"""
import json, sys
import numpy as np, torch

path, tag, out = sys.argv[1:4]
L = torch.load(path, map_location="cpu", weights_only=False)
print("keys", list(L.keys())[:20])
z = L["Z1"][tag].float().numpy()            # [N, T, 128]
A = L["A_raw"].float().numpy()               # [N, T, 2] raw
ok = L["ok"].numpy() if torch.is_tensor(L["ok"]) else np.asarray(L["ok"])
print("Z1", z.shape, "A_raw", A.shape, "ok", ok.shape)
SC = np.array([2.0, 0.4], dtype=np.float32)   # acc / 2.0, yaw / 0.4 (card 2.1 step 1); R2 is scale invariant
a = (A / SC)[:, :-1]
v = ok[:, :-1] & ok[:, 1:] & np.isfinite(a).all(-1)
N = z.shape[0]
widx = np.broadcast_to(np.arange(N)[:, None], v.shape)[v]
X1 = z[:, :-1][v]; X2 = np.concatenate([z[:, :-1][v], z[:, 1:][v]], 1); Y = a[v]
print("transitions", len(Y), "windows", N)
rng = np.random.RandomState(0)
GROUP = sys.argv[4] if len(sys.argv) > 4 else "window"      # "window" or "eid" (episode: windows of one clip overlap by >= 77 of 91 frames, so a window split can leak)
if GROUP == "eid":
    eid = np.asarray(L["eid"]); ue = np.unique(eid); print("episodes", len(ue))
    fe = {e: i for e, i in zip(ue, rng.permutation(len(ue)) % 5)}
    fold_of_w = np.array([fe[e] for e in eid])
else:
    fold_of_w = rng.permutation(N) % 5
fold = fold_of_w[widx]

def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1]), Xs.T @ (Y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ W + ym

def cross_r2(X, Y):
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    P = np.zeros_like(Y)
    chosen = []
    for f in range(5):
        tr, te = fold != f, fold == f
        # inner selection on train folds only (3 inner folds by window group)
        inner = (fold[tr] + 1) % 3
        best, bl = 1e18, None
        for lam in lams:
            e = 0.0
            for g in range(3):
                m = ridge_fit(X[tr][inner != g], Y[tr][inner != g], lam)
                e += ((m(X[tr][inner == g]) - Y[tr][inner == g]) ** 2).sum()
            if e < best: best, bl = e, lam
        chosen.append(bl)
        P[te] = ridge_fit(X[tr], Y[tr], bl)(X[te])
    r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    # window-cluster bootstrap CI
    ws = np.unique(widx); boots = []
    brng = np.random.RandomState(1)
    byw = {w: np.where(widx == w)[0] for w in ws}
    for _ in range(200):
        pick = brng.choice(ws, len(ws))
        ii = np.concatenate([byw[w] for w in pick])
        boots.append(1 - ((P[ii] - Y[ii]) ** 2).sum(0) / ((Y[ii] - Y[ii].mean(0)) ** 2).sum(0))
    boots = np.array(boots)
    return r2, np.percentile(boots, [2.5, 97.5], 0), chosen

r_s, ci_s, l_s = cross_r2(X1, Y)
r_p, ci_p, l_p = cross_r2(X2, Y)
res = {"group": GROUP, "file": path, "tag": tag, "n_transitions": int(len(Y)), "n_windows": int(N), "rule": "z_t-only R2 > 0.25 * pair R2 on a channel => trigger", "channels": {}}
trig = False
for c, nm in enumerate(("acc", "yaw_rate")):
    ratio = float(r_s[c] / r_p[c]) if r_p[c] > 0 else None
    t = bool(r_s[c] > 0.25 * r_p[c])
    trig |= t
    res["channels"][nm] = {"r2_z_only": float(r_s[c]), "ci_z_only": ci_s[:, c].tolist(), "r2_pair": float(r_p[c]), "ci_pair": ci_p[:, c].tolist(),
                           "threshold_0.25xpair": float(0.25 * r_p[c]), "ratio": ratio, "trigger": t}
res["TRIGGER_ANY_CHANNEL"] = trig
res["lambda_chosen"] = {"z_only": l_s, "pair": l_p}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
print("ZZ BCAL DONE ZZ")
