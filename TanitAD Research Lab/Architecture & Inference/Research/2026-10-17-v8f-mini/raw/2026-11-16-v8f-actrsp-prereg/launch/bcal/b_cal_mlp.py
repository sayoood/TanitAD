"""B-CAL supplement (diagnostic, NOT the card's rule): same TRAIN latents, cross-fitted by window. Reports R2 per channel for
 (1) linear on z_t, (2) linear on [z_t, z_{t+1}], (3) linear on z_{t+1}-z_t only, (4) MLP on z_t, (5) MLP on [z_t, z_{t+1}] (the shape of the level-1 IDM head).
Usage: python b_cal_mlp.py <latents.pt> <tag> <out.json>"""
import json, sys
import numpy as np, torch
torch.set_num_threads(2)
path, tag, out = sys.argv[1:4]
L = torch.load(path, map_location="cpu", weights_only=False)
z = L["Z1"][tag].float().numpy(); A = L["A_raw"].float().numpy(); ok = np.asarray(L["ok"])
a = (A / np.array([2.0, 0.4], dtype=np.float32))[:, :-1]
v = ok[:, :-1] & ok[:, 1:] & np.isfinite(a).all(-1)
N = z.shape[0]; widx = np.broadcast_to(np.arange(N)[:, None], v.shape)[v]
zt, zn, Y = z[:, :-1][v], z[:, 1:][v], a[v]
fold = (np.random.RandomState(0).permutation(N) % 5)[widx]
feats = {"z_t": zt, "pair": np.concatenate([zt, zn], 1), "diff_only": zn - zt}

def lin(X, Y, tr, te, lam=1e-3):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = (X - mu) / sd; ym = Y[tr].mean(0)
    W = np.linalg.solve(Xs[tr].T @ Xs[tr] + lam * tr.sum() * np.eye(X.shape[1]), Xs[tr].T @ (Y[tr] - ym))
    return Xs[te] @ W + ym

def mlp(X, Y, tr, te, ep=12, seed=0):
    torch.manual_seed(seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xs = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 2))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2); idx = np.where(tr)[0]
    for e in range(ep):
        np.random.RandomState(e).shuffle(idx)
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]; loss = ((net(Xs[b]) - Yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net(Xs[te]).numpy()

res = {"file": path, "tag": tag, "n_transitions": int(len(Y)), "note": "diagnostic; the card's rule is the linear z_t vs linear pair (b_cal.py)"}
for name, X in feats.items():
    for kind, fn in (("linear", lin), ("mlp", mlp)):
        if kind == "linear" and name == "diff_only" or (kind == "mlp" and name == "diff_only"): 
            if kind == "mlp": continue
        P = np.zeros_like(Y)
        for f in range(5):
            te = fold == f; P[te] = fn(X, Y, ~te, te)
        r2 = 1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        res[f"{kind}_{name}"] = {"acc": float(r2[0]), "yaw_rate": float(r2[1])}
        print(kind, name, r2, flush=True); json.dump(res, open(out + ".partial", "w"), indent=1)
for ch in ("acc", "yaw_rate"):
    res[f"mlp_ratio_{ch}"] = res["mlp_z_t"][ch] / res["mlp_pair"][ch] if res["mlp_pair"][ch] > 0 else None
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1)); print("ZZ BCAL-MLP DONE ZZ")
