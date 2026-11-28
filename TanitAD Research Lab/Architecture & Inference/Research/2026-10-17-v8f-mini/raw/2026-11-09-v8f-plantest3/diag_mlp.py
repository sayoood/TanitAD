"""Beside PREREG_V8F_PLANTEST3, deciding nothing (Master Mind, 2026-11-09): is speed / yaw rate present in the level-1 latents but nonlinearly? A small MLP readout on the card's TUNE inner split
(same held-out episodes as the gate); early stopping on 20 % of the fit EPISODES, so the held-out episodes are read once. No SCORE window. CPU. Shuffled-target control per row.
Tier T0, a readout diagnostic on encoded latents, never a driving result."""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
import diag_multiframe as M
import plantest_lib as P
import plantest2_lib as Q
import plantest3_lib as L3


def r2(y, p):
    return 1.0 - float(((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def train(xa, ya, xv, yv, xh, seed, epochs=120, bs=512):
    torch.manual_seed(seed)
    mu, sd = xa.mean(0, keepdim=True), xa.std(0, keepdim=True) + 1e-6
    ym, ys = ya.mean(), ya.std() + 1e-9
    xa, xv, xh = (xa - mu) / sd, (xv - mu) / sd, (xh - mu) / sd
    t = (ya - ym) / ys
    net = nn.Sequential(nn.Linear(xa.shape[1], 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, bad = 1e9, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xa), generator=g)
        for i in range(0, len(xa), bs):
            j = perm[i:i + bs]
            loss = ((net(xa[j])[:, 0] - t[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(xv)[:, 0] * ys + ym - yv) ** 2).mean())
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: b.clone() for k, b in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return (net(xh)[:, 0] * ys + ym).numpy(), ep + 1


def read(tag, ev, TRK):
    eid = [str(e) for e in ev["eid"]]
    tune, _ = Q.split_indices(np.array(eid))
    fit_ids, held_ids = L3.inner_split([eid[i] for i in tune])
    es_ids = {e for e in fit_ids if hashlib.sha256(f"plantest3-mlp-es|{e}".encode()).digest()[0] < 51}
    tr = np.array([i for i in tune if eid[i] in fit_ids and eid[i] not in es_ids])
    va = np.array([i for i in tune if eid[i] in es_ids])
    he = np.array([i for i in tune if eid[i] in held_ids])
    assert not (set(tr) & set(he)) and not (set(va) & set(he)) and not (set(tr) & set(va))
    Z = ev["Z1"].double().numpy()
    A = np.asarray(ev["A"], np.float64)
    n = len(eid)
    vk = np.full((n, 16), np.nan)
    for i, (e, t) in enumerate(zip(eid, ev["t"])):
        g = int(t) - 15
        r = TRK.get(e)
        if r is None or g < 0:
            continue
        r = np.asarray(r)
        for k in range(16):
            if 2 * (g + k) < len(r):
                vk[i, k] = float(r[2 * (g + k), 3])
    rows = []
    for m in (1, 4):
        X = M.stack(Z, m)
        tg = {"speed": vk[:, m - 1:], "yaw_behind": (A[:, m - 2:-1, 1] if m >= 2 else np.concatenate([np.full((n, 1), np.nan), A[:, :-1, 1]], axis=1))}
        row = dict(m=m, n_train_windows=len(tr), n_es_windows=len(va), n_held_windows=len(he))
        for name, y in tg.items():
            def part(ix):
                a, b = X[ix].reshape(-1, X.shape[2]), y[ix].reshape(-1)
                ok = np.isfinite(b)
                return torch.from_numpy(a[ok]).float(), torch.from_numpy(b[ok]).float()
            xa, ya = part(tr); xv, yv = part(va); xh, yh = part(he)
            p, ep = train(xa, ya, xv, yv, xh, seed=0)
            ys = ya[torch.randperm(len(ya), generator=torch.Generator().manual_seed(1))]
            ps, _ = train(xa, ys, xv, yv[torch.randperm(len(yv), generator=torch.Generator().manual_seed(2))], xh, seed=0)
            row[name] = dict(r2_heldout=r2(yh.numpy(), p), r2_shuffled=r2(yh.numpy(), ps), epochs=ep, n_held=int(len(yh)))
        rows.append(row)
        print(tag, row, flush=True)
    return rows


def main():
    TRK = torch.load(M.TRACKS, weights_only=False)["tracks"]
    res = {}
    for tag in ("M2b_s0", "M2b_s1"):
        res[tag] = read(tag, P.load_latents(M.EVAL, tag), TRK)
    json.dump(res, open(os.path.join(HERE, "DIAG_MLP.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
