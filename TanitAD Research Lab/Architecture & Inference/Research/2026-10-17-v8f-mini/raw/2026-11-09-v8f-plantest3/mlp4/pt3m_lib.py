"""PREREG_V8F_PT3M library: the per-target reader (speed: four-frame ridge; yaw rate: DECN's four-frame MLP), its cost class, and the ceilings. Tier T0, CPU. Imported from PLANTEST3 / DECN, nothing copied except train_net (DM.train with an lr argument that returns the network)."""
import hashlib
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PT3 = os.path.abspath(os.path.join(HERE, ".."))
THEORY = os.path.abspath(os.path.join(PT3, ".."))
sys.path.insert(0, PT3)
sys.path.insert(0, os.path.join(THEORY, "decinfo"))
sys.path.insert(0, os.path.join(THEORY, "decinfo", "decn"))
import numpy as np
import torch
import torch.nn as nn

import decn_lib as DN
import plantest3_lib as L3
import plantest_lib as P

DL = DN.DL
M = 4
KNOT_IDX = [11, 10, 9, 8]        # rollout / path index of knots 12, 11, 10, 9 (newest first, as DL.stack_ch)
CAP = DN.CAP_BIG
LRS = (1e-3, 3e-4, 1e-4)
DT = P.DT


def make_net(d):
    return nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))


def train_net(xa, ya, xv, yv, seed=0, epochs=CAP, bs=512, lr=1e-3):
    """diag_mlp.train with the same operations in the same order; adds lr, returns the network and its standardisers, and the best early-stopping loss."""
    torch.manual_seed(seed)
    mu, sd = xa.mean(0, keepdim=True), xa.std(0, keepdim=True) + 1e-6
    ym, ys = ya.mean(), ya.std() + 1e-9
    xa, xv = (xa - mu) / sd, (xv - mu) / sd
    t = (ya - ym) / ys
    net = make_net(xa.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
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
    return dict(net=net, mu=mu, sd=sd, ym=ym, ys=ys, epochs=ep + 1, es_loss=best, lr=lr)


def mlp_predict(m, x):
    with torch.no_grad():
        return (m["net"]((x - m["mu"]) / m["sd"])[:, 0] * m["ys"] + m["ym"]).numpy()


def ridge_weights(xf, yf, ridge=DL.RIDGE, chunk=DN.CHUNK):
    """DN.ridge_pred_chunked's closed form, returning (W (D,), b, target std): prediction = (x @ W + b) * sd."""
    sd = float(yf.std())
    y = torch.as_tensor(yf / sd, dtype=torch.double)
    n, d = xf.shape
    G = torch.zeros(d, d, dtype=torch.double); bv = torch.zeros(d, dtype=torch.double); sx = torch.zeros(d, dtype=torch.double)
    for i in range(0, n, chunk):
        xc = torch.as_tensor(xf[i:i + chunk], dtype=torch.double)
        G += xc.T @ xc; bv += xc.T @ y[i:i + chunk]; sx += xc.sum(0)
    mu, ym = sx / n, y.mean()
    Gc = G - n * torch.outer(mu, mu) + ridge * n * torch.eye(d, dtype=torch.double)
    W = torch.linalg.solve(Gc, bv - n * mu * ym)
    return W, ym - mu @ W, sd


class Reader(nn.Module):
    """Input (..., 512): four latents newest first. Output (..., 2): (speed, yaw rate) in fit-standard units. No parameter requires grad."""

    def __init__(self, Ws, bs, net, mu, sd, ym, ys, ystd):
        super().__init__()
        self.register_buffer("Ws", torch.as_tensor(Ws, dtype=torch.float32)); self.register_buffer("bs", torch.as_tensor(float(bs), dtype=torch.float32))
        self.net = net
        self.register_buffer("mu", mu.squeeze(0).float()); self.register_buffer("sd", sd.squeeze(0).float())
        self.register_buffer("ym", torch.as_tensor(float(ym))); self.register_buffer("ys", torch.as_tensor(float(ys)))
        self.ystd = np.asarray(ystd, np.float64)
        self.register_buffer("ystd_yaw", torch.as_tensor(float(ystd[1]), dtype=torch.float32))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        s = x @ self.Ws + self.bs
        w = (self.net((x - self.mu) / self.sd)[..., 0] * self.ys + self.ym) / self.ystd_yaw
        return torch.stack([s, w], -1)

    def read_raw(self, x):
        with torch.no_grad():
            return self(x).numpy().astype(np.float64) * self.ystd

    def digest(self):
        h = hashlib.sha256()
        for t in (self.Ws, self.bs, self.mu, self.sd, self.ym, self.ys):
            h.update(t.numpy().tobytes())
        for k in sorted(self.net.state_dict()):
            h.update(self.net.state_dict()[k].numpy().tobytes())
        h.update(self.ystd.tobytes())
        return h.hexdigest()


def save_reader(rd, path):
    torch.save(dict(Ws=rd.Ws, bs=rd.bs, net=rd.net.state_dict(), mu=rd.mu, sd=rd.sd, ym=rd.ym, ys=rd.ys, ystd=rd.ystd, sha256=rd.digest()), path)


def load_reader(path):
    d = torch.load(path, weights_only=False)
    net = make_net(d["mu"].shape[0]); net.load_state_dict(d["net"])
    rd = Reader(d["Ws"], d["bs"], net, d["mu"].unsqueeze(0), d["sd"].unsqueeze(0), d["ym"], d["ys"], d["ystd"])
    assert rd.digest() == d["sha256"], "reader digest"
    return rd


def stack_knots(Z, k):
    """Z (..., T, D), k: the newest knot index in Z -> (..., 4 D), frames k, k-1, k-2, k-3."""
    return torch.cat([Z[..., k - j, :] for j in range(M)], dim=-1)


class ReaderCost(L3.CostSpec):
    """FLAT1 only. mode 'path' routes the 12-knot goal path into top_cost through the unchanged plan_all3 / goal3; dec is the N1053 ridge (used only for the decoded-monotonicity diagnostic in plan_all3)."""

    def __init__(self, c, reader, ridge_dec):
        super().__init__("path", c, ridge_dec)
        self.reader = reader

    def top_cost(self, st, roll, goal, level, last_k):
        x = roll[..., KNOT_IDX, :].flatten(-2)          # (B, S, 512)
        with torch.no_grad():
            g = self.reader(goal[:, KNOT_IDX, :].flatten(-2))   # (B, 2)
        return (self.reader(x) - g.unsqueeze(1)).pow(2).sum(-1)

    def sub_cost(self, st, lifted, sub, lam):
        raise NotImplementedError("PT3M is FLAT1 only")


class LinearK12:
    """Control 3: the N1053 single-frame ridge in the Reader interface; reads the newest of the four frames."""

    def __init__(self, dec, d):
        self.W, self.b, self.d = dec.W1, dec.b1, d

    def __call__(self, x):
        return x[..., :self.d] @ self.W + self.b


# ------------------------------------------------------------------------------------------------ ceilings (raw units, n x 12 x 2)
def kin_plan_m4(reader, Zk, hold):
    """Zk (n, 13, D) encoded latents at knots 0..12; hold (n, 12, 2). Knots 3..11 from the reader, knots 0..2 stay at HOLD."""
    Y = np.stack([reader.read_raw(stack_knots(Zk, k)) for k in range(3, 13)], 1)   # (n, 10, 2), knots 3..12
    plan = np.array(hold, np.float64)
    plan[:, 3:12, 0] = (Y[:, 1:, 0] - Y[:, :-1, 0]) / DT
    plan[:, 3:12, 1] = Y[:, :-1, 1]
    return plan


def end_plan(v0, v12, w0, w12):
    """Constant acceleration to v12 over 2.4 s; yaw rate ramp from w0 to w12 over the 12 knots. Arrays (n,), NaN passes through."""
    ramp = (np.arange(1, 13, dtype=np.float64) / 12.0)[None]
    acc = np.repeat(((v12 - v0) / (12 * DT))[:, None], 12, axis=1)
    yaw = w0[:, None] + (w12 - w0)[:, None] * ramp
    return np.stack([acc, yaw], -1)
