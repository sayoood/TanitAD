"""PREREG_V8F_IMAG1 probes: ridge / logistic (LATSEM2B's, unchanged) and a batched 2-layer MLP bank (hidden 64, 3 probe seeds, early stop on an inner split). CPU, torch, no model."""
import warnings
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

LAM, HID, NCLS = 10.0, 64, 7
MLP_CFG = dict(lr=3e-3, bs=256, max_epochs=15, patience=3, probe_seeds=(0, 1, 2))
CONV = {"fits": 0, "not_converged": 0}


# ------------------------------------------------------------------------------------------------ ridge / logistic
def fit_lin(kind, X, y):
    """kind cont: ridge lambda 10 on the standardised features; bin / cls: logistic C 0.1 lbfgs (LATSEM2's rules). Returns a tuple."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    if kind == "cont":
        ym = y.mean()
        return ("cont", mu, sd, np.linalg.solve(Xs.T @ Xs + LAM * np.eye(X.shape[1]), Xs.T @ (y - ym)), ym)
    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000, tol=1e-4)
    with warnings.catch_warnings(record=True) as wr:
        warnings.simplefilter("always")
        clf.fit(Xs, y.astype(np.int64))
    CONV["fits"] += 1
    CONV["not_converged"] += int(any(issubclass(x.category, ConvergenceWarning) for x in wr))
    vals, cnt = np.unique(y, return_counts=True)
    return ("cls", mu, sd, clf, float(vals[cnt.argmax()]))


def predict_lin(P, X):
    Xs = (X - P[1]) / P[2]
    return Xs @ P[3] + P[4] if P[0] == "cont" else P[3].predict(Xs).astype(np.float64)


def majority(y):
    vals, cnt = np.unique(y, return_counts=True)
    return float(vals[cnt.argmax()])


# ------------------------------------------------------------------------------------------------ batched MLP bank
class MLPBank:
    """M independent MLPs (D -> 64 -> 7, ReLU) trained at once: one per (target, probe seed). Targets: kind cont (standardised y, MSE), bin (BCE on logit 0), cls (CE on 7 logits).
    Rows with a NaN label do not enter that model's loss. Adam lr 3e-3, minibatches of 256 shared by all models, at most 15 epochs, early stop on the inner-val loss
    (patience 3 epochs; the best weights per model are kept). Prediction of a target = the mean over its 3 probe seeds (cont: value; bin: mean probability >= 0.5; cls: argmax of the mean softmax)."""

    def __init__(self, targets, kinds, D, seed):
        self.targets, self.kinds, self.D = list(targets), dict(kinds), D
        self.models = [(q, s) for q in self.targets for s in MLP_CFG["probe_seeds"]]
        M = len(self.models)
        self.W1, self.b1 = torch.zeros(M, D, HID), torch.zeros(M, 1, HID)
        self.W2, self.b2 = torch.zeros(M, HID, NCLS), torch.zeros(M, 1, NCLS)
        for i, (q, s) in enumerate(self.models):       # probe seed s decorrelates the init of the three copies of one target
            gs = torch.Generator().manual_seed((int(seed) * 7919 + 131 * s + 17 * self.targets.index(q) + 5) % (2 ** 31))
            self.W1[i] = (torch.rand(D, HID, generator=gs) * 2 - 1) / np.sqrt(D)
            self.b1[i] = (torch.rand(1, HID, generator=gs) * 2 - 1) / np.sqrt(D)
            self.W2[i] = (torch.rand(HID, NCLS, generator=gs) * 2 - 1) / np.sqrt(HID)
            self.b2[i] = (torch.rand(1, NCLS, generator=gs) * 2 - 1) / np.sqrt(HID)
        self.ymu = torch.zeros(M)
        self.ysd = torch.ones(M)
        self.epochs_run = 0
        self.seed = int(seed)

    def _fwd(self, X, sl=None):
        W1, b1, W2, b2 = (self.W1, self.b1, self.W2, self.b2) if sl is None else (self.W1[sl], self.b1[sl], self.W2[sl], self.b2[sl])
        M = W1.shape[0]
        h = (X @ W1.permute(1, 0, 2).reshape(self.D, M * HID)).reshape(X.shape[0], M, HID).permute(1, 0, 2) + b1
        return torch.bmm(torch.relu(h), W2) + b2                       # [M, n, 7]

    @staticmethod
    def _loss(out, Y, K):
        """per-model masked mean loss. Y [M, n] (NaN = no label), K [M] kind code 0 cont / 1 bin / 2 cls."""
        m = torch.isfinite(Y)
        y0 = torch.where(m, Y, torch.zeros_like(Y))
        l_c = (out[..., 0] - y0) ** 2
        l_b = torch.nn.functional.softplus(out[..., 0]) - y0 * out[..., 0]
        l_k = torch.nn.functional.cross_entropy(out.reshape(-1, NCLS), y0.long().clamp(0, NCLS - 1).reshape(-1), reduction="none").reshape(Y.shape)
        loss = torch.where((K == 0)[:, None], l_c, torch.where((K == 1)[:, None], l_b, l_k)) * m
        return loss.sum(1) / m.sum(1).clamp(min=1)

    def fit(self, X, ys, inner_val):
        """X [n, D] standardised, ys {target: y [n] with NaN}, inner_val bool [n]: the rows held out for early stopping."""
        cfg = MLP_CFG
        X = torch.as_tensor(X, dtype=torch.float32)
        M = len(self.models)
        Y = torch.stack([torch.as_tensor(ys[q], dtype=torch.float32) for q, _ in self.models])                       # [M, n]
        K = torch.tensor([{"cont": 0, "bin": 1, "cls": 2}[self.kinds[q]] for q, _ in self.models])
        trm = torch.as_tensor(~inner_val)
        for i, (q, _) in enumerate(self.models):
            if self.kinds[q] == "cont":
                f = torch.isfinite(Y[i]) & trm
                if f.sum() > 1:
                    self.ymu[i], self.ysd[i] = Y[i][f].mean(), Y[i][f].std().clamp(min=1e-6)
                Y[i] = (Y[i] - self.ymu[i]) / self.ysd[i]
        tr = np.flatnonzero(~inner_val)
        va = torch.as_tensor(np.flatnonzero(inner_val))
        params = [self.W1, self.b1, self.W2, self.b2]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=cfg["lr"])
        best = torch.full((M,), float("inf"))
        bad = torch.zeros(M, dtype=torch.long)
        snap = [p.detach().clone() for p in params]
        gen = np.random.default_rng(self.seed + 12345)
        Xv, Yv = X[va], Y[:, va]
        has_val = torch.isfinite(Yv).any(1)
        for ep in range(cfg["max_epochs"]):
            order = gen.permutation(tr)
            for s in range(0, len(order), cfg["bs"]):
                ix = torch.as_tensor(order[s:s + cfg["bs"]])
                loss = self._loss(self._fwd(X[ix]), Y[:, ix], K).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                vl = self._loss(self._fwd(Xv), Yv, K)
                imp = (vl < best - 1e-6) | ~has_val
                best = torch.where(imp, vl, best)
                bad = torch.where(imp, torch.zeros_like(bad), bad + 1)
                for p, sn in zip(params, snap):
                    sn[imp] = p.detach()[imp]
            self.epochs_run = ep + 1
            if (bad >= cfg["patience"]).all():
                break
        with torch.no_grad():
            for p, sn in zip(params, snap):
                p.copy_(sn)
                p.requires_grad_(False)
        return self

    @torch.no_grad()
    def predict(self, X, only=None):
        """{target: prediction [n]} (seed-ensembled). ``only`` restricts to a subset of targets (their models only)."""
        X = torch.as_tensor(X, dtype=torch.float32)
        qs = self.targets if only is None else [q for q in self.targets if q in only]
        idx = [i for i, (q, _) in enumerate(self.models) if q in qs]
        out = self._fwd(X, sl=torch.as_tensor(idx))
        res = {}
        for q in qs:
            j = [n for n, i in enumerate(idx) if self.models[i][0] == q]
            o = out[j]
            kd = self.kinds[q]
            if kd == "cont":
                mu = torch.stack([self.ymu[idx[n]] for n in j])
                sd = torch.stack([self.ysd[idx[n]] for n in j])
                res[q] = (o[..., 0] * sd[:, None] + mu[:, None]).mean(0).double().numpy()
            elif kd == "bin":
                res[q] = (torch.sigmoid(o[..., 0]).mean(0) >= 0.5).double().numpy()
            else:
                res[q] = torch.softmax(o, -1).mean(0).argmax(-1).double().numpy()
        return res
