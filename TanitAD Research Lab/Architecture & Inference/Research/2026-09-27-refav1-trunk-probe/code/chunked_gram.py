"""Memory-lean drop-in for probe_sp.GramReadout: the SAME Gram-trick PCA (basis fit on the training rows only),
computed in column chunks straight from the fp16 feature matrix, so no full float32 copy of an n x 82k matrix ever
exists. A2 crashed (MemoryError, 632 MiB) holding several such copies; the math here is unchanged:
    Z = (X - mu) / sd  (column-wise, training rows)      G = Z Z^T = sum over column chunks
    eigh(G) -> top-m U, S=sqrt(eig)                        V^T = (Z^T U / S)^T, built chunk by chunk
    projection of any rows = ((X_rows - mu) / sd) @ V      (chunked)
`selfcheck()` asserts equality with GramReadout on a random matrix before any real use."""
import numpy as np
import probe as P
import probe_sp as S

CH = 4096


class ChunkedGramReadout(P.Readout):
    def __init__(self, img, kf, rows=None):
        """img: the FULL fp16/fp32 feature matrix; rows: the training row indices (None = all)."""
        self.kmu, self.ksd = kf.mean(0), kf.std(0) + 1e-8
        self.has_img = img is not None
        if not self.has_img:
            return
        r = np.arange(img.shape[0]) if rows is None else np.asarray(rows)
        n, d = len(r), img.shape[1]
        self.imu = np.empty(d, np.float64); self.isd = np.empty(d, np.float64)
        G = np.zeros((n, n), np.float64)
        for c0 in range(0, d, CH):
            blk = img[r, c0:c0 + CH].astype(np.float32)
            mu = blk.mean(0, dtype=np.float64); sd = blk.std(0, dtype=np.float64) + 1e-6
            self.imu[c0:c0 + CH], self.isd[c0:c0 + CH] = mu, sd
            z = ((blk - mu.astype(np.float32)) / sd.astype(np.float32))
            G += (z @ z.T).astype(np.float64)
            del blk, z
        ev, U = np.linalg.eigh(G)
        order = np.argsort(ev)[::-1][: max(P.MS)]
        ev, U = np.clip(ev[order], 1e-12, None), U[:, order].astype(np.float32)
        s = np.sqrt(ev).astype(np.float32)
        self.vt = np.empty((len(order), d), np.float32)
        for c0 in range(0, d, CH):
            blk = img[r, c0:c0 + CH].astype(np.float32)
            z = (blk - self.imu[c0:c0 + CH].astype(np.float32)) / self.isd[c0:c0 + CH].astype(np.float32)
            self.vt[:, c0:c0 + CH] = (z.T @ U / s).T
            del blk, z

    def project(self, img, rows, m):
        r = np.arange(img.shape[0]) if rows is None else np.asarray(rows)
        out = np.zeros((len(r), m), np.float64)
        for c0 in range(0, img.shape[1], CH):
            blk = img[r, c0:c0 + CH].astype(np.float32)
            z = (blk - self.imu[c0:c0 + CH].astype(np.float32)) / self.isd[c0:c0 + CH].astype(np.float32)
            out += (z @ self.vt[:m, c0:c0 + CH].T).astype(np.float64)
            del blk, z
        return out

    def design_rows(self, img, rows, kf_rows, m):
        blocks = [(kf_rows - self.kmu) / self.ksd]
        if self.has_img:
            blocks.append(self.project(img, rows, m))
        return np.concatenate(blocks, 1)

    def fit_rows(self, img, rows, kf_rows, y, lam, m):
        X = self.design_rows(img, rows, kf_rows, m)
        self.lam, self.m, self.d = lam, m, X.shape[1]
        self.xmu, self.ymu, self.W = self.ridge(X, y, lam)
        return self

    def predict_rows(self, img, rows, kf_rows):
        return (self.design_rows(img, rows, kf_rows, self.m) - self.xmu) @ self.W + self.ymu


def fit_select_rows(img, rows, kf, y, base, g, groups, use_img):
    """== probe_sp.fit_select, but img stays the FULL matrix and folds are row indices (no copies)."""
    rows = np.asarray(rows)
    ug = np.unique(groups)
    inner = {e: i % P.N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    ms = P.MS if use_img else (0,)
    score = {(l, m): [] for l in P.LAMS for m in ms}
    for f in range(P.N_INNER):
        tr, te = fold != f, fold == f
        ro = ChunkedGramReadout(img if use_img else None, kf[tr], rows[tr] if use_img else None)
        for m in ms:
            Xtr = ro.design_rows(img, rows[tr], kf[tr], m) if use_img else (kf[tr] - ro.kmu) / ro.ksd
            Xte = ro.design_rows(img, rows[te], kf[te], m) if use_img else (kf[te] - ro.kmu) / ro.ksd
            for lam in P.LAMS:
                xmu, ymu, W = P.Readout.ridge(Xtr, y[tr], lam)
                yh = (Xte - xmu) @ W + ymu
                score[(lam, m)].append(P.ade(base[te] + yh.reshape(-1, *base.shape[1:]), g[te]))
    best = min(score, key=lambda k_: float(np.concatenate(score[k_]).mean()))
    ro = ChunkedGramReadout(img if use_img else None, kf, rows if use_img else None)
    ro.fit_rows(img, rows, kf, y, *best) if use_img else ro.fit(None, kf, y, *best)
    return ro, best


def selfcheck():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 9000)).astype(np.float16)
    kf = rng.standard_normal((300, 8)); y = rng.standard_normal((300, 20))
    tr = np.arange(0, 220); te = np.arange(220, 300)
    a = S.GramReadout(X[tr].astype(np.float32), kf[tr]).fit(X[tr].astype(np.float32), kf[tr], y[tr], 10.0, 16)
    b = ChunkedGramReadout(X, kf[tr], tr).fit_rows(X, tr, kf[tr], y[tr], 10.0, 16)
    pa = a.predict(X[te].astype(np.float32), kf[te]); pb = b.predict_rows(X, te, kf[te])
    err = float(np.abs(pa - pb).max()); scale = float(np.abs(pa).max())
    assert err <= 1e-3 * max(scale, 1.0), f"chunked != GramReadout: max|diff| {err} (scale {scale})"
    return err, scale


if __name__ == "__main__":
    print("selfcheck max|diff|, scale:", selfcheck())
