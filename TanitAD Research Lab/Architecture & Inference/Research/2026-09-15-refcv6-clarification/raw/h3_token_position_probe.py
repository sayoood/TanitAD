"""H3 — do refcv5-v2's stride-32 tokens carry their own position?

The anchor decoder attends to the 160 tokens with NO positional encoding
(refc.py:2292), so the only way it can know where a token is, is from the
token's content. This probe asks whether a LINEAR read of a 704-d token
recovers its column (20-way) and row (8-way).

Design (fixed before looking at results):
  * data: tokens_s32_fp16.npy [27,664 rows, 704, 8, 20] of 139 EVAL clips
    (frozen refcv5-v2 trunk, banked 2026-09-13); every 40th frame per clip
  * split: clip-disjoint by sha12 -> int(sha12, 16) % 4 == 0 is TEST;
    of the rest, % 8 == 1 is VAL (carved from FIT clips only)
  * model: ridge regression onto one-hot targets (closed form), features
    z-scored with FIT statistics; lambda picked on VAL only
  * controls that must read known values:
      - majority class           -> exactly 1/20 (col) and 1/8 (row)
      - labels permuted within each frame (same tokens, positions shuffled)
                                 -> ~1/20 and ~1/8
      - the probe refit on the permuted labels scored on TRUE labels -> ~chance
  * reported: top-1 accuracy, |col error| <= 1 accuracy, n tokens, n clips, d
  * a linear probe that FAILS is not proof of absence (function class stated);
    a probe that SUCCEEDS shows position is linearly available, not that the
    decoder uses it.
"""
import json
import sys
import numpy as np

C = r"C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens"
STRIDE = 40
rng = np.random.default_rng(0)

ix = np.load(C + "/index.npz", allow_pickle=True)
clip_ord = ix["clip_ordinal"]
sha = ix["clip_sha12"]
tok = np.load(C + "/tokens_s32_fp16.npy", mmap_mode="r")
N, D, H, W = tok.shape
assert (D, H, W) == (704, 8, 20), tok.shape

sel = []
for c in np.unique(clip_ord):
    rows = np.flatnonzero(clip_ord == c)
    sel.extend(rows[::STRIDE].tolist())
sel = np.array(sorted(sel))
clip_of = clip_ord[sel]
bucket = np.array([int(sha[c], 16) for c in clip_of])
is_test = bucket % 4 == 0
is_val = (~is_test) & (bucket % 8 == 1)
is_fit = (~is_test) & (~is_val)


def gather(mask):
    r = sel[mask]
    x = np.asarray(tok[r], dtype=np.float32)            # [n, 704, 8, 20]
    n = x.shape[0]
    x = x.transpose(0, 2, 3, 1).reshape(n * H * W, D)  # token rows
    col = np.tile(np.arange(W), n * H)
    row = np.tile(np.repeat(np.arange(H), W), n)
    frame = np.repeat(np.arange(n), H * W)
    return x, col, row, frame, n


xf, colf, rowf, frf, nf = gather(is_fit)
xv, colv, rowv, frv, nv = gather(is_val)
xt, colt, rowt, frt, nt = gather(is_test)
mu = xf.mean(0)
sd = xf.std(0) + 1e-6
xf = (xf - mu) / sd
xv = (xv - mu) / sd
xt = (xt - mu) / sd


def permute_within_frame(lbl, frame, n):
    # tokens are laid out as contiguous blocks of H*W per frame (see gather)
    assert np.array_equal(frame, np.repeat(np.arange(n), H * W))
    return rng.permuted(lbl.reshape(n, H * W), axis=1).reshape(-1)


XB_FIT = np.hstack([xf, np.ones((xf.shape[0], 1), np.float32)])
GRAM_FIT = (XB_FIT.T.astype(np.float64) @ XB_FIT.astype(np.float64))   # computed once


def ridge_fit(x, y, k, lam):
    assert x is xf, "ridge_fit reuses the FIT Gram matrix"
    xty = np.zeros((XB_FIT.shape[1], k))
    for c in range(k):                                  # X^T one_hot(y), column by column
        xty[:, c] = XB_FIT[y == c].sum(0, dtype=np.float64)
    g = GRAM_FIT.copy()
    g[np.diag_indices_from(g)] += lam * x.shape[0]
    g[-1, -1] -= lam * x.shape[0]                      # no penalty on bias
    return np.linalg.solve(g, xty)


def predict(wt, x):
    return (x @ wt[:-1].astype(np.float32) + wt[-1].astype(np.float32)).argmax(1)


res = {"n_tokens": {"fit": int(xf.shape[0]), "val": int(xv.shape[0]), "test": int(xt.shape[0])},
       "n_frames": {"fit": int(nf), "val": int(nv), "test": int(nt)},
       "n_clips": {"fit": int(len(np.unique(clip_of[is_fit]))), "val": int(len(np.unique(clip_of[is_val]))),
                   "test": int(len(np.unique(clip_of[is_test])))},
       "d": D, "stride_frames": STRIDE}
for name, k, yf, yv, yt in (("column", W, colf, colv, colt), ("row", H, rowf, rowv, rowt)):
    best = None
    for lam in (1e-4, 1e-2, 1.0, 1e2):
        w = ridge_fit(xf, yf, k, lam)
        acc_v = float((predict(w, xv) == yv).mean())
        if best is None or acc_v > best[0]:
            best = (acc_v, lam, w)
    acc_v, lam, w = best
    p = predict(w, xt)
    r = {"lambda_picked_on_val": lam, "val_acc": round(acc_v, 4),
         "test_acc": round(float((p == yt).mean()), 4), "chance": round(1.0 / k, 4)}
    if name == "column":
        r["test_acc_within_1"] = round(float((np.abs(p - yt) <= 1).mean()), 4)
    # controls
    maj = np.bincount(yf, minlength=k).argmax()
    r["ctrl_majority_test_acc"] = round(float((yt == maj).mean()), 4)
    y_perm_fit = permute_within_frame(yf, frf, nf)
    w_perm = ridge_fit(xf, y_perm_fit, k, lam)
    r["ctrl_permuted_labels_probe_test_acc"] = round(float((predict(w_perm, xt) == yt).mean()), 4)
    y_perm_test = permute_within_frame(yt, frt, nt)
    r["ctrl_true_probe_vs_permuted_test_labels"] = round(float((p == y_perm_test).mean()), 4)
    res[name] = r
    print(name, json.dumps(r), flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print(json.dumps(res, indent=2))
