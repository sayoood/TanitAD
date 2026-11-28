#!/usr/bin/env python
"""closeread_fit.py -- card PREREG_V8F_PROOF_ADD77 (CLOSEREAD), sections 3 to 5: can a closing speed be read from the token reader's own per-frame outputs over a history?
Reads the npz files of closeread_dump.py (per reader seed: a fit file and a score file, and a lists file) and does arms L and N for H = 5, 10, 25, the beside rows,
the controls c1 to c7, the names, the result JSON and a short verdict file. Tier T0 (a probe on cached tokens of logged frames). Nothing is trained but a ridge and tiny nets on CPU.

  python closeread_fit.py --dump-dir D --out-dir O [--episode-map MAP.json] [--rulelog0 NPZ --rulelog1 NPZ --qmap0 JSON --qmap1 JSON] [--seeds 0,1] [--boot 2000] [--n-epochs 120]
  python closeread_fit.py --selftest
The output folder must not exist. The controls are printed first; if one fails, no name and no score value is written or printed (not to stdout either).
Last stdout line: ZZ CLOSEREAD rc <rc> controls_all_pass <bool>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT_S_DEFAULT = 0.2                 # frame period of the cache (dist_rule.py:25 DT = 0.2; used by LabelSource, v8f_t1_eval_rule.py:110)
RUN_MIN = 3                        # a label closing speed needs a run of >= 3 frames (v8f_t1_eval_rule.py:39 RUN_MIN)
HS = (5, 10, 25)
ARMS = ("L", "N")
N_SEEDS = (0, 1, 2)
BAR_R = 0.30
FIT_MIN_FRAMES, SCORE_MIN_FRAMES, SCORE_MIN_EPISODES = 2000, 500, 20
DUMP_KEYS = ("clips", "cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")
BANNED = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no lo" + "nger", "clo" + "sed loop", "driving " + "performance"]


# ============================================================================================ the held-out rule (tok_probe_lr.py:317, tok_reader.py:42)
def heldout_rule(clip_id):
    return int(hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:8], 16) % 10 == 0  # MUT-HELDOUT


def sha12(clip_id):
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


def recompute_heldout(manifest_ids, exclude):
    """the held-out list as the reader's trainer builds it: clips of the manifest minus the sha12-order slice [lo:hi] (v8f_mini_train.py:2720), then the rule; sorted (tok_probe_lr.py:311)."""
    lo, hi = (int(x) for x in exclude) if exclude else (0, 0)
    ex = set(sorted(manifest_ids, key=sha12)[lo:hi])
    return sorted(c for c in manifest_ids if c not in ex and heldout_rule(c)), sorted(ex)


def list_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# ============================================================================================ data
def load_dump(path):
    z = np.load(path, allow_pickle=False)
    for k in DUMP_KEYS:
        if k not in z.files:
            raise SystemExit("REFUSED: %s lacks %s" % (path, k))
    D = {k: np.asarray(z[k]) for k in z.files}
    D["clips"] = np.asarray([str(c) for c in D["clips"]])
    D["dt_s"] = float(D["dt_s"]) if "dt_s" in D else DT_S_DEFAULT
    check_dump(D, path)
    return D


def check_dump(D, name=""):
    n = len(D["frame"])
    for k in DUMP_KEYS[1:]:
        if len(D[k]) != n:
            raise SystemExit("REFUSED: %s array %s has %d rows, expected %d" % (name, k, len(D[k]), n))
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    same = cid[1:] == cid[:-1]
    if not (np.all(cid[1:] >= cid[:-1]) and np.all(fr[1:][same] == fr[:-1][same] + 1)):
        raise SystemExit("REFUSED: %s rows are not sorted by (clip, frame) with consecutive frames" % name)


def ep_index(D, ep_map):
    """per-row episode id as a string; ep_map: clip id -> episode id (None: the clip is its own episode, stated in the result)."""
    cl = D["clips"]
    e = np.asarray([str(ep_map.get(c, c)) if ep_map else c for c in cl])
    return e[np.asarray(D["cid"], np.int64)]


def build_rows(D, H, ep_map=None):
    """rows of one cell: the last frame r has a label with a closing speed and the H frames r-H+1..r of the clip all carry a read (full five-frame window). X [n, H, 3] = (gap read, presence, closing output), oldest first."""
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    n = len(cid)
    f3 = np.stack([np.asarray(D["gap"], np.float64), np.asarray(D["p"], np.float64), np.asarray(D["clos"], np.float64)], 1)
    ok = np.asarray(D["valid"], bool) & np.isfinite(f3).all(1)
    cs = np.concatenate([[0], np.cumsum(~ok)])
    idx = np.arange(H - 1, n)
    win = (cs[idx + 1] - cs[idx - H + 1] == 0) & (cid[idx] == cid[idx - H + 1]) & (fr[idx] - fr[idx - H + 1] == H - 1)
    lab = np.asarray(D["lab_present"], bool) & (np.asarray(D["lab_run"], np.int64) >= RUN_MIN) & np.isfinite(np.asarray(D["lab_closing"], np.float64))
    sel = idx[win & lab[idx]]
    X = np.stack([f3[sel - H + 1 + k] for k in range(H)], 1) if len(sel) else np.zeros((0, H, 3))
    y = np.asarray(D["lab_closing"], np.float64)[sel]  # MUT-TARGET
    ep = ep_index(D, ep_map)[sel] if len(sel) else np.asarray([], str)
    return {"X": X, "y": y, "clip": D["clips"][cid[sel]] if len(sel) else np.asarray([], str), "frame": fr[sel], "ep": ep, "H": H, "dt_s": float(D.get("dt_s", DT_S_DEFAULT))}


def assert_no_leak(fit_clips, score_clips, train_ids=None):
    """a hard refusal inside the fit path: no fit clip is a score clip, none is in the reader's training list, every fit clip is held out by the rule."""
    f, s = set(map(str, fit_clips)), set(map(str, score_clips))
    if f & s:
        raise SystemExit("REFUSED: %d fit clips are score clips" % len(f & s))
    if train_ids is not None and f & set(train_ids):
        raise SystemExit("REFUSED: %d fit clips are in the reader's training list" % len(f & set(train_ids)))
    if not all(heldout_rule(c) for c in f):
        raise SystemExit("REFUSED: a fit clip is not held out by the rule")


# ============================================================================================ estimators
def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    n, d = Z.shape
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * n * np.eye(d), Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def ridge_predict(m, X):
    return ((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"]


def split_by_clip(clips, frac=0.25, seed=0):
    """-> boolean mask of validation rows; a split of the fit set BY CLIP (no clip on both sides)."""
    u = np.array(sorted(set(map(str, clips))))
    rng = np.random.default_rng(seed)
    v = set(rng.permutation(u)[:max(1, int(math.ceil(frac * len(u))))].tolist())
    return np.asarray([str(c) in v for c in clips])


LAMBDAS = np.logspace(-6, 1, 15)


def select_ridge(Xtr, ytr, Xva, yva):
    best = None
    for lam in LAMBDAS:
        m = ridge_fit(Xtr, ytr, lam)
        e = float(np.mean((ridge_predict(m, Xva) - yva) ** 2))
        if best is None or e < best[0]:
            best = (e, float(lam))
    return best[1]


def train_mlp(Xtr, ytr, Xva, yva, seed, max_epochs=120, patience=10, bs=256, lr=1e-3):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda a: torch.from_numpy(((a - mu) / sd).astype(np.float32))
    xt, yt, xv, yv = f(Xtr), torch.from_numpy(ytr.astype(np.float32)), f(Xva), torch.from_numpy(yva.astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hub = torch.nn.HuberLoss(delta=1.0)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, ep_best = float("inf"), None, 0, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            hub(net(xt[j])[:, 0], yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(hub(net(xv)[:, 0], yv))
        if v < best - 1e-9:
            best, bad, ep_best = v, 0, ep + 1
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(f(X))[:, 0].numpy().astype(np.float64)
    return predict, {"best_epoch": ep_best, "val_huber": best}


def run_cell(arm, fit_rows, score_rows, n_epochs=120, perm_seed=None, split_seed=0, train_ids=None):
    """fit one arm on the fit rows of one reader seed, estimate the score rows. -> est [n_score], info. The validation split is a split of the FIT rows by clip; the score rows enter nowhere but the final estimate."""
    assert_no_leak(fit_rows["clip"], score_rows["clip"], train_ids)  # MUT-LEAKGUARD
    fit_rows = fit_rows  # MUT-FITROWS
    Xf, yf = fit_rows["X"].reshape(len(fit_rows["y"]), -1), np.array(fit_rows["y"], np.float64)
    Xs = score_rows["X"].reshape(len(score_rows["y"]), -1)
    if perm_seed is not None:
        yf = np.random.default_rng(perm_seed).permutation(yf)
    va = split_by_clip(fit_rows["clip"], 0.25, split_seed)
    Xtr, ytr, Xva, yva = Xf[~va], yf[~va], Xf[va], yf[va]  # MUT-RIDGE-VAL
    if arm == "L":
        lam = select_ridge(Xtr, ytr, Xva, yva)
        m = ridge_fit(Xf, yf, lam)
        return ridge_predict(m, Xs), {"lambda": lam, "n_fit": int(len(yf)), "n_val_clips": int(len(set(fit_rows["clip"][va])))}
    preds, infos = [], []
    for s in N_SEEDS:
        pr, inf = train_mlp(Xtr, ytr, Xva, yva, s, max_epochs=n_epochs)
        preds.append(pr(Xs))
        infos.append(inf)
    return np.mean(preds, 0), {"n_fit": int(len(yf)), "nets": infos}


def beside(score_rows, H):
    """the two rows that name nothing: the reader's own closing output at the frame and the two-point change of the gap read over H frames."""
    X, dt = score_rows["X"], score_rows["dt_s"]
    return {"reader_closing": X[:, -1, 2], "gap_two_point": (X[:, -1, 0] - X[:, 0, 0]) / ((H - 1) * dt)}


# ============================================================================================ statistics
KEYS = ("n", "sx", "sy", "sxx", "syy", "sxy", "sae", "say")


def episode_sums(est, y, epi, E):
    """per-episode sufficient statistics of one reader seed: n, sums for Pearson, sum |est - y| and sum |y| (the constant 0)."""
    bc = lambda w: np.bincount(epi, weights=w, minlength=E).astype(np.float64)
    return {"n": bc(np.ones(len(y))), "sx": bc(est), "sy": bc(y), "sxx": bc(est * est), "syy": bc(y * y), "sxy": bc(est * y), "sae": bc(np.abs(est - y)), "say": bc(np.abs(y))}


def pearson_sums(t):
    num = t["n"] * t["sxy"] - t["sx"] * t["sy"]
    den = np.sqrt(np.maximum(t["n"] * t["sxx"] - t["sx"] ** 2, 0.0) * np.maximum(t["n"] * t["syy"] - t["sy"] ** 2, 0.0))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def boot_totals(stats, B, seed):
    """two-level bootstrap: reader seeds with replacement, then evaluation episodes with replacement (the episode draw is shared by the drawn seeds: the same frames).
    -> point totals (all seeds, all episodes) and bootstrap totals [B] per key."""
    S, E = len(stats), len(stats[0]["n"])
    rng = np.random.default_rng(seed)
    W = np.zeros((B, E))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, E, E), minlength=E)
    sd = rng.integers(0, S, (B, S))
    point = {k: float(sum(st[k].sum() for st in stats)) for k in stats[0]}
    boot = {}
    for k in stats[0]:
        M = np.stack([W @ st[k] for st in stats])
        boot[k] = np.take_along_axis(M, sd.T, axis=0).sum(0)
    return point, boot


def pair_summary(stats, B, seed):
    """paired difference of two absolute errors on the same frames: (sum sae - sum say) / n; -> point, low, high (percentile 2.5 / 97.5)."""
    point, boot = boot_totals(stats, B, seed)
    d = (boot["sae"] - boot["say"]) / np.where(boot["n"] > 0, boot["n"], np.nan)
    return float((point["sae"] - point["say"]) / point["n"]), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def cell_summary(ests, score_rows_list, B, seed):
    """ests / score_rows_list: per reader seed. -> dict of r (pooled, per seed, interval), the two absolute errors, their paired difference with its interval."""
    allep = sorted(set(np.concatenate([r["ep"] for r in score_rows_list]).tolist()))
    emap = {e: i for i, e in enumerate(allep)}
    E = len(allep)
    stats = [episode_sums(est, r["y"], np.asarray([emap[e] for e in r["ep"]], np.int64), E) for est, r in zip(ests, score_rows_list)]
    point, boot = boot_totals(stats, B, seed)
    r_pool = float(pearson_sums(point))
    rb = pearson_sums(boot)
    r_seed = [float(pearson_sums({k: v.sum() for k, v in st.items()})) for st in stats]
    d_pt, d_lo, d_hi = pair_summary(stats, B, seed)
    return {"r_pooled": r_pool, "r_ci": [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))], "r_per_seed": r_seed, "mae_est": float(point["sae"] / point["n"]), "mae_zero": float(point["say"] / point["n"]),
            "diff_abs_err": d_pt, "diff_ci": [d_lo, d_hi], "n_frames": int(point["n"]), "n_episodes": int((sum(st["n"] for st in stats) > 0).sum()), "n_frames_per_seed": [int(st["n"].sum()) for st in stats]}


def name_of(arm, H, s):
    ok = bool(s["r_pooled"] >= BAR_R and s["r_ci"][0] > 0 and all(r >= BAR_R for r in s["r_per_seed"]) and s["diff_ci"][1] < 0)
    return "CLOSEREAD-%s-H%d-%s" % (arm, H, "READS-CLOSING" if ok else "NO-CLOSING"), ok


def a_name(oks):
    return "CLOSEREAD-A-CLOSING-IS-READABLE" if any(oks) else "CLOSEREAD-NOT-READABLE-FROM-READER-OUTPUTS"


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================================================ controls
def c1_no_leak(fit_dumps, score_dumps, lists):
    """per reader seed: fit clips are not score clips, not in the reader's training list, the rule recomputed from the manifest gives the clips of the dump, count and list hash equal the reader's train_meta."""
    out, ok = {}, True
    for k in sorted(fit_dumps):
        D, S, L = fit_dumps[k], score_dumps[k], lists[k]
        fit_ids = sorted(set(D["clips"].tolist()))
        rec, ex = recompute_heldout(L["manifest_ids"], L.get("exclude"))
        r = {"n_fit_clips": len(fit_ids), "shared_with_score": len(set(fit_ids) & set(S["clips"].tolist())), "in_training_list": len(set(fit_ids) & set(L["train_ids"])),
             "rule_list_equals_fit": rec == fit_ids, "n_equals_meta": len(rec) == int(L["meta_n_heldout"]) and len(fit_ids) == int(L["meta_n_heldout"]),
             "sha_equals_meta": (list_sha256(rec) == L["meta_heldout_sha256"]) if L.get("meta_heldout_sha256") else None,
             "train_list_is_manifest_minus_heldout_minus_excluded": sorted(L["train_ids"]) == sorted(c for c in L["manifest_ids"] if c not in set(rec) and c not in set(ex))}
        r["pass"] = bool(r["shared_with_score"] == 0 and r["in_training_list"] == 0 and r["rule_list_equals_fit"] and r["n_equals_meta"] and r["sha_equals_meta"] is not False and r["train_list_is_manifest_minus_heldout_minus_excluded"])
        out["seed%d" % k] = r
        ok = ok and r["pass"]
    return {"pass": bool(ok), "detail": out}


def two_point_label(D, lag, dt):
    """per row r with frames r-lag..r of one clip all carrying a lead label: (gap[r] - gap[r-lag]) / (lag * dt); NaN elsewhere."""
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    pres = np.asarray(D["lab_present"], bool) & np.isfinite(np.asarray(D["lab_gap"], np.float64))
    g = np.asarray(D["lab_gap"], np.float64)
    n = len(cid)
    out = np.full(n, np.nan)
    cs = np.concatenate([[0], np.cumsum(~pres)])
    idx = np.arange(lag, n)
    okk = (cs[idx + 1] - cs[idx - lag] == 0) & (cid[idx] == cid[idx - lag]) & (fr[idx] - fr[idx - lag] == lag)
    out[idx[okk]] = (g[idx[okk]] - g[idx[okk] - lag]) / (lag * dt)
    return out


def c2_definition(D, bar=0.70):
    """the two-point change of the LABEL gap over 5 frames against the label closing speed: r >= 0.70, positive; shifted by +-10 frames the correlation is lower."""
    dt = float(D.get("dt_s", DT_S_DEFAULT))
    ch = two_point_label(D, 4, dt)
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    lab = np.asarray(D["lab_present"], bool) & (np.asarray(D["lab_run"], np.int64) >= RUN_MIN) & np.isfinite(np.asarray(D["lab_closing"], np.float64))
    y = np.asarray(D["lab_closing"], np.float64)
    m = np.isfinite(ch) & lab
    r0 = pearson(ch[m], y[m])
    out = {"r0": r0, "n": int(m.sum())}
    shifted_ok = True
    for s in (10, -10):
        n = len(cid)
        i = np.arange(n)
        j = i + s
        v = (j >= 0) & (j < n)
        i, j = i[v], j[v]
        v = (cid[i] == cid[j]) & (fr[j] - fr[i] == s) & np.isfinite(ch[i]) & lab[j]
        rs = pearson(ch[i[v]], y[j[v]])
        out["r_shift%+d" % s] = rs
        out["n_shift%+d" % s] = int(v.sum())
        shifted_ok = shifted_ok and bool(np.isfinite(rs) and rs < r0)
    out["pass"] = bool(np.isfinite(r0) and r0 >= bar and shifted_ok)
    return out


def qmap_apply(g, xs, ys):
    """g_used = min(g, Q(g)) of qcal_map.apply_map (distrule/qcal_map.py:33); NaN stays NaN."""
    g = np.asarray(g, np.float64)
    out = np.array(g, copy=True)
    fin = np.isfinite(g)
    gg = g[fin]
    q = np.interp(gg, xs, ys)
    lo = gg < xs[0]
    q[lo] = ys[0] * gg[lo] / xs[0]
    hi = gg > xs[-1]
    q[hi] = gg[hi] - (xs[-1] - ys[-1])
    out[fin] = np.minimum(gg, q)
    return out


def c3_dump_is_rule_reader(score_dump, rule_log, qmap, tol=1e-3):
    """presence and gap read of the rule log (p_src; g_src = min(g, Q(g)) with the map of the run) equal the per-frame dump at the same (clip, frame): largest difference <= tol on every row with the window in the clip."""
    key = {(int(c), int(f)): i for i, (c, f) in enumerate(zip(score_dump["clip_index"], score_dump["frame"]))}
    ci, j = np.asarray(rule_log["ci"], np.int64), np.asarray(rule_log["j"], np.int64)
    kk = np.asarray(rule_log["k"], np.int64) if "k" in rule_log else np.zeros(len(ci), np.int64)
    rows = np.array([key.get((int(c), int(f)), -1) for c, f in zip(ci, j)], np.int64)
    inclip = rows >= 0
    r = {"n_log_rows": int(len(ci)), "n_in_clip": int(inclip.sum())}
    if not inclip.any():
        r["pass"] = False
        return r
    rr = rows[inclip]
    pl, gl = np.asarray(rule_log["p_src"], np.float64)[inclip], np.asarray(rule_log["g_src"], np.float64)[inclip]
    valid = np.asarray(score_dump["valid"], bool)[rr]
    pd_, gd = np.asarray(score_dump["p"], np.float64)[rr], qmap_apply(np.asarray(score_dump["gap"], np.float64)[rr], *qmap)
    fin_agree = bool(np.array_equal(np.isfinite(pl), valid) and np.array_equal(np.isfinite(gl), valid))
    both = valid & np.isfinite(pl) & np.isfinite(gl)
    dp = np.abs(pl[both] - pd_[both])
    dg = np.abs(gl[both] - gd[both])
    first = both & (kk[inclip] == 0)
    r.update({"n_compared": int(both.sum()), "n_first_tick": int(first.sum()), "finite_where_window_exists": fin_agree,
              "max_abs_diff_p": float(dp.max()) if len(dp) else float("nan"), "max_abs_diff_gap_after_map": float(dg.max()) if len(dg) else float("nan"), "tol": tol})
    r["pass"] = bool(fin_agree and both.sum() > 0 and first.sum() > 0 and dp.max() <= tol and dg.max() <= tol)
    return r


def c7_counts(fit_rows, score_rows):
    nf, ns = len(fit_rows["y"]), len(score_rows["y"])
    ne = len(set(score_rows["ep"].tolist()))
    return {"fit_frames": int(nf), "score_frames": int(ns), "score_episodes": int(ne), "pass": bool(nf >= FIT_MIN_FRAMES and ns >= SCORE_MIN_FRAMES and ne >= SCORE_MIN_EPISODES)}


def c6_estimator(B=300, seed=5):
    """a pair with itself reads exactly 0; a swap reads the negative (point and interval)."""
    rng = np.random.default_rng(seed)
    E, S = 25, 2
    stats = []
    for _ in range(S):
        y = rng.normal(0, 1, 400)
        est = y + rng.normal(0, 0.5, 400)
        ep = np.arange(400) % E
        st = episode_sums(est, y, ep, E)
        stats.append(st)
    selfp = [{"n": st["n"], "sae": st["sae"], "say": st["sae"]} for st in stats]
    p0 = pair_summary(selfp, B, seed)
    fwd = pair_summary([{"n": st["n"], "sae": st["sae"], "say": st["say"]} for st in stats], B, seed)
    swp = pair_summary([{"n": st["n"], "sae": st["say"], "say": st["sae"]} for st in stats], B, seed)
    ok_self = p0 == (0.0, 0.0, 0.0)
    ok_swap = bool(abs(swp[0] + fwd[0]) < 1e-12 and abs(swp[1] + fwd[2]) < 1e-12 and abs(swp[2] + fwd[1]) < 1e-12 and fwd[0] != 0.0)
    return {"self_pair": list(p0), "swap_equals_negative": ok_swap, "pass": bool(ok_self and ok_swap)}


# ---- synthetic dumps (controls c5, the selftest and the tests)
def make_synth(seed=0, n_fit=60, n_score=60, frames=101, signal=True, reads_noise=False, dt=0.2, noise_gap=1.0, ep_size=2, seeds=(0, 1)):
    """-> (fit dumps, score dumps, ep_map) keyed by reader seed. Gap of the lead changes at a known rate per clip; the read is the gap plus noise (signal) or pure noise (reads_noise). The label closing speed is the rate."""
    rng = np.random.default_rng(seed)

    def mk(ids, off, rs):
        cols = {k: [] for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
        for c, cid_ in enumerate(ids):
            amp, per, ph = rng.normal(0, 1.0), rng.uniform(150, 250), rng.uniform(0, 2 * np.pi)
            g0 = rng.uniform(25, 35)
            t = np.arange(frames)
            rate = amp * np.sin(2 * np.pi * t / per + ph)
            g = np.clip(g0 + np.cumsum(rate) * dt, 2.0, 58.0)
            rate_eff = np.gradient(g, dt)
            reads = g + rng.normal(0, noise_gap, frames)
            if reads_noise:
                reads = rng.uniform(5, 55, frames)
            cols["cid"].append(np.full(frames, c))
            cols["clip_index"].append(np.full(frames, off + c))
            cols["frame"].append(t)
            cols["p"].append(np.clip(0.95 + rng.normal(0, 0.02, frames), 0, 1))
            cols["gap"].append(reads)
            cols["clos"].append(rng.normal(0, 1.0, frames))
            cols["valid"].append(t >= 4)
            cols["lab_present"].append(np.ones(frames, bool))
            cols["lab_gap"].append(g)
            cols["lab_closing"].append(rate_eff + rng.normal(0, 0.05, frames))
            cols["lab_run"].append(np.full(frames, 5))
        D = {k: np.concatenate(v) for k, v in cols.items()}
        D["clips"] = np.asarray(ids)
        D["dt_s"] = dt
        return D
    fit_ids, score_ids, i = [], [], 0
    while len(fit_ids) < n_fit or len(score_ids) < n_score:
        c = "synthclip-%06d" % i
        i += 1
        if heldout_rule(c):
            if len(fit_ids) < n_fit:
                fit_ids.append(c)
        elif len(score_ids) < n_score:
            score_ids.append(c)
    fits, scores = {}, {}
    for k in seeds:
        fits[k] = mk(fit_ids, 0, k)
        scores[k] = mk(score_ids, 0, k)
    # the score labels and clip ids do not depend on the reader seed: copy those of the first seed and redraw only the reads
    k0 = seeds[0]
    for k in seeds[1:]:
        for kk in ("lab_gap", "lab_closing", "lab_present", "lab_run", "valid", "clips", "cid", "clip_index", "frame"):
            scores[k][kk] = scores[k0][kk]
        scores[k]["gap"] = scores[k0]["lab_gap"] + rng.normal(0, noise_gap, len(scores[k0]["gap"])) if not reads_noise else rng.uniform(5, 55, len(scores[k0]["gap"]))
    ep_map = {c: "ep%03d" % (j // ep_size) for j, c in enumerate(score_ids)}
    return fits, scores, ep_map


def c5_planted_signal(n_epochs=20, B=400):
    """the tool reads a planted signal: arm L at H = 25 on sequences whose gap changes at a known rate: r >= 0.90 and READS-CLOSING; on noise alone NO-CLOSING."""
    res = {}
    for tag, kw in (("signal", {}), ("noise", {"reads_noise": True})):
        fits, scores, epm = make_synth(seed=11, **kw)
        fr = {k: build_rows(fits[k], 25) for k in fits}
        sr = {k: build_rows(scores[k], 25, epm) for k in scores}
        ks = sorted(fits)
        ests = [run_cell("L", fr[k], sr[k], n_epochs)[0] for k in ks]
        s = cell_summary(ests, [sr[k] for k in ks], B, 3)
        nm, ok = name_of("L", 25, s)
        res[tag] = {"r_pooled": s["r_pooled"], "name": nm, "reads": ok}
    res["pass"] = bool(res["signal"]["r_pooled"] >= 0.90 and res["signal"]["reads"] and not res["noise"]["reads"])
    return res


def c4_negative(fit_rows, score_rows, n_epochs, B, seed=7):
    """both arms fitted on closing labels permuted over the fit frames (history 25): the pooled r has an interval that includes 0. Only the flag is reported (no value of the score set)."""
    ok = {}
    for arm in ARMS:
        ests = [run_cell(arm, fit_rows[k], score_rows[k], n_epochs, perm_seed=100 + k)[0] for k in sorted(fit_rows)]
        s = cell_summary(ests, [score_rows[k] for k in sorted(score_rows)], B, seed)
        ok[arm] = bool(s["r_ci"][0] <= 0.0 <= s["r_ci"][1])
    return {"pass": bool(all(ok.values())), "includes_zero": ok}


# ============================================================================================ the run
def load_dir(dump_dir, seeds):
    fits, scores, lists = {}, {}, {}
    for k in seeds:
        fits[k] = load_dump(os.path.join(dump_dir, "reader_s%d_fit.npz" % k))
        scores[k] = load_dump(os.path.join(dump_dir, "reader_s%d_score.npz" % k))
        lists[k] = json.load(open(os.path.join(dump_dir, "reader_s%d_lists.json" % k), encoding="utf-8"))
    return fits, scores, lists


def run_all(fits, scores, lists, ep_map, rulelogs, out_dir, B=2000, n_epochs=120, quiet=False, c5_epochs=20):
    """-> (rc, controls_all_pass). Writes CLOSEREAD_CONTROLS.json always; the result and verdict files only when every control passes."""
    if os.path.exists(out_dir):
        print("REFUSED: output folder exists: " + out_dir)
        return 2, False
    os.makedirs(out_dir)
    ks = sorted(fits)
    P = (lambda *a: None) if quiet else print
    controls = {}
    P("== controls (printed before any estimate) ==")
    controls["c1"] = c1_no_leak(fits, scores, lists)
    P("c1 no leak (fit clips, training list, held-out rule, count and hash): %s" % ("PASS" if controls["c1"]["pass"] else "FAIL"))
    same_lab = all(np.array_equal(scores[ks[0]][q], scores[k][q], equal_nan=True) for k in ks[1:] for q in ("lab_gap", "lab_closing", "lab_present", "lab_run", "clip_index", "frame", "valid"))
    controls["c2"] = c2_definition(scores[ks[0]])
    controls["c2"]["labels_equal_across_seeds"] = bool(same_lab)
    controls["c2"]["pass"] = bool(controls["c2"]["pass"] and same_lab)
    P("c2 definition and clock (two-point change of the label gap against the label closing speed; shifted by +-10 frames): %s" % ("PASS" if controls["c2"]["pass"] else "FAIL"))
    c3 = {}
    for k in ks:
        rl = rulelogs.get(k)
        c3["seed%d" % k] = c3_dump_is_rule_reader(scores[k], rl["log"], rl["qmap"]) if rl else {"pass": False, "reason": "no rule log given"}
    controls["c3"] = {"pass": bool(all(v["pass"] for v in c3.values())), "detail": c3}
    P("c3 the dump is the reader the rule gets: %s" % ("PASS" if controls["c3"]["pass"] else "FAIL"))
    fr = {H: {k: build_rows(fits[k], H) for k in ks} for H in HS}
    sr = {H: {k: build_rows(scores[k], H, ep_map) for k in ks} for H in HS}
    c7 = {"H%d_seed%d" % (H, k): c7_counts(fr[H][k], sr[H][k]) for H in HS for k in ks}
    controls["c7"] = {"pass": bool(all(v["pass"] for v in c7.values())), "detail": c7}
    P("c7 fit and score sets are large enough: %s" % ("PASS" if controls["c7"]["pass"] else "FAIL"))
    controls["c6"] = c6_estimator()
    P("c6 estimator (self pair reads 0, swap reads the negative): %s" % ("PASS" if controls["c6"]["pass"] else "FAIL"))
    controls["c5"] = c5_planted_signal(c5_epochs, min(B, 400))
    P("c5 the tool reads a planted signal and no signal in noise: %s" % ("PASS" if controls["c5"]["pass"] else "FAIL"))
    if controls["c1"]["pass"] and controls["c7"]["pass"]:
        controls["c4"] = c4_negative(fr[25], sr[25], n_epochs, B)
    else:
        controls["c4"] = {"pass": False, "reason": "not run: c1 or c7 failed"}
    P("c4 negative control (permuted fit labels, both arms, H = 25): %s" % ("PASS" if controls["c4"]["pass"] else "FAIL"))
    allp = all(c["pass"] for c in controls.values())
    json.dump({"controls": controls, "controls_all_pass": bool(allp)}, open(os.path.join(out_dir, "CLOSEREAD_CONTROLS.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    if not allp:
        P("a control did not pass: no name and no score value is written or printed. The failure goes into TIMES_NOTE.md before any repair; the re-read goes into a new folder.")
        return 1, False
    cells, names, oks, beside_rows = {}, {}, [], {}
    for H in HS:
        for arm in ARMS:
            ests, info = [], {}
            for k in ks:
                e, inf = run_cell(arm, fr[H][k], sr[H][k], n_epochs, train_ids=lists[k]["train_ids"])
                ests.append(e)
                info["seed%d" % k] = inf
            s = cell_summary(ests, [sr[H][k] for k in ks], B, 1000 + H)
            nm, ok = name_of(arm, H, s)
            cells["%s_H%d" % (arm, H)] = dict(s, fit_info=info, name=nm)
            names["%s_H%d" % (arm, H)] = nm
            oks.append(ok)
        bs = [beside(sr[H][k], H) for k in ks]
        beside_rows["H%d" % H] = {nm: cell_summary([b[nm] for b in bs], [sr[H][k] for k in ks], B, 2000 + H) for nm in ("reader_closing", "gap_two_point")}
    top = a_name(oks)
    npass = int(sum(oks))
    res = {"card": "PREREG_V8F_PROOF_ADD77", "tier": "T0", "dt_s": float(scores[ks[0]].get("dt_s", DT_S_DEFAULT)), "reader_seeds": ks, "bootstrap": {"B": B, "levels": "reader seeds, then evaluation episodes", "interval": "percentile 2.5 / 97.5"},
           "episode_source": "map file" if ep_map else "the clip is its own episode (no map given)", "cells": cells, "beside_rows_no_names": beside_rows, "names": names, "A_name": top, "n_cells_passing_of_six": npass,
           "note_six_cells": "six arm-history cells are read; with six cells one may pass by chance: a single passing cell is one of six, and the wrapper arm that follows is its test on rollout windows." if npass == 1 else "six arm-history cells are read.",
           "families": {"LONGITUDINAL": "not measured by this read", "LATERAL": "not measured by this read", "TACTICAL": "not measured by this read", "STRATEGIC": "NOT COMPUTED (n = 0; a token probe reads no strategic quantity)"}, "controls_all_pass": True}
    json.dump(res, open(os.path.join(out_dir, "CLOSEREAD_RESULT.json"), "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    lines = ["# CLOSEREAD verdict (card ADD77, tier T0: a probe on cached tokens of logged frames)", "", "All controls pass. Frame period %.3f s. Episodes: %s." % (res["dt_s"], res["episode_source"]), "",
             "| cell | r pooled | r 95 % | r per seed | MAE est | MAE zero | paired diff (est - 0) 95 % | name |", "|---|---|---|---|---|---|---|---|"]
    for key, c in cells.items():
        lines.append("| %s | %+.3f | [%+.3f, %+.3f] | %s | %.3f | %.3f | %+.3f [%+.3f, %+.3f] | %s |" % (key, c["r_pooled"], c["r_ci"][0], c["r_ci"][1], ", ".join("%+.3f" % r for r in c["r_per_seed"]), c["mae_est"], c["mae_zero"],
                                                                                                   c["diff_abs_err"], c["diff_ci"][0], c["diff_ci"][1], c["name"]))
    lines += ["", "Beside (no names): the reader's own closing output and the two-point change of the gap read over H frames.", "", "| H | row | r pooled | r per seed | MAE | MAE zero |", "|---|---|---|---|---|---|"]
    for h, d in beside_rows.items():
        for nm, c in d.items():
            lines.append("| %s | %s | %+.3f | %s | %.3f | %.3f |" % (h, nm, c["r_pooled"], ", ".join("%+.3f" % r for r in c["r_per_seed"]), c["mae_est"], c["mae_zero"]))
    lines += ["", "**%s** (%d of six cells read READS-CLOSING). %s" % (top, npass, res["note_six_cells"]), "",
              "Families: this probe measures none of them. LONGITUDINAL, LATERAL, TACTICAL: not measured by this read. STRATEGIC: NOT COMPUTED (n = 0)."]
    open(os.path.join(out_dir, "CLOSEREAD_VERDICT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    P("== cells ==")
    for key, c in cells.items():
        P("%s r %+.3f [%+.3f, %+.3f] diff %+.3f [%+.3f, %+.3f] -> %s" % (key, c["r_pooled"], c["r_ci"][0], c["r_ci"][1], c["diff_abs_err"], c["diff_ci"][0], c["diff_ci"][1], c["name"]))
    P(top)
    return 0, True


def load_rulelog(path_log, path_qmap):
    z = np.load(path_log, allow_pickle=True)
    q = json.load(open(path_qmap, encoding="utf-8"))
    return {"log": {k: z[k] for k in z.files}, "qmap": (np.asarray(q["xs"], np.float64), np.asarray(q["ys"], np.float64))}


# ============================================================================================ the self-test
def synth_lists(fits, seed=0, n_manifest=1000, exclude=(40, 50)):
    """a manifest whose held-out list (after the exclusion) is exactly the clips of the synthetic fit dump: ids are drawn so that the rule gives them; the rest are training clips."""
    out = {}
    for k, D in fits.items():
        fit_ids = sorted(set(D["clips"].tolist()))
        man, i = list(fit_ids), 0
        while len(man) < n_manifest:
            c = "synthtrain-%06d" % i
            i += 1
            if not heldout_rule(c):
                man.append(c)
        ex = set(sorted(man, key=sha12)[exclude[0]:exclude[1]])
        if ex & set(fit_ids):
            raise SystemExit("synthetic manifest: an excluded clip is a fit clip; change the exclusion")
        held, ex = recompute_heldout(man, exclude)
        out[k] = {"manifest_ids": man, "exclude": list(exclude), "train_ids": sorted(c for c in man if c not in set(held) and c not in ex), "meta_n_heldout": len(held), "meta_heldout_sha256": list_sha256(held)}
        assert held == fit_ids
    return out


def synth_rulelogs(scores, qmap=((5.0, 20.0, 60.0), (3.0, 15.0, 50.0)), perturb=0.0):
    xs, ys = np.asarray(qmap[0]), np.asarray(qmap[1])
    out = {}
    for k, S in scores.items():
        rows = np.where(S["valid"] & (S["frame"] % 10 == 0))[0]
        g = qmap_apply(S["gap"][rows], xs, ys) + perturb
        out[k] = {"log": {"ci": S["clip_index"][rows], "j": S["frame"][rows], "k": (S["frame"][rows] - 10) % 30, "p_src": S["p"][rows].astype(np.float32).astype(np.float64), "g_src": g.astype(np.float32).astype(np.float64)}, "qmap": (xs, ys)}
        out[k]["log"]["k"] = np.where(np.arange(len(rows)) % 3 == 0, 0, 10)
    return out


def selftest():
    import io
    import contextlib
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        if not cond:
            fails.append(name)
    srcs = [os.path.join(HERE, f) for f in sorted(os.listdir(HERE)) if f.endswith((".py", ".md", ".sh", ".json")) and not f.startswith("tmp")]
    bad = [(os.path.basename(p), x) for p in srcs for x in BANNED if x in open(p, encoding="utf-8", errors="replace").read().lower()]
    chk("the text of every file of the folder holds none of the banned words (%d files)" % len(srcs), not bad)
    fits, scores, epm = make_synth(seed=3)
    lists = synth_lists(fits)
    rl = synth_rulelogs(scores)
    chk("c1 passes on clean synthetic lists", c1_no_leak(fits, scores, lists)["pass"])
    f2 = {k: dict(v) for k, v in fits.items()}
    f2[0]["clips"] = f2[0]["clips"].copy()
    f2[0]["clips"][0] = scores[0]["clips"][0]
    chk("c1 fails on a planted shared clip id", not c1_no_leak(f2, scores, lists)["pass"])
    l2 = {k: dict(v) for k, v in lists.items()}
    l2[0]["meta_heldout_sha256"] = "0" * 64
    chk("c1 fails on a wrong list hash", not c1_no_leak(fits, scores, l2)["pass"])
    chk("c2 passes on clean synthetic labels", c2_definition(scores[0])["pass"])
    sh = dict(scores[0])
    sh["lab_closing"] = np.roll(scores[0]["lab_closing"], 1000)
    chk("c2 fails on labels from another clock", not c2_definition(sh)["pass"])
    chk("c3 passes on a synthetic rule log", c3_dump_is_rule_reader(scores[0], rl[0]["log"], rl[0]["qmap"])["pass"])
    rl2 = synth_rulelogs(scores, perturb=0.01)
    chk("c3 fails on a planted gap difference of 0.01", not c3_dump_is_rule_reader(scores[0], rl2[0]["log"], rl2[0]["qmap"])["pass"])
    chk("c6 estimator", c6_estimator()["pass"])
    c5 = c5_planted_signal(20, 400)
    chk("c5 planted signal read (r %.2f) and noise not read" % c5["signal"]["r_pooled"], c5["pass"])
    for tag, kw, want_top in (("signal in the reads", {}, "CLOSEREAD-A-CLOSING-IS-READABLE"), ("noise in the reads", {"reads_noise": True}, "CLOSEREAD-NOT-READABLE-FROM-READER-OUTPUTS")):
        f_, s_, e_ = make_synth(seed=4, **kw)
        l_ = synth_lists(f_)
        r_ = synth_rulelogs(s_)
        d = os.path.join(tempfile.mkdtemp(prefix="closeread_selftest_"), "out")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, allp = run_all(f_, s_, l_, e_, r_, d, B=300, n_epochs=25, quiet=False, c5_epochs=10)
        res = json.load(open(os.path.join(d, "CLOSEREAD_RESULT.json"), encoding="utf-8")) if os.path.exists(os.path.join(d, "CLOSEREAD_RESULT.json")) else {}
        chk("end to end, %s: controls pass, rc 0, top name %s" % (tag, want_top), rc == 0 and allp and res.get("A_name") == want_top)
        if tag.startswith("noise"):
            chk("noise in the reads: no cell reads closing", not any(v.endswith("READS-CLOSING") for v in res.get("names", {}).values()))
        chk("the output folder that exists is refused (%s)" % tag, run_all(f_, s_, l_, e_, r_, d, quiet=True)[0] == 2)
    # a failing control: labels that do not follow the label gap (c2 fails): no name, no score value in the output
    f_, s_, e_ = make_synth(seed=4)
    for k in s_:
        s_[k]["lab_closing"] = np.random.default_rng(1).normal(0, 1, len(s_[k]["lab_closing"]))
    d = os.path.join(tempfile.mkdtemp(prefix="closeread_selftest_"), "out")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc, allp = run_all(f_, s_, synth_lists(f_), e_, synth_rulelogs(s_), d, B=300, n_epochs=25, c5_epochs=10)
    out = buf.getvalue()
    chk("a failing control: rc 1, no result file, no name and no score line on stdout", rc == 1 and not allp and not os.path.exists(os.path.join(d, "CLOSEREAD_RESULT.json")) and "CLOSEREAD-" not in out.replace("CLOSEREAD-NOT", "X") and " r +" not in out and "== cells ==" not in out)
    ok = not fails
    print("ZZ CLOSEREAD rc %d controls_all_pass %s" % (0 if ok else 1, ok))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dump-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--episode-map", default="")
    ap.add_argument("--seeds", default="0,1")
    for i in (0, 1):
        ap.add_argument("--rulelog%d" % i, default="")
        ap.add_argument("--qmap%d" % i, default="")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--n-epochs", type=int, default=120)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.dump_dir or not a.out_dir:
        print("REFUSED: --dump-dir and --out-dir are needed")
        print("ZZ CLOSEREAD rc 2 controls_all_pass False")
        return 2
    seeds = [int(x) for x in a.seeds.split(",")]
    fits, scores, lists = load_dir(a.dump_dir, seeds)
    epm = json.load(open(a.episode_map, encoding="utf-8")) if a.episode_map else None
    rls = {k: load_rulelog(getattr(a, "rulelog%d" % k), getattr(a, "qmap%d" % k)) for k in seeds if getattr(a, "rulelog%d" % k, "") and getattr(a, "qmap%d" % k, "")}
    rc, allp = run_all(fits, scores, lists, epm, rls, a.out_dir, B=a.boot, n_epochs=a.n_epochs)
    print("ZZ CLOSEREAD rc %d controls_all_pass %s" % (rc, allp))
    return rc


if __name__ == "__main__":
    sys.exit(main())
