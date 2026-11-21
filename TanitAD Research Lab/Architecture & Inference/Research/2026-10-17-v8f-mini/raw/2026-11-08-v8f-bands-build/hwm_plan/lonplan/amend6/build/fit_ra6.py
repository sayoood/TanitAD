"""AMEND_HWMPLAN_6, section 4: the RA fit, the K-control and the FIT-CURVE row. CPU only. The procedure is amend1_kstep.py's frozen-readout procedure (pure numpy below; no model needed except for `scale_actions`).
  python fit_ra6.py kcontrol --chassis s0|s1     the fit on the ORIGINAL 141 clips must reproduce AMEND_5's readout (lambda, n_pairs, mu, sd, W, ym within 1e-8); the 140-clip mutation must read RED
  python fit_ra6.py fit --chassis s0|s1 [--new-clips N]   fit on 141 + the first N new clips (default: the largest complete checkpoint of CHUNKS.json); writes readout_in/readout_K10_R35_<k>.pt + RA6_FIT_<k>.json
  python fit_ra6.py curve --chassis s0|s1        FIT-CURVE: held-out R2 / r at the nested totals (beside, no bar)
"""
import argparse, hashlib, io, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import numpy as np

K = 10
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LAT141 = "F:/TanitAD-scratch/hwm_plan/latents_train_R35_{k}.pt"
CK = "C:/Users/Admin/t2v8f_pulls/ladder_R35_{k}/ckpt_last.pt"
RA5 = "F:/TanitAD-scratch/actrsp/lonplan/amend5/readout_in/readout_K10_R35_{k}.pt"
OUT_DIR = ROOT + "/readout_in"
CHECKPOINTS_NEW = (141, 423, 859)


# ---------------------------------------------------------------------------------------------- pure
def pairs(LT, scale_fn, k=K):
    """-> Dr (n,128) f64, Yf (n,1) f64, EP (n,) clip index (clips sorted by id), clips. As amend1_kstep.py: frame table (first window wins), 11 consecutive frames, finite actions, mean scaled acceleration over 10 transitions."""
    import torch
    tag = list(LT["Z1"])[0]
    Zw, Aw, okw = LT["Z1"][tag].float(), scale_fn(LT["A_raw"].float()), LT["ok"].bool()
    tw, eidw = np.asarray(LT["t"]), list(LT["eid"])
    clips = sorted(set(eidw)); cix = {c: i for i, c in enumerate(clips)}
    tab = {}
    for w in range(Zw.shape[0]):
        for j in range(Zw.shape[1]):
            if bool(okw[w, j]):
                tab.setdefault((cix[eidw[w]], int(tw[w]) - (Zw.shape[1] - 1) + j), (Zw[w, j], Aw[w, j]))
    byc = {}
    for (c, f) in tab:
        byc.setdefault(c, []).append(f)
    EP, ZT, ZK, AB = [], [], [], []
    for ci in range(len(clips)):
        fr = sorted(byc.get(ci, []))
        pos = {int(f): i for i, f in enumerate(fr)}
        Z = torch.stack([tab[(ci, f)][0] for f in fr]); A = torch.stack([tab[(ci, f)][1] for f in fr])
        for i, f in enumerate(fr):
            f = int(f)
            if all((f + m) in pos for m in range(k + 1)):
                js = [pos[f + m] for m in range(k)]
                if torch.isfinite(A[js]).all():
                    EP.append(ci); ZT.append(Z[i]); ZK.append(Z[pos[f + k]]); AB.append(A[js].mean(0))
    zt, zk, ab = (torch.stack(x).numpy().astype(np.float64) for x in (ZT, ZK, AB))
    return zk - zt, ab[:, :1], np.array(EP), clips


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd; ym = y.mean(0)
    return mu, sd, np.linalg.solve(Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1]), Xs.T @ (y - ym)), ym


def ridge_pred(m, X):
    return ((X - m[0]) / m[1]) @ m[2] + m[3]


def fold_map(n_clips, EP):
    return (np.random.RandomState(0).permutation(n_clips) % 5)[EP]


def choose_lam(X, y, fo):
    best, bl = np.inf, None
    for lam in LAMS:
        err = sum(((ridge_pred(ridge_fit(X[fo != g], y[fo != g], lam), X[fo == g]) - y[fo == g]) ** 2).sum() for g in np.unique(fo))
        if err < best:
            best, bl = err, lam
    return bl


def fit_ra(Dr, Yf, EP, n_clips):
    lam = choose_lam(Dr, Yf, fold_map(n_clips, EP))
    mu, sd, W, ym = ridge_fit(Dr, Yf, lam)
    return dict(lam=lam, mu=mu, sd=sd, W=W, ym=ym, n_pairs=int(len(Dr)), n_clips=int(n_clips))


def compare(a, b, tol=1e-8):
    d = dict(lam_equal=bool(a["lam"] == b["lam"]), n_pairs_equal=bool(a["n_pairs"] == b["n_pairs"]), n_clips_equal=bool(a["n_clips"] == b["n_clips"]),
             **{f"max_abs_d{x}": float(np.abs(np.asarray(a[x], np.float64) - np.asarray(b[x], np.float64)).max()) for x in ("mu", "sd", "W", "ym")}, tol=tol)
    d["ok"] = bool(d["lam_equal"] and d["n_pairs_equal"] and d["n_clips_equal"] and max(d[f"max_abs_d{x}"] for x in ("mu", "sd", "W", "ym")) <= tol)
    return d


def kcontrol_pure(LT141, scale_fn, ra5):
    """ra5: dict(lam, mu, sd, W, ym, n_pairs, n_clips) of AMEND_5's frozen readout. -> green (141 clips) and the regression arm (one clip dropped; must read not ok)."""
    Dr, Yf, EP, clips = pairs(LT141, scale_fn)
    green = compare(fit_ra(Dr, Yf, EP, len(clips)), ra5)
    keep = EP != 0
    Dr2, Yf2, EP2 = Dr[keep], Yf[keep], EP[keep] - 1
    red = compare(fit_ra(Dr2, Yf2, EP2, len(clips) - 1), ra5)
    return dict(green=green, regression_drop_one_clip=red, passed=bool(green["ok"] and not red["ok"]))


def merge_latents(parts):
    """Concatenate latent dicts (the 141 file and the chunks): Z1 {tag}, A_raw, ok, t, eid. Refuses a repeated clip across parts."""
    import torch
    tag = list(parts[0]["Z1"])[0]
    seen = set()
    for p in parts:
        c = set(p["eid"])
        if seen & c:
            raise ValueError(f"{len(seen & c)} clips appear in more than one part")
        seen |= c
    return dict(Z1={tag: torch.cat([p["Z1"][tag] for p in parts])}, A_raw=torch.cat([p["A_raw"] for p in parts]), ok=torch.cat([p["ok"] for p in parts]),
                t=np.concatenate([np.asarray(p["t"]) for p in parts]), eid=sum((list(p["eid"]) for p in parts), []))


def chunk_parts(tag, n_new, chunks_json=None, lat_dir=None):
    """The new-clip parts of the first n_new clips (a whole number of chunks of 50, or the last partial chunk's clips filtered by clip id). sha256 of every chunk checked against CHUNKS.json."""
    import torch
    J = json.load(open(chunks_json or ROOT + "/lat6/CHUNKS.json", encoding="utf-8"))
    order = json.load(open(ROOT + "/clips6.json", encoding="utf-8"))["clips"][:n_new]
    want = set(order)
    parts = []
    for ch in J["chunks"]:
        p = (lat_dir or ROOT + "/lat6") + "/" + ch["file"]
        raw = open(p, "rb").read()
        if hashlib.sha256(raw).hexdigest() != ch["sha256"]:
            raise RuntimeError(f"{ch['file']}: sha256 differs from CHUNKS.json")
        if not (set(ch["clips"]) & want):
            continue
        d = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        keep = np.array([e in want for e in d["eid"]])
        if not keep.all():
            idx = torch.from_numpy(np.where(keep)[0])
            d = dict(Z1={tag: d["Z1"][tag][idx]}, A_raw=d["A_raw"][idx], ok=d["ok"][idx], t=np.asarray(d["t"])[keep], eid=[e for e, k_ in zip(d["eid"], keep) if k_])
        else:
            d = dict(Z1={tag: d["Z1"][tag]}, A_raw=d["A_raw"], ok=d["ok"], t=np.asarray(d["t"]), eid=list(d["eid"]))
        parts.append(d)
    got = {e for p in parts for e in p["eid"]}
    if got != want:
        raise RuntimeError(f"chunks hold {len(got)} of the {len(want)} requested new clips")
    return parts


def largest_checkpoint(chunks_json=None):
    J = json.load(open(chunks_json or ROOT + "/lat6/CHUNKS.json", encoding="utf-8"))
    have = sum(len(c["clips"]) for c in J["chunks"])
    ok = [c for c in CHECKPOINTS_NEW if c <= have]
    if not ok:
        raise RuntimeError(f"only {have} new clips are banked; the smallest committed checkpoint is {CHECKPOINTS_NEW[0]}")
    return max(ok)


def curve_pure(Dr, Yf, EP, n_clips, totals, n_boot=1000):
    """FIT-CURVE (beside): for each nested total (clips 0..t-1 by the order of `clips` as passed), 5-fold clip-grouped cross-fit: held-out R2 and r, with a clip-cluster bootstrap CI. Dr/EP must be ordered 141 first, then the new clips in selection order."""
    out = {}
    for t in totals:
        sel = EP < t
        X, y, e = Dr[sel], Yf[sel], EP[sel]
        fo = (np.random.RandomState(0).permutation(t) % 5)[e]
        pred = np.zeros(len(y))
        for f in range(5):
            tr, te = fo != f, fo == f
            lam = choose_lam(X[tr], y[tr], fo[tr])
            pred[te] = ridge_pred(ridge_fit(X[tr], y[tr], lam), X[te])[:, 0]
        yy = y[:, 0]
        g = [np.where(e == c)[0] for c in range(t)]

        def stat(ii):
            p_, y_ = pred[ii], yy[ii]
            return np.array([1 - ((p_ - y_) ** 2).mean() / ((y_ - y_.mean()) ** 2).mean(), np.corrcoef(p_, y_)[0, 1]])
        pt = stat(np.arange(len(yy)))
        r = np.random.RandomState(0)
        bs = np.array([stat(np.concatenate([g[c] for c in r.randint(0, t, t)])) for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        out[str(t)] = dict(n_clips=int(t), n_pairs=int(sel.sum()), r2=float(pt[0]), r2_ci=[float(lo[0]), float(hi[0])], r=float(pt[1]), r_ci=[float(lo[1]), float(hi[1])])
    return out


# ---------------------------------------------------------------------------------------------- impure (real files)
def _scale_fn(k):
    sys.path.insert(0, HERE)
    import amend6_env  # noqa: F401
    import ra_l1 as R
    m, cfg, _ = R.D.load_model(CK.format(k=k))
    return m.wm.scale_actions


def _ra5(k):
    import torch
    r = torch.load(RA5.format(k=k), map_location="cpu", weights_only=False)
    ra = {x: np.asarray(r["RA"][x], np.float64) for x in ("mu", "sd", "W", "ym")}
    return r, dict(lam=float(r["lambda"]), **ra, n_pairs=8306, n_clips=141)


def _load141(k):
    import torch
    return torch.load(LAT141.format(k=k), map_location="cpu", weights_only=False)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["kcontrol", "fit", "curve"]); ap.add_argument("--chassis", required=True); ap.add_argument("--new-clips", type=int, default=None)
    a = ap.parse_args()
    k = a.chassis; tag = f"R35_{k}"
    sc = _scale_fn(k)
    LT = _load141(k)
    if a.cmd == "kcontrol":
        _, ra5 = _ra5(k)
        res = kcontrol_pure(LT, sc, ra5)
        os.makedirs(ROOT + "/controls", exist_ok=True)
        json.dump(res, open(ROOT + f"/controls/kcontrol_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1)
        print("K-CONTROL", k, "passed" if res["passed"] else "RED", res["green"])
        sys.exit(0 if res["passed"] else 1)
    n_new = a.new_clips or largest_checkpoint()
    parts = [LT] + chunk_parts(tag, n_new)
    M = merge_latents(parts)
    Dr, Yf, EP, clips = pairs(M, sc)
    if a.cmd == "curve":
        c141 = list(json.load(open("F:/TanitAD-scratch/hwm_plan/ts/clips141.json", encoding="utf-8"))["clips"])
        order = c141 + json.load(open(ROOT + "/clips6.json", encoding="utf-8"))["clips"][:n_new]
        remap = {c: i for i, c in enumerate(order)}
        EPn = np.array([remap[clips[i]] for i in EP])
        tot = [t for t in (141, 282, 564, 1000) if t <= len(order)]
        res = curve_pure(Dr, Yf, EPn, len(order), tot)
        json.dump(res, open(ROOT + f"/readout/FIT_CURVE_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1)
        print("FIT-CURVE", k, {t: (round(v["r2"], 4), round(v["r"], 4)) for t, v in res.items()})
        return
    f = fit_ra(Dr, Yf, EP, len(clips))
    r5, _ = _ra5(k)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/readout_K10_R35_{k}.pt"
    torch.save({"tag": tag, "K": K, "lambda": f["lam"], "RA": {x: f[x] for x in ("mu", "sd", "W", "ym")}, "fit": dict(n_pairs=f["n_pairs"], n_clips=f["n_clips"], n_new_clips=n_new, amend="AMEND_HWMPLAN_6")}, p)
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rec = dict(chassis=k, readout=p, sha256=sha, lam=f["lam"], n_pairs=f["n_pairs"], n_clips=f["n_clips"], n_new_clips=n_new, lam_amend5=float(r5["lambda"]),
               cos_W_vs_RA5=float((f["W"][:, 0] @ np.asarray(r5["RA"]["W"], np.float64)[:, 0]) / (np.linalg.norm(f["W"]) * np.linalg.norm(np.asarray(r5["RA"]["W"], np.float64)))),
               chunks_sha256=json.load(open(ROOT + "/lat6/CHUNKS.json", encoding="utf-8"))["chunks"] and [c["sha256"] for c in json.load(open(ROOT + "/lat6/CHUNKS.json", encoding="utf-8"))["chunks"]])
    json.dump(rec, open(ROOT + f"/RA6_FIT_{k}.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print("FIT", k, "n_pairs", f["n_pairs"], "n_clips", f["n_clips"], "lam", f["lam"], "sha", sha[:12], "cos(W6,W5)", round(rec["cos_W_vs_RA5"], 4))


if __name__ == "__main__":
    main()
