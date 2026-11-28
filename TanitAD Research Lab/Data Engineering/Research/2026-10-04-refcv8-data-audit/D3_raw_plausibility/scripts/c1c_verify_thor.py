#!/usr/bin/env python3
"""D3 check 1c -- classify and VERIFY the same-recording matches found by c1_overlap_thor_v2.py (independent channels).

Channel 1 (used by the finder)      : speed + yaw-rate profile (frame-free).
Channel 2 (independent, this script): the (x, y) PATH SHAPE.  For a pair (B window, A log, tau) take A's positions on [tau, tau+20 s],
   B's on [0, 20 s], fit the best rigid transform (Kabsch, rotation + translation, no scale) and report the RMS residual [m].
   A genuine same-recording pair re-observes the same path => small residual; the NEGATIVE CONTROL is the identical pair with tau
   shifted by +5.0 s (same clips, wrong alignment) whose residual must be much larger, and a random-pair control.
Channel 3 (independent, images)     : for pairs with tau < 19 s the two 20 s clips share wall-clock time, so B's first frame must look like
   A's frame at B's start.  Normalised cross-correlation of 52x128 grey thumbnails against A's rows r-1..r+1 versus the same
   statistic against a random frame of a random other clip (null).
Then: threshold table, recording components (union-find on exact matches), and per-eval-clip leak table (sha12 only).
"""
import hashlib, io, json, os, sys, time, collections
import numpy as np
import pandas as pd

OUT = "/home/nvidia/refcv8_audit/D3/run2"
EGO = "/home/nvidia/data/obstacle/egomotion_alpamayo"
CACHE_TR = "/home/nvidia/data/refcv6-b1-416x1024-train"
CACHE_EV = "/home/nvidia/data/refcv6-b1-416x1024-eval139"
sha12 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:12]
CLIP_S = 20.2


def kabsch_rms(P, Q):
    """RMS residual after the best rotation+translation mapping P -> Q (both [n,2])"""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, d]) @ Vt
    return float(np.sqrt(((Pc @ R - Qc) ** 2).sum(1).mean()))


_cache = {}
def xy(cid):
    if cid in _cache:
        return _cache[cid]
    d = pd.read_parquet(f"{EGO}/{cid}.parquet", columns=["timestamp", "x", "y"])
    t = d.timestamp.to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    r = (t[o], d.x.to_numpy(np.float64)[o], d.y.to_numpy(np.float64)[o])
    if len(_cache) > 300:
        _cache.clear()
    _cache[cid] = r
    return r


def path_resid(cidB, cidA, tau):
    tb, xb, yb = xy(cidB); ta, xa, ya = xy(cidA)
    s = np.arange(0.0, 20.0, 0.1)
    if tau + 20.0 > ta[-1]:
        return None
    B = np.stack([np.interp(s, tb, xb), np.interp(s, tb, yb)], 1)
    A = np.stack([np.interp(tau + s, ta, xa), np.interp(tau + s, ta, ya)], 1)
    return kabsch_rms(A, B)


def main():
    t0 = time.time()
    train = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/refcv6_train_clips.txt") if l.strip()]
    ev = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/eval139_ids.txt") if l.strip()]
    ids = train + ev
    n = len(ids)
    split = np.array([0] * len(train) + [1] * len(ev))
    S = np.load(f"{OUT}/c1_series_summary.npz")
    informative = S["informative"]; spans = S["span_s"]
    Pz = np.load(f"{OUT}/c1_pairs_lt6.npz")
    b, a, tau, sc, rv, rw = Pz["b"], Pz["a"], Pz["tau"].astype(np.float64) / 10.0, Pz["score"], Pz["rmse_v"], Pz["rmse_w"]
    out = {"n_pairs_lt6": int(len(b))}
    # ---- threshold table -------------------------------------------------------------------------------
    best = np.full(n, np.inf)
    np.minimum.at(best, b, sc)
    inf_idx = np.where(informative)[0]
    thr_tab = {}
    for thr in (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 2.0, 4.0):
        m = sc < thr
        wins = np.unique(b[m]); logs = np.unique(a[m])
        thr_tab[str(thr)] = {"pairs": int(m.sum()), "windows_with_match": int(len(wins)), "pct_of_informative_windows": round(100 * len(wins) / len(inf_idx), 2),
                             "eval_windows_with_match_to_train_log": int(len(np.unique(b[m & (split[b] == 1) & (split[a] == 0)]))),
                             "train_windows_with_match_to_eval_log": int(len(np.unique(b[m & (split[b] == 0) & (split[a] == 1)])))}
    out["threshold_table"] = thr_tab
    # histogram of per-window best score (log10)
    bb = best[inf_idx]
    h, e = np.histogram(np.log10(np.clip(bb[np.isfinite(bb)], 1e-9, None)), bins=np.arange(-9, 1.01, 0.5))
    out["hist_log10_best_score"] = {"edges": e.tolist(), "counts": h.tolist(), "n_inf_best": int((~np.isfinite(bb)).sum())}
    # best score by window excitation (so a low-excitation window's chance matches are visible)
    vstd, wstd = S["vstd"], S["wstd"]
    # ---- channel 2: path shape on the best pair of every window with best score < 1 ------------------------
    # one pair per window (its best), plus ALL pairs with score < 0.1
    sel = []
    order = np.argsort(sc)
    seen_w = set()
    for k in order:
        if sc[k] >= 1.0:
            break
        if int(b[k]) in seen_w and sc[k] >= 0.1:
            continue
        seen_w.add(int(b[k]))
        sel.append(k)
    rows = []
    for k in sel:
        r1 = path_resid(ids[b[k]], ids[a[k]], float(tau[k]))
        r2 = path_resid(ids[b[k]], ids[a[k]], float(tau[k]) + 5.0)      # wrong-alignment control
        rows.append((int(b[k]), int(a[k]), float(tau[k]), float(sc[k]), r1, r2))
    ok = [r for r in rows if r[4] is not None and r[5] is not None]
    r1v = np.array([r[4] for r in ok]); r2v = np.array([r[5] for r in ok]); scv = np.array([r[3] for r in ok])
    # random-pair control: window b vs a random OTHER log at a random feasible tau
    rng = np.random.default_rng(1)
    rnd = []
    for r in ok[:400]:
        for _ in range(1):
            a2 = int(rng.integers(n))
            if a2 == r[0]:
                continue
            tmax = spans[a2] - 20.2
            if tmax < 1:
                continue
            q = path_resid(ids[r[0]], ids[a2], float(rng.uniform(0, tmax)))
            if q is not None:
                rnd.append(q)
    out["path_shape_check"] = {"n_pairs_checked": len(ok),
        "resid_rms_m_by_score_band": {f"{lo}-{hi}": (lambda m: {"n": int(m.sum()), "median": float(np.median(r1v[m])) if m.any() else None, "p90": float(np.quantile(r1v[m], .9)) if m.any() else None,
                                                                "frac_lt_1m": float((r1v[m] < 1.0).mean()) if m.any() else None})((scv >= lo) & (scv < hi))
                                      for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 1.0))},
        "CONTROL_same_pair_tau_plus_5s": {"median": float(np.median(r2v)), "frac_lt_1m": float((r2v < 1.0).mean())},
        "CONTROL_random_pair": {"n": len(rnd), "median": float(np.median(rnd)), "frac_lt_1m": float((np.array(rnd) < 1.0).mean())}}
    # ---- classification & components --------------------------------------------------------------------------
    thr_exact = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    m = sc < thr_exact
    # a pair is verified if path-shape residual < 1.5 m (independent channel)
    ver = {}
    for r in ok:
        ver[(r[0], r[1])] = r[4]
    good = [k for k in np.where(m)[0] if ver.get((int(b[k]), int(a[k])), 9e9) < 1.5 or (int(b[k]), int(a[k])) not in ver]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k in good:
        ra, rb = find(int(a[k])), find(int(b[k]))
        if ra != rb:
            parent[ra] = rb
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    sizes = collections.Counter(len(v) for v in comp.values())
    mixed = [v for v in comp.values() if len(v) > 1 and len({split[i] for i in v}) == 2]
    out["components"] = {"thr_exact": thr_exact, "n_pairs_used": len(good), "n_components_all": len(comp), "n_multi_clip_components": sum(1 for v in comp.values() if len(v) > 1),
                         "size_hist": {str(k): v for k, v in sorted(sizes.items())}, "largest": max(len(v) for v in comp.values()),
                         "n_components_with_train_and_eval": len(mixed), "clips_in_multi_clip_components": sum(len(v) for v in comp.values() if len(v) > 1),
                         "train_clips_in_multi": sum(1 for v in comp.values() if len(v) > 1 for i in v if split[i] == 0),
                         "eval_clips_in_multi": sum(1 for v in comp.values() if len(v) > 1 for i in v if split[i] == 1),
                         "eval_clips_sharing_recording_with_a_train_clip": len({i for v in mixed for i in v if split[i] == 1}),
                         "train_clips_sharing_recording_with_an_eval_clip": len({i for v in mixed for i in v if split[i] == 0})}
    # tau distribution for matches
    tt = tau[good]
    out["tau_distribution_s"] = {"n": int(len(tt)), "tau_lt_0.5": int((tt < 0.5).sum()), "0.5_20": int(((tt >= 0.5) & (tt < 20.2)).sum()), "20.2_40": int(((tt >= 20.2) & (tt < 40)).sum()),
                                 "40_80": int(((tt >= 40) & (tt < 80)).sum()), "80_plus": int((tt >= 80).sum())}
    # eval-clip table: gap between the 20 s clips (negative = overlapping content)
    gaps = collections.defaultdict(list)
    for k in good:
        bi, ai = int(b[k]), int(a[k])
        if split[bi] != split[ai]:
            gap = float(tau[k]) - CLIP_S
            ei = bi if split[bi] == 1 else ai
            gaps[ei].append(gap)
    mg = np.array([min(abs(x) if x >= 0 else x for x in v) for v in gaps.values()]) if gaps else np.array([])
    # per eval clip: smallest gap (signed: negative = time-overlap)
    min_gap = {ei: min(v) for ei, v in gaps.items()}
    vals = np.array(list(min_gap.values()))
    out["eval_train_links"] = {"n_eval_clips_linked": len(min_gap), "of_eval": int((split == 1).sum()),
        "min_gap_s_bands": {"overlapping_time (gap<0)": int((vals < 0).sum()), "0-5 s": int(((vals >= 0) & (vals < 5)).sum()), "5-30 s": int(((vals >= 5) & (vals < 30)).sum()),
                            "30-60 s": int(((vals >= 30) & (vals < 60)).sum()), "60+ s": int((vals >= 60).sum())},
        "eval_clips_sha12_first": [sha12(ids[i]) for i in list(min_gap)[:10]]}
    # informative-only denominator: eval windows are testable only when informative
    out["eval_testable"] = {"n_eval": int((split == 1).sum()), "n_eval_informative": int(informative[split == 1].sum())}
    # ---- channel 3: images for tau < 19 s ---------------------------------------------------------------------
    try:
        import torch
        from PIL import Image
        def frames(cid, sel_rows):
            p = f"{CACHE_EV if cid in ev_set else CACHE_TR}/{cid}.v2ep.pt"
            d = torch.load(p, map_location="cpu", weights_only=False)
            buf = d["jpeg_buf"].numpy(); ln = d["jpeg_len"].numpy(); off = np.concatenate([[0], np.cumsum(ln)])
            outf = {}
            for r in sel_rows:
                if 0 <= r < len(ln):
                    im = Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("L").resize((128, 52), Image.BILINEAR)
                    outf[r] = np.asarray(im, np.float32)
            return outf
        ev_set = set(ev)
        side = {}
        for l in open("/home/nvidia/data/refcv6_clip_clock_sidecar.jsonl"):
            r = json.loads(l); side[int(r["sid"])] = r
        mt = torch.load("/home/nvidia/data/refcv6-b1-416x1024-train/_v2manifest.pt", map_location="cpu", weights_only=False)
        me = torch.load("/home/nvidia/data/refcv6-b1-416x1024-eval139/_v2manifest.pt", map_location="cpu", weights_only=False)
        uid = {c: int(u) for c, u in zip(list(mt["clip_id"]) + list(me["clip_id"]), list(mt["episode_uid"]) + list(me["episode_uid"]))}
        def gs_dt(c):
            r = side.get(uid[c]); return (float(r["grid_start_s"]), float(r["dt_s"])) if r else (0.113, 0.100667)
        def ncc(x, y):
            x = (x - x.mean()) / (x.std() + 1e-6); y = (y - y.mean()) / (y.std() + 1e-6)
            return float((x * y).mean())
        cand = [k for k in good if 0.5 <= tau[k] < 19.0][:60]
        res_img = []
        for k in cand:
            B, A = ids[b[k]], ids[a[k]]
            gB, dB = gs_dt(B); gA, dA = gs_dt(A)
            rA = (tau[k] + gB - gA) / dA
            fB = frames(B, [0])[0]
            fA = frames(A, [int(round(rA)) - 1, int(round(rA)), int(round(rA)) + 1])
            res_img.append(max(ncc(fB, v) for v in fA.values()) if fA else None)
        nul = []
        for k in cand[:40]:
            B = ids[b[k]]; A2 = ids[int(rng.integers(len(train)))]
            fB = frames(B, [0])[0]; fA = frames(A2, [int(rng.integers(20, 150))])
            nul.append(max(ncc(fB, v) for v in fA.values()))
        rv_ = np.array([x for x in res_img if x is not None])
        out["image_check_tau_lt_19s"] = {"n_pairs": int(len(rv_)), "ncc_matched_median": float(np.median(rv_)) if len(rv_) else None, "ncc_matched_min": float(rv_.min()) if len(rv_) else None,
                                         "ncc_null_random_frame_median": float(np.median(nul)) if nul else None, "ncc_null_max": float(max(nul)) if nul else None,
                                         "frac_matched_gt_null_max": float((rv_ > max(nul)).mean()) if len(rv_) and nul else None}
    except Exception as e:
        out["image_check_tau_lt_19s"] = {"error": repr(e)[:200]}
    # ---- full leak table (sha12 only) ----------------------------------------------------------------------------
    lk = []
    for k in good:
        bi, ai = int(b[k]), int(a[k])
        if split[bi] != split[ai]:
            lk.append({"window_clip": sha12(ids[bi]), "window_split": "eval" if split[bi] else "train", "log_clip": sha12(ids[ai]), "log_split": "eval" if split[ai] else "train",
                       "tau_s": float(tau[k]), "gap_between_clips_s": round(float(tau[k]) - CLIP_S, 1), "score": float(sc[k])})
    lk.sort(key=lambda r: r["gap_between_clips_s"])
    out["eval_train_leak_pairs"] = lk
    json.dump(out, open(f"{OUT}/c1c_verify_result.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "eval_train_leak_pairs"}, indent=1)[:7000])
    print("elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
