#!/usr/bin/env python3
"""D3 check 1b -- do any two cached clips share a SOURCE RECORDING (time-overlapping egomotion)?

Why this is the right instrument.  Every clip's egomotion log is 100 Hz over the 20 s clip window
(offset 0 of the recording, MEASURED in egomotion_source.py docstring) and then CONTINUES, sparsely
(~4-10 Hz), out to ~20-140 s.  If clip B was cut from the same drive as clip A, B's 20 s speed and
yaw-rate profile must reappear inside A's log at some offset tau (B starts tau seconds after A).
No clip/recording identifier exists in the label records (MM 2026-10-04: only horizon.recording_span_s),
and egomotion carries no GNSS, so the motion profile itself is the only join key.

Feature (frame-free, so it is invariant to the clip-local origin): speed v(t)=hypot(vx,vy) [m/s]
and yaw-rate w(t)=d(unwrapped quaternion yaw)/dt [rad/s] on a 10 Hz grid.
Score(B, A, tau) = 0.5 * ( MSE_v / 0.3^2 + MSE_w / 0.02^2 ),  tau on the 0.1 s grid, tau in [0, spanA-20 s].
Literal thresholds (road/sensor physics, NOT fitted to the data): a genuine overlap re-samples the SAME
vehicle signal, so its error is interpolation-limited: RMSE_v < 0.3 m/s and RMSE_w < 0.02 rad/s  =>  score < 1.

Window excitation: a window whose v-std < 0.5 m/s AND w-std < 0.02 rad/s (steady cruise / standing still)
matches almost anything, so it is reported as UNINFORMATIVE and never counted as "no overlap found".

Controls (all computed in this script, all must read their known value):
  C1 self-match   : B = A's own first 20 s must give tau = 0.0 and score ~ 0 for every excited clip.
  C2 shifted+noise: B = A[tau0 : tau0+20 s] + N(0, 0.1 m/s) / N(0, 0.005 rad/s), tau0 = 37.3 s, on every A
                    with span >= 60 s; must be recovered at tau0 within 0.1 s.
  C3 null         : the distribution of best score over unrelated pairs (min over tau) is reported; its
                    minimum over informative pairs is the empirical false-match floor.
Ids are written as sha12 only.
"""
from __future__ import annotations
import glob, hashlib, json, os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd

EGO = "/home/nvidia/data/obstacle/egomotion_alpamayo"
OUT = "/home/nvidia/refcv8_audit/D3/run2"
HZ = 10.0
WIN = 200            # 20 s at 10 Hz
NFFT = 2048
SV, SW = 0.3, 0.02   # literal normalisers (m/s, rad/s)


def sha12(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def yaw_from_q(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def series(cid):
    d = pd.read_parquet(f"{EGO}/{cid}.parquet", columns=["timestamp", "qx", "qy", "qz", "qw", "vx", "vy"])
    t = d.timestamp.to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    t = t[o]
    v = np.hypot(d.vx.to_numpy(np.float64)[o], d.vy.to_numpy(np.float64)[o])
    yaw = np.unwrap(yaw_from_q(*(d[c].to_numpy(np.float64)[o] for c in ("qx", "qy", "qz", "qw"))))
    tend = np.floor(t[-1] * HZ) / HZ
    g = np.arange(0.0, tend + 1e-9, 1.0 / HZ)
    g = g[:NFFT]
    vg = np.interp(g, t, v)
    yg = np.interp(g, t, yaw)
    wg = np.diff(yg) * HZ
    return vg.astype(np.float64), wg.astype(np.float64), float(t[-1])


def load_all(ids):
    V, W, S = {}, {}, {}
    for c in ids:
        v, w, s = series(c)
        V[c], W[c], S[c] = v, w, s
    return V, W, S


_G = {}


def _init(Vs, Ws, lens, wins_v, wins_w, informative):
    _G.update(Vs=Vs, Ws=Ws, lens=lens, wv=wins_v, ww=wins_w, inf=informative)
    n = len(lens)
    # spectra of every log (padded to NFFT) and sliding energies
    FAv = np.fft.rfft(Vs, NFFT, axis=1)
    FAw = np.fft.rfft(Ws, NFFT, axis=1)
    cv = np.concatenate([np.zeros((n, 1)), np.cumsum(Vs ** 2, axis=1)], axis=1)
    cw = np.concatenate([np.zeros((n, 1)), np.cumsum(Ws ** 2, axis=1)], axis=1)
    _G.update(FAv=FAv, FAw=FAw, cv=cv, cw=cw)


def score_B(b):
    """score of window b against every log A; returns arrays over A: best score, tau index, rmse_v, rmse_w."""
    Vs, Ws, lens = _G["Vs"], _G["Ws"], _G["lens"]
    wv, ww = _G["wv"][b], _G["ww"][b]
    Bv = np.zeros(NFFT); Bv[:WIN] = wv
    Bw = np.zeros(NFFT); Bw[:WIN - 1] = ww
    FBv = np.fft.rfft(Bv); FBw = np.fft.rfft(Bw)
    cv = np.fft.irfft(_G["FAv"] * np.conj(FBv), NFFT, axis=1)    # [nA, NFFT]
    cw = np.fft.irfft(_G["FAw"] * np.conj(FBw), NFFT, axis=1)
    tmax = NFFT - WIN
    n = len(lens)
    tau = np.arange(0, tmax)
    EBv, EBw = float((wv ** 2).sum()), float((ww ** 2).sum())
    # sliding energies of A over [tau, tau+WIN) (v) and [tau, tau+WIN-1) (w)
    cvv, cww = _G["cv"], _G["cw"]
    L = cvv.shape[1]
    iv1 = np.minimum(tau + WIN, L - 1); iw1 = np.minimum(tau + WIN - 1, L - 1)
    EAv = cvv[:, iv1] - cvv[:, np.minimum(tau, L - 1)]
    EAw = cww[:, iw1] - cww[:, np.minimum(tau, L - 1)]
    ssd_v = EBv - 2 * cv[:, :tmax] + EAv
    ssd_w = EBw - 2 * cw[:, :tmax] + EAw
    mv, mw = ssd_v / WIN, ssd_w / (WIN - 1)
    sc = 0.5 * (mv / SV ** 2 + mw / SW ** 2)
    valid = tau[None, :] <= (lens[:, None] - WIN)       # B must lie fully inside A's 10 Hz series
    sc = np.where(valid, sc, np.inf)
    k = np.argmin(sc, axis=1)
    r = np.arange(n)
    return (sc[r, k].astype(np.float32), k.astype(np.int16),
            np.sqrt(np.maximum(mv[r, k], 0)).astype(np.float32), np.sqrt(np.maximum(mw[r, k], 0)).astype(np.float32))


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    train = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/refcv6_train_clips.txt") if l.strip()]
    ev = [l.strip().replace(".v2ep.pt", "") for l in open("/home/nvidia/data/eval139_ids.txt") if l.strip()]
    ids = train + ev
    split = np.array([0] * len(train) + [1] * len(ev), dtype=np.int8)
    print("clips", len(train), len(ev), flush=True)
    V, W, S = load_all(ids)
    print("series loaded", round(time.time() - t0, 1), "s", flush=True)
    n = len(ids)
    lens = np.array([len(V[c]) for c in ids])
    spans = np.array([S[c] for c in ids])
    Vs = np.zeros((n, NFFT)); Ws = np.zeros((n, NFFT))
    for i, c in enumerate(ids):
        Vs[i, :len(V[c])] = V[c]; Ws[i, :len(W[c])] = W[c]
    wv = np.stack([V[c][:WIN] for c in ids]); ww = np.stack([W[c][:WIN - 1] for c in ids])
    vstd, wstd, vmean = wv.std(1), ww.std(1), wv.mean(1)
    informative = (vstd >= 0.5) | (wstd >= 0.02)
    print("informative windows", int(informative.sum()), "of", n, flush=True)
    np.savez_compressed(f"{OUT}/c1_series_summary.npz", sha12=np.array([sha12(c) for c in ids]), split=split,
                        span_s=spans, vstd=vstd, wstd=wstd, vmean=vmean, informative=informative, lens=lens)

    # ---- pair scoring (all windows vs all logs) --------------------------------------------------
    # window b is scored only if informative (others are reported as uninformative, not as "no overlap")
    todo = [b for b in range(n) if informative[b]]
    nproc = int(os.environ.get("D3_NPROC", "6"))
    best = np.full((n, n), np.inf, dtype=np.float32)
    tau_i = np.zeros((n, n), dtype=np.int16)
    rv = np.zeros((n, n), dtype=np.float32); rw = np.zeros((n, n), dtype=np.float32)
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc, initializer=_init, initargs=(Vs, Ws, lens, wv, ww, informative)) as pool:
        for k, (b, res) in enumerate(zip(todo, pool.imap(score_B, todo, chunksize=8))):
            best[b], tau_i[b], rv[b], rw[b] = res
            if k % 500 == 0:
                print("scored", k, "/", len(todo), round(time.time() - t0, 1), "s", flush=True)
    # persist every pair with score < 6 (ids by index; sha12 mapping saved in c1_series_summary.npz order)
    _self = np.eye(n, dtype=bool)
    bb, aa = np.where((best < 6.0) & ~_self)
    np.savez_compressed(f"{OUT}/c1_pairs_lt6.npz", b=bb.astype(np.int32), a=aa.astype(np.int32), tau=tau_i[bb, aa],
                        score=best[bb, aa], rmse_v=rv[bb, aa], rmse_w=rw[bb, aa])
    print("saved pairs lt6", len(bb), flush=True)
    # self pairs are the control C1; mask them out of the "other clip" search
    diag_score = np.array([best[i, i] for i in range(n)]); diag_tau = np.array([tau_i[i, i] for i in range(n)])
    for i in range(n):
        best[i, i] = np.inf

    out = {"n_clips": n, "n_train": len(train), "n_eval": len(ev), "n_informative_windows": int(informative.sum()),
           "thresholds_literal": {"score_match_lt": 1.0, "rmse_v_ms": SV, "rmse_w_rads": SW,
                                  "window_informative_if": "vstd>=0.5 m/s or wstd>=0.02 rad/s"}}
    # C1
    inf_idx = np.where(informative)[0]
    c1_ok = (diag_score[inf_idx] < 0.05) & (diag_tau[inf_idx] == 0)
    out["C1_self_match"] = {"n": int(len(inf_idx)), "n_recovered_tau0": int(c1_ok.sum()),
                            "max_score": float(diag_score[inf_idx].max()), "median_score": float(np.median(diag_score[inf_idx]))}
    # matches
    def pairs_below(thr):
        ii, jj = np.where(best < thr)
        return ii, jj
    ii, jj = pairs_below(1.0)
    pr = []
    for b, a in zip(ii, jj):
        pr.append({"window_clip": sha12(ids[b]), "window_split": "eval" if split[b] else "train",
                   "log_clip": sha12(ids[a]), "log_split": "eval" if split[a] else "train",
                   "tau_s": float(tau_i[b, a]) / HZ, "score": float(best[b, a]),
                   "rmse_v": float(rv[b, a]), "rmse_w": float(rw[b, a]),
                   "vstd_B": float(vstd[b]), "wstd_B": float(wstd[b])})
    pr.sort(key=lambda r: r["score"])
    out["matches_score_lt_1"] = {"n_pairs": len(pr), "pairs_first_200": pr[:200]}
    # eval-vs-train breakdown
    def cnt(ws, ls):
        return sum(1 for r in pr if r["window_split"] == ws and r["log_split"] == ls)
    out["match_breakdown"] = {f"{w}_window_in_{l}_log": cnt(w, l) for w in ("train", "eval") for l in ("train", "eval")}
    # null floor: minimum best score over informative windows vs other logs, and quantiles of per-window best
    per_window_best = np.where(informative, best.min(1), np.nan)
    pw = per_window_best[~np.isnan(per_window_best)]
    out["C3_null_per_window_best_score"] = {"n": int(len(pw)), "min": float(pw.min()),
        "q001": float(np.quantile(pw, 0.001)), "q01": float(np.quantile(pw, 0.01)), "q05": float(np.quantile(pw, 0.05)),
        "median": float(np.median(pw)), "n_lt_1": int((pw < 1.0).sum()), "n_lt_2": int((pw < 2.0).sum()), "n_lt_4": int((pw < 4.0).sum())}
    # near-misses to show the gap
    flat = np.argsort(best.ravel())[:60]
    nm = []
    for f in flat:
        b, a = divmod(int(f), n)
        nm.append({"window_clip": sha12(ids[b]), "log_clip": sha12(ids[a]), "tau_s": float(tau_i[b, a]) / HZ,
                   "score": float(best[b, a]), "rmse_v": float(rv[b, a]), "rmse_w": float(rw[b, a])})
    out["lowest_60_scores"] = nm
    # eval windows: best score against TRAIN logs and best against ALL logs
    ev_idx = np.where(split == 1)[0]; tr_idx = np.where(split == 0)[0]
    out["eval_windows_vs_train_logs"] = {
        "n_eval_informative": int(informative[ev_idx].sum()),
        "min_best_score": float(np.min(best[np.ix_(ev_idx[informative[ev_idx]], tr_idx)])),
        "n_lt_1": int((best[np.ix_(ev_idx[informative[ev_idx]], tr_idx)].min(1) < 1.0).sum())}
    out["train_windows_vs_eval_logs"] = {
        "n_train_informative": int(informative[tr_idx].sum()),
        "min_best_score": float(np.min(best[np.ix_(tr_idx[informative[tr_idx]], ev_idx)])),
        "n_lt_1": int((best[np.ix_(tr_idx[informative[tr_idx]], ev_idx)].min(1) < 1.0).sum())}
    out["uninformative"] = {"n_total": int((~informative).sum()),
                            "n_eval": int((~informative[ev_idx]).sum()), "n_train": int((~informative[tr_idx]).sum())}
    out["span_s"] = {"median": float(np.median(spans)), "min": float(spans.min()), "max": float(spans.max()),
                     "n_span_lt_21": int((spans < 21).sum())}
    out["elapsed_s_main"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{OUT}/c1_overlap_result.json", "w"), indent=1)
    print("main done", out["elapsed_s_main"], flush=True)

    # ---- C2 shifted+noise control (serial, 40 random A with span >= 60 s) -------------------------
    rng = np.random.default_rng(7)
    cand = np.where(lens >= 600)[0]
    pick = rng.choice(cand, size=min(40, len(cand)), replace=False)
    _init(Vs, Ws, lens, wv, ww, informative)
    ok = 0; rows = []
    tau0 = 373   # 37.3 s
    for a in pick:
        bv = Vs[a, tau0:tau0 + WIN] + rng.normal(0, 0.1, WIN)
        bw = Ws[a, tau0:tau0 + WIN - 1] + rng.normal(0, 0.005, WIN - 1)
        _G["wv"] = np.vstack([bv]); _G["ww"] = np.vstack([bw])
        s, k, r_v, r_w = score_B(0)
        s2 = s.copy();
        got = int(k[a]); sc = float(s[a])
        hit = (got == tau0) and sc < 1.0
        ok += int(hit)
        # best over ALL logs should be A itself
        top = int(np.argmin(s))
        rows.append({"A": sha12(ids[a]), "tau_found": got / HZ, "score": sc, "argmin_is_A": top == int(a)})
    json.dump({"n": len(pick), "n_recovered": ok, "tau0_s": tau0 / HZ, "rows": rows},
              open(f"{OUT}/c1_control_shift_noise.json", "w"), indent=1)
    print("C2", ok, "/", len(pick), flush=True)
    print("elapsed", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
