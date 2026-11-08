#!/usr/bin/env python3
"""SCORER1 Stage 0 (PREREG_SCORER1 section 1), CPU analysis: the tie-correct within-token gap on a navtrain held-out set, and the committed Z1 / Z2 reading.

    python stage0_gap.py --scores DIR/scores_<tag>_base.npz --sets <heldout sets dir> --targets <train_grow dir> --pdm <heldout_pdm dir> --out stage0.json
    python stage0_gap.py --selftest            # synthetic planted-gap / null fixtures through the SAME file readers (no data needed)

Inputs (all on the SAME 24 held-out logs, `splits/sft_heldout_logs.txt`):
  scores   u400's per-proposal logits, scorer_finetune.py --dump-scores: scores_<tag>_<arm>.npz (log, token, step, rank, scores [N, 64, C], space='logit'); the `base` arm IS u400 (the loaded scorer).
  sets     the on-policy sets (onpolicy_*.jsonl: key = (log_name, token, step, rank); `traj` [64, 20, 2] = the proposals).
  targets  targets_rank*.jsonl of --targets (key (log_name, token, step, rank); `ego` slot 6 = v0, slot 2 = ax0; `traj` [20, >=2] = the GT future).
  pdm      the true PDM labels pdm_*.jsonl (key, nc dac ep ttc c ddc, each 64 long).
Statistic (the PRE-REGISTERED estimator, section 8.1): per token, over the 64 proposals, rho_h = Spearman(AVERAGE ranks) of the head's value against v4/v0, for the PREDICTED sigmoid(logit_h) and for the TRUE label;
a token enters head h only if BOTH rank vectors vary (the true labels vary) and >= 5 proposals; gap = rho_pred - rho_true per token on the tokens where both are defined; CRUISE only (the ego-probe definition
v0 > 5 m/s, |ax0| < 0.3, |GT v4 - v0| < 1 m/s; v4 = the 0.2 s chord at step 19 of the trajectory, GT's from `traj` of the targets row: UNVERIFIED that this equals the probe's 'last 0.5 s chord' of human_future_poses).
Interval: paired LOG-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile (the programme's draws).  Reading, committed in the card:
  Z1 (gap present on navtrain held-out): mean of the TTC and NC head gaps <= -0.20 AND its CI-hi < 0  => run Stage 1.
  Z2 (gap absent): the CI includes 0 OR |mean| < 0.10                                                  => Stage 1 is not the lever (localise the shift navtrain -> navtest).
  anything else => inconclusive: run Stage 1, record Stage 0 as inconclusive.
Z1 is tested first.  The head mean in a draw is the mean of the two heads' draw means (the same resampled logs for both).
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

HEADS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3}
PDM = ("nc", "dac", "ep", "ttc", "c", "ddc")
V_IDX, DT = 19, 0.2
SEED, B = 20260927, 10000


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")


def read_jsonl_dir(d, pattern, want_key, fields):
    """{(log, token, step, rank): {field: value}} over d/pattern; `want_key(row) -> key tuple or None`"""
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, pattern))):
        with _open(fp) as fh:
            for ln in fh:
                if not ln.startswith("{"):
                    continue
                r = json.loads(ln)
                k = want_key(r)
                if k is not None:
                    out[k] = {f: r[f] for f in fields if f in r}
    return out


def key_set(r):
    return (r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))


def key_pdm(r):
    k = r.get("key")
    return (k[0], k[1], int(k[2]), int(k[3])) if k and len(k) == 4 else None


def chord_v(xy, idx=V_IDX, dt=DT):
    xy = np.asarray(xy, np.float64)[..., :2]
    i = min(idx, xy.shape[-2] - 1)
    prev = np.zeros_like(xy[..., 0, :]) if i == 0 else xy[..., i - 1, :]
    return np.linalg.norm(xy[..., i, :] - prev, axis=-1) / dt


def load_tokens(a):
    """-> per scored set: dict(key, ratio [64], P [64, 4], Y [64, 4], cruise bool) for every scored key found in all four sources"""
    z = np.load(a.scores, allow_pickle=False)
    keys = [(str(l), str(t), int(s), int(r)) for l, t, s, r in zip(z["log"], z["token"], z["step"], z["rank"])]
    sc = z["scores"]
    sets = read_jsonl_dir(a.sets, "onpolicy_*.jsonl*", key_set, ("traj",))
    tg = {}
    for fp in sorted(glob.glob(os.path.join(a.targets, "targets_rank*.jsonl*"))):
        with _open(fp) as fh:
            for ln in fh:
                if ln.startswith("{"):
                    r = json.loads(ln)
                    tg[(r["log_name"], r.get("token", ""), int(r["step"]), int(r.get("rank", 0)))] = {"ego": r["ego"], "traj": r["traj"]}
    pdm = read_jsonl_dir(a.pdm, "pdm_*.jsonl*", key_pdm, PDM)
    toks, miss = [], {"sets": 0, "targets": 0, "pdm": 0}
    for i, k in enumerate(keys):
        if k not in sets:
            miss["sets"] += 1
        elif k not in tg:
            miss["targets"] += 1
        elif k not in pdm or any(len(pdm[k].get(c, ())) != 64 for c in PDM):
            miss["pdm"] += 1
        else:
            v4 = chord_v(sets[k]["traj"])
            ego = tg[k]["ego"]
            v0, ax0 = float(ego[6]), float(ego[2])
            gv4 = float(chord_v(np.asarray(tg[k]["traj"])[None, ...])[0])
            cruise = v0 > 5.0 and abs(ax0) < 0.3 and abs(gv4 - v0) < 1.0
            P = 1.0 / (1.0 + np.exp(-sc[i][:, :4].astype(np.float64)))
            Y = np.stack([np.asarray(pdm[k][c], np.float64) for c in PDM[:4]], -1)
            toks.append({"key": k, "ratio": v4 / max(v0, 1e-6), "P": P, "Y": Y, "cruise": bool(cruise)})
    return toks, miss, len(keys)


def rho(x, y):
    if x.size < 5:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse(toks, boot=B, seed=SEED):
    cr = [t for t in toks if t["cruise"]]
    logs = np.array([t["key"][0] for t in cr]) if cr else np.array([])
    res = {"n_scored_tokens": len(toks), "n_cruise_tokens": len(cr), "n_cruise_logs": int(len(set(logs.tolist()))), "estimator": "average-rank Spearman; paired log-cluster bootstrap",
           "boot": boot, "seed": seed, "heads": {}}
    if not cr:
        res["reading"] = "NO CRUISE TOKENS: Stage 0 cannot be read"
        return res
    ul, li = np.unique(logs, return_inverse=True)
    draws = np.random.default_rng(seed).choice(len(ul), size=(boot, len(ul)), replace=True)
    per = {}
    for h, j in HEADS.items():
        gp, gt, ok = np.full(len(cr), np.nan), np.full(len(cr), np.nan), np.zeros(len(cr), bool)
        for n, t in enumerate(cr):
            rp, rt = rho(t["ratio"], t["P"][:, j]), rho(t["ratio"], t["Y"][:, j])
            if np.isfinite(rp) and np.isfinite(rt):
                gp[n], gt[n], ok[n] = rp, rt, True
        gap = gp - gt
        s = np.bincount(li[ok], weights=gap[ok], minlength=len(ul))
        c = np.bincount(li[ok], minlength=len(ul)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bm = s[draws].sum(1) / c[draws].sum(1)
        bm = np.where(np.isfinite(bm), bm, np.nan)
        pt = float(s.sum() / max(c.sum(), 1.0)) if ok.any() else float("nan")
        per[h] = bm
        res["heads"][h] = {"n_tokens": int(ok.sum()), "pred_mean": float(np.nanmean(gp)) if ok.any() else None, "true_mean": float(np.nanmean(gt)) if ok.any() else None,
                           "gap_mean": pt, "gap_lo": float(np.nanpercentile(bm, 2.5)) if ok.any() else None, "gap_hi": float(np.nanpercentile(bm, 97.5)) if ok.any() else None}
    m = np.nanmean(np.stack([per["TTC"], per["NC"]]), axis=0)
    ok = np.isfinite(m)
    pt = float(np.mean([res["heads"]["TTC"]["gap_mean"], res["heads"]["NC"]["gap_mean"]]))
    lo, hi = float(np.percentile(m[ok], 2.5)), float(np.percentile(m[ok], 97.5))
    res["ttc_nc_mean_gap"] = {"point": pt, "lo": lo, "hi": hi, "n_draws_used": int(ok.sum())}
    z1 = pt <= -0.20 and hi < 0.0
    z2 = (lo <= 0.0 <= hi) or abs(pt) < 0.10
    res["Z1"], res["Z2"] = bool(z1), bool(z2 and not z1)
    res["reading"] = ("Z1: gap PRESENT on navtrain held-out -> run Stage 1" if z1 else "Z2: gap ABSENT (a distribution shift) -> Stage 1 is not the lever; localise the shift by city/scenario/speed band"
                      if z2 else "INCONCLUSIVE -> run Stage 1, record Stage 0 as inconclusive")
    return res


def selftest():
    """planted fixtures written in the REAL file formats: (a) predicted head anti-correlated with speed while the truth is not -> Z1; (b) predicted == truth ranks -> Z2; (c) too few cruise tokens handled"""
    import tempfile
    rng = np.random.default_rng(3)
    outs = {}
    for name, planted in (("planted_gap", True), ("null", False)):
        with tempfile.TemporaryDirectory() as td:
            sd, tdir, pd = (os.path.join(td, x) for x in ("sets", "targets", "pdm"))
            for d in (sd, tdir, pd):
                os.makedirs(d)
            keys, sc, setr, tgr, pdr = [], [], [], [], []
            for n in range(240):
                log = f"LOG{n % 24:02d}"
                k = (log, f"tok{n:04d}", 0, 0)
                v0 = 8.0 + rng.random() * 4
                speed = v0 * (0.5 + rng.random(64))                                    # 64 proposals, constant-speed rays at 0.2 s steps
                traj = np.stack([np.cumsum(np.full(20, s * 0.2)) for s in speed], 0)[..., None]
                traj = np.concatenate([traj, np.zeros_like(traj)], -1)
                setr.append({"kind": "onpolicy_set", "log_name": log, "token": k[1], "step": 0, "rank": 0, "traj": traj.tolist()})
                gt = np.stack([np.cumsum(np.full(20, v0 * 0.2)), np.zeros(20), np.zeros(20)], -1)
                tgr.append({"log_name": log, "token": k[1], "step": 0, "rank": 0, "ego": [0, 0, 0.0, 0, 0, 0, v0], "traj": gt.tolist()})
                y = rng.random((64, 4))                                                 # truth: unrelated to speed
                logit = (y if not planted else -speed[:, None] / 4.0 + 0.05 * rng.standard_normal((64, 4)))
                pdr.append({"key": list(k), "nc": y[:, 0].tolist(), "dac": y[:, 1].tolist(), "ep": y[:, 2].tolist(), "ttc": y[:, 3].tolist(), "c": y[:, 0].tolist(), "ddc": y[:, 0].tolist()})
                sc.append(np.concatenate([logit * 3.0, np.zeros((64, 3))], 1))
                keys.append(k)
            for d, fn, rows in ((sd, "onpolicy_x.jsonl", setr), (tdir, "targets_rank0.jsonl", tgr), (pd, "pdm_x.jsonl", pdr)):
                with open(os.path.join(d, fn), "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in rows))
            sp = os.path.join(td, "scores_t_base.npz")
            np.savez(sp, log=np.array([k[0] for k in keys]), token=np.array([k[1] for k in keys]), step=np.zeros(len(keys), np.int64), rank=np.zeros(len(keys), np.int64),
                     ckpt_step=np.zeros(len(keys), np.int64), scores=np.stack(sc), space=np.asarray("logit"))
            ns = argparse.Namespace(scores=sp, sets=sd, targets=tdir, pdm=pd)
            toks, miss, n = load_tokens(ns)
            assert n == 240 and miss == {"sets": 0, "targets": 0, "pdm": 0} and len(toks) == 240 and all(t["cruise"] for t in toks), (n, miss)
            outs[name] = analyse(toks, boot=2000)
    assert outs["planted_gap"]["Z1"] is True and outs["planted_gap"]["ttc_nc_mean_gap"]["hi"] < 0, outs["planted_gap"]["ttc_nc_mean_gap"]
    assert outs["null"]["Z2"] is True and outs["null"]["Z1"] is False, outs["null"]["ttc_nc_mean_gap"]
    # a missing PDM label drops the token (counted), it never fills a default
    empty = analyse([], boot=10)
    assert "cannot be read" in empty["reading"]
    print("stage0 selftest ok: planted", outs["planted_gap"]["ttc_nc_mean_gap"], "null", outs["null"]["ttc_nc_mean_gap"])
    print("ZZTEST_PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores")
    ap.add_argument("--sets")
    ap.add_argument("--targets")
    ap.add_argument("--pdm")
    ap.add_argument("--out", default="stage0.json")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.scores and a.sets and a.targets and a.pdm):
        sys.exit("need --scores --sets --targets --pdm (or --selftest)")
    toks, miss, n = load_tokens(a)
    res = analyse(toks, a.boot)
    res["inputs"] = {"scores": a.scores, "n_scored": n, "dropped_missing": miss}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_scored_tokens", "n_cruise_tokens", "ttc_nc_mean_gap", "reading") if k in res}, indent=1))


if __name__ == "__main__":
    main()
