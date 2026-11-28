#!/usr/bin/env python
"""preserr_t0.py -- card PREREG_V8F_PROOF_ADD116 (PRESERR-T0). Tier T0: the token reader's presence on recorded evaluation frames against the label (a diagnostic of the
reader, not driving performance). CPU only; reads banked rule logs and their side files, and the evaluation join; runs no rollout and no fit.

Frames of a model seed: the unique (clip index, frame index) pairs of the rule log and the later-look rule log of the banked reader-fed run I10RQ50KF5Hb20_s<m> with a full
history (`ok` of the side file). Label of a frame: join row off[ci] + j; a lead L = marking 1 and a finite gap from 0 to 60 m. For a threshold t the reader marks a frame
when p >= t. Miss rate M(t) = share of L frames not marked; extra-mark rate X(t) = share of frames without L that are marked. Pooled = mean of the two seeds.
Intervals: bootstrap over clips, the same draw for both seeds.

  python preserr_t0.py --stage controls --t1 DIR --join FILE --tests FILE --out CONTROLS.json     prints h<i> PASS|FAIL and one summary line; no value
  python preserr_t0.py --stage names --controls CONTROLS.json --out RESULT.json                   refuses unless h0 to h5 all pass
  python preserr_t0.py --selftest DIR
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD116"
TIER = "T0 (the reader's reads on recorded evaluation frames; a diagnostic of the reader, not driving performance)"
RUN_RR, RUN_RLC = "RQ50KF5Hb20", "RLcKF5Hb20"
SEEDS = (0, 1)
GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 40.0), (40.0, 60.0))
GAP_MAX = 60.0
TOL_P = 1e-6
CONTROLS = ("h0", "h1", "h2", "h3", "h4", "h5")
PRE = "PRESERR-T0-"


def sha_file(p):
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def log_paths(t1, run, m):
    d = os.path.join(t1, "I10%s_s%d" % (run, m), "ticks")
    return [os.path.join(d, "rule_log_I10%s_s%d.npz" % (run, m)), os.path.join(d, "rule_log_mid_I10%s_s%d.npz" % (run, m))]


def label_of(JL, ci, j):
    """-> (maps [n] bool, L [n] bool, gap [n] float64) for logged steps (ci, j): join row off[ci] + j; L = marking 1 and a finite gap from 0 to 60 m"""
    off, n = np.asarray(JL["off"]).astype(np.int64), np.asarray(JL["n"]).astype(np.int64)
    lv, lg = np.asarray(JL["lead_valid"]), np.asarray(JL["lead_gap"], dtype=np.float64)
    ci, j = np.asarray(ci).astype(np.int64), np.asarray(j).astype(np.int64)
    maps = (ci >= 0) & (ci < len(n))
    maps[maps] = (j[maps] >= 0) & (j[maps] < n[ci[maps]])
    row = np.zeros(len(ci), np.int64)
    row[maps] = off[ci[maps]] + j[maps]
    g = np.where(maps, lg[row], np.nan)
    L = maps & (lv[row] == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX)
    return maps, L, g


def load_seed(t1, m):
    """the two logs of the reader-fed run of seed m with their side files, row arrays concatenated, plus the per-log control facts of h1"""
    cols = {k: [] for k in ("ci", "j", "present", "src_bad", "p", "ok")}
    bad, files = [], []
    for lp in log_paths(t1, RUN_RR, m):
        sp = lp[:-4] + "_rd.npz"
        z, s = npz(lp), npz(sp)
        files += [lp, sp]
        nm = os.path.basename(lp)
        n = len(z["ci"])
        if len(s["p"]) != n or len(s["ok"]) != n:
            bad.append("%s: %d rows, side file %d" % (nm, n, len(s["p"])))
            continue
        p, ps = np.asarray(s["p"], np.float64), np.asarray(z["p_src"], np.float64)
        dp = np.abs(p - ps)
        n_p = int((~(dp <= TOL_P)).sum())
        if n_p:
            bad.append("%s: %d rows where p of the side file is not p_src of the log within %g" % (nm, n_p, TOL_P))
        rec = (p >= 0.5) & np.asarray(s["ok"], bool) & ~np.asarray(z["src_bad"], bool)
        n_r = int((rec != np.asarray(z["present"], bool)).sum())
        if n_r:
            bad.append("%s: %d rows where present is not p >= 0.5 and ok and not src_bad" % (nm, n_r))
        for k, v in (("ci", z["ci"]), ("j", z["j"]), ("present", z["present"]), ("src_bad", z["src_bad"]), ("p", p), ("ok", s["ok"])):
            cols[k].append(np.asarray(v))
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    return out, bad, files


def frames_of(rows):
    """unique (ci, j) of the rows with a full history -> dict(ci, j, p, present, n_src_bad, spread): p and present of the first row of the frame; spread = the largest
    difference of p between repeats of one frame"""
    ok = np.asarray(rows["ok"], bool)
    ci, j, p = rows["ci"][ok].astype(np.int64), rows["j"][ok].astype(np.int64), np.asarray(rows["p"], np.float64)[ok]
    pres, sb = np.asarray(rows["present"], bool)[ok], np.asarray(rows["src_bad"], bool)[ok]
    key = ci * 1000000 + j
    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    mx, mn = np.full(len(u), -np.inf), np.full(len(u), np.inf)
    np.maximum.at(mx, inv, p)
    np.minimum.at(mn, inv, p)
    return {"key": u, "ci": ci[first], "j": j[first], "p": p[first], "present": pres[first], "n_src_bad": int(sb.sum()), "spread": float((mx - mn).max()) if len(u) else 0.0}


def clip_stats(F, L, g, n_clip):
    """per-clip counts of one seed: nL, nN, miss[t], mark[t], and per gap bin nLb, missb (at 0.5)"""
    ci, p = F["ci"], F["p"]
    T, B = len(GRID), len(BINS)
    S = {"nL": np.zeros(n_clip), "nN": np.zeros(n_clip), "miss": np.zeros((T, n_clip)), "mark": np.zeros((T, n_clip)), "nLb": np.zeros((B, n_clip)), "missb": np.zeros((B, n_clip))}
    np.add.at(S["nL"], ci[L], 1.0)
    np.add.at(S["nN"], ci[~L], 1.0)
    for k, t in enumerate(GRID):
        mk = p >= t
        np.add.at(S["miss"][k], ci[L & ~mk], 1.0)
        np.add.at(S["mark"][k], ci[~L & mk], 1.0)
    m5 = p >= 0.5
    for b, (lo, hi) in enumerate(BINS):
        inb = L & (g >= lo) & ((g < hi) if hi < GAP_MAX else (g <= hi))
        np.add.at(S["nLb"][b], ci[inb], 1.0)
        np.add.at(S["missb"][b], ci[inb & ~m5], 1.0)
    return S


def _div(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def measures(stats, w):
    """stats: list (one per seed) of clip_stats; w [..., n_clip] clip weights (1 for the point, bootstrap counts for a draw) -> pooled arrays over the leading axes of w"""
    M = np.mean([_div((s["miss"][:, None, :] * w[None]).sum(-1), (s["nL"] * w).sum(-1)[None]) for s in stats], axis=0)          # [T, draws]
    X = np.mean([_div((s["mark"][:, None, :] * w[None]).sum(-1), (s["nN"] * w).sum(-1)[None]) for s in stats], axis=0)
    i5 = GRID.index(0.5)
    lg, sm = [2, 3], [0, 1]
    Ml = np.mean([_div((s["missb"][lg].sum(0) * w).sum(-1), (s["nLb"][lg].sum(0) * w).sum(-1)) for s in stats], axis=0)
    Ms = np.mean([_div((s["missb"][sm].sum(0) * w).sum(-1), (s["nLb"][sm].sum(0) * w).sum(-1)) for s in stats], axis=0)
    return {"M": M, "X": X, "rM": _div(M, np.where(M[i5] > 0, M[i5], 0.0)[None]), "rX": _div(X, np.where(X[i5] > 0, X[i5], 0.0)[None]), "M_large": Ml, "M_small": Ms, "D": Ml - Ms}


def _iv(a):
    a = np.asarray(a, np.float64)
    if a.size == 0 or np.mean(~np.isfinite(a)) > 0.05:
        return None
    return [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]


def auc(p, L):
    n1, n0 = int(L.sum()), int((~L).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    r = np.empty(len(p), np.float64)
    ps = p[order]
    i = 0
    while i < len(ps):                                                    # mid ranks for ties
        k = i
        while k + 1 < len(ps) and ps[k + 1] == ps[i]:
            k += 1
        r[order[i:k + 1]] = 0.5 * (i + k) + 1.0
        i = k + 1
    return float((r[L].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def read_values(per_seed, n_clip, n_boot, seed=20261122):
    """per_seed: list of (F, L, g) -> the payload of the card"""
    stats = [clip_stats(F, L, g, n_clip) for F, L, g in per_seed]
    has = np.zeros(n_clip, bool)
    for s in stats:
        has |= (s["nL"] + s["nN"]) > 0
    idx = np.flatnonzero(has)
    sub = [{k: v[..., idx] for k, v in s.items()} for s in stats]
    pt = measures(sub, np.ones((1, len(idx))))
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)) for _ in range(n_boot)]).astype(np.float64)
    bs = measures(sub, w)
    i5 = GRID.index(0.5)
    out = {"n_clips_with_frames": int(len(idx)), "n_boot": int(n_boot), "grid": list(GRID),
           "per_seed": [{"n_frames": int(len(F["p"])), "n_L": int(L.sum()), "n_not_L": int((~L).sum()), "M_0.5": float(_div(s["miss"][i5].sum(), s["nL"].sum())),
                         "X_0.5": float(_div(s["mark"][i5].sum(), s["nN"].sum())), "auc": auc(F["p"], L),
                         "M_by_t": [float(_div(s["miss"][k].sum(), s["nL"].sum())) for k in range(len(GRID))],
                         "X_by_t": [float(_div(s["mark"][k].sum(), s["nN"].sum())) for k in range(len(GRID))],
                         "bins": [{"lo": lo, "hi": hi, "n_L": int(s["nLb"][b].sum()), "n_miss": int(s["missb"][b].sum())} for b, (lo, hi) in enumerate(BINS)]}
                        for (F, L, g), s in zip(per_seed, stats)],
           "pooled": {"M_0.5": {"value": float(pt["M"][i5, 0]), "interval": _iv(bs["M"][i5])}, "X_0.5": {"value": float(pt["X"][i5, 0]), "interval": _iv(bs["X"][i5])},
                      "by_t": [{"t": t, "M": float(pt["M"][k, 0]), "X": float(pt["X"][k, 0]),
                                "M_ratio": {"value": (float(pt["rM"][k, 0]) if np.isfinite(pt["rM"][k, 0]) else None), "interval": _iv(bs["rM"][k])},
                                "X_ratio": {"value": (float(pt["rX"][k, 0]) if np.isfinite(pt["rX"][k, 0]) else None), "interval": _iv(bs["rX"][k])}} for k, t in enumerate(GRID)],
                      "M_large_gaps": {"value": float(pt["M_large"][0]), "interval": _iv(bs["M_large"])}, "M_small_gaps": {"value": float(pt["M_small"][0]), "interval": _iv(bs["M_small"])},
                      "large_minus_small": {"value": float(pt["D"][0]), "interval": _iv(bs["D"])}}}
    return out


def names_of(payload):
    """card section 4"""
    P = payload["pooled"]
    by = {round(r["t"], 1): r for r in P["by_t"]}
    tstar = None
    for t in (0.4, 0.3, 0.2, 0.1):
        v = by[t]["M_ratio"]["value"]
        if v is not None and v <= 0.5:
            tstar = t
            break
    room = False
    if tstar is not None and P["M_0.5"]["value"] > 0 and P["X_0.5"]["value"] > 0:
        iv, xr = by[tstar]["M_ratio"]["interval"], by[tstar]["X_ratio"]["value"]
        room = iv is not None and iv[1] < 0.75 and xr is not None and xr <= 2.0
    out = {"THRESHOLD": PRE + ("THRESHOLD-HAS-ROOM" if room else "THRESHOLD-HAS-NO-ROOM"), "t_star": tstar}
    ml, ms, d = P["M_large_gaps"]["value"], P["M_small_gaps"]["value"], P["large_minus_small"]["interval"]
    if np.isfinite(ml) and np.isfinite(ms) and d is not None and ml >= 2.0 * ms and d[0] > 0.0:
        out["WHERE"] = PRE + "MISSES-AT-LARGE-GAPS"
    elif np.isfinite(ml) and np.isfinite(ms) and d is not None and ms >= 2.0 * ml and d[1] < 0.0:
        out["WHERE"] = PRE + "MISSES-AT-SMALL-GAPS"
    else:
        out["WHERE"] = PRE + "MISSES-NOT-ORDERED-BY-GAP"
    return out


def _scalar_text(v):
    a = np.asarray(v).reshape(-1)
    x = a[0] if a.size else ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def run_controls(a):
    JL = npz(a.join)
    n_clip = len(np.asarray(JL["n"]))
    join_sha = sha_file(a.join)
    c, files = {}, []
    bad0, bad5 = [], []
    for m in SEEDS:                                                                            # h0: label-form logs against the join
        for lp in log_paths(a.t1, RUN_RLC, m):
            z = npz(lp)
            files.append(lp)
            maps, L, _ = label_of(JL, z["ci"], z["j"])
            n_un, n_dis = int((~maps).sum()), int((np.asarray(z["present"], bool) != L).sum())
            if n_un or n_dis:
                bad0.append("%s: %d rows map to no join row, %d rows where present is not the lead of the join row" % (os.path.basename(lp), n_un, n_dis))
            named = _scalar_text(z["join_sha256"]).lower() if "join_sha256" in z else ""
            if not (len(named) >= 16 and (join_sha.startswith(named) or named.startswith(join_sha))):
                bad5.append("%s names the join hash %r, the join read has %s" % (os.path.basename(lp), named[:16], join_sha[:16]))
    c["h0"] = {"pass": not bad0, "failures": bad0}
    per, bad1, bad2, bad3 = [], [], [], []
    for m in SEEDS:
        rows, b1, fl = load_seed(a.t1, m)
        bad1 += ["seed %d: %s" % (m, x) for x in b1]
        files += fl
        F = frames_of(rows)
        maps, L, g = label_of(JL, F["ci"], F["j"])
        if F["spread"] > TOL_P:
            bad2.append("seed %d: p differs by %g between repeats of a frame" % (m, F["spread"]))
        if int((~maps).sum()):
            bad2.append("seed %d: %d frames map to no join row" % (m, int((~maps).sum())))
        if F["n_src_bad"]:
            bad3.append("seed %d: %d rows with src_bad" % (m, F["n_src_bad"]))
        mk = F["p"] >= 0.5
        if int((mk != F["present"]).sum()):
            bad3.append("seed %d: %d frames where p >= 0.5 is not present of the log" % (m, int((mk != F["present"]).sum())))
        per.append((F, L, g))
    if len(per) == 2 and not (len(per[0][0]["key"]) == len(per[1][0]["key"]) and np.array_equal(per[0][0]["key"], per[1][0]["key"])):
        bad2.append("the two seeds do not have the same set of frames (%d, %d)" % (len(per[0][0]["key"]), len(per[1][0]["key"])))
    if any(len(F["key"]) == 0 for F, _, _ in per):
        bad2.append("a seed has no frame")
    c["h1"] = {"pass": not bad1, "failures": bad1}
    c["h2"] = {"pass": not bad2, "failures": bad2}
    c["h3"] = {"pass": not bad3, "failures": bad3}
    tt = io.open(a.tests, encoding="utf-8", errors="replace").read() if a.tests and os.path.isfile(a.tests) else ""
    hits = re.findall(r"(\d+) passed", tt)
    bad4 = []
    if not (hits and int(hits[-1]) > 0):
        bad4.append("the lead's test output has no count of passed tests above 0")
    if re.search(r"\b\d+ (failed|errors?)\b", tt):
        bad4.append("the lead's test output has a failed or error count")
    c["h4"] = {"pass": not bad4, "failures": bad4}
    c["h5"] = {"pass": not bad5, "failures": bad5}
    ok = all(c[k]["pass"] for k in CONTROLS)
    res = {"card": CARD, "tier": TIER, "controls": c, "all_pass": ok,
           "inputs": {"join_sha256": join_sha, "files": {os.path.basename(f): sha_file(f) for f in files}, "tool_sha256": sha_file(os.path.abspath(__file__))}}
    if ok:
        res["payload"] = read_values(per, n_clip, a.n_boot)
    return res


def print_names(payload, names):
    P = payload["pooled"]
    f = lambda x: "none" if x is None else "%.4f" % x
    iv = lambda x: "none" if x is None else "[%.4f, %.4f]" % (x[0], x[1])
    L = ["card %s, tier %s" % (CARD, TIER), "clips with frames %d, bootstrap draws %d" % (payload["n_clips_with_frames"], payload["n_boot"])]
    for m, s in enumerate(payload["per_seed"]):
        L.append("seed %d: frames %d, with a lead of the label %d, without %d, M(0.5) %s, X(0.5) %s, auc %s" % (m, s["n_frames"], s["n_L"], s["n_not_L"], f(s["M_0.5"]), f(s["X_0.5"]), f(s["auc"])))
        L.append("seed %d bins (lo-hi m: misses/leads): %s" % (m, ", ".join("%g-%g: %d/%d" % (b["lo"], b["hi"], b["n_miss"], b["n_L"]) for b in s["bins"])))
    L.append("pooled M(0.5) %s %s, X(0.5) %s %s" % (f(P["M_0.5"]["value"]), iv(P["M_0.5"]["interval"]), f(P["X_0.5"]["value"]), iv(P["X_0.5"]["interval"])))
    for r in P["by_t"]:
        L.append("t %.1f: M %s, X %s, M/M(0.5) %s %s, X/X(0.5) %s %s" % (r["t"], f(r["M"]), f(r["X"]), f(r["M_ratio"]["value"]), iv(r["M_ratio"]["interval"]), f(r["X_ratio"]["value"]), iv(r["X_ratio"]["interval"])))
    L.append("pooled miss rate at gaps from 20 m: %s %s; below 20 m: %s %s; difference %s %s" % (
        f(P["M_large_gaps"]["value"]), iv(P["M_large_gaps"]["interval"]), f(P["M_small_gaps"]["value"]), iv(P["M_small_gaps"]["interval"]),
        f(P["large_minus_small"]["value"]), iv(P["large_minus_small"]["interval"])))
    L.append("t_star %s" % names["t_star"])
    L.append("NAME %s" % names["THRESHOLD"])
    L.append("NAME %s" % names["WHERE"])
    return L


# ============================================================================================ synthetic world (tests and selftest)
def make_world(base, miss_p=(0.30, 0.30), extra_p=0.05, sep=4.0, fault=None, n_clip=12, n_fr=60, seed=5):
    """a small t1 folder and join with known behaviour. L frames get a logit around +sep/2 shifted so that about miss_p of them fall below 0.5; frames without L around -sep.
    `sep` large makes a lower threshold recover misses at few extra marks (room); see the tests."""
    rng = np.random.default_rng(seed)
    os.makedirs(base, exist_ok=True)
    t1 = os.path.join(base, "t1")
    off = np.arange(n_clip) * n_fr
    lv = np.zeros(n_clip * n_fr, np.int64)
    gap = np.full(n_clip * n_fr, np.nan)
    for c in range(n_clip):
        a0 = int(rng.integers(5, 20))
        lv[off[c] + a0: off[c] + a0 + 30] = 1
        gap[off[c] + a0: off[c] + a0 + 30] = np.linspace(5.0, 58.0, 30)
    gap[off[1] + 10] = 75.0 if lv[off[1] + 10] else gap[off[1] + 10]
    jp = os.path.join(base, "join.npz")
    np.savez(jp, clip=np.array(["c%02d" % i for i in range(n_clip)]), off=off, n=np.full(n_clip, n_fr), lead_valid=lv, lead_gap=gap)
    jsha = sha_file(jp)
    JL = npz(jp)
    ci = np.repeat(np.arange(n_clip), 20)
    j_t = np.tile(np.arange(0, 40, 2), n_clip)
    for m in SEEDS:
        for run in (RUN_RR, RUN_RLC):
            os.makedirs(os.path.join(t1, "I10%s_s%d" % (run, m), "ticks"), exist_ok=True)
        r2 = np.random.default_rng(100 + m)
        for mid, jj in ((False, j_t), (True, j_t + 5)):
            _, L, g = label_of(JL, ci, jj)
            u = r2.random(len(ci))
            # L frames: a share miss_p[m] gets a logit in (-sep/8 - 0.4, -0.05) ("near misses" when sep is large: p between about 0.3 and 0.49), the rest above 0.5
            lo_p = 1.0 / (1.0 + np.exp(sep / 8.0 + 0.4))
            pL = np.where(u < miss_p[m], lo_p + (0.49 - lo_p) * r2.random(len(ci)), 0.55 + 0.44 * r2.random(len(ci)))
            pN = np.where(r2.random(len(ci)) < extra_p, 0.55 + 0.44 * r2.random(len(ci)), (0.02 + 0.2 * r2.random(len(ci))) if sep >= 2.0 else (0.02 + 0.47 * r2.random(len(ci))))
            p = np.where(L, pL, pN)
            ok = np.ones(len(ci), bool) if mid else (jj >= 2)
            sb = np.zeros(len(ci), bool)
            pres = (p >= 0.5) & ok
            if fault == "h1" and m == 0 and not mid:
                pres = pres.copy()
                pres[3] = ~pres[3]
            if fault == "h3" and m == 1 and mid:
                sb = sb.copy()
                k = int(np.flatnonzero(pres)[0])
                sb[k] = True
                pres = pres & ~sb
            nm = "rule_log_%sI10%s_s%d" % ("mid_" if mid else "", RUN_RR, m)
            d = os.path.join(t1, "I10%s_s%d" % (RUN_RR, m), "ticks")
            jlog = jj if not (fault == "h2" and m == 1 and mid) else np.where(np.arange(len(jj)) == 7, n_fr + 3, jj)
            np.savez(os.path.join(d, nm + ".npz"), ci=ci, j=jlog, present=pres, src_bad=sb, p_src=p.astype(np.float64))
            np.savez(os.path.join(d, nm + "_rd.npz"), p=(p.astype(np.float32) if not mid else p), ok=ok)
            presL = L.copy()
            if fault == "h0" and m == 0 and mid:
                presL[5] = ~presL[5]
            nm2 = "rule_log_%sI10%s_s%d" % ("mid_" if mid else "", RUN_RLC, m)
            np.savez(os.path.join(t1, "I10%s_s%d" % (RUN_RLC, m), "ticks", nm2 + ".npz"), ci=ci, j=jj, present=presL, join_sha256=np.array(jsha if fault != "h5" else "0" * 64))
    tp = os.path.join(base, "lead_test.out")
    io.open(tp, "w", encoding="utf-8", newline="\n").write("....\n4 passed in 0.1s\n" if fault != "h4" else "..F.\n1 failed, 3 passed in 0.1s\n")
    return t1, jp, tp


def selftest(base):
    ok = True
    ns = argparse.Namespace
    t1, jp, tp = make_world(os.path.join(base, "clean"))
    r = run_controls(ns(t1=t1, join=jp, tests=tp, n_boot=300))
    ok &= r["all_pass"] is True and "payload" in r
    for fault in ("h0", "h1", "h2", "h3", "h4", "h5"):
        t1, jp, tp = make_world(os.path.join(base, "f_" + fault), fault=fault)
        r = run_controls(ns(t1=t1, join=jp, tests=tp, n_boot=50))
        ok &= r["all_pass"] is False and r["controls"][fault]["pass"] is False and "payload" not in r
    print("ZZ PRESERR-T0 SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", default="")
    for nm in ("t1", "join", "tests", "controls", "out"):
        ap.add_argument("--" + nm, default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.selftest)
    if not a.stage or not a.out:
        print("REFUSED: --stage and --out are required")
        return 2
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    if a.stage == "controls":
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if res["controls"][k]["pass"] else "FAIL"))
        print("ZZ PRESERR-T0 controls all_pass %s" % res["all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any((v or {}).get("pass") is not True for v in cj["controls"].values()):
        print("REFUSED: the controls file is not all-pass with h0 to h5 present: no name is written")
        print("ZZ PRESERR-T0 names refused")
        return 3
    names = names_of(cj["payload"])
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps({"card": CARD, "tier": TIER, "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}, indent=1))
    for ln in print_names(cj["payload"], names) + ["ZZ PRESERR-T0 names done"]:
        print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
