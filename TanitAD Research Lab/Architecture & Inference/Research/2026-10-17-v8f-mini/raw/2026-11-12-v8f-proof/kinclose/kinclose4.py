#!/usr/bin/env python
"""kinclose4.py -- tool of card PREREG_V8F_PROOF_ADD91 (GAPRANGE), tier T0 (a diagnostic on logged frames; no rollout).

The reader's gap error against the label gap over range bins, and slopes of the reader's own gap: RG_W (run found from the reader's gap with the
label's rule), RL_W (run found from the label gap, a bound), RP_W (plain window). Numpy only. Controls q0 to q3 are computed by the read, q4 is the
self-test. With one control failed no name is written and every value is held under the key "withheld".

  python kinclose4.py --selftest [--closeread-tools DIR]
  python kinclose4.py read --dump-dir D --episode-map E --add81-controls C --out-dir O [--closeread-tools DIR]
  python kinclose4.py show --result R --controls C
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kinclose2 as K2  # noqa: E402
import kinclose3 as K3  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD91"
WS = (5, 10, 25)
BINS = (("ALL", -np.inf, np.inf), ("NEAR", -np.inf, 20.0), ("MID", 20.0, 40.0), ("FAR", 40.0, np.inf))
RUN_MIN, STEP_MAX_M = 3, 3.0
REQ_R, BOOT_B, BOOT_SEED = 0.9, 2000, 91
SD_IN_M, SD_OUT_M = 1.0, 2.0
MIN_FRAMES, MIN_EPISODES = 100, 5
Q1_GAP_TOL, Q1_SLOPE_TOL, Q_TOL = 1e-9, 1e-6, 1e-9
PLANT_OFFSET_M, PLANT_SDS = 2.0, (1.0, 8.0)
PLANT_BIN_SD = (("NEAR", 1.0), ("MID", 8.0), ("FAR", 8.0))
FAULTS = set()  # in-process regression switches of the self-test; the command line never sets them


# ============================================================================================ frames
def make_ctx(D, ep_map):
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    nc, T = len(D["clips"]), int(fr.max()) + 1
    f3 = np.stack([np.asarray(D["gap"], np.float64), np.asarray(D["p"], np.float64), np.asarray(D["clos"], np.float64)], 1)
    ok = np.asarray(D["valid"], bool) & np.isfinite(f3).all(1)
    G, RM = np.full((nc, T), np.nan), np.zeros((nc, T), bool)
    G[cid[ok], fr[ok]] = f3[ok, 0]
    RM[cid[ok], fr[ok]] = True
    lg = np.asarray(D["lab_gap"], np.float64)
    lp = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    LG, LM = np.full((nc, T), np.nan), np.zeros((nc, T), bool)
    LG[cid[lp], fr[lp]] = lg[lp]
    LM[cid[lp], fr[lp]] = True
    rows = K2.CF.build_rows(D, 25, ep_map)
    name2i = {str(c): i for i, c in enumerate(D["clips"])}
    ci = np.asarray([name2i[str(c)] for c in rows["clip"]], np.int64)
    return {"clips": [str(c) for c in D["clips"]], "G": G, "RM": RM, "LG": LG, "LM": LM, "rows": rows, "ci": ci, "fr": np.asarray(rows["frame"], np.int64),
            "dt": float(D["dt_s"]), "T": T, "y": np.asarray(rows["y"], np.float64), "ep": np.asarray(rows["ep"])}


def meas(c):
    """scored rows whose own frame has a valid reader gap and a label gap."""
    return c["RM"][c["ci"], c["fr"]] & c["LM"][c["ci"], c["fr"]]


def bin_masks(c):
    lg, m = c["LG"][c["ci"], c["fr"]], meas(c)
    out = {}
    for name, lo, hi in BINS:
        if name == "NEAR" and "bin_edge" in FAULTS:
            hi = 25.0
        with np.errstate(invalid="ignore"):
            out[name] = m & (lg >= lo) & (lg < hi)
    return out


def gap_err(c):
    """reader's gap minus label gap at the frame of every scored row (NaN where one is missing)."""
    e = c["G"][c["ci"], c["fr"]] - c["LG"][c["ci"], c["fr"]]
    return -e if "sign_flip" in FAULTS else e


# ============================================================================================ runs and slopes
def run_index(c, W, src):
    """per scored row the frames of the run, newest first, at most W. src 'reader': going back while the reader's gap is valid and its step is at most
    3.0 m. src 'label': while label and reader value exist and the step of the label gap is at most 3.0 m. None when the row's own frame has no value."""
    RM, LM = c["RM"], c["LM"]
    X = c["G"] if src == "reader" else c["LG"]
    gate = "no_step_gate" not in FAULTS
    out = []
    for i, j in zip(c["ci"], c["fr"]):
        def ok(f):
            return bool(RM[i, f]) if src == "reader" else bool(RM[i, f] and LM[i, f])
        if not ok(j):
            out.append(None)
            continue
        idx, f = [int(j)], int(j) - 1
        while f >= 0 and len(idx) < W and ok(f) and ((not gate) or abs(X[i, f + 1] - X[i, f]) <= STEP_MAX_M):
            idx.append(f)
            f -= 1
        out.append(idx)
    return out


def run_slope(c, runs):
    """least-squares slope against time of the reader's gap over each run; a run below 3 frames gives 0.0 (as the rule's source does); no run -> NaN."""
    G, dt = c["G"], c["dt"]
    est, short = np.full(len(runs), np.nan), np.zeros(len(runs), bool)
    for r, (i, idx) in enumerate(zip(c["ci"], runs)):
        if idx is None:
            continue
        if len(idx) < RUN_MIN:
            est[r], short[r] = 0.0, True
            continue
        g = G[i, idx[::-1]]
        t = np.arange(len(idx), dtype=np.float64) * dt
        tb = t.mean()
        est[r] = float(((t - tb) * (g - g.mean())).sum() / ((t - tb) ** 2).sum())
    return est, short


def plain_slope(c, W):
    sl, _ = K2.slope_rows(c["G"], c["RM"], c["ci"], c["fr"], W, c["dt"])
    return np.where(c["RM"][c["ci"], c["fr"]], sl, np.nan)


# ============================================================================================ statistics
def point_err(es):
    e = np.concatenate([np.asarray(x, np.float64) for x in es]) if es else np.zeros(0)
    if len(e) < 2:
        return {"n": int(len(e)), "mean": float("nan"), "sd": float("nan"), "mean_abs": float("nan")}
    return {"n": int(len(e)), "mean": float(e.mean()), "sd": float(e.std()), "mean_abs": float(np.abs(e).mean())}


def boot_err(es, eps, B, seed):
    """mean, standard deviation about the mean and mean absolute value of the pooled errors, with a two-level bootstrap (reader seeds, then episodes; one episode draw shared by the seeds)."""
    out = point_err(es)
    out["per_seed"] = [point_err([e]) for e in es]
    alle = sorted(set(np.concatenate([np.asarray(x) for x in eps]).tolist())) if out["n"] else []
    out["n_episodes"] = int(len(alle))
    if out["n"] < 2 or len(alle) < 2:
        return out
    emap = {e: i for i, e in enumerate(alle)}
    ne, S = len(alle), []
    for e, ep in zip(es, eps):
        e = np.asarray(e, np.float64)
        idx = np.asarray([emap[x] for x in np.asarray(ep).tolist()], np.int64)
        S.append(np.stack([np.bincount(idx, minlength=ne).astype(np.float64), np.bincount(idx, e, ne), np.bincount(idx, e * e, ne), np.bincount(idx, np.abs(e), ne)]))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        ks = rng.integers(0, len(S), len(S))
        w = np.bincount(rng.integers(0, ne, ne), minlength=ne).astype(np.float64)
        tot = sum(S[k] @ w for k in ks)
        if tot[0] < 2:
            continue
        m = tot[1] / tot[0]
        d.append((m, np.sqrt(max(tot[2] / tot[0] - m * m, 0.0)), tot[3] / tot[0]))
    d = np.asarray(d)
    for col, key in ((0, "mean_ci"), (1, "sd_ci"), (2, "mean_abs_ci")):
        out[key] = [float(np.percentile(d[:, col], 2.5)), float(np.percentile(d[:, col], 97.5))]
    return out


# ============================================================================================ controls q0 to q3
def controls(ctxs, ks, add81_ctl):
    ctl = {}
    want = [int(add81_ctl["controls"]["t0"]["frames"]["H25_seed%d" % k]["n_tool"]) for k in ks]
    got = [int(len(c["fr"])) for c in ctxs]
    ctl["q0"] = {"n_frames": got, "n_add81": want, "pass": bool(got == want)}
    q1 = {"max_abs_gap_error_m": 0.0, "max_abs_slope_diff_ms": 0.0, "n_rows": 0, "n_without_value": 0}
    for c in ctxs:
        c1 = dict(c, G=c["LG"].copy(), RM=c["LM"].copy())
        m1 = meas(c1)
        e = gap_err(c1)[m1]
        if len(e):
            q1["max_abs_gap_error_m"] = max(q1["max_abs_gap_error_m"], float(np.abs(e).max()))
        est, _ = run_slope(c1, run_index(c1, 5, "reader"))
        f = np.isfinite(est)
        q1["n_rows"] += int(len(est))
        q1["n_without_value"] += int((~f).sum())
        if f.any():
            q1["max_abs_slope_diff_ms"] = max(q1["max_abs_slope_diff_ms"], float(np.abs(est[f] - c["y"][f]).max()))
    q1["pass"] = bool(q1["n_rows"] > 0 and q1["n_without_value"] == 0 and q1["max_abs_gap_error_m"] <= Q1_GAP_TOL and q1["max_abs_slope_diff_ms"] <= Q1_SLOPE_TOL)
    ctl["q1"] = q1

    def unit_on(c, mask):
        n = K3.unit_noise(c, "R")[c["ci"][mask], c["fr"][mask]]
        return n if "no_normalise" in FAULTS else (n - n.mean()) / n.std()

    q2 = {"levels": [], "pass": True}
    for s in PLANT_SDS:
        es = []
        for c in ctxs:
            m = meas(c)
            G2 = c["G"].copy()
            G2[c["ci"][m], c["fr"][m]] = c["LG"][c["ci"][m], c["fr"][m]] + PLANT_OFFSET_M + s * unit_on(c, m)
            es.append(gap_err(dict(c, G=G2))[m])
        p = point_err(es)
        ok = bool(abs(p["mean"] - PLANT_OFFSET_M) <= Q_TOL and abs(p["sd"] - s) <= Q_TOL)
        q2["levels"].append({"planted_sd_m": s, "planted_mean_m": PLANT_OFFSET_M, "mean_m": p["mean"], "sd_m": p["sd"], "n": p["n"], "pass": ok})
        q2["pass"] = bool(q2["pass"] and ok)
    ctl["q2"] = q2

    q3 = {"partition": True, "n_measured": [], "n_in_bins": [], "planted": [], "pass": True}
    planted_es = {b: [] for b, _ in PLANT_BIN_SD}
    skipped = set()
    for c in ctxs:
        bm, m = bin_masks(c), meas(c)
        cnt = sum(bm[b].astype(np.int64) for b, _ in PLANT_BIN_SD)
        q3["partition"] = bool(q3["partition"] and (cnt[m] == 1).all() and (cnt[~m] == 0).all() and bool((bm["ALL"] == m).all()))
        q3["n_measured"].append(int(m.sum()))
        q3["n_in_bins"].append({b: int(bm[b].sum()) for b, _ in PLANT_BIN_SD})
        G3 = c["G"].copy()
        for b, s in PLANT_BIN_SD:
            if int(bm[b].sum()) < 2:
                skipped.add(b)
                continue
            G3[c["ci"][bm[b]], c["fr"][bm[b]]] = c["LG"][c["ci"][bm[b]], c["fr"][bm[b]]] + s * unit_on(c, bm[b])
        e3 = gap_err(dict(c, G=G3))
        for b, _ in PLANT_BIN_SD:
            planted_es[b].append(e3[bm[b]])
    for b, s in PLANT_BIN_SD:
        if b in skipped:
            q3["planted"].append({"bin": b, "planted_sd_m": s, "skipped": "a reader seed has fewer than 2 frames in the bin"})
            continue
        p = point_err(planted_es[b])
        ok = bool(abs(p["sd"] - s) <= Q_TOL and abs(p["mean"]) <= Q_TOL)
        q3["planted"].append({"bin": b, "planted_sd_m": s, "sd_m": p["sd"], "mean_m": p["mean"], "n": p["n"], "pass": ok})
        q3["pass"] = bool(q3["pass"] and ok)
    q3["pass"] = bool(q3["pass"] and q3["partition"])
    ctl["q3"] = q3
    return ctl


# ============================================================================================ arms and names
def measure(ctxs, B):
    rl = [c["rows"] for c in ctxs]
    bms = [bin_masks(c) for c in ctxs]
    errs = [gap_err(c) for c in ctxs]
    est, short = {}, {}
    for c in ctxs:
        for W in WS:
            for src, tag in (("reader", "RG"), ("label", "RL")):
                e, s = run_slope(c, run_index(c, W, src))
                est.setdefault("%s%d" % (tag, W), []).append(e)
                short.setdefault("%s%d" % (tag, W), []).append(s)
            est.setdefault("RP%d" % W, []).append(plain_slope(c, W))
    out = {"frames": {"scored": [int(len(c["fr"])) for c in ctxs], "measured": [int(meas(c).sum()) for c in ctxs]}, "gap_error": {}, "arms": {}, "pairs": {}}
    for b, lo, hi in BINS:
        ms = [bm[b] for bm in bms]
        ge = boot_err([e[m] for e, m in zip(errs, ms)], [c["ep"][m] for c, m in zip(ctxs, ms)], B, BOOT_SEED)
        out["gap_error"][b] = ge
        ys = np.concatenate([c["y"][m] for c, m in zip(ctxs, ms)])
        arms = {"_label_closing_sd_ms": float(ys.std()) if len(ys) > 1 else float("nan")}
        for name, e in est.items():
            masks = [m & np.isfinite(x) for m, x in zip(ms, e)]
            if min(int(m.sum()) for m in masks) < 3:
                arms[name] = None
                continue
            a = K2.summarize(e, rl, masks, B, BOOT_SEED)
            if a is not None and name in short:
                n = sum(int(m.sum()) for m in masks)
                a["short_run_share"] = float(sum(int((s & m).sum()) for s, m in zip(short[name], masks)) / n)
            arms[name] = a
        out["arms"][b] = arms
    both = [bm["ALL"] & np.isfinite(a) & np.isfinite(b) for bm, a, b in zip(bms, est["RG10"], est["RL10"])]
    out["pairs"]["RG10_minus_RL10_ALL"] = K2.paired(est["RG10"], est["RL10"], rl, both, B, BOOT_SEED) if min(int(m.sum()) for m in both) >= 3 else None
    return out


def no_data(ge):
    return bool(ge["n"] < MIN_FRAMES or ge.get("n_episodes", 0) < MIN_EPISODES or "sd_ci" not in ge)


def r_name(a, stem, suffix=""):
    if not a or a.get("r_ci") is None or not np.all(np.isfinite(a["r_ci"])):
        return stem + "-NOT-DECIDED"
    if a["r_ci"][0] >= REQ_R:
        return stem + "-REACHES" + suffix
    if a["r_ci"][1] < REQ_R:
        return stem + "-BELOW" + suffix
    return stem + "-NOT-DECIDED"


def names_from(P):
    names = []
    for b, _, _ in BINS:
        ge = P["gap_error"][b]
        stem = "GAPRANGE-GAP-ERROR-%s" % b
        if no_data(ge):
            names.append(stem + "-NO-DATA")
        elif ge["sd_ci"][1] <= SD_IN_M:
            names.append(stem + "-WITHIN-1M")
        elif ge["sd_ci"][0] > SD_OUT_M:
            names.append(stem + "-ABOVE-2M")
        else:
            names.append(stem + "-NOT-DECIDED")
    for W in WS:
        names.append(r_name(P["arms"]["ALL"].get("RG%d" % W), "GAPRANGE-READER-RUN-%d" % W, "-THE-REQUIREMENT"))
    if no_data(P["gap_error"]["NEAR"]):
        names.append("GAPRANGE-READER-RUN-10-NEAR-NO-DATA")
    else:
        names.append(r_name(P["arms"]["NEAR"].get("RG10"), "GAPRANGE-READER-RUN-10-NEAR"))
    pr = P["pairs"].get("RG10_minus_RL10_ALL")
    names.append("GAPRANGE-RUN-FINDING-COSTS" if (pr and pr["ci"][0] > 0) else "GAPRANGE-RUN-FINDING-NOT-SEPARATED")
    return names


# ============================================================================================ read and show
def run_read(a, log=print):
    ks = [int(x) for x in a.seeds.split(",")]
    cpath, rpath = os.path.join(a.out_dir, "GAPRANGE_CONTROLS.json"), os.path.join(a.out_dir, "GAPRANGE_RESULT.json")
    if os.path.exists(cpath) or os.path.exists(rpath):
        K2.refuse("an output file exists in " + a.out_dir)
    ep_map = K2.read_json(a.episode_map)
    add81_ctl = K2.read_json(a.add81_controls)
    ctxs = [make_ctx(K2.CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)), ep_map) for k in ks]
    ctl = controls(ctxs, ks, add81_ctl)
    npass = sum(1 for v in ctl.values() if v["pass"])
    allp = bool(npass == len(ctl))
    os.makedirs(a.out_dir, exist_ok=True)
    K2.write_json(cpath, {"card": CARD, "reader_seeds": ks, "controls": ctl, "n_passed": npass, "n_controls": len(ctl), "controls_all_pass": allp})
    for nm in sorted(ctl):
        log("CONTROL %s %s" % (nm, "PASS" if ctl[nm]["pass"] else "FAIL"))
    P = measure(ctxs, a.boot)
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks, "bins_label_gap_m": {b: [None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi] for b, lo, hi in BINS},
           "bootstrap": {"B": a.boot, "seed": BOOT_SEED, "levels": "reader seeds, then evaluation episodes (episode draw shared by the seeds)", "interval": "percentile 2.5 / 97.5"},
           "controls_sha256": K2.sha256_file(cpath), "controls_all_pass": allp}
    if allp or "names_despite_fail" in FAULTS:
        res.update(P)
        res["names"] = names_from(P)
    else:
        res.update({"names": [], "withheld": P})
    K2.write_json(rpath, res)
    log("ZZ GAPRANGE-CONTROLS %d/%d all_pass %s" % (npass, len(ctl), allp))
    return 0 if allp else 1


def run_show(a):
    C, R = K2.read_json(a.controls), K2.read_json(a.result)
    if not C.get("controls_all_pass") or C.get("n_passed") != C.get("n_controls"):
        K2.refuse("the controls did not all pass; no name and no value is shown")
    if R.get("controls_sha256") != K2.sha256_file(a.controls):
        K2.refuse("the result is not bound to this control file (sha256 differs)")
    if "withheld" in R or not R.get("names") or "arms" not in R:
        K2.refuse("the result carries no names or arms")
    print("GAPRANGE (card ADD91, tier T0); frames scored %s measured %s" % (R["frames"]["scored"], R["frames"]["measured"]))
    for n in R["names"]:
        print("NAME " + n)
    for b, ge in R["gap_error"].items():
        if "sd_ci" not in ge:
            print("GAPERR %-4s n %d episodes %d (no interval)" % (b, ge["n"], ge.get("n_episodes", 0)))
            continue
        print("GAPERR %-4s sd %.2f [%.2f, %.2f] mean %+.2f [%+.2f, %+.2f] mean_abs %.2f n %d episodes %d per-seed sd %s" % (
            b, ge["sd"], ge["sd_ci"][0], ge["sd_ci"][1], ge["mean"], ge["mean_ci"][0], ge["mean_ci"][1], ge["mean_abs"], ge["n"], ge["n_episodes"],
            ["%.2f" % x["sd"] for x in ge["per_seed"]]))
    for b, arms in R["arms"].items():
        print("BIN %s label closing sd %.3f" % (b, arms["_label_closing_sd_ms"]))
        for nm, v in arms.items():
            if nm.startswith("_"):
                continue
            if not v:
                print("  %-5s (no frames)" % nm)
                continue
            print("  %-5s r %+.3f [%+.3f, %+.3f] mae %.3f err_sd %.3f n %d%s" % (nm, v["r_pooled"], v["r_ci"][0], v["r_ci"][1], v["mae"], v["err_sd"], v["n_frames"],
                                                                              (" short %.3f" % v["short_run_share"]) if "short_run_share" in v else ""))
    pr = R["pairs"].get("RG10_minus_RL10_ALL")
    if pr:
        print("PAIR RG10-RL10 ALL abs err %+.3f [%+.3f, %+.3f] n %d" % (pr["diff_abs_err"], pr["ci"][0], pr["ci"][1], pr["n_frames"]))
    return 0


# ============================================================================================ self-test (control q4)
def selftest():
    import tempfile
    tmp = os.path.join(HERE, "tmp_work")
    os.makedirs(tmp, exist_ok=True)
    bad, lines = [], []

    def chk(name, cond):
        lines.append("%s %s" % ("ok  " if cond else "FAIL", name))
        if not cond:
            bad.append(name)

    if K2.CF is None:
        K2.load_tools(None)
    with tempfile.TemporaryDirectory(dir=tmp) as root:
        dumps, epm = K3.synth()
        K2.write_json(os.path.join(root, "epm.json"), epm)
        n = [len(K2.CF.build_rows(dumps[k], 25, epm)["y"]) for k in (0, 1)]
        K2.write_json(os.path.join(root, "c81.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k]} for k in (0, 1)}}}})
        K2.write_json(os.path.join(root, "c81b.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k] + (k == 1)} for k in (0, 1)}}}})
        off = {k: dict(D, gap=np.where(np.asarray(D["lab_present"], bool), np.asarray(D["lab_gap"], np.float64) + 3.0, np.asarray(D["gap"], np.float64)),
                       valid=np.asarray(D["valid"], bool) & np.asarray(D["lab_present"], bool)) for k, D in dumps.items()}
        noff = [len(K2.CF.build_rows(off[k], 25, epm)["y"]) for k in (0, 1)]
        K2.write_json(os.path.join(root, "c81off.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": noff[k]} for k in (0, 1)}}}})
        for tag, W in (("dump", dumps), ("dumpoff", off)):
            os.makedirs(os.path.join(root, tag))
            for k, D in W.items():
                np.savez(os.path.join(root, tag, "reader_s%d_score.npz" % k), **D)
        cnt = [0]

        def go(faults=(), c81="c81.json", dd="dump"):
            cnt[0] += 1
            od = os.path.join(root, "o%d" % cnt[0])
            a = argparse.Namespace(dump_dir=os.path.join(root, dd), episode_map=os.path.join(root, "epm.json"), add81_controls=os.path.join(root, c81), out_dir=od, boot=200, seeds="0,1")
            FAULTS.clear(), FAULTS.update(faults)
            try:
                rc = run_read(a, log=lambda s: None)
            finally:
                FAULTS.clear()
            return rc, K2.read_json(os.path.join(od, "GAPRANGE_CONTROLS.json")), K2.read_json(os.path.join(od, "GAPRANGE_RESULT.json")), od

        rc, C, R, od = go()
        chk("clean world: rc 0, four controls pass, names and arms present, nothing withheld", rc == 0 and C["n_passed"] == 4 and bool(R["names"]) and "arms" in R and "withheld" not in R)
        chk("q1 on a world with planted steps: the run rule on the reader path equals the port of the rule's source (largest difference %.1e)" % C["controls"]["q1"]["max_abs_slope_diff_ms"],
            C["controls"]["q1"]["pass"])
        c0 = make_ctx(dumps[0], epm)
        lgr = c0["LG"][c0["ci"], c0["fr"]][meas(c0)]
        chk("the planted world has measured rows between 20 and 25 m (needed by the bin-edge regression)", int(((lgr >= 20) & (lgr < 25)).sum()) > 0)
        chk("the reader's gap of the world (label plus white noise of 1 m) has an error standard deviation near 1 m on ALL", 0.8 < R["gap_error"]["ALL"]["sd"] < 1.2)
        chk("RG_10 and RL_10 read the label closing speed with a positive r on ALL", R["arms"]["ALL"]["RG10"]["r_pooled"] > 0 and R["arms"]["ALL"]["RL10"]["r_pooled"] > 0)
        chk("names: four GAP-ERROR names, three READER-RUN names on ALL, one NEAR name, one RUN-FINDING name",
            sum(1 for x in R["names"] if "GAP-ERROR-" in x) == 4 and sum(1 for x in R["names"] if x.startswith("GAPRANGE-READER-RUN-") and "-NEAR-" not in x) == 3
            and sum(1 for x in R["names"] if "-NEAR-" in x and "READER-RUN" in x) == 1 and sum(1 for x in R["names"] if "RUN-FINDING" in x) == 1 and len(R["names"]) == 9)
        rco, Co, Ro, _ = go(dd="dumpoff", c81="c81off.json")
        chk("a world whose reader gap is the label gap plus 3.0 m: controls pass (failed: %s)" % [k for k, v in Co["controls"].items() if not v["pass"]], rco == 0 and "arms" in Ro)
        if "arms" in Ro:
            chk("in that world the gap error has mean +3.0 and standard deviation 0", abs(Ro["gap_error"]["ALL"]["mean"] - 3.0) < 1e-9 and Ro["gap_error"]["ALL"]["sd"] < 1e-9)
            chk("in that world RG_5 reads the label closing speed at r above 0.99", Ro["arms"]["ALL"]["RG5"]["r_pooled"] > 0.99)
        for fault, ctl in (("sign_flip", "q2"), ("no_normalise", "q2"), ("bin_edge", "q3"), ("no_step_gate", "q1")):
            rcf, Cf, Rf, _ = go([fault])
            chk("regression %s turns %s red, the read exits 1, no name, values withheld" % (fault, ctl),
                (not Cf["controls"][ctl]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf and "arms" not in Rf and "gap_error" not in Rf)
        rcf, Cf, Rf, _ = go(c81="c81b.json")
        chk("regression: a frame count that differs from card ADD81 turns q0 red", (not Cf["controls"]["q0"]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf)
        rcd, Cd, Rd, odd = go(["sign_flip", "names_despite_fail"])
        chk("regression: names written although a control failed is caught (the healthy tool writes none)", bool(Rd["names"]) and not Cd["controls_all_pass"])
        try:
            run_show(argparse.Namespace(controls=os.path.join(odd, "GAPRANGE_CONTROLS.json"), result=os.path.join(odd, "GAPRANGE_RESULT.json")))
            shown = True
        except SystemExit:
            shown = False
        chk("show refuses a result whose controls did not all pass", not shown)
        ge = lambda sd_ci, n=500, ne=20: {"n": n, "n_episodes": ne, "sd_ci": sd_ci}  # noqa: E731
        fake = {"gap_error": {"ALL": ge([7.0, 9.0]), "NEAR": ge([0.5, 0.9]), "MID": ge([1.5, 2.5]), "FAR": ge([7.0, 9.0], n=50)},
                "arms": {"ALL": {"RG5": {"r_ci": [0.1, 0.3]}, "RG10": {"r_ci": [0.91, 0.99]}, "RG25": {"r_ci": [0.5, 0.95]}}, "NEAR": {"RG10": {"r_ci": [0.2, 0.6]}}},
                "pairs": {"RG10_minus_RL10_ALL": {"ci": [0.1, 0.4]}}}
        nf = names_from(fake)
        chk("name rule on planted intervals",
            nf == ["GAPRANGE-GAP-ERROR-ALL-ABOVE-2M", "GAPRANGE-GAP-ERROR-NEAR-WITHIN-1M", "GAPRANGE-GAP-ERROR-MID-NOT-DECIDED", "GAPRANGE-GAP-ERROR-FAR-NO-DATA",
                   "GAPRANGE-READER-RUN-5-BELOW-THE-REQUIREMENT", "GAPRANGE-READER-RUN-10-REACHES-THE-REQUIREMENT", "GAPRANGE-READER-RUN-25-NOT-DECIDED",
                   "GAPRANGE-READER-RUN-10-NEAR-BELOW", "GAPRANGE-RUN-FINDING-COSTS"])
        fake["gap_error"]["NEAR"] = ge([0.5, 0.9], ne=3)
        fake["pairs"]["RG10_minus_RL10_ALL"] = {"ci": [-0.1, 0.4]}
        nf2 = names_from(fake)
        chk("name rule: a bin with fewer than 5 episodes has no data, also for the NEAR run name; a pair that spans 0 is not separated",
            "GAPRANGE-GAP-ERROR-NEAR-NO-DATA" in nf2 and "GAPRANGE-READER-RUN-10-NEAR-NO-DATA" in nf2 and "GAPRANGE-RUN-FINDING-NOT-SEPARATED" in nf2)
        rng = np.random.default_rng(5)
        e1, e2 = rng.normal(1.0, 2.0, 4000), rng.normal(1.0, 2.0, 4000)
        ep = np.asarray(["e%02d" % (i % 40) for i in range(4000)])
        bs = boot_err([e1, e2], [ep, ep], 300, 1)
        chk("bootstrap of the error statistics: the interval of the standard deviation holds the planted 2.0 and the point value",
            bs["sd_ci"][0] < 2.0 < bs["sd_ci"][1] and bs["sd_ci"][0] <= bs["sd"] <= bs["sd_ci"][1] and bs["n_episodes"] == 40)
    for ln in lines:
        print(ln)
    print("ZZ GAPRANGE-SELFTEST rc %d" % (1 if bad else 0))
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--closeread-tools", default=None)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("read")
    for x in ("--dump-dir", "--episode-map", "--add81-controls", "--out-dir"):
        r.add_argument(x, required=True)
    r.add_argument("--closeread-tools", dest="crt", default=None)
    r.add_argument("--boot", type=int, default=BOOT_B)
    r.add_argument("--seeds", default="0,1")
    s = sub.add_parser("show")
    s.add_argument("--result", required=True)
    s.add_argument("--controls", required=True)
    a = p.parse_args(argv)
    if a.selftest:
        K2.load_tools(a.closeread_tools)
        return selftest()
    if a.cmd == "read":
        K2.load_tools(a.crt or a.closeread_tools)
        return run_read(a)
    if a.cmd == "show":
        return run_show(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
