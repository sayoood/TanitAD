#!/usr/bin/env python
"""kinclose3.py -- tool of card PREREG_V8F_PROOF_ADD90 (KINCLOSE3), tier T0: what a gap history can carry. Label gap only (no reader output, no ego speed, no fit).

Arms: BG_W = least-squares slope of the label gap over the run with cap W (the run rule of v8f_t1_eval_rule.py lines 92-112 with W in place of 5); planted arms = the same slope on
(label gap + noise), noise white / smooth (AR 0.90) / reader-like (AR 0.94), planted with mean 0 and standard deviation exactly sigma over the frames with a label gap.
Subcommands: read (controls g0..g3, then arms and names; names only when every control passed, else the values go under the key withheld), show, --selftest (control g4).
Uses kinclose2.py beside it for the bootstrap summary and closeread_fit.py for the scored rows. Numpy only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kinclose2 as K2  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD90"
WS = (5, 10, 25)
NAME_WS = (10, 25)
SIGMAS = ((0.25, "S025"), (0.5, "S050"), (1.0, "S100"), (2.0, "S200"), (4.0, "S400"), (8.0, "S800"))
KINDS = (("W", "WHITE", 0.0), ("S", "SMOOTH", 0.90), ("R", "READERLIKE", 0.94))
LAG_BANDS = {"W": (-0.05, 0.05), "S": (0.85, 0.95), "R": (0.89, 0.97)}
RUN_MIN, STEP_MAX_M = 3, 3.0
REQ_R, BOOT_B, BOOT_SEED, PLANT_SEED = 0.9, 2000, 90, 90
G0_TOL, G1_TOL, G2_TOL = 1e-6, 1e-9, 1e-9
ROW147_SD = 1.24
FAULTS = set()  # in-process regression switches of the self-test; the command line never sets them


def make_ctx(D, ep_map):
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    nc, T = len(D["clips"]), int(fr.max()) + 1
    lg = np.asarray(D["lab_gap"], np.float64)
    lp = np.asarray(D["lab_present"], bool) & np.isfinite(lg)
    LG, LM = np.full((nc, T), np.nan), np.zeros((nc, T), bool)
    LG[cid[lp], fr[lp]] = lg[lp]
    LM[cid[lp], fr[lp]] = True
    rows = K2.CF.build_rows(D, 25, ep_map)
    name2i = {str(c): i for i, c in enumerate(D["clips"])}
    ci = np.asarray([name2i[str(c)] for c in rows["clip"]], np.int64)
    return {"clips": [str(c) for c in D["clips"]], "LG": LG, "LM": LM, "rows": rows, "ci": ci, "fr": np.asarray(rows["frame"], np.int64), "dt": float(D["dt_s"]), "T": T}


def run_index(ctx, W):
    """per scored row: the frames of the run with cap W, newest first, from the label gap (missing label or a step of more than 3.0 m ends the run)."""
    LG, LM = ctx["LG"], ctx["LM"]
    out = []
    for i, j in zip(ctx["ci"], ctx["fr"]):
        if not LM[i, j]:
            out.append([])
            continue
        idx, f = [int(j)], int(j) - 1
        while f >= 0 and len(idx) < W and LM[i, f] and (("no_step_gate" in FAULTS) or abs(LG[i, f + 1] - LG[i, f]) <= STEP_MAX_M):
            idx.append(f)
            f -= 1
        out.append(idx)
    return out


def run_slope(ctx, runs, X):
    """least-squares slope against time of X over each run (>= 3 frames), else NaN."""
    dt = ctx["dt"] * (0.5 if "dt_wrong" in FAULTS else 1.0)
    out = np.full(len(runs), np.nan)
    for r, (i, idx) in enumerate(zip(ctx["ci"], runs)):
        if len(idx) < RUN_MIN:
            continue
        g = X[i, idx[::-1]]
        t = np.arange(len(idx), dtype=np.float64) * dt
        tb = t.mean()
        out[r] = float(((t - tb) * (g - g.mean())).sum() / ((t - tb) ** 2).sum())
    return out


def unit_noise(ctx, kind):
    """[clips, T] noise of one kind: one draw per clip id, then shifted and scaled so that over all frames with a label gap the mean is 0 and the standard deviation 1."""
    coef = {k: c for k, _, c in KINDS}[kind]
    T = ctx["T"]
    out = np.zeros((len(ctx["clips"]), T))
    for i, c in enumerate(ctx["clips"]):
        h = int(hashlib.sha256(("%s|%s|%d" % (c, kind, PLANT_SEED)).encode("utf-8")).hexdigest()[:8], 16)
        z = np.random.default_rng(h).standard_normal(T)
        if coef > 0:
            e = np.zeros(T)
            e[0] = z[0]
            for t in range(1, T):
                e[t] = coef * e[t - 1] + np.sqrt(1.0 - coef ** 2) * z[t]
            z = e
        out[i] = z
    if "no_normalise" not in FAULTS:
        m = ctx["LM"]
        out = (out - out[m].mean()) / out[m].std()
    return out


def lag1(N, LM):
    pr = LM[:, 1:] & LM[:, :-1]
    return float(K2.CF.pearson(N[:, :-1][pr], N[:, 1:][pr]))


def build(ctxs):
    """arms of every reader seed: {name: [est per seed]}, plus the control records g0..g2 (g3 is added by the caller)."""
    est = {}
    g0 = {"max_abs_diff_ms": 0.0, "n_frames": 0, "n_without_value": 0, "tol": G0_TOL}
    g1 = {"max_abs_diff": 0.0, "tol": G1_TOL}
    for c in ctxs:
        runs = {W: run_index(c, W) for W in WS}
        for W in WS:
            est.setdefault("BG%d" % W, []).append(run_slope(c, runs[W], c["LG"]))
        b5, y = est["BG5"][-1], np.asarray(c["rows"]["y"], np.float64)
        f = np.isfinite(b5)
        g0["n_frames"] += int(len(y))
        g0["n_without_value"] += int((~f).sum())
        if f.any():
            g0["max_abs_diff_ms"] = max(g0["max_abs_diff_ms"], float(np.abs(b5[f] - y[f]).max()))
        for kind, _, _ in KINDS:
            N = unit_noise(c, kind)
            for W in WS:
                z0 = run_slope(c, runs[W], c["LG"] + 0.0 * N)
                bg = est["BG%d" % W][-1]
                same = np.isfinite(z0) == np.isfinite(bg)
                d = float(np.abs(z0[np.isfinite(z0) & np.isfinite(bg)] - bg[np.isfinite(z0) & np.isfinite(bg)]).max()) if (np.isfinite(z0) & np.isfinite(bg)).any() else 0.0
                g1["max_abs_diff"] = max(g1["max_abs_diff"], d if bool(same.all()) else float("inf"))
                for sg, code in SIGMAS:
                    est.setdefault("P%s%d_%s" % (kind, W, code), []).append(run_slope(c, runs[W], c["LG"] + sg * N))
    g0["pass"] = bool(g0["n_frames"] > 0 and g0["n_without_value"] == 0 and g0["max_abs_diff_ms"] <= G0_TOL)
    g1["pass"] = bool(g1["max_abs_diff"] <= G1_TOL)
    g2 = {"kinds": {}, "pass": True}
    c = ctxs[0]
    for kind, _, _ in KINDS:
        N = unit_noise(c, kind)
        m, s, l1 = float(N[c["LM"]].mean()), float(N[c["LM"]].std()), lag1(N, c["LM"])
        lo, hi = LAG_BANDS[kind]
        ok = bool(abs(m) <= G2_TOL and abs(s - 1.0) <= G2_TOL and lo <= l1 <= hi)
        g2["kinds"][kind] = {"mean": m, "sd": s, "lag1": l1, "band": [lo, hi], "pass": ok}
        g2["pass"] = bool(g2["pass"] and ok)
    return est, {"g0": g0, "g1": g1, "g2": g2}


def names_from(arms):
    names, reach = [], [5]
    for W in NAME_WS:
        b = arms.get("BG%d" % W)
        if b and b["r_ci"][0] >= REQ_R:
            names.append("KINCLOSE3-RUN-%d-REACHES-THE-REQUIREMENT" % W)
            reach.append(W)
        elif b and b["r_ci"][1] < REQ_R:
            names.append("KINCLOSE3-RUN-%d-BELOW-THE-REQUIREMENT" % W)
        else:
            names.append("KINCLOSE3-RUN-%d-NOT-DECIDED" % W)
    reader_codes = []
    for W in reach:
        for kind, kname, _ in KINDS:
            best = None
            for sg, code in SIGMAS:
                a = arms.get("P%s%d_%s" % (kind, W, code))
                if a and a["r_ci"][0] >= REQ_R:
                    best = code
                else:
                    break
            names.append(("KINCLOSE3-GAP-ERROR-W%d-%s-UP-TO-%s" % (W, kname, best)) if best else ("KINCLOSE3-GAP-ERROR-W%d-%s-BELOW-S025" % (W, kname)))
            if kind == "R":
                reader_codes.append(best)
    names.append("KINCLOSE3-READER-GAP-ERROR-ABOVE-EVERY-TOLERATED-LEVEL" if all(c != "S800" for c in reader_codes) else "KINCLOSE3-READER-GAP-ERROR-WITHIN-A-TOLERATED-LEVEL")
    return names


def run_read(a, log=print):
    ks = [int(x) for x in a.seeds.split(",")]
    cpath, rpath = os.path.join(a.out_dir, "KINCLOSE3_CONTROLS.json"), os.path.join(a.out_dir, "KINCLOSE3_RESULT.json")
    if os.path.exists(cpath) or os.path.exists(rpath):
        K2.refuse("an output file exists in " + a.out_dir)
    ep_map = K2.read_json(a.episode_map)
    add81_ctl = K2.read_json(a.add81_controls)
    dumps = {k: K2.CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)) for k in ks}
    ctxs = [make_ctx(dumps[k], ep_map) for k in ks]
    est, ctl = build(ctxs)
    want = [int(add81_ctl["controls"]["t0"]["frames"]["H25_seed%d" % k]["n_tool"]) for k in ks]
    got = [int(len(c["fr"])) for c in ctxs]
    ctl["g3"] = {"n_frames": got, "n_add81": want, "pass": bool(got == want)}
    npass = sum(1 for v in ctl.values() if v["pass"])
    allp = bool(npass == len(ctl))
    os.makedirs(a.out_dir, exist_ok=True)
    K2.write_json(cpath, {"card": CARD, "reader_seeds": ks, "controls": ctl, "n_passed": npass, "n_controls": len(ctl), "controls_all_pass": allp})
    for nm in sorted(ctl):
        log("CONTROL %s %s" % (nm, "PASS" if ctl[nm]["pass"] else "FAIL"))
    rl = [c["rows"] for c in ctxs]
    A = {n: K2.summarize(e, rl, [K2.fin(x) for x in e], a.boot, BOOT_SEED) for n, e in est.items()}
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks, "bootstrap": {"B": a.boot, "seed": BOOT_SEED, "levels": "reader seeds, then evaluation episodes (episode draw shared by the seeds)",
           "interval": "percentile 2.5 / 97.5"}, "controls_sha256": K2.sha256_file(cpath), "controls_all_pass": allp, "row147_error_sd_ms": ROW147_SD}
    if allp or "names_despite_fail" in FAULTS:
        res.update({"arms": A, "names": names_from(A)})
    else:
        res.update({"names": [], "withheld": {"arms": A}})
    K2.write_json(rpath, res)
    log("ZZ KINCLOSE3-CONTROLS %d/%d all_pass %s" % (npass, len(ctl), allp))
    return 0 if allp else 1


def run_show(a):
    C, R = K2.read_json(a.controls), K2.read_json(a.result)
    if not C.get("controls_all_pass") or C.get("n_passed") != C.get("n_controls"):
        K2.refuse("the controls did not all pass; no name and no value is shown")
    if R.get("controls_sha256") != K2.sha256_file(a.controls):
        K2.refuse("the result is not bound to this control file (sha256 differs)")
    if "withheld" in R or not R.get("names") or "arms" not in R:
        K2.refuse("the result carries no names or arms")
    print("KINCLOSE3 (card ADD90, tier T0)")
    for n in R["names"]:
        print("NAME " + n)
    for nm, v in R["arms"].items():
        if not v:
            print("%-10s (no frames)" % nm)
            continue
        print("%-10s r %+.3f [%+.3f, %+.3f] mae %.3f err_sd %.3f n %d" % (nm, v["r_pooled"], v["r_ci"][0], v["r_ci"][1], v["mae"], v["err_sd"], v["n_frames"]))
    return 0


# ============================================================================================ self-test (control g4)
def port_label_closing(gap, ok, j, dt):
    """line-by-line port of LabelSource.read (v8f_t1_eval_rule.py lines 92-112): -> closing speed of frame j or None when the frame has no label / a run below 3 frames gives 0.0."""
    if j < 0 or j >= len(gap) or not ok[j]:
        return None
    idx, f = [j], j - 1
    while f >= 0 and len(idx) < 5 and ok[f] and abs(gap[f + 1] - gap[f]) <= 3.0:
        idx.append(f)
        f -= 1
    if len(idx) < 3:
        return 0.0
    idx = idx[::-1]
    G = gap[idx]
    t = np.arange(len(idx), dtype=np.float64) * dt
    tb = t.mean()
    return float(((t - tb) * (G - G.mean())).sum() / ((t - tb) ** 2).sum())


def synth(seed=4, n_clips=30, T=101, dt=0.2):
    """dumps of two reader seeds with a label gap that has planted steps (changes of the lead) and holes; the label closing speed comes from the port of the rule's source."""
    rng = np.random.default_rng(seed)
    clips = ["synthclip-%04d" % i for i in range(n_clips)]
    cols = {n: [] for n in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
    for i in range(n_clips):
        v = np.cumsum(rng.normal(0, 0.4, T)) * 0.5
        g = 30.0 + np.cumsum(v * dt)
        for s in rng.integers(10, T - 5, 2):
            g[s:] += rng.choice([-9.0, 7.0, 12.0])
        g = np.clip(g, 2.0, 59.0)
        ok = np.ones(T, bool)
        if i % 4 == 0:
            ok[40:47] = False
        clo, run = np.zeros(T), np.zeros(T, np.int64)
        for j in range(T):
            c = port_label_closing(g, ok, j, dt)
            if c is None:
                continue
            idx, f = 1, j - 1
            while f >= 0 and idx < 5 and ok[f] and abs(g[f + 1] - g[f]) <= 3.0:
                idx, f = idx + 1, f - 1
            run[j], clo[j] = idx, c
        cols["cid"].append(np.full(T, i)); cols["clip_index"].append(np.full(T, 1000 + i)); cols["frame"].append(np.arange(T))
        cols["p"].append(np.full(T, 0.9)); cols["gap"].append(g + rng.normal(0, 1.0, T)); cols["clos"].append(rng.normal(0, 1.0, T))
        cols["valid"].append(np.arange(T) >= 4); cols["lab_present"].append(ok); cols["lab_gap"].append(np.where(ok, g, np.nan))
        cols["lab_closing"].append(clo); cols["lab_run"].append(run)
    D = {n: np.concatenate(v) for n, v in cols.items()}
    D["clips"], D["dt_s"], D["seed"] = np.asarray(clips), np.float64(dt), np.int64(0)
    return {0: D, 1: dict(D, seed=np.int64(1))}, {c: "ep%03d" % (i // 2) for i, c in enumerate(clips)}


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
        dumps, epm = synth()
        dd = os.path.join(root, "dump")
        os.makedirs(dd)
        for k, D in dumps.items():
            np.savez(os.path.join(dd, "reader_s%d_score.npz" % k), **D)
        K2.write_json(os.path.join(root, "epm.json"), epm)
        n = [len(K2.CF.build_rows(dumps[k], 25, epm)["y"]) for k in (0, 1)]
        K2.write_json(os.path.join(root, "c81.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k]} for k in (0, 1)}}}})
        cnt = [0]

        def go(faults=(), c81=None):
            cnt[0] += 1
            od = os.path.join(root, "o%d" % cnt[0])
            a = argparse.Namespace(dump_dir=dd, episode_map=os.path.join(root, "epm.json"), add81_controls=c81 or os.path.join(root, "c81.json"), out_dir=od, boot=200, seeds="0,1")
            FAULTS.clear(), FAULTS.update(faults)
            try:
                rc = run_read(a, log=lambda s: None)
            finally:
                FAULTS.clear()
            return rc, K2.read_json(os.path.join(od, "KINCLOSE3_CONTROLS.json")), K2.read_json(os.path.join(od, "KINCLOSE3_RESULT.json")), od

        rc, C, R, od = go()
        chk("clean world: rc 0, four controls pass, names and arms present, nothing withheld", rc == 0 and C["n_passed"] == 4 and bool(R["names"]) and "arms" in R and "withheld" not in R)
        chk("g0 on a world with planted steps: the run rule equals the port of the rule's source (largest difference %.1e)" % C["controls"]["g0"]["max_abs_diff_ms"], C["controls"]["g0"]["pass"])
        chk("BG_5 reads the label at r 1", abs(R["arms"]["BG5"]["r_pooled"] - 1.0) < 1e-9)
        chk("a longer run reads the label less well than the run of five (lag)", R["arms"]["BG25"]["r_pooled"] < R["arms"]["BG5"]["r_pooled"])
        chk("planted ladder: white noise of 8 m over five frames reads worse than 0.25 m", R["arms"]["PW5_S800"]["r_pooled"] < R["arms"]["PW5_S025"]["r_pooled"])
        chk("names: one RUN name per window, GAP-ERROR names for W 5 in three kinds, one reader-gap name",
            sum(1 for x in R["names"] if "-RUN-" in x) == 2 and sum(1 for x in R["names"] if "GAP-ERROR-W5-" in x) == 3 and sum(1 for x in R["names"] if "READER-GAP-ERROR" in x) == 1)
        for fault, ctl in (("no_step_gate", "g0"), ("dt_wrong", "g0"), ("no_normalise", "g2")):
            rcf, Cf, Rf, _ = go([fault])
            chk("regression %s turns %s red, the read exits 1, no name, values withheld" % (fault, ctl), (not Cf["controls"][ctl]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf and "arms" not in Rf)
        K2.write_json(os.path.join(root, "c81b.json"), {"controls": {"t0": {"frames": {"H25_seed%d" % k: {"n_tool": n[k] + (k == 1)} for k in (0, 1)}}}})
        rcf, Cf, Rf, _ = go(c81=os.path.join(root, "c81b.json"))
        chk("regression: a frame count that differs from card ADD81 turns g3 red", (not Cf["controls"]["g3"]["pass"]) and rcf == 1 and Rf["names"] == [])
        rcd, Cd, Rd, odd = go(["dt_wrong", "names_despite_fail"])
        chk("regression: names written although a control failed is caught (the healthy tool writes none)", bool(Rd["names"]) and not Cd["controls_all_pass"])
        try:
            run_show(argparse.Namespace(controls=os.path.join(odd, "KINCLOSE3_CONTROLS.json"), result=os.path.join(odd, "KINCLOSE3_RESULT.json")))
            shown = True
        except SystemExit:
            shown = False
        chk("show refuses a result whose controls did not all pass", not shown)
        names_fake = names_from({"BG10": {"r_ci": [0.95, 0.99]}, "BG25": {"r_ci": [0.5, 0.8]}, **{"P%s%d_%s" % (k, w, c): {"r_ci": [0.95 if s <= 1.0 else 0.5, 1.0]} for k, _, _ in KINDS for w in WS for s, c in SIGMAS}})
        chk("name rule on planted intervals: RUN-10 reaches, RUN-25 below, codes stop at S100, reader gap above every level",
            "KINCLOSE3-RUN-10-REACHES-THE-REQUIREMENT" in names_fake and "KINCLOSE3-RUN-25-BELOW-THE-REQUIREMENT" in names_fake and "KINCLOSE3-GAP-ERROR-W10-READERLIKE-UP-TO-S100" in names_fake
            and not any("W25" in x for x in names_fake) and "KINCLOSE3-READER-GAP-ERROR-ABOVE-EVERY-TOLERATED-LEVEL" in names_fake)
        names_fake2 = names_from({"BG10": {"r_ci": [0.5, 0.95]}, "BG25": {"r_ci": [0.5, 0.8]}, **{"P%s%d_%s" % (k, w, c): {"r_ci": [0.95, 1.0]} for k, _, _ in KINDS for w in WS for s, c in SIGMAS}})
        chk("name rule: RUN-10 not decided, all levels tolerated gives the within name", "KINCLOSE3-RUN-10-NOT-DECIDED" in names_fake2 and "KINCLOSE3-READER-GAP-ERROR-WITHIN-A-TOLERATED-LEVEL" in names_fake2)
    for ln in lines:
        print(ln)
    print("ZZ KINCLOSE3-SELFTEST rc %d" % (1 if bad else 0))
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
