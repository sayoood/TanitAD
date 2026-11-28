#!/usr/bin/env python
"""kinclose6.py -- tool of card PREREG_V8F_PROOF_ADD93 (GAPSPLIT), tier T0 (a diagnostic on logged frames; no rollout).

The reader's gap error against the label gap, split by clip: CONST (standard deviation of the clip means), FAST (root mean square about the clip
mean), STEP (root mean square of the change between neighbouring frames), absolute and proportional, read against the same statistics of the planted
bar levels of card ADD92. Numpy only. Controls u0 to u3 are computed by the read, u4 is the self-test. With one control failed no name is written
and every value is held under the key "withheld".

  python kinclose6.py --selftest [--closeread-tools DIR]
  python kinclose6.py read --dump-dir D --episode-map E --add81-controls C --add92-result R92 --out-dir O [--closeread-tools DIR]
  python kinclose6.py show --result R --controls C
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
import kinclose4 as K4  # noqa: E402
import kinclose5 as K5  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD93"
MIN_CLIP_FRAMES = 5
NEAR_M = 20.0
BOOT_B, BOOT_SEED = 2000, 93
TOL = 1e-9
BARS = {"fast_abs": ("R", "ABS", 1.0, "S100"), "fast_prop": ("R", "PROP", 0.025, "P025"), "step_abs": ("W", "ABS", 0.5, "S050"), "const_prop": ("C", "PROP", 0.20, "P200")}
U3_ARMS = (("R", "ABS", 1.0, "S100", 10), ("C", "PROP", 0.20, "P200", 10))
U2 = {"const_abs": 3.0, "fast_abs": 1.0, "const_prop": 0.2, "fast_prop": 0.05}
STATS = ("const_abs", "fast_abs", "step_abs", "const_prop", "fast_prop", "step_prop", "near_fast_abs", "near_step_abs")
NAME_RULES = (("fast_abs", "GAPSPLIT-FAST-ABS", "THE-BAR"), ("fast_prop", "GAPSPLIT-FAST-PROP", "THE-BAR"), ("step_abs", "GAPSPLIT-STEP-ABS", "THE-BAR"),
              ("const_prop", "GAPSPLIT-CONSTANT-PROP", "THE-CARRIED-LEVEL"))
FAULTS = set()  # in-process regression switches of the self-test; the command line never sets them


def frame_set(c):
    """[clips, T] frames with a valid reader gap and a label gap, in clips that have at least 5 of them."""
    m = c["RM"] & c["LM"]
    return m & (m.sum(1) >= MIN_CLIP_FRAMES)[:, None]


def clip_sums(c, G, M):
    """per clip a vector of sums (see stat) of the error of gap G against the label gap on the frames M; -> [clips, 15], residual lag-1 pairs."""
    LG = c["LG"]
    out = np.zeros((M.shape[0], 15))
    ra, rb = [], []
    gm_e = gm_q = 0.0
    if "fast_global" in FAULTS:
        gm_e = float((G[M] - LG[M]).mean()) if M.any() else 0.0
        gm_q = float((G[M] / LG[M] - 1.0).mean()) if M.any() else 0.0
    for i in range(M.shape[0]):
        idx = np.flatnonzero(M[i])
        n = len(idx)
        if n == 0:
            continue
        e = G[i, idx] - LG[i, idx]
        q = e if "prop_as_abs" in FAULTS else (G[i, idx] / LG[i, idx] - 1.0)
        me, mq = float(e.mean()), float(q.mean())
        r = e - (gm_e if "fast_global" in FAULTS else me)
        rq = q - (gm_q if "fast_global" in FAULTS else mq)
        adj = np.flatnonzero(np.diff(idx) == 1)
        de, dq = e[adj + 1] - e[adj], q[adj + 1] - q[adj]
        near = LG[i, idx] < NEAR_M
        nadj = near[adj + 1]
        out[i] = [n, n * me, n * me * me, float((r * r).sum()), len(adj), float((de * de).sum()), n * mq, n * mq * mq, float((rq * rq).sum()), len(adj), float((dq * dq).sum()),
                  int(near.sum()), float((r[near] ** 2).sum()), int(nadj.sum()), float((de[nadj] ** 2).sum())]
        ra.append(r[adj])
        rb.append(r[adj + 1])
    ra = np.concatenate(ra) if ra else np.zeros(0)
    rb = np.concatenate(rb) if rb else np.zeros(0)
    return out, (ra, rb)


def stat(t):
    """statistics from a total of the clip sums."""
    def rms(s2, n):
        return float(np.sqrt(s2 / n)) if n > 0 else float("nan")

    def sd(n, s1, s2):
        return float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0))) if n > 0 else float("nan")
    return {"const_abs": sd(t[0], t[1], t[2]), "fast_abs": rms(t[3], t[0]), "step_abs": rms(t[5], t[4]), "const_prop": sd(t[0], t[6], t[7]), "fast_prop": rms(t[8], t[0]),
            "step_prop": rms(t[10], t[9]), "near_fast_abs": rms(t[12], t[11]), "near_step_abs": rms(t[14], t[13])}


def pooled(ctxs, gaps, Ms):
    return stat(sum(clip_sums(c, G, M)[0].sum(0) for c, G, M in zip(ctxs, gaps, Ms)))


def clip_episodes(c, ep_map):
    return [str(ep_map.get(n, n)) if ep_map else n for n in c["clips"]]


def boot(ctxs, gaps, Ms, ep_map, B, seed):
    per = [clip_sums(c, G, M)[0] for c, G, M in zip(ctxs, gaps, Ms)]
    eps = [clip_episodes(c, ep_map) for c in ctxs]
    alle = sorted(set(e for ep, s in zip(eps, per) for e, row in zip(ep, s) if row[0] > 0))
    emap = {e: i for i, e in enumerate(alle)}
    S = []
    for ep, s in zip(eps, per):
        m = np.zeros((len(alle), s.shape[1]))
        for e, row in zip(ep, s):
            if row[0] > 0:
                m[emap[e]] += row
        S.append(m)
    point = stat(sum(m.sum(0) for m in S))
    rng = np.random.default_rng(seed)
    d = {k: [] for k in STATS}
    for _ in range(B):
        ks = rng.integers(0, len(S), len(S))
        w = np.bincount(rng.integers(0, len(alle), len(alle)), minlength=len(alle)).astype(np.float64)
        st = stat(sum(w @ S[k] for k in ks))
        for k in STATS:
            d[k].append(st[k])
    out = {}
    for k in STATS:
        v = np.asarray(d[k], np.float64)
        v = v[np.isfinite(v)]
        out[k] = {"value": point[k], "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None,
                  "per_seed": [stat(m.sum(0))[k] for m in S]}
    return out, {"n_frames": [int(m[:, 0].sum()) for m in S], "n_step_pairs": [int(m[:, 4].sum()) for m in S], "n_near_frames": [int(m[:, 11].sum()) for m in S],
                 "n_episodes": int(len(alle)), "n_clips": [int((p[:, 0] > 0).sum()) for p in per]}


# ============================================================================================ controls u0 to u3
def planted_split(c, M, prop):
    """reader's gap replaced on M by the label gap plus an exactly planted constant-per-clip part and fast part (control u2)."""
    n = M.sum(1).astype(np.float64)
    cc = K5.unit(c, "C")[:, 0].copy()
    N = float(n.sum())
    cc = cc - float((n * cc).sum() / N)
    cc = cc / float(np.sqrt((n * cc * cc).sum() / N))
    u = np.where(M, K3.unit_noise(c, "R"), 0.0)
    u = np.where(M, u - (u.sum(1) / np.maximum(n, 1.0))[:, None], 0.0)
    u = u / float(np.sqrt((u * u).sum() / N))
    G = c["G"].copy()
    if prop:
        G[M] = (c["LG"] * (1.0 + U2["const_prop"] * cc[:, None] + U2["fast_prop"] * u))[M]
    else:
        G[M] = (c["LG"] + U2["const_abs"] * cc[:, None] + U2["fast_abs"] * u)[M]
    return G


def controls(ctxs, ks, add81_ctl, add92_res, Ms):
    ctl = {}
    want = [int(add81_ctl["controls"]["t0"]["frames"]["H25_seed%d" % k]["n_tool"]) for k in ks]
    got = [int(len(c["fr"])) for c in ctxs]
    ctl["u0"] = {"n_frames": got, "n_add81": want, "pass": bool(got == want)}
    s1 = pooled(ctxs, [c["LG"] for c in ctxs], Ms)
    v1 = [s1[k] for k in STATS if k not in ("near_fast_abs", "near_step_abs")]
    ctl["u1"] = {"max_abs_statistic": float(np.nanmax(np.abs(v1))) if np.isfinite(v1).any() else float("nan"), "all_finite": bool(np.isfinite(v1).all())}
    ctl["u1"]["pass"] = bool(ctl["u1"]["all_finite"] and ctl["u1"]["max_abs_statistic"] <= TOL)
    sa = pooled(ctxs, [planted_split(c, M, False) for c, M in zip(ctxs, Ms)], Ms)
    sp = pooled(ctxs, [planted_split(c, M, True) for c, M in zip(ctxs, Ms)], Ms)
    u2 = {"abs": {"const_abs": sa["const_abs"], "fast_abs": sa["fast_abs"]}, "prop": {"const_prop": sp["const_prop"], "fast_prop": sp["fast_prop"]}, "planted": U2}
    u2["pass"] = bool(abs(sa["const_abs"] - U2["const_abs"]) <= TOL and abs(sa["fast_abs"] - U2["fast_abs"]) <= TOL
                      and abs(sp["const_prop"] - U2["const_prop"]) <= TOL and abs(sp["fast_prop"] - U2["fast_prop"]) <= TOL)
    ctl["u2"] = u2
    u3 = {"arms": [], "pass": True}
    rl = [c["rows"] for c in ctxs]
    for kind, typ, lv, code, W in U3_ARMS:
        nm = K5.arm_name("OWN", kind, typ, code, W)
        est = [K5.slopes(c, K5.plant(c, K5.unit(c, kind), typ, lv), W)["OWN"][0] for c in ctxs]
        a = K2.summarize(est, rl, [np.isfinite(x) for x in est], 20, K5.BOOT_SEED)
        ref = (add92_res.get("arms", {}).get("ALL", {}).get(nm) or {}).get("r_pooled")
        ok = bool(a is not None and ref is not None and abs(a["r_pooled"] - ref) <= TOL)
        u3["arms"].append({"arm": nm, "r_here": a["r_pooled"] if a else None, "r_add92": ref, "pass": ok})
        u3["pass"] = bool(u3["pass"] and ok)
    ctl["u3"] = u3
    return ctl


# ============================================================================================ measure and names
def measure(ctxs, Ms, ep_map, B):
    reader, frames = boot(ctxs, [c["G"] for c in ctxs], Ms, ep_map, B, BOOT_SEED)
    lag = []
    for c, M in zip(ctxs, Ms):
        ra, rb = clip_sums(c, c["G"], M)[1]
        lag.append(float(K2.CF.pearson(ra, rb)) if len(ra) >= 3 else float("nan"))
    bars = {}
    for key, (kind, typ, lv, code) in BARS.items():
        st = pooled(ctxs, [K5.plant(c, K5.unit(c, kind), typ, lv) for c in ctxs], Ms)
        bars[key] = {"planted": "%s-%s-%s" % (kind, typ, code), "value": st[key], "all_statistics": st}
    ca, fa = reader["const_abs"]["value"], reader["fast_abs"]["value"]
    return {"frames": frames, "reader": reader, "residual_lag1_per_seed": lag, "share_constant_abs": float(ca * ca / (ca * ca + fa * fa)) if (ca * ca + fa * fa) > 0 else float("nan"),
            "bars": bars, "ratio_to_bar": {k: (float(reader[k]["value"] / bars[k]["value"]) if bars[k]["value"] else float("nan")) for k in BARS}}


def names_from(P):
    names = []
    for key, stem, tail in NAME_RULES:
        ci, bar = P["reader"][key]["ci"], P["bars"][key]["value"]
        if ci is None or not np.all(np.isfinite(ci)) or not np.isfinite(bar):
            names.append(stem + "-NOT-DECIDED")
        elif ci[0] > bar:
            names.append(stem + "-ABOVE-" + tail)
        elif ci[1] <= bar:
            names.append(stem + "-WITHIN-" + tail)
        else:
            names.append(stem + "-NOT-DECIDED")
    return names


# ============================================================================================ read and show
def run_read(a, log=print):
    ks = [int(x) for x in a.seeds.split(",")]
    cpath, rpath = os.path.join(a.out_dir, "GAPSPLIT_CONTROLS.json"), os.path.join(a.out_dir, "GAPSPLIT_RESULT.json")
    if os.path.exists(cpath) or os.path.exists(rpath):
        K2.refuse("an output file exists in " + a.out_dir)
    ep_map = K2.read_json(a.episode_map)
    add81_ctl = K2.read_json(a.add81_controls)
    add92_res = K2.read_json(a.add92_result)
    ctxs = [K4.make_ctx(K2.CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)), ep_map) for k in ks]
    Ms = [frame_set(c) for c in ctxs]
    keep = K3.PLANT_SEED
    if "other_seed" in FAULTS:
        K3.PLANT_SEED = keep + 1
    try:
        ctl = controls(ctxs, ks, add81_ctl, add92_res, Ms)
        npass = sum(1 for v in ctl.values() if v["pass"])
        allp = bool(npass == len(ctl))
        os.makedirs(a.out_dir, exist_ok=True)
        K2.write_json(cpath, {"card": CARD, "reader_seeds": ks, "controls": ctl, "n_passed": npass, "n_controls": len(ctl), "controls_all_pass": allp})
        for nm in sorted(ctl):
            log("CONTROL %s %s" % (nm, "PASS" if ctl[nm]["pass"] else "FAIL"))
        P = measure(ctxs, Ms, ep_map, a.boot)
    finally:
        K3.PLANT_SEED = keep
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks, "min_clip_frames": MIN_CLIP_FRAMES,
           "bootstrap": {"B": a.boot, "seed": BOOT_SEED, "levels": "reader seeds, then evaluation episodes (episode draw shared by the seeds)", "interval": "percentile 2.5 / 97.5"},
           "controls_sha256": K2.sha256_file(cpath), "controls_all_pass": allp}
    if allp or "names_despite_fail" in FAULTS:
        res.update(P)
        res["names"] = names_from(P)
    else:
        res.update({"names": [], "withheld": P})
    K2.write_json(rpath, res)
    log("ZZ GAPSPLIT-CONTROLS %d/%d all_pass %s" % (npass, len(ctl), allp))
    return 0 if allp else 1


def run_show(a):
    C, R = K2.read_json(a.controls), K2.read_json(a.result)
    if not C.get("controls_all_pass") or C.get("n_passed") != C.get("n_controls"):
        K2.refuse("the controls did not all pass; no name and no value is shown")
    if R.get("controls_sha256") != K2.sha256_file(a.controls):
        K2.refuse("the result is not bound to this control file (sha256 differs)")
    if "withheld" in R or not R.get("names") or "reader" not in R:
        K2.refuse("the result carries no names or values")
    print("GAPSPLIT (card ADD93, tier T0); %s" % R["frames"])
    for n in R["names"]:
        print("NAME " + n)
    for k in STATS:
        v = R["reader"][k]
        ci = v["ci"] or [float("nan"), float("nan")]
        bar = R["bars"].get(k)
        print("%-14s %.4f [%.4f, %.4f] per seed %s%s" % (k, v["value"], ci[0], ci[1], ["%.4f" % x for x in v["per_seed"]],
                                                       (" | bar %s %.4f ratio %.2f" % (bar["planted"], bar["value"], R["ratio_to_bar"][k])) if bar else ""))
    print("residual lag-1 per seed %s; share of the constant part (absolute) %.3f" % (["%.3f" % x for x in R["residual_lag1_per_seed"]], R["share_constant_abs"]))
    for k, b in R["bars"].items():
        s = b["all_statistics"]
        print("BAR %s (%s): const_abs %.3f fast_abs %.3f step_abs %.3f const_prop %.4f fast_prop %.4f step_prop %.4f" % (
            b["planted"], k, s["const_abs"], s["fast_abs"], s["step_abs"], s["const_prop"], s["fast_prop"], s["step_prop"]))
    return 0


# ============================================================================================ self-test (control u4)
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
        os.makedirs(os.path.join(root, "dump"))
        for k, D in dumps.items():
            np.savez(os.path.join(root, "dump", "reader_s%d_score.npz" % k), **D)
        a92 = argparse.Namespace(dump_dir=os.path.join(root, "dump"), episode_map=os.path.join(root, "epm.json"), add81_controls=os.path.join(root, "c81.json"),
                                 out_dir=os.path.join(root, "o92"), boot=20, seeds="0,1")
        rc92 = K5.run_read(a92, log=lambda s: None)
        chk("the tool of card ADD92 reads the planted world (its result file is the reference of control u3)", rc92 == 0)
        cnt = [0]

        def go(faults=(), c81="c81.json"):
            cnt[0] += 1
            od = os.path.join(root, "o%d" % cnt[0])
            a = argparse.Namespace(dump_dir=os.path.join(root, "dump"), episode_map=os.path.join(root, "epm.json"), add81_controls=os.path.join(root, c81),
                                   add92_result=os.path.join(root, "o92", "GAPREQ_RESULT.json"), out_dir=od, boot=200, seeds="0,1")
            FAULTS.clear(), FAULTS.update(faults)
            try:
                rc = run_read(a, log=lambda s: None)
            finally:
                FAULTS.clear()
            return rc, K2.read_json(os.path.join(od, "GAPSPLIT_CONTROLS.json")), K2.read_json(os.path.join(od, "GAPSPLIT_RESULT.json")), od

        rc, C, R, od = go()
        chk("clean world: rc 0, four controls pass (failed: %s), names and values present, nothing withheld" % [k for k, v in C["controls"].items() if not v["pass"]],
            rc == 0 and C["n_passed"] == 4 and len(R.get("names", [])) == 4 and "reader" in R and "withheld" not in R)
        if "reader" in R:
            chk("the reader of the world (label plus white noise of 1 m): FAST near 1 m, STEP near 1.41 m, CONST small",
                0.9 < R["reader"]["fast_abs"]["value"] < 1.1 and 1.3 < R["reader"]["step_abs"]["value"] < 1.55 and R["reader"]["const_abs"]["value"] < 0.3)
            chk("the interval of FAST holds its point value", R["reader"]["fast_abs"]["ci"][0] <= R["reader"]["fast_abs"]["value"] <= R["reader"]["fast_abs"]["ci"][1])
            chk("bars: the white level of 0.5 m has a STEP near 0.71 m and the constant level of 20 percent a CONST near 0.2",
                0.6 < R["bars"]["step_abs"]["value"] < 0.8 and 0.15 < R["bars"]["const_prop"]["value"] < 0.25)
            chk("the white reader of 1 m is named above the bar of the step", "GAPSPLIT-STEP-ABS-ABOVE-THE-BAR" in R["names"])
        for fault, ctl in (("fast_global", "u2"), ("prop_as_abs", "u2"), ("other_seed", "u3")):
            rcf, Cf, Rf, _ = go([fault])
            chk("regression %s turns %s red, the read exits 1, no name, values withheld" % (fault, ctl),
                (not Cf["controls"][ctl]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf and "reader" not in Rf)
        rcf, Cf, Rf, _ = go(c81="c81b.json")
        chk("regression: a frame count that differs from card ADD81 turns u0 red", (not Cf["controls"]["u0"]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf)
        rcd, Cd, Rd, odd = go(["fast_global", "names_despite_fail"])
        chk("regression: names written although a control failed is caught (the healthy tool writes none)", bool(Rd["names"]) and not Cd["controls_all_pass"])
        try:
            run_show(argparse.Namespace(controls=os.path.join(odd, "GAPSPLIT_CONTROLS.json"), result=os.path.join(odd, "GAPSPLIT_RESULT.json")))
            shown = True
        except SystemExit:
            shown = False
        chk("show refuses a result whose controls did not all pass", not shown)
        fake = {"reader": {"fast_abs": {"ci": [2.0, 3.0]}, "fast_prop": {"ci": [0.01, 0.02]}, "step_abs": {"ci": [0.5, 0.9]}, "const_prop": {"ci": [0.25, 0.4]}},
                "bars": {"fast_abs": {"value": 0.8}, "fast_prop": {"value": 0.02}, "step_abs": {"value": 0.7}, "const_prop": {"value": 0.2}}}
        chk("name rule on planted intervals", names_from(fake) == ["GAPSPLIT-FAST-ABS-ABOVE-THE-BAR", "GAPSPLIT-FAST-PROP-WITHIN-THE-BAR", "GAPSPLIT-STEP-ABS-NOT-DECIDED",
                                                                   "GAPSPLIT-CONSTANT-PROP-ABOVE-THE-CARRIED-LEVEL"])
    for ln in lines:
        print(ln)
    print("ZZ GAPSPLIT-SELFTEST rc %d" % (1 if bad else 0))
    return 1 if bad else 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--closeread-tools", default=None)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("read")
    for x in ("--dump-dir", "--episode-map", "--add81-controls", "--add92-result", "--out-dir"):
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
