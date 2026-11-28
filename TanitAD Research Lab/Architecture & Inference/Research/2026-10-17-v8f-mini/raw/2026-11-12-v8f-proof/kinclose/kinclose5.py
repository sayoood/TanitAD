#!/usr/bin/env python
"""kinclose5.py -- tool of card PREREG_V8F_PROOF_ADD92 (GAPREQ), tier T0 (a diagnostic on logged frames; no rollout).

Planted gap error on the label gap (absolute and proportional; white, reader-like and constant per clip) and the slope of the planted gap over the
run found from the planted gap itself (OWN) or from the label gap (EXACT), read against the label closing speed. Numpy only. Controls h0 to h3 are
computed by the read, h4 is the self-test. With one control failed no name is written and every value is held under the key "withheld".

  python kinclose5.py --selftest [--closeread-tools DIR]
  python kinclose5.py read --dump-dir D --episode-map E --add81-controls C --out-dir O [--closeread-tools DIR]
  python kinclose5.py show --result R --controls C
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kinclose2 as K2  # noqa: E402
import kinclose3 as K3  # noqa: E402
import kinclose4 as K4  # noqa: E402

CARD = "PREREG_V8F_PROOF_ADD92"
WS = (5, 10)
LEVELS = {"ABS": ((0.5, "S050"), (1.0, "S100"), (2.0, "S200"), (4.0, "S400")), "PROP": ((0.025, "P025"), (0.05, "P050"), (0.10, "P100"), (0.20, "P200"))}
PLANS = (("W", "ABS"), ("R", "ABS"), ("W", "PROP"), ("R", "PROP"), ("C", "PROP"))
RUNS = (("OWN", "reader"), ("EXACT", "label"))
LAG_BANDS = {"W": (-0.05, 0.05), "R": (0.89, 0.97), "C": (0.999, 1.000001)}
REQ_R, BOOT_B, BOOT_SEED, PLANT_SEED_C = 0.9, 2000, 92, 92
H1_SLOPE_TOL, H_TOL = 1e-6, 1e-9
C_ABS_CHECK_M = 4.0
PAIR_AT = ("R", "ABS", "S100", 10)
NEAR_W = 10
FAULTS = set()  # in-process regression switches of the self-test; the command line never sets them


def unit(c, kind):
    """[clips, T] unit noise of one kind, mean 0 and standard deviation 1 exactly over the frames with a label gap."""
    if kind in ("W", "R"):
        N = K3.unit_noise(c, kind)
    else:
        N = np.zeros((len(c["clips"]), c["T"]))
        for i, name in enumerate(c["clips"]):
            h = int(hashlib.sha256(("%s|C|%d" % (name, PLANT_SEED_C)).encode("utf-8")).hexdigest()[:8], 16)
            N[i, :] = np.random.default_rng(h).standard_normal()
        m = c["LM"]
        N = (N - N[m].mean()) / N[m].std()
    return N * (1.1 if "no_normalise" in FAULTS else 1.0)


def plant(c, N, typ, lv):
    if typ == "ABS" or "prop_as_abs" in FAULTS:
        return c["LG"] + lv * N
    return c["LG"] * (1.0 + lv * N)


def slopes(c, P, W):
    """-> {run form: (estimate per scored row, short-run flag per row)} for the gap P put in the place of the reader's gap."""
    c2 = dict(c, G=P, RM=c["LM"].copy())
    return {run: K4.run_slope(c2, K4.run_index(c2, W, src)) for run, src in RUNS}


def arm_name(run, kind, typ, code, W):
    return "%s-%s-%s-%s-W%d" % (run, kind, typ, code, W)


# ============================================================================================ controls h0 to h3
def controls(ctxs, ks, add81_ctl):
    ctl = {}
    want = [int(add81_ctl["controls"]["t0"]["frames"]["H25_seed%d" % k]["n_tool"]) for k in ks]
    got = [int(len(c["fr"])) for c in ctxs]
    ctl["h0"] = {"n_frames": got, "n_add81": want, "pass": bool(got == want)}
    h1 = {"max_abs_own_minus_exact_ms": 0.0, "max_abs_run5_minus_label_ms": 0.0, "max_abs_shift_of_constant_offset_ms": 0.0, "n_rows": 0, "n_without_value": 0}
    for c in ctxs:
        NC = unit(c, "C")
        for W in WS:
            s0 = slopes(c, c["LG"], W)
            sc = slopes(c, c["LG"] + C_ABS_CHECK_M * NC, W)
            o, x = s0["OWN"][0], s0["EXACT"][0]
            h1["n_rows"] += int(len(o))
            h1["n_without_value"] += int((~np.isfinite(o)).sum() + (~np.isfinite(x)).sum())
            f = np.isfinite(o) & np.isfinite(x)
            if f.any():
                h1["max_abs_own_minus_exact_ms"] = max(h1["max_abs_own_minus_exact_ms"], float(np.abs(o[f] - x[f]).max()))
                if W == 5:
                    h1["max_abs_run5_minus_label_ms"] = max(h1["max_abs_run5_minus_label_ms"], float(np.abs(o[f] - c["y"][f]).max()))
            for run in ("OWN", "EXACT"):
                a, b = s0[run][0], sc[run][0]
                same = np.isfinite(a) == np.isfinite(b)
                g = np.isfinite(a) & np.isfinite(b)
                d = float(np.abs(a[g] - b[g]).max()) if g.any() else 0.0
                h1["max_abs_shift_of_constant_offset_ms"] = max(h1["max_abs_shift_of_constant_offset_ms"], d if bool(same.all()) else float("inf"))
    h1["pass"] = bool(h1["n_rows"] > 0 and h1["n_without_value"] == 0 and h1["max_abs_own_minus_exact_ms"] <= H_TOL and h1["max_abs_run5_minus_label_ms"] <= H1_SLOPE_TOL
                      and h1["max_abs_shift_of_constant_offset_ms"] <= H_TOL)
    ctl["h1"] = h1
    c = ctxs[0]
    m = c["LM"]
    h2 = {"kinds": {}, "pass": True}
    for kind in ("W", "R", "C"):
        N = unit(c, kind)
        mean, sd, l1 = float(N[m].mean()), float(N[m].std()), K3.lag1(N, m)
        lo, hi = LAG_BANDS[kind]
        ok = bool(abs(mean) <= H_TOL and abs(sd - 1.0) <= H_TOL and lo <= l1 <= hi)
        h2["kinds"][kind] = {"mean": mean, "sd": sd, "lag1": l1, "band": [lo, hi], "pass": ok}
        h2["pass"] = bool(h2["pass"] and ok)
    ctl["h2"] = h2
    h3 = {"levels": [], "label_gap_min_m": float(c["LG"][m].min()), "pass": bool(c["LG"][m].min() > 0)}
    for kind, typ in PLANS:
        N = unit(c, kind)
        for lv, code in LEVELS[typ]:
            P = plant(c, N, typ, lv)
            d = (P[m] - c["LG"][m]) if typ == "ABS" else (P[m] / c["LG"][m] - 1.0)
            # note N1 of the card: the sign of the planted gap is judged for the proportional form only; for an absolute level the count at or below 0 is recorded
            ok = bool(abs(float(d.std()) - lv) <= H_TOL and abs(float(d.mean())) <= H_TOL and (typ == "ABS" or float(P[m].min()) > 0))
            h3["levels"].append({"kind": kind, "type": typ, "code": code, "level": lv, "realised": float(d.std()), "mean": float(d.mean()), "min_planted_gap_m": float(P[m].min()),
                                 "n_at_or_below_0": int((P[m] <= 0).sum()), "n_frames": int(m.sum()), "pass": ok})
            h3["pass"] = bool(h3["pass"] and ok)
    ctl["h3"] = h3
    return ctl


# ============================================================================================ arms and names
def measure(ctxs, B):
    rl = [c["rows"] for c in ctxs]
    allm = [np.ones(len(c["fr"]), bool) for c in ctxs]
    near = [K4.bin_masks(dict(c, G=c["LG"], RM=c["LM"]))["NEAR"] for c in ctxs]
    est, short = {}, {}
    for c in ctxs:
        for W in WS:
            s0 = slopes(c, c["LG"], W)
            est.setdefault("L0-W%d" % W, []).append(s0["OWN"][0])
        for kind, typ in PLANS:
            N = unit(c, kind)
            for lv, code in LEVELS[typ]:
                P = plant(c, N, typ, lv)
                for W in WS:
                    for run, (e, s) in slopes(c, P, W).items():
                        nm = arm_name(run, kind, typ, code, W)
                        est.setdefault(nm, []).append(e)
                        short.setdefault(nm, []).append(s)
    out = {"frames": {"scored": [int(len(c["fr"])) for c in ctxs], "near": [int(m.sum()) for m in near]}, "arms": {"ALL": {}, "NEAR": {}}, "pairs": {}, "prop_levels_in_metres": {}}

    def summ(nm, masks0):
        e = est[nm]
        masks = [m & np.isfinite(x) for m, x in zip(masks0, e)]
        if min(int(m.sum()) for m in masks) < 3:
            return None
        a = K2.summarize(e, rl, masks, B, BOOT_SEED)
        if a is not None and nm in short:
            n = sum(int(m.sum()) for m in masks)
            a["short_run_share"] = float(sum(int((s & m).sum()) for s, m in zip(short[nm], masks)) / n)
        return a

    for nm in est:
        out["arms"]["ALL"][nm] = summ(nm, allm)
        if nm.endswith("-W%d" % NEAR_W):
            out["arms"]["NEAR"][nm] = summ(nm, near)
    k, t, code, W = PAIR_AT
    a, b = est[arm_name("OWN", k, t, code, W)], est[arm_name("EXACT", k, t, code, W)]
    both = [np.isfinite(x) & np.isfinite(y) for x, y in zip(a, b)]
    out["pairs"]["OWN_minus_EXACT_%s_%s_%s_W%d" % (k, t, code, W)] = K2.paired(a, b, rl, both, B, BOOT_SEED) if min(int(m.sum()) for m in both) >= 3 else None
    c = ctxs[0]
    m = c["LM"]
    with np.errstate(invalid="ignore"):
        mn = m & (c["LG"] < 20.0)
    for kind, typ in PLANS:
        if typ != "PROP":
            continue
        N = unit(c, kind)
        for lv, code in LEVELS[typ]:
            d = plant(c, N, typ, lv) - c["LG"]
            out["prop_levels_in_metres"]["%s-%s" % (kind, code)] = {"all_frames_sd_m": float(d[m].std()), "below_20m_sd_m": float(d[mn].std()) if int(mn.sum()) > 1 else float("nan")}
    return out


def names_from(P):
    names = []
    A = P["arms"]["ALL"]
    for run, _ in RUNS:
        for kind, typ in PLANS:
            for W in WS:
                best = None
                for _, code in LEVELS[typ]:
                    a = A.get(arm_name(run, kind, typ, code, W))
                    if a and a.get("r_ci") is not None and np.all(np.isfinite(a["r_ci"])) and a["r_ci"][0] >= REQ_R:
                        best = code
                    else:
                        break
                stem = "GAPREQ-%s-%s-%s-W%d" % (run, kind, typ, W)
                names.append((stem + "-UP-TO-" + best) if best else (stem + "-BELOW-" + LEVELS[typ][0][1]))
    k, t, code, W = PAIR_AT
    pr = P["pairs"].get("OWN_minus_EXACT_%s_%s_%s_W%d" % (k, t, code, W))
    if pr and pr["ci"][0] > 0:
        names.append("GAPREQ-RUN-FINDING-COSTS")
    elif pr and pr["ci"][1] < 0:
        names.append("GAPREQ-RUN-FINDING-HELPS")
    else:
        names.append("GAPREQ-RUN-FINDING-NOT-SEPARATED")
    return names


# ============================================================================================ read and show
def run_read(a, log=print):
    ks = [int(x) for x in a.seeds.split(",")]
    cpath, rpath = os.path.join(a.out_dir, "GAPREQ_CONTROLS.json"), os.path.join(a.out_dir, "GAPREQ_RESULT.json")
    if os.path.exists(cpath) or os.path.exists(rpath):
        K2.refuse("an output file exists in " + a.out_dir)
    ep_map = K2.read_json(a.episode_map)
    add81_ctl = K2.read_json(a.add81_controls)
    ctxs = [K4.make_ctx(K2.CF.load_dump(os.path.join(a.dump_dir, "reader_s%d_score.npz" % k)), ep_map) for k in ks]
    ctl = controls(ctxs, ks, add81_ctl)
    npass = sum(1 for v in ctl.values() if v["pass"])
    allp = bool(npass == len(ctl))
    os.makedirs(a.out_dir, exist_ok=True)
    K2.write_json(cpath, {"card": CARD, "reader_seeds": ks, "controls": ctl, "n_passed": npass, "n_controls": len(ctl), "controls_all_pass": allp})
    for nm in sorted(ctl):
        log("CONTROL %s %s" % (nm, "PASS" if ctl[nm]["pass"] else "FAIL"))
    P = measure(ctxs, a.boot)
    res = {"card": CARD, "tier": "T0", "reader_seeds": ks,
           "bootstrap": {"B": a.boot, "seed": BOOT_SEED, "levels": "reader seeds, then evaluation episodes (episode draw shared by the seeds)", "interval": "percentile 2.5 / 97.5"},
           "controls_sha256": K2.sha256_file(cpath), "controls_all_pass": allp}
    if allp or "names_despite_fail" in FAULTS:
        res.update(P)
        res["names"] = names_from(P)
    else:
        res.update({"names": [], "withheld": P})
    K2.write_json(rpath, res)
    log("ZZ GAPREQ-CONTROLS %d/%d all_pass %s" % (npass, len(ctl), allp))
    return 0 if allp else 1


def run_show(a):
    C, R = K2.read_json(a.controls), K2.read_json(a.result)
    if not C.get("controls_all_pass") or C.get("n_passed") != C.get("n_controls"):
        K2.refuse("the controls did not all pass; no name and no value is shown")
    if R.get("controls_sha256") != K2.sha256_file(a.controls):
        K2.refuse("the result is not bound to this control file (sha256 differs)")
    if "withheld" in R or not R.get("names") or "arms" not in R:
        K2.refuse("the result carries no names or arms")
    print("GAPREQ (card ADD92, tier T0); frames scored %s near %s" % (R["frames"]["scored"], R["frames"]["near"]))
    for n in R["names"]:
        print("NAME " + n)

    def cell(v):
        if not v:
            return "(none)"
        return "%+.3f [%+.3f, %+.3f] e%.2f s%.2f" % (v["r_pooled"], v["r_ci"][0], v["r_ci"][1], v["err_sd"], v.get("short_run_share", 0.0))

    for W in WS:
        print("L0 W%d %s" % (W, cell(R["arms"]["ALL"].get("L0-W%d" % W))))
    for b in ("ALL", "NEAR"):
        for kind, typ in PLANS:
            for W in WS:
                if b == "NEAR" and W != NEAR_W:
                    continue
                for run, _ in RUNS:
                    print("%s %s-%s W%d %s: %s" % (b, kind, typ, W, run, " | ".join("%s %s" % (code, cell(R["arms"][b].get(arm_name(run, kind, typ, code, W)))) for _, code in LEVELS[typ])))
    for k, v in R["pairs"].items():
        if v:
            print("PAIR %s abs err %+.3f [%+.3f, %+.3f] n %d" % (k, v["diff_abs_err"], v["ci"][0], v["ci"][1], v["n_frames"]))
    for k, v in R["prop_levels_in_metres"].items():
        print("METRES %s all %.2f below20 %.2f" % (k, v["all_frames_sd_m"], v["below_20m_sd_m"]))
    return 0


# ============================================================================================ self-test (control h4)
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
        cnt = [0]

        def go(faults=(), c81="c81.json"):
            cnt[0] += 1
            od = os.path.join(root, "o%d" % cnt[0])
            a = argparse.Namespace(dump_dir=os.path.join(root, "dump"), episode_map=os.path.join(root, "epm.json"), add81_controls=os.path.join(root, c81), out_dir=od, boot=100, seeds="0,1")
            FAULTS.clear(), FAULTS.update(faults)
            K4.FAULTS.clear()
            if "no_step_gate" in faults:
                K4.FAULTS.add("no_step_gate")
            try:
                rc = run_read(a, log=lambda s: None)
            finally:
                FAULTS.clear(), K4.FAULTS.clear()
            return rc, K2.read_json(os.path.join(od, "GAPREQ_CONTROLS.json")), K2.read_json(os.path.join(od, "GAPREQ_RESULT.json")), od

        rc, C, R, od = go()
        chk("clean world: rc 0, four controls pass (failed: %s), names and arms present, nothing withheld" % [k for k, v in C["controls"].items() if not v["pass"]],
            rc == 0 and C["n_passed"] == 4 and bool(R["names"]) and "arms" in R and "withheld" not in R)
        if "arms" in R:
            A = R["arms"]["ALL"]
            chk("level 0 over the run of 5 reads the label closing speed at r 1", abs(A["L0-W5"]["r_pooled"] - 1.0) < 1e-9)
            chk("names: twenty ladder names and one run-finding name", len(R["names"]) == 21 and sum(1 for x in R["names"] if "RUN-FINDING" in x) == 1)
            chk("ladder: white absolute error of 4 m reads worse than 0.5 m on exact runs", A["EXACT-W-ABS-S400-W5"]["r_pooled"] < A["EXACT-W-ABS-S050-W5"]["r_pooled"])
            chk("a scale error that is constant over a clip reads better than a white one of the same level (P200, W 10, exact runs)",
                A["EXACT-C-PROP-P200-W10"]["r_pooled"] > A["EXACT-W-PROP-P200-W10"]["r_pooled"])
            chk("the proportional levels are reported in metres", len(R["prop_levels_in_metres"]) == 12 and all(v["all_frames_sd_m"] > 0 for v in R["prop_levels_in_metres"].values()))
            chk("NEAR holds the arms of W 10 only", bool(R["arms"]["NEAR"]) and all(k.endswith("-W10") for k in R["arms"]["NEAR"]))
        for fault, ctl in (("no_normalise", "h2"), ("prop_as_abs", "h3"), ("no_step_gate", "h1")):
            rcf, Cf, Rf, _ = go([fault])
            chk("regression %s turns %s red, the read exits 1, no name, values withheld" % (fault, ctl),
                (not Cf["controls"][ctl]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf and "arms" not in Rf)
        rcf, Cf, Rf, _ = go(c81="c81b.json")
        chk("regression: a frame count that differs from card ADD81 turns h0 red", (not Cf["controls"]["h0"]["pass"]) and rcf == 1 and Rf["names"] == [] and "withheld" in Rf)
        rcd, Cd, Rd, odd = go(["no_normalise", "names_despite_fail"])
        chk("regression: names written although a control failed is caught (the healthy tool writes none)", bool(Rd["names"]) and not Cd["controls_all_pass"])
        try:
            run_show(argparse.Namespace(controls=os.path.join(odd, "GAPREQ_CONTROLS.json"), result=os.path.join(odd, "GAPREQ_RESULT.json")))
            shown = True
        except SystemExit:
            shown = False
        chk("show refuses a result whose controls did not all pass", not shown)
        fake = {"arms": {"ALL": {}}, "pairs": {"OWN_minus_EXACT_R_ABS_S100_W10": {"ci": [-0.3, -0.1]}}}
        for run, _ in RUNS:
            for kind, typ in PLANS:
                for W in WS:
                    for i, (_, code) in enumerate(LEVELS[typ]):
                        good = (i <= 1) if run == "EXACT" else (i == 0 and kind == "C")
                        fake["arms"]["ALL"][arm_name(run, kind, typ, code, W)] = {"r_ci": [0.95, 0.99] if good else [0.5, 0.95]}
        nf = names_from(fake)
        chk("name rule on planted intervals: exact runs up to the second level, own runs below the smallest except kind C, run finding helps",
            "GAPREQ-EXACT-R-ABS-W10-UP-TO-S100" in nf and "GAPREQ-EXACT-C-PROP-W5-UP-TO-P050" in nf and "GAPREQ-OWN-R-ABS-W10-BELOW-S050" in nf
            and "GAPREQ-OWN-C-PROP-W10-UP-TO-P025" in nf and "GAPREQ-OWN-W-PROP-W5-BELOW-P025" in nf and nf[-1] == "GAPREQ-RUN-FINDING-HELPS" and len(nf) == 21)
        fake["pairs"]["OWN_minus_EXACT_R_ABS_S100_W10"] = {"ci": [0.1, 0.3]}
        chk("name rule: a pair above 0 names the cost of run finding", names_from(fake)[-1] == "GAPREQ-RUN-FINDING-COSTS")
        fake["pairs"]["OWN_minus_EXACT_R_ABS_S100_W10"] = {"ci": [-0.1, 0.3]}
        chk("name rule: a pair that spans 0 is not separated", names_from(fake)[-1] == "GAPREQ-RUN-FINDING-NOT-SEPARATED")
    for ln in lines:
        print(ln)
    print("ZZ GAPREQ-SELFTEST rc %d" % (1 if bad else 0))
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
