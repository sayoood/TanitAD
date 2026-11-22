#!/usr/bin/env python3
"""read_lookswitch.py -- reader of card PREREG_V8F_PROOF_ADD109 (LOOKSWITCH-T0). Tier T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance).
The mix S_k: the gap of the fixed-window fit D_k while the head's direction of the frame is inside the fixed narrow window (|azimuth| <= 8 and |elevation + 2| <= 8 degrees), the gap of
the fit at the head's direction P_k otherwise; fit seeds k = 0, 1, 2. CPU, numpy only. Imports the readers of cards ADD108 (read_lookgap.py) and ADD107 (read_lookat.py) unchanged.
  read_lookswitch.py --stage controls --out CONTROLS.json [paths]     prints only the lines w<i> PASS|FAIL and a last line; the values go to the file
  read_lookswitch.py --stage names --controls CONTROLS.json --out RESULT.json --log NAMES.log     refuses unless w0, w1, w2 all pass
  read_lookswitch.py --selftest
"""
import argparse
import importlib.util
import io
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD109"
PRE = "LOOKSWITCH-T0"
N_BOOT, BOOT_SEED = 2000, 109
THETA, EL0 = 8.0, -2.0
REF_E = {"D0": 3.960, "D1": 3.913, "D2": 4.112, "P0": 3.895, "P1": 3.699, "P2": 3.689}
REF_S = {"S4": 0.8748, "S8_out": 0.5844}
REF_MIN_D = 3.913
ROWS = 3841
V8 = "/workspace/v8proof"
TIER = "T0 (recorded frames, no rollout; a diagnostic of a perception head, not driving performance)"
FAMILIES = "LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC: not read (n = 0 for each; reason: T0, no rollout and no plan in this card, card section 5)"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inside_mask(az, el, theta=THETA):
    return (np.abs(az) <= theta) & (np.abs(el - EL0) <= theta)


def boot_ratio(num, den, n_boot=N_BOOT, seed=BOOT_SEED):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    nc = len(num)
    idx = np.random.RandomState(seed).randint(0, nc, size=(n_boot, nc))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    r = r[np.isfinite(r)]
    if r.size < n_boot // 2:
        return None
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def mix(ins, dg, pg):
    return np.where(ins, dg, pg)


def mix_second(ins, dg, pg):
    """second routine, written apart: a plain loop"""
    out = []
    for i in range(len(ins)):
        out.append(float(dg[i]) if bool(ins[i]) else float(pg[i]))
    return np.array(out)


def ratio_block(lab, ci, nc, A, B, n_boot=N_BOOT, seed=BOOT_SEED):
    """A, B: lists of gap arrays (three fits each). -> E of each, ratio of the means, interval (per-clip sums, mean over the fits)."""
    ea = [np.abs(g - lab) for g in A]
    eb = [np.abs(g - lab) for g in B]
    num = sum(np.bincount(ci, weights=e, minlength=nc) for e in ea) / float(len(ea))
    den = sum(np.bincount(ci, weights=e, minlength=nc) for e in eb) / float(len(eb))
    Ea, Eb = [float(e.mean()) for e in ea], [float(e.mean()) for e in eb]
    return {"E_a": Ea, "E_b": Eb, "ratio": float(np.mean(Ea) / np.mean(Eb)), "iv": boot_ratio(num, den, n_boot, seed), "sums": {"num": num.tolist(), "den": den.tolist()}}


def compute(lab, ci, nc, D, P, az_dir, el_dir, az_lab=None, n_boot=N_BOOT, seed=BOOT_SEED, bins=None):
    """lab [rows], ci clip index [rows], D and P lists of three gap arrays, direction of the head per row. -> payload"""
    ins = inside_mask(az_dir, el_dir)
    S = [mix(ins, D[k], P[k]) for k in range(3)]
    out = {"n_rows": int(len(lab)), "n_inside": int(ins.sum()), "n_outside": int((~ins).sum()), "n_clips_with_outside": int(len(np.unique(ci[~ins]))), "n_clips": int(nc),
           "n_boot": n_boot, "seed": seed, "theta_deg": THETA, "tier": TIER, "families": FAMILIES}
    rb = ratio_block(lab, ci, nc, S, D, n_boot, seed)
    out["E"] = {"S%d" % k: rb["E_a"][k] for k in range(3)}
    out["E"].update({"D%d" % k: rb["E_b"][k] for k in range(3)})
    out["E"].update({"P%d" % k: float(np.abs(P[k] - lab).mean()) for k in range(3)})
    out["R"], out["iv"], out["sums"] = rb["ratio"], rb["iv"], rb["sums"]
    out["second_routine_max_abs_diff"] = float(max(np.max(np.abs(mix_second(ins, D[k], P[k]) - S[k])) for k in range(3)))
    out["parts"] = {}
    for nm, m in (("inside", ins), ("outside", ~ins)):
        out["parts"][nm] = {"n": int(m.sum()), "E": {"%s%d" % (a, k): (float(np.abs(g[k][m] - lab[m]).mean()) if m.any() else None) for a, g in (("S", S), ("D", D), ("P", P)) for k in range(3)}}
    inv = [mix(~ins, D[k], P[k]) for k in range(3)]
    ib = ratio_block(lab, ci, nc, inv, D, n_boot, seed)
    out["inverse"] = {"E": ib["E_a"], "ratio": ib["ratio"], "iv": ib["iv"]}
    out["other_bounds"] = {}
    for th in (4.0, 16.0):
        m = inside_mask(az_dir, el_dir, th)
        tb = ratio_block(lab, ci, nc, [mix(m, D[k], P[k]) for k in range(3)], D, n_boot, seed)
        out["other_bounds"]["%g" % th] = {"n_outside": int((~m).sum()), "E": tb["E_a"], "ratio": tb["ratio"], "iv": tb["iv"]}
    if bins is not None and az_lab is not None:
        gb, ab = bins
        e0 = S[0] - lab
        out["bins_gap_S0"] = [{"bin": "%d-%d m" % (lo, hi), "n": int(m.sum()), "E": float(np.abs(e0[m]).mean()) if m.any() else None, "signed": float(e0[m].mean()) if m.any() else None,
                               "short5": float((e0[m] < -5.0).mean()) if m.any() else None} for lo, hi, m in gb(lab)]
        out["bins_az_S0"] = [{"bin": nm, "n": int(m.sum()), "E": float(np.abs(e0[m]).mean()) if m.any() else None} for nm, m in ab(az_lab)]
    return out


def name_of(P):
    iv = P.get("iv")
    if iv is None:
        return None
    if iv[1] < 1.0:
        return PRE + ("-CUTS" if all(P["E"]["S%d" % k] < REF_MIN_D for k in range(3)) else "-CUTS-NOT-EACH-FIT")
    if iv[0] > 1.0:
        return PRE + "-WORSE"
    return PRE + "-NOT-SEPARATED"


def name_second(P):
    """second derivation from the per-clip sums with plain loops and a hand-made percentile"""
    num, den = P["sums"]["num"], P["sums"]["den"]
    nc = len(num)
    idx = np.random.RandomState(P["seed"]).randint(0, nc, size=(P["n_boot"], nc)).tolist()
    v = []
    for row in idx:
        d = sum(den[j] for j in row)
        if d > 0:
            v.append(sum(num[j] for j in row) / d)
    if len(v) < P["n_boot"] // 2:
        return None
    v.sort()

    def pct(q):
        pos = (len(v) - 1) * q / 100.0
        a = int(pos)
        b = min(a + 1, len(v) - 1)
        return v[a] + (v[b] - v[a]) * (pos - a)
    lo, hi = pct(2.5), pct(97.5)
    if hi < 1.0:
        return PRE + ("-CUTS" if all(P["E"]["S%d" % k] < REF_MIN_D for k in range(3)) else "-CUTS-NOT-EACH-FIT")
    if lo > 1.0:
        return PRE + "-WORSE"
    return PRE + "-NOT-SEPARATED"


def ctl_w0(P, dirs_stats):
    bad = []
    for k, v in REF_E.items():
        e = P["E"].get(k)
        if e is None or abs(e - v) > 1e-3:
            bad.append("E(%s) %s, expected %.3f within 1e-3" % (k, e, v))
    if P["n_rows"] != ROWS:
        bad.append("%d rows, expected %d" % (P["n_rows"], ROWS))
    for k, v in REF_S.items():
        s = dirs_stats.get(k)
        if s is None or abs(s - v) > 1e-4:
            bad.append("%s %s, expected %.4f within 1e-4" % (k, s, v))
    return {"pass": not bad, "failures": bad}


def ctl_w1(P):
    bad = []
    if P["second_routine_max_abs_diff"] != 0.0:
        bad.append("the second routine differs by %s" % P["second_routine_max_abs_diff"])
    if P["n_inside"] + P["n_outside"] != P["n_rows"]:
        bad.append("inside and outside rows do not add up")
    return {"pass": not bad, "failures": bad}


def selftest():
    """synthetic world: a planted +5 m on the outside rows of P raises E(S_k) by exactly 5 m times the share of outside rows; a swap of inside and outside is caught by the second routine."""
    rng = np.random.RandomState(5)
    nc, per = 12, 40
    n = nc * per
    ci = np.repeat(np.arange(nc), per)
    lab = rng.uniform(5, 55, n)
    az = rng.normal(0, 7, n)
    el = rng.normal(-2, 3, n)
    D = [lab + np.abs(rng.normal(0, 3, n)) for _ in range(3)]
    Pg = [lab + np.abs(rng.normal(0, 3, n)) for _ in range(3)]
    ins = inside_mask(az, el)
    a = compute(lab, ci, nc, D, Pg, az, el, n_boot=200)
    P2 = [g + np.where(ins, 0.0, 5.0) for g in Pg]
    b = compute(lab, ci, nc, D, P2, az, el, n_boot=200)
    share = float((~ins).mean())
    ok = []
    ok.append(all(abs((b["E"]["S%d" % k] - a["E"]["S%d" % k]) - 5.0 * share) < 1e-9 for k in range(3)))
    ok.append(ctl_w1(a)["pass"])
    swapped = mix(~ins, D[0], Pg[0])
    ok.append(float(np.max(np.abs(mix_second(ins, D[0], Pg[0]) - swapped))) > 0.0)
    ok.append(name_of(a) == name_second(a))
    ok.append(0 < a["n_outside"] < n)
    print("selftest checks %s" % ok)
    print("ZZ READ-LOOKSWITCH SELFTEST %s" % ("OK" if all(ok) else "FAIL"))
    return all(ok)


def run_controls(a):
    RG = load_mod(a.gap_reader, "read_lookgap")
    LK = RG.load_lookat(a.lookat_reader)[0]
    ctl = {"w2": {"pass": bool(selftest_quiet()), "failures": []}}
    J = LK.load_npz(a.eval_join)
    paths = {"D0": V8 + "/cropread/run1/dump/D_H5/reader_s0_score.npz", "D1": V8 + "/cropread/run1/dump/D_H5/reader_s1_score.npz", "D2": V8 + "/visseed10k/fitroot/dump/D_H5/reader_s2_score.npz",
             "D3": V8 + "/visseed10k/fitroot/dump/D_H5/reader_s3_score.npz", "D4": V8 + "/visseed10k/fitroot/dump/D_H5/reader_s4_score.npz",
             "B0": V8 + "/cropread/run1/dump/B_H5/reader_s0_score.npz", "B1": V8 + "/cropread/run1/dump/B_H5/reader_s1_score.npz"}
    for k in range(3):
        paths["P%d" % k] = a.lookgap_root + "/run/dump/B_H5/reader_s%d_score.npz" % k
    L = {k: RG.lay(LK, J, LK.load_npz(p)) for k, p in paths.items()}
    d0 = L["D0"]
    sel = d0["lab_present"] & np.isfinite(d0["lab_gap"])
    for k in L:
        sel &= np.isfinite(L[k]["gap"])
    rows = np.flatnonzero(sel)
    n = np.asarray(J["n"], np.int64)
    nc = len(n)
    ci = np.repeat(np.arange(nc), n)[rows]
    lab = d0["lab_gap"][rows]
    clips = [str(c) for c in J["clip"]]
    box = LK.box_on_rows(J, RG.tracks_of(LK, a.box_dir, clips))
    dirs = LK.box_on_rows(J, RG.tracks_of(LK, a.dirs_dir, clips))
    azd, eld, azl, ell = dirs["az"][rows], dirs["el"][rows], box["az"][rows], box["el"][rows]

    def gb(lb):
        return [(lo, hi, RG.gap_bin_mask(lb, lo, hi)) for lo, hi in RG.GAP_BINS]

    def ab(azv):
        return [(RG.bin_name_az(lo, hi), RG.az_bin_mask(azv, lo, hi)) for lo, hi in RG.AZ_BINS]
    P = compute(lab, ci, nc, [L["D%d" % k]["gap"][rows] for k in range(3)], [L["P%d" % k]["gap"][rows] for k in range(3)], azd, eld, azl, a.n_boot, BOOT_SEED, (gb, ab))
    st, _, _, _ = LK._stats(np.abs(azd - azl), np.abs(eld - ell), lab, azl)
    P["dirs"] = {"S4": st["S4"], "S8": st["S8"], "S8_out": st["S8_out"], "n_no_dir": int((~dirs["has"][rows]).sum())}
    ctl["w0"] = ctl_w0(P, P["dirs"])
    ctl["w1"] = ctl_w1(P)
    allp = all(ctl[k]["pass"] for k in ("w0", "w1", "w2"))
    res = {"card": CARD, "tier": TIER, "controls": ctl, "all_pass": bool(allp), "reader_sha256": LK.sha_file(os.path.abspath(__file__)), "gap_reader_sha256": LK.sha_file(a.gap_reader),
           "lookat_reader_sha256": LK.sha_file(a.lookat_reader), "inputs": {k: LK.sha_file(p) for k, p in paths.items()}}
    res["payload" if allp else "withheld"] = P
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh)
    for k in ("w0", "w1", "w2"):
        print("%s %s" % (k, "PASS" if ctl[k]["pass"] else "FAIL"))
    print("ZZ READ-LOOKSWITCH controls all_pass %s" % allp)
    return 0 if allp else 1


def selftest_quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return selftest()
    finally:
        sys.stdout = old


def fiv(iv):
    return "no interval" if iv is None else "[%.4f, %.4f]" % (iv[0], iv[1])


def run_names(a):
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != ["w0", "w1", "w2"] or any(v.get("pass") is not True for v in cj["controls"].values()):
        print("REFUSED: the controls file is not all-pass with w0 to w2 present: no name is written")
        return 3
    P = cj["payload"]
    n1, n2 = name_of(P), name_second(P)
    if n1 != n2:
        print("STOP: the two derivations of the name disagree: %s | %s" % (n1, n2))
        return 4
    if os.path.exists(a.out) or os.path.exists(a.log):
        print("REFUSED: %s or %s exists" % (a.out, a.log))
        return 2
    E = P["E"]
    L = ["tier: " + TIER, "families: " + FAMILIES,
         "rows %d of %d clips; head's direction inside the fixed narrow window on %d rows, outside on %d rows (%d clips have outside rows)" % (P["n_rows"], P["n_clips"], P["n_inside"], P["n_outside"], P["n_clips_with_outside"]),
         "E (mean absolute gap error, m): " + "; ".join("%s %.3f" % (k, E[k]) for k in ("S0", "S1", "S2", "D0", "D1", "D2", "P0", "P1", "P2")),
         "R_S = mean E(S0..S2) / mean E(D0..D2) = %.4f; interval %s (%d draws, RandomState(%d), two-sided 95 %% percentile, cluster bootstrap over the evaluation clips, fits held fixed)" % (P["R"], fiv(P["iv"]), P["n_boot"], P["seed"]),
         "name: %s" % n1]
    for nm in ("inside", "outside"):
        q = P["parts"][nm]
        L.append("%s rows (n %d): " % (nm, q["n"]) + "; ".join("%s %s" % (k, "n/a" if v is None else "%.3f" % v) for k, v in q["E"].items()))
    L.append("inverse mix (P inside, D outside) against D: E %s; ratio %.4f interval %s" % (", ".join("%.3f" % x for x in P["inverse"]["E"]), P["inverse"]["ratio"], fiv(P["inverse"]["iv"])))
    for th, q in P["other_bounds"].items():
        L.append("mix with the bound of %s degrees (outside rows %d): E %s; ratio %.4f interval %s" % (th, q["n_outside"], ", ".join("%.3f" % x for x in q["E"]), q["ratio"], fiv(q["iv"])))
    L.append("S0 by label gap: " + "; ".join("%s n %d E %.3f signed %.3f short>5m %.3f" % (b["bin"], b["n"], b["E"], b["signed"], b["short5"]) for b in P["bins_gap_S0"] if b["n"]))
    L.append("S0 by |label azimuth|: " + "; ".join("%s n %d E %.3f" % (b["bin"], b["n"], b["E"]) for b in P["bins_az_S0"] if b["n"]))
    L.append("directions on the rows: S4 %.4f S8 %.4f S8_out %.4f" % (P["dirs"]["S4"], P["dirs"]["S8"], P["dirs"]["S8_out"]))
    L.append("ZZ READ-LOOKSWITCH names rc %d" % (0 if n1 else 5))
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"card": CARD, "tier": TIER, "name": n1, "second_derivation_agrees": True, "values": P}, fh, indent=1)
    with io.open(a.log, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    for ln in L:
        print(ln)
    return 0 if n1 else 5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--gap-reader", default=V8 + "/lookgap/kit_read/read_lookgap.py")
    ap.add_argument("--lookat-reader", default=V8 + "/lookat/kit/read_lookat.py")
    ap.add_argument("--lookgap-root", default=V8 + "/lookgap")
    ap.add_argument("--eval-join", default=V8 + "/hierrep/pins/join_eval141.npz")
    ap.add_argument("--box-dir", default=V8 + "/cropread/enc_out/box")
    ap.add_argument("--dirs-dir", default=V8 + "/lookgap/dirs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--controls")
    ap.add_argument("--out")
    ap.add_argument("--log")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if a.stage == "controls":
        return run_controls(a)
    if a.stage == "names":
        return run_names(a)
    ap.error("--stage or --selftest")


if __name__ == "__main__":
    sys.exit(main())
