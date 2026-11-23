"""read_boxfuse.py -- card PREREG_V8F_PROOF_ADD129 (BOXFUSE-T0): the box read of cards ADD127/ADD128 joined with the token head (arm D) under an agreement gate.
Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. CPU only; no fit; the reader of card ADD127 is imported
unchanged (identity by sha256) and supplies the picks, the map and the estimator.

Arms on the 3,841 evaluation rows, each against the three fits of arm D:
  FG  picked frame (locator's pick, vision only) whose box gap and arm D's gap differ by at most GATE_FACTOR: geometric mean; else arm D's gap
  FM  geometric mean on every picked frame (beside)        FL  as FG with the label's pick (bound, not a candidate)
Modes: controls (f0..f5; prints pass/fail, counts and the gate shares; no error of an arm) and names (refuses unless the controls file says all pass).
Extra arguments beside those of read_boxread.py: --add128-controls, --add128-result (the files of card ADD128's read)."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD129"
CARD_128 = "PREREG_V8F_PROOF_ADD128"
PRE = "BOXFUSE-T0"
RB_PREFIX = "aaac070b11e827c0"
GATE_FACTOR = 1.5
F1_MAX_RATIO = 0.75
F3_MIN_SHARE = 0.30
IDENT_TOL = 1e-6
BAR_R = 0.90
FAR_HALF = 0.5
N_BOOT, BOOT_SEED, PERM_SEED = 2000, 129, 129
CONTROLS = tuple("f%d" % i for i in range(6))
SKIP_INPUT_KEYS = ("read_boxread_c.py", "prior_controls")
LIMITS = ("One detector, one set of weights, one fit of the locator, 141 evaluation clips used a second time (row 197 read the same clips); two parameters fixed without a fit; "
          "the interval answers what another draw of clips would say, not what another fit, another detector or another choice of the two parameters would say.")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_reader():
    m = sys.modules.get("read_boxread")
    if m is not None and getattr(m, "__file__", None) and os.path.isfile(m.__file__):
        if not _sha(m.__file__).startswith(RB_PREFIX):
            raise SystemExit("REFUSED: the loaded read_boxread.py is not the reader of card ADD127 (%s)" % RB_PREFIX)
        return m
    for p in (os.environ.get("BOXREAD_READER"), "/workspace/v8proof/boxread/kit/read/read_boxread.py", "F:/TanitAD-scratch/v8proof/boxread/read/read_boxread.py"):
        if p and os.path.isfile(p):
            hx = _sha(p)
            if not hx.startswith(RB_PREFIX):
                raise SystemExit("REFUSED: %s has sha256 %s..., expected %s" % (p, hx[:16], RB_PREFIX))
            d = os.path.dirname(os.path.abspath(p))
            if d not in sys.path:
                sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location("read_boxread", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["read_boxread"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: read_boxread.py not found (set BOXREAD_READER)")


RB = load_reader()
BS = RB.BS
ARM_D = RB.ARM_D


# ============================================================================================ the reads
def load_world(a):
    """the picks and gaps of card ADD128 on the evaluation rows (the calls of the imported reader's names stage). No label enters gw, pw or the fits of arm D."""
    J = BS.load_npz(a.eval_join)
    L = {k: BS.lay_dump(J, BS.load_npz(p)) for k, p in RB.dump_paths(a).items()}
    store = RB.DetStore(a.det_dir)
    tclips = BS.ids_of(BS.parse_kv(a.list_file)["train_list"])
    Gw = RB.build_map(RB.train_reads(a.box_dir, store, tclips, "wide"))
    if Gw is None:
        raise ValueError("no wide map could be fitted")
    Pz = BS.lay_pred(J, RB.load_locator(RB.locator_path(a)))
    cache = {}

    def det_for(cam, c):
        if cam != "wide":
            return None
        if c not in cache:
            cache[c] = store.get("wide", c)
        return cache[c]
    wide = RB.eval_pick_locator(J, Pz["az"], Pz["el"], det_for, lambda c: None, False)["wide"]
    lab = RB.eval_pick_label(J, a.box_dir, det_for)
    n = np.asarray(J["n"], np.int64)
    rows = np.flatnonzero(BS.t3_rows(L))
    W = {"nc": len(n), "ci": np.repeat(np.arange(len(n)), n)[rows], "labg": L["D_0"]["lab_gap"][rows], "gD": {k: np.asarray(L[k]["gap"][rows], np.float64) for k in ARM_D}}

    def gmap(rd):
        p = rd["pick"][rows]
        g = np.full(len(rows), np.nan)
        if p.any():
            g[p] = Gw.predict(rd["w"][rows][p], rd["h"][rows][p])
        return g, p & np.isfinite(g) & (g > 0)
    W["gw"], W["pw"] = gmap(wide)
    W["gl"], W["pl"] = gmap(lab)
    W["same_pick"] = W["pw"] & W["pl"] & (wide["w"][rows] == lab["w"][rows]) & (wide["h"][rows] == lab["h"][rows])
    return W


def gate_of(gbox, pick, gd, factor=None):
    f = GATE_FACTOR if factor is None else factor
    ok = pick & np.isfinite(gd) & (gd > 0)
    out = np.zeros(len(gd), bool)
    out[ok] = np.abs(np.log(gbox[ok]) - np.log(gd[ok])) <= math.log(f)
    return out


def arms_of(W):
    """-> gaps {name: [rows]}, gates {fit index: bool [rows]} (the gate of FG)"""
    gaps, gates = {}, {}
    for i, k in enumerate(ARM_D):
        gd = W["gD"][k]
        gaps[k] = gd
        gw_ok = W["pw"] & np.isfinite(gd) & (gd > 0)
        gl_ok = W["pl"] & np.isfinite(gd) & (gd > 0)
        mean_w = np.sqrt(np.where(gw_ok, W["gw"], 1.0) * np.where(gw_ok, gd, 1.0))
        mean_l = np.sqrt(np.where(gl_ok, W["gl"], 1.0) * np.where(gl_ok, gd, 1.0))
        gates[i] = gate_of(W["gw"], W["pw"], gd)
        gaps["BW_%d" % i] = np.where(W["pw"], W["gw"], gd)
        gaps["FG_%d" % i] = np.where(gates[i], mean_w, gd)
        gaps["FM_%d" % i] = np.where(gw_ok, mean_w, gd)
        gaps["FL_%d" % i] = np.where(gate_of(W["gl"], W["pl"], gd), mean_l, gd)
    return gaps, gates


def fits(X):
    return list(ARM_D) if X == "D" else ["%s_%d" % (X, i) for i in range(3)]


# ============================================================================================ controls
def _res(bad, **kw):
    d = {"pass": not bad, "failures": list(bad)}
    d.update(kw)
    return d


def input_hashes(a):
    out = RB.input_hashes(a)
    out["read_boxfuse.py"] = _sha(os.path.abspath(__file__))
    for k in ("add128_controls", "add128_result"):
        p = getattr(a, k, "")
        out[k] = _sha(p) if p and os.path.isfile(p) else None
    return out


def ctl_f0(a, W, gaps):
    bad = []
    cj = BS.try_json(a.add128_controls) if a.add128_controls else None
    rj = BS.try_json(a.add128_result) if a.add128_result else None
    if not isinstance(cj, dict) or cj.get("card") != CARD_128 or cj.get("all_pass") is not True:
        bad.append("no all-pass controls file of card ADD128 given")
    else:
        now, was = RB.input_hashes(a), cj.get("inputs") or {}
        diff = sorted(k for k in was if k not in SKIP_INPUT_KEYS and not k.startswith("tests_") and now.get(k) != was.get(k))
        if diff:
            bad.append("inputs differ from those of card ADD128: %s" % ",".join(diff[:5]))
    flags = {}
    if not isinstance(rj, dict) or rj.get("card") != CARD_128 or not isinstance(rj.get("measures"), dict):
        bad.append("no result file of card ADD128 given")
    else:
        if isinstance(cj, dict) and rj.get("controls_sha256") != _sha(a.add128_controls):
            bad.append("the result file of card ADD128 does not belong to its controls file")
        E = rj["measures"].get("E") or {}
        for k in fits("D") + fits("BW"):
            e = float(np.abs(gaps[k] - W["labg"]).mean())
            flags[k] = bool(k in E and E[k] is not None and abs(e - float(E[k])) <= IDENT_TOL)
            if not flags[k]:
                bad.append("%s: its error is not the banked value of card ADD128 within %g" % (k, IDENT_TOL))
    return _res(bad, identical_to_card_add128=flags)


def gate_shares(W, gates):
    npick = int(W["pw"].sum())
    in_place = float(np.mean([gates[i].sum() / npick for i in range(3)])) if npick else None
    rng = np.random.RandomState(PERM_SEED)
    perm = []
    for k in ARM_D:
        gd = W["gD"][k][rng.permutation(len(W["labg"]))]
        perm.append(gate_of(W["gw"], W["pw"], gd).sum() / npick if npick else np.nan)
    return npick, in_place, (float(np.mean(perm)) if npick else None)


def ctl_f1(a, W, gates):
    bad = []
    npick, sp, sm = gate_shares(W, gates)
    if not npick or sp is None or sp <= 0:
        bad.append("no picked frame passes the gate")
    elif sm > F1_MAX_RATIO * sp:
        bad.append("with arm D's gaps permuted the gate passes on %.4f of the picked frames, above %.2f of %.4f" % (sm, F1_MAX_RATIO, sp))
    return _res(bad, n_picked=npick, share_pass=sp, share_pass_permuted=sm)


def ctl_f2(a):
    a2 = copy.copy(a)
    a2.locator_sha256 = ""
    r = RB.ctl_b4(a2, {"scratch": []})
    return _res(list(r["failures"]), locator_sha256_prefix=r.get("locator_sha256_prefix"))


def ctl_f3(a, W, gates):
    share = float(np.mean([gates[i].mean() for i in range(3)])) if len(W["labg"]) else None
    bad = [] if (share is not None and share >= F3_MIN_SHARE) else ["the gate passes on %s of the evaluation frames (bar %.2f)" % ("none" if share is None else "%.4f" % share, F3_MIN_SHARE)]
    return _res(bad, share=share, n_rows=int(len(W["labg"])))


def ctl_f4(a):
    before = BS.parse_kv(a.test_before, float)
    tests = {k: {"text": BS.read_text(p), "mtime": os.path.getmtime(p) if os.path.isfile(p) else None, "before": before.get(k)} for k, p in BS.parse_kv(a.test_out).items()}
    r = RB.ctl_b6(a, {"tests": tests})
    bad = list(r["failures"])
    if "fuse" not in tests:
        bad.append("no test output named fuse")
    return _res(bad, n_test_outputs=len(tests))


def ctl_f5(a, W, gaps):
    bad = []
    nrow, nclip = int(len(W["labg"])), int(len(np.unique(W["ci"])))
    if a.expect_rows and nrow != int(a.expect_rows):
        bad.append("%d rows, expected %d" % (nrow, int(a.expect_rows)))
    if a.expect_clips and W["nc"] != int(a.expect_clips):
        bad.append("%d clips in the join, expected %d" % (W["nc"], int(a.expect_clips)))
    for k, g in gaps.items():
        if len(g) != nrow or not np.isfinite(g).all():
            bad.append("arm %s does not cover the same finite rows" % k)
    return _res(bad, n_rows=nrow, n_clips=int(W["nc"]), n_clips_with_rows=nclip)


def controls_main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        W = load_world(a)
        gaps, gates = arms_of(W)
        ctl = {"f0": ctl_f0(a, W, gaps), "f1": ctl_f1(a, W, gates), "f2": ctl_f2(a), "f3": ctl_f3(a, W, gates), "f4": ctl_f4(a), "f5": ctl_f5(a, W, gaps)}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as ex:
        why = "the controls could not be computed: %s: %s" % (type(ex).__name__, str(ex)[:60])
        ctl = {k: _res([why]) for k in CONTROLS}
    res = {"card": CARD, "tier": RB.TIER, "controls": ctl, "all_pass": bool(sorted(ctl) == sorted(CONTROLS) and all(v["pass"] for v in ctl.values()))}
    try:
        res["inputs"] = input_hashes(a)
    except (OSError, ValueError, TypeError) as ex:
        res["inputs"] = {"error": type(ex).__name__}
    BS.wt(RB.fresh(os.path.join(a.out_dir, "FUSE_CONTROLS.json")), json.dumps(res, indent=1, default=BS._jdefault))
    for k in CONTROLS:
        c = ctl[k]
        print("ctl %s PASS" % k if c["pass"] else "ctl %s FAIL %s" % (k, (c["failures"][0] if c["failures"] else "")[:110]))
    f1, f3 = ctl["f1"], ctl["f3"]
    if f1.get("share_pass") is not None:
        print("f1 shares %.4f %.4f" % (f1["share_pass"], f1["share_pass_permuted"]))
    if f3.get("share") is not None:
        print("f3 share %.4f" % f3["share"])
    print("ZZ BOXFUSE CONTROLS ALL-PASS" if res["all_pass"] else "ZZ BOXFUSE CONTROLS FAILED")
    return 0 if res["all_pass"] else 2


# ============================================================================================ names
def name_ratio(iv, R):
    if iv is None or R is None:
        return None
    if iv[1] < 1.0:
        return PRE + ("-BETTER" if R <= BAR_R else "-SMALL-GAIN")
    if iv[0] > 1.0:
        return PRE + "-WORSE"
    return PRE + "-NOT-SEPARATED"


def compute_values(W, n_boot=N_BOOT, seed=BOOT_SEED):
    gaps, gates = arms_of(W)
    labg, ci, nc = W["labg"], W["ci"], W["nc"]
    absk = {k: np.abs(g - labg) for k, g in gaps.items()}
    sgn = {k: g - labg for k, g in gaps.items()}
    sums = {k: np.bincount(ci, weights=absk[k], minlength=nc) for k in gaps}

    def msum(X):
        return sum(sums[k] for k in fits(X)) / 3.0
    arms = ("D", "BW", "FG", "FM", "FL")
    P = {"n_rows": int(len(labg)), "n_clips": int(nc), "n_boot": n_boot, "seed": seed, "gate_factor": GATE_FACTOR}
    P["E"] = {k: BS.fnum(absk[k].mean()) for k in gaps}
    P["mean_E"] = {X: BS.fnum(np.mean([P["E"][k] for k in fits(X)])) for X in arms}
    P["R"] = {X: BS.fnum(P["mean_E"][X] / P["mean_E"]["D"]) for X in arms if X != "D"}
    P["iv"] = {X: BS.boot_ratio(msum(X), msum("D"), n_boot, seed) for X in arms if X != "D"}
    P["R"]["FG_over_BW"] = BS.fnum(P["mean_E"]["FG"] / P["mean_E"]["BW"])
    P["iv"]["FG_over_BW"] = BS.boot_ratio(msum("FG"), msum("BW"), n_boot, seed)
    far_m = BS.gap_bin_mask(labg, 40, 60)
    far_n = np.bincount(ci[far_m], minlength=nc).astype(np.float64)
    far_s = {k: np.bincount(ci[far_m], weights=sgn[k][far_m], minlength=nc) for k in gaps}
    nfar = float(far_n.sum())
    P["far"] = {"n": int(nfar), "n_clips": int((far_n > 0).sum()),
                "signed": {X: (float(np.mean([far_s[k].sum() / nfar for k in fits(X)])) if nfar > 0 else None) for X in arms},
                "iv_diff_FG_minus_D": BS.boot_diff_over(sum(far_s[k] for k in fits("FG")) / 3.0, sum(far_s[k] for k in fits("D")) / 3.0, far_n, n_boot, seed) if nfar > 0 else None}
    both = W["pw"] & W["pl"]
    same, other = W["same_pick"], both & ~W["same_pick"]
    eb = np.abs(W["gw"] - labg)
    g_any = np.mean([gates[i] for i in range(3)], axis=0)
    bs = {"picks": {"n_both": int(both.sum()), "n_same": int(same.sum()), "share_same": BS.fnum(same.sum() / both.sum()) if both.any() else None,
                    "E_BW_same": BS.fnum(eb[same].mean()) if same.any() else None, "E_BW_other": BS.fnum(eb[other].mean()) if other.any() else None,
                    "E_D_same": BS.fnum(np.mean([absk[k][same].mean() for k in ARM_D])) if same.any() else None,
                    "E_D_other": BS.fnum(np.mean([absk[k][other].mean() for k in ARM_D])) if other.any() else None},
          "gate": {"share_of_rows": BS.fnum(g_any.mean()), "share_of_picked": BS.fnum(g_any[W["pw"]].mean()) if W["pw"].any() else None,
                   "share_same_pick": BS.fnum(g_any[same].mean()) if same.any() else None, "share_other_pick": BS.fnum(g_any[other].mean()) if other.any() else None,
                   "E_FG_on_gated": BS.fnum(np.mean([absk["FG_%d" % i][gates[i]].mean() for i in range(3) if gates[i].any()])) if any(gates[i].any() for i in range(3)) else None,
                   "E_D_on_gated": BS.fnum(np.mean([absk[ARM_D[i]][gates[i]].mean() for i in range(3) if gates[i].any()])) if any(gates[i].any() for i in range(3)) else None}}
    P["beside"] = bs
    P["bins"] = {}
    for X in arms:
        out = []
        for lo, hi in RB.GAP_BINS:
            m = BS.gap_bin_mask(labg, lo, hi)
            row = {"bin": "%d-%d m" % (lo, hi), "n": int(m.sum())}
            if m.any():
                row.update({"E": BS.fnum(np.mean([absk[k][m].mean() for k in fits(X)])), "signed": BS.fnum(np.mean([sgn[k][m].mean() for k in fits(X)])), "gate_share": BS.fnum(g_any[m].mean())})
            out.append(row)
        P["bins"][X] = out
    P["sums"] = {"abs": {k: sums[k].tolist() for k in gaps}, "far_n": far_n.tolist()}
    return P


def derive_names(P):
    names = [name_ratio(P["iv"]["FG"], P["R"]["FG"])]
    s, iv = P["far"]["signed"], P["far"]["iv_diff_FG_minus_D"]
    less = s["FG"] is not None and s["D"] is not None and iv is not None and iv[0] > 0.0 and abs(s["FG"]) <= FAR_HALF * abs(s["D"])
    names.append(PRE + ("-FAR-LEADS-LESS-SHORT" if less else "-FAR-LEADS-STAY-SHORT"))
    return [n for n in names if n]


def fv(x, d=3):
    return "None" if x is None else ("%." + str(d) + "f") % x


def fiv(iv):
    return "[%.4f, %.4f]" % (iv[0], iv[1]) if iv else "None"


def verdict_lines(P, names):
    L = ["# BOXFUSE-T0 verdict (card %s)" % CARD, "", "Tier: " + RB.TIER, "", "## Families", ""] + ["* " + x for x in RB.FAMILY_LINES] + ["", "## Names", ""]
    L += ["* `%s`" % n for n in names]
    L += ["", "## Measure: mean absolute gap error on %d evaluation frames of %d clips (m); gate factor %.2f" % (P["n_rows"], P["n_clips"], P["gate_factor"]), ""]
    L.append("* arm D: %s; mean %s" % ("; ".join("%s %s" % (k, fv(P["E"][k])) for k in ARM_D), fv(P["mean_E"]["D"], 4)))
    for X, what in (("BW", "box read, locator's pick (row 197)"), ("FG", "gated geometric mean, locator's pick, vision only"), ("FM", "geometric mean with no gate (beside)"),
                    ("FL", "gated geometric mean, label's pick (bound, not a candidate)")):
        L.append("* arm %s (%s): %s; mean %s; R = %s, interval %s" % (X, what, "; ".join(fv(P["E"][k]) for k in fits(X)), fv(P["mean_E"][X], 4), fv(P["R"][X], 4), fiv(P["iv"][X])))
    L.append("* FG divided by BW: %s, interval %s" % (fv(P["R"]["FG_over_BW"], 4), fiv(P["iv"]["FG_over_BW"])))
    L.append("* interval: %d draws, seed %d; two-sided 95 %% percentile, cluster bootstrap over the evaluation clips, fits held fixed" % (P["n_boot"], P["seed"]))
    f = P["far"]
    L += ["", "## Far leads (label gap 40 to 60 m): n %d in %d clips" % (f["n"], f["n_clips"]), "",
          "* mean signed error (read minus label), mean of three fits: " + ", ".join("%s %s" % (X, fv(f["signed"][X])) for X in ("D", "BW", "FG", "FM", "FL")),
          "* paired difference FG minus D, interval %s" % fiv(f["iv_diff_FG_minus_D"]), "", "## Beside, naming nothing", ""]
    p, g = P["beside"]["picks"], P["beside"]["gate"]
    L.append("* frames with both picks %d; the locator's pick is the label's pick on %d (share %s); error of the box gap on these %s m (arm D %s m), on the others %s m (arm D %s m)"
             % (p["n_both"], p["n_same"], fv(p["share_same"], 4), fv(p["E_BW_same"]), fv(p["E_D_same"]), fv(p["E_BW_other"]), fv(p["E_D_other"])))
    L.append("* the gate passes on %s of the rows and %s of the picked frames; on %s of the frames whose pick is the label's and on %s of the others; on the gated frames FG %s m, arm D %s m"
             % (fv(g["share_of_rows"], 4), fv(g["share_of_picked"], 4), fv(g["share_same_pick"], 4), fv(g["share_other_pick"], 4), fv(g["E_FG_on_gated"]), fv(g["E_D_on_gated"])))
    for X in ("D", "BW", "FG", "FL"):
        L.append("* by label gap, arm %s: " % X + "; ".join("%s n %d E %s signed %s gate %s" % (r["bin"], r["n"], fv(r.get("E")), fv(r.get("signed")), fv(r.get("gate_share"), 4)) for r in P["bins"][X]))
    L += ["", "## Limits of the card", "", LIMITS, ""]
    return L


def names_main(a):
    cpath = os.path.join(a.out_dir, "FUSE_CONTROLS.json")
    cj = BS.try_json(cpath) or {}
    cs = cj.get("controls", {}) if isinstance(cj.get("controls"), dict) else {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or sorted(cs) != sorted(CONTROLS) or any(v.get("pass") is not True for v in cs.values()):
        print("REFUSED: the controls file in the out dir is not all-pass with f0 to f5 present: no name and no value is opened")
        return 3
    now, was = input_hashes(a), cj.get("inputs") or {}
    diff = sorted(k for k in set(now) | set(was) if now.get(k) != was.get(k))
    if diff:
        print("REFUSED: inputs differ from those of the controls: %s" % diff[:6])
        return 3
    P = compute_values(load_world(a), a.B)
    names = derive_names(P)
    res = {"card": CARD, "tier": RB.TIER, "names": names, "measures": {k: v for k, v in P.items() if k not in ("beside", "sums")}, "beside": P["beside"],
           "input_hashes": now, "controls_sha256": _sha(cpath), "sums": P["sums"]}
    BS.wt(RB.fresh(os.path.join(a.out_dir, "FUSE_RESULT.json")), json.dumps(res, indent=1, default=BS._jdefault))
    BS.wt(RB.fresh(os.path.join(a.out_dir, "FUSE_VERDICT.md")), "\n".join(verdict_lines(P, names)))
    for n in names:
        print("ZZ BOXFUSE NAME %s" % n)
    return 0


def main(argv=None):
    ap = RB.build_parser()
    ap.add_argument("--add128-controls", default="")
    ap.add_argument("--add128-result", default="")
    a = ap.parse_args(sys.argv[1:] if argv is None else list(argv))
    return controls_main(a) if a.mode == "controls" else names_main(a)


if __name__ == "__main__":
    sys.exit(main())
