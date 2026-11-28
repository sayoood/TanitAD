"""read_boxfuse_loc.py -- card PREREG_V8F_PROOF_ADD130 (BOXFUSE-T0-LOCFIT): the joined read of card ADD129 under the locator's further fits LOC_s1 and LOC_s2.
Tier T0: recorded frames, no rollout; a diagnostic of a perception read and not driving performance. CPU only; no fit. The tool of card ADD129 (read_boxfuse.py)
is imported unchanged (identity by sha256) and supplies arms, gate, values and the controls f1 to f5 (the card's g1 to g5).

Modes:
  controls  one fit (--locator-fit LOC_s1|LOC_s2, its own --out-dir): g0 and g6 are stored under the key f0, g1..g5 under f1..f5
  names     one fit: refuses unless this fit's controls AND the other fit's controls (--other-controls) say all pass; writes the values, prints no name
  summary   <out_s1> <out_s2> <out_dir> <FUSE_RESULT.json of card ADD129>: the card's two names from the two result files
Extra arguments: --add128-controls, --add128-result, --add129-result, --other-controls."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD130"
CARD_129 = "PREREG_V8F_PROOF_ADD129"
PRE = "BOXFUSE-T0-LOCFIT"
RF_PREFIX = "6b25176e0bff7442"
FITS = ("LOC_s1", "LOC_s2")
BAR_R = 0.90
IDENT_TOL = 1e-6
LIMITS = ("Three fits of one locator on the same train clips, one detector, the same 141 evaluation clips read a third time; the interval per fit answers what another "
          "draw of clips would say with the fits held fixed.")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_fuse():
    m = sys.modules.get("read_boxfuse")
    if m is not None and getattr(m, "__file__", None) and os.path.isfile(m.__file__):
        if not _sha(m.__file__).startswith(RF_PREFIX):
            raise SystemExit("REFUSED: the loaded read_boxfuse.py is not the tool of row 198 (%s)" % RF_PREFIX)
        return m
    for p in (os.environ.get("BOXFUSE_READER"), "/workspace/v8proof/boxread/kit_f/read_boxfuse.py", "F:/TanitAD-scratch/v8proof/boxread/fuse/read_boxfuse.py"):
        if p and os.path.isfile(p):
            hx = _sha(p)
            if not hx.startswith(RF_PREFIX):
                raise SystemExit("REFUSED: %s has sha256 %s..., expected %s" % (p, hx[:16], RF_PREFIX))
            spec = importlib.util.spec_from_file_location("read_boxfuse", p)
            m = importlib.util.module_from_spec(spec)
            sys.modules["read_boxfuse"] = m
            spec.loader.exec_module(m)
            return m
    raise SystemExit("REFUSED: read_boxfuse.py not found (set BOXFUSE_READER)")


RF = load_fuse()
RB, BS = RF.RB, RF.BS
ORIG = {k: getattr(RF, k) for k in ("CARD", "LIMITS", "ctl_f0", "input_hashes", "derive_names")}


def ctl_g0(a, W, gaps):
    """the card's g0 and g6, stored under the key f0"""
    bad = []
    if a.locator_fit not in FITS:
        bad.append("the locator's fit is not one of %s" % ",".join(FITS))
    cj = BS.try_json(a.add128_controls) if a.add128_controls else None
    rj = BS.try_json(a.add128_result) if a.add128_result else None
    flags, g6 = {}, {}
    if not isinstance(cj, dict) or cj.get("card") != RF.CARD_128 or cj.get("all_pass") is not True or not isinstance(cj.get("inputs"), dict):
        bad.append("no all-pass controls file of card ADD128 given")
    else:
        now, was = RB.input_hashes(a), cj["inputs"]
        skip = tuple(RF.SKIP_INPUT_KEYS) + ("locator", "locator_meta")
        diff = sorted(k for k in was if k not in skip and not k.startswith("tests_") and now.get(k) != was.get(k))
        if diff:
            bad.append("inputs differ from those of card ADD128: %s" % ",".join(diff[:5]))
        if now.get("locator") is None or now.get("locator") == was.get("locator"):
            bad.append("the locator's file is missing or is that of LOC_s0")
    if not isinstance(rj, dict) or rj.get("card") != RF.CARD_128 or not isinstance(rj.get("measures"), dict):
        bad.append("no result file of card ADD128 given")
    else:
        E = rj["measures"].get("E") or {}
        for k in RF.ARM_D:
            e = float(np.abs(gaps[k] - W["labg"]).mean())
            flags[k] = bool(k in E and abs(e - float(E[k])) <= IDENT_TOL)
            if not flags[k]:
                bad.append("%s: its error is not the banked value of card ADD128 within %g" % (k, IDENT_TOL))
    lp = RB.locator_path(a)
    meta = BS.try_json(os.path.splitext(lp)[0] + "_meta.json")
    hs = RB.sha_of(lp)
    if not isinstance(meta, dict) or hs is None or str(meta.get("pred_sha256", "")).lower() != hs:
        bad.append("the locator's file does not hash to its banked record of card ADD107")
    r9 = BS.try_json(a.add129_result) if a.add129_result else None
    if not isinstance(r9, dict) or r9.get("card") != CARD_129 or not isinstance(r9.get("measures"), dict):
        bad.append("no result file of card ADD129 given")
    else:
        a0 = copy.copy(a)
        a0.locator_fit = "LOC_s0"
        W0 = RF.load_world(a0)
        gaps0, _ = RF.arms_of(W0)
        E9 = r9["measures"].get("E") or {}
        for k in RF.fits("FG"):
            e = float(np.abs(gaps0[k] - W0["labg"]).mean())
            g6[k] = bool(k in E9 and abs(e - float(E9[k])) <= IDENT_TOL)
            if not g6[k]:
                bad.append("%s with LOC_s0: its error is not the banked value of row 198 within %g" % (k, IDENT_TOL))
    return RF._res(bad, arm_d_identical_to_card_add128=flags, g6_fg_of_row_198_reproduced=g6, locator_fit=a.locator_fit, locator_sha256_prefix=(hs or "")[:16])


def input_hashes_loc(a):
    out = ORIG["input_hashes"](a)
    out["read_boxfuse_loc.py"] = _sha(os.path.abspath(__file__))
    p = getattr(a, "add129_result", "")
    out["add129_result"] = _sha(p) if p and os.path.isfile(p) else None
    out["locator_fit"] = a.locator_fit
    return out


PATCH = {"CARD": CARD, "LIMITS": LIMITS, "ctl_f0": ctl_g0, "input_hashes": input_hashes_loc, "derive_names": lambda P: []}


def all_pass_file(path):
    cj = BS.try_json(path) or {}
    cs = cj.get("controls", {}) if isinstance(cj.get("controls"), dict) else {}
    return cj.get("card") == CARD and cj.get("all_pass") is True and sorted(cs) == sorted(RF.CONTROLS) and all(v.get("pass") is True for v in cs.values()), cj


def fit_main(argv):
    ap = RB.build_parser()
    for k in ("--add128-controls", "--add128-result", "--add129-result", "--other-controls"):
        ap.add_argument(k, default="")
    a = ap.parse_args(argv)
    for k, v in PATCH.items():
        setattr(RF, k, v)
    try:
        if a.mode == "controls":
            return RF.controls_main(a)
        ok, oj = all_pass_file(a.other_controls) if a.other_controls else (False, {})
        mine = (BS.try_json(os.path.join(a.out_dir, "FUSE_CONTROLS.json")) or {}).get("inputs", {}).get("locator_fit")
        if not ok or oj.get("inputs", {}).get("locator_fit") not in FITS or oj["inputs"]["locator_fit"] == mine:
            print("REFUSED: the controls file of the other fit is not all-pass: no value is opened")
            return 3
        return RF.names_main(a)
    finally:
        for k, v in ORIG.items():
            setattr(RF, k, v)


def derive_names(per_fit):
    """per_fit: {fit: {"R": float, "iv": [lo, hi]}} for the two further fits -> the card's two names"""
    rep = all(per_fit[f]["iv"] is not None and per_fit[f]["iv"][1] < 1.0 for f in FITS)
    bar = rep and all(per_fit[f]["R"] is not None and per_fit[f]["R"] <= BAR_R for f in FITS)
    return [PRE + ("-REPLICATED" if rep else "-NOT-REPLICATED"), PRE + ("-BAR-3-OF-3" if bar else "-BAR-NOT-3-OF-3")]


def summary_main(argv):
    if len(argv) != 4:
        print("REFUSED: usage: summary <out_s1> <out_s2> <out_dir> <FUSE_RESULT.json of card ADD129>")
        return 3
    outs, out_dir, r9p = dict(zip(FITS, argv[:2])), argv[2], argv[3]
    per, rows = {}, {}
    for f in FITS:
        cp, rp = os.path.join(outs[f], "FUSE_CONTROLS.json"), os.path.join(outs[f], "FUSE_RESULT.json")
        ok, cj = all_pass_file(cp)
        rj = BS.try_json(rp) or {}
        if not ok or cj.get("inputs", {}).get("locator_fit") != f or rj.get("card") != CARD or rj.get("controls_sha256") != _sha(cp):
            print("REFUSED: %s has no all-pass controls with its result: no name is opened" % f)
            return 3
        m, b = rj["measures"], rj["beside"]
        per[f] = {"R": m["R"]["FG"], "iv": m["iv"]["FG"]}
        rows[f] = {"R_FG": m["R"]["FG"], "iv_FG": m["iv"]["FG"], "mean_E_FG": m["mean_E"]["FG"], "mean_E_D": m["mean_E"]["D"], "R_BW": m["R"]["BW"], "iv_BW": m["iv"]["BW"],
                   "share_same_pick": b["picks"]["share_same"], "gate_share_of_rows": b["gate"]["share_of_rows"], "gate_share_same_pick": b["gate"]["share_same_pick"],
                   "gate_share_other_pick": b["gate"]["share_other_pick"], "far_signed_FG": m["far"]["signed"]["FG"], "result_sha256": _sha(rp), "controls_sha256": _sha(cp)}
    r9 = BS.try_json(r9p) or {}
    if r9.get("card") != CARD_129:
        print("REFUSED: no result file of card ADD129")
        return 3
    rows["LOC_s0"] = {"R_FG": r9["measures"]["R"]["FG"], "iv_FG": r9["measures"]["iv"]["FG"], "mean_E_FG": r9["measures"]["mean_E"]["FG"], "R_BW": r9["measures"]["R"]["BW"],
                      "share_same_pick": r9["beside"]["picks"]["share_same"], "row": 198}
    names = derive_names(per)
    rs = [rows[f]["R_FG"] for f in ("LOC_s0",) + FITS]
    res = {"card": CARD, "tier": RB.TIER, "names": names, "per_fit": rows, "R_FG_three_fits": rs, "R_FG_spread": [min(rs), max(rs)], "bar_R": BAR_R}
    os.makedirs(out_dir, exist_ok=True)
    BS.wt(RB.fresh(os.path.join(out_dir, "LOCFIT_RESULT.json")), json.dumps(res, indent=1, default=BS._jdefault))
    L = ["# BOXFUSE-T0-LOCFIT verdict (card %s)" % CARD, "", "Tier: " + RB.TIER, "", "## Families", ""] + ["* " + x for x in RB.FAMILY_LINES] + ["", "## Names", ""]
    L += ["* `%s`" % n for n in names] + ["", "## The joined read FG against arm D, per fit of the locator (3,841 frames of 141 clips)", ""]
    for f in ("LOC_s0",) + FITS:
        r = rows[f]
        L.append("* %s%s: R(FG) %s, interval %s; box read alone R %s; the locator's pick is the label's pick on %s of the frames with both picks"
                 % (f, " (row 198)" if f == "LOC_s0" else "", RF.fv(r["R_FG"], 4), RF.fiv(r["iv_FG"]), RF.fv(r["R_BW"], 4), RF.fv(r["share_same_pick"], 4)))
    for f in FITS:
        r = rows[f]
        L.append("* %s beside: FG %s m against arm D %s m; the gate passes on %s of the rows, on %s of the frames with the label's pick and %s of the others; far leads signed %s m"
                 % (f, RF.fv(r["mean_E_FG"], 4), RF.fv(r["mean_E_D"], 4), RF.fv(r["gate_share_of_rows"], 4), RF.fv(r["gate_share_same_pick"], 4), RF.fv(r["gate_share_other_pick"], 4), RF.fv(r["far_signed_FG"])))
    L += ["* R(FG) over the three fits: %s to %s; bar of the first fit %.2f" % (RF.fv(min(rs), 4), RF.fv(max(rs), 4), BAR_R), "", "## Limits of the card", "", LIMITS, ""]
    BS.wt(RB.fresh(os.path.join(out_dir, "LOCFIT_VERDICT.md")), "\n".join(L))
    for n in names:
        print("ZZ LOCFIT NAME %s" % n)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "summary":
        return summary_main(argv[1:])
    return fit_main(argv)


if __name__ == "__main__":
    sys.exit(main())
