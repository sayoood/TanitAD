#!/usr/bin/env python
"""read_longroll10k_n3.py -- the read of record of card PREREG_V8F_PROOF_ADD70 by its note N3: a wrapper around the unedited reader read_longroll10k.py (same folder).

Three faults of the reader's controls were found at its first run on data (note N3, section 1); the reader printed and wrote no name, no level and no difference. This wrapper replaces
exactly those three controls and withholds the headway shares of the P = 60 grid; every other line of the reader runs as it is.
  l12  the tick files of R0LrCarry and R0LrFresh are compared on every key except the checkpoint scalar and the run name `tag` (tolerance 1e-6; a key in one file only fails)
  l11  every P = 60 run: two pre-roll segments per window and the expected number of segment rows (windows of the grid times kinds, as in the reader); the envelope counter of the
       pre-roll (clipped_windows_in_preroll) is reported, not judged
  P60  the headway reader's own controls are not required on the P = 60 grid (73 lead windows under its bar of 200); the headway shares of that grid are NOT reported: this wrapper removes
       them from the result file and from the printed beside block. The family cells and the strategic read of the P = 60 pairs stay as beside values if the family controls and the window
       identity pass on that grid.
The first result of each replaced control is kept under `as_first_read`. Usage: the arguments of read_longroll10k.py, unchanged; `--selftest-n3 <empty folder>` runs the planted-fault
checks of this wrapper on synthetic files written into that folder (nothing is removed by this tool).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import read_longroll10k as R  # noqa: E402

np = R.np
NOT_REPORTED = "not reported (note N3): the headway instrument is not qualified on the P = 60 grid (73 lead windows, its own bar is 200)"
SKIP_L12 = {"ckpt", "tag"}


def l12_n3(a):
    pc = os.path.join(R.tick_dir(a.pull, "R0LrCarry", 0), "ticks_%s_m10_causal_heads.npz" % R.key_of("R0LrCarry", 0))
    pf = os.path.join(R.tick_dir(a.pull, "R0LrFresh", 0), "ticks_%s_m10_causal_heads.npz" % R.key_of("R0LrFresh", 0))
    zc, zf = R.npz_load(pc), R.npz_load(pf)
    if zc is None or zf is None:
        return {"PASS": None, "why": "tick file missing"}
    per = {k: R.maxdiff(zc[k], zf[k]) for k in sorted(set(zc.files) & set(zf.files)) if k not in SKIP_L12}
    only = sorted(set(zc.files) ^ set(zf.files))
    mxx = max(per.values()) if per else float("inf")
    return {"largest_difference": mxx, "tolerance": R.TOL, "keys_compared": len(per), "keys_skipped": sorted(SKIP_L12), "keys_in_one_only": only,
            "worst_keys": [[k, per[k]] for k in sorted(per, key=lambda k: -per[k])[:5]], "PASS": bool(per and not only and mxx <= R.TOL)}


def l11_n3(a, keys_by_grid, first):
    out = {"runs": {}, "envelope_counter_reported_not_judged": {}}
    ok = []
    for run, k in R.NEW_RUN_LIST:
        form, ro, P, nav, g, ex, every, kap = R.ALL_RUNS[run]
        if P != 60:
            continue
        N = int(len(keys_by_grid[g]))
        key = R.key_of(run, k)
        td = R.tick_dir(a.pull, run, k)
        pl = R.jl(os.path.join(td, "preroll_log_%s.json" % key))
        po = R.npz_load(os.path.join(td, "preroll_offsets_%s.npz" % key))
        if pl is None or po is None:
            out["runs"][key] = {"problems": ["pre-roll log or offsets file missing"]}
            ok.append(False)
            continue
        kinds = sorted(pl.get("windows", {}))
        nseg = int(len(po["seg_ci"]))
        segs = pl.get("segments_per_window")
        prob = []
        if segs != 2:
            prob.append("segments per window %s" % segs)
        if not kinds or nseg != N * len(kinds):
            prob.append("segment rows %d, expected %d" % (nseg, N * len(kinds)))
        out["runs"][key] = {"problems": prob, "segments_per_window": segs, "seg_rows": nseg, "expected_seg_rows": N * len(kinds), "kinds": kinds}
        out["envelope_counter_reported_not_judged"][key] = pl.get("clipped_windows_in_preroll")
        ok.append(not prob)
    out["PASS"] = bool(ok and all(ok))
    out["as_first_read"] = first
    return out


def file_controls_n3(a, keys_by_grid, names, log):
    C = _orig_file_controls(a, keys_by_grid, names, log)
    first12, first11 = C.get("l12_recorder_frame"), C.get("l11_segments")
    n12 = l12_n3(a)
    n12["as_first_read"] = first12
    C["l12_recorder_frame"] = n12
    C["l11_segments"] = l11_n3(a, keys_by_grid, {"PASS": (first11 or {}).get("PASS"), "problems": {k: v.get("problems") for k, v in ((first11 or {}).get("runs") or {}).items() if v.get("problems")}})
    return C


def read_grid_n3(a, E, grid_name, keys, runs, spread_run, pairs, dod_defs, log):
    G = _orig_read_grid(a, E, grid_name, keys, runs, spread_run, pairs, dod_defs, log)
    if grid_name == "60" and isinstance(G.get("controls"), dict) and "family_q2_q3_q4" in G["controls"]:
        c = G["controls"]
        qc = c.get("family_q2_q3_q4")
        fam_ok = bool(isinstance(qc, dict) and qc and all(isinstance(v, dict) and v.get("PASS") is True for v in qc.values()))
        c["ALL_PASS_as_first_read"] = c.get("ALL_PASS")
        c["note_n3"] = "headway controls not required on this grid; headway shares of this grid are not reported"
        c["ALL_PASS"] = bool(c.get("same_windows_every_dump") is True and fam_ok)
    return G


def strip_p60_headway(res):
    b = res.get("beside")
    if isinstance(b, dict):
        for nm, p in (b.get("P60_pairs") or {}).items():
            if isinstance(p, dict) and "headway" in p:
                p["headway"] = NOT_REPORTED
        if "levels_P60" in b:
            b["levels_P60"] = NOT_REPORTED
        if isinstance(b.get("spread"), dict) and "P60" in b["spread"]:
            b["spread"]["P60"] = NOT_REPORTED
    d = res.get("grid_reads_detail_controls_only")
    if isinstance(d, dict) and isinstance(d.get("60"), dict) and isinstance(d["60"].get("spread"), dict) and "headway" in d["60"]["spread"]:
        d["60"]["spread"]["headway"] = NOT_REPORTED


def selftest_n3(folder):
    """planted-fault checks of the three replaced controls and of the removal of the P = 60 headway shares; synthetic files only"""
    if os.path.exists(folder) and os.listdir(folder):
        print("ZZ SELFTEST-N3 ok: False (folder not empty)")
        return 1
    items = []

    class A:                                                                                   # noqa: D401
        pass

    def ticks(root, run, tag, bump=0.0, extra=False):
        td = R.tick_dir(root, run, 0)
        os.makedirs(td, exist_ok=True)
        d = {"tag": np.asarray(tag), "ckpt": np.asarray("x"), "P": np.arange(6.0).reshape(3, 2), "ci": np.arange(3), "plan": np.ones((3, 4)) + bump, "split": np.asarray("eval")}
        if extra:
            d["only_here"] = np.zeros(2)
        np.savez(os.path.join(td, "ticks_%s_m10_causal_heads.npz" % R.key_of(run, 0)), **d)

    def p60(root, segs=2, short=0, clipped=83):
        kb = {"60": np.zeros((7, 2), int)}
        for run, k in R.NEW_RUN_LIST:
            if R.ALL_RUNS[run][2] != 60:
                continue
            td = R.tick_dir(root, run, k)
            os.makedirs(td, exist_ok=True)
            key = R.key_of(run, k)
            json.dump({"windows": {"model": 7, "floor": 7}, "segments_per_window": segs, "clipped_windows_in_preroll": {"model": clipped, "floor": 126}}, io.open(os.path.join(td, "preroll_log_%s.json" % key), "w", encoding="utf-8"))
            np.savez(os.path.join(td, "preroll_offsets_%s.npz" % key), seg_ci=np.zeros(14 - short, int))
        return kb

    def case(name, got, want):
        items.append((name, got is want))

    for i, (name, kw, want) in enumerate((("l12 equal arrays, other run name", {}, True), ("l12 planted array difference", {"bump": 1e-3}, False), ("l12 planted key in one file", {"extra": True}, False))):
        a = A()
        a.pull = os.path.join(folder, "l12_%d" % i)
        ticks(a.pull, "R0LrCarry", "I10R0LrCarry_s0", **kw)
        ticks(a.pull, "R0LrFresh", "I10R0LrFresh_s0")
        case(name, l12_n3(a)["PASS"], want)
    a = A()
    a.pull = os.path.join(folder, "l12_missing")
    case("l12 file missing is not a pass", l12_n3(a)["PASS"] is None, True)
    for i, (name, kw, want) in enumerate((("l11 two segments, full rows, envelope counter above 0", {}, True), ("l11 planted one segment", {"segs": 1}, False), ("l11 planted short segment rows", {"short": 1}, False))):
        a = A()
        a.pull = os.path.join(folder, "l11_%d" % i)
        kb = p60(a.pull, **kw)
        r = l11_n3(a, kb, None)
        case(name, r["PASS"], want)
        if i == 0:
            case("l11 reports the envelope counter", all(v == {"model": 83, "floor": 126} for v in r["envelope_counter_reported_not_judged"].values()) and len(r["runs"]) == 8, True)
    a = A()
    a.pull = os.path.join(folder, "l11_missing")
    case("l11 missing files is not a pass", l11_n3(a, {"60": np.zeros((7, 2), int)}, None)["PASS"], False)
    global _orig_read_grid
    keep = _orig_read_grid
    try:
        for name, grid, same, fam, want in (("grid 60: family controls pass, headway controls not required", "60", True, True, True), ("grid 60: planted family control failure", "60", True, False, False),
                                            ("grid 60: planted window mismatch", "60", False, True, False), ("grid 30: headway controls stay required", "30", True, True, False)):
            _orig_read_grid = lambda *x, _s=same, _f=fam: {"controls": {"same_windows_every_dump": _s, "headway_h1_h6": {"ALL_PASS": False}, "family_q2_q3_q4": {"q2": {"PASS": _f}, "q3": {"PASS": True}}, "ALL_PASS": False}}  # noqa: E731
            case(name, read_grid_n3(None, None, grid, None, None, None, None, None, None)["controls"]["ALL_PASS"], want)
        _orig_read_grid = lambda *x: {"controls": {"ALL_PASS": False, "why": "runs missing"}}  # noqa: E731
        case("grid 60: an early stop of the reader stays a stop", read_grid_n3(None, None, "60", None, None, None, None, None, None)["controls"]["ALL_PASS"], False)
    finally:
        _orig_read_grid = keep
    res = {"beside": {"P60_pairs": {"X@P60": {"A": "a", "B": "b", "headway": {"U": 1}, "cells": {"c": 2}}}, "levels_P60": {"r": 1}, "levels_P30": {"r": 3}, "spread": {"P30": 4, "P60": 5}, "AT_FRESH_pairs": {"Y": {"headway": {"U": 6}}}},
           "pairs": {"AGE-CONV": {"headway": {"U": 7}}}, "grid_reads_detail_controls_only": {"30": {"spread": {"headway": 8}}, "60": {"spread": {"headway": 9, "family": 10}}}}
    strip_p60_headway(res)
    s = json.dumps(res)
    case("P60 headway shares removed", res["beside"]["P60_pairs"]["X@P60"]["headway"] == NOT_REPORTED and res["beside"]["levels_P60"] == NOT_REPORTED and res["beside"]["spread"]["P60"] == NOT_REPORTED
         and res["grid_reads_detail_controls_only"]["60"]["spread"]["headway"] == NOT_REPORTED, True)
    case("P60 cells and everything at P = 30 kept", res["beside"]["P60_pairs"]["X@P60"]["cells"] == {"c": 2} and res["beside"]["levels_P30"] == {"r": 3} and res["beside"]["spread"]["P30"] == 4
         and res["pairs"]["AGE-CONV"]["headway"] == {"U": 7} and res["beside"]["AT_FRESH_pairs"]["Y"]["headway"] == {"U": 6} and res["grid_reads_detail_controls_only"]["30"]["spread"]["headway"] == 8
         and res["grid_reads_detail_controls_only"]["60"]["spread"]["family"] == 10 and '"U": 1' not in s and "9" not in s.replace("19", ""), True)
    for n, okk in items:
        print("   %-62s %s" % (n, "ok" if okk else "NOT OK"))
    allok = all(o for _, o in items)
    print("ZZ SELFTEST-N3 ok: %s (%d of %d)" % (allok, sum(1 for _, o in items if o), len(items)))
    return 0 if allok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest-n3" in argv:
        return selftest_n3(argv[argv.index("--selftest-n3") + 1])
    if "--selftest" in argv:
        return R.main(argv)
    out_path = argv[argv.index("--out") + 1]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(argv)
    text = buf.getvalue().split("\n")
    cut_at = next((i for i, ln in enumerate(text) if ln.startswith("== beside, naming nothing")), None)
    head = text if cut_at is None else text[:cut_at + 1]
    print("\n".join(head))
    cap = None
    if rc == 0 and os.path.exists(out_path):
        res = json.load(io.open(out_path, encoding="utf-8"))
        strip_p60_headway(res)
        res["note_n3"] = "read of record by note N3 (wrapper read_longroll10k_n3.py): l12 without the run name, l11 without the envelope counter, no headway shares from the P = 60 grid"
        json.dump(res, io.open(out_path, "w", encoding="utf-8", newline="\n"), indent=1)
        cap = res.get("controls_all_pass")
        if cut_at is not None and isinstance(res.get("beside"), dict):
            for k, v in res["beside"].items():
                print("   %s %s" % (k, json.dumps(v)[:2500]))
    print("ZZ READ-LONGROLL-N3 rc %s controls_all_pass %s" % (rc, cap))
    return rc


_orig_file_controls = R.file_controls
_orig_read_grid = R.read_grid
R.file_controls = file_controls_n3
R.read_grid = read_grid_n3

if __name__ == "__main__":
    sys.exit(main())
