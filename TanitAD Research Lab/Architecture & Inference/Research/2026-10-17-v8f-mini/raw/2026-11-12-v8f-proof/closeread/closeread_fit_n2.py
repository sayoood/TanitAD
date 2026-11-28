#!/usr/bin/env python
"""closeread_fit_n2.py -- card PREREG_V8F_PROOF_ADD78 (CLOSEREAD2): the fit tool of card ADD77 (closeread_fit.py, imported, not changed) with the two control forms of card ADD78
and the name prefix CLOSEREAD2.
  c2  on score frames whose trailing label run has all five frames: the two-point change of the label gap over the five frames against the label closing speed, r >= 0.70 positive,
      regression slope of the closing speed on the two-point change in [0.8, 1.2], at least 500 frames, lower r when the labels are shifted by +-10 frames.
      Beside, reported and not judged: c2 of card ADD77 as written.
  c3  the text of the card: on every rule-log row with the five-frame window in the clip, presence and gap after the map equal the dump within 1e-3, rows of the first replan
      tick among them, and no such row without a read in the rule log. Rule-log rows where the dump has no window are counted and reported, not judged.
Everything else (arms, bootstrap, bars, the other controls, the refusal to write a name when a control fails) is the imported tool.

  python closeread_fit_n2.py --dump-dir D --out-dir O --episode-map MAP.json --rulelog0 NPZ --qmap0 JSON --rulelog1 NPZ --qmap1 JSON [--seeds 0,1] [--boot 2000]
  python closeread_fit_n2.py --selftest2
Last stdout line: ZZ CLOSEREAD rc <rc> controls_all_pass <bool>   (the line of the imported tool; --selftest2 prints ZZ CLOSEREAD2-SELFTEST rc <rc>)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closeread_fit as F  # noqa: E402

BASE_SHA16 = "b71f481cdc57e6ec"
RUN_FULL = 5
BAR_R, SLOPE_LO, SLOPE_HI, MIN_FRAMES = 0.70, 0.8, 1.2, 500
BASE_C2, BASE_C3, BASE_NAME, BASE_ANAME = F.c2_definition, F.c3_dump_is_rule_reader, F.name_of, F.a_name


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def c2_full_run(D, bar=BAR_R):
    """card ADD78 c2. -> dict with pass; 'as_written_card_add77' holds the reading of the old form (not judged)."""
    dt = float(D.get("dt_s", F.DT_S_DEFAULT))
    ch = F.two_point_label(D, 4, dt)
    cid, fr = np.asarray(D["cid"], np.int64), np.asarray(D["frame"], np.int64)
    run = np.asarray(D["lab_run"], np.int64)
    y = np.asarray(D["lab_closing"], np.float64)
    full = np.asarray(D["lab_present"], bool) & (run == RUN_FULL) & np.isfinite(y)
    m = np.isfinite(ch) & full
    out = {"n": int(m.sum()), "form": "frames whose trailing label run has all five frames"}
    r0 = F.pearson(ch[m], y[m])
    va = float(np.var(ch[m])) if m.sum() > 2 else 0.0
    slope = float(np.mean((ch[m] - ch[m].mean()) * (y[m] - y[m].mean())) / va) if va > 0 else float("nan")
    out.update({"r0": r0, "slope": slope})
    shifted_ok = True
    n = len(cid)
    for s in (10, -10):
        i = np.arange(n)
        j = i + s
        v = (j >= 0) & (j < n)
        i, j = i[v], j[v]
        v = (cid[i] == cid[j]) & (fr[j] - fr[i] == s) & m[i] & full[j]
        rs = F.pearson(ch[i[v]], y[j[v]])
        out["r_shift%+d" % s] = rs
        out["n_shift%+d" % s] = int(v.sum())
        shifted_ok = shifted_ok and bool(np.isfinite(rs) and rs < r0)
    old = BASE_C2(D)
    out["as_written_card_add77"] = {"r0": old.get("r0"), "n": old.get("n"), "pass": bool(old.get("pass")), "judged": False}
    out["pass"] = bool(np.isfinite(r0) and r0 >= bar and np.isfinite(slope) and SLOPE_LO <= slope <= SLOPE_HI and out["n"] >= MIN_FRAMES and shifted_ok)
    return out


def c3_card_text(score_dump, rule_log, qmap, tol=1e-3):
    """card ADD78 c3: equality on every rule-log row with the window in the clip; a rule-log read where the dump has no window is counted, not judged."""
    key = {(int(c), int(f)): i for i, (c, f) in enumerate(zip(score_dump["clip_index"], score_dump["frame"]))}
    ci, j = np.asarray(rule_log["ci"], np.int64), np.asarray(rule_log["j"], np.int64)
    kk = np.asarray(rule_log["k"], np.int64) if "k" in rule_log else np.zeros(len(ci), np.int64)
    rows = np.array([key.get((int(c), int(f)), -1) for c, f in zip(ci, j)], np.int64)
    inclip = rows >= 0
    r = {"n_log_rows": int(len(ci)), "n_in_clip": int(inclip.sum()), "tol": tol}
    if not inclip.any():
        r["pass"] = False
        return r
    rr = rows[inclip]
    pl, gl = np.asarray(rule_log["p_src"], np.float64)[inclip], np.asarray(rule_log["g_src"], np.float64)[inclip]
    valid = np.asarray(score_dump["valid"], bool)[rr]
    pd_, gd = np.asarray(score_dump["p"], np.float64)[rr], F.qmap_apply(np.asarray(score_dump["gap"], np.float64)[rr], *qmap)
    log_fin = np.isfinite(pl) & np.isfinite(gl)
    no_read_where_window = int((valid & ~log_fin).sum())
    both = valid & log_fin
    dp = np.abs(pl[both] - pd_[both])
    dg = np.abs(gl[both] - gd[both])
    first = both & (kk[inclip] == 0)
    r.update({"n_compared": int(both.sum()), "n_first_tick": int(first.sum()), "n_window_rows_without_log_read": no_read_where_window,
              "n_log_reads_without_window_not_judged": int((~valid & log_fin).sum()),
              "max_abs_diff_p": float(dp.max()) if len(dp) else float("nan"), "max_abs_diff_gap_after_map": float(dg.max()) if len(dg) else float("nan")})
    r["pass"] = bool(both.sum() > 0 and first.sum() > 0 and no_read_where_window == 0 and dp.max() <= tol and dg.max() <= tol)
    return r


def name_of2(arm, H, s):
    nm, ok = BASE_NAME(arm, H, s)
    return "CLOSEREAD2" + nm[len("CLOSEREAD"):], ok


def a_name2(oks):
    return "CLOSEREAD2" + BASE_ANAME(oks)[len("CLOSEREAD"):]


def patch():
    F.c2_definition, F.c3_dump_is_rule_reader, F.name_of, F.a_name = c2_full_run, c3_card_text, name_of2, a_name2


def with_jumps(D):
    """a copy of a synthetic dump with a step of 30 m in the label gap at frame 50 of every clip; the run length after the step is 1, 2, 3, 4, then 5 again."""
    D = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in D.items()}
    fr = np.asarray(D["frame"], np.int64)
    D["lab_gap"] = D["lab_gap"] + np.where(fr >= 50, 30.0, 0.0)
    run = np.array(D["lab_run"], np.int64)
    for off, val in ((0, 1), (1, 2), (2, 3), (3, 4)):
        run[fr == 50 + off] = val
    D["lab_run"] = run
    return D


def selftest2():
    fails = []

    def chk(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        if not cond:
            fails.append(name)
    base = os.path.join(HERE, "closeread_fit.py")
    chk("the imported tool is the reviewed one (sha256 %s)" % BASE_SHA16, sha16(base) == BASE_SHA16)
    fits, scores, epm = F.make_synth(seed=3)
    S = scores[0]
    c = c2_full_run(S)
    chk("c2 passes on clean synthetic labels (r %.3f, slope %.3f)" % (c["r0"], c["slope"]), c["pass"])
    sh = dict(S)
    sh["lab_closing"] = np.roll(S["lab_closing"], 1000)
    chk("c2 fails on labels of another clock", not c2_full_run(sh)["pass"])
    sg = dict(S)
    sg["lab_closing"] = -S["lab_closing"]
    chk("c2 fails with the sign turned", not c2_full_run(sg)["pass"])
    un = dict(S)
    un["lab_closing"] = 5.0 * S["lab_closing"]
    cu = c2_full_run(un)
    chk("c2 fails with the unit times 5 at r %.3f (slope %.2f)" % (cu["r0"], cu["slope"]), not cu["pass"] and cu["r0"] >= 0.70)
    J = with_jumps(S)
    cj = c2_full_run(J)
    chk("c2 passes on labels with a step of 30 m per clip, and the form of card ADD77 reads lower there (%.3f against %.3f)" % (cj["as_written_card_add77"]["r0"], cj["r0"]),
        cj["pass"] and cj["as_written_card_add77"]["r0"] < cj["r0"] - 0.05)
    few = dict(S)
    few["lab_run"] = np.where(np.arange(len(S["lab_run"])) < 300, 5, 4)
    chk("c2 fails with fewer than 500 frames of a full run", not c2_full_run(few)["pass"])
    rl = F.synth_rulelogs(scores)
    chk("c3 passes on a synthetic rule log", c3_card_text(S, rl[0]["log"], rl[0]["qmap"])["pass"])
    rl2 = F.synth_rulelogs(scores, perturb=0.01)
    chk("c3 fails on a planted gap difference of 0.01", not c3_card_text(S, rl2[0]["log"], rl2[0]["qmap"])["pass"])
    inv = np.where(~np.asarray(S["valid"], bool) & (np.asarray(S["frame"]) == 3))[0][:20]
    lg = {k: np.array(v, copy=True) for k, v in rl[0]["log"].items()}
    lg["ci"] = np.concatenate([lg["ci"], S["clip_index"][inv]])
    lg["j"] = np.concatenate([lg["j"], S["frame"][inv]])
    lg["k"] = np.concatenate([lg["k"], np.zeros(len(inv), lg["k"].dtype)])
    lg["p_src"] = np.concatenate([lg["p_src"], np.full(len(inv), 0.5)])
    lg["g_src"] = np.concatenate([lg["g_src"], np.full(len(inv), 20.0)])
    c3x = c3_card_text(S, lg, rl[0]["qmap"])
    chk("c3 passes with rule-log reads on frames without a window (%d counted) while the tool of card ADD77 fails there" % c3x["n_log_reads_without_window_not_judged"],
        c3x["pass"] and c3x["n_log_reads_without_window_not_judged"] == len(inv) and not BASE_C3(S, lg, rl[0]["qmap"])["pass"])
    lg2 = {k: np.array(v, copy=True) for k, v in rl[0]["log"].items()}
    lg2["p_src"][0] = np.nan
    chk("c3 fails when a row with the window has no read in the rule log", not c3_card_text(S, lg2, rl[0]["qmap"])["pass"])
    patch()
    for tag, kw, want in (("signal in the reads", {}, "CLOSEREAD2-A-CLOSING-IS-READABLE"), ("noise in the reads", {"reads_noise": True}, "CLOSEREAD2-NOT-READABLE-FROM-READER-OUTPUTS")):
        f_, s_, e_ = F.make_synth(seed=4, **kw)
        d = os.path.join(tempfile.mkdtemp(prefix="closeread2_selftest_"), "out")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, allp = F.run_all(f_, s_, F.synth_lists(f_), e_, F.synth_rulelogs(s_), d, B=300, n_epochs=25, quiet=False, c5_epochs=10)
        p = os.path.join(d, "CLOSEREAD_RESULT.json")
        res = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        chk("end to end, %s: controls pass, rc 0, top name %s" % (tag, want), rc == 0 and allp and res.get("A_name") == want and all(v.startswith("CLOSEREAD2-") for v in res.get("names", {}).values()))
    f_, s_, e_ = F.make_synth(seed=4)
    for k in s_:
        s_[k]["lab_closing"] = 5.0 * s_[k]["lab_closing"]
    d = os.path.join(tempfile.mkdtemp(prefix="closeread2_selftest_"), "out")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc, allp = F.run_all(f_, s_, F.synth_lists(f_), e_, F.synth_rulelogs(s_), d, B=300, n_epochs=25, c5_epochs=10)
    out = buf.getvalue()
    chk("a target in another unit: c2 fails, rc 1, no result file, no name and no cell line on stdout", rc == 1 and not allp and not os.path.exists(os.path.join(d, "CLOSEREAD_RESULT.json")) and "CLOSEREAD2-" not in out and "== cells ==" not in out)
    ok = not fails
    print("ZZ CLOSEREAD2-SELFTEST rc %d" % (0 if ok else 1))
    return 0 if ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        print("REFUSED: the self-test of this tool is --selftest2 (the self-test of the imported tool is run from closeread_fit.py)")
        return 2
    if "--selftest2" in argv:
        return selftest2()
    if sha16(os.path.join(HERE, "closeread_fit.py")) != BASE_SHA16:
        print("REFUSED: closeread_fit.py is not the reviewed file (sha256 %s expected)" % BASE_SHA16)
        print("ZZ CLOSEREAD rc 2 controls_all_pass False")
        return 2
    if "--episode-map" not in argv:
        print("REFUSED: --episode-map is needed (note N1 of card ADD77)")
        print("ZZ CLOSEREAD rc 2 controls_all_pass False")
        return 2
    patch()
    rc = F.main(argv)
    out_dir = argv[argv.index("--out-dir") + 1] if "--out-dir" in argv else ""
    dump_dir = argv[argv.index("--dump-dir") + 1] if "--dump-dir" in argv else ""
    if out_dir and os.path.isdir(out_dir) and not os.path.exists(os.path.join(out_dir, "CLOSEREAD2_CARD.json")):
        files = {f: sha16(os.path.join(dump_dir, f)) for f in sorted(os.listdir(dump_dir))} if dump_dir and os.path.isdir(dump_dir) else {}
        info = {"card": "PREREG_V8F_PROOF_ADD78", "note": "the result and verdict files are written by the imported tool and carry its card field (ADD77); the card of this read is ADD78, the names carry the prefix CLOSEREAD2",
                "imported_tool_sha256_16": BASE_SHA16, "this_tool_sha256_16": sha16(os.path.abspath(__file__)), "dump_files_sha256_16": files, "rc": int(rc)}
        with open(os.path.join(out_dir, "CLOSEREAD2_CARD.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(info, fh, indent=1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
