"""check_shufclose.py -- pod check of the wrapper v8f_t1_eval_reader_rule_rzgfhqet.py (card ADD83, control u0: the table source is the estimate).
Part A (always; numpy and the standard library only): static facts of the wrapper text, none of the banned words in the new text. --pure-only stops here.
Part B (on the pod; needs the tool tree, the checkpoint, the data, the tables of make_surrogate.py and the banked run): four rollouts of the NEW wrapper on a subset of the pinned grid (the
  first --n-clips clips with all their windows), fresh (no pre-roll), model seed 0, readout K (--readout causal_g --acc-gain 0 --path-exec --kappa-max 0.2), --rule-every 5, q50 map, reader seed 0:
    sk_rr     form RR (the reference of the same wrapper, same grid, same batches)
    sk_ident  form RRe --closing-table ident_L25_s0.npz  (the estimate table itself, identity map)
    sk_zero   form RRe --closing-table zero_table.npz
    sk_c2     form RRe --closing-table const2_table.npz  (a planted +2 m/s)
  plus three refusals (form RR with --closing-table; form RRe with --closing-est and --closing-table together; form RRe with neither). Checks:
    U0A  sk_ident against the BANKED run I10RQ50eL25KF5Hb20_s0 (--banked-est, card ADD79) on the rows of the subset clips (replan and mid-look rule logs keyed by (ci, t0, j, k[, q]); first plans
         keyed by (ci, t0)): every subset row is a banked row; the closing column c_src within --tol-banked (default 1e-3 m/s) on every common row; the same rows marked full (est_ok equal exactly);
         first plans within --tol-banked; rows present with a full history exist (the check is not empty). The remaining columns are compared and reported, not judged.
    U0B  sk_zero equals sk_rr: dump arrays, tick file, rule log, rule-log side file, mid-look log, mid-look side file, first-plan file; every common array, largest difference 0; c_src is 0 on every
         row of the logs of both; c_est / est_ok present row for row in the logs of sk_zero (NaN exactly where est_ok is False)
    U0C  sk_c2: c_src is 2.0 on every present row with est_ok and 0.0 on every present row without; v_L = max(0, v_l + c_src); c_est is 2.0 on every est_ok row (replan and mid-look logs)
    U0L  the run text of sk_ident / sk_zero / sk_c2 has the line "est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>": counts equal those recomputed from the logs
         (preroll 0 0), the sha256 is that of the table file; sk_rr has no such line
    U0F  the three refusals: non-zero return code, text with REFUSED and the flag(s) named
  Nothing here removes, moves or stops anything; part B writes only inside --workdir (refused if it exists, except with --analyze-only, which only reads the run folders already there).
Usage on the pod:
  python check_shufclose.py --new NEW.py --tree TREE --ckpt CKPT --tracks native_tracks.pt --eval-cache DIR --eval-join J.npz --grid PINNED_GRID.json --reader READER.pt --qmap MAP.json
         --workdir NEWDIR --tables DIR --est-file closest_L25_s0.npz --distrule-dir DIR --closest-dir DIR --banked-est T1RUNDIR [--python PY] [--device cuda] [--n-clips 3] [--dry] [--analyze-only]
Last line: ZZ CHECK-SHUFCLOSE PASS <counts>  or  ZZ CHECK-SHUFCLOSE FAIL <names>   (--pure-only: ZZ CHECK-SHUFCLOSE-PURE PASS or FAIL <names>)
--selftest: the analysis part on synthetic logs, with planted failures that each have to be caught (ZZ CHECK-SHUFCLOSE-SELFTEST PASS or FAIL)."""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAD = ("vo" + "id", "super" + "seded", "retr" + "acted", "for" + "merly", "corr" + "ection", "with" + "drawn", "no " + "longer", "closed " + "loop")
LINE = re.compile(r"^est looks scored full (\d+) short (\d+) preroll full (\d+) short (\d+) file ([0-9a-f]{64})$")
MARKED = ("c_est", "est_ok")


# ==================================================================================================================== analysis (pure functions, used by the selftest)
def keys_of(L, with_q):
    cols = ["ci", "t0", "j", "k"] + (["q"] if with_q else [])
    return list(zip(*[np.asarray(L[c]).astype(np.int64).tolist() for c in cols]))


def compare_rows(A, B, with_q, tol):
    """rows of A (subset run) against the rows of B (banked) with the same key (ci, t0, j, k[, q]). -> dict: counts, c_src largest difference, est_ok mismatches, the other columns reported."""
    ka, kb = keys_of(A, with_q), keys_of(B, with_q)
    ib = {k: i for i, k in enumerate(kb)}
    ia = [i for i, k in enumerate(ka) if k in ib]
    jb = [ib[ka[i]] for i in ia]
    out = {"rows_subset": len(ka), "rows_common": len(ia), "rows_not_in_banked": len(ka) - len(ia)}
    if not ia:
        out.update(c_src_largest_diff=float("nan"), c_src_bad=0, est_ok_mismatch=0, full_rows=0, present_full_rows=0, other_bad_columns={}, other_largest=0.0)
        return out
    d = np.abs(np.asarray(A["c_src"], np.float64)[ia] - np.asarray(B["c_src"], np.float64)[jb])
    out["c_src_largest_diff"] = float(d.max())
    out["c_src_bad"] = int((~(d <= tol)).sum())
    ea, eb = np.asarray(A["est_ok"]).astype(bool)[ia], np.asarray(B["est_ok"]).astype(bool)[jb]
    out["est_ok_mismatch"] = int((ea != eb).sum())
    out["full_rows"] = int(ea.sum())
    out["present_full_rows"] = int((ea & np.asarray(A["present"]).astype(bool)[ia]).sum())
    out["c_src_nonzero_full"] = int(((np.asarray(A["c_src"])[ia] != 0.0) & ea).sum())
    bad_cols, worst = {}, 0.0
    for c in A:
        if c in MARKED or c == "c_src" or c not in B:
            continue
        x, y = np.asarray(A[c]), np.asarray(B[c])
        if x.ndim != 1 or y.ndim != 1 or x.shape != (len(ka),) or y.shape != (len(kb),) or x.dtype.kind not in "fiub" or y.dtype.kind not in "fiub":
            continue
        x, y = x[ia], y[jb]
        if x.dtype.kind in "iub" or y.dtype.kind in "iub":
            ne = x.astype(np.int64) != y.astype(np.int64)
        else:
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            with np.errstate(invalid="ignore"):
                ne = ~((np.isnan(xf) & np.isnan(yf)) | (np.abs(xf - yf) <= tol))
            fin = np.isfinite(xf) & np.isfinite(yf)
            if fin.any():
                worst = max(worst, float(np.max(np.abs(xf[fin] - yf[fin]))))
        if ne.any():
            bad_cols[c] = int(ne.sum())
    out["other_bad_columns"], out["other_largest"] = bad_cols, worst
    return out


def plan_compare(PA, PB):
    kz = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(PA["ci"], PA["t0"]))}
    kb = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(PB["ci"], PB["t0"]))}
    common = [k for k in kz if k in kb]
    d = float(max((np.abs(PA["plan"][kz[k]].astype(np.float64) - PB["plan"][kb[k]].astype(np.float64)).max() for k in common), default=np.nan))
    return {"plans_subset": len(kz), "plans_common": len(common), "plan_largest_diff": d}


def analyze_u0a(RLA, MLA, P1A, RLB, MLB, P1B, tol):
    """-> (pass, detail). The table of the estimate through the new form against the banked run of the same estimator."""
    cr, cm = compare_rows(RLA, RLB, False, tol), compare_rows(MLA, MLB, True, tol)
    pc = plan_compare(P1A, P1B)
    ok = True
    for c in (cr, cm):
        ok = ok and c["rows_common"] == c["rows_subset"] > 0 and c["c_src_bad"] == 0 and c["est_ok_mismatch"] == 0 and c["full_rows"] > 0 and c["present_full_rows"] > 0 and c["c_src_nonzero_full"] > 0
    ok = ok and pc["plans_common"] == pc["plans_subset"] > 0 and bool(pc["plan_largest_diff"] <= tol)
    return bool(ok), {"replan": cr, "mid": cm, "plans": pc, "tol": tol}


def maxdiff_dicts(a, b, skip=("tag", "form", "mix", "mix_counts", "closing_est", "closing_table", "reader", "qmap")):
    """-> (largest absolute difference over the common numeric arrays, problems, keys only b has)."""
    a = {k: v for k, v in a.items() if k not in skip}
    b = {k: v for k, v in b.items() if k not in skip}
    extra = sorted(set(b) - set(a))
    b = {k: v for k, v in b.items() if k in a}
    worst, prob = 0.0, []
    if sorted(a) != sorted(b):
        prob.append("keys differ %s" % (sorted(set(a) ^ set(b))[:4],))
    for k in sorted(set(a) & set(b)):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            prob.append("shape of %s" % k)
            continue
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(np.float64), y.astype(np.float64)
            if not np.array_equal(np.isnan(xf), np.isnan(yf)):
                prob.append("nan pattern of %s" % k)
                continue
            if xf.size and np.isfinite(xf).any():
                worst = max(worst, float(np.nanmax(np.abs(xf - yf))))
        elif x.dtype.kind in "US":
            continue
        elif not np.array_equal(x, y):
            prob.append("content of %s" % k)
    return worst, prob, extra


def analyze_u0b(DR, DZ, FR, FZ, RLZ, MLZ, RLR, MLR):
    """DR / DZ: dump files {name: arrays} of the RR run and the zero-table run; FR / FZ: {label: arrays or None} of the tick, rule-log, side, mid-look and first-plan files of both;
    RLZ, MLZ, RLR, MLR: the rule logs. -> (pass, detail)."""
    worst, prob, files, newk = 0.0, [], 0, {}
    if sorted(DR) != sorted(DZ) or not DR:
        prob.append("dump file lists differ or empty")
    for f in sorted(set(DR) & set(DZ)):
        w_, p_, _ = maxdiff_dicts(DR[f], DZ[f])
        worst, prob, files = max(worst, w_), prob + p_, files + 1
    for label in sorted(set(FR) | set(FZ)):
        a, b = FR.get(label), FZ.get(label)
        if (a is None) != (b is None):
            prob.append("%s present only in %s" % (label, "RR" if a is not None else "zero"))
            continue
        if a is None:
            continue
        w_, p_, ex = maxdiff_dicts(a, b)
        worst, prob, files = max(worst, w_), prob + ["%s: %s" % (label, x) for x in p_], files + 1
        if ex:
            newk[label] = ex
    c_zero = bool(len(RLZ["c_src"]) and len(MLZ["c_src"]) and np.all(RLZ["c_src"] == 0.0) and np.all(MLZ["c_src"] == 0.0) and np.all(RLR["c_src"] == 0.0) and np.all(MLR["c_src"] == 0.0))
    aligned = all(len(RLZ[k]) == len(RLZ["ci"]) for k in MARKED) and all(len(MLZ[k]) == len(MLZ["ci"]) for k in MARKED)
    nan_pat = bool(aligned and np.array_equal(np.isnan(RLZ["c_est"]), ~np.asarray(RLZ["est_ok"]).astype(bool)) and np.array_equal(np.isnan(MLZ["c_est"]), ~np.asarray(MLZ["est_ok"]).astype(bool)))
    ok = worst == 0.0 and not prob and files > 0 and c_zero and aligned and nan_pat and "rule log" in newk and "mid-look log" in newk
    return bool(ok), {"largest_diff": worst, "files": files, "problems": prob[:4], "new_keys": newk, "c_src_zero": c_zero, "columns_aligned": aligned, "nan_pattern": nan_pat,
                      "tick_rows": int(len(RLZ["ci"])), "mid_rows": int(len(MLZ["ci"]))}


def analyze_u0c(RL, ML):
    res, ok = {}, True
    for label, L in (("replan", RL), ("mid", ML)):
        pres, okk = np.asarray(L["present"]).astype(bool), np.asarray(L["est_ok"]).astype(bool)
        a2 = bool(np.all(L["c_src"][pres & okk] == 2.0))
        a0 = bool(np.all(L["c_src"][pres & ~okk] == 0.0))
        avl = bool(np.allclose(L["v_L"][pres], np.maximum(0.0, L["v_l"][pres] + L["c_src"][pres])))
        ae = bool(np.all(L["c_est"][okk] == 2.0) and np.isnan(L["c_est"][~okk]).all())
        res[label] = {"present_full": int((pres & okk).sum()), "present_short": int((pres & ~okk).sum()), "c_src_2": a2, "c_src_0": a0, "v_L": avl, "c_est": ae}
        ok = ok and a2 and a0 and avl and ae and bool((pres & okk).any())
    return bool(ok), res


def analyze_line(lines, rl, ml, table_sha):
    """lines: the run-text lines that start with 'est looks'; rl, ml: the rule logs. -> (ok, detail)"""
    full = int(np.asarray(rl["est_ok"]).astype(bool).sum() + np.asarray(ml["est_ok"]).astype(bool).sum())
    short = int((~np.asarray(rl["est_ok"]).astype(bool)).sum() + (~np.asarray(ml["est_ok"]).astype(bool)).sum())
    m_ = LINE.match(lines[0]) if len(lines) == 1 else None
    good = bool(m_ and int(m_.group(1)) == full and int(m_.group(2)) == short and int(m_.group(3)) == 0 and int(m_.group(4)) == 0 and m_.group(5) == table_sha)
    return good, {"lines": len(lines), "full": full, "short": short, "line_ok": good}


def analyze_refusal(rc, text, names):
    return bool((rc is None or rc != 0) and "REFUSED" in text and all(n in text for n in names))


# ==================================================================================================================== selftest
def _synth_log(n, with_q, seed, c_src_value=None):
    r = np.random.default_rng(seed)
    L = {"ci": np.repeat(np.arange(3), n // 3 + 1)[:n].astype(np.int64), "t0": (np.arange(n) // 2).astype(np.int64), "j": np.arange(n, dtype=np.int64) + 10, "k": (np.arange(n) % 2).astype(np.int64),
         "present": r.uniform(size=n) < 0.8, "v_l": r.uniform(2, 10, n), "gap_src": r.uniform(5, 50, n)}
    if with_q:
        L["q"] = np.full(n, 5, np.int64)
    L["est_ok"] = r.uniform(size=n) < 0.85
    c = r.normal(0, 1.2, n) if c_src_value is None else np.full(n, float(c_src_value))
    L["c_est"] = np.where(L["est_ok"], c, np.nan)
    L["c_src"] = np.where(L["est_ok"], c, 0.0)
    L["v_L"] = np.maximum(0.0, L["v_l"] + L["c_src"])
    return L


def _synth_p1(seed):
    r = np.random.default_rng(seed)
    return {"ci": np.array([0, 0, 1, 1, 2], np.int64), "t0": np.array([0, 5, 0, 5, 0], np.int64), "plan": r.normal(0, 3, (5, 10, 2)).astype(np.float32)}


def selftest():
    res = []

    def need(name, cond):
        print(("PASS " if cond else "FAIL ") + name, flush=True)
        res.append(bool(cond))

    def cp(L):
        return {k: np.array(v, copy=True) for k, v in L.items()}
    RL, ML, P1 = _synth_log(60, False, 1), _synth_log(60, True, 2), _synth_p1(3)
    # a banked run has more rows, other arrays order and tiny differences
    perm = np.random.default_rng(4).permutation(60)
    RLB, MLB = {k: v[perm] for k, v in cp(RL).items()}, {k: v[perm] for k, v in cp(ML).items()}
    RLB["c_src"] = RLB["c_src"] + 4e-4
    P1B = {k: v[::-1].copy() for k, v in cp(P1).items()}
    P1B["plan"] = P1B["plan"] + 4e-4
    ok, d = analyze_u0a(RL, ML, P1, RLB, MLB, P1B, 1e-3)
    need("u0a: the identity case passes (permuted banked rows, differences 4e-4)", ok)
    bad = cp(RLB)
    bad["c_src"] = bad["c_src"] + 2e-3
    need("u0a: closing column off by 2e-3 fails", not analyze_u0a(RL, ML, P1, bad, MLB, P1B, 1e-3)[0])
    bad = cp(MLB)
    i = int(np.nonzero(bad["est_ok"])[0][0])
    bad["est_ok"][i] = False
    need("u0a: one row marked full in one log and short in the other fails", not analyze_u0a(RL, ML, P1, RLB, bad, P1B, 1e-3)[0])
    bad = cp(P1B)
    bad["plan"] = bad["plan"] + 2e-3
    need("u0a: plans of the first look off by 2e-3 fails", not analyze_u0a(RL, ML, P1, RLB, MLB, bad, 1e-3)[0])
    bad = {k: v[:-1] for k, v in cp(RLB).items()}
    need("u0a: a subset row that the banked run does not have fails", not analyze_u0a(RL, ML, P1, bad, MLB, P1B, 1e-3)[0])
    zr, zm = cp(RL), cp(ML)
    for L in (zr, zm):
        L["c_src"][:] = 0.0
        L["v_L"] = np.maximum(0.0, L["v_l"])
    rr_r = {k: v for k, v in zr.items() if k not in MARKED}
    rr_m = {k: v for k, v in zm.items() if k not in MARKED}
    D = {"ep0.npz": {"plan": np.arange(6.0), "x": np.array([1, 2])}}
    FR = {"rule log": rr_r, "mid-look log": rr_m, "first-plan file": P1}
    FZ = {"rule log": zr, "mid-look log": zm, "first-plan file": cp(P1)}
    ok, d = analyze_u0b(D, {k: cp(v) for k, v in D.items()}, FR, FZ, zr, zm, rr_r, rr_m)
    need("u0b: zero table equals RR, new keys only in the zero run", ok)
    D2 = {"ep0.npz": {"plan": np.arange(6.0) + 1e-9, "x": np.array([1, 2])}}
    need("u0b: a difference of 1e-9 in a dump array fails", not analyze_u0b(D, D2, FR, FZ, zr, zm, rr_r, rr_m)[0])
    FZ2 = {k: cp(v) for k, v in FZ.items()}
    FZ2["first-plan file"]["plan"] = FZ2["first-plan file"]["plan"] + np.float32(1e-6)
    need("u0b: a difference in the first-plan file fails", not analyze_u0b(D, {k: cp(v) for k, v in D.items()}, FR, FZ2, zr, zm, rr_r, rr_m)[0])
    zr2 = cp(zr)
    zr2["c_src"][3] = 0.01
    need("u0b: a non-zero closing input fails", not analyze_u0b(D, {k: cp(v) for k, v in D.items()}, FR, FZ, zr2, zm, rr_r, rr_m)[0])
    zr3 = cp(zr)
    zr3["c_est"] = np.zeros_like(zr3["c_est"])
    need("u0b: a NaN pattern that does not follow est_ok fails", not analyze_u0b(D, {k: cp(v) for k, v in D.items()}, FR, FZ, zr3, zm, rr_r, rr_m)[0])
    c2r, c2m = _synth_log(60, False, 5, 2.0), _synth_log(60, True, 6, 2.0)
    need("u0c: constant 2.0 passes", analyze_u0c(c2r, c2m)[0])
    b = cp(c2m)
    i = int(np.nonzero(b["present"] & b["est_ok"])[0][0])
    b["c_src"][i] = 2.1
    b["v_L"][i] = max(0.0, b["v_l"][i] + 2.1)
    need("u0c: one look with 2.1 fails", not analyze_u0c(c2r, b)[0])
    b = cp(c2r)
    j = int(np.nonzero(b["present"] & ~b["est_ok"])[0][0])
    b["c_src"][j] = 2.0
    b["v_L"][j] = max(0.0, b["v_l"][j] + 2.0)
    need("u0c: a short look that feeds 2.0 fails", not analyze_u0c(b, c2m)[0])
    sha = "ab" * 32
    full = int(c2r["est_ok"].sum() + c2m["est_ok"].sum())
    short = 120 - full
    line = "est looks scored full %d short %d preroll full 0 short 0 file %s" % (full, short, sha)
    need("u0l: a correct line passes", analyze_line([line], c2r, c2m, sha)[0])
    need("u0l: a wrong count fails", not analyze_line([line.replace("full %d short" % full, "full %d short" % (full + 1))], c2r, c2m, sha)[0])
    need("u0l: another file hash fails", not analyze_line([line], c2r, c2m, "cd" * 32)[0])
    need("u0l: no line fails", not analyze_line([], c2r, c2m, sha)[0])
    need("u0f: a refusal with the names passes", analyze_refusal(1, "REFUSED: --closing-table belongs to form RRe only", ["--closing-table"]))
    need("u0f: return code 0 fails", not analyze_refusal(0, "REFUSED: --closing-table", ["--closing-table"]))
    need("u0f: a text without REFUSED fails", not analyze_refusal(1, "Traceback --closing-table", ["--closing-table"]))
    need("u0f: a text without the flag name fails", not analyze_refusal(1, "REFUSED: something", ["--closing-table"]))
    ok = all(res)
    print("ZZ CHECK-SHUFCLOSE-SELFTEST " + ("PASS" if ok else "FAIL") + " checks %d failed %d" % (len(res), len(res) - sum(res)))
    return 0 if ok else 1


# ==================================================================================================================== command line
def build_parser():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--new", default=os.path.join(HERE, "v8f_t1_eval_reader_rule_rzgfhqet.py"))
    ap.add_argument("--pure-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    for k in ("tree", "ckpt", "tracks", "eval-cache", "eval-join", "grid", "reader", "qmap", "workdir", "banked-est", "tables", "est-file"):
        ap.add_argument("--" + k)
    ap.add_argument("--distrule-dir", default="")
    ap.add_argument("--closest-dir", default="")
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--tol-banked", type=float, default=1e-3)
    return ap


def main():
    ARGS = build_parser().parse_args()
    if ARGS.selftest:
        sys.exit(selftest())
    FAILED, COUNTS = [], {}

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (" -- " + detail if detail else ""), flush=True)
        if not cond:
            FAILED.append(name.split(" ")[0])

    SRC = open(ARGS.new, encoding="utf-8").read()
    print("wrapper", ARGS.new, hashlib.sha256(SRC.encode("utf-8")).hexdigest()[:16], flush=True)
    # ------------------------------------------------------------------------------------------------ part A
    texts = [
        ("A1 the flag --closing-table exists and is refused outside form RRe and together with --closing-est",
         'ap.add_argument("--closing-table", default="")' in SRC and "--closing-table belongs to form RRe only" in SRC and "--closing-est and --closing-table together" in SRC),
        ("A2 form RRe loads the table when the flag is given", 'if mine.closing_table:\n                load_table(mine.closing_table, mine.rule_log)' in SRC),
        ("A3 the look reads the table at the clip index and the frame of the look", 'return _table_at(states, e_off)' in SRC and "e = np.array([int(s.j) + int(e_off) for s in states], np.int64)" in SRC),
        ("A4 a missing row or a value that is not finite is short with the closing input 0.0", "est_ok = hit & np.isfinite(v)" in SRC and "return np.where(est_ok, c_est, 0.0)" in SRC),
        ("A5 the run line and the log columns are those of the parent", 'dict(meta, **est_tick_arrays(log)) if f == "RRe" else meta' in SRC and "        write_est_line()\n" in SRC),
    ]
    for n_, c_ in texts:
        check(n_, bool(c_))
    mark = "Text of the parent (rzgfhqe) follows."
    head_txt = SRC[:SRC.index(mark)].lower() if mark in SRC else ""
    own = open(os.path.abspath(__file__), encoding="utf-8").read().lower()
    check("A6 the new header text and this check have none of the banned words", bool(head_txt) and not [w for w in BAD if w in head_txt or w in own], str([w for w in BAD if w in head_txt or w in own]))
    if ARGS.pure_only:
        print("ZZ CHECK-SHUFCLOSE-PURE " + ("PASS" if not FAILED else "FAIL " + "; ".join(FAILED)))
        sys.exit(0 if not FAILED else 1)

    # ------------------------------------------------------------------------------------------------ part B
    for k in ("tree", "ckpt", "tracks", "eval_cache", "eval_join", "grid", "reader", "qmap", "workdir", "banked_est", "tables", "est_file"):
        if not getattr(ARGS, k):
            print("ZZ CHECK-SHUFCLOSE STOP missing --" + k.replace("_", "-"))
            sys.exit(3)
    W = ARGS.workdir
    M = ARGS.replan_steps
    DISTRULE = ARGS.distrule_dir or os.path.dirname(os.path.abspath(ARGS.new))
    CLOSEST = ARGS.closest_dir or os.path.dirname(os.path.abspath(ARGS.new))
    TIMEOUT = shutil.which("timeout")
    if os.path.exists(W) and not ARGS.analyze_only:
        print("ZZ CHECK-SHUFCLOSE STOP OUTPUT-EXISTS " + W)
        sys.exit(3)
    if not ARGS.dry and not os.path.exists(W):
        os.makedirs(W)
    T_IDENT = os.path.join(ARGS.tables, "ident_L25_s0.npz")
    T_ZERO = os.path.join(ARGS.tables, "zero_table.npz")
    T_C2 = os.path.join(ARGS.tables, "const2_table.npz")
    for p in (T_IDENT, T_ZERO, T_C2, ARGS.est_file):
        if not os.path.isfile(p):
            print("ZZ CHECK-SHUFCLOSE STOP missing-input " + p)
            sys.exit(3)
    GRID_SUB = os.path.join(W, "grid_subset.json")
    if not ARGS.dry and not os.path.exists(GRID_SUB):
        g = json.load(open(ARGS.grid, encoding="utf-8"))
        order = list(g["order"])[:ARGS.n_clips]
        g2 = dict(g)
        g2["order"], g2["windows"] = order, {c: g["windows"][c] for c in order}
        g2["n_windows"] = int(sum(len(g["windows"][c]) for c in order))
        g2["shufclose_subset"] = {"n_clips": len(order), "source": os.path.basename(ARGS.grid)}
        with open(GRID_SUB, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(g2, fh)

    RUNS = [("sk_rr", "RR", None), ("sk_ident", "RRe", T_IDENT), ("sk_zero", "RRe", T_ZERO), ("sk_c2", "RRe", T_C2)]

    def tag_of(name):
        return name + "_m%d_causal_heads" % M

    def run_cmd(name, form, table, out=None, extra=()):
        out = out or os.path.join(W, name)
        tb = ["--closing-table", table] if table else []
        cmd = [ARGS.python, ARGS.new, "--tree", ARGS.tree, "--form", form, "--reader", ARGS.reader, "--qmap", ARGS.qmap] + tb + list(extra) + [
            "--rule-log", os.path.join(out, "ticks", "rule_log_%s.npz" % name), "--rule-every", "5",
            "--readout", "causal_g", "--readout-log", os.path.join(out, "ticks", "readout_log_%s.json" % name), "--acc-gain", "0", "--path-exec", "--kappa-max", "0.2",
            "--preroll-steps", "0",
            "run", "--ckpt", ARGS.ckpt, "--tag", tag_of(name), "--tracks", ARGS.tracks, "--eval-cache", ARGS.eval_cache, "--eval-grid", GRID_SUB, "--eval-join", ARGS.eval_join,
            "--out-dir", out, "--replan-steps", str(M), "--nav", "deadreckon", "--scene", "logged", "--schedule", "periodic", "--limit", "causal",
            "--dump-ticks", os.path.join(out, "ticks"), "--dump-heads", "--no-analyze", "--device", ARGS.device]
        if TIMEOUT:
            cmd = [TIMEOUT, "1800"] + cmd
        return out, cmd

    def env_run():
        return dict(os.environ, DISTRULE_DIR=DISTRULE, CLOSEST_DIR=CLOSEST, PYTHONPATH=os.path.join(ARGS.tree, "stack") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")

    def run_one(name, form, table):
        out, cmd = run_cmd(name, form, table)
        if ARGS.dry:
            print("CMD: DISTRULE_DIR=%s CLOSEST_DIR=%s PYTHONPATH=%s %s" % (DISTRULE, CLOSEST, os.path.join(ARGS.tree, "stack"), " ".join(cmd)))
            return 0
        if os.path.exists(out):
            if ARGS.analyze_only:
                return 0
            raise SystemExit("OUTPUT-EXISTS " + out)
        os.makedirs(os.path.join(out, "ticks"))
        with open(os.path.join(out, "ticks", "run_%s.txt" % name), "w", encoding="utf-8") as fh:
            fh.write("arm %s form %s closing-table %s\n" % (name, form, table if table else "none"))
        with open(os.path.join(out, "t1.log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, env=env_run(), stdout=fh, stderr=subprocess.STDOUT).returncode
        print("RUN %s form=%s table=%s rc=%d" % (name, form, os.path.basename(table) if table else "none", rc), flush=True)
        return rc

    if not ARGS.analyze_only:
        for name, form, table in RUNS:
            rc = run_one(name, form, table)
            if rc != 0:
                print("ZZ CHECK-SHUFCLOSE FAIL run-%s-rc-%d (see %s)" % (name, rc, os.path.join(W, name, "t1.log")))
                sys.exit(1)
    elif ARGS.dry:
        for name, form, table in RUNS:
            run_one(name, form, table)

    def refusal(label, form, table, extra):
        out = os.path.join(W, "refusal_" + label)
        _, cmd = run_cmd("refusal_" + label, form, table, out=out, extra=extra)
        if TIMEOUT:
            cmd[1] = "600"
        path = os.path.join(W, "refusal_%s.log" % label)
        if ARGS.dry:
            print("CMD: " + " ".join(cmd))
            return None, ""
        if os.path.exists(path) and ARGS.analyze_only:
            return None, open(path, encoding="utf-8").read()
        cp = subprocess.run(cmd, env=env_run(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = cp.stdout.decode("utf-8", "replace")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return cp.returncode, text

    ref_a = refusal("tablenoRRe", "RR", T_ZERO, ())
    ref_b = refusal("both", "RRe", T_ZERO, ["--closing-est", ARGS.est_file])
    ref_c = refusal("none", "RRe", None, ())
    if ARGS.dry:
        print("ZZ CHECK-SHUFCLOSE dry run: nothing executed beyond part A")
        sys.exit(0)

    # ------------------------------------------------------------------------------------------------ analysis
    def load_npz(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def ticks_dir(name):
        return os.path.join(W, name, "ticks")

    def tfile(name, stem):
        return os.path.join(ticks_dir(name), stem % name)

    def banked_file(folder, stem):
        nm = os.path.basename(os.path.normpath(folder))
        return os.path.join(folder, "ticks", stem % nm)

    def dumps(name):
        d = os.path.join(W, name, "dump_" + tag_of(name))
        return {os.path.basename(f): load_npz(f) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))}

    def run_text(name):
        return open(os.path.join(ticks_dir(name), "run_%s.txt" % name), encoding="utf-8").read().splitlines()

    STEMS = [("ticks_%s_m%d_causal_heads.npz" % ("%s", M), "tick file"), ("rule_log_%s.npz", "rule log"), ("rule_log_%s_rd.npz", "rule-log side file"),
             ("rule_log_mid_%s.npz", "mid-look log"), ("rule_log_mid_%s_rd.npz", "mid-look side file"), ("p1_first_%s.npz", "first-plan file")]

    # ---- U0A
    BRL, BML, BP1 = banked_file(ARGS.banked_est, "rule_log_%s.npz"), banked_file(ARGS.banked_est, "rule_log_mid_%s.npz"), banked_file(ARGS.banked_est, "p1_first_%s.npz")
    RL_I, ML_I = load_npz(tfile("sk_ident", "rule_log_%s.npz")), load_npz(tfile("sk_ident", "rule_log_mid_%s.npz"))
    if all(os.path.isfile(p) for p in (BRL, BML, BP1)):
        ok, d = analyze_u0a(RL_I, ML_I, load_npz(tfile("sk_ident", "p1_first_%s.npz")), load_npz(BRL), load_npz(BML), load_npz(BP1), ARGS.tol_banked)
        COUNTS.update(u0a_replan_common=d["replan"]["rows_common"], u0a_mid_common=d["mid"]["rows_common"], u0a_plans_common=d["plans"]["plans_common"])
        check("U0A sk_ident against the banked run %s on the subset rows: %s (bar %.0e)" % (os.path.basename(os.path.normpath(ARGS.banked_est)), json.dumps(d), ARGS.tol_banked), ok)
    else:
        check("U0A banked files exist", False, str([p for p in (BRL, BML, BP1) if not os.path.isfile(p)]))

    # ---- U0B
    DR, DZ = dumps("sk_rr"), dumps("sk_zero")
    FR = {lab: (load_npz(tfile("sk_rr", st)) if os.path.isfile(tfile("sk_rr", st)) else None) for st, lab in STEMS}
    FZ = {lab: (load_npz(tfile("sk_zero", st)) if os.path.isfile(tfile("sk_zero", st)) else None) for st, lab in STEMS}
    RL_Z, ML_Z = load_npz(tfile("sk_zero", "rule_log_%s.npz")), load_npz(tfile("sk_zero", "rule_log_mid_%s.npz"))
    RL_R, ML_R = load_npz(tfile("sk_rr", "rule_log_%s.npz")), load_npz(tfile("sk_rr", "rule_log_mid_%s.npz"))
    ok, d = analyze_u0b(DR, DZ, FR, FZ, RL_Z, ML_Z, RL_R, ML_R)
    COUNTS.update(zero_tick_rows=d["tick_rows"], zero_mid_rows=d["mid_rows"], zero_files=d["files"])
    check("U0B sk_zero equals sk_rr: largest difference %.3g (%d files); c_src is 0 on all %d + %d rows; new keys %s" % (d["largest_diff"], d["files"], d["tick_rows"], d["mid_rows"], json.dumps(d["new_keys"])),
          ok, json.dumps(d["problems"]))

    # ---- U0C
    RL_2, ML_2 = load_npz(tfile("sk_c2", "rule_log_%s.npz")), load_npz(tfile("sk_c2", "rule_log_mid_%s.npz"))
    ok, d = analyze_u0c(RL_2, ML_2)
    check("U0C sk_c2: %s" % json.dumps(d), ok)

    # ---- U0L
    L1, ok_l = {}, True
    for name, tb in (("sk_ident", T_IDENT), ("sk_zero", T_ZERO), ("sk_c2", T_C2)):
        RLg, MLg = load_npz(tfile(name, "rule_log_%s.npz")), load_npz(tfile(name, "rule_log_mid_%s.npz"))
        lines = [x for x in run_text(name) if x.startswith("est looks")]
        sha = hashlib.sha256(open(tb, "rb").read()).hexdigest()
        good, dd = analyze_line(lines, RLg, MLg, sha)
        L1[name] = dd
        ok_l = ok_l and good
    no_line = not [x for x in run_text("sk_rr") if x.startswith("est looks")]
    L1["sk_rr_has_no_line"] = no_line
    check("U0L run line counts equal the counts recomputed from the logs, file sha256 equals the table file: %s" % json.dumps(L1), ok_l and no_line)

    # ---- U0F
    okf = analyze_refusal(ref_a[0], ref_a[1], ["--closing-table"]) and analyze_refusal(ref_b[0], ref_b[1], ["--closing-est", "--closing-table"]) and analyze_refusal(ref_c[0], ref_c[1], ["--closing-table"])
    check("U0F refusals: RR with --closing-table (rc %s), RRe with both flags (rc %s), RRe with neither (rc %s) stop with REFUSED naming the flags" % (ref_a[0], ref_b[0], ref_c[0]), okf,
          " | ".join(x[1].strip()[-100:] for x in (ref_a, ref_b, ref_c)))

    print("counts " + json.dumps(COUNTS, sort_keys=True))
    print("ZZ CHECK-SHUFCLOSE " + ("PASS checks-failed 0 " + "tick-rows %d mid-rows %d" % (COUNTS["zero_tick_rows"], COUNTS["zero_mid_rows"]) if not FAILED else "FAIL " + "; ".join(FAILED)))
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
