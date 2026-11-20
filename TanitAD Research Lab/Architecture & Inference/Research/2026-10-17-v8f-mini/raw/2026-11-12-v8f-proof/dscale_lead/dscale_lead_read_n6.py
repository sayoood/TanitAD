"""dscale_lead_read_n6: note N6 of card ADD45 around the unchanged reader and the wrappers of notes N4 and N5.
  python dscale_lead_read_n6.py --k2-tol 1e-3 <every other argument of dscale_lead_read.py>
  python dscale_lead_read_n6.py --selftest
Control d4 (p5_u) of a probe passes if it passes under note N5, or if every other clause holds (header checks, run complete, gap clause as frozen or under note N5) and the log holds the record of
step 1 and the lead term of the last 1,000 steps (in the form of the probe) is at most START_RATIO_MAX times the lead term at step 1.
Nothing else changes; the verdict under note N5 is kept beside (pass_under_note_n5)."""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dscale_lead_read as D  # noqa: E402
import dscale_lead_read_n4 as N4  # noqa: E402
import dscale_lead_read_n5 as N5  # noqa: E402

START_RATIO_MAX = 0.80
_FROZEN_P5_U = N5._FROZEN_P5_U


def start_clause_n6(body_all, sh, last_mean, window, total):
    first = [x for x in body_all if x.get("first") and int(x.get("step", -1)) == 1]
    keys = ("lead_sup_bce",) if sh else ("lead_sup_bce", "lead_sup_gap")
    s1 = None
    if first and all(k in first[0] for k in keys):
        s1 = float(sum(float(first[0][k]) for k in keys))
    ratio = (float(last_mean) / s1) if (s1 is not None and s1 > 0 and last_mean is not None) else None
    body = [x for x in body_all if not x.get("first")]
    gl = [float(x["lead_sup_gap_mean"]) for x in body if x["step"] > total - window and "lead_sup_gap_mean" in x]
    return {"pass": bool(ratio is not None and ratio <= START_RATIO_MAX), "lead_term_at_step_1": s1, "last_mean": last_mean, "ratio_last_to_step_1": ratio, "bar": START_RATIO_MAX,
            "gap_part_last_window_reported_not_judged": (sum(gl) / len(gl)) if gl else None}


def p5_u_n6(path, sh, expect_total, expect_clip_kept=None, window=1000):
    r = N5.p5_u_n5(path, sh, expect_total, expect_clip_kept, window)
    _, body_all = D._read_log(path)
    st = start_clause_n6(body_all, sh, r.get("last_mean"), window, int(r.get("total") or 0))
    other = bool(r.get("last_step") is not None and r.get("last_step") == r.get("total") and r.get("header_checks") and all(r["header_checks"].values())
                 and (r.get("note_n5_gap_clause") or {}).get("pass") is True)
    out = dict(r)
    out["pass_under_note_n5"] = bool(r["pass"])
    out["note_n6_other_clauses"] = other
    out["note_n6_start_clause"] = st
    out["pass"] = bool(r["pass"] or (other and st["pass"]))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    assert D.p5_u is _FROZEN_P5_U or D.p5_u is p5_u_n6
    D.p5_u = p5_u_n6
    rc = N4.main(argv)
    out_dir, _ = N4._argv_pop(argv, "--out-dir")
    if out_dir:
        jp, mp = out_dir + "/" + D.OUT_JSON, out_dir + "/" + D.OUT_MD
        if os.path.exists(jp):
            r = json.load(open(jp, encoding="utf-8"))
            d4 = {k: v for k, v in r.get("controls", {}).items() if k.startswith("d4_train_log")}
            det = lambda v: (v.get("detail") or {})                                              # noqa: E731
            r["d4_note_n5"] = {"short_frac": N5.SHORT_FRAC, "short_share_max": N5.SHORT_SHARE_MAX, "n_probes": len(d4),
                               "n_pass_as_frozen": sum(1 for v in d4.values() if det(v).get("pass_as_frozen")),
                               "n_pass_under_note": sum(1 for v in d4.values() if det(v).get("pass_under_note_n5"))}
            r["d4_note_n6"] = {"start_ratio_max": START_RATIO_MAX, "n_probes": len(d4), "n_pass_under_note_n5": sum(1 for v in d4.values() if det(v).get("pass_under_note_n5")),
                               "n_pass_under_note_n6": sum(1 for v in d4.values() if v.get("pass"))}
            json.dump(r, open(jp, "w", encoding="utf-8"), indent=1)
        if os.path.exists(mp):
            with open(mp, "a", encoding="utf-8") as f:
                f.write("\ncontrol d4 read under notes N5 and N6 of card ADD45 (N5: gap clause on full-batch records; N6: the lead term of the last 1,000 steps at most %r of the lead term at step 1, "
                        "where the clause on the first and last 1,000 steps does not hold)\n" % START_RATIO_MAX)
    return rc


def _write_log(path, total, recs, sh=False, clip_kept=None):
    hdr = {"header": True, "total": total, "w_clos": 0.0, "eval_guard": {"n_shared": 0, "n_eval_ids": 48}}
    if sh:
        hdr["shuffle_gap"] = {"seed": 1}
    if clip_kept is not None:
        hdr["clip_list"] = {"n_kept": clip_kept}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(hdr) + "\n")
        for x in recs:
            f.write(json.dumps(x) + "\n")


def _recs(total, every=50, kind="level", start=0.75, level=0.40, step1=True, stop_early=False, full_zero_at=()):
    """kind: 'fall' (keeps falling over the run, passes as frozen), 'level' (falls within the first 100 steps, then level with a small rise), 'dead' (stays at its start value)"""
    out, n = [], total // every
    if step1:
        out.append({"step": 1, "first": True, "lead_sup_bce": 0.92 * start, "lead_sup_gap": 0.08 * start, "lead_sup_bce_mean": 0.92 * start, "lead_sup_gap_mean": 0.08 * start, "lead_sup_clos_mean": 0.0,
                    "n_pres": 670.0, "n_gap": 200.0, "n_clos": 0.0})
    for i in range(1, n + 1):
        st = i * every
        if stop_early and i == n:
            break
        if kind == "fall":
            lvl = 0.45 - 0.30 * st / total
        elif kind == "dead":
            lvl = start
        else:
            lvl = (start if st <= 50 else level) + 0.02 * st / total
        x = {"step": st, "first": False, "lead_sup_bce_mean": 0.95 * lvl, "lead_sup_gap_mean": 0.05 * lvl, "lead_sup_clos_mean": 0.0, "n_pres": 670.0 + (i % 7), "n_gap": 200.0 + (i % 11), "n_clos": 0.0}
        if st in full_zero_at:
            x["n_gap"] = 0.0
        out.append(x)
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="n6_selftest_") as td:
        return _selftest_in(td)


def _selftest_in(td):
    total, res = 20000, {}

    def run(name, recs, sh=False, clip_kept=None):
        p = os.path.join(td, name + ".jsonl")
        _write_log(p, total, recs, sh=sh, clip_kept=clip_kept)
        fr = _FROZEN_P5_U(p, sh, total, clip_kept)
        n5 = N5.p5_u_n5(p, sh, total, clip_kept)
        n6 = p5_u_n6(p, sh, total, clip_kept)
        res[name] = (bool(fr["pass"]), bool(n5["pass"]), bool(n6["pass"]), n6["note_n6_start_clause"]["ratio_last_to_step_1"])
        return res[name][:3]

    ok = True
    ok &= run("falls_over_the_run", _recs(total, kind="fall")) == (True, True, True)
    ok &= run("level_after_early_fall", _recs(total)) == (False, False, True)
    ok &= run("planted_stays_at_start", _recs(total, kind="dead")) == (False, False, False)
    ok &= run("planted_ends_at_0.87_of_start", _recs(total, level=0.87 * 0.75 - 0.02)) == (False, False, False)
    ok &= run("planted_incomplete", _recs(total, stop_early=True)) == (False, False, False)
    ok &= run("planted_full_batch_without_gap", _recs(total, full_zero_at=(9000,))) == (False, False, False)
    ok &= run("planted_no_step_1_record", _recs(total, step1=False)) == (False, False, False)
    ok &= run("presence_only_level", _recs(total), sh=True) == (False, False, True)
    ok &= run("presence_only_stays_at_start", _recs(total, kind="dead"), sh=True) == (False, False, False)
    ok &= run("corpus_list_level", _recs(total), clip_kept=1143) == (False, False, True)
    r_lvl = res["level_after_early_fall"][3]
    ok &= bool(r_lvl is not None and 0.5 < r_lvl < 0.6)
    for k, v in res.items():
        print("selftest %-32s frozen %-5s note_n5 %-5s note_n6 %-5s ratio %s" % (k, v[0], v[1], v[2], "none" if v[3] is None else "%.3f" % v[3]))
    print("ZZ N6-SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
