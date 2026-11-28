"""dscale_lead_read_n5: note N5 of card ADD45 around the unchanged reader and the wrapper of note N4.
  python dscale_lead_read_n5.py --k2-tol 1e-3 <every other argument of dscale_lead_read.py>
  python dscale_lead_read_n5.py --selftest
Control d4 (p5_u) of a probe passes if it passes as frozen, or if every other clause holds as frozen and the gap clause holds on full-batch records: a logged record is a full-batch record if its n_pres is
at least SHORT_FRAC times the median n_pres of the run's logged records; every full-batch record has n_gap above 0; the short records are at most SHORT_SHARE_MAX of the logged records.
Nothing else changes; the frozen verdict is kept beside (pass_as_frozen)."""
import json
import os
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dscale_lead_read as D  # noqa: E402
import dscale_lead_read_n4 as N4  # noqa: E402

SHORT_FRAC, SHORT_SHARE_MAX = 0.5, 0.05
_FROZEN_P5_U = D.p5_u


def gap_clause_n5(body_all):
    if not body_all:
        return {"pass": False, "reason": "no logged record"}
    med = float(statistics.median(float(x["n_pres"]) for x in body_all))
    short = [x for x in body_all if float(x["n_pres"]) < SHORT_FRAC * med]
    full = [x for x in body_all if float(x["n_pres"]) >= SHORT_FRAC * med]
    min_full = min(float(x["n_gap"]) for x in full) if full else None
    share = len(short) / float(len(body_all))
    ok = bool(full and min_full is not None and min_full > 0 and share <= SHORT_SHARE_MAX)
    return {"pass": ok, "median_n_pres": med, "short_below_n_pres": SHORT_FRAC * med, "n_records": len(body_all), "n_short": len(short), "short_share": share,
            "short_share_max": SHORT_SHARE_MAX, "short_steps": [int(x["step"]) for x in short][:40], "n_short_without_gap_label": sum(1 for x in short if float(x["n_gap"]) <= 0),
            "min_n_gap_full_batches": min_full}


def p5_u_n5(path, sh, expect_total, expect_clip_kept=None, window=1000):
    r = _FROZEN_P5_U(path, sh, expect_total, expect_clip_kept, window)
    _, body_all = D._read_log(path)
    g = gap_clause_n5(body_all)
    fi, la = r.get("first_mean"), r.get("last_mean")
    other = bool(fi is not None and la is not None and la < fi and r.get("last_step") is not None and r.get("last_step") == r.get("total") and r.get("header_checks") and all(r["header_checks"].values()))
    out = dict(r)
    out["pass_as_frozen"] = bool(r["pass"])
    out["note_n5_other_clauses"] = other
    out["note_n5_gap_clause"] = g
    out["pass"] = bool(r["pass"] or (other and g["pass"]))
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    assert D.p5_u is _FROZEN_P5_U or D.p5_u is p5_u_n5
    D.p5_u = p5_u_n5
    rc = N4.main(argv)
    out_dir, _ = N4._argv_pop(argv, "--out-dir")
    if out_dir:
        jp, mp = out_dir + "/" + D.OUT_JSON, out_dir + "/" + D.OUT_MD
        if os.path.exists(jp):
            r = json.load(open(jp, encoding="utf-8"))
            d4 = {k: v for k, v in r.get("controls", {}).items() if k.startswith("d4_train_log")}
            r["d4_note_n5"] = {"short_frac": SHORT_FRAC, "short_share_max": SHORT_SHARE_MAX, "n_probes": len(d4),
                               "n_pass_as_frozen": sum(1 for v in d4.values() if (v.get("detail") or {}).get("pass_as_frozen")),
                               "n_pass_under_note": sum(1 for v in d4.values() if v.get("pass"))}
            json.dump(r, open(jp, "w", encoding="utf-8"), indent=1)
        if os.path.exists(mp):
            with open(mp, "a", encoding="utf-8") as f:
                f.write("\ncontrol d4 read under note N5 of card ADD45 (gap clause on full-batch records; short below %r of the median n_pres, at most %r of the records)\n" % (SHORT_FRAC, SHORT_SHARE_MAX))
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


def _recs(total, every=50, fall=True, short_at=(), short_gap=0.0, full_zero_at=(), stop_early=False):
    out, n = [], total // every
    for i in range(1, n + 1):
        st = i * every
        if stop_early and i == n:
            break
        lvl = (0.4 - 0.35 * st / total) if fall else 0.2
        x = {"step": st, "first": False, "lead_sup_bce_mean": lvl / 2, "lead_sup_gap_mean": lvl / 2, "lead_sup_clos_mean": 0.0, "n_pres": 670.0 + (i % 7), "n_gap": 200.0 + (i % 11), "n_clos": 0.0}
        if st in short_at:
            x["n_pres"], x["n_gap"] = 12.0, short_gap
        if st in full_zero_at:
            x["n_gap"] = 0.0
        out.append(x)
    return out


def selftest():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="n5_selftest_") as td:
        return _selftest_in(td)


def _selftest_in(td):
    total, res = 20000, {}

    def run(name, recs, **kw):
        p = os.path.join(td, name + ".jsonl")
        _write_log(p, total, recs, clip_kept=1143)
        fr = _FROZEN_P5_U(p, False, total, 1143)
        n5 = p5_u_n5(p, False, total, 1143)
        res[name] = (bool(fr["pass"]), bool(n5["pass"]), n5["note_n5_gap_clause"].get("n_short"), n5["note_n5_gap_clause"].get("n_short_without_gap_label"))
        return res[name]

    ok = True
    ok &= run("clean", _recs(total))[:2] == (True, True)
    ok &= run("one_short_no_gap", _recs(total, short_at=(8950,)))[:3] == (False, True, 1)
    ok &= run("short_with_gap", _recs(total, short_at=(8950,), short_gap=3.0))[:2] == (True, True)
    ok &= run("planted_full_zero", _recs(total, full_zero_at=(9000,)))[:2] == (False, False)
    ok &= run("planted_full_zero_and_short", _recs(total, short_at=(8950,), full_zero_at=(9000,)))[:2] == (False, False)
    many = tuple(range(50, total + 1, 150))[: int(0.30 * (total // 50))]
    ok &= run("planted_30pct_short", _recs(total, short_at=many))[:2] == (False, False)
    ok &= run("planted_no_fall", _recs(total, fall=False, short_at=(8950,)))[:2] == (False, False)
    ok &= run("planted_incomplete", _recs(total, short_at=(8950,), stop_early=True))[:2] == (False, False)
    for k, v in res.items():
        print("selftest %-28s frozen %-5s note_n5 %-5s n_short %s without_gap %s" % (k, v[0], v[1], v[2], v[3]))
    print("ZZ N5-SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
