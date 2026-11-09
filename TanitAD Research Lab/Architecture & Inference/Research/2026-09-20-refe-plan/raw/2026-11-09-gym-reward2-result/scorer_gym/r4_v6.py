"""R4 + G-REPRO of PREREG_REWARD_2 section 3, run in the chain BEFORE any state of seed 20261111 is drawn. Re-simulates the six-rung ladder (log mode, floor 0.20) of the 39 ELIGIBLE states of the
seed-20261110 list (USED; pilot_v5/PREREG_REWARD_1_STATES.json) through the real caller (run_v6.do_state -> GymState -> RewardEvaluator) with curve_min_seg_m = 0.05, and checks
 R4      : the verdicts reproduce CURVE_SPEED_DEFECT.md: 34 KNOWN, the not-KNOWN are exactly 31, 46, 121, 125, 134 (and the reference table, raw/resim, says the same).
 G-REPRO : crawl/best within 1 % of the table, and the 0.9x and 1.0x returns within 1 % (the table = pilot_v6/raw/resim/resim_<idx>.json, the exact re-simulation behind CURVE_SPEED_DEFECT.md).
 extra   : hard flags equal the stored M-GYM-5 hard flags on every rung (scripted rungs: they must not move).
usage: r4_v6.py OUT.json [--workers 4]
Prints R4_PASS or R4_FAIL; exit 0 / 1. Reads only used states."""
import gzip, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gym_env as g
import reward1 as r1
import reward2 as r2
import ramguard
import run_v6

P6, P5 = r2.P6, r2.P5
NOT_KNOWN_EXPECTED = (31, 46, 121, 125, 134)        # PREREG_REWARD_2 section 3, R4
REL = 0.01


def eligible_idx():
    res = json.load(open(os.path.join(P5, "raw", "results_v5.json"), encoding="utf-8"))
    return [e["idx"] for e in res["B3"]["eligible_detail"]]


def reference(idx):
    """the table: exact re-simulation rets of CURVE_SPEED_DEFECT.md"""
    d = json.load(open(os.path.join(P6, "raw", "resim", f"resim_{idx}.json"), encoding="utf-8"))
    assert d["arc"] == 0.05 and d["floor"] == 0.20
    return {k: v["ret"] for k, v in d["ladder"].items()}, {k: bool(v["hard"]) for k, v in d["ladder"].items()}


_STORED = {}


def stored_hard(idx):
    if not _STORED:
        for sub in ("run_main", "ladder_20_160"):
            for l in gzip.open(os.path.join(P5, "raw", sub, "states.jsonl.gz"), "rt", encoding="utf-8"):
                r = json.loads(l)
                _STORED[r["idx"]] = {n: any(c["hard"] for c in v["logs"]) for n, v in r["ladder"].items()}
    return _STORED[idx]


def best(r):
    return max(r["f0.9"], r["f1.0"])


def judge(idx, rets, hards):
    """pure comparison of one re-simulated state against the table (tests mutate it)"""
    ref, ref_h = reference(idx)
    inf, kn = g.ladder_known(rets, hards)
    rinf, rkn = g.ladder_known(ref, ref_h)
    c, cref = rets["crawl1.5"] / best(rets), ref["crawl1.5"] / best(ref)
    e_crawl = abs(c - cref) / abs(cref)
    e_09, e_10 = abs(rets["f0.9"] - ref["f0.9"]) / abs(ref["f0.9"]), abs(rets["f1.0"] - ref["f1.0"]) / abs(ref["f1.0"])
    return dict(idx=idx, informative=inf, known=kn, ref_informative=rinf, ref_known=rkn, crawl_over_best=c, ref_crawl_over_best=cref,
                err_crawl_over_best=e_crawl, err_f09=e_09, err_f10=e_10, bit_identical_all_rungs=all(rets[k] == ref[k] for k in ref),
                hard_equal_stored=(hards == stored_hard(idx)), gate_crawl=bool(e_crawl <= REL), gate_f09_f10=bool(e_09 <= REL and e_10 <= REL),
                verdict_equals_table=bool(inf == rinf and kn == rkn))


def decide(ok_rows, errs):
    """(R4, G_REPRO) from the judged rows; pure so a test can feed it perturbed rows"""
    not_known = sorted(r["idx"] for r in ok_rows if not r["known"])
    r4 = bool(not errs and len(ok_rows) == 39 and sum(bool(r["known"]) for r in ok_rows) == 34 and not_known == list(NOT_KNOWN_EXPECTED) and all(r["verdict_equals_table"] for r in ok_rows))
    gr = bool(not errs and len(ok_rows) == 39 and all(r["gate_crawl"] and r["gate_f09_f10"] for r in ok_rows) and all(r["hard_equal_stored"] for r in ok_rows))
    return r4, gr


def main():
    out_path = sys.argv[1]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    t0 = time.time()
    sts = r1.load_states_v5()
    el = eligible_idx()
    assert len(el) == 39
    rows, errs = {}, []

    def on_result(r, task):
        if "error" in r:
            errs.append((r["idx"], r["error"]))
            return
        assert r["curve_min_seg_m"] == r2.CURVE_MIN_SEG_M
        rets = {n: v["ret"] for n, v in r["ladder"].items()}
        hards = {n: any(x["hard"] for x in v["logs"]) for n, v in r["ladder"].items()}
        rows[r["idx"]] = dict(judge(r["idx"], rets, hards), rets=rets, limit=r["limit"])
        print(time.strftime("%H:%M:%S"), "state", r["idx"], "done", flush=True)

    ramguard.run_tasks(run_v6.do_state, [(i, sts[i], "ladder", r1.FLOOR, r2.CURVE_MIN_SEG_M) for i in el], workers, on_result,
                       label_fn=lambda t: f"R4 used state {t[0]}", log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))
    ok_rows = [rows[i] for i in el if i in rows]
    not_known = sorted(r["idx"] for r in ok_rows if not r["known"])
    res = dict(prereg="PREREG_REWARD_2.md section 3 R4 / G-REPRO", curve_min_seg_m=r2.CURVE_MIN_SEG_M, floor=r1.FLOOR, wall_s=time.time() - t0, n_eligible=len(el),
               n_run=len(ok_rows), errors=errs, n_known=sum(bool(r["known"]) for r in ok_rows), not_known_idx=not_known, not_known_expected=list(NOT_KNOWN_EXPECTED),
               max_err_crawl_over_best=max(r["err_crawl_over_best"] for r in ok_rows) if ok_rows else None,
               max_err_f09=max(r["err_f09"] for r in ok_rows) if ok_rows else None, max_err_f10=max(r["err_f10"] for r in ok_rows) if ok_rows else None,
               n_bit_identical=sum(r["bit_identical_all_rungs"] for r in ok_rows), per_state=ok_rows)
    res["R4"], res["G_REPRO"] = decide(ok_rows, errs)
    res["PASS"] = bool(res["R4"] and res["G_REPRO"])
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("n_run", "n_known", "not_known_idx", "max_err_crawl_over_best", "max_err_f09", "max_err_f10", "n_bit_identical", "R4", "G_REPRO", "PASS")}))
    print("R4_PASS" if res["PASS"] else "R4_FAIL", flush=True)
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
