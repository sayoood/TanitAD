"""tswap_result: the READ and the RESULT of PREREG_V8F_TSWAP.md (T1-kin, eval-141, scene `reindexed`), the card's outcomes applied mechanically.
Instrument = the pinned T1-kin reader's (lf_read.py -> llr_read.py -> the RECOVER-NEXT reader, loaded BY FILE PATH, as ../t1_blind/t1blind_read.py): arrays per rollout through `rr.extract`, the 19 rows of a paired contrast by
`rr.rows_of` (gapdecomp.decompose, paired episode-cluster bootstrap B 2,000 seed 0), P10 = the `b_` part of the same arrays. Contrasts per seed: TSWAP - OWN, TSWAP - P10, OWN - P10. STRATEGIC: PAUSED (PI item 25), listed beside.
Outcomes (card): T-CARRIES | T-PARTIAL | T-NOT | INVALID; `outcome()` is a pure function of the three M1 intervals per seed and of the controls.
T share = (M1(OWN) - M1(TSWAP)) / (M1(OWN) - M1(P10)): per-window M1 of the three arms on the windows where all three are finite, one episode-cluster draw (gapdecomp.cached_draws) per replicate for numerator and denominator.
usage: tswap_result.py --out-root F:/.../tswap/out --work-dir F:/.../tswap/work --raw <pk>/raw --out-dir <pk> [--gates-dir <raw>] ;  --subset: REHEARSAL on S_OWN/S_TSWAP (banner SUBSET, writes SUBSET_* files only)"""
import os
import sys
import json
import time
import argparse
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
import numpy as np

ART = "E:/Projects/TanitAD-artifacts"
LF_PATH = f"{ART}/dagland/motion_c/lf_read/code/lf_read.py"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
M1, ADE = "M1 crawl share", "ADE6"
ORDER = ("PRIMARY", "crawl", "LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
SEEDS = (0, 1)


# ---------------------------------------------------------------------------------------------- the card's rules (pure)
def m1_better(r):                    # M1 crawl share: lower is better (gapdecomp.sep_worse: every row but TAC is lower-better); separated and below 0
    return bool(r["hi"] < 0)


def m1_worse(r):
    return bool(r["lo"] > 0)


def outcome(facts, controls_pass):
    """facts = {seed: {"ts_vs_own": M1 row, "ts_vs_p10": M1 row, "own_vs_p10": M1 row}} (each {delta, lo, hi}). The card's section 'Outcomes, committed now'."""
    if not controls_pass:
        return "INVALID"
    better_both = all(m1_better(f["ts_vs_own"]) for f in facts.values())
    excess_removed = all(not m1_worse(f["ts_vs_p10"]) for f in facts.values())
    if better_both and excess_removed:
        return "T-CARRIES"
    if better_both:
        return "T-PARTIAL"
    return "T-NOT"


NEXT = {"T-CARRIES": "T trained on its own states, the DT2 / RECOVER recipe on T's labels, under its own card.",
        "T-PARTIAL": "The T share is reported. Next: T trained on its own states, AND the twin read on O, each under its own card.",
        "T-NOT": "The crawl does not sit in T's own-state decision. Next: the twin read on O (O fed the logged ego history), under its own card.",
        "INVALID": "A control failed: no row is read as a result."}


def t_share(a_own, a_ts, b_p10, eid, n_boot, GD, seed=0):
    """Bootstrap ratio (M1(OWN) - M1(TSWAP)) / (M1(OWN) - M1(P10)) over the windows where all three are finite. -> dict."""
    a_own, a_ts, b_p10 = (np.asarray(x, np.float64) for x in (a_own, a_ts, b_p10))
    ok = np.isfinite(a_own) & np.isfinite(a_ts) & np.isfinite(b_p10)
    ep = np.asarray([str(x) for x in eid])
    uniq = np.unique(ep[ok])
    ei = np.searchsorted(uniq, ep[ok])
    num, den = a_own[ok] - a_ts[ok], a_own[ok] - b_p10[ok]
    Sn, Sd, N = (np.bincount(ei, weights=w, minlength=len(uniq)) for w in (num, den, np.ones(ok.sum())))
    M = GD.cached_draws(uniq, n_boot, seed)
    Nb = M @ N
    nb, db = (M @ Sn) / Nb, (M @ Sd) / Nb
    with np.errstate(divide="ignore", invalid="ignore"):
        rb = nb / db
    f = rb[np.isfinite(rb)]
    lo, hi = (float(x) for x in np.percentile(f, [2.5, 97.5])) if len(f) else (None, None)
    return {"share": float(num.mean() / den.mean()), "lo": lo, "hi": hi, "numerator_mean_m1": float(num.mean()), "denominator_mean_m1": float(den.mean()), "n_windows": int(ok.sum()),
            "n_episodes": int(len(uniq)), "n_draws_denominator_nonpositive": int((db <= 0).sum()), "n_boot": int(n_boot), "estimator": "episode-cluster bootstrap ratio, windows with all three arms finite"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def status_of(SR, GD, lab, r):
    return "better" if SR.better(GD, lab, r) else ("worse" if SR.worse(GD, lab, r) else ("separated" if (r["lo"] > 0 or r["hi"] < 0) else "n.s."))


def read(a):
    LF = _load("lf_read_for_tswap", LF_PATH)
    LF.init()
    rr, SR, GD = LF.rr, LF.SR, LF.GD
    rr.PK = HERE
    rr.WORK = f"{a.work_dir}/rig"
    arr_dir = f"{a.work_dir}/arrays"
    seeds = (0,) if a.subset else SEEDS
    names = {"own": "S_OWN" if a.subset else "OWN_s{s}", "ts": "S_TSWAP" if a.subset else "TSWAP_s{s}"}
    gates = {}
    if not a.subset:
        for s in seeds:
            g = json.load(open(f"{a.gates_dir}/GATES_s{s}.json"))
            gates[f"s{s}"] = g
    controls_pass = bool(a.subset or all(g["PASS"] for g in gates.values()))
    res = {"card": "PREREG_V8F_TSWAP.md", "subset_rehearsal": bool(a.subset), "tier": "T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2). TSWAP feeds the logged state (privileged): a diagnostic.",
           "estimator": f"paired episode-cluster bootstrap B={a.n_boot} seed 0 (gapdecomp.decompose) over the windows' episodes; answers another draw of EPISODES; inference is deterministic at a fixed thread count; the training-run question is answered by the two seeds",
           "strategic_family": "PAUSED (PI item 25): rows listed beside, never decisive", "controls_PASS": controls_pass, "gates": gates, "reader_sha256": LF.sha256_file(__file__), "lf_read_sha256": LF.sha256_file(LF_PATH), "seeds": {}}
    if not controls_pass:
        res["outcome"] = "INVALID"
        res["next"] = NEXT["INVALID"]
        return res, None
    t0 = time.time()
    tags = {s: {k: v.format(s=s) for k, v in names.items()} for s in seeds}
    for s in seeds:
        for k in ("own", "ts"):
            if not a.subset:                                              # the pinned extractor needs all 141 episodes: the subset rehearsal reads pre-built arrays from arr_dir (make_rehearsal_arrays.py)
                rr.extract(tags[s][k], s, a.out_root, arr_dir)
    facts, stat_rows = {}, {}
    for s in seeds:
        A = {k: rr.read_arrays(f"{arr_dir}/{tags[s][k]}.npz") for k in ("own", "ts")}
        eid = A["own"]["eid"]
        for k in ("eid", "ws", "v0"):
            assert np.array_equal(A["ts"][k], A["own"][k]), f"{k} differs between the arms"
        p10_same = all(np.array_equal(A["ts"][f"b_{i}"], A["own"][f"b_{i}"], equal_nan=True) for i in range(len(GD.LABELS)))
        cls, _ = GD.classify(A["own"]["v0"], A["own"]["g"])
        c = {}
        c["TSWAP_minus_OWN"], c["M1_per_class_split_TSWAP_minus_OWN"] = rr.rows_of(rr.Aof(A["ts"]), rr.Aof(A["own"]), eid, cls, a.n_boot, M1)
        c["TSWAP_minus_P10"] = rr.rows_of(rr.Aof(A["ts"]), rr.Bof(A["ts"]), eid, cls, a.n_boot)[0]
        c["OWN_minus_P10"] = rr.rows_of(rr.Aof(A["own"]), rr.Bof(A["own"]), eid, cls, a.n_boot)[0]
        i1 = list(GD.LABELS).index(M1)
        share = t_share(A["own"][f"a_{i1}"], A["ts"][f"a_{i1}"], A["own"][f"b_{i1}"], eid, a.n_boot, GD)
        excess_sep = bool(c["OWN_minus_P10"][M1]["lo"] > 0)
        f = {"ts_vs_own": c["TSWAP_minus_OWN"][M1], "ts_vs_p10": c["TSWAP_minus_P10"][M1], "own_vs_p10": c["OWN_minus_P10"][M1]}
        for k in ("ts_vs_own", "ts_vs_p10"):                              # the pure rules agree with the pinned reader's direction words
            assert m1_better(f[k]) == bool(SR.better(GD, M1, f[k])) and m1_worse(f[k]) == bool(SR.worse(GD, M1, f[k])), "outcome rules disagree with the reader's direction words"
        facts[s] = f
        st = {nm: {lab: status_of(SR, GD, lab, c[nm][lab]) for lab in c[nm]} for nm in ("TSWAP_minus_OWN", "TSWAP_minus_P10", "OWN_minus_P10")}
        res["seeds"][f"s{s}"] = {"tags": tags[s], "rows": {nm: c[nm] for nm in ("TSWAP_minus_OWN", "TSWAP_minus_P10", "OWN_minus_P10")}, "status": st, "M1_per_class_split_TSWAP_minus_OWN": c["M1_per_class_split_TSWAP_minus_OWN"],
                                 "M1_facts": f, "OWN_excess_separated_above_0": excess_sep, "p10_arrays_identical_in_both_arms": bool(p10_same),
                                 "T_share": share if excess_sep else {"reported": False, "reason": "OWN - P10 on M1 is not separated above 0 at this seed", "would_be": share},
                                 "ts_separated_worse_rows": sorted(l for l, v in st["TSWAP_minus_OWN"].items() if v == "worse"), "ts_separated_better_rows": sorted(l for l, v in st["TSWAP_minus_OWN"].items() if v == "better"),
                                 "family_of": {lab: SR.FAMILY[lab] for lab in c["TSWAP_minus_OWN"]}}
        print(f"s{s}: M1 TSWAP-OWN {f['ts_vs_own']['delta']:+.4f} [{f['ts_vs_own']['lo']:+.4f}, {f['ts_vs_own']['hi']:+.4f}]  TSWAP-P10 {f['ts_vs_p10']['delta']:+.4f} [{f['ts_vs_p10']['lo']:+.4f}, {f['ts_vs_p10']['hi']:+.4f}]  "
              f"OWN-P10 {f['own_vs_p10']['delta']:+.4f}  share {share['share']:+.3f}", flush=True)
    res["outcome"] = outcome(facts, controls_pass)
    res["next"] = NEXT[res["outcome"]]
    res["wall_s"] = time.time() - t0
    return res, facts


def fmt(r):
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def render(res):
    sub = res["subset_rehearsal"]
    md = [("# SUBSET REHEARSAL OF THE TSWAP RESULT WRITER: NOT A RESULT" if sub else "# RESULT_TSWAP (T1-kin, eval-141, scene `reindexed`)"), ""]
    if sub:
        md += ["**SUBSET: 10 episodes / 170 windows, arm S_OWN / S_TSWAP, seed 0 only. This file rehearses the writer; the outcome word below is NOT the card's outcome.**", ""]
    md += [f"**Outcome by the card: {res['outcome']}{' (subset rehearsal, not a result)' if sub else ''}.**  {res['tier']}", f"Estimator: {res['estimator']}.", "",
           "Controls (card): " + (("SUBSET: the gates are in raw/SUB_GATES.json" if sub else "all PASS") if res["controls_PASS"] else "NOT ALL PASS -> INVALID, no row is read"), ""]
    for s, g in res["gates"].items():
        md.append(f"* {s}: " + ", ".join(f"{k} {'PASS' if v['PASS'] else 'FAIL'}" for k, v in g.items() if k.startswith("G-")))
    md.append("")
    for s, d in res["seeds"].items():
        f = d["M1_facts"]
        sh = d["T_share"]
        md += [f"## {s}: the named rows", "", "| row | TSWAP - OWN [95 % CI] | status | TSWAP - P10 [CI] | status | OWN - P10 [CI] | status |", "|---|---|---|---|---|---|---|"]
        for lab in (M1, ADE):
            md.append(f"| {lab} | {fmt(d['rows']['TSWAP_minus_OWN'][lab])} | {d['status']['TSWAP_minus_OWN'][lab]} | {fmt(d['rows']['TSWAP_minus_P10'][lab])} | {d['status']['TSWAP_minus_P10'][lab]} | "
                      f"{fmt(d['rows']['OWN_minus_P10'][lab])} | {d['status']['OWN_minus_P10'][lab]} |")
        md.append("")
        md.append("T share: " + (f"{sh['share']:+.3f} [{sh['lo']:+.3f}, {sh['hi']:+.3f}] (numerator {sh['numerator_mean_m1']:+.4f}, denominator {sh['denominator_mean_m1']:+.4f}, {sh['n_windows']:,} windows, draws with non-positive denominator {sh['n_draws_denominator_nonpositive']})"
                                    if sh.get("reported", True) else f"not reported ({sh['reason']}); would be {sh['would_be']['share']:+.3f}"))
        md += ["", f"### {s}: all 19 rows", "", "| family | row | TSWAP - OWN [CI] | status | TSWAP - P10 [CI] | status | OWN - P10 status |", "|---|---|---|---|---|---|---|"]
        labs = sorted(d["rows"]["TSWAP_minus_OWN"], key=lambda l: (ORDER.index(d["family_of"][l]), list(d["rows"]["TSWAP_minus_OWN"]).index(l)))
        for lab in labs:
            fam = d["family_of"][lab] + (" (PAUSED, decides nothing)" if d["family_of"][lab] == "STRATEGIC" else "")
            md.append(f"| {fam} | {lab} | {fmt(d['rows']['TSWAP_minus_OWN'][lab])} | {d['status']['TSWAP_minus_OWN'][lab]} | {fmt(d['rows']['TSWAP_minus_P10'][lab])} | {d['status']['TSWAP_minus_P10'][lab]} | {d['status']['OWN_minus_P10'][lab]} |")
        md += ["", f"TSWAP separated-better than OWN on: {', '.join(d['ts_separated_better_rows']) or 'none'}.  TSWAP separated-worse than OWN on: {', '.join(d['ts_separated_worse_rows']) or 'none'}.", ""]
    md += ["## Next, committed by the card for this outcome", "", res["next"], "", "STRATEGIC family: PAUSED (PI item 25), listed, decides nothing. Mechanism read at T1-kin, not a driving claim; TSWAP feeds a privileged signal."]
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--raw", default=f"{HERE}/raw")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--gates-dir", default=f"{HERE}/raw")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--subset", action="store_true")
    a = ap.parse_args(argv)
    res, _ = read(a)
    pre = "SUBSET_" if a.subset else ""
    os.makedirs(a.raw, exist_ok=True)
    json.dump(res, open(f"{a.raw}/{pre}READ_TSWAP.json", "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(f"{a.out_dir}/{pre}RESULT_TSWAP.md", "w", encoding="utf-8", newline="\n").write(render(res))
    print("OUTCOME TSWAP:", res["outcome"], "(subset rehearsal, not a result)" if a.subset else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
