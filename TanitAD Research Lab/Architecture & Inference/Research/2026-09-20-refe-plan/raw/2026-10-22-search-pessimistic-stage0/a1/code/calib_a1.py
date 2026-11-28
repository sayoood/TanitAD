#!/usr/bin/env python3
"""AMENDMENT A1 calibration: how many seed models S must each fold model average for the Stage 0 instrument to be valid?

REAL FEATURES, SYNTHETIC LABELS ONLY.  The real world is loaded for its FEATURES and immediately stripped of its real labels (`blind_world`: every label array is replaced by ones, and `assert_blind` is checked in
every process), so no IN / OOF / concordance / regret / gap can be computed on a real label.  Every label that is scored here comes from `stage0.plant_labels`, the C-II-b generator.

For each S in S_LIST (1, 3, 5, 8):
  * the four PLANTED worlds kappa in {0.5, 1, 2, 4}, generator seed 7771 (exactly the sealed run's C-II-b arms): the planted arm and its exclusion-deleted twin, seed-averaged over S models per fold;
  * 24 NULL worlds (kappa = 0, generator seeds 8001..8024, same features): the primary arm, the seed-replicate arm, the exclusion-deleted arm, C-I (evaluation x100 and trained x10), C-II-a, C-III-real, and the
    decision rule `stage0.decide`, i.e. what the real read would conclude about a world in which NO session effect exists.  (R1, the descriptive secondary arm, never enters the decision and is not run.)
C-II-b does not depend on any label (the planted worlds are built from the features alone), so it is the SAME 4 planted worlds for all 24 null worlds: it is computed once per S and joined to each null world's
decision afterwards.  The null-world INVALID rate is therefore reported EXCLUDING C-II-b (a failed C-II-b would otherwise make 24 of 24 worlds INVALID, a single event counted 24 times; it is gated separately by
the selection rule's condition (a)).

THE SELECTION RULE (committed in DRAFT_AMENDMENT_A1.md before this script was launched; implemented once, in `apply_selection_rule`):
  choose the SMALLEST S in {3, 5, 8} such that
    (a) C-II-b passes at some kappa <= 2,
    (b) the null-world false-positive count is <= 2 of 24   (FP = the full rule reads GAP on a kappa = 0 world, with C-II-b taken as passed),
    (c) the null-world INVALID count is <= 4 of 24          (INVALID by any control other than C-II-b).
  If no S qualifies the outcome is INSTRUMENT-LIMITED: no real read.

    CUDA_VISIBLE_DEVICES=-1 python calib_a1.py --workers 16 --out ../raw/calib_a1.json         (last line printed: ZZCALIB_DONE)
    CUDA_VISIBLE_DEVICES=-1 python calib_a1.py --identity-only                                  (S = 1, the four planted worlds, compared with the sealed run's excerpt)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage0 as S0  # noqa: E402

# ====================================================================================================================== the committed design
S_LIST = (1, 3, 5, 8)                     # S = 1 is the identity check and the baseline; only S_CANDIDATES can be selected
S_CANDIDATES = (3, 5, 8)
KAPPAS = tuple(S0.PLANT_KAPPAS)           # (0.5, 1, 2, 4), seed S0.PLANT_SEED = 7771: the sealed run's planted worlds
KAPPA_MAX_FOR_SELECTION = 2.0             # condition (a): C-II-b must pass at some kappa <= this
NULL_SEEDS = tuple(range(8001, 8025))     # 24 kappa = 0 worlds
FP_MAX = 2                                # condition (b): false positives of the full rule over the 24 null worlds
INVALID_MAX = 4                           # condition (c): INVALID (any control but C-II-b) over the 24 null worlds
SEALED_EXCERPT = os.path.join(HERE, "..", "raw", "RESULT_STAGE0_controls_only.json")
IDENTITY_TOL = 1e-12                      # the sealed run and this one are the same float pipeline on the same machine: the expectation is exactly 0.0


# ====================================================================================================================== the label-blind real world
def blind_world(root: str | None = None, verify_sha: bool = True):
    """The REAL features (sha256-verified like the sealed read) with every label replaced by ones.  Nothing downstream can score a real label."""
    w, man = S0.load_world(root or S0.ARTIFACTS, verify_sha=verify_sha)
    dummy = S0.Labels(np.ones((w.N, w.K, 6)), np.ones((w.N, w.K)), np.ones((w.N, w.K), bool), np.ones((w.Ns, w.Q, 6)), np.ones((w.Ns, w.Q)), np.ones((w.Ns, w.Q), bool))
    wb = w.with_labels(dummy)
    wb.meta = {**wb.meta, "label_blind": True}
    assert_blind(wb)
    return wb, man


def assert_blind(w) -> None:
    L = w.labels
    if not (np.all(L.sub == 1.0) and np.all(L.pdms == 1.0) and np.all(L.p_sub == 1.0) and np.all(L.p_pdms == 1.0) and bool(w.meta.get("label_blind"))):
        raise RuntimeError("A1-BLIND: the calibration world carries labels that are not the constant dummy: refusing to compute anything on it")


class LazyShuffles:
    """The within-token label shuffles of C-I (evaluation), generated ONE AT A TIME.  `stage0.control_shuffle` builds a list of all 100 label sets (about 100 x 8.5 MB = 850 MB), which would cap the number of worker
    processes on this box; `chance_check` only needs `len()` and a single pass, so a lazy sequence yields exactly the same (cache, labels) pairs in the same order.  Pinned equal to `control_shuffle` by a test."""

    def __init__(self, w, cache, seeds=None):
        self.w, self.cache, self.seeds = w, cache, tuple(S0.SHUF_SEEDS if seeds is None else seeds)

    def __len__(self):
        return len(self.seeds)

    def __iter__(self):
        for sd in self.seeds:
            yield self.cache, S0.shuffled_labels(self.w, sd)


# ====================================================================================================================== one world, the decision parts (mirrors stage0.run_all steps 1, 2, 4, 5 and 7)
def decision_parts(wl, d, idx, prim, repl) -> dict:
    """Everything `run_all` computes for a labelled world EXCEPT C-II-b and the descriptive R1 arm.  `wl` carries the labels to be scored (here always synthetic).  Returns the ingredients of `stage0.decide`."""
    t = {}
    t0 = time.time()
    a0 = S0.run_arm(wl, d, S0.R0, prim, idx)
    t["a0"] = time.time() - t0
    t0 = time.time()
    a1 = S0.run_arm(wl, d, S0.R0, repl, idx)
    t["a1"] = time.time() - t0
    t0 = time.time()
    am = S0.run_arm(wl, d, S0.R0, prim, idx, fit=S0.fit_all_unexcluded(wl, d, S0.R0, prim))
    t["am"] = time.time() - t0
    t0 = time.time()
    c_shuf = S0.chance_check(wl, d, LazyShuffles(wl, a0["cache"]))                      # == S0.control_shuffle(wl, d, a0["cache"]) with the label sets streamed
    t["c_shuf"] = time.time() - t0
    t0 = time.time()
    c_shuf_world = S0.control_shuffle_world(wl, d, S0.R0, idx, train_seeds=prim)
    t["c_shuf_world"] = time.time() - t0
    cc = S0.control_collapse(a0, am)
    controls = {
        "C-STRUCT": {"pass": all(x["own_fold_tokens"] == 0 and x["search_tokens"] == 0 and x["own_fold_sessions"] == 0 for x in a0["fit"]["audit"] + a1["fit"]["audit"])},
        "C-I-eval": {"pass": c_shuf["pass"], "failed": [k for k, v in c_shuf["checks"].items() if not v], "uninformative": c_shuf["uninformative"]},
        "C-I-trained": {"pass": c_shuf_world["pass"], "failed": [k for k, v in c_shuf_world["checks"].items() if not v], "uninformative": c_shuf_world["uninformative"]},
        "C-II-a": S0.control_seen(a0),
        "C-III-real": {"pass": cc["pass"], "M1": cc["M1"]},
    }
    return {"primary": a0["gaps"], "replicate": a1["gaps"], "swap": a0["swap_null"], "controls": controls, "certified": {"M3": bool(c_shuf["informative"]["M3"])},
            "levels": a0["levels"], "seen_minus_oof_M1": a0["seen_minus_oof"]["M1"]["point"], "seconds_parts": t}


def planted_parts(w, d, idx, prim, kappa: float) -> dict:
    """The sealed run's C-II-b arm for one kappa: planted labels on the real features, the normal arm and its exclusion-deleted twin, `control_planted`'s verdict."""
    t0 = time.time()
    wp = w.with_labels(S0.plant_labels(w, kappa, seed=S0.PLANT_SEED))
    a = S0.run_arm(wp, d, S0.R0, prim, idx)
    m = S0.run_arm(wp, d, S0.R0, prim, idx, fit=S0.fit_all_unexcluded(wp, d, S0.R0, prim))
    cp = S0.control_planted(a, m)
    g = a["gaps"]["M1"]
    return {"control": cp, "M1": {"point": g["point"], "se": g["se"], "lo_bonf": g["lo_bonf"], "hi_bonf": g["hi_bonf"], "loso_min": g["loso_min"]}, "swap_null": a["swap_null"], "seconds": time.time() - t0}


def run_planted(w, d, S: int, kappa: float) -> dict:
    S0.N_SEEDS = S
    prim, _ = S0.seed_sets()
    idx = S0.boot_indices(w.S)
    r = planted_parts(w, d, idx, prim, kappa)
    return {"kind": "planted", "S": S, "kappa": kappa, **r}


def run_null(w, d, S: int, i: int) -> dict:
    S0.N_SEEDS = S
    prim, repl = S0.seed_sets()
    idx = S0.boot_indices(w.S)
    t0 = time.time()
    wn = w.with_labels(S0.plant_labels(w, 0.0, seed=NULL_SEEDS[i]))
    t_gen = time.time() - t0
    r = decision_parts(wn, d, idx, prim, repl)
    r["seconds_parts"]["labels"] = t_gen
    return {"kind": "null", "S": S, "i": i, "label_seed": NULL_SEEDS[i], **r, "seconds": time.time() - t0}


# ====================================================================================================================== worker side
_W: dict = {}


def _world():
    if "w" not in _W:
        wb, man = blind_world()
        _W.update(w=wb, d=S0.make_design(wb), man=man)
    assert_blind(_W["w"])
    return _W["w"], _W["d"]


def run_task(spec: tuple) -> dict:
    kind, S, arg = spec
    t0 = time.time()
    try:
        w, d = _world()
        rec = run_planted(w, d, S, arg) if kind == "planted" else run_null(w, d, S, arg)
        rec["seconds"] = time.time() - t0
        rec["pid"] = os.getpid()
        return S0.jsonable(rec)
    except Exception as e:                                                          # noqa: BLE001  a failed task must be VISIBLE, never a silent hole
        return {"kind": kind, "S": S, "arg": arg, "error": repr(e), "traceback": traceback.format_exc(), "seconds": time.time() - t0, "pid": os.getpid()}


# ====================================================================================================================== master side: summaries, the selection rule, the identity check
def _nanmean(x):
    x = [v for v in x if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(np.mean(x)) if x else float("nan")


def summarise_S(S: int, planted: list, nulls: list) -> dict:
    """planted: the run_planted records of this S; nulls: the run_null records.  Pure function of the records."""
    per = {r["kappa"]: r["control"] for r in planted if "error" not in r}
    grid = S0.planted_grid_from_per(per) if per else {"per_kappa": {}, "kappas_read": [], "kappas_passing": [], "smallest_kappa_read": None, "pass": False}
    c2b_ok_le = any(k <= KAPPA_MAX_FOR_SELECTION for k in grid["kappas_passing"])
    table = {f"{k:g}": {"M1_gap": v["M1_gap"], "M1_lo_bonf": v["M1_lo_bonf"], "nu_M1": v["swap_null_M1"], "lo_minus_nu": v["M1_lo_bonf"] - v["swap_null_M1"], "gap_ge_min": v["M1_gap"] >= v["min_required"],
                   "planted_gap_read": v["planted_gap_read"], "seen_minus_oof": v["seen_minus_oof_planted"], "seen_beats_oof": v["seen_beats_oof_planted"], "collapse_pass": v["collapse"]["pass"],
                   "collapsed_M1_gap": v["collapse"]["M1"]["gap_mutated"], "pass": v["pass"]} for k, v in sorted(per.items())}
    ok_nulls = [r for r in nulls if "error" not in r]
    rows, counts_full, counts_wo, failed_counts = [], {}, {}, {}
    for r in ok_nulls:
        wo = S0.decide(r["primary"], r["replicate"], r["swap"], r["controls"], certified=r["certified"])                                  # C-II-b taken as passed
        full = S0.decide(r["primary"], r["replicate"], r["swap"], {**r["controls"], "C-II-b": {"pass": grid["pass"]}}, certified=r["certified"])
        ign = S0.decide(r["primary"], r["replicate"], r["swap"], {}, certified=r["certified"])                                           # controls ignored: does the GAP conjunct fire?
        counts_full[full["outcome"]] = counts_full.get(full["outcome"], 0) + 1
        counts_wo[wo["outcome"]] = counts_wo.get(wo["outcome"], 0) + 1
        for c in wo.get("failed_controls", []):
            failed_counts[c] = failed_counts.get(c, 0) + 1
        pm = {m: wo["per_measure"][m] for m in ("M1", "M2", "M3")}
        rows.append({"i": r["i"], "label_seed": r["label_seed"], "outcome_wo_c2b": wo["outcome"], "outcome_full": full["outcome"], "gap_if_controls_ignored": ign["outcome"] == "GAP", "failed_controls": wo.get("failed_controls", []),
                     "gap_measures": wo.get("gap_measures", []), "M1_point": pm["M1"].get("point"), "M1_lo_bonf": pm["M1"].get("lo_bonf"), "M1_nu": pm["M1"].get("swap_null_nu"),
                     "M1_separated": pm["M1"].get("separated_above_swap_null"), "M1_ge_delta": pm["M1"].get("magnitude_ge_delta")})
    n = len(ok_nulls)
    fp_n = counts_wo.get("GAP", 0)
    inv_n = counts_wo.get("INVALID", 0)
    m1 = [r["M1_point"] for r in rows]
    secs = {k: _nanmean([r["seconds_parts"].get(k, float("nan")) for r in ok_nulls]) for k in ("a0", "a1", "am", "c_shuf", "c_shuf_world", "labels")}
    return {"S": S, "planted_table": table, "c2b": {"kappas_read": grid["kappas_read"], "kappas_passing": grid["kappas_passing"], "pass_any": grid["pass"], "pass_at_kappa_le_max": bool(c2b_ok_le)},
            "null": {"n_expected": len(NULL_SEEDS), "n_ok": n, "n_errors": len(nulls) - n, "fp_n": fp_n, "fp_rate": (fp_n / n if n else float("nan")), "invalid_wo_c2b_n": inv_n, "invalid_wo_c2b_rate": (inv_n / n if n else float("nan")),
                     "gap_conjunct_fires_ignoring_controls_n": int(sum(r["gap_if_controls_ignored"] for r in rows)), "invalid_with_c2b_n": counts_full.get("INVALID", 0), "outcomes_wo_c2b": counts_wo,
                     "outcomes_with_c2b": counts_full, "failed_control_counts": failed_counts, "M1_gap_mean": _nanmean(m1), "M1_gap_sd": (float(np.nanstd(m1, ddof=1)) if len(m1) > 1 else float("nan")),
                     "M1_lo_bonf_mean": _nanmean([r["M1_lo_bonf"] for r in rows]), "M1_nu_mean": _nanmean([r["M1_nu"] for r in rows]), "M1_separated_n": int(sum(bool(r["M1_separated"]) for r in rows)),
                     "per_world": rows},
            "seconds": {"planted_total": float(sum(r.get("seconds", 0.0) for r in planted if "error" not in r)), "null_mean_parts": secs,
                        "null_total": float(sum(r.get("seconds", 0.0) for r in ok_nulls)), "null_mean": (float(np.mean([r["seconds"] for r in ok_nulls])) if ok_nulls else float("nan"))},
            "n_planted_ok": len(per)}


def apply_selection_rule(summaries: dict) -> dict:
    """The committed rule.  summaries = {S: summarise_S(...)}.  Returns {chosen: S or None, verdict, per_S: {...}}."""
    per, chosen = {}, None
    for S in S_CANDIDATES:
        s = summaries.get(S)
        if s is None:
            per[S] = {"eligible": False, "reason": "not calibrated"}
            continue
        complete = bool(s["null"]["n_ok"] == len(NULL_SEEDS) and s["n_planted_ok"] == len(KAPPAS))
        a = bool(s["c2b"]["pass_at_kappa_le_max"])
        b = bool(s["null"]["fp_n"] <= FP_MAX)
        c = bool(s["null"]["invalid_wo_c2b_n"] <= INVALID_MAX)
        ok = complete and a and b and c
        per[S] = {"complete": complete, "a_c2b_passes_at_kappa_le_2": a, "b_false_positives_le_2_of_24": b, "c_invalid_le_4_of_24": c, "fp_n": s["null"]["fp_n"], "invalid_n": s["null"]["invalid_wo_c2b_n"], "qualifies": ok}
        if ok and chosen is None:
            chosen = S
    verdict = f"SELECT S={chosen}" if chosen is not None else "INSTRUMENT-LIMITED"
    return {"chosen_S": chosen, "verdict": verdict, "per_S": {str(k): v for k, v in per.items()},
            "rule": f"smallest S in {list(S_CANDIDATES)} with C-II-b passing at some kappa <= {KAPPA_MAX_FOR_SELECTION:g}, null-world false positives <= {FP_MAX}/{len(NULL_SEEDS)}, null-world INVALID (excluding C-II-b) <= {INVALID_MAX}/{len(NULL_SEEDS)}"}


def identity_check(planted_S1: list, sealed_path: str = SEALED_EXCERPT) -> dict:
    """S = 1 must reproduce the sealed run's planted numbers (the controls-only excerpt: planted worlds are label-free, so this excerpt holds no real-token statistic)."""
    sealed = json.load(open(sealed_path))["controls"]["C-II-b"]["per_kappa"]
    rows, worst = {}, 0.0
    for r in sorted((x for x in planted_S1 if "error" not in x), key=lambda x: x["kappa"]):
        k = f"{r['kappa']:g}"
        old, new = sealed[k], r["control"]
        pairs = {"M1_gap": (old["M1_gap"], new["M1_gap"]), "M1_lo_bonf": (old["M1_lo_bonf"], new["M1_lo_bonf"]), "swap_null_M1": (old["swap_null_M1"], new["swap_null_M1"]),
                 "seen_minus_oof_planted": (old["seen_minus_oof_planted"], new["seen_minus_oof_planted"]),
                 "collapse_M1_gap_mutated": (old["collapse"]["M1"]["gap_mutated"], new["collapse"]["M1"]["gap_mutated"]),
                 "collapse_M1_swap_null_mutated": (old["collapse"]["M1"]["swap_null_mutated"], new["collapse"]["M1"]["swap_null_mutated"])}
        diffs = {q: abs(float(a) - float(b)) for q, (a, b) in pairs.items()}
        flags_equal = bool(old["planted_gap_read"] == new["planted_gap_read"] and old["pass"] == new["pass"])
        worst = max(worst, max(diffs.values()))
        rows[k] = {"sealed": {q: a for q, (a, _) in pairs.items()}, "now": {q: b for q, (_, b) in pairs.items()}, "abs_diff": diffs, "flags_equal": flags_equal}
    ok = bool(len(rows) == len(KAPPAS) and worst <= IDENTITY_TOL and all(v["flags_equal"] for v in rows.values()))
    return {"per_kappa": rows, "max_abs_diff": worst, "bit_identical": bool(worst == 0.0), "tolerance": IDENTITY_TOL, "pass": ok}


def estimate_real_read_seconds(s: dict) -> float:
    """ESTIMATED wall of the real read at this S (serial, one process): the null-world parts (a0, a1, am, C-I x2) + the R1 arm (~ a0) + the four planted pairs, measured here on the real features."""
    p = s["seconds"]["null_mean_parts"]
    return float(p["a0"] + p["a1"] + p["am"] + p["c_shuf"] + p["c_shuf_world"] + p["a0"] + s["seconds"]["planted_total"])


# ====================================================================================================================== main
def fmt(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def print_summary(S: int, s: dict) -> None:
    print(f"--- S = {S}: C-II-b planted table (M1 gap | lo | nu | lo-nu | SEEN-OOF | collapse | pass)", flush=True)
    for k, v in s["planted_table"].items():
        print(f"    kappa {k:>4}: gap {fmt(v['M1_gap'])} lo {fmt(v['M1_lo_bonf'])} nu {fmt(v['nu_M1'])} lo-nu {fmt(v['lo_minus_nu'])} SEEN-OOF {fmt(v['seen_minus_oof'])} collapse {v['collapse_pass']} read {v['planted_gap_read']} PASS {v['pass']}", flush=True)
    n = s["null"]
    print(f"    C-II-b kappas passing {s['c2b']['kappas_passing']}  read {s['c2b']['kappas_read']}  pass at kappa<=2: {s['c2b']['pass_at_kappa_le_max']}", flush=True)
    print(f"    null worlds {n['n_ok']}/{n['n_expected']} ok: FP {n['fp_n']}/{n['n_ok']}  INVALID(excl C-II-b) {n['invalid_wo_c2b_n']}/{n['n_ok']}  gap-conjunct-fires-ignoring-controls {n['gap_conjunct_fires_ignoring_controls_n']}  "
          f"INVALID(incl C-II-b) {n['invalid_with_c2b_n']}  failed controls {n['failed_control_counts']}", flush=True)
    print(f"    null M1: mean gap {fmt(n['M1_gap_mean'])} sd {fmt(n['M1_gap_sd'])} mean lo {fmt(n['M1_lo_bonf_mean'])} mean nu {fmt(n['M1_nu_mean'])}; separated {n['M1_separated_n']}/{n['n_ok']}", flush=True)
    print(f"    runtime: null world mean {fmt(s['seconds']['null_mean'], 1)} s, planted total {fmt(s['seconds']['planted_total'], 1)} s; ESTIMATED real read (serial) {fmt(estimate_real_read_seconds(s), 0)} s", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=12, help="worker processes (the brief caps this at 20)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "calib_a1.json"))
    ap.add_argument("--s-list", default=",".join(str(s) for s in S_LIST))
    ap.add_argument("--n-null", type=int, default=len(NULL_SEEDS))
    ap.add_argument("--identity-only", action="store_true", help="S = 1, the four planted worlds only, then compare with the sealed excerpt")
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise SystemExit("ABORT: run under CUDA_VISIBLE_DEVICES=-1 (CPU only)")
    if a.workers > 20:
        raise SystemExit("ABORT: at most 20 workers")
    s_list = tuple(int(x) for x in a.s_list.split(","))
    n_null = min(a.n_null, len(NULL_SEEDS))
    if a.identity_only:
        s_list, n_null = (1,), 0
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, os.path.splitext(os.path.basename(a.out))[0] + ".partial.jsonl")
    open(partial, "w").close()
    t_start = time.time()
    wb, man = blind_world()
    d = S0.make_design(wb)
    print(f"[calib_a1] real FEATURES loaded and label-blinded (assert_blind ok): N {wb.N} S {wb.S} D {wb.D} Ns {wb.Ns}; constants_sha256 {S0.constants_sha256()}; workers {a.workers}; S list {s_list}; null worlds {n_null}", flush=True)
    del wb
    specs = [("null", S, i) for S in sorted(s_list, reverse=True) for i in range(n_null)] + [("planted", S, k) for S in sorted(s_list, reverse=True) for k in KAPPAS]
    specs.sort(key=lambda sp: -sp[1])                                                         # the longest tasks (largest S) first; stable, so nulls precede planted inside an S
    results = []
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_task, sp): sp for sp in specs}
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            results.append(r)
            with open(partial, "a") as f:
                f.write(json.dumps(r) + "\n")
            sp = futs[fu]
            print(f"[calib_a1] {k}/{len(specs)} done {sp} {'ERROR ' + r['error'] if 'error' in r else ''} {r['seconds']:.1f}s elapsed {time.time() - t_start:.0f}s", flush=True)
    errors = [r for r in results if "error" in r]
    summaries = {}
    for S in s_list:
        pl = [r for r in results if r.get("kind") == "planted" and r["S"] == S]
        nu = [r for r in results if r.get("kind") == "null" and r["S"] == S]
        summaries[S] = summarise_S(S, pl, nu)
        print_summary(S, summaries[S])
    ident = identity_check([r for r in results if r.get("kind") == "planted" and r["S"] == 1]) if 1 in s_list else None
    if ident is not None:
        print(f"[calib_a1] IDENTITY S=1 vs the sealed run's planted numbers: max |diff| = {ident['max_abs_diff']:.3e} (bit identical: {ident['bit_identical']}); flags equal: "
              f"{all(v['flags_equal'] for v in ident['per_kappa'].values())}; PASS {ident['pass']}", flush=True)
        for k, v in ident["per_kappa"].items():
            print(f"    kappa {k:>4}: sealed M1 gap {v['sealed']['M1_gap']:.12f} now {v['now']['M1_gap']:.12f}  lo {v['sealed']['M1_lo_bonf']:.12f} / {v['now']['M1_lo_bonf']:.12f}  nu {v['sealed']['swap_null_M1']:.12f} / {v['now']['swap_null_M1']:.12f}", flush=True)
    sel = apply_selection_rule(summaries) if not a.identity_only else None
    if sel is not None:
        print(f"[calib_a1] SELECTION RULE: {sel['rule']}", flush=True)
        for S, v in sel["per_S"].items():
            print(f"    S={S}: {v}", flush=True)
        print(f"[calib_a1] VERDICT: {sel['verdict']}", flush=True)
    est = {str(S): estimate_real_read_seconds(summaries[S]) for S in s_list if summaries[S]["null"]["n_ok"]}
    res = {"schema": "spess0-a1-calib-v1", "constants_sha256": S0.constants_sha256(), "evidence_class": "MEASURED (ours), REAL features + SYNTHETIC labels only; no real label was scored", "s_list": list(s_list), "kappas": list(KAPPAS),
           "null_seeds": [NULL_SEEDS[0], NULL_SEEDS[-1], len(NULL_SEEDS)], "n_null_run": n_null, "rule": {"fp_max": FP_MAX, "invalid_max": INVALID_MAX, "kappa_max": KAPPA_MAX_FOR_SELECTION, "candidates": list(S_CANDIDATES)},
           "summaries": {str(S): summaries[S] for S in s_list}, "identity": ident, "selection": sel, "estimated_real_read_seconds_serial": est, "inputs": man, "n_errors": len(errors), "errors": errors,
           "wall_seconds": time.time() - t_start, "workers": a.workers}
    json.dump(S0.jsonable(res), open(a.out, "w"), indent=1)
    print(f"[calib_a1] wrote {os.path.abspath(a.out)}; errors {len(errors)}; wall {time.time() - t_start:.0f}s", flush=True)
    print("ZZCALIB_DONE", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:                                                                   # noqa: BLE001  a crash must be loud and must NOT look like completion
        traceback.print_exc()
        print("ZZCALIB_FAILED", flush=True)
        sys.exit(2)
