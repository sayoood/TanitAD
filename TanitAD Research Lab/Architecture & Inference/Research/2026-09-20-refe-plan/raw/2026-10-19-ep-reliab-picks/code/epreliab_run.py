"""EP-RELIAB-PICKS runner.  CPU only.  REGISTERED SOURCE: PREREG_EP_RELIAB_PICKS.md (sha256 pinned in epreliab_lib.PREREG_SHA256; the stage refuses on a mismatch).

    epreliab_run.py analyze   # arrays (sha256-verified, READ ONLY) -> per-seed quantities -> the gate (controls) -> refuses on a red control -> decision -> C-NULL / E-NULL -> run/results/result.json

Env: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4.  No git, no GPU.  Stage, never push.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import epreliab_lib as L                                                                  # noqa: E402

ROOT = "C:/Users/Admin/par18/epreliab"
RUN = f"{ROOT}/run"
ARRAYS = f"{L.PARENT}/pickrule/run/prepare/arrays.npz"
PREP_REPORT = f"{L.PARENT}/pickrule/run/prepare/prepare_report.json"
LANDED_RESULT = f"{L.PARENT}/pickrule/run/results/result.json"
POSTHOC = f"{L.PARENT}/ncfastcal/run/results/posthoc_diag.json"
PLANNER = f"{L.REF}/planner.py"


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_ram(min_gb: float = L.MIN_AVAIL_GB, poll_s: float = 20.0, max_wait_s: float | None = None) -> float:
    import psutil
    t0 = time.time()
    while True:
        av = psutil.virtual_memory().available / 2 ** 30
        if av >= min_gb:
            return av
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise L.EpReliabError(f"available RAM {av:.2f} GB < {min_gb} GB after {max_wait_s} s")
        log(f"[ram] available {av:.2f} GB < {min_gb} GB; waiting {poll_s:.0f} s")
        time.sleep(poll_s)


def all_zero(ci: dict) -> bool:
    return all(float(ci[k]) == 0.0 for k in ("point", "lo", "hi"))


def close(a: float, b: float, tol: float) -> bool:
    return bool(abs(float(a) - float(b)) <= tol)


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the gate (prereg s7): every control that must hold before the decision is read
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gate_checks(arr: dict, ep_cols: dict, per_seed: dict, P, E, F, S, landed: dict, posthoc: dict) -> dict:
    z6, lab, top8 = arr["z6"], arr["lab"], arr["top8"]
    N = len(z6)
    chk, ok = {}, True

    def rec(name, passed, detail):
        nonlocal ok
        chk[name] = {"pass": bool(passed), "detail": detail}
        ok = ok and bool(passed)
    pk_p = P.rule_pick(z6)
    # C-ID-RULE
    n_eq, n_eq_e = int((pk_p == top8[:, 0].astype(np.int64)).sum()), int((pk_p == E.picks_np(z6)).sum())
    pin = P.check_planner_pin(PLANNER)
    rec("C-ID-RULE", n_eq == N and n_eq_e == N and pin["status"] != "FAIL",
        {"rule_equals_banked_pick": f"{n_eq}/{N}", "equals_epr1_picks_np": f"{n_eq_e}/{N}", "planner_pin": {k: v for k, v in pin.items() if k != "rule_lines_text"},
         "planner_copy_sha256": L.sha256_file(PLANNER), "planner_copy_equals_pin": bool(L.sha256_file(PLANNER) == L.PLANNER_SHA256), "note": "NOT RUN is reported as NOT RUN, not PASS"})
    # C-PDMS and the safe mask
    pd_mine, pd_epr1 = P.pdms_from_labels(lab), E.pdms_np(lab)
    rec("C-PDMS", bool(np.array_equal(pd_mine, pd_epr1)) and tuple(P.V1_W) == (5.0, 5.0, 2.0) and bool(np.array_equal(L.safe_mask(lab), E.safe_np(lab))),
        {"pdms_bit_equal_to_epr1_pdms_np": bool(np.array_equal(pd_mine, pd_epr1)), "weights": list(P.V1_W), "weights_source": "refe/planner.py:520 V1_W (navsim @ 3e8291b pdm_scorer.py:38-42) via pickrule_lib.V1_W",
         "safe_mask_equals_epr1_safe_np": bool(np.array_equal(L.safe_mask(lab), E.safe_np(lab)))})
    # C-ID-LANDED and C-ID-R53 and C-ATTR, per seed
    for s in L.SEEDS:
        ps = per_seed[s]
        zE = z6.copy()
        zE[..., L.EP] = ep_cols[s]
        pk_e = P.rule_pick(zE)
        pd_e, pd_p = P.at(pd_mine, pk_e).astype(np.float64), P.at(pd_mine, pk_p).astype(np.float64)
        mine = P.scale_ci(S.mean(pd_e - pd_p), 100.0)
        ref = landed["analysis"]["composites"][f"OOF_s{s}"]["sweep"]["none"]
        rd = ref["d_pdms_x100"]
        rec(f"C-ID-LANDED[seed{s}]", all(close(mine[k], rd[k], L.TOL_ID_LANDED) for k in ("point", "lo", "hi")) and int((pk_e != pk_p).sum()) == int(ref["n_changed_vs_p2"]),
            {"mine": {k: mine[k] for k in ("point", "lo", "hi")}, "landed": {k: rd[k] for k in ("point", "lo", "hi")}, "n_changed": int((pk_e != pk_p).sum()), "n_changed_landed": int(ref["n_changed_vs_p2"]), "tol": L.TOL_ID_LANDED})
        nc_mine = ps["reliability"]["NC"]["composite_picks"]["deficit_form_sum_1my_over_sum_1msigma"]
        nc_ref = posthoc["arms"][f"U_s{s}"]["ratio_observed_over_predicted"]
        rec(f"C-ID-R53[seed{s}]", all(close(nc_mine[k], nc_ref[k], L.TOL_ID_LANDED) for k in ("point", "lo", "hi")),
            {"mine": {k: nc_mine[k] for k in ("point", "lo", "hi")}, "posthoc_diag": {k: nc_ref[k] for k in ("point", "lo", "hi")}, "tol": L.TOL_ID_LANDED})
        at_ = ps["attribution"]
        rec(f"C-ATTR[seed{s}]", at_["max_abs_seq_sum_minus_total_per_set"] <= L.TOL_ATTR_EXACT and at_["max_abs_shapley_sum_minus_total_per_set"] <= L.TOL_ATTR_EXACT
            and close(at_["mean_total_x100_float64"], rd["point"], L.TOL_ATTR_LANDED_X100),
            {"max_abs_seq_sum_minus_total_per_set": at_["max_abs_seq_sum_minus_total_per_set"], "max_abs_shapley_sum_minus_total_per_set": at_["max_abs_shapley_sum_minus_total_per_set"],
             "float64_total_x100": at_["mean_total_x100_float64"], "landed_float32_d_pdms_x100": rd["point"], "abs_diff": abs(at_["mean_total_x100_float64"] - rd["point"]), "tol_landed": L.TOL_ATTR_LANDED_X100,
             "tol_exact": L.TOL_ATTR_EXACT})
    # C-ID (P2 vs P2): the full analysis with the composite replaced by P2's own EP column must read exactly zero
    p2 = L.analyze_seed(arr, z6[..., L.EP], S)
    G, A = p2["G"], p2["attribution"]
    zero_cis = [G["delta"], G["pred_gain"], G["real_gain"], G["D_M"], A["total_x100"]] + list(A["sequential_x100"].values()) + list(A["shapley_x100"].values())
    zero_mass = float(G["M_e"]["point"]) == float(G["M_p"]["point"])
    rec("C-ID", p2["n_changed"] == 0 and all(all_zero(c) for c in zero_cis) and zero_mass,
        {"n_changed": p2["n_changed"], "every_point_lo_hi_equals_0.0": all(all_zero(c) for c in zero_cis), "n_intervals_checked": len(zero_cis), "M_e_equals_M_p": zero_mass})
    # C-REACH: synthetic planted bank (0 < n_changed < N by construction) and the identity of the arithmetic on the real arrays
    arr_s, ep_s, li_s, ch_s = L.synthetic_bank()
    draws_s = F.boot_draws(12, 500, 1)
    S_s = P.Stats(li_s, draws_s, F, n_logs=12)
    zs = arr_s["z6"].copy()
    zs[..., L.EP] = ep_s
    pe_s, pp_s = P.rule_pick(zs), P.rule_pick(arr_s["z6"])
    syn = L.reach_check(S_s, P.sigmoid64(zs[..., L.EP]), arr_s["lab"][..., L.EP].astype(np.float64), pp_s, pe_s)
    syn["synthetic_picks_as_designed"] = bool((pp_s == 0).all() and np.array_equal(pe_s == 1, ch_s) and 0 < int(ch_s.sum()) < len(ch_s))
    reals = {}
    y = lab[..., L.EP].astype(np.float64)
    for s in L.SEEDS:
        zE = z6.copy()
        zE[..., L.EP] = ep_cols[s]
        reals[str(s)] = L.reach_check(S, P.sigmoid64(zE[..., L.EP]), y, pk_p, P.rule_pick(zE))
    rec("C-REACH", syn["pass"] and syn["synthetic_picks_as_designed"] and all(r["pass"] for r in reals.values()), {"synthetic_planted_bank": syn, "real_arrays_identity_of_the_arithmetic": reals})
    # C-POWER (Amendment 1): the NEUTRAL label field on the real picks and the real sigma; no real EP label at a pick is used or recorded
    pw = {}
    for s in L.SEEDS:
        zE = z6.copy()
        zE[..., L.EP] = ep_cols[s]
        r = L.power_check(S, P.sigmoid64(zE[..., L.EP]), y, pk_p, P.rule_pick(zE))
        pw[str(s)] = {f: {"plus_class": r[f]["plus"]["class"], "minus_class": r[f]["minus"]["class"], "zero_class": r[f]["zero"]["class"], "zero_is_exactly_0_0": r[f]["zero_is_exactly_0_0"], "pass": r[f]["pass"]}
                      for f in ("DELTA", "MASS")}
        pw[str(s)]["pass"] = bool(r["pass"])
    rec("C-POWER", all(v["pass"] for v in pw.values()), {"planted_shift": L.PLANT, "by_seed": pw})
    # C-SCALE (planted)
    draws_k = F.boot_draws(10, 500, 2)
    sc = {}
    for kind, want in (("exact", True), ("ok_1.05", True), ("half", False), ("high_1.2", False)):
        x, yy, sf, li, nl = L.planted_slope_bank(kind)
        Sk = P.Stats(li, draws_k, F, n_logs=nl)
        ci = L.scale_slope(Sk, x, yy, sf)
        sc[kind] = {"slope": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "on": L.scale_on(ci), "expected_on": want}
    x, yy, sf, li, nl = L.planted_slope_bank("exact")
    Sk = P.Stats(li, draws_k, F, n_logs=nl)
    off = L.scale_slope(Sk, np.clip(x + 0.2, 0, 2), yy, sf)["point"] - L.scale_slope(Sk, x, yy, sf)["point"]
    sc["offset_invariance_abs_change_of_slope"] = float(abs(off))
    rec("C-SCALE", all(v["on"] == v["expected_on"] for k, v in sc.items() if isinstance(v, dict) and "expected_on" in v) and sc["offset_invariance_abs_change_of_slope"] < 1e-9, sc)
    # C-REACH-DECISION
    mr = L.mutation_report()
    rec("C-REACH-DECISION", len(mr["unmutated_failures"]) == 0 and mr["n_caught"] == mr["n_mutations"], mr)
    return {"ok": ok, "checks": chk}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_analyze(a) -> int:
    t0 = time.time()
    prereg_sha = L.require_prereg(a.prereg)
    os.makedirs(f"{RUN}/results", exist_ok=True)
    avail = wait_for_ram()
    log(f"analyze: RAM available {avail:.2f} GB; prereg sha256 {prereg_sha[:12]}")
    P, E, F = L.load_pickrule(), L.load_epr1(), L.load_fastneg()
    prep = json.load(open(PREP_REPORT, "r", encoding="utf-8"))
    sha = L.sha256_file(ARRAYS)
    if sha != L.ARRAYS_SHA256 or sha != prep["arrays_npz_sha256"]:
        raise L.EpReliabError(f"arrays.npz sha256 {sha} != the registered {L.ARRAYS_SHA256} / the audit's report {prep['arrays_npz_sha256']}")
    z = np.load(ARRAYS, allow_pickle=False)
    arr = {k: z[k] for k in ("z6", "lab", "top8", "log_idx", "fold", "ep_OOF_s0", "ep_OOF_s1")}
    N = len(arr["z6"])
    if N != 15808 or arr["z6"].shape != (15808, 64, 6) or len(np.unique(arr["log_idx"])) != L.N_LOGS:
        raise L.EpReliabError("unexpected array size")
    ep_cols = {s: arr[f"ep_OOF_s{s}"] for s in L.SEEDS}
    draws = F.boot_draws(L.N_LOGS, L.BOOT_B, L.BOOT_SEED)
    if not np.array_equal(draws, E.boot_draws(L.N_LOGS, L.BOOT_B, L.BOOT_SEED)):
        raise L.EpReliabError("fastneg_lib.boot_draws differs from epr1.boot_draws")
    S = P.Stats(arr["log_idx"].astype(np.int64), draws, F)
    landed = json.load(open(LANDED_RESULT, "r", encoding="utf-8"))
    posthoc = json.load(open(POSTHOC, "r", encoding="utf-8"))
    log("computing the per-seed quantities (held in memory until the gate has passed)")
    per_seed = {s: L.analyze_seed(arr, ep_cols[s], S) for s in L.SEEDS}
    log("gate: running the controls")
    gate = gate_checks(arr, ep_cols, per_seed, P, E, F, S, landed, posthoc)
    gate["prereg_sha256"] = prereg_sha
    with open(f"{RUN}/results/gate_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(L.jsonable(gate), indent=1, sort_keys=True))
    for k, v in gate["checks"].items():
        log(f"gate {k}: {'PASS' if v['pass'] else 'FAIL'}")
    if not gate["ok"]:
        log("GATE FAILED: no decision is taken (write an amendment before the number it governs)")
        return 3
    dec = L.decide(per_seed)
    real_pts = {s: float(per_seed[s]["G"]["delta" if dec["deciding_form"] == "DELTA" else "D_M"]["point"]) for s in L.SEEDS}
    log(f"gate PASS; deciding form {dec['deciding_form']}; computing C-NULL (5) and E-NULL (200)")
    null = L.run_null(arr, ep_cols, S, dec["deciding_form"], real_points=real_pts)
    label = L.final_label(dec["outcome"], True, null["c_null"]["void"])
    out = {"schema": L.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "prereg_sha256": prereg_sha, "n_sets": N, "n_logs": L.N_LOGS,
           "bootstrap": {"B": L.BOOT_B, "seed": L.BOOT_SEED, "unit": "log", "percentiles": [2.5, 97.5]}, "inputs": {"arrays_npz_sha256": sha, "prepare_report_sha256": L.sha256_file(PREP_REPORT),
                                                                                                                  "landed_result_sha256": L.sha256_file(LANDED_RESULT), "posthoc_diag_sha256": L.sha256_file(POSTHOC)},
           "gate": gate, "per_seed": {str(s): per_seed[s] for s in L.SEEDS}, "decision": dec, "null": null, "final_label": label, "next_arm": L.next_arm(label),
           "code_sha256": {n: L.sha256_file(os.path.join(HERE, n)) for n in ("epreliab_lib.py", "epreliab_run.py")}, "elapsed_s": round(time.time() - t0, 1)}
    with open(f"{RUN}/results/result.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(L.jsonable(out), indent=1, sort_keys=True))
    log(f"ANALYZE_DONE: {label}; deciding form {dec['deciding_form']}; form_sensitive {dec['form_sensitive']}; dominant {dec['dominant_term']}; C-NULL fires {null['c_null']['fires']}/5; {out['elapsed_s']} s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["analyze"])
    ap.add_argument("--prereg", default=os.path.join(ROOT, "PREREG_EP_RELIAB_PICKS.md"))
    a = ap.parse_args(argv)
    return {"analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
