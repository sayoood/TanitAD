"""NC-FASTCAL runner.  CPU only.  REGISTERED SOURCE: PREREG_NC_FASTCAL.md (sha256 pinned in ncfastcal_lib.PREREG_SHA256; every stage refuses on a mismatch).

    ncfastcal_run.py analyze    # gate (C-JOIN-STRATA, C-ID, C-REACH, C-FOLD, C-REACH-DECISION) -> refuses on a red control -> per-stratum Platt fit, the arms, the decision, C-NULL; writes run/results/

Env: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8.  No git.  Stage, never push.  Reads the audit's arrays READ ONLY.
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
import ncfastcal_lib as L                                                                 # noqa: E402

ROOT = "C:/Users/Admin/par18/ncfastcal"
RUN = f"{ROOT}/run"
ARRAYS = "C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz"                           # READ ONLY (the audit's prepare output)
PREPARE_REPORT = "C:/Users/Admin/par18/pickrule/run/prepare/prepare_report.json"          # READ ONLY
LANDED_RESULT = "C:/Users/Admin/par18/pickrule/run/results/result.json"                   # READ ONLY (the audit's result; C-ID (c))
SPEEDS = f"{L.REF}/speeds.npz"
STRATA_JSON = f"{L.REF}/strata.json"


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_ram(min_gb: float = L.MIN_AVAIL_GB, poll_s: float = 20.0, max_wait_s: float | None = 600.0) -> float:
    import psutil
    t0 = time.time()
    while True:
        av = psutil.virtual_memory().available / 2 ** 30
        if av >= min_gb:
            return av
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise L.NcFastcalError(f"available RAM {av:.2f} GB < {min_gb} GB after {max_wait_s} s")
        log(f"[ram] available {av:.2f} GB < {min_gb} GB; waiting {poll_s:.0f} s")
        time.sleep(poll_s)


def write_json(path: str, obj, P) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(obj), indent=1, sort_keys=True))


def stage_analyze(a) -> int:
    prereg_sha = L.require_prereg(a.prereg)
    os.makedirs(f"{a.run_dir}/results", exist_ok=True)
    avail = wait_for_ram()
    P = L.load_pickrule()
    F = L.load_fastneg()
    E = L.load_epr1()
    t0 = time.time()
    log(f"analyze: RAM available {avail:.2f} GB; prereg {prereg_sha[:8]}; pinned pickrule_lib {L.PICKRULE_LIB_SHA256[:8]}, epr1 {L.EPR1_SHA256[:8]}, fastneg_lib {L.FASTNEG_LIB_SHA256[:8]}")
    # ---- inputs, hash-checked
    with open(PREPARE_REPORT, "r", encoding="utf-8") as f:
        prep = json.load(f)
    arrays_sha = L.sha256_file(ARRAYS)
    if arrays_sha != prep["arrays_npz_sha256"]:
        raise L.NcFastcalError(f"arrays.npz sha256 {arrays_sha} != the audit's prepare report {prep['arrays_npz_sha256']}")
    for path, want in ((SPEEDS, L.SPEEDS_NPZ_SHA256), (STRATA_JSON, L.STRATA_JSON_SHA256)):
        got = L.sha256_file(path)
        if got != want:
            raise L.NcFastcalError(f"{path} sha256 {got} != the pinned {want}")
    z = np.load(ARRAYS, allow_pickle=False)
    arr = {k: z[k] for k in z.files}
    sp = np.load(SPEEDS, allow_pickle=False)
    speeds = {k: sp[k] for k in ("key_log", "key_token", "pick", "r", "v_b", "code")}
    with open(STRATA_JSON, "r", encoding="utf-8") as f:
        strata_json = json.load(f)
    with open(LANDED_RESULT, "r", encoding="utf-8") as f:
        landed = json.load(f)
    z6, lab, fold, log_idx = arr["z6"], arr["lab"], arr["fold"].astype(np.int64), arr["log_idx"].astype(np.int64)
    code = speeds["code"].astype(np.uint8)
    N = len(z6)
    eps = {sd: arr[f"ep_OOF_s{sd}"] for sd in L.SEEDS}
    draws = F.boot_draws(L.N_LOGS, L.BOOT_B, L.BOOT_SEED)
    if not np.array_equal(draws, E.boot_draws(L.N_LOGS, L.BOOT_B, L.BOOT_SEED)):
        raise L.NcFastcalError("fastneg_lib.boot_draws differs from epr1.boot_draws")
    if len(np.unique(log_idx)) != L.N_LOGS or N != 15808:
        raise L.NcFastcalError(f"unexpected size: {N} sets / {len(np.unique(log_idx))} logs")
    S = P.Stats(log_idx, draws, F)
    pdms = P.pdms_from_labels(lab)
    pk_p = P.rule_pick(z6)
    # ================= the gate =================
    chk, ok = {}, True

    def rec(name, passed, detail, gating=True):
        nonlocal ok
        chk[name] = {"pass": bool(passed), "detail": detail, "gating": bool(gating)}
        if gating:
            ok = ok and bool(passed)
    sj = L.strata_join_checks(arr, speeds, strata_json)
    rec("C-JOIN-STRATA", sj["pass"], sj["detail"])
    # C-ID (b) rule reproduction on P2
    bank = arr["top8"][:, 0].astype(np.int64)
    n_b, n_e = int((pk_p == bank).sum()), int((pk_p == E.picks_np(z6)).sum())
    rec("C-ID-b", n_b == N and n_e == N, {"rule_pick_equals_banked_pick": f"{n_b}/{N}", "equals_epr1_picks_np": f"{n_e}/{N}"})
    # C-ID (a) identity calibration == the uncalibrated composite, bit for bit, both seeds
    x64 = z6[..., L.NC].astype(np.float64)
    nc_id = L.apply_calibration(x64, code, fold, L.identity_params())
    det = {"nc_column_bit_equal_to_p2": bool(np.array_equal(nc_id, x64))}
    id_ok = det["nc_column_bit_equal_to_p2"]
    for sd, ep in eps.items():
        pk_u = P.rule_pick(L.composite(z6, ep=ep))
        pk_i = P.rule_pick(L.composite(z6, ep=ep, nc=nc_id))
        det[f"seed{sd}_picks_equal"] = f"{int((pk_u == pk_i).sum())}/{N}"
        id_ok = id_ok and bool((pk_u == pk_i).all())
    rec("C-ID-a", id_ok, det)
    # C-ID (c) the uncalibrated composite reproduces the landed audit's numbers
    cmp_ = {}
    c_ok = True
    for sd, ep in eps.items():
        mine = P.sweep_one(L.composite(z6, ep=ep), lab, pdms, pk_p, S, None)
        cmp_[f"seed{sd}"] = L.compare_landed(mine, landed["analysis"]["composites"][f"OOF_s{sd}"]["sweep"]["none"])
        c_ok = c_ok and cmp_[f"seed{sd}"]["pass"]
    rec("C-ID-c", c_ok, cmp_)
    # C-REACH (a) + C-FOLD: planted synthetic bank, the real code path; (b) the same planted table on the real arrays (transform statement; picks moved INFORMATIONAL)
    reach_bad = L.reach_table_failures(L)
    rec("C-REACH-a-C-FOLD (planted synthetic bank)", not reach_bad, {"failures": reach_bad})
    p_plant = L.identity_params()
    p_plant[:, 4, 1] = -3.0
    zc = L.apply_calibration(x64, code, fold, p_plant)
    b_ok = bool(((zc != x64) == (code == 4)).all()) and bool(np.array_equal(zc[code != 4], x64[code != 4]))
    moved = {}
    for sd, ep in eps.items():
        moved[f"seed{sd}"] = int((P.rule_pick(L.composite(z6, ep=ep, nc=zc)) != P.rule_pick(L.composite(z6, ep=ep))).sum())
    rec("C-REACH-b (planted Q4 shift on the real arrays: the transform)", b_ok, {"planted": "b = -3.0 in stratum Q4, all folds", "real_picks_moved_INFORMATIONAL_not_a_gate": moved,
                                                                              "proposals_shifted": int((zc != x64).sum()), "q4_proposals": int((code == 4).sum())})
    bt = L.branch_table_failures(L)
    rec("C-REACH-DECISION (branch table)", not bt, {"failures": bt})
    gate = {"ok": ok, "checks": chk, "prereg_sha256": prereg_sha, "arrays_npz_sha256": arrays_sha}
    write_json(f"{a.run_dir}/results/gate_report.json", gate, P)
    for k, v in chk.items():
        log(f"gate {k}: {'PASS' if v['pass'] else 'FAIL'}")
    if not ok:
        log("GATE FAILED: no decision quantity is computed (write an amendment before the number it governs)")
        return 3
    # ================= the fit and the arms =================
    log("gate PASS: fitting the per-stratum Platt NC calibration (OOF) and the global companion")
    tf = time.time()
    nc_cal, params, diag = L.calibrated_nc(z6, lab, code, fold)
    nc_glob, params_g, diag_g = L.calibrated_nc(z6, lab, np.zeros_like(code), fold)
    n_fail = [d for d in diag if not d["ok"]]
    log(f"fit done in {time.time() - tf:.1f} s; per-stratum fit failures {len(n_fail)} of {len(diag)}" + (f" ({[(d['fold'], d['stratum_name'], d['reason']) for d in n_fail]})" if n_fail else ""))
    for s in range(L.N_STRATA):
        log(f"  stratum {L.STRATA_NAMES[s]}: a by fold {np.round(params[:, s, 0], 3).tolist()}  b by fold {np.round(params[:, s, 1], 3).tolist()}")
    p2_row = P.sweep_one(z6, lab, pdms, pk_p, S, None)
    arms = {"P2_unvetoed": p2_row}
    zU, zC, zG = {}, {}, {}
    for sd, ep in eps.items():
        zU[sd], zC[sd], zG[sd] = L.composite(z6, ep=ep), L.composite(z6, ep=ep, nc=nc_cal), L.composite(z6, ep=ep, nc=nc_glob)
        arms[f"U_s{sd}"] = P.sweep_one(zU[sd], lab, pdms, pk_p, S, None)
        arms[f"C_s{sd}"] = P.sweep_one(zC[sd], lab, pdms, pk_p, S, None)
        arms[f"G_s{sd}"] = P.sweep_one(zG[sd], lab, pdms, pk_p, S, None)
    arms["PC"] = P.sweep_one(L.composite(z6, nc=nc_cal), lab, pdms, pk_p, S, None)
    # ---- the registered decision (PRIMARY arm C_s, both seeds); the same pairs are re-derived by the shared `decide` path and must agree
    pairs = [(arms[f"C_s{sd}"]["d_nc_unsafe_rate"], arms[f"C_s{sd}"]["d_pdms_x100"]) for sd in L.SEEDS]
    rec_flag = L.recovers(pairs)
    dd = L.decide(P, z6, lab, eps, nc_cal, pdms, pk_p, S)
    if dd["recovers"] != rec_flag:
        raise L.NcFastcalError("decide() disagrees with the reported sweep rows")
    decision = {"recovers": rec_flag, "per_seed": {str(sd): {"cond_i_upper_dNC_lt_0.001": L.cond_i(pairs[i][0]), "cond_ii_lower_dPDMS_ge_0": L.cond_ii(pairs[i][1]),
                                                              "dNC": pairs[i][0], "dPDMS_x100": pairs[i][1]} for i, sd in enumerate(L.SEEDS)}}
    unc_pairs = [(arms[f"U_s{sd}"]["d_nc_unsafe_rate"], arms[f"U_s{sd}"]["d_pdms_x100"]) for sd in L.SEEDS]
    decision["uncalibrated_beside"] = {"would_recover_by_the_same_rule": L.recovers(unc_pairs), "per_seed": {str(sd): {"cond_i": L.cond_i(unc_pairs[i][0]), "cond_ii": L.cond_ii(unc_pairs[i][1])} for i, sd in enumerate(L.SEEDS)}}
    glob_pairs = [(arms[f"G_s{sd}"]["d_nc_unsafe_rate"], arms[f"G_s{sd}"]["d_pdms_x100"]) for sd in L.SEEDS]
    decision["global_platt_companion_no_authority"] = {"would_recover_by_the_same_rule": L.recovers(glob_pairs), "per_seed": {str(sd): {"cond_i": L.cond_i(glob_pairs[i][0]), "cond_ii": L.cond_ii(glob_pairs[i][1])} for i, sd in enumerate(L.SEEDS)}}
    log(f"decision (before the null): RECOVERS = {rec_flag}; per seed (i, ii) = {[(L.cond_i(a_), L.cond_ii(b_)) for a_, b_ in pairs]}")
    # ---- diagnostics (prereg s8; no authority)
    sig_c, sig_raw = P.sigmoid64(nc_cal), P.sigmoid64(z6[..., L.NC])
    bce_g = L.heldout_bce(x64, lab[..., L.NC], np.zeros_like(code), fold, params_g)
    diagn = {"heldout_bce_by_stratum": L.heldout_bce(x64, lab[..., L.NC], code, fold, params), "heldout_bce_global_platt": {"all_proposals": bce_g.get("S")},
             "reliability_at_picks": {"P2": L.reliability_at_picks(z6[..., L.NC], nc_cal, lab[..., L.NC], pk_p, P)}, "q1_like_C": {}, "n_picks_changed_C_vs_U": {}}
    for sd in L.SEEDS:
        pk_u, pk_c = P.rule_pick(zU[sd]), P.rule_pick(zC[sd])
        diagn["reliability_at_picks"][f"U_s{sd}"] = L.reliability_at_picks(z6[..., L.NC], nc_cal, lab[..., L.NC], pk_u, P)
        diagn["reliability_at_picks"][f"C_s{sd}"] = L.reliability_at_picks(z6[..., L.NC], nc_cal, lab[..., L.NC], pk_c, P)
        q = P.q1_q2(lab[..., L.NC], sig_c, pk_c, pk_p, S)
        diagn["q1_like_C"][f"seed{sd}"] = {k: q[k] for k in ("q1", "n_A", "n_over", "sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max")}
        q_u = P.q1_q2(lab[..., L.NC], sig_raw, pk_u, pk_p, S)
        diagn["q1_like_C"][f"seed{sd}_uncalibrated_for_reference"] = {k: q_u[k] for k in ("q1", "n_A", "n_over", "sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max")}
        diagn["n_picks_changed_C_vs_U"][f"seed{sd}"] = int((pk_c != pk_u).sum())
    # ================= C-NULL =================
    log("computing C-NULL (strata codes permuted within token, 5 replicates; each replicate refits the calibration)")
    tn = time.time()
    groups = L.token_groups(speeds["key_token"])
    null = L.run_null(P, z6, lab, eps, code, fold, groups, pdms, pk_p, S)
    log(f"C-NULL done in {time.time() - tn:.1f} s: RECOVERS fires on {null['fires']} of {null['of']}; condition (i) alone on both seeds fires on {null['cond_i_only_fires']} of {null['of']}; void {null['void']}")
    final = L.final_label(rec_flag, null["void"])
    out = {"schema": L.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "prereg_sha256": prereg_sha, "n_sets": int(N), "n_logs": L.N_LOGS,
           "bootstrap": {"B": L.BOOT_B, "seed": L.BOOT_SEED, "unit": "log", "percentiles": [2.5, 97.5]}, "gate": gate, "n_tokens": len(groups),
           "fit": {"params_a_b_by_fold_stratum": params, "diag": diag, "n_fit_failed": len(n_fail), "global_params": params_g, "global_diag": diag_g},
           "arms": arms, "decision": decision, "diagnostics": diagn, "null": null, "final_outcome": final, "next_arm": L.next_arm(final),
           "code_sha256": {n: L.sha256_file(os.path.join(HERE, n)) for n in ("ncfastcal_lib.py", "ncfastcal_run.py")}, "elapsed_s": round(time.time() - t0, 1)}
    write_json(f"{a.run_dir}/results/result.json", out, P)
    np.savez_compressed(f"{a.run_dir}/results/nc_cal_oof.npz", nc_cal_primary=nc_cal.astype(np.float32), nc_cal_global=nc_glob.astype(np.float32), params_primary=params, params_global=params_g)
    log(f"ANALYZE_DONE in {out['elapsed_s']} s: final {final}; next arm: {out['next_arm']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["analyze"])
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--prereg", default=os.path.join(ROOT, "PREREG_NC_FASTCAL.md"))
    a = ap.parse_args(argv)
    return {"analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
