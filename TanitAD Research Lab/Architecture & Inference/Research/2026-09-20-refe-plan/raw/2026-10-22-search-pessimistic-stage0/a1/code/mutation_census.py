#!/usr/bin/env python3
"""Mutation census of stage0.py: apply each textual mutation to a COPY of the package, run the test suite on the copy, and require it to go RED.
(The programme's rule: 'MUTATE, do not inspect'.)  A mutant that survives is either an equivalent mutant (say why) or a hole in test_stage0.py.

    CUDA_VISIBLE_DEVICES=-1 python mutation_census.py --out ../raw/MUTATION_CENSUS.json [--only name1,name2]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# name -> (old, new[, why a survivor would be equivalent])
MUTANTS = {
    "role_shift_oof_is_in": ("mem = _members(models[(g + k) % N_FOLDS])", "mem = _members(models[(g + k + 1) % N_FOLDS])"),
    "m3_sign_flipped": ("out[\"M3\"] = cluster_gap(rd.sess, S, rd.m3_reg[b_role], rd.m3_reg[a_role],", "out[\"M3\"] = cluster_gap(rd.sess, S, rd.m3_reg[a_role], rd.m3_reg[b_role],"),
    "a3_threshold_strict": ("m = (np.abs(pp - pa) >= THR_PAIR - 1e-12) & ok", "m = (np.abs(pp - pa) > THR_PAIR + 1e-9) & ok"),
    "pair_mask_wrong_direction": ("return (dp >= THR_PAIR - 1e-12) & valid[:, :, None] & valid[:, None, :]", "return (dp <= -THR_PAIR + 1e-12) & valid[:, :, None] & valid[:, None, :]"),
    "regret_uses_argmin": ("pick = np.argmax(a, axis=1)", "pick = np.argmin(a, axis=1)"),
    "pairwise_gradient_dropped": ("gW[R] += s.T @ (xp - xa)", "gW[R] += 0.0 * (s.T @ (xp - xa))"),
    "ddc_row_becomes_trainable": ("    R = list(ROWS)\n    gW = np.zeros_like(W)", "    R = list(ROWS) + [DDC]\n    gW = np.zeros_like(W)"),
    "decide_drops_loso": ("gap_m = S and Mg and Rp and Ls", "gap_m = S and Mg and Rp"),
    "decide_drops_replicate": ("gap_m = S and Mg and Rp and Ls", "gap_m = S and Mg and Ls"),
    "decide_drops_separation": ("gap_m = S and Mg and Rp and Ls", "gap_m = Mg and Rp and Ls"),
    "decide_ignores_failed_controls": ("    if failed:\n        return {\"outcome\": \"INVALID\"", "    if False:\n        return {\"outcome\": \"INVALID\""),
    "bootstrap_never_draws_last_session": ("return np.random.default_rng(seed).integers(0, S, size=(B, S))", "return np.random.default_rng(seed).integers(0, S - 1, size=(B, S))"),
    "half_split_unbalanced": ("is_search[order[1::2]] = True", "is_search[order[1:2]] = True"),
    "folds_not_balanced": ("f = int(np.argmin(load))", "f = int(np.argmax(load))"),
    "training_includes_search_half": ("& ~d.is_search)[0]\n\n\ndef check_exclusion", ")[0]\n\n\ndef check_exclusion"),
    "adam_no_bias_correction": ("c1, c2 = 1 - b1 ** step, 1 - b2 ** step", "c1, c2 = 1.0, 1.0"),
    "seed_ignored": ("rng = np.random.default_rng(int(seed))", "rng = np.random.default_rng(0)"),
    "swap_null_degenerate": ("pairs = [(1, 2), (2, 3), (3, 4)]", "pairs = [(1, 1), (2, 2), (3, 3)]"),
    "seen_control_accepts_anything": ("\"pass\": bool(t[\"z\"] > -Z_MAX)}", "\"pass\": True}"),
    "planted_world_does_not_require_seen_to_beat_oof": ("\"pass\": bool(read and coll[\"pass\"] and seen_pt > SEEN_MARGIN)}", "\"pass\": bool(read and coll[\"pass\"])}"),
    "seen_token_z_sign_flipped": ("seen.m1_num[1] - seen.m1_num[0], seen.m1_den.sum()), \"M3\":", "seen.m1_num[0] - seen.m1_num[1], seen.m1_den.sum()), \"M3\":"),
    "chance_residual_centre_wrong": ("R[\"m1c\"].append(rd.m1_num[0] - 0.5 * rd.m1_den)", "R[\"m1c\"].append(rd.m1_num[0] - 0.4 * rd.m1_den)"),
    "shuffle_leaves_fan_labels_intact": ("pf = np.argsort(rng.random((w.N, w.K)), axis=1)", "pf = np.tile(np.arange(w.K), (w.N, 1))"),
    "shuffle_pooled_deal_is_identity": ("pu = np.argsort(rng.random((w.Ns, w.K + w.Q)), axis=1)", "pu = np.tile(np.arange(w.K + w.Q), (w.Ns, 1))"),
    "shuffle_deals_fan_and_perturbations_separately": ("pu = np.argsort(rng.random((w.Ns, w.K + w.Q)), axis=1)",
                                                         "pu = np.concatenate([np.argsort(rng.random((w.Ns, w.K)), axis=1), w.K + np.argsort(rng.random((w.Ns, w.Q)), axis=1)], axis=1)"),
    "m3_not_certified_by_control": ("certified={\"M3\": bool(c_shuf[\"informative\"][\"M3\"])}", "certified={\"M3\": True}"),
    "reverse_gap_never_flagged": ("\"reverse_gap\": bool(g[\"hi_bonf\"] < -nu),", "\"reverse_gap\": False,"),
    "step_a_perturbation_features_misaligned": ("agg_p[(g, k)] = ens_score(w.Xp[s_idx], mem)", "agg_p[(g, k)] = ens_score(w.Xp[s_idx][:, ::-1], mem)"),
    "pairwise_labels_sign_flipped": ("y = np.sign(dif) * (np.abs(dif)", "y = -np.sign(dif) * (np.abs(dif)"),
    "exclusion_line_deleted_in_source": ("train_folds = [f for f in range(N_FOLDS) if f != h]      # EXCLUDE-LINE: model h never sees fold h", "train_folds = list(range(N_FOLDS))      # EXCLUDE-LINE: model h never sees fold h"),
    "search_and_fit_halves_swapped_in_scoring": ("sel = d.is_search if half == \"search\" else ~d.is_search", "sel = ~d.is_search if half == \"search\" else d.is_search"),
    "informativeness_guard_removed": ("informative = bool(4 * st[\"se\"] <= DELTA[m])", "informative = True"),
    "chance_z_test_removed": ("at_chance = bool(abs(st[\"z\"]) <= Z_MAX)", "at_chance = True"),
    "decide_drops_magnitude": ("gap_m = S and Mg and Rp and Ls", "gap_m = S and Rp and Ls"),
    "prereg_digest_not_checked": ("    if constants_sha256() not in txt:", "    if False:"),
    "collapse_control_accepts_anything": ("\"pass\": bool(abs(gm) <= bound), \"gating\"", "\"pass\": True, \"gating\""),
    "collapse_bound_uses_only_the_mutated_scale": ("bound = max(nu, nu_m, 2.0 * se_m, COLLAPSE_FRAC * abs(g))", "bound = max(nu_m, 2.0 * se_m, COLLAPSE_FRAC * abs(g)) * 0.0"),
    "structural_check_disabled": ("    if in_fold or in_search or sess_overlap:", "    if False:"),
    # ---- AMENDMENT A1 (seed averaging): mutants of the changed lines in stage0.py ...
    "a1_average_weights_not_composite": ("    acc = None\n    for W, b in members:\n        a = aggregate(head_logits(X, W, b))\n        acc = a if acc is None else acc + a\n    return acc if len(members) == 1 else acc / len(members)",
                                         "    W = np.mean([m[0] for m in members], 0)\n    b = np.mean([m[1] for m in members], 0)\n    return aggregate(head_logits(X, W, b))"),
    "a1_ensemble_sums_instead_of_averaging": ("return acc if len(members) == 1 else acc / len(members)", "return acc"),
    "a1_ensemble_uses_first_member_only": ("    for W, b in members:\n        a = aggregate(head_logits(X, W, b))", "    for W, b in members[:1]:\n        a = aggregate(head_logits(X, W, b))"),
    "a1_fan_scores_use_first_member_only": ("a = ens_score(w.X[idx[g]], mem)", "a = ens_score(w.X[idx[g]], mem[:1])"),
    "a1_step_a_perturbations_use_first_member_only": ("agg_p[(g, k)] = ens_score(w.Xp[s_idx], mem)", "agg_p[(g, k)] = ens_score(w.Xp[s_idx], mem[:1])"),
    "a1_replicate_seeds_overlap_the_primary": ("REPL_SEED_BASE = 100 ", "REPL_SEED_BASE = 0 "),
    "a1_disjointness_guard_removed": ("    if not set(prim).isdisjoint(rep):", "    if False:"),
    "a1_seed_sets_ignore_N_SEEDS": ("    prim, rep = tuple(range(s)), tuple(range(REPL_SEED_BASE, REPL_SEED_BASE + s))", "    prim, rep = (0,), (REPL_SEED_BASE,)"),
    "a1_every_seed_trains_the_first_seed": ("per = [fit_all(w, d, rec, s, touch_ddc) for s in seeds]", "per = [fit_all(w, d, rec, seeds[0], touch_ddc) for s in seeds]"),
    "a1_run_all_replicate_not_averaged": ("a1 = run_arm(w, d, R0, repl, idx)", "a1 = run_arm(w, d, R0, repl[:1], idx)"),
    "a1_run_all_exclusion_arm_not_averaged": ("am = run_arm(w, d, R0, prim, idx, fit=fit_all_unexcluded(w, d, R0, prim))", "am = run_arm(w, d, R0, prim, idx, fit=fit_all_unexcluded(w, d, R0, prim[:1]))"),
    "a1_run_all_planted_arm_not_averaged": ("planted[kp] = (run_arm(wp, d, R0, prim, idx), ", "planted[kp] = (run_arm(wp, d, R0, prim[:1], idx), "),
    "a1_run_all_planted_exclusion_arm_not_averaged": ("run_arm(wp, d, R0, prim, idx, fit=fit_all_unexcluded(wp, d, R0, prim)))", "run_arm(wp, d, R0, prim, idx, fit=fit_all_unexcluded(wp, d, R0, prim[:1])))"),
    "a1_run_all_shuffle_world_not_averaged": ("c_shuf_world = control_shuffle_world(w, d, R0, idx, train_seeds=prim)", "c_shuf_world = control_shuffle_world(w, d, R0, idx, train_seeds=(0,))"),
    "a1_shuffle_world_default_trains_one_seed": ("        fa = fit_all_seeds(ws, d, rec, tseeds)", "        fa = fit_all_seeds(ws, d, rec, tseeds[:1])"),
    "a1_exclusion_deleted_fit_not_averaged": ("    return mod.fit_all_seeds(w, d, rec, seed)", "    return mod.fit_all_seeds(w, d, rec, _as_seeds(seed)[:1])"),
    "a1_seed_constants_not_digested": ('"n_seeds": N_SEEDS, "repl_seed_base": REPL_SEED_BASE,\n        "primary_seeds": list(seed_sets()[0]), "replicate_seeds": list(seed_sets()[1]),', ""),
    # ---- ... and of calib_a1.py (the selection rule, the identity check, the blind guard).  Names starting calib_ mutate calib_a1.py
    "calib_fp_bound_off_by_one": ('b = bool(s["null"]["fp_n"] <= FP_MAX)', 'b = bool(s["null"]["fp_n"] < FP_MAX)'),
    "calib_invalid_bound_off_by_one": ('c = bool(s["null"]["invalid_wo_c2b_n"] <= INVALID_MAX)', 'c = bool(s["null"]["invalid_wo_c2b_n"] < INVALID_MAX)'),
    "calib_selects_the_largest_S": ("        if ok and chosen is None:", "        if ok:"),
    "calib_c2b_accepts_any_kappa": ('c2b_ok_le = any(k <= KAPPA_MAX_FOR_SELECTION for k in grid["kappas_passing"])', 'c2b_ok_le = any(True for k in grid["kappas_passing"])'),
    "calib_S1_is_selectable": ("S_CANDIDATES = (3, 5, 8)", "S_CANDIDATES = (1, 3, 5, 8)"),
    "calib_incomplete_calibration_qualifies": ('complete = bool(s["null"]["n_ok"] == len(NULL_SEEDS) and s["n_planted_ok"] == len(KAPPAS))', "complete = True"),
    "calib_fp_counted_with_c2b_as_is": ('fp_n = counts_wo.get("GAP", 0)', 'fp_n = counts_full.get("GAP", 0)'),
    "calib_invalid_counts_c2b": ('inv_n = counts_wo.get("INVALID", 0)', 'inv_n = counts_full.get("INVALID", 0)'),
    "calib_identity_tolerance_loose": ("IDENTITY_TOL = 1e-12 ", "IDENTITY_TOL = 1e-3 "),
    "calib_blind_guard_removed": ("    if not (np.all(L.sub == 1.0) and np.all(L.pdms == 1.0)", "    if False and not (np.all(L.sub == 1.0) and np.all(L.pdms == 1.0)"),
    "calib_lazy_shuffles_use_half_the_seeds": ("tuple(S0.SHUF_SEEDS if seeds is None else seeds)", "tuple(S0.SHUF_SEEDS[::2] if seeds is None else seeds)"),
    "calib_replicate_arm_scored_as_primary": ('return {"primary": a0["gaps"], "replicate": a1["gaps"],', 'return {"primary": a1["gaps"], "replicate": a0["gaps"],'),
    "calib_null_worlds_are_planted": ("S0.plant_labels(w, 0.0, seed=NULL_SEEDS[i])", "S0.plant_labels(w, 1.0, seed=NULL_SEEDS[i])"),
    "calib_planted_world_wrong_seed": ("S0.plant_labels(w, kappa, seed=S0.PLANT_SEED)", "S0.plant_labels(w, kappa, seed=S0.PLANT_SEED + 1)"),
}


def run_one(name, spec, workdir_root, timeout):
    old, new = spec[0], spec[1]
    target = "calib_a1.py" if name.startswith("calib_") else "stage0.py"
    root = tempfile.mkdtemp(prefix=f"mut_{name}_", dir=workdir_root)
    d = os.path.join(root, "code")                                  # code/ and raw/ side by side, as in the package, so the tests that read ../raw (the sealed excerpt) run instead of skipping
    os.makedirs(d)
    os.makedirs(os.path.join(root, "raw"))
    excerpt = os.path.join(HERE, "..", "raw", "RESULT_STAGE0_controls_only.json")
    if os.path.exists(excerpt):
        shutil.copy(excerpt, os.path.join(root, "raw", "RESULT_STAGE0_controls_only.json"))
    for fn in ("stage0.py", "synth.py", "test_stage0.py", "geometry_real.json", "calib_a1.py", "golden_digests.py"):
        shutil.copy(os.path.join(HERE, fn), os.path.join(d, fn))
    p = os.path.join(d, target)
    src = open(p, encoding="utf-8").read().replace("\r\n", "\n")
    if src.count(old) != 1:
        return {"name": name, "status": "BAD_MUTANT", "occurrences": src.count(old), "target": target}
    open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    # A1 and calib mutants: the A1 tests (which were written to kill them) run FIRST; a survivor is then re-run against the WHOLE suite before it is called a survivor
    stages = [("A1 tests", ["-k", "A1 or digest"]), ("whole suite", [])] if name.startswith(("a1_", "calib_")) else [("whole suite", [])]
    res = None
    try:
        for label, extra in stages:
            r = subprocess.run([PY, "-m", "pytest", "test_stage0.py", "-x", "-q", "--no-header", "-p", "no:cacheprovider", *extra], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
            tail = (r.stdout or "").strip().splitlines()[-3:]
            killed = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")][:2]
            res = {"name": name, "target": target, "status": "KILLED" if r.returncode != 0 else "SURVIVED", "killed_by": label if r.returncode != 0 else None, "rc": r.returncode, "first_failures": killed, "tail": tail[-1] if tail else ""}
            if r.returncode != 0:
                break
    except subprocess.TimeoutExpired:
        res = {"name": name, "target": target, "status": "KILLED", "rc": "timeout", "note": "the mutant hangs the suite"}
    res["seconds"] = round(time.time() - t0, 1)
    shutil.rmtree(root, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "raw", "MUTATION_CENSUS.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    names = [n for n in MUTANTS if not a.only or n in a.only.split(",")]
    root = tempfile.mkdtemp(prefix="mutcensus_")
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(lambda n: run_one(n, MUTANTS[n], root, a.timeout), names))
    shutil.rmtree(root, ignore_errors=True)
    surv = [r["name"] for r in rows if r["status"] == "SURVIVED"]
    bad = [r["name"] for r in rows if r["status"] == "BAD_MUTANT"]
    res = {"what": "mutation census of stage0.py and calib_a1.py against test_stage0.py", "n": len(rows), "killed": sum(r["status"] == "KILLED" for r in rows), "survived": surv, "bad_mutants": bad, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['status']:10} {r['name']:44} {r.get('first_failures', [''])[:1]}")
    print("killed", res["killed"], "of", res["n"], "survived", surv, "bad", bad)
    return 1 if surv or bad else 0


if __name__ == "__main__":
    sys.exit(main())
