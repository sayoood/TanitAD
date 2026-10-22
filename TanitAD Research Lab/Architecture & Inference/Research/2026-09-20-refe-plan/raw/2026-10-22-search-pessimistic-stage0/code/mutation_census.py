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
    "role_shift_oof_is_in": ("W, b = models[(g + k) % N_FOLDS]", "W, b = models[(g + k + 1) % N_FOLDS]"),
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
    "step_a_perturbation_features_misaligned": ("agg_p[(g, k)] = aggregate(head_logits(w.Xp[s_idx], W, b))", "agg_p[(g, k)] = aggregate(head_logits(w.Xp[s_idx][:, ::-1], W, b))"),
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
}


def run_one(name, spec, workdir_root, timeout):
    old, new = spec[0], spec[1]
    d = tempfile.mkdtemp(prefix=f"mut_{name}_", dir=workdir_root)
    for fn in ("stage0.py", "synth.py", "test_stage0.py", "geometry_real.json"):
        shutil.copy(os.path.join(HERE, fn), os.path.join(d, fn))
    p = os.path.join(d, "stage0.py")
    src = open(p, encoding="utf-8").read().replace("\r\n", "\n")
    if src.count(old) != 1:
        return {"name": name, "status": "BAD_MUTANT", "occurrences": src.count(old)}
    open(p, "w", encoding="utf-8", newline="\n").write(src.replace(old, new))
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    try:
        r = subprocess.run([PY, "-m", "pytest", "test_stage0.py", "-x", "-q", "--no-header", "-p", "no:cacheprovider"], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
        tail = (r.stdout or "").strip().splitlines()[-3:]
        killed = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("FAILED") or ln.startswith("ERROR")][:2]
        res = {"name": name, "status": "KILLED" if r.returncode != 0 else "SURVIVED", "rc": r.returncode, "first_failures": killed, "tail": tail[-1] if tail else ""}
    except subprocess.TimeoutExpired:
        res = {"name": name, "status": "KILLED", "rc": "timeout", "note": "the mutant hangs the suite"}
    res["seconds"] = round(time.time() - t0, 1)
    shutil.rmtree(d, ignore_errors=True)
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
    res = {"what": "mutation census of stage0.py against test_stage0.py", "n": len(rows), "killed": sum(r["status"] == "KILLED" for r in rows), "survived": surv, "bad_mutants": bad, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['status']:10} {r['name']:44} {r.get('first_failures', [''])[:1]}")
    print("killed", res["killed"], "of", res["n"], "survived", surv, "bad", bad)
    return 1 if surv or bad else 0


if __name__ == "__main__":
    sys.exit(main())
