"""K-MUT driver (PREREG_V8F_N1DM draft, section 5): for each mutant, edit the build tree (one or more exact source replacements), run the NAMED tests (each must go RED by an assertion, not by a crash of the collection),
restore the file, verify its sha256 equals the pre-run value. Writes mutation_table_n1dm.json. One pytest at a time, CPU only. usage: run_mutants_n1dm.py <tree> <outjson> <tmpdir> [mutant ids...]"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TREE, OUT, TMP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ONLY = set(sys.argv[4:])
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T = "tests/test_v8f_n1dm.py::"
TH = "tests/test_v8f_n1dh.py::"
S = "tests/test_strat1_n1dm_check.py::"
MODEL, TRAIN, LOADER, CHECK = "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_t1_eval_rbd.py", "stack/scripts/strat1_check.py"
REACH = T + "test_reach_every_branch_gets_a_nonzero_gradient_and_the_controls_read_exactly_zero"
NOOWN = T + "test_reach_a_batch_with_no_own_code_row_gives_exactly_zero_ce_and_the_branches_still_train"
FORM = T + "test_formula_the_plan_is_head_o_plus_the_p_weighted_branches_recomputed_from_hooks"
PERM = T + "test_formula_p_permuted_across_rows_changes_the_plan_to_the_formula_with_the_permuted_p"
NOARG = T + "test_formula_the_selection_is_one_deterministic_soft_form_no_argmax_no_sampling"
DETACH = T + "test_the_posterior_is_detached_decoder_gradient_on_route_head_is_exactly_zero_after_training_steps"
Z1 = T + "test_z1_modes_at_step_0_equals_off_bit_for_bit_except_the_route_term"
Z0C = T + "test_z0c_the_modes_build_draws_nothing_from_the_global_rng"
E2E = T + "test_end_to_end_n1dm_arm_probe_levers_group_class_record_checkpoint_and_the_checker_reads_the_real_selfcheck"
PROBE = T + "test_probe_modes_reads_every_branch_the_permutation_the_identity_and_the_exact_zero_controls"
GDVB = T + "test_gdvb_refuses_a_nonzero_start_wrong_shape_bias_a_wrong_count_and_a_missing_optimiser_entry"
GDVB2 = T + "test_gdvb_refuses_declared_modes_built_without_or_with_the_wrong_modules"
INIT = T + "test_init_ckpt_fresh_keys_and_resume_refusal"
CONST = T + "test_the_snapshot_is_the_1f217bf0_model_and_the_modes_constants"
LOAD = T + "test_roundtrip_each_loader_builds_the_branches_from_the_checkpoint_cfg_strict_and_equal[v8f_t1_eval_rbd.py]"
CE_LINE = '        total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total      # PREREG_V8F_N1DH: ONE line (a mutation deletes it); off adds nothing: the tip\'s total\n'
ADD_LINE = "                    coef = coef + p_mode[:, k:k + 1] * br(hf)\n"
P_LINE = "p_mode = torch.softmax(self._route_logits.float(), -1).detach()"

MUTANTS = [
    ("a_delete_the_branch_add", [(MODEL, ADD_LINE, "                    pass\n")], [REACH, FORM, E2E]),
    ("b_permute_p_across_rows", [(MODEL, P_LINE, P_LINE + ".roll(1, 0)")], [FORM, PERM]),
    ("c_no_detach_on_p", [(MODEL, P_LINE, "p_mode = torch.softmax(self._route_logits.float(), -1)")], [REACH, DETACH, PROBE]),
    ("d_hard_argmax_selection", [(MODEL, P_LINE, "p_mode = torch.nn.functional.one_hot(self._route_logits.argmax(-1), 3).float()")], [FORM, NOARG]),
    ("e_branch_default_nonzero_init", [(MODEL, "                    nn.init.zeros_(br.weight)", "                    nn.init.normal_(br.weight, std=0.02)")], [Z1, GDVB]),
    ("f_delete_the_ce_loss_line", [(MODEL, CE_LINE, "")], [REACH, E2E]),
    ("g_remove_the_own_code_mask", [(MODEL, 'rm = (batch["route_ok"].bool() & out["route_own"]).float()', 'rm = batch["route_ok"].bool().float()')], [NOOWN]),
    ("h_branches_not_in_the_route_group", [(MODEL, "for m_ in (self.route_head, self.route_emb, self.route_br) if m_ is not None]", "for m_ in (self.route_head, self.route_emb) if m_ is not None]")], [Z1]),
    ("i_modules_not_built_under_a_forked_rng", [(MODEL, "            with torch.random.fork_rng(devices=[]):\n                self.route_head = nn.Linear(fl, N_ROUTE)", "            if True:\n                self.route_head = nn.Linear(fl, N_ROUTE)")], [Z0C, Z1]),
    ("j_branches_with_a_bias", [(MODEL, "int(self.head_o[2].out_features), bias=False) for _ in range(N_ROUTE)])", "int(self.head_o[2].out_features), bias=True) for _ in range(N_ROUTE)])")], [Z1, GDVB]),
    ("k_branch_prefix_not_admitted_by_the_init_ckpt_gate", [(MODEL, "+ ROUTE_KEY_PREFIXES + ROUTE_BR_KEY_PREFIXES      #", "+ ROUTE_KEY_PREFIXES      #")], [CONST, INIT]),
    ("l_gdvb_modes_clause_dropped", [(TRAIN, '    if a.aux_route == "modes":', '    if False and a.aux_route == "modes":')], [GDVB]),
    ("m_probe_permutation_hook_is_a_noop", [(TRAIN, "lambda mod_, i_, out_: out_.roll(1, 0))", "lambda mod_, i_, out_: out_)")], [PROBE, E2E]),
    ("n_probe_not_wired_into_the_selfcheck", [(TRAIN, '        sc["aux_route_probe"] = arp\n', "        pass\n")], [E2E]),
    ("o_checker_drops_the_m_gate_read", [(CHECK, "    bad += m_gate_problems(kind, toks, sc) ", "    bad += [] ")], [S + "test_n1dm_checks_pass_on_the_real_checker[gate]"]),
    ("p_one_eval_loader_drops_the_route_keys_v8f_t1_eval_rbd", [(LOADER, 'cfg = T.M.V8FMiniConfig(**ck["cfg"])', 'cfg = T.M.V8FMiniConfig(**{k: v for k, v in ck["cfg"].items() if k not in ("aux_route", "w_aux_route")})')], [LOAD]),
    ("q_ce_weight_ignored_in_the_total", [(MODEL, 'total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total', 'total = total + 0.0 * T["route"] if cfg.aux_route != "off" else total')], [REACH]),
    ("r_every_branch_reads_p_column_0", [(MODEL, ADD_LINE, "                    coef = coef + p_mode[:, 0:1] * br(hf)\n")], [FORM]),
    ("s_p_is_the_constant_one_third_not_the_route_head", [(MODEL, P_LINE, "p_mode = torch.full_like(self._route_logits.float(), 1.0 / 3.0)")], [FORM, PERM]),
    ("t_label_permuted_within_the_batch_n1dh_mutation_c", [(TRAIN, "    return lat.long(), ok\n", "    return torch.roll(lat.long(), 1), ok\n")], [TH + "test_klabel_route_stage_equals_the_label_source_called_directly_on_analytic_arcs"]),
    ("u_route_emb_present_beside_the_branches_gdvb", [(TRAIN, "    wrong_extra = has_emb != (a.aux_route == \"cond\") or has_br != (a.aux_route == \"modes\")", "    wrong_extra = False")], [GDVB2]),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows = []
for mid, edits, tests in MUTANTS:
    if ONLY and mid not in ONLY:
        continue
    files = sorted({e[0] for e in edits})
    raws = {f: (TREE / f).read_bytes() for f in files}
    before = {f: hashlib.sha256(raws[f]).hexdigest() for f in files}
    texts = {f: raws[f].decode("utf-8") for f in files}
    for f, old, new in edits:
        assert texts[f].count(old) == 1, (mid, f, old[:60], texts[f].count(old))
        texts[f] = texts[f].replace(old, new)
    for f in files:
        (TREE / f).write_bytes(texts[f].encode("utf-8"))
    res = []
    try:
        for tid in tests:
            run_tmp = TMP / f"mut_{mid[:1]}"
            run_tmp.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONUTF8": "1", "TMP": str(run_tmp), "TEMP": str(run_tmp), "PYTHONPATH": str(TREE / "stack")}
            r = subprocess.run([PY, "-m", "pytest", tid, "-q", "-x", "-p", "no:cacheprovider", f"--basetemp={run_tmp / 'bt'}"], cwd=str(TREE / "stack"), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) or " failed" in ln or " passed" in ln or " error" in ln][-3:]
            res.append({"test": tid.split("::")[-1], "rc": r.returncode, "red": r.returncode != 0, "red_by_a_failed_test_not_a_collection_error": r.returncode == 1 and " failed" in r.stdout, "tail": tail})
    finally:
        for f in files:
            (TREE / f).write_bytes(raws[f])
    after = {f: sha(TREE / f) for f in files}
    rows.append({"mutant": mid, "files": files, "edits": len(edits), "tests": res, "named_test_red": all(x["red"] for x in res), "any_named_test_red": any(x["red"] for x in res),
                 "all_red_by_failed_tests": all(x["red_by_a_failed_test_not_a_collection_error"] for x in res), "file_restored": all(before[f] == after[f] for f in files), "sha256_after": after})
    print(mid, "RED" if rows[-1]["any_named_test_red"] else "STILL GREEN", [x["red"] for x in res], flush=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
