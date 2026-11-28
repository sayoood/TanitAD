"""K-MUT driver (PREREG_V8F_N1DH section 5): for each mutant, edit the build tree (one or more exact source replacements), run the NAMED tests (each must go RED), restore the file, verify its sha256 equals the pre-run value.
Writes mutation_table_n1dh.json. One pytest at a time, CPU only. usage: run_mutants_n1dh.py <tree> <outjson> <tmpdir> [mutant ids...]"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TREE, OUT, TMP = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ONLY = set(sys.argv[4:])
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
T = "tests/test_v8f_n1dh.py::"
S = "tests/test_strat1_n1dh_check.py::"
MODEL, TRAIN, LOADER = "stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_t1_eval_rbd.py"
REACH = T + "test_reach_nonzero_gradients_on_both_modules_and_the_controls_read_exactly_zero"
NOOWN = T + "test_reach_a_batch_with_no_own_code_row_gives_exactly_zero_ce_value_and_gradient"
OWN = T + "test_own_the_ce_rows_are_exactly_the_rows_where_o_read_t_own_code"
Z1 = T + "test_z1_cond_at_step_0_equals_off_bit_for_bit_except_the_route_term_and_the_rng_streams_are_the_off_models"
KL = T + "test_klabel_route_stage_equals_the_label_source_called_directly_on_analytic_arcs"
KLR = T + "test_klabel_real_d1ho_sample_class_counts_equal_the_label_source_called_directly"
E2E = T + "test_end_to_end_n1dh_arm_probe_levers_group_class_record_and_checkpoint"
HON = T + "test_gdvb_passes_an_honoured_argv_and_names_the_levers_in_built_levers"
FLAGS = T + "test_flags_default_off_have_flag_checks_a_choice_check_and_a_wm_only_default"
CONDLINE = T + "test_probe_refuses_a_model_whose_conditioning_line_is_deleted"
CELINE = T + "test_probe_refuses_a_model_whose_ce_term_is_deleted_from_the_total"
MASKP = T + "test_probe_refuses_a_model_without_the_own_code_mask"
DETP = T + "test_probe_refuses_a_model_whose_posterior_is_not_detached"
CE_LINE = '        total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total      # PREREG_V8F_N1DH: ONE line (a mutation deletes it); off adds nothing: the tip\'s total\n'
COND_LINE = "                hf = hf + self.route_emb(torch.softmax(self._route_logits, -1).detach())          # ONE line (a mutation deletes it): zero-init route_emb, so hf + 0.0 == hf at step 0\n"

MUTANTS = [
    ("a_delete_the_conditioning_line", [(MODEL, COND_LINE, "")], [REACH, CONDLINE]),
    ("b_delete_the_ce_loss_line", [(MODEL, CE_LINE, "")], [REACH, CELINE]),
    ("c_remove_the_own_code_mask", [(MODEL, 'rm = (batch["route_ok"].bool() & out["route_own"]).float()', 'rm = batch["route_ok"].bool().float()')], [OWN, NOOWN, MASKP]),
    ("d_no_detach_on_the_posterior", [(MODEL, "torch.softmax(self._route_logits, -1).detach())", "torch.softmax(self._route_logits, -1))")], [REACH, DETP]),
    ("e_route_emb_default_nonzero_init", [(MODEL, "            nn.init.zeros_(self.route_emb.weight)\n", "            pass\n")], [Z1]),
    ("f_route_modules_not_built_under_a_forked_rng", [(MODEL, "            with torch.random.fork_rng(devices=[]):\n                self.route_head = nn.Linear(fl, N_ROUTE)", "            if True:\n                self.route_head = nn.Linear(fl, N_ROUTE)")], [Z1]),
    ("g_own_mask_inverted", [(MODEL, "                own = ~use_t ", "                own = use_t ")], [OWN, NOOWN]),
    ("h_label_permuted_within_the_batch_card_mutation_c", [(TRAIN, "    return lat.long(), ok\n", "    return torch.roll(lat.long(), 1), ok\n")], [KL, KLR]),
    ("i_remove_the_gdvb_entries", [(TRAIN, '    "aux_route": "lever:aux_route", "w_aux_route": "lever:w_aux_route",\n', "")], [FLAGS, HON, E2E]),
    ("j_flags_not_applied_in_make_config", [(TRAIN, "                           aux_route=a.aux_route, w_aux_route=a.w_aux_route,", "                           ")], [FLAGS, HON]),
    ("k_declared_vs_built_clause_not_called", [(TRAIN, "    bad += bad_aux_route(a, cfg, model, opt) ", "    bad += [] ")],
     [T + "test_gdvb_refuses_declared_cond_but_built_without_the_modules", T + "test_gdvb_refuses_a_zero_weight_with_cond_and_a_weight_with_off"]),
    ("l_probe_not_wired_into_the_selfcheck", [(TRAIN, '        sc["aux_route_probe"] = arp\n', "        pass\n")], [E2E]),
    ("m_new_key_prefixes_not_admitted_by_the_init_ckpt_gate", [(MODEL, "O_CODE_KEY_PREFIXES + ROUTE_KEY_PREFIXES ", "O_CODE_KEY_PREFIXES ")],
     [FLAGS, T + "test_fine_tune_from_a_checkpoint_without_the_route_keys_passes_the_init_ckpt_gate_and_only_with_the_flag"]),
    ("n_drop_the_route_keys_from_one_eval_loader_v8f_t1_eval_rbd", [(LOADER, 'cfg = T.M.V8FMiniConfig(**ck["cfg"])', 'cfg = T.M.V8FMiniConfig(**{k: v for k, v in ck["cfg"].items() if k not in ("aux_route", "w_aux_route")})')],
     [T + "test_roundtrip_each_loader_builds_the_route_modules_from_the_checkpoint_cfg_strict_and_equal[v8f_t1_eval_rbd.py]"]),
    ("o_loop_never_calls_route_stage", [(TRAIN, "                    route_lab = route_stage(b)\n", "                    pass\n")], [E2E]),
    ("p_checker_drops_the_h_gate_read", [("stack/scripts/strat1_check.py", "    bad += h_gate_problems(kind, toks, sc) ", "    bad += [] ")], [S + "test_n1dh_checks_pass_on_the_real_checker[gate]"]),
    ("q_ce_weight_ignored_in_the_total", [(MODEL, 'total = total + cfg.w_aux_route * T["route"] if cfg.aux_route != "off" else total', 'total = total + 0.0 * T["route"] if cfg.aux_route != "off" else total')], [REACH, CELINE]),
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
            tail = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR")) or " failed" in ln or " passed" in ln][-3:]
            res.append({"test": tid.split("::")[-1], "rc": r.returncode, "red": r.returncode != 0, "tail": tail})
    finally:
        for f in files:
            (TREE / f).write_bytes(raws[f])
    after = {f: sha(TREE / f) for f in files}
    rows.append({"mutant": mid, "files": files, "edits": len(edits), "tests": res, "named_test_red": all(x["red"] for x in res), "any_named_test_red": any(x["red"] for x in res),
                 "file_restored": all(before[f] == after[f] for f in files), "sha256_after": after})
    print(mid, "RED" if rows[-1]["any_named_test_red"] else "STILL GREEN", [x["red"] for x in res], flush=True)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
