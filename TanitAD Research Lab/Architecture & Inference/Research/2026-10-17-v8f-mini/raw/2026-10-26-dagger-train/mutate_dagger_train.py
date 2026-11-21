"""Mutation proof for the DAgger lever's TRAINER side (PREREG_V8F_DAGGER): one mutant per rule, each must turn ``stack/tests/test_v8f_dagger_train.py`` RED.

R77: the mutants run on a COPY. This script builds its own minimal copy of the tree (``--work``, default E:/Projects/TanitAD-artifacts/dt/mut) from the tree it is given, mutates ONLY the copy, restores the
copy's trainer after every mutant, and at the end checks that every delivered file is byte-identical to its sha256 at start (it refuses to start if ``--work`` is inside the delivery tree).

usage:  python mutate_dagger_train.py <delivery tree root> <python> [--work DIR] [--check] [--list] [--only NAME[,NAME]] [--group G] [--log FILE]
  --check   build the copy, verify that every anchor is found EXACTLY ONCE in its region and every mutated text compiles; run nothing
  --list    print the mutants
Per group: the UNMUTATED copy is run green first on the group's whole selector; then each mutant of the group runs its own (narrower) selector with -x and must fail (rc != 0, and not a collection error).
Only one pytest process runs at a time (CUDA_VISIBLE_DEVICES=-1, basetemp under the work dir). A mutant that survives is printed as SURVIVOR; the exit code is 1 then.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import time

TRAIN = "stack/scripts/v8f_mini_train.py"
TEST = "stack/tests/test_v8f_dagger_train.py"
PKG = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
REF = PKG + "/raw/2026-10-26-dagger-train/pre_dagger_v8f_mini_train.py.ref"
COPY_TREES = ["stack/tanitad", "taniteval/taniteval"]
COPY_FILES = ["stack/pyproject.toml", "stack/tests/conftest.py", "taniteval/conftest.py", TEST, TRAIN, REF, PKG + "/raw/eval_grid_windows.json"]
DELIVERED = [TRAIN, TEST, REF, "stack/tanitad/data/v8f_dagger.py", "stack/scripts/v8f_dagger_bank.py", "stack/tests/test_v8f_dagger.py"]
PROBE = ("def dagger_probe(", "# ------------------------------------------------------------------------------------------------ the encoder (PREREG_V8F_LORA)")
BDAG = ("def bad_dagger(", "def bad_init_ckpt(")
BDATA = ("def build_data(", "def to_dev(")

# ---- selectors (pytest -k): GROUP = the union run green first; each mutant's own is a subset
K_SHA = ("test_a_matching_declared_sha256 or test_a_declared_sha256 or test_the_gate_rehashes or test_a_declared_value_that_is_not or test_the_sha256_shape or "
         "test_required_at_p_above_zero or test_a_wrong_declared_sha256")                                                      # the FOURTH flag, --dagger-bank-sha256
K_FLAGS = ("test_the_four_flags or test_a_flag_without_an_entry or test_g_dvb_passes_what_was_built or test_g_dvb_refuses_a_swap_built or test_g_dvb_refuses_an_illegal or "
           "test_g_dvb_refuses_the_swap_in_steer or test_a_declared_flag_with_no_swap or " + K_SHA)
K_GBANK = "test_g_dvb_refuses_a_bank_generated or test_g_dvb_refuses_a_bank_whose_history or test_g_dvb_refuses_missing_or_failed or test_the_clean_bank_loads"
K_KB = "test_k_b3 or test_k_b2 or test_coverage_below or test_a_missing_bank or test_p_zero_reads_no_bank or test_the_bank_is_re_indexed"
K_PROBE = "test_the_probe"
K_E2E = ("test_a_default_run or test_dag_0 or test_flags_to_config or test_p_one or test_a_real_run_gives or test_the_check_that_the_loss or test_the_swap_is_what or test_the_log_interval or "
         "test_a_resumed_run or test_a_refused_declaration or test_a_bank_refused_at_load or test_the_gate_refuses_a_swap")
K_IDENT = "test_defaults_reproduce or test_the_saved_reference or test_the_single_applying_lines"
K_WSHA = "test_every_run_records_the_train_window_list_sha256 or test_defaults_reproduce"
GROUPS = {"flags": K_FLAGS, "gbank": K_GBANK, "kb": K_KB, "probe": K_PROBE, "e2e": K_E2E, "ident": K_IDENT, "wsha": K_WSHA}


DISK_FULL = ("No space left", "badbit", "Errno 28", "unexpected pos", "ENOSPC")


def M(name, rule, old, new, group, k, region=None):
    return dict(name=name, rule=rule, old=old, new=new, group=group, k=k, region=region)


MUTANTS = [
    # ---- the flags and their G-DVB entries
    M("F01_p_default_is_not_off", "--dagger-p defaults to OFF (DG.DAGGER_DEFAULTS, 0.0)", 'type=float, default=DG.DAGGER_DEFAULTS["p"],', "type=float, default=0.1,", "flags", "test_the_four_flags"),
    M("F02_seed_has_no_gdvb_entry", "every flag has a FLAG_CHECKS entry (dagger_seed)", ', "dagger_seed": "dagger:dagger_seed",', ",", "flags", "test_the_four_flags or test_a_flag_without_an_entry"),
    M("F03_bank_has_no_gdvb_entry", "every flag has a FLAG_CHECKS entry (dagger_bank)", '"dagger_bank": "dagger:dagger_bank", ', "", "flags", "test_the_four_flags or test_a_flag_without_an_entry"),
    M("F04_gate_does_not_compare_p", "G-DVB: the built p is the declared p", '    for k in ("dagger_p", "dagger_seed"):\n        if lv.get(k) != declared[k]:', '    for k in ("dagger_seed",):\n        if lv.get(k) != declared[k]:', "flags", "test_g_dvb_refuses_a_swap_built", BDAG),
    M("F05_gate_does_not_compare_seed", "G-DVB: the built seed is the declared seed", '    for k in ("dagger_p", "dagger_seed"):\n        if lv.get(k) != declared[k]:', '    for k in ("dagger_p",):\n        if lv.get(k) != declared[k]:', "flags", "test_g_dvb_refuses_a_swap_built", BDAG),
    M("F06_gate_does_not_compare_kin_mode", "G-DVB: the built swap's kin_mode is the declared one", '    if lv.get("kin_mode") != a.kin_mode:', "    if False:", "flags", "test_g_dvb_refuses_the_swap_in_steer", BDAG),
    M("F07_gate_ignores_the_swaps_own_problems", "G-DVB: a lever declared without an effect / illegal law is refused (swap.problems())", '    bad += [f"DAgger swap: {m}" for m in dag.problems()]', "    bad += []", "flags", "test_g_dvb_refuses_an_illegal or test_g_dvb_refuses_the_swap_in_steer", BDAG),
    M("F08_ego_exclusivity_dropped", "--dagger-p > 0 with --ego-perturb-p > 0 is refused", "    if a.dagger_p > 0 and a.ego_perturb_p > 0:", "    if False:", "flags", "test_g_dvb_refuses_an_illegal", BDAG),
    M("F09_no_swap_built_not_refused", "a declared flag with no swap built is refused (a flag that builds nothing)", '        return [f"{moved} are set but no DAgger swap was built: a flag that builds nothing is refused, not ignored"] if moved else []', "        return []", "flags", "test_a_declared_flag_with_no_swap", BDAG),
    M("F10_declared_bank_with_no_bank_built", "a bank declared at p = 0 builds nothing and is refused", '    if dag.bank is None:\n        bad.append(f"lever dagger_bank: declared', '    if False:\n        bad.append(f"lever dagger_bank: declared', "flags", "test_g_dvb_refuses_an_illegal", BDAG),
    # ---- G-DVB on the bank
    M("B01_bank_path_not_compared", "G-DVB: the bank's path against the declaration", "    if want != got:", "    if False:", "flags", "test_g_dvb_refuses_a_swap_built", BDAG),
    M("B02_bank_sha_not_read_back", "G-DVB: the bank's sha256 against the file on disk", '    if lv.get("dagger_bank_sha256") != disk:', "    if False:", "flags", "test_g_dvb_refuses_a_swap_built", BDAG),
    M("B03_schema_not_checked", "G-DVB: the bank's schema", '    if hdr.get("schema") != DG.SCHEMA:', "    if False:", "gbank", "test_g_dvb_refuses_a_bank_generated", BDAG),
    M("B04_rollout_not_checked", "G-DVB: the bank's configuration equals PREREG section 1 (rollout)", '    if hdr.get("rollout") != DG.ROLLOUT:', "    if False:", "gbank", "test_g_dvb_refuses_a_bank_generated", BDAG),
    M("B05_constants_not_checked", "G-DVB: the bank's matching constants sha256", '    if hdr.get("constants_sha256") != DG.constants_sha256():', "    if False:", "gbank", "test_g_dvb_refuses_a_bank_generated", BDAG),
    M("B06_cfg_not_checked", "G-DVB: the bank was generated with this model's cfg", '            if (hdr.get("cfg") or {}).get(k) != getattr(cfg, k):', "            if False:", "gbank", "test_g_dvb_refuses_a_bank_generated", BDAG),
    M("B07_hmax_not_checked", "G-DVB: the bank's history length is the model's hmax", '        if lv.get("bank_hmax") != cfg.hmax:', "        if False:", "gbank", "test_g_dvb_refuses_a_bank_whose_history", BDAG),
    M("B08_controls_record_not_required", "G-DVB: K-B2 / K-B3 / COVERAGE were read (the record exists)", "    if not ctl:", "    if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    M("B09_controls_bank_not_matched", "G-DVB: the controls were read on THIS bank", '        if ctl.get("bank_sha256") != lv.get("dagger_bank_sha256"):', "        if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    M("B10_kb2_record_not_matched", "G-DVB: K-B2 was read against this bank's generating checkpoint", '        if (ctl.get("kb2") or {}).get("ckpt_sha256") != hdr.get("ckpt_sha256"):', "        if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    M("B11_init_ckpt_not_reread", "G-DVB: K-B2 re-read at the gate (the run's init checkpoint sha256)", '        if init_sha != hdr.get("ckpt_sha256"):', "        if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    M("B12_kb3_record_not_matched", "G-DVB: K-B3 was read with this bank's window list and an empty eval overlap", '        if (ctl.get("kb3") or {}).get("window_list_sha256") != hdr.get("window_list_sha256") or (ctl.get("kb3") or {}).get("overlap_with_eval") != 0:', "        if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    M("B13_coverage_record_not_required", "G-DVB: COVERAGE is recorded as passed", '        if not isinstance(ctl.get("coverage"), dict) or ctl["coverage"].get("blocked") is not False:', "        if False:", "gbank", "test_g_dvb_refuses_missing_or_failed", BDAG),
    # ---- the load: K-B3 (three checks), K-B2, COVERAGE, the bank's identity
    M("K01_bare_shard_accepted", "the bank is a merged bank (its manifest's sha256 is its identity)", "    if bank0.manifest is None:", "    if False:", "kb", "test_a_missing_bank"),
    M("K02_eval_overlap_not_checked", "K-B3 check 3: bank clips are disjoint from eval-141", "kb3 = bank0.check_train_only(train_clips=clips, eval_clips=eval_clips,", "kb3 = bank0.check_train_only(train_clips=clips, eval_clips=[],", "kb", "test_k_b3_each_check or test_k_b3_the_banks_recorded"),
    M("K03_train_list_is_the_join", "K-B3 check 1: every bank clip is in the ARM's TRAIN clip list (not merely in the join)", "kb3 = bank0.check_train_only(train_clips=clips,", 'kb3 = bank0.check_train_only(train_clips=list(map(str, J["clip"])),', "kb", "test_k_b3_each_check or test_k_b3_a_clip_excluded"),
    M("K04_window_sha_is_the_banks_own", "K-B3 check 2: the bank's window-list sha256 equals the TRAINER's", "train_window_list_sha256=wsha)", 'train_window_list_sha256=bank0.header.get("window_list_sha256"))', "kb", "test_k_b3_each_check"),
    M("K05_kb2_not_checked", "K-B2: the bank's generating checkpoint is the --init-ckpt", '    kb2 = bank0.check_on_policy(init["sha256"])', '    kb2 = {"ckpt_sha256": bank0.header.get("ckpt_sha256")}', "kb", "test_k_b2"),
    M("K06_init_ckpt_not_required", "K-B2 needs --init-ckpt", '    if not init.get("sha256"):', "    if False:", "kb", "test_k_b2"),
    M("K07_coverage_not_enforced", "COVERAGE < 0.5 blocks the launch", '    if cov["blocked"]:', "    if False:", "kb", "test_coverage_below"),
    M("K08_bank_not_re_indexed_by_name", "the bank is re-indexed to the trainer's join BY NAME", '    bank = DG.DaggerBank(bank0.a, bank0.header, clip_names=J["clip"], manifest=bank0.manifest, sha256=bank0.sha256, path=bank0.path)',
      "    bank = DG.DaggerBank(bank0.a, bank0.header, manifest=bank0.manifest, sha256=bank0.sha256, path=bank0.path)", "kb", "test_the_bank_is_re_indexed"),
    M("K09_controls_not_recorded", "the controls that were read are recorded in info (selfcheck.json) and read back by the gate", '        info["dagger"] = ctl ', '        info["dagger_unrecorded"] = ctl ', "kb", "test_the_clean_bank_loads or test_g_dvb_refuses_missing_or_failed"),
    M("K10_grid_not_required", "K-B3: the canonical eval-141 grid is REQUIRED (a check that cannot run is a refusal)", "    if not os.path.exists(grid):", "    if False:", "kb", "test_a_missing_bank"),
    M("K11_header_eval_list_ignored", "K-B3: the bank's own recorded eval list is also read", '    if hdr:\n        src["bank_header"] = hdr', '    if False:\n        src["bank_header"] = hdr', "kb", "test_k_b3_the_banks_recorded"),
    M("K12_cache_eval_manifest_ignored", "K-B3: the cache's eval manifest is also read", '    if os.path.exists(os.path.join(root, "eval", "manifest.jsonl")):', "    if False:", "kb", "test_k_b3_the_banks_recorded"),
    M("K13_vacuous_grid_accepted", "K-B3: a grid that lists no clip is refused", '    if not src["grid"]:', "    if False:", "kb", "test_k_b3_the_banks_recorded"),
    M("K14_synthetic_grid_path", "K-B3: --synthetic reads the synthetic world's own grid", 'os.path.join(a.out, "_synthetic", "eval_grid_windows.json") if a.synthetic else DAGGER_EVAL_GRID', "DAGGER_EVAL_GRID", "kb", "test_k_b3_the_banks_recorded"),
    M("K15_bank_read_at_p_zero", "--dagger-p 0 reads no bank", "    if a.dagger_bank and a.dagger_p > 0:", "    if a.dagger_bank:", "kb", "test_p_zero_reads_no_bank"),
    M("K16_wrong_seed_built", "argv --dagger-seed becomes the built seed", "seed=a.dagger_seed, kin_mode=a.kin_mode)", "seed=0, kin_mode=a.kin_mode)", "flags", "test_g_dvb_passes_what_was_built"),
    M("K17_wrong_p_built", "argv --dagger-p becomes the built p", "DG.DaggerSwap(bank, p=a.dagger_p,", "DG.DaggerSwap(bank, p=min(1.0, 2 * a.dagger_p),", "flags", "test_g_dvb_passes_what_was_built"),
    # ---- G-LIVE: dagger_probe
    M("P01_off_not_skipped", "the probe does nothing when the lever is off", "    if not dag.active:", "    if False:", "probe", "test_the_probe_passes", PROBE),
    M("P02_no_eligible_window_accepted", "the probe needs an eligible window", "    if n_sel < 1:", "    if False:", "probe", "test_the_probe_refuses_a_batch_with_no_eligible", PROBE),
    M("P03_batch_keys_not_compared", "the swap changes no key of the batch", "    if set(pert) != set(batch):", "    if False:", "probe", "test_the_probe_refuses_a_swap_that_touches_a_target_or_adds", PROBE),
    M("P04_stray_input_accepted", "every key forward_infer reads is replaced or kept", "    if stray:", "    if False:", "probe", "test_the_probe_refuses_an_input_the_swap", PROBE),
    M("P05_shape_dtype_not_compared", "the swap keeps every replaced tensor's shape and dtype", "        if pert[k].shape != batch[k].shape or pert[k].dtype != batch[k].dtype:", "        if False:", "probe", "test_the_probe_refuses_a_swap_that_changes_an_unselected", PROBE),
    M("P06_unselected_row_change_accepted", "no unselected row changes in any replaced tensor", "        if bool(ch[k][~sel].any()):", "        if False:", "probe", "test_the_probe_refuses_a_swap_that_changes_an_unselected", PROBE),
    M("P07_record_not_the_windows_own", "a selected row is swapped for ITS window's own matched, uncut state", '    if not (np.array_equal(rec["r"].astype(np.int64), batch["t"][sel_rows].numpy())', '    if False and (np.array_equal(rec["r"].astype(np.int64), batch["t"][sel_rows].numpy())', "probe", "test_the_probe_refuses_a_record", PROBE),
    M("P08_kin3_not_the_banks", "the swapped kin3 IS the bank record's", '    if not bool((pert["kin3"][sel_rows] == torch.from_numpy(rec["kin3"]).to(batch["kin3"].dtype)).all()):', "    if False:", "probe", "test_the_probe_refuses_a_swap_that_changes_nothing", PROBE),
    M("P09_no_identical_exemption", "a selected row whose own ego is bit-identical to the logged one is exempted and counted", '    identical = sel & ~torch.stack([ch[k] for k in ("p10f", "base", "kin3", "p10_t")]).any(0)', "    identical = torch.zeros_like(sel)", "probe", "test_the_probe_exempts_exactly", PROBE),
    M("P10_all_identical_accepted", "at least one non-identical selected row must exist", "    if not bool(need.any()):", "    if False:", "probe", "test_the_probe_exempts_exactly", PROBE),
    M("P11_no_base_at_rest_exemption", "a base that cannot move at rest is exempted and counted", "    base_rest = (need & (batch", "    base_rest = (need & torch.zeros_like(need) & (batch", "probe", "test_the_probe_exempts_a_base", PROBE),
    M("P12_changed_rows_not_required", "every selected row changes in p10f, base, kin3 and p10_t", "        if not bool(ch[k][req].all()):", "        if False:", "probe", "test_the_probe_refuses_a_swap_that_leaves_one_tensor", PROBE),
    M("P13_p10f_not_in_loop", "... p10f", 'for k in ("p10f", "base", "kin3", "p10_t"):\n        req', 'for k in ("base", "kin3", "p10_t"):\n        req', "probe", "test_the_probe_refuses_a_swap_that_leaves_one_tensor", PROBE),
    M("P14_base_not_in_loop", "... base", 'for k in ("p10f", "base", "kin3", "p10_t"):\n        req', 'for k in ("p10f", "kin3", "p10_t"):\n        req', "probe", "test_the_probe_refuses_a_swap_that_leaves_one_tensor", PROBE),
    M("P15_p10_t_not_in_loop", "... p10_t", 'for k in ("p10f", "base", "kin3", "p10_t"):\n        req', 'for k in ("p10f", "base", "kin3"):\n        req', "probe", "test_the_probe_refuses_a_swap_that_leaves_one_tensor", PROBE),
    M("P16_kept_objects_not_checked", "every kept key is the SAME object", "    if kept_moved:", "    if False:", "probe", "test_the_probe_refuses_a_swap_that_touches_a_target_or_adds", PROBE),
    M("P17_lab_s_more_than_rc", "lab_s differs in rc only", '    if set(ls0) != set(ls1) or [k for k in ls0 if k != "rc" and ls1[k] is not ls0[k]]:', "    if False:", "probe", "test_the_probe_refuses_labels", PROBE),
    M("P18_rc_unselected_change", "lab_s.rc is untouched on unselected rows", '    if bool(changed(ls1["rc"], ls0["rc"])[~sel].any()):', "    if False:", "probe", "test_the_probe_refuses_labels", PROBE),
    M("P19_rc_frame_change_not_checked", "lab_s.rc equals its frame change to 1e-5", '    if not res["rc_max_abs_err"] <= 1e-5:', "    if False:", "probe", "test_the_probe_refuses_labels", PROBE),
    M("P20_no_g_live_on_swapped_batch", "live_check on the swapped batch", "    lp = live_check(m, pert, cfg)", '    lp = {"pass": True, "terms": {}, "control_prior_grad_sum": 0.0, "groups_grad": {}}', "probe", "test_the_probe_runs_g_live", PROBE),
    M("P21_determinism_not_required", "control: the same loss twice differs by exactly 0.0", '    if res["control_repeat_loss_diff"] != 0.0 or res["control_repeat_grad_diff"] != 0.0:', "    if False:", "probe", "test_the_probe_refuses_when_the_loss_is_not_reproducible", PROBE),
    M("P22_p0_identity_not_required", "control: p = 0 returns the same object, loss and gradient differ by exactly 0.0", '    if not res["control_p0_same_object"] or res["control_p0_loss_diff"] != 0.0 or res["control_p0_grad_diff"] != 0.0:', "    if False:", "probe", "test_the_probe_refuses_a_p_zero", PROBE),
    M("P23_loss_move_not_required", "the swap moves the loss and the gradient", '    if not (res["loss_diff"] != 0.0 and res["grad_diff_all"] > 0.0):', "    if False:", "probe", "test_the_probe_refuses_when_the_loss_never_reads", PROBE),
    M("P24_p10f_alone_not_required", "replacing p10f alone moves the loss", '    if not only["p10f"] != 0.0:', "    if False:", "probe", "test_the_probe_refuses_when_p10f_alone", PROBE),
    M("P25_prior_control_not_required", "control: the frozen P10 prior's gradient difference is exactly 0.0", '    if 0.0 != res["control_prior_grad_diff"]:', "    if False:", "probe", "test_the_probe_refuses_a_prior_gradient", PROBE),
    M("P26_wm_control_not_required", "control: the world-model auxiliary's gradient difference is exactly 0.0", '    if res["control_wm_grad_diff"] not in (0.0, None):', "    if False:", "probe", "test_the_probe_refuses_a_prior_gradient", PROBE),
    M("P27_p0_control_is_p1", "control: the p = 0 batch is dag.apply(p = 0)", "    b_zero = dag.apply(batch, dag.PROBE_STEP, p=0.0)", "    b_zero = dag.apply(batch, dag.PROBE_STEP, p=1.0)", "probe", "test_the_probe_passes", PROBE),
    M("P28_probe_not_at_p1", "the probe swaps at p = 1 (every eligible row)", "    pert = dag.apply(batch, dag.PROBE_STEP, p=1.0)", "    pert = dag.apply(batch, dag.PROBE_STEP, p=0.5)", "probe", "test_the_probe_passes", PROBE),
    # ---- the caller: main()
    M("W01_apply_line_deleted", "the loop applies the swap (b = dag.apply(b, step))", "            b = dag.apply(b, step)  ", "            b = b  ", "e2e", "test_a_real_run_gives or test_the_check_that_the_loss"),
    M("W02_swap_never_built", "the swap main() builds is make_dagger(...)", "    dag = make_dagger(a, J, wins, clips, info)", "    dag = DG.DaggerSwap()", "e2e", "test_flags_to_config or test_p_one"),
    M("W03_gate_not_given_the_swap", "declared_vs_built reads the BUILT swap", "bad = declared_vs_built(a, cfg, model, opt, info, enc, ego, lr_at, dag)", "bad = declared_vs_built(a, cfg, model, opt, info, enc, ego, lr_at)", "e2e", "test_a_refused_declaration or test_flags_to_config"),
    M("W04_probe_not_run", "the in-process gate runs dagger_probe", "dagp = dagger_probe(model, probe, cfg, dag)", 'dagp = {"pass": True}', "e2e", "test_the_gate_refuses_a_swap_whose_probe_fails"),
    M("W05_flat_levers_not_recorded", "levers.dagger_p / levers.dagger_bank_sha256 are recorded", 'levers["dagger_p"], levers["dagger_bank_sha256"] = dag.p, dag.bank_sha256', "pass", "e2e", "test_a_default_run or test_flags_to_config"),
    M("W06_levers_dagger_not_recorded", "levers.dagger (the built lever dict) is recorded", 'levers["dagger"] = dag.built_levers()', "pass", "e2e", "test_a_default_run or test_flags_to_config"),
    M("W07_summary_dagger_n_is_zero", "summary.json carries dagger_n", '"dagger_n": dag_tot["n"],', '"dagger_n": 0,', "e2e", "test_flags_to_config or test_p_one"),
    M("W08_selected_not_counted", "the run total dagger_n counts the swapped rows", '                c_["n"] += dag.last["n_selected"]', '                c_["n"] += 0', "e2e", "test_flags_to_config or test_p_one"),
    M("W09_eligible_counts_selected", "the run total dagger_eligible counts the eligible rows", '                c_["eligible"] += dag.last["n_eligible"]', '                c_["eligible"] += dag.last["n_selected"]', "e2e", "test_a_real_run_gives or test_flags_to_config"),
    M("W10_rows_not_counted", "the run total dagger_rows counts the samples seen", '                c_["rows"] += dag.last["n_rows"]', '                c_["rows"] += 0', "e2e", "test_a_default_run or test_flags_to_config"),
    M("W11_resume_forgets_totals", "a resumed run keeps counting", '        dag_tot.update(ck.get("dagger_totals") or {})', "        pass", "e2e", "test_a_resumed_run"),
    M("W12_checkpoint_drops_totals", "the checkpoint carries the totals", '"dagger_totals": dict(dag_tot)}', '"dagger_totals": {}}', "e2e", "test_a_resumed_run or test_flags_to_config"),
    M("W13_interval_counts_dropped", "the log carries per-interval counts when the lever is on", "                if dag.active:\n                    rec[", "                if False:\n                    rec[", "e2e", "test_the_log_interval"),
    M("W14_log_keys_always", "a default run's log is the pre-change log (no dagger keys)", "                if dag.active:\n                    rec[", "                if True:\n                    rec[", "e2e", "test_a_default_run"),
    M("W15_interval_not_reset", "the interval counters restart at every log line", "                dag_int.update(rows=0, eligible=0, n=0, nav_kept=0)", "                pass", "e2e", "test_the_log_interval"),
    M("I01_default_log_keys_differ", "the default run IS the pre-change run, key for key (no dagger keys in the log at p = 0)", "                if dag.active:\n                    rec[", "                if True:\n                    rec[", "ident", "test_defaults_reproduce"),
    M("I02_default_summary_loses_a_pre_change_key", "the default run IS the pre-change run: the summary keeps every pre-change key", '"ego_perturb_nav_kept": ego_tot["nav_kept"],\n               "dagger_n"', '"ego_perturb_rows": ego_tot["rows"],\n               "dagger_n"', "ident", "test_defaults_reproduce"),
    # ---- the FOURTH flag, --dagger-bank-sha256 (PREREG_V8F_DAGGER section 4; integration build di): one mutant per rule
    M("S01_sha_default_is_not_off", "--dagger-bank-sha256 defaults to none (off = every earlier arm)", 'dest="dagger_bank_sha256", default="",', 'dest="dagger_bank_sha256", default="0" * 64,', "flags", "test_the_four_flags"),
    M("S02_sha_has_no_gdvb_entry", "every flag has a FLAG_CHECKS entry (dagger_bank_sha256)", '"dagger_bank_sha256": "dagger:dagger_bank_sha256", ', "", "flags", "test_the_four_flags or test_a_flag_without_an_entry"),
    M("S03_sha_not_among_the_no_swap_defaults", "a declared --dagger-bank-sha256 with no swap built is refused", '("dagger_p", "dagger_bank", "dagger_bank_sha256", "dagger_seed")}', '("dagger_p", "dagger_bank", "dagger_seed")}', "flags", "test_a_declared_flag_with_no_swap", BDAG),
    M("S04_sha_not_required_at_p_above_zero", "--dagger-bank-sha256 is REQUIRED when --dagger-p > 0 (DAG names all four)", "    if a.dagger_p > 0 and not a.dagger_bank_sha256:", "    if False:", "flags", "test_g_dvb_refuses_an_illegal or test_required_at_p_above_zero", BDAG),
    M("S05_sha_not_refused_at_p_zero", "--dagger-bank-sha256 is REFUSED when --dagger-p is 0 (declared and does nothing; DAG-0 names only --dagger-p 0)", "    if a.dagger_p <= 0 and a.dagger_bank_sha256:", "    if False:", "flags", "test_g_dvb_refuses_an_illegal or test_required_at_p_above_zero", BDAG),
    M("S06_sha_without_a_bank_not_refused", "a declared --dagger-bank-sha256 with no --dagger-bank at p > 0 is refused (nothing to hash)", "        if a.dagger_bank_sha256 and a.dagger_p > 0:", "        if False:", "flags", "test_g_dvb_refuses_an_illegal", BDAG),
    M("S07_gate_does_not_compare_declared_to_disk", "the gate re-hashes bank.json and refuses a declared hash that differs", "    if a.dagger_bank_sha256 and a.dagger_bank_sha256.lower() != disk:", "    if False:", "flags", "test_the_gate_rehashes", BDAG),
    M("S08_gate_compares_case_sensitively", "the declared hash is compared as a hash (hex is case-blind)", "a.dagger_bank_sha256.lower() != disk", "a.dagger_bank_sha256 != disk", "flags", "test_a_matching_declared_sha256", BDAG),
    M("S09_shape_check_dropped", "a declared value that is not a full sha256 is refused with its own message", "    if a.dagger_bank_sha256 and not is_sha256_hex(a.dagger_bank_sha256):", "    if False:", "flags", "test_a_declared_value_that_is_not", BDAG),
    M("S10_shape_accepts_a_wrong_length", "a sha256 is exactly 64 hex digits (a truncated hash identifies nothing)", "len(s) == 64 and ", "len(s) >= 12 and ", "flags", "test_a_declared_value_that_is_not or test_the_sha256_shape"),
    M("S11_shape_accepts_non_hex", "a sha256 is hex", 'all(c in "0123456789abcdefABCDEF" for c in s)', "True", "flags", "test_a_declared_value_that_is_not or test_the_sha256_shape"),
    M("S12_shape_rejects_capitals", "hex is case-blind", '"0123456789abcdefABCDEF"', '"0123456789abcdef"', "flags", "test_the_sha256_shape or test_a_matching_declared_sha256"),
    M("S13_load_ignores_the_declared_sha", "the bank is loaded against the declared hash (a bank with another manifest is refused at load)", "expect_sha256=a.dagger_bank_sha256 or None)", "expect_sha256=None)", "flags", "test_a_declared_sha256_that_is_not_the_banks"),
    # ---- the TRAIN window-list digest on EVERY run (Gap B: the reader's K-B3.2 key `train_window_list_sha256`, beside `train_sha256`)
    M("W01_window_sha_not_in_info", "build_data records the TRAIN window-list digest for every run", '    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)', "    pass", "wsha", "test_every_run_records_the_train_window_list_sha256", BDATA),
    M("W02_window_sha_not_in_summary", "summary.json carries the digest at the level of train_sha256", '               "train_window_list_sha256": info.get("train_window_list_sha256"),\n', "", "wsha", "test_every_run_records_the_train_window_list_sha256"),
    M("W03_window_sha_only_for_dagger_arms", "the digest is recorded on EVERY run (default and DAG-0 too), not only when the lever is on", '    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)', '    if a.dagger_p > 0:\n        info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)', "wsha", "test_every_run_records_the_train_window_list_sha256", BDATA),
    M("W04_window_sha_of_a_shorter_list", "the digest is over the window list the trainer BUILT (all of it)", '    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)', '    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins[1:])', "wsha", "test_every_run_records_the_train_window_list_sha256", BDATA),
    M("W05_window_sha_of_the_clip_list", "the digest is a hash of WINDOWS (clip:t), not of the clip list", '    info["train_window_list_sha256"] = DG.windows_sha256(J["clip"], wins)', '    info["train_window_list_sha256"] = info["train_sha256"]', "wsha", "test_every_run_records_the_train_window_list_sha256", BDATA),
    M("W06_summary_key_renamed", "the key is the reader's name, train_window_list_sha256", '               "train_window_list_sha256": info.get("train_window_list_sha256"),', '               "window_list_sha256": info.get("train_window_list_sha256"),', "wsha", "test_every_run_records_the_train_window_list_sha256 or test_defaults_reproduce"),
    M("S14_declared_sha_not_recorded", "levers.dagger_bank_sha256_declared records the declared hash", 'levers["dagger_bank_sha256_declared"] = a.dagger_bank_sha256 or None', 'levers["dagger_bank_sha256_declared"] = None', "e2e", "test_flags_to_config"),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_copy(root, work):
    if os.path.commonpath([os.path.abspath(root), os.path.abspath(work)]) == os.path.abspath(root):
        raise SystemExit(f"REFUSED (R77): --work {work} is inside the delivery tree {root}")
    shutil.rmtree(work, ignore_errors=True)
    for t in COPY_TREES:
        shutil.copytree(os.path.join(root, t), os.path.join(work, t), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "results", "tests"))
    for f in COPY_FILES + ["stack/scripts/" + n for n in os.listdir(os.path.join(root, "stack/scripts")) if n.endswith(".py")]:
        if os.path.exists(os.path.join(root, f)):
            os.makedirs(os.path.dirname(os.path.join(work, f)), exist_ok=True)
            shutil.copyfile(os.path.join(root, f), os.path.join(work, f))
    os.makedirs(os.path.join(work, "stack/tests"), exist_ok=True)
    for f in ("test_v8f_dagger.py", "test_v8f_mini.py"):                                       # imported by other suites; harmless here
        if os.path.exists(os.path.join(root, "stack/tests", f)):
            shutil.copyfile(os.path.join(root, "stack/tests", f), os.path.join(work, "stack/tests", f))


def region_bounds(text, region):
    if region is None:
        return 0, len(text)
    a = text.index(region[0])
    b = text.index(region[1], a + 1)
    return a, b


def apply(text, m):
    a, b = region_bounds(text, m["region"])
    seg = text[a:b]
    n = seg.count(m["old"])
    if n != 1:
        raise SystemExit(f"ANCHOR for {m['name']} found {n} times (need exactly 1) in {'its region' if m['region'] else 'the file'}: {m['old'][:90]!r}")
    return text[:a] + seg.replace(m["old"], m["new"]) + text[b:]


def run_pytest(py, work, k, x, log, tag):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
    for _ in range(720):                                                                         # one pytest process at a time: wait for a slot (< 2 runs active across the machine)
        r = subprocess.run([py, "C:/Users/Admin/orcread/pytest_runs.py"], capture_output=True, text=True)
        if r.stdout.strip().isdigit() and int(r.stdout.strip()) < 2:
            break
        time.sleep(30)
    os.makedirs(os.path.join(work, "bt"), exist_ok=True)                                        # pytest --basetemp needs its parent to exist
    bt = os.path.join(work, "bt", "cur")                                                        # ONE basetemp, reused: pytest clears it at the start of every run (a basetemp per mutant filled the disk: 640 MB per run on a 1 MiB-cluster exFAT)
    cmd = [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", bt, TEST, "-k", k] + (["-x"] if x else [])
    t = time.time()
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=3 * 3600)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    log(f"    rc={p.returncode} {tail}  [{time.time() - t:.0f}s]")
    return p.returncode, p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("python")
    ap.add_argument("--work", default="E:/Projects/TanitAD-artifacts/dt/mut")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--log", default="")
    a = ap.parse_args(argv)
    muts = [m for m in MUTANTS if (not a.only or m["name"] in a.only.split(",") or any(m["name"].startswith(x) for x in a.only.split(","))) and (not a.group or m["group"] == a.group)]
    if a.list:
        for m in muts:
            print(f"{m['name']:42s} [{m['group']}] {m['rule']}")
        print(len(muts), "mutants")
        return 0
    lf = open(a.log, "a", encoding="utf-8") if a.log else None

    def log(s):
        print(s, flush=True)
        if lf:
            lf.write(s + "\n")
            lf.flush()
    before = {f: sha(os.path.join(a.root, f)) for f in DELIVERED if os.path.exists(os.path.join(a.root, f))}
    for f, h in before.items():
        log(f"BEFORE {h} {f}")
    build_copy(a.root, a.work)
    pristine = open(os.path.join(a.work, TRAIN), encoding="utf-8").read()
    names = [m["name"] for m in MUTANTS]
    assert len(names) == len(set(names)), "duplicate mutant names"
    for m in MUTANTS:                                                                            # every anchor exactly once in its region, every mutated text compiles
        ast.parse(apply(pristine, m))
    log(f"{len(MUTANTS)} mutants, each anchor found exactly once in its region, each mutated text compiles")
    if a.check:
        return 0
    killed, survivors = [], []
    py = a.python
    for g, gk in GROUPS.items():
        gm = [m for m in muts if m["group"] == g]
        if not gm:
            continue
        log(f"== group {g}: baseline (unmutated copy) on  -k \"{gk[:120]}...\"")
        rc, out = run_pytest(py, a.work, gk, False, log, f"base_{g}")
        if rc != 0:
            log(f"BASELINE RED for group {g}: no mutant of this group is judged")
            log(out[-3000:])
            survivors += [m["name"] + " (baseline red)" for m in gm]
            continue
        for m in gm:
            open(os.path.join(a.work, TRAIN), "w", encoding="utf-8", newline="\n").write(apply(pristine, m))
            log(f"  {m['name']}: {m['rule']}")
            try:
                rc, out = run_pytest(py, a.work, m["k"], True, log, f"mut_{m['name']}")
            finally:
                open(os.path.join(a.work, TRAIN), "w", encoding="utf-8", newline="\n").write(pristine)
            if rc == 1 and any(x in out for x in DISK_FULL):                                         # a test that died because the DISK was full proves nothing about the rule
                survivors.append(m["name"])
                log(f"    SURVIVOR {m['name']}  (rc=1 but the output shows a full disk: INCONCLUSIVE, not a test failure)")
            elif rc == 1 and (" failed" in out or " error" in out):                                  # pytest's own "tests failed" exit; a collection error (2) or "no tests ran" (5) is NOT a kill
                killed.append(m["name"])
                log(f"    KILLED {m['name']}")
            else:
                survivors.append(m["name"])
                log(f"    SURVIVOR {m['name']}  (rc={rc}: {'INCONCLUSIVE, not a test failure' if rc in (2, 5) else 'the suite stayed green'})")
    assert open(os.path.join(a.work, TRAIN), encoding="utf-8").read() == pristine
    after = {f: sha(os.path.join(a.root, f)) for f in before}
    changed = [f for f in before if before[f] != after[f]]
    log(f"delivered files changed during the run: {changed or 'none'}")
    log(f"{len(killed)} of {len(killed) + len(survivors)} mutants killed" + (f"; SURVIVORS: {survivors}" if survivors else ""))
    return 0 if not survivors and not changed else 1


if __name__ == "__main__":
    sys.exit(main())
