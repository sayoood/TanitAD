import hashlib, json, os, re, glob
ROOT = "F:/TanitAD-scratch/rollhist/build"; B1 = ROOT + "/b1"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def rd(p, d=None):
    try: return json.load(open(p, encoding="utf-8"))
    except Exception: return d
mut = rd(B1 + "/mutation_table_rollhist.json", [])
reg = rd(B1 + "/regression_rollhist.json", [])
final = {}
for f in ("test_v8f_t1_readout", "test_rollhist_pair_check", "test_v8f_rollhist"):
    p = f"{B1}/logs/regression_new_{f}.log"
    t = open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""
    m = re.findall(r"(\d+ passed[^\n]*|\d+ failed[^\n]*)", t)
    final[f] = m[-1] if m else "NO RESULT"
rep = {
  "task": "PREREG_V8F_ROLLHIST build B1 (N1-N4, chain checker, R4/R5, R1-R3 on CPU)",
  "tip": "7b26a8aca32e8a1b9169271418305f43af7272f8",
  "blobs_at_tip_equal_fb686f4a": {"v8f_mini_train.py": "591142ad", "v8f_dagger.py": "482d7e2a", "v8f_dagger_bank.py": "78c87c0f", "same_as_fb686f4a": True},
  "export_verified": "git hash-object of the 3 exported files equals their tip blobs",
  "final_test_results_new_files": final,
  "controls": {
    "N1_promoted_vs_study_bit_identical": "2880 arrays compared, bit-identical True (b1/logs/study_vs_promoted.log); golden sha256 1c42d85fdbe040c9e10c3967c4df34fa405fe39a0ad45c91a5f5a9cf0cd4a8e9",
    "N1_gain1_equals_causal_gain0_equals_causal_z": "array_equal on 40 plans x path_exec{F,T} (test_gain_one_is_causal_and_gain_zero_is_causal_z_array_for_array); forced general formula within 1e-12",
    "R4_pinned_none_vs_todays_generator": "max abs difference 0.0, content_sha256 equal, bank.json sha equal default vs named (test_r4_control_pinned_none...)",
    "R4_nonvacuous": "RH arrays differ from pinned by > 1e-6; the pinned bank reads a3 deviation > 1e-3 on the same check",
    "R4_rh_a3": "stored last-knot acceleration minus 1.0 x executed <= 1e-6 on every finite record; per-call heading step <= 0.2 x distance + 1e-12 on every frame of every call (>50 calls); positions equal backward readout",
    "R5": "6 mismatch cases (both flags, both directions) refused with 'loaded bank header'; p=0 declarations refused with 'does nothing'; legacy header (no generator) reads pinned/none; control cases return []",
    "R1": "dagger_probe PASS on the RH bank; control_p0_loss_diff 0.0, control_p0_grad_diff 0.0, control_prior_grad_diff 0.0, control_wm_grad_diff 0.0",
    "R2_R3": "S (ckpt term, read_s/head_s) and O (ade term, read_o/head_o) gradient difference swapped-vs-logged > 0; same at p = 0 EXACTLY 0.0",
  },
  "hops_flag_to_gdvb": rd(B1 + "/hops.json", {}),
  "mutants": [{k: m.get(k) for k in ("mutant", "went_red", "red_tests", "restored_sha_ok", "pattern_found")} for m in mut],
  "regression": reg,
  "cpu_only_limits": "The fixture is the trainer's synthetic world and a tiny random model: it proves the executor is installed, bounded and read back, not that the real checkpoint's rollout on real data holds the bound with the coverage the pod bank needs; see HANDOFF_B1.md pod pilot.",
  "card_vs_brief_differences_for_MM": [
    "G-PAIR: --dagger-seed is allowed to differ or not between B0 and RH (the card counts it among 'the four DAgger flags'); every other differing flag is a problem.",
    "The checker is a new stack script (stack/scripts/rollhist_pair_check.py); the study checkers block3_prep.sh / chk_arm3.sh / dscale10k_check.py were not edited. Plug-in: call pair_problems(rh_argv, b0_argv, bank_sha256) beside dscale10k_check.dagger_problems and mint only on [].",
  ],
}

rep["fix_round"] = {
  "date": "2026-11-20",
  "fix1_generator_record_only_when_non_default": {
    "change": "stack/scripts/v8f_dagger_bank.py: is_default_generator(gen); the record (header npz, shard plan + provenance, bank.json common, PROVENANCE.json) is written ONLY for readout != pinned or kappa != none",
    "proof": "test_n2_a_default_bank_is_byte_identical_to_todays_generator (vs a null-executor run) and test_n2_a_default_bank_is_byte_identical_to_the_unpatched_generators_files (runs the UNPATCHED tip export's generator in a subprocess; ROLLHIST_ORIG_STACK): shard npz and bank.json byte-equal, same file list, provenance equal after the clock/argv/tool-sha/seconds/process-counter fields. bank sha256 d424f9e8... on the fixture in both",
    "built_levers": "an absent record still reads pinned / none (test_n3, test_r5_a_header_without_the_field)",
    "M5": "now 'a NON-default bank carries no generator record': RED (4 tests)",
  },
  "fix2_declaration_names_the_whole_readout": {
    "function": "tanitad/data/v8f_dagger.py: readout_token(readout, acc_gain, path_exec) (+ parse_readout_token for canonical spelling); pinned -> pinned; causal_g -> causal_g{gain:g} + x for path-exec (causal_g1x, causal_g0.5)",
    "used_by": "built_levers (from the loaded header), the trainer flag --dagger-bank-readout (type=parse; the bare 'causal_g' and non-canonical 'causal_g1.0x' are refused), bad_dagger (declared vs built on the token), rollhist_pair_check.py (RH must name causal_g1x and kappa 0.2)",
    "tests": "gain 1 / path-exec off bank with causal_g1x declared: REFUSED; gain 0.5 / path-exec on with causal_g1x declared: REFUSED; the matching token accepted (test_r5_the_declaration_names_the_whole_readout...); token spelling test; pair-check token tests",
    "new_mutant": "M14_compare_readout_base_name_only: RED (1 test)",
  },
  "regression_finding": "test_v8f_orc_chain 13 passed / 32 skipped and test_v8f_ego_perturb 80 passed / 2 skipped on the patched AND the unpatched tree, with F: above 4 GB: B1's 13 errors and 3 failures did not reproduce (consistent with the ENOSPC cut, not code). Pre-existing identical failures on both trees: test_v8f_oacc (1), test_v8f_flat (1), test_v8f_cz (1), test_v8f_kscore (4), test_v8f_kindx_s1 (149, whole file on both trees), test_v8f_lora (14).",
}
json.dump(rep, open(B1 + "/REPORT_B1.json", "w", encoding="utf-8"), indent=1)
lines = []
for f in sorted(glob.glob(B1 + "/*")) + sorted(glob.glob(B1 + "/logs/*")):
    if os.path.isfile(f) and not f.endswith("MANIFEST_SHA256.txt"):
        lines.append(f"{sha(f)}  {f.replace(ROOT + '/', '')}")
open(B1 + "/MANIFEST_SHA256.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("report written", final)
