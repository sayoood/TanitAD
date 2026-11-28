# IMPL PLAN SAFEOBJ: executable by a Sonnet agent (0 GPU; about 1 day, INHERITED from D-REFE-SAFEHEAD-1)

**Scope.** Build `--epo-con` exactly as `SPEC_SAFEOBJ.md` says, plus the profile `refe_rls1d`, the chain wiring and the reader.

**Rules:**
- No launch and no training.
- Stage, never push, unless the brief says otherwise.
- End with a deliverable manifest.
- PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Work in a SHORT path (MAX_PATH), e.g. `C:/sobj`.
- Run every test named below in its **FULL** form: no `--static`, no `--no-e2e`, no `--quick`.
- A ZZTEST_SKIP is NOT a pass.
- Mutation-first: write each RED mutation BEFORE the code it guards, and see it fail.

## Steps (priority order; bank incrementally)

### 1. Freeze the base

Record these blobs at the tip, so step 2 can prove the reversal:

| file | blob |
|---|---|
| `refe/scorer_finetune.py` | `7c8071ea…` |
| `refe/launch_gate.py` | `e6e000ca…` |
| `refe/gate_checks.py` | `14e32f1b…` |
| `refe/rls1_checks.py` | `f28a106e…` |
| `refe/rls1_pod.sh` | `d1ba34a6…` |
| `refe/rl_reward.py` | `cd5c5144…` (must NOT change) |

### 2. Write `refe/test_rl_safeobj.py` first

Copy the skeleton of `refe/test_rl_r0c.py`: `check`, `bits`, the mutation table, `apply_edits`.
- Define `SAFEOBJ_FILE_REVERSALS` as (new, old) pairs. Each "new" occurs exactly once.
- Reversing the pairs must give blob `7c8071ea…` (I5).
- Implement I1-I5, C1 and M1-M9 (SPEC §6) on a tiny CPU scorer, the way test_rl_epo builds one.
- Print `ZZTEST_PASS <n>  (1 unmutated suite + k RED mutations)`.

### 3. Edit the trainer (`refe/scorer_finetune.py`)

Purely additive, except one line inside the pinned `train_losses`.

**3a. New module-level code** (unpinned; after `epo_reference_built`, :206-216):
- `EPO_CON_MODES = ("off", "g5")` and `EPO_CON_DEFAULT = "off"`.
- `epo_con_cols()`: reads `rl_reward.COMPONENTS.index("ttc")` and `.index("nc")`.
- `epo_con_term(sxB, sxA_det, tg, rule_fn, T, eps, lam: dict, parts)`, implementing SPEC §1.2:
  - float32;
  - reuse `epo_policy`;
  - the `REFE_RL_MUTATE_CON_DETACH` and `REFE_RL_MUTATE_CON_ATTACH_A` hooks.
- `class EpoDual`, with:
  - `active`
  - `lam`
  - `step(acc)`, implementing SPEC §1.4
  - `state_dict()` / `load_state_dict()`
  - `built_facts()`: mode from comps, cols, eps, lr, max, lam0, and ref read by sentinel.
- `epo_con_refusals(a, ap)`, implementing SPEC §2.
- `g13_epo_con_live_and_identity(armA, armB, probe, kl_tol)`, mirroring `g11_epo_live_and_identity` (:478-498) and implementing SPEC §5.2.

**3b. argparse.** Add the five flags after `--epo-ref` (:575-576). Update the module docstring's flag list (:25-34).

**3c. main().**
- Call `epo_con_refusals` beside :591.
- Build `dual` after :682-690.
- Add `info["epo_con"] = dual.built_facts()` for every `--epo` run (:704-707).

**3d. The one edit inside the pinned segment `main.train_losses` (:792-800).**
- After `lB = lB + ep` (:799), add the two-line `if dual.active:` block.
- Register it in SAFEOBJ_FILE_REVERSALS.

**3e. The loop.**
- `con_acc` beside `epo_acc` (:1371, :1385-1386).
- `dual.step` between :1395 and :1402.
- The `con_*` row fields (:1404-1416).
- The `EPO_CON` block in `scorer_states_last.pt` (:1420-1423) and in `save_full` meta (:981-1001).

**3f. Preflight.** G13 after G11 (:1292-1297). The speed probe (:1342) must include the term.

### 4. Update the pin chain

This is the same pattern R0c used (`test_rl_s1b.py:57-58`, `test_rl_r0c.py:94-120`):
- **`test_rl_r0c.py`**: apply `test_rl_safeobj.reverse_edits` BEFORE its own `reverse_edits`. The R0c → `d34d5cf6` proof must still hold.
- **`test_rl_s1b.py`**: the P-pins hash `undo_r0c(undo_safeobj(segment))`.
- **Every other pinned segment** must stay byte-identical. If one changes, STOP and report.

### 5. Gate (`refe/gate_checks.py`, `refe/launch_gate.py`), per SPEC §5

**gate_checks.py:**
- `DVB_KINDS_SCORER` gains the five flags.
- New `SCORER_EPO_CON_KEYS`.
- `scorer_dvb_rows` gains the `epo_con` block.
- `SCORER_GATES["G13"]`.
- `scorer_required_gates`.
- `judge_g13`.

**launch_gate.py:**
- `RLS1D_*` constants.
- `PROFILES["refe_rls1d"]`.
- `rls1_refusals` reads per-arm `arm_epo_registered` and `arm_must_name`, and calls `mod.epo_con_refusals`.
- `refe_rls1`, `refe_rls1b` and `refe_rls1c` refuse any `--epo-con` other than off.
- `consume_scorer_preflight` re-judges G13.
- Update `--arm-name` help (:988) and `_profile_for` (:876).

**Tests:** extend `refe/test_launch_gate.py` with accept/refuse rows for R0d, R1d and R1o (omitted flag, default value, wrong seed, `--epo-con g5` on R1o, `--epo-con` on R0c), plus the M7 / M8 arms.

### 6. Arms file (`eval/rls1d_arms.json`)

- Derived from R0's banked token argv, exactly as `eval/rls1c_arms.json` was.
- `self_check`: BASE with R0 and seed 0 hashes to `24451b62…`.
- `order`: `["R0d", "R1d", "R1o"]`.
- `budget`: 10.5.

### 7. Chain (`refe/rls1_pod.sh`, `refe/rls1_checks.py`)

**rls1_pod.sh:**
- `REGISTERED_ARMS` += `R0d R1d R1o`.
- `arm_profile` maps these arms to `refe_rls1d` (:87-89).
- `arm_args` epo strings (:194-197).
- `REFDIR="$ARMDIR/scores"` for all three: own-A, as for R0c (:493-497).
- C0 is the second control.

**rls1_checks.py:**
- `monitor --observe-only` for R1o: it logs `ZZRLS1_K3_OBSERVED` and never writes `KILLED.json`. K2 and K6 untouched.
- R1o's step refuses unless `R0d/safeobj_S.json` and `R1d/safeobj_S.json` both read `"S": "PASS"` (step 8).

**Tests:** extend `test_rls1_chain.py` and `test_rls1_queue.py`, including M9 and the kill-stops-chain rule for R0d and R1d.

### 8. Reader (`eval/safeobj_read.py`)

**Input:** an arm dir's dumps plus C0's.

**Reads, through `rl_val_eval.run`** (paired log-cluster bootstrap, 10,000, seed 20260927):
- bars S and G (PREREG §5);
- the four families (PREREG §8);
- the λ trajectory and the O4/O5/O6 discriminants from `sft.jsonl`.

**Writes:**
- `safeobj_S.json`, with S PASS/FAIL per check;
- `safeobj_read.json`, with outcome O1-O7 by the committed rules, applied mechanically.

**Selftest:** `eval/test_safeobj_read.py`. A synthetic arm equal to its A must read S PASS, G FAIL and O3. A synthetic arm at dose.json's alpha 0.25 must sit exactly on G's boundary.

### 9. Families work item

Verify that the navtest RR3 family reader used for `RESULT_P2REP.md` (`raw/2026-10-16-p2s1-read/p2rep_read.py`) accepts RL-S1 B and A checkpoints.
- If it does not, extend it or write the adapter.
- LATERAL at navtest must exist before Stage 3. A missing family is a work item, not an omission.

### 10. Run the suites in FULL form and report every token

**Tests that name the edited files:**

| edited file | tests |
|---|---|
| `scorer_finetune.py` | `test_aux_g12_bf16`, `test_aux_s1_wiring`, `test_dump_scores`, `test_label_dir_refusal`, `test_launch_gate` (default mode, i.e. FULL: `ZZLG_FULL_PASS`), `test_rl_epo`, `test_rl_r0c`, `test_rl_s1b`, `test_rl_terms_label`, `test_rls1_chain`, `test_rls1_queue`, `test_s1aux_chain`, `test_s1c` (no `--no-e2e`), `eval/test_aux_navtest_forward` |
| `launch_gate.py` / `gate_checks.py` | also `test_gate_gpu_probes`, `test_gate_suite_prerun`, `test_p2r_chain`, `test_p2rep_chain`, `test_p6_chain` (`ZZP6CHAIN_PASS`), `test_p6_levers`, `test_p6_cpu_probe`, `test_p6_overlap` |
| `rls1_checks.py` / `rls1_pod.sh` | also `test_flock_rw`, `test_rl_tlane_label` |
| new | `test_rl_safeobj`, `eval/test_safeobj_read`, `eval/test_rl_val_eval` |

**Expected tokens:**
- `test_aux_g12_bf16` and the CUDA check of `test_gate_gpu_probes` print a LOUD skip without CUDA. Run them on a CUDA box, or report them as NOT RUN.
- The bash chain tests need Git Bash (and perl for the queue test).

**Baseline:** record the pre-edit counts first, so a regression is a number, not an impression:
- `test_rl_r0c` 26
- `test_rl_s1b` 18
- `test_rl_epo` 42

These are INHERITED from `PREREG_R0C.md`; re-measure them on the tip before editing.

### 11. Pre-run the G-SUITE on the pod for `refe_rls1d`

Pre-run it after shipping the tree to `/workspace/refe-plan-rls1` (the 2026-10-16 pod-prerun pattern). Then run `launch_gate.py run --profile refe_rls1d --arm-name R0d` to mint the token.

⛔ **Do not launch.** The launch is the Master Mind's call, after the PI's budget decision for Stage 2.

### 12. Hand back

≤ 300 words plus a manifest: every file, its blob, every test token, and every RED mutation.
