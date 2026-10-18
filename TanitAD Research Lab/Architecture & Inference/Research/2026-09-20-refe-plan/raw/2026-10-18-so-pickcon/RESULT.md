# SO-PICKCON RESULT -- the constraint's own temperature (`--epo-con-temp`) and the near-tie-robust A2 read

Date 2026-10-18. Base = mirror tip `0a5958a`. CPU only; the pod was never touched. Nothing committed or pushed; nothing in the registers edited. Everything is staged under `C:/Users/Admin/pickcon/out/` (mirrors repo paths; `paths.json`; base blobs in `BASE_BLOBS.json`).

**Evidence class and tier.** Deliverable 1 is a WIRING result: MEASURED on the repo's own CPU test rig (a tiny synthetic scorer and the REAL `main()` with CPU stubs for the GPU parts), not a driving result. Deliverable 2 is MEASURED on the banked R0 states (`scores_update{250,500,1000}_{A,B}.npz`, 3,000 sets / 75 logs) at the **offline selection tier**, NOT T1; the estimator is the paired log-cluster bootstrap (10,000, seed 20260927), which answers "another draw of LOGS" only -- not another training run, not another inference run (one R0 run per state, no replicate). Longitudinal (TTC) family only; lateral and tactical have no instrument at this tier; **STRATEGIC is PAUSED (PI ruling 2026-09-19, ITEM 25) -- reported as paused, never as a pass.**

---------------------------------------------------------------------------------------------------

## DELIVERABLE 1 -- `--epo-con-temp`

### The flag
`--epo-con-temp <float>`, separate from `--epo-temp` (the selection / EPO reward policy T). **Default `None` = INHERIT `--epo-temp`** (registered 0.1), declared as the named constant `EPO_CON_TEMP_DEFAULT = None` (patched `refe/scorer_finetune.py:234`, parser `:837`).
* **The T the constraint used before** (`0a5958a:refe/scorer_finetune.py:1049`): `ct = epo_con_term(sxB, sxA, tg, sel_score, a.epo_temp, a.epo_con_eps, dual.lam, con_last)` -- the SAME `a.epo_temp` that the EPO term used one function above (`:1038`), `--epo-temp` default 0.1 (`:800`). So the one number SO-CARVE says must change (0.1 -> ~0.02) could not move without moving the EPO reward term too.
* **Bit-for-bit default:** with the flag absent the dual is built with `temp = a.epo_temp` (patched `:954`), so every arm before PICKCON (R0d, R1d, R1o, whatever `--epo-temp` it set) gets exactly the value it had. The test proves it with a default-vs-explicit-0.1 identity at EXACTLY 0.0 and an inheritance check at `--epo-temp 0.05`.
* ⚠️ **A DEFAULT THAT SILENTLY DOES NOTHING MUST BE NAMED.** Because the default reproduces R0d, an arm that means to test T = 0.02 and does NOT carry `--epo-con-temp 0.02` runs R0d and calls it the new arm -- the "default carries the experiment" trap. Documented in three places: the trainer's module docstring (`NOTE: that default is a SILENT reproduction of R0d`), the flag's `--help`, and the gate -- `arm_must_name` for the arm refuses an argv that omits the flag and the refusal says both the registered value and what the parser default would have given (`G10`, `G_future_arm_fixture_R0p_with_the_flag_OMITTED_is_refused_NAMED_and_registered_value_and_the_parser_DEFAULT`). **The arm's argv must name `--epo-con-temp 0.02` (or whatever T the PI picks) explicitly.**
* Refusals added to `epo_con_refusals`: a non-finite or non-positive T is refused; naming `--epo-con-temp` while `--epo-con off` (or without `--epo`) is refused, because the term is OFF and the flag would be a no-op carried in an argv.

### The hop trace (argparse -> namespace -> constraint term -> loss -> `.backward()`; patched trainer line numbers, printed by `test_rl_pickcon.py` from the live source)
| hop | what | where |
|---|---|---|
| 1 | argparse declares `--epo-con-temp` (float, default `None`) | `refe/scorer_finetune.py:837` |
| 2 | namespace -> the dual: `EpoDual(..., temp=(a.epo_temp if a.epo_con_temp is None else a.epo_con_temp))` | `:954` |
| 3 | `EpoDual.temp` holds it and `built_facts()` publishes it in the data event's `epo_con.temp` | `:303` |
| 4 | `train_losses` passes `dual.temp` to the constraint term (no longer `a.epo_temp`) | `:1068` |
| 5a/5b | `epo_con_term` runs the policy of arm B and of arm A at that T (no literal 0.1 in the function) | `:274` / `:275` |
| 6 | `epo_policy` divides the log score by T | `:201` |
| 7 | `ct` joins arm B's loss (`lB = lB + ct`) | `:1069` |
| 8 | the loop calls `((lA + lB) / a.accum).backward()` | `:1678` |
| 9 | G13 receives the dual's T (`temp=dual.temp`) and re-runs the loss assembly at 2T | `:1593` |

### Reachability tests (all in `refe/test_rl_pickcon.py`, `ZZTEST_PASS 20` = 1 unmutated suite + 19 RED mutations)
* **(a) The flag moves the constraint's gradient.** The constraint's OWN gradient on the scorer parameters (the run's `con_probe` construction: the real `train_losses` with the dual ON minus the same assembly with it OFF) differs between 0.1 and 0.02, arm A's gradient does not, and the logged `cB_ttc` differs. Twin: the same T twice reads EXACTLY 0.0 (the probe is not vacuous). Magnitudes: see the MEASURE block below.
* **(b) Identity control.** Default vs explicit `--epo-con-temp 0.1`: loss bits, arm B gradient, constraint gradient, arm A gradient, the logged parts all **identical, max |diff| = exactly 0.0**. Inheritance: with `--epo-temp 0.05` the default resolves to 0.05, equals an explicit 0.05 at 0.0 and differs from an explicit 0.1 (so the default is not a hidden constant 0.1). Zero control: lambda = 0 reads EXACTLY 0.0 in value and gradient.
* **(c) T -> 0 is the hard pick, exactly.** At T = 1e-6 and 1e-8 the policy equals the one-hot of the pick, and `cA_ttc`, `cB_ttc`, `g_ttc` equal the hard-pick TTC means / gap to the last bit (labels on a dyadic grid so float32 sums are exact; the first seed whose two arms' picks carry different TTC labels is used, so a g of exactly 0 cannot make the equality vacuous). Twin: at T = 0.1 the quantity is NOT the pick (the wedge SO-CARVE measured).
* **(d) Mutation: hard-code the old T.** In a temp copy of the source -- (M1) `train_losses` passes `a.epo_temp` again; (M2) the old 0.1 hard-coded inside `epo_con_term` -- **check (a) goes RED** (and the inheritance check, and the static hop checks R4/R5). Full RED record (the actual test output):

```
M1 (d) the OLD T hard-coded at the call: train_losses passes a.epo_temp again            U  R_hop_4_train_losses_passes_dual_temp_as_the_T_of_epo_con_term_and_not_a_epo_temp, U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not, U_b_the_default_INHERITS_epo_temp_for_any_value_of_it_it_is_not_a_constant_0.1
M2 (d) the OLD T hard-coded INSIDE the term (policy of B and of A at a literal 0.1)      U  R_hop_5_epo_con_term_runs_BOTH_policies_at_its_T_parameter_with_no_literal_0.1_in_it, U_a_the_constraint_s_gradient_on_the_scorer_params_DIFFERS_between_0.1_and_0.02_and_arm_A_does_not, U_b_the_default_INHERITS_epo_temp_for_any_value_of_it_it_is_not_a_constant_0.1
M9 (d, preflight) the OLD T hard-coded at the call: the PREFLIGHT must FAIL (G13 temp_reach)   E  E_t002_the_real_main_runs_preflight_PASS_then_trains
```
  The full 19-row mutation table (M1-M9 trainer-side, G1-G10 gate-side) is in `logs/` of the patched lane (`logs/tests_patched/test_rl_pickcon.py.log`) and in the test's own stdout; every row turns >= 1 expected check RED (a mutation that stayed GREEN would fail the test).
* **End to end through the REAL `main()`** (CPU stubs for GPU parts only): default == explicit 0.1 bit for bit through the whole run (rows + saved tensors, max |diff| 0.0); 0.02 moves arm B and leaves arm A untouched; the data event records the BUILT T; G13's `temp_reach` is live and re-judged by the gate; with the old T hard-coded the PREFLIGHT FAILS (M9).
* **G13 `temp_reach` (new, in-run).** The run's own preflight re-runs its `train_losses` assembly with `dual.temp` doubled and requires the L_con-only gradient to move (and arm A to stay at exactly 0.0). This catches a hard-coded old T that the data event alone would not reveal (the event reports what the dual HOLDS, `temp_reach` reports what the loss READS).

### G-DVB entry and the deliberate-regression arm
* **Which gate:** the scorer trainer (`scorer_finetune.py`) is governed by `refe/launch_gate.py` + `refe/gate_checks.py` (profile `refe_rls1d`); `stack/scripts/launch_gate*.py` do not cover it (0 hits for scorer / epo).
* **Entry:** `DVB_KINDS_SCORER["epo_con_temp"] = "built"` (`gate_checks.py:649`), observed through the data event's `epo_con.temp` (`SCORER_EPO_CON_KEYS += "temp"`, `:662`); the DECLARED side is the effective value (`scorer_effective`, `:667`: `None` -> `--epo-temp`), the BUILT side is `EpoDual.temp`. **A new trainer flag without an entry is refused by the gate itself** (existing: "trainer flags with NO declared-vs-built entry (a new lever needs its check)", `check_dvb_scorer`; its `unknown_flag` arm; and the dead-lever scan, which now sees `main` read `epo_con_temp`). Deleting the new entry turns `G_dvb_registry_classifies_epo_con_temp_as_built` RED (G2).
* **Regression arm `con_temp_ignored`** (`gate_checks.py:1080`): sets the data event's `epo_con.temp` to twice the built/effective value -- "the trainer ignored the flag" -- and **FAILS the gate on both the named and the unnamed argv** (`G_dvb_REGRESSION_ARM_con_temp_ignored_FAILS_on_the_named_argv`, `..._unnamed_argv`; leaves no residue). `judge_g13` re-judges `temp_reach` from raw fields with a negative twin per clause (absent, wrong T, not exactly 2x, not moved, arm A touched).
* **Launch profile:** `--epo-con-temp` is added to `AUXS1_EPO_FLAGS` (the aux profiles keep forbidding every EPO flag); `rls1_refusals` compares the EFFECTIVE value, so a registered arm refuses a changed T and the message says what the argv resolves to (G6/G7).
* **Draft registration of the next arm (NOT applied -- arm name and T are a PI / Master Mind decision).** Shape, exercised by the `R0p` fixture in `test_rl_pickcon.py`: `RLS1D_ARM_EPO_REGISTERED["R0p"] = {**RLS1D_CON_ON, "epo_con_temp": 0.02}`; `RLS1D_ARM_MUST_NAME["R0p"] = RLS1D_CON_FLAGS + ("--epo-con-temp",)`; plus `RLS1D_ARM_CLASS / SEEDS / CON_TERMS` rows and the arm in the suite list.

### Pinned-chain bookkeeping (why the edit is bigger than 40 lines)
`test_rl_safeobj / test_rl_r0c / test_rl_s1b` pin the trainer by git blob through an ordered chain of reversal tables. Every trainer edit therefore adds a reversal layer: `test_rl_s1b.py` gained `PICKCON_REVERSALS` / `undo_pickcon` / `pickcon_reverse_file` (19 generated pairs) and `E2D4422_TRAINER_BLOB = e063928807b7...` (= `0a5958a:refe/scorer_finetune.py`); `test_rl_pickcon.py` P-checks re-derive that blob EXACTLY from the patched file. `test_rl_safeobj.py` and `test_launch_gate.py` got the matching snippet / row-count updates (observed rows 23 -> 24).

### Dependent test sweep (full form, not `-k`; baseline = pristine `0a5958a` export run in the same lane harness)
> FIRST PASS, kept as the record. Its two REGRESSION rows (`test_dump_scores.py`, `test_safeobj_outcome.py`) are superseded by **Regression fix** below.
<!--TESTS-->
Lanes complete: True. Regressions: ['test_dump_scores.py', 'test_safeobj_outcome.py']. Failing at baseline too: ['test_aux_navtest_forward.py', 'test_p6_chain.py', 'test_rls1_chain.py', 'test_rls1_queue.py'].

| test file (run in full form) | baseline rc | baseline token | patched rc | patched token | patched s | verdict |
|---|---|---|---|---|---|---|
| `test_aux_g12_bf16.py` | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 36.9 | **PASS = PASS** |
| `test_aux_navtest_forward.py` | 1 | AssertionError: the head module is not the D: copy | 1 | AssertionError: the head module is not the D: copy | 2.2 | **FAILS AT BASELINE TOO (environmental)** |
| `test_aux_s1_wiring.py` | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 250.0 | **PASS = PASS** |
| `test_dump_scores.py` | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 1 | AssertionError: the unmutated source FAILS its own checks: ['parser: exactly ONE new dest  | 4.6 | **REGRESSION** |
| `test_gate_gpu_probes.py` | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 3.0 | **PASS = PASS** |
| `test_gate_suite_prerun.py` | 0 | ZZTEST_PASS 144 | 0 | ZZTEST_PASS 144 | 262.3 | **PASS = PASS** |
| `test_label_dir_refusal.py` | 0 | ZZTEST_PASS 6 | 0 | ZZTEST_PASS 6 | 4.7 | **PASS = PASS** |
| `test_launch_gate.py_static_no-banked` | 0 | ZZLG_STATIC_PASS 674 | 0 | ZZLG_STATIC_PASS 674 | 55.8 | **PASS = PASS** |
| `test_p2r_chain.py` | 0 | ZZTEST_PASS 139 | 0 | ZZTEST_PASS 139 | 5.1 | **PASS = PASS** |
| `test_p2rep_chain.py` | 0 | ZZTEST_PASS 353 | 0 | ZZTEST_PASS 353 | 257.1 | **PASS = PASS** |
| `test_p6_chain.py` | 1 | ZZGATE-FAIL-000000000000ZZ | 1 | ZZGATE-FAIL-000000000000ZZ | 146.6 | **FAILS AT BASELINE TOO (environmental)** |
| `test_p6_cpu_probe.py` | 0 | ZZTEST_PASS 111 | 0 | ZZTEST_PASS 111 | 125.2 | **PASS = PASS** |
| `test_p6_levers.py` | 0 | ZZP6_PASS 156 | 0 | ZZP6_PASS 156 | 349.0 | **PASS = PASS** |
| `test_p6_overlap.py` | 0 | ZZTEST_PASS 54 | 0 | ZZTEST_PASS 54 | 1.4 | **PASS = PASS** |
| `test_rl_epo.py` | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 133.8 | **PASS = PASS** |
| `test_rl_pickcon.py` | None | (not run) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 39.1 | **NEW TEST** |
| `test_rl_r0c.py` | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 116.1 | **PASS = PASS** |
| `test_rl_reward.py` | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 1.4 | **PASS = PASS** |
| `test_rl_s1b.py` | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 32.6 | **PASS = PASS** |
| `test_rl_safeobj.py` | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 408.0 | **PASS = PASS** |
| `test_rl_terms_label.py` | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 3.4 | **PASS = PASS** |
| `test_rls1_chain.py` | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 79.4 | **FAILS AT BASELINE TOO (environmental)** |
| `test_rls1_queue.py` | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 65.9 | **FAILS AT BASELINE AND PATCHED, DIFFERENT TAIL** |
| `test_s1aux_chain.py` | 0 | ZZTEST_PASS 256 | 0 | ZZTEST_PASS 256 | 78.6 | **PASS = PASS** |
| `test_s1c.py` | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 153.3 | **PASS = PASS** |
| `test_safeobj_outcome.py` | 0 | ZZTEST_PASS 133  (1 unmutated suite of 172 checks + the CLI end to end, 132 RED mutations) | 1 | AssertionError: the UNMUTATED tool fails: ['outcome:o1->O1', 'outcome:o1_ttc->O1', 'outcom | 71.5 | **REGRESSION** |
| `test_sft4r_chain.py` | 0 | ZZTEST_PASS 31 | 0 | ZZTEST_PASS 31 | 3.2 | **PASS = PASS** |
| `test_sft4r_gate.py` | 0 | ZZTEST_PASS 153 | 0 | ZZTEST_PASS 153 | 11.0 | **PASS = PASS** |
| `test_suite_nesting.py` | 0 | ZZTEST_PASS 95 | 0 | ZZTEST_PASS 95 | 8.5 | **PASS = PASS** |
<!--/TESTS-->

### Measured magnitudes behind (a), (b), (c) (`so_pickcon_measure.py`, read from the test's own closures)
<!--MEASURE-->
(MEASURED by `so_pickcon_measure.py` -> `pickcon_measure.json`, the test's own closures; tiny synthetic scorer, a wiring magnitude, not a driving result)

* (a) max |constraint grad| on arm B: 0.327485 at T 0.1, 0.882004 at T 0.02; max |diff| between them 0.564079 (relative to the T 0.1 max: 1.722); arm A's gradient differs by 0; logged cB_ttc 0.463184 -> 0.472639; twin (same T twice) max |diff| 0.
* (b) default vs explicit 0.1: max |diff| of total gradient B 0, constraint gradient 0, gradient A 0; loss bits equal True, logged parts equal True; effective T 0.1 vs 0.1; zero control (lambda 0) max |constraint grad| 0; inheritance at --epo-temp 0.05: effective T 0.05, diff vs explicit 0.05 = 0, vs explicit 0.1 = 0.280822.
* (c) hard-pick exactness:
  * T1e-06: |policy - one-hot| max 0; cA_ttc 0.3125 (pick label mean 0.3125); cB_ttc 0.21875 (pick label mean 0.21875); g_ttc 0.09375, equals the hard-pick gap exactly: True; g + p1gap (the wedge) 0
  * T1e-08: |policy - one-hot| max 0; cA_ttc 0.3125 (pick label mean 0.3125); cB_ttc 0.21875 (pick label mean 0.21875); g_ttc 0.09375, equals the hard-pick gap exactly: True; g + p1gap (the wedge) 0
  * T0.1: |policy - one-hot| max 0.654203; cA_ttc 0.334293 (pick label mean 0.3125); cB_ttc 0.298258 (pick label mean 0.21875); g_ttc 0.0360354, equals the hard-pick gap exactly: False; g + p1gap (the wedge) -0.0577146
<!--/MEASURE-->

---------------------------------------------------------------------------------------------------

## DELIVERABLE 2 -- SO-CARVE A2 control 3 redesigned near-tie-robust, re-run on the banked R0 states (CPU)

**Why control 3 failed.** "T = 1e-4 must reproduce L_P1 to 1e-6" cannot hold on sets whose top two selection scores differ by less than a few T in log space: the softmax puts mass on a runner-up the argmax ignores. That is a property of the data and of the tolerance, not of the machinery.

### A2R -- the pre-registered robust control (`PREREG_SOPICKCON_A2R_NEARTIE.md`, sha256 `55fb333c...a8a5`, written BEFORE any margin / runner-up / coverage number)
* **epsilon derived from the tolerance, not from the data:** runner-up mass `r = sum_{j != argmax} exp((l_j - l_max)/T_c)` at `T_c = 1e-4`; retained set `S* = {r_A, r_B <= rho}`, `rho = 5e-9` (a retained set contributes <= 100(rho + rho) = 1e-6 points, so the mean error over ANY retained subset is <= the original 1e-6 tolerance); equivalent top-2 margin `eps_eq = T_c ln(63/rho) = 2.3257e-3`; coverage floor 0.50; five outcomes committed (CONFIRMED / SHIFTED / NONE / CONTROL-FAILS / UNDERPOWERED).
* **Controls: all PASS.** (1) T = 0.1 on all sets reproduces `tempsweep_a2.json` L_T (<= 1e-9) at every state; (2) B := A reads exactly 0.0; (3') T_c = 1e-4 on S* reproduces L_P1(S*) to **<= 2.2e-10** at all three states (the original control read -0.0149 / +0.0259 / -0.0174 on all sets -- the near-ties); (4) planted-array self-test PASS (an exact tie is excluded and the all-sets control FAILS on it; a 1e-2 margin is retained).
* **Outcome (as committed): `A2R-UNDERPOWERED`.** Coverage |S*|/3,000 = **0.458 / 0.449 / 0.439 < 0.50**. Near-ties are the NORM for this scorer: 1,426 / 1,433 / 1,483 of 3,000 sets have a top-2 margin below eps_eq in arm A (no exact ties). No tracking verdict is drawn from S*; **A2 stays PROVISIONAL by A2R's own commitment.**

### A2R2 -- the follow-up the failed arm leaves behind (`PREREG_SOPICKCON_A2R2_TIEAWARE.md`, sha256 `81a8f3c9...596d`, written AFTER `a2r_neartie.json` and BEFORE any A2R2 number; post-hoc-motivated by A2R and said so in its section 0)
The pick on a near-tie is decided by numerical noise, so its loss there is an arbitrary draw. A2R2 defines the pick tie-aware: the label averaged over the candidates within tau of the top, `tau_0 = eps_eq` (fixed by A2R, not chosen), over ALL 3,000 sets. Same bootstrap, labels, states. Controls (T = 0.1 reproduction, B := A zero, tau = 0 reproduces L_P1 to 1e-12, planted array) all PASS.
* **Outcome (as committed): `A2R2-ALIGNED(T* = 0.02)`.** `|L_T - L_P1^{tau_0}|` <= 0.25 at all three states for T = 0.02 and 0.01, not for 0.05 or 0.1.

| T | u250 diff [95 % CI] | u500 diff [95 % CI] | u1000 diff [95 % CI] | max abs | aligned (<= 0.25) |
|---|---|---|---|---|---|
| 0.1  | -0.452 [-0.878, +0.004] | -0.567 [-1.113, -0.054] | -0.760 [-1.258, -0.272] | 0.760 | no |
| 0.05 | -0.258 [-0.619, +0.130] | -0.360 [-0.831, +0.083] | -0.471 [-0.893, -0.057] | 0.471 | no |
| **0.02** | -0.121 [-0.385, +0.160] | -0.141 [-0.491, +0.184] | -0.208 [-0.521, +0.118] | **0.208** | **yes** |
| 0.01 | -0.103 [-0.307, +0.111] | -0.032 [-0.293, +0.213] | -0.103 [-0.334, +0.134] | 0.103 | yes |

(diff = L_T(all sets) - L_P1^{tau_0}(all sets), points of TTC loss; negative = the tempered expectation UNDERSTATES the pick's loss. Source: `a2r2_tieaware.json`.)

### Answer: does T ~ 0.02 still track the pick?
**Yes, on the committed bar -- with a thin margin and three caveats.**
1. **Thin:** the worst state (u1000) is 0.208 against the 0.25 bar (margin 0.042); T = 0.01 has 0.103 and is the more robust choice. T = 0.1 (today's behaviour) misses at every state and its CI excludes 0 at u500 and u1000, so the SO-CARVE wedge survives the near-tie treatment (it was not an artefact of near-ties).
2. **The bar is partly noise:** the pick's own loss moves with the tie-resolution tau in {0, 1e-4, 1e-3, tau_0, 1e-2} by up to ~0.3 points (u500: L_P1^tau 0.633 -> 0.925; u250 and u1000 move ~0.1), i.e. the thing T is being matched to is tie-break noisy at the scale of the 0.25 bar. The sensitivity (descriptive, NOT a verdict): 0.02 is aligned for tau in {0, 1e-4, 1e-3, tau_0} but NOT at tau = 1e-2; 0.01 is aligned at every tau.
3. **No lever-effect claim; the CIs for T = 0.02 and 0.01 all span 0** (the difference to the pick is not distinguishable from 0 at "another draw of logs"), the estimator says nothing about training or inference variance, R0 is one run per state, and A2R2 is post-hoc-motivated (its pre-registration postdates A2R's UNDERPOWERED outcome, and says so). A2R's own tag stays UNDERPOWERED / A2 PROVISIONAL.

**Recommendation for the Master Mind (a choice, not a decision):** the next arm names `--epo-con-temp 0.02` (the SO-CARVE T*, supported on both reads) and, if the budget allows one more arm, a sibling at 0.01, which is the more robust under the sensitivity read. Either way the flag is in the argv, never the default.

### Four metric families (this deliverable)
LONGITUDINAL: TTC only (the constraint quantity itself). LATERAL / TACTICAL: no instrument at this tier, not computable here (n = 3,000 sets, no lateral / tactical labels in the dumps). STRATEGIC: **PAUSED** (PI ruling 2026-09-19). Per family, never pooled; estimator = log-cluster bootstrap, "another draw of logs".

---------------------------------------------------------------------------------------------------

## Regression fix (second pass, 2026-10-18) -- `test_dump_scores.py` and `test_safeobj_outcome.py` (baseline rc 0, first-pass patched rc 1)

Both were the patch's *legitimate* schema additions meeting a tool or test that pinned the OLD schema by exact equality. Neither half of the patch was wrong; the pinned side learned the addition. No check was weakened, removed or relaxed (the test edits only ADD).

### R1 -- `refe/test_dump_scores.py` (the test was behind)
* **Root cause.** The test pins the parser's dest set: `dests - BASE_DESTS - aux_dests - epo_dests == {"dump_scores"}`. The patch adds exactly ONE dest, `epo_con_temp` (the failing message's dest list shows no other new one), and puts it in the right place: `ap.add_argument("--epo-con-temp", ...)` directly after `--epo-con-lam0`, and into `EPO_CON_DESTS` (the SAFEOBJ group). So the parser is right and the test's `epo_dests` still named the six pre-PICKCON EPO dests.
* **Fix (the test side, the way SAFEOBJ did it: it extended `epo_dests` in place by five).** `epo_dests |= {"epo_con_temp"}` -- the all-or-none rule now covers seven dests -- and the check also requires the default `getattr(ns_, "epo_con_temp", None) is None` (= inherits `--epo-temp`, every pre-PICKCON arm bit for bit). Stricter, not looser: a missing member of the group, an unexpected extra dest, or a non-None default now fails.
* **Measured (MEASURED, full form, `fix/` lane + a direct run).** `ZZTEST_PASS 16` = the baseline's 16 (41 checks + 14 RED mutations + coverage), rc 0.

### R2 -- `eval/safeobj_outcome.py` + `eval/test_safeobj_outcome.py` (the TOOL was behind)
* **Root cause.** `EpoDual.built_facts()` now returns a ninth key, `temp` (the T the constraint's softmax runs at); the trainer's `data` event therefore carries a nine-key `epo_con` block. `read_log` compares that block for EXACT equality with `REGISTERED_EPO_CON` (eight keys, `eval/safeobj_outcome.py:361`), so every log written by the real (patched) dual read `NOT_READ: the data event's BUILT epo_con block ... is not the registered ...`, and with it ~70 `outcome:*` / `e2e:*` checks. Not a renamed constant, log field or JSON schema: an ADDED key in a block that is compared by equality. Only `scorer_finetune.py` among the patched files feeds it (through `built_facts`).
* **Which side is wrong: the tool.** It has to read BOTH (i) the banked R0d / R1d logs, written by trainers that predate PICKCON (the real banked `raw/2026-10-18-r0d-outcome/raw/sft.jsonl` carries the eight-key block), and (ii) logs of the patched trainer, whose block has `temp` = `--epo-temp` = 0.1 when `--epo-con-temp` is unnamed. PREREG_SAFEOBJ section 2 registers the constraint's policy at **T = 0.1** and says "a second temperature would be a second lever" -- so a nine-key block is the registered block **iff `temp` is exactly 0.1**; any other value (0.02 = the PICKCON arm's T, `None`, a bool, a string) is still refused with the same message.
* **Fix.** `eval/safeobj_outcome.py`: `REGISTERED_CON_TEMP = 0.1` and `_registered_view(blk)` (strips `temp` iff it is exactly the registered T; a legacy block passes through unchanged), applied to `data["epo_con"]` immediately before the UNCHANGED comparison line `if data.get("epo_con") != registered:`. `REGISTERED_EPO_CON` is untouched (so the existing "registered lr is wrong" mutation anchor and the PREREG pins stay valid).
* **Test additions (ADD only).** `sim_log` builds the real dual as `main` now does (`temp=EPO_TEMP`); 3 new modified arms (`legacy_block` = temp stripped, `n_log_temp` = 0.02, `n_log_temp_none`); 5 new checks (`notread:n_log_temp`, `notread:n_log_temp_none`, `tolerated:a_legacy_data_block_without_temp_is_read_exactly_as_before_PICKCON`, `compat:the_banked_R0d_trainer_log_has_the_legacy_epo_con_block_and_is_read` -- a REAL producer's log, `real_main:the_real_trainers_epo_con_block_carries_temp_equal_to_the_registered_T_and_to_epo_temp`); the dual pin now also ties `REGISTERED_CON_TEMP` = the prereg text "`epo_policy` at T = 0.1" = the trainer's `--epo-temp` default = the test constant, and the real dual's built facts = registered + `temp`; 3 new RED mutations (T not enforced -> RED exactly at `notread:n_log_temp`; legacy block refused -> RED at the legacy and banked-log checks; `temp` never stripped -> RED at 59 checks). Counts: checks 172 -> 177, mutations 132 -> 135, expected `ZZTEST_PASS 136`.
* **Measured so far (quick harness `fix/quick_outcome.py`, patched tree).** Unmutated suite: 0 failing of 177 checks. Mutations run there: the 3 new ones and "the BUILT dual is not compared with the registered one" -- all RED where they exist to (`expected subset` true). The full 135-mutation sweep is the lane below.

### Environmental confirm -- `test_rls1_queue.py`
Baseline and patched die in the SAME assertion on the SAME cause, `ps: unknown option -- o` (Git-bash `ps` has no `-o pgid=`), log line `REFUSED: the trainer of R0 is not its own process group (pgid , pid N)`; the tails differ only in pid / timestamp / seconds -- environmental, not a PICKCON effect.

### Fix-pass test table (final verdicts; first-pass rows that were not re-run stand)
<!--FIX-->
Fix pass complete: True. Regressions after the fix: none. Fixed: ['test_dump_scores.py', 'test_safeobj_outcome.py']. Failing at baseline too (environmental): ['test_aux_navtest_forward.py', 'test_p6_chain.py', 'test_rls1_chain.py', 'test_rls1_queue.py']. Token mismatches on re-run files: none.

Tested == staged (sha256 of `out/` file equals the `patched_full` file the lane ran): `safeobj_outcome.py` True, `test_safeobj_outcome.py` True, `test_dump_scores.py` True

| test file (full form) | re-run in fix pass | baseline rc | baseline token | patched rc (first pass) | patched rc (fix pass / final) | patched token (final) | patched s | verdict |
|---|---|---|---|---|---|---|---|---|
| `test_aux_g12_bf16.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 0 | 0 | ZZTEST_PASS 7  (1 unmutated suite of 30 checks + 6 RED mutations) | 36.9 | **PASS = PASS** |
| `test_aux_navtest_forward.py` | no (first-pass row stands) | 1 | AssertionError: the head module is not the D: copy | 1 | 1 | AssertionError: the head module is not the D: copy | 2.2 | **FAILS AT BASELINE TOO (environmental)** |
| `test_aux_s1_wiring.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 0 | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 250.0 | **PASS = PASS** |
| `test_dump_scores.py` | yes | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 1 | 0 | ZZTEST_PASS 16  (1 unmutated suite of 41 checks + 14 RED mutations + coverage) | 28.8 | **FIXED (first pass REGRESSION) -> PASS = PASS** |
| `test_gate_gpu_probes.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 0 | 0 | ZZTEST_PASS 16  (incl. the CUDA probe on this GPU) | 3.0 | **PASS = PASS** |
| `test_gate_suite_prerun.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 144 | 0 | 0 | ZZTEST_PASS 144 | 262.3 | **PASS = PASS** |
| `test_label_dir_refusal.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 6 | 0 | 0 | ZZTEST_PASS 6 | 4.7 | **PASS = PASS** |
| `test_launch_gate.py_static_no-banked` | yes | 0 | ZZLG_STATIC_PASS 674 | 0 | 0 | ZZLG_STATIC_PASS 674 | 42.8 | **PASS = PASS** |
| `test_p2r_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 139 | 0 | 0 | ZZTEST_PASS 139 | 5.1 | **PASS = PASS** |
| `test_p2rep_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 353 | 0 | 0 | ZZTEST_PASS 353 | 257.1 | **PASS = PASS** |
| `test_p6_chain.py` | no (first-pass row stands) | 1 | ZZGATE-FAIL-000000000000ZZ | 1 | 1 | ZZGATE-FAIL-000000000000ZZ | 146.6 | **FAILS AT BASELINE TOO (environmental)** |
| `test_p6_cpu_probe.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 111 | 0 | 0 | ZZTEST_PASS 111 | 125.2 | **PASS = PASS** |
| `test_p6_levers.py` | no (first-pass row stands) | 0 | ZZP6_PASS 156 | 0 | 0 | ZZP6_PASS 156 | 349.0 | **PASS = PASS** |
| `test_p6_overlap.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 54 | 0 | 0 | ZZTEST_PASS 54 | 1.4 | **PASS = PASS** |
| `test_rl_epo.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 0 | 0 | ZZTEST_PASS 42  (1 unmutated suite + 41 RED mutations) | 133.8 | **PASS = PASS** |
| `test_rl_pickcon.py` | yes | None | (not run) | 0 | 0 | ZZTEST_PASS 20  (1 unmutated suite + 19 RED mutations) | 17.3 | **NEW TEST** |
| `test_rl_r0c.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 0 | 0 | ZZTEST_PASS 26  (1 unmutated suite + 25 RED mutations) | 116.1 | **PASS = PASS** |
| `test_rl_reward.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 0 | 0 | ZZTEST_PASS 31  (1 unmutated suite + 29 RED mutations + coverage) | 1.4 | **PASS = PASS** |
| `test_rl_s1b.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 0 | 0 | ZZTEST_PASS 18  (1 unmutated suite + 17 RED mutations) | 32.6 | **PASS = PASS** |
| `test_rl_safeobj.py` | yes | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 0 | 0 | ZZTEST_PASS 61  (1 unmutated suite + 60 RED mutations) | 208.7 | **PASS = PASS** |
| `test_rl_terms_label.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 0 | 0 | ZZTEST_PASS 39  (1 unmutated suite + 37 RED mutations + coverage) | 3.4 | **PASS = PASS** |
| `test_rls1_chain.py` | no (first-pass row stands) | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 1 | 1 | AssertionError: MODEL_REGISTRY.md not found above the package | 79.4 | **FAILS AT BASELINE TOO (environmental)** |
| `test_rls1_queue.py` | no (first-pass row stands) | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 1 | 1 | AssertionError: {'rc': 1, 'out': "ps: unknown option -- o\nTry `ps --help' for more inform | 65.9 | **FAILS AT BASELINE TOO (environmental)** |
| `test_s1aux_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 256 | 0 | 0 | ZZTEST_PASS 256 | 78.6 | **PASS = PASS** |
| `test_s1c.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 0 | 0 | ZZTEST_PASS 220  (2 skipped, not counted) | 153.3 | **PASS = PASS** |
| `test_safeobj_outcome.py` | yes | 0 | ZZTEST_PASS 133  (1 unmutated suite of 172 checks + the CLI end to end, 132 RED mutations) | 1 | 0 | ZZTEST_PASS 136  (1 unmutated suite of 177 checks + the CLI end to end, 135 RED mutations) | 586.1 | **FIXED (first pass REGRESSION) -> PASS = PASS** |
| `test_sft4r_chain.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 31 | 0 | 0 | ZZTEST_PASS 31 | 3.2 | **PASS = PASS** |
| `test_sft4r_gate.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 153 | 0 | 0 | ZZTEST_PASS 153 | 11.0 | **PASS = PASS** |
| `test_suite_nesting.py` | no (first-pass row stands) | 0 | ZZTEST_PASS 95 | 0 | 0 | ZZTEST_PASS 95 | 8.5 | **PASS = PASS** |
<!--/FIX-->

## Manifest (all under `C:/Users/Admin/pickcon/out/`, mirroring repo paths; `paths.json` lists them as `[repo path, abs src, "raw"]`)
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`
| artifact | repo path | edit of |
|---|---|---|
| the trainer (flag, dual, G13 `temp_reach`, refusals) | `PK/refe/scorer_finetune.py` | base blob in `BASE_BLOBS.json` |
| gate checks (G-DVB row, `con_temp_ignored`, `judge_g13`) | `PK/refe/gate_checks.py` | idem |
| launch gate (profile flags, effective-value compare) | `PK/refe/launch_gate.py` | idem |
| reversal layer for the pinned chain | `PK/refe/test_rl_s1b.py` | idem |
| snippet / count updates | `PK/refe/test_rl_safeobj.py`, `PK/refe/test_launch_gate.py` | idem |
| NEW reachability + gate test | `PK/refe/test_rl_pickcon.py` | new file |
| regression fix: outcome tool accepts the legacy and the registered-T block | `PK/eval/safeobj_outcome.py` | `0a5958a` blob in `BASE_BLOBS.json` |
| regression fix: outcome tool's test (adds only) | `PK/eval/test_safeobj_outcome.py` | idem |
| regression fix: parser-dest pin learns `epo_con_temp` | `PK/refe/test_dump_scores.py` | idem |
| this package | `PK/raw/2026-10-18-so-pickcon/` : `RESULT.md`, `PREREG_SOPICKCON_A2R_NEARTIE.md/.sha256`, `PREREG_SOPICKCON_A2R2_TIEAWARE.md/.sha256`, `so_a2r_neartie.py`, `so_a2r2_tieaware.py`, `so_pickcon_measure.py`, `a2r_neartie.json`, `a2r2_tieaware.json`, `pickcon_measure.json`, `TEST_RESULTS.md`, `test_results.json`, `BASE_BLOBS.json`, `PATCH_*.diff`, `logs/` | new |

Not committed, not pushed. To land: copy the TEN files (`PK/refe/`: scorer_finetune, gate_checks, launch_gate, test_rl_s1b, test_rl_safeobj, test_launch_gate, test_rl_pickcon, test_dump_scores; `PK/eval/`: safeobj_outcome, test_safeobj_outcome -- the staged copies are the TESTED ones), keep `raw/2026-10-18-so-pickcon/` whole. `PK/SHIP_MANIFEST.json` lists `refe/test_dump_scores.py` (and other edited files) by sha256 -- regenerate it at landing (`launch_gate.py ship-manifest`), it is not part of this package.
Honest limits: no pod run, no GPU, no T1; the tests prove the flag REACHES the loss and the gate REFUSES a mismatch -- they say nothing about whether T = 0.02 improves driving; that is the next arm's question.

## Draft BACKLOG status line (not applied)
`SO-PICKCON -- 2026-10-18 -- (1) DONE at 0 GPU: --epo-con-temp (default None = inherits --epo-temp, bit-identical; old T = `a.epo_temp` at 0a5958a:scorer_finetune.py:1049), G-DVB row + con_temp_ignored arm + G13 temp_reach, test_rl_pickcon ZZTEST_PASS 20 (identity exactly 0.0, mutation RED), dependent sweep see TEST_RESULTS.md; the next arm MUST NAME --epo-con-temp. (2) A2R (near-tie exclusion) = UNDERPOWERED (coverage 0.44-0.46 < 0.50, control 3' passes <= 2.2e-10); A2R2 (tie-aware pick) = ALIGNED(T*=0.02), margin 0.208/0.25 thin, T=0.01 more robust, CIs span 0, post-hoc-motivated. STRATEGIC family PAUSED. Next: PI picks the arm name + T (0.02, optional 0.01 sibling); register in refe_rls1d; launch-gate it.`
