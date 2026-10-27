# oaccflat_check: do the trainer's flags COMPOSE for DAG-OACC and FLAT-DAG?

Date 2026-10-27. Code under test = git tip `0a820606` (every one of the 1,565 `.py` files of `stack/` + `taniteval/` in `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/` re-hashed against the tip with
`git hash-object --no-filters`: 1565 equal, 0 differ, 0 absent — `verify_tree.py`). Nothing committed, pushed, or edited in a checkout. No box-2 contact, no GPU, no optimiser step.

## HEADLINE

1. **DAG-OACC composes. PASS at both seeds** (24/24 checks each). The trainer's REAL in-process gate accepts DAG_s's argv + `--w-acc 49.6969`, with the acceleration term and the DAgger swap both live in the same gate run.
2. **FLAT-DAG composes. PASS at both seeds** (25/25 each) — **but only if the argv NAMES `--o-interface cut`, and the gate does not enforce that.**
3. ⛔ **F1 (launch blocker for FLAT-DAG, fix STAGED): the gate ACCEPTS a FLAT-DAG argv that forgets `--o-interface cut`.** `INIT_CKPT_MATCH` (`v8f_mini_train.py:506`) does not contain `o_interface`, so the argv builds a FULL
   model, loads FLAT's cut-trained weights into it, and every probe passes (MEASURED, both seeds). Only C-F3 would refuse it, after the GPU hours. **Fix staged:** one key in `INIT_CKPT_MATCH`, read only when the checkpoint has it
   (every M2b checkpoint has none, so DAG / DAG-OACC are unaffected). New test: RED on the tip trainer (3 failed, 4 controls pass), GREEN on the fix (7 passed). **Needs the Master Mind to land it before the FLAT-DAG stage runs;
   and the queue's FLAT-DAG stages must name `--o-interface cut` regardless (PREREG §1).**
4. **The iteration-1 bank cannot stand in for the FLAT bank, not even for composition.** K-B2 ties the bank to `--init-ckpt`: FLAT-DAG on an M2b-generated bank is REFUSED by the real gate (both seeds). The bank generator
   accepts the FLAT checkpoint and FLAT's argv, and the model it builds from FLAT IS the cut policy.
5. ⚠ **F2 (gap in the gate, not a composition failure):** the gate does not cover the training loop's own `b = dag.apply(b, step)` line (`v8f_mini_train.py:1605`). Deleting it leaves the gate GREEN (MEASURED, mutant `loop_dag_apply`),
   although `dagger_probe`'s docstring says the gate would refuse. The repo's unit tests do cover it (4 tests re-run here, 4 pass). The run's own `summary.json` `dagger_n / dagger_eligible` is the post-hoc check:
   DAG's real run read 14,688 / 145,037 = 0.1013. **The DAG-OACC and FLAT-DAG readers must keep that share check (0.1 ± 0.02).**

6. **F3 (already RED on the tip, unrelated):** `test_v8f_orc_chain.py` has 2 failing queue tests (stale vs the DAG stages), identical with and without the staged trainer. `pytest -q` is not green on the tip; whoever lands next should know.

## Scope: what is real and what is not (read this before quoting anything)

| | |
|---|---|
| trainer, model, DAgger library, bank generator, gate probes | **REAL**, tip `0a820606`, called as a library exactly as `dagq/real_gate_proof.py` does |
| model dimensions | **REAL** (d 256, n_q 4, nb 12, hist 4, hmax 16, pool4x10) |
| `--init-ckpt` | **REAL checkpoints**: `M2b_s{0,1}` (sha256 `601cb435…` / `6104fb15…`) and `FLAT_s{0,1}` (`8496934a…` / `6fdfc9b0…`), loaded strictly, 0 tensors differ after load, `source_step` 19164 |
| the argv | derived mechanically from the REAL banked argv.json (DAG: `v8f-dagger/M2b_s{s}_dag/argv.json`; FLAT: `raw/2026-10-26-flat/runs/M2b_s{s}_flat/argv.json`); only the data/path/device/batch flags are re-pointed |
| data | ⚠ **SYNTHETIC world** (6 train / 2 eval clips, T 90). The real corpus (`cache_full`, v9 labels, P10 prior file) exists on box 2 only. The dev-box route `real_gate_proof.py` is also synthetic. |
| banks | made by the REAL generator (`v8f_dagger_queue_tool.py bank` → `v8f_dagger_bank.py gen` ×2 shards + merge) from the REAL M2b / FLAT checkpoint on the synthetic world, NOT the iteration-1 banks (they name real TRAIN clips) |
| device | CPU, `CUDA_VISIBLE_DEVICES=-1`, one torch process at a time |

So "the gate accepts it" is MEASURED for the real code on a synthetic world. What real data could change (probe batch contents, eligible-row counts, COVERAGE of the FLAT bank on 4,512 real clips) is UNVERIFIED below.

## ARM 1 — DAG-OACC = DAG_s argv verbatim + `--w-acc 49.6969`

Instrument: `arm1_dag_oacc.py` (`CG_SEED=0|1`), results `out/arm1_s0.json`, `out/arm1_s1.json`; records `out/arm1_s{S}/{selfcheck_treatment,selfcheck_control,composed_probe_treatment,composed_probe_control}.json`.

### The hops (flag → namespace → model attribute → forward → loss → backward)

| hop | where (tip) |
|---|---|
| argv `--w-acc`, default 0.0 | `stack/scripts/v8f_mini_train.py:120` |
| namespace → config | `make_config` `:243`, `w_acc=a.w_acc` `:250` → `V8FMiniConfig.w_acc` `stack/tanitad/models/v8f_mini.py:187` (validated finite ≥ 0 `:206`) |
| G-DVB declared == built | `v8f_mini_train.py:595-599` (+ the `cfg.w_acc` loop `:613`); the built lever `v8f_mini.py:924` |
| forward → term | `v8f_mini.py:871` `T["acc"] = accel_term(out["traj30"], gt, gok)` (def `:379`; always logged) |
| term → total | `v8f_mini.py:888` `total = total + cfg.w_acc * T["acc"] if cfg.w_acc > 0.0 else total` |
| total → backward | loop `v8f_mini_train.py:1621` `model.loss(b, …)` → `:1626` `T["total"].backward()` |
| the gate's proof | `acc_probe` `:864`, called `:1518` |
| DAgger argv | `:140` `--dagger-p`, `:143` bank, `:146` sha256, `:150` seed; `--init-ckpt` `:152` |
| bank → swap object | `make_dagger` `:394` (called `:1476`) → `load_dagger_bank` `:365` (K-B2 `:384`, K-B3, COVERAGE) → G-DVB `bad_dagger` `:409` (via `declared_vs_built` `:574`) |
| swap → batch | gate: `dagger_probe` `:1061`, called `:1528`; loop: `b = dag.apply(b, step)` `:1605` → `DaggerSwap.apply` `stack/tanitad/data/v8f_dagger.py:656`, the applying line `:685` `out.update(upd)` |
| composition | the swap runs on the batch BEFORE `model.loss` (`:1605` → `:1621`); `accel_term` reads O's plan and the (kept) GT, so the term is computed on the swapped state with no extra wiring |

### Measured (synthetic world, real weights). Seed 0 / seed 1

| check | s0 | s1 |
|---|---|---|
| REAL gate (`--selfcheck-only`) accepts the argv, `declared_vs_built` PASS | PASS | PASS |
| init = M2b_s{s} (`601cb435…` / `6104fb15…`), applied, `n_mismatch_after_load` | 0 | 0 |
| G-DVB levers: `w_acc` | 49.6969 | 49.6969 |
| G-DVB levers: `dagger_p`, `dagger_seed`, bank sha256 declared == built, `ego_perturb_p` | 0.1, 0, ✓, 0.0 | 0.1, 1, ✓, 0.0 |
| acc term at the probe batch | 0.008411 | 0.005334 |
| `total(w) − total(0)` vs `w × acc` | 0.417988 vs 0.417986 | 0.265076 vs 0.265080 |
| **O's head (`head_o`) gradient from the term** (sum abs diff) | **485.70** | **229.91** |
| controls, exactly 0.0: repeat total / repeat grad / frozen P10 prior grad / WM grad | 0.0 / 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 / 0.0 |
| `dagger_probe`: loss diff, grad diff on the swapped batch; p = 0 control loss / grad | 2.647, 24588; **0.0 / 0.0** | 1.647, 16593; **0.0 / 0.0** |
| K-B2 (bank ↔ init sha), K-B3, COVERAGE ≥ 0.5 read and passed | ✓ (0.661) | ✓ (0.664) |
| **swap share at the declared p = 0.1** (400 steps of the loop's own `dag.apply` schedule; eligible draws) | **0.1009** (3,200) | **0.0919** (3,600) |
| COMPOSED, on the SWAPPED batch: acc term (swapped vs logged) | 0.009683 vs 0.008411 | 0.008669 vs 0.005334 |
| COMPOSED: contribution vs `w × acc` on the swapped batch | 0.48123 vs 0.48124 | 0.43082 vs 0.43082 |
| **COMPOSED: `head_o` gradient from the term on the swapped batch** | **467.99** | **305.36** |
| COMPOSED controls (repeat total/grad, prior grad) | 0.0 | 0.0 |
| **CONTROL (same argv WITHOUT `--w-acc`)**: built `w_acc`; term's contribution at the built weight; its `head_o` gradient | **0.0; exactly 0.0; exactly 0.0** | **0.0; exactly 0.0; exactly 0.0** |
| CONTROL: swap just as live (same share, `dagger_probe` PASS) | ✓ | ✓ |
| `--w-acc -1.0` refused by the real code (`V8FMiniConfig` `ValueError`, before G-DVB) | ✓ | ✓ |

Real-corpus reference points (INHERITED from the banked DAG run, not re-measured): DAG's real `summary.json` `dagger_n / dagger_eligible = 14,688 / 145,037 = 0.1013`; its real gate read acc = 0.0107 at probe weight 1.0.

### Argv: one lever (C-O1)

* The treatment gate argv equals DAG's REAL argv token for token (every non-data flag, same order and values), plus `--w-acc 49.6969` named exactly once. Control gate argv = the same without it. They differ in `--w-acc` and `--out` only.
* Every DAG lever is NAMED on the argv (`--dagger-p 0.1`, `--ego-perturb-p 0`, `--dagger-seed s`, …): no lever relies on a default.

### Amendment 2 of PREREG_V8F_OACC and the one-lever rule from RECORDS — `arm1_pair_records.py`, `out/arm1_pair_records.json` (16/16)

* The tip's `check_x_run` (`v8f_oacc_read.py`, byte-exact from `git show 0a820606:…`, unchanged) returns **no refusal** for the DAG vs DAG-OACC pair. ⚠ **The X side is SIMULATED** (no DAG-OACC run exists): DAG's REAL
  `summary.json` / `selfcheck.json` with only the DAG-OACC fields replaced (argv + `--w-acc`, `argv_sha`, `levers.w_acc`, `acc_probe` = the real gate's). UNVERIFIED until a real DAG-OACC run writes its own records.
* Why the Amendment-2 class cannot bite here: DAG's own summary and selfcheck `levers` are IDENTICAL key for key (no differing key); `residual_zero_init` reads **False in both** (the init checkpoint loads `head_o.2` before the readout);
  DAG's `w` record carries `w_tac` / `w_cons` itself (no older-format B). The control stays strict: a `w_tac` change is refused; a `residual_zero_init` difference alone is not.
* The one-lever rule is assertable from `argv.json` alone, and goes RED (reader refuses, each measured) when: a second lever differs (`--lr`), `--dagger-seed`, the bank sha256, `--init-ckpt`, `--dagger-p`; when `--w-acc` is named twice; when the built `w_acc` ≠ the argv's; when the comparator argv already names `--w-acc`.
* Note: no DAG-OACC reader exists yet; `check_x_run` is the closest committed analogue and is hard-wired to "B = banked M2b". The DAG-OACC reader needs B = DAG (`rsm`/`rargv` = DAG's records) — as exercised here.

## ARM 2 — FLAT-DAG = FLAT_s + DAG's recipe, `--o-interface cut` kept

Instrument: `arm2_flat_dag.py`, results `out/arm2_s0.json`, `out/arm2_s1.json` (25/25 each); records `out/arm2_s{S}/{selfcheck_flat_dag,cut_gradient_readings}.json`.

### The hops

| hop | where (tip) |
|---|---|
| argv `--o-interface` (`full`/`cut`) | `v8f_mini_train.py:116` → `make_config` `:250` → `V8FMiniConfig.o_interface` `v8f_mini.py:185` |
| G-DVB declared == built | `:592` |
| the cut | `v8f_mini.py:764-767` (`if cfg.o_interface == "cut":` `ck_for_o = self.ck_nom…`, `ck_ok = zeros`, `code = zeros_like(code)` AFTER every override) → `run_o` |
| the gate's cut test | `o_interface_probe` `:790`, called `:1514`: perturb everything above O (donor / GT / random T code, S checkpoints shifted by 2.5); under cut O is handed the constants and `traj60` is bit-for-bit unchanged; T still reads S |
| init weights vs interface | `load_init_ckpt` `:510`, `INIT_CKPT_MATCH` `:506` — **`o_interface` is NOT in it (F1)** |
| swap | as Arm 1 |

### Measured. Seed 0 / seed 1

| check | s0 | s1 |
|---|---|---|
| FLAT checkpoint sha256 prefix; its own cfg reads `o_interface: cut` (M2b's has no such key) | `8496934a` ✓ | `6fdfc9b0` ✓ |
| REAL gate accepts FLAT-DAG; init = FLAT_s, applied, `n_mismatch_after_load` 0 | PASS | PASS |
| G-DVB: `o_interface` cut, `dagger_p` 0.1, seed, bank sha256 declared == built | ✓ | ✓ |
| **trainer's cut probe: max abs d(traj60) under donor / GT / random code / shifted S checkpoints** | **0.0 / 0.0 / 0.0 / 0.0** | **0.0 ×4** |
| O is handed code 0, ck = nominal, ok 0 (all four perturbations); T's goal logits move with S (cut leaves S→T) | ✓ (7.70) | ✓ |
| G-LIVE: declared-dead set under the cut | `['e_code.weight']` = actual dead set | same |
| **gradient of the O-PLAN loss (ade/fde) on the LOGGED batch: `e_code.weight`; S groups; T groups; O groups** | **0.0; 0.0; 0.0; 2448** | **0.0; 0.0; 0.0; 2454** |
| same on the **SWAPPED** batch (cut + swap composed) | **0.0; 0.0; 0.0; 3198** | **0.0; 0.0; 0.0; 2293** |
| `e_code.bias`, `e_ck.*` gradient | 3.68, 11.76 (learned CONSTANTS under the cut, by design) | 4.23, 13.54 |
| `dagger_probe`: loss diff / grad diff; p = 0 control | 1.950 / 17720; **0.0 / 0.0** | 2.701 / 19524; **0.0 / 0.0** |
| **swap share at p = 0.1** | **0.0956** (3,200 draws) | **0.0902** (4,400) |
| `live_check` on the swapped batch under the cut | PASS | PASS |
| K-B2 reads FLAT_s as both bank generator and init; COVERAGE | ✓ (0.646) | ✓ (0.649) |
| **CONTROL for the cut reading** (same argv WITHOUT `--o-interface cut`, builds `full`): `e_code.weight`; S; T gradient | **86.79; 349.3; 366.6** | **87.20; 311.5; 280.7** |
| FLAT-DAG-0 (`--dagger-p 0`, no bank flags): gate accepts, cut live, dagger off | ✓ | ✓ |
| pair vs DAG (same world): FLAT-DAG adds exactly `--o-interface cut`; values differ only in `--out`, `--init-ckpt`, bank, bank sha256 | ✓ | ✓ |

The "zero gradient on `e_code`" of the brief is exactly **`e_code.weight`** (the gate itself declares it the dead weight). `e_code.bias` and `e_ck.*` still train: they read constants, so they are learned constants — consistent with PREREG_V8F_FLAT.

### (a) Does the trainer tie the bank to the init checkpoint? YES (K-B2). So no stand-in.

`load_dagger_bank` `:365` → `bank.check_on_policy(init sha256)` (`v8f_dagger.py:321`) and `bad_dagger` `:409+`. **FLAT-DAG on a bank generated by M2b_s{s} is REFUSED by the real gate** ("K-B2: the bank was generated by checkpoint 601cb4352673, the arm
fine-tunes 8496934a2ce2: the states are not on-policy for this init"), s0 and s1. So the composition gate ran on a bank the real generator made from FLAT_s (below). The iteration-1 banks (M2b-generated) cannot even gate FLAT-DAG.

### (b) The bank generator accepts FLAT

* `v8f_dagger_bank.train_list()` parses FLAT's REAL banked argv.json as `--m2b-argv` (it carries `--o-interface cut`; `ns.o_interface == "cut"`, `ns.synthetic` False).
* `v8f_dagger_bank.load_ckpt()` builds the FLAT checkpoint with `cfg.o_interface == "cut"` and loads it strictly; the trainer's own `o_interface_probe`, run on THAT model, passes (traj60 invariant to S and T): **the bank is rolled out by the cut policy**.
* The queue's own bank tool (2 shards + merge) runs with `--ckpt FLAT_s` and an `--m2b-argv` carrying `--o-interface cut`: rc 0; `bank.json common.ckpt_sha256` = FLAT_s's sha256 (C-F1), K-B4 / K-B5 PASS, not partial, COVERAGE 0.646 (s0) on the synthetic world.
* ⚠ On the REAL 4,512 TRAIN clips the FLAT bank's COVERAGE is UNVERIFIED (BLOCKED is a pre-registered outcome).

## F1 — the staged fix

`stage/stack/scripts/v8f_mini_train.py` (= the tip file + one key in `INIT_CKPT_MATCH`, 642 bytes added incl. the comment; LF; diff against the tip is that one hunk), `stage/stack/tests/test_v8f_init_ckpt_o_interface.py` (7 tests).

| | tip trainer | staged trainer |
|---|---|---|
| cut checkpoint + argv without the cut | **accepted** (the hole) | refused: `init-ckpt … o_interface: 'cut' != 'full'` |
| full checkpoint + `--o-interface cut` | **accepted** | refused |
| cut ckpt + cut argv; full ckpt + default argv | accepted | accepted |
| a checkpoint without the key (every M2b) under either argv | accepted | accepted (unchanged) |
| `INIT_CKPT_MATCH` names `o_interface` once; `w_acc` NOT in it | fails | passes |
| **result** | **3 failed, 4 passed** (`logs/pytest_nofix.log`) | **7 passed** (`logs/pytest_fixed.log`) |

`w_acc` deliberately stays OUT of the tuple: DAG-OACC fine-tunes M2b (no `w_acc` / 0.0) with `--w-acc > 0`. `code_mix`, `cons_absent`, `window_start` are also absent from the tuple; not changed (scope).
Regression of the staged trainer on the three repo test files that use `--init-ckpt`: see "Regression" below.

## F2 — the loop line is not covered by the gate

Mutant `loop_dag_apply` (delete `b = dag.apply(b, step)` at `:1605`) leaves Arm 1's gate **GREEN** (24/24, rc 0): `dagger_probe` calls `dag.apply` itself, so it cannot see the loop's call.
The repo's tests do: `test_the_single_applying_lines_exist_exactly_once`, `test_a_real_run_gives_the_loss_the_swapped_batch…`, `test_the_check_that_the_loss_received_swapped_tensors_can_see_a_missing_apply`, `test_the_swap_is_what_changes_the_training`
— re-run on the tip here: **4 passed** (`logs/pytest_tip_loop.log`). No fix staged (it needs a training step to prove, which this task excludes); the protection for the two new arms is those tests plus the run's own share record.

## Mutations (in-memory, one line each; the file on disk is never touched; each mutation applied exactly once or the run aborts) — `run_mutants.py`, `out/MUTANTS.json`

| mutant (deleted line) | caught by | result |
|---|---|---|
| `total = total + cfg.w_acc * T["acc"] …` (`v8f_mini.py:888`) | the REAL gate's `acc_probe`: "the total moved by 0.0 when the weight went from 0 to 49.6969" | **RED** |
| `out.update(upd)` (`v8f_dagger.py:685`), Arm 1 and Arm 2 | the REAL gate's `dagger_probe`: "the swapped kin3 … is not the bank record's own kin3" | **RED**, **RED** |
| `ck_for_o = self.ck_nom…` (`v8f_mini.py:765`), the cut's checkpoint half | the REAL gate's `o_interface_probe`: "cut is declared but O was handed a non-constant" | **RED** |
| `code = torch.zeros_like(code)` (`v8f_mini.py:767`), the cut's code half | the REAL gate's G-LIVE: `e_code.weight` is no longer dead | **RED** |
| `b = dag.apply(b, step)` (`v8f_mini_train.py:1605`), the loop line | nothing in the gate | **GREEN (F2)** |

## Regression of the staged trainer

Staged trainer vs the three repo test files that use `--init-ckpt` (`test_v8f_dagger_train.py`, `test_v8f_ego_perturb.py`, `test_v8f_orc_chain.py`), run through `run_pytest_staged.py` (the staged module is imported first as `v8f_mini_train`):
**212 passed, 10 skipped, 2 failed**, 503 s (`logs/pytest_staged_regress.log`). The 2 failures — `test_v8f_orc_chain.py::test_queue_is_crlf_every_older_label_is_byte_identical_and_only_the_four_orc_labels_are_added` and
`::test_queue_orc_stages_are_one_flat_command_string_each_that_the_dry_run_prints_and_the_real_run_executes` — **fail identically on the UNMODIFIED tip trainer** (re-run: 2 failed, 0.5 s). They compare the queue `.cmd` with the ORC-era
label set ("only the four ORC labels are added"); the DAG stages (`dagarm`, `bankdag`, `evaldag`, …) landed after them. **F3: a RED pair already on the tip, unrelated to this change** (it concerns the queue file, which another agent owns).
The full suite (`pytest -q`) was NOT run.

## MEASURED vs UNVERIFIED

**MEASURED** (ours; artifact paths under `E:/Projects/TanitAD-artifacts/oaccflat_check/`): everything in the tables above, from `out/*.json`.

**UNVERIFIED — do not quote as measured:**
* Any property on the REAL corpus: real probe batches, the real iteration-1 bank (`v8f-dagger/bank_s{0,1}`) under DAG-OACC, the real FLAT bank's K-B0…K-B5 and COVERAGE ≥ 0.5 at 4,512 clips, cuda numerics. The box-2 stage's own gate (`gate_*` stages) is the real test; this note only says the code path composes.
* The DAG-OACC records (`argv.json`, `summary.json`, `selfcheck.json`) of a real run, and the reader that will consume them: the pair test used DAG's real records with the DAG-OACC fields substituted.
* That the box-2 `waccoacc` stage writes 49.6969 (offline recomputation `dagoacc_dry/wacc_offline_DAG_s0.json`: 49.6969, INHERITED, not re-run here). The check here is that the trainer accepts and uses whatever value the argv carries.
* The queue (`v8f_mini_box2_queue.cmd`): not read for FLAT-DAG / DAG-OACC stages (another agent owns it); whether its stages name `--o-interface cut`, `--w-acc` and all DAG levers is unchecked.
* F2's protection on the real run (the share reads in `summary.json`) — the DAG-OACC / FLAT-DAG reader code does not exist yet.
* Seeds beyond {0, 1}; any behaviour of a trained DAG-OACC / FLAT-DAG model (this is a launch-gate composition check, not a result).

## Re-run

```
T=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a
export PYTHONPATH="$T/stack;$T/taniteval;$T/stack/scripts" CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
cd E:/Projects/TanitAD-artifacts/oaccflat_check && ./final_all.sh        # arm1 s0,s1 · pair records · arm2 s0,s1 · all mutants  (~10 min, one torch process at a time)
```
