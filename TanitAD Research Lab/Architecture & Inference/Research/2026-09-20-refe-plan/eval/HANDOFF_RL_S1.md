# HANDOFF RL-S1 part 1 (2026-10-12) -- priorities 2 and 3 are NOT built; read this, then continue

Ended at the ~200k-context rule (PI token budget), not because of a blocker. Priority 1 (reward) and 4 (prereg draft) are banked.
D3 is DECIDED (R2 in, Master Mind 5e4a998): K and L are built and tested to the core standard; the `refe_rls1` profile must accept the R2 argv with them ON.

## Done (all under `PKG = TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`)
* `refe/rl_reward.py` -- pure numpy; `reward(tg, cfg, d=, k=, lane=)`; `RewardCfg` (lam_d .15, w_prog .5, guard ON, K/L flags OFF). Missing input for an ON term RAISES.
* `refe/test_rl_reward.py` -- `ZZTEST_PASS 31`: hand-computed values for PDMS, D, g2, K, L (+ the section 3.3 guards), 29 source mutations all RED. Run:
  `PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe refe/test_rl_reward.py`
* `eval/PREREG_RL_S1.md` -- DRAFT; OPEN-BASE has both variants side by side; D3 marked decided.
* Baseline to preserve: `refe/test_launch_gate.py --static` = **76** (`--no-banked`) / **90** (with the banked generic gate). Re-run before and after every gate edit.

* **P4 ruling (Master Mind, 2026-10-12, after this handoff was written):** the progress guard is capped by `rl_reward.guard_penalty` so it can never push a
  safe proposal to or below an unsafe one it beat (the uncapped `w_prog = 0.5` did). `w_prog` stays 0.5. `reward()` now also returns `out["guard"]`.

## Findings that change the build (each read from source, file:line at the tip)
1. **Selection-rule conflict.** Review 5.2 says `A_i = NC*DAC*DDC*(5EP+5TTC+4C)/14` (train.py:611-614) = rule `v2_shape`. u400 deploys **`navsim_v1_lane`**
   (`planner.py:552-557`, `RESULT_SFT4.md` l.23: NC*DAC*(5EP+5TTC+2C)/12 * p6); P2 will deploy `navsim_v1`. The EPO policy must use the BASE's rule:
   reuse `scorer_finetune.sel_score` (it already does lane-head x p6). Do not use `_pdm_aggregate`.
2. **`scorer_finetune.py` cannot start from u400.** It builds `REFeConfig.for_backbone(a.backbone)` (6 outputs) and strict-loads (L161-164) without
   `ckpt_io.config_for_checkpoint(cfg, a.ckpt)`; and `--lane-head` (L177-188) copies `old.weight` (7 rows) into `new.weight[:6]`. Fix additively: size the cfg
   from the checkpoint, skip the head-extension when the head already has 7 rows. Default-preserving (meta absent -> unchanged).
3. **The existing `--b-mode expreward` already optimises E_pi[true v1 score]** (SFT-3/SFT-4 recipe, `expected_reward_loss`, L68-75, tau_s .1). The new EPO differs by:
   reward R (rl_reward, not agg_v1(labels)), the KL anchor to `pi_SFT` (`base` ScorerCopy already gives sx0 each micro-batch), the 0->1 ramp over 200 updates.
4. **Arms A/B of `scorer_finetune` are paired by construction**: A = BCE only (= C_s), B = BCE + EPO (= R_s). EPO goes on B only; `--epo-lam 0` must make B == A (K1).
5. **Gate coupling.** `DVB_KINDS` is pinned equal to train.py's parser dests by `test_launch_gate.py` S5 (`set(C.DVB_KINDS)==dests`, `len(dests)==55`): **do not add scorer flags
   to it.** Add a separate `DVB_KINDS_SCORER` + `check_dvb_scorer` (unknown flag FAILS, stale entry FAILS, same two arms `unknown_flag`/`stale_registry`). `profile_refusals`
   hard-codes `import train as T` / `train.py` (launch_gate.py): give profiles `parser_module` + `trainer_src` keys, default = today's. `scorer_finetune` has no
   `build_parser()` (parser inline, L106-145): extract it VERBATIM. `instrumented_run`/G-LIVE/G-EVAL/G-CKPT are train.py-only: for `refe_rls1` consume
   `scorer_finetune --preflight`'s JSON like `consume_p2_gate` consumes p2_gate.py (parsed, never grepped; file sha256 bound into the token).
6. **Batch has no row key** (`TargetBank.__getitem__` returns `ci` = per-proposal ckpt_step; the row index only when `online`). To look up D/K/L side files per sample,
   wrap `Subset` so it also yields the row index, ONLY when `--epo` is on (OFF path unchanged => bit-identical).
7. Bash heredocs in this environment eat a backslash level: write with Write/Edit.

## Next steps, in order (bank each)
1. `scorer_finetune.py`, additive, default OFF: `build_parser()` extraction; item 2 fix; flags `--epo`, `--epo-temp .1`, `--epo-beta .1`, `--epo-ramp 200`, `--epo-lam 1.0`,
   `--epo-lam-d .15`, `--epo-w-prog .5`, `--epo-k` + `--epo-lam-k .05`, `--epo-l` + `--epo-lam-l .05`, `--epo-terms-train/-heldout <dir>` (D/K/L side files,
   schema `{"kind":"rl_terms","key":[log,token,step,rank],"ckpt_step":c,"d":[64],"k":[64],"lc_executed":[64],"label_fires":bool,"dither":[64]}` -- part-2 labeller writes it).
   `epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg)` module-level, torch, reward from `rl_reward` as a detached tensor. Refuse `--epo` without `--epo-terms-train` unless `--epo-lam-d 0` is NAMED.
2. Preflight gate `G11_epo_live_and_identity`: EPO-only backward non-zero on each scorer module; `--epo` off / lam 0 reads EXACTLY 0.0; `REFE_RL_MUTATE_EPO_DETACH=1` -> RED.
3. `refe/test_rl_epo.py` (CPU, tiny scorer modules): OFF path == the verbatim baseline closures (embed them; blob `scorer_finetune.py` = 92b52dac) on loss AND on a multi-step param hash across b-modes
   listnet/compw/expreward +- lane head; ON: non-zero grad, OFF control exactly 0.0; mutations (delete the EPO line; `--epo` default flipped; detach) RED.
   Dynamic reachability: AST-extract `losses6` from `main()` and exec the SOURCE text (so a mutation of the real caller is seen).
4. `refe_rls1` profile + `DVB_KINDS_SCORER` + tests in `test_launch_gate.py` (static): R0/R1 argv clean, R2 argv (K/L ON) clean, unknown flag FAIL, missing required flag refused, count >= 76/90.

## Part 2 (not started, not in any brief yet): on-policy proposal dumper (~51,000 scenes), extended labeller (D, K, L -> `rl_terms` side files; G-REWARD-ID `g>0 iff DAC=0` on 10,000 labelled proposals),
RL-train / RL-val log-disjoint partition, pod chain script, RL-val lookup evaluator. Also OPEN: base checkpoint (PREREG_RL_S1 OPEN-BASE).

---

# 2026-10-12 -- RL-S1 build, steps 1-3 of "Next steps" DONE (scorer_finetune.py + test_rl_epo.py); step 4 (`refe_rls1` profile) is the NEXT agent's

## What is built (all in `refe/`, all default OFF; baseline = blob 92b52dac, verified byte-equal to the disk file BEFORE editing)
* `scorer_finetune.py`: `build_parser()` extracted VERBATIM (38 lines moved by script, not retyped); u400 fix = `cfg = ckpt_io.config_for_checkpoint(...)` + `extend_lane_head()` skips a head that already
  has 7 rows (and consumes no RNG); `native7` flag so preflight G3/G5/restore treat a trained 7-row head as an ordinary 7-wide base; a 7-row ckpt without `--lane-head` is REFUSED.
  EPO: module-level `epo_loss(sxB, sx_ref, R, rule_fn, T, beta, lam_pg, parts=None)`, `epo_policy`, `epo_ramp`, `epo_reward` (-> `rl_reward.reward`), `load_rl_terms`, `epo_refusals(a, ap)`,
  `IndexedSubset`/`make_train_set`/`split_rows` (row index rides the batch ONLY with `--epo`), `g11_epo_live_and_identity`; in main(): closures `epo_term`, `train_losses` (THE loss assembly: loop, speed
  probe and G11 all call it), `epo_probe`. Flags exactly as listed in "Next steps" 1. Policy = softmax(log(A + 1e-6)/T) with A = main()'s own `sel_score` (navsim_v1 x lane output), never `_pdm_aggregate`.
* Preflight gate `G11_epo_live_and_identity` (only with `--epo`): EPO-only backward non-zero + finite on EACH of score_q_mlp/score_dec/score_head of arm B; controls `lam0` (graph live, weight 0.0) and `off`
  read EXACTLY 0.0; KL(pi_SFT||pi) at step 0 = 0.0 (u400/6-col) or <= 1e-2 (freshly extended lane head); train rows resolve in the terms; held-out terms matched > 0 when given.
  `REFE_RL_MUTATE_EPO_DETACH=1` -> RED and no PREFLIGHT_PASS.json is written.
* `test_rl_epo.py`: `ZZTEST_PASS 42` = unmutated suite + 41 source mutations all RED. Run: `PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe refe/test_rl_epo.py` (~5 min, CPU).
  It proves: (a) OFF path == the embedded baseline closures, AST-equal AND bit-identical (loss bits + 3-update AdamW parameter hash) across b-mode listnet/compw/expreward x lane head +/-;
  (b) epo_loss against hand-computed values (J, KL 0.41903392 and its reverse, gradient, ramp); (c) every hop flag -> `.backward()` with the real caller TEXT exec'd (the train-loop body, train_losses, epo_term,
  epo_probe are AST-extracted from main()); (d) the REAL `main()` end to end on CPU with stubbed REFe/banks/ckpt-loader: preflight (G1-G11) -> PASS json -> train, 6-col and native-7+lane-head, OFF, `--epo-lam 0`
  (arm B == arm A bitwise at the END of a run = K1 at main level), and EPO never touches arm A (A bitwise equal to the OFF run's A).

## Findings that change the plan (read before step 4 / part 2)
1. **The existing G4 gate would have failed EVERY `--lam 0` run** (`float(bB) != float(bA)` demands the ranking term be live). Arm B must be BCE-only + EPO, i.e. `--b-mode listnet --lam 0`. G4 now accepts
   `lam == 0` (identity still asserted); for lam > 0 it is unchanged. Pinned by a mutation.
2. `DataLoader` calls `Subset.__getitems__`, which bypasses an overridden `__getitem__` -- `IndexedSubset` overrides both (a mutation without the second goes RED on a real DataLoader).
3. `scorer_finetune.py` is UNTRACKED in this D: index (blob 92b52dac exists in the object store; HEAD has no entry) -- `git add` stages it as a NEW file; the lander must land it as a modification of 92b52dac on the mirror tip.

## DESIGN CALLS I MADE that the Master Mind may overrule (each is one `epo_refusals` line or one constant)
* `--epo` is REFUSED unless `--pdm-labels-train` AND `--require-pdm` (R is built from PDM sub-scores; no set may mix label sources) and unless `--b-mode listnet --lam 0` (R vs C must differ by the EPO term alone).
* EPO levers named WITHOUT `--epo` are REFUSED (a flag whose default disables it must be named in the arm's argv).
* `log(A + 1e-6)` as the review writes it, not `clamp_min(1e-6)` as `expected_reward_loss` does (the clamp zeroes the gradient exactly where a proposal needs raising).
* `--epo-terms-heldout` is loaded, validated and logged (and G11 requires >= 1 matched set), but `evaluate()` does NOT yet compute R-at-pick: that is the part-2 RL-val lookup evaluator.
* Side-file name convention (part-2 labeller must write it): `<dir>/rl_terms_*.jsonl`, schema as in "Next steps" 1; a line counts only for the set it was computed on (same key AND ckpt_step, OnPolicyBank's rule);
  d must be [64] in [0,1]; k [64] >= 0; lc_executed [64] bool; dither [64]; label_fires JSON bool. The training index is restricted to sets that have a record.

## Step 4 (NOT done): what the `refe_rls1` profile can rely on
* `scorer_finetune.build_parser()` (import has side effects: torch, train, model); `scorer_finetune.epo_refusals(a, ap)` returns the refusal messages for an argv -- the profile can call it for R0/R1/R2.
* New dests for `DVB_KINDS_SCORER`: epo, epo_temp, epo_beta, epo_ramp, epo_lam, epo_lam_d, epo_w_prog, epo_k, epo_lam_k, epo_l, epo_lam_l, epo_terms_train, epo_terms_heldout (EPO_DESTS = all but `epo`).
* `--preflight` writes `preflight.jsonl` (last `event=preflight` row, `gates` -> `G11_epo_live_and_identity`) and `PREFLIGHT_PASS.json` ONLY if every gate is ok; parse, never grep. R2 argv = R0 + `--epo-k --epo-l`
  (both need `--epo-terms-train` with k / lc_executed / label_fires / dither present). R0/R1 argv needs `--epo-ramp 200` etc. only if they differ from the defaults (defaults ARE the registered values).
* u400 argv shape that the CPU smoke passes (stubs): `--lane-head --teacher-lane-train <d> --teacher-lane-heldout <d> --pdm-labels-train <d> --pdm-labels-heldout <d> --require-pdm --b-mode listnet --lam 0 --epo --epo-terms-train <d>`.
* NOT validated on a real checkpoint: the stubs replace REFe/OnPolicyBank/TargetBank/ckpt_io's loader, so the first real preflight (A40 smoke) is still the final check of the u400 strict load.

## Still open (unchanged): part 2 (dumper, extended labeller writing `rl_terms_*`, G-REWARD-ID, RL-train/RL-val partition, chain script, RL-val lookup evaluator) and OPEN-BASE.

---

# 2026-10-12 -- RL-S1 part 2, slice S1: the clean pool MEASURED, partition tool, RL-val evaluator, dump fitness check (nothing launched, no pod touched)

## Pool numbers (evidence in `raw/rl_pool/rl_pool_by_base.json`; instrument `eval/rl_pool_measure.py`; paragraph appended to `PREREG_RL_S1.md`)
Eligible population [M]: navtrain.yaml 103,288 tokens / 1,192 logs, 0 in navtest, minus the 24 SFT held-out logs = **101,583 tokens**.

| | trained on (tokens / sets) | UNSEEN tokens (eligible) | vs the 51,000 planned |
|---|---|---|---|
| **B: P2** (3 epochs, all served train sets) | 99,495 / 174,912 [M] | **2,088** (1,837 placeable: 251 have no recoverable log), >= 632 logs | **4 %**; below the 3,000 RL-val alone needs |
| **A: u400** (update 400 x 64 = 25,600 sets) | at most 25,600 tokens [M bound] | **>= 75,983 certain** (73,895 + 2,088), ~77.5k expected [E]; **members NOT identifiable** | exceeds 51,000 in count only |
| both, strict lineage (model_final's scorer, main run) | fixed bank epochs 1-11 + on-policy epochs 12-27 | at most 8,531 escaped the on-policy phase | neither is clean |
| no on-policy set at all | - | 2,126 tokens (2,088 eligible) | need a dump in either variant |

* The banked bank reproduces the carve exactly (177,836 served = 174,912 + 2,924), so **B's trained-key list needs no pod fetch** (`raw/rl_pool/served_keys.tsv.gz`).
* **A's draw is unrecoverable**, not merely unlocated: it is `randperm` seed 0 over the launch-time ordered index (n between 149,377 and 149,440). The only files that could rebuild it are on the SFT-4 pod, which is gone: `/workspace/data/refe_sft4/sft.jsonl` (the `data` event) and `/workspace/data/refe_sft2/train_pdm/pdm_*.jsonl` as they stood at launch. The P2 pod's recomputed labels are a different set at a different moment and cannot stand in. The review's "28,727 labels not written at arming" are SETS: mostly rank-1 views, so only ~3,027 whole tokens [E] had nothing labelled.
* My per-epoch reconstruction of what the main run could train on (from each set's own label time) matches the trainer's own `onpolicy_sets` counts within 0.1 % at all 16 epoch boundaries: that is the validation of the reconstruction, not of the reading below.
* **Reading (mine, ESTIMATED argument, not a measurement):** pool size decides against B. A's contamination (~24 % of a random scene's tokens touched by u400) hits C and R identically; the contrast R - C needs only RL-train / RL-val log-disjointness. Both bases share the model_final exposure.

## Built (all staged; none of the do-not-edit files touched)
| file | what | check |
|---|---|---|
| `refe/rl_partition.py` | deterministic LOG-disjoint RL-train / RL-val split: drops navtest tokens+logs, held-out logs, the base's trained tokens, unknown-log scenes; logs ordered by sha256(seed\|log); RL-val capped at n_val // min_val_logs scenes per log (3000 // 60 = 50) so it spans >= 60 logs; train = whole logs after; short pool -> exit 3, nothing written (`--allow-short`); `verify()` re-checks every invariant before writing (exit 5) | `refe/test_rl_partition.py`: **ZZTEST_PASS 20** = 1 + 18 source mutations, all RED (log split, held-out/navtest/trained admitted, urandom seed, seed ignored, input-order, cap removed, no trim, silent shortfall, blind verify, wrong manifest hash, ...) |
| `eval/rl_val_eval.py` | RL-val LOOKUP: scores + `pdm_*.jsonl` + `rl_terms_*.jsonl` (+ `onpolicy_*.jsonl` for path length) -> PDMS and six sub-scores of the pick, R and D at the pick, skill, zero picks, slow-pick telemetry (pick/fan-mean path, median length rank, shorter-than-median share, quintile shares, shortest-bin share) with the review's g3 bounds, g5 and K4 flags; pairing = `proxy_eval.boot` (the stage-3 estimator); every run carries the self-pair == exactly 0 and arm<->base swap checks | `eval/test_rl_val_eval.py`: **ZZTEST_PASS 31** = 1 + 29 RED mutations; values hand-derived with exact fractions |
| `eval/rl_pool_measure.py` | the pool instrument (needs Python >= 3.14 for `compression.zstd`) | regenerates `raw/rl_pool/*`; reproduces 177,836 / 174,912 / 2,924 |
| `eval/rl_ckpt_equality.py` | ckpt_last@10075 vs model_final.pt | 276/276 tensors bit-identical, equal frozen fingerprints: the 22,742 step-10,075 served sets ARE the frozen fan |
| `raw/rl_pool/` | `rl_pool_by_base.json`, `navtrain_token_table.tsv.gz` (token -> log, the partition's pool), `served_keys.tsv.gz` (P2's trained keys), `dump_fitness.json`, `ckpt_last_vs_model_final.json`, `partition_runs/` | manifests of two real runs: A (no list) 48,000 train / 3,000 val over 576 / 75 logs; B refused (val 1,837/3,000) |

## Findings that change the plan
1. **`scorer_finetune.py` writes NO per-proposal scores** (its `evaluate()` aggregates in-process). The evaluator's input is therefore new: `rl_val_eval.write_scores_npz(path, keys, ckpt_steps, scores[N,M,C], space)`. Add `--dump-scores DIR` to `evaluate()` (a few lines at the `sx0/sxA/sxB` site) before any RL-val read; pass `--epo-terms-heldout` through as `--terms`. NOT built (scorer_finetune is the EPO owner's file).
2. **The dumper cannot do the frozen-fan re-dump as shipped** (`raw/rl_pool/dump_fitness.json`): F1 no key list / shard, F2 checkpoint path+step hard-wired to a live run dir (a FULL model_final.pt has no `state`; pointing `--run-dir` at the finished run's `ckpt_last.pt` works today), **F3 it writes `onpolicy_props`, the two labellers read only `onpolicy_set`** (only `onpolicy_label.py`, the teacher simulator, produces those: **10.2 s per set, 144 worker-hours for 51,000, AVOID**), F4 `train.OnPolicyBank` needs a `targets` field (a lean writer can emit `[{}]*64`; PDM labels overwrite all six components; untested on the train side), F5 dump numerics (bf16/TF32) vs the eval planner's are unmeasured, F6 images and DBs are pod-only.
3. **Cost (E)**: dump 3.28 h for 51,000 (0.2313 s per sample, M), 2.40 h if the 13,713 plain final-fan sets already in the bundle are reused; labels 44 CPU-h (PDM 1.78 + rl_terms 1.34 CPU-s per set) = 1.8 h on 24 vCPU, 4.0 h on 11. The 13,713 are a contiguous slice of a seeded permutation, so reusing them introduces no selection by token (argued from source).
4. The 251 yaml tokens with no target-bank row have no log and no images: they cannot be RL scenes in any variant.

## Next (nothing here is launched)
1. Master Mind: decide OPEN-BASE with the table above; freeze the partition seed/params in the prereg (default seed 20261012 [H]).
2. Build, in this order: dumper edits F1-F4 + lean `onpolicy_set` writer + test; `--dump-scores`; the G11-style gate and chain script (`refe_rls1`, item 4 of the part-1 list is still open).
3. Pod, after P2: F5 probe (20 navtest tokens), F4 smoke (`scorer_finetune --preflight` on a lean set), then dump + labels.


---

# 2026-10-12 -- RL-S1 step 4 DONE: the `refe_rls1` launch-gate profile (`gate_checks.py`, `launch_gate.py`, `test_launch_gate.py`)

Scope kept: `scorer_finetune.py`, `rl_reward.py` and `rl_terms*` were NOT touched. Every existing profile is unchanged -- proved by their own suites (numbers below), not asserted.

## Numbers (MEASURED, dev box, CPU, before my first edit and after my last)
| suite | before | after |
|---|---|---|
| `test_launch_gate.py --static` (with the banked generic gate) | **90** | **316** |
| `test_launch_gate.py --static --no-banked` (what the gate's own G-SUITE runs) | **76** | **302** |
| `test_p6_levers.py --fast` / `test_p6_chain.py` / `test_p2r_chain.py` | 123 / 204 / 139 | 123 / 204 / 139 |
| source-mutation sweep of the NEW code (47 mutations in `launch_gate.py` + `gate_checks.py`, on a copy of `refe/`, `--static --no-banked`) | -- | **47 / 47 RED** (the first sweep caught 2 GREEN: `check_hyg` silently reading train.py's functions, and the CLI not handing `--gate-preflight-json` to `run_checks`/`finalize`; both now have tests, +3 checks) |
| `test_launch_gate.py` FULL (real trainer rehearsal, all regression arms) | -- | **344** (ZZLG_FULL_PASS, rc 0) on the code BEFORE the last two edits (one evidence-detail line in `check_dvb_scorer`, +3 test checks); **NOT re-run on the final bytes -- UNVERIFIED** |

**Verification gap, stated plainly:** after those last edits the machine ran out of memory and C: space (an unrelated `ladder_eval_diag.py`, PID 45724, held 10-11 GB; `WinError 1455` / `ENOSPC` / OpenBLAS allocation failures in every torch process), so `test_p6_levers.py --fast` (123), `test_p6_chain.py` (204) and the FULL run above were NOT re-run on the final bytes. What WAS re-run on the final bytes and passed: `--static` 316, `--static --no-banked` 302, `test_p2r_chain.py` 139, and the 47-mutation sweep. The last edits cannot reach the P6 paths (they touch only `check_dvb_scorer`, a docstring and the tests), but a fresh agent should re-run those three once memory is free:
`PYTHONIOENCODING=utf-8 C:/Users/Admin/venvs/tanitad/Scripts/python.exe refe/test_p6_levers.py --fast; ... refe/test_p6_chain.py; ... refe/test_launch_gate.py`. Run them ONE AT A TIME (six at once exhausted the pagefile here).

## What is built
* `launch_gate.py`: profiles carry `parser_module` + `trainer_src` (`_BASE` = `train` / `train.py`; `profile_refusals` reads them, default-preserving). New profile **`refe_rls1`** (`runner: scorer`, `arm_rules: rls1`,
  trainer `refe/scorer_finetune.py`, checks `G-HYG G-DVB G-PREFLIGHT G-SUITE`, `G-CLOCK` NA as P2). `rls1_refusals` = the arm rules; `rls1_arm_class`, `rls1_control_argv` (the matched control of an R argv: the same argv minus every EPO flag);
  `consume_scorer_preflight` (G-PREFLIGHT); `run_checks` / `finalize` / `gate_inputs_of` gain `preflight_json` (additive, last); CLI: `--arm-name` (run, refusals), `--gate-preflight-json` (run). `run` REFUSES to mint a profile with arm rules
  without `--arm-name`; `refusals` without it reads the arm off the argv.
* `gate_checks.py`: `DVB_KINDS_SCORER` (46 dests: 15 data, 13 built, 14 consumed, 3 run, 1 control) -- a table SEPARATE from `DVB_KINDS` (S5 still pins that one equal to train.py's 55; I did not add one scorer flag to it);
  `check_dvb_scorer` with regression arms `unknown_flag`, `stale_registry` (the two asked for) and `dead_lever`, `unbound_data`, `lever_ignored`; `load_scorer_preflight`, `parse_scorer_argv`, `scorer_required_gates`, `scorer_dvb_rows`;
  `check_hyg(profile=)` and `namespace_reads(src, funcs)` (default = today's behaviour).
* `eval/REFE_LAUNCH_GATE.md` section 4: one bullet for `refe_rls1`.

## The exact argv per arm that passes the static gate (`launch_gate.py refusals --profile refe_rls1 --arm-name <ARM> --args "..."` -> `ZZGATE-CLEANZZ`)
Common to EVERY arm (the control must see the same labels, loss, optimiser and step budget as the EPO arm, or R vs C is not the EPO term alone):
```
--ckpt <base.pt> --backbone vitl16 --targets <train targets> --images <pix> --calib <calib.json> --onpolicy <RL-train sets> --heldout <RL-val sets> --heldout-logs <RL-val logs.txt> --out <arm dir>
--pdm-labels-train <dir> --pdm-labels-heldout <dir> --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3 --seed <S>
--lane-head --teacher-lane-train <dir> --teacher-lane-heldout <dir>        # variant A (u400, 7 outputs). Variant B (P2 model_final, 6 outputs): omit these three
```
* **C0 / C1** (control, BCE only): the common argv, `--seed 0 / 1`. Carries NO `--epo*` flag (not even one named at its default).
* **R0 / R1** (core reward): common + `--epo --epo-terms-train <dir>` (+ optionally `--epo-terms-heldout <dir>`), `--seed 0 / 1`. K / L flags are REFUSED here.
* **R2**: R0's argv + `--epo-k --epo-l`, `--seed 0`.
Pinned values (PREREG_RL_S1 DRAFT section 1; the EFFECTIVE value, named or the parser default, must equal them -- so `RLS1_EPO_REGISTERED` is also pinned to equal the parser defaults): `--epo-temp .1 --epo-beta .1 --epo-ramp 200 --epo-lam 1.0
--epo-lam-d .15 --epo-w-prog .5` (R2 also `--epo-lam-k .05 --epo-lam-l .05`); recipe `--lr 1e-4 --warmup 50 --batch 8 --accum 8 --max-updates 1000`. Forbidden (named-but-inert, or a second label source): `--preflight --lane-labels-train
--lane-labels-heldout --b-compw --tau-s --tau-t`. The chain order is: `scorer_finetune ... --preflight` into `<arm dir>` (also what unlocks the trainer's own `PREFLIGHT_PASS.json` check) -> `launch_gate.py run --profile refe_rls1
--arm-name <ARM> ... --gate-preflight-json <arm dir>/preflight.jsonl` -> launch -> `verify`.

## DESIGN CALLS I MADE (the Master Mind may overrule; each is one constant or one `if` in `launch_gate.py`)
1. **The recipe and lever values are pinned from a DRAFT prereg.** Change them in `RLS1_RECIPE` / `RLS1_EPO_REGISTERED` / `RLS1_R2_REGISTERED` / `RLS1_ARM_SEEDS` -- nowhere else. `--epochs` is deliberately NOT pinned (see "left", item 2).
2. **The arm is declared at mint** (`--arm-name`), because an arm read off the argv cannot notice an omitted flag (an `--epo` left out of R0 reads as a clean control). The arm name is not in the token; the arm CLASS is a pure function of the
   bound argv, and the declaration is written to nothing but the refusal logic.
3. **Forbidden-as-inert flags** (`RLS1_FORBIDDEN`) and **`--lane-head` <-> `--teacher-lane-*` must pair**; `--epo-lam-k/-l` without `--epo-k/-l` is refused (scorer_finetune would silently ignore it).
4. **Exact flag spelling**: argparse accepts a unique prefix (`--require-p` parses as `--require-pdm`); every gate flag test is on exact tokens, so the profile refuses an abbreviation.
5. `--epo-terms-heldout` is OPTIONAL (the RL-val lookup evaluator that consumes it is part 2).
6. **`G-PREFLIGHT` wants `preflight.jsonl`**: its `data` event (b_mode, and the whole `epo` block) is what binds the preflight to this argv's levers; `PREFLIGHT_PASS.json` carries the preflight event alone and is refused by G-DVB ("no preflight `data` event").
   Without that binding any `PREFLIGHT_PASS.json` in `--out` unlocks any argv, which is how the trainer behaves today.
7. G11 is **re-judged from its own fields** (EPO-only gradient finite and non-zero on all three scorer modules, both controls exactly 0.0, KL at step 0, `mutate_detach` False), not trusted on its `ok`. Every other gate: `ok is True`.

## Findings (from reading source; not fixed -- not mine)
* **There is no G6.** The trainer numbers its preflight gates 1-5 and 7-11 (`gates["G..."]` in `scorer_finetune.py`); "every G1-G11" means those ten. The consumer's table is pinned to the source by a test, so a new G12 FAILS the gate until the consumer learns it.
* **`scorer_finetune.epo_refusals` cannot see an EPO lever named AT its default value without `--epo`** (it compares value != default): `--epo-temp 0.1` alone passes it. The profile's token test ("a control must not carry any `--epo*` flag") closes it.
* **A separate C run is NOT set-identical to its R run.** The training index is restricted to sets that have an `rl_terms` record only inside `if a.epo:` (`tr_idx = [... in terms_train]`); a C run (no `--epo`, and `--epo-terms-train` without `--epo` is refused) trains on EVERY PDM-labelled set. They match only if the part-2 labeller writes terms for every PDM-labelled
  train set. The paired pass has no such risk: arm A of the R run IS the matched control (same sets, order, steps, trunk forward). The gate cannot see this (it judges one argv).
* The preflight log does not record the planned update count, so `--max-updates 1000` binding (1,000 updates needs `--epochs >= 1.34` at 48,000 sets, batch 8 x accum 8) is not machine-checked; the default `--epochs 1.0` would silently end at 750.

## Real-shape proof
The unit fixtures are built from the trainer's own gate-key spellings, but the strongest check ran the REAL `main()` (test_rl_epo's CPU stub harness) end to end for `--epo` 6-column, `--epo` native-7 + `--lane-head`, and OFF, and fed the resulting `preflight.jsonl` to
`consume_scorer_preflight` and `check_dvb_scorer`: PASS / PASS on all three (13 observed lever rows with `--epo`, 3 without; 8 / 9 / 7 gates). Scratch only (`scratchpad/real_shape.py`, `real_consume.py`); not banked, because it needs the 5-minute harness.

## What is left (in priority order)
1. **Land before anything can be minted.** `require_commit` is on and `ship-manifest` refuses a file that is not its blob at the commit: `scorer_finetune.py` (staged as a NEW file on this index; HEAD has no entry), `rl_reward.py`, `test_rl_reward.py`, `test_rl_epo.py` and the three gate files must be on the mirror tip. The G-SUITE of `refe_rls1` also runs `test_rl_epo.py` (~5 min CPU) on the pod.
2. **Part 2** (unchanged): dumper, `rl_terms` labeller + G-REWARD-ID, RL-train / RL-val partition, the pod chain script (`rls1_pod.sh`: preflight -> gate `run --arm-name` -> launch -> `verify`), RL-val lookup evaluator, OPEN-BASE. Two small asks for `scorer_finetune.py`'s owner:
   write the planned update count into the preflight row (so the gate can assert 1,000), and log `train_sets` (already in the `data` event) so a cross-arm check can assert C and R trained on the same number of sets.
3. A cross-arm check (C vs R `train_sets` equal) is a pair-level gate; this profile judges one argv.
4. The first real preflight on the A40 is still the final check of the u400 strict load (unchanged from the previous section).


---

# 2026-10-12 -- RL-S1 priorities 1-3 DONE on the CPU side: frozen-fan dumper, image coverage, `--dump-scores` (GPU identity evidence PENDING)

Scope kept: `rl_reward.py`, `rl_terms_label.py`, `rl_partition.py`, `launch_gate.py`, `gate_checks.py`, `test_launch_gate.py` and `onpolicy_dump.py` were NOT touched. Everything below is STAGED, nothing committed or pushed.

## 1. Frozen-fan dumper -- `refe/rl_fan_dump.py` (sibling of `onpolicy_dump.py`, which `code/op_pipeline.sh` still calls)
* Key list (`--keys` token<TAB>log TSVs, checked by sha256 against `--partition-manifest`, which must read `verify == "clean"`), `--shard I/N` by LOG (`crc32(log) % N`, the function `rl_terms_label.shard_of` uses, so dump shard i and label shard i hold the same logs), a FIXED `--ckpt` (full `model_final.pt` or partial `ckpt_last.pt`; `--expect-ckpt-sha256`, `--ckpt-step` must agree with the file), `--ranks 0`, resume, `--plan-only` (no model, no GPU: coverage in seconds), `--compare-sets` (exit 5 on any difference).
* Output: `<out>/onpolicy_rlfan_<tag>_s<i>of<n>.jsonl` = `onpolicy_set` records, key fields FIRST (the PDM labeller's `--served-only` reads only the head 2000 + tail 400 characters of a line), `targets = [{}] * 64`, `label_version 0`, `lean: true`, no timestamps (byte-reproducible).
* Numerics = the old dumper's, line for line (trainer's `TargetBank.__getitem__` inputs, 1 thread, TF32, eval, no_grad, bf16 autocast, batch 4, `calib` only for a non-empty rig, `traj.float().cpu()`, the same two-step rounding).
* **Suite: `refe/test_rl_fan_dump.py` ZZTEST_PASS 28** = 1 unmutated suite of 73 checks + 26 RED mutations. The four required ones are in it: wrong checkpoint loaded, shard overlap (and not-exhaustive, and a split log), dropped proposal rank (and order reversed, size guard removed), and the manifest-hash variants. Consumers in the suite are the REAL `train.OnPolicyBank`, `onpolicy_relabel_pdm.served_keys` and `select_bank_subset.head_fields`.
* **Consumer proof (MEASURED, dev box, CPU): `eval/rl_fan_labeller_proof.py --run-labellers`** ran the two production labellers UNMODIFIED in the navsim venv on 6 lean sets (built from banked step-10,075 sets, because the GPU was busy) and loaded the results through `OnPolicyBank(pdm_only)`, `scorer_finetune.load_rl_terms` and `rl_val_eval.load_pdm/load_terms/load_paths`: ZZLABPROOF_PASS, 6/6 at every hop. (!) It is on lean sets built FROM banked ones; the re-run on genuinely dumped records is part of the pending GPU step.

## 2. Image coverage -- NO re-partition needed; the partition_id stays `272cfd35...`
`raw/rl_fan/probe_image_coverage.py` -> `raw/rl_fan/image_coverage.json` (`ZZCOVERAGE_COMPLETE`), MEASURED from banked artifacts, two routes that do not share an assumption: (a) the main run's `train.log` has 29 load lines, every one "N/N tuples resolve ALL cameras" (last 189,888/189,888, 413,152 frames indexed); (b) arithmetic: 103,288 yaml tokens x 4 = 413,152 exactly, the rank-0 bank names 412,148 distinct frames, and the 1,004 = 251 x 4 difference is exactly the 251 tokens with no bank row. All 51,000 partition scenes are bank rows (logs agree).
* (!) That covers the ORIGINAL A40 pod's tree. The P2 pod (where RL-S1 will run) is **UNVERIFIED -- INHERITED** (same 32 OpenScene shards via `code/pod_dataprep.sh`). Its measurement is `rl_fan_dump.py --plan-only` (exit 3 + `unavailable_scenes_*.tsv` on any gap).
* If that ever finds a gap, the only admissible remedy is `refe/rl_pool_available.py` (filters the POOL token table; `rl_partition.py` stays byte-identical; suite `test_rl_pool_available.py` ZZTEST_PASS 10 = 15 checks + 8 RED mutations, with property tests on the REAL `partition()`). Its sequence is in its docstring. Log OLD and NEW partition_id in the prereg BEFORE the first RL number.
* The partition TSVs are now in the repo: `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012/{rl_train_scenes.tsv, rl_val_scenes.tsv}` (sha256 equal to the manifest's).

## 3. `--dump-scores <dir>` in `scorer_finetune.py` -- default OFF, bit-identical when off
* New parser dest: **`dump_scores`** (default `""`, one new dest, nothing else). (!!) **Master Mind: add `dump_scores` to `DVB_KINDS_SCORER` in `gate_checks.py`** (kind: run / observational -- it changes no weight, loss or logged number; the suite proves that on the real `main()`). I did not edit the gate files.
* Writes `scores_<tag>_<base|A|B>.npz` in exactly `rl_val_eval.write_scores_npz`'s layout (`log`, `token`, `step`, `rank`, `ckpt_step`, `scores` float64 [N,M,C], scalar `space="logit"`), keyed per proposal from the batch's own rows (not from a re-derived order).
* **Suite: `refe/test_dump_scores.py` ZZTEST_PASS 16** = 41 checks + 14 RED mutations (token column reversed, proposal axis flipped, arms swapped, ckpt_step zero, row offset forgotten, last-batch keys truncated, sigmoid written as logit, space prob, float32, step column dropped, dump alters the scores, dump always on, flag default on, unguarded edit inside `evaluate()`). OFF = the AST of `evaluate()` minus the guarded blocks is identical to the pre-edit function. End to end: the dump goes through the real `rl_val_eval.run` and PDMS equals the hand-derived value; the real `main()` smoke (test_rl_epo's CPU harness) gives bit-identical weights and logged numbers with the dump on and off. `test_rl_epo.py` re-run after the edit: ZZTEST_PASS 42.

## 4. EVIDENCE STATUS
| claim | status |
|---|---|
| dumped record format is read by `OnPolicyBank`, `served_keys`, both labellers and the 3 `rl_val_eval` loaders | **MEASURED** (suite + labeller proof, above) |
| dumper code-path equals the old dumper's (same GPU), batch-size / composition / worker invariance | **A PASS** (500/500 bit-identical); **C FAILED its EXACT bar** (batch size and grouping change values by up to 0.25) -- and the OLD dumper is batch-variant by the identical 59,156 values, new == old bit for bit at batch 1 and at batch 4: inherited from the forward pass. MEASURED 2026-10-12, `raw/rl_fan/RESULT_IDENTITY.md` |
| dumped fan equals the banked 22,742 step-10,075 sets on every key the dev box holds images for | **MEASURED 2026-10-12: NOT bit-identical** on 1,365/1,365 keys, max abs delta 0.25, the old dumper likewise (B2) -- cross-GPU plus batch dependence; the bit-identical pod gate of section 6 would refuse a correct dumper (see `raw/rl_fan/RESULT_IDENTITY.md`). Original row: (!) Cross-hardware: the dev box is an RTX 4060 (Ada), the banked sets came from an A40 (Ampere); bf16 / TF32 kernels may differ, so B is MEASURED, never assumed, and a difference is reported with its magnitude as a DEFECT, not a footnote. If B differs while A is exact, the pod gate below decides. |
| dev-box s/set and peak GPU | 0.520 s/set total, 0.420 forward, peak 2.27 GB (RTX 4060; MEASURED 2026-10-12, `raw/rl_fan/identity_devbox.json`) |
| P2 pod image coverage | **UNVERIFIED -- INHERITED** (`--plan-only` on the pod) |
The dev-box GPU was held by another agent's `ladder_eval_diag.py` (7.7 GB, OOM at my model load) the whole session; I did not kill it. A waiter (`_rlfan_work/wait_and_run_identity.sh`) starts the identity run on its own once the GPU is idle -- see `raw/rl_fan/HANDOFF_IDENTITY.md`.

## 5. Cost estimate for the A40 (ESTIMATED, from the MEASURED old-dumper median of 0.2313 s/set)
51,000 x 0.2313 s = **3.28 h** of dump (the dev-box rate replaces this when measured); lean sets measure ~37.2 KB each (6 real lean records) => **~1.9 GB**; labels ~1.78 CPU-s (PDM) + ~1.34 CPU-s (D/K/L) per set => **~44 CPU-h** (both labellers have `--workers`; `rl_terms_label.py` also `--shard i/n`; PDM has no `--shard`, split by running different `--sets` dirs).

## 6. Pod command lines (shipped files only; no pod was touched)
```
# 0. preflight, seconds, no GPU: coverage of the frozen partition on THIS pod (exit 3 + unavailable_scenes_*.tsv on any gap)
python refe/rl_fan_dump.py --plan-only --ckpt <model_final.pt> --keys <rl_train_scenes.tsv> <rl_val_scenes.tsv> --partition-manifest <A_no_trained_list_seed20261012.manifest.json> --targets <train_grow> --images <navtrain_pixels> --calib <calib_table.json> --out <d_plan>
# 1. the GATE before the 51,000-scene dump: dump a slice of the banked step-10,075 keys WITH --compare-sets; every dumped key that has a banked set at this ckpt_step
#    must be bit-identical, else exit 5 (a rig/kernel difference on THIS pod is then a defect to diagnose, not a footnote)
python refe/rl_fan_dump.py --ckpt <model_final.pt> --keys <tsv of banked keys> --targets ... --images ... --calib ... --out <d_cmp> --compare-sets <banked onpolicy_*.jsonl files or dirs> --limit 2000
# 2. dump, once per side of the partition (scorer_finetune reads --onpolicy and --heldout as two dirs)
python refe/rl_fan_dump.py --ckpt <model_final.pt> --expect-ckpt-sha256 <MODEL_REGISTRY sha> --keys <rl_train_scenes.tsv> --partition-manifest <manifest> --targets ... --images ... --calib ... --out <sets_train> --shard I/N
python refe/rl_fan_dump.py ... --keys <rl_val_scenes.tsv> --out <sets_val>
# 3. labels on every proposal (NOT --served-only: RL needs all 64), navsim venv
python refe/onpolicy_relabel_pdm.py --sets <sets_X> --out <pdm_X> --workers N        # then rl_terms_label.py --sets <sets_X> --out <terms_X> --workers N [--shard I/N]
# 4. RL-val log list for --heldout-logs:  cut -f2 rl_val_scenes.tsv | sort -u
```
(!) `--compare-sets` is a flag of the dump run itself (after writing, it compares every dumped key against the reference sets at the same ckpt_step); it is exercised by the suite's compare-gate mutations.

## 7. HAZARDS (for the Master Mind / gate)
1. (!!) **A lean set is only meaningful with PDM labels.** `targets` are placeholders (an all-ones row); `OnPolicyBank._apply_pdm_labels` overwrites all six components. The RL argv MUST name `--require-pdm` (already in the refe_rls1 common argv). **Add a gate check: an `--onpolicy` / `--heldout` dir that contains `onpolicy_rlfan_*` implies `--require-pdm` AND `--pdm-labels-*`.**
2. New gate dest: **`dump_scores`** (see section 3).
3. The tool is one-checkpoint-one-`--out` dir by design (`--tag`); resume refuses a different `ckpt_step`. Do not mix a fan from another checkpoint into `sets_train`.
4. `scorer_finetune.py` is still staged as a NEW file on this index (see the gate agent's "What is left" item 1): land it before any token can be minted; my +49 lines ride on top of it.

## Deliverable manifest (all STAGED; none committed or pushed)
`refe/rl_fan_dump.py` - `refe/test_rl_fan_dump.py` - `refe/rl_pool_available.py` - `refe/test_rl_pool_available.py` - `refe/scorer_finetune.py` (modified, +`--dump-scores`) - `refe/test_dump_scores.py` - `eval/rl_fan_identity.py` - `eval/rl_fan_labeller_proof.py` - `raw/rl_fan/{probe_image_coverage.py, image_coverage.json, HANDOFF_IDENTITY.md}` - `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012/{rl_train_scenes.tsv, rl_val_scenes.tsv, .gitattributes}` - this section. All paths relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`.
**Single-location (dev box D: only, NOT in the repo):** `D:/Projects/TanitAD-artifacts/_rlfan_work/` (inputs: dev bank rows, proxy sets, key lists; the identity run's output `identity/` appears there). Nothing is on any pod.

---

# 2026-10-12 -- RL-S1 TL-1 (PI: "Paper-pure, smaller pool"): the route, its measured cost, the partition re-cut, the chain wiring

Scope kept: `refe/gate_checks.py` and `refe/launch_gate.py` were NOT touched. Nothing ssh'd, nothing on a pod. Everything below is STAGED, nothing committed or pushed.

## 1. The route (MEASURED) and its cost per set
* **What a queue row is.** `onpolicy_dump.py` writes `{key, ckpt_step, props [64][20][3], teacher, aug}` (line 141: `"teacher": row["traj"]`). **`teacher` is the BANKED teacher trajectory of `TargetBank.rows[i]`, not a model output**; `props` are the generator's proposals. Neither needs a queue or a teacher model at label time: the proposals are in the `rl_fan_dump.py` record (`traj` + `yaw`, the dumper's own rounding), the trajectory is a row of `targets_rank0.jsonl`.
* **The route:** `refe/rl_tlane_label.py` rebuilds the queue row from (dump record, bank row) and calls `onpolicy_relabel_teacher_lane.label_log` UNMODIFIED (same Scorer, DriveRL calculators, stride 2, `--repair-last-heading` as SFT-4's rows carry). Zero extra GPU, zero extra dump. The three options in the brief: (a) `rl_fan_dump` emitting queue rows: unnecessary, the join needs nothing the dump lacks and the dumper stays byte-for-byte untouched; (b) the old `onpolicy_dump.py` with its gaps fixed: ruled out by `dump_fitness.json` (F1-F4: no key list, a checkpoint hard-wired to a live run dir, the wrong output kind; `rl_fan_dump.py` already is that fix, so editing the old file again would only risk `op_pipeline.sh`); (c) reusing the 22,742 frozen-fan sets: the simulator cost is per set whatever produced the proposals, no PDM label survives, and the files are on the dev box (see `PREREG_RL_S1.md` section 9).
* **Cost per set: about 10.2 CPU-s** (the labeller is one Scorer pass over teacher + 64 proposals; there is no cheaper paper-pure variant, the calculators are the cost). MEASURED: the banked held-out job of SFT-4 (`status_r0_hw0..3.json`, rank 0, 1,705 sets): `sec_scoring` 10.15 / 10.18 / 10.16 / 9.44 s per sample on the pod (the NAVSIM DAC part is only 0.22-0.23 s of it, and this tool does not run it); the dev box, this tool, 1 worker, steady state (second chunk) 10.4 s/set; the original command with 2 workers 10.7 s/set/worker (`raw/rl_tlane/dev_run10.log`, `dev_run_old.log`; the dev box was shared with another job). **19,000 sets = 53.8 CPU-h = 5.4 h at 10 workers (4.9 h at 11 vCPU).** With the PDM (0.9-1.5 h) and `rl_terms` (0.7 h) labels the chain's dump + labels is about 7.0-7.6 h wall, against 6.5-7 h for the dropped 51,000-scene plan without lane labels.
* **Identity (MEASURED, `raw/rl_tlane/`, instrument `eval/rl_tlane_identity.py`):** (1) rebuilt queue rows equal REAL queue rows float for float: 3,137 of 3,137 held-out rows (rank 0 and 1) and 1,920 of 1,920 train rows, through the real `rl_fan_dump.set_record`; (2) the queue's `teacher` equals the bank row's `traj`: 1,920 of 1,920; (3) this tool vs the ORIGINAL `onpolicy_relabel_teacher_lane.py --queue ...` command on the same 24 held-out sets, same machine: **14,040 of 14,040 leaf values bit-identical**; (4) vs SFT-4's banked POD labels for the same 24 keys: 13,608 of 14,040 equal, every event / categorical leaf equal, the centre-line distance differs on 337 of 1,560 values (171 by <= 1e-6, 14 in (1e-6, 1e-2], 152 in (1e-2, 0.074]), the trainer's soft target by <= 0.042; `REFE_SIM_HZ` 10 and 20 give identical output. **(4) is not "reproduces SFT-4 exactly": it is the ORIGINAL labeller's cross-machine non-determinism (Windows dev box vs the Linux pod; the `onpolicy_label.py` selftest comment documents near-tie lane flips of the same kind), and (3) shows the route adds none.** A bit-for-bit reproduction of SFT-4's pod labels needs the pod: UNVERIFIED.

## 2. The partition re-cut (priority 2) -- `PREREG_RL_S1.md` section 9
`rl_partition.py` already had `--n-train`; no tool edit. Same tool, seed 20261012, `--n-train 16000`: OLD `partition_id` `272cfd357b81...` -> NEW `9995a39aeeac...` (manifest sha256 `f8a1640c...`). The tool re-produced the frozen 48,000 / 3,000 byte for byte first; the new RL-train list is an exact prefix of the old; **RL-val is byte-identical**. Banked beside the old run: `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000{.manifest.json,/}`. `rls1_checks.PARTITION` is re-pinned (n_train 16000). The "prefer keys whose queue rows exist" rule was NOT adopted (it saves under an hour of dump for a selection; section 9 has the arithmetic).

## 3. The chain (priority 3) -- `refe/rls1_pod.sh`, `refe/rls1_checks.py`, `refe/test_rls1_chain.py` `ZZTEST_PASS 238` (was 177; `test_p2r_chain` 139, `test_rl_partition` 20, `test_rl_fan_dump` 28 unchanged and green)
* the old pre-compute `tlane-check` (labels supplied from outside, `RLS1_TLANE_*` env) is GONE; a test refuses its return. Stage `tlane` is now a 3-set labeller **smoke on the RL-val dump** (`ZZTLANE_DONE 3 0 0` + the trainer's row acceptance); `label_one` runs the labeller per shard after PDM and `rl_terms`; stage 6b `rls1_checks.py tlane-exact` (**new**) requires exactly one accepted row per dumped set, `--expect` the partition's count (16,000 / 3,000), none foreign / doubled / rejected; the bank stage tars `tlane_train` and `tlane_heldout` too; the join-gate counts (16000 / 3000), the remedy (`--n-train 16000`) and its logged old id are checked against `RC.PARTITION` by the test.
* **`refe/rl_tlane_label.py`** (new; `refe/test_rl_tlane_label.py` `ZZTEST_PASS 36` = 42 checks + 34 RED mutations): shard by log (same function as the dump), resume by key + `ckpt_step`, refusals for a foreign checkpoint / rank-1 set / doubled key / missing or doubled bank row / torn dump line / missing `--repair-last-heading`. **The suite found a real defect in my first draft: a kill leaves a torn last line, and appending after it fused the next row onto the fragment and lost it; fixed (`terminate_torn_line`), pinned by a mutation.** A second pipelining hazard is pinned the same way: the chain labels shard i while the dump is still writing shard i+1 into the same directory, so the labeller of shard i reads only `onpolicy_rlfan_*_s<i>of<n>.jsonl` (it would otherwise refuse on a neighbour's half-written last line); mutation `shard_reads_every_dump_file` turns exactly that check red.
* new regression arms in the chain test, each RED when broken: label override back, no smoke, smoke unchecked, exact gate missing / without the count / unchecked, labels unaudited, labeller without the repair / SIM_HZ / the training bank / the partition's keys / keeping fd 200, lane dirs not banked, update budget unchecked, `--epochs` back to 2, join-gate / remedy / manifest drift to the 48,000 partition. `rls1_checks.tlane_exact` and `updates_check` have 14 source mutations of their own.

## 4. THE ARGV CHANGE the Master Mind must know (report, not a gate edit)
**`--epochs 2` -> `--epochs 4`** in `rls1_pod.sh` `arm_args` and in the fenced block of "The exact argv per arm" above (the chain test compares the two token for token). Reason: the trainer ends at `min(ceil(n / 8 x epochs / 8), --max-updates)`; on 16,000 scenes `--epochs 2` is **500 updates**, silently, and the chain reads `update1000`. 16,000 / 64 x 4 = 1,000 exactly. `--epochs` is NOT pinned by `refe_rls1` (`RLS1_RECIPE` carries `--max-updates`), so the real gate still calls all five argv CLEAN (the test runs it). A new guard, `rls1_checks.py updates-check`, refuses any arm whose planned updates differ from `--max-updates`. **Design consequence for the PI:** each scene is seen 4 times instead of 1.33 (every arm identically); the alternative (500 updates) changes the ramp and the K3/K4 schedule. If the gate's `RLS1_RECIPE` is ever extended to pin `--epochs`, pin 4.

## 5. Hazards / open (nothing hidden)
1. **Landing:** `SHIP_MANIFEST.json` must be regenerated after landing (new `refe/rl_tlane_label.py`, `test_rl_tlane_label.py`; changed `rls1_pod.sh`, `rls1_checks.py`); the chain's required-file list now also names `refe/onpolicy_relabel_teacher_lane.py onpolicy_label.py planner.py scorer_gate.py score_proposals.py route_rank.py map_feature_cache.py query_cache.py navtrain_scenarios.py build_scorer_targets.py code/augment_routes.py` (all exist; the test checks it) and the `_n16000` partition files.
2. **UNVERIFIED on a pod:** the Scorer on the pod's DBs / maps / `DRIVERL_EVAL_DEVICE=cuda` env (the SFT-4 labels ran in the same env), 10 spawn workers at about 0.7 GB each, the 5.4 h. The `tlane` stage is the cheap discriminator (it runs right after the 3,000-set RL-val dump, about 15 min of GPU).
3. **Pre-existing, not mine:** the chain exports no `REFE_SIM_HZ` to `onpolicy_relabel_pdm.py` / `rl_terms_label.py` (both import `navtrain_scenarios`, whose default is 20; the dev-box proof harnesses run them with 10). My labeller sets 10 (the bank's and `op_env.sh`'s rate; 10 and 20 gave identical output on the 24 held-out step-0 sets). Whether it matters to those two tools was not checked.
4. **Cross-machine label noise** (section 1 item 4) is a property of the original labeller; a pod re-measurement of the 3,137 held-out keys against SFT-4's banked labels would price it exactly (needs the held-out queue: `D:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04/refe_heldout_v5.tar.zst`, 65 MB).

## Deliverable manifest (all STAGED; none committed or pushed)
`refe/rl_tlane_label.py` - `refe/test_rl_tlane_label.py` - `eval/rl_tlane_identity.py` - `refe/rls1_pod.sh` (modified) - `refe/rls1_checks.py` (modified) - `refe/test_rls1_chain.py` (modified) - `eval/RLS1_POD_RUNBOOK.md` - `eval/PREREG_RL_S1.md` (section 9) - this section and the `--epochs 4` argv line - `raw/rl_pool/partition_runs/A_no_trained_list_seed20261012_n16000{.manifest.json,/rl_train_scenes.tsv,/rl_val_scenes.tsv,/.gitattributes}` - `raw/rl_pool/partition_runs/.gitattributes` - `raw/rl_tlane/{identity_*.json, dev_run10.log, dev_run20.log, dev_run_old.log, devbox_env.sh}`. All paths relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`.
**Single-location (dev box D: only, NOT in the repo):** `D:/Projects/TanitAD-artifacts/_tlane_work/` (extracted held-out and train queue rows, the 24-set smoke inputs and outputs `smoke/out10`, `out20`, `out_old`).
