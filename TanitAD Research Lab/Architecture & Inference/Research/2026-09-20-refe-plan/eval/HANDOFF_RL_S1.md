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
