# RESULT — v7F R3: ALL tactical labels now reach the tactical layer's training loss

PI R3 (BINDING, 2026-09-27): *"All our tactical labels must be used to train the tactical layer to
estimate and choose the right tactical behaviors and goals which MUST condition the operative planning."*

Base: origin tip **c36b6ddd** (`stack/`, snapshot `C:/Users/Admin/tipsnap/c36b6ddd/`). Evidence class of
every number below: **MEASURED** (ours; this package), unless marked otherwise. Nothing here is a
capability claim: every run is a CPU plumbing smoke on a tiny model.

## 0. Headline

1. **Reaches a loss now, did not before:** the lateral + longitudinal tactical ACTION ids (CE on
   `a_lat`/`a_lon`), the 22-token tactical GOAL set incl. **every traffic-light colour** (multi-label BCE
   on `g_tac`, D-TLIGHT-1 closed for v7F), and SPEED_BAND's target interval (v_lo, v_hi) (L1 on
   `g_tac.args` slots 0/1) — in S-T (where the label file was not even loaded before) and S-J.
2. **Flag:** `--w-tac-label-all W` (float, default 0.0 = absent). Companions, same names/semantics as the
   refcv3 trainer: `--tac-goal-negatives {measured*,geometry,all,cot-absence-negative}`,
   `--cot-negative-sidecar PATH`. Requires `--s2-labels <v7.2 blob>` and `--goal-multilabel`; refused with
   `--goal-factored`, in S-W/S-S, off the v7.0 vocabulary, and without the planner cut.
3. **Tests:** new `test_tactical_label_reach_v6.py` **39/39 pass** (incl. 3 deliberate-regression arms that
   go RED, and G-DVB mutation arms). 88 test files touching the changed modules: baseline c36b6ddd
   **1950 passed / 47 skipped / 8 failed / 10 errors**; modified **1989 passed / 47 skipped / 8 failed /
   10 errors** — **0 outcome changes on the 2,015 common tests**; the 18 failures/errors are the identical
   set in both trees (out-of-tree artifacts, §4).
4. **Bit-identical when OFF, three independent proofs:** 8/8 in-suite cases vs the c36b6ddd trainer
   module; **16/16 cross-process runs** vs the untouched snapshot tree (loss, every term, log keys, global
   RNG, EVERY parameter gradient, state_dict); CLI `main()` dry-run step rows identical in 4/4 stages.
   Only metadata changes when off: 5 new `config.json` keys + one effective-weights row.
5. **Needs integration / decisions:** (a) the launch gate has NO v6/v7F profile — the G-DVB entry is
   built as a v6-scoped module (`declared_vs_built_v6.py`) and wired into the trainer; wiring it into
   `launch_gate.py` is a **PROPOSAL** (§6); (b) PI/MM decisions: negatives policy (the banked
   absence-as-negative sidecar is over a DIFFERENT blob md5 and is refused on the v7.2 pin), the PROPOSED
   SPEED_BAND slot mapping, and **behaviours (`a_lat`/`a_lon`) still do not condition the operative
   planner** (R4 territory) — §7.

## 1. What was built

**Term** (`train_v6_staged.tac_label_all_loss`, in force where `layer_tac` trains):

    L_R3 = CE(a_lat.logits, tac_lat_id) + CE(a_lon.logits, tac_lon_id)
         + BCE_w(g_tac.logits, tac_goal_y; tac_goal_w, pos_weight, class_mask)      # tac_goal_head.tac_goal_loss, reused
         + |g_tac.args - tac_goal_args| . tac_goal_arg_mask / SPEED_SCALE            # SPEED_BAND v_lo/v_hi, m/s -> /10
    loss += w_tac_label_all * L_R3

* Targets come from the ONE rule (`v7_labels.tactical_goal_targets`) through the EXISTING v7.2 join
  (`V72WindowSupervision`), on the same windows and the same tactical band as the action ids (±(hi-lo)/2
  = ±2 s of `t0`). `class_mask` = `tac_goal_head.mask_report(goal_supervision_census(...))` and
  `pos_weight` = `goal_pos_weight(...)`, both computed on the LOADED split — the refcv3 D-TACGOAL recipe,
  so the two trainers cannot disagree about what "supervised" means.
* HEADS ONLY (MEASURED, §3.2): gradient reaches `goal_head_tac`, `act_head_lat`, `act_head_lon` and — via
  the `e_g_str` cond `g_tac` reads — `goal_head_str`/`vocab_str` (frozen in S-T); never encoder, readout,
  adapters or predictors.
* The supervised goal is the one that conditions the operative planner: `--goal-multilabel` makes
  `e_g_tac` (-> `predictor_op` intent, -> the 6 s emission) read the sigmoid **gates** the BCE trains;
  `--goal-factored` would build `e_g_tac` from a different head pair and is refused.
* NO new module, NO new state_dict key, NO RNG draw (a flag-on step draws the same global stream).

**Files** (`code/fix/`, full; `code/diffs_vs_c36b6ddd/`, unified diffs that `git apply` cleanly onto the
c36b6ddd blobs — `raw/diff_apply_check.txt`):

| file | change |
|---|---|
| `stack/scripts/train_v6_staged.py` | +711 / −7, 26 hunks (table below) |
| `stack/tanitad/train/declared_vs_built_v6.py` | NEW, 214 lines — G-DVB for the v6 trainer's new levers |
| `stack/tests/test_tactical_label_reach_v6.py` | NEW, 39 tests |
| `stack/tests/test_v6_effective_weights.py` | pinned zeroing counts S-W 10→11, S-S 14→15 (the new term is zeroed there; named in the docstring) |

Trainer hunks (new-file line ranges from the diff; every `−` line is an intended modification):

| hunk | what |
|---|---|
| `+347,19` | `V6LossWeights.w_tac_label_all: float = 0.0` (+ contract comment) |
| `+377,9` / `+405,10` | `for_stage`: zeroed in S-W / S-S (`−` = the two `replace(...)` closings) |
| `+2596,75` | `V72WindowSupervision.enable_tac_label_targets` (per-record `(y, w)` + SPEED_BAND args, built once) |
| `+2675,13` | `report()`: names the R3 consumer only once enabled (`−` = `return super().report() \| {`) |
| `+2689,15` / `+2709,12` | `batch()`: emits the six `TAC_LABEL_BATCH_KEYS` only when enabled |
| `+3003,295` | R3 section: `TAC_LABEL_BATCH_KEYS`, `TAC_GOAL_ARG_SLOTS_V7` (PROPOSED), `_tac_axis_ce`, `tac_label_all_loss`, `tac_label_policy`, `build_tac_label_targets` |
| `+4606,10` | `v6_loss_step` docstring: batch contract |
| `+5114,46` | `v6_loss_step`: the term (refuses without the planner cut, on a factored or non-multilabel head) |
| `+6203,50` | `synthetic_tac_label_batch` (dry-run stand-in, own generator) |
| `+6370,45` / `+6441,9` / `+6569,10` | `dry_run`: keep the loaded label set (`−` = 2 lines); G-DVB v6 + policy on the real blob; synthetic keys per step; `dry_run.json["tac_label_all"]` only when ON |
| `+6609,7` | `_weights_from_args` |
| `+7177,10` | `train()`: load `--s2-labels` when `w_s2_goal` **or** `w_tac_label_all` is in force (`−` = the old condition) |
| `+7211,38` | `train()`: G-DVB v6, `build_tac_label_targets`, and re-reading the join report into `config.json` |
| `+7659,10` | `train()`: `config.json["tac_label_all"]` (only when ON) |
| `+10534,38` | parser: `--w-tac-label-all`, `--tac-goal-negatives`, `--cot-negative-sidecar` |
| `+10735,7` / `+10779,19` / `+10809,11` | effective-weights audit: `W_TERM_FLAGS`, `_term_precondition`, `W_TERM_MASKS` |
| `+10901,97` | `_preflight_tac_label_all` (every refusal, milliseconds, before the corpus) |
| `+11468,9` / `+11507,7` / `+11529,8` | `preflight`: read `w_tl`; the incumbent "`--s2-labels` with `--w-s2-goal 0`" refusal is SCOPED to not fire when R3 is the consumer (`−` = that condition); call the R3 refusals |

## 2. What reaches a loss now — per family (before -> after)

MEASURED on the canonical v7.2 TRAIN blob (md5 `0ff90213…`, 4,572 records; `raw/audit_census_train_blob.txt`)
unless marked. Hop-by-hop citations of the "before" column: `AUDIT.md`.

| family | before (c36b6ddd) | after, `--w-tac-label-all > 0` |
|---|---|---|
| lat action `a_tac.lat` | NO (ids joined only in S-S/S-J, read by no loss) | **YES** — CE on `a_lat` (5 present classes; LANE_CHANGE_L/R, ABORT_LC absent from the blob) |
| lon action `a_tac.lon` | NO | **YES** — CE on `a_lon` (7 present; YIELD_MERGE absent) |
| goal SET (22) | NO (never projected) | **YES** — `measured` (default): **17/22** classes trained; `all` / `cot-absence-negative`: **21/22** |
| traffic light RED / GREEN / YELLOW / colourless | NO (D-TLIGHT-1) | **YES under every policy** — `measured` pos/neg: RED 376/403, GREEN 363/416, YELLOW 22/757, colourless 18/761 (negatives ENTAILED by the other colours) |
| SPEED_BAND (constant token, 4,572/4,572) | NO | token masked (no negative under any policy); its **(v_lo, v_hi) reach the loss** via `g_tac.args` slots 0/1 (PROPOSED mapping) |
| masked under `measured` | — | YIELD, CORRIDOR_OFFSET, GAP_TARGET, REACT_ON_ONCOMING (CoT tokens with ZERO supervised negatives) + SPEED_BAND — they reach under `all` / `cot-absence-negative` |
| other per-goal args (STOP_POINT within/hold, TURN radius/dyaw/within/by_time, GAP time_gap_s, …), action args (`within_m`, `lat_peak_m`, `v_target_ms`), `g_tac.anchor` | NO | **still NO** — needs the DataFlyWheel slot encoder (§7) |

## 3. Proofs

### 3.1 Flag OFF is bit-identical to c36b6ddd (three independent proofs)
1. **In-suite** (`test_OFF_the_loss_is_BIT_IDENTICAL_to_the_c36b6ddd_trainer`, 8 cases = 4 stages × batch
   with/without every R3 key present): the c36b6ddd trainer loaded as a module (content-anchored:
   `$TANITAD_R3_REF_TRAINER` or `git show c36b6ddd:`), same stack, same batch, same seeds — loss, every
   term (`torch.equal`), log-key set and global RNG state equal. Negative control: ON differs.
2. **Cross-process, cross-tree** (`code/proof/bit_identity_off.py`, `raw/off_compare.txt`): the untouched
   snapshot tree (its own `tanitad` + trainer, no bytecode written into it) vs the modified tree over the
   SAME saved batches — **16/16 identical** on loss (`float.hex`), every term, log keys, global RNG after
   the step, a digest of EVERY parameter gradient after backward, and the freshly built state_dict
   (2 configs [default, `--goal-multilabel`] × 2 batches [plain, + S2/tac/R3 keys from the REAL join] ×
   4 stages).
3. **CLI** (`raw/cli_off_dryrun_compare.txt`): the real `main()` (preflight + dry-run), S-W/S-T/S-S/S-J —
   step rows identical 4/4; control: the untouched tree matches itself. Record deltas, exhaustively:
   `config.json` gains `args.{w_tac_label_all, tac_goal_negatives, cot_negative_sidecar}`,
   `loss_weights.w_tac_label_all`, `loss_weights_in_force.w_tac_label_all` (all at default), and the
   effective-weights table one row. ⚠️ Found in passing, PRE-EXISTING: `dry_run()` never seeds the global
   RNG before building the model (`train()` does), so two dry-runs of the SAME untouched tree differ; the
   comparison seeds it outside the trainer (`code/proof/seeded_main.py`).

Tree hygiene: the baseline copy used for (3) is byte-identical to the snapshot
(`raw/tree_identity_base_vs_snapshot.txt`); my tree differs from the snapshot in exactly the 4 files above
(`raw/tree_diff_mine_vs_snapshot.txt`).

### 3.2 Flag ON: every family reaches a finite, non-zero-gradient term
* Literal fixture through the REAL v7.2 join (9 clips reproducing the train blob's provenance structure):
  under `all` the goal rows of `g_tac.type_head` receiving gradient are **exactly** the 21 non-SPEED_BAND
  tokens (all four traffic-light tokens included); SPEED_BAND reaches through `arg_head` rows **{0, 1}
  exactly**; under `measured` the five masked rows get **exactly 0.0**; lat/lon CE tok-counts equal the
  fixture's literal counts; RED<->GREEN swap: OFF moves nothing (D-TLIGHT-1 as originally measured), ON
  moves the term.
* HEADS ONLY, on the real autograd graph: the R3 term alone reaches exactly the top-level modules
  {`goal_head_tac`, `act_head_lat`, `act_head_lon`, `goal_head_str`, `vocab_str`}; under the real S-T
  freeze the full S-T backward populates `.grad` only in `layer_tac`/`planner`.
* **Real records** (`code/proof/smoke_real_eval_labels.py`, `raw/smoke_real_eval_labels*`): the canonical
  v7.2 EVAL blob (md5 `aa12c948…`, 147 records) through the real join, one in-band window per clip:
  `measured` reaches 13/22 goal tokens, `all` 17/22 — **RED (8 pos), GREEN (15), YELLOW (3) reached in both**;
  the colourless token and 3 others have **zero positives in the eval split** (masked for that reason).
  Tactical-band coverage: **6,027 / 27,930 windows = 21.6 %** (190 windows/clip, window 4).
* **Real `train()`** (`raw/train_smoke_ST_summary.txt`): 3 real clips (RED/GREEN/YELLOW) from the local
  eval v2 cache, S-W 2 steps -> S-T 6 steps via `--init-from/--prev-gate` (INCONCLUSIVE gate overridden
  WITH a stated reason; stamped `--no-require-parity --exclude-eval-clips none
  --allow-eval-clips-in-train` — a plumbing smoke, not an arm). ON terms `[plan, seam, t1, tac_label_all]`;
  RED, GREEN and YELLOW all appear as supervised positives; step-1 incumbent terms (t1, seam, fan ADE)
  identical ON vs OFF; in-band 123/405 windows (30.4 % at max_horizon 60). ⭐ This smoke caught a real
  record defect no unit test saw: `config.json["s2"]["join"]["tactical_consumer"]` said "NONE" while the
  term was in force (the report was snapshotted before enabling) — fixed (`+7211` hunk) and pinned
  (`test_the_join_report_names_the_consumer_only_once_R3_is_enabled`).

### 3.3 Deliberate regressions (each must go RED — and does)
* join unwired (`enable_tac_label_targets` monkeypatched to a no-op) -> the term **refuses** (`ValueError:
  … missing …`), it never silently skips;
* goal BCE dropped (`tac_goal_loss` -> 0) -> the reach check fires (`goal reach moved`);
* colour dropped at the join (`tactical_goal_targets` zeroes the four TL cells) -> the reach check fires;
* G-DVB v6: a stack built without `--goal-multilabel`, with `--goal-factored`, on the v6.0 vocabulary, or
  without the planner cut; the weight the loss reads ≠ argv (S-W zeroing, or ON behind argv's back); a
  flag missing from the parser or the registry — each is a named `Mismatch`.

## 4. Test results (CPU, dev box; `CUDA_VISIBLE_DEVICES=-1`; `tanitad` verified to import from the copy)

* `stack/tests/test_tactical_label_reach_v6.py`: **39 passed** (`raw/tests_v7f_r3_final.log`).
* 88 test files that import `train_v6_staged` / `declared_vs_built` / `v7_labels` / `tac_goal_head`
  (`raw/test_files_touching.txt`), JUnit per test id (`raw/tests_outcome_compare.txt`):

  | tree | passed | skipped | failed | errors | total |
  |---|---|---|---|---|---|
  | c36b6ddd (untouched copy) | 1950 | 47 | 8 | 10 | 2015 |
  | modified | 1989 | 47 | 8 | 10 | 2054 |

  0 outcome changes on the 2,015 common tests; +39 new, all pass. The 18 failures/errors, identical in
  both trees, all read artifacts outside `stack/` that this checkout does not carry
  (`raw/preexisting_failure_reasons.txt`): 15× `test_eval_contamination` (Research Lab files absent —
  `alpamayo_clip_ids.txt` ×8, `alpamayo_IN_parity_train…` ×2, the s2-v1 `labels_v2` delivery ×3,
  `parity_ls.txt`, the val40 lead index), `test_launch_closure_audit` (`Project Steering/GATE_PROTOCOL.md`
  absent), `test_v6_st_launch_fixes::test_E4_…` (the E4 resolution doc absent),
  `test_launch_gate::test_G_SUITE_the_pinned_env_list_…` (the pinned sha is over the LF blob; this
  checkout is CRLF). Verbatim in the two full logs.
* `test_runbook_commands.py` needs two out-of-tree docs; copied into both test trees (read-only copies):
  16/16 pass on both.

## 5. CPU smokes (the trainer's own tiny path)

* `--dry-run`, tiny geometry (`raw/cli_dry_runs_summary.txt`): S-T ON exit 0, terms `[plan, seam, t1,
  tac_label_all]`, step-1 `tac_lat_ce 2.055 · tac_lon_ce 2.050 · tac_goal_bce 0.695 ·
  tac_speedband_l1_ms 14.34`; S-J ON exit 0 (terms + o1/o2/o3/o5/o6/s1); S-T OFF exit 0, terms
  `[plan, seam, t1]`. The ON dry-run ran the label policy on the real eval blob: 13/22 trainable
  (eval split) and G-DVB v6 PASS.
* Refusals exit 2 with the reason (`raw/cli_refusal_*.log`): ON in S-W; ON without `--goal-multilabel`.
* 30-step S-T optimisation on real eval records + random pixels (`raw/smoke_real_eval_labels_trajectory.txt`;
  first-5 vs last-5 step means): lat CE 1.84 -> 1.03, lon CE 2.01 -> 1.83, goal BCE 1.12 -> 1.04,
  SPEED_BAND L1 11.30 -> 7.82 m/s. ⚠️ Random pixels ⇒ the heads can only learn label MARGINALS; this
  shows the optimisation path works, nothing about driving.
* At initialisation the goal conditioning has ZERO forward effect by design (the operative FiLM and the
  emission are zero-init) — the gradient path is live (MEASURED: S-T/S-J gradients differ between the
  softmax and gates readings while the losses are identical, `raw/off_v7f_r3.json`).

## 6. G-DVB and the launch gate

* **Built:** `stack/tanitad/train/declared_vs_built_v6.py` — the `declared_vs_built` API/rules
  (`Lever`/`Mismatch`, literal expectations, readers on the BUILT stack) for the v6 trainer:
  `REGISTRY_V6` = `w_tac_label_all` (**loss**: the weight `v6_loss_step` reads for the stage == argv; the
  three heads built on the frozen v7 vocabularies at 22/8/8; `goal_head_tac.multilabel`; no factored pair;
  the planner cut; ≥ 2 arg slots), `tac_goal_negatives` (**data**), `cot_negative_sidecar` (**data**).
  `train()` and `dry_run()` call `refuse_on_mismatch_v6` whenever the term is in force.
* **Why not in `declared_vs_built.REGISTRY`:** its `check()` runs every registered lever against the
  model it is given (a `RefCV3Model`), and two of the new dests already name refcv3 entries.
* **NOT wired into `stack/scripts/launch_gate.py` — PROPOSAL.** `launch_gate.PROFILES` has only
  `refc`/`refcv6`/`refcv7`, all driving `refc_v3_train.py`; there is no v6/v7F profile to hang the entry
  on. Proposed: a `v7f` profile (`trainer: stack/scripts/train_v6_staged.py`) whose G-DVB step calls
  `declared_vs_built_v6.check_v6(stack, args, weights_in_force=V6LossWeights…for_stage(stage),
  parser=build_parser())`, with `required_on=("--goal-multilabel",)` and `--goal-factored` in
  `forbidden_levers` whenever `--w-tac-label-all > 0`, and `required_positive=("--w-tac-label-all",)` if
  the PI makes R3 mandatory for v7F. ⚠️ MEASURED gap it must close first: **230 of the 233**
  `train_v6_staged.build_parser()` dests have no G-DVB entry (`uncovered_v6`).

## 7. NOT done / open questions / decisions needed

1. **Behaviours do not condition the operative planning (R3's second half / R4).** `a_lat`/`a_lon` ->
   `e_a_tac` -> `predictor_tac` ONLY (`v6.py:5958-5989`); the operative planner receives only the GOAL
   embedding `e_g_tac` (`v6.py:5996-5998`, `:6035`). This change makes the behaviours *estimated from
   labels*; making them *condition the plan* is an architecture change (a new port), not done here.
2. **Negatives policy for the v7F launch** (PI/MM): `measured` 17/22 vs `cot-absence-negative` (the PI's
   2026-09-16 ruling, 21/22) vs `all` (21/22, the same `(y, w)` on this blob but without the ruling's
   stamp). ⚠️ The only banked sidecar (`…/2026-09-16-flywheel-negatives/cot_absence_negative_v8.0_train.json.gz`)
   declares `source_blob_md5 fa89ea55…` ≠ the v7.2 train pin `0ff90213…` the trainer enforces, so
   `load_cot_negative_sidecar` refuses it on the canonical blob — it must be rebuilt over the v7.2 blob
   (or v7F moved to that blob). Which blob `fa89ea55` is: UNVERIFIED (the name says v8.0).
3. **SPEED_BAND slot mapping (arg0 = v_lo_ms, arg1 = v_hi_ms) is a PROPOSAL.** Every other per-goal arg,
   every action arg and `g_tac.anchor` stay unsupervised until the DataFlyWheel slot encoder
   (SPEC_V7_LABEL_TRAINER_WIRING §3.2) lands; config.json says so (`args_unsupervised`).
4. **`--goal-factored` under v7.0 is incoherent:** its LAT vocabulary is the v6 partition
   (`ANCHOR_GOAL`, `CORRIDOR_OFFSET`, `EVADE_IN_CORRIDOR`, `LAT_UNCONSTRAINED`, unversioned) and its LON
   vocabulary is all 22 v7 tokens. Refused with R3; a v7 LAT/LON goal partition is a decision.
5. **Weighting** (declared, not tuned): unit sub-weights inside the term (CE + CE + BCE + L1/SPEED_SCALE)
   under one flag weight; lat/lon CE UNWEIGHTED (the refcv3 z_tac convention — `v7_labels.class_weights`
   exists: e.g. LANE_KEEP 0.31 vs TURN_R 3.53). BCE pos_weight on the train blob: **3** of 22 classes sit
   ON the cap of 50 under `measured` (YIELD_FOR_TURN_L/R, OVERTAKE_VEHICLE), **9** under `all` — for
   those the cap, not the data, sets the weight (the refcv3 caveat, same numbers path).
6. **Sparse supervision:** only 21.6–30.4 % of windows sit inside a record's tactical band, i.e. ~3–5
   labelled windows per batch of 16. Band-aware sampling is NOT implemented (it would change every other
   term's data distribution — a design decision).
7. **Docs describing D-TLIGHT-1 as open** (per `test_tactical_label_reach.py`'s `DOCS_CARRYING_THE_CLAIM`):
   `Project Steering/GOALS_AND_CLAIMS.md` (D-TLIGHT-1), `CLAUDE.md`, `…/2026-09-06-traffic-light-chain/
   TRAFFIC_LIGHT_CHAIN.md` — must be updated WHEN this lands; not touched here (outside my paths).
8. Minor, pre-existing, found in passing: `dry_run()` does not seed the global RNG (§3.1); the generic
   effective-weights text for an unmet R3 precondition says "the loss term is skipped" while
   `v6_loss_step` in fact refuses.

## 8. Integration notes

* Apply onto **c36b6ddd (or later origin)** — NOT onto the D: working tree (`agent/arch-inf-20260803`),
  which is behind it (it lacks `stack/tanitad/train/declared_vs_built.py`, imported by the new module).
* Use `code/fix/` (full files) or `code/diffs_vs_c36b6ddd/` (`git apply` validated). Blob line endings
  differ per file in the repo: the trainer blob is LF, `test_v6_effective_weights.py`'s blob is CRLF; each
  diff carries its target blob's convention, and the diffs folder's `.gitattributes` (`*.diff -text`) keeps
  those bytes exact through `core.autocrlf=true`.
* Lazy imports only: the trainer's import-time closure is unchanged (`test_runbook_commands`
  `test_the_static_closure_matches_the_runtime_one` passes); a pod needs `declared_vs_built_v6.py` only
  when the flag is used.
* Example S-T launch fragment: `--stage S-T --w-tac-label-all 1 --goal-multilabel --s2-labels
  <…/s2_labels_v7.2_train.jsonl.gz> [--tac-goal-negatives measured]` (+ the stage's usual
  `--init-from/--prev-gate`, `--nav-cond --nav-labels`).
* Reproduce every artifact in `raw/` (dev-box paths): `code/proof/rerun_all.sh` (all smokes/proofs on
  the modified tree; `raw/rerun_all.log`), then `code/proof/summarize_all.py`; the baseline-side inputs
  are `code/proof/bit_identity_off.py` / `seeded_main.py` run against the untouched tree.
  `code/proof/copy_deliverable.py` is the copy into this folder with the clip-id scan against the 4,719
  v7.2 clip ids (`raw/clipid_redaction_report.json`: 0 bare UUIDs and 0 real clip-id prefixes in the
  final copy — the train-smoke summary names its three clips only by traffic-light colour).
