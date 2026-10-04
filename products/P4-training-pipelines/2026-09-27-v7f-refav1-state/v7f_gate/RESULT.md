# RESULT — a `v7f` profile for the BINDING launch gate (`train_v6_staged.py`), on the v7F merge WITH the nav fix

**2026-09-27 · TrainingFlyWheel (P4) · launch-gate agent · CPU only (CUDA hidden) · STAGED, never committed.**
Built on origin tip `c36b6ddd`, re-pointed (coordinator ~16:00) to the v7F MERGE package
`…/2026-09-27-v7f-refav1-state/v7f_merge/` (R1+R3+R4+R2 fixes+R6), and **updated (coordinator ~17:00) after the
merge MAPPED `--nav-cond`** (`build_stack_from_args`: `nav_cond=bool(getattr(a, "nav_cond", False))`), and
**again (coordinator ~17:40) after the merge FIXED F7 and F8** (`_NavBoundPredictor` binds the operative nav term
to the WM-loss helpers; `synthetic_train_batch` draws nav keys). Design: `DESIGN.md`. Evidence: `raw/`. All times
Europe/Berlin unless marked UTC.

## Headline
1. **v7F has a gate profile** (`--profile v7f`, trainer FAMILY `family: "v6"`): all seven checks defined; on CPU the
   dev-box stage RUNS **G-HYG, G-DVB, G-EVAL, G-LIVE, G-CKPT** as a REHEARSAL (tiny geometry + synthetic corpus +
   **synthetic nav records through the trainer's real join / NavEmitter**, the real trainer code) whose evidence
   `finalize` never counts ⇒ token INCOMPLETE; G-CLOCK (v6) is ERROR (no clock reference for the v7F cache);
   G-SUITE is not run here.
2. **The nav update (coordinator's 4 asks), MEASURED:** (1) the S-W rehearsal predecessor carries `--nav-cond` +
   the synthetic nav source (recorded in the evidence); (2) G-DVB's nav pins are flipped — the compliant S-T and S-W
   rehearsals **PASS G-DVB with 0 mismatches** and a nav-less forward now RAISES `NavTokenMissing`; the regression
   arm `v6_nav_not_mapped` (plus a SOURCE mutation deleting the mapping line) FAILS; (3) the G-HYG `KeyError`s are
   gone — they were downstream of the refused predecessor; G-HYG now names only the 4 open config classes;
   (4) the open item `R2-NAV-NOT-BUILT` is **CLOSED** — `open_items == ()`, an all-green evidence set finalizes
   to **PI-DECISION** (D1 `--tac-op-cond`), never PASS.
3. **Tests (MEASURED, on the merge with the F7/F8 fixes):** `stack/tests/test_launch_gate_v7f.py` **52 passed**;
   the merge's 7 DVB-related files + it **243 passed, 0 skipped** (baselines ON); refcv7 `tests/test_launch_gate.py`
   **125 passed + 1 failed, identical to the tip baseline** (the environmental CRLF sha pin, §2).
4. **F7 and F8 — found by this gate once nav was built AND fed — are CLOSED by the merge, MEASURED:** the compliant
   S-W smoke now **PASSES G-LIVE with no acknowledgement**, every operative nav leaf reached (4/4 leaf modules, 6/6
   tensors), the trainer's own `grad_reach.json` agreeing (nothing unreached), and the zero-init
   `nav.layer_proj.operative` is OFF 0.0 in the S-W smoke checkpoint (absmax 9.8e-5 after 4 steps); the F7 pins are flipped, with the arm `v6_nav_unbound_wm` and a source mutation of the
   three WM-loss call sites each FAILING G-LIVE naming exactly the operative nav leaves. The trainer's own
   `--dry-run` returns **0 with nav on** (F8; the nav-off control too). **Dead flags: none** among all 239 dests; no
   declared-not-built lever left.
5. **Master Mind must integrate / PI decides:** the 3 gate files (full files, or `code/diffs_vs_merge_copy_current/`
   onto the CURRENT merge copy); `@strict_fields` on the 4 v6 config classes (G-HYG FAILS); RNG/data position in the
   v6 checkpoint (G-CKPT FAILS); `v6_chain.py`'s S-T command; a G-CLOCK reference. PI: D1 `--tac-op-cond` mode; D3
   `--strategic-off` (REQUIRED here); R6 vs `--uplink ema`; o11/o13/o14 not zeroed in S-T.

## 1. What was built (repo paths; full files in `code/fix/`, LF like the tip blobs)
| file | what |
|---|---|
| `stack/scripts/launch_gate.py` | `PROFILES["v7f"]` (family v6, stages devbox=rehearsal / thor=real / all), the v6 jobs `job_model_v6` (G-HYG, G-DVB, G-EVAL), `job_smoke_v6` (G-LIVE run 1; G-CKPT runs 2 = resume, 3 = uninterrupted reference), `job_clock_v6` (ERROR), `JOB_FUNCS_V6`; the rehearsal machinery (declared tiny overlay, synthetic corpus seam **incl. a synthetic nav label source served through the trainer's real join + `NavEmitter`**, a trained tiny S-W predecessor **with nav** for the S-T ladder seam, data-lever departures, preflight classification); `v6_profile_rules` (R1–R6, SEL-1; stage-dependent rows; re-derived in `finalize`); `finalize` refuses REHEARSAL evidence by name and records each instrument's rehearsal verdict in the token; `run --rehearsal {auto,on,off}`. refc code paths untouched (the family dispatch is one lookup). Nav-update changes: `V7F_REHEARSAL_NAV_LABELS`, the nav seam in `v6_corpus_seam`, the pre-nav departure REMOVED, the behaviour probe restructured (full forward + a forward without the nav keys), `_V7F_OPEN_ITEMS = ()`, arm `v6_nav_not_mapped`, the predecessor's argv recorded (`gate_predecessor_argv.json`), G-LIVE reports `live_top_modules` and per-leaf `leaf_live` [tensors with a gradient, tensors]. F7/F8 update: arm `v6_nav_unbound_wm` (swaps `train_v6_staged._NavBoundPredictor` for the bare predictor; if the trainer has no `_NavBoundPredictor` the arm records an exit, so it can never pass). |
| `stack/tanitad/train/declared_vs_built_v6.py` | the merge's union module VERBATIM + `GATE_TIME`/`register_gate` + **232 gate-time entries** (the tip's 231 dests + R6's `strategic_off`) + `check_all` + `kinds_census`; `_run` gains `include_gate_levers` (default False) so the trainer's `check` (build time) and `check_v6` (R3 path) run exactly the merge's 7 levers — **unchanged**. `uncovered_v6(parser) == []` on the merged trainer (239 dests). Nav update: `_c_nav_cond`'s docstring records the fix (the reader stays as the guard). |
| `stack/tests/test_launch_gate_v7f.py` | **52 tests**: profile + argv rules (R1–R6, SEL-1, stage rows), finalize (rehearsal refusal, re-derived rules, **no open items ⇒ PI-DECISION on D1**, a re-opened item still blocks), the registry (full coverage, a NEW flag refused, the nav reader both ways, the trainer-side checks unchanged), the positive rehearsal (G-DVB **PASS**, the predecessor carries nav; **S-W G-LIVE PASS with every operative nav leaf reached**), every arm (incl. **`v6_nav_not_mapped` on S-W and S-T**, **`v6_nav_unbound_wm`**, and **two source mutations**: the nav mapping line, the three WM-loss call sites), two MEASURED open findings (uplink ema, o14 in S-T). The two F7 pins (the finding test, the acknowledged-PASS test) are RETIRED into the positive + the arm + the mutation. |

## 2. Test counts (all CPU, `CUDA_VISIBLE_DEVICES=-1`)
| run | result | log |
|---|---|---|
| `test_launch_gate_v7f.py` on the merged tree (nav mapped, F7 + F8 fixed) | **52 passed** | `raw/pytest_v7f_on_merge_final.txt` |
| the merge's 7 DVB-related files + `test_launch_gate_v7f.py`, bit-identity baselines ON (the coordinator's sets: 10 failed / 223 passed before the nav update; 2 failed / 240 passed after the F7/F8 fix, both failures my F7 pins) | **243 passed, 0 skipped** | `raw/pytest_merge_dvb_related_plus_gate_navfixed.txt` |
| the merge's 7 DVB-related files alone, baselines ON | **191 passed, 0 skipped** | `raw/pytest_merge_dvb_related_after_with_baselines.txt` |
| the same, baselines OFF | 178 passed, 13 skipped (the baseline-gated tests) | `raw/pytest_merge_dvb_related_after.txt` |
| refcv7 `test_launch_gate.py` — tip snapshot, BEFORE any change | 125 passed, 1 failed | `raw/tip_c36b6ddd/pytest_launch_gate_baseline_tip.txt` |
| refcv7 `test_launch_gate.py` — merged tree, AFTER | **125 passed, 1 failed — the same test** | `raw/pytest_launch_gate_refcv7_on_merge_after.txt` |

The 7 files: `test_v7f_r1r4.py`, `test_tactical_label_reach_v6.py`, `test_v7f_r2_nav_fixes.py`,
`test_v7f_r6_strategic_off.py`, `test_nav_v6stack.py`, `test_v6_stage_init_introduction.py`,
`test_v6_effective_weights.py`. The 1 refcv7 failure is ENVIRONMENTAL, MEASURED:
`test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile` pins sha256 `fc1214e5…` of
`stack/ops/launch_gate_thor_env_failures.json`; the git blob at `c36b6ddd` hashes to `fc1214e5…`, the CRLF checkout
(tipsnap / working copies) to `9b6aa814…`. Same failure before and after.

**Regression arms — each FAILS through the gate** (`ctx.arm`, the child's job functions):
| check | arm | how it goes RED | test |
|---|---|---|---|
| G-HYG | `v6_undeclared_attribute` | the walker names `cfg.encoder.gate_arm_undeclared_lever` | `test_ARM_v6_undeclared_attribute_FAILS_G_HYG_by_name` |
| G-DVB | **`v6_nav_not_mapped`** (S-W) | the config handed to `V6Stack` has `nav_cond` False: both nav reads (`declared True but BUILT False`) + the behaviour probe (a nav-less forward RUNS) | `test_ARM_v6_nav_not_mapped_FAILS_G_DVB_on_S_W_…` |
| G-DVB | **`v6_nav_not_mapped`** (S-T) | the nav-carrying predecessor is REFUSED by the trainer's ladder seam (`nav.*` unexpected) | `test_ARM_v6_nav_not_mapped_on_S_T_is_REFUSED_at_the_ladder_seam` |
| G-DVB | **source mutation** | the mapping line deleted from `train_v6_staged.py` (compiled under its own path) → both nav reads + the behaviour probe | `test_ARM_SOURCE_deleting_the_merges_nav_mapping_line_FAILS_G_DVB` |
| G-DVB | `v6_candidates_not_built` | registry: `--n-candidates: declared 1 but BUILT 8`; the gate's probe: the forward RAISES | `test_ARM_G_DVB_each_FAILS_naming_its_defect[…]` |
| G-DVB | `missing_dvb_v6_module` | "declared_vs_built_v6 is not importable" | idem |
| G-DVB | `v6_uncovered_flag` | an unregistered parser flag is named by `uncovered_v6` | idem |
| G-LIVE | `v6_strategic_leaf_trainable` (no `--strategic-off`) | R6 (a): `vocab_str.table.weight` CHANGED; `.grad` populated reported ≥ 1 | `test_ARM_…_FAILS_G_LIVE_on_R6_bit_identity` |
| G-LIVE | `v6_strategic_leaf_trainable` (with `--strategic-off`) | no gradient reaches `layer_str` (MEASURED) → the dead-trainable-leaf rule | `test_ARM_…_under_strategic_off_FAILS_as_a_dead_trainable_leaf` |
| G-LIVE | `v6_drop_term` | declared term `t1` absent from every step | `test_ARM_v6_drop_term_FAILS_G_LIVE_naming_the_term` |
| G-LIVE | **`v6_nav_unbound_wm`** (S-W; F7 re-introduced) | `_NavBoundPredictor` → the bare predictor: the 4 operative nav leaf modules (6 tensors) get ZERO gradient, named; the trainer's `grad_reach.json` names the same 6 | `test_ARM_v6_nav_unbound_wm_reintroduces_F7_and_FAILS_G_LIVE_…` |
| G-LIVE | **source mutation** (S-W; F7 re-introduced) | the three WM-loss call sites handed `stack.predictor_op` again in the trainer SOURCE → the same named FAIL | `test_ARM_SOURCE_unbinding_the_three_WM_loss_call_sites_FAILS_G_LIVE` |
| G-CKPT | `v6_resume_drops_opt` | `resume_opt_state_vs_saved` differs | `test_ARM_v6_resume_drops_opt_FAILS_G_CKPT_on_the_optimizer` |
| G-EVAL | `v6_loader_drops_vocab_version` | the strict rebuild fails (v6.0 vocabulary) | `test_ARM_v6_loader_drops_vocab_version_FAILS_G_EVAL` |
| G-CLOCK | — | `job_clock_v6` is ERROR, never PASS | `test_G_CLOCK_v6_is_ERROR_never_a_pass_until_it_is_built` |
| finalize | a re-opened open item | INCOMPLETE, the item named | `test_an_OPEN_ITEM_still_blocks_a_v7f_PASS` |
The shared arms (`missing_hygiene_module`, `venv_lacks_module`) reach the v6 G-HYG through the shared helpers.

## 3. The dev-box rehearsal, end to end through the CLI (merged tree, nav mapped, F7/F8 fixed; `raw/merge_rehearsal_{st,sw}/`)
`launch_gate.py run --profile v7f --stage devbox --cpu-only …` on the merge-compliant S-T argv (and the S-W one):
token **INCOMPLETE** (rc 2) — every check MISSING because its evidence is REHEARSAL; **no OPEN ITEM line any more** —
carrying what each instrument said:
| check | S-T (merge-compliant) | S-W | what it measured |
|---|---|---|---|
| G-HYG | FAIL | FAIL | ONLY the 4 open config classes (EncoderConfig, PredictorConfig, ReadoutConfig, V6Config); import closure clean |
| G-DVB | **PASS** (was FAIL) | **PASS** (was FAIL) | 0 mismatches; coverage 239/239; nav BUILT and consumed (a nav-less forward raises `NavTokenMissing`); every R1/R3/R4/R6 lever BUILT as declared; stage table S-T `layer_tac, planner` / S-W `aux, encoder, predictor_op, readout` ✓; positive control live |
| G-EVAL | PASS | PASS | strict rebuild via `load_v6_from_ck` (with nav); state dict, V6Config and every forward output identical; negative control moves outputs |
| G-LIVE | PASS | **PASS** (was FAIL on F7) | S-T: `plan`/`seam`/`t1` present+finite on 100/100 steps, each reaches trainable params, 81 trainable tensors / 43 leaf modules / 0 dead (nav's tactical leaves live), **57 `layer_str` tensors bit-identical, `.grad` populated on 0**, the trainer's `grad_reach.json` agrees. S-W: o1/o2/o3/o5/o6 on 100/100 steps, 86 tensors / 45 leaf modules / 0 dead — **every operative nav leaf live** (`nav.embed` 1/1, `nav.arg_proj` 2/2, `nav.gate` 1/1, `nav.layer_proj.operative` 2/2), the trainer's `grad_reach.json` names nothing unreached; R6 holds (57 tensors bit-identical, `.grad` on 0) |
| G-CKPT | FAIL | FAIL | save→load exact; S-T: the resumed run drew `[[1,5],[1,23]],[[2,10],[2,7]]` = the uninterrupted run's FIRST batches, not its steps 101–102 `[[1,21],[1,10]],[[1,6],[1,18]]`; 101 tensors differ at the end (S-W: 86) |
Recorded departures (never hidden): tiny overlay (frame 64×128, enc 32×1, pred 32×1, d_tac 32, d_str 16, window 3,
batch 2 — the readout CELL grid kept); `--v2-cache` → an EMPTY dir; host paths dropped; **`--nav-labels` → the named
synthetic source** (3 records, one per synthetic clip, `LabelManifest` md5 `gate-rehearsal-synthetic`); S-T ladder
seam = a trained 1-step tiny S-W predecessor WITH nav + the recorded INCONCLUSIVE override; smoke: R1/R3 data levers
removed; build: preflight problems naming dropped paths classified expected.

## 4. Findings (each MEASURED; the gate catches every one)
| # | finding | evidence | owner |
|---|---|---|---|
| F1 | **CLOSED** — `--nav-cond` was DECLARED-NOT-BUILT through the CLI (tip + first merge); FIXED in the merge (one mapping line); G-DVB now PASSES and the arm/mutation that removes the line FAILS | history: `raw/probe_navcond_merge.out.txt` (first merge), `raw/tip_c36b6ddd/probe_navcond_tip.out.txt`; now: `raw/merge_rehearsal_*/G-DVB.json` | done (merge) |
| F2 | G-HYG: EncoderConfig / PredictorConfig / ReadoutConfig / V6Config accept undeclared attributes | G-HYG evidence | model owner (`@strict_fields`) |
| F3 | G-CKPT: a resume replays the sampler/RNG stream; resumed ≠ uninterrupted | G-CKPT evidence | v6 trainer owner |
| F4 | R6 + `--uplink ema`: `stack.ema_update()` moves 3 `ema_adapter_str.*` tensors every S-T step (EMA of a frozen source, float rounding) | `raw/probe_uplink_ema_R6.out.txt`, test `TIP_FINDING_uplink_ema…` | PI (R6 scope) / trainer (skip EMA of frozen layers) |
| F5 | `for_stage("S-T")` keeps o11/o13/o14: `--w-o14 1.0` in S-T is audited `TRAINS` by the trainer's own table and has NO autograd graph | `raw/probe_o14_in_ST.out.txt`, test `TIP_FINDING_o14…` | trainer owner (zero or refuse in S-T) |
| F7 | **CLOSED by the merge** — R2's operative nav channel trained nothing: in S-W, O1 (`stage_a_losses`) and O2/O3/O5 (`rollout_transitions`) called `predictor_op` WITHOUT `nav_cond`, so the six `predictor_op` nav tensors got NO gradient (gate per-step grads AND the trainer's `grad_reach.json`), S-T froze them, and `nav.layer_proj.operative` (zero-init) was **0.0 exactly** in the S-W, predecessor and S-T checkpoints. FIX (merge): `_NavBoundPredictor` binds `stack.nav(..., "operative")` to the predictor those three call sites roll (nav off = the same object). NOW MEASURED: S-W G-LIVE PASS with every operative nav leaf live, nothing unreached in `grad_reach.json`, `nav.layer_proj.operative` absmax 9.8e-5 after the 4-step S-W smoke. Guarded by the arm `v6_nav_unbound_wm` and a source mutation (each FAILS naming the 4 leaf modules). ⚠️ The S-T REHEARSAL's 1-step tiny predecessor still carries `nav.layer_proj.operative` = 0.0: the FiLM `to_scale_shift` it feeds is zero-init too, so step 1 cannot reach it (the coordinator's "waking" note) — a property of a 1-step predecessor, not of a real S-W run | before: `raw/probe_nav_operative_inert.json`; after: `raw/probe_nav_operative_after_F7_fix.json`, `raw/merge_rehearsal_sw/G-LIVE.json`, tests `positive_S_W_G_LIVE_PASSES_…`, `ARM_v6_nav_unbound_wm_…`, `ARM_SOURCE_unbinding_…` | done (merge) |
| F8 | **CLOSED by the merge** — the trainer's own `--dry-run` RAISED `NavTokenMissing` on every nav launch (`synthetic_train_batch` emitted no nav keys). FIX (merge): nav keys drawn LAST when the stack has nav. NOW MEASURED: `--dry-run` returns **0 with nav on** (and 0 on the nav-off control); both dry-run phases of the dynamic trace return 0 | before: `raw/probe_dryrun_nav.out.txt`; after: `raw/probe_dryrun_nav_after_F8_fix.out.txt`, `raw/dynamic_reads_merge.json` phases | done (merge) |
| F6 | instrument lessons fixed BEFORE banking: a count-based DVB control reads blind when the lever is already mismatched; a probe that swallowed a raising forward; a negative control upstream of a zero-init layer moved 0/52 outputs; a 2×2 rehearsal cell grid masked every O3 cell; **a nav-less rehearsal smoke hid F7** (hence the synthetic nav seam: nav must be built AND fed to be judged) | this session | — |

## 5. What the siblings' flags need (all now IN the merge) — and every future flag
| flag | registry entry | G-LIVE | profile |
|---|---|---|---|
| `--nav-cond` (R2) | `built`, gate-time (`_c_nav_cond`: cfg field + module) | the S-W smoke runs WITH nav (synthetic records): every operative nav leaf must get a gradient (F7, fixed) | REQUIRED; `--i-know-this-arm-predates-nav` REFUSED |
| `--tac-op-cond` (R4) | `built`, trainer-side (merge) | its port trains via `plan` (reach/leaf checks) | S-T REQUIRED, `off` refused, MODE in `pi_pending_values` (D1) |
| `--max-speed-input-v6` (R1a) | `built`, trainer-side (merge) | removed in the rehearsal smoke (needs the sidecar): Thor | S-T REQUIRED |
| `--plan-vmax-cap` (R1b) | `built`, trainer-side (merge) | idem; ⚠️ its inference-only property is NOT checked yet | S-T REQUIRED |
| `--speed-max-sidecar-v6` (R1) | `data` (merge) | — | S-T REQUIRED; `meta_required_flags` binds its `.meta.json` |
| `--w-tac-label-all` (R3) | `loss`, needs weights (merge) | `V6_TERM_KEYS["w_tac_label_all"] = "tac_label_all"` | S-T `> 0` REQUIRED; `--goal-multilabel` REQUIRED; `--goal-factored` REFUSED |
| `--tac-goal-negatives`, `--cot-negative-sidecar` (R3) | `data` (merge) | — | — |
| `--strategic-off` (R6) | `built`, gate-time | R6 bit-identity holds with it | REQUIRED; `--tac-goal-cond` REFUSED |
A FUTURE flag: (1) a registry entry (`register`/`register_v6` trainer-side, or `register_gate`) — else `uncovered_v6`
FAILS G-DVB; (2) a new loss term → the trainer's effective-weight row AND `V6_TERM_KEYS`; a new `v6_loss_step`
argument → `V6_LOSS_KNOBS`; (3) a ruled value → `required_on_v6(_stage)`/`required_values`, an open one →
`pi_pending_values`; (4) S-T-only → `V6_SW_REFUSED`; host data → `V7F_REHEARSAL_DROP` / `V7F_SMOKE_DATA_LEVERS`, or a
synthetic source served through the trainer's own reader (the nav pattern) when the lever must be judged live.

## 6. NOT done (stated, not narrowed)
* **G-CLOCK (v6) not implemented** — `job_clock_v6` is ERROR; needs an independent clock reference for the v7F
  cache first. R3 (labels read at a window time) makes it load-bearing.
* **G-SUITE not run** (heavy; this box is CPU-only by rule); the v7f profile uses the shared run-mode `job_suite`.
* **The Thor stage never ran** (no data here): the non-rehearsal paths of `job_model_v6`/`job_smoke_v6` (real
  corpus, the preflight on real paths, the sentinel mapping, real nav labels) are **UNVERIFIED**.
* The rehearsal exercises nav through the real join/emitter on SYNTHETIC records (the label CONTENT and its
  `allow_oracle_nav` policy are Thor's); it cannot exercise R1/R3 terms (no joinable sidecar/labels); G-EVAL's fixed
  batch is the trainer's `synthetic_batch`; `--plan-vmax-cap`'s training-inactive / eval-active property is not
  checked by G-LIVE.
* F7 and F8 were REPORTED by the gate and FIXED by the merge (trainer code is not the gate's to change); the gate
  does not special-case nav: the generic dead-leaf rule + the trainer's `grad_reach.json` control caught F7.
* The S-T rehearsal's 1-step tiny predecessor cannot wake the zero-init operative nav path (FiLM `to_scale_shift`
  and `nav.layer_proj.operative` are both zero-init); a real S-W predecessor is thousands of steps.
* The full registry is NOT wired into the trainer (gate-time only, by design — an integration decision).
* Not re-run: the full 107-file v6 suite with my module (only the 7 DVB-related files; they are the ones importing
  it).
* Environment: my working trees are `C:/Users/Admin/v7f_gate/tree` (tip, SUPERSEDED) and `…/tree_m` (merge copy +
  the coordinator's nav fix, synced from `C:/Users/Admin/v7f_merge/`, which I did not edit) — nested one level so
  the HMAC test key (`…/v7f_gate/tmp_keys/`) stays inside my workspace but OUTSIDE the code tree the gate refuses
  keys in. No key was written anywhere else; the default `~/.config/tanitad/launch_gate.key` was never touched.

## 7. G-DVB coverage table — every flag (the inventory's 232 at `c36b6ddd` + the merge's 8 = 240 option rows, 239 dests)
Method (`raw/coverage_table.json`, regenerated after the nav, F7 and F8 fixes): kind = `declared_vs_built_v6.REGISTRY_V6`; static
= AST reads of the ARGS object in the trainer + the helpers it hands the namespace to; dynamic = every
`argparse.Namespace` read on the CPU rehearsal (S-W/S-T jobs WITH nav + both dry-runs, which return 0 since the F8 fix), argparse internals, `effective_weights.explicit_dests` (a whole-namespace re-parse) and the
gate's own reads excluded. **FINDING rows are now READ from the rehearsal's G-DVB evidence** (never hard-coded):
`dvb_findings_from_rehearsal == {}`.

| status | n | meaning |
|---|---|---|
| COVERED-CPU | 94 | a `built`/`loss` reader on the BUILT stack ran in G-DVB on the CPU rehearsal (incl. `nav_cond`, now built — its row carries the history) |
| COVERED-CPU (named guard) | 62 | `elsewhere`: its named guard (G-LIVE term/knob capture, a trainer guard) ran on CPU |
| GUARD-NOT-EXERCISED-ON-CPU | 7 | `o14_shuffle_targets o7_model o9_mask_frac o9_momentum o9_neighbour_k psg_enc_only psg_eval_every` — their branch is off in the rehearsal |
| NOT-COVERABLE-ON-CPU (data) | 23 dests / 24 rows | host paths, content-bound on the launch host (list in the table) |
| NON-LEVER (runtime / record) | 44 / 8 | not model levers; each entry names its role |
| REFUSED-BY-TRAINER | 1 | `--v2-val-cache` (train() raises: P4-5) |
| FINDING: DECLARED-NOT-BUILT | **0** | (was 1: `--nav-cond`, fixed in the merge) |
| DEAD (read nowhere) | **0** | every dest has an args read site; the 18 never consumed on the exercised paths all live in unexercised branches: `dump_seam_plan_degenerate ema_decay_end ema_decay_start no_amp o14_shuffle_targets o7_model o9_mask_frac o9_momentum o9_neighbour_k obs_monitor_dims obs_monitor_window predates_nav psg_enc_only psg_eval_every require_parity t3_floor trunk_anchor_model v2_lru` (`nav_semantics` is now consumed by `train`; `predates_nav` is no longer passed by any rehearsal) |

## 8. Deliverable manifest
| artifact | where | copies |
|---|---|---|
| `DESIGN.md`, `RESULT.md` | `repo:products/P4-training-pipelines/2026-09-27-v7f-refav1-state/v7f_gate/` | repo (staged) |
| `code/fix/stack/scripts/launch_gate.py` (sha256 `f1534182a874e082…`) | repo (staged) + `C:/Users/Admin/v7f_gate/tree_m/stack/scripts/` | 2 |
| `code/fix/stack/tanitad/train/declared_vs_built_v6.py` (`0df4bf43d28ecfd1…`) | repo (staged) + `…/tree_m/stack/tanitad/train/` | 2 |
| `code/fix/stack/tests/test_launch_gate_v7f.py` (`428a647b5340a0bf…`) | repo (staged) + `…/tree_m/stack/tests/` | 2 |
| `code/diffs_vs_merge_copy_current/*.diff` (2: `launch_gate.py`, `test_launch_gate_v7f.py` vs the coordinator's CURRENT merge copy `C:/Users/Admin/v7f_merge/stack/…`, which holds my previous banked files; `declared_vs_built_v6.py` is IDENTICAL to the merge copy's — no diff, recorded in `bank_proof.json`), `code/diffs_vs_c36b6ddd/*.diff` (3), `code/diffs_vs_v7f_merge/*.diff` (1, vs the merge PACKAGE's registry -- which since 16:56 carries my PREVIOUS registry, so it equals the merge-copy diff for that file) — each PROVEN to `git -c core.autocrlf=false apply` and reproduce the banked file byte-for-byte (`raw/bank_proof.json`) | repo (staged) | 1 (regenerable: `code/tools/bank.py`) |
| `code/tools/*.py` (census / trace / consumers / table / bank / rehearsal-bank / assembly generators + the two new probes) | repo (staged) + `C:/Users/Admin/v7f_gate/work/` | 2 |
| `raw/` logs, rehearsal evidence + `rehearsal_verdicts*.json` (token SUMMARIES — the tokens themselves are NOT banked), probes (F7: `probe_nav_operative_inert.json` before / `probe_nav_operative_after_F7_fix.json` after; F8: `probe_dryrun_nav.out.txt` before / `probe_dryrun_nav_after_F8_fix.out.txt` after), census, coverage table, `staging_verification.json` | repo (staged) | 1 |
| full gate run dirs (smoke dirs, arm runs, the pre-fix and tip-era evidence), the tip-era working tree `C:/Users/Admin/v7f_gate/tree/` | `C:/Users/Admin/v7f_gate/gate_runs/`, `…/tree/` | **C: only** (scratch; everything quoted is copied into `raw/`) |
| the rehearsal TOKENS (INCOMPLETE, test-key-signed, fake commit `7f7f…`) | `C:/Users/Admin/v7f_gate/gate_runs/*/INCOMPLETE_*.json` | **C: only, by design** |
| the HMAC TEST key | `C:/Users/Admin/v7f_gate/tmp_keys/gate.key` | **C: only, by design** (never in the repo; the default key path never touched) |

### Full table
| flag | dest | default | inventory class | DVB kind | status | static read sites | CPU consumers |
|---|---|---|---|---|---|---|---|
| `--v2-val-cache` | `v2_val_cache` | `[]` | DATA | data | **REFUSED-BY-TRAINER** | T:train | train |
| `--a-max` | `a_max` | `4.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--adapter-hidden` | `adapter_hidden` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--agent-slots` | `agent_slots` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--anchor-goal` | `anchor_goal` | `none` | MODEL | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--anchor-table` | `anchor_table` | `None` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--d-goal-embed` | `d_goal_embed` | `128` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-str` | `d_str` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-t2-hidden` | `d_t2_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-t2-proj` | `d_t2_proj` | `128` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--d-tac` | `d_tac` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-hidden` | `diffusion_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-noise-rho` | `diffusion_noise_rho` | `0.9` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-sigma-a` | `diffusion_sigma_a` | `2.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-sigma-k` | `diffusion_sigma_k` | `0.1` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--diffusion-steps` | `diffusion_steps` | `4` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--dt` | `dt` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight, T:train | build_stack_from_args, train |
| `--ema-decay` | `ema_decay` | `0.996` | OPTIM | built | **COVERED-CPU** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | build_stack_from_args |
| `--enc-depth` | `enc_depth` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--enc-dim` | `enc_dim` | `384` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--enc-grad-checkpoint` | `enc_grad_checkpoint` | `auto` | OPTIM | built | **COVERED-CPU** | T:build_stack_from_args | resolve_gc |
| `--enc-heads` | `enc_heads` | `6` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-blocks` | `f_blocks` | `3` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-hidden-str` | `f_hidden_str` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--f-hidden-tac` | `f_hidden_tac` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--fallback-calibration` | `fallback_calibration` | `None` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--fallback-roll-k` | `fallback_roll_k` | `10` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--fallback-trigger` | `fallback_trigger` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--frame-h` | `frame_h` | `256` | MODEL | built | **COVERED-CPU** | T:_preflight_subframe, T:build_stack_from_args, T:subframe_desync, geo | frame_from_args, build_stack_from_args |
| `--frame-w` | `frame_w` | `640` | LOSS | built | **COVERED-CPU** | T:_preflight_subframe, T:build_stack_from_args, T:subframe_desync, geo | frame_from_args, build_stack_from_args |
| `--goal-cat-args` | `goal_cat_args` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--goal-factored` | `goal_factored` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:_term_precondition, T:build_stack_from_a | _preflight_tac_label_all, _term_precondition, build_stack_fr |
| `--goal-multilabel` | `goal_multilabel` | `False` | DATA | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:_term_precondition, T:build_stack_from_a | _preflight_tac_label_all, _term_precondition, build_stack_fr |
| `--horizons` | `horizons` | `[1, 2, 4]` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--in-channels` | `in_channels` | `9` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--kappa-max` | `kappa_max` | `0.2` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--max-speed-input-v6` | `max_speed_input_v6` | `False` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:assert_r1r4_record, T:build_stack_from_args, T:r | _preflight_r1_r4, assert_r1r4_record, build_stack_from_args, |
| `--mpc-lr` | `mpc_lr` | `0.05` | OPTIM | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-refine` | `mpc_refine` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--mpc-roll-k` | `mpc_roll_k` | `0` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-steps` | `mpc_steps` | `3` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-topk` | `mpc_topk` | `2` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-consist` | `mpc_w_consist` | `0.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-goal` | `mpc_w_goal` | `1.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--mpc-w-kin` | `mpc_w_kin` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-agent-slots` | `n_agent_slots` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-anchors` | `n_anchors` | `256` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-candidates` | `n_candidates` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-lat-bins` | `n_lat_bins` | `16` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-registers` | `n_registers` | `4` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--n-slot-queries` | `n_slot_queries` | `100` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--nav-cond` | `nav_cond` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--newest-frame-only` | `newest_frame_only` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, train_v58f_unicycle_head:build_train_episodes | build_stack_from_args |
| `--no-isolate-interp` | `no_isolate_interp` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--no-isolate-planner` | `no_isolate_planner` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:build_stack_from_args, T:preflight | _preflight_tac_label_all, build_stack_from_args, preflight |
| `--no-isolate-uplink` | `no_isolate_uplink` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--o5-target` | `o5_target` | `live` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--param-budget` | `param_budget` | `300000000` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--patch` | `patch` | `16` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--per-layer-encoders` | `per_layer_encoders` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args |
| `--plan-steps` | `plan_steps` | `60` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--plan-vmax-cap` | `plan_vmax_cap` | `False` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:assert_r1r4_record, T:build_stack_from_args, T:d | _preflight_r1_r4, assert_r1r4_record, build_stack_from_args, |
| `--plan-wta-eps` | `plan_wta_eps` | `0.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-depth` | `pred_depth` | `6` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-dim` | `pred_dim` | `768` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-heads` | `pred_heads` | `12` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--pred-modern` | `pred_modern` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--proposals` | `proposals` | `query` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--readout-dim` | `readout_dim` | `128` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--readout-grid` | `readout_grid` | `4` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--readout-grid-w` | `readout_grid_w` | `None` | LOSS | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--selector` | `selector` | `none` | UNCLASSIFIED | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--selector-mlp-hidden` | `selector_mlp_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--selector-tau-m` | `selector_tau_m` | `1.0` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-free-dims` | `sigreg_free_dims` | `0` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-slices` | `sigreg_slices` | `512` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--sigreg-subspaces` | `sigreg_subspaces` | `1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-depth` | `slot_depth` | `3` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-heads` | `slot_heads` | `8` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-hidden` | `slot_hidden` | `256` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--slot-src` | `slot_src` | `cells` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--strategic-off` | `strategic_off` | `False` | (merge) | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--t2-contrastive` | `t2_contrastive` | `False` | UNCLASSIFIED | built | **COVERED-CPU** | T:_term_precondition, T:build_stack_from_args, T:preflight | _term_precondition, build_stack_from_args, preflight |
| `--t2-tau` | `t2_tau` | `0.1` | UNCLASSIFIED | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--tac-goal-cond` | `tac_goal_cond` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args, T:preflight | build_stack_from_args, preflight |
| `--tac-op-cond` | `tac_op_cond` | `off` | (merge) | built | **COVERED-CPU** | T:_preflight_r1_r4, T:build_stack_from_args, T:r1r4_config_block | _preflight_r1_r4, build_stack_from_args, r1r4_config_block |
| `--tac-vocab-version` | `tac_vocab_version` | `v7.0` | MODEL | built | **COVERED-CPU** | T:_preflight_tac_label_all, T:build_stack_from_args | _preflight_tac_label_all, build_stack_from_args |
| `--uplink` | `uplink` | `stopgrad` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--vit5-encoder` | `vit5_encoder` | `False` | MODEL | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--w-anchor` | `w_anchor` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_preflight_r1_r4, T:_weights_from_args, T:preflight | _preflight_r1_r4, _weights_from_args, preflight |
| `--w-o14` | `w_o14` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:build_stack_from_args, T:train | _weights_from_args, build_stack_from_args, train |
| `--w-select` | `w_select` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-t2-contrast` | `w_t2_contrast` | `0.0` | LOSS | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-tac-label-all` | `w_tac_label_all` | `0.0` | (merge) | loss | **COVERED-CPU** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--window` | `window` | `6` | DATA | built | **COVERED-CPU** | T:build_stack_from_args | build_stack_from_args |
| `--anchor-axis-w` | `anchor_axis_w` | `[1.0, 1.0]` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--anchor-objective` | `anchor_objective` | `metric` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--bptt-truncate` | `bptt_truncate` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--cond-param` | `cond_param` | `steer_accel_v` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--daccel` | `daccel` | `2.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--dkappa` | `dkappa` | `0.02` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--freeze-encoder` | `freeze_encoder` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--freeze-readout` | `freeze_readout` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--grad-checkpoint` | `grad_checkpoint` | `False` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:resolve_gc | resolve_gc |
| `--lambda-plan` | `lambda_plan` | `None` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:preflight, T:resolve_lambda_plan, T:v6_weight_specs | resolve_lambda_plan, v6_weight_specs |
| `--o11-k` | `o11_k` | `6` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o11-negs` | `o11_negs` | `1` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o11-tau` | `o11_tau` | `1.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o13-k` | `o13_k` | `4` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o13-seed` | `o13_seed` | `1300` | RUNTIME | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o14-k` | `o14_k` | `4` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run |
| `--o14-mode` | `o14_mode` | `fut` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run |
| `--o1-detach-encoder` | `o1_detach_encoder` | `False` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--o1-k` | `o1_k` | `10` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o1-stopgrad-factual` | `o1_stopgrad_factual` | `False` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--o2-tau-s` | `o2_tau_s` | `2.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-band-rows` | `o3_band_rows` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-block-h` | `o3_block_h` | `2` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-block-w` | `o3_block_w` | `2` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-blocks` | `o3_blocks` | `2` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o3-mode` | `o3_mode` | `action` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-form` | `o5_form` | `l1` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-k` | `o5_k` | `20` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--o5-mode` | `o5_mode` | `uniform` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o5-target-crop` | `o5_target_crop` | `0.0` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:build_stack_from_args, T:train | build_stack_from_args, train |
| `--o6-innovation` | `o6_innovation` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--o6-innovation-shuffle` | `o6_innovation_shuffle` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rand-daccel-max` | `rand_daccel_max` | `3.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rand-dkappa-max` | `rand_dkappa_max` | `0.05` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--rollout-grad-checkpoint` | `rollout_grad_checkpoint` | `auto` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:train | resolve_gc |
| `--s1-multi-k` | `s1_multi_k` | `2` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:preflight, T:train | preflight |
| `--sigreg-accum` | `sigreg_accum` | `1` | OPTIM | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--stage` | `stage` | `None` | RUNTIME | elsewhere | **COVERED-CPU (named guard)** | T:_declared_freeze_preflight, T:_preflight_effective_weights, T:_prefl | _declared_freeze_preflight, _preflight_effective_weights, _p |
| `--t2-negative` | `t2_negative` | `lane_mirror` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--t2-positive` | `t2_positive` | `photometric` | MODEL | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--t5-lag` | `t5_lag` | `0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:preflight, T:train | preflight, train |
| `--t5-pairs` | `t5_pairs` | `False` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:_term_precondition, T:preflight, T:train | _term_precondition, train |
| `--t5-w-kappa` | `t5_w_kappa` | `1.0` | UNCLASSIFIED | elsewhere | **COVERED-CPU (named guard)** | T:dry_run, T:train | dry_run, train |
| `--w-o10-psg` | `w_o10_psg` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train, T:v6_weight_specs | train, v6_weight_specs |
| `--w-o11-cf` | `w_o11_cf` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o13-ego` | `w_o13_ego` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-ctrl` | `w_o1_ctrl` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-fact` | `w_o1_fact` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o1-scene` | `w_o1_scene` | `0.3` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o2` | `w_o2` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o3` | `w_o3` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o5` | `w_o5` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o6` | `w_o6` | `0.1` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-o7-distill` | `w_o7_distill` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-o8-pixel` | `w_o8_pixel` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-o9-ema` | `w_o9_ema` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:train | train |
| `--w-s1` | `w_s1` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args | _weights_from_args |
| `--w-s1-multi` | `w_s1_multi` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-s2-goal` | `w_s2_goal` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-t1` | `w_t1` | `1.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args |
| `--w-t5-consist` | `w_t5_consist` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:_weights_from_args, T:preflight | _weights_from_args, preflight |
| `--w-trunk-anchor` | `w_trunk_anchor` | `0.0` | LOSS | elsewhere | **COVERED-CPU (named guard)** | T:assert_trunk_anchor_preflight, T:build_trunk_anchor | assert_trunk_anchor_preflight, build_trunk_anchor |
| `--o14-shuffle-targets` | `o14_shuffle_targets` | `False` | MODEL | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o7-model` | `o7_model` | `facebook/dinov3-vitl16-p` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-mask-frac` | `o9_mask_frac` | `0.5` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-momentum` | `o9_momentum` | `0.996` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--o9-neighbour-k` | `o9_neighbour_k` | `0` | MODEL | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--psg-enc-only` | `psg_enc_only` | `False` | UNCLASSIFIED | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--psg-eval-every` | `psg_eval_every` | `3` | RUNTIME | elsewhere | **GUARD-NOT-EXERCISED-ON-CPU** | T:train | - |
| `--allow-any-labels` | `allow_any_labels` | `False` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:preflight, T:train | preflight |
| `--allow-eval-clips-in-train` | `allow_eval_clips_in_train` | `False` | OPTIM | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_resolve_eval_exclusion | _eval_excl_key, _resolve_eval_exclusion |
| `--cot-negative-sidecar` | `cot_negative_sidecar` | `None` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_tac_label_all, T:tac_label_policy | _preflight_tac_label_all |
| `--domain-strata` | `domain_strata` | `` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:preflight, T:train | preflight, train |
| `--dump-seam-plan` | `dump_seam_plan` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_seam_dump, T:seam_dump_import_error, T:train | seam_dump_import_error, train |
| `--exclude-eval-clips` | `exclude_eval_clips` | `auto` | OPTIM | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_resolve_eval_exclusion | _eval_excl_key, _resolve_eval_exclusion |
| `--frame-hfov` | `frame_hfov` | `120.0` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | geometry:frame_from_args | frame_from_args |
| `--gate-probes` | `gate_probes` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--init-encoder-from/--enc-init-from` | `init_encoder_from` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:apply_encoder_seed, T:assert_trunk_anchor_preflight | apply_encoder_seed |
| `--init-from` | `init_from` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:apply_encoder_seed, T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--nav-labels` | `nav_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:eval_exclusion_roots, T:label_blob_md5_for_vmax, T | _eval_excl_key, eval_exclusion_roots, preflight, train |
| `--prev-gate` | `prev_gate` | `None` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:dry_run, T:train | dry_run, train |
| `--projection` | `projection` | `cylindrical` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | geometry:frame_from_args | frame_from_args |
| `--psg-labels` | `psg_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:train, T:v6_weight_specs | v6_weight_specs |
| `--require-parity` | `require_parity` | `True` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--no-require-parity` | `require_parity` | `True` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--s2-labels` | `s2_labels` | `None` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_term_precondition, T:dry_run, T:eval_exclusion_ro | _eval_excl_key, _term_precondition, dry_run, eval_exclusion_ |
| `--speed-max-sidecar-v6` | `speed_max_sidecar_v6` | `None` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_r1_r4, T:dry_run, T:r1r4_config_block, T:train | _preflight_r1_r4, dry_run, r1r4_config_block, train |
| `--t3-scores` | `t3_scores` | `` | UNCLASSIFIED | data | **NOT-COVERABLE-ON-CPU (data)** | T:preflight, T:train | preflight, train |
| `--tac-goal-negatives` | `tac_goal_negatives` | `measured` | (merge) | data | **NOT-COVERABLE-ON-CPU (data)** | T:_preflight_tac_label_all, T:tac_label_policy | _preflight_tac_label_all |
| `--trunk-anchor-model` | `trunk_anchor_model` | `facebook/dinov3-vitb16-p` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:build_trunk_anchor | - |
| `--v2-cache` | `v2_cache` | `[]` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | T:_eval_excl_key, T:_preflight_eval_exclusion, T:_resolve_eval_exclusi | _eval_excl_key, _preflight_eval_exclusion, _resolve_eval_exc |
| `--v2-lru` | `v2_lru` | `64` | DATA | data | **NOT-COVERABLE-ON-CPU (data)** | train_v58f_unicycle_head:build_train_episodes | - |
| `--v2-subframe` | `v2_subframe` | `None` | MODEL | data | **NOT-COVERABLE-ON-CPU (data)** | T:subframe_desync, train_flagship_v4:resolve_v2_frames | resolve_v2_frames, subframe_desync |
| `--batch` | `batch` | `16` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_warn_rank_gate_unrulable, T:train, T:x4_monitor_from_args | _warn_rank_gate_unrulable, train, x4_monitor_from_args |
| `--clip` | `clip` | `1.0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:dry_run, T:train | dry_run, train |
| `--device` | `device` | `cuda` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:_run_config, T:train | _run_config, train |
| `--domain-max-amp` | `domain_max_amp` | `20.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--domain-min-stratum` | `domain_min_stratum` | `8` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--domain-tau` | `domain_tau` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--dry-batch` | `dry_batch` | `2` | OPTIM | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--dry-k` | `dry_k` | `12` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--dry-run` | `dry_run` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:_preflight_eval_exclusion, T:_preflight_r1_r4, T:_preflight_tac_labe | _preflight_eval_exclusion, _preflight_r1_r4, _preflight_tac_ |
| `--dry-steps` | `dry_steps` | `2` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run | dry_run |
| `--ema-decay-end` | `ema_decay_end` | `None` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | - |
| `--ema-decay-ramp` | `ema_decay_ramp` | `off` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | build_stack_from_args |
| `--ema-decay-start` | `ema_decay_start` | `0.99` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_stack_from_args, T:dry_run, T:train | - |
| `--eps-per-batch` | `eps_per_batch` | `4` | OPTIM | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--force-rerun` | `force_rerun` | `False` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--log-every` | `log_every` | `50` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--lr` | `lr` | `0.0001` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer | build_trunk_optimizer |
| `--max-horizon` | `max_horizon` | `None` | MODEL | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--nav-semantics` | `nav_semantics` | `t0_constant` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--no-amp` | `no_amp` | `False` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | - |
| `--no-step-ckpts` | `no_step_ckpts` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--o4-alpha` | `o4_alpha` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | train |
| `--o4-floor` | `o4_floor` | `0.25` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--obs-monitor-dims` | `obs_monitor_dims` | `32` | MODEL | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor | - |
| `--obs-monitor-every` | `obs_monitor_every` | `0` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:assert_trunk_anchor_preflight, T:build_observer_monitor | build_observer_monitor |
| `--obs-monitor-window` | `obs_monitor_window` | `256` | DATA | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor | - |
| `--out` | `out` | `None` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run, T:preflight, T:train | dry_run, preflight, train |
| `--print-launch` | `print_launch` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:main | main |
| `--refuse-unreached` | `refuse_unreached` | `False` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:dry_run, T:train | dry_run, train |
| `--resume` | `resume` | `auto` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--save-every` | `save_every` | `1000` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--seed` | `seed` | `0` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:build_observer_monitor, T:dry_run, T:train, T:x4_monitor_from_args | dry_run, train |
| `--spectrum-accum` | `spectrum_accum` | `1` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_warn_rank_gate_unrulable, T:train, T:x4_monitor_from_args | _warn_rank_gate_unrulable, train, x4_monitor_from_args |
| `--spectrum-ci-reps` | `spectrum_ci_reps` | `0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:train, T:x4_monitor_from_args | train, x4_monitor_from_args |
| `--spectrum-every` | `spectrum_every` | `200` | RUNTIME | runtime | **NON-LEVER (runtime)** | T:train | train |
| `--steps` | `steps` | `30000` | OPTIM | runtime | **NON-LEVER (runtime)** | T:_ema_tau_record, T:build_lr_scheduler, T:build_stack_from_args, T:dr | build_lr_scheduler, train |
| `--t3-alpha-end` | `t3_alpha_end` | `1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--t3-alpha-start` | `t3_alpha_start` | `-1.0` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--t3-floor` | `t3_floor` | `0.25` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | - |
| `--t3-warmup-frac` | `t3_warmup_frac` | `0.5` | OPTIM | runtime | **NON-LEVER (runtime)** | T:preflight, T:train | preflight |
| `--trunk-lr-scale` | `trunk_lr_scale` | `1.0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer, T:trunk_lr_factor, T:trunk_lr_split_active | trunk_lr_split_active |
| `--trunk-lr-warmup-steps` | `trunk_lr_warmup_steps` | `0` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer, T:trunk_lr_factor, T:trunk_lr_split_active | trunk_lr_split_active |
| `--wd` | `wd` | `0.05` | OPTIM | runtime | **NON-LEVER (runtime)** | T:build_trunk_optimizer | build_trunk_optimizer |
| `--x4-spectrum-layers` | `x4_spectrum_layers` | `tac,str` | UNCLASSIFIED | runtime | **NON-LEVER (runtime)** | T:x4_monitor_from_args | x4_monitor_from_args |
| `--allow-discarded-weights` | `allow_discarded_weights` | `False` | RUNTIME | record | **NON-LEVER (record)** | T:_preflight_effective_weights, T:effective_weights_stamp | _preflight_effective_weights, effective_weights_stamp |
| `--allow-inconclusive-gate` | `allow_inconclusive_gate` | `False` | RUNTIME | record | **NON-LEVER (record)** | T:dry_run, T:preflight, T:train | preflight, train |
| `--allow-unreached` | `allow_unreached` | `[]` | RUNTIME | record | **NON-LEVER (record)** | T:dry_run, T:train | dry_run, train |
| `--i-know-this-is-the-control-arm` | `control_arm_ack` | `False` | MODEL | record | **NON-LEVER (record)** | T:_preflight_r1_r4, T:main, T:preflight | _preflight_r1_r4, main, preflight |
| `--dump-seam-plan-degenerate` | `dump_seam_plan_degenerate` | `False` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:train | - |
| `--expect-n-trainable` | `expect_n_trainable` | `None` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:_declared_freeze_preflight | _declared_freeze_preflight |
| `--gate-off-reason` | `gate_off_reason` | `` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:dry_run, T:preflight, T:train | preflight, train |
| `--i-know-this-arm-predates-nav` | `predates_nav` | `False` | UNCLASSIFIED | record | **NON-LEVER (record)** | T:preflight | - |
