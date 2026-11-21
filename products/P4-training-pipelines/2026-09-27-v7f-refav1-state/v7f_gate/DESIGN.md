# DESIGN — the `v7f` profile of the BINDING launch gate (`stack/scripts/launch_gate.py`)

**2026-09-27 · TrainingFlyWheel (P4) · launch-gate agent.** Designed first against the origin-tip snapshot
`c36b6ddd`, then **re-pointed (coordinator, ~16:00) to the v7F MERGE** (`…/2026-09-27-v7f-refav1-state/v7f_merge/`:
R1 + R3 + R4 + the three R2 fixes + R6 `--strategic-off`). `file:line` citations are **tip `c36b6ddd`** lines (the
merge shifts them; the FUNCTION names are the anchors and are unchanged). Every claim marked MEASURED has a raw
artifact under `raw/`; see `RESULT.md` for the numbers. **Updated (coordinator, ~17:00): the merge now MAPS
`--nav-cond`** — the rehearsal keeps nav end to end (§2), G-DVB's nav pins are flipped to the fixed state with a
regression arm that removes the mapping, and the open item is closed (§4). **Updated again (~17:40): the merge
FIXED F7** (operative nav untrained in S-W — found by G-LIVE once nav was fed) **and F8** (the trainer's `--dry-run`
crashed on nav): the F7 pins are flipped to the positive with an arm + a mutation that must FAIL (§3 G-LIVE).

## 0. What v7F's first compliant run is (the thing the profile certifies)

PI directive R1–R6 (2026-09-27): **S-W** then **S-T**, strategic layer OFF.

| stage | trains (`v6.py:4567` `STAGE_GROUPS`) | frozen | loss terms in force (`train_v6_staged.py:351-396` `for_stage`) |
|---|---|---|---|
| S-W | `encoder, readout, predictor_op, aux` | `layer_tac, layer_str, planner, interp` | o1, o2, o3, o5, o6 (+ o11/o13/o14 when weighted) |
| S-T | `layer_tac, planner` | `encoder, readout, predictor_op, aux, layer_str, interp` | `plan`, `seam`, `t1` (+ `tac_label_all` with R3; o11/o13/o14 are NOT zeroed in S-T — see G-LIVE) |

S-T's first configuration (the brief + the merge): `--n-candidates 1 --proposals query --selector none --nav-cond
--strategic-off --max-speed-input-v6 --plan-vmax-cap --speed-max-sidecar-v6 … --w-tac-label-all >0
--goal-multilabel --s2-labels … --tac-op-cond {detached|e2e}`.

## 1. Architecture — a trainer FAMILY, not a patched refc path

The refc checks are written against refc's hooks (`RefCV3Model.__init__`, `compute_losses_v3`, `build_optimizer`,
`ResumableEpochSampler`, `_pin_trainer_cfg`, `refcv6_loader.py`); none exist in the v6 trainer. So:

* profile key `family`: absent = today's refc path, **untouched**; `"v6"` = parallel job functions
  (`job_model_v6`, `job_smoke_v6`, `job_clock_v6`) chosen in `cmd_check` by ONE lookup (`JOB_FUNCS_V6`).
  `finalize`, the token, `verify`, `exec`, the data manifest and G-SUITE are SHARED — one definition of the binding.
* every trainer seam the v6 jobs touch is a module attribute (`train_v6_staged.{train, apply_stage_freeze,
  build_trunk_optimizer, v6_loss_step, load_resume, preflight, _print_refusal, _run_config}`,
  `v6.V6Stack.__init__`, `train_flagship4b.FlagshipWindowDataset.__getitem__`,
  `train_v58f_unicycle_head.build_train_episodes`), patched with the refc `_Patcher` and always restored.
* the child loads the trainer as `train_v6_staged_gate` (never the refc module name).
* the gate parses with `build_parser()` **plus** `--i-know-this-is-the-control-arm`, which `main()` adds after it
  (`:11026`) — a `build_parser()`-only parse refuses a legal argv.
* `main()` RETURNS 2 on a preflight refusal (`:11042-11048`, it does not raise): the gate records the printed
  problems (`_print_refusal`) and FAILs with them.

## 2. The stages, and why the dev box can never mint a v7F token

| stage | host | argv | corpus | evidence counts toward a token? |
|---|---|---|---|---|
| `devbox` | CPU | the launch argv + the profile's DECLARED tiny overlay (geometry shrunk; the readout CELL grid kept — it is a lever); host-only paths dropped; `--v2-cache` → an explicitly named EMPTY dir (the preflight requires the flag and lists it) | **synthetic** (`build_train_episodes` → seeded `ToyEpisode`s; parity/geometry binding NOT exercised — recorded) | **NO — REHEARSAL.** Every evidence file carries `rehearsal: {departures}` and `inputs_read["gate:rehearsal"]`; `finalize` rejects it BY NAME; the token (INCOMPLETE) carries each instrument's rehearsal verdict |
| `thor` | the launch host | the launch argv exactly (the smoke changes only `--steps`/`--out`, as refcv7) | the real caches + labels | yes |

Rehearsal-only mechanics (each a recorded departure): S-T trains its own **1-step tiny S-W predecessor** and runs
the real ladder seam on it (`--init-from`, `--prev-gate`, the trainer's recorded INCONCLUSIVE override); the
predecessor carries `--nav-cond` (S-T may not INTRODUCE `nav.*` — `STAGE_MAY_INTRODUCE`; a nav-less predecessor is
refused by the ladder seam, MEASURED); its argv is recorded (`gate_predecessor_argv.json`, evidence
`argv_local.rehearsal_predecessor.sw_argv`). **Nav is KEPT everywhere** (updated after the merge's nav fix):
`--nav-labels` → the named synthetic source `__gate_rehearsal_synthetic_nav__/nav.jsonl.gz`, which the corpus seam
(`v6_corpus_seam`) serves as seeded nav records (one per synthetic clip, a `LabelManifest` marked
`gate-rehearsal-synthetic`) through the trainer's REAL `assert_cache_join` → `NavEmitter` → batch splat; synthetic
clip ids come from the patched `join_clip_ids` for the EMPTY rehearsal cache only. The smoke removes only the
data-fed levers `V7F_SMOKE_DATA_LEVERS` (R1 input + cap + sidecar, R3 weight + labels); the BUILD job keeps every
lever and classifies preflight problems that NAME a dropped flag as expected (recorded, never hidden — any other
problem still refuses).

## 3. The seven checks for `train_v6_staged.py`

### G-HYG — config classes refuse undeclared attributes; the launch venv imports the trainer
* **Asserts (S-W, S-T):** on the config tree the trainer built for this argv (captured at `V6Stack.__init__(cfg)`):
  no undeclared attribute (`config_hygiene.undeclared_attributes`), every config dataclass `is_strict`, the ACTIVE
  probe (an undeclared attribute set on a deep copy must RAISE); plus the static import closure of the trainer
  (shared `import_closure`/`judge_import_closure`).
* **Hooks:** `V6Config` (`v6.py:3871`), `EncoderConfig`/`PredictorConfig`/`ReadoutConfig` (`tanitad/config.py`).
* **MEASURED (tip AND merge): FAILS** — all four classes are open (`is_strict` False). The D-REFCV6-EQUALIZE-DROPPED
  class is open on v7F. Fix (model owner): `@strict_fields` on the four classes.
* **Arm `v6_undeclared_attribute`:** the build sets `cfg.encoder.gate_arm_undeclared_lever` — named by the walker.

### G-DVB — every argv lever equals the BUILT stack's value; every trainer flag has an entry
* **Asserts:** `declared_vs_built_v6.check_all(stack, args, weights_in_force=<the stage's weights>, parser)` == [] —
  EVERY registry entry on the FULLY built and frozen stack, the parser coverage of EVERY dest (`uncovered_v6`), and
  the config-tree walk; the gate's OWN probes, independent of the registry: per-group trainability after the
  trainer's freeze == the gate's LITERAL stage table; a forward on the trainer's own `synthetic_batch` (eval, forked
  RNG) must run and emit a fan of exactly `--n-candidates` (a forward that RAISES is a FAIL); with `--nav-cond`, a
  forward WITHOUT the nav keys must raise `NavTokenMissing` (running = nav not consumed; any other exception = a
  broken build); the registry's positive control (widen `cand_queries` → `--n-candidates` must be reported
  BUILT at the new width — keyed on the VALUE, not a count); the profile's argv rules (§4).
* **The registry** (`stack/tanitad/train/declared_vs_built_v6.py`, ONE module): the merge's union file VERBATIM
  (its 7 levers run in the TRAINER: `check` at build, `check_v6` on the R3 path) **+ 232 GATE-TIME entries**
  (`register_gate`, set `GATE_TIME`) for every other dest — run only by `check_all`, never inside the trainer,
  because the trainer's build-time call site runs before the O5 teacher / O14 head / anchor table / fallback
  calibration are installed. The merged trainer is therefore bit-identical (MEASURED: the merge's 7 DVB-related files 191/191 with my module, baselines ON (after the F7/F8 fixes)).
* **History — MEASURED at the tip AND on the first merge: FAILED on `--nav-cond`** (declared, preflight REQUIRES it,
  never mapped into `V6Config` by `build_stack_from_args`: `stack.cfg.nav_cond` False, `stack.nav` None, a forward
  without a nav token RAN). **FIXED in the merge** (one mapping line, pinned by
  `test_v7f_r2_nav_fixes.py::test_REAL_LAUNCH_*`). **Now MEASURED: PASSES** on the compliant S-T and S-W
  rehearsals — 0 mismatches, nav built, the nav-less forward raises `NavTokenMissing`; every R1/R3/R4/R6 lever BUILT
  as declared.
* **Arms:** `v6_nav_not_mapped` (the fix removed at the config the trainer hands `V6Stack`: S-W → both nav reads +
  the behaviour probe FAIL; S-T → the nav-carrying predecessor is REFUSED by the ladder seam) and, in the tests, a
  SOURCE mutation (the mapping line deleted from the trainer, compiled under its own path) → G-DVB FAILS;
  `v6_candidates_not_built` (the fan is born 8 wide against `--n-candidates 1`: the registry reports BUILT 8 AND the
  forward RAISES — MEASURED, which is why a forward that raises is itself a FAIL); `missing_dvb_v6_module`;
  `v6_uncovered_flag` (a new unregistered flag in the parser).

### G-LIVE — the REAL trainer smoke: declared terms live, gradients where declared, R6 literal
`steps = max(smoke_steps, 2 × --log-every)` rounded up to a multiple of `--log-every` (rows land at
`step % log_every == 0`, `:7699`). Asserts:
1. `summary.json` done at `steps` from step 0 (the artifact, not the exit code); one training `v6_loss_step` call
   per step;
2. **declared terms = the trainer's OWN registry** `effective_weight_rows(a)` (`W_TERM_FLAGS` + the `o10_psg`
   spec): every row that builds a graph → its `v6_loss_step` key present and finite on EVERY step; every row that
   does not → absent. The row→key table `V6_TERM_KEYS` is a LITERAL and EXHAUSTIVE (a registry term it does not
   name FAILS — the merge's `w_tac_label_all` → `tac_label_all` is in it);
3. **per-term gradient reach** at step 1 (`autograd.grad`, retain_graph, forked RNG, no `.grad` write): every term
   in force reaches ≥ 1 trainable parameter — **MEASURED FINDING:** `--w-o14 1.0` carried into S-T is audited
   `TRAINS` by the trainer's own table but has NO autograd graph (`for_stage("S-T")` does not zero o11/o13/o14);
4. every optimizer leaf module gets a non-zero gradient over the smoke unless the argv acknowledges it on
   `--allow-unreached`; the trainer's own `grad_reach.json` must agree (control). **F7 — MEASURED on the merge once
   nav was built AND fed, now FIXED there:** in S-W the six `predictor_op` nav tensors (`nav.embed`, `nav.arg_proj`,
   `nav.layer_proj.operative`, `nav.gate.operative`) got NO gradient — O1 (`stage_a_losses`) and O2/O3/O5
   (`rollout_transitions`) called `predictor_op` without `nav_cond`; S-T freezes the group, so the zero-init
   operative projection stayed EXACTLY 0 through S-W → S-T. The merge's `_NavBoundPredictor` binds the operative nav
   term to the predictor those call sites roll: the compliant S-W smoke now PASSES with every operative nav leaf live
   (per-leaf `leaf_live` in the evidence), and the trainer's census agrees;
5. **R6 (coordinator precision):** (a) DECISIVE — every `layer_str` parameter (EMA copies included) BIT-IDENTICAL
   between optimizer construction (after `--init-from` + freeze) and the end of the smoke, and equal to the smoke's
   own `ckpt.pt`; (b) REPORTED, not gated — how many `layer_str` tensors had `.grad` populated. **MEASURED
   FINDING:** under `--uplink ema`, `stack.ema_update()` moves 3 `ema_adapter_str.*` tensors every S-T step (an EMA
   of a frozen source drifts by float rounding) → (a) FAILS and names the mechanism `EMA`;
6. every OTHER frozen group bit-identical (EMA teachers outside `layer_str` are reported, not failed);
7. the total loss is not constant; the final `train_log.jsonl` row carries the last step's `terms` and `loss`
   (control);
8. no strategic term (`s1`, `s1_multi`, `s2`) in force — R6 at the loss level;
9. the `v6_loss_step` ARGUMENTS as called (signature defaults applied) == argv, every step (`V6_LOSS_KNOBS`).
* **Arms:** `v6_strategic_leaf_trainable` (one `layer_str` leaf made trainable after the trainer's freeze: WITHOUT
  `--strategic-off` it MOVES → R6 (a) FAILS; WITH it — MEASURED on the merge — no gradient reaches `layer_str`, the
  leaf cannot move, and the dead-trainable-leaf rule FAILS instead); `v6_drop_term` (a declared term silently
  leaves the loss — the F3-whitelist class); `v6_nav_unbound_wm` (F7 re-introduced: `train_v6_staged.
  _NavBoundPredictor` swapped for the bare predictor → the 4 operative nav leaf modules are named dead; a trainer
  without `_NavBoundPredictor` makes the arm record an exit, never a pass) and, in the tests, a SOURCE mutation of
  the three WM-loss call sites → the same named FAIL.

### G-CLOCK — labels read at the true window time
* **Design:** the trainer's OWN label time (`V72WindowSupervision._t_now_s`, `:2563-2566`: `(t + window − 1 + offs)
  × dt` on the NOMINAL `--dt`; `NavEmitter.__call__`, `v7_labels.py:1309`: `t_last × dt`) against an INDEPENDENT
  measurement of the v7F cache's clock; the legacy `(t + w − 1) × 0.1` clock must read RED (positive control); the
  arithmetic `judge_clock_windows` is shared with refcv7.
* **NOT IMPLEMENTED:** `job_clock_v6` returns ERROR (never PASS): no independent reference exists for the v7F cache
  (`…-w120-256x640cyl`; refcv6's q4c reference was measured on another extraction). R3 makes this load-bearing.

### G-EVAL — the eval loader rebuilds the IDENTICAL model
* **Asserts:** `tanitad.eval.v6_probe_trunk.load_v6_from_ck` (rebuild from `config["args"]` via
  `build_stack_from_args`, STRICT load) applied to a checkpoint in the trainer's layout (`_save_ckpt` payload), whose
  run record is the TRAINER's own `_run_config(args, stack, freeze)` for THIS argv, JSON round-tripped (on a
  non-rehearsal host the smoke's config.json must carry the same built levers): strict 0/0, every tensor identical,
  the V6Config identical, every forward output identical on a fixed batch (determinism control first), and a
  perturbed copy must DIFFER (the negative control perturbs `encoder.patch.weight` — MEASURED: perturbing an
  emission weight upstream of its ZERO-initialised final layer moved 0 of 52 outputs, a blind control).
* **Arm `v6_loader_drops_vocab_version`:** the loader rebuilds without `tac_vocab_version` (the round trip the
  trainer documents at `:5358-5362`) → v6.0 vocabulary → the strict load fails.

### G-CKPT — the trainer's own save → load and resume
* **Asserts:** run 1 saves at `N` → saved `stack`/`opt` == in-memory; run 2 resumes (`resume_guard` →
  `assert_resume_lineage` → `load_resume`) to `N+k` → post-load == saved, param groups equal, `summary.json` done at
  `N+k` resumed from `N`, `launch_mode` resume; run 3 is the UNINTERRUPTED `N+k` reference → the windows drawn at
  steps `N+1..N+k` (captured `(ep_idx, t_last)`) and the final weights must equal run 2's.
* **MEASURED (tip AND merge): FAILS** — the resumed run draws the uninterrupted run's FIRST batches (`train()`
  re-seeds torch/random/the sampler generator from `--seed` at every launch, `:6475-6478`; the payload is
  `{stack, opt, step, config}` — no RNG state, no data position); the final weights differ. Save→load itself is exact.
* **Arm `v6_resume_drops_opt`:** the resume loads the model but not the optimizer → the post-load optimizer differs.

### G-SUITE — shared, trainer-agnostic (`job_suite`, run mode). Not in the dev-box stage (and CPU-only there).

## 4. Profile-level argv rules (`v6_profile_rules`, re-derived inside `finalize`)
| rule | source |
|---|---|
| `--stage` ∈ {S-W, S-T}; S-S / S-J REFUSED | R6 |
| `--nav-cond` REQUIRED; `--i-know-this-arm-predates-nav` REFUSED | R2 |
| `--strategic-off` REQUIRED; `--tac-goal-cond` REFUSED | R6 (the merge's D3 mechanism; coordinator 2026-09-27) |
| S-T only: `--max-speed-input-v6`, `--plan-vmax-cap`, `--speed-max-sidecar-v6` REQUIRED | R1 (refused by the trainer in S-W) |
| S-T only: `--w-tac-label-all > 0`, `--goal-multilabel` REQUIRED; `--goal-factored` REFUSED | R3 |
| S-T only: `--tac-op-cond` REQUIRED, not `off`; its MODE is **PI-DECISION** (`pi_pending_values`, D1) | R4 |
| `--selector goal/mlp` REFUSED | SEL-1 (`v6_chain.py:501`) |
| `--w-s1-multi > 0`, `--w-s2-goal > 0` REFUSED | R6 at the loss level |
| control-arm acknowledgements / mis-wired isolation arms REFUSED | not v7F |
| OPEN ITEMS: **none** (`R2-NAV-NOT-BUILT` CLOSED by the merge's nav fix, 2026-09-27; the mechanism stays — an open item makes the token INCOMPLETE, never PASS or PI-DECISION) | the MEASURED nav defect, fixed |

## 5. How a NEW flag slots in (the siblings' and every future one)
1. a registry entry in `declared_vs_built_v6.py` — trainer-side (`register` build-time / `register_v6` needing the
   loss weights) or gate-time (`register_gate`); without it `uncovered_v6` FAILS G-DVB;
2. a new loss term → a row in the trainer's effective-weight registry AND in the gate's `V6_TERM_KEYS`; a new
   `v6_loss_step` argument → a `V6_LOSS_KNOBS` row;
3. a PI-fixed value → a `required_values`/`required_on_v6(_stage)` row; an undecided one → `pi_pending_values`;
4. S-T-only and refused in S-W → `V6_SW_REFUSED` (the rehearsal predecessor); needs host data → `V7F_REHEARSAL_DROP`
   / `V7F_SMOKE_DATA_LEVERS`.
