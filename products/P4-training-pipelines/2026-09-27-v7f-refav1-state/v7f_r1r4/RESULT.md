# v7F — R1 (max speed: tactical input + hard cap) and R4 (tactical → operative conditioning): RESULT

**2026-09-27 · TrainingFlyWheel (P4) · built on a private copy of origin tip `c36b6ddd`
(`C:/Users/Admin/v7f_r1r4/stack`, CPU only) · STAGED, not committed · needs INTEGRATION (below).**

## Headline

1. **R4 = PARTIAL at the tip → BUILT.** The tactical GOAL already reaches the operative plan
   (`e_g_tac` → `emit`); the tactical BEHAVIOURS (`a_lat` × `a_lon`) do not (MEASURED: swapping a
   behaviour moves the plan by exactly 0.0). `--tac-goal-cond` is strategic→tactical, confirmed.
   Built `--tac-op-cond {off,detached,e2e}` (`AUDIT_R4.md`).
2. **R1 flags:** `--max-speed-input-v6` + `--speed-max-sidecar-v6` (the refcv6 4-bin one-hot into
   the tactical decision latent) and `--plan-vmax-cap` (refav1's cap on every candidate, at
   inference, before selection; the training forward stays uncapped by design).
3. **Tests:** new `tests/test_v7f_r1r4.py` **63 passed / 0 skipped**; **14/14 mutants killed**;
   all 107 v6-related test files: **2,589 passed** vs the tip's 2,526 (+63 = the new file), and the
   IDENTICAL 32 pre-existing failures+errors on both (8 F / 24 E — environmental: repo documents
   absent from a bare `stack/` copy, one CRLF-vs-LF sha pin). **No new failure.**
4. **OFF = bit-identical to the tip:** state_dict per tensor + key order + RNG stream (tiny ×2
   vocabularies, and the production default 87,899,224 params / 405 keys), every forward output
   (woken emission, not a vacuous zero plan), `emit`, the default batch/config.json/dry_run.json.
5. **Needs integration / PI:** (a) G-DVB entries are a **PROPOSAL module**
   (`tanitad/train/declared_vs_built_v6.py`, wired into `build_stack_from_args`) — `launch_gate.py`
   has NO `train_v6_staged` profile; (b) `v6_chain.py` does not carry the new flags; (c) T1 eval
   tools must pass `plan_v_max_ms`; (d) **two MEASURED R2 defects** — a real `--nav-cond` v7F launch
   cannot start at the tip (§7); (e) PI decisions D1–D4 (§8).

## 1. What was built

| flag (default) | model | group | params (prod) | enters | gradient |
|---|---|---|---|---|---|
| `--tac-op-cond off\|detached\|e2e` (`off`) | `V6Config.tac_op_cond` → `V6Stack.tac_op_port` = zero-init `Linear(2·d_goal_embed → d_goal_embed)` | `planner` | +32,896 (d_goal_embed 128) | `e_plan = e_g_tac + tac_op_port(e_a_tac)` → `emit` (fan, anchor head, selector) | plan loss → port; `e2e` also → `act_head_lat/lon`; `detached` never |
| `--max-speed-input-v6` (off) | `V6Config.max_speed_input_v6` → `V6Stack.vmax_tac` = zero-init `Linear(4 → d_tac)` | `layer_tac` | +2,560 (d_tac 512) | `z_tac_p += vmax_tac(onehot(bin(v_hi)))` — the decision view every tactical head reads; the WM-side `z_tac` untouched | plan loss (via goal/behaviour) and t1 (via behaviour → P_T) |
| `--speed-max-sidecar-v6 PATH` (None) | — | — | 0 | the refcv6 sidecar (RAW `v_hi_ms`, joined on the stable id) → batch keys `v_max_ms`/`v_max_valid` | — |
| `--plan-vmax-cap` (off) | `V6Config.plan_vmax_cap` (+ `emit(v_max=)`, `forward(plan_v_max_ms=)`) | — | 0 | every candidate's accel capped BEFORE anchor/selector/MPC/fallback; eval-mode emit REFUSES without a limit | none (no parameters) |

Production default geometry, MEASURED (`raw/prod_geometry_byte_identity.out.json`): all three ON
= +35,456 params, exactly the four port keys, no pre-existing tensor moved.

**An R1+R4 S-T launch adds:** `--tac-op-cond detached --max-speed-input-v6 --plan-vmax-cap
--speed-max-sidecar-v6 <refcv6_speed_max_v8_train.jsonl>` (the sidecar built over the SAME blob as
`--nav-labels`). All three are REFUSED in S-W; S-T introduces the two ports over an S-W checkpoint
(`STAGE_MAY_INTRODUCE["S-T"]`); inference must pass `plan_v_max_ms` =
`plan_speed_cap.plan_limit_from_vhi(v_hi, valid)` and, for R1a, `v_max_ms`/`v_max_valid`.

### R4 — what the conditioning is computed from (PI 2026-08-03 rule)
`e_a_tac` = `[vocab_a_lat.encode(act_head_lat(z_tac_p)), vocab_a_lon.encode(act_head_lon(z_tac_p))]`
— the LAT×LON behaviour posteriors over the vision-derived tactical latent (plus the declared
max-speed input when R1a is on). No situation-classifier output in any form. `detached` follows the
F-1 downward-port rule (the plan loss trains the port, never the behaviour heads, which stay
t1-/label-trained); `e2e` lets the plan loss shape the behaviours too. **Liveness**
(`raw/r4_counterfactual_values.json`, MEASURED): with the emission and port woken (N(0, 0.5)
surrogate) a LAT behaviour swap moves the 6 s plan by up to **13.11 m** and a LON swap by **25.96 m**
(both modes); OFF the same swaps move it **0.0** while the decision itself moved by 1.0 (the
negative control = the tip defect); from ZERO through S-T's own `v6_loss_step` (4 Adam steps,
lr 1e-2) the port leaves zero (|W| sum 9.25) and the swaps then move the plan by 0.0002 m / 0.0007 m —
liveness acquired by training, small because the port has only begun to learn. Gradient: the plan
loss reaches `tac_op_port.weight` even at zero init; in `e2e` it also reaches `act_head_lat/lon`
and `vocab_a_lat/lon`; in `detached` it reaches none of them (the goal path is unchanged).

### R1a — encoding: the 4-bin one-hot, not a normalised scalar
The PI ruled four discrete values (2026-09-16: 30/50/100/120 km/h); the one-hot is the refcv6
channel verbatim — same reader, same stamp (`speed_max_derivation_v6`, refused in both directions),
same ladder applied ONCE on the model side; a one-hot has no ordering for a regressor to exploit
(an obedience test stays a test of obedience); an invalid row is ALL-ZERO ("unknown" ≠ 30 km/h, the
X15 rule) with no companion flag. **Real labels** (`raw/real_sidecar_check.out.json`, the local v8
EVAL blob, 147 clips): limits {30: 49, 50: 52, 100: 39, 120: 7} km/h, model-side bin = sidecar bin on
every row, cap limit = `limit_ms_of_bin(bin)`, a wrong label md5 REFUSED.

### R1b — the cap (refav1 `_cap_speed` semantics, `tanitad/models/plan_speed_cap.py`)
Per step `a ← min(a, (v_max − v)/dt)` floored at `−a_max`; binds EVERY candidate (live fan, the
diffusion arm's query reference, the MPC top-K) BEFORE selection; non-binding = bit-identical
(`torch.where` keeps the original waypoints); `v0 > v_max` brakes at `−a_max` and is reported as an
INPUT property (`vmax_v0_over_limit`). **One deliberate deviation:** the speed is tracked with v6's
floor at 0 (`train_v58f_unicycle_head.py:136`); refav1's unfloored formula applied to this integrator
lets a stop-then-go plan reach **2.2 m/s under a 1.2 m/s limit** (literal test + control).
**Training is uncapped** because the label exceeds its own bin on decelerating windows and on the
>120 km/h clips (the imitation target would be unreachable) — decision D2.

Literal tests (dt 0.1, a_max 4): v0 10, a +4, limit 11 → `[4, 4, 2, 0, …]`; v0 15, limit 11, a 0 →
`[−4]×10 + [0, 0]`; stop-then-go → `[−4]×5 + [4, 4, 4, 0, 0, 0]` (peak 1.2) vs refav1-unfloored
`[−4]×5 + [4]×5 + [2]` (peak 2.2). **Discriminating control on the real eval-mode forward:** every
candidate commands +3 m/s² from 10 m/s under a 13.8889 m/s (50 km/h) limit — uncapped peak 28.0 m/s,
capped ≤ 13.8889 + 1e-4 on every candidate; the emitted waypoints are exactly
`unicycle_rollout(capped a)`.

## 2. Tests (CPU, `-m "not slow"`, raw logs in `raw/`)

| run | result |
|---|---|
| `tests/test_v7f_r1r4.py` (new; pre-change module = tip `v6.py` via `TANITAD_V6_PRE_R1R4`) | **63 passed, 0 skipped** (`raw/pytest_new_tests_final.txt`, `-v`) |
| mutation harness (`raw/mutation_harness.py`): 14 single-point mutants of the new code | **14/14 KILLED** |
| 24-file v6 subset — tip baseline (`raw/pytest_v6_subset_baseline_tip.txt`) | 786 passed, **1 failed**, 19 skipped (`test_E4_…` reads a repo doc absent from a bare `stack/` copy — environmental) |
| same 24 + the new file — after, BEFORE the pin extension | 846 passed, 4 failed: the E4 one + 3 EXACT pins of `STAGE_MAY_INTRODUCE["S-T"]` (designed to fail "first" when the allowance grows) |
| same 24 + the new file — after, FINAL code (`raw/pytest_v6_subset_after_final.txt`) | **849 passed** (= 786 + the 63 new), 1 failed (the same environmental E4), 19 skipped |
| every stack test file that touches v6 / the trainer / DVB / the ladder (`raw/v6_related_tests.txt`, 106 at the tip) — tip baseline | 2,526 passed / 8 failed / 24 errors / 43 skipped |
| the same 106 + the new file — after (`raw/pytest_v6_related_after.txt`) | **2,589 passed** / 8 failed / 24 errors / 43 skipped — the IDENTICAL 32 failures+errors (set-compared), none new; all 32 environmental (repo docs under `TanitAD Research Lab/` absent from a bare `stack/` copy; one CRLF-vs-LF sha pin in `test_launch_gate.py`). ⚠️ this run started seconds before a DOCSTRING-only edit to `v6.py`; the final-code rows above re-ran the subset + the mutation harness |

`tests/test_v6_stage_init_introduction.py`: the two exact pins extended "consciously", the way every
entry after `cand_score.` was (docstring records the 2026-09-27 extension).

## 3. Trainer smoke (the REAL `--dry-run` path, CPU, synthetic tensors, S-T, 4 steps)

`raw/trainer_smoke.py` → `raw/trainer_smoke.json` (seeded driver — the tip's CLI dry-run is not
reproducible across processes, §7.3). Every arm: preflight clean, finite, X3 isolation PASS.

| arm | params | step losses 1–4 | bit-identical to OFF |
|---|---|---|---|
| off | 8,706,120 | 46.67498 · 38.30475 · 34.48569 · 42.64952 | — |
| `--tac-op-cond e2e` / `detached` | +528 | 46.67498 · 38.30475 · 34.48557 · 42.64840 | steps 1–2 ✅ (zero-init; the port's first gradient arrives at step 2), then live |
| `--max-speed-input-v6` | +160 | 46.67498 · 38.30475 · 34.48569 · 42.64953 | step 1 ✅ exactly; the TOTAL stays equal through step 3 only because the t1 term's change from step 2 on (0.7746302 → 0.7746307) is below the float32 resolution of a ~38 sum — the port trains from step 1 via t1 |
| `--plan-vmax-cap` | +0 | identical | **all 4 steps ✅** (the cap never touches training) |
| all three (e2e) | +688 | 46.67498 · 38.30475 · 34.48557 · 42.64840 | steps 1–2 ✅ |

Cap smoke through the eval-mode forward (4 regimes): limits 8.33 / 8.33 / 33.33 / none m/s,
`n_v0_over_limit` 1 (braked), 8 of 32 candidates bound, **`n_plan_over_limit` 0**,
`max_plan_excess_ms` (t0 under the limit) 0.0.

## 4. Bit-identical when OFF — the proof, not the claim
Against the TIP `v6.py` loaded as a separate module (content-anchored, never HEAD): state_dict key
order and every tensor `torch.equal`, and the global RNG stream after construction, at the tiny
geometry for vocab v6.0 AND v7.0, and at the production default (87,899,224 / 405); the ENTIRE
forward output tree `torch.equal` (plan, goals, latents, `planner_side`, `uplink_side`) with a woken
emission; `emit` likewise; the default synthetic batch key set; a default `config.json` and
`dry_run.json` carry no R1/R4 key; mutant M1 (build the port unconditionally) is killed by it.

## 5. G-DVB — written as a PROPOSAL, wired locally
`tanitad/train/declared_vs_built_v6.py`: a separate registry (a shared one would collide on
`max_speed_input_v6`, which `refc_v3_train.py` also defines with a RefCV3Model reader), same
`Mismatch`/`Lever`/KINDS, `built` readers on the BUILT `V6Stack` (module presence, shapes, group,
the cfg field the forward branches on) against LITERAL argv expectations; `data` entry for the
sidecar. `build_stack_from_args` calls `refuse_on_mismatch` (so every v6 build, including the eval
loader's rebuild from recorded args, refuses a declared-but-not-built lever). ⛔ **Not in
`launch_gate.py`: its PROFILES are `refc`/`refcv6`/`refcv7` (all `refc_v3_train.py`); a
`train_v6_staged` profile with G-DVB over all ~232 flags is the gate agent's item.**

## 6. What is NOT done
* No real training data run: the train sidecar (`refcv6_speed_max_v8_train.jsonl`) and the corpus
  are on Thor; the join is tested on a synthetic sidecar + the real EVAL labels with stand-in
  episodes. ⚠️ UNVERIFIED: that Thor's train sidecar's `source_md5` equals the md5 of the blob a v7F
  launch passes as `--nav-labels` (the reader REFUSES otherwise, by design).
* `v6_chain.py` (the ladder's launch-line builder) is not taught the three flags; the S-S/S-J
  CARRY rule for `tac_op_port.`/`vmax_tac.` is recorded in `STAGE_MAY_INTRODUCE`'s comment only
  (moot under R6 while S-S is skipped).
* T1/eval tools do not yet pass `plan_v_max_ms`: an eval-mode forward of a `--plan-vmax-cap`
  checkpoint REFUSES until they do (intended: mandatory at inference).
* The training-time seam dump banks the uncapped training-forward plan (unchanged). A tool that
  RE-EMITS from a forward's outputs (e.g. `seam_probe.py`'s documented `stack.emit(z_op, e_g_tac,
  v0)`) must pass `out["e_plan"]` (present only when `--tac-op-cond` is on) for an R4 arm, or it
  re-emits without the behaviour conditioning.
* R3 (the tactical label loss) is another agent's; until it lands the behaviour heads are shaped
  only by t1 (and, under `e2e`, the plan loss).
* No hard (argmax) tactical-choice inference mode; conditioning is on posteriors.

## 7. Found, NOT fixed (R2 was declared out of scope) — needs an owner
1. **Build time (MEASURED):** `build_stack_from_args` runs `assert_isolation` on
   `synthetic_batch` (`v6.py:6181-6201` at the tip), which carries no nav token → with
   `--nav-cond` the stack build raises `NavTokenMissing` before any corpus mounts
   (`raw/r2_nav_defects_probe.out.json`).
2. **Step 1 (MEASURED statically):** in `train()`'s batch dict (`train_v6_staged.py:7279`) the
   NavEmitter splat (`:7314`) is followed by the literal `"nav_token": b.get("nav_token")` (`:7336`);
   the later key wins and the dataset item has no nav token → `None` → `NavTokenMissing` at step 1.
   `tests/test_nav_v6stack.py:195` PINS the overriding line. ⇒ R2 exists in the model but a real
   `--nav-cond --nav-labels` v7F run cannot start at the tip. (My R1 keys are merged with
   `batch |= …` AFTER the literal and `synthetic_batch` carries them, so R1 has neither defect.)
3. `dry_run()` never seeds the global RNG (only `train()` does, `:6475`): two identical dry-runs
   differ (46.5779 vs 46.5834 at step 1) — a same-seed smoke is a fresh draw.
4. R6: the tactical goal head reads `cond=e_g_str` from the UNTRAINED strategic head, and the plan
   loss reaches 13 `layer_str` tensors (frozen in S-T, trained in S-J) — `AUDIT_R4.md` §5.

## 8. Decisions needed (PI)
* **D1 — R4 mode.** Recommended `detached` (the F-1 rule: behaviours stay label-grounded once R3
  lands; the plan learns to FOLLOW the tactical choice) vs `e2e` (the plan loss also shapes the
  choice — more end-to-end, less attributable).
* **D2 — cap in training?** Built inference-only (the GT exceeds its own bin on decelerating and
  >120 km/h clips). Capping the S-T plan too would make the plan loss see the constraint, at the
  cost of an unreachable target on those windows.
* **D3 — R6 literally off?** Zero or bypass `e_g_str` into the tactical goal head when the strategic
  layer is off (today it is a fixed random embedding).
* **D4 — fix the two R2 defects** (small; `synthetic_batch` nav keys + drop/reorder the `.get` line
  and its pin) — who owns it.

## 9. Hunks (for merging with the tactical-label agent) — `raw/hunk_index.json`, full diffs in `code/diffs_vs_c36b6ddd/`
**`stack/tanitad/models/v6.py`** (17 hunks, +209/−3; tip line anchors): module docstring note (`@@ -57`);
`V6Config` R1/R4 fields after `nav_cond` (`@@ -4055`); `__post_init__` `tac_op_cond` check before the
`uplink` check (`@@ -4387`); `__init__` END: `tac_op_port`/`vmax_tac` after `t2_head` (`@@ -5387`);
`_GROUP_PREFIXES` `vmax_tac.`→layer_tac after `cond_tac_dyn.` (`@@ -5439`) and `tac_op_port.`→planner
after `prop_diffusion.` (`@@ -5463`); `emit`: signature `v_max`, docstring, eval-mode refusal, cap
after the fan, cap of the MPC top-K (`@@ -5735…-5864`); `forward`: signature, max-speed refusal,
`z_tac_p += vmax_tac(…)` after the cut, `e_plan` + `emit(..., v_max=plan_v_max_ms)`, `out["e_plan"]`
(`@@ -5864…-6138`); `synthetic_batch` keys (`@@ -6196`).
**`stack/scripts/train_v6_staged.py`** (15 hunks, +384/−3): `STAGE_MAY_INTRODUCE["S-T"]` (`@@ -485`);
`v6_loss_step` forward kwargs `v_max_ms`/`v_max_valid` (`@@ -4259`); `build_stack_from_args` cfg
mapping (`@@ -5307`) + G-DVB-v6 call (`@@ -5361`); `synthetic_train_batch` keys (`@@ -5710`);
`dry_run` sidecar read, cap smoke, `r1_r4` record (`@@ -5899`, `-6012`, `-6056`); NEW helper block
before `_run_config` + `_run_config` return→assert (`@@ -6325`, `-6383`); `train()` sidecar join
after the nav join, `r1_max_speed_join` record, batch keys after the S2 keys (`@@ -7040`, `-7106`,
`-7412`); argparse after `--tac-goal-cond` (`@@ -9447`); `preflight` tail call (`@@ -10871`).
**`stack/tests/test_v6_stage_init_introduction.py`** (2 hunks): the exact `S-T` allowance pins.
**NEW:** `stack/tanitad/models/plan_speed_cap.py`, `stack/tanitad/train/declared_vs_built_v6.py`,
`stack/tests/test_v7f_r1r4.py`.
Likely conflict zones with a tactical-loss change: `v6_loss_step` (only the forward call's kwargs
here), `V6LossWeights` (untouched here), the argparse block (mine sits right after
`--tac-goal-cond`), `_run_config` (its opening `return {` became `cfg_json = {`).
