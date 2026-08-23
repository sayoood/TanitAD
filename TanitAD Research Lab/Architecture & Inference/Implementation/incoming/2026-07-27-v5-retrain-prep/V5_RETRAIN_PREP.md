# v5 retrain — the validated changes, WIRED

**Date:** 2026-07-27 · **Host:** dev box (CPU + RTX 4060), **no pod touched** ·
**Spec:** `Project Steering/Gates/flagship-v5-retrain.PREP.md`
**Status:** implementation complete and staged. ⛔ **NOT AUTHORISED TO LAUNCH** — §5 is a
ready command, and the PI approves the spend, not an agent. §4 lists what is still gating.

> The dev-box cache is keyed `14231cd29c74`, **not** the parity key `e438721ae894`, so **no
> parity work was done here** and none of these numbers is a corpus measurement. Every change
> below is CPU-testable by construction and was tested that way.

---

## 0. Headline — three findings that change the card

| # | finding | evidence class |
|---|---|---|
| 1 | ⚠️ **Defect #2 was still live.** The card marks "the four selection diagnostics" **✅ fixed**. It was **inert**: `_training_loop`'s row tuple listed `sel_gap` / `rank_acc` / `frac_sel_2x_worse_than_oracle` behind `if k in log`, but `v4_loss_step` never merged them out of `plan_l` — only `ade`/`oracle_ade` were lifted. The guard evaluated False every step and all three were dropped **exactly as before**, with a fix in place that looked done. Same double-filter shape as the `g_op_fwd_ade_m` bug. **Found by a test, now fixed.** | **MEASURED** (ours; `test_the_WRITTEN_ROW_carries_the_pair_and_the_four_selection_diagnostics` was RED before the fix, GREEN after) |
| 2 | ⚠️ **The card's own open item is a false absence.** "`speed_benefit_recovered_frac` and `deploy_tick_p99_ms` have **NO EMITTER** anywhere in the codebase" is **wrong on both counts**. Both live in `stack/scripts/gate_emitters.py` (2026-07-23) and **both produce a value today, off git-tracked artifacts, with zero GPU**: `deploy_tick_p99_ms = 18.7641` (PASS ≤ 50) and `speed_benefit_recovered_frac = 0.8184` (PASS ≥ 0.70). **Do not strike them.** Exact card edit in §3. | **MEASURED** (ours; both emitters executed on the dev box, §3) |
| 3 | The **reachability clamp** needed no implementation — it shipped complete for v1.5 and is correctly **OFF on v4**. What was missing is now supplied: a proof that flipping it on v4 **cannot perturb any loss term** (Δ = 0.0 exactly), which reduces the pending v4 flip to exactly one open question — the sibling's paired-Δ on v4's own fan. | **MEASURED** (ours; `test_flipping_the_clamp_ON_V4_cannot_perturb_any_LOSS_TERM`) |

---

## 1. What was wired — per change

Priority order was the brief's: the mid-run gate first, then the instrument, the diagnostics,
rank-16, the clamp, C8, the emitters. All seven landed.

### 1.1 ⭐ The mid-run held-out gate — §1.6, the single most valuable item

**The waste it removes, MEASURED:** ~29.5 GPU-h — half the v4 30k run — spent training past the
best checkpoint while every training term improved.

**New module** `stack/tanitad/train/heldout_gate.py`. Deliberately split in two so the
DECISION is testable with no GPU, no dataset and no model:

- `HeldoutGate.observe(step, per_window, eid)` — pure arithmetic; the stop rule lives here.
- `HeldoutGate.probe(step, world, head, episodes)` — runs pseudo-simulation, then `observe`.
- `DeployableSurfacePlanner` — the `.traj` adapter: `encode_window → head → out["wp_seq"]`,
  i.e. **the SELECTED trajectory**. Not the oracle. v4's regression *was* in selection, so a
  probe that read the oracle would have reported the run healthy while it decayed.

**The primary is the map-free composite** (`taniteval.pseudosim`), **never `ade_0_2s`**. ADE
travels as a labelled diagnostic and the rule never consults it. Pre-registered decision rule,
fixed before any checkpoint exists:

1. first probe = incumbent;
2. every later probe compared by **paired episode-cluster bootstrap** (B=2000, unit = held-out
   episode) on windows finite in both;
3. `separated_worse` = CI excludes zero **and** delta < 0;
4. the incumbent advances **only on a separated improvement** — a lucky point estimate must
   never become the bar the stop rule fires against;
5. **two consecutive** `separated_worse` probes stop the run. One does not.

Also pinned: the **admitted component set is PINNED at the first probe** (re-deriving
`discriminative_range` per probe would let the composite change definition mid-run and compare
two different metrics — the forking-paths failure `GATE_PROTOCOL` §0.3 forbids); windows are
alignment-checked by digest and a mismatch **raises** (`WindowAlignmentError`) rather than
running an unpaired "paired" test; an unusable probe **raises** (`GateNotUsableError`) rather
than silently disabling the gate.

**Trainer integration** (`stack/scripts/train_flagship_v4.py`):
- probes at `--heldout-every`, writes the full record to `train_log.jsonl` and stdout;
- archives **`ckpt_best.pt`** at every new incumbent — the run's peak survives the tail;
- **breaks the loop** on stop and writes `EARLY_STOP` + the reason into the run's own log;
- gate state (streak / incumbent / pinned ranges) round-trips through the checkpoint, and a
  resume of an already-stopped run **refuses to start** rather than spending the hours the gate
  just saved;
- `metrics.json` carries the verdict, so *"why did this run end"* is answerable from the record;
- **ON by default.** `--no-heldout-gate` exists and **trips preflight**.

**Tests that pin it** — `stack/tests/test_heldout_gate.py` (18) + 4 in `test_train_flagship_v4.py`:

| test | the input designed to make it fail |
|---|---|
| `..._decays_on_the_composite_while_ADE_IMPROVES_is_stopped` | the v4 30k failure's own shape: ADE falls monotonically while the composite decays. **Must stop.** |
| `test_the_ADE_control_does_NOT_stop_on_the_same_run` | the **falsifier** — the same run gated on ADE sails on. Proves the axis choice is load-bearing, not decorative. |
| `test_a_recovery_between_two_bad_probes_RESETS_the_streak` | bad→good→bad. A cumulative counter would kill a recovered run. |
| `test_a_worse_but_UNSEPARATED_probe_never_counts` | exact construction: half the episodes −0.50, half +0.48 → point estimate worse, interval covers zero. The case a point-estimate comparison gets wrong. |
| `test_the_incumbent_advances_only_on_a_SEPARATED_improvement` | an unseparated upward blip must not move the bar. |
| `test_misaligned_windows_RAISE_instead_of_being_compared` | a probe over a different episode set. |
| `test_the_streak_survives_a_checkpoint_roundtrip` | runs the **control first**: a fresh gate cannot stop on its first probe, so a lost streak would launder a decayed run past the gate. |
| `test_probe_grid_keeps_a_perturbed_heading_or_recovery_is_all_NaN` | builds the degenerate `dyaw=0`-only grid and shows `recovery` is entirely NaN — a plausible-looking composite that is structurally blind to the collapse the gate exists to catch. |
| `test_planner_reads_the_SELECTED_trajectory_not_the_oracle` | a head whose oracle is excellent and whose pick is awful. |
| `test_planner_REFUSES_a_non_dense_head` | the tactical instance's own horizon tuple — a coarse plan scored as dense divides every derivative by the wrong dt. |
| `test_the_training_LOOP_stops_and_banks_the_best_checkpoint` | end-to-end on the **real** `_training_loop`, plus a no-gate control that runs to the end, so the early stop is attributable. |

**Still gating (§4):** the cadence/episode budget is a defensible default, not a measurement —
the probe's wall-clock on the real corpus has **not** been measured (dev box has no parity cache).

### 1.2 Vision at rank ≈ 16 — §1.2

**New module** `stack/tanitad/models/vision_rank.py`; wired into `V15Config` (inherited by
`V4Config`) and into the v4 **factorised LAT/LON/DIST heads**, which are v4's flat reader — they
took `states[:, -1]`, the raw 2048-d state, straight into a `Linear`, i.e. exactly the shape the
swamping dose-response was measured on.

- `vision_rank: int = 16` — first-class, validated **at config construction** so a bad rank costs
  zero GPU seconds;
- **raw-2048 is impossible to select by accident**: `0`, `-1`, `None`, `2048`, `4096` and a
  missing key all land on `RawVisionRankRefused`. Raw needs `allow_raw_vision=True` **and** a
  non-empty written reason — a boolean can be flipped absent-mindedly, a sentence cannot;
- **decode-side only.** v4 is at 2 of 2 encoder-touching levers and `encoder_touching_levers ≤ 2`
  is a KILL secondary; a trunk-side projection would breach it. Pinned by a test.
- the measured ladder (`DOSE_RESPONSE`) ships **with** the lever, including the narrow claim
  verbatim: *at k=16 vision stops **destroying** the ego signal; it is +0.026× over ego alone,
  i.e. indistinguishable from it — **not** "vision adds value"*.

**Bit-identity where inertness is claimed:** the explicitly-allowed raw path is an exact identity
with **zero parameters** — `max|Δ| == 0.0` and `torch.equal`, proven not asserted.

**Compat:** a `config.json` written before this lever has no `vision_rank` key and its checkpoint
has no projection weights. `eval_flagship_v4.py` reads that absence as *legacy raw arm* and routes
it through the named override, so old checkpoints still load STRICT.

**Tests:** `stack/tests/test_vision_rank.py` (18).

**Still gating (§4):** the evidence is a **linear probe on a frozen state**. It does not
establish that a *trained non-linear* head with a learned 2048→128 projection degrades the same
way. The module says so in its own docstring; the rank is now an explicit recorded choice, and
the lever does **not** claim the reduction is free.

### 1.3 Reachability clamp — §1.1

**No implementation was needed.** `reachability_mask` / `reach_band` / the argmax-only masking
shipped with v1.5; `V15Config.sel_reach_clamp = True`, `V4Config.sel_reach_clamp = False`, and
`tactical_config()` inherits OFF. Already pinned by 5 tests in `test_flagship_v15.py` and 2 in
`test_flagship_v4.py`, including the `sel_score` bit-identity and the empty-survivor fallback.

**What I added** — the half of the v4 flip that is measurable *today*, on v4's own head:
`test_flipping_the_clamp_ON_V4_cannot_perturb_any_LOSS_TERM` proves that with the clamp ON vs OFF
the v4 head returns a **bit-identical `sel_score`** (`max|Δ| == 0.0`) and bit-identical `cls` /
`cls_refined` / `traj` / `loss`, and that the clamp was actually live in the ON arm (a guard that
did nothing proves nothing). **Kept OFF by default** as instructed.

**⭐ THE GATING ITEM LANDED WHILE I WORKED — AND IT PASSES.** The sibling's v4-fan zero-change
test was committed mid-task (`abc864a`), artifact
`…/incoming/2026-07-27-v4-instrument/raw/v4_reach_clamp.json`. On **v4's own fan**
(`flagship-v4-fromscratch`, step 29999, **967 windows / 44 episode clusters**), it reproduces the
v1.5 property:

| | 4-wp convention | dense 20-wp |
|---|---|---|
| candidates removed | **75.29 %** | 75.29 % |
| windows with an empty survivor set | **0.00 %** | 0.00 % |
| oracle survives | **100 %** | 100 % |
| windows where the pick MOVES | **0** | 0 |
| paired Δ ADE (episode-cluster, B=2000) | **0.0**, not separated | **0.0**, not separated |
| `miss@2m` | 0.0641 → 0.0641 | 0.0341 → 0.0341 |
| speed-up | **4.05×** | 4.05× |

**Evidence class: INHERITED** — this is the sibling's artifact; I read it, I did not re-run it.
Per operating-standard rule 1 I therefore **did not flip the default**, and the brief's
instruction ("keep it OFF for v4 by default") is honoured as written.

⇒ **The flip is now a one-line change with both halves measured**: *free* (the sibling's table
above) and *code-safe* (my Δ = 0.0 loss-term proof). Flipping means
`V4Config.sel_reach_clamp = True` **plus re-deriving two tests that assert it is False** —
`test_the_reachability_clamp_is_OFF_on_v4_until_it_is_measured_there` and
`test_the_selector_never_truncates_the_candidate_set`, the latter of which says in its own body
*"if it is ever enabled on v4, this test must be re-derived, not silently relaxed."*
**Escalated in §6 — it is a decision, not an implementation.**

### 1.4 The grounding instrument — §1.4

Both halves now reach the log, from the **same batch and the same forward pass**:
`g_op_mid_de_m` (metric inverse dynamics on **real** pairs) and `g_op_fwd_ade_m` (forward
consistency on the **imagined** rollout). The diagnostic quantity is their **ratio**; with only
the imagined half a rise is unattributable, because encoder drift and predictor drift are
indistinguishable — which is why all three v4 logs were undiagnosable.

**Two filters had to change** (`v4_loss_step`'s `wm_log` comprehension *and* the row-writer's key
tuple). Patching one leaves the other starved and the fix silently inert — the documented failure.

**Tests:** `test_joint_step_log_carries_the_GROUNDING_PAIR_not_just_the_imagined_half`, plus the
written-row test below, which is the one that can see a starved filter.

### 1.5 The four selection diagnostics — §1.5

⚠️ **The card's "✅ fixed" was wrong** — see §0.1. `v15_losses` returns `sel_gap`, `rank_acc` and
`frac_sel_2x_worse_than_oracle` in `plan_l`, and `v4_loss_step` never merged them into `log`, so
the row-writer's `if k in log` dropped all three every step. Now merged as detached floats
(`sel_gate` / `sel_pen_span` arrive separately via `out["telemetry"]` when the longitudinal term
is active). LOG-ONLY: no loss term, no parity effect.

**The test that catches a starved filter** —
`test_the_WRITTEN_ROW_carries_the_pair_and_the_four_selection_diagnostics` reads the **actual
`train_log.jsonl` the loop wrote** and requires each key in **every** step row. A test on
`v4_loss_step`'s dict alone cannot see this class of defect; that is exactly how it survived.

### 1.6 C8 — calibrated readout selection — §1.3

**New module** `stack/tanitad/models/readout_selection.py`. **The rule, not a fitter:**
`op` for lead ≤ 0.5 s (the registered constant `C8_SWITCH_STEP = 5`, low end of the measured
0.5–0.8 s range — the two score identically), `str` beyond.

**Semantics as validated:** roll the predictor **once**, decode the full path with each needed
head, read path `r[j]` at index `j`. **Not** a per-step Δpose splice inside one SE(2)
accumulation — that object was never validated. Cost is one extra decode, **zero extra rollout**,
pinned by a test that counts predictor calls.

**The artefact is reported, not hidden:** the concatenated path can be **discontinuous** at the
switch; `switch_discontinuity_m` / `_max_m` are returned so nobody discovers a metre-scale step in
a "smooth" trajectory downstream.

**No fitter, enforced adversarially:** `test_the_module_ships_a_rule_and_CANNOT_fit_one` inspects
**every** public signature for any parameter that could carry ground truth (`target`, `gt`, `ade`,
`loss`, `traj_tgt`, `future_poses`, …) and fails if one exists. Fitting the switch to ADE moves it
to 1.0 s, which buys **0.0063 m (0.7 %)** of ADE and pays **3.1× of deployable `T_blind`**
(2.5 s → 0.8 s). `test_the_switch_is_NOT_at_the_ADE_optimum` fails if the constant is ever
"optimised" to 10.

**Graceful degradation:** REF-A/B/C carry a single bare readout. `resolve_readout_bank` degrades
to one-readout-for-all-lead-times and **says so** (`c8_available: False`, *"NOT a C8 number"*)
rather than raising or pretending.

**Bit-identity:** at `switch_step = 0` and `switch_step ≥ k` the output is **`torch.equal`** to the
existing `rollout_decode` on that head — C8 is a strict generalisation, not a rewrite.

**Scope, carried in `C8_PROVENANCE` so it cannot be quoted bare:** measured on **v1
`flagship-30k` only**; and the rule is a 12.5 % improvement on a number **still 41 % worse than
`hold_v0` (0.5933)** — readout selection is real and free, and it does **not** rescue the
deployable regime.

**Not yet attached to the trainer's `canary_rollout`.** Deliberate: C8's evidence is v1-only, and
swapping the canary's readout mid-programme would move every canary number and break comparability
with the pre-registered `wm_canary_ade_2s ≤ 0.55` secondary. It is a callable instrument; adopting
it as the canary's default is a PI call. **Escalated in §6.**

**Tests:** `stack/tests/test_readout_selection.py` (12).

---

## 2. Explicitly NOT implemented (§2 of the card), and confirmed absent from the diff

**C1**, **C3**, **λ/τ prior strength**, **the action filter**, **λ_plan** (a gradient scale; a
documented no-op at 1.0) and **the anchor vocabulary** were not touched. `λ_plan` stays 1.0 and the
anchors are untouched — the diff contains no change to `default_anchors`, `load_anchors`, the
anchor buffer, or `lambda_plan`'s semantics.

---

## 3. The card's own open item — SETTLED, and the exact edit

**Verdict: do NOT strike them. The claim is false; the two `null`s had two different real causes.**

MEASURED on the dev box, zero GPU, off git-tracked artifacts:

```
$ python stack/scripts/gate_emitters.py deploy-tick \
      --eff-json taniteval/results/eff_levers_flagship-30k.json
  deploy_tick_p99_ms = 18.7641   pass: true   (A40; all_levers; ade delta -6.6e-05 m)

$ python stack/scripts/gate_emitters.py speed-benefit \
      --arm-log taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl --repo-root .
  speed_benefit_recovered_frac = 0.8184   pass: true   (n_arm_rows 40)

$ ... --arm-log taniteval/results/trainlogs/flagship-v4.1-10k_train_log.jsonl
  speed_benefit_recovered_frac = null     n_arm_rows: 0   n_nospeed_rows: 40
```

The last line is the whole diagnosis: the emitter **ran**, and found **zero** rows carrying
`g_op_fwd_ade_m` in v4.1's log — the starved filter, **fixed in §1.4**. So:

| metric | why it read `null` | precondition for v5 | status |
|---|---|---|---|
| `speed_benefit_recovered_frac` | the **arm's own log** had no `g_op_fwd_ade_m` rows | the v5 log must carry `g_op_fwd_ade_m` in the (8000, 10000] bucket | ✅ **fixed today** (§1.4) — no further work |
| `deploy_tick_p99_ms` | no `taniteval.efficiency` **lever panel** was ever measured on a v4 checkpoint | run the efficiency lever panel on the v5 checkpoint (GPU, minutes) | ⬜ a measurement to schedule, **not** code to write |

This is the **CLAUDE.md rule-2 class** — *absence found at one location is not absence*. Striking
them would have removed two criteria that are measurable today.
Pinned by `test_BOTH_disputed_emitters_EXIST_and_produce_a_value_off_committed_artifacts` and
`test_the_v4_null_was_an_INPUT_defect_not_a_missing_emitter`.

### The exact card edit (replaces §4's two ⚠️ lines)

Replace, in `Project Steering/Gates/flagship-v5-retrain.PREP.md` §4:

> ⚠️ **`speed_benefit_recovered_frac` and `deploy_tick_p99_ms` have NO EMITTER** — they were recorded
> `null` last time, and the run was called a *"formal 8-metric gate"* when it was a **6-metric gate**.
> **Either build the emitters or strike them from the card. Do not carry an unmeasurable criterion.**

with:

> ⚠️ **CORRECTED 2026-07-27 — both metrics DO have emitters and are NOT struck.** The "no emitter"
> claim was a stale absence (CLAUDE.md rule 2). Both live in `stack/scripts/gate_emitters.py` and
> both produce a value off git-tracked artifacts with **zero GPU**: `deploy_tick_p99_ms = 18.7641`
> (PASS ≤ 50) and `speed_benefit_recovered_frac = 0.8184` (PASS ≥ 0.70) on v1. The last gate's two
> `null`s had two **different** causes, neither of them a missing emitter:
> - **`speed_benefit_recovered_frac`** — the emitter ran and found **0 rows** carrying
>   `g_op_fwd_ade_m` in v4.1's `train_log.jsonl` (`n_arm_rows: 0`): a starved log filter, **fixed
>   2026-07-27** in `train_flagship_v4.py`. **No further work; v5's log will feed it.**
> - **`deploy_tick_p99_ms`** — the emitter ran; no `taniteval.efficiency` **lever panel** has ever
>   been measured on a v4/v5 checkpoint. **Schedule that panel on the v5 checkpoint** (GPU, minutes).
>   Until it is run the metric is UNMEASURED-for-this-arm, which is a scheduling item, not a reason
>   to strike a criterion.
>
> ⇒ the v5 gate is an **8-metric gate** provided the efficiency lever panel is run on the checkpoint.

---

## 4. What remains GATING

The card's five §3 items are all still in flight and **none of them is unblocked by this work**:
T3's verdict, v4's grounding instrument, the clamp's zero-change property on v4's own fan, the
latent-ablation verdict, and the pseudo-simulation arm panel. Added by this work:

1. ~~**The clamp's v4 flip**~~ — **CLEARED** while this task ran: the sibling's v4-fan test
   returned 75.29 % removed / oracle 100 % / **paired Δ = 0.0, pick never moves** / 4.05× cheaper
   on 967 windows (§1.3). Both halves are now measured. Left OFF because the number is INHERITED
   and the flip is a decision → **§6**.
2. **The efficiency lever panel on the v5 checkpoint** — the one remaining precondition for an
   8-metric gate (§3).
3. **The gate's probe cost on the real corpus is UNMEASURED.** The dev box has no parity cache
   (`14231cd29c74`, not `e438721ae894`), so `--heldout-every` / `--heldout-episodes` are defensible
   defaults, not a measured budget. First real run should time probe #1 before trusting the cadence.
4. **Rank-16's scope limit** — measured as a linear probe on a frozen state; the trained non-linear
   case is not established (§1.2).
5. **C8 is not the canary's default** and should not become one without a PI call (§1.6, §6).

---

## 5. The launch command — ⛔ **NOT AUTHORISED**

**This is a staged command, not an instruction to run.** The PI approves the spend, not an agent,
and §4 above is not empty. Every flag is present so nothing has to be reconstructed later.

```bash
# ⛔ NOT AUTHORISED — DO NOT RUN. Staged for the PI. Restart budget: this spends 1 of 2.
PYTHONPATH=/workspace/TanitAD/stack \
/workspace/venv/bin/python /workspace/TanitAD/stack/scripts/train_flagship_v4.py \
  --train-cache /workspace/epcache/physicalai-train-e438721ae894 \
  --val-cache   /workspace/epcache/physicalai-val-0c5f7dac3b11 \
  --anchors-dense /workspace/anchors/dense_1_20_256.pt \
  --out /workspace/runs/flagship-v5 \
  --from-scratch \
  --steps 30000 --gate-step 10000 \
  --batch 16 --accum 4 \
  --lr-head 1e-4 --lr-trunk 1e-4 \
  --lambda-plan sched --phase-a-steps 2000 --phase-b-steps 8000 \
  --lam-mult-floor 0.25 \
  --strategic full --d-strat 128 --long-horizon-k 50 \
  --lat-weight 0.05 --lon-weight 0.05 --dist-weight 0.05 \
  --jerk-w 0.02 --curv-w 0.01 \
  --strat-goal-weight 0.1 --strat-pred-weight 0.5 --strat-scalar-weight 0.05 \
  --ego-null-row --rollout-k 4 --dense-plan \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 \
  --heldout-patience 2 --heldout-stride 8 --heldout-nboot 2000 \
  --warmup 2000 --workers 4 \
  --log-every 50 --eval-every 500 --save-every 1000 --eval-episodes 40 \
  --device cuda --seed 0
```

Dry-run it first (**this one is safe — it starts nothing**):

```bash
python stack/scripts/train_flagship_v4.py --print-launch   # prints the command + preflight gates
```

Notes the PI needs before approving:
- **`--rollout-k 4` is v1-verbatim and must not be raised** before `speed_benefit_recovered_frac`
  unlocks it ([PM] #2); preflight enforces it.
- **the held-out gate is ON**; `--no-heldout-gate` trips preflight by design.
- **vision enters at rank 16** by default; raw-2048 cannot be selected without an explicit flag
  **and** a written reason.
- **the clamp stays OFF on v4/v5** until the sibling's v4-fan measurement returns.
- **C8 is available but not wired into the canary** — deliberate (§1.6).
- parity is asserted **before any GPU allocation** (count + uid digest vs the committed manifest).

---

## 6. ESCALATE — integration decisions that need a person

0. ⭐ **FLIP THE REACHABILITY CLAMP ON v4/v5?** Its gating measurement **landed and passed** during
   this task (§1.3): 75.29 % of the fan removed, oracle survives 100 %, **the pick moves on 0 of
   967 windows**, paired Δ **exactly 0.0** and not separated, `miss@2m` unchanged, **4.05× cheaper**
   — on **v4's own fan**, which is the surface the property was previously unmeasured on. My own
   test adds that the flip **cannot perturb any loss term** (Δ = 0.0 on `sel_score`, `cls`,
   `cls_refined`, `traj`, `loss`). I did **not** flip it: the number is INHERITED (the sibling's
   artifact, not re-run by me) and the brief said keep it OFF. **The flip is now one config line
   plus re-deriving two named tests** (§1.3). This is the cheapest remaining win on the card and it
   wants a yes/no.
1. **C8 as the canary/eval default.** `readout_selection.calibrated_rollout_decode` is built,
   tested and free, but every production caller still hard-wires `grounding.step["op"]`. Adopting
   it would move every canary number and break comparability with the pre-registered
   `wm_canary_ade_2s ≤ 0.55`. **PI call.** Named call sites: `taniteval/rollout.py::collect`,
   `train_flagship_v4.py::canary_rollout`, `train_flagship_v16.py::canary_rollout`,
   `taniteval/blindimag.py::blind_rollout`.
2. **The card edit in §3 must be applied to `flagship-v5-retrain.PREP.md`.** I did not edit the
   card — it is a Project Steering gate document. The exact replacement text is in §3.
3. **`RETRACTION_LOG.md` needs two entries** (root-cause classes, per operating-standard rule 4):
   *stale absence-claim* (the "no emitter" line) and *inert fix / double-filter* (the "✅ fixed"
   diagnostics). Both are recurrences of already-logged classes, which is itself the finding.

---

## 7. Test counts

| suite | before | after | delta |
|---|---|---|---|
| `stack` | 1135 passed, 7 skipped | **1191 passed, 7 skipped** | **+56** |
| `taniteval` | 559 passed | **559 passed** | 0 (unchanged; no taniteval source was modified) |

⚠️ The brief quoted the `taniteval` baseline as **514**. The measured baseline on this checkout
**before any of my changes** was **559** — the brief's figure is stale (INHERITED). I re-measured
both baselines before touching anything rather than assuming.

---

## 8. Deliverable manifest

Everything is in the repo. **I committed nothing, pushed nothing, switched no branch, and touched
no pod.** Nothing here lives in only one place.

⚠️ **BUT — four of my files were swept into a sibling's commit while I worked.** I staged them at
the "bank incrementally" checkpoint; a concurrent agent then ran a pathspec-free commit, which
takes the WHOLE INDEX, and `abc864a` (*"THE LATENT DICHOTOMY IS A FALSE ALTERNATIVE…"*) now
contains `stack/tanitad/train/heldout_gate.py`, `stack/tests/test_heldout_gate.py`,
`stack/scripts/train_flagship_v4.py` and `stack/tests/test_train_flagship_v4.py` under a message
that has nothing to do with them. **This is exactly the hazard CLAUDE.md §"Git hygiene" documents,
observed a third time.** Nothing is lost or corrupted — I verified `git diff HEAD` is empty for all
four, so the committed content **is** the current content. It is a provenance defect, not a data
loss: anyone reading `abc864a`'s message will not know the mid-run held-out gate is inside it.

| artifact | where | state |
|---|---|---|
| `heldout_gate.py` — the mid-run gate | `repo:stack/tanitad/train/heldout_gate.py` | **new** — swept into `abc864a` (see note above) |
| `vision_rank.py` — the rank lever | `repo:stack/tanitad/models/vision_rank.py` | **new**, staged |
| `readout_selection.py` — C8 | `repo:stack/tanitad/models/readout_selection.py` | **new**, staged |
| v4 trainer: gate wiring, grounding pair, selection diagnostics, CLI, preflight | `repo:stack/scripts/train_flagship_v4.py` | modified — swept into `abc864a` |
| v4 head: rank-16 on the factorised readers | `repo:stack/tanitad/models/flagship_v4.py` | modified, staged |
| `V15Config.vision_rank` + construction-time validation | `repo:stack/tanitad/models/flagship_v15.py` | modified, staged |
| legacy vision-rank compat on the eval path | `repo:stack/scripts/eval_flagship_v4.py` | modified, staged |
| tests — held-out gate (18) | `repo:stack/tests/test_heldout_gate.py` | **new** — swept into `abc864a` |
| tests — vision rank (18) | `repo:stack/tests/test_vision_rank.py` | **new**, staged |
| tests — C8 (12) | `repo:stack/tests/test_readout_selection.py` | **new**, staged |
| tests — trainer integration + written-row + preflight (+5) | `repo:stack/tests/test_train_flagship_v4.py` | modified — swept into `abc864a` |
| tests — the v4 clamp-flip bit-identity (+1) | `repo:stack/tests/test_flagship_v4.py` | modified, staged |
| tests — the emitter falsifiers (+2) | `repo:stack/tests/test_gate_emitters.py` | modified, staged |
| this report + the launch command + the card edit | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-v5-retrain-prep/V5_RETRAIN_PREP.md` | **new**, staged |

**Nothing exists in only ONE place.** No pod path, no worktree, no scratchpad-only artifact.
