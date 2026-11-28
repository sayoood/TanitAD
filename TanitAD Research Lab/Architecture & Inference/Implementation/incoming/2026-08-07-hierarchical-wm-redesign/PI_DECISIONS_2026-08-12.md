# PI DECISION SHEET — 2026-08-12 · v6 start
### six decisions, each with a recommendation. Two block the launch; four do not.

**How to read this:** every row carries its **evidence class** (`MEASURED` / `PUBLISHED` /
`INHERITED` / `ESTIMATED` / `HYPOTHESIS`) and the **primary source** — `MODEL_REGISTRY.md §x`
or a raw gate JSON. Nothing here is quoted from a summary or a weekly report.
**Sources re-verified on the dev box 2026-08-12** unless marked otherwise.

**Tier stamp:** nothing on this page is a driving-performance claim. Capability claims are
**T1** only (`taniteval/tools/t1_eval.py`, `EVAL_DOCTRINE.md`).

---

## 0. The two-line answer

| | |
|---|---|
| **S-W (world stage) can start tomorrow on code.** | Everything dev-side is DONE and re-MEASURED tonight: 80/80 tests, dry-run OK, params 87.89 M / 300 M, X3 isolation `pass=True`. |
| **It is blocked on exactly two PI decisions — D1 (cost authorisation) and D2 (which pod).** | Both are provision/spend. Neither is a technical unknown. Everything else on this sheet can wait days or weeks without delaying v6. |

---

## D1 — How many A40-hours are authorised for S-W before the first gate? ⛔ BLOCKS THE LAUNCH

**Question:** S-W is ESTIMATED **175–290 A40-hours** (7–12 A40-days) for 30 k steps. Do we
authorise the full ladder up front, a re-cost checkpoint, or a shortened ladder?

| option | what it buys | what it risks |
|---|---|---|
| **(a)** authorise the full 30 k up front | no interruption | if the estimate is wrong the way this programme's estimates have been wrong, we discover it after ~2.5 GPU-days of overspend |
| **(b) ⭐ authorise ~12 A40-h to the step-500 re-cost, then re-decide** | the band becomes MEASURED before the big commitment | one deliberate decision point ~12 h in |
| **(c)** authorise a 20 k ladder now | linear saving | the ladder is shorter and must be stated in the run row; does not remove the estimate risk, only the exposure |

**Evidence**

| fact | class | source |
|---|---|---|
| flagship v1 `flagship4b-speedjerk-30k`: `wallclock_s` **191 206.2** / 30 000 steps = **6.374 s/step** on A40 | **MEASURED** | `MODEL_REGISTRY.md §1.2` (`summary.json` literal) |
| `flagship-v4-fromscratch`: **59.04 h** = `wallclock_s` **212 544.6** / 30 000 = **7.085 s/step** on A40 | **MEASURED** | `MODEL_REGISTRY.md §1.5.5` (`metrics.json` literal, re-read off pod2 2026-07-27) |
| S-W's 21–35 s/step and 175–290 A40-h | **ESTIMATED** | `V6_TRAINER_DESIGN.md §5` — extrapolation from the two rows above (26 encoder passes/sample vs v1's ~8; ~80 predictor rolls/sample vs v1's handful) |
| this programme's estimates ran **~11 % low** once, understating spend by **≈2.5 GPU-days** | **MEASURED** | `MODEL_REGISTRY.md §1.5.5` — the superseded row's *"~53 h ESTIMATED"* vs MEASURED 59.04 h |
| `step_s` is logged **already divided**, with `step_s_note` naming the divisor | **MEASURED** (mine, dev-box dry-run 2026-08-12) | `train_v6_staged.py`; the older trainers' accumulated counter is what produced the false *"430 s/step"* alarm |

> ⚠️ **Registry-vs-doc conflict, reported per the rules.** `V6_TRAINER_DESIGN.md §5` attributes
> the 59.04 h basis to **"v4.2"**. The registry says that number belongs to
> **`flagship-v4-fromscratch` (§1.5.5)**; `flagship-v4.2-30k` (§1.5.3) is a *different* arm,
> **killed at ~step 5 k**, and has no 30 k cost. **The registry wins.** The estimate itself is
> unaffected — the s/step basis is the same two MEASURED runs — but the design note's basis line
> needs the name corrected (filed as F4 in `BACKLOG.md`).

### ⭐ RECOMMENDATION — (b), with the re-cost rule pre-registered NOW

Authorise **≤ 12 A40-hours** (≈ step 1 200–2 000 at the top of the band), which reaches the
step-500 `step_s` read and a first O6 spectrum series. Then apply this rule, **both outcomes
committed in advance**:

| MEASURED s/step at step 500 | pre-registered action |
|---|---|
| **< 21** | band was pessimistic — proceed to 30 k and re-cost the whole ladder down in the run row |
| **21–35** | proceed to 30 k as planned. No further decision needed |
| **35–50** | apply the cost levers in `V6_TRAINER_DESIGN.md §5` order — **`--o5-k 10` first** — relaunch, re-measure at step 500. ⛔ Never silently shorten `--plan-steps`: that is §4b and it is binding |
| **> 50** | ⛔ **STOP and diagnose.** That is >2× the top of a band built from two MEASURED A40 runs, so it is a **defect** (dataloader, thread count, horizon, MooseFS) and not a cost. Cutting steps would hide it |

**Cost to defer:** **v6 does not start.** This is the binding decision.

---

## D2 — Which pod runs S-W, and does one have to be provisioned? ⛔ BLOCKS THE LAUNCH

**Question:** S-W occupies one A40 for **7–12 days**. Both current pods have assigned tracks.
Which yields, or is a third provisioned?

| option | consequence |
|---|---|
| **(a) ⭐ provision a dedicated A40 for S-W** | v6 starts tomorrow; neither existing track is disturbed. Costs one pod for ~2 weeks |
| **(b)** S-W on pod5 after the v5.8f release row + W5 close | v6 slips by the release row's remaining wall-clock; pod5 also owns W5 (see D3) |
| **(c)** S-W on pod4 after PH0, deferring PH1 | v6 starts sooner, but PH1 is the source of S2 — and S2 already gates S-S (D6). Survivable *only* because S-S is two stages away |

**Evidence**

| fact | class | source |
|---|---|---|
| pod4 = the VLM track (PH0 → PH1); pod5 = the v5.8f release track (T1 → four families → W5) | **INHERITED** | `OVERNIGHT_PLAN_2026-08-11.md §0` — ⚠️ *not* re-verified against the live fleet by this agent (no pod access) |
| ⛔ *"Never add GPU/RAM load to a pod that is training, and never eval on a training pod"* | binding | `CLAUDE.md` Invariants |
| **BOTH pod4 and pod5 refuse connections on their own `$RUNPOD_PUBLIC_IP:$RUNPOD_TCP_PORT_22`** (sshd up, both directions tried) — stale mappings, no `.runpod.internal` DNS | **MEASURED 2026-08-11** | `CLAUDE.md` traps preflight |
| ⛔ pods have **no git credentials**: `git fetch` HANGS, and a `checkout -B` after a failed fetch RESETS the tree to an ancient HEAD and destroys shipped files (pod5 HEAD `6d714ad`, weeks old, tree current) | **MEASURED 2026-08-11** | `CLAUDE.md` traps preflight |

### ⭐ RECOMMENDATION — (a), and treat file-ship as a first-class step

Provision one A40 for S-W. **Whatever pod is chosen, the code arrives by md5-verified FILE-SHIP
(xz+b64 PTY push or the HF stack-tar relay), never by git**, and the direct SSH mapping is
**probed with `ssh -n … 'echo OK'` before any transfer is built on it** — both mappings were
measured dead yesterday. The design note's §3.0 (md5 + `grep -c assert_isolation` + `--dry-run`)
is the pre-launch gate and takes ~20 s.

**Cost to defer:** **v6 does not start.** Co-blocking with D1.

---

## D3 — Does W5/E-H1 (v5.8f's 6 s baseline) run BEFORE v6, or in parallel?

**Question:** `HIERARCHY_VOCABULARY §4b` promotes W5/E-H1 to a **REQUIRED precursor**. Does that
mean *before the launch*, or *before the comparison*?

| option | consequence |
|---|---|
| **(a) serial** — W5 completes, then S-W launches | v6 slips by W5's wall-clock for **zero technical gain** |
| **(b) ⭐ parallel** — W5 on pod5, S-W on its own pod | v6 starts tomorrow; the yardstick lands long before it is needed |

**Evidence**

| fact | class | source |
|---|---|---|
| *"E-H1/W5 … is promoted from queued to REQUIRED precursor — it baselines v5.8f at 6 s before v6 trains against it"* | binding directive | `HIERARCHY_VOCABULARY.md §4b` |
| **S-W consumes no v5.8f MEASUREMENT.** S-W's required gate is **P1 / P3 / P6**, none of which reference v5.8f, and no path reads a v5.8f **checkpoint, gate JSON or eval result**. ⚠️ v6 *does* import v5.8f **code** (`A_MAX`/`KAPPA_MAX`, `build_train_episodes`, `make_sampler`, `UnicycleEmission`) — deliberate reuse of gated parts, not a dependency on a v5.8f number | **MEASURED** (mine, source read 2026-08-12) | `train_v6_staged.py` `STAGE_GATE_SPEC["S-W"]`; imports at `train_v6_staged.py:116, 1001, 1159` and `v6.py:1096` |
| W5 is **1 GPU**, queued on pod5 | **INHERITED** | `BACKLOG.md` E3; `OVERNIGHT_PLAN_2026-08-11.md` B5 |

### ⭐ RECOMMENDATION — (b) PARALLEL, with the constraint stated rather than the schedule bent

The precursor binds the **claim**, not the **launch**. Pre-register it that way:

> **No v6-vs-v5.8f comparison at 6 s is quotable until W5/E-H1 lands.** S-W's own gate
> (P1/P3/P6) does not reference v5.8f and is unaffected. If S-W passes its gate before W5
> lands, the gate stands and the *comparison* waits.

**Cost to defer:** if the PI picks serial anyway, v6 slips by W5's wall-clock. Nothing else.

---

## D4 — Accept the 6 s horizon's window loss, or rebuild the episode cache?

**Question:** the 6 s horizon costs windows per episode, because `max_horizon` enters the
windowing as `t_max = frames − window − max_horizon`. Accept, or rebuild the cache with longer
episodes?

### ⭐ First — the loss is SMALLER and MORE LOCALISED than the design note implies

**MEASURED by me 2026-08-12** (arithmetic on `train_v6_staged.py:1069–1084` + `V6Config`
defaults, re-instantiated on this box). v6 derives `max_horizon` **per stage, from that stage's
live loss terms** — so the 6 s cost lands only where a 6 s target actually exists:

| stage | live terms | derived `max_horizon` | windows per 120-frame episode | vs v5f's 94 |
|---|---|---|---|---|
| **S-W** (`--o1-k 10 --o5-k 20`, as specced §3.1) | O1/O2/O3/O5/O6 | **20** | **94** | **unchanged** |
| S-W with `--o5-k 60` (the LF4 lever) | + 60-step O5 roll | 60 | 54 | −42.6 % |
| **S-T** (λ_plan 1.0 ⇒ 60-step plan target) | T1 + plan | **60** | **54** | **−42.6 %** |
| **S-S** (`stride_str` 20) | S1 | **20** | **94** | **unchanged** |
| S-J (all terms) | all | 60 | 54 | −42.6 % |

⇒ **the window loss does not apply to S-W as specced, and does not apply to S-S at all.** It is
**unavoidable on S-T/S-J** — a 6 s planner has no target inside a shorter window — and it is
**optional on S-W**, where it is a consequence of choosing `--o5-k 60` (see the decided-by-evidence
list below).

| option | consequence |
|---|---|
| **(a) ⭐ accept** | S-W and S-S run at v5f's 94 windows; S-T/S-J at 54. Record the count in the run row |
| **(b)** rebuild the cache with longer episodes | recovers windows on S-T/S-J only, at the cost of a full cache rebuild — and **does not restore comparability with v5f anyway**, because a different episode length is a different window distribution regardless |

**Evidence**

| fact | class | source |
|---|---|---|
| `t_max = ep.frames.shape[0] − window − max_horizon` | **MEASURED** (mine, source read) | `stack/tanitad/data/_contract.py:120` (`EpisodeWindowDataset`, the class the v2 path uses) |
| v4's inherited `plan.max_horizon` is **20**; v6 refuses to inherit it and prints its own beside it | **MEASURED** | `V6_TRAINER_DESIGN.md §3.6` + `train_v6_staged.py:1077–1096` |
| per-stage derived horizons 20/60/20/60 and the window counts above | **MEASURED** (mine, 2026-08-12) | computed from `V6Config` (`stride_tac` 5, `stride_str` 20, `plan_steps` 60, `window` 6) and the `need` expression at `train_v6_staged.py:1071–1075` |
| **F = 120 frames per episode** | **INHERITED** | `V6_TRAINER_DESIGN.md §3.6`, consistent with the cache name `…-w120-…`. ⚠️ **Not verified by this agent** (no pod access). The general form is `(F−66)/(F−26)`; **verify on the pod from the trainer's own `[v6] windowing:` line, which prints the realised count** |

### ⭐ PARITY — settled, and it does not block either option

**Parity is EPISODE SELECTION**, not windowing: `physicalai-train-e438721ae894`, **2376
episodes**, skip-hash **`f09e44db`** (`CLAUDE.md` Invariants). Therefore:

| | verdict |
|---|---|
| changing `max_horizon` | ✅ **does not touch parity** — it is a windowing/training-config choice inside episodes that were already selected. `--require-parity` stays on and still guards the corpus key |
| a cache rebuild that re-extracts **longer clips from the SAME 2376 episodes** | ✅ admissible — no re-selection |
| a cache rebuild that **re-picks which clips/episodes enter** | ⛔ **REFUSE.** *"Anything that re-selects episodes breaks cross-arm comparability and must be refused"* |
| O4 interaction weighting | ✅ **reweights the draw, never removes a window** (`floor > 0`), so every arm still sees all 2376 episodes |

⇒ option (b) is *not* automatically a parity violation, but it **must be built as a re-extraction
of the same episode list, and that must be proven by the corpus key + skip-hash surviving the
rebuild** — otherwise it is refused.

### ⭐ RECOMMENDATION — (a) accept, and record the window count in every run row

**Cost to defer:** **zero on the critical path.** S-W and S-S are unaffected, so this decision is
not needed until S-T — i.e. after S-W's 7–12 days plus its gate. Defer it freely.

---

## D5 — E-ENC: one shared encoder, or per-layer encoders?

**Question:** does each layer get its own encoder, and do we spend a second full S-W to find out?

| option | params | consequence |
|---|---|---|
| **(a) ⭐ shared encoder + per-layer adapters** (default) | **87 893 449** (87.89 M) | one visual substrate; the field's shape; cheapest |
| **(b)** per-layer encoders (`--per-layer-encoders`) | **120 743 881** (120.74 M) | each layer sees the input at its own rate; must EARN +32.85 M |
| **(c)** run the matched pair as a pre-registered arm | (b) vs shared at `--pred-dim 960` = **118 105 417** (118.11 M), **gap 2.19 %** | the only attributable comparison — and it **doubles the S-W spend** |

**Evidence**

| fact | class | source |
|---|---|---|
| 87 893 449 / 120 743 881 / 118 105 417 at `--pred-dim 960`, `gap_frac` **0.0219** | **MEASURED (mine, re-instantiated on the dev box 2026-08-12)** | `V6Stack.param_report()` + `matched_param_config()`; reproduces `V6_TRAINER_DESIGN.md §2` exactly |
| all three arms are inside the **sub-300M INVARIANT**, and `build_stack_from_args` refuses otherwise **before any GPU time** | **MEASURED** (mine, dry-run) | `[v6] params 87.89 M / budget 300 M` |
| every frontier system — **V-JEPA 2, DINO-WM, Drive-JEPA** — uses ONE encoder with downstream consumers | **PUBLISHED** | `JEPA_PHYSICS_SURVEY.md §1, §4` (arXiv 2506.09985 · 2411.04983 · 2601.22032) |
| matching by eye is how an arm wins on **capacity** and is read as winning on **architecture** — the C6 confound class | binding | `V6_TRAINER_DESIGN.md §2`; `CLAUDE.md` operating standard 5 |

### ⭐ RECOMMENDATION — (a) for the first S-W; run the matched pair SHORT, not full-length

1. **The first S-W runs the shared encoder.** The field prior is PUBLISHED and unanimous, the
   arm is 32.85 M cheaper, and a tie already goes to it by the pre-registered rule.
2. **Do not spend a second 175–290 A40-hour S-W to answer E-ENC.** The cheapest discriminating
   experiment is the matched pair at a **reduced, equal step count** (both arms, same steps, same
   seed, same corpus), decided on **per-layer P-battery pass rate**, tie → shared encoder.
3. If the arm runs, it must be the matched pair — `--per-layer-encoders` (120.74 M) vs shared
   `--pred-dim 960` (118.11 M) — and the **2.19 % residual gap is quoted**, because "matched"
   with a 30 % gap is not matched.

**Cost to defer:** **zero.** The default is well-supported; the arm can run at any point after
the first S-W gate.

---

## D6 — S-S's only REQUIRED gate probe cannot be computed before PH2. ⚠️ NEW — found in the source

**Question:** S-S's gate requires the **STRATEGIC family**, whose supervision (**S2 `g_str`**)
arrives only from the VLM pipeline's **PH2**. So what happens when S-S finishes?

**What the code will actually do — MEASURED, not predicted**

| fact | class | source |
|---|---|---|
| `STAGE_GATE_SPEC["S-S"]["required"] = ("STRATEGIC_family",)`, criterion *"computable at all (measured vs n/a today)"* | **MEASURED** (mine, source read 2026-08-12) | `stack/scripts/train_v6_staged.py:231–241` |
| *"**S2 (`g_str` supervision) is not wired here and must not be faked.** It arrives from the PH0→PH1→PH2 VLM/geometric pipeline"* | binding | `V6_TRAINER_DESIGN.md §3.3` |
| `pass: null` (a required probe did not run) is **⛔ INCONCLUSIVE, and inconclusive is NOT a pass** — refused unless `--allow-inconclusive-gate` **AND** a non-empty `--gate-off-reason`, stamped into `config.json` and printed as a banner | **MEASURED** | `V6_TRAINER_DESIGN.md §4.1`; `train_v6_staged.py` |

⇒ **S-S will terminate at `pass: null` unless PH2 has landed, and S-J will then refuse to
launch.** This is the gate working correctly, not a bug — but it is a *scheduling* fact that
should be decided now rather than discovered at day 20.

| option | consequence |
|---|---|
| **(a)** hold S-S until PH2 delivers `g_str` | the ladder stalls at S-T for however long PH1+PH2 take |
| **(b)** run S-S on **S1 only** and pass `--allow-inconclusive-gate --gate-off-reason "S2 unwired: PH2 pending"` | honest and auditable, but ships an inconclusive stage into S-J |
| **(c) ⭐ pre-registered gate amendment**: promote **`S1_ade_8_30s`** from `reported` to `required` for the PH2-pending case, keeping STRATEGIC as `n/a` **with its reason and its n** | S-S gets a real, computable criterion (*"S1 ADE(8–30 s) beats CV/corridor baselines at T1"*) instead of an override |

### ⭐ RECOMMENDATION — (c), and it is a ~1-hour code change in `stack/`, not a GPU decision

(c) keeps X5's rule intact (*a failed stage never propagates upward*) while removing the need for
a blanket override, and it satisfies the four-families rule the honest way: **the STRATEGIC family
is reported `n/a` with its reason and its n**, never silently dropped. The amendment must be
pre-registered **before** S-S runs, not after seeing S1's number.

**Cost to defer:** low for now — S-S is two stages and ≥8 A40-days away. But the amendment must
land **before** S-S starts, so it belongs in the backlog now (filed as F1).

---

## Decided by evidence — NOT escalated (each has a sound default and a stated reopen trigger)

`CLAUDE.md`: *a blocked PI decision blocks ONE item, not the programme.* These were decidable
from the sources, so they are decided.

| # | question | decision | basis | what would reopen it |
|---|---|---|---|---|
| E1 | S-W `--o5-k`: 20 or 60? | **20** — as `V6_TRAINER_DESIGN §3.1` ships it | the 175–290 h band is built for k=20; the S-W gate (P1/P3/P6) is a 2 s-scale battery; k=60 triples the O5 roll **and** costs 42.6 % of windows (D4) | LF4 (`JEPA_PHYSICS_SURVEY §3`) being promoted to a flagship lever — then it is **re-costed first**, not flag-flipped |
| E2 | uplink: `stopgrad` or `ema`? | **`stopgrad`** (default) | `--uplink ema --ema-decay 0.996` is listed as a pre-registered **control arm**, not a default | the control arm being scheduled |
| E3 | `--o4-alpha` | **1.0** (default) | O4 **reweights the draw, never removes a window**; `alpha=0` reproduces uniform exactly and is the attributability control | attribution dispute on O4 |
| E4 | gradient isolation | **ON, both levers** | X3 is binding; the off arm needs `--i-know-this-is-the-control-arm` and preflight refuses without it | the isolation control arm being scheduled |
| E5 | schedule S-J up front? | **No** — run it only if S-T/S-S plateau | `JEPA_PHYSICS_SURVEY §5`; `V6_TRAINER_DESIGN §3.4` | an S-T/S-S plateau |
| E6 | do we still need the erosion argument for staging? | **No** — staging stands on the **field evidence** and on **consumer invalidation**, not on erosion | **H-COTRAIN REJECTED within the measured range** — MEASURED, `h_cotrain_curve.json`: every probed variable *rose* (curvature 0.213→0.513 encoded, 0.225→0.704 predicted), effective rank **expanded 53 %** (participation ratio 4.53→6.94 of 2048), P1 went FAIL→PASS at 20 k | a λ_plan **0** control arm (which S-W *is*) contradicting it |
| E7 | SIGReg (O6): keep? | **Keep, λ 0.1** | **SIGReg VALIDATED** — MEASURED retention **1.532** vs the ≥0.8× gate (`h_cotrain_curve.json`); PI kept it 2026-08-11 | rank retention <0.8× in S-W's own spectrum series |

---

## Appendix — the numbers this sheet depends on, with their artifacts

| number | value | class | artifact |
|---|---|---|---|
| v1 s/step (A40) | 6.374 | MEASURED | `MODEL_REGISTRY.md §1.2` |
| `flagship-v4-fromscratch` s/step (A40) | 7.085 | MEASURED | `MODEL_REGISTRY.md §1.5.5` |
| v6 shared-encoder params | 87 893 449 | MEASURED (mine 2026-08-12) | `V6Stack.param_report()` |
| v6 per-layer params | 120 743 881 | MEASURED (mine 2026-08-12) | `V6Stack.param_report()` |
| v6 matched arm (`--pred-dim 960`) | 118 105 417 · gap 2.19 % | MEASURED (mine 2026-08-12) | `matched_param_config()` |
| v6 test suite | **80 passed**, 0 warnings | MEASURED (mine 2026-08-12) | `pytest tests/test_v6_staged.py -q -W error::UserWarning` |
| SIGReg rank retention | **1.532** (gate ≥0.8×) | MEASURED | `h_cotrain_curve.json` |
| W7-FULL `sel_gap` | **3.2075** (gate ≤0.4505 m) — **FAIL** | MEASURED, **T0** | `w7_full_gate.json` |
| W7 roll-cost argmin error-rank | **132.32 of 256** (median = 128) | MEASURED, EXPLORATORY | `w7_selection_rules.json` |
| consumer invalidation on trunk repair | frozen selector **0.7933 → 4.4159** | MEASURED, **T0** | `w7_full_gate.json` (`reference.frozen_argmax_selected_ade`, `mini_eval.frozen_selected_ade_in_run`) |
| v5.8f deployed arm (the incumbent v6 must beat) | selected **0.4815** · oracle **0.1077** · accel MAE **0.515** | MEASURED, **T0**, 881 grid | `MODEL_REGISTRY.md §1.14` |
