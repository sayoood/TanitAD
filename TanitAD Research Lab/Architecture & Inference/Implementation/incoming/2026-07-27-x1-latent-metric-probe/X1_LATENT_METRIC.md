# X1 — IS THE METRIC EGO-MOTION IN THE LATENT AT ALL? plus X4 across every arm, X1b's rig split, and C8's selection rule

**Stream:** Architecture & Inference — implementation. **Date:** 2026-07-27 (Europe/Berlin).
**Host:** dev box only. **⛔ No pod was contacted** — pod1 (training), pod2 (owed controls), pod3
(classifier build) and the eval pod (pseudo-simulation) were never touched. GPU use: one encoder pass
on the local RTX 4060, memory-checked before allocating because a sibling agent shares it.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID, no frame, no raw content appears in this folder.

**Runs what the source research pre-registered** —
`…/Research/2026-07-27-imagination-perception-manifold/MANIFOLD_MISMATCH_RESEARCH.md` §6 (X1, X1b, X4)
and §5 (C1, C3/C6, C8), with `PRE_REGISTRATION.md`'s falsifiers F1–F4. Nothing here re-derives that
design; it executes it.

---

# 0. VERDICT

> ## 🔴 **X4 FIRST, BECAUSE IT WAS TIME-CRITICAL: v4 CANNOT INHERIT v1's CONCLUSION — IT DOES NOT CARRY THE INSTRUMENT.** All three committed v4 logs (v4.1 to 12 500 steps, v4-from-scratch to **29 999**) contain **no `g_*` key at all**, and `train_flagship_v4.py` / `flagship_v4.py` contain **no reference to `grounding_losses`, `MetricInverseDynamics` or `StepDisplacementReadout`**. The real-vs-imagined decode gap is **UNMEASURED on v4 and unmeasurable from any committed v4 artifact.**
>
> ## ⇒ **The registry row E1 asks for must be scoped to the v1 family (v1 / nospeed / v2 / v3enc / expA). Writing it as a program property would hand v4 a claim never tested on it.**

> ## ⛔ **X4's UNIVERSALITY FALSIFIER FIRED, AND THE WAY IT FIRED IS THE RESULT.** The gap widens on every instrumented arm — except on the **no-speed control at the `str` level**, where it is flat (1.93 → 1.70 → 1.76 → 1.77 over 20 k steps). With `v0` injected the gap grows **×14.2 / ×13.8 / ×3.3** (op/tac/str); without it, **×2.4 / ×2.6 / ×0.9**.
>
> ## ⇒ **The widening is not a property of the architecture. It is manufactured ~6× faster by the injected `v0` channel.** And the attributing ablation — one level in the source report — now holds at **all three**: removing `v0` moves the PERCEIVED decode by **+3.0 / +3.8 / +3.1 %** and the IMAGINED decode by **×6.87 / ×6.27 / ×5.75**.

> ## ⭐ **X1's ANSWER IS HORIZON-DEPENDENT, AND NEITHER PRE-REGISTERED OUTCOME IS THE WHOLE TRUTH.** At **k = 1 (0.1 s)** a fresh 4×-width probe reaches **0.1888 m [0.1635, 0.2158]** against the trained head's **0.4587 m** — **58.8 % better, separated**, and it meets the source's explicit *"≲ 0.2 m ⇒ the information IS there"* bar. At **k = 4** the margin is **0.019 m, not separated**. At **k = 20** the **trained head WINS by 19.4 %, separated** — a fresh probe cannot reach it.
>
> ## ⇒ **The head is mis-attached at 0.1–0.2 s (C1's premise holds, and only there) and is at or above the frozen-latent ceiling from 0.4 s on (C1 cannot help, and could hurt).**

> ## ⭐⭐ **AND THE MECHANISM IS SETTLED FURTHER THAN THAT: ONE SCALAR BEATS EVERY LATENT PROBE.** The true-`v0` straight-line integrator — no latent at all — scores **0.0076 / 0.0236 / 0.0860 / 2.0143 m** at k = 1/2/4/20, i.e. **24.8× / 15.4× / 8.3× / 1.7× better** than the best probe the latent can support. A ridge probe from the latent to speed gets **R² ≈ 0.61**, flat across horizons, and **integrating that perceived speed reproduces the entire pair-probe's accuracy** (0.1998 vs 0.2014 at k = 1). The **second latent slot adds ~nothing** (0.2016 single-slot vs 0.2014 paired).
>
> ## ⇒ **The latent carries an R² ≈ 0.61 speed reading and almost no other metric ego-motion. The readout's competence is dominated by integrating a speed it is HANDED. `C3` — teach the ENCODER to perceive metric ego-motion — is the item; `C1` is a nearly-free short-horizon repair, not the fix.**

> ## ⛔ **X1b: the two-rig confound is REFUTED as the binding constraint.** Within-cluster fits are **2.4–3.2 % WORSE** than pooled, nowhere near the pre-registered **≥ 20 % better**. The matched **random-split control** costs **7.5–8.6 %** (separated) against the rig proxy's 2.4–3.1 % (not separated) — so the proxy is structural but an order of magnitude too small to matter. **⇒ C3/C6 are not futile for the rig reason.**

> ## ⭐ **C8: the selection rule beats BOTH flat answers — and the ADE-OPTIMAL RULE IS THE WRONG ONE.** In the deployable regime a per-lead-time rule gives `ade_0_2s` **0.8355 [0.7871, 0.8855]** vs `always_op` 0.9554 (**−0.1199 [−0.1718, −0.0724]**, sep) and `always_str` 0.8710 (**−0.0355 [−0.0572, −0.0142]**, sep) — but fitting the switch point to ADE costs **3.1× of deployable `T_blind` (2.5 s → 0.8 s) to buy 0.7 % of ADE.**
>
> ## ⇒ **The rule the data supports is `op` for lead times ≤ 0.5–0.8 s, `str`/`tac` beyond — Pareto-dominant over the flat `always_str` swap (ADE 0.8597 vs 0.8710 at identical `T_blind` 2.5 s). Do NOT fit the switch point to `ade_0_2s`.** In the privileged regime the same rule also **widens the beats-CV window at BOTH ends — 0.4 s … 18.5 s, where `always_op` gives 0.4 … 7.4 s and the flat `str` swap only 0.6 … 18.5 s.** ⚠️ And readout selection does **not** rescue the deployable regime: even the per-window oracle over the whole bank (0.5901) only just reaches the `hold_v0` floor (0.5933), and **no rule beats CV at any lead time there.**

---

# 1. PRE-REGISTRATION — what was fixed before any number existed

The falsifiers for X1, X1b and X4 were **already committed** in the source report (§6) and are quoted
verbatim in each script's docstring. This job adds three things, and all three were written into the
code **before the checkpoint finished downloading**, i.e. before any X1 number could exist. File
mtimes are the evidence and are reproducible from the working tree:

| file | written (Europe/Berlin) | what it fixed in advance |
|---|---|---|
| `scripts/x4_log_sweep.py` | 01:40 | the five step bands, the 8-number reproduction gate, the universality falsifier |
| `scripts/hf_pull_ckpt.py` | 01:42 | — (transport) |
| `scripts/x1_cache_latents.py` | 01:46 | ⭐ the **fidelity tolerance** (imagined decode ≤ 3× 0.0304 m; real-pair decode ≤ 2× 1.0129 m) and the **shuffle control that must fail** |
| `scripts/x1_probe.py` | 01:49 | ⭐ the **five ways this probe could report "no metric information" spuriously**, each with its ruling-out check (R1–R5), and the X1b random-split control |
| `scripts/c8_selection_rule.py` | 01:58 | the C8 rule family, the cross-validation, the two-objective table |

**v1's checkpoint became loadable at ~01:59.** Every X1 design decision above predates it.

## 1.1 Both outcomes of X1, committed in advance (source §6.1, restated so this file stands alone)

* **→ C1 ("head mis-attached").** A fresh probe recovers Δpose well from real latents ⇒ the metric
  information **is** in the latent, the trained head's ~1.0 m is an optimisation / multi-task-
  interference failure ⇒ the fix is the ~10-line C1 loss term at ≈0 % step time.
* **→ C3/C6 ("the latent has no metric ego-motion").** No probe recovers it ⇒ a loss term on the head
  cannot help; only a change to the **representation** (encoder-level ego-motion supervision, or a
  metric-competent teacher) can, and the registry must say the metric readout is an
  **action-integrator**, not a perception decoder.

## 1.2 ⚠️ What would make this probe report "no metric information" SPURIOUSLY

Stated before the run, each with the check that rules it out (all implemented in the two scripts):

| # | spurious-null mechanism | ruling-out check |
|---|---|---|
| **R1** | wrong preprocessing → the cached latents are not the latents the model was trained on | the **fidelity check**: the trained `step["op"]` decoding the predictor's *imagined* rollout on these very latents must stay small |
| **R2** | the probe simply under-fits | **train error reported next to test error**, plus the 4×-width arm; if train error is also ~1 m the ceiling is informational, not optimisational |
| **R3** | target or unit error (`pose_scale` 10.0, `SPEED_SCALE` 10.0, dt) | the **dt self-check** (realised displacement ÷ logged speed) and the true-`v0` integrator baseline, which must land at a sane metre value |
| **R4** | too few pairs | the **sample-size curve** at 25 / 50 / 100 % of the training pairs |
| **R5** | an unlucky episode split | episode-cluster bootstrap intervals; the split is recorded |

And the opposite direction, so the instrument is validated **both ways**: the **shuffle control** —
the same trained head fed *mismatched* real pairs — **must fail**. If it does not, the head ignores its
input and every "recovery" number is an artefact.

## 1.3 ⚠️ Corpus parity — stated, not buried

The dev-box episode cache is keyed **`physicalai-val-bb543bdf7836`** (100 eps), **not** the parity key
`e438721ae894` and not the eval pod's `physicalai-val-0c5f7dac3b11`. **Nothing measured on it is
cross-arm comparable and none of it may enter `MODEL_REGISTRY.md` as an arm result.** It is admissible
for X1 because X1's question is answered by a **within-run** contrast — fresh probe vs the trained head
vs baselines, all on one and the same windows, all with the same encoder.

**X4 and C8 do not have this problem at all:** X4 reads committed training logs, and C8 reads the
committed per-window dumps from the parity val split (`0c5f7dac3b11`, 600 eps / 596 clusters).

---

# 2. REPRODUCTION GATES — passed before anything new was quoted

`MEASURED` · `artifacts/x4_log_sweep.json` → `reproduction_gate`, `artifacts/c8_selection_rule.json`
→ `reproduction_gate`.

| # | committed number | source | reproduced | tol |
|---|---|---|---|---|
| 1–5 | v1 `op` real/imagined at the five bands: **2.2486/0.9541 (n=20)**, 1.4833/0.2442 (40), 1.2821/0.1034 (**59**), 1.1004/0.0516 (40), **1.0129/0.0304 (41)** | `MANIFOLD_MISMATCH_RESEARCH.md` §2.2 | ✅ **exact, including every row count** | 1e-4 |
| 6–7 | v1 `str` 16.0284/4.5320 → 1.8735/0.1606 | ibid. | ✅ exact | 1e-4 |
| 8 | nospeed `op` @19–21k **1.1333/0.3543** | ibid. §2.3 | ✅ exact | 1e-4 |
| 9 | `ade_0_2s` of the `op` readout, imagination + true actions = **0.3839** | `BLIND_IMAGINATION.md` | ✅ 0.3839 | 5e-4 |
| 10 | `ade_0_2s` with the `str` readout = **0.1950** | ibid. | ✅ 0.1950 | 5e-4 |
| 11 | `de@0.5s` paired `str` − `op` = **0.0449 [0.0350, 0.0549]**, separated | `rung0_own_readout_short_horizon.json` | ✅ exact (`c8_selection_rule.json` → `regimes.own_kinematic_DEPLOYABLE.paired_str_minus_op_at_grid["0.5s"]`) — **and it is the `own_kinematic` regime, not the true-action one** (see §7.1) | interval |
| 12 | beats-CV window **0.4 s … 7.4 s = 71 / 185 steps** | `BLIND_IMAGINATION.md` §2.5 / line 325 | ✅ exact (`reproduction_gate.beats_cv_window_convention` = steps [4, 74], 71) — **and it recovered the CONVENTION** (see §7.1) | steps |
| 13 | v1 imagined-decode magnitude on a *fresh, held-out* corpus | X1 fidelity, §4.1 | ✅ 0.0437 m vs the 0.0304 m train band — the defect reproduces off-corpus at **13.4×** | 3× |

⇒ Every number below is quoted on top of a passing gate.

---

# 3. ⭐ X4 — THE REAL-VS-IMAGINED CONTRAST ACROSS EVERY COMMITTED ARM

`MEASURED` · tier **DECISION-GRADE** for the two headline findings (both are reproduced-gated,
multi-arm, and carry their falsifier) · `artifacts/x4_log_sweep.json` · 14 logs scanned, 9 carrying the
grounding instrument, 6 distinct arms · 0 GPU, ~20 CPU-seconds.

## 3.1 🔴 FINDING 1 — **v4 DOES NOT CARRY THE INSTRUMENT AT ALL.** It cannot inherit v1's conclusion, and it cannot refute it either.

The brief's time-critical question was "run X4 on v4 **before v4 inherits v1's conclusion**". The
answer is stronger and more awkward than either outcome anticipated:

| log | steps | grounding keys |
|---|---|---|
| `taniteval/results/trainlogs/flagship-v4.1-10k_train_log.jsonl` | 0 … 12 500 | **none** |
| `…/2026-07-23-v4-eval-harness/flagship-v4.1-10k_train_log.jsonl` | 0 … 12 500 | **none** |
| `…/2026-07-26-v4-restart-lever/raw/v4fs_train_log.jsonl` | 0 … **29 999** | **none** |

v4's trainer logs `plan_ade`, `oracle_ade`, `wm`, `planner`, `lambda_plan`, four `gnorm_*` — and **no
`g_{lvl}_{mid,fwd}` at all**. Source-confirmed, not inferred from the logs alone:
`stack/scripts/train_flagship_v4.py` and `stack/tanitad/models/flagship_v4.py` contain **no reference**
to `grounding_losses`, `MetricInverseDynamics` or `StepDisplacementReadout`; the v4 loop is a different
objective (planner + world-model terms), and v4.1 *warm-starts* trunk+grounding from v1's checkpoint
(`MODEL_REGISTRY.md`:719) without re-training or re-logging the grounding terms.

> ### ⇒ **The real-vs-imagined decode gap is UNMEASURED on v4, and no committed v4 artifact can measure it.** Writing v1's F4 row into the registry as a program-wide property would make v4 inherit a claim that has never been tested on it. **The registry row must be scoped to the v1 family (v1 / nospeed / v2 / v3enc / expA), and v4 needs an eval-side measurement — which is now the cheapest open item in this stream (~20 GPU-min, the same instrument as X1's fidelity stage).**

*(Absence-at-one-location rule honoured: three separate log families were globbed — the curated
`taniteval/results/trainlogs` mirror, the research-hub `raw/` dumps, and the four worktree run dirs —
and the v4 source was read directly rather than inferred from the logs.)*

## 3.2 🔴 FINDING 2 — the pre-registered falsifier FIRED. The widening is **not** universal; it is **manufactured by the injected `v0`**.

Pre-registered: *"REFUTES universality if any arm shows the ratio SHRINKING with training."*
**It shrinks on one arm — and it is exactly the arm with no `v0` channel, at the level with the longest
imagined rollout.**

| arm | `speed_input` | `op` gap growth | `tac` | `str` |
|---|---|---:|---:|---:|
| **v1 `flagship4b-speedjerk-30k`** | **True** | **×14.16** (2.36 → 33.37) | ×13.75 (2.79 → 38.40) | ×3.30 (3.54 → 11.67) |
| **`flagship4b-phase0-30k` (no-speed)** | **False** | ×2.44 (1.31 → 3.20) | ×2.57 (1.65 → 4.24) | **×0.92 (1.93 → 1.77) ⛔ SHRINKS** |
| `flagship4b-v2-30k` (7.7 k) | True | ×1.68 | ×1.71 | ×1.28 |
| `flagship4b-v3enc-30k` (10.8 k) | True | ×1.79 | ×1.56 | ×1.07 |
| `expA-nodrop` (2.0 k, own-range) | True | ×2.65 | ×3.12 | ×2.34 |

The no-speed arm's `str` ratio sequence is **1.93 → 1.70 → 1.76 → 1.77** — flat, not widening, over
20 000 steps.

**Direction of the correction:** the honest statement is not "the gap widens universally" but
**"the gap widens on every arm, and it widens ~6× faster when `v0` is injected."** The two long runs
are the only ones that reach the late bands, and they differ in exactly the three fields §2.3 of the
source report identified.

## 3.3 ⭐⭐ FINDING 3 — the attributing ablation, extended from ONE level to ALL THREE, and the signature is tight

Matched band **19–21 k** (both arms alive), same corpus, same seed, same weights, same horizons —
config-diff is `action_dim` 2→3, `aux_accel`, `jerk_weight` and nothing else:

| level | REAL pair (`*_mid_de_m`) v1 → nospeed | IMAGINED (`*_fwd_ade_m`) v1 → nospeed |
|---|---|---|
| `op` | 1.1004 → 1.1333 · **+3.0 %** | 0.0516 → 0.3543 · **×6.87** |
| `tac` | 4.7312 → 4.9127 · **+3.8 %** | 0.1846 → 1.1582 · **×6.27** |
| `str` | 2.4216 → 2.4957 · **+3.1 %** | 0.2450 → 1.4082 · **×5.75** |

> ### **Removing the injected speed channel leaves the PERCEIVED decode unmoved to within 3–4 % at every level of the hierarchy, and degrades the IMAGINED decode by 5.8–6.9× at every level.** The source report had this at `op` only; it now holds three times over, with a spread of ±0.4 pp on the real side and ±0.6× on the imagined side.

The encoder never sees `v0` — it is an **action** into the predictor — so the real side is *expected*
to be unmoved and is. What is **not** forced by construction is that the imagined side collapses
without it, at all three timescales, by nearly the same factor. `MEASURED`, `CONFIRMED` on two arms at
parity, now at three levels.

## 3.4 Free provenance check, worth recording

The four worktree run dirs (`.claude/worktrees/fervent-perlman-1dde67/stack/experiments/*`) and the
curated `taniteval/results/trainlogs/*` mirror produce **bit-identical band statistics** for all four
shared arms. The mirror is faithful; either may be quoted.

## 3.5 ⚠️ The aggregation caveat travels with every ratio above

`*_mid_de_m` is an **endpoint** displacement error averaged over the level's horizons; `*_fwd_ade_m` is
an **ADE over waypoints 1…k**. For a monotonically growing error the endpoint statistic is the larger,
typically by ~2×. **Every ratio in this section is a bound of the right order, not a clean effect
size**; `x4_log_sweep.json` publishes `ratio_agg_corrected_x0.5` beside every raw ratio (28–30 k `op`
becomes ≈16.7×). **The trend — and every cross-arm comparison in §3.2/§3.3, which is the load-bearing
part — is invariant to any fixed aggregation factor**, because the same factor divides out.

---

# 4. X1 — THE FROZEN-LATENT PROBE CEILING

`MEASURED` · tier **DECISION-GRADE** (falsifier pre-registered, both outcomes committed, instrument
validated in both directions, paired estimator) · `artifacts/x1_cache_fidelity.json`,
`artifacts/x1_probe.json`.

**Setup.** v1 `flagship4b-speedjerk-30k` step **29999**, pulled from the gated HF repo
`Sayood/tanitad-flagship-4b-speedjerk` (3.309 GB, strict load of both `model` and `grounding`),
`state_dim` **2048**. 100 local val episodes × 199 frames = **19 900 frozen-encoder latents**, encoded
in **151 s** on the dev-box RTX 4060 (GPU checked at 6.93 GiB free before allocating). Episode-disjoint
split **60 train / 40 test**, seed 0. ~11 900 training pairs and ~1 950 test pairs per horizon.
⚠️ **NON-PARITY corpus** — §1.3.

## 4.1 The instrument passed in both directions before any probe was fitted

| pre-registered check | rule | got | |
|---|---|---:|---|
| **fidelity — imagined decode must stay small** (R1) | ≤ 3 × 0.0304 m | **0.0437 m** | ✅ |
| **fidelity — real-pair decode must be in range** | ≤ 2 × 1.0129 m | **0.5842 m** | ✅ |
| **shuffle control — must FAIL** | mismatched pairs > 1.5 × matched | **1.3117 m** (2.25×) | ✅ |
| **dt self-check** (R3) | realised displacement ÷ logged speed ≈ 0.1 s | **0.1007 s** | ✅ |

⇒ the cached latents are the latents the model was trained on, and the trained head does react to its
input. **The real-vs-imagined defect reproduces on held-out data with a fresh corpus: 0.5842 / 0.0437 =
13.4×** (train-band was 33.4×; different corpus and different statistic — see §3.5 — but the same sign
and the same order).

## 4.2 🔴 THE RESULT: the answer is HORIZON-DEPENDENT, and that splits the two diagnoses

All numbers metres, on the **same held-out pairs**, `MEASURED`:

| k (lead time) | ridge | MLP h512 | **MLP 4× (h2048)** | **trained `invdyn`** | probe − head, paired [95 % CI] | ⛔ falsifier |
|---|---:|---:|---:|---:|---|---|
| **1 (0.1 s)** | 0.2014 | 0.1944 | **0.1888** | 0.4587 (`op`) | **−0.2699 [−0.3348, −0.2078] ✅ sep** | **does NOT fire — probe beats head by 58.8 %** |
| **2 (0.2 s)** | 0.4019 | 0.3668 | **0.3645** | 0.5204 (`op`) | −0.1559 [−0.2161, −0.0967] ✅ sep | fires by **0.04 pp** (29.96 % vs the 30 % bar) |
| **4 (0.4 s)** | 0.7970 | **0.7136** | 0.7659 | 0.7849 (`op`) | −0.0190 [−0.1105, +0.0747] ❌ not sep | **FIRES** — 2.4 % |
| **20 (2.0 s)** | 3.8675 | 3.6513 | **3.5136** | **2.9419** (`str`) | **+0.5717 [+0.0911, +1.0932] ✅ sep** | **FIRES DECISIVELY — the trained head BEATS every fresh probe by 19.4 %** |

> ### ⭐ **At 0.1–0.2 s the metric information IS in the latent and the trained head is leaving ~59 % of it on the table. From 0.4 s onward there is nothing left to recover: a 4×-width MLP with 11 900 training pairs cannot beat the head, and at 2 s it LOSES to it by 19 %.**
>
> ### **The source report's explicit "information IS there" threshold — *"if a fresh probe reaches ≲ 0.2 m at k = 1"* — is met exactly: 0.1888 m [0.1635, 0.2158].**

**R2 (under-fitting) ruled out:** train errors are reported beside test (k=1: ridge 0.1435 / h512 0.1129
/ h2048 0.1218; k=20: 2.9497 / 1.4424 / 1.6609). At k=20 the MLPs fit the *training* pairs to 1.44–1.66 m
and still test at 3.5 m — that is a generalisation ceiling, not an optimisation failure.
**R4 (too few pairs) partly live and stated:** the ridge sample-size curve still improves with data
(k=1: 0.2526 → 0.2352 → **0.2014** at 25/50/100 %), so the true k=1 ceiling may be *slightly better*
than 0.1888 — which strengthens the k=1 conclusion and does not touch the k≥4 one, where more data
does not close a 19 % deficit.

## 4.3 ⭐⭐ The finding that decides the mechanism: a single scalar beats every latent probe

| k | best latent probe | **true `v0` integrator** *(one scalar, no latent at all)* | latent probe ÷ `v0` |
|---|---:|---:|---:|
| 1 | 0.1888 | **0.0076** | **24.8×** |
| 2 | 0.3645 | **0.0236** | **15.4×** |
| 4 | 0.7136 | **0.0860** | **8.3×** |
| 20 | 3.5136 | **2.0143** | **1.74×** |

`baseline_true_v0_integrator_de_m` is Δx = v·k·dt, Δy = 0, Δyaw = 0 using the logged current speed —
i.e. **exactly what the `v0` action channel hands the predictor.**

And the decomposition that shows *why*:

* **A ridge probe from the latent to SPEED gets R² = 0.617 / 0.610 / 0.609 / 0.602** — mediocre, and
  flat across horizons (it is a property of the encoder, not of the horizon).
* **Integrating that perceived speed in a straight line reproduces the FULL pair-probe's accuracy**:
  0.1998 vs 0.2014 (k=1), 0.4024 vs 0.4019 (k=2), 0.8076 vs 0.7970 (k=4). ⇒ **essentially ALL of the
  latent pair's metric content is speed.**
* **The second latent slot adds almost nothing.** A ridge from `z_t` **alone** scores 0.2016 vs the
  pair's 0.2014 at k=1, and 0.8085 vs 0.7970 at k=4 (it adds ~8 % only at k=20, 4.2025 vs 3.8675).

> ### ⇒ **The "pair" is barely a pair. The latent carries a ~R² 0.61 reading of the current speed and almost no other metric ego-motion, and the readout's job is dominated by integrating a speed it is HANDED rather than one it perceives.** This is direct, within-run, held-out support for **M2 (the injected-`v0` shortcut)** — and it substantially weakens **M1**'s specific claim that the defect lives in a *contracted second slot*, because the second slot barely matters even to a probe that is free to use it.

---

# 5. X1b — RIG-CONDITIONED SCALE: the two-rig confound is REFUTED as the binding constraint

`MEASURED` · tier **CONFIRMED** (single instrument, but with a matched negative control) ·
`artifacts/x1_probe.json` → `x1b_rig`, `x1b_rig_split_meta`.

⚠️ **Method deviation, declared.** The pre-registration says *"split by per-clip `cy`"*. **The cached
episodes carry no `clip_id` and no intrinsics** — `ToyEpisode` is `frames / actions / poses /
episode_id / maneuvers` — so `cy` is not readable from the cache and the intended split is not
available on this host. Substitute: a **measured image statistic** — the per-episode mean
row-intensity profile, whose vertical structure is precisely what a ~215 px principal-point difference
moves — clustered into two groups by 1-D k-means on its first principal component (PC1 explains
**64.5 %** of profile variance; cluster separation **1.90 SD**; sizes **35 / 65**). **The cluster
identity as "rig A / rig B" is `UNVERIFIED`** — it is a proxy, and it is labelled as one.

| k | pooled fit | within-group fit | improvement | **random-split control** (same group sizes) | excess over random |
|---|---:|---:|---:|---:|---:|
| 1 | 0.2014 | 0.2062 | **−2.39 %** | −7.45 % | +5.06 pp |
| 2 | 0.4019 | 0.4129 | **−2.72 %** | −7.83 % | +5.11 pp |
| 4 | 0.7970 | 0.8216 | **−3.08 %** | −8.60 % | +5.52 pp |
| 20 | 3.8675 | 3.9915 | **−3.21 %** | −3.41 % | +0.20 pp |

> ### ⛔ **Pre-registered falsifier: "REFUTES the two-rig-confound hypothesis if within-rig probe error is not ≥ 20 % better than pooled." It is not 20 % better — it is 2.4–3.2 % WORSE. The hypothesis is REFUTED at every horizon.**

**But the control makes the negative result informative rather than empty.** Splitting the training
data in two always costs accuracy (half the pairs per fit). A **random** split of the same sizes costs
**7.5–8.6 %** and that loss is **separated** from zero; the **rig-proxy** split costs only 2.4–3.1 %
and that loss is **not separated** from zero at k = 1, 2, 4. ⇒ **the proxy is capturing something
structural — a fit transfers within a cluster better than across a random half — but the effect is an
order of magnitude too small to be what caps metric visual odometry.**

⇒ **Good news for C3/C6:** the source report's §5.1 worry that *"our corpus contains two incompatible
scale priors, which would cap any learned metric VO"* is **not** the binding constraint. C3 and C6 are
not futile for that reason. The binding constraint is §4.3: **the encoder perceives speed at R² ≈ 0.61
and that is the ceiling.**

---

# 6. THE C1-vs-C3/C6 VERDICT

> ## ⭐ **NEITHER OUTCOME AS WRITTEN. The pre-registration offered a binary — "the information is in the latent" (⇒ C1) or "it is not" (⇒ C3/C6). The measured answer is: it is there at 0.1–0.2 s and gone by 0.4 s, and even where it is there it is worth 25× less than the `v0` the model is already handed. Both outcomes were committed in advance and the honest report is that the truth is between them, with a clear ranking falling out.**

| candidate | what X1 says about it | verdict |
|---|---|---|
| **C1** — symmetric/mixed-source supervision of the metric head (~10 lines, ≈0 % step time) | ✅ **Its premise holds, but only at k ≤ 2 and its ceiling is small.** A fresh probe recovers **0.2699 m [0.2078, 0.3348]** of real-pair error at 0.1 s and **0.1559 m** at 0.2 s that the trained head does not. ⛔ At k = 4 the recoverable margin is **0.019 m, not separated**; at k = 20 it is **negative** — the head is already better than the frozen-latent ceiling, so C1 cannot help there and a mixed-source term risks *degrading* it. | ⭐ **DO IT — it is nearly free and its premise is measured — but scope it to the SHORT horizons and pre-register a rollout-quality guard.** It is a 0.1–0.2 s repair, **not** a fix for the 9.4×/33× gap. |
| **C3** — encoder-level metric ego-motion supervision | ⭐ **This is the item.** The latent's entire metric content is a speed reading at **R² ≈ 0.61**, flat across horizons; integrating it reproduces the whole pair-probe. Supervising the *encoder* to perceive speed is aimed exactly at the measured ceiling. ⚠️ Untested in our program in the *perception* form — the historic 0.61 → 0.965 win was **injection**, not perception. | ⭐ **RANK #1 among training fixes.** X1b removes the two-rig objection to it. |
| **C6** — distillation from a metric-competent teacher | not directly probed; bounded by the same monocular-scale-prior physics, and X1b says the rig conflict is not the blocker | keep as listed — weeks, not scheduled |
| **C7** — re-fit `grounding.step` on real pairs, frozen trunk | ⛔ already refuted by §3, and X1 adds the ceiling: at k ≥ 4 a *fresh, 4×-width* probe cannot beat the existing head | remains ⛔ **REFUTED** |
| **the registry claim** that the metric readout is a *perception* decoder | 🔴 X1 §4.3 is direct within-run evidence that it is dominated by **integrating an injected scalar**: one scalar beats every latent probe by 25× at 0.1 s | ⇒ **E4 (below) is now supported by a second, independent measurement** — but X2's surgical `v0`-scaling test is still the clean proof and is still owed |

---

# 7. C8 — THE HORIZON-BANKED READOUT SELECTION RULE

`MEASURED` · tier **DECISION-GRADE** · `artifacts/c8_selection_rule.json` · **zero GPU, no model
load** — only the committed per-window dense-DE dumps, v1 `flagship-30k` step 29999, parity val
`physicalai-val-0c5f7dac3b11`, **599 windows / 596 episode clusters**, K = 185 steps, dt 0.1 s.
Estimator: paired episode-cluster bootstrap, B = 2000, seed 0.

**Integrity gate first:** the three dumps (`blind-imagination`, `tblind-ladder`, `tblind-rung1`) must
describe the same window set — `eid` and `t0` are asserted equal element-by-element across all three
before any array is combined. ✅ passed. That is what makes the **complete 3-readout bank** available in
the deployable regime: `op` and `str` come from the BI dump, `tac` (`…__own__roTAC`, for both the
imagination arm and the frozen-last control) only exists in the ladder dump.

## 7.1 ⭐ Two conventions recovered by reproduction, both of which were being quoted loosely

1. **`de@0.5s` paired `str` − `op` = +0.0449 m [+0.0350, +0.0549], separated.** This reproduces the
   committed interval **exactly** — and it is the **`own_kinematic` (deployable)** regime. Under
   **true actions** the same contrast is **+0.0382 m [+0.0318, +0.0450]**. The source report quotes
   −0.0449 in a sentence about the readout swap generally; it belongs to the deployable regime
   specifically. *(Sign: positive = `str` has the larger error, i.e. `str` is worse at 0.5 s — the
   crossover, in the same direction the literature reports.)*
2. 🔴 **"beats-CV horizon" means SEPARATED-beats, not point-mean-beats.** A point-mean rule
   (`arm ≤ floor`) puts v1's `op` readout at steps **4…88**; the committed artifact says
   **0.4 s … 7.4 s = 71 / 185 steps**. The rule that reproduces **4…74 = 71 steps exactly** is the
   paired episode-cluster bootstrap with the CI lower bound of (floor − arm) > 0.
   ⇒ **the point-mean version overstates the end of the window by ~1.4 s.** Pinned in code
   (`c8_selection_rule.GATE_BEATS_CV`) so it cannot drift again.

## 7.2 The rule family, fitted out-of-fold

Five-fold cross-validation **grouped by episode cluster**; each rule is fitted on the training folds
and scored on the held-out fold, so no rule is scored on the data that chose it. Rules:
`always_{op,tac,str}` · `argmin_per_step` (per-lead-time argmin) · `crossover_op_str` (one switch step)
· `crossover_op_tac_str` (two switch steps) · plus a per-window **oracle** (not deployable, an upper
bound) and the two floors.

### `own_kinematic` — **the deployable regime**

| arm | `ade_0_2s` [95 % CI] | vs `always_op` (paired) | vs `always_str` (paired) |
|---|---|---|---|
| `always_op` — *what the harness uses today* | 0.9554 [0.9003, 1.0129] | — | — |
| `always_str` — *the flat swap that was about to be adopted* | 0.8710 [0.8179, 0.9231] | — | — |
| `always_tac` | 0.8463 [0.7954, 0.8981] | — | — |
| **`rule_crossover_op_str`** (oof) | 0.8534 [0.8029, 0.9028] | **−0.1020 [−0.1525, −0.0545] ✅ sep** | **−0.0176 [−0.0251, −0.0094] ✅ sep** |
| **`rule_argmin_per_step`** (oof) | **0.8355 [0.7871, 0.8855]** | **−0.1199 [−0.1718, −0.0724] ✅ sep** | **−0.0355 [−0.0572, −0.0142] ✅ sep** |
| `rule_crossover_op_tac_str` (oof) | 0.8355 [0.7871, 0.8855] | −0.1199 [−0.1718, −0.0724] ✅ sep | −0.0355 [−0.0572, −0.0142] ✅ sep |
| *oracle per window (NOT deployable)* | *0.5901* | — | — |
| floor `hold_v0` | 0.5933 | | |
| floor constant velocity | 0.6083 | | |

**Fitted rules (on all data):** `argmin_per_step` → `op` for the first **1.1 s**, then `tac`.
`crossover_op_str` → `op` for **1.3 s**, then `str`. `crossover_op_tac_str` → `op` 1.2 s, `tac` to
18.4 s, then `str`.

⇒ **A selection rule beats BOTH flat answers, separated, in the deployable regime** — but it is a
**12.5 %** improvement on a number that is still **41 % worse than doing nothing** (`hold_v0`, 0.5933).
Even the per-window oracle over the whole bank (0.5901) only just reaches the floor. **Readout
selection is real and free; it does not rescue the deployable regime.**

### `true_actions` — the privileged regime, where the rule also **widens the capability window at both ends**

| arm | `ade_0_2s` | **beats-CV window** (separated) |
|---|---|---|
| `always_op` | 0.3839 [0.3598, 0.4106] | **0.4 s … 7.4 s** ← the committed number, reproduced |
| `always_str` | 0.1950 [0.1767, 0.2139] | 0.6 s … 18.5 s |
| `always_tac` | 0.1865 [0.1684, 0.2056] | 0.6 s … 18.5 s |
| **`rule_argmin_per_step`** (oof) | **0.1817 [0.1641, 0.2005]** | ⭐ **0.4 s … 18.5 s** |
| `rule_crossover_op_tac_str` (oof) | 0.1819 [0.1642, 0.2006] | ⭐ 0.4 s … 18.5 s |
| `rule_crossover_op_str` (oof) | 0.1854 [0.1673, 0.2039] | ⭐ 0.4 s … 18.5 s |
| *oracle (NOT deployable)* | *0.1398* | *0.4 s … 18.5 s* |

The rule beats `always_str` by **−0.0133 [−0.0176, −0.0089] ✅** and `always_tac` by
**−0.0048 [−0.0072, −0.0023] ✅** — both separated, both small. Fitted rule: `op` → 0.7 s → `tac` →
1.9 s → `str`.

> ### ⭐ **The selection rule is not just a smaller ADE — it recovers a capability the flat swap loses.** `always_op` beats CV from **0.4 s**; the flat `str`/`tac` swap only from **0.6 s**; the rule keeps `op`'s early start **and** gains `str`'s reach: **0.4 s … 18.5 s**. That is the GraphCast/FuXi/Pangu argument, measured on our own bank, for free.

⚠️ In the **deployable** regime **no arm beats CV at any lead time** — including every rule. The only
deployable-regime arm with a non-empty window is the *non-deployable* per-window oracle, at
0.6 s … 1.0 s. **The readout bank does not create deployable capability; it improves an arm that is
still below the floor.**

## 7.3 🔴 The answer to "what rule" is **not** the ADE-optimal one — and this is the finding

The two objectives the program actually reports disagree. `T_blind` (imagination vs the frozen-last
control, contiguous separation from step 2 — `t_blind.json`'s rule) against `ade_0_2s`, over the
op→`str` switch step, deployable regime:

| switch at | `ade_0_2s` | `T_blind` |
|---:|---:|---:|
| 0.0–0.3 s (= `always_str`) | 0.8710 | **2.5 s** |
| **0.5 s** | **0.8597** | **2.5 s** |
| 0.8 s | 0.8597 | **2.5 s** |
| 1.0 s | **0.8534** ⬅ ADE-optimal | **0.8 s** ⬅ collapse |
| 1.3 s | 0.8534 | 0.8 s |
| ≥ 2.0 s (= `always_op`) | 0.9554 | 0.8 s |

`always_op` 0.8 s · `always_tac` 2.1 s · `always_str` 2.5 s (the committed 0.8 s → 2.5 s readout-swap
result, reproduced).

> ### ⛔ **The ADE-greedy rule buys 0.0063 m of `ade_0_2s` (0.7 %) and pays 3.1× of deployable `T_blind` (2.5 s → 0.8 s).** The rule the data supports is **`op` for lead times ≤ 0.5–0.8 s, `str` (or `tac`) beyond** — which is **Pareto-dominant over the flat `always_str` swap**: strictly better ADE (0.8597 vs 0.8710) at identical `T_blind` (2.5 s). **⇒ C8's recommendation: switch at step 5–8 (0.5–0.8 s). Do NOT fit the switch point to `ade_0_2s`.**

## 7.4 ⚠️ Selection semantics — the thing that is easy to implement wrongly

This selects **which head's trajectory is read at lead time j** — each specialist decodes the *same*
latent rollout, and you read the one for your lead time. That is exactly the GraphCast / FuXi /
Pangu-Weather pattern. It is **not** a per-step splice of Δposes inside one SE(2) accumulation (that
would need a re-decode, and no published system does it). **Consequence the harness owner must be told:
the concatenated path can be DISCONTINUOUS at the switch step.** For a planner consuming waypoints at
fixed lead times this is irrelevant; for a smooth-path consumer it is not, and a short blend window
would be needed. Implementation surface is unchanged from the source report's estimate:
`taniteval/rollout.py::collect` plus the two `canary_rollout`s.

---

# 8. LIMITATIONS AND COUNTER-EVIDENCE

## 8.1 What each experiment can and cannot support

| experiment | corpus | can support | ⛔ cannot support |
|---|---|---|---|
| **X4** | committed **training logs** of 6 arms (parity corpus, by construction — these are the runs themselves) | that the real/imagined split is a property of *training*, and that its rate is set by the `v0` channel | it is a **train-set** quantity. It says nothing directly about val generalisation; X1's fidelity stage is the val-side check and it agrees |
| **X1 / X1b** | dev-box **`physicalai-val-bb543bdf7836`**, 100 eps, **NON-PARITY** | within-run contrasts: fresh probe vs the trained head vs baselines, all on identical windows | any cross-arm or leaderboard comparison; any registry arm row |
| **C8** | committed per-window dumps on the **parity** val split `0c5f7dac3b11` (600 eps / 596 clusters) | the selection rule and its intervals, for **v1 only** | v4 or any other arm — those dumps do not exist |

## 8.2 ⚠️ Counter-evidence that must not be dropped: PlaNet Fig. 7

`PUBLISHED (cited)` — Hafner et al., PlaNet ([arXiv:1811.04551](https://ar5iv.labs.arxiv.org/html/1811.04551)):
latent overshooting *"can substantially improve the performance of the DRNN and other models … but
**slightly reduces performance of our RSSM**"*. **Multi-horizon / multi-source training is not a free
win.** It repairs a *misspecified or capacity-limited* predictor and can hurt an adequate one.

This bears directly on C1: C1 is a mixed-source term on the metric **head**, and PlaNet's result is
about a term on the **latent**. The two are not the same intervention — but the *shape* of the risk is
the same and the program should carry it: **any C1 arm must be pre-registered with a rollout-quality
guard** (`ade_0_2s` and `wm_canary_ade_2s` must not regress), not only with a real-pair-decode target.
C8 supplies the matching empirical caution from our own data (§7): the ADE-greedy selection rule
**degrades** the capability metric.

## 8.3 The aggregation caveat, carried

Restated from §3.5 because it applies to every ratio quoted anywhere in this document:
`*_mid_de_m` is an **endpoint** error, `*_fwd_ade_m` an **ADE** — for a growing error the endpoint is
larger by ~2×, so the 33.4× at 28–30 k is generously ≈**16.7×**. Every raw ratio in
`x4_log_sweep.json` ships with `ratio_agg_corrected_x0.5` beside it. **The trend and every cross-arm
comparison are aggregation-invariant.** X1 does **not** inherit this caveat: it compares
endpoint-to-endpoint throughout (probe DE vs trained-head DE at the same k on the same pairs).

---

# 9. 🔴 ESCALATIONS — raised in the headline, not written into a README

**E1′ (sharpens the source report's E1). The registry row is owed, and X4 SCOPES it.**
`Project Steering/MODEL_REGISTRY.md` still carries **no** real-vs-imagined row and **no** blind-horizon
row (grepped: the only `invdyn` mentions are training args and `v2_invdyn_gradscale`). The row should
be entered **for the v1 family only** — v1, `flagship4b-phase0-30k`, v2, v3enc, expA-nodrop — with the
per-level numbers in §3.3 and the aggregation caveat. ⛔ **It must NOT be entered as a program-wide
property: v4 does not carry the instrument (§3.1) and would inherit an untested claim.**
*Owner: registry owner. Blocked on nobody.*

**E2′ (answers the source report's E2). C8's open question is answered, and the answer contradicts the
obvious rule.** The decision is **not** "which readout" and **not** "the ADE-optimal switch". It is
**`op` for lead times ≤ 0.5–0.8 s, `str`/`tac` beyond** — Pareto-dominant over the flat `always_str`
swap the program was about to adopt. ⚠️ Two implementation notes that must travel with it: the
concatenated path can be **discontinuous at the switch step** (§7.4), and **"beats-CV horizon" means
SEPARATED-beats** (§7.1) — the point-mean reading overstates it by ~1.4 s.
*Owner: eval-harness owner — `taniteval/rollout.py::collect` + both `canary_rollout`s.*

**E3′ (re-scopes the source report's E3). C1 should still land before the next run — but SCOPED, and
with a guard.** Its premise is measured true only at k ≤ 2 (§4.2). At k = 4 the recoverable margin is
0.019 m and not separated; at k = 20 the trained head already **beats** the frozen-latent ceiling, so a
mixed-source term there risks a PlaNet-Fig.-7-style regression (§8.2). ⛔ **Any C1 arm must be
pre-registered with a rollout-quality guard (`ade_0_2s`, `wm_canary_ade_2s` must not regress), not only
a real-pair-decode target.**
*Owner: trainer owner — `stack/tanitad/models/metric_dynamics.py::grounding_losses`.*

**E4′ (the source report's E4, now with a second independent measurement).** *"The metric readout may be
an action-integrator, not a perception decoder"* was `HYPOTHESIS` with three supports. X1 adds a
fourth, of a different kind — **held-out, within-run, on the frozen encoder**: a single true-`v0`
scalar beats every latent probe by **24.8× at 0.1 s**, the latent's whole metric content is a
**R² ≈ 0.61** speed reading, and the second latent slot adds nothing (§4.3). ⛔ **X2 (the surgical
`v0`-scaling test, ~20 GPU-min) is still the clean proof and is still owed** — but the peek /
duty-cycle / re-anchoring streams should stop treating the readout as a perception decoder now.

**E5′ — NEW, and it is the cheapest open item in the program.** **v4 needs the grounding instrument.**
Two ways, both small: (a) **eval-side**, ~20 GPU-min — run `x1_cache_latents.py`'s fidelity stage
against a v4 checkpoint (v4.1 warm-started v1's `grounding`, so the heads exist in the checkpoint even
though the trainer never re-trains or logs them); or (b) **train-side**, ~5 lines — log
`g_{lvl}_{mid,fwd}` from the v4 loop so the next v4 run is measurable at all. ⛔ **Until one of these
happens, every statement about v4's imagination/perception behaviour is UNVERIFIED.**

---

# 10. DELIVERABLE MANIFEST

**Everything below is in the repo working tree and STAGED (`git add`). This agent committed nothing,
pushed nothing, and switched no branch.** Verified with `git ls-files --stage` — **and the scoped
`git status --short` again under-reported, showing 4 of 10 staged files while `ls-files --stage` showed
all 10.** The brief's warning is confirmed a second time; use `ls-files --stage`.

⚠️ **NOTE FOR THE ORCHESTRATOR — the documented whole-index sweep happened again, to this work.**
While this stream was running, a sibling agent's pathspec-free `git commit` swept the already-staged
deliverables into **`8a5a998`** (*"THE (λ,τ) CURVE WAS NOT MEASURED …"*) and a partial set into
`5a5a905`. **Nothing was lost** — the files are in `HEAD` and this is the *benign* direction of the
hazard — but nine of the ten artifacts below now live under a commit message about an unrelated
experiment, which is exactly the `CLAUDE.md` git-hygiene failure ("a quick commit of my thing silently
sweeps in a sibling's work"). Only the final `X1_LATENT_METRIC.md` revision remains staged-not-committed.
**If the program wants this work findable by message, it needs a follow-up commit or a note in
`DECISIONS.md`; searching the log for "X1" or "latent metric" will not find it.**

**Deliverable path:**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-x1-latent-metric-probe/`

| artifact | what it is | where it lives | only one place? |
|---|---|---|---|
| `X1_LATENT_METRIC.md` | ⭐⭐ this report | **repo** (staged) | no — content is derived from the JSON below |
| `scripts/x4_log_sweep.py` | X4: band sweep over every committed log + the 8-number reproduction gate + the attributing ablation at all 3 levels | **repo** (staged) | no |
| `scripts/x1_cache_latents.py` | X1 stage 1: frozen-encoder latent cache + the pre-registered fidelity and shuffle validations | **repo** (staged) | no |
| `scripts/x1_probe.py` | X1 stage 2 + X1b: ridge / MLP-h512 / MLP-4× probes, baselines, controls, rig split | **repo** (staged) | no |
| `scripts/c8_selection_rule.py` | C8: the rule family, 5-fold episode-grouped CV, the two-objective table, both reproduction gates | **repo** (staged) | no |
| `scripts/hf_pull_ckpt.py` | parallel-range gated-HF pull for the dev box (truststore, token read in place) | **repo** (staged) | no |
| `artifacts/x4_log_sweep.json` | every X4 number, per arm, per level, per band | **repo** (staged) | no |
| `artifacts/x1_cache_fidelity.json` | the three pre-registered validations + the trained heads' error on our windows | **repo** (staged) | no |
| `artifacts/x1_probe.json` | every X1/X1b number incl. train errors, sample-size curves, all bootstraps | **repo** (staged) | no |
| `artifacts/c8_selection_rule.json` | every C8 number incl. fitted rules, per-grid paired contrasts, the two-objective sweep | **repo** (staged) | no |

## 10.1 ⚠️ Working files deliberately NOT staged (too large for the repo), each with its rebuild command

| file | size | where | why it is safe |
|---|---:|---|---|
| `v1_speedjerk_ckpt.pt` | 3.309 GB | `devbox:C:/Users/Admin/tanitad-data/eval/` | **not our artifact** — it is HF `Sayood/tanitad-flagship-4b-speedjerk` (gated), plus the pod copies. Re-pull: `python scripts/hf_pull_ckpt.py --repo Sayood/tanitad-flagship-4b-speedjerk --file ckpt.pt --out <path>` |
| `x1_latents.pt` | 49 MB | `devbox:C:/Users/Admin/tanitad-data/eval/x1_latents/` | fully regenerable in **151 s** from the ckpt + the local val cache: `python scripts/x1_cache_latents.py --ckpt <ckpt> --cache-dir …/physicalai-val-bb543bdf7836 --episodes 100` |
| `x1_grounding_heads.pt` | 54 MB | ibid. | a slice of the ckpt; same command |
| `x1_fidelity_perwindow.pt` | 46 KB | ibid. | same command. *(Small enough to stage if the program wants the per-window fidelity values; say so and it will be.)* |

**Nothing that took real effort lives in only one place.** The only single-disk items are regenerable
in minutes by a committed script from a source that is itself in three places.

## 10.2 Reproduction — end to end on the dev box

```
py="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"; D=".../2026-07-27-x1-latent-metric-probe"

# X4 — 0 GPU, ~20 CPU-seconds, includes its own 8-number gate
$py "$D/scripts/x4_log_sweep.py"

# C8 — 0 GPU, 782 s measured (the bootstrap-heavy beats-CV windows dominate)
$py "$D/scripts/c8_selection_rule.py"

# X1 — one GPU pass (151 s) then CPU/GPU probe fitting (~70 s)
$py "$D/scripts/hf_pull_ckpt.py" --repo Sayood/tanitad-flagship-4b-speedjerk \
    --file ckpt.pt --out C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt
$py "$D/scripts/x1_cache_latents.py" --ckpt C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt \
    --cache-dir C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836 \
    --episodes 100 --out-dir C:/Users/Admin/tanitad-data/eval/x1_latents --batch 12
$py "$D/scripts/x1_probe.py" --latents .../x1_latents.pt --heads .../x1_grounding_heads.pt \
    --device cuda --stride-train 1 --stride-eval 4 --epochs 60
```

## 10.3 What this unblocks

| stream | what it can now do |
|---|---|
| **the next flagship run** | C1 can be written **scoped to k ≤ 2 with a rollout guard** (E3′), and **C3 is ranked #1** with its premise measured rather than assumed (§4.3) and the two-rig objection removed (§5) |
| **the eval harness** | C8's selection rule is specified, cross-validated and Pareto-checked (E2′); two quoting conventions are pinned in code (§7.1) |
| **the registry** | the F4 row can be entered with the correct scope and the per-level numbers (E1′) |
| **the peek / duty-cycle / re-anchoring streams** | they can stop building on "the readout is a perception decoder" (E4′) |
| **v4** | knows it is UNINSTRUMENTED, and has two small routes to fix that (E5′) |
