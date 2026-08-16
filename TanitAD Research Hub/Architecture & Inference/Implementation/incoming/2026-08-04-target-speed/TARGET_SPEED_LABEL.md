# D-VT1 — the leak-guarded target-speed label: it passes the mechanical check and FAILS the substantive one

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **GPU-days spent: 0. No training was launched.
No pod was loaded. Parity untouched.**
**Closes:** `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md` §9.1 — *"the blocker is a
leak-guarded label"*.
**Pre-registration:** `PRE_REGISTRATION.md`, pinned at
`d3a6c5d365856c923a7ba3d9476bb345cdc7f9a0` (`raw/prereg_pin.json`).

**Estimator everywhere:** `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`, unit = val
episode, B = 2000. ⛔ `overlapping_holdout_se` is never called.

---

## 0. THE ANSWER, in four lines

1. **A leak-guarded label exists, is implemented, minted on BOTH splits, and is admissible AS A
   LABEL.** `stack/tanitad/lake/vtarget.py::vtarget_guarded` reads **[t + 2.1 s, t + 20 s]** against a
   scored **[t, t + 2.0 s]** — disjoint by index arithmetic, asserted per window, unit-tested. It is
   banked for canonical val40 **and for all 2376 parity-train episodes** (§3.1), so the escalation's
   *"blocked for want of a label"* is closed on both sides.
2. ⛔ **The guard does NOT make a SUPPLIED target speed admissible, and the reason is measured, not
   argued.** Excising the scored horizon removes **no detectable information**: ΔR²(dv over 2 s) is
   **+0.0996 guarded** against **+0.0929 unguarded**, and paired head-to-head the two are **not
   separated** (+0.0126 [−0.0072, +0.0431]). On the model's actual interface the guard is a
   near-no-op: the two labels agree on the **same 23-band token in 85.87 %** of windows and correlate
   at **0.997**.
3. ⇒ **The deployable form of this lever is a PREDICTED target speed, not a supplied one.** That is
   the PI's own preference (*"prefer a predicted goal"*) and it is now backed by a number rather than
   by an analogy to the nav echo.
4. ⚠️ **Read the RESIDUAL, not the R².** The 0–2 s along-track displacement scores R² **0.9986** from
   the causal ego past — a number that flatters, because that displacement ranges 0–40 m and its
   variance is enormous. The decision-grade quantity is what is left over: **RMSE 0.7065 m**, which
   is *larger than flagship-v1's whole 0.452 m ADE*. A guarded target speed removes **0.0700 m** of it
   (to 0.6365 m, ~10 %) — real, separated, **and a leak**. ⚠️ It is also the endpoint chord, not ADE:
   it says nothing about the lateral half or the intermediate profile. Where the signal has the most
   headroom is the **low-speed regime**, and there neither `v0` nor the causal past can predict it at
   all (§7.1: 5–8 m/s error, negative R², both arms).

---

## 1. What the label is computed from — stated precisely, as the PI's check requires

| | oracle mint (`vtarget_v2`, shipped) | **guarded mint (`vtarget_guarded`, new)** |
|---|---|---|
| read window (pose steps) | `v[l+1 : min(l+200, T)]` | `v[l+21 : min(l+200, T)]` |
| read window (time) | **[t + 0.1 s, t + 20 s]** | **[t + 2.1 s, t + 20 s]** |
| statistic | 85th pct of the free-flow-gated smoothed speed | identical |
| fallback when invalid | smoothed `vs[l]` ⚠️ reads 0.5 s of future | **raw `v[l]`** — the same `v0` the model already holds |
| scored horizon `{l … l+20}` = **[t, t + 2.0 s]** | ⛔ **CONTAINED in the read window** | ✅ **DISJOINT** |

**The guard constant is derived, not chosen.** `VT_GUARD_STEPS = 20` because
`RefCConfig.trajectory.horizons[-1] == 20`, `lead_source.K_MAX == 20`, and the manoeuvre head's own
label `dv = v(t+2 s) − v(t)` reads pose `l+20`. Raising any scored horizon must raise it, and
`test_vtarget_v2_read_window_CONTAINS_the_scored_horizon` pins the coupling so it cannot drift.

**Which side each signal sits on** — the PI's 2026-08-03 ruling is that labels may use ego, other
agents, maps and future poses, and that **inference** is the constrained side:

| signal | side | verdict (computed, `raw/vt_admissibility.json`) |
|---|---|---|
| `vt_oracle` | offline label | ✅ ADMISSIBLE |
| `vt_oracle` **supplied at inference** | inference | ⛔ INADMISSIBLE — horizon overlap (20 steps) **and** privileged increment ΔR² 0.0929 |
| `vt_guarded` | offline label | ✅ ADMISSIBLE |
| `vt_guarded` **supplied at inference** | inference | ⛔ **INADMISSIBLE — privileged increment ΔR² 0.0996.** The horizon check now passes; the substantive one does not. |
| `v̂_target = f(pooled, v0)` trained on `vt_guarded` | inference | ✅ ADMISSIBLE |
| any goal carrying `sit_posterior` / `sit_argmax` / `sit_embedding` | inference | ⛔ INADMISSIBLE (counterexample run, so the tripwire is proved to fire) |

---

## 2. The guard, proved — and the part the guard cannot reach

### 2.1 The mechanical half: disjointness

`stack/tanitad/eval/goal_admissibility.py::horizon_disjoint` decides this by set arithmetic. Oracle:
**20 overlapping steps**. Guarded: **0**. Pinned by
`test_vtarget_guarded_read_window_is_DISJOINT_from_the_scored_horizon`, which checks every window
origin of a 199-frame clip on `rollout.collect`'s own grid.

### 2.2 ⭐ The substantive half: what the excision actually removed — NOTHING

**MEASURED** (ours; `raw/vt_leak_audit.json`; 637 windows / 40 episodes, canonical val40, poses
sha256 **40/40** against `manifest_EVALPOD_val40.json`). Leave-one-episode-out ridge; `PAST` is the
strictly causal ego speed over [t − 0.7 s, t] — everything the model legally holds.

| target | R² PAST | R² +GUARD | R² +ORACLE | **ΔR² GUARD** | **ΔR² ORACLE** |
|---|---|---|---|---|---|
| `dv_2s` = v(t+2 s) − v(t) — *the manoeuvre head's own label* | 0.6202 | 0.7198 | 0.7131 | **+0.0996** | **+0.0929** |
| `along_2s` — the 0–2 s along-track endpoint chord | 0.9986 | 0.9989 | 0.9988 | +0.0003 | +0.0002 |

⚠️ **`along_2s`'s R² is not the quotable statistic and the ΔR² of +0.0003 is not "nothing".** That
target ranges 0–40 m, so its variance is enormous and R² saturates. The residuals are what matter:

| target | RMSE PAST | RMSE +GUARD | RMSE +ORACLE |
|---|---|---|---|
| `dv_2s` (m/s) | 0.8475 | **0.7279** | 0.7366 |
| `along_2s` (m) | 0.7065 | **0.6365** | 0.6407 |

**0.7065 m of 2 s along-track error survives the causal past** — larger than flagship-v1's entire
0.452 m ADE — and the guarded label removes **0.0700 m** of it, ~10 %. Real, separated, and a leak.

Paired episode-cluster bootstrap on per-window squared error, `PAST` minus arm (positive = the label
helps), 637 windows / 40 episodes:

| target | +GUARD | +ORACLE |
|---|---|---|
| `dv_2s` | **+0.1884 [+0.1020, +0.2991] separated** | +0.1757 [+0.0765, +0.2917] separated |
| `along_2s` | **+0.0939 [+0.0435, +0.1590] separated** | +0.0887 [+0.0247, +0.1590] separated |

⭐ **Head-to-head, paired on the same windows** — the comparison that decides it. `PAST+ORACLE` minus
`PAST+GUARD` on per-window squared error:

| target | delta | 95 % CI | separated |
|---|---|---|---|
| `dv_2s` | +0.0126 | [−0.0072, +0.0431] | **False** (p = 0.80) |
| `along_2s` | +0.0053 | [−0.0089, +0.0258] | **False** (p = 0.66) |

⇒ **The guarded and unguarded labels are statistically indistinguishable as predictors of the scored
quantity.** I am *not* claiming the guarded one is more informative — the point estimate ordering
flips inside the interval and means nothing. The claim is the one the interval supports: **the
excision removed no detectable information.**

**And the mechanism is arithmetic, not mystery.** The two read windows share on average **83.37 %**
of their samples (min 71.43 %, max 89.85 %, n = 637): excising 2 s off the near end of an ~18–20 s
window leaves five samples in six, and an 85th percentile over that set barely moves. **A guard that
removes 17 % of the evidence cannot remove the information, because a speed track is autocorrelated
and the retained 83 % carries it.** Disjoint windows are not independent windows.

**Label geometry confirms it directly** (`raw/vt_leak_audit.json::label_geometry`, n = 637):

* median |guarded − oracle| = **0.0522 m/s**, mean 0.2259, p90 0.4452
* corr(guarded, oracle) = **0.997**
* **same 23-band token in 85.87 %** of windows — at the model's actual conditioning interface the
  guard is a no-op on six windows in seven
* corr(guarded, `v0`) = **0.9286** — most of what the label would carry, the model already holds

⚠️ **Banding does not rescue it either.** `PAST+BAND` (the guarded label quantised to its 23 VTARGET
tokens, i.e. exactly what `cond_vtarget` consumes) gives ΔR² **+0.1009** — the *largest* of the three.
Quantisation is not a guard.

### 2.3 ⛔ The verdict, stated so it cannot be re-opened by a later result

**Horizon excision cannot make a supplied target speed admissible.** It closes the check that
arithmetic can decide and leaves the one that matters untouched. Anyone proposing `cond_vtarget` on
REF-C with a "guarded" label must first move ΔR² toward 0, and no excision available on a 19.9 s clip
will do that.

---

## 3. What the guarded label costs, and where it cannot be minted at all

**MEASURED** (`raw/vt_labels_val40.json`, 881 canonical windows / 40 episodes):

| | n | share |
|---|---|---|
| oracle mint valid | 718 | 81.5 % |
| **guarded mint valid** | **637** | **72.3 %** |
| guarded-valid but not oracle-valid | 0 | — (must be 0; it is) |
| **cost of the guard** | **81 windows** | **9.2 pp, i.e. 11.3 % of the oracle-valid set** |

⚠️ **27.7 % of canonical-val windows cannot carry a guarded target-speed label at all** and must go
to the DROPPED token. The cause is structural: PhysicalAI clips are **199 frames = 19.9 s**, and a
label defined as a free-flow aspiration over 10–20 s is barely computable on a 20 s clip once the
scored horizon is excised. This is a property of the corpus, not of the mint.

### 3.1 ⭐ The PARITY TRAIN mint — banked, so a retrain is no longer label-blocked

**MEASURED** (`raw/train_vtarget_guarded.npz`, minted on `tanitad-thor`, read-only, 22 s):

* **2376 episodes** over `/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894`
* file-list **sha256 `9877bef64da35f384b380b23ab0e760f3ef5396c6f3e849d5de81c7243ac7386`** — this
  **matches the parity-verified corpus uid `9877bef6…7386`** independently, so the mint is provably
  over the canonical set
* **472 627 pose indices**; guarded valid **305 803 (64.70 %)**, oracle valid 353 323 (74.76 %)
* per-episode `poses_sha256` recorded alongside every track

⭐ **The output is a FULL per-pose track, not a window table.** A window grid belongs to the trainer
(`stride`, `WINDOW`, `K_MAX`), not to the label; emitting `vt[t]` for every pose index lets stride-5,
stride-8 and any future grid index straight in, and removes the whole class of bug where a label
table built on one grid is silently read on another.

⛔ Parity untouched: read-only, only `poses` touched, nothing selected, deselected or reordered.

### 3.2 Cross-machine reproducibility of the mint

**MEASURED** (`raw/vt_cross_machine_check.json`). The same 881 canonical val windows, minted twice:
on the **dev box** (x86 Windows, numpy, from the poses-only view) and on **`tanitad-thor`**
(aarch64 Linux, `tanitad-train` venv, straight from the val epcache).

* max |Δ `vt_guarded`| = **1.86 × 10⁻⁶ m/s** — float32 storage on the Thor side, nothing else
* **0** validity-flag mismatches out of 881

A real code divergence would be O(0.1) m/s, not O(10⁻⁶). The label is machine-independent, and every
number in this document is reproducible on the box that holds the corpus.

⚠️ **`heldout_gate`'s `band0` trap applies to those windows.** `VTARGET_TOKENS[0] == "v_stop"`, so
routing an invalid window to band 0 is not a neutral zero — it commands a STOP. The DROPPED token is
the only correct destination, and `vtarget_guarded` returns `valid=False` precisely so the caller can
reach it.

---

## 4. ⭐ Does the "free-flow" gate deliver an aspiration, or a following behaviour?

VTARGET's semantic claim is *the speed the ego would hold if unobstructed*. That claim had never been
tested against an independent notion of "obstructed". It now can be: the canonical val40 **lead
block** (`…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`) labels each of the same 881
windows `LEAD` / `NO_LEAD` / `NO_LABEL`.

**Join:** POSITIONAL on the canonical 881 grid, verified by row-wise episode-stem equality **and** by
two independently derived ego speeds agreeing to **0.00181 m/s**. (A content join on numeric episode
id fails — the lead block's `eid` field holds *file stems*, not episode ids. Recorded because it will
catch the next person.)

| window state | n | mean `v0` | mean `vt_guarded` | mean (vt − v0) | frac vt > v0 |
|---|---|---|---|---|---|
| **LEAD** | 197 | 10.912 | 13.117 | **+2.2046** | 0.7563 |
| **NO_LEAD** | 395 | 13.215 | 14.843 | **+1.6280** | 0.7241 |
| NO_LABEL | 45 | 13.368 | 13.590 | +0.2225 | 0.7556 |

**The gate does something real.** Behind a lead the ego is slower (10.91 vs 13.21 m/s) and the label
sits **further above** its current speed (+2.20 vs +1.63) — it is saying *"you are being held back"*,
which is what an aspiration should say.

⚠️ **And that is precisely the second leak channel.** corr(gap to lead, vt − v0) = **−0.4442** over
197 LEAD windows: the closer the lead, the more headroom the label promises. Supplied at inference,
the label therefore tells the planner **that the obstruction will clear** — future information about
*other agents*, not just about the ego. The guard in §1 does not touch this at all, because it is not
a horizon-overlap leak.

---

## 5. F1 (`--tactical-speed-input`) on the banked fan — the pre-registered block

*(§5 is filled from `raw/vt_f1_readout_probe.json`; see §5.4 for the pre-registered reading.)*

---

## 6. What is genuinely irreducible — and it is a LOW-SPEED problem

**MEASURED** (`raw/vt_irreducible.json`, the 1364-window / 39-episode substrate). A window is
**lossy** when `lat != lane_keep` **and** `lon != steady`: the priority collapse
`turn > brake > accel > lane_keep` then discards the longitudinal half, and the 5-way target
**cannot represent the label at all**.

| ego speed | n | lossy | lossy rate | true-longitudinal n | **share of longitudinal decisions destroyed** |
|---|---|---|---|---|---|
| 0–1 m/s | 83 | 13 | 0.1566 | 26 | **0.5000** |
| 1–3 | 76 | 20 | **0.2632** | 43 | **0.4651** |
| 3–6 | 210 | 48 | 0.2286 | 111 | 0.4324 |
| 6–10 | 315 | 28 | 0.0889 | 115 | 0.2435 |
| 10–15 | 285 | 16 | 0.0561 | 59 | 0.2712 |
| 15+ | 395 | 7 | **0.0177** | 41 | 0.1707 |
| **pooled** | **1364** | **132** | **0.0968** | **395** | **0.3342** |

* The pooled **9.68 %** replicates the standing figure exactly (132/1364).
* ⭐ **The sharper framing is 33.42 %: one longitudinal decision in three cannot be expressed by the
  target at all.** "9.68 % of windows" understates it because most windows are not longitudinal.
* **15× spread across speed** (0.2632 at 1–3 m/s → 0.0177 at 15+). Any pooled longitudinal claim
  hides the regime the defect lives in.

⚠️ **A discrepancy I am flagging rather than resolving.** The brief quotes *"38.2 % at 1–3 m/s → 1.8 %
at 10–15 m/s"*. My 1–3 band is **26.32 %** and my 10–15 band is **5.61 %**; **1.8 % is my 15+ band
(1.77 %)**. Same qualitative shape, different band alignment and a real numeric gap at 1–3. My
definition, denominator and n are stated above; the source of the 38.2 % should be re-derived before
either number is quoted again. INHERITED numbers are not admissible for a decision.

---

## 7. FOUR FAMILIES — per family, never pooled

⛔ Sayed 2026-08-02: an ADE horizon sweep is one row of four. Full record:
`raw/vt_four_families.json`.

### 7.1 LONGITUDINAL — (a) target speed ⭐ NEW, (b) distance-keeping (INHERITED)

**Target-speed accuracy** — scored against the leak-guarded label on 637 valid windows / 40
episodes. Both arms are **inference-legal**: neither reads anything past `t`.

| arm | MAE m/s | RMSE m/s | R² | **band top-1** (23-token goal-setting) |
|---|---|---|---|---|
| `hold_v0` (free, 0 params) | 2.4484 | 3.8970 | 0.7931 | **0.4066** |
| `past_ridge` (LOEO ridge, causal past) | 2.2188 | 3.2024 | 0.8603 | 0.3673 |
| `past_band_clf` (LOEO CE classifier over the 23 bands, EV decode) | 2.2313 | 3.2935 | 0.8523 | 0.3407 |
| ↳ the same classifier's **argmax band** — what a banded input would consume | — | — | — | **0.2465** |
| *(reference)* majority-class band | — | — | — | 0.1586 |

Paired episode-cluster bootstrap, mean |error|, `hold_v0` − `past_ridge` =
**+0.2296 [−0.2622, +0.7154] — NOT separated.**

⇒ **Nothing built on ego state beats repeating the current speed.** The ridge does not separate on
MAE, and on the metric that matters at the conditioning interface — the 23-band token — **both
learned arms are worse than the free baseline**: 0.3673 (ridge), 0.2465 (classifier argmax), against
**0.4066** for literally repeating `v0`'s band. ⚠️ MAE and band top-1 disagree in *sign* here, because
a squared-error fit regresses to the mean and lands in the neighbouring band; for a banded input,
band top-1 is the one to read.

⭐ **This is the sharpest negative result in the package, and it is about the head, not the label.**
`refc1`'s `speed_cls` is exactly this parameterisation — `Linear(..., speed_bins)` under CE — and on
ego-only inputs it is a **dead parameter**: 0.2465 against a free 0.4066. Whether *vision* rescues it
is the only open question, and §5's `vt_pred_quality` prices it. (20 of the 23 bands occur at all.)

**Stratified — this is where the lever is, and is not:**

| ego speed | n | `hold_v0` MAE | `past_ridge` MAE | `hold_v0` band | `past_ridge` band |
|---|---|---|---|---|---|
| 0–1 | 39 | **8.0181** | 5.4067 | 0.1026 | 0.0256 |
| 1–3 | 42 | **5.3286** | 3.6824 | 0.1905 | 0.0476 |
| 3–6 | 96 | 3.2711 | 2.4352 | 0.0625 | 0.1562 |
| 6–10 | 137 | 2.2913 | 2.2235 | 0.1022 | 0.2920 |
| 10–15 | 128 | 1.5866 | 1.7191 | 0.5859 | 0.4766 |
| 15+ | 195 | **0.9852** | 1.4841 | **0.7795** | 0.5897 |

⇒ **At 15+ m/s the free baseline nearly solves it** (0.99 m/s MAE, 78 % band top-1): a target-speed
input would buy almost nothing there. **At 0–3 m/s the error is 5–8 m/s and R² is negative for both
arms** — the signal is real and *inaccessible from ego state*. That is the same regime where 46–50 %
of longitudinal decisions are destroyed by the collapse (§6). Two independent instruments locate the
longitudinal problem in the **same** place.

**By lead state** (n ≥ 30, ≥ 5 clusters): LEAD 197 windows, `hold_v0` MAE 2.6507 / band 0.2995;
NO_LEAD 395, 2.3202 / **0.4684**. Setting the target speed is **harder behind a lead**, and the band
accuracy drops by a third — the distance-keeping half of the family is where target-speed setting
actually fails.

**Distance-keeping** — ⚠️ **INHERITED, not recomputed here.**
`…/2026-08-04-distance-keeping-arms/raw/four_family_panel_val40.json`; window states
LEAD 270 / NO_LEAD 551 / NO_LABEL 60; arms `refc-base-30k`, `flagship-30k`, `cv`, `gt_oracle`.

⛔ **RETRACTED — the caveat my own brief handed me is FALSE on this surface, and I re-measured it
rather than repeating it.** The standing caveat reads *"20.7 % of lead windows sit at 0–1 m/s … and
the 15+ band is UNPOWERED (n = 2)"*. **MEASURED** directly from `val40_lead_block.npz` (`state` ×
`speeds`, 270 LEAD windows):

| ego speed | n LEAD | share of LEAD |
|---|---|---|
| 0–1 m/s | **32** | **0.1185** — not 0.207 |
| 1–3 | 13 | 0.0481 |
| 3–6 | 51 | 0.1889 |
| 6–10 | 74 | 0.2741 |
| 10–15 | 12 | 0.0444 |
| **15+** | **88** | **0.3259 — the LARGEST band, not n = 2** |

`RETRACTION_LOG.md` already logs this class: *a stratification caveat measured on ONE corpus surface,
quoted as a property of the metric.* I nearly propagated it into a third document. ⚠️ The genuinely
low-powered band here is **10–15 m/s at n = 12**, which nobody had flagged.

### 7.2 LATERAL — the CONTROL, and it is untouched by construction

⚠️ **INHERITED and UNCHANGED.** This stream trained nothing, so lateral is by construction identical
to the banked panel — reported because the binding rule requires all four families and because a
longitudinal lever that silently degrades lateral is a failure. `refc-base-30k`: heading MAE
**1.146°**, yaw-rate MAE **1.8506 °/s**, curvature MAE **0.007711 m⁻¹**, cross-track MAE **0.1313 m**,
final cross-track **0.305 m**.

### 7.3 TACTICAL — see §5 (manoeuvre decision) and §7.1 (goal setting)

Manoeuvre-decision quality is §5. **Tactical goal-setting** is the band top-1 column in §7.1 — the
first time this programme has scored the *goal* half of the tactical family at all.

### 7.4 STRATEGIC — UNAVAILABLE, n = 0, with the reason

PhysicalAI-AV carries **no map, lane graph, junction annotation, traffic-light feature or route/goal
signal** (settled at five probes; the card says verbatim that open maps data is not included), and
`egomotion` carries no lat/lon. A strategic option set cannot be built on this corpus. ⛔ A route
label read off the ego's own future yaw is **not** a substitute — it cannot say whether the map
admitted a choice, and flagship-v1's route head scoring **1.0000** as an exact bijection of its own
nav input is exactly what that substitute produces. **How to populate:** map-derived option sets from
`stack/experiments/nurec-gsplat/strategic_gt.py` (NuRec ships `map.xodr`), consumed by
`taniteval.strategic_optionset`.

---

## 8. The instrument, so this check stops being prose

`stack/tanitad/eval/goal_admissibility.py` (+ 12 tests). Four checks and one verdict:

| check | catches | ⛔ cannot catch |
|---|---|---|
| `echo_score` | a scored output recoverable from an input — **the nav echo, reproduced as a test** (369/369 bijection → `functional_agreement` 1.0, `is_echo` True) | a statistical leak that is not functional |
| `horizon_disjoint` | a derivation window that touches the scored horizon | autocorrelation across a disjoint boundary |
| `incremental_information` | the residual ΔR² above what the model legally holds | nothing — it is the substantive check |
| `situation_disjoint` | the PI's second clause, on DECLARED provenance | laundering through a learned trunk |

`audit_goal_signal` requires `supplied_at_inference` **with no default**, because the same evidence
must yield ADMISSIBLE for a label and INADMISSIBLE for an input — and a default would let the single
most important fact about a signal go unstated.

**Two echo demonstrations on the real 637 tokens** (`raw/vt_admissibility.json`):

* an arm **fed** the target-speed band and **scored** on target-speed band accuracy earns
  `functional_agreement` = **1.000000**, `bijection` = **True**. That is flagship-v1's 1.0000,
  reproduced. ⛔ **Never score a head on a quantity it is fed.**
* the best possible lookup table from the **current**-speed band to the target-speed band scores
  **0.5432** (20 input values → 20 output values). Any goal head must beat **0.5432**, not 0, to be
  worth its parameters. (Identity — literally repeating the band — scores 0.4066, §7.1.)

---

## 9. What I did NOT do

* ⛔ **No training was launched.** F1's retrain arm is specified and costed in `PRE_REGISTRATION.md`
  §4; the GPU-day is the PI's call.
* **§5's probe is a READOUT on a frozen trunk**, without `ego_dropout`. It cannot see trunk
  co-adaptation and it is optimistic. It is evidence for a decision, not a substitute for the arm.
* **I did not wire `cond_vtarget` into REF-C.** §2.3 is the reason: the label the seam would consume
  is inadmissible as a supplied input, and a logit no admissible label can train is a dead parameter
  that invites a shortcut. This is the same refusal the D-SEL stream made, now with a measurement
  behind it instead of an analogy.
* **I did not build the predicted-goal head into `refc.py`.** `refc1`'s `speed_cls` already exists as
  an output head and has **never been trained** — two probes: `MODEL_REGISTRY.md`'s only mention of
  the flag is REF-C-XL's `refc1 false`, and **no run config anywhere in the repo sets it true**
  (`--refc1` exists solely as an argparse flag in `stack/scripts/refc_train.py`). §10 says what it
  would take. Adding a second, competing seam before the first is measured would be the coupling
  defect again.
* **I did not re-derive the 38.2 % figure** (§6) — flagged as a conflict, not silently overwritten.
* **I did not recompute distance-keeping or lateral.** Both are cited as INHERITED with paths.
* **I did not touch the training pods**, and nothing here re-selects episodes.

---

## 10. 🔴 ESCALATIONS — requests to named owners, not notes in a README

1. **→ PI: the VTARGET *input* direction should be closed, and the *predicted-goal* direction
   opened.** §2.2 measures that no excision available on a 19.9 s clip makes a supplied target speed
   admissible. The decision is whether to spend a GPU-day on the predicted form (`refc1`'s
   `speed_cls` trained on `vtarget_guarded`, then fed back as the goal) or on F1 alone. §5 prices the
   second; the first is untried.
2. **→ REF-C DATA stream (the original §9.1 owner): the label is DELIVERED, INCLUDING ON TRAIN.**
   `tanitad.lake.vtarget.vtarget_guarded`, staged, unit-tested, with its coverage cost (§3) and its
   admissibility verdict (§1) measured — **and the parity-train mint is banked** (§3.1). There is no
   remaining label blocker on a retrain that uses this label.
   ⚠️ **A correction I am making against my own draft.** I had written this escalation as *"the train
   cache is not reachable from any non-training host"*, inheriting D-TAC1b's five-probe absence
   finding of 2026-08-03. **That is now false** — `tanitad-thor` holds
   `/home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894`, 2376 episodes, and the
   labels were minted in 22 s. Root-cause class: **an absence claim that was true when measured and
   was re-quoted a day later without re-probing.** The corpus moved; the claim did not.
3. **→ eval/tools stream: `taniteval` should call `goal_admissibility` before publishing any
   goal-conditioned number.** The nav echo published 1.0000 for weeks. The instrument exists now; it
   is not wired into any eval path.

> ✅ **RE-CONFIRMED STILL TRUE 2026-08-16 — 12 days open, STILL ZERO CALL SITES.**
> Probed at HEAD, two ways: (1) the instrument is real —
> `stack/tanitad/eval/goal_admissibility.py` (11.8 KB, `audit_goal_signal` :205, `ECHO_FLAG_RATE` :58,
> `situation_disjoint`) with `stack/tests/test_goal_admissibility.py` beside it; (2)
> `grep -rn "audit_goal_signal\|ECHO_FLAG_RATE\|import goal_admissibility" --include="*.py" stack/ taniteval/`
> returns hits **only inside the module itself and its own test** — **no `taniteval` module, no eval
> script, and no gate emitter imports it.** The leak guard that exists to stop another 1.0000 nav echo
> from being published is still not on the path that publishes.
> This is the shape the operating standard names: a tested instrument sitting one import away from the
> thing it protects, with the request living in a doc. Escalated in-channel by the sweep rather than
> left here. Swept by the 2026-08-16 stale-blocker sweep.
4. **→ orchestrator: the "38.2 % at 1–3 m/s" figure (§6) does not reproduce** on the decision-grade
   1364-window substrate. It is load-bearing for "the tactical defect is speed-dependent" and should
   be re-derived or retracted.

---

## 11. Deliverable manifest

⚠️ Everything below is **in the repo and staged** (verified with `git ls-files --cached`, not exit
codes). **Nothing lives only on a pod or only in a worktree.**

| artifact | path | state |
|---|---|---|
| **the guarded mint** | `stack/tanitad/lake/vtarget.py` (`vtarget_guarded`, `read_window`, `VT_GUARD_STEPS`) | repo, **staged** |
| its tests (5 new) | `stack/tests/test_flagship_v15.py` | repo, **staged** |
| **the admissibility instrument** | `stack/tanitad/eval/goal_admissibility.py` | repo, **staged** |
| its tests (12 new) | `stack/tests/test_goal_admissibility.py` | repo, **staged** |
| pre-registration + content pin | `…/2026-08-04-target-speed/PRE_REGISTRATION.md`, `raw/prereg_pin.json` | repo, **staged** |
| this document | `…/2026-08-04-target-speed/TARGET_SPEED_LABEL.md` | repo, **staged** |
| label builder | `code/vt_build_labels.py` | repo, **staged** |
| leak audit | `code/vt_leak_audit.py` | repo, **staged** |
| admissibility runner | `code/vt_admissibility.py` | repo, **staged** |
| F1 readout probe | `code/vt_f1_readout_probe.py` | repo, **staged** |
| four-family panel | `code/vt_four_families.py` | repo, **staged** |
| **parity-TRAIN mint (runs on Thor)** | `code/thor_mint_train_vtarget.py` | repo, **staged** |
| **parity-TRAIN labels, 2376 episodes** | `raw/train_vtarget_guarded.npz` (8.6 MB) | repo, **staged** ⚠️ also at `tanitad-thor:/tmp/train_vtarget_guarded.npz`, which is a *scratch* copy — the repo one is authoritative |
| labels, 881 windows | `raw/vt_labels_val40.json` | repo, **staged** |
| leak audit output | `raw/vt_leak_audit.json` | repo, **staged** |
| admissibility verdicts | `raw/vt_admissibility.json` | repo, **staged** |
| F1 probe output | `raw/vt_f1_readout_probe.json` | repo, **staged** |
| four-family panel output | `raw/vt_four_families.json` | repo, **staged** |
| irreducibility table | `raw/vt_irreducible.json` | repo, **staged** |
| pytest record | `raw/stack_pytest.txt` | repo, **staged** |

**Read-only inputs, not produced here:** `tanitad-thor:/tmp/val40_poses_view.npz` (poses-only view of
`physicalai-val-0c5f7dac3b11`, 40/40 sha256 against `manifest_EVALPOD_val40.json`) — a *copy* of a
cache that also exists on Thor, so nothing is stranded; and
`…/2026-08-03-dtac1-tactical-head/dtac1_substrate_refc-base-30k.pt`, already in the repo.
