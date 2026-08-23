# Evidence dossier A — selection / longitudinal / horizon / MPC measured record

**Provenance:** compiled 2026-08-18 by a read-only research subagent for the REF-C improvement
review (`REFC_IMPROVEMENT_REVIEW.md`, same directory). Worktree
`C:\Users\Admin\wt-tanitad-local`, branch `agent/arch-inf-20260803`, HEAD `d2ede52b`. All line
numbers verified in that tree at compile time. Evidence class per claim: **[M]** MEASURED at the
cited artifact, **[P]** PUBLISHED (registry/retraction text quoting its own artifact), **[I]**
INHERITED. This file is the verbatim agent report, banked for provenance; the review cites it.

---

## 1. C101 — CEM planner vs CV, closed-loop T1

**Registry block: `Project Steering/MODEL_REGISTRY.md:2622-2660`** (heading at `:2622`). **[P→M]**

Verbatim, `:2637-2639`:
> | **planner − CV** | **+0.2585 m [+0.0869, +0.4309]**, CI-separated, **p(δ>0) = 0.9975** |
> | ⇒ | **the CEM planner is 35.8 % WORSE than constant velocity, closed-loop** |
> | **operative under TRUE actions − CV** | **−0.3151 m** ⇒ the WM rolls out *better* than CV when handed true actions |

`:2641-2642`: *"⇒ ⭐⭐ **THE LOSS IS IN THE ACTION SEARCH, NOT IN THE WORLD MODEL.** The CEM cannot
find actions that exploit a predictor that demonstrably works."*

`:2644-2648` (per family, never pooled):
> **LONGITUDINAL 1.9062 vs CV 1.6705 m**, speed error **0.9431 vs 0.7607 m/s**, bias **+0.2737 vs
> −0.0995 m/s** … **TACTICAL** and **STRATEGIC** are genuine **N/A with reasons**: the CEM emits no
> manoeuvre class, and the cost carries **no route/goal term**. Distance-keeping/TTC uncomputable —
> no lead-agent track.

**Eval JSON — `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-planner-beats-cv-redrive/raw/planner_beats_cv_banked_analysis.json`** **[M, read directly]**, key `4_closedloop_planner_vs_cv_NEW`:
- `_tier` = `"T1 (action-closed loop: the model is conditioned on its OWN actions)"`
- `paired_planner_minus_cv`: delta **0.2585**, lo **0.0869**, hi **0.4309**, ci95 **0.172**, `p_delta_gt0` **0.9975**, separated **true**, n_windows **221**, n_episodes **20**, n_boot **2000**, estimator `paired_episode_cluster_bootstrap`
- `paired_operative_minus_cv`: delta **−0.3151** [−0.6277, −0.0602], ci95 0.2838, `p_delta_gt0` **0.008**, separated **true**
- arms (episode_cluster_bootstrap, 221/20): `closed_bike` **0.9799** [0.7456, 1.2312]; `open_grnd` **0.4063** [0.3293, 0.4907]; `cv` **0.7214** [0.4680, 1.0360]
- `closedloop_planner_beats_cv` = **false**
- `5_four_metric_families_closedloop` — planner LON per-horizon RMSE 0.1441/0.5472/1.1520/**1.9062**; CV 0.1136/0.4238/0.9427/**1.6705**; planner LAT crosstrack@2s **1.9637**, CV **1.7115**; TACTICAL/STRATEGIC `_computable: false` with reasons citing `planner_p2.py:280` and `planner_p2.py:44-51`

**Scope correction that supersedes the older block** (`:2624-2630`): `planner_beats_cv` is computed
in `analyze_openloop` (**`planner_p2.py:621`, fn at `:555`**) — **OPEN LOOP, 881 win / 40 ep /
stride 8**; the banked `p2win_flagship-30k.pt` is **closed-loop 221/20/stride 16**. The open-loop
verdict **remains UNDECIDED** (`:2650-2654`); flip needs −6.589 % against local envelope
−6.909 %…+5.877 %; blocked on a **4.70 GB val cache** (PI decision).

**Retraction entry: `Project Steering/RETRACTION_LOG.md:5252-5321`.** `:5286-5291` repeats the T1
result and the localisation verbatim. `:5302-5304` — a data defect caught in passing: curvature on
**11 stopped-ego windows**, GT `|κ|` mean **34.83 1/m** vs median **0.00081** (max **23,004**),
**masked not clamped**. Corrections it carries: `:5308-5310` — the artifact holds **14 boolean
instances across 6 distinct names**, not five (C91 collapsed the 9 `beats_head` entries).
Reproduction gate (JSON `2_reproduction_gate`): `cv` and `open_grnd` **bit-exact (rel_drift 0.0)**,
`closed_bike` drift **0.0193 %**. CEM now seeded (`cem_seed`, default 0, `fa4b3d1`, 9 pinning
tests) — `:2655-2657`.

---

## 2. v5f — "the SELECTOR is the defect"

**`MODEL_REGISTRY.md:1096`** — section heading, verbatim: *"#### ⭐ What the same table shows
instead — the fan is good and the SELECTOR is the defect"*. **[P]**

`:1100-1101`: *"`sel_gap` = `plan_ade − oracle_ade` **does not close**: 0.4510 / 0.4878 / 0.5681 /
0.3980 / 0.3432 / 0.4715 — no trend across 2,650 steps."*
`:1098-1099`: `oracle_ade` improves monotonically 0.9450 → 0.5902 → 0.5663 → **0.5254**.
`:1102-1103`: `rank_acc` **0.000–0.375**; `frac_sel_2x_worse_than_oracle` **0.25–0.50**.
`:1104-1105`: at step 3,650 `plan_ade` **1.0251** vs `oracle_ade` **0.5254** — *"~2× better if it
merely chose correctly"*. Block-median table at `:1079-1086` (500-step block medians, n ≥ 4/block).

⚠️ **Scope stamp that must travel with it — `:1112`:** *"**These are TRAINER-log numbers, not eval
output.**"*

**Eval-grade sel_gap, same arm** — `:1039-1047`, flagship-v5f-w120-30k **[TIER T0]**,
`eval_flagship_v4.py`, 600-ep w120 val corpus, **881 windows**: ade@2s **0.4011 m**, oracle_ade@2s
**0.1975 m**, **sel_gap 0.2036 m** — *"the selector still leaves ~half"*; miss@2m 0.1487.

**Corrections/supersessions:**
- `:1307` — **W1 pre-registered gate REFUTED: −16.7 %.** Kinematic re-rank of top-8 *worsens*: sel ADE 0.4011 → **0.4351**; gap 0.2036 → **0.2376**.
- `:1425-1426` — v5.8f: *"The whole deficit is SELECTION (sel_gap 0.374 vs v5f's 0.204)"*, over a **feasible** fan, 0.37 m recoverable headroom.
- `:1478` — W7 (WM-roll re-rank, topk=256, `winner_in_shortlist_frac` = 1.0): oracle **0.1273**, selected **3.3348**, **sel_gap 3.207**.
- `:2350` / `:2430` — REF-C scale A/B sel_gap: small 0.3048 · base **0.2813** · base@64 0.1895 · XL **0.3075** · XL@64 0.0346.
- `:1384` — tactical fan: *"the 8-candidate goal fan BEATS CV at 4 s and 6 s; the selector throws the advantage away"* (`sel_gap_tac 8.95`, mixed-unit); artifact `e44_gate.json`.

Implementation: `taniteval/taniteval/selgap.py`; tests `taniteval/tests/test_selgap.py`. Research
Hub sources: `…/2026-08-03-esel-verdict/raw/esel_probe_refc-{xl,base,small}-30k.json`,
`…/2026-08-03-s1-climbout/raw/*.json`.

---

## 3. SEL-1 — refusal of the L1 roll-consistency re-ranker

### 3a. The winner's-curse law (E-WC), `Project Steering/V6F_PLANNER_DESIGN.md:46-71` **[M]**
Source: `stack/scripts/sel_winners_curse_law.py` on the banked in-repo REF-C-XL fan, **881 windows /
40 episodes / 256 candidates**.

`:50-57` verbatim: roll-consistency (`cons_score`, the W7 quantity) *"has a normalised argmax
error-rank that **RISES** with the candidate count (0.241 at N=4 → **0.286** at N=256, where 0.5 is
a coin flip) while its lower-tail hit rate **COLLAPSES** (0.57 → **0.28**) and its ADE stays pinned
at ~6.2–6.45 m while the oracle falls 4.606 → **0.164**; whereas REF-C's **supervised** selector on
the *same fan* moves the opposite way — rank **0.099 → 0.014**, lower-tail **0.77 → 0.99**, ADE
**5.365 → 0.471**"*.
Replication (REF-C-base, 104.2 M, 881 win, 128 cand), `:56-57` and `:334-336`: roll-consistency
0.243 → 0.283 / p10 0.41 → 0.25; supervised 0.082 → **0.021** / 0.62 → **0.98**.
Full N-law table: **`:316-324`**. Fan references `:311-312`: oracle **0.1639** · fan mean
**13.9564** · shipped supervised selector **0.4714 [0.3896, 0.5556]** · CV **0.8377**.
Top-m refuted as remedy, `:59-61`: top-8 medoid **+0.1294 m [+0.0645, +0.2029] WORSE** than argmax,
paired-separated.
Registry cross-entry `MODEL_REGISTRY.md:1480-1485`: within-window ρ **0.445 / 0.497**, across-window
ρ **0.3185 [0.2064, 0.4086]**, argmin **26× the oracle** ⇒ *"That combination is the **winner's
curse**"*. Design-refusal row `V6F_PLANNER_DESIGN.md:258`: **argmin error-rank 132 of 256 — the
median.**

### 3b. `GoalDistanceScorer` — the opposite shape
`V6F_PLANNER_DESIGN.md:428` (SEL-1): `score_i = −‖endpoint_i − ĝ‖/τ + b_i`, `ĝ = W·e_g_tac + c`,
**+267 params (MEASURED, test-pinned)**.
`:385-386`: *"The goal rule's rank IMPROVES with N at every σ ≤ 2 — it is in the supervised
selector's family, not the roll-cost's, because **a candidate-independent reference has no
degenerate minimiser**."* `:441`: *"the goal rule's normalised rank **falls** with N at every
σ ≤ 2; the roll-cost's **rises**."*
`:387-389`: at **σ = 16 m** the goal rule becomes indistinguishable from the roll-cost (6.598 /
rank 0.302 vs 6.450 / 0.286). Zero-information control, `:67-68`: goal replaced by its corpus
marginal sits at **7.8237 m**.

### 3c. Admission criterion σ* ≈ 0.8 m at 2 s
`V6F_PLANNER_DESIGN.md:380-384` **[M]**: *"⭐ **THE ADMISSION THRESHOLD, MEASURED:** the crossover
sits between σ 0.5 (better, separated) and σ 1.0 (worse, separated) ⇒ **σ\* ≈ 0.8 m** on this fan …
**σ\* ≈ 1.7 × (incumbent selected ADE)** and **≈ 4.9 × (fan oracle)**."* Effect sizes `:64-65`:
σ=0.5 m beats the trained selector by **−0.1591 m [−0.2300, −0.0894] separated**; σ=1.0 m loses by
**+0.0943 [+0.0241, +0.1650]**.
Unit pin — `stack/scripts/e_wc2_sigma_star.py:45-56`: σ is **per-axis, not radial**; 0.8/0.4714 =
**1.70** and 0.8/0.1639 = **4.88** reproduce the published ratios; a radial RMS would inflate by
**1.414** and flip FUNDED→INCONCLUSIVE on arithmetic alone.

### 3d. E-WC2 FIRED — SEL-1 REFUSED
`V6F_PLANNER_DESIGN.md:645-686` **[M]**, banner `:645`: *"⛔⛔ **FIRED 2026-08-16 — SEL-1 IS
REFUSED. S-T MUST NOT LAUNCH WITH IT.**"*

| quantity | value | CI95 | n | line |
|---|---|---|---|---|
| σ(2 s) per-axis | **4.7104 m** | [3.8087, 5.6860] | 881 win / 40 ep | `:649` |
| σ(6 s) per-axis | **18.3519 m** | [15.8621, 20.9608] | 681 / 40 | `:650` |
| **σ/ADE** | **9.9915** | **[7.4492, 13.5119]** | vs incumbent ADE 0.4714 | `:651` |
| σ/oracle | **28.7307** | — | vs oracle 0.1639 | `:652` |

REF-C-base agrees σ/ADE **9.6337** (`:654`). Estimator: point estimates full-set; intervals
episode-cluster bootstrap (`taniteval/ci.py`), **2000 draws**; `overlapping_holdout_se` used
nowhere (`:654-656`). Lower bound 7.4492 is **2.48×** the 3.0 refusal threshold (`:659`). §5.3
REDERIVE also fires: σ(6 s) = **3.7481×** σ(2 s) (base 3.7241) on matched 681 windows ⇒
`threshold_6s: null` (`:662-664`).
**But the estimand survives** (`:666-671`): a **0-parameter constant-yaw-rate** goal reaches
σ(2 s) = **1.1888 m**, **3.96× better** than the ridge on frozen REF-C latents — *"these latents
are the wrong surface"*; even that floor sits at **σ/ADE 2.52 — still not FUNDED**.
⚠️ Scope (`:673-677`): this is the **REF-C** surface, **not** frozen S-W latents (never dumped).
Evidence class **EXPLORATORY** for absolutes, tier **T0-DIAGNOSTIC** — *"no T1 claim may cite
it."* Package: `…/incoming/2026-08-16-ewc2-result/EWC2_RESULT.md`.

### 3e. Pre-registered reopening thresholds on frozen S-W latents
`V6F_PLANNER_DESIGN.md:679-684`: **σ ≤ 0.80 m ⇒ FUNDED** (5.89× better than REF-C, 1.49× better
than the kinematic floor, from vision alone); **σ ≤ 1.41 m** merely to leave REFUSED; **σ > 1.41 m
⇒ REFUSED again and `ANCHOR_GOAL` is the line.** One **~10–25 GPU-min** dump at the S-W→S-T
boundary is the only remaining input.
Encoded in `stack/scripts/v6_chain.py:216-237` (`SW_LATENT_ADMISSION`): `funded_at_or_below_m:
0.80` (`:231`), `refused_above_m: 1.41` (`:232`); adjudicator `:343-347`. Unit trap refused by name
at `:285-295`. Estimator thresholds in `stack/scripts/e_wc2_sigma_star.py:161-171`:
`fund_at_or_below: 1.7`, `refuse_at_or_above: 3.0`. `NO_VERDICT` ≠ `REFUSED` (`:93-101`).

### 3f. The S-W latent dumper landing (2026-08-18)
**`stack/scripts/v6_dump_sw_latents.py`** header, `:1-11`: *"⭐ STEP 1 of the selector-admission
recipe — dump **v6 S-W latents** in E-WC2's dump contract. This is the one script the PI's
*'eventually we need a tactical selector'* was blocked on, and it was a SCRIPT, not a decision."*
Four-step recipe at `:13-17` (`v6_chain.sw_admission_recipe`). Dumped blocks + admissibility
classes at `:41-56`; VISION-ONLY measured not asserted, `:58-66` (`v6.py:4045-4066`;
`--vision-only-control` default ON, permutes `v0` **and** `actions`, requires bit-identical
latents; failure = `instrument_fail`). Second dead-gate coupling at `:30-38`
(`e_wc2_sigma_star.py:799-810`, `:443-449`, `:800-806`).
Package **`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-sw-latent-dumper/SW_LATENT_DUMPER.md`**, `:11` **[M]** — end-to-end join on a **planted σ at all three
verdicts**: *"planted 0.30 → recovered **0.3046** → **FUNDED** and the selector launch is ADMITTED;
1.10 → **1.1274** → **INCONCLUSIVE**; 2.00 → **1.9856** → **REFUSED**. The script existing is not
the dump existing."*
`:13` — **the script is NOT on Thor**, `git fetch` on Thor hangs (`raw/thor_readiness.json`).
`:17` — expect `NO_VERDICT` on the real run (live S-W arm has `selector: null` in its own
`config.json`) — *"An operator who sees `NO_VERDICT` and stops has thrown away a good
measurement."* Contract table `:31-45`; coupling-two `--max-horizon 60` must not be inherited
(`:53-55`; grid built at `K_MAX_GRID = 20`). Raw: `…/raw/sw_dumper_roundtrip.json`.

**⇒ Admission path today: REFUSED** (E-WC2 on REF-C). Reopening is FUNDED/INCONCLUSIVE/REFUSED
against σ on frozen S-W latents; producer exists, **measurement outstanding**. Related:
`RETRACTION_LOG.md:4924` (SEL-1's pre-registered reopening path), `:6137` (*"SEL-1 could not report
FUNDED"*), `stack/scripts/e_ag1_anchor_floor.py:5-29`, `stack/tests/test_e_wc2_sigma_star.py`,
`stack/tests/test_v6_dump_sw_latents.py`.

---

## 4. C121 — F-9's gate, band edges, pooled P7

**`RETRACTION_LOG.md:6729-6805`.** Package **`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-p7-per-stratum/`** (commit `a6039363`). **[M]**

`:6733-6738` — `refc-xl-30k` / selector entropy, **881 windows / 40 episodes, T0**:

| cut | ρ | interval |
|---|---|---|
| **pooled** | **0.4656** | [0.3663, 0.5324] — **passes** |
| **`LEAD_20_40m`** | **0.0973** | [−0.2664, 0.4400] — **fails** |

`:6740-6742`: *"**62.5 % of val40 windows are FREE-FLOW, and they carry the pooled number.** …
**F-9's gate row is computable today.**"* (`P7_PER_STRATUM.md:20`: 551/881.)
`:6744-6751`: *"⭐⭐ **AND THE VERDICT MOVES WITH THE BAND EDGES** … the **3-band splits FAIL at
20/40, 15/35, 14/45 and the median split** for entropy, and fail **only at 20/40** for dispersion.
⇒ ⛔ **Until the strata are PRE-REGISTERED, the T3 row can be made to pass or fail by choosing
where the bands sit.**"*
Five-cut matrix — `P7_PER_STRATUM.md:28-32`: LEAD vs NO_LEAD (edge-free, n=270/21 ep) **PASS/PASS**;
20/40 m **FAIL/FAIL** (mid band); 15/35 m **FAIL**(mid)/PASS; 14/45 m **FAIL**(far)/PASS; median
split at **24.58 m** **FAIL**(far half)/PASS.
Root-cause class `:6752-6755`: *"**EXACTLY THE FIT-WINDOW-DEPENDENT EXPONENT, IN A GATE** … the
same log yields −0.387 to −0.738 depending on the window. **A stratum boundary is a window.**"*
Proposed pre-registration `:6756-6758`: primary = **edge-free LEAD vs NO_LEAD (n = 270 / 21 ep)**;
the 3-band split **reported always, gating never**.
Admissibility `:6760-6771`: source is `obstacle.offline` via banked `val40_lead_block.npz`;
`assert_stratifier_admissible` **raises** on ego/model-derived kinds; join verified —
`gt`/`cv`/`speed` byte-identical, `eid` **881/881**, lead-block speeds matching to **1.8e-3 m/s**.
Killed control `:6773-6781`: one shuffle + cluster bootstrap reported **ρ +0.1998, CI [0.0133,
0.4009] — significant, from pure noise**; replaced by `permutation_null`.
Power `:6783-6792`: positive control fires in **every** reported stratum incl. the failing one;
permutation null there spans **[−0.236, 0.218] at p = 0.376**; interval half-widths **0.35 / 0.16 /
0.09** at n = **84 / 270 / 551**. Only `episode_cluster_bootstrap_percentile_95` decides.
Not computable `:6794-6800`: `NO_LABEL` gets no ρ (**5 episodes**); the local 100-ep val cache is a
**different split** (best residual **2.47 m/s**); fine bands need **≈4×** the lead-carrying
episodes. Code/artifacts: `taniteval/taniteval/p7_strata.py`, `taniteval/tools/p7_per_stratum.py`,
`taniteval/tests/test_p7_strata.py`, `raw/p7_per_stratum.json`, `raw/p7_lead_vs_nolead.json`,
`raw/p7_sens_{15_35,14_45,24.58_1000}.json`.

---

## 5. C122 — distance-keeping, lead attach, tier0 collision (commit `d2ede52b`)

**`RETRACTION_LOG.md:6809-6891`**; package **`TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-refa-reconciliation/`**.

**The three metrics (distance-keeping), `REFA_RECONCILIATION.md:238-244` — [metric-suite tier T0],
paired episode-cluster bootstrap, n_used ≈ 218–240 windows in 19 episode clusters:**

| arm | headway_min (m) vs GT | time_gap_min (s) vs GT | min_TTC (s) vs GT |
|---|---|---|---|
| **flagship v1** | **−0.0801** [−0.2412, +0.1497] ⛔ not sep | **−0.0102** [−0.0356, +0.0209] ⛔ not sep | **−0.1566** [−1.1020, +0.7065] ⛔ not sep |
| **REF-A DINOv2** | **−1.4180** [−2.3983, −0.3815] ✅ sep | **−0.3152** [−0.5797, −0.0881] ✅ sep | **−5.8223** [−9.3414, −2.0561] ✅ sep |
| **REF-A dyn-in** | **−1.9295** [−3.4288, −0.5172] ✅ sep | **−0.4959** [−0.9334, −0.1241] ✅ sep | **−5.5187** [−9.5892, −2.1061] ✅ sep |

Direct REF-A − flagship (`:246-248`): headway **−0.9819** [−1.9782, −0.0594] · time-gap **−0.2944**
[−0.5603, −0.0616] · min-TTC **−3.9094** [−6.7124, −1.4463] — all separated, all unsafe direction.
REF-A closes on the lead in **136 of 247** windows against the human's **103**.
Three caveats travel with it (`:250-260`): (1) **not** an independent test of encoder usability —
same predicted path as ADE; (2) **TTC is dt-dependent** — banked spacing **dt = 0.5 s**, scales as
1/dt (`lead_metrics.py:134`), not comparable to dense dt = 0.1 s; headway and time-gap **are**
dt-invariant; (3) censoring **111/247** (REF-A) and **113/228** (flagship) at `TTC_CAP_S = 30 s`.

**Lead block attach, row-for-row — `REFA_RECONCILIATION.md:231-236`:** *"⛔ **`driving.py:608`'s
'no lead-agent state exists' is a STALE BLOCKER.** … **881 block rows = 881 dump rows, episode
partition identical, speed correlation 1.0** (a 0.0018 m/s max difference is expected … realised
spacing ~0.1007 s). **LEAD 270 / NO_LEAD 551 / NO_LABEL 60.**"*

**`tier0` name collision — `REFA_RECONCILIATION.md:70-81`:** `driving_refa-dinov2.json#block` reads
`taniteval.driving/tier0`; that `tier0` is the **METRIC-SUITE** tier (4 waypoints 0.5 s apart vs
the dense 20-step path), **not** `EVAL_DOCTRINE`'s T0. `:83-86` — the §6 rank table where **2.1675**
and **0.4271** sit side by side carries **no tier stamp at all**. Retraction form
`RETRACTION_LOG.md:6828-6832`: *"I told the PI REF-A's **2.1675** was a **T1** driving number.
**It is T0.** `rollout.py:144-153`: the predictor 'is fed the expert's true future actions'."*

**"91× reads better, 5.1× drives worse" — the artefact finding, `RETRACTION_LOG.md:6811-6824`:**
three non-comparabilities — (1) **different arms**: `MODEL_REGISTRY.md:3615` states C104's substrate
as *"frozen `v6F-SW-30k` snapshots"* vs REF-A's opponent `flagship4b-speedjerk-30k` (v1); (2)
**different quantities**: `ego_v0` is supplied as an input to both driving arms (`rollout.py:80`),
`lead_gap`/`lead_closing` sit in the family `driving.py:608` refuses; (3) **different budgets**:
flagship **277,404,073** params vs REF-A **160,514,460 (57.9 %)** — a **116.9 M** gap.
**E-RECON-1 contrast, `:6842-6849` / `REFA_RECONCILIATION.md:294-305`:** LEAD (n = 270, 21 ep)
deficit **+1.7150** [+1.2273, +2.1929]; NO_LEAD (n = 551, 33 ep) **+1.7295** [+1.4313, +2.0274];
**contrast −0.0146 [−0.5988, +0.5551] — NOT SEPARATED**. Longitudinal members point the **wrong
way** for O1 (`long_abs_2s_m` +0.0989, `speed_mae_mps` +0.0914, `progress_abs_err_m` +0.0954 —
none separated). Power bound: a lead-presence benefit larger than **23–39 %** of the deficit is
excluded.
Also `:6861-6867`: `MODEL_REGISTRY.md:1899`'s *"differ in exactly two things"* is **false — at
least four** (SIGReg `full_relaxed` = two changes incl. `free_dims`; h15 imagination **22.06 M**;
grounding depth **13.43 M vs 4.48 M**). `:6874-6875`: SIGReg is **not inert** under a frozen
encoder — state term puts **|grad| 8.93e+01** on the adapter, **0** on the predictor.
**Superseded by C123** (`:6895-6957`): C104's reading is **narrowed to a READOUT claim** — under
REF-A's own geometry the frozen features read `lead_gap` **0.52850** vs C104's condition
**0.44997** (`:6916-6917`); *"'Swap in a stronger encoder' is UNSUPPORTED by our own strongest test
of it"* (`:6940`).
Artifacts: `raw/refa_vs_flagship_families.json`, `raw/refa_lead_rung.json`,
`raw/reproduction_gate_rows.json` (reproduction max diff **4.9e-05**), `code/refa_lead_rung.py`,
`code/refa_vs_flagship_families.py`.

---

## 6. C89 / C89b — tactical band curvature κ

**`RETRACTION_LOG.md:4664-4771`.** C89 retracted *"both axes peak at 2.0 s"*; **C89b retracted
C89's own replacement**.

`:4724-4725`: *"⛔ **`0.2331 / 0.4040` ARE NOT BAND VALUES. Do not quote them as such** — they are
what C89 above told the next reader to use, so this is the highest-propagation-risk number in the
log."*
**How the defect was found — `:4727-4733`:**
`…/2026-08-16-tactical-labels/code/tac_a4_horizon_sweep.py:140-148` anchors **every** horizon at
`t0`:
```python
k  = min(int(round(t0 + H * POSE_HZ)), poses.shape[0] - 1)
dv = float(v[k] - v[t0])          # <-- anchored at t0, at EVERY horizon
```
⇒ the "2.0" row is **(0.0, 2.0]** = `OP_BAND_S`; the "6.0" row is **(0.0, 6.0]**. **No row of that
sweep is the tactical band** — the quantity had to be computed fresh, *"anchored at `t0+20`, read
across `t0+21 … t0+60`"* (`:4739`).

**THE ACTUAL BAND VALUES — `:4741-4748`.** MEASURED, **201 clips**, episode-cluster bootstrap
(`taniteval/ci.py`), PRODUCTION thresholds, statistic **`mean_band`** (mean in-band deviation from
the band start):

| window | LON κ [95 % CI] | LAT κ [95 % CI] | n (LON/LAT) |
|---|---|---|---|
| ⭐ **`TAC_BAND_S` (2.0, 6.0]** | **0.1428** [0.0540, 0.2250] | **0.1777** [0.0658, 0.2953] | 201 / 193 |
| ⚠️ seam (0.0, 2.0] = `OP_BAND_S` | 0.3270 [0.2289, 0.4192] | 0.3132 [0.1973, 0.4323] | 201 / 193 |
| ⚠️ full horizon (0.0, 6.0] *(the "0.2331/0.4040" row)* | 0.2210 [0.1165, 0.3167] | 0.3806 [0.2587, 0.4911] | 201 / 193 |

`:4750-4751`: paired band−seam **LON Δκ −0.1843** [−0.2746, −0.0961] · **LAT Δκ −0.1354**
[−0.2707, −0.0162], **both CI-separated**; *"The true band is **worse than either number C89
offered**."* Third argmax caught (`:4764-4767`): the 2.0 s sheet's thresholds (`Δv 0.75 / Δyaw
0.05`) were **also** κ-maximising.
Artifacts (`:4770-4771`): `…/2026-08-16-tactical-review/code/tacrev_band_agreement.py` ·
`…/raw/b1_band_agreement.json`; rebuilt sheet `…/review/TACTICAL_VISUAL_REVIEW_BAND_2_6S.html`.

---

## 7. Longitudinal ledger

**"88.7 % of oracle gap is longitudinal" — `MODEL_REGISTRY.md:1582-1583`** verbatim: *"Load-bearing
consequence: 88.7 % of the oracle gap is longitudinal, and this is the missing longitudinal state
variable."* Context `:1575-1581` — P1 lead-gap resolution, MEASURED 2026-08-11 ~17:20Z, two runs
pod4, n = **266** vehicle-lead windows, R²(enc) ≤ 0; 2-layer MLP ceiling **−0.334**. Artifacts
`:1590`: `p12_gate_clsfilter.json`, `p1_lead_transforms.json`.
Sharpened at **`:1634-1635`**: *"This sharpens the standing '88.7 % of the oracle gap is
longitudinal' (§1.14, T0) to **~99 % at T1**"* — supporting numbers `:1630-1632`: repaired arm `cl`
ADE **9.3697**, LON along-track MAE **9.2655** vs LAT cross-track MAE **0.7446**; v5f **23.8965 of
23.9837** vs **0.9993** lateral; LON speed MAE **9.73 / 26.94 m/s**.
Repeated at **`:1732`**: *"the half where 88.7 % of the T0 oracle gap was measured to live."* Path
annotation `:1734-1736`: the instrument is **`taniteval/tools/build_lead_block.py`**, not
`tools/build_lead_block.py`.

**`taniteval/taniteval/driving.py:608` — the stale blocker.** Current text (verified in tree,
`:606-608`):
```
"refused": {
    "headway_ttc_distance_keeping":
        "no lead-agent state exists (lead_state is a None stub)",
```
Also stated in the module docstring at `:65`. Introduced **2026-07-25**, commit `df32781a`
(`git log -S`) — **still present, unchanged**. C122 (`RETRACTION_LOG.md:6836-6838`) calls it a
**STALE BLOCKER**: `lead_source.py` and a val40 lead block exist and attach row-for-row, *"and
**REF-A had never been scored on it**."* Sibling refusals in the same dict `:609-617`
(`vtarget_referenced_speed_at_2s` refuted 1.65 vs 0.475 MAE; `lane_centre_deviation`;
`curvature_mae_at_this_resolution` measured 24× the signal).

**`taniteval/taniteval/lead_source.py`** — *"Turn `obstacle.offline` into the `win["lead"]` block
`four_families` consumes"* (`:2`). Provides the **registration** (episode pose index → clip time)
plus **per-window lead assembly**, both pure-numpy (`:16-18`). Key design points: `:20-29` —
`register_poses_to_time` matches the episode's own `poses` (x, y) against the egomotion track and
returns clip time per pose index **with fit residual reported**, needing no camera clock.
`:31-36` — **three window states, never two**: `LEAD` / `NO_LEAD` / `NO_LABEL`; **2.44 %** of the
corpus has no `obstacle.offline`; egomotion runs 20–140 s while labels stop at ~20 s. `:38-47` —
conventions inherited unchanged: rig frame **x forward, y left** (MEASURED: of **2,778** tracks
living ≥ 2 s, **1,756 (63.2 %)** are world-static under x-fwd/y-left vs 236 mirrored, ~32 under
either axis swap — `…/2026-08-03-obstacle-offline-join/raw/frame_convention.json`);
`gap = along − size_x/2` (rig origin to the lead's **rear face**); lead selection at t0 **strictly
causal** (cuboids ≤ t0 only). Constants `:53-60`: `WINDOW = 8`, `K_MAX = 20`, `STRIDE = 8`,
`LEAD_LAT_M = 2.0`. Functions: `RegistrationError` `:88`, `window_last_indices` `:95`, `_theil_sen`
`:120`, `register_poses_to_time` `:131`, `select_lead_causal` `:235`, `lead_track_in_window`
`:274`, `lead_block` `:329`.

**Obstacle join — 12.1 M boxes.** `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-17-train-obstacle-join/TRAIN_OBSTACLE_JOIN.md:1` — *"The train-corpus obstacle join exists —
2,308 episodes, 12.1 M agent boxes, verified by its own consumer"*; `:12` `n_agent_boxes
12,122,129`. Corroborated: `…/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md:77`
(**12 122 129**); `Project Steering/Reports/2026-08-17-2319-program-report.md:83` (**2,308 eps /
433,040 frames / 12,122,129 boxes**, HF + 3-way md5). ⚠️ Correction attached:
`…/2026-08-18-data-strategy-refresh/DATA_STRATEGY_REFRESH.md:138` — the D1 slot-probe claim is
**WITHDRAWN** (probe failed its own positive control, 6.319 m on a tensor a ridge reads at
1.016 m, r = +0.979); **the join itself is unaffected and still MEASURED.**

---

## 8. planner_p2 — cost function and vtarget constants

**File: `taniteval/taniteval/planner_p2.py`** (present in this worktree; note
`MODEL_REGISTRY.md:2689-2691` still carries a 🟥 *"RECONSTRUCTION RISK — P2 is uncommitted … exists
only on `tanitad-eval`"* — **stale relative to this tree**).

Docstring cost spec, **`:33-37`** verbatim:
```
THE COST  J(plan) = w_v·(v̂ − v_target)²                 [track the minted target]
                  + w_c·(accel² + jerk²) + w_s·steer_rate²  [comfort / smoothness]
                  − w_p·progress                          [along-track progress]
                  (+ gap/TTC barrier — SKIPPED in v0: no lead-agent labels in our
                   front-cam+pose data, per the spec's "skip gap term v0")
```
Honest scope, **`:44-50`**: *"The P2 cost is LONGITUDINAL + comfort + progress only. It carries NO
lateral / route / goal term (the strategic goal module is P3)."*
Weights, **`:139-140`**: `W = dict(v=1.0, c=0.10, s=50.0, p=0.02)` — *"engineered cost weights
(physical scales; NOT fit to GT ADE)"*.
Implementation `cost_fn` **`:177-194`**: `speed_err = ((vhat - v_target[:,None])**2).mean(dim=1)`
`:184`; `jerk = (accel[:,1:] - accel[:,:-1]) / DT` `:187`; `comfort = (accel**2).mean(1) +
(jerk**2).mean(1)` `:189`; `steer_smooth = (steer_rate**2).mean(1)` `:190`; `progress = seg.sum(1)
+ traj[:,0].norm(dim=-1)  # arc length to 2 s` `:192`; return `:193-194` = `w["v"]*speed_err +
w["c"]*comfort + w["s"]*steer_smooth − w["p"]*progress`. **No gap/TTC barrier term exists in the
code.** Action envelope `:104-114`: data `|steer| ≤ 0.016`, `|accel| ≤ 1.9`; `sig_steer=0.006,
sig_accel=0.5, min_steer=0.0008, min_accel=0.05`. `vtarget_for` at **`:146-171`**: 85th-pct
free-flow future speed over the next 10–20 s per window.

**`stack/tanitad/lake/vtarget.py`** — the `vtarget_for` port. Constants **`:58-79`**:
`DT = 0.1` `:58` · `VT_LOOK_LO = 100` (10 s — *"documented floor, NOT enforced by vtarget_raw"*)
`:61` · `VT_LOOK_HI = 200` (20 s) `:62` · `VT_MIN_STEPS = 30` (3 s) `:63` · `VT_PCTL = 0.85` `:64`
· `VT_HARD_DECEL = 1.5` m/s² `:65` · `SMOOTH_WIN = 11` (1.1 s Savitzky-Golay) `:68` ·
`SMOOTH_POLY = 2` `:69` · `VT_MIN_LOOKAHEAD = 50` (5 s, the honest floor v2 enforces) `:70` ·
**`VT_GUARD_STEPS = 20`** `:79`.
Three measured defects in the header: `:11-16` lookahead floor never enforced (episodes are **199
frames = 19.9 s**, so realised lookahead decays to 3 s); `:18-24` pose jitter drives the free-flow
gate (dt = 0.1 s amplifies jitter **10×**; ±0.2 m/s wobble fabricates **±2.8 m/s²**; the gate fires
**asymmetrically** ⇒ upward bias before the 85th percentile); `:32-42` both mints read
`v[l+1 : l+200]`, a **superset of the scored horizon** — the nav-echo defect (flagship-v1's route
head an exact bijection, **369/369**, score **1.0000**). `VT_GUARD_STEPS = 20` is **derived, not
chosen** (`:73-78`): `RefCConfig.trajectory.horizons[-1] == 20`, `lead_source.K_MAX == 20`, and the
manoeuvre head's label `dv = v(t+2 s) − v(t)`.

---

## 9. F-cells — definitions and current status

Canonical definitions: **`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md`**.

| cell | prio | definition (verbatim) | line | status |
|---|---|---|---|---|
| **F-7** | P3 | *"T2 manoeuvre contrastives (time-reversal + lane-mirror negatives on `z_tac`) — label-free, S-T-stage loss + weight."* | `:212` | **BUILT, inert at default** |
| **F-8** | P3 | *"T5 temporal-consistency selection loss — needs consecutive-window batches (sampler change) + plan-switch-rate logging (its gate row asks for LATERAL yaw/curvature MAE at selection level)."* | `:213` | **BUILT, inert at default** |
| **F-9** | P3 | *"T3 interaction curriculum (multi-agent entropy from the P8 occupancy readout) — gated on P8 maturity."* | `:214` | **BUILT, inert at default**; gate row now computable (C121) |
| **F-11** | P3 | *"S1 multi-tick strategic rollout (8–30 s = 4–15 strategic ticks) — currently 1-tick; the gate reports `S1_ade_8_30s` against a capability the loss never exercises."* | `:216` | **BUILT**; specified horizon **arithmetically unreachable** |

**F-1** (the `g_str → P_T` port) is not in that table but lives in code:
`stack/scripts/v6_chain.py:576` (*"the g_str->P_T port (F-1). GEOMETRY, exactly like `selector`"*),
`:629-637` (**ON BY DEFAULT**; `--no-tac-goal-cond` *"reproduces the pre-F-1 ladder as a declared
decision, never a default"*), `:1391-1423`, `:1790`, `:2185-2187`. F-15 is sequenced behind it:
`DIAGRAM_CONFORMANCE.md:220` — *"Do not build before the selection question (F-1/SEL-1) settles."*

**Program state — `Project Steering/Reports/2026-08-18-0904-program-report.md:58`:** *"| F-7/F-8,
F-9/F-11 cells | built, inert at default; C115 + C119 found | integrated |"* **[I — the report's
own §2 header at `:45-46` marks this table INHERITED from agent reports, not re-derived.]**

**F-7/F-8 build facts — `…/incoming/2026-08-18-f7-f8-cells/F7_F8_CELLS.md`** (base HEAD `6784455`)
**[M]**: default build **UNCHANGED at 87,893,449 params / 405 keys** (`:10-12`). Costs — **F-7
+164,225 params / +5 keys; F-8 +0 / +0** (`RETRACTION_LOG.md:6331`). Earliest legal insertion —
F-7 at **S-T** (`STAGE_MAY_INTRODUCE["S-T"] += ("t2_head.",)`), no fresh S-W run; F-8 needs no
insertion point (`:6333-6334`).
**C115 (`RETRACTION_LOG.md:6288`)**: *"`z_tac` HAS NO TEMPORAL MIXING: IT IS A FUNCTION OF THE LAST
FRAME ALONE, SO HALF OF CATALOG T2 IS NOT EXPRESSIBLE"* ⇒ `lane_mirror` is the default hard
negative; `time_reverse` built, measured, **excluded** (`F7_F8_CELLS.md:15-23`). **F-8 is
DEGENERATE ALONE** — a flat plan scores **exactly 0**, and the unicycle emission outputs exactly
zero controls at init (`:25-30`; `RETRACTION_LOG.md:6317`). T2 trivial-proxy control now refuses
below `T2_CONTROL_MIN_N = 32` (null ratio range at n=4: **0.397–3.361**; n=256: 0.829–1.036 —
`F7_F8_CELLS.md:36-45`).

**F-9/F-11 build facts — `…/incoming/2026-08-18-f9-f11-cells/F9_F11_CELLS.md`** (base HEAD
`45b8e44`) **[M]**: default build unchanged **87,893,449 / 405**, verified four ways through the
real `build_stack_from_args` (default / F-9 / F-11 / both) — **delta (0, 0) every time**
(`:10-13`; `RETRACTION_LOG.md:6615-6618`). Both cells structurally **zero-parameter**.
**C119 (`RETRACTION_LOG.md:6560-6638`)** — F-9's entropy functional is **inverted 3.9×**: bare
spatial entropy reads **0.9649** on an EMPTY road vs **0.2500** on a dense one; the shipped
mass-gated functional reads **0.0064 / 0.1863 / 0.4886** (empty / one agent / four agents) and an
exactly-empty raster scores **exactly 0** (`:6564-6578`; `F9_F11_CELLS.md:46-55`). F-11:
`t_max = frames − window − max_horizon`, `max_horizon = K·stride_str` ⇒ windows/episode =
**114 − 20K**; K=4 (8 s) **−64 %**, K=5 (10 s) **−85 %**, **K=6 (12 s) ZERO windows** — *"The
catalog asks for 4–15 ticks; only 4 and 5 exist. 30 s is longer than a 12 s episode."*
(`:6582-6591`). K=1 row (94 windows/ep) reproduces `PI_DECISIONS_2026-08-12.md` §D4.
Suites (`:6636-6638`): `stack` **4084 passed / 7 skipped / 2 xfailed**, `taniteval` **1136
passed**, new suites **40 (F-9) + 29 (F-11)**. ⛔ **"Neither cell has been trained — no claim is
made that either improves anything."**
F-9's own gate row before C121 (`:6606-6611`): *"not computable today"* at two probes —
`w7_roll_rerank.py` has zero stratification support. **Superseded by C121**
(`RETRACTION_LOG.md:6742`: *"F-9's gate row is computable today"*).

---

## 10. Horizon spec — `stack/tanitad/models/v6.py` §4b

Header comment `:143`: `# §4b — THE BINDING HORIZON SPEC (PI 2026-08-11, HIERARCHY_VOCABULARY.md)`.
Verbatim `:147-151`:
```
147  PLAN_STEPS = 60
148  DT = 0.1                       # 10 Hz tick — the dense-horizon contract
149  HORIZON_S = PLAN_STEPS * DT    # 6.0 s
150  OP_BAND_S = (0.0, 2.0)         # operative band — fine control authority
151  TAC_BAND_S = (2.0, 6.0)        # tactical band — same controls, g_tac-shaped
```
Exported at `:119-120`; config mirror `:2968-2972` (`plan_steps: int = PLAN_STEPS`); module
docstring `:29` (*"§4b the **binding 6 s horizon spec**"*), `:45` (*"§4b seam-free by construction
… ONE 60-step (a, κ)@10 Hz control"*). Enforcement `:3273`, accessors `:3440`, `:3473`, emission
`:4526`. This is the spec C89 cites at `RETRACTION_LOG.md:4671-4677` and the reason **2.0 s is the
SEAM, not the tactical band**.

---

## 11. C117 — "three probes of the same shape are one probe"

**Rule statement, recorded in `Project Steering/Reports/2026-08-18-0904-program-report.md:139-141`**
(§7 *Incidents — honestly*), verbatim:
> ⛔ **I reported "no credential scanner exists" from three probes. One has existed since
> 2026-07-25.** All three asked "is a scanner PRODUCT installed?" — **three probes of the same
> shape are one probe.**

**The rule in one paragraph.** Absence is only established by probes that differ in *shape*, not in
*instance*. The C117 pass asserted "no credential scanner exists" from three checks
(`RETRACTION_LOG.md:6469-6473`) — but all three asked the same question, and therefore shared one
blind spot: a bespoke in-repo scanner (`safe_commit.py`, existing since 2026-07-25). Repeating a
query against more hosts, paths, or file lists multiplies instances, not independence. Corollaries
already logged: `RETRACTION_LOG.md:2728-2729` (*"a negative probe must vary the PATH SHAPE, not
only the host"*); `:3476` (*"Two probes that share a blind spot are one probe"*). Root canon:
Operating-Standard rule 2 (`MODEL_REGISTRY.md:3341-3342`).
⚠️ Note: in `RETRACTION_LOG.md` the class **C117 (`:6421`)** is titled about the unified perception
corpus being on one disk; the three-probes finding sits inside that entry (`:6467-6473`) and its
scanner half is itself superseded (2026-08-18 program report `:61`, `:142-145` — C111 was
misdiagnosed: the old scanner catches the token's shape; it leaked because nothing called it).

---

## Source manifest

See the agent report's full listing; principal files: `Project Steering/MODEL_REGISTRY.md`,
`Project Steering/RETRACTION_LOG.md`, `Project Steering/V6F_PLANNER_DESIGN.md`,
`Project Steering/Reports/2026-08-18-0904-program-report.md`, `taniteval/taniteval/planner_p2.py`,
`taniteval/taniteval/driving.py`, `taniteval/taniteval/lead_source.py`,
`stack/tanitad/models/v6.py`, `stack/tanitad/lake/vtarget.py`,
`stack/scripts/v6_dump_sw_latents.py`, `stack/scripts/e_wc2_sigma_star.py`,
`stack/scripts/v6_chain.py`, and the incoming packages `2026-08-18-planner-beats-cv-redrive`,
`2026-08-18-refa-reconciliation`, `2026-08-18-p7-per-stratum`, `2026-08-18-sw-latent-dumper`,
`2026-08-18-f7-f8-cells`, `2026-08-18-f9-f11-cells`, `2026-08-16-diagram-conformance`,
`2026-08-17-train-obstacle-join`. Not opened (any number from them would be INHERITED):
`EWC2_RESULT.md`, `E4_SELECTOR_RESOLUTION.md`, `JACK_IN_GATES.md`, `b1_band_agreement.json`,
`tac_a4_horizon_sweep.py`, `EVAL_DOCTRINE.md`, `PI_DECISIONS_2026-08-12.md`.
No files were created, modified, staged, or committed by the compiling agent.
