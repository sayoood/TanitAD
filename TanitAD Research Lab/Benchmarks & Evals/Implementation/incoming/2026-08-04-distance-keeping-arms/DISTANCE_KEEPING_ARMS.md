# The first real distance-keeping numbers — `refc-base-30k` and `flagship-v1` on the canonical val40

- **Date:** 2026-08-04 · **Discipline:** Benchmarks & Eval · **Status:** PENDING orchestrator triage
- **Evidence class:** `MEASURED (ours, this run)` unless a row says otherwise. Artifacts in `raw/`.
- **Cost:** dev-box CPU only. **0 GPU. No training pod touched** (`tanitad-new` / `tanitad-pod4` untouched).
- **Estimator:** paired **episode-cluster bootstrap** (`taniteval/ci.py`), B = 2000, seed 0.
  ⛔ Never `overlapping_holdout_se`.

---

## Lead

**LONGITUDINAL distance-keeping is no longer `NOT COMPUTABLE, n = 0`. It is computed, on real arms,
on the canonical 881-window val40 surface.**

| | |
|---|---|
| **Window states** | **LEAD 270 · NO_LEAD 551 · NO_LABEL 60**, over 881 windows / 40 episodes |
| **The registered prediction** | **270 LEAD / 550 NO_LEAD / 61 NO_LABEL** — ⭐ **LEAD LANDED EXACTLY (270 = 270)**; NO_LABEL out by 1 |
| **Registration** | 40/40 episodes registered, **0 refusals**; probe residual median-of-medians **0.00132 m**, worst **0.00979 m** |
| **Grid spacing** | **0.100496 – 0.101006 s** — the eval grid is confirmed **NOT 10 Hz** |
| **The headline finding** | on distance-keeping the two arms are **statistically indistinguishable**, and **both are indistinguishable from the human** — but flagship-v1 carries a **separated longitudinal speed bias** and an **early-overrun rate the human never produces** |
| **⚠️ Not done** | TACTICAL and STRATEGIC are **UNAVAILABLE on this surface, n = 0**, with reasons in §7 |

---

## 1. How this was made runnable at all — and the parity proof

The eval pod is **gone**: `tanitad-eval` (`69.30.85.106:22073`) answers `Connection refused`, and the
`windows_<arm>.pt` re-analysis surface my brief pointed at is no longer reachable there.

**Absence at one location is not absence.** Three probes of different *path shape*, not just
different hosts:

| what | where it actually was | note |
|---|---|---|
| banked per-window `pred` dumps | ⭐ **already committed in this repo** — `taniteval/results/windows_refc-base-30k.pt`, `taniteval/results/windows_flagship-30k.pt` (27 arms total) | the pod was never needed |
| the canonical val40 episodes | `tanitad-thor:/home/nvidia/valdata/physicalai-val-0c5f7dac3b11` (40/40 sha256 OK, 4.70 GB) | LAN, not a training pod |
| `obstacle.offline` + `egomotion` label chunks | already local, `C:/Users/Admin/tanitad-data/physicalai/labels/` | **37/37 needed chunks present for both kinds**, verified before running |

⛔ **I did not move 4.7 GB.** The scorer reads only `poses` and `episode_id`; `frames_u8` is
117 MB/episode of dead weight for this job. `code/thor_extract_poses.py` shipped a **poses-only
view — 207,990 bytes** — and `code/build_local_val40_view.py` rebuilt 40 `ep_*.pt` stubs from it.

**Parity proof, not parity assertion** (`raw/val40_view_verify.json`):

| check | result |
|---|---|
| `poses_sha256` vs the committed `manifest_EVALPOD_val40.json` | **40 / 40 identical, 0 mismatches** |
| `episode_id` and `T` vs the same manifest | **40 / 40 identical** |
| windows from `window_last_indices` over those `T` | **881** — the canonical count |

Nothing in this package writes into any episode cache, so `physicalai-val-0c5f7dac3b11` and
skip-hash `f09e44db` are untouched **by construction**, not by argument.

### ⛔ The row-alignment precondition — checked before a single metric was computed

A lead block is attached to a banked `pred` **by row position**. A permuted order puts every lead on
the wrong window and still returns a plausible number. So `code/verify_alignment_and_export.py`
rebuilds `gt` from the local poses view with the scorer's own formulae and compares
(`raw/alignment_report.json`):

- **GT max |error| = 7.629e-06 m over 881 × 4 × 2 values** — that is **exactly 1 float32 ULP at the
  largest |GT| present (73.54 m)**. Tightest agreement a float32 dump can give. Alignment proven.
- `eid` compared as a **partition**, not by literal value: passes.
- **Independent confirmation:** the ADE recomputed here reproduces the published headlines to the
  4th decimal — `refc-base-30k` **0.4728**, `flagship-30k` **0.4271**, CV floor **0.8377**
  (`raw/ade_by_lead_state.json` vs `taniteval/results/driving_*.json`).

⚠️ **One real discrepancy found, and it is a definitional one worth recording.** The
`score_val40_lead.py` runner's *internal* "CV" is `[poses[:,3]·t, 0]` — that is **`go_straight` on
the recorded speed channel**, not the canonical floor. The banked dumps' `cv` is
`baseline_waypoints()['constant_velocity']` — the finite-difference last-step velocity vector,
which has a lateral component. They differ by up to **1.343 m** on these windows. **Everything
below uses the canonical `constant_velocity` floor**, passed through `--pred-npz` as its own arm, so
these numbers sit on the same floor every `driving_*.json` headline uses.

---

## 2. ⭐ The falsifiable prediction — it held

The 2026-08-03 package registered an ESTIMATE and said plainly that if the eval-host run did not
land near it, the registration was wrong and the numbers should not be trusted.

| | LEAD | NO_LEAD | NO_LABEL | lead rate over labelled |
|---|---|---|---|---|
| **registered prediction** (ESTIMATED, `t₀` from a prior) | **270** | 550 | **61** | 32.9 % |
| **MEASURED here** | **270** | 551 | **60** | **32.9 %** |

**LEAD is exact. NO_LABEL is out by one window (60 vs 61).** The registration is confirmed
independently of the prediction by the probe residuals (median-of-medians 1.32 mm, worst 9.79 mm)
and by 0 refusals across 40 episodes.

Also confirmed, both by two probes: **`ep_00037.pt` is the single val40 episode with no
`obstacle.offline`** — its **22 windows are NO_LABEL** and are their own denominator, never free
flow. The remaining **38 NO_LABEL windows** are horizons leaving the ~20 s labelled span, in
`ep_00016` (2), `ep_00028` (9), `ep_00034` (8), `ep_00038` (19).

---

## 3. LONGITUDINAL — the family Sayed made binding, both halves

### 3a. Distance keeping — **the numbers that did not exist until today**

`raw/four_family_panel_val40.json`. Denominators: **270 LEAD windows of 881**; each arm's `n` is
lower because **an arm that drives out of the lead's corridor loses its lead** (by design).

| arm | n (of 270 LEAD) | mean min-headway (m) | mean min-time-gap (s) | n time-gap | mean min-TTC (s) | n closing |
|---|---|---|---|---|---|---|
| **GT (human)** | **247** | 28.886 | 3.161 | 221 | 25.023 | 103 |
| **`refc-base-30k`** | **241** | 28.593 | 3.113 | 217 | **25.074** | 92 |
| **`flagship-30k` (v1)** | **228** | **30.572** | 3.130 | 223 | 24.738 | 115 |
| CV floor (`constant_velocity`) | 236 | 27.733 | 3.216 | 209 | 22.930 | 100 |

⛔ **Do not read flagship's 30.572 m as the best distance-keeping.** It is over the **fewest windows
(228)**, and §4 shows those windows are selected by the arm's own overrun. On the **paired**
subset the advantage vanishes.

**Paired deltas — the admissible comparison** (paired episode-cluster bootstrap, B = 2000):

| contrast | Δ min-headway (m) | Δ min-time-gap (s) | Δ min-TTC (s) |
|---|---|---|---|
| **GT − CV** *(the instrument's own control on THIS surface)* | **+0.404 [+0.083, +0.842]** ✅ | **+0.089 [+0.018, +0.191]** ✅ | **+1.148 [+0.222, +2.484]** ✅ |
| `refc-base-30k` − CV | +0.287 [−0.009, +0.763] ✗ | **+0.058 [+0.006, +0.142]** ✅ | **+1.206 [+0.401, +2.545]** ✅ |
| `flagship-30k` − CV | +0.371 [−0.012, +0.930] ✗ | **+0.082 [+0.014, +0.187]** ✅ | **+1.411 [+0.330, +2.999]** ✅ |
| **`refc-base-30k` − `flagship-30k`** | **−0.022 [−0.246, +0.160]** ✗ | **−0.016 [−0.051, +0.010]** ✗ | **−0.086 [−0.940, +0.864]** ✗ |
| GT − `refc-base-30k` | +0.124 [−0.140, +0.396] ✗ | +0.028 [−0.017, +0.084] ✗ | +0.101 [−0.835, +0.996] ✗ |
| GT − `flagship-30k` | +0.080 [−0.140, +0.252] ✗ | +0.010 [−0.019, +0.035] ✗ | +0.157 [−0.682, +1.085] ✗ |

n paired = 195–222 over **19 episodes** (only 19 of 40 val episodes carry jointly-valid lead
windows — that is the real cluster count and it is why these intervals are wide).

**Three readings, in order of confidence:**

1. ✅ **The instrument separates on THIS surface too.** GT − CV is separated on all three metrics,
   reproducing the D-LEAD-1 control on the val40 window set rather than inheriting it.
2. ⛔ **The two arms are indistinguishable on distance-keeping.** All three intervals contain 0,
   and the point estimates are within 0.09 s / 0.02 m of each other.
3. ⚠️ **Both arms are also indistinguishable from the human** on all three metrics — while GT − CV
   *is* separated. Read carefully: this is **not** "the arms drive like humans". It is **"on the
   distance-keeping axis, 19 episodes of val40 cannot resolve the arm-to-human gap, though they can
   resolve the human-to-CV gap"**. The arms sit between GT and CV and the study is underpowered
   for the upper half. The fix is more lead-bearing episodes, not a stronger claim.

### 3b. Target speed — where the arms are **not** indistinguishable

Sparse 4-waypoint view, **`dt = 0.5 s`**. ⛔ `speed ~ 1/dt`, `accel ~ 1/dt²`; `along_*` and the
distance-keeping metrics are dt-invariant. These are **not** comparable to a dense 10 Hz run.

| arm | speed MAE (m/s) | **speed bias (m/s)** | along MAE (m) | **along final bias (m)** | accel MAE (m/s²) | ego-progress ratio |
|---|---|---|---|---|---|---|
| `refc-base-30k` | 0.4460 | **+0.0206** | 0.4166 | **+0.0584** | 0.4976 | 1.0119 |
| `flagship-30k` | 0.4710 | **+0.1911** | 0.3936 | **+0.3375** | 0.7717 | 0.9970 |
| CV floor | 0.4678 | −0.0545 | 0.4401 | +0.0347 | 0.4573 | 0.9914 |

**Paired speed bias** (+ = too fast):

| contrast | Δ speed bias (m/s) | separated |
|---|---|---|
| **GT − `flagship-30k`** | **−0.1911 [−0.2846, −0.0922]** | ✅ **flagship is significantly too fast** |
| GT − `refc-base-30k` | −0.0206 [−0.1060, +0.0631] | ✗ **REF-C is indistinguishable from the human** |
| **`refc-base-30k` − `flagship-30k`** | **−0.1705 [−0.2846, −0.0562]** | ✅ |
| `flagship-30k` − CV | +0.2456 [+0.1399, +0.3430] | ✅ |
| `refc-base-30k` − CV | +0.0751 [+0.0151, +0.1400] | ✅ |

⭐ **On the longitudinal axis the 104 M-parameter REF-C is better calibrated than the 263 M
flagship, and the gap is separated.** Speed MAE does not see it (Δ −0.025 [−0.124, +0.082], not
separated) — the **bias** does, and so does `along_final_bias` (+0.058 m vs +0.338 m).

---

## 4. ⭐ Why flagship's headway "looks best": it over-runs the lead

`raw/lead_retention_by_arm.json`, `raw/overshoot_depth.json`, `raw/early_overshoot_ci.json`.

| arm | keeps the lead | **loses it entirely** | of those, lost by overrun | lost by leaving the corridor |
|---|---|---|---|---|
| GT (human) | **247 / 270** | 23 | 3 | 20 |
| `refc-base-30k` | 241 / 270 | 29 | 7 | 22 |
| CV floor | 236 / 270 | 34 | 2 | 32 |
| **`flagship-30k`** | **228 / 270** | **42** | **25** | 17 |

⚠️ **The naive "gap < 0 somewhere" indicator is NOT a clean contact precursor and I am not quoting
it as one.** It fires **28/270 for both arms** — paired Δ **exactly +0.0000 [−0.012, +0.010]** — and
also for GT (6) and the CV floor (3), with overruns up to ~39 m, because the ego legitimately passes
a stopped or turning-off lead. Reporting it as tailgating would be wrong.

**What discriminates is WHEN the overrun starts** (first horizon step with `gap < 0`):

| arm | 0.5 s | 1.0 s | 1.5 s | 2.0 s |
|---|---|---|---|---|
| GT (human) | **0** | 2 | 2 | 2 |
| CV floor | **0** | 1 | 1 | 1 |
| `refc-base-30k` | 5 | 4 | 11 | 8 |
| **`flagship-30k`** | **17** | 7 | 3 | 1 |

**EARLY OVERSHOOT — already past the lead's rear face at the FIRST waypoint (t + 0.5 s):**

| arm | n / 270 LEAD | rate [CI95] | vs GT | vs CV floor |
|---|---|---|---|---|
| **GT (human)** | **0 / 270** | 0.0000 [0.0000, 0.0000] | — | — |
| CV floor | **0 / 270** | 0.0000 [0.0000, 0.0000] | — | — |
| `refc-base-30k` | 5 / 270 | 0.0185 [0.0034, 0.0400] | **+0.0185 [+0.0034, +0.0400]** ✅ | ✅ |
| **`flagship-30k`** | **17 / 270** | 0.0630 [0.0038, 0.1504] | **+0.0630 [+0.0038, +0.1504]** ✅ | ✅ |
| paired `flagship − refc` | — | — | **+0.0444 [−0.0084, +0.1264]** ✗ **not separated** | |

**Both arms produce a first-waypoint overrun the human and the trivial floor produce zero times in
270 windows.** flagship's point rate is **3.4×** REF-C's, but the paired interval **includes zero**
over 21 episodes — so the ordering is **suggestive, not established**, and I am not calling it.

⇒ **flagship's 30.572 m mean headway is survivorship.** Its 25 terminal overruns are exactly the
tight windows, and they leave its mean. On the 219-window paired subset the headway difference is
**−0.022 m [−0.246, +0.160]**. **Report the mean with its `n`, or not at all.**

---

## 5. LATERAL

| arm | heading MAE (°) | yaw-rate MAE (°/s) | curvature MAE (1/m) | cross-track MAE (m) | cross-track bias (m) |
|---|---|---|---|---|---|
| `refc-base-30k` | 1.146 | 1.851 | 0.007711 | 0.1313 | +0.0020 |
| **`flagship-30k`** | **0.807** | **1.453** | **0.003289** | **0.1152** | −0.0057 |
| CV floor | 3.827 | 3.686 | 0.011723 | 0.5259 | −0.0281 |

**Paired `refc-base-30k` − `flagship-30k`** (positive = REF-C worse):

| metric | Δ | separated |
|---|---|---|
| heading MAE | **+0.402° [+0.165, +0.684]** | ✅ flagship better |
| curvature MAE | **+0.0044 [+0.0024, +0.0069] 1/m** | ✅ flagship better |
| yaw-rate MAE | +0.400 [−0.007, +0.857] °/s | ✗ |
| cross-track MAE | +0.016 [−0.010, +0.045] m | ✗ |

Both arms are separated from the CV floor on **all four** lateral metrics, and both are separated
**from GT** on all four — the lateral gap to the human is real and resolvable, unlike the
distance-keeping gap.

⇒ **The two arms split the families.** flagship wins LATERAL (separated on 2 of 4); REF-C wins the
LONGITUDINAL speed bias (separated). **A pooled score would have cancelled this to "a tie".**

---

## 6. Speed-stratified — and the R0 stratification does **not** transfer

⚠️ **The val40 population is far faster than the 500-clip R0 population the 2026-08-03 stratified
read was measured on.** MEASURED, and cross-checked four ways (egomotion `hypot(vx,vy)` at the
registered `t₀`, the episode's own `poses[:,3]`, the banked dump's `speed`, and per-step
displacement ÷ fitted `grid_dt`; max disagreement **0.002 m/s**, and `poses[:,3]` vs the dump is
**bit-identical**):

| band (m/s) | val40 windows | val40 LEAD | R0 500-clip windows | R0 LEAD |
|---|---|---|---|---|
| 0–1 | **50** | 32 | 1,804 | 183 |
| 1–3 | 52 | 13 | 991 | 73 |
| 3–6 | 127 | 51 | 2,510 | 122 |
| 6–10 | 203 | 74 | 4,293 | 267 |
| 10–15 | 179 | 12 | 1,347 | 38 |
| **15+** | **270** | **88** | 59 | **2 (UNPOWERED)** |

⇒ **The R0 caveats invert on val40.** The "20.7 % of leads sit at 0–1 m/s where the metric cannot
discriminate" defect is **1.9 % here (5 of 270)**; the "15+ band is UNPOWERED (n = 2)" caveat is
false here — **15+ is the LARGEST lead-bearing stratum (88 leads)**. **Neither the R0 pooling defect
nor its band-level deltas may be carried onto val40.**

Per-band min-TTC (episode-cluster bootstrap; bands under 30 lead-bearing windows are reported
**UNPOWERED**, never quoted):

| band | LEAD | `refc-base-30k` | `flagship-30k` | CV floor | GT |
|---|---|---|---|---|---|
| 0–1 | 32 | UNPOW (n=27) | UNPOW (n=8) | 25.74 [18.7, 27.7] | UNPOW (n=29) |
| 1–3 | 13 | UNPOW | UNPOW | UNPOW | UNPOW |
| 3–6 | 51 | 22.91 [16.6, 27.7] | 23.94 [17.5, 27.5] | 19.61 [12.6, 24.0] | 23.90 [17.4, 27.3] |
| 6–10 | 74 | 22.65 [16.5, 26.3] | 23.45 [17.9, 26.7] | 20.14 [12.1, 25.4] | 22.38 [16.2, 26.1] |
| 10–15 | 12 | UNPOW | UNPOW | UNPOW | UNPOW |
| **15+** | **88** | 27.03 [18.5, 30.0] | 25.78 [18.0, 29.8] | 26.50 [18.6, 30.0] | 27.11 [18.9, 30.0] |

⚠️ **Every band's arm-vs-arm intervals overlap.** No band rescues a separation the pooled read did
not have. The 0–1 band's "UNPOWERED for three arms but OK for CV" is itself informative: the arms
lose the lead in the crawl regime more often than the trivial floor does.

---

## 7. ⛔ TACTICAL and STRATEGIC — UNAVAILABLE **on this surface**, with the reason and the n

A missing metric is a work item, not an excuse, so here is exactly what is missing and what it costs.

### TACTICAL — `status: UNAVAILABLE`, **n = 0**

**Reason (MEASURED from the artifacts, not asserted):** the banked `windows_<arm>.pt` dumps carry
only `pred · gt · cv · eid · speed · head_deg · wp_steps`. `maneuver_pred` and `maneuver_gt` are
**absent**. `rollout.collect` is a **world-model fidelity decode fed the expert's true future
actions** — its own `pc2` record stamps `actions_source="expert_future"` and `pc2_pass=False` — so it
**structurally cannot** produce a manoeuvre decision.

⭐ **But the family's DENOMINATOR does exist and is quantified here**, which turns a vague gap into a
scoped task. The val40 cache carries `maneuvers`; on the 881 canonical window origins:

| class | n | share |
|---|---|---|
| `lane_keep` | 511 | 58.0 % |
| `turn_left` | 117 | 13.3 % |
| `turn_right` | 74 | 8.4 % |
| `accelerate` | 93 | 10.6 % |
| `brake_stop` | 86 | 9.8 % |

⇒ **What is missing is one thing only: each arm's declared manoeuvre on these 881 windows.** That is
a hierarchy-traversing forward pass, not a new instrument. **This is the single highest-value
follow-up in this package**, because the programme's top known defect — the 5-way softmax mixing
lat+lon, `0/881 accelerate` — lives in exactly the 93 `accelerate` and 86 `brake_stop` windows above.

⛔ **I did not manufacture a substitute.** Deriving a pseudo-manoeuvre from the predicted trajectory
and scoring it against these labels would measure trajectory geometry, not the head's decision, and
presenting it as TACTICAL would be the C63 failure again.

### STRATEGIC — `status: UNAVAILABLE`, **n = 0**

Same cause: `route_pred` / `route_gt` are absent from the dumps.

⛔ **The nearest MEASURED strategic numbers are on a DIFFERENT SURFACE and one of them is
inadmissible.** `INHERITED` from `…/2026-08-03-nav-known-bit/NAV_KNOWN_BIT.md` — a **closed-loop
NuRec scene, one clip, n = 155**, not these 881 windows:

| | flagship-v1 | `refc-base` |
|---|---|---|
| `route_head_eq_logged` @ known = 1 | ⛔ **1.0000 — INADMISSIBLE** | **0.6452 [0.3694, 0.9154]** |
| echo control | **`ECHO: True` · `DETERMINISTIC_ECHO: True`**, argmax a bijection of its own nav input on **181/181** ticks | `ECHO: False` |

⇒ **flagship-v1's 1.0000 is an echo of its own input read as skill and must never be quoted as
strategic performance.** Only REF-C's 0.6452 is admissible, **and only on that scene**. Neither
number belongs in this panel's table.

⛔ **`flagship-30k` has no trajectory fan at all** (`anchor_decoder is None`; four unimodal
`Linear(d,2)` heads). Every fan-based statistic — goal/anchor selection rank, top-k, picked-rate — is
**undefined** for it. The corresponding REF-C figures (GT-nearest anchor median rank 2, top-5 76.4 %,
picked 24.8 %) exist but are `INHERITED` and on the NuRec scene, so no arm-vs-arm goal-selection
comparison is possible today. **Reported as undefined, not as zero.**

---

## 8. P4 — what this means for the programme's headline claim

**The claim:** 88.7 % of the oracle gap is longitudinal, and the selection gap is independently
87.6–89.9 % longitudinal. Distance-keeping is the first **direct** instrument on that axis.

**⚠️ The result is the interesting one, because it partly contradicts the framing — and I am not
reconciling that away.**

1. **Distance-keeping does NOT reproduce the ADE ordering, and it does not contradict it either —
   because the ADE ordering is not itself separated.** On all 881 windows
   `refc − flagship = +0.0457 [−0.0555, +0.1506]`, **not separated**; on the 270 LEAD windows
   `+0.0727 [−0.0911, +0.2314]`, **not separated**. Distance-keeping agrees: no separation. **The
   "flagship 0.4271 beats REF-C 0.4728" reading was never a separated result** and distance-keeping
   independently declines to break the tie.

2. ⭐ **The distance-keeping axis shows a MUCH SMALLER model-over-floor margin than ADE does.**
   Same arms, same estimator, restricted to the same 270 LEAD windows (`raw/ade_by_lead_state.json`):

   | subset | n | CV floor ADE | `refc − CV` | `flagship − CV` |
   |---|---|---|---|---|
   | all windows | 881 | 0.8377 | −0.365 [−0.556, −0.201] ✅ | −0.411 [−0.624, −0.205] ✅ |
   | **LEAD only** | **270** | **0.5532** | **−0.137 [−0.308, −0.008]** ✅ | **−0.210 [−0.383, −0.038]** ✅ |
   | NO_LEAD only | 551 | 0.8800 | −0.399 [−0.618, −0.213] ✅ | −0.422 [−0.699, −0.168] ✅ |

   **In lead-following windows the trivial constant-velocity floor is already at 0.5532 m and the
   models' advantage over it shrinks by ~2.7×.** Following a lead is the regime where a
   never-braking floor is *hardest to beat*, not easiest — the opposite of the intuition that
   longitudinal error concentrates where a lead exists.

3. **The longitudinal defect that IS separated is a speed BIAS, not a distance-keeping failure.**
   flagship-v1 runs **+0.1911 m/s [+0.0922, +0.2846] fast relative to the human**, REF-C does not
   (−0.0206, interval contains 0). And that bias has a **physical consequence the distance-keeping
   instrument can see and ADE cannot**: flagship is already past the lead's rear face at the first
   waypoint in **17 of 270** lead windows, where the human does it **0 times**.

4. **Therefore:** the "88.7 % longitudinal" framing survives, but it should be **re-attributed**.
   On this evidence the longitudinal gap is **not** primarily a distance-keeping failure — it is a
   **speed-setting bias that manifests as lead overrun**. An arm can hold correct headway
   statistics and still drive through the car in front. ⇒ **Optimising headway/TTC directly would
   target the wrong half of the family; the separated, actionable lever is the speed bias.**

⚠️ **Power caveat that binds every claim above.** The paired distance-keeping deltas rest on
**19 episodes** and 195–222 windows. This surface can resolve GT − CV and it can resolve speed bias;
it **cannot** resolve arm-vs-arm distance-keeping or arm-vs-human distance-keeping. The honest
statement is *"no separation on 19 clusters"*, never *"the arms keep distance as well as humans"*.

---

## 8b. ⭐ Bonus — the programme's first CROSS-ARM distance-keeping ranking (25 model arms + 2 floors)

Because `raw/val40_lead_block.npz` exists, escalation item 2 was cheap enough to do in the same
turn. Of the **27** `windows_*.pt` dumps in `taniteval/results/`, **25 were admitted** (`gt`
bit-identical to the reference, 881 rows) and scored alongside GT and the CV floor;
`refc-v12-smoke-reg` and `refc-v12-smoke-t0` were **REFUSED** (88 rows — a different surface, not
rescaled). `raw/crossarm_distance_keeping.json`, `code/rank_all_arms.py`.

⛔ **Read `kept` beside every row, and do NOT read "above GT" as "better than the human".** An arm
that hangs back, or that loses the tight windows, scores a higher min-TTC on a smaller and easier
set. GT keeps the most windows (247) of any arm.

| arm | kept /270 | headway (m) | time-gap (s) | min-TTC (s) | Δ min-TTC vs CV [CI95] |
|---|---|---|---|---|---|
| `flagship-v4.2-step4000` | 218 | 30.46 | 3.287 | 24.43 | **+1.466 [+0.46, +2.80]** ✅ |
| `flagship-30k` (v1) | 228 | 30.57 | 3.130 | 24.74 | **+1.411 [+0.33, +3.00]** ✅ |
| `flagship-speed` (19 k) | 236 | 29.31 | 3.100 | 24.40 | **+1.219 [+0.01, +2.83]** ✅ |
| `refc-base-30k` | 241 | 28.59 | 3.113 | **25.07** | **+1.206 [+0.40, +2.55]** ✅ |
| **GT (human)** | **247** | 28.89 | 3.161 | 25.02 | **+1.148 [+0.22, +2.48]** ✅ |
| `refc-v12` | 242 | 28.56 | 3.080 | 24.93 | +1.101 [−0.21, +2.97] ✗ |
| `flagship-v16-ab-ft` | 232 | 29.69 | 3.070 | 24.37 | +1.066 [+0.03, +2.65] ✅ |
| `refc-v12-k16reg` | 242 | 28.58 | 3.082 | 24.88 | +1.024 [−0.14, +2.69] ✗ |
| `flagship-v4.1-10k` | 210 | 30.79 | 3.068 | 24.08 | +1.020 [+0.05, +2.33] ✅ |
| `refb-v2-30k` | 229 | 29.30 | 3.097 | 24.51 | +0.974 [+0.16, +2.43] ✅ |
| `refc-xl` / `refc-xl-30k` / `refc-v12-identity` | 230 / 242 | 30.54 / 28.62 | 3.156 / 3.089 | 23.57 / 24.43 | +0.728 / +0.695, both ✗ |
| `refb-v2-20k`, `refb`, `flagship-v3enc-10k`, `refc-xl-live`, `refb-10k` | 221–238 | 27.71–29.84 | 3.078–3.254 | 22.98–23.96 | +0.50 → −0.38, all ✗ |
| **`flagship-nospeed`** *(the no-speed ablation control)* | 240 | 27.94 | 2.897 | 21.13 | **−2.905 [−5.49, +0.22]** ✗ |
| `overfit_refa-dynin-15k` | 175 | 20.58 | 3.207 | 19.58 | −2.995 [−6.91, +1.77] ✗ |
| **`refa-dinov2`** | 247 | 27.18 | 2.927 | 19.18 | **−4.484 [−8.28, −0.77]** ✅ **worse than CV** |
| **`refa-dynin-30k`** | 235 | 27.13 | 2.652 | 19.33 | **−4.582 [−8.39, −0.92]** ✅ **worse than CV** |
| `overfit_refa-dynin-5k` | 172 | 20.18 | 3.309 | 17.89 | **−4.922 [−8.08, −1.08]** ✅ worse |
| `overfit_refa-dynin-20k` | 243 | 26.91 | 2.645 | 17.59 | **−6.090 [−10.97, −1.26]** ✅ worse |
| **`flagship-v2-6k`** | 218 | 23.89 | 2.971 | 17.28 | **−6.412 [−9.48, −2.88]** ✅ worse |
| CV floor | 236 | 27.73 | 3.216 | 22.93 | — |

⭐ **The instrument recovers a known finding it was never tuned on.** The arms it places
**significantly below the trivial floor** are exactly the programme's documented speed-blind ones —
the whole REF-A family and **`flagship-nospeed`, the no-speed ablation control** — while every
speed-input arm sits above it. That is independent evidence the distance-keeping metric is measuring
longitudinal competence and not noise. ⚠️ It is **not** a pre-registered validation; it is a
consistency check found after the fact, and it should be labelled as such.

---

## 9. ⛔ What I did NOT do — plainly

1. **No new inference. No arm was re-run.** Every `pred` is the banked 2026-07-21 dump. No GPU, no
   checkpoint load, no training pod touched. `tanitad-new` and `tanitad-pod4` were never contacted.
2. **TACTICAL and STRATEGIC are not populated on this surface** — §7. I did not substitute a
   trajectory-derived pseudo-decision, and I did not import the NuRec closed-loop numbers into the
   val40 table.
3. **`flagship-30k` = flagship-v1 (`flagship4b-speedjerk-30k`, step 29999)**, per
   `MODEL_REGISTRY.md` §1.2 (`TanitEval keys: flagship-30k` FINAL, `flagship-speed` = the 19 k
   relay). I did **not** score `flagship-nospeed`, which is the no-speed ablation control the
   registry warns is routinely mistaken for v1.
4. **Dense-path distance-keeping was not computed.** These dumps carry only the sparse
   4-waypoint view, so `dt = 0.5 s`. Headway / time-gap / TTC are dt-invariant in definition but
   **TTC's closing rate is estimated over 0.5 s steps**, which smooths it; a dense
   `pred_dense` re-score would sharpen TTC and is a follow-up, not a correction.
5. **No episode was re-selected, no cache written.** Parity artifacts were verified by hash rather
   than asserted (§1).
6. **I did not re-litigate D-LEAD-1** or re-derive the metric; I reproduced its control on this
   surface (GT − CV separated on all three) and moved on.
7. **The cross-arm table in §8b is distance-keeping ONLY.** The other 23 arms were not given a
   four-family panel, a lateral read or a paired arm-vs-arm matrix — that is a follow-up, and the
   ranking must not be quoted as a general leaderboard.
8. **No situation-classifier, goal or nav signal is touched.** Lead state here is an **EVAL LABEL**
   built offline from `obstacle.offline`, never an inference-time input, so neither the vision-only
   rule nor the goal-input rule is engaged. ⚠️ If anyone later feeds lead state to a model, both
   rules apply and this package is **not** the precedent for it.

---

## 10. Tests

| suite | result |
|---|---|
| `pytest -q` in `taniteval/` | ✅ **937 passed**, 1 warning, 202.69 s — matches the brief's expectation exactly |
| `pytest -q` in `stack/` (run 1, 08:44) | 🟡 **2 failed, 2041 passed**, 12 skipped, 2 xfailed, 410 s |
| the same 2 tests in isolation | ✅ **2 passed in 9.18 s** |
| **`pytest -q` in `stack/` (run 2, 08:49)** | ✅ **2068 passed**, 12 skipped, 2 xfailed, 1038.67 s — **GREEN, 0 failures** |

**No test was added or changed by this package** — it is a scoring run over existing instruments,
and it touches **nothing** under `stack/` or `taniteval/`.

⚠️ **The run-1 failures were transient and are now explained — I chased them rather than reporting
"green" on the first pass.** Both were
`tests/test_nav_known_channel.py::test_refc_train_exposes_nav_known_channel_and_it_reaches_the_config`
and `tests/test_refc_labels_v3_wiring.py::test_labels_v3_is_selectable`. Three pieces of evidence:

1. Both **pass in isolation** (2 passed in 9.18 s).
2. `git status` shows **concurrent sibling edits in flight on exactly the files under test** —
   `stack/scripts/refc_train.py` and `stack/tanitad/refs/refc.py` both ` M` — during a 410 s run.
3. ⭐ **The suite GREW between the two runs: 2043 collected → 2082 (2068 + 12 + 2).** A sibling
   stream landed **39** tests while run 1 was executing.

⇒ **Root-cause class: a full-suite run over a working tree that other agents are editing is not a
point-in-time measurement.** The failures were a mid-run source change, not a regression, and
`stack/` is green. ⛔ I did **not** `git stash` to isolate — sibling agents have unstaged work there
and stashing would have destroyed it. ⚠️ **Worth generalising:** any agent quoting a suite count on
this box should state the run's start time, because the tree moves underneath it.

---

## 11. Deliverable manifest

Everything is in the repo working tree, **staged, never pushed**. Nothing is stranded on a pod.

| artifact | path |
|---|---|
| **This report** | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-04-distance-keeping-arms/DISTANCE_KEEPING_ARMS.md` |
| ⭐ **The val40 lead block** (28 KB; makes any banked dump re-scorable with **zero** label I/O) | `…/raw/val40_lead_block.npz` |
| Four-family panel, all arms, paired + stratified | `…/raw/four_family_panel_val40.json` |
| Scorer output, per arm | `…/raw/lead_refc-base-30k.json`, `…/raw/lead_flagship-30k.json`, `…/raw/lead_cv-canonical.json` |
| ADE by lead state | `…/raw/ade_by_lead_state.json` |
| Lead retention / overshoot depth / early-overshoot CI / safety tail | `…/raw/lead_retention_by_arm.json`, `overshoot_depth.json`, `overshoot_ci.json`, `early_overshoot_ci.json`, `safety_tail.json` |
| ⭐ **Cross-arm distance-keeping ranking, 27 arms** | `…/raw/crossarm_distance_keeping.json` |
| Test logs (both suites, both `stack` runs) | `…/raw/taniteval_pytest.txt`, `…/raw/stack_pytest.txt`, `…/raw/stack_pytest_rerun.txt` |
| **Parity proof** (40/40 `poses_sha256` vs the committed manifest) | `…/raw/val40_view_verify.json` |
| **Row-alignment proof** (GT to 1 float32 ULP) | `…/raw/alignment_report.json` |
| Code (12 scripts, all re-runnable on the dev box) | `…/code/` |
| Poses-only val40 view (rebuildable in ~30 s from Thor) | scratchpad only — **deliberately not committed**, it is a 4.7 GB cache's derivative and `code/` regenerates it |

**Off-repo state I created:** `tanitad-thor:/tmp/thor_extract_poses.py` and
`tanitad-thor:/tmp/val40_poses_view.npz` (208 KB, `/tmp` only). **Nothing on Thor outside `/tmp` was
written or overwritten**, so the sibling stream mirroring instruments to Thor is unaffected.

---

## 12. ⛔ ESCALATION — integration, not a note in a README

1. ⭐ **Produce `maneuver_pred` / `route_pred` on the canonical 881 windows for both arms.** This is
   the ONLY thing between the programme and a complete four-family panel (§7). The labels, the
   surface, the estimator and the lead block all now exist; what is missing is one
   hierarchy-traversing forward pass per arm. **This is a GPU item and it is the top follow-up.**
2. ✅ **DONE in this turn — §8b.** All 25 admissible banked arms are ranked
   (`raw/crossarm_distance_keeping.json`). Remaining: give them the other three families too.
3. **Fix the floor definition in `score_val40_lead.py`.** Its internal "CV" is `go_straight` on the
   recorded speed channel, not `constant_velocity` (§1) — they differ by up to 1.343 m. Either
   rename it `hold_v0` in the output or switch it to `baseline_waypoints()['constant_velocity']`.
   Left unfixed, a future reader will compare it to a `driving_*.json` CV number and be wrong.
4. **Correct the stratification caveat where it is quoted.** The "0–1 m/s dominates / 15+ is
   UNPOWERED" caveat is an **R0 500-clip** property and is **false on val40** (§6). It has already
   travelled into the briefs.
5. **Record `tanitad-eval` as gone.** `69.30.85.106:22073` refuses connections and the memory note
   `evalpod-banked-window-dumps` still points there. The dumps are in the repo; the val40 cache is
   on Thor. The note should say so before someone concludes the surface is lost.
6. ✅ **CLOSED, not escalated.** The two run-1 `stack` failures were chased to ground: a clean
   re-run is **2068 passed, 0 failed**, and the suite grew by 39 tests mid-run (§10). No owner
   needed. Worth adding to `RETRACTION_LOG.md` as a **class**, though: *a suite count taken while
   sibling agents edit the tree is not a point-in-time measurement* — quote the run's start time.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Reason:**
- **Landed at:**
