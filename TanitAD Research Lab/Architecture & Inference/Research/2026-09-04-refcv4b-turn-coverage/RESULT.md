# RESULT — refcv4b's 6 s TURN COVERAGE: the flagged regression is an END-BEARING
# ARTIFACT, and the corpus never asks for what the clamp excludes

**tag** `refcv4b-turn-coverage-6s` · **date** 2026-09-04 · **agent** Arch+Inference
FlyWheel · **surface** 4,823 windows / 141 B1-v7.2 EVAL clips ·
**evidence class MEASURED (ours)** · ⛔ **MODEL-FREE** (no checkpoint, no forward
pass — `GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`) · **no T-tier applies**: nothing
here is a model number, and no ceiling below may be compared to an achievement.

---

## VERDICT — **NO ACTION.** Let `refcv4b-b1-v72-40k` run.

> The launch package flagged `8.85 %` of windows with **no candidate whose
> END-BEARING turns > 30°** and explicitly did not measure the TERMINAL-HEADING
> equivalent. **That number is now measured and it is `0.00 %`** — for both the
> coarse 5→6 s slot-segment definition the launch instrument used *and* the exact
> integrated-yaw definition, and it stays `0.00 %` after a μ = 0.7 friction-circle
> filter. **The decision number — the fraction of windows whose GT 6 s heading
> demand exceeds the best available candidate — is `0/4823 = 0.00 %`.** There is
> no manoeuvre in this corpus that the `a_lat_max = 3.0 m/s²` clamp excludes.
> Every amendment that widens `a_lat` makes the oracle-in-vocabulary ADE
> **WORSE**, separated, and raises the friction load; appending a high-`a_lat`
> sub-family changes the oracle ADE by **−0.0000 m**, because those candidates
> are never selected on any window.

Run state at the time of this measurement (read-only probe): step **1,000 of
40,284**, GPU 100 %, 43.4 GB — i.e. the decision was taken at the cheap moment
the brief asked for, and the answer is not to spend it.

---

## 1. What was measured, and against which bytes

| | |
|---|---|
| vocabulary | the **LIVE artifact pulled from the pod**, `sha256 e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb`, byte-identical to the `file_sha256` in the run's own `anchors.build.json` and to `config.json[anchors]`; 117 controls = 13 `a_lon` × 9 `a_lat`, `a_lon ∈ [−4.0, +2.9167]`, `a_lat ∈ [−3.0, +3.0]` |
| curvature derivation | `kappa = clamp(a_lat / max(v0, 4.0)², ±0.12)` per window — the decoder's own line, `refc.py:1351-1356` |
| integrator | `tanitad.models.kinematic.rollout_unicycle`, the function `refa_v1_plan.unicycle_paths` wraps and `roll_bank` calls |
| reach clamp | `refc_select.anchor_reachability_mask(..., accel_max=2.0, horizon_s=6.0)` — the run's own `selection.sel_accel_max` |
| windows | the banked `refcv3-40284-openloop-dump` grid: 4,823 windows / 141 episodes, stride 5 |
| ground truth | ego poses from `/root/data/eval/_v2manifest.pt`, `md5 4c146bbbd982ce79ee5ee6954d186438` (pulled and re-verified locally). ⚠️ **The banked dump's `g` is a 2 s grid**; the 6 s GT does not exist in it and had to come from the poses. |

⚠️ **The 6 s horizon is END-CLAMPED for 22.83 % of windows.** `refc_v3_train.py:366`
clamps `future_poses_ext` at the episode end, so 1,101 of 4,823 windows have a GT
6 s target that repeats the last pose. Every demand statistic is therefore given
on **3,722 true-6 s windows** and, separately, on all 4,823 — never silently
pooled.

### Controls that had to read known values (all PASS)

| control | required | measured |
|---|---|---|
| **K1** dump `v0` vs `poses[ws, 3]` | ≈ 0 or the window→pose mapping is wrong and every GT number is void | **0.000e+00** (exact, all 4,823) |
| **K2** my vectorised roll vs `rollout_unicycle` (float64 `state0`) | < 1e-12 m | **2.84e-14 m** |
| **K3** the `{a_lon=0, a_lat=0}` anchor (index 67) | `y ≡ 0`, `yaw(6 s) ≡ 0` | **0.000e+00** both |
| **K4** constant-zero predictor ADE == mean ‖GT‖ (independent float64) | < 1e-12 m | **1.42e-14 m** |
| **K5** instrument parity with the launch package | reproduce its printed counts | **8.85 %** no >30° end-bearing and **48.3** >30° terminal-heading candidates/window — **both reproduced exactly** |

⚠️ `unicycle_paths` builds `state0` at the torch default dtype (float32,
`refa_v1_plan.py:319`), so its first integration step runs in float32 whatever
the controls are; that path differs from float64 by **1.43e-07 m**. A *dtype*
difference, reported as such rather than as a parity failure.

---

## 2. SUPPLY — the three turn metrics side by side, so they are never conflated again

`%` = windows with **NO** surviving candidate exceeding θ. n = 4,823 / 141 ep,
d = 117 candidates.

| metric | > 15° | **> 30°** | > 45° | > 60° | median max available |
|---|---|---|---|---|---|
| **END-BEARING** `\|atan2(y₆, x₆)\|` (the chord) | 0.00 % | **8.85 %** | 19.49 % | 29.82 % | 93.0° |
| **TERMINAL HEADING**, 5→6 s slot segment *(the launch instrument's own definition)* | 0.00 % | **0.00 %** | 6.03 % | 11.71 % | 161.7° |
| **TERMINAL HEADING**, exact integrated yaw | 0.00 % | **0.00 %** | 3.42 % | 8.67 % | 189.4° |

The reach clamp at `horizon_s = 6.0` is **inert**: the ±12.0 m/s band kills
**0.00 %** of candidates (survivors 117.0/117, no empty window). The clamped and
unclamped rows are identical. *(The 26–37 % kill rates quoted for earlier
vocabularies are 2 s numbers and do not transfer — the run's own `config.json`
already carries that note.)*

**Restricted to physically admissible candidates** (peak realised
|a| ≤ μ·g on the rolled path, so a candidate that accelerates into its own
curvature is charged for it):

| admissibility | surv/win | no > 30° TH | no > 45° TH | no > 60° TH | demand > supply |
|---|---|---|---|---|---|
| reach clamp only | 117.0 | **0.00 %** | 3.42 % | 8.67 % | **0.00 %** |
| + Kamm μ ≤ 0.9 | 102.2 | **0.00 %** | 3.42 % | 9.58 % | **0.00 %** |
| + Kamm μ ≤ 0.7 | 98.0 | **0.00 %** | 5.04 % | 12.79 % | **0.08 %** |

Per-window counts of > 30° candidates: **30.9** by end-bearing, **48.3** by the
coarse terminal-heading segment (the launch figure), **55.1** by exact yaw.

**SIGNED control** (a window turning left needs a *left* candidate, not merely a
large-|turn| one): 0.00 % / 3.42 % / 8.67 % at 30/45/60°, and signed demand >
signed supply on **0.00 %** — identical to the unsigned figures, as a grid
symmetric in `a_lat` requires. That is the control, not a second result.

---

## 3. DEMAND — what the corpus actually asks for over 6 s

| metric | n | > 5° | > 15° | > 30° | > 45° | > 60° | > 90° | p95 | max |
|---|---|---|---|---|---|---|---|---|---|
| GT terminal heading, exact `\|Δyaw\|` (true-6 s windows) | 3,722 | 37.70 % | 22.00 % | **14.05 %** | 8.81 % | 5.21 % | 1.08 % | 61.3° | 109.0° |
| GT end-bearing | 3,722 | 26.68 % | 12.82 % | 5.99 % | 2.47 % | 0.73 % | 0.32 % | 33.6° | 179.9° |
| GT terminal heading, 5→6 s segment | 3,722 | 35.68 % | 20.77 % | 12.71 % | 7.98 % | 4.54 % | 0.99 % | 57.4° | 179.8° |
| GT terminal heading, exact (**all** windows, incl. end-clamped) | 4,823 | 35.81 % | 20.09 % | 12.52 % | 7.71 % | 4.48 % | 0.83 % | 57.0° | 109.0° |

The corpus **does** turn — 14 % of windows swing more than 30° of heading in 6 s
and 5 % more than 60°. **This is not a "the corpus is all straight" dismissal.**
The vocabulary simply supplies all of it.

---

## 4. THE DECISION NUMBER — demand ∩ supply

| statistic | all 4,823 | true-6 s 3,722 |
|---|---|---|
| **windows where GT terminal-heading demand > best candidate** | **0 (0.00 %)** | **0 (0.00 %)** |
| windows where GT *end-bearing* demand > best candidate | 12 (0.25 %) | 11 (0.30 %) |
| windows where GT segment-terminal-heading demand > best candidate | 5 (0.10 %) | 5 (0.13 %) |

**ADE penalty attributable to a terminal-heading coverage hole: 0.0000 m, n = 0
by construction.**

The 12 end-bearing exceedances are **not turns**. They sit at
`v0 ∈ [0.05, 1.41] m/s` (median 1.31) in **2 episodes**, their GT 6 s
displacement is **0.001–7.661 m (median 4.98 m)** against a corpus median of
**54.17 m**, and their GT `|Δyaw|` median is **0.3°**. Their oracle residual is
**2.8166 m ALONG and 0.0152 m LATERAL** — the error is entirely longitudinal
(matching a creep/stop profile with a 13-point `a_lon` grid), and the lateral
term is 1.5 cm. A bearing measured on a 5 m chord at 1 m/s is direction noise,
not a manoeuvre the vocabulary failed to supply. n = 12 in 2 episodes is
**UNDERPOWERED** and is reported as a residual, never as an effect.

### Where the 8.85 % actually lives — it is exactly where the corpus drives straight

| `v0` band (m/s) | n | no > 30° end-bearing | median max EB | median max TH | GT `\|Δyaw\|` p95 | GT end-bearing p95 | demand > supply (EB) |
|---|---|---|---|---|---|---|---|
| [0, 5) | 981 | 0.00 % | 167.3° | 443.9° | 62.1° | 37.9° | 1.22 % |
| [5, 10) | 1,395 | 0.00 % | 139.8° | 285.2° | 70.6° | 39.1° | 0.00 % |
| [10, 15) | 1,222 | 0.00 % | 71.4° | 145.3° | 60.5° | 28.8° | 0.00 % |
| [15, 20) | 585 | 0.00 % | 44.8° | 91.2° | 28.2° | 13.9° | 0.00 % |
| [20, 25) | 291 | 26.80 % | 31.8° | 64.8° | 16.2° | 9.0° | 0.00 % |
| [25, 30) | 194 | **100.00 %** | 24.5° | 49.9° | **4.5°** | **2.3°** | 0.00 % |
| [30, ∞) | 155 | **100.00 %** | 20.6° | 42.0° | **3.2°** | **1.7°** | 0.00 % |

The hole is **entirely** at `v0 ≥ 23.17 m/s` (median 28.50). At those speeds the
corpus's 95th-percentile heading swing is **3–5°** and its 95th-percentile
end-bearing is **1.7–2.3°**. The launch package's own physics is right — a 30°
end-bearing at 27 m/s needs 0.53 g — **and nothing in this corpus asks for it.**

The 427 hole windows carry **no localised ADE penalty**. Their oracle ADE is
**1.1279 m [0.9440, 1.3262]** against **1.1096 m [1.0381, 1.1852]** for the other
4,396 — fully overlapping. *(Disjoint window sets, so a PAIRED estimator does not
apply and is not used; each mean carries its own episode-cluster bootstrap,
`taniteval/ci.py`, n_boot 2000, seed 0. ⛔ `overlapping_holdout_se` is used
nowhere.)* Their mean GT `|Δyaw|` is **1.78°** against 11.65° for the rest.

---

## 5. Oracle-in-vocabulary by GT turn magnitude — ALONG and LATERAL apart

⛔ **This is a MODEL-FREE CEILING on the anchor PRIOR, not an achievement.** The
live decoder adds a per-anchor offset (`refc.py::_decode` → `offset_head`) on top
of the bank, so the model is not confined to these values and they may not be
compared to any model ADE.

| GT `\|Δyaw\|` bin | n | n_ep | **ADE** | **ALONG** | **LAT** | straight anchor | zero path | demand > supply |
|---|---|---|---|---|---|---|---|---|
| < 5° | 3,096 | 137 | 0.8696 | 0.6874 | 0.3563 | 2.0455 | 32.0310 | 0.00 % |
| 5–15° | 758 | 94 | 1.3501 | 0.8320 | 0.8806 | 3.3107 | 27.5862 | 0.00 % |
| 15–30° | 365 | 60 | 1.5756 | 0.9110 | 1.0794 | 5.4978 | 29.6954 | 0.00 % |
| 30–45° | 232 | 46 | 1.6427 | 1.0275 | 1.0776 | 6.6453 | 23.1750 | 0.00 % |
| 45–90° | 332 | 34 | 1.8337 | 1.1814 | 1.1536 | 8.9075 | 21.2437 | 0.00 % |
| > 90° | 40 | **8** | 1.9717 | 1.0510 | 1.4540 | 12.5800 | 19.6605 | 0.00 % ⚠️ **UNDERPOWERED** |
| **pooled** | **4,823** | **141** | **1.1112** | **0.7805** | **0.5921** | **3.2866** | **29.8845** | **0.00 %** |

With episode-cluster intervals: turning windows (`|Δyaw| ≥ 30°`, n = 604 / 46 ep)
**ADE 1.7695 [1.5717, 1.9695]**, ALONG **1.1136 [0.9798, 1.2515]**, LAT
**1.1443 [1.0055, 1.2803]**; straight windows (`< 5°`, n = 3,096 / 137 ep)
**ADE 0.8696 [0.8136, 0.9254]**, ALONG **0.6874 [0.6458, 0.7263]**, LAT
**0.3563 [0.2977, 0.4247]**.

**Reading it.** The ceiling degrades with turn magnitude, 0.87 → 1.97 m, and the
degradation is **split roughly evenly between ALONG and LAT** — LAT rises 0.36 →
1.45 m while ALONG rises 0.69 → 1.05 m. ⚠️ **That is a RESOLUTION / model-class
limit, not a coverage hole**, and the distinction is the whole verdict: the
`demand > supply` column is `0.00 %` in **every** bin, including > 90°. A constant
`(a_lon, a_lat)` arc held for 6 s cannot trace a real 6 s trajectory however wide
the grid is made — and §6 shows widening makes it *worse*.

---

## 6. The counterfactual — every widening amendment LOSES

Same 4,823 windows, same clamp, **paired episode-cluster bootstrap**
(`taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, seed 0, cluster =
episode). Negative = better than shipped. ⛔ `overlapping_holdout_se` not used.

| amendment | n cand | ADE (all) | **Δ vs shipped (all)** | verdict | Δ on GT turn ≥ 30° (n 604 / 46 ep) |
|---|---|---|---|---|---|
| **SHIPPED 13×9, `a_lat_max` 3.0** | 117 | **1.1112** | — | — | — |
| 13×9, `a_lat_max` **4.5** | 117 | 1.1963 | **+0.0850 [+0.0582, +0.1143]** | **WORSE, separated** | **+0.2330 [+0.1009, +0.3703] WORSE** |
| 13×9, `a_lat_max` **6.0** | 117 | 1.3028 | **+0.1915 [+0.1458, +0.2436]** | **WORSE, separated** | **+0.5887 [+0.4201, +0.7626] WORSE** |
| shipped **+ 12 high-`a_lat`** (±4.5, ±6.0 at 3 `a_lon`) | 129 | 1.1112 | **−0.0000 [−0.0001, −0.0000]** | **no effect** | −0.0000, not separated |
| shipped, `kappa_cap` 0.12 → **0.25** | 117 | 1.1111 | −0.0002 [−0.0004, +0.0001] | not separated | −0.0011, not separated |
| 13×**11**, `a_lat_max` 3.0 *(finer, same range, +26 anchors)* | 143 | 1.0695 | −0.0418 [−0.0566, −0.0286] | better, separated | −0.0578 [−0.1167, +0.0118], **not** separated |
| 13×11, `a_lat_max` 4.5 | 143 | 1.1414 | +0.0301 [+0.0151, +0.0462] | WORSE, separated | +0.0565, not separated |

Two things settle it:

1. **Widening the lateral range at fixed budget is strictly harmful** — it spends
   the 9 `a_lat` levels on manoeuvres the corpus never performs, and the loss is
   **largest on exactly the turning windows** the widening was meant to serve
   (+0.2330 m at 4.5, +0.5887 m at 6.0). This is the resolution-versus-range
   trade the launch package's own `a_lon` finding already exhibited, on the other
   axis.
2. **Appending a high-`a_lat` sub-family costs nothing and buys nothing.** Adding
   candidates can only lower a min-over-vocabulary, and the measured gain is
   **−0.0000 m** to four decimals: the 12 extra candidates are the oracle choice
   on effectively no window. That is the minimal intervention the brief asked me
   to price, and its price is that it does not exist.

The only improvement available is **finer `a_lat` at the SAME range** (13×11,
−0.0418 m separated) — but it is a **+26-anchor budget change**, it is **not
separated on the turning windows**, and it does not address turn *coverage*,
which is not broken. It is a candidate for the next vocabulary revision, not a
reason to abort a live run.

### Kamm circle of each family (μ = 0.7), on the ROLLED path

`a_lat_realised = v(t)² · kappa`, so a candidate that accelerates **raises its own
lateral load quadratically** — the constant-`a_lat` label is only true at t = 0.

| family | v0 = 10 | v0 = 18 | v0 = 27 | v0 = 36 |
|---|---|---|---|---|
| **SHIPPED 13×9 `a_lat_max` 3.0** | 18/117, peak **2.28 g** | 10/117, 1.21 g | 4/117, 0.87 g | 2/117, 0.73 g |
| 13×9 `a_lat_max` 4.5 | 26/117, peak **3.41 g** | 18/117, 1.78 g | 12/117, 1.27 g | 8/117, 1.05 g |
| 13×9 `a_lat_max` 6.0 | 36/117, peak **4.54 g** | 28/117, 2.36 g | 26/117, 1.67 g | 22/117, 1.37 g |
| shipped + 12 high-`a_lat` | 22/129, 2.28 g | 14/129, 1.21 g | 8/129, 0.97 g | 4/129, 0.88 g |

Widening to 4.5 m/s² would put **26 of 117** candidates outside a μ = 0.7 circle
at 10 m/s and raise the peak to 3.41 g — for an ADE that gets **worse**.

**Does the unphysical tail matter today?** Barely: the oracle picks a μ > 0.7
candidate on **0.31 %** of windows (2.48 % of the 604 turning windows), the
oracle's median peak load is **0.06 g** (p95 0.30 g, max 1.20 g), and filtering
the vocabulary to μ ≤ 0.7 costs **+0.0043 m** of oracle ADE. It is a small
cosmetic defect of the shipped set, not a reason to restart. ⇒ **logged for the
next revision, not for this run.**

---

## 7. Two instrument findings that outlive this decision

**(a) ⛔ END-BEARING IS NOT A TURN METRIC, AND IT FAILS IN BOTH DIRECTIONS.** The
chord angle to the 6 s waypoint under-reads turn capability where the path is
long (a 6 s arc at 27 m/s ends far ahead, so a real 30° heading change shows as a
~15° chord — hence a fictitious 8.85 % "hole"), and over-reads it where the path
is short (a 5 m creep at 1 m/s gives a chord bearing that is pure noise — hence
12 fictitious "exceedances"). **Both artifacts appear in one 117-anchor set on one
corpus.** ⇒ A turn claim carries the **heading**, and if the coarse
slot-segment version is used it is named as such: the segment and exact-yaw
counts differ by 48.3 vs 55.1 candidates/window and by 6.03 % vs 3.42 % at the
45° threshold. Same family as the `df` / Thor `free` / `step_s` traps: *a
quantity that answers a nearby question, read as the answer.*

**(b) ⚠️ THE BANKED KAMM FIGURES UNDER-REPORT PEAK LOAD BY 1.21–1.85×.**
`anchors.build.json` (and `coverage6s.py::kamm`, `zero_gpu_checks.py`) finite-
difference the **8 slot waypoints**, whose spacing is 0.5–1.0 s, while the
integration step is 0.1 s and the load grows quadratically along the path:

| v0 | banked slot-FD | per-step exact | ratio |
|---|---|---|---|
| 10.09 m/s | 12/117, **1.23 g** | 18/117, **2.26 g** | **1.84×** |
| 18.0 m/s | 4/117, 0.84 g | 10/117, 1.21 g | 1.43× |
| 27.27 m/s | **0/117, 0.68 g** | **4/117, 0.87 g** | 1.28× |
| 36.0 m/s | 0/117, 0.61 g | 2/117, 0.73 g | 1.21× |

The published *"`over_mu_0.7_at_v0_27.27_ms`: 0, `peak_g`: 0.68"* row is **0 and
0.68 g under the slot-FD instrument and 4 and 0.87 g under the exact one**. The
qualitative conclusion (the alat family is far inside the friction circle where
the flat-κ family was at 3.96 g) is unchanged; the specific counts should be
re-stated with their instrument. ⇒ **This is a correction to bank, not a defect
in the run.**

*(I also caught this class in my own instrument mid-analysis: the first C1 pass
compared m/s² against a `g` threshold and printed "114/117 over μ = 0.7, peak
22.4 g". It was caught only because part 2 had already printed 2.28 g for the
same family — a second, independently-written path over the same quantity.)*

---

## 8. The four metric families (binding, per `CLAUDE.md`)

This is a **capacity measurement of an action vocabulary**, not an eval of a
model, so the families are reported as properties of the candidate set. Nothing
is dropped silently.

| family | reported | value |
|---|---|---|
| **LONGITUDINAL** | ALONG-track oracle residual, per turn bin and pooled; the `a_lon` range and its speed-band reach | pooled ALONG **0.7805 m**; 0.6874 → 1.0510 m across the turn bins; on the 12 end-bearing exceedances the residual is **entirely** longitudinal (2.8166 ALONG / 0.0152 LAT) |
| **LATERAL** | LAT oracle residual; heading supply at three definitions; curvature bound and its realised friction load | pooled LAT **0.5921 m**; 0.3563 → 1.4540 m; heading supply table §2; Kamm table §6 |
| **TACTICAL** | manoeuvre-magnitude strata — can the vocabulary *express* each decision class, and what does each cost | §5, all six bins, `demand > supply` **0.00 %** in every one; > 90° bin flagged UNDERPOWERED (n = 40 / 8 ep) |
| **STRATEGIC** | ⚠️ **NOT COMPUTED, with reason** — the anchor vocabulary is a tactical action space and carries no route or goal channel; the strategic path in refcv4b is `nav_cmd` + `lan`, which this artifact does not touch. n/a, n = 0. Measuring it needs a model forward pass and belongs to the T1 arm, not to a MODEL-FREE vocabulary study. |

---

---

## 8b. The ego-dropout regime — turn coverage holds; the GEOMETRIC ceiling does not

The run carries `ego_dropout 0.5`, and `refc.py:1341-1344` rolls the bank for a
**withheld** row at `ref_speed_ms = 10.0` instead of the measured `v0`, so **half
of training rows are supervised against a fixed 10 m/s bank**. Supply was
re-measured there.

⛔ **A defect I nearly manufactured, and the code forbids it.** My first pass
applied the reach band at the row's *true* `v0` to that fixed bank and got **130
empty windows**. That is not what runs: `refc.py:1538` and `:1681` both do
`keep = keep | (~ego_keep)[:, None]`, precisely so a withheld channel cannot
decide which candidates exist — a withheld row keeps **all 117**. The corrected
figure is **0 empty**. *(Reported because the wrong version looked exactly like a
finding, and only reading the gate line refuted it.)*

**Turn coverage is unaffected.** The 10 m/s bank holds **66 / 48 / 40** of 117
candidates above 30° / 45° / 60° of terminal heading, max **191.9°** ⇒ windows
with no > 30° candidate **0.00 %**, and GT demand > supply **0.00 %**. The verdict
does not change.

**The geometric ceiling does change, sharply, and it is a mechanical consequence
of a fixed-speed bank meeting a speed-varying corpus** — not a coverage hole:

| `v0` band | n | OIV ADE, withheld (10 m/s bank) | OIV ADE, measured `v0` |
|---|---|---|---|
| [0, 5) | 981 | 6.9130 | 0.8694 |
| [5, 10) | 1,395 | **2.3750** | 1.2310 |
| [10, 15) | 1,222 | **2.5692** | 1.0835 |
| [15, 20) | 585 | 7.8632 | 1.2372 |
| [20, 25) | 291 | 19.5285 | 1.2414 |
| [25, ∞) | 349 | 37.5622 | 1.0897 |
| **pooled** | **4,823** | **7.5941** | **1.1112** |

paired: **+6.4829 m [+4.9994, +8.0756], SEPARATED** (episode-cluster, n_boot
2000, seed 0).

⚠️ **Scope, stated rather than escalated.** The anchor-classification target is
computed against `out["anchor_bank"]`, i.e. *the bank actually decoded*, so on a
withheld row `a_star` is the nearest anchor in a bank whose geometry does not fit
that window. This bounds what the anchor-classification loss can teach on **~50 %
of rows** — but (a) the offset head is unclamped and corrects on top of the
prior, (b) speed-blindness is the *registered purpose* of E11'/X15 ego-dropout,
and (c) it is orthogonal to turn coverage, which is intact in both regimes. ⇒ **a
measured property of the registered design, for the Master Mind's attention at
the next revision — NOT a reason to touch this run**, and it would need a
pre-registration to change.

## 9. Escalations for the Master Mind

1. ⭐ **NO ACTION on `refcv4b-b1-v72-40k`. Do not abort.** The flagged regression
   does not exist as a turn-coverage defect: `0/4823` windows demand more heading
   than the vocabulary supplies, at every threshold and under a friction filter.
   The minimal intervention (a high-`a_lat` sub-family) is measured at
   **−0.0000 m**; every wider grid is **separated WORSE**. *(I have not touched
   the run; the only pod contact was `cat`/`md5sum`/`base64` of two small files
   and a `tail` of the log.)*
2. **A correction to bank** — the refcv4b Kamm figures in `anchors.build.json`
   and in the vocabulary package's `COVERAGE6S*.json` are slot-finite-difference
   values and under-report peak load by up to **1.85×**. The verdict they
   supported is unaffected; the numbers should be re-stated with their
   instrument. Suggested register class: the `df`/`step_s` scope family.
3. **For the NEXT vocabulary revision, not this run:** (a) 13×**11** at the same
   `a_lat_max` 3.0 is **−0.0418 m [−0.0566, −0.0286]** separated on all windows
   (but **not** separated on turning windows) at +26 anchors — resolution, not
   range, is the live lever on both axes now; (b) **18 of 117** shipped
   candidates leave a μ = 0.7 circle at `v0 = 10 m/s` because constant curvature
   under positive `a_lon` raises `v²κ` — a per-candidate Kamm filter at build
   time costs **+0.0043 m** of oracle ADE and removes the tail.
4. ⚠️ **A measured scope note on the registered ego-dropout lever (§8b), not an
   abort reason:** the withheld-row bank is rolled at a fixed 10 m/s, and its
   oracle ceiling on this corpus is **7.5941 m** against **1.1112 m** at the
   measured `v0` (**+6.4829 [+4.9994, +8.0756]**, separated), rising to
   **37.5622 m** above 25 m/s. Turn coverage there is intact (0.00 % missing);
   it is the *geometry* that does not transfer. Worth a pre-registered look at
   the next revision — e.g. a speed-bucketed reference roll.
5. **The end-bearing metric should be retired from turn-coverage gates** and
   replaced by terminal heading with its definition stamped (§7a). It is the
   metric that generated this whole investigation.

---

## 10. Deliverable manifest

| artifact | where it lives | note |
|---|---|---|
| `RESULT.md` (this file) | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4b-turn-coverage/RESULT.md` | staged |
| `raw/TURNCOV6S.json` | same dir | supply / demand / intersection / turn-bin ceiling, + controls K1–K5 |
| `raw/TURNCOV6S_PART2.json` | same dir | speed-resolved hole, Kamm per family, paired amendment bootstraps |
| `raw/TURNCOV6S_PART3.json` | same dir | Kamm instrument comparison, admissible supply, oracle friction load |
| `raw/TURNCOV6S_PART4.json` | same dir | launch-package reconciliation (48.3 / 8.85 %) + signed-direction control |
| `raw/TURNCOV6S_PART5.json` | same dir | episode-cluster CIs on the hole/rest and turn/straight contrasts |
| `raw/TURNCOV6S_PART6.json` | same dir | the ego-dropout / reference-speed regime (§8b) |
| `raw/per_window.npz` | same dir | 4,823-row per-window bank (v0, ws, eid, demand, supply, oracle ADE/ALONG/LAT, turn bin) — re-analysable with zero GPU and zero pod contact |
| `raw/scripts/turncov6s{,_b,_c,_d,_e,_f}.py` | same dir | the six instruments, runnable off-mount with `PYTHONPATH=<clone>/stack` |
| `raw/anchors_live_refcv4b.pt` | same dir | the LIVE pod artifact, `sha256 e86cf507…e8fb` verified against the pod |
| `raw/_v2manifest_b1eval141.pt` | same dir | ego poses + actions for the 141 eval clips, `md5 4c146bbb…6438` verified against the pod — **this is what makes the 6 s GT reproducible without the pod**; it existed in ONE place (pod `/root/data/eval/`) before this run |

⚠️ **Nothing produced here exists in only one place.** The two pulled binaries
are copies of pod artifacts and are now in the repo with their checksums.

⛔ **Nothing on the pod was modified.** Pod contact was limited to `ls`,
`md5sum`, `sha256sum`, `cat`, `base64` (two files totalling 774 KB), `tail`,
`free`, and one `python3 -c json.load` of `config.json`.
