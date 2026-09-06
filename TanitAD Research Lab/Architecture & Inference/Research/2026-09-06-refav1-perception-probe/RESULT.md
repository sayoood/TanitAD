# RESULT — refav1 frozen-trunk PERCEPTION PROBE (H-REFAV1-PERCEPT-1)

**Evidence class: MEASURED (ours).** Rig: Thor (`tanitad-thor-wifi`).
Checkpoint `/home/nvidia/refav1_lon/ckpt/ckpt.pt`, **step 21109**, loaded with
**0 missing / 0 unexpected** into the code bundled beside it
(`/home/nvidia/refav1_lon/code/stack`, `refa_v1.py` 2,880 lines).
Pre-registration: `SPEC.md`, committed `092db55` **before** these numbers existed.
Raw: `raw/probe_results.json`, `raw/lead_census.json`, `raw/bank_meta.json`,
`raw/probe.log`, `raw/percprobe_lead.mp4.assert.json`.

⛔ **TIER.** This is a **representation diagnostic**, not a driving number. It is
neither T0 nor T1: nothing here is rolled out and nothing here is a capability
claim. It answers one question — *what is IN the latent* — and its verdict
constrains where refav1's longitudinal defect can live, nothing more.

---

## 0. The one-line answer

⭐ **refav1's longitudinal failure is a PLANNER/COST defect, not a perception
defect — for the GAP. But the CLOSING RATE is absent from the latent entirely,
and distance-keeping needs both.**

---

## 1. The freeze is PROVEN, not asserted

| check | value |
|---|---|
| trunk tensors fingerprinted (params + buffers) | **407** |
| sha256 mismatches before vs after | **0** |
| `max |delta|` over every trunk tensor | **0 exactly** |

The head is a **closed-form ridge solve** on pre-computed features, so the trunk
never receives a gradient — but it is fingerprinted before and after regardless,
because "it cannot have changed" is an argument and this is a measurement.

## 2. What was probed, and why it is the right tensor

`plan()` computes `field = self.encode(feats)` then `last = self._last_state(field)`
— and `last` is the root every iCEM rollout starts from, i.e. **the latent the
planner's cost is evaluated in**. That is the tensor this probe decodes. The arms
named in the register (`wk15`, `ha0_ext`, `lonseam`) are **cost-weight triples
over this one shared trunk**, so a single checkpoint is the correct object.

Corpus: the **193 clips** carrying BOTH the agent join and cached DINOv3
features → **18,477 rows**, **3,615 lead rows**. Split **116 fit / 77 scored
clips**, episode-disjoint, seeded. Pooling `16x40 → 8x20` (20 azimuth bins,
6°/bin). Every hyper-parameter — both PCA stages and the ridge λ — fit on the
FIT split only, λ by clip-grouped 5-fold CV inside FIT.

**Alignment VERIFIED, not assumed:** 201 episode frames → 101 feature rows, so
row *j* ↔ episode frame *2j*; re-checked by an offset sweep (§6b).

---

## 3. THE DECISIVE CELL — `lead_gap_m` (lead ≤ 30 m, vehicles only)

n = **1,586** scored rows over **42** bootstrap clusters. `d` printed per arm.
⚠️ **42, not 77.** The split has **77** scored clips, but only **42** of
them carry a labelled lead within 30 m — and it is the **cluster count, not the
row count and not the split size, that is the n behind every interval here.**

| arm | R²_skill | 95 % CI (episode-cluster) | within-clip | d |
|---|---|---|---|---|
| `constant` (control) | **+0.000000** | — | — | 0 |
| `pix` — RAW-PIXEL FLOOR | **−0.0513** | [−0.1702, +0.0643] | −0.1123 | 128 |
| `dino` — frozen DINOv3 | +0.2225 | [+0.0744, +0.3761] | +0.2726 | 128 |
| ⭐ **`field` — refav1 latent** | **+0.3632** | **[+0.2069, +0.5088]** | **+0.3995** | 128 |
| `pix_rff` (nonlinear) | −0.0693 | [−0.3475, +0.2035] | −0.2686 | 1024 |
| `dino_rff` (nonlinear) | **+0.5883** | [+0.4860, +0.6754] | +0.3811 | 1024 |
| `field_rff` (nonlinear) | +0.4411 | [+0.3027, +0.5610] | +0.3429 | 1024 |
| `shuffle_within_clip` | +0.0307 | [+0.0018, +0.0643] | −0.0013 | 128 |
| `shuffle_within_clip_rff` | +0.0000 | [+0.0000, +0.0000] | +0.0000 | 1024 |

**PAIRED deltas** (same resampled clips score both arms — overlapping marginal
CIs are not a null result):

| comparison | delta | 95 % CI | |
|---|---|---|---|
| ⭐ **`field` − `pix`** | **+0.4145** | **[+0.2018, +0.6120]** | **EXCLUDES 0** |
| `field` − `dino` | +0.1407 | [+0.0616, +0.2103] | EXCLUDES 0 |
| `dino` − `pix` | +0.2738 | [+0.0694, +0.4794] | EXCLUDES 0 |
| `field_rff` − `pix_rff` | +0.5104 | [+0.2347, +0.7991] | EXCLUDES 0 |
| ⚠️ **`field_rff` − `dino_rff`** | **−0.1472** | **[−0.2708, −0.0247]** | **EXCLUDES 0** |
| `field_rff` − `field` | +0.0779 | [−0.0824, +0.2419] | spans 0 |

⭐ **PASS on the pre-registered criterion.** `field` beats the raw-pixel floor
(paired **+0.4145**, CI excludes zero) and beats the constant.
⚠️ **The within-clip shuffle is small but NOT zero: +0.0307, CI
[+0.0018, +0.0643], which EXCLUDES zero.** The honest readable quantity is
therefore **TRUE − SHUFFLED = 0.3632 − 0.0307 = +0.3325**, not the raw +0.3632.
The leakage-immune statistic agrees: `field`'s **within-clip** skill is
**+0.3995** against the shuffle's **−0.0013** — so it tracks the lead **as it
moves**, not merely which clip it is looking at.

⇒ **The lead's LONGITUDINAL POSITION IS in the latent. A cost in this space CAN
express "keep distance". The objective is REPRESENTABLE, and refav1's
longitudinal gap on the GAP term is therefore a PLANNER/COST defect.**

⚠️ **The third pre-registered outcome ALSO fired, in the nonlinear class only.**
Linearly `field` > `dino` (+0.1407, excludes 0); **nonlinearly `dino` > `field`
(−0.1472, excludes 0)**. Read together: refav1's adapter has made lead
information more *linearly* accessible while ending up with **less of it in
total** than the frozen encoder it was built on. That is a real, separate defect
with its own fix (adapter / objective), and it must not be folded into the PASS.

---

## 4. THE SECOND FINDING — the CLOSING RATE is not there at all

n = **1,491** scored rows.

| arm | R²_skill | 95 % CI | |
|---|---|---|---|
| `constant` | +0.000000 | — | control reads its known value |
| `pix` | −0.0001 | [−0.0003, +0.0002] | |
| `dino` | +0.0114 | [−0.0575, +0.0718] | spans 0 |
| `field` | +0.0061 | [−0.0406, +0.0513] | spans 0 |
| `dino_rff` | +0.0086 | [−0.0295, +0.0416] | spans 0 |
| `field_rff` | −0.0009 | [−0.0667, +0.0507] | spans 0 |

Every paired delta spans zero. **Nothing decodes the closing rate — not refav1,
not frozen DINOv3, not linearly, not nonlinearly.**

⛔ **AND THIS IS A NULL ABOUT THE REPRESENTATION, NOT A NOISY LABEL.** The
obvious objection is that a one-step finite difference of quantised cuboid
centres is mostly noise. It is not: the closing rate's **lag-1 autocorrelation is
+0.7836**, strongly POSITIVE, where a finite difference of white noise reads
**−0.5**. The target is a smooth physical signal (mean −0.64 m/s, **std
2.61 m/s** over 3,429 rows). So the null stands against a real target.

⭐ **Why this matters more than it looks.** Distance-keeping is a function of
**gap AND rate**. The gap is present; the rate is absent. A cost built in this
latent can say *"the lead is 18 m away"* but **cannot say "and I am closing at
2 m/s"** — which is precisely the term a following controller needs. This is a
**representation** work item, and it is upstream of any cost re-weighting.

---

## 5. Controls, and the one that changed a conclusion

* **`constant` read EXACTLY +0.000000 on every target.** The metric is
  `1 − SSE_model/SSE_fitmean`, so this is true by construction — and it is
  reported because when it is *not* exactly zero the panel is unreadable.
* **The raw-pixel floor is a real floor**, not a formality: on the decisive cell
  it reads **−0.0513**, i.e. worse than predicting the mean.
* ⭐ **THE WITHIN-CLIP SHUFFLE EARNED ITS PLACE — it demolished the agent-count
  cell.** On `log1p_n_agents_visible` (n = 7,510) `field` reads **+0.5726** and
  looks like a strong result; the **within-clip shuffle reads +0.4958**, and
  *every* arm's within-clip skill is **negative** (`field` −0.1321, `dino`
  −0.2691, `pix` −0.2158). ⇒ **almost the whole agent-count decode is CLIP
  IDENTITY, not per-frame perception**; the readable quantity is
  TRUE − SHUFFLED ≈ **+0.077**.
  ⚠️ **Scope this honestly.** This does NOT reconcile the register's open
  disagreement on `n_agents` decodability (LAB_BACKLOG **P-5**: +0.1220/+0.0998
  lead-matched vs +0.3881/+0.2754) — those are different arms, corpora and
  pooling. What it does say is that **an `n_agents` number without a within-clip
  shuffle control is not interpretable**, and none of the disputed panels
  reported one at this strength.
* ⚠️ The same control is **substantial at 80 m** (`shuffle` +0.1517 vs `field`
  +0.2486): the wider the gap cap, the more between-clip variance there is to
  exploit. This is the reason the **30 m cap is the decisive cell** — it is the
  car-following regime *and* the one where clip identity buys least.

---

## 6. Every cell, for completeness

| target | n | `constant` | `pix` FLOOR | `dino` | **`field`** | `shuffle_wc` | field−pix (paired) |
|---|---|---|---|---|---|---|---|
| **`lead_gap_m` ≤30 m** | 1,586 | +0.000000 | −0.0513 | +0.2225 | **+0.3632** | +0.0307 | **+0.4145 EXCLUDES 0** |
| `lead_gap_m` ≤80 m | 3,117 | +0.000000 | +0.0149 | +0.2402 | +0.2486 | +0.1517 | +0.2337 EXCLUDES 0 |
| `lead_lat_m` ≤30 m | 1,586 | +0.000000 | +0.0000 | +0.0866 | +0.0475 | −0.0319 | +0.0475 spans 0 |
| **`lead_closing_mps`** | 1,491 | +0.000000 | −0.0001 | +0.0114 | **+0.0061** | +0.0037 | +0.0062 spans 0 |
| `log1p_n_agents_vis` | 7,510 | +0.000000 | +0.1596 | +0.5317 | +0.5726 | **+0.4958** | +0.4130 EXCLUDES 0 |
| `lead_present` ≤30 m | 7,510 | +0.000000 | +0.0100 | +0.4868 | **+0.5137** | +0.2622 | **+0.5037 EXCLUDES 0** |

⭐ **`lead_present` is the cleanest secondary result**: `field` **+0.5137**
against a pixel floor of **+0.0100**, paired **+0.5037** excluding zero, and —
unlike `n_agents` — its **within-clip** skill is genuinely positive (`field`
**+0.2740**, `dino` +0.2607) against a within-clip shuffle of **−0.1630**. The
trunk knows **whether** there is a lead and **where** it is; it does not know how
fast it is approaching.

⚠️ **`lead_lat_m`'s pixel floor is uninformative BY CONSTRUCTION, not a
finding:** its λ selected the **top of the grid (1e6)**, i.e. the ridge collapsed
to the constant predictor and read +0.0000 with a near-zero-width CI. That is the
2026-08-22 failure signature, caught here by the control rather than published.
The lead's lateral offset is also low-variance by selection (|cy| ≤ 1.75 m), so
this cell has little to explain.

---

## 6b. Alignment sweep — reported against the letter of the pre-registration

Target shifted ±2 rows (±0.4 s) against the `field` arm on the decisive cell:

| offset | −2 (−0.4 s) | −1 (−0.2 s) | **0** | +1 (+0.2 s) | +2 (+0.4 s) |
|---|---|---|---|---|---|
| R²_skill | +0.3633 | **+0.3684** | **+0.3632** | +0.3435 | +0.3194 |

⚠️ **The letter of the criterion said "best offset must be 0", and the best is
−1, by +0.0052.** Reported as it came out. That margin is ~3 % of the cell's CI
half-width (±0.15), so the sweep **does not resolve ±1 row** — which is exactly
what a target with **lag-1 autocorrelation +0.7836** should do.

What the sweep *does* establish is the thing it was for: **no gross
misalignment** (a 2-row error would show a clear drop; offsets −2..0 are flat
within noise while +2 falls 0.044). And its **asymmetry is mechanistically
expected**: `_last_state` is built from a **4-frame window ending at row j** plus
a motion term `field[:,-1] − field[:,-2]`, so its temporal centre of mass sits
slightly *before* j — a mild preference for −1 is what that construction
predicts, not a bug.

---

## 7. THE VISUALISATION (P2 — what the PI asked to see)

`raw/percprobe_lead.mp4` — **10,875,839 bytes**, md5
`88774a6a9466a44ebc55393832e02a0b`.
Also on Thor at `/home/nvidia/percprobe/raw/percprobe_lead.mp4`.
Receipt: `raw/percprobe_lead.mp4.assert.json`.

Standing viz standard, honoured: **camera projection + metric BEV inset
TOGETHER + text overlay**. Both panes draw the **decoded** lead against the
**ground-truth** track, *and* the **raw-pixel floor's** decode beside it — the
control is in the picture, because without it the trunk's marker means nothing.
Only **SCORED-SPLIT** clips are drawn: the head never saw them.

⛔ **CONTENT ASSERTIONS — the exit code is not the check** (the last render was
stretched 3.9 % on every frame while every exit code read 0). The written mp4 is
**decoded back** and asserted:

| assertion | value | |
|---|---|---|
| frames written == frames decoded back | **398 == 398** | PASS |
| geometry written == geometry decoded | **[1700, 586] == [1700, 586]** | PASS |
| mean luminance of decoded frames > 1 | **55.32** (source 75.67) | PASS |
| projected markers actually in frame | **1,193** | PASS |
| **ALL_ASSERTIONS_PASS** | **true** | |

**Projection preflight** (run before the render, not after): HFOV reads
**120.00°** from the cylindrical formula — the rig's own name is
`camera_front_wide_120fov`, while the pinhole formula gives 92.6° and looks
entirely plausible. A rig point 10 m straight ahead projects to **col 319.1**
against an image centre of 319.5; at 20 m it moves **up** the image (row 186.4 →
152.6); and **+y (LEFT) 3 m lands LEFT of centre (col 208.2)** while −3 m lands
right (col 430.5) — so the handedness of the whole chain is confirmed, not
assumed.

⭐ **Per-clip extrinsics are demonstrably live, not a constant:** the four drawn
clips report camera heights **1.3172 / 1.5597 / 1.2856 / 1.3335 m** and pitches
**+0.561 / +0.301 / −0.459 / +0.510°**. (554 distinct heights exist over the
2,400 parity clips; a constant would be wrong for almost all of them.)

⚠️ `*.mp4` is git-ignored in this repo, so the video is banked **on disk in the
package directory and on Thor**, with its md5 and its assertion receipt committed.

---

## 8. What this does NOT say

* ⛔ **Not a driving claim.** No tier, no rollout. A decodable latent is
  **necessary, not sufficient** (C131: flagship v1 had the highest rank ever
  measured here and no environment interpretation).
* ⛔ **The estimator answers ONE question.** The episode-cluster bootstrap says
  *"would another draw of EPISODES say this?"* It does not answer *another
  training run* (`H-ESTIM-SEED-1`) or *another inference run*. Here the trunk is
  **frozen** and the head is a **closed-form solve** — there is no training or
  inference stochasticity to price, so the episode draw and the split seed are
  the only randomness. **That is why this panel's separated CIs are admissible
  where a one-seed trained-arm comparison's would not be.**
* ⚠️ **`field_rff` is thin on the capped cell**: d = 1024 against ~2,169 fit
  rows (n/d ≈ 2). The linear arm (d = 128, n/d ≈ 17) carries the verdict; the
  nonlinear arm is reported with its `n` and `d` beside it and is not the basis
  of the PASS.
* The corpus is the 193 clips that already had DINOv3 features — **not** selected
  for lead presence, so it does not carry the LEAD130 selection confound that
  flipped an earlier panel from −0.3515 to +0.1220.

---

## 9. What it delivers beyond the verdict

⭐ **The lead track, recovered.** The 433,040-record / 2,308-clip agent join
(`joins/train2400_agents.jsonl.xz`) had been lost to a dead session scratchpad.
It is re-pulled from HF `Sayood/tanitad-ph0-aug120` and verified **md5
`24cbdca8c3b23aafc2fb17e6bf99cf76` == the banked expected value**. Egomotion is
present on Thor for **all 55 chunks** covering these clips (spot-checked to the
per-clip parquet). ⇒ **Both inputs `build_lead_tracks.py` needs for the
`distance_keeping` family now exist on one box** — the family refav1 has never
reported (`UNAVAILABLE, n = 0`). ⚠️ The builder as written reads the raw label
zips, so pointing it at the join needs a small adapter; that is a named, cheap
work item, not a claim that the metric has been run.
