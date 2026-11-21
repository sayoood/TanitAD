# FAN WIDTH — how many trajectory hypotheses does REF-C actually need?

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** `arch-inf` · **Branch:** `agent/arch-inf-20260803`
**GPU cost:** ~4 minutes of an idle Jetson AGX Thor (latency only). Everything else is 0-GPU on
banked fans. ⛔ **No training pod was touched** (`tanitad-new`, `tanitad-pod4` are training).

**Pre-registration:** `…/2026-08-04-fan-width/PREREG_FAN_WIDTH.md`
**Blob id, re-verified at the end of the pass:** `1bffa9db6a6047325dceff1ef787d67ab2fd5152` —
`git ls-files -s` and `git hash-object` **MATCH**, so §4/§5/§6's thresholds are the ones staged
before any sweep ran.

**n = the canonical 881 windows / 40 episodes**, all three arms.
**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap` / `episode_cluster_bootstrap`,
unit = **episode**, `n_boot = 2000`. ⛔ **`overlapping_holdout_se` is never called.**

---

## 0. THE HEADLINE

> ### 0.1 ⛔ The fan is NOT ~4× too wide. Truncating it costs accuracy at EVERY rung.
>
> **Neither saturation curve saturates.** On the FPS-prefix ladder, *every* rung below full width
> is **separated-worse** than the full fan — for the **selected** ADE *and* for the **oracle-in-fan**
> — on both arms. `N*(selected) = N*(oracle) = N_max`, i.e. **unreached**.
>
> | XL, N | selected ADE | Δ vs 256 | oracle-in-fan | Δ vs 256 |
> |---|---|---|---|---|
> | 32 | 1.3914 | **+0.9200** [+0.7428, +1.0955] worse | 0.8061 | **+0.6421** [+0.5415, +0.7421] worse |
> | 64 | 0.9213 | **+0.4498** [+0.3410, +0.5787] worse | 0.4368 | **+0.2728** [+0.2296, +0.3187] worse |
> | 128 | 0.5888 | **+0.1174** [+0.0799, +0.1573] worse | 0.2624 | **+0.0984** [+0.0739, +0.1255] worse |
> | 256 | **0.4714** | — | **0.1640** | — |
>
> ⇒ **My registered prediction (O1, `N*(selected) ≤ 32`) is REFUTED.** The registered branch that
> fires is **O4** — the oracle is still improving at the widest fan we own — together with a
> selected curve that is *also* still improving. On the truncation ladder the answer is **O2,
> JOINTLY BOUND**. I said in §4.1 that if O2 fired I would say the compute lever is dead. On *this*
> operationalisation, it is.

> ### 0.2 ⭐ But the answer is **92 of 256** — and it is BIT-EXACT, not a tolerance.
>
> Filter the **anchors** (not the decoded fan) through the bounded-acceleration band — `v0` is known
> before any decode, so this costs nothing — and decode only the first **N_suff** surviving anchors
> in FPS order:
>
> | arm | full fan | **fixed N_suff** | saving | **variable width** (decode exactly the survivors) | **avg saving** | selected ADE | selection index vs full fan |
> |---|---|---|---|---|---|---|---|
> | `refc-small-30k` | 64 | **23** | 2.78× | mean **17.3** (med 19, max 25) | **3.70×** | 0.5261 | **881 / 881 identical** |
> | `refc-base-30k` | 128 | **46** | 2.78× | mean **36.0** (med 39, max 51) | **3.55×** | 0.4728 | **881 / 881 identical** |
> | `refc-xl-30k` | 256 | **92** | 2.78× | mean **74.0** (med 83, max 102) | **3.46×** | 0.4714 | **881 / 881 identical** |
>
> Not "within 0.02 m" — **the same integer index on every window**, so the emitted trajectory, all
> four metric families and distance-keeping are byte-for-byte unchanged (paired Δ **exactly 0.0**,
> every component, §5).
>
> **Two policies, both bit-exact HERE — but their guarantees differ, and it matters for shipping.**
>
> * **Variable** (decode exactly this window's survivors): the guarantee is **structural**. The
>   subset contains every survivor by construction, and the winner is always a survivor, so the
>   argmax cannot move. Better on average (**3.46–3.70×**), needs a ragged batch shape.
> * **Fixed `N_suff`** (a static tensor shape, easiest to ship): the guarantee is **EMPIRICAL**.
>   XL's worst window has **102** survivors, more than the budget of 92 — the policy works because
>   the winner's rank among survivors never exceeded **92** on these 881 windows (median 38, p95 82).
>   ⚠️ **That is a calibration, not a theorem**, and a new corpus could exceed it. Ship it with the
>   runtime guard in §8 item 1, or ship the variable policy.
>
> `raw/variable_width_policy.json`.
>
> **Why it works, and it is the literal content of a result we already had.** *"S2 is exactly
> ADE-inert"* means `argmax(logits)` over the full fan **never leaves the band** —
> MEASURED here at **1.0000 of windows, n_fail = 0**, and (the part that had to be measured
> separately) **1.0000 under the ANCHOR-level band too**, which agrees with the decoded band on only
> ~96 %. A subset that contains every survivor therefore cannot change the argmax.
>
> ⇒ **~64 % of what we decode is provably dead weight.** The clamp is not a no-op; it is a
> **precondition that frees budget**, exactly as prereg §6's **R1** branch registered.

> ### 0.3 ⚠️ And the honest deflation: on Thor that 2.78× is worth **6.5 %** end-to-end.
>
> MEASURED on Thor (aarch64, torch 2.13.0+cu130, batch 1, `steps = 2`, warmed, p50):
>
> | XL | encoder (N-independent) | decoder @256 | decoder @92 | end-to-end | Hz |
> |---|---|---|---|---|---|
> | full fan | 30.04 ms | 11.79 ms | — | **41.83 ms** | **23.9** |
> | N_suff = 92 | 30.04 ms | — | 9.08 ms | **39.13 ms** | **25.6** |
>
> The decoder is only **28.2 %** of the frame, and is itself fixed-cost dominated (8 → 256
> candidates, a 32× candidate increase, costs only **2.18×**: 5.41 → 11.79 ms). On base the saving is
> **0.86 ms = 3.3 %** (25.77 → 24.92 ms).
>
> ⇒ **Fan width is not where REF-C's latency lives. The encoder is.** Take the 2.78× — it is free and
> bit-exact — but do not budget a Thor real-time plan against it.

> ### 0.4 ⭐ The width that *would* have been the lever is 128, and the evidence is a RETRAINED ladder
>
> Paired, on the same 881 windows (⚠️ **confounded with decoder capacity** — small/base/XL differ in
> size too; stated, not buried):
>
> | step | SELECTED ADE | ORACLE-in-fan |
> |---|---|---|
> | 64 → 128 | **+0.0533** [+0.0167, +0.0925] **separated** | **+0.0299** [+0.0204, +0.0398] **separated** |
> | 128 → 256 | **+0.0013** [−0.0281, +0.0316] **NOT separated** | **+0.0275** [+0.0142, +0.0405] **separated** |
>
> ⇒ From 128 to 256 the fan gets **measurably better** and the shipped trajectory does **not**. That
> is **O1, SELECTOR-BOUND** — on the retrained ladder, which is the one that can actually be shipped.
> **The second half of a 256-anchor vocabulary is oracle-only.**
>
> **Compounded, and this is the recommendation:** train at **128** (0.4728, not separated from XL's
> 0.4714) and decode **46** of them ⇒ **5.6× fewer decodes than the 256-anchor arm**, for a selected
> ADE that is statistically indistinguishable from it.

---

## 1. What was run, and where

| | |
|---|---|
| arms | `refc-xl-30k` (256 anchors), `refc-base-30k` (128), `refc-small-30k` (64), all step **29,999** |
| banks | `taniteval/results/fan_refc-{base,xl}-30k.pt`; `…/2026-07-22-refc-small-30k/fan_refc-small-30k.pt` |
| anchors | `decoder.anchors` buffers pulled from the two checkpoints on Thor → `raw/refc_anchor_vocab.pt` |
| lead block | `…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`, row-aligned, **no re-inference** |
| decode | `nav_mode = follow_constant`, `steps = 2`, window 8 / stride 8 — the banks' own protocol |
| latency host | **Jetson AGX Thor** (`thor6`), GPU idle before and after; ⛔ no training pod touched |

---

## 2. Why a prefix of the bank IS an exact decode at width N (and where that stops)

### 2.1 The nested-FPS-prefix relation — VERIFIED bit-exactly, and extended

`furthest_point_sample` is greedy FPS, so `chosen[:N]` **is** the FPS-N solution for the same pool
and seed. MEASURED on the actual checkpoint buffers:

| relation | result |
|---|---|
| `xl256[:128] == base128` | **bit-exact, maxabs 0.0** ← *not previously verified anywhere* |
| `base128[:64] == xl256[:64] == small64` | **bit-exact, maxabs 0.0** |

The banked note in `refc_anchors_small64.pt` claimed only the 64-rung. The 128-of-256 rung is
established here for the first time, which is what licenses using XL's buffer for the arms whose own
buffer was not pulled (asserted in code, not assumed).

### 2.2 The decoder has no candidate axis — C-flags PASSES

`CrossAttnLayer.forward` (`stack/tanitad/refs/refc.py:1007`) is
`q + cross(norm_q(q), kv, kv)` with `kv` = the **image** conv map, then a per-token FiLM-MLP.
**There is no attention over the candidate axis.** Every graft present on these arms
(`decoder.maneuver_to_anchor`, an `nn.Linear(n_man, N, bias=False)`) is a per-anchor row.

**C-flags** asserted from both checkpoints that the two couplings that *would* break this —
`_goal_along_prior`'s across-fan z-score and `_apply_grafts`'s group-norm clamp — are **absent**:
no `goal_gate` / `goal_dist_gate` / `route_to_anchor` / `cons_gate` parameter, no `seam_clamp` in
either `config.json`, `grounded_selector: false`. ⇒ **§2.2 holds; nothing is withdrawn.**

### 2.3 ⛔ What the prefix identity does NOT license — and the measurement that proves it matters

It is a statement about **inference at width N from these weights**. It is *not* a statement about a
model **retrained** at width N — and the difference is large, not academic:

* `refc-xl-30k` **truncated** to 128 → **0.5888**
* `refc-base-30k`, **trained** at 128 → **0.4728**

A 0.116 m gap between the same width reached two ways. Any claim that reads a truncation ladder as a
capacity ladder is wrong by about that much.

---

## 3. P1 — the two saturation curves (prereg §4)

**Saturation criterion, fixed in advance:** the smallest N whose paired bootstrap vs N_max is **not
separated** *and* `|Δ| ≤ 0.02 m` (= `free_win_m`, transcribed from
`PREREG_D-SEL_REFC_SELECTION_SURFACE.md` §6.3, not invented here).

### 3.1 `refc-xl-30k` (256), FPS prefix, selection rule UNCHANGED at every rung

| N | selected ADE | oracle-in-fan | sel_gap | rank_acc | frac_2×_worse | Δsel vs 256 | Δoracle vs 256 |
|---|---|---|---|---|---|---|---|
| 1 | 12.0153 | 12.0153 | 0.0000 | 1.0000 | 0.0000 | +11.5439 worse | +11.8514 worse |
| 2 | 9.4413 | 9.4267 | 0.0145 | 0.9864 | 0.0000 | +8.9698 worse | +9.2628 worse |
| 4 | 3.7710 | 3.5346 | 0.2364 | 0.8854 | 0.0102 | +3.2996 worse | +3.3706 worse |
| 8 | 2.8294 | 2.2737 | 0.5557 | 0.7741 | 0.0545 | +2.3579 worse | +2.1097 worse |
| 16 | 2.3159 | 1.2257 | 1.0902 | 0.5187 | 0.2577 | +1.8444 worse | +1.0617 worse |
| 32 | 1.3914 | 0.8061 | 0.5854 | 0.5131 | 0.2236 | +0.9200 worse | +0.6421 worse |
| 64 | 0.9213 | 0.4368 | 0.4845 | 0.4427 | 0.3235 | +0.4498 worse | +0.2728 worse |
| 128 | 0.5888 | 0.2624 | 0.3264 | 0.4506 | 0.3644 | +0.1174 worse | +0.0984 worse |
| **256** | **0.4714** | **0.1640** | 0.3075 | 0.3110 | 0.4540 | — | — |

`refc-base-30k` (128) is the same shape: N=64 → selected **0.6685** (Δ **+0.1958** [+0.1301,+0.2710]
worse), oracle **0.2833** (Δ **+0.0919** worse).

**Three-sided, every rung: `worse` at every N < N_max, both curves, both arms. `not separated` occurs
only at N_max, where the delta is 0 by construction.**

### 3.2 A structural caveat the numbers force

The FPS prefix at small N is a set of manifold **extremes** — that is what furthest-point sampling
seeds with — so N = 1 scores **12.0 m**. This makes the small-N end of the ladder a statement about
*extremes*, not about small fans in general. It is why the ladder's real information is in
**N ≥ 32**, and why the controls below matter.

### 3.3 Controls — all PASS, with their **direction** predicates

| control | result | verdict |
|---|---|---|
| **C-flags** | no forbidden param, no `seam_clamp`, both arms | **PASS** — §2.2 stands |
| **C-full-rung** | XL selected 0.471439 vs published 0.4714 (**dev 3.94e-5**), oracle 0.163950 vs 0.1640 (**4.97e-5**); base 0.472773 / 0.191423 (**2.71e-5 / 2.31e-5**). Tolerance 1.5e-4. **Two-sided** — beating published would fail too; `red_flag_fired: false` on both | **PASS** |
| **C-monotone-oracle** | nested prefixes ⇒ oracle must not increase; no increase at any rung | **PASS** (a check on my own pipeline that could genuinely have fired) |
| **C-random-subset** (24 seeds) | FPS prefix ≤ random on oracle at **8/9** rungs (XL) and **7/8** (base); threshold was ≥ 7/9 | **PASS** — the "principled subset" premise **holds**, so §3's curves are about *fan width*, not about *which* subset |
| **C-stride** | reported, no verdict rides on it; mostly worse than the prefix at small N, mixed mid-ladder | reported |
| **C-band-fidelity** | anchor-vs-decoded band agreement **0.9657** (XL) / **0.9606** (base); wrongly-admitted **2.21 % / 2.93 %**, trigger was 20 % | **PASS** — P2b's saving is real, not illusory |

⛔ **`C-shuffled` was deliberately not used** as a subset control: permute-then-argmax is a uniform
random pick for any score, so it is vacuous by construction here.

⚠️ **The "oracle" here is a genuine per-window upper bound** — `min` over the same nested set the
selector picks from — so the failure mode where an "oracle" is beaten by arms it should bound
(a per-fold AP maximiser against a pooled-AP headline) cannot arise: there is no fold aggregation.

---

## 4. P2 — reallocating capacity into the reachable set (prereg §6): **R1 fires**

⛔ The clamp's ADE-inertness at full width is already MEASURED and is **not** re-reported as a
finding. What follows is what its freed budget buys.

### 4.1 Selection containment — the measurement the whole claim rests on

| | `refc-base-30k` | `refc-xl-30k` | `refc-small-30k` |
|---|---|---|---|
| full-fan argmax inside the **DECODED** band | **1.0000** | **1.0000** | **1.0000** |
| full-fan argmax inside the **ANCHOR** band | **1.0000** (n_fail 0) | **1.0000** (n_fail 0) | **1.0000** (n_fail 0) |
| anchor-band survivors / window | med 39, mean 36.0, min 11, max 51 | med 83, mean 74.0, min 24, max **102** | med 19, mean 17.3, min 6, max 25 |
| FPS rank of the winner among survivors | med 21, p95 38, **max 46** | med 38, p95 82, **max 92** | med 9, p95 21, **max 23** |

**N_suff is the max of that last row** — the worst window's winner rank. Note that XL's worst window
has **102 survivors but N_suff = 92**: the fixed policy does *not* always contain every survivor,
and works because the *winner* is always within 92. That is the empirical half of §0.2's caveat, and
`selection_bit_identical_to_full_fan: true` in `raw/reachable_budget.json` is the check that it
holds on all 881 — verified, not argued. The **median** window's winner sits at rank **38 of 256**.

### 4.2 The budget ladder, paired against the FULL fan (`refc-xl-30k`)

| decodes | saving | selected ADE | Δ vs full fan | verdict | selection identical |
|---|---|---|---|---|---|
| 16 | 16.0× | 1.0302 | +0.5587 [+0.4238, +0.7029] | worse | 0.1748 |
| 32 | 8.0× | 0.6810 | +0.2096 [+0.1414, +0.2972] | worse | 0.3927 |
| 48 | 5.33× | 0.5437 | +0.0722 [+0.0394, +0.1060] | worse | 0.6311 |
| 64 | 4.0× | 0.4971 | +0.0256 [+0.0067, +0.0476] | worse | 0.7923 |
| **92** | **2.78×** | **0.4714** | **+0.0000 [0, 0]** | **not separated** | **1.0000** |
| 128 | 2.0× | 0.4714 | +0.0000 | not separated | 1.0000 |

`refc-base-30k`: N=32 (4.0×) → 0.5480, Δ **+0.0752** [+0.0322, +0.1319] worse; **N=46 (2.78×) → 0.4728, Δ exactly 0, selection identical**.

⚠️ **The 0.0000 rows are `degenerate` in `taniteval.ci`'s own sense** — the arms are *identical*, so
`separated=False` there is arithmetic, not evidence. That is the correct reading: it is not a
statistical tie, it is the **same trajectory**.

### 4.3 P2a vs P2b — information or compute

**P2a** ranks survivors of the **decoded** band; **P2b** ranks survivors of the **anchor** band.
They become **identical at and above N_suff** — where both subsets contain every survivor — and
differ only below it, and only slightly on XL: at N = 64, P2a **0.4984** vs P2b **0.4971**. On base
the gap below N_suff is larger (N = 32: P2a **0.5168** vs P2b **0.5480**), because base's smaller
survivor set makes the ~4 % anchor-vs-decoded band disagreement bite sooner.

⛔ **Only P2b is a compute claim.** The anchor band is evaluable before any decode; **P2a needs the
decode it is supposed to save** and is reported purely to show the two bands are interchangeable at
the operating point — which C-band-fidelity independently confirms (agreement 0.9657 XL / 0.9606
base, wrongly-admitted 2.21 % / 2.93 % against a 20 % trigger).

### 4.4 ρ hygiene — binding, and it bites here exactly as warned

| | ρ(shipped logits, −ADE), full candidate axis | **restricted to reachable survivors** | selection ADE beside it |
|---|---|---|---|
| `refc-xl-30k` | **0.9071** [0.8966, 0.9172] | **0.6125** [0.5903, 0.6353] | 0.4714 |
| `refc-base-30k` | **0.8838** [0.8642, 0.9022] | **0.5272** [0.4960, 0.5588] | 0.4728 |

⛔ The full-axis ρ ≈ 0.9 is **not** evidence of a good selector — it is mostly the score correctly
ranking the ~71 % of candidates that could never be chosen. Restricted to the candidates the
selector actually competes among it falls by **0.29–0.36**, while the selection ADE stays 0.4714
against an oracle of 0.1640. **Never quote the full-axis number.**

---

## 5. P4 — FOUR METRIC FAMILIES, per family, never pooled

⛔ An ADE horizon sweep is **one row of four**. `Δt = 0.5 s`, derived by `four_families.infer_dt`
from `wp_steps = [5,10,15,20]` — never hard-coded (a hard-coded 0.1 s inflates speed 5× and accel
25×, R-2026-08-03-c).

### 5.1 At N_suff and above — every family is **exactly** unchanged

Because the selection index is identical on 881/881 windows, every family is identical by
construction: paired Δ **exactly 0.0** on `speed_abs_err`, `along_abs_err`, `cross_abs_err`,
`heading_abs_err`, `curvature_abs_err`, `yaw_rate_abs_err`, and on distance-keeping. **The 2.78×
saving is family-neutral because it is an identity, not a trade.**

### 5.2 ⭐ Below N_suff the cost is almost purely LONGITUDINAL — the axis ADE hides

`refc-xl-30k`, paired vs the full fan:

| N (saving) | LONG speed_abs (m/s) | LONG along_abs (m) | LAT cross_abs (m) | LAT heading (°) | LAT yaw-rate (°/s) |
|---|---|---|---|---|---|
| 46 (5.6×) | **+0.0726** [+0.0367,+0.1111] **worse** | **+0.0735** [+0.0395,+0.1089] **worse** | **−0.0004** [−0.0047,+0.0037] **not separated** | +0.0407 worse | +0.0604 worse |
| 64 (4.0×) | **+0.0249** [+0.0042,+0.0490] **worse** | **+0.0222** [+0.0031,+0.0447] **worse** | +0.0030 [−0.0008,+0.0074] **not separated** | +0.0262 worse | +0.0294 worse |
| 32 (8.0×) | **+0.2013** [+0.1339,+0.2859] **worse** | **+0.1982** worse | +0.0311 [+0.0154,+0.0503] worse | +0.2874 worse | +0.4257 worse |

`refc-base-30k` at N=32 (4.0×): speed **+0.0822** [+0.0351,+0.1446] **worse**, cross-track **+0.0018**
[−0.0029,+0.0059] **not separated**, heading **+0.0044** [−0.0332,+0.0347] **not separated**.

⇒ **Fan width buys longitudinal resolution, not lateral.** Cutting below N_suff degrades
target-speed and along-track while cross-track is *statistically untouched* — consistent with the
programme's standing measurement that 87.6–89.9 % of the selection gap is longitudinal, and a clean
demonstration of why an ADE-only table would have been unreadable here.

⭐ **Independently corroborated.** E-EXP-1 (`7d8ed27`, landed while this pass ran) measures a
matched-DoF **lateral** control **9.9× / 11.7× smaller** than its along-path effect, using a
completely different instrument (radial refinement off the candidate set). Two instruments, one
axis. See §8 item 6 for the integration.

⚠️ **Signed components carry no verdict.** `speed_signed_err` moves to **−0.068 m/s** at N=46 against
**+0.021** at full fan — the under-budget fan **under-predicts speed**. A negative paired delta on a
*bias* is a direction, not an improvement, and the JSON labels those rows `n/a` rather than "better".

### 5.3 LONGITUDINAL — distance-keeping (MEASURED, no re-inference)

Lead block: **881 rows**, row-aligned with the banks; **270 LEAD**, 551 NO_LEAD, **60 NO_LABEL**.
At the full fan **242** windows carry a scorable lead (the *predicted* path decides corridor
membership, so the denominator moves slightly with the arm: 241–248 across the ladder).

| N | n with lead | mean min-headway (m) | mean min time-gap (s) | n(time-gap) | mean min-TTC (s) | n closing |
|---|---|---|---|---|---|---|
| 32 | 244 | 28.798 | 3.116 | 222 | 25.114 | 90 |
| 46 | 241 | 28.593 | 3.079 | 219 | 24.641 | 91 |
| **92 … 256** | **242** | **28.623** | **3.089** | **220** | **24.429** | **98** |

⚠️ **TTC is heavily censored** — only **98 of 242** windows ever close on the lead; the rest are
capped. Quote `n_closing` beside the mean or do not quote the mean.

**Stratified by speed** (⛔ never pooled), full fan, `min_stratum_n = 30`:

| band (m/s) | windows | with lead | status | min-headway (m) | min time-gap (s) | min-TTC (s) |
|---|---|---|---|---|---|---|
| 0–1 | 50 | **25** | **UNPOWERED** (< 30) | — | — | — |
| 1–3 | 52 | **12** | **UNPOWERED** (< 30) | — | — | — |
| 3–6 | 127 | 46 | OK | 15.02 [10.75, 25.24] | 3.15 [2.42, 5.10] | 21.02 [16.13, 27.34] |
| 6–10 | 203 | 65 | OK | 27.21 [21.09, 34.11] | 3.51 [2.85, 4.43] | 21.71 [15.93, 25.74] |
| 10–15 | 179 | **8** | **UNPOWERED** (< 30) | — | — | — |
| **15+** | 270 | **86** | **OK** | 43.30 [29.91, 60.96] | **1.82 [1.35, 2.33]** | 26.94 [18.71, 29.84] |

⚠️ **CORRECTION TO AN INHERITED NUMBER.** My brief carried *"the 15+ band is UNPOWERED (n = 2)"*.
**MEASURED here on the canonical 881 windows: the 15+ band has n = 86 lead-bearing windows and is
POWERED.** The band that is starved at the fast end is **10–15 (n = 8)**, not 15+. Whoever owns the
`n = 2` figure should re-derive it — this pass cannot tell whether it came from a different arm, a
different corridor width, or a stale stratification, and I am not asserting which. What I *can*
assert is the number above and its provenance
(`raw/budget_four_families_refc-xl-30k.json → …/by_speed_band`).
⚠️ The 0–1 m/s band remains **UNPOWERED (25)**, so the brief's warning that the crawling regime
cannot discriminate stands.

⭐ The 15+ band is the one that should worry a reviewer: **1.82 s** mean minimum time-gap at
motorway speed, against 3.15–3.51 s in the mid bands. That is a LONGITUDINAL safety read, it is
powered, and it is unchanged by fan width.

**Distance-keeping does not separate across the budget ladder.** Paired, budget-46 vs the full fan:
min-headway **+0.0032 m** [−0.1028, +0.1025] **not separated** (n = 241); min time-gap **+0.0039 s**
[−0.0111, +0.0197] **not separated** (n = 219); min-TTC **+0.2347 s** **not separated**. Headway,
time-gap and TTC are set by whether a lead exists and how far it is, not by the sub-metre trajectory
differences a narrower fan produces.

### 5.4 TACTICAL — half measured, half a work item

* **Goal/anchor SELECTION half — MEASURED**, and it is the half fan width acts on: `rank_acc`
  0.3110 (XL, full fan) falling from 1.0000 at N = 1; `sel_gap` 0.3075; `frac_sel_2×_worse` 0.4540.
  ⚠️ `frac_sel_2×_worse` **rises** monotonically with width (0.0 → 0.4540): as the fan grows the
  selector gets absolutely better and *relatively* worse against its own best candidate.
* **Manoeuvre-DECISION half — UNAVAILABLE, n = 0.** `refc_rerank.dump` stores no decoded manoeuvre
  logits. **A WORK ITEM, not a pass.**

### 5.5 STRATEGIC — **UNAVAILABLE, n = 0 of 881**

No route/goal label in a fan bank, and the decode ran with `nav_mode='follow_constant'`, so the route
input was never exercised. **A WORK ITEM, not a pass.**

---

## 6. Latency (prereg §7) — warmed, p50/p95, never a first call

Thor, aarch64, torch 2.13.0+cu130, **batch 1**, `steps = 2`, 15 warm-up + 60 timed iterations,
`torch.cuda.synchronize()` around every region. Memory is in-process
`torch.cuda.max_memory_allocated()` only — ⚠️ on Thor `mem_get_info` / `free` / `tegrastats` /
`VmRSS` all lie.

**`refc-xl-30k` — encoder 30.04 / 30.91 ms (p50/p95), N-INDEPENDENT**

| N | decoder p50 | p95 | end-to-end p50 | Hz | decoder share | peak MiB |
|---|---|---|---|---|---|---|
| 32 | 6.21 | 6.25 | 36.25 | 27.59 | 0.171 | 979.5 |
| 64 | 7.58 | 8.01 | 37.62 | 26.58 | 0.202 | 980.2 |
| **92** | **9.08** | **9.49** | **39.13** | **25.56** | 0.232 | 980.9 |
| 128 | 9.46 | 9.85 | 39.50 | 25.31 | 0.240 | 981.7 |
| **256** | **11.79** | **12.09** | **41.83** | **23.91** | 0.282 | 985.2 |

**`refc-base-30k` — encoder 20.81 / 21.03 ms**; decoder 128 → **4.96** ms, 46 → **4.11** ms;
end-to-end **25.77 → 24.92 ms** (38.8 → 40.1 Hz), i.e. **3.3 %**.

Three things the table says that a headline "2.78× cheaper" would hide:

1. **The decoder is 16–28 % of the frame.** The end-to-end win from N_suff is **6.5 % (XL)** and
   **3.3 % (base)**.
2. **The decoder is fixed-cost dominated.** 8 → 256 candidates is 32× the work for **2.18×** the
   time; below ~32 candidates the curve is flat to noise (base reads 4.00 ms at N=8 and 3.77 at
   N=16 — kernel-selection noise, not a saving).
3. **Peak memory barely moves** (985.2 → 980.9 MiB, −0.4 %). Fan width is not a memory lever either.

⇒ If Thor latency is the goal, **the encoder is the target, not the fan.**

---

## 7. What I did NOT do — plainly

1. ⛔ **No arm was retrained.** Every number is inference on banked fans or a timed forward. The
   "train at 128" recommendation in §0.4 rests on a **capacity-confounded** cross-arm comparison, not
   on a width-only retrain. The clean experiment is a `refc-xl` re-trained at 128 anchors.
2. ⛔ **The truncation ladder is not a retrained ladder**, and §2.3 measures the gap (0.116 m) rather
   than assuming it away.
3. **The variable-width policy is measured for ACCURACY but not for LATENCY.** Its selection is
   bit-identical and its mean budget is 74.0 / 36.0 / 17.3 (§0.2), but every timed rung on Thor used
   a **static** tensor shape. A ragged per-window fan may not realise the 3.46× on a real engine —
   dynamic shapes cost kernel re-selection, and TensorRT would need a shape profile. **Untested.**
4. **No end-to-end closed-loop run.** All ADE is open-loop on the 881 canonical windows; this
   programme has measured open-loop 0.45 m → closed-loop 1.69 m before, so nothing here should be
   read as a closed-loop claim.
5. **The band's own parameters were not tuned** (`accel_max = 2.5`, `horizon_s = 2.0`, taken from
   `flagship_v15` unchanged). A wider band would raise N_suff; a narrower one would risk the
   containment property that makes the saving bit-exact. **Untested.**
6. **TACTICAL manoeuvre-decision and STRATEGIC are UNAVAILABLE (n = 0)**, with reasons in §5.4/§5.5.
   They are work items.
7. **Only the shipped t=0 classifier ranker was swept.** The augmented banks carry `refined_logits`,
   `cons_score`, `cons_deploy`, `emitted_logits`; the interaction of fan width with those rankers is
   not measured. (⚠️ It matters: E-SEL-0 measured the refined ranker **0.84–0.92 m worse**, so a
   different ranker could well have a different N_suff — containment is a property of *the score*.)
8. **`refc-small-30k` has no timed rung** — only base and XL were timed on Thor.

---

## 8. ESCALATIONS — things that need a decision, not a README

1. ⭐ **SHIP THE ANCHOR-LEVEL BAND AS A DECODE FILTER.** It is a ~15-line change in
   `AnchoredDiffusionDecoder.forward` (evaluate `reachability_mask` on `self.anchors` against `v_ms`
   *before* `_decode`, decode the survivors, scatter back), it is **bit-exact** on all three arms,
   and it is **2.78× fewer decodes at a fixed shape / 3.46–3.70× variable**. It needs an owner and a
   PR. Ship the **fixed `N_suff`** first — same static shape, no engine work — and treat the
   variable policy as a follow-up once §7.3's dynamic-shape question is answered.
   ⚠️ It must carry the containment assertion as a **runtime guard**, because containment is a
   property of the **score**, not a theorem — see §7.7. If a future ranker (S1/S3/S5/S6) puts its
   argmax outside the band, the guard must fire rather than silently ship a different trajectory.
2. ⚠️ **RE-SCOPE THE "73.8 % UNREACHABLE ⇒ 4× TOO MANY HYPOTHESES" FRAMING.** The two readings in
   the brief were *compute* or *accuracy*. The measured answer is **compute, at 2.78× not 4×, and
   only via the band — never by truncation**, whose cost is separated-worse at every rung.
3. ⚠️ **THE 256-ANCHOR ARM'S SECOND HALF IS ORACLE-ONLY** (§0.4). Before any future arm is trained
   at 256, someone should own the decision to train at 128 instead. The discriminating experiment is
   a **width-only** retrain of one arm — the current evidence is capacity-confounded.
4. ⚠️ **DO NOT BUDGET THOR REAL-TIME AGAINST FAN WIDTH.** 23.9 Hz at 256 → 25.6 Hz at 92. The
   encoder is 72 % of the frame. Any Thor latency plan should target the encoder.
5. ⚠️ **`ρ = 0.907` ON THE FULL CANDIDATE AXIS IS IN CIRCULATION AS A SELECTOR QUALITY NUMBER.**
   Restricted to reachable survivors it is **0.61 / 0.53**. §4.4 is the correction; the trap has now
   caught three streams.

6. ⭐ **INTEGRATE WITH E-EXP-1 (`7d8ed27`, landed mid-pass) — the two results compose into one
   recommendation, and neither says it alone.**
   E-EXP-1 measures that **leaving** the fan (a radial refinement of the *already-selected*
   trajectory) is worth **+0.159 / +0.165 m**, i.e. **53.7–56.5 %** of the oracle gap, against K1's
   ≤ 8.4 % ceiling on *re-ranking a fixed fan*. This pass measures that **64–71 % of the fan is dead
   weight** and that shrinking it is **free and bit-exact**.
   ⇒ **The reallocation that pays is OUTSIDE the candidate set, not inside it.** My P2 asked whether
   N reachable slots beat N mixed slots; the answer (R1, §4) is that the band frees budget but the
   *extra candidates* were never the lever. E-EXP-1 says where the budget should go. **The joint
   recommendation is: cut the fan with the anchor band, then spend the freed decoder time on an
   off-fan refinement operator.** Someone should own that as one design, not two documents.
   ⭐ **Convergent evidence, from two independent instruments:** E-EXP-1's matched-DoF lateral
   control is **9.9× / 11.7× smaller** than its along-path effect; §5.2 here finds that cutting the
   fan degrades speed and along-track **separated-worse** while cross-track is **not separated at
   all**. Both instruments put the mass on the along-path axis.
   ⚠️ **One caution before composing them:** E-EXP-1 refines the *selected* trajectory, so its gain
   is measured downstream of exactly the selection this pass leaves bit-unchanged. That makes the
   two composable in principle — but the composition is **UNMEASURED**, and the cheap check is to
   re-run E-EXP-1's operator on the N_suff-budget output and confirm the delta is unchanged (it
   should be **exactly** unchanged, since the input trajectory is bit-identical). That is a
   0-GPU, ~10-minute confirmation and it has not been done.

---

## 9. DELIVERABLE MANIFEST

⚠️ **Nothing here lives in only one place.** Every file is in the repo working tree, **staged**
(verified with `git ls-files --cached`, not with an exit code), and mirrored off-Drive at
`C:/Users/Admin/tanitad-mirror/2026-08-04-fan-width/`.

| artifact | where | what it is |
|---|---|---|
| `PREREG_FAN_WIDTH.md` | repo, staged | pre-registration, blob `1bffa9db…` |
| `FAN_WIDTH.md` | repo, staged | this document |
| `code/fan_width_sweep.py` | repo, staged | P1 ladder + C-flags-adjacent controls + P2a/P2b + ρ hygiene |
| `code/reachable_budget.py` | repo, staged | N_suff, selection containment, budget ladder, retrained ladder |
| `code/budget_four_families.py` | repo, staged | four families per budget rung + distance-keeping |
| `code/thor_fan_latency.py` | repo, staged · also `tanitad-thor:/home/nvidia/thor_fan_latency.py` | Thor latency harness |
| `raw/fan_width_refc-{base,xl}-30k.json` | repo, staged | P1 curves, controls, families |
| `raw/reachable_budget.json` | repo, staged | the decisive budget result |
| `raw/budget_four_families_refc-{base,xl}-30k.json` | repo, staged | four-family panels |
| `raw/variable_width_policy.json` | repo, staged | the variable-width policy |
| `raw/fan_latency_{xl,base}.json` | repo, staged · also `tanitad-thor:/home/nvidia/` | Thor p50/p95 |
| `raw/refc_anchor_vocab.pt` | repo, staged · also `tanitad-thor:/home/nvidia/refc_anchor_vocab.pt` | the two `decoder.anchors` buffers — **previously existed only inside 3 GB checkpoints on one host** |

⛔ **Staged, never committed, never pushed.** The index also contains sibling streams' work; whoever
commits must read `git diff --cached --name-only` first (CLAUDE.md's git-hygiene rule).

---

## 10. Provenance

| | |
|---|---|
| prereg blob (staged == worktree, re-verified) | `1bffa9db6a6047325dceff1ef787d67ab2fd5152` |
| bank sha256 (XL) | recorded in `raw/fan_width_refc-xl-30k.json → /bank/sha256` |
| bank sha256 (base) | recorded in `raw/fan_width_refc-base-30k.json → /bank/sha256` |
| anchors | `raw/refc_anchor_vocab.pt`, pulled from `decoder.anchors` at step 29,999 |
| full suite | **2093 passed, 12 skipped, 2 xfailed** (729 s) — `raw/stack_pytest.txt`. ⚠️ The collected count is drifting live as sibling streams land tests: **2043** (my brief) -> **2068** (sibling banked 09:07) -> **2093** (this run). This stream added NO file under `stack/` or `taniteval/`, so the suite result is independent of it by construction. |

🔒 **Parity untouched.** Nothing re-selected an episode; every number is on the canonical 881
windows / 40 episodes. 🔒 No token was read, printed or written. ⛔ Nothing pushed to HF. ⛔ Nothing
committed — deliverables are **staged only**.
