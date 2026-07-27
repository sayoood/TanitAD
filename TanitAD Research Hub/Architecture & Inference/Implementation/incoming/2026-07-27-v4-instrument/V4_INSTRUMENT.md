# V4 INSTRUMENT — the real-vs-imagined decode gap MEASURED on flagship-v4, the `v0` shortcut proven surgically, and v4's fan geometry dumped at last

**Stream:** Architecture & Inference — implementation. **Date:** 2026-07-27 (Europe/Berlin; pods are UTC).
**Host:** dev box + **pod3** (A40, MEASURED idle 0 MiB / 0 % before launch, restored to its original
file state afterwards). ⛔ **pod1 (training), pod2 (arm panel) and the eval pod (T3 + λ/τ) were never
contacted.** Total pod time: **830.7 s = 13.8 min** across five runs (sum of the `elapsed_s` in the
five artifact JSONs: 179.1 + 77.8 + 135.6 + 59.6 + 378.6), plus a 15 s checkpoint pull.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID, no frame, no raw content appears in this folder.

**Pre-registration:** `PRE_REGISTRATION.md` in this folder, written **03:06 Europe/Berlin, before the
first v4 grounding number existed** (both outcomes, the `v0` predictions, the clamp verdict rule, and
the five validation gates are all fixed there and are quoted here unchanged).

---

# 0. VERDICT

> ## 🔴 **THE PREMISE OF THIS BRIEF WAS WRONG IN A WAY THAT MAKES THE FIX FREE: v4 ALREADY CARRIES THE INSTRUMENT AND HAS BEEN TRAINING IT FOR 30 000 STEPS. THE GAP IS ONE LINE OF `log`, NOT AN ABSENT MEASUREMENT.**
> `train_flagship_v4.py:95` calls `flagship_loss(world, grounding, …)`; `flagship_losses.py:359` calls
> `grounding_losses(...)` and merges its log verbatim (`**g_log`, `:425`); `train_flagship_v4.py:861`
> puts `grounding.parameters()` in the optimizer. **v4 computes all six `g_*` numbers every step and
> trains the heads.** `train_flagship_v4.py:159` forwards exactly ONE of them (`g_op_fwd_ade_m`) into
> the JSONL row — and never the paired real-side partner. `MEASURED`, tier **DECISION-GRADE**.

> ## ⭐⭐ **AND THE ANSWER IS OUTCOME A AT ALL THREE LEVELS: v4 SHOWS THE GAP. `real ÷ imagined` = 20.14× / 21.93× / 14.72× (op/tac/str), against v1's 36.98× / 44.12× / 32.62× ON THE IDENTICAL 967 WINDOWS.**
> Ratio-of-ratios v4 ÷ v1 = **0.545 / 0.497 / 0.451** — inside the pre-registered `[1/3, 3]` band that
> was fixed before the run. **⇒ the finding GENERALISES and the registry row widens beyond the v1
> family.** v4's gap is real, present at every level, and roughly **half** v1's.

> ## ⭐⭐⭐ **THE `v0` SHORTCUT IS NOW PROVEN SURGICALLY, WITHIN ONE MODEL, AND IT IS ~6–14× LARGER THAN THE CROSS-ARM ESTIMATE. ZEROING THE `v0` ACTION CHANNEL DEGRADES THE IMAGINED DECODE ×93.7 (v1) AND ×39.4 (v4) WHILE LEAVING THE PERCEIVED DECODE *BIT-EXACTLY* UNCHANGED (max |Δ| = 0.0 on every level of both arms).**
> X4's cross-arm ablation put this at ×6.87. That comparison had two different encoders and 30 k steps
> of different training in it. **The surgical version — same weights, same windows, same forward pass —
> is ×39–94.** This is **X2**, the clean proof the program has owed since E4′, and it is now paid.

> ## ⛔ **AND THE DISCRIMINATOR SAYS IT IS `v0` SPECIFICALLY, NOT "ACTIONS": ZEROING BOTH CAN ACTION CHANNELS AND KEEPING `v0` COSTS ONLY ×1.32 (v1) / ×1.07 (v4).**
> A **71× / 37× separation** between "remove `v0`" and "remove the steering and throttle commands".
> **The imagined metric decode is very nearly a pure integrator of the scalar it is handed, and is
> almost indifferent to what the ego is actually commanded to do.**

> ## ⭐ **GAP 2: v4's FAN GEOMETRY IS DUMPED (967 windows × 256 candidates, first time in the program) AND THE CLAMP IS EXACTLY FREE ON IT — 75.29 % of candidates removed, ADE-oracle 100 %, pick moved on 0 of 967 windows, paired Δ 0.000000 [0.000000, 0.000000], miss@2m unchanged, 4.05× cheaper.**
> Independently cross-checked: our fresh dump's max last-waypoint along-track is **100.14 m**
> (**180.31 km/h**) against the separately rescued `fan_last_along_v4.pt`'s **100.57 m / 181.03 km/h** —
> a **0.43 %** agreement across a different window set and a different v4 checkpoint.

> ## 🔴 **BUT THE DEFAULT STAYS OFF, AND NOT FOR THE REASON ANYONE EXPECTED. THE PRE-REGISTERED BOTH-DIRECTIONS GUARD FIRED: ON v4 A *TIGHTER* BAND MAKES ADE *BETTER*, SEPARATED (−0.0818 [−0.1498, −0.0265] at `accel_max` 0.5; −0.1340 [−0.2329, −0.0454] at 0.2; miss@2m 0.0641 → 0.0331).**
> FIX 2's own committed test says it in one line: *"if a tight band ever **helped**, the clamp would be
> a tuned selector rather than physics."* On REF-C-XL it did not. **On v4 it does.** ⇒ The zero at 2.5
> is real, but it is the zero of a knob sitting on a live ADE gradient, not the zero of an inert
> physical bound. **I did not relax the test, and I do not recommend flipping the default.** What the
> data DOES support is a *separate, pre-registered* experiment worth more than the flip: **a free
> −0.08…−0.13 m of ADE and a halved miss@2m from a purely physical prior.**

> ## ⭐ **AND THE "BLAMELESS VOCABULARY" FRAMING DOES NOT SURVIVE CONTACT WITH v4's FAN. 94.36 % of the candidates the clamp removes have an ANCHOR that was ALREADY unreachable; the offset head moves a candidate's implied mean speed by only 1.67 m/s on average (p99 5.80, max 7.26). On v4 the clamp is removing VOCABULARY, not offset-head excess** — the opposite of the REF-C-XL reading, and the lever it points at is a **state-conditioned anchor set** (CoverNet), not a refinement bound.

---

# 1. THE CORRECTION THE WHOLE JOB RESTS ON — and why it is a textbook Operating-Standard-rule-2 catch

`MEASURED`, tier **DECISION-GRADE**. Reproduced before anything new was quoted.

X4 (`…/2026-07-27-x1-latent-metric-probe/X1_LATENT_METRIC.md` §3.1) reported that all three committed
v4 logs carry **no `g_*` key at all**, and inferred **"v4 does not carry the instrument … the
real-vs-imagined decode gap is UNMEASURED on v4, and no committed v4 artifact can measure it."**

**The log-level fact reproduces exactly** (`scripts/` re-derived it, not inherited):

| log | rows | steps | `g_*` keys |
|---|---:|---|---|
| `taniteval/results/trainlogs/flagship-v4.1-10k_train_log.jsonl` | 276 | 0 … 12 500 | **0** |
| `…/2026-07-26-v4-restart-lever/raw/v4fs_train_log.jsonl` | 661 | 0 … **29 999** | **0** |

**The inference does not.** A second probe — following the call instead of grepping the file — finds
the instrument alive inside v4's loop:

| where | what it does |
|---|---|
| `stack/scripts/train_flagship_v4.py:95` | `wm_total, wm_log, _ = flagship_loss(world, **grounding**, batch, …)` |
| `stack/tanitad/train/flagship_losses.py:359` | `loss_ground, g_parts, g_log = grounding_losses(...)` |
| `stack/tanitad/train/flagship_losses.py:425` | returns `{…, **g_log}` — all six numbers, verbatim |
| `stack/scripts/train_flagship_v4.py:220 / 291 / 825` | `grounding = build_grounding(world.state_dim)` |
| `stack/scripts/train_flagship_v4.py:861` | `head_group = list(head.parameters()) + list(grounding.parameters())` — **trained, at `lr_head`** |
| `stack/scripts/train_flagship_v4.py:159` | `**{k: v for k, v in wm_log.items() if k in ("g_op_fwd_ade_m",)}` — **one key survives to disk** |

> ### ⇒ **v4 has computed `g_{op,tac,str}_{mid_de_m,fwd_ade_m}` on every one of its 30 000 steps and has been TRAINING the heads that produce them. What was missing is a `log` filter, not a measurement.** The string-level claim *"`train_flagship_v4.py` contains no reference to `grounding_losses`"* is TRUE and is exactly why the absence looked total: the reference is one call-frame away. **This is Operating Standard rule 2 — "absence found at ONE location is not absence" — firing inside a report that was itself written to honour that rule.**

**Two consequences that change what the fix costs:**

1. **E5′ option (b) is not "~5 lines", it is ~1 line**, and it is already half-written: the comment at
   `train_flagship_v4.py:145-158` documents a previous fix to this very line that was *silently inert*
   because it filtered on the unprefixed name. Widening the tuple to
   `("g_op_mid_de_m", "g_op_fwd_ade_m", "g_tac_mid_de_m", "g_tac_fwd_ade_m", "g_str_mid_de_m",
   "g_str_fwd_ade_m")` makes every future v4 run measurable at zero cost. ⚠️ **Even after that earlier
   fix, only the IMAGINED half is forwarded — the paired REAL partner still never reaches disk, so the
   ratio is still not loggable.** That is the actual open defect.
2. **A v4 checkpoint's `grounding` is a genuine v4-trained instrument, not inherited v1 weights.** On
   the arm used here — **v4-from-scratch** — it is *purely* v4: `train_flagship_v4.py:974` records that
   from-scratch simply does not call `_warmstart_trunk`, so `build_grounding()`'s random init is what
   trained. There is no v1 in this measurement at all.

---

# 2. SETUP, CORPUS AND PARITY — stated, not buried

| | |
|---|---|
| **arm A** | v1 `flagship4b-speedjerk-30k`, step **29999**, `pod3:/workspace/tmp/idm/ckpt.pt` (3,308,556,862 B = byte-count identical to HF `Sayood/tanitad-flagship-4b-speedjerk/ckpt.pt`) |
| **arm B** | **`flagship-v4-fromscratch`**, step **29999**, pulled fresh from gated HF `Sayood/flagship-v4-fromscratch/ckpt.pt` (3,243,109,310 B, 15 s at 216 MB/s) |
| **both** | `model` **and** `grounding` loaded `strict=True`; `state_dim` 2048; `action_dim` **3** (`predictor.act_emb.0.weight` = [768, 3]) — **v4 has the `v0` channel** |
| **corpus** | `pod3:/workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6`, **44 episodes**, whole-split consume of the first N files — **nothing re-selects episodes** |
| **windows** | **967**, window 8, stride 8, `LEVEL_CFG = {op: ((1,2,4), 4), tac: ((8,16), 16), str: ((20,), 20)}` — the identical dict `grounding_losses` is called with |
| **estimator** | paired **episode-cluster bootstrap**, `taniteval/taniteval/ci.py`, **B = 2000**, seed 0, unit = episode cluster (**44**). ⛔ `overlapping_holdout_se` is used nowhere; every central value is the `full_set` mean |

## 2.1 ⚠️ CORPUS PARITY — the honest statement

The 44-episode cache is keyed **`physicalai-val-heldout-79d4e3d2d4c6`**, **not** the parity val key
`physicalai-val-0c5f7dac3b11`. **No number in this document is a leaderboard number and none may enter
`MODEL_REGISTRY.md` as an arm result.** Two facts make it nonetheless the right corpus:

* **It is genuinely held out, MEASURED not assumed.** Intersecting per-episode `poses_sha256` against
  `manifest_train_parity.json` (2376 episodes, the canonical `physicalai-train-e438721ae894`):
  **0 of 44 overlap — 0.0 %.**
* **Both arms trained on that same parity train corpus**, so the split is equally held out for both,
  and every claim here is a **within-window contrast** (v4 vs v1 on identical windows; baseline vs
  ablation inside one model) that a shared corpus property divides out of.

⚠️ **`physicalai-val-f1b378f295ae` — the other frames-carrying cache on pod3 — was REFUSED.** It is a
registered leaky split, and we re-measured it: **62 of 80 episodes, 77.5 %, overlap the parity train
corpus** (`parity.py` records 78.5 %; the two builds differ slightly in size). The canonical 600-episode
parity val exists on pod3 only as a **poses-only view** (`s3parity/views/…`, no `frames_u8`), so the
canonical **881**-window set is not reachable from this host without touching a forbidden pod.

---

# 3. ⭐ GAP 1 — THE v4 REAL-VS-IMAGINED BANDS

`MEASURED` · tier **DECISION-GRADE** (pre-registered with both outcomes, instrument validated in both
directions, paired estimator, matched-arm control) · `raw/grounding_bands_v4fs.json`,
`raw/grounding_bands_v1.json`, `raw/grounding_ci.json`, `raw/grounding_perwindow_*.pt`.

## 3.1 The instrument passed before any band was quoted — and one gate FAILED, which is reported

| # | pre-registered check | rule | v1 | v4fs | |
|---|---|---|---:|---:|---|
| **V1** | fidelity, imagined decode | `op_fwd` ≤ 3 × 0.0304 m | **0.0384** ✅ | 0.0774 ✅ | the anchor |
| **V2** | fidelity, real-pair decode | `op_mid` ≤ 2 × 1.0129 m | **1.4185** ✅ | 1.5580 ✅ | the anchor |
| **V3** | within-episode shuffle **must FAIL** | > 1.5 × matched | 2.8263 = **1.99×** ✅ | 2.2894 = **1.47×** 🔴 **FAILS** | |
| **V3b** | **cross-episode** shuffle **must FAIL** | > 1.5 × matched | 3.5161 = **2.48×** ✅ | 3.2027 = **2.06×** ✅ | |
| **V4** | strict load of `model` **and** `grounding` | no missing/unexpected keys | ✅ | ✅ | |
| **V5** | dt self-check | 0.09 ≤ realised Δp ÷ v ≤ 0.11 s | **0.10063** ✅ | 0.10063 ✅ | |

> ### 🔴 **V3 FAILED ON v4 — at 1.47× against a 1.5× bar — and the diagnosis is the CONTROL, not the head.** The pre-registered shuffle permutes the invdyn head's partner latent **within a ≤32-window chunk of one episode**, where consecutive latents are nearly interchangeable; it is a weak mismatch by construction. **V3b, which takes the partner from a DIFFERENT EPISODE, passes on both arms (v4 2.06×, v1 2.48×).** ⇒ v4's `invdyn['op']` does react to its input; the pre-registered control was under-powered. **Reported as a failure, not re-baselined into a pass.**
>
> ⚠️ **V3b IS NOT PRE-REGISTERED — declared, not glossed.** It was written *after* V3 failed, as the
> diagnosis of that failure, and it is a **control** (a deliberately-wrong input that must degrade), not
> a fitted quantity. It is admissible as a diagnosis and **inadmissible as a substitute for V3**: the
> pre-registered gate stands as FAILED on v4 and everything downstream of the *real-pair* side on v4
> carries that caveat. **Nothing in §4 depends on it** — those are imagined-vs-imagined contrasts.

**v1's committed anchor reproduces off-corpus.** v1's 28–30 k train band is `1.0129 / 0.0304`
(≈**33.3×**). On this fresh held-out corpus the same weights give **`1.4185 / 0.0384` = 36.98×** — the
defect reproduces at **1.11×** of its committed ratio on data it never saw. Every band below sits on
top of that.

## 3.2 🔴 THE RESULT — v4 shows the gap, at roughly half v1's magnitude

All metres, **967 windows / 44 episode clusters, IDENTICAL window sets** (asserted element-by-element
before anything was combined; the script refuses if they differ).

| level | arm | **REAL** `g_*_mid_de_m` [95 % CI] | **IMAGINED** `g_*_fwd_ade_m` [95 % CI] | **ratio** | agg-corrected ×0.5 |
|---|---|---|---|---:|---:|
| **op** | v1 | 1.4185 [1.1691, 1.6691] | 0.03836 [0.0331, 0.0444] | **36.98×** | 18.49× |
| **op** | **v4fs** | 1.5580 [1.2969, 1.8172] | 0.07737 [0.0663, 0.0893] | **20.14×** | 10.07× |
| **tac** | v1 | 6.0505 [5.0169, 7.0943] | 0.13713 [0.1168, 0.1581] | **44.12×** | 22.06× |
| **tac** | **v4fs** | 6.3220 [5.1886, 7.4324] | 0.28832 [0.2431, 0.3436] | **21.93×** | 10.96× |
| **str** | v1 | 6.0992 [4.9068, 7.4239] | 0.18699 [0.1602, 0.2138] | **32.62×** | 16.31× |
| **str** | **v4fs** | 5.8768 [4.6419, 7.2202] | 0.39919 [0.3331, 0.4819] | **14.72×** | 7.36× |

**Against the brief's v1 calibration** (train-log, 28–30 k, `1.0129 / 0.0304` ⇒ 33.3×; and 0–1 k
`2.2486 / 0.9541` ⇒ 2.36×): v4 at 30 k sits at **20.14×** — **8.5× above v1's 0–1 k early band and
0.60× of v1's own late band**. It is unambiguously in the "late, widened" regime, not the early one.

### The pre-registered decision

| level | v1 ratio | v4 ratio | **v4 ÷ v1** | band fixed in advance | **OUTCOME** |
|---|---:|---:|---:|---|---|
| op | 36.9788 | 20.1363 | **0.5445** | [1/3, 3] | **A — comparable** |
| tac | 44.1229 | 21.9274 | **0.4970** | [1/3, 3] | **A — comparable** |
| str | 32.6185 | 14.7219 | **0.4513** | [1/3, 3] | **A — comparable** |

> ### ⇒ **OUTCOME A, unanimously. The real-vs-imagined decode gap is NOT a v1 idiosyncrasy. The registry row widens beyond the v1 family and names v4 with its own measured numbers.** ⛔ The scoping E1′ imposed — *"v1 family only, because v4 cannot be asked"* — is now **superseded**: v4 could always be asked; nobody had asked it.

### Where the two arms actually differ (paired, same windows)

| level | side | v4 − v1 [95 % CI] | separated |
|---|---|---|---|
| op | REAL | +0.1395 [+0.0398, +0.2250] | ✅ |
| op | IMAGINED | **+0.0390 [+0.0286, +0.0505]** | ✅ |
| tac | REAL | +0.2715 [−0.0501, +0.5605] | ❌ |
| tac | IMAGINED | **+0.1512 [+0.1176, +0.1941]** | ✅ |
| str | REAL | −0.2224 [−1.0749, +0.5716] | ❌ |
| str | IMAGINED | **+0.2122 [+0.1615, +0.2821]** | ✅ |

> ### ⭐ **The two arms are statistically indistinguishable on the PERCEIVED decode at `tac` and `str`, and separated on the IMAGINED decode at ALL THREE levels — always with v4 worse.** ⇒ **v4's smaller ratio is not a better-perceiving encoder. It is a WORSE imagination**: v4's imagined decode is 2.02× / 2.10× / 2.13× v1's, while its perceived decode is within noise. Joint planner training left the perception where it was and degraded the imagined metric readout — exactly the coupling the v4 line's "WM yanked off-manifold" hypothesis predicts, now measured on the grounding instrument instead of on the canary.

---

# 4. ⭐⭐ THE `v0`-SHORTCUT MECHANISM — the surgical test (X2), paid at last

`MEASURED` · tier **DECISION-GRADE** · `raw/grounding_bands_*.json` → `imagined_degradation_x_vs_baseline`,
`raw/grounding_ci.json` → `within_arm_paired_ablation_minus_baseline`.

## 4.1 The prediction that was written down first, and how it scored

`PRE_REGISTRATION.md` §2.2 was committed at 03:06 with `action_dim = 3` already measured:

| prediction | outcome |
|---|---|
| REAL side **bit-exactly unchanged** under every `v0` ablation, all levels, both arms | ✅ **EXACT.** `max |Δ|` over the raw per-window arrays = **0.0** for 6 ablations × 3 levels × 2 arms. Falsifier **F-A did not fire** — the ablation is wired right |
| IMAGINED side degrades **≥ 2× at `op`**; "expect ×3…×10" | ✅ direction right, ⛔ **magnitude badly under-predicted: ×93.7 (v1) / ×39.4 (v4)**. I was low by 4–13× |
| worse, never better, at every level | ✅ every level, every ablation, both arms |
| `v0_shuffled` ≥ `v0_zero` (a wrong speed worse than no speed) | ⛔ **WRONG** — see §4.3 |
| monotone in \|Δv0\| | ✅ (`half` ×48.5 < `zero` ×93.7 ≈ `double` ×90.0 on v1) |

**F-B** (< 1.2× ⇒ mechanism refuted on v4) **did not fire**. **F-C** (any improvement) **did not fire**.

⚠️ **TWO OF THE SIX ABLATIONS ARE NOT PRE-REGISTERED — declared.** `PRE_REGISTRATION.md` §2.1 fixed
`v0_zero`, `v0_shuffled`, `v0_half`, `v0_double`. **`v0_gshuffled` and `act_zero` were added after a
4-episode smoke and before the decision-grade run**, because the smoke showed the within-episode
shuffle was a 2.11 m/s perturbation (too weak to be the "wrong speed" control it was meant to be) and
because nothing in the pre-registration separated *`v0`* from *actions in general*. Both are **controls
with a forced direction** — removing information cannot help a metric readout — not tuned quantities,
and neither was selected from a set of candidates. **`act_zero` carries the report's most striking
number (×1.07) and its post-hoc status is stated here rather than buried.** The pre-registered
`v0_zero` result (×93.7 / ×39.4) stands on its own and does not need either of them.

## 4.2 🔴 The numbers

Imagined decode `g_*_fwd_ade_m`, **× vs that arm's own baseline** on the same 967 windows. Every paired
contrast below is **separated** (episode-cluster bootstrap, B = 2000).

| ablation | mean \|Δv0\| | v1 `op` | v1 `tac` | v1 `str` | **v4 `op`** | **v4 `tac`** | **v4 `str`** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v0_zero` | 15.07 m/s | **×93.73** | ×91.25 | ×82.02 | **×39.43** | ×41.86 | ×38.23 |
| `v0_double` | 15.07 m/s | ×90.01 | ×91.86 | ×83.61 | ×47.48 | ×36.19 | ×32.74 |
| `v0_gshuffled` | 10.83 m/s | ×67.18 | ×65.98 | ×59.53 | ×32.71 | ×29.46 | ×27.28 |
| `v0_half` | 7.54 m/s | ×48.51 | ×46.92 | ×42.59 | ×21.44 | ×21.74 | ×19.67 |
| `v0_shuffled` (within-episode) | 2.11 m/s | ×13.78 | ×13.28 | ×12.00 | ×6.35 | ×6.26 | ×5.70 |
| **`act_zero` — CAN channels zeroed, `v0` KEPT** | **0.00** | **×1.32** | ×2.85 | ×3.08 | **×1.07** | ×1.34 | ×1.38 |

*(corpus mean speed 15.07 m/s. `op` paired deltas: v1 `v0_zero` +3.5571 [+2.9455, +4.1900]; v4
`v0_zero` +2.9733 [+2.4563, +3.4630]; v1 `act_zero` +0.0123 [+0.0087, +0.0160]; v4 `act_zero` +0.0054
[+0.0015, +0.0098] — all separated.)*

> ### ⭐⭐ **THE DISCRIMINATOR IS THE FINDING. Deleting BOTH commanded action channels and keeping `v0` costs ×1.32 (v1) and ×1.07 (v4). Deleting `v0` alone and keeping both commands costs ×93.7 and ×39.4. That is a 71× / 37× separation between "what the car is told to do" and "how fast it is going".** The imagined metric trajectory is, to a very good approximation, **`v0` integrated forward** — it barely consults the actions, and X4's cross-arm ×6.87 was an **order-of-magnitude under-estimate** because it compared two differently-trained encoders instead of ablating one.

**Why this is stronger than the cross-arm version:** the cross-arm ablation (v1 vs the no-speed control)
moved the *perceived* decode by +3.0/+3.8/+3.1 %, and that residual is the confound — two encoders,
two 30 k trainings. Here the perceived decode moves by **exactly zero**, by construction: `invdyn` reads
only encoder latents and the encoder never sees `v0`. **The `v0` effect is isolated to machine precision.**

## 4.3 ⚠️ The prediction that was WRONG, and what it teaches

I predicted a *wrong* speed would hurt more than *no* speed. It does not: `v0_zero` (×93.7) is far worse
than `v0_gshuffled` (×67.2), which is far worse than the within-episode `v0_shuffled` (×13.8). The
ordering is **monotone in \|Δv0\|**, not in "wrongness" — 2.11 / 7.54 / 10.83 / 15.07 m/s maps onto
×13.8 / ×48.5 / ×67.2 / ×93.7. **⇒ the readout is not "confused by a bad input"; it is linearly
integrating whatever scalar it is handed, and the error is the integral of the offset.** That is a
sharper statement of M2 than the shortcut framing, and it is the *reason* `act_zero` is inert.

⚠️ **Method note, declared:** `v0_shuffled` permutes within a ≤32-window chunk of one episode, where
speed is strongly autocorrelated — hence its tiny 2.11 m/s displacement. It is kept in the table as the
low-\|Δv0\| point of the response curve, **not** as a "wrong speed" control; `v0_gshuffled` (global pool
over all 44 episodes, 10.83 m/s) is that control.

## 4.4 What this settles and what it does not

* **E4′ is upgraded from `HYPOTHESIS` to `MEASURED` on two arms.** *"The metric readout is an
  action-integrator, not a perception decoder"* now has the surgical proof X2 was owed. `PENDING` →
  **CLOSED**.
* ⛔ It does **not** say the encoder has no metric content — X1 measured that separately (a ridge probe
  to speed at R² ≈ 0.61). It says the **imagined** readout is dominated by the injected scalar.
* ⛔ It does **not** transfer to the deployed planner path, which reads `out["traj"]`, not the grounding
  readout. It transfers to every claim about *imagination quality* that quotes `g_*_fwd_ade_m`.

---

# 5. ⭐ GAP 2 — v4's FAN GEOMETRY, DUMPED

`MEASURED` · tier **DECISION-GRADE** for the zero-change result, **CONFIRMED** for the vocabulary
decomposition (single surface) · `raw/fan_v4fs_meta.json`, `raw/fan_v4fs_reduced.pt`,
`raw/v4_reach_clamp.json` · 6.3 GPU-min.

## 5.1 The dump

**967 windows × 256 candidates × 20 waypoints × 2**, from `flagship-v4-fromscratch` step 29999, built
through `FlagshipV4Dataset` (MODE B) with `goal_mode = oracle` — i.e. `_goal_inputs` verbatim, the
as-evaluated path — at `lambda_plan = 1.0`, `sel_reach_clamp = False` (asserted, twice: on the run's own
config **and** on the shipped `V4Config` default, so the test cannot become circular).

⚠️ **`base_rank` is not read anywhere.** It is `[as-trained pick] ++ [anchor index order]` and carries
zero score information (**retraction class C15**, hit independently by three streams). The ranking used
is the head's own **`sel_score`** — `refined_logits` *after* v4's factorised LAT×LON×DIST grafts and
after the VTARGET term, i.e. the exact tensor `select` argmaxes.

**Fidelity, both directions:** the recorded `sel_idx` equals `argmax(sel_score)` on **1.000000** of
windows (the test script *refuses to run* below 1.0), and the head config is taken from the run's own
`config.json['head_cfg']` — a default-built head does not even load (`cond_imagination` was `False`).

**Independent cross-check** against the separately rescued `fan_last_along_v4.pt`
([881, 256] fp32, md5 `63dd7b77…`, `_src = v5_v4_windows.pt fan[:,:,-1,0]`), a **different** window set
and a **different** v4 checkpoint:

| quantity | rescued surface | **this dump** | agreement |
|---|---:|---:|---:|
| max last-waypoint along-track | 100.5707 m | **100.1391 m** | **0.43 %** |
| implied max mean speed | 50.285 m/s = **181.03 km/h** | 50.087 m/s = **180.31 km/h** | 0.40 % |

⇒ the two dumps describe the same object. The 181 km/h fan is reproduced.

## 5.2 🔴 THE ZERO-CHANGE TEST — the clamp is exactly free on v4

`accel_max = 2.5` m/s², `horizon_s = 2.0` ⇒ band `v0 ± 5.0` m/s — **the head's own `sel_accel_max`**
(`flagship_v15.py:139,468`), applied to the candidates. **Nothing tuned on held-out error.**

| quantity | REF-C-XL (FIX 2, 881 windows) | **flagship-v4 (this work, 967 windows)** |
|---|---:|---:|
| candidates removed | 72.08 % | **75.29 %** |
| windows with an empty survivor set | 0.00 % | **0.00 %** |
| ADE-oracle survives | 100 % | **100 %** (0.2553 → 0.2553) |
| **windows where the pick moves** | 0 of 881 | **0 of 967** |
| **paired Δ ADE** (episode-cluster, B = 2000) | 0.0000 [0.0000, 0.0000] | **0.000000 [0.000000, 0.000000]** |
| miss@2m | 0.0159 → 0.0159 | **0.0641 → 0.0641** |
| ⇒ per-candidate compute | 3.58× cheaper | **4.05× cheaper** |

Identical on the **dense 20-waypoint** metric the head is actually trained on: ADE 0.5912 → 0.5912,
oracle 0.2054 → 0.2054, 0 picks moved, Δ = 0.000000 [0, 0].

## 5.3 ⛔ AND THE BOTH-DIRECTIONS GUARD FIRED. THE DEFAULT STAYS OFF.

**Δ = 0 is evidence only if the instrument CAN move the pick.** Driving the identical code at tighter
bands (`fan_err4`, same windows, same estimator):

| `accel_max` | removed | oracle survives | as-trained ADE | clipped ADE | **picks moved** | **paired Δ [95 % CI]** | sep |
|---:|---:|---:|---:|---:|---:|---|---|
| **2.5** (the head's own) | 75.29 % | **100 %** | 0.7480 | **0.7480** | **0 / 967** | **0.000000 [0.000000, 0.000000]** | — |
| 0.5 | 93.61 % | 86.66 % | 0.7480 | **0.6662** | 168 | **−0.0818 [−0.1498, −0.0265]** | ✅ |
| 0.2 | 97.21 % | 64.84 % | 0.7480 | **0.6140** | 445 | **−0.1340 [−0.2329, −0.0454]** | ✅ |
| 0.05 | 99.27 % | 24.20 % | 0.7480 | 0.7159 | 675 | −0.0321 [−0.1682, +0.1042] | ❌ |

* **Direction 1 PASSES:** the instrument demonstrably moves the pick (168 / 445 / 675 windows). The zero
  at 2.5 is therefore a real zero, not a dead guard (the **C13** class).
* **Direction 2 FAILS:** FIX 2's own committed test requires that a tight band make the ADE **strictly
  worse** — *"if a tight band ever helped, the clamp would be a tuned selector rather than physics."*
  **On v4 it helps, separated, at two of three tightenings, and miss@2m falls 0.0641 → 0.0331.**

> ### ⇒ **VERDICT: `V4Config.sel_reach_clamp` STAYS `False`, and I am saying so rather than relaxing the test.** The brief's primary bar (pick unchanged, Δ = 0.0000, oracle surviving) **is met on v4** — that part of the escalation is now MEASURED and can be retired. But the guard that makes the zero mean *physics* rather than *a knob at a lucky setting* **fails**, and v4's standing invariant (*"`q` must never exist in the deployment path"*) exists for precisely the temptation the row above creates. **Flipping the default would be inert today and would place a live ADE knob inside the deployed selector.**

⚠️ **A tighter band destroys coverage while improving the realised pick** — the oracle survives only
86.66 % at 0.5 and its ADE degrades 0.2553 → 0.3180, yet the *selected* ADE improves. That is the
LLM-Assist / CoverNet result on our own fan: **a big fan is an adversarial search against an
approximate scorer**, and v4's scorer is losing in it. `PUBLISHED (cited)`.

## 5.4 ⭐ The "blameless vocabulary" claim does NOT hold on v4's fan

| quantity | value |
|---|---:|
| emitted max implied mean speed | **50.09 m/s = 180.31 km/h** |
| emitted p99 | 45.84 m/s = 165.03 km/h |
| **anchor VOCABULARY max** | **46.96 m/s = 169.07 km/h** |
| anchor vocabulary p99 | 44.38 m/s = 159.77 km/h |
| offset-head contribution, mean \|Δ\| | **1.67 m/s** |
| offset-head contribution, p99 / max \|Δ\| | 5.80 / 7.26 m/s |
| **share of REMOVED candidates whose ANCHOR was already unreachable** | **94.36 %** |
| val GT max implied mean speed | 34.88 m/s = 125.5 km/h |

> ### ⇒ **On v4, the reachability clamp removes VOCABULARY, not offset-head excess.** The 256 anchors are still bitwise-real human windows — but they are **state-independent**, drawn by furthest-point-sampling from a 200 000-window pool, so by construction they over-sample the tails: a 169 km/h anchor is offered to a 5 m/s window. The offset head shifts a candidate's implied speed by only **1.67 m/s** on average. **The FIX 2 sentence *"the vocabulary is blameless; it is the unbounded offset head that emits 171.5 km/h"* is a REF-C-XL statement and does not transfer to v4** — and the lever it points at on v4 is **CoverNet-style state-conditioned set construction**, not a refinement bound. `MEASURED`, one surface, tier **CONFIRMED**.

---

# 6. 🔴 ESCALATIONS — in the headline, not in a README

**E-A. `MODEL_REGISTRY.md`'s real-vs-imagined row must be entered WIDER than E1′ scoped it, and E1′'s
reason is void.** E1′ scoped the row to the v1 family *"because v4 does not carry the instrument"*. v4
does carry it (§1), and asked, it answers: **20.14× / 21.93× / 14.72×** at op/tac/str, step 29999,
44 clean held-out episodes / 967 windows, with the aggregation caveat. Enter v4 as its own row, flagged
NON-PARITY-val so it can never be mistaken for a leaderboard number.
*Owner: registry owner. Blocked on nobody.*

**E-B. One line in `train_flagship_v4.py` makes every future v4 run self-measuring, and the earlier fix
to that line only got HALF of it.** `:159` forwards `g_op_fwd_ade_m` and nothing else — the paired REAL
partner still never reaches disk, so the ratio remains unloggable even after the documented repair.
Widen the tuple to all six `g_{lvl}_{mid_de_m,fwd_ade_m}`. **Log-only, no loss term, no parity effect.**
*Owner: trainer owner.*

**E-C. `V4Config.sel_reach_clamp` STAYS `False` — but the escalation's stated blocker is now cleared and
should be rewritten, not left standing.** `flagship_v4.py:124-140` and
`CONFIRMED_FIXES.md` §FIX 2 both say the blocker is *"v4's fan geometry was never dumped"*. It has been
(§5.1), and the zero-change property **holds** (§5.2). The remaining reason to keep it off is different
and stronger: **the tight-band guard fails on v4** (§5.3). The comment and
`test_the_reachability_clamp_is_OFF_on_v4_until_it_is_measured_there` should be updated to cite the real
reason, or the next agent will re-dump a fan that already exists.
*Owner: whoever owns FIX 2. **Do not flip the default on the strength of §5.2 alone.***

**E-D — NEW, and it is worth more than the flip. A purely physical prior buys v4 −0.0818 [−0.1498,
−0.0265] m of ADE and halves miss@2m (0.0641 → 0.0331), separated, for free.** That is a real lever
sitting behind an invariant written for a different object (a *score*-based `q`, not a *kinematic*
band). It must **not** be taken by loosening a default; it should be a **pre-registered arm** with both
outcomes committed, because the same measurement shows it destroys 13 % of oracle coverage and the
program has a standing rule against fitting a selector to `ade`. **Companion finding for the same
arm:** §5.4 says the cheaper and more principled version is a **state-conditioned anchor set**, which
attacks the cause rather than filtering the symptom.
*Owner: v4/v5 planner owner. Discriminating experiment, not a config change.*

**E-F — 🔴 TIME-CRITICAL, FOUND WHILE THIS JOB RAN, AND IT IS NOT MINE. THE WORKING TREE CAN NO LONGER
LOAD ANY EXISTING v4 CHECKPOINT.** A sibling stream added `VisionRankProjection` to
`FlagshipV4Head.__init__` (`flagship_v4.py`, uncommitted working-tree change on top of `abc864a`;
`V15Config.vision_rank = 16`). At rank 16 the projection is **not** parameter-free, and the factorised
heads' first Linear changes from `2048 →128` to `16 →128`. **MEASURED on `flagship-v4-fromscratch`'s own
`head` state dict** (98 tensors vs the new head's 100):

```
Missing key(s):  vision_rank_proj.basis_loaded, vision_rank_proj.proj.weight
size mismatch:   lat_head.0.weight  [128, 2048] -> [128, 16]
                 lon_head.0.weight  [128, 2048] -> [128, 16]
                 dist_head.0.weight [128, 2048] -> [128, 16]
```

`eval_flagship_v4.load_v4_from_ck:290` does `head.load_state_dict(ck["head"])` **STRICT**, so as the
tree stands **`flagship-v4.1-10k`, `flagship-v4.2-step4000` and `flagship-v4-fromscratch` are all
unloadable and every committed v4 number is unreproducible.** The change is right on its merits
(decode-side, `encoder_touching_levers` unchanged) — it needs a back-compat path: `vision_rank =
state_dim` for legacy configs, or a migration in `load_v4_from_ck`. ⚠️ **This is exactly why the GAP 2
dump here used the pre-change `flagship_v15.py` md5 `ceae820d…` / `flagship_v4.py` md5 `81ee86cd…` —
the as-trained architecture. Re-running `v4_fan_dump.py` against today's tree will fail at the strict
load, not produce a different number.**
*Owner: the v5-prep / vision-rank stream. Verified by construction + strict-load attempt, not inferred.*

**E-E. X1_LATENT_METRIC.md §3.1 and its §0 verdict need a correction entry, by root-cause CLASS.** The
class is *"absence of a STRING taken as absence of a CAPABILITY; the call was one frame away"* — the
same class as the Vulkan-ICD and `ps -C python3` retractions, and it fired inside a report written to
warn about that class. The corrected statement: **"no committed v4 LOG carries a `g_*` key"** (true,
reproduced here) — **not** "v4 does not carry the instrument".
*Owner: `Project Steering/RETRACTION_LOG.md`.*

---

# 7. LIMITATIONS AND COUNTER-EVIDENCE

| | can support | ⛔ cannot support |
|---|---|---|
| **GAP 1 bands** | v4 vs v1 on identical windows; the presence, sign and order of the gap on v4; the registry row's scope | any cross-arm leaderboard number — **NON-PARITY val key**. The absolute metre values are not comparable to `MODEL_REGISTRY` rows measured on `0c5f7dac3b11` |
| **the `v0` ablations** | that the IMAGINED metric readout integrates the injected scalar, isolated to machine precision within one model | any claim about the deployed planner path (`out["traj"]`), which does not read the grounding readout |
| **GAP 2** | v4's own fan geometry and the clamp's exact effect on it | the canonical **881**-window set — that split's frames are not reachable from pod3, so this is 967 different (clean) windows |

**The aggregation caveat travels with every ratio here.** `*_mid_de_m` is an **endpoint** displacement
error averaged over the level's horizons; `*_fwd_ade_m` is an **ADE over waypoints 1…k**. For a growing
error the endpoint statistic is the larger, typically by ~2×, so **every ratio in §3.2 is a bound of the
right order, not a clean effect size** — `grounding_ci.json` publishes `ratio_agg_corrected_x0.5`
beside every raw ratio. **The v4-vs-v1 comparison and the ratio-of-ratios are aggregation-invariant**,
because the same factor divides out; the §4 ablation factors are imagined-vs-imagined and do not inherit
the caveat at all.

**Counter-evidence carried, not dropped.** PlaNet Fig. 7 (`PUBLISHED (cited)`, Hafner et al.,
[arXiv:1811.04551](https://ar5iv.labs.arxiv.org/html/1811.04551)) — multi-source/multi-horizon training
repairs a misspecified predictor and can *hurt* an adequate one. §4 says v4's imagined readout is
`v0`-dominated; **that does not license removing `v0`**. Removing it makes the imagined decode ×39–94
worse. The right reading is that the readout's competence is *not evidence of a competent world model*,
not that the channel should go.

---

# 8. DELIVERABLE MANIFEST

**Everything below is in the repo working tree and STAGED (`git add`). This agent committed nothing,
pushed nothing, and switched no branch.** Verified with **`git ls-files --stage`**, not a scoped
`git status --short`.

**Deliverable path:**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-v4-instrument/`

| artifact | what it is | where it lives | only one place? |
|---|---|---|---|
| `V4_INSTRUMENT.md` | ⭐⭐ this report | **repo** (staged) | no — derived from the JSON below |
| `PRE_REGISTRATION.md` | ⭐ both outcomes, the `v0` predictions and the clamp verdict rule, written before the first number | **repo** (staged) | no |
| `scripts/v4_grounding_probe.py` | GAP 1 + §4: the paired real/imagined bands, 6 `v0`/action ablations, 5 validation gates | **repo** (staged) | also `pod3:/workspace/v4instr/` |
| `scripts/v4_grounding_ci.py` | GAP 1 stage 2: paired episode-cluster bootstrap, the outcome rule in code | **repo** (staged) | no |
| `scripts/v4_fan_dump.py` | GAP 2 stage 1: v4's per-candidate fan geometry + its real `sel_score` | **repo** (staged) | also `pod3:/workspace/v4instr/` |
| `scripts/v4_reach_clamp_test.py` | GAP 2 stage 2: the zero-change test, the both-directions guard, the vocabulary/offset decomposition | **repo** (staged) | no |
| `raw/grounding_bands_v1.json`, `raw/grounding_bands_v4fs.json` | every band + ablation + validation gate, per arm | **repo** (staged) | no |
| `raw/grounding_v3b_v1.json`, `raw/grounding_v3b_v4fs.json` | the V3b cross-episode shuffle control | **repo** (staged) | no |
| `raw/grounding_perwindow_v1.pt`, `raw/grounding_perwindow_v4fs.pt` | 967-window arrays for all 7 conditions — the input to every interval | **repo** (staged) | no |
| `raw/grounding_ci.json` | every interval, the cross-arm paired contrasts, the ratio-of-ratios verdict | **repo** (staged) | no |
| `raw/fan_v4fs_meta.json` | the fan dump's fidelity + speed statistics | **repo** (staged) | no |
| `raw/fan_v4fs_reduced.pt` (5.98 MB) | **v4's fan geometry**: per-candidate mean speed, ADE (dense + 4wp), `sel_score`, `v0`, terminal xy, anchor speeds | **repo** (staged) | no |
| `raw/v4_reach_clamp.json` | the zero-change result, the tight-band sweep, the vocabulary decomposition, the verdict | **repo** (staged) | no |

## 8.1 Working files deliberately NOT staged, each with its rebuild command

| file | size | where | why it is safe |
|---|---:|---|---|
| `v4fs_ckpt.pt` | 3.243 GB | `pod3:/workspace/v4instr/` | **not our artifact** — gated HF `Sayood/flagship-v4-fromscratch/ckpt.pt`. Re-pull in 15 s |
| `fan_v4fs_full.pt` | ~39.6 MB | `pod3:/workspace/v4instr/out/` | the full [967, 256, 20, 2] geometry; `fan_v4fs_reduced.pt` (staged) carries everything the test needs. Rebuild: `v4_fan_dump.py`, 6.3 GPU-min |
| v1 ckpt | 3.309 GB | `pod3:/workspace/tmp/idm/ckpt.pt` | pre-existing; HF `Sayood/tanitad-flagship-4b-speedjerk` + pod copies |

**Nothing that took real effort lives in only one place.**

## 8.2 pod3 was left as it was found — and one drift finding

`flagship_v15.py` / `flagship_v4.py` were temporarily replaced with the repo versions for the GAP 2 dump
(the pod's copies predate FIX 2 and have **no** `reachability_mask` / `sel_reach_clamp`) and **restored
byte-identically afterwards** (md5 `ef6ab811…` / `5ca5dde3…`, verified). No trainer was touched, no PID
killed, `pytest -q` is unaffected because **no file under `stack/` or `taniteval/` was modified**.

🔴 **Drift, for the nightly `pod_git_drift.py` owner:** `pod3:/workspace/TanitAD` sits at commit
`0f93b98` and its `flagship_v15.py`, `flagship_v4.py`, `config.py` and `flagship_losses.py` all differ
from the repo. A pod carrying a pre-FIX-2 `flagship_v15.py` will silently produce unclamped v1.5
behaviour if anything is launched from it.

⚠️ **Exact code provenance of the GAP 2 dump, because the repo moved under it mid-job.** The fan was
dumped with `flagship_v15.py` md5 **`ceae820d7ba1e7842112ebfcc407cc08`** and `flagship_v4.py` md5
**`81ee86cd59f6b66d68aa9006e5ac761b`** — the working-tree state at 03:0x, which strict-loads the v4
checkpoint. By 03:4x a sibling stream had changed both (E-F). GAP 1 used the pod's own
`metric_dynamics.py` and `fourbrain.py`, **byte-identical to the repo** (md5 `c9374e73…` / `96ab4404…`),
so no drift touches it at all.

## 8.3 Reproduction — end to end

```
# pod3, 830.7 s of pod time total (MEASURED)
scp scripts/*.py tanitad-pod3:/workspace/v4instr/
# 1) GAP 1 + the v0 ablations, both arms (179.1 s + 77.8 s MEASURED)
PYTHONPATH=/workspace/TanitAD/stack python v4_grounding_probe.py \
    --ckpt <ckpt> --arm <v1|v4fs> \
    --cache-dir /workspace/v4run/valcache/physicalai-val-heldout-79d4e3d2d4c6 \
    --episodes 44 --out /workspace/v4instr/out
# 2) GAP 2 fan dump (378.6 s MEASURED) -- --head-config is REQUIRED
PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts python v4_fan_dump.py \
    --ckpt /workspace/v4instr/v4fs_ckpt.pt --head-config <run config.json> \
    --cache-dir <same> --episodes 44 --stride 8 --out /workspace/v4instr/out

# dev box, ZERO GPU
python scripts/v4_grounding_ci.py --v1 raw/grounding_perwindow_v1.pt \
    --v4 raw/grounding_perwindow_v4fs.pt --out raw/grounding_ci.json
python scripts/v4_reach_clamp_test.py --dump raw/fan_v4fs_reduced.pt \
    --out raw/v4_reach_clamp.json
```

## 8.4 What this unblocks

| stream | what it can now do |
|---|---|
| **the registry** | enter the real-vs-imagined row **program-wide**, with v4's own numbers, instead of scoping it to v1 because v4 "could not be asked" (E-A) |
| **the v4/v4.x trainer** | make every future run self-measuring with a one-line log change (E-B) — and stop shipping v4 gates that render INCOMPLETE for want of a grounding secondary |
| **FIX 2 / the v1.5 clamp owner** | the v4 fan exists; the zero-change property is MEASURED on it; the reason to keep the default OFF is now a *different, stronger* one that should replace the stale comment (E-C) |
| **the v5 imagination-selection stream** | a real v4 fan with a real ranking (not `base_rank`), plus the measured result that pruning to a physical band **helps** v4's scorer (E-D) and that the excess is **vocabulary**, pointing at state-conditioned anchors (§5.4) |
| **the peek / duty-cycle / re-anchoring streams** | X2 is paid: they can stop treating `g_*_fwd_ade_m` as evidence of imagination quality — it is ×39–94 an artefact of the injected `v0` |
| **RETRACTION_LOG** | a new root-cause class instance with a clean, reproduced correction (E-E) |
