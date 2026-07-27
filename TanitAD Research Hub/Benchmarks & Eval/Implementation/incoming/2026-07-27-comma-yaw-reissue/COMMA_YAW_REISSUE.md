# comma2k19 `yaw_rate` — the re-issue

**Date:** 2026-07-27 · **Agent:** `comma-yaw-reissue` · **HEAD at start:** `37ccfea`
**Host:** dev box only. **pod1 (training), pod2 (small validation) and pod3 were not touched.**
**Nothing was recomputed.** Every number below is *read* from an artifact that already existed.
**Evidence class:** `MEASURED (ours + artifact path)` unless a line says otherwise.
**Estimator:** `taniteval.ci.(paired_)episode_cluster_bootstrap`, unit = the 36 val episodes, B = 2000.
**`overlapping_holdout_se` is not used anywhere in this work.**
**Reported per corpus, never pooled** — which is the entire point here, because PhysicalAI is
unaffected and comma2k19 is not.

**Reads with:** `…/Architecture & Inference/…/incoming/2026-07-27-idm-v3/IDM_V3.md` §4 (the finding),
`Project Steering/RETRACTION_LOG.md` class **C29**.
**Raw:** `raw/comma_yaw_anchor.json` (the anchor + honesty condition), `raw/comma_yaw_inventory.json`
(all **75** locations, machine-readable).

---

## 0. The headline, stated so it cannot be over-read

> **Every comma2k19 `yaw_rate` number this program published before 2026-07-27 was scored against a
> heading label that is undefined at standstill.** They are **not wrong about the model** — the model
> was right and the label was wrong (**C29**) — but they are **stale as published**.
>
> **75 locations** carry such a number or an inference built on one. **4** could be corrected with a
> repaired measurement that already exists. **46** are **STALE-PENDING**: they were measured on a
> substrate where **no repaired number exists**, and this pass deliberately did **not** invent one.
> **6** are not stale numbers at all but **conclusions that C29 overturns** — chiefly *"comma is
> disqualified for yaw"*, which gated the YouTube-IDM rotation channel for **five days**.
> **2** are a plain factual error found on the way (a `0.83` that should read `0.8108`).
>
> ⭐ **And the correction must be stated against its own interest, or it misleads in the other
> direction.** On comma2k19 alone the repair moves **R² +0.0114 → +0.3308** and **MAE −42.5 %** — but
> **medAE moves only −1.1 %**, **nMedAE gets 8.0 % WORSE**, and **Spearman ρ is flat (+0.001)**.
> **The repair fixes the tail and the summary statistic, not typical accuracy.** A re-issue that
> quotes only *"0.105 → 0.811"* overstates what changed.

---

## 1. The defect, and why PhysicalAI is out of scope

comma2k19's heading is `arctan2(enu_v_north, enu_v_east)` — the direction of the **ENU velocity
vector**, which is **undefined when the vehicle is stationary**
(`stack/tanitad/data/comma2k19.py`, `HEADING_MODE_LEGACY`, **still the loader default**).

`MEASURED · DECISION-GRADE` — `…/2026-07-27-idm-v3/results/labels_v3.json → yaw_audit_by_speed`,
all 64 comma val segments:

| speed bin (m/s) | n frames | **% physically impossible (\|ω\| > 1.5)** | max \|ω\| |
|---|---:|---:|---:|
| **[0.0, 0.5)** | 217 | **26.27 %** | **15.53 rad/s** |
| [0.5, 1.0) … [20, 100) | 18 855 | **0.000 % in every bin** | ≤ 0.51 |

**PhysicalAI: zero impossible frames in every bin** — its heading comes from an orientation
quaternion, which is standstill-robust. The repair changes **`n_pai_changed = 0`** windows
(asserted in the artifact), and PhysicalAI's yaw R² is **bit-identical** before and after.

⛔ **Therefore: no PhysicalAI yaw number in this repo is stale, and none was re-issued.**
This is the reason the brief's "per corpus, never pooled" rule is load-bearing rather than stylistic —
**a pooled number over a mix containing comma IS in scope**, and three of the highest-traffic numbers
in the program (the v1 card's `0.0104`, the C5 retraction's `0.010`, the `0.105`) are exactly that.

---

## 2. The anchor — the only repaired measurement that exists

**The DEPLOYED head (`idm_head_v1` = "A0", 2,899,724 params, frozen `flagship4b-speedjerk-30k`
latents), NOTHING RETRAINED, scored twice on the IDENTICAL windows.**
Protocol: **v3 val split, 4,195 windows / 36 episodes** (comma 2,992 / 22 eps; PhysicalAI 1,203 /
14 eps), **`v_min` 0.5**. Source: `…/2026-07-27-idm-v3/results/compare_v3.json →
LABEL_FIX_deployed_head/yaw_rate`. `MEASURED · DECISION-GRADE`

| corpus | statistic | `heading_repair` **OFF** | `heading_repair` **ON** | change |
|---|---|---:|---:|---:|
| **comma2k19** (n = 2 992) | **R²** | **+0.011430** | **+0.330822** | +0.319 |
| | MAE | 0.043978 | 0.025300 | **−42.5 %** |
| | **medAE** | 0.0200874 | 0.0198625 | **−1.1 %** |
| | **nMedAE** | 1.43714 | 1.55247 | **+8.0 % WORSE** |
| | Spearman ρ | 0.343827 | 0.344786 | +0.001 |
| **PhysicalAI** (n = 1 203) | R² | +0.903482 | +0.903482 | **bit-identical** |
| *POOLED* ⚠️ *do not quote* | R² | +0.104592 | +0.810826 | +0.706 |
| | medAE | 0.0211812 | 0.0210731 | −0.5 % |
| | nMedAE | 1.79840 | 1.90438 | **+5.9 % WORSE** |

**Why the R² moves so far while medAE does not.** The repair touches **50 windows (1.19 %)**;
4,145 are bit-identical (`max_abs_label_delta = 0.0`). On those 50 the vehicle is **stationary**
(GT speed max **0.528 m/s**, mean 0.042), the legacy label claimed up to **9.47 rad/s (543 °/s)**, and
**A0 had predicted 0.023**. R² has an unbounded left tail, so 50 impossible rows dominated it. A
median does not notice them — which is precisely why **medAE barely moves**.

### 2.1 The repair beats the earlier deletion fix, and discards nothing

| protocol | n | pooled | PhysicalAI | **comma2k19** |
|---|---:|---:|---:|---:|
| legacy — **what we published** | 4 195 | +0.1046 | +0.9035 | **+0.0114** |
| 9 impossible windows **deleted** (IDM v2's fix) | 4 186 | +0.4967 | +0.9035 | **+0.0719** |
| **`heading_repair` ON, `v_min` 0.5** | **4 195** | **+0.8108** | +0.9035 | **+0.3308** |

### 2.2 The other repaired comma values that exist — and what each may replace

| what | comma2k19 yaw R² | protocol | may be quoted for |
|---|---:|---|---|
| A0, deployed head, nothing retrained | **+0.3308** | repair ON, `v_min` 0.5, v3 val split | **only** the v3-val-split A0 numbers |
| `R0LEG` (legacy-trained) **re-scored** on repaired labels | **+0.5894** | repair ON (scoring), legacy training | IDM v2's `B0` row only — bit-identical seed-0 match |
| `R0` (retrained on repaired labels), 3-seed | **+0.6553** | repair ON, training **and** scoring | the v3 rotation expert |
| `V3F` shipped composite | **+0.6791** | repair ON | the shipped v3 card |
| v4 steer-ladder rung 757 | pooled **+0.9188** | repair ON (`idm4_steer.py:43`) | the fleet-refill ladder |

⛔ **None of these substitutes for a number measured on a different substrate.** That is the single
rule that produces the stale-pending list in §4.

---

## 3. Corrections made — superseded values kept visible

Every edit **preserves the superseded value and its date** and adds the protocol beside it, the way
the `observed_frac` and `heldout`/`full_set` corrections were handled. **No number was deleted.**

### 🔴 3.1 THE FLAGGED REGISTRY DIFF — review this first

**`Project Steering/MODEL_REGISTRY.md` §8.1 item 6** (the supervised-IDM cross-domain finding).
The cell **`comma2k19 … yaw R² 0.000`** is **left in place** and now carries a `LABEL-PROTOCOL
CORRECTION 2026-07-27 (C29)` block stating: the protocol (`heading_repair` OFF, **no `v_min`**),
that **`0.000` is a fact about the label**, that it is **STALE-PENDING and NOT replaced** (that
substrate has 12,420 comma windows and no repaired measurement), what *is* measured elsewhere
(+0.3308 / +0.679), the **honesty condition**, and that **PhysicalAI/rig-B numbers in §8.1 #6 and
§10 are unaffected and must not be re-issued.**
*Registry diff is additive only — no existing figure in the registry was altered.*

### 3.2 The rest

| file | what | action |
|---|---|---|
| `Benchmarks & Eval/LEADERBOARD.md` §7.3 | comma `yaw R² 0.000` | annotated; **states the FAIL verdict is unchanged** — the gate's primary is *speed* (0.657) and ADE ratio 2.40, neither touched by the heading label |
| `Project Steering/LOOP_STATE.md` | pod3 IDM entry, `yaw 0.0005` | annotated; **NO-GO left standing** |
| `…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` | ⭐ the deployed head's own card — named by IDM_V3 escalation #2 | added `label_protocol_reissue_2026_07_27` + one `usage_caveat`. **No existing value altered** (both yaw fields re-read after the edit). Per-number verdicts inside: pooled `0.010433` **stale-pending**, parity-val `0.807472` **not stale** |
| `Project Steering/RETRACTION_LOG.md` C5 row | *"R² 0.010 at scale (n=9,420)"* | annotated with a forward pointer: **deletion was the wrong fix**; row not rewritten |
| `Paper/TANITAD_PAPER.md` §H7 + §7.9 | *"yaw ≈ 0 cross-class"* | annotated in both places; notes the three-measurement H7 argument never rested on the yaw channel |
| `Project Steering/PROGRAM_OVERVIEW.md` | *"yaw ≈ 0"* | annotated inline |
| `…/2026-07-26-idm-v2/IDM_V2_RESULTS.md` §3.2 | the per-corpus yaw table | annotated with the three-protocol table; **B0/B1/V3sB/V3wB explicitly marked stale-pending** |
| `…/2026-07-26-idm-v2/IDM_DIAGNOSIS.md` | comma yaw **ceiling 0.352** | annotated — see §3.3 |
| `…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md` | comma `R² 0.0719` + "defect #2 still live" | annotated — **defect #2 is now closed**, and its prediction was right |
| `…/2026-07-24-idm-pipeline-derisk/RESULTS_idm_pipeline_derisk.md` | *"comma cannot test yaw pseudo-labels"* | annotated — **conclusion overturned**, cell left stale-pending |
| `…/2026-07-24-branchb-transfer-eval/v1-encoder-char/RESULTS_v1_encoder_char.md` | caveat **C6** | annotated — **disqualification lifted** |
| `…/2026-07-24-branchb-transfer-eval/MANIFEST.md` | *"untestable on comma"* | annotated |
| `stack/tanitad/data/comma2k19.py` | 🔴 **factual error** | **corrected** — see §3.4 |
| `stack/tests/test_comma2k19.py` | 🔴 same error, second copy | **corrected** |

### 3.3 ⭐ The correction with the largest downstream consequence

Three documents concluded, from the *same* control (a comma-co-trained head reading comma yaw
**R² −0.00003**), that **comma is disqualified for yaw** and *"cross-class YAW transfer is
UNVERIFIED — comma cannot test it."* That conclusion **gated the YouTube-IDM rotation channel**.

**The MECHANISM those documents named was right** — they said it was a label artifact, not a transfer
failure, and C29 confirms it exactly. **The DISQUALIFICATION is overturned:** on repaired labels the
*deployed* head reads comma yaw **+0.3308** and a retrained head **+0.679**. **comma can test yaw.**

⚠️ **But it is testable, not answered** — and the honesty condition is why. The repair lifts the
aggregate, not per-window precision (**medAE −1.1 %**, **nMedAE +8.0 % worse**). "comma yaw is now
usable" is a claim about aggregate quality only.

Same class, in IDM v2's `IDM_DIAGNOSIS.md`: **comma's yaw "smooth-fit ceiling" R² 0.352** was the
premise for *"no model can score well against a label that noisy."* **That ceiling is a property of
the broken label.** The deployed head already reads **+0.3308** (≈ the old ceiling) and a retrained
head **+0.679** — *above* it. ⛔ **The repaired ceiling has NOT been recomputed. Do not quote 0.352;
there is currently no measured repaired ceiling.**

### 3.4 🔴 A plain factual error found on the way (not staleness)

`stack/tanitad/data/comma2k19.py` and `stack/tests/test_comma2k19.py` both stated:

> *"the deployed IDM head's pooled `yaw_rate` R2 read **0.105** against these labels and **0.83**
> against the repaired ones"*

**The deployed head's repaired pooled R² is `0.8108`**, read from
`compare_v3.json → LABEL_FIX_deployed_head/yaw_rate/repaired/pooled/r2`. **`0.83` is the level a head
*retrained* on repaired labels reaches** (`R0`, pooled `0.8413`). Both copies are corrected, and both
now carry the per-corpus split and the honesty condition. The error inflates the *label-only* effect
by attributing a retrained arm's gain to the label — the exact direction the brief warns about.

*(Origin: `IDM_V3.md` §0's headline says "0.105 to 0.83", while its own §4.2 table and §9 escalation
say **0.8108/0.811**. §0 is internally inconsistent — **escalated, not silently rewritten**, since it
is a sibling stream's document and its authoritative sections are correct.)*

---

## 4. ⛔ STALE-PENDING — named, not filled in

**46 locations.** Each was measured on a substrate where **no repaired measurement exists**. Naming
them is the deliverable; inventing a value would not be.

**Why the anchor cannot be pasted into them:** the anchor is the deployed head on **4,195 windows /
36 episodes at `v_min` 0.5**. The pre-repair work below used **different heads, different splits, and
no `v_min` gate at all**. Two numbers that differ by substrate *and* by label protocol are not
comparable in either direction.

| substrate | numbers stranded | what a repaired value would require |
|---|---|---|
| **`idm_head_v1_card.json` → `val_heldout_traindomain`** (90 clips / **9,420 windows**, POOLED rig-A + rig-B + comma) — ⭐ **highest priority: this is the card a consumer reads** | yaw R² **+0.010433**, MAE **0.11814** | ✅ **ACTIONED 2026-07-27** by the `heading-default` pass — see below |
| **`2026-07-22-idm-proof`** (PhysicalAI-trained head → 90 comma val clips, **12,420 windows**, no `v_min`) — 14 values across `results/regate_comma/multirig_primary/multirig_symm/results_multirig/results_regate` + 3 run logs | comma cross yaw R² **+0.00019 … +0.00051** | rebuild the comma window cache with `HEADING_MODE_HOLD`, re-score the same head |
| **`2026-07-22-own-dynamics-encoder`** `in_comma_heldout` — the *"comma yaw unreadable in-domain"* control | **−0.0000292 / −0.0000886** | same rebuild, re-score both camcond arms |
| **`2026-07-24-branchb-transfer-eval/v1-encoder-char`** | `H1 cross_comma` **−0.005529**, `H3 cross_comma` **+0.000216**, and `H2 indom` **+0.835024** (a POOLED rig-A **+ comma** mix ⇒ comma-contaminated) | same rebuild, re-fit the multi-domain heads |
| **`2026-07-24-idm-pipeline-derisk`** `comma_crossCLASS` | zero-shot **+0.000259**, calib-ceiling **+0.000676**, MAE **0.251737** | same rebuild |
| **`2026-07-26-idm-v2` arms B1 / V2 / V3 / V3wB** (v3 val split, but legacy-trained *and* legacy-scored) | comma R² **0.1313 / 0.1412 / 0.1422 / 0.1407** | re-score those checkpoints against repaired labels (the machinery exists — `R0LEG` proves it) |
| **`IDM_DIAGNOSIS.md` comma yaw smooth-fit ceiling** | **0.3521** (w9) / **0.2245** (w17) | recompute the ceiling on repaired labels |

### ✅ 4.1 UPDATE 2026-07-27 — the first row is actioned, and the answer is not the anchor

*(added by the `heading-default` pass; the row's original text is preserved above.
Full record: `…/incoming/2026-07-27-heading-default/HEADING_DEFAULT.md` §5, raw
`raw/idm_head_v1_comma_rescore.json`. `MEASURED`, dev box, **no pod touched, nothing retrained**.)*

The persisted head was re-scored on the **comma half** of that exact split — `cm_[40:70]`,
**30 clips / 4,140 windows**, `episode_id`-disjoint from its own 40 comma training clips.
⛔ **The POOLED 9,420 number is still NOT re-issued** (the rig half was not recomputed, and
PhysicalAI is unaffected anyway) — **per corpus, never pooled**, exactly as §1 argues.

| | repair **OFF** (no `v_min` — the card's own protocol) | repair **ON** (`v_min` 0.5) |
|---|---:|---:|
| comma yaw **R²** | **+0.000048** | **−0.000421** |
| MAE | 0.228804 | 0.152695 (**−33.3 %**, paired CI [−0.150, −0.012], separated) |
| medAE / nMedAE / ρ | 0.02503 / 2.262 / +0.1980 | 0.02438 / **2.490 (worse)** / +0.1987 |

⭐ **The repair does NOT move R² on this substrate.** The honesty condition reproduces **in shape**
(MAE down hard, medAE barely, nMedAE worse, ρ flat) but the **R² lift does not** — anyone who had
pasted the anchor's `+0.3308` into the card would have published **+0.33 where the measurement is
≈ 0**. That is this document's own §2.2 rule ("none of these substitutes for a number measured on a
different substrate") vindicated by measurement rather than by argument.

**MEASURED cause:** one **wholly-stationary clip** (`cm_00045`, 300 frames, **zero** observable
frames, v_max 0.039 m/s). `hold_heading_through_standstill` deliberately leaves such a segment
unchanged, so its 138 windows keep a garbage label — **84 physically impossible, up to 15.28 rad/s** —
and R² has an unbounded left tail. **Raising `v_min` cannot help** (84–85 survivors at 1.0/2.0/4.0).
🔴 The repair's own `observable` mask is the fix and **no caller in the repo uses it** — escalated.

⚠️ **And one confound this pass could NOT close:** the anchor's comma val lives on
`comma2k19-val-**76b6e94a97a1**` (64 segs) while A0 trained on `**61c46fca8f7f**` (90 eps), so the
`cm_*` indices are **not comparable** and **whether the anchor's val overlaps A0's training clips is
UNKNOWN** (`HYPOTHESIS`). Both draw from the same 21 routes of `raw_data/Chunk_1`. The probe is one
command on the eval pod and half of it is staged — see `HEADING_DEFAULT.md` §5.5/§7.

⚠️ **One rounding discrepancy, worth fixing when someone re-issues these:** `LOOP_STATE.md` quotes
comma yaw as **0.0005** where `LEADERBOARD.md` and `MODEL_REGISTRY.md` quote **0.000** — the same run
(`results.json` = `0.00047288938804967984`), **two roundings in circulation**.

---

## 5. NOT stale — 14 locations checked and left alone

⛔ **Do not assume all comma-adjacent yaw numbers are stale.** These were checked and are correct:

- **Every PhysicalAI-only yaw number** — `n_pai_changed = 0`, bit-identical: idm-proof in-corpus
  **+0.9244**; the v1 card's `val_parityval` **+0.807472**; A0 **+0.9035**; VALIDATION.md **0.940 /
  0.784 / 0.8075**; the parity-val pilot's **0.55 → 0.75**; the rig-B cells **+0.504 / −0.109 /
  ≈ 0**; `aux_yaw_r2` training telemetry.
- **Every post-repair comma number** — v3 `R0` / `V2R` / `V3F`, the fleet-refill v4 rungs, and
  `A0_on_repaired_labels`. `idm4_steer.py:43` declares its protocol **in code**, which is why it was
  classifiable without re-reading the run.
- **YouTube pseudo-label yaw statistics** in `DB_RETRY.md` / `YT_DB_RETRY.md` — YouTube corpus, no
  comma labels involved.

⚠️ **One confound flagged rather than corrected:** `v1-encoder-char` `H2 cross_rigB` yaw **+0.5035**
is evaluated on **PhysicalAI** (clean), but its **head was fit on a mix containing broken comma
labels**. The eval corpus is clean; the *arm* is confounded. Flagged, not re-issued.

---

## 6. Second probe — because absence at one location is not absence

The JSON sweep and the prose/code sweep were run **independently** and reconciled. The second probe
added **21** locations the first missed (the `IDM_DIAGNOSIS` ceiling family, the four IDM-v2 arm rows,
the `PRE_REGISTRATION_IDMV2` inference, three run logs, the second `0.83` copy, `PROGRAM_OVERVIEW`,
`R3_hypothesis_portfolio`, three code docstrings, three program-report restatements).

**Absences confirmed at ≥ 2 spellings each:** `PROJECT_STATE.md`, `README.md`, `CLAUDE.md`,
`DECISIONS.md`, `taniteval/`, and `Ressources/` carry **no comma2k19 yaw metric**.

**And the absence that mattered most:** before this pass, **`MODEL_REGISTRY.md` contained no
`0.105` / `0.811` / `0.841` / `0.679` at all** — its only comma yaw number was the stale `0.000`.
IDM_V3 escalation #2 (*"`MODEL_REGISTRY.md` and `idm_head_v1_card.json` need re-issuing"*) had **not**
been actioned. It is actioned now. Likewise `Paper/` carried **no** IDM-v3 comma yaw number.

---

## 7. Deliberately NOT edited

| file | why |
|---|---|
| `Project Steering/Mission Plan.md` | ⛔ out of bounds by standing rule — **not opened** |
| `Project Steering/Reviews/2026-07-25-independent-chief-scientist-review/*` | a dated external review is a historical record; correcting it would erase what the reviewer was actually shown |
| `Project Steering/Reports/*-program-report.md` | append-only dated series; the correction belongs in the registry and here |
| `…/2026-07-26-idm-v2/PRE_REGISTRATION_IDMV2.md` | a frozen pre-registration must never be rewritten once the outcome is known. ⚠️ It contains an inference C29 **refutes** — *"B1 does not move comma yaw_rate ⇒ the yaw ceiling is NOT the label."* The ceiling **was** the label; winsorising simply was not the right repair. Recorded here instead |
| `…/2026-07-27-idm-v3/IDM_V3.md` §0 | sibling stream's document; §4.2 and §9 already carry `0.8108`. §0's `0.83` is escalated, not overwritten |

---

## 8. 🔴 Escalations — these need an owner, not a note in a file

1. ✅ **CLOSED 2026-07-27** by the `heading-default` pass. ~~The loader default is still
   `HEADING_MODE_LEGACY`.~~ `DEFAULT_HEADING_MODE` is now **`HEADING_MODE_HOLD`**, and legacy raises
   `LegacyHeadingRefused` unless the caller passes `allow_legacy=True` **and a written reason**. No
   existing cache changed meaning — legacy contributes no cache-key fragment, so every pre-flip
   comma cache dir keeps its exact name; the repair contributes one. The trainer **and** the lake
   ingestor are wired (the lake's `build_params_hash` had carried no label regime at all — it is
   exported to HF). **23 tests**, every refusal demonstrated firing.
   → `…/incoming/2026-07-27-heading-default/HEADING_DEFAULT.md` §1–§3.
   ⚠️ **A corpus rebuild is still NOT done** — the flip changes what a *new* build produces; no
   existing comma cache was rebuilt, deliberately.
2. ✅ **ACTIONED 2026-07-27** by the same pass, on the dev box with no GPU training —
   **and the result is not the anchor: the repair does NOT move R² on that substrate**
   (comma-only **+0.000048 → −0.000421**; MAE −33.3 %). See §4.1 above for the numbers, the measured
   cause (one wholly-stationary clip) and the confound that remains open. The **pooled** 9,420
   number is deliberately still **NOT re-issued** and stays stale-pending.
3. **The YouTube-IDM rotation channel was gated on a conclusion that no longer holds.** *"comma
   disqualified for yaw"* is lifted (§3.3). The owner of that line should decide whether the
   cross-class yaw question is now worth re-opening — **on repaired labels**, and reading the
   honesty condition first.
4. **`IDM_V3.md` §0 says `0.83` where its own §4.2 says `0.8108`.** Its author should reconcile the
   headline; two copies of the propagated `0.83` are already fixed.
5. **Two roundings of one number are in circulation** (`0.000` vs `0.0005`, §4).

---

## 9. Suite

| suite | result | vs the brief's baseline |
|---|---|---|
| `stack/` `pytest -q` | **1534 passed, 12 skipped**, 2 warnings | ✅ exact match, **zero new skips** |
| `taniteval/` `pytest -q` | **661 passed**, 1 warning | ✅ exact match |

Run with the project venv (`C:\Users\Admin\venvs\tanitad`); system `python` 3.14 has no pytest.
Only comments and documents were edited — no executable path was touched.
