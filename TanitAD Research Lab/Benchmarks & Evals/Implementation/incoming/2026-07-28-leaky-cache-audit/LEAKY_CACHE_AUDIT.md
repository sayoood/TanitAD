# What was scored on the 78 %-leaked cache — and what is withdrawn

**Agent:** `leaky-cache-audit` · **Date:** 2026-07-28 · **Repo HEAD:** `31134bb`
**Evidence class:** `MEASURED (ours)` unless a line says otherwise. Read-only on every cache; pod1 and
pod2 were never contacted. 🔒 Gated-confidential: counts, window masses and cache-local `ep_XXXXX` file
tags only — **no clip UUIDs, no `episode_id` values.**

---

## 0. Headline — the answer is NOT "nothing"

> **Six published result sets were scored on `physicalai-val-f1b378f295ae`, and the worst of them is a
> model card that calls its numbers "held-out".**
>
> **`idm_head_v1`'s card `val_parityval` — speed R² 0.885254, yaw R² 0.807472, steer R² 0.782080,
> ADE@2s 2.703221, n = 3,517 — is 80.0 % train-contaminated by window mass, and 40.0 % of its windows
> are bit-identical to clips the scoring head ITSELF trained on.** Only 8 of its 40 clips / 702 of its
> 3,517 windows (20.0 %) are content-disjoint from anything the encoder or the head saw.
> ⛔ **Withdrawn as a generalisation claim.**

**The thing that makes the size legible — and it has been sitting in the repo since 2026-07-25.** The
*same head* (md5 `fa4462f0…`), *same frozen encoder*, *same protocol* (k=4, stride 2, 40 clips), scored
on the content-verified **clean** split `physicalai-val-0c5f7dac3b11` (`recon_metrics.json`
`aggregate_40ep_stride2`, n=3,521 — within 4 windows of the card's 3,517):

| metric | card `val_parityval` — **80 % leaked**, n=3,517 | clean `0c5f7dac3b11`, n=3,521 | shift |
|---|---:|---:|---:|
| **ADE@2s** | **2.703221** | **3.856014** | **+42.7 %** |
| speed MAE | 2.072577 | 2.966595 | +43.1 % |
| speed R² | 0.885254 | 0.824929 | −0.0603 |
| yaw R² | 0.807472 | 0.784160 | −0.0233 |
| steer R² | 0.782080 | 0.669771 | −0.1123 |
| **long_accel R²** | **+0.081120** | **−0.184694** | **−0.2658 — SIGN FLIP: better-than-mean becomes worse-than-mean** |

**Every one of the six moves the same way, and one reverses in sign.** `VALIDATION.md` attributed this
gap entirely to *"clip selection … ADE scales with speed"* — an explanation offered without knowing that
80 % of the comparison set was memorised.
⚠️ **Stated against my own case:** this pair is matched on head, encoder, protocol and n, but **not on
episodes** — the two 40-clip sets share only 8 clips — so it does **not** isolate the leak from clip
difficulty. It is an **upper bound** on the leak effect and the only matched-head clean measurement that
exists. A clean isolate needs the head re-scored on `0c5f7dac3b11`'s windows *and* on the 8 shared
clips separately, which is a GPU job (§9.3).

**Every `taniteval` ADE and closed-loop number is CLEAR** — `runner.VAL` has pointed at the clean split
since the harness's first commit, and all 36 committed result JSONs carry `n_windows = 881`. The blast
radius is entirely in the **IDM / Branch-B / encoder-characterisation family on pod3**, which is the one
host where the leaky split is the *only* PhysicalAI val present.

**And a second class of defect, cheaper to fix and just as damaging:** four places — `MODEL_REGISTRY`
R8, `MODEL_REGISTRY` §2.2, `taniteval/registry.py:85`, `Paper/TANITAD_PAPER.md` §7.2 (plus
`DOC_CORRECTION_SWEEP.md`) — instruct a future analyst to *"re-evaluate on the **clean** `f1b378` val"*.
**They prescribe the 77.5 %-leaked split as the cure for a leak.** R4 F6 flagged this on 2026-07-25; it
has stood for three days and is still in the paper.

---

## 1. Where I looked — the coverage, not just the catch

⚠️ *Absence at one location is not absence.* Recorded as data in `search_coverage.json`.

| # | probe | scope | what it returned |
|---|---|---|---|
| 1 | literal `f1b378f295ae` / `f1b378` | whole repo, every file type | **866 occurrences in 139 files** |
| 2 | regex `physicalai-val-[0-9a-f]+` — *enumerate every val key that exists*, not just the one I was told about | whole repo | `0c5f7dac3b11` ×833 · **`f1b378f295ae` ×803** · `bb543bdf7836` ×45 · `8c0d3047924e` ×4 · `-demo` (fixture) |
| 3 | regex over **all** corpora `(comma2k19\|cosmos\|physicalai\|nuscenes\|waymo)-(val\|train\|test)-[0-9a-f]+` | whole repo | 27 distinct keys; **no second cache is content-equivalent to `f1b378`** (§2.3) |
| 4 | JSON provenance fields — `val_dir`, `val_cache`, `cache_dir`, `cache_name`, `source_dir`, `dir` — whose **value** contains the key | whole repo | 7 artifacts |
| 5 | **argument defaults** in committed commands: `--val-cache`, `--pai-val-cache`, `--val` | `stack/scripts`, `taniteval`, every `incoming/` | **11 scripts** name it as a default |
| 6 | latent-cache **tag** indirection (`va_a_*`, `va_b_*`) — the way most of the damage actually travels, since the cache name never appears in the result JSON | whole repo | 10 scripts, 5 result sets |
| 7 | downstream inheritance (`clean17`) | whole repo | 2 further artifacts re-read it |
| 8 | the sibling split `physicalai-val-heldout-79d4e3d2d4c6` | whole repo | 18 files (§5) |
| 9 | the superseded figure `62/79` / `78.5` | whole repo | 40+ files |
| 10 | **pod-side ground truth** — `ls`, rig tables | pod3 (idle, ours), read-only | pod3's `/workspace/pai_epcache` holds **only** the leaky split as its PhysicalAI val |

**Probe 6 is the one that mattered.** Four of the six affected result sets record *no cache name at
all* — they name latent tags (`va_b_00017`) written by a separate encode step. A search by cache key
alone would have found the scripts and missed the numbers.

### 1.1 How I proved I had the right episode set — twice, arithmetically

The result JSONs do not list their episodes. I reconstructed each set from pod3's rig tables and
`select_episodes`' ordering, then **checked the reconstruction against a number the artifact already
published**:

| set | windows I compute | windows the artifact records | |
|---|---:|---:|---|
| `idm_head_v1` card `val_parityval` (`va_a[:20]+va_b[:20]`) | **3,517** | **3,517** | ✅ exact |
| `idm-proof` `in_corpus_heldout_paival` (all 80) | **7,028** | **7,028** | ✅ exact |
| `idm_pipeline_derisk` `rigBval_crossRIG` (`va_b[:54]`) | **4,742** | **4,742** | ✅ exact |
| `e1a_horizon_clean17` at K=20 (17 eps) | **374** | **374** | ✅ exact |

Four independent exact matches. The episode identification is not an inference.

---

## 2. The fact, re-derived here rather than inherited

### 2.1 The overlap

Recomputed from the staged fingerprints, independently of the source agent's intersect code:

| | |
|---|---:|
| `physicalai-val-f1b378f295ae` episodes | **80** |
| overlap with `physicalai-train-e438721ae894` by `poses_sha256` | **62** |
| overlap by `frames_sha256` (raw sensor bytes) | **62** — *same 62 tags* |
| **fraction of episodes** | **77.5 %** |
| **fraction of frames** (12,328 / 15,906) | **77.51 %** |
| **fraction of windows** (IDM protocol, k=4 stride 2) | **77.5 %** |

**The honest denominator is 80, not 79** — 80 files carrying 80 distinct poses hashes and 80 distinct
frames hashes, but only **79 distinct `episode_id`s**: two files share an id. `62/79 = 78.5 %` is what
the registry carried from 2026-07-23; the content figure is **`62/80 = 77.5 %`**.

### 2.2 ⭐ The 62-by-content figure is not new today — and that matters

`PARITY_LEAK_CHECK.md` calls its 2026-07-28 result *"the first content confirmation"*. It is the
**second**. `…/2026-07-26-s3-decision-grade/disjointness_result.json` already measured
**62 / 80 = 77.5 % by sha256 of raw `poses` bytes on 2026-07-26**, and `V4_INSTRUMENT.md` re-measured
**62 of 80** on 2026-07-27. The 2026-07-28 run is the first at the **sensor** level (`frames_u8`) and
across six families — genuinely stronger, and the corroboration across three independent runs is worth
more than any one of them. **What is new is the blast radius, which is what nobody had computed.**

⚠️ Also from the 2026-07-26 file, unrelated but load-bearing: `physicalai-train-51f40f5ebc21` (the
320-episode subset) overlaps the parity train on **256 / 320 = 80 %**, by content.

### 2.3 Is any other cache content-equivalent to it? No — but 8 episodes travel

Intersecting the staged hashes of `f1b378` @80 against the deployed clean val @40:

> **Exactly 8 episodes are shared, agreeing on `poses_sha256` AND `frames_sha256`** — and **all 8 are
> among `f1b378`'s 18 content-clean episodes. Not one shared episode is train-leaked.**

This is the consistency check the two instruments owe each other, and they pass it: the deployed 40 are
content-verified train-disjoint, so any episode they share with `f1b378` *must* fall in `f1b378`'s clean
18. It also independently upgrades `VALIDATION.md`'s *"8 clips shared (pose-fingerprint dist 0.0)"* from
a fingerprint claim to a **sha256 content** claim.

No other cache reproduces `f1b378`. `physicalai-val-bb543bdf7836` (100 eps, dev box) is a different
non-parity build, already labelled as such everywhere it is used, and is **not** in scope.

---

## 3. Every hit, its claim, and the verdict

Full machine-readable form: `hits_inventory.json`. Mass table: `leak_mass_by_result_set.json`.

### 3.1 ⛔ INVALIDATED — generalisation claims scored on contaminated windows

| # | artifact · node | claim | leaked clips | **leaked window mass** | verdict |
|---|---|---|---:|---:|---|
| **A1** | `…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` · `val.val_parityval` | speed R² **.885254**, yaw R² **.807472**, steer R² **.782080**, long_accel R² **.081120**, MAE 2.072577, **ADE@2s 2.703221**, n=3,517 | **32 / 40 = 80.0 %** | **2,815 / 3,517 = 80.0 %** · ⛔ **1,407 / 3,517 = 40.0 % in the HEAD's OWN train clips** | ⛔ **WITHDRAWN as held-out** |
| **A2** | `…/2026-07-22-idm-proof/results.json` · `physicalai_to_comma2k19.val.in_corpus_heldout_paival` | speed R² .929725, **yaw R² .924424**, steer R² .857885, ADE@2s 2.733018, n=7,028 | **62 / 80 = 77.5 %** | **5,448 / 7,028 = 77.5 %** (23.8 % in the head's own clips) | ⛔ **WITHDRAWN as held-out**; it is also the denominator of `ade_ratio` |
| **A3** | `…/2026-07-24-idm-pipeline-derisk/results_idm_pipeline_derisk.json` · `rigBval_crossRIG` | zero-shot cross-rig, labelled ***"rigB-val (episode-disjoint)"***, n=4,742 | **41 / 54 = 75.9 %** | **3,600 / 4,742 = 75.9 %** (27.8 % in the head's own clips) | ⛔ **the "episode-disjoint" label is FALSE** |
| **A6** | `…/2026-07-24-branchb-transfer-eval/results_branchb_transfer_e50_CONVERGED.json` + `MODEL_REGISTRY §10.1` · `rig_val`, `multirig_val` | paired ΔR² **−0.755 [−1.336, −0.108]** and **−1.325 [−2.295, −0.801]**, published as ***"clean, disjoint"*** and ***"episode-leakage controlled"*** | rig-A **21 / 26 = 80.8 %** · rig-B **41 / 54 = 75.9 %** | rig-A **1,848 / 2,286** · rig-B **3,600 / 4,742** | ⛔ **the labels are FALSE; the paired ΔR² is CONFOUNDED** (§4) |

### 3.2 ⚠️ AFFECTED — ratio-structured, partly protected, but not clean

| # | artifact | claim | leaked mass | verdict |
|---|---|---|---:|---|
| **A4** | `…/2026-07-24-idm-downstream-ablation/…json` · `domains.rigB_sameClass` | pseudo-benefit 2.4346, ceiling 2.5317, **fraction_of_ceiling 0.960** | **3,600 / 4,742 = 75.9 %** | floor/pseudo/ceiling share one substrate, so a common-mode inflation **partly divides out of the ratio** — the absolute per-arm R² and the phrase *"held-out real-labeled test"* do not survive |
| **A5** | `…/2026-07-24-idm-parity-validation/…json` · all arms, `fraction_of_ceiling`, verdict **"GO-decision-grade"** | speed 1.092 / yaw 0.710 / ADE 1.074 of ceiling | **5,448 / 7,028 = 77.5 %** | same structure — **but this one carried a GO decision**, and the word *"validation"* in its name is doing work it cannot support |
| **A7** | `stack/scripts/run_v1_encoder_char.py` (`v1-encoder-char`) | encoder characterisation on `va_b[:54]` | **75.9 %** | affected by construction; **no committed result JSON in-repo** → which numbers it published is **UNVERIFIED** |
| **A8** | `…/2026-07-24-youtube-idm-pilot/run_youtube_pilot_downstream.py` | pilot downstream on `VA`/`VB` | affected by construction | pod-side artifacts only; **UNVERIFIED** which numbers reached a doc |

### 3.3 ✅ NOT INVALIDATED — and each is a measured clearance, not an assumption

| # | artifact | why it stands |
|---|---|---|
| **A9** | `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_clean17.json` (374 windows, 17 eps, `val_dir = …f1b378f295ae`) | The 17 were selected by **`episode_id`** — an inadmissible key — but **in this cache id-clean is a strict subset of content-clean**: all 62 content-leaked files also match by id (`episode_id_agrees` true for every one of the 62), so a file whose id is absent from train **cannot** be content-leaked. Whichever 17 of the 18 content-clean files were used, **all 17 are content-clean.** ⚠️ It was **luck, not method** — C43's comma case is precisely a leak with *zero* name-level evidence. Residual: it used **17 of the 18** available clean episodes; the id key is blind to the two files sharing one id, costing **1 episode / 22 windows (5.6 % of the clean episodes available)**. E1a's *headline* (the C6 horizon retraction) is on a different cache entirely. |
| **A10** | `…/2026-07-21-vlm-production-semantic/` — 12 artifacts, episodes 40–79 (**31 / 40 = 77.5 % leaked**, 2,722 windows) | It is a **VLM output-quality audit**: `parse_failure_rate 0.0`, `enum_violation_rate 0.0`, `truncation_rate 0.0`, evidence `unique_rate 1.0`, and the finding that the `confidence` field is **degenerate** (one value on 100 % of windows). Cosmos-Reason2-8B never trained on our parity corpus, so "overlap with `e438721ae894`" is not a property that can bias any of these. This is the sanctioned `note_leaky_audit` class. **Flagged, not investigated:** if any label produced on those 31 leaked windows was later consumed as a *training* label, that is a train-side question. |
| **A11** | `route_label_audit.py`, `vlm_route_labels.py`, `vlm_kin_crossval.py`, `taniteval/label_overlay.py` | Label scoring, no model metric. Since 2026-07-25 they call `parity.note_leaky_audit(...)` and stamp `decision_grade: False`. |

### 3.4 ✅ RULED OUT — the places it never reached

| what | why | class |
|---|---|---|
| **Every `taniteval` decision-grade ADE and closed-loop number** | `runner.VAL` = `/root/valdata/physicalai-val-0c5f7dac3b11` since the harness's **first** commit (`a91bef8`); `git log --all -S f1b378f295ae -- taniteval/` hits only `label_overlay.py`, a video renderer. All 36 committed result JSONs carry `n_windows = 881`. | MEASURED (`VAL_PARITY_REPORT.md` §4.1–4.2), re-read here |
| **The `sorted(glob("*val*"))[-1]` swap ever firing** | The two split dirs never co-exist under any epcache root on pod1/pod2/pod3/eval. Re-confirmed by us 2026-07-28: pod3 holds the leaky split **alone**. | MEASURED |
| **idm-v2 / idm-v3 — `A0`, `R0`, `V2R`, `V3F`, and `physicalai_yaw_r2 +0.903482`** | Both pre-registrations name `physicalai-val-0c5f7dac3b11`; `idm2_encode.py` reads `/root/valdata/physicalai-val-0c5f7dac3b11`. **C29's forward pointer "PhysicalAI is unaffected, +0.903482" is on the clean split and stands.** | MEASURED |
| pod2's small validation, the 600-episode build | `f1b378` is not on pod2 at all; 0/600 by content | MEASURED |

---

## 4. The one place the leak's DIRECTION was called wrong

`MODEL_REGISTRY §10.1` has argued since 2026-07-25:

> *"the leak inflates **both** arms' val R² equally … so the WORSE-Branch-B ordering is **conservative,
> not manufactured**."*

**That is not established, and the asymmetry points the other way.**

- **flagship-v1**, the paired control, is the **parity-trained** encoder. It **provably** trained on the
  62 leaked episodes — MEASURED, bit-identical.
- **Branch B** trained on a **different** 2,466-clip set (`MODEL_REGISTRY` §10.1: *"Never the parity key
  `e438721ae894`"*). Its content overlap with `f1b378` is **UNMEASURED**.

Both encoders are frozen and a fresh IDM head is fitted on identical windows, so an encoder that
memorised the evaluation episodes yields more informative latents on exactly the 75.9–80.8 % of windows
being scored. **A memorisation advantage for the control is a live candidate explanation for the very
gap the paired ΔR² reports.** The mitigation is withdrawn, not softened.

**What survives, stated precisely:**

1. ✅ **Finding 1 survives.** Branch B's own cross-rig speed R² is **−0.667** against a **+0.9** gate. An
   absolute failure by 1.57 R² does not need a clean control.
2. ✅ **Finding 2 survives.** In-domain rig-A and Branch B's own 40 k head in-sample on rig-B (**0.24**)
   are not measured on this substrate.
3. ⛔ **Finding 3 — the paired ΔR² on `rig_val` and `multirig_val` — is CONFOUNDED** and must not be
   quoted as a clean contrast until Branch B's own overlap with `f1b378` is measured by content. That is
   a cheap probe: the fingerprints for `f1b378` are already staged; only Branch B's clip list is needed.

---

## 5. `episode_id` is now measured wrong in BOTH directions — every such claim, flagged

⛔ **An "episode-disjoint" claim resting on `episode_id` is not evidence.** Both error directions are now
measured on our own corpus:

| direction | measurement |
|---|---|
| **false negatives** — content leak with no name-level trace | C43: **2 of 22** comma evaluation episodes bit-identical to training clips, *no* filename/tag/id comparable |
| **false positives** — id collision with zero shared content | `physicalai-val-0c5f7dac3b11` @600: **20 of 600** share an id with a train episode and share **no content whatsoever** |
| **not even a key** | the parity train has **2,342 distinct ids for 2,376 episodes** — 33 ids reused across 67 episodes |
| **wrong denominator** | `f1b378`: 80 files, **79 distinct ids** → `62/79 = 78.5 %` instead of `62/80 = 77.5 %` |

Every `episode_id`-founded disjointness claim I encountered, flagged whether or not it turned out true:

| claim | key | status |
|---|---|---|
| `physicalai-val-heldout-79d4e3d2d4c6` is 0.0 % overlapping — **the substrate of E1a's headline, E1b and E1c** | built by `episode_id ∉ train ids` | ⚠️ **the claim happens to hold** — subsequently content-verified **0/44 at the POSES level** (`disjointness_result.json`, 2026-07-26). ⛔ **`frames_u8` was never checked**, and the *key that built the split* is still not evidence. **UNVERIFIED at sensor level.** |
| `f1b378` 62/79 (78.5 %) | `episode_id` | corroborated at 62 by content, **wrong denominator** |
| `physicalai-val-0c5f7dac3b11` @600 "episode-disjoint" | `episode_id` would report **20 overlaps** | content says **0** — the id key would have manufactured a 20-episode leak |
| `run_branchb_transfer.py` docstring + `RESULTS_branchB.md`: *"`_val` … episode-disjoint"* | assumed from the directory name | ⛔ **FALSE**, 75.9–80.8 % leaked |
| `results_idm_pipeline_derisk.json`: *"rigB-val (episode-disjoint)"* | assumed | ⛔ **FALSE**, 75.9 % leaked |
| `idm_head_v1_train.py`: *"two EPISODE-DISJOINT val sets that were never in train"* | assumed | ⛔ **FALSE** for `val_parityval`: 80.0 % leaked, 40.0 % in the head's own clips |

---

## 6. The registry correction — applied

**Edited in `Project Steering/MODEL_REGISTRY.md`, four places. The superseded value is preserved inline
with its date, and the upgrade from an id claim to a content claim is stated.**

| § | before | after |
|---|---|---|
| §10.1 Leakage bullet | *"62 of its 79 populated episodes (78.5 %)"*, `episode_id` intersection, **2026-07-25** | **62 of 80 = 77.5 %, MEASURED BY CONTENT** (6 families incl. raw `frames_u8`; frame mass 12,328 / 15,906) + the superseded figure and its date kept verbatim + the reason the denominator was wrong + the C43 both-directions rule + the withdrawn direction argument (§4) + per-rig leaked mass |
| §10.1 provenance line | *"val rigA 26 / rigB 54, **episode-disjoint**"* | struck; replaced with **rig-A 21/26 (80.8 %), rig-B 41/54 (75.9 %) bit-identical to parity-train episodes** |
| §10.1 results table | `rig_val` / `multirig_val` labelled ***"(clean, disjoint)"*** | struck; **"75.9 % LEAKED"**, with a note that the paired ΔR² is confounded and which findings survive |
| §7 R8 | *"re-evaluate on the `f1b378` val before any comparative claim"* | 🟥 **the inverted remedy is corrected** — points at the content-verified clean split, and names the **three other files carrying the same inversion** |
| §2.2 REF-A ijepa-4b | quotes `registry.py`'s *"clean number lives on the f1b378 val"* | quote left intact (it is a quote), **flagged inline** with the measured 77.5 % and a pointer to the source that must be fixed |

⛔ **Nothing was re-selected, re-registered or re-scored. Parity is untouched.** No `overlapping_holdout_se`
number is quoted anywhere in this document. Nothing is held to v1's 0.4271 (`wm_fidelity_ade_2s`, a
different question).

---

## 7. Withdrawn — stated plainly

**Yes. Published claims are withdrawn.**

1. ⛔ **`idm_head_v1`'s entire `val_parityval` block** — speed R² 0.885254, yaw R² 0.807472, steer R²
   0.782080, long_accel R² 0.081120, speed MAE 2.072577, **ADE@2s 2.703221** — **is withdrawn as a
   held-out / generalisation number.** 80.0 % of its mass is bit-identical to the frozen encoder's
   training corpus; 40.0 % is bit-identical to the scoring head's own training clips.
   ⭐ **It survives as an in-sample reproduction fingerprint** — which is exactly what `idmval_vacheck.py`
   used it for ("reproduces the card to every digit"), and *that* control remains valid: it proves the
   weights and the metric code are intact. Reproducing a number is not validating it.
2. ⛔ **`idm-proof`'s `in_corpus_heldout_paival`** (yaw R² **+0.9244** among them) — **withdrawn as a
   held-out number**; 77.5 % contaminated. Its `ade_ratio = 2.3988` is not a quotable magnitude. The
   `PASS = false` verdict is **not** withdrawn — the contamination direction makes it conservative.
3. ⛔ **The "episode-disjoint" / "clean, disjoint" / "leakage-controlled" labels** on A3, A5, A6 and the
   `idm_head_v1` card **are withdrawn as statements of fact.**
4. ⛔ **The Branch-B paired ΔR² on the two `*_val` rows is withdrawn as a clean contrast** (§4). Branch
   B's absolute gate failure is **not** withdrawn.
5. ✅ **Not withdrawn, and it would have been an error in the other direction to withdraw them:** every
   `taniteval` ADE, every closed-loop number, `e1a_horizon_clean17`, the whole VLM label-audit family,
   and idm-v2/v3's `A0`/`R0`/`V3F` including **`physicalai_yaw_r2 +0.903482`**. Four of the six IDM
   result sets are on the leaky cache; two are not, and the difference is measured, not assumed.

**Training facts are untouched throughout** — weights, `weights_md5 fa4462f0b898b036be729c790278b823`,
params 2,899,724, recipes, step counts, the parity key. A val leak invalidates generalisation claims,
not training facts.

---

## 8. Escalations — decisions, not README lines

1. 🟥 **The inverted remedy is still live in 3 files I did not edit** — `taniteval/registry.py:85-87`,
   `Paper/TANITAD_PAPER.md` §7.2, `…/2026-07-26-doc-correction-sweep/DOC_CORRECTION_SWEEP.md`. Each
   tells a future analyst to fix a leak by evaluating on the leaked split. **R4 F6 flagged this on
   2026-07-25 and it is still in the paper.** Code and paper owners.
2. 🟥 **`idm_head_v1_card.json` needs a leak block.** The card's own `provenance.evidence_class` reads
   *"val metrics below are held-out"* and its `per_number_verdict` says
   *"val.val_parityval.r2.yaw_rate = 0.807… **NOT STALE** — PhysicalAI-only. Do not re-issue."* That
   verdict was about the **heading repair** and is correct in its own scope — but it now reads as a
   clean bill of health for a number that is 80 % train-contaminated. Same for
   `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` §5 *"NOT stale — 14 locations checked and left
   alone"*, which lists three leaky-substrate numbers as *"correct"*.
3. ⚠️ **One cheap probe closes §4:** hash Branch B's own 2,466 training clips against the staged
   `f1b378` fingerprints. Fingerprints already exist; only the clip list is needed. Until then the
   Branch-B paired ΔR² has no clean reading.
4. ⚠️ **`physicalai-val-heldout-79d4e3d2d4c6` is content-verified at the POSES level only.** It carries
   E1a's headline, E1b and E1c. `frames_u8` was never checked on it, and it was *built* by the key C43
   disqualifies. One re-run of the existing `fingerprint_pai_cache.py --mode full` settles it.
5. ⚠️ **Two result families have no committed metric JSON** (`v1-encoder-char`, `youtube-idm-pilot`
   downstream). Which numbers they published is **UNVERIFIED** — that is a stranding problem as much as
   a leak problem.
6. ⚠️ **`PARITY_LEAK_CHECK.md`'s "first content confirmation" should read "first at sensor level"** —
   `disjointness_result.json` got there by content on 2026-07-26 (§2.2). Minor, but it is a provenance
   claim and this program has been burned by those.
7. ⚠️ **The superseded `62/79` / `78.5 %` figure is in 40+ files.** Substance unchanged; only the
   denominator moves. A mechanical sweep, listed in `hits_inventory.json` §C.
8. ⚠️ **`taniteval/` is NOT green in the working tree, and it is not this audit's doing — cause proven.**
   `tests/test_control.py::test_a_pure_offset_moves_placement_and_NOT_heading` fails (**1 failed, 725
   passed**; briefed 726/0). The tree carries uncommitted sibling edits to
   `taniteval/taniteval/control.py` and `taniteval/taniteval/pseudosim.py`; substituting **HEAD's**
   versions of exactly those two files into an otherwise identical tree runs the same file **34/34
   green**. The closed-loop control/recovery workstream owes the fix or an updated assertion.
   `stack/` is **1,576 passed / 12 skipped — exactly as briefed.**
   *(Second occurrence of this pattern in two days: `PARITY_LEAK_CHECK.md` §8.4 reported a `taniteval`
   edit breaking a `stack` test yesterday. A cross-suite guard would catch it at the edit.)*

---

## 9. What this does NOT establish

1. **Only PhysicalAI, only these two caches.** `physicalai-val-bb543bdf7836` (100 eps, dev box) is a
   different non-parity build and was **not** checked against any training corpus here.
2. **The 32/40 and 16/40 counts in A1 rest on reconstructing the clip set** from pod3's rig tables and
   `select_episodes`' documented ordering. **The reconstruction is verified by an exact window-count
   match (3,517 = 3,517)** — but it is a reconstruction, not a stamped provenance record. The fix is
   the one `VAL_PARITY_REPORT.md` §4.4 already shipped for `taniteval`: stamp the val cache into the
   result JSON. `run_idm_*` still does not.
3. **I did not re-score anything on the clean split.** Every "what would it read clean" number here is
   either already published (A1's 3.856) or absent. Re-scoring the IDM family on
   `physicalai-val-0c5f7dac3b11` is the obvious follow-up and is a GPU job.
4. **The 8 clean clips in A1 are too few to re-issue a number on.** 702 windows over 8 episodes is not a
   substitute measurement, and sub-selecting them would be a **re-selection**, which is refused.
5. **REF-A's I-JEPA leak vs its own 320-episode subset remains UNVERIFIED** — a different overlap.
   *(Related and MEASURED 2026-07-26: `physicalai-train-51f40f5ebc21` @320 overlaps the parity train on
   256/320 = 80 %.)*

---

## 10. Evidence classes

| claim | class |
|---|---|
| 62/80 = 77.5 % overlap; 12,328/15,906 frames; the 18 clean tags | **MEASURED (ours)** — recomputed from `…/2026-07-28-parity-leak-check/raw/hashes_*.json` + `KNOWNPOSITIVE_*.json` |
| 8 episodes shared between `f1b378` and the deployed 40, all content-clean | **MEASURED (ours)** — set intersection of `poses_sha256` and `frames_sha256` |
| the `val_parityval` / `paival` / `rigBval` clip sets and their window counts | **MEASURED (ours)** — pod3 rig tables (read-only) + `select_episodes` ordering, each verified by an exact window-count match against the artifact |
| 32/40 clips and 2,815/3,517 windows leaked; 16/40 and 1,407/3,517 in the head's own train set | **MEASURED (ours)** |
| ADE@2s 2.703 (leaky) vs 3.856 (clean), same head | **PUBLISHED (cited)** — `idm_head_v1_card.json`, `VALIDATION.md`, `recon_metrics.json`; the *attribution* of the gap is ours |
| `taniteval` ADE path never used the leaky split | **MEASURED** (`VAL_PARITY_REPORT.md` §4.2, `git log --all -S`), re-read here |
| the two split dirs never co-exist under one root | **MEASURED** (`ADDENDUM_coexistence_probe_RESOLVED.md`, 2026-07-25) + **MEASURED (ours)** for pod3, 2026-07-28 |
| `heldout-79d4e3d2d4c6` is 0/44 by poses sha256 | **MEASURED** (`disjointness_result.json`, 2026-07-26) — **not re-verified here, and NOT checked at sensor level** |
| Branch B's own overlap with `f1b378` | ⛔ **UNVERIFIED** — the gap that keeps §4 from being resolvable today |
| which numbers `v1-encoder-char` / `youtube-idm-pilot` published | ⛔ **UNVERIFIED** — no committed result JSON |
