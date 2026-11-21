# `DATA_STRATEGY.md` refreshed to v4.0 — and the brief's own premise was wrong in three places

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **GPU consumed: zero** ·
**Pods touched: none** · **Thor touched: not at all, not even read-only**

**Deliverable:** `DataEng/DATA_STRATEGY.md` **v4.0** (in place, supersedes v3.0 of 2026-08-17).

---

## 0. TL;DR

1. ⛔ **The brief said the file was "a month stale". It was ONE DAY stale.** It is at **v3.0,
   2026-08-17**, not the v1.0 of 2026-07-06 that
   `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` complained about. The
   "month stale" phrasing is a **correct 2026-08-15 observation, inherited forward past its own
   repair.** *Same class as every stale-blocker case in this programme: a true statement whose
   subject moved.* The refresh was still needed — v3.0 was overtaken inside 24 h — but a reader
   given "a month stale" would have expected to find far less than was there.
2. ⛔ **The brief said the SAM3 perception corpus is "published gated-public". MEASURED: it is
   not published at all.** `pushed_to_hf: false` on both the unified 201-clip corpus and the 86
   re-detected records; only the older **115-clip leg** is on HF, in a **PRIVATE** (not gated-public)
   repo. **The unified corpus lives on one disk on the dev box.** That is the definition-of-done
   failure the operating standard exists to prevent, and it is now §12 row 5 of the strategy.
3. ⚠️ **The brief said "102 stranded / 117 rescued" without the reconciliation, and `BACKLOG.md`
   A11 says "98".** Both are right about different quantities: **102 stranded by content-hash in the
   A11 roots → 98 pulled (4 left with reasons) + 19 gate/summary JSONs found beyond those roots =
   117 files.** I re-counted the banked tree: **98 + 19 = 117 files.**
4. ⭐ **Every load-bearing parity number in v4.0 is MEASURED BY ME, re-derived in-repo through the
   committed oracle** — not copied from `contamination.json` and not from prose. Script + banked
   JSON in this package; `ALL_CLAIMS_REPRODUCED: true`.
5. ⭐ **One of v3.0's open items closed by measurement, two by execution, and I verified the code
   fix at HEAD myself** rather than inheriting the claim.

---

## 1. What I re-derived rather than inherited

`code/verify_strategy_claims.py` → `raw/strategy_claims_verified.json`. **Zero GPU, zero pod, ~2 s.**
Run: `PYTHONUTF8=1 python code/verify_strategy_claims.py --json raw/strategy_claims_verified.json`
*(`PYTHONUTF8=1` is required on this box for the same reason `build_obstacle_join.py` needs it: the
default cp1252 stdout encoder dies on the artifacts' non-ASCII characters. It cost me one run.)*

### 1.1 ⭐ Set IDENTITY, not a matching count

**MEASURED.** Three files, `parity.uid_digest` (sha256 over the newline-joined sorted ids):

| file | n | sorted-id sha256 |
|---|---:|---|
| `…/2026-08-17-aug120-refuse/raw/fused_aug120_v2_index.jsonl` | 201 | `80632f17292eb5fc484956338a0aad7b40f91ff66502faa11a5cad49f9a0439e` |
| `…/2026-08-17-perception-floor-unify/raw/fused_aug120_v3_index.jsonl` | 201 | *(identical)* |
| `…/2026-08-17-thor-concurrency-pilot/alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` | 201 | *(identical)* |

`all_three_identical: true` · `set_equality: true`. Through the oracle: the cohort is
**201/201 inside parity train** and **0/201 inside the deployed val**.
⇒ This is the C113 half of the correction reproduced from primaries: **the aug120 perception cohort
does not *coincide with* the exclusion list — it IS it.**

### 1.2 ⭐ The three contamination rates, computed not copied

| rate | MEASURED | matches `contamination.json` |
|---|---|---|
| catalogue | **201 / 4,729 = 0.042504** | ✅ |
| **buildable today** | **201 / 257 = 0.782101** | ✅ |
| **deployed val swallowed** | **6 / 40 = 0.1500** | ✅ |

Digest sets loaded and self-checked on read: `parity_train_clip_digests.json` **2,400**,
`deployed_val40_clip_digests.json` **40**.

### 1.3 The feature read-set, by layer

**MEASURED** from `stack/tests/test_physicalai_feature_readset.py`: **2 / 5 / 6 of 36**, program-wide
names `camera_front_wide_120fov, camera_intrinsics, egomotion, obstacle.offline, sensor_extrinsics,
vehicle_dimensions`. **I did not hand-write the number into the strategy** — v4.0 carries the
three-row table and points at the pin.

### 1.4 Pinning tests re-run on this checkout

| test file | result |
|---|---|
| `stack/tests/test_physicalai_feature_readset.py` | **9 passed** |
| `stack/tests/test_eval_contamination.py` | **17 passed** |

*(Run with the `tanitad` venv's interpreter. ⚠️ `python -m pytest` on the system Python 3.14 gives
`No module named pytest` — that is an interpreter-selection error, not a missing dependency, and it
would read as a broken suite.)*
⚠️ **I did NOT run the full `stack` suite.** Per C114, a full run taken while sibling agents edit the
tree measures a **torn snapshot**, not the code — and three agents are live in this tree. The two
files above are the ones this document's claims depend on.

### 1.5 ⭐ The VLM `lon3` fix, verified at HEAD from source rather than inherited

v3.0 listed *"fix the VLM `lon3` mapping"* as an unassigned open item. **MEASURED, from
`stack/scripts/ph1_fuse.py` at HEAD:** the substring tables are gone, replaced by
`VLM_VERB_TO_A_TAC` — an explicit **total dict over a closed vocabulary** that **raises
`UnmappedActionVerb`** on an unknown key, with `hold_corridor → ("LANE_KEEP", NO_CLAIM)` and
`reduce_to → (NO_CLAIM, "BRAKE_TO")` — i.e. defects (1) and (2) fixed *at the mechanism*, plus four
declared sentinels separating "said nothing" from "spoke but untypeable" from "no v6 token".
⇒ **v4.0 records this as FIXED and narrows the open item to the half that is still open** (the
shared-input non-independence, §5.3b), rather than carrying a closed item forward.

### 1.6 ⛔ No credential scanner exists — three probes

C111's rule is *"any bulk import is SCANNED FOR CREDENTIAL PATTERNS BEFORE IT IS STAGED"*.
**MEASURED, three independent probes, all negative:**

| probe | result |
|---|---|
| `stack/scripts/` + `stack/tests/` by name (`secret|scan|cred|token`) | only `rig_band_scan.py` (unrelated) |
| repo-wide grep `detect-secrets\|trufflehog\|secret.scan\|credential.pattern` over `.py/.sh/.yml/.yaml/.toml/.cfg` | **zero hits** |
| repo root for `pre-commit`/`gitleaks`; `.github/workflows/`; `AGENT_OPERATING_STANDARD.md` | no pre-commit config; workflows are `pod-exec.yml` + `pod-telemetry.yml`; the standard carries exactly one line, and it is the `Keys.txt` invariant |

⇒ **The rule is stated and unimplemented.** Filed as strategy §12 row 14, alongside the
time-sensitive PI action: **the token is still in plaintext on Thor**, so rotation is not optional.

### 1.7 ⚠️ Absence at one location is not absence — it caught me once here

Checking the `parity.py` §9 confidentiality contradiction, my **first** `git ls-files` probe used
`…/2026-08-17-thor-concurrency-pilot/raw/alpamayo_clip_ids.txt` and returned **empty** — which would
have supported *"the escalation is stale, the files are gone"*, the weaker and wronger conclusion.
The files are at the **package root**, not under `raw/`. Second and third probes (`git ls-files` by
basename; filesystem `find`) found both, **tracked**: `alpamayo_clip_ids.txt` **4,729 lines** and
`alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt` **201 lines** = **4,930 raw PhysicalAI-AV clip ids
in plaintext**, against `parity.py` §9's stated *"the repo carries only the digests"*.

---

## 2. What v4.0 marks as WITHDRAWN — and what still stands

⛔ **The brief's instruction was to mark withdrawal explicitly and to say what survives.** Over-
correction has cost this programme once already, so each row states both halves.

| v3.0 said | v4.0 status | what still stands |
|---|---|---|
| §4.3 *"the last two disagree and it is UNRESOLVED… the honest statement is 'not on aug120', not 'G1 was wrong'"* | ⛔ **WITHDRAWN** (C87). It was an **instrument** — G1's cropper: 0/54 tiles are a tight crop, 45 padded to a ~96 px floor, 5 are the whole frame, none outlined. 2.7 % vs 88.9 % **on the same detections** | G1's ~71 % is **not withdrawn** and G1 **read its crops correctly**. Sign **KIND and TEXT stay forbidden** — the two highest-scoring FPs are a dashboard `30` roundel (0.927) and a hoarding (0.778), i.e. above true signs |
| §4.3's third population, **2/23 ≈ 9 %** | ✅ **UNTOUCHED by C87** — it is a *different instrument* (VLM-box ↔ SAM3-box **location**, PH0 pilot frames) | **B3 stays DEMOTED to diagnostic-only** on that basis. v4.0 warns explicitly against merging it with the SAM3 content rows |
| §5.1 *"corroborations 88 / conflicts 10 are stale in BOTH directions… do not quote them; re-fuse first"* | ⭐ **DISCHARGED** — the re-fuse happened; the unified corpus measures **88 / 10** | ⭐ **The withdrawal was still correct**: it was withdrawn as *uninterpretable* (2 of 6 checks could not fire), not as known wrong. And it still **must not** be read as independent-observer agreement while §5.3b stands |
| §5.4 *"Re-fuse is a work item with no owner and no date"* | ✅ **CLOSED.** `_provenance.vlm` lie **201/201 → 0**; `perception.absent` **115 → 0**; prose "no agents" **119 → 0** | ⚠️ any **archived** pre-re-fuse copy still carries the false vision-only claim |
| §1.1 *"the F-18 slot probe already returned NEGATIVE (D1)… this join does not change that result"* | ⛔ **WITHDRAWN** — D1 is withdrawn; the probe failed its own positive control (6.319 m on a tensor a ridge reads at 1.016 m, r = +0.979) | The join itself is unaffected and **still MEASURED**: 2,308 eps / 433,040 frames / 12,122,129 boxes. What changed is that the re-read can now run on the **full parity corpus** |
| §5.1 *"SAM3 records available 201/201"* | ⚠️ **TRUE BUT MISLEADING as written** — 115 at floor 0.25 and **86 still at 0.5**. Now **201/201 at ONE floor, ONE schema, residual 0** | The v1-vs-v2 non-pooling rule was correct while it applied, and is **LIFTED** only now |
| §5.3(a) VLM `lon3` is *"a code fix, not a re-labelling job"* | ⭐ **the fix LANDED** — verified at HEAD by me (§1.5) | the historical κ = 0.0000 measurement stands as the record of what was wrong |
| §0 (443×), §7 (augmentation ≠ S-W fix), §8 (parity sacred), §9 (firewall + license conflict), §10 (flywheel), §3 (A2 labels complete) | ✅ **UNCHANGED, and said so explicitly in v4.0** | — |
| §6's *"≈6.8 h at 8 shards, ≈179 GB"* | ⚠️ **NOT withdrawn — it is a DIFFERENT QUANTITY.** That is local decode/encode + output size; the new 1.73 TB / 41.8 h is **HF egress**. v4.0 puts both in one table and labels the axis | the 19.4 s/clip basis is unchanged |

⛔ **Where the document stated a plan that measurement has overtaken, v4.0 says so rather than
deleting it** — per the brief. Examples kept in place with their date and their fate: v1.0's
60/25/15 training mix (*"never shipped and is withdrawn"*), the labelling→extraction
reclassification of 2026-08-16, and now extraction→*extraction-behind-a-ruling* (§3.1, §6.1).

---

## 3. What is genuinely new in v4.0

| § | content |
|---|---|
| **§1.2** | ⛔⛔ the parity contamination, both directions, with the buildable-rate correction and the blast-radius statement |
| **§1.2b** | ⭐ the membership oracle: the two call sites, the committed digest files, `filter_*` vs `assert_*`, the `sanctioned_audit` escape hatch, counts-only disclosure, what the mint refuses, and the pin |
| **§4.1b** | ⭐ the floor unification; the guard that **fired and was left firing**; the dev-box GPU finding; Triton refuted; the pyarrow-after-torch segfault; ⛔ **the one-disk risk** |
| **§4.3** | ⭐ C87 rewritten from "unresolved" to "resolved, and it was the instrument" — with the prereg amendment stated |
| **§6.1–§6.4** | the mandatory `filter_train_clips()` pre-build call; the measured concurrency result and the abort criterion that could not fire; the density-skew cost table; the launch-path defect and why the crash was the good outcome |
| **§11** | ⭐ **new section** — standing data-engineering hygiene: credential scanning (C111), *a census is a claim about the filter* (C110), and the verification discipline (C77/C18, C87, positive controls, set-equality) |
| **§12** | four new rows (13–16), one closed (9), one replaced (5), two sharpened (2, 6) |
| header | the version-citation warning, and an explicit note that v3.0's §11 is now §12 |

---

## 4. ⛔ Flagged, not fixed — owned by other streams

| item | owner | why I did not touch it |
|---|---|---|
| **`MODEL_REGISTRY.md` §11.2 is stale in three places** — SAM3 coverage still **86/201 = 42.8 %** with *"115 clips carry NO SAM3 record"*; the retired 2-of-3 estimator still quoted as **178/201 LAT / 61/201 LON** (corrected: **115/201 LAT, 0/201 LON**); "Known gaps" item 1 still names the 115-clip hole as open | **registry owner** | `BACKLOG.md` F9 states registry rows are **registry-owner only**. ⭐ §12.4 **is** current — cite that one for parity |
| **`BACKLOG.md` A11 says "98" where C110 says 102/117** | the backlog owner | both are right about different quantities; the row would read more clearly with the split spelled out (§0 item 3) |
| **`parity.py` §9's confidentiality sentence is FALSE as written** (§1.7) | **PI** — it is a gating-compliance question, and deleting the files breaks the eval-contamination pin | already escalated by the parity package; I only re-verified it and recorded it in the strategy |
| **The unified perception corpus is unpushed** | **PI** — a write to a public-facing platform | the owning agent declined on the same ground; I agree and did not attempt it |
| **The 476-clip w120 pilot corpus (18.33 GB) on Thor**, and BACKLOG **A14**'s 24 un-pulled dumps | their producing streams | recorded risks with staged recipes, not stranded artifacts |
| **`SAM3_CONCEPT_RELIABILITY.md` §4.1** superseded by C87 | its author | that package asked for it to be **annotated, not rewritten**, and changed not one byte of it itself |

---

## 5. Two cautions for anyone quoting these packages

1. ⚠️ **Three different `stack` suite totals appear across this week's packages** — 3,763 · 3,770 ·
   3,782 (and the parity package reports 3,935 with 3 pre-existing failures). They are different
   points in time with several agents in one tree, and each document pairs its own delta.
   **Cite a suite number with its document, never bare.** (C114.)
2. ⚠️ **"Detections" and "tracks" are different units and both appear as per-concept tables** — e.g.
   `traffic sign` on the unified 201 is **4,869 detections** vs **4,357 tracks**. At stride-6 a
   "track" ≈ a detection and is **not** an object count; 87.7 % of tracks on the v2 leg are
   single-frame. State the unit.

---

## 6. Deliverable manifest

**Everything below is in the repo working tree and STAGED. Nothing is on a pod, in a worktree, or in
a scratchpad. Nothing was pushed.**

| artifact | path (repo-relative) |
|---|---|
| ⭐ **the refreshed strategy (v4.0, in place)** | `DataEng/DATA_STRATEGY.md` |
| this record | `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-18-data-strategy-refresh/DATA_STRATEGY_REFRESH.md` |
| re-derivation script (zero GPU, zero pod, ~2 s) | `…/2026-08-18-data-strategy-refresh/code/verify_strategy_claims.py` |
| its banked output | `…/2026-08-18-data-strategy-refresh/raw/strategy_claims_verified.json` |

**Nothing else was modified.** No registry row, no backlog row, no test, no source file — every
correction I found in a file owned by another stream is in §4 above and in the strategy's §12, named
rather than edited.

**Reproduce (zero GPU, zero pod):**
```
cd "<repo>/TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-18-data-strategy-refresh"
PYTHONUTF8=1 <tanitad-venv>/python.exe code/verify_strategy_claims.py --json raw/strategy_claims_verified.json
cd "<repo>/stack" && <tanitad-venv>/python.exe -m pytest \
    tests/test_physicalai_feature_readset.py tests/test_eval_contamination.py -q
```
Expected: `ALL_CLAIMS_REPRODUCED: true`, exit 0; and `26 passed`.

⛔ **Escalation, not a doc line:** §12 rows **5** (push the unified perception corpus off one disk),
**13** (4,930 plaintext clip ids vs the stated digests-only rule) and **14** (rotate the exposed HF
token; implement the credential scan) are **PI decisions with a time-sensitive component**. They are
in the strategy's open-decisions table with owners named, and they are repeated here because a
request that lives only inside a document is the failure mode that left an orthogonality instrument
unmerged for 10 days.
