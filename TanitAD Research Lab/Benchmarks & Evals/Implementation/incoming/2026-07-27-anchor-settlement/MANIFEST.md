# MANIFEST — `2026-07-27-anchor-settlement`

**Agent:** `anchor-settlement` · **Date:** 2026-07-27 · **HEAD at start:** `84028f4`
**STAGED, NEVER PUSHED.** No commit, no push, no branch switch.
**Hosts:** dev box (RTX 4060) + `tanitad-eval` (A40, idle). ⛔ **pod1 (training) and pod2 (small
validation) were never contacted.**

## The one-line answer

**The comma-yaw anchor `+0.3308` is WITHDRAWN — 2 of its 22 comma evaluation episodes are, by
content (sha256 of raw pose bytes *and* raw sensor bytes), bit-identical to 2 of the deployed head's
own comma TRAINING clips; without them it reads −0.746.** The other half of the
disqualification-lifting claim (`+0.679`, a *retrained* head) has **no leak** and stands, but reads
**+0.3038 [+0.054, +0.479]** on the 20 content-clean episodes. ⇒ **comma yaw is testable; the
deployed head does not do it.**

## Artifacts

| file | kind | produced by | note |
|---|---|---|---|
| `ANCHOR_SETTLEMENT.md` | document | — | the settlement: pre-registration, overlap, admissibility, what the claim is worth |
| `MANIFEST.md` | document | — | this file |
| `code/fingerprint_comma_cache.py` | code | — | content fingerprints of a comma episode cache; run on **both** hosts (~20–25 s each) |
| `code/intersect_by_content.py` | code | — | the overlap + hash-family agreement + duplicate + self-overlap checks |
| `code/resettle_anchor.py` | code | — | the anchor re-measured, leave-2-out, admissibility, PhysicalAI control |
| `code/resettle_arms.py` | code | — | all 18 persisted v3 arms, leave-2-out |
| `code/admissibility_consequence_61c.py` | code | — | the 2nd corpus, **through the shipped API** |
| `raw/fp_61c46fca8f7f.json` | raw | (2) dev box | 90 eps × 6 content hashes + per-frame digests + exact pose bytes |
| `raw/fp_76b6e94a97a1.json` | raw | (2) `tanitad-eval` | 64 eps, ditto |
| `raw/anchor_overlap.json` | raw | (3) | ⭐ **the answer** — pre-registration, every path, every count |
| `raw/anchor_resettlement.json` | raw | (4) `tanitad-eval` | §3, §5.3B, the PhysicalAI control |
| `raw/arms_resettlement.json` | raw | (5) `tanitad-eval` | §4, all 18 arms |
| `raw/admissibility_consequence_61c.json` | raw | (6) dev box | §5.3A |

## Repo files changed outside this directory

**Code (2):** `stack/tanitad/data/comma2k19.py` (the admissibility API + its own anchor quotes
amended), `stack/tests/test_yaw_admissibility.py` (**new, 19 tests**).
**Documents corrected (16):** listed with their before/after meaning in `ANCHOR_SETTLEMENT.md` §9.1 —
the **three** documents that lifted the disqualification, plus `COMMA_YAW_REISSUE.md`,
`HEADING_DEFAULT.md`, `IDM_V3.md`, `MODEL_CARD_IDM_V3.md`, `IDM_DIAGNOSIS.md`, `IDM_V2_RESULTS.md`,
`DB_RETRY.md`, `idm_head_v1_card.json`, `MODEL_REGISTRY.md`, `RETRACTION_LOG.md` (**new class C43**),
`LEADERBOARD.md`, `LOOP_STATE.md`, `PROGRAM_OVERVIEW.md`, `TANITAD_PAPER.md`, and two test docstrings.

⛔ **Not opened:** `…/2026-07-26-idm-v2/PRE_REGISTRATION_IDMV2.md`, `Project Steering/Mission Plan.md`.
⛔ **No PhysicalAI number re-issued** (`n_pai_changed = 0`, re-measured). ⛔ **No repaired ceiling
derived.** ⛔ **Nothing pushed to HF.** 🔒 Counts only, no clip UUIDs. 🔒 Parity untouched.

## Lives in only one place?

**Nothing.** `/root/fp_76b6e94a97a1.json`, `/root/anchor_resettlement.json` and
`/root/arms_resettlement.json` on `tanitad-eval` are byte-identical to the staged copies (md5
verified on both sides); the three scripts also sit at `tanitad-eval:/root/*.py` and are staged here.
The 30 re-encoded latents in the scratchpad are **not** staged — regenerable in ~90 s from md5-pinned
artifacts (the `heading-default` pass's decision, unchanged).

## Suites

`stack/` **1576 passed, 12 skipped** (baseline 1557/12 → **+19 = exactly the new test file**, zero
new skips) · `taniteval/` **663 passed, 0 skipped** (baseline 663, unchanged).
Run with the project venv `C:\Users\Admin\venvs\tanitad`.

## 🔴 Needs an owner (full text in `ANCHOR_SETTLEMENT.md` §8)

1. **The YouTube-IDM rotation gate** was re-opened on `+0.3308 / +0.679`. The surviving evidence is
   **one** retrained-head number at **+0.3038**. Its owner should re-take the go/no-go. **I did not.**
2. **Run the same probe on `physicalai-val-0c5f7dac3b11` × `physicalai-train-e438721ae894`.** The
   registry's *"episode-disjoint from train ✅"* is an `episode_id` claim; a **78 % leak (62/79)**
   has already been measured once on a sibling val cache. ~25 s per cache, torch only.
3. **`idm2_lib.py:19` / `idm3_a0.py` insert `/root/taniteval` unconditionally**, which on
   `tanitad-eval` is a **different `ci.py`** from HEAD (`ef925f06…` vs `c92618a0…`). **Every published
   v3 interval came through it.** `stack_check` structurally cannot see this.
