# The Alpamayo eval leak is closed in code — and the dangerous direction was the other one

**2026-08-18 · closing RETRACTION_LOG C112 · all figures MEASURED (ours), re-derivable with
`code/derive_contamination.py`, banked in `raw/contamination.json`**

---

## Headline, in the order that matters

1. ⛔ **The contamination rate that matters is not 4.3 %, it is 78.2 %.** 4.3 % is 201 of the
   4 729 Alpamayo *records*. But **only 257 of those 4 729 have w120 video built**, and a split
   can only contain clips that exist. **The Alpamayo eval split buildable TODAY is 257 clips, of
   which 201 = 78.21 % are in the parity train corpus** — REF-A-I-JEPA scale (~80 %), not "the
   same class at smaller scale".
2. ⛔ **The 201 are not a subset of the aug120 perception corpus — they ARE it, exactly.**
   `fused_aug120_v2_index.jsonl` and `fused_aug120_v3_index.jsonl` each hold 201 clips whose
   sorted-id sha256 is `80632f17…`, byte-identical to the exclusion list's. **201/201 = 100 %** are
   in `physicalai-train-e438721ae894`. The pilot recorded this as a *coincidence of count*; it is
   **set identity**, and the mechanism is visible in `stack/scripts/aug120_pipeline.py:53`
   (`todo = (records ∩ w120_corpus) − done`) — the cohort was *selected from* the parity corpus.
3. ⭐ **The direction nobody was looking at is worse: 6 of the 40 canonical val episodes (15.0 %)
   are inside the Alpamayo record set.** Not "an eval split contains train clips" but "a train
   corpus is about to swallow the deployed val". Blast radius **today is zero** — nothing trains on
   those labels — and the trigger is already scheduled (the 4 472-clip build).
4. ✅ **Blast radius on published numbers: ZERO.** No model-eval number has ever been computed on
   this corpus. Stated positively because it is the good outcome and it is worth being explicit.
5. ✅ **The fix is a derived membership oracle, not a list.** `parity.py` §10/§10b answers *"is this
   clip in the parity train split?"* / *"…in the deployed val?"* for **any** clip, so the next 4 472
   Alpamayo clips and every future corpus are covered with nothing to update.

---

## 1. Root cause, and why the fix is shaped the way it is

C112's class: **a non-overlap ASSUMED FROM PROVENANCE ("different source ⇒ disjoint") rather than
COMPUTED FROM IDS.**

⚠️ **The assumption was not laziness — it was UNANSWERABLE.** The committed manifest carries
`clip_membership.clip_id_sha256_sorted`, a digest of the **whole sorted list**. A whole-list digest
is a set *identity*; you cannot test one element against it. And the clip ids themselves are
gated-confidential and live only on pods. So on any other host the question *"is clip X in the
parity train split?"* **had no answer at all**, and an unanswerable question is exactly the one that
gets answered by provenance.

⇒ The fix is not a reminder. It is **the missing oracle**: `stack/tanitad/data/parity_train_clip_digests.json`
commits `sha256(clip_id)` for each of the 2 400 parity train clips. Membership becomes exact,
enumeration stays impossible, and §9's rule ("the repo carries only the digests") is preserved.

**THE GENERATION IS THE PROOF.** `scripts/make_parity_clip_digests.py` refuses to write unless its
source reproduces the committed corpus digest — the same contract as `register_v2_geometry_sibling`.
A digest set cannot exist for a wrong, truncated or re-selected clip set.

**And the chain closes inside the repo, on this box, with no pod access.** The banked pilot listing
`…/2026-08-17-thor-concurrency-pilot/parity_ls.txt` reproduces
`e61a04553df5b9d52a0810be32cf31927bd92644d9d12ada563910b8a0ada4de` — the manifest's committed
digest — exactly. `tests/test_eval_contamination.py` re-walks that chain (listing → manifest digest →
per-clip digests → the committed file) on every run.

---

## 2. Where an Alpamayo eval split is or could be defined — DERIVED, not hand-listed

Two independent sweeps (mine + a dedicated search agent, six locations, six naming conventions):

| finding | evidence |
|---|---|
| **No Alpamayo eval/val split constructor exists in the repo.** | `records.parquet` has no `split` column (`…/2026-08-06-alpamayo-augmentation/a2_records_stats.json`, 12 columns). No function anywhere partitions Alpamayo clips. `taniteval` has **2** Alpamayo hits, both prose — `registry.py:288` lists exactly three eval corpora (`physicalai`, `comma`, `cosmos`). |
| **The one place a split IS declared** is `label_split` in the S2 label index. | `…/2026-08-16-s2-v1-labels/review/labels_v2/clip_index.json`, consumed by `stack/scripts/s2_labels.py:543`. |
| **The selectors that exist** choose by coverage/throughput, never by eval role. | `ph0_sample_clips.stratified_sample`, `v2_to_pilot.pick_clips`, `aug120_pipeline` (the cohort deriver), `s2_lab_lib.derive_sam3_gap`. |
| **The machinery a new split would reuse.** | `…/2026-08-02-v2-clean-val-selector/clean_val_select.py` (the program's one true val-split constructor), `parity.py` `guard_val_split`/`assert_v2_splits_disjoint`, `lake/curation.is_eval_holdout`. |

⛔ **`label_split` is a PROVENANCE TAG, not a partition — and that is the trap.** MEASURED:

| `label_split` | n clips | in parity **train** | in Alpamayo records | verdict |
|---|---:|---:|---:|---|
| **`aug120`** | 201 | **201 (100 %)** | 201 | ⛔ a **TRAIN** leg. Correct as supervision, catastrophic as a holdout. |
| `w120val` | 600 | **0 (0 %)** | 56 | ✅ the genuine held-out leg. |

The name `aug120` reads like an independent augmentation corpus. Nothing in the name, the file or
the schema says otherwise — so the fact is now **computed from ids and written into the run record**
(§3).

---

## 3. The fix, at the point that cannot be bypassed

### `stack/tanitad/data/parity.py` §10 — refuse a train-contaminated eval split

| function | what it does |
|---|---|
| `clip_digest(id)` / `parity_train_clip_digests()` | the membership oracle |
| `clips_in_parity_train(ids)` | which ids are contaminated (**in-process**; discloses nothing the caller did not supply) |
| **`assert_eval_clips_disjoint_from_parity_train(ids, label=…)`** | ⛔ **REFUSES**. Per-split PASS/FAIL, never a percentage threshold — a threshold would have waved the 4.3 % case through |
| `filter_eval_clips(ids, label=…)` | the sanctioned repair; reports `n_kept` so a report can never quote the pre-filter n |
| `assert_v2_eval_cache(dirs, label=…)` | the same for a v2 cache dir. ⚠️ Distinct from `assert_v2_splits_disjoint`, which needs **both** dirs in hand — an evaluator handed one `--v2-cache` (the normal case) could not use it |

The only way past is `sanctioned_audit="<why>"`, mirroring `note_leaky_audit`: it takes the **reason**,
prints the disclosure and stamps `decision_grade: False`. A label census over train clips stays
possible; a silent one does not. 🔒 Every refusal prints **counts only** — pinned by a test.

### `stack/tanitad/data/parity.py` §10b — refuse a train corpus that swallows the deployed val

`assert_train_clips_disjoint_from_deployed_val(ids, label=…)` / `filter_train_clips(...)`, against
`deployed_val40_clip_digests.json`.

⚠️ **Its proof is weaker than §10's and says so.** A 40-of-600 deployment is a subset and **cannot**
reproduce the corpus digest. The substitute is a **second-source cross-check**: every episode's
independently banked `clip_sha8` (`…/2026-08-04-instrument-durability/raw/val40_lead_index_ANON.json`,
a different stream) must equal `sha256(clip_id)[:8]` of the id being minted. **40/40 agree.**
`load_clip_digests` **refuses** a subset file that carries no complete cross-check, so the weaker
proof can never be silently skipped.

### `stack/scripts/s2_labels.py` — disclosure that cannot be forgotten, plus a refusal

* `S2LabelSet.parity_contamination()` is **unconditional** and rides in `report()`, which
  `train_v6_staged.py:2495` writes into **every v6 run's `config.json`**. A fact nobody asked for is
  exactly the fact that goes unnoticed.
* `load_s2_labels(path, role="eval")` refuses each leg that overlaps parity train — MEASURED: it
  refuses the canonical artifact, naming `aug120`, which is the correct outcome.
* `role="train"` (the default) is **behaviour-identical**; the `parity.py` diff is **412 insertions,
  0 deletions**.
* ⚠️ If `tanitad.data.parity` cannot be imported at all, `role="eval"` **refuses**. A guard that
  no-ops when its oracle is missing is C112 wearing a green suite.

---

## 4. The pin — `stack/tests/test_eval_contamination.py`, 17 tests, RED without the fix

Every refusal test feeds a **real** contaminated clip id read at run time from the banked list, and
every one is paired with a **positive control** on a disjoint set — a guard that raised
unconditionally would fail here rather than look strict.

**C107 discharged by construction.** Each guard was neutered in turn and the suite required to go red:

| neutered | result |
|---|---|
| `assert_eval_clips_disjoint_from_parity_train` | **4 failed**, 13 passed |
| `clips_in_parity_train` | **8 failed**, 9 passed |
| `assert_train_clips_disjoint_from_deployed_val` | **1 failed**, 16 passed |

Also pinned: the digest file's self-check actually fires (drop one entry from a copy → refusal); the
mint refuses a wrong source; the 201 are **derived without reading the banked list**; the refusal
never prints a clip id; the aug120 leg is 201/201 and w120val 0/600; `role='eval'` refuses through
the real loader on the real artifact.

⚠️ Missing artifacts `pytest.fail`, never `skip`. A skipped leak test is the absent check that
produced C112.

---

## 5. Blast radius — enumerated by opening artifacts

**No model-eval number has ever been computed on the Alpamayo/aug120 corpus. Zero retractions
follow from C112.**

| probe | result |
|---|---|
| all 73 `taniteval/results/*.json` opened and walked for corpus keys | **zero** name aug120/alpamayo |
| `taniteval/taniteval/registry.py:288` `CORPORA` | three corpora: `physicalai`, `comma`, `cosmos` |
| `MODEL_REGISTRY.md` | every alpamayo/aug120 hit is confined to **§11 "PRODUCED DATASETS"**; no model row evaluates on it |
| the aug120 numbers that DO exist | label-quality only — fusion counts (88 corroborated / 10 conflicts), SAM3 precision, refuse-deltas, κ agreement. Correct as published; **must never be quoted as held-out** |

⚠️ **One trap worth naming.** `…/2026-08-05-alpamayo2-super/comparison/` carries real ADE numbers
beside the word "Alpamayo" — those are the **Alpamayo-2-Super 34.3 B model** scored on the 290-clip
**OOD-val** corpus (`corpus_key physicalai-oodval-6f4b94e4c7ce-q90`), a different corpus entirely.
Not contaminated by this.

---

## 6. ⛔ ESCALATIONS — these need a decision, not a doc

1. **§10b is a guard placed BEFORE a scheduled failure.** The 4 472-clip Alpamayo build
   (`…/2026-08-15-aug120-fusion/NEXT_4472_BUILD_INPUTS.md`) is the trigger. **Whoever runs it must
   call `parity.filter_train_clips()` on the clip list before building** — 6 clips out of 4 729, and
   without it 15 % of the episode set behind every published open-loop number enters training.
   Nothing existing would notice: §9 checks a cache against *its own* corpus digest, and an
   augmentation corpus is a different corpus by construction.
2. **`parity.py` §9's confidentiality claim is now FALSE and I did not silently fix it.** It says
   *"The repo carries only the digests"*, but `…/2026-08-17-thor-concurrency-pilot/` commits
   **4 729 + 201 raw PhysicalAI-AV clip ids in plaintext** (`alpamayo_clip_ids.txt`,
   `alpamayo_IN_parity_train_EXCLUDE_FROM_EVAL.txt`, tracked at `6784455`). My enforcement data is
   digests precisely so it does not add a third copy — but **the existing two are a live
   gating-compliance question and are load-bearing evidence for the tests above.** Deleting them
   would break the pin; keeping them contradicts the stated rule. **PI decision.**
3. **3 pre-existing suite failures are NOT mine and need their owner.**
   `tests/test_v6_stage_init_introduction.py` — 3 failures because `STAGE_MAY_INTRODUCE["S-T"]`
   grew a `t2_head.` entry in the working tree while the test still expects the old tuple.
   `t2_head` is **absent from HEAD** in both `train_v6_staged.py` and `models/v6.py`; the untracked
   `tests/test_v6_t2_contrastive.py` / `test_v6_t5_consistency.py` show that stream is live. Also
   `tests/test_v5_trainer_v2_val.py` fails **3 tests when run in isolation** with
   `ModuleNotFoundError: No module named 'taniteval'` and passes in the full run — a pre-existing
   path-isolation defect, unrelated to this work.
4. **The parity VAL 600-clip oracle does not exist.** §10b covers the **deployed 40**, which is what
   every published statistic is quoted over. Overlap with the other 560 is not a leak but is a
   comparability hazard. `corpus_key=` is the hook; minting it needs the 600 clip ids, which live
   only on a pod.

---

## 7. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `parity.py` §10 + §10b (412 insertions, 0 deletions) | `repo:stack/tanitad/data/parity.py` | no |
| **the membership oracle**, 2 400 digests | `repo:stack/tanitad/data/parity_train_clip_digests.json` | no |
| the deployed-val oracle, 40 digests | `repo:stack/tanitad/data/deployed_val40_clip_digests.json` | no |
| the mint (generation-is-the-proof) | `repo:stack/scripts/make_parity_clip_digests.py` | no |
| S2 loader wiring (`role=`, `parity_contamination`) | `repo:stack/scripts/s2_labels.py` | no |
| **the pin**, 17 tests | `repo:stack/tests/test_eval_contamination.py` | no |
| this report | `repo:…/incoming/2026-08-18-alpamayo-parity-exclusion/ALPAMAYO_PARITY_EXCLUSION.md` | no |
| the re-derivation script | `repo:…/2026-08-18-alpamayo-parity-exclusion/code/derive_contamination.py` | no |
| raw derived figures | `repo:…/2026-08-18-alpamayo-parity-exclusion/raw/contamination.json` | no |
| A15 provenance correction | `repo:…/2026-08-02-thor-deployment-profile/PROVENANCE_CORRECTION.md` | no |

Suites at the time of writing: **`stack` 3 935 passed / 3 failed** (all three are escalation 3, not
this work) · **`taniteval` 1 136 passed, clean** · **`test_eval_contamination.py` 17 passed**.
