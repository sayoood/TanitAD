# PH1 fusion of the aug120 batches — 201/201 clips fused, and the SAM3 leg was only 43 % covered

**2026-08-15.** Closes open item **§4.3a** of `Project Steering/STOP_2026-08-15_RESUME_RUNBOOK.md`
("aug120 batch fusion — labels exist, fusion not run"). Strategy and jurisdiction rules unchanged:
`…/2026-08-07-hierarchical-wm-redesign/PH1_FUSION_STRATEGY.md`; fuser `stack/scripts/ph1_fuse.py`.

Every number below is **MEASURED** by this run against the far side of
`Sayood/tanitad-ph0-aug120` unless stamped otherwise. Artifacts: `raw/` in this directory
(summary, per-batch accounting, per-clip label sources, coverage) and
HF `Sayood/tanitad-ph0-aug120/fused_aug120/` (204 files, far-side verified).

---

## 1. The headline — beside the val-600 baseline, never pooled with it

| | **aug120 (this run)** | **val-600 (baseline, `fused_w120val/_summary.json`)** |
|---|---|---|
| clips fused | **201** / 201 v2 records | 600 / 600 |
| **corroborations** | **88** | **175** |
| **conflicts** | **10** | **41** |
| **with the Alpamayo layer** | **201** (100 %) | **56** (9.3 %) |
| SAM3 records available | **86** (42.8 %) | 596 (99.3 %) |
| clips fused as **named SAM3-absent partials** | **115** | 0 — but see §4 |

⛔ **These two columns must not be added or averaged.** Three reasons, each measured:

1. **Different populations.** aug120 = the Alpamayo-augmented clips that have a w120 cache and
   were not already done; val-600 = the w120 *val* split. Disjoint by construction
   (`todo = (records ∩ w120) − done`).
2. **Different Alpamayo coverage.** 100 % vs 9.3 % — the aug120 set is *defined* by having an
   Alpamayo record, so its tactical 2-of-3 vote always has three voters and the val set's
   usually has two. A pooled corroboration rate would silently mix a 3-voter and a 2-voter
   estimator.
3. **Different SAM3 coverage** (§3) — 42.8 % vs 99.3 %. Corroboration counts are *counts of
   computable checks that agreed*; with SAM3 absent on 115 clips, two of the six checks cannot
   fire at all. **The aug120 corroboration count is therefore a floor, not a rate**, and the
   fair per-clip comparison is restricted to the 86 SAM3-covered clips — which this run records
   per clip (`_label_sources.json`) so the restriction is reconstructable, but does **not**
   itself publish as a headline rate, because the val-600 side would need its own per-record
   re-derivation to match and those records were not re-pulled.

**Corroboration inventory (aug120, 201 clips)** — what actually fired:

| check | corroborated | conflict | not_computable | flagged |
|---|---|---|---|---|
| `speed_sign_vs_ego` | 83 | 9 | 0 | — |
| `red_light_vs_stop` | 5 | 1 | 0 | — |
| `goal_evidence` | 15 grounded | — | 16 (SAM3 absent) | — |
| `census_vs_scene` | — | — | 71 (SAM3 absent) | 2 empty-urban |
| `scene_vs_situations` | 0 | 0 | 0 | — |

`scene_vs_situations` never fires because the frozen situation detectors did not run on these
batches and the v2 records carry no `situations` key — the fuser's `not_computable` path only
triggers on a *junction/intersection* scene claim, and none of the 201 VLM records made one.
**Absence of an instrument, recorded as absence.**

**Vocabulary emitted** (the real v6 tokens, imported — `tanitad.models.v6`):
`g_str` FOLLOW_MAIN_ROAD 151 · ROUTE_TO 31 · TURN_LEFT 17 · TURN_RIGHT 2.
`g_tac_lat` LANE_KEEP 186 · LANE_CHANGE_L 7 · LANE_CHANGE_R 7 · null 1.
`g_tac_lon` CRUISE 112 · BRAKE_TO 52 · HOLD 36 · YIELD_MERGE 1.
A true 2-of-3 majority (≥2 voters agreeing) backs **178/201 lateral** and **61/201 longitudinal**
tokens; Alpamayo cast a vote on **201/201** clips. Winning-vote sources across both axes:
ego 316, alpamayo 245, vlm 211.

## 2. `batch_00184` — handled as a NAMED partial, and it was not alone

**Yes.** Runbook §6.11 flags `batch_00184` (8 clips) as having v2 labels but no SAM3 directory
(`B184_SAM3_ABSENT`). Far-side listing confirms it: **it is the only batch prefix with 1 file
instead of 2**. Its 8 clips fused from ego + VLM + Alpamayo with
`perception.absent = "AUG120_SAM3_STAGE_GAP"` stamped per record, and every SAM3-dependent
corroboration degraded to `not_computable` — **never a silent zero, and never a fabricated
verdict**. 2 of its 8 clips do have SAM3 from `batch_00160` (the overlapping 40-run pass) and
were fused with it; 6 have none anywhere.

⚠️ **But the runbook's framing understates the problem** — this is the finding of the run:
`batch_00184` is **not** the only gap. **115 of 201 clips (57.2 %) have no SAM3 record at all**,
spread across *every* batch (§3). Treating §6.11 as "8 clips to redo" would have banked a
perception layer that is 43 % covered while looking complete.

## 3. ⛔ ROOT CAUSE: `aug120_pipeline.py` never passed `--n` to `ph0_sam3.py`, whose default is 4

MEASURED, and the mechanism is in the source, not inferred:

- `stack/scripts/ph0_sam3.py:411` — `for rec in d.get("clips", [])[:a.n]`, with
  `--n` **defaulting to 4** (`ph0_sam3.py:387`).
- `stack/scripts/aug120_pipeline.py` passed `--n` to the bridge and to the VLM, and **omitted it
  for SAM3**.
- Consequence, verified per file: **every** `batch_*/sam3/sam3.json` holds exactly **4** clip
  records, and those 4 are exactly the first 4 of that batch's v2 order (checked on three
  batches: `s_ids == v_ids[:4]` True). 25 sam3 files × 4 = 93 records, 86 distinct clips.
- The stage still printed `SAM3_RC=0`. **A stage that succeeds on the wrong scope looks like an
  answer** — the same class as `df` hiding the MooseFS quota and `tegrastats` on Thor.

**Fixed in place** (`stack/scripts/aug120_pipeline.py`, this change): `--n str(len(batch))` with
the measurement in a comment so the next reader cannot re-derive it wrong. The fix does **not**
retro-fill the labels — the 115 clips need a SAM3 re-run on a GPU pod (§6).

## 4. The defect this run also found in the val-600 baseline

The baseline summary reads `n_v2: 600, n_sam3: 596`. Under the fuser as it stood, those **4
clips were fused with a silently empty perception layer** — no marker, and their
`census_vs_scene` / `goal_evidence` verdicts were computed from a detector that never ran.
That is the same shape as the 0-of-600 join defect the campaign pinned in `2da0799`, one level
down. Two changes make it structurally impossible (both pinned by new tests):

1. **A v2 clip with no SAM3 record now refuses the run loudly** unless the operator names the
   reason: `--missing-sam3-ok REASON`.
2. **With the flag**, the record carries `perception.absent = REASON`, the summary carries
   `sam3_missing` + `missing_sam3_reason`, and `goal_evidence` / `census_vs_scene` return
   `not_computable` instead of `provisional` / `flagged_empty_urban` — an absent detector may
   not manufacture a verdict, the rule already applied to `scene_vs_situations`.

⚠️ **`fused_w120val/` on HF still contains those 4 unmarked records.** Not corrected here (it
would silently re-baseline the published 175/41/56); flagged in §6 as a work item.

## 5. Method — one fuse invocation, val-600 shape, plus a declared merge

The far side is **two overlapping pipeline passes**, not one: a partial `BATCH=8` pass (tags
`batch_00000…00184` step 8; its `_00192` slice was never pushed) and a completed `BATCH=40` pass
(`00000/00040/00080/00120/00160/00200`, the `AUG120_DONE` provenance). Their v2 union is
**201 clips, exactly the reconstructed `todo`**, with **152 clips present in two files**. Naive
per-directory fusion would have fused those 152 twice.

So the run merges first, under a declared policy, and asserts the fact the policy depends on:
**duplicate v2 records are content-identical** — 151/152 differ *only* in the `_calls` meta
field, **0 substantive diffs** (assertion in `code/aug120_fuse_run.py`; the run aborts if a
substantive diff ever appears). Per-clip source tags are recorded in `_label_sources.json`.

The population was reconstructed from primary sources, not from a saved list — `records.parquet`
(4,729 clips) ∩ w120 corpus listing (3,000 shards) − `w120val_600/clips.json` (600) = **201**,
matching `AUG120 … todo=201` in the pipeline's own log line. Per-batch v2 sets equal the
corresponding `todo` slices exactly (`==todo[b0:b0+8]` / `[b0:b0+40]` for all 25 tags), which is
what makes the reconstruction trustworthy rather than merely plausible.

Ego spine: all 201 `bridged_w120train_2400/ego/<clip>.npz` pulled and verified to load with
`poses [T,4]`; `speed_profile.src == "ego_npz"` on all 201 fused records. *(Note: that archive's
`failures.json` lists all 2,400 clips with `ModuleNotFoundError` — it is the **superseded**
failure log of the pre-fix bridge run, not a live defect; the `clips.json` beside it lists 2,400
successes and the npz files exist and load.)*

The fuse itself is one invocation with the val-600 argument shape (`--v2-json --sam3 --ego-root
--records --out`), plus `--missing-sam3-ok AUG120_SAM3_STAGE_GAP`. **The Alpamayo layer attaches
exactly as the val-600 run attached it** — `--records records.parquet`, grouped by `clip_id`;
1,005 rows = 201 clips × 5 tasks, all five task keys present on all 201 records.

## 6. Per-batch accounting (records in → fused out)

`v2 in` / `sam3 in` are the far-side file contents; `attributed` counts clips whose **chosen**
v2 source was that tag (so each of the 201 is counted once). Full table:
`raw/fused_aug120_batch_accounting.json`.

| tag | v2 in | sam3 in | attributed | corrob. | confl. | alpamayo | sam3-absent |
|---|---|---|---|---|---|---|---|
| batch_00000 | 40 | 4 | 24 | 8 | 4 | 24 | 20 |
| batch_00008 | 8 | 4 | 4 | 2 | 0 | 4 | 0 |
| batch_00016 | 8 | 4 | 4 | 1 | 0 | 4 | 0 |
| batch_00024 | 8 | 4 | 4 | 3 | 0 | 4 | 0 |
| batch_00032 | 8 | 4 | 4 | 2 | 0 | 4 | 0 |
| batch_00040 | 40 | 4 | 26 | 14 | 2 | 26 | 22 |
| batch_00048 | 8 | 4 | 3 | 2 | 0 | 3 | 0 |
| batch_00056 | 8 | 4 | 4 | 2 | 0 | 4 | 0 |
| batch_00064 | 8 | 4 | 3 | 2 | 0 | 3 | 0 |
| batch_00072 | 8 | 4 | 4 | 3 | 0 | 4 | 0 |
| batch_00080 | 40 | 4 | 26 | 11 | 1 | 26 | 22 |
| batch_00088 | 8 | 4 | 3 | 2 | 0 | 3 | 0 |
| batch_00096 | 8 | 4 | 4 | 1 | 0 | 4 | 0 |
| batch_00104 | 8 | 4 | 3 | 1 | 0 | 3 | 0 |
| batch_00112 | 8 | 4 | 4 | 1 | 1 | 4 | 0 |
| batch_00120 | 40 | 4 | 26 | 8 | 1 | 26 | 22 |
| batch_00128 | 8 | 4 | 3 | 1 | 0 | 3 | 0 |
| batch_00136 | 8 | 4 | 4 | 1 | 0 | 4 | 0 |
| batch_00144 | 8 | 4 | 3 | 1 | 0 | 3 | 0 |
| batch_00152 | 8 | 4 | 4 | 1 | 0 | 4 | 0 |
| batch_00160 | 40 | 4 | 33 | 18 | 1 | 33 | 29 |
| batch_00168 | 8 | 4 | 3 | 2 | 0 | 3 | 0 |
| batch_00176 | 8 | 4 | 4 | 1 | 0 | 4 | 0 |
| **batch_00184** | **8** | **0** | **0** | 0 | 0 | 0 | 0 |
| batch_00200 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| **TOTAL** | **353** | **93** | **201** | **88** | **10** | **201** | **115** |

*Reading the table:* `batch_00184`'s 8 clips are all attributed to the 40-run tags that also hold
their v2 (that is why its `attributed` is 0 while its `v2 in` is 8) — the 40-run rows therefore
carry the `sam3-absent` counts. The dedup is why 353 v2 records in yield 201 fused out.

## 7. Verification

- **Record-level, all 201:** schema `ph1-fused-v1`; `inference_admissible == {perception,
  semantics}` (ego and alpamayo excluded — *labels may use ego; inference is vision-only*);
  `sign_text_status == pending_g1_gate` on 201/201 (**G1 is CLOSED at 0/31 — sign text stays
  extraction-only**); every vocab token in the imported v6 lists; the goal/situation
  disjointness assert holds on every record; all 5 Alpamayo task keys present; ego spine
  `src == ego_npz`. **0 failures.**
- **Absent-partial semantics:** 115/115 carry the absent marker with empty tracks and **zero**
  fabricated `flagged_empty_urban` or `provisional` verdicts; 86/86 SAM3-covered clips carry no
  spurious marker (82 of those produced ≥1 track; 4 produced none, which is valid abstention).
- **Independent recount** of corroborations/conflicts by re-reading the 201 records:
  **88 / 10**, identical to `_summary.json`.
- **Far side** (never the push log — `POD_HANDOVER_2026-08-13.md §4b`): fresh
  `dataset_info` listing → **204/204 files present, 0 missing, 0 extra, 0 size mismatches**;
  then a **byte round-trip** of `batch_00008`'s 4 attributed records + both meta files,
  re-downloaded with `force_download=True`: **6/6 md5-identical**, and the re-read records
  re-assert schema, clip_id and whitelist. `FARSIDE_VERIFY PASS`.
- **Tests:** `stack/tests/test_ph1_fuse.py` **14 passed** (12 + 2 new pinning the named-partial
  behaviour). **Full suite `pytest -q` with `PYTHONUTF8=1`: 2,812 passed / 0 failed / 17 skipped /
  2 xfailed** in 239.7 s — exactly the 2026-08-15 baseline (2,810/0/17/2) **+2**, my two new tests.
  ⚠️ `test_ph1_fuse.py` also gained the suite's standard `sys.path` insert for `scripts/`: it had
  none and only imported because an alphabetically-earlier test happened to do the insert first —
  running that file alone failed with `ModuleNotFoundError: ph1_fuse`. Collection-order luck, now
  removed.

## 8. What this run did NOT do

1. **Did not re-run SAM3** on the 115 uncovered clips — needs a GPU pod; none is rented
   (all four pods stopped 2026-08-15). Fusion is re-runnable per clip once labels exist: the
   fuser skips existing outputs, so a later pass only writes the new records.
2. **Did not correct `fused_w120val/`** — its 4 unmarked SAM3-absent records stand; re-fusing
   would re-baseline the published 175/41/56 without a decision.
3. **Did not re-derive the val-600 per-record numbers**, so no like-for-like restricted-population
   rate is published (§1.3).
4. **Did not pull any video or cache payload** — labels + ego npz only, ~34 MB down, 4.3 MB up.
5. **Did not commit or push to git** (staged only, per the operating standard).
6. **Did not touch the `_00192` gap** — the BATCH=8 pass's missing slice is fully covered by the
   40-run pass, so it costs nothing; noted so it is not mistaken for data loss.

## 9. Work items this creates

| # | item | why | cost |
|---|---|---|---|
| 1 | **SAM3 re-run for 115 clips** (incl. 6 of `batch_00184`'s 8) with the `--n` fix | perception layer is 43 % covered; the fix is already in `aug120_pipeline.py` | ~30 min GPU (batch of 115 at the pilot's rate) + a re-fuse |
| 2 | **Re-fuse those 115** after (1), then re-verify far side | fuser resumes; only new records written | minutes, 0 GPU |
| 3 | **Decide on `fused_w120val/`'s 4 unmarked records** | published baseline vs correctness | PI call |
| 4 | **SAM3 sign-class threshold study** (runbook §6.7) is now *more* load-bearing | 15 `goal_evidence: grounded` verdicts rest on SAM3 sign tracks whose class is flagged unreliable (⅔ of best crops had no sign) | scoped elsewhere |

**Escalation:** items 1–2 are the only things standing between this and a complete aug120
perception layer, and they need **one GPU pod for ~30 minutes**. Everything else here is done.
