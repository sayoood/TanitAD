# What the 4,472-clip build needs from this fusion pass — one page, no execution

**Scope:** runbook `§4.3b` — the 4,472 Alpamayo-augmented clips with **no w120 cache**, built via
chunk-index → `v2_compressed.py build --only-clips` → the aug120 flow. **Nothing here is
executed**; this is the list of things the aug120 pass measured that change how that build should
be run. Evidence class on each item.

## 1. The scale, restated honestly

The aug120 pass processed **201 clips** and produced **86 SAM3-covered** ones. The 4,472 build is
**22× the clips**, and it *starts* from a state the aug120 pass never faced: no w120 cache at all,
so a **build** stage precedes bridge → VLM → SAM3 → fuse. Every per-batch defect below is a defect
that would repeat 22× and be 22× more expensive to redo. *(MEASURED counts; scale ratio arithmetic.)*

## 2. Five things that must be fixed or decided BEFORE the first batch

| # | what | evidence | consequence at 4,472 scale |
|---|---|---|---|
| **1** | **`--n` on every stage, asserted against the batch size** — the aug120 run omitted it for SAM3 (`ph0_sam3.py` default 4) and silently covered 4 clips per batch while returning rc=0 | MEASURED: 86/201 covered; fix now in `aug120_pipeline.py` | would have produced ~4 SAM3 records per batch across ~112 batches — a perception layer covering **~4 %**, discovered only at fusion |
| **2** | **A per-batch coverage assertion after each stage**, not just an rc check: `n_out == n_in` per stage, written into the batch dir and pushed with it | the aug120 batches carry no such record; coverage had to be reconstructed from the far side afterwards | at 22× the batches, post-hoc reconstruction is the only alternative and it does not scale |
| **3** | **Batch tags must be unique per pass.** The aug120 far side holds two overlapping passes (BATCH=8 and BATCH=40) writing the SAME tag namespace (`batch_%05d` of the todo offset), so 152 of 201 clips exist in two files | MEASURED; harmless only because duplicates proved content-identical (0 substantive diffs) | with a build stage in front, a re-run after a crash is *expected*; colliding tags would make "which pass produced this label" unanswerable |
| **4** | **A parity decision, in writing, before the build.** These clips are not in `physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`); building them creates a **new** selection | INHERITED from `aug120_pipeline.py`'s own note ("would need building under a NEW parity key") + the standing parity rule | the parity rule forbids anything that re-selects episodes for cross-arm comparability. The build is admissible as a **separate labelled corpus**, not as an extension of the parity set — say so explicitly or the first trainer that eats it breaks comparability |
| **5** | **Disk + push discipline is already right — keep it.** Per-batch pull → process → push → delete bounded peak disk to one batch | the aug120 run completed 201/201 on a pod with a MooseFS quota that has killed jobs before | at 22× nothing else survives; a whole-corpus staging step would hit the quota trap |

## 3. What the fusion side already handles, and what it still cannot

**Handled (no work needed):**
- Partial SAM3 legs are now **named** (`--missing-sam3-ok REASON`, `perception.absent`,
  `sam3_missing` in the summary) and SAM3-dependent checks degrade to `not_computable`. A 4,472
  build with a partial perception leg will fuse *honestly* rather than silently.
- The fuser **resumes** (skips existing outputs), so an interrupted fuse costs one record.
- Vocabulary tokens are imported from `tanitad.models.v6`; the goal/situation disjointness assert
  and the ego-not-in-`inference_admissible` whitelist hold per record.
- The Alpamayo layer attaches by `--records records.parquet` grouped on `clip_id` — it will
  attach for **100 %** of these clips too, since the 4,472 are by definition augmented clips.

**Still cannot, and should be planned around:**
- **`situations` never fires.** The frozen situation detectors do not run in this flow and the v2
  records carry no `situations` key, so `scene_vs_situations` is structurally dead — 0/201 here.
  If the strategic/tactical supervision wants that check, the detectors must be wired into the
  bridge stage, not the fuser. *(MEASURED: 0 of 201 fired, and the reason is in the record.)*
- ⛔ **`goal_evidence: grounded` NO LONGER EXISTS — RETIRED 2026-08-16, and so is `provisional`.**
  *(Primary: `stack/scripts/ph1_fuse.py`, `GOAL_EVIDENCE_RETIRED = ("grounded", "provisional")`.)*
  The verdict is now always `not_computable` with the gap NAMED, and the one measured fact survives
  under an honest name — **`sign_like_object_present`**, plus the raw `sam3_sign_tracks` count and
  `evidence_sign_kind` (⚠️ a **VLM self-report**, never corroboration). Emission over **aug120**:
  `grounded` **15/201 → 0/201**. ⇒ **the 4,472 build has no `grounded` supervision channel to plan
  around**; it has a per-clip presence flag. *(MEASURED —
  `…/Architecture & Inference/…/incoming/2026-08-16-evidence-and-flake/` §1.4.)*
- ⚠️ **AND THE DETECTOR IS NOT "~⅔ GARBAGE" — that figure was a DIFFERENT CORPUS, and this bullet
  used to quote it as a property of SAM3's `traffic sign` class.** The threshold study asked for
  above **has now landed** (`…/Data Engineering/…/incoming/2026-08-16-sam3-concept-reliability/`),
  and it names this paragraph as the misread. Every number below carries its corpus, because the
  corpus is the whole disagreement:

  | number | value | **corpus** |
  |---|---|---|
  | `traffic sign` precision, uniform draw | **0.880** [0.795, 0.958] · n=64 over 33 clips · episode-cluster bootstrap | **`aug120`** (83 records, 538 sign detections) |
  | same, ❓-counted-wrong (the pessimistic bracket — **both are reported on purpose**) | **0.688** [0.552, 0.800] | **`aug120`** |
  | G1's own max-area rule, reproduced on this corpus | **0.926** [0.815, 1.000] · n=32 · **ZERO** empty boxes | **`aug120`** |
  | G1's *"no sign visible at all"* — ~22 of 31 crops (~⅔) | ⛔ **does not transfer, in either direction** | **`w120val`** (600 clips, 4,048 sign detections) |

  ⚠️ **The reconciliation is PARTIAL and the study says so.** G1's two candidate mechanisms were
  both tested and both REFUTED — the **selection** (max-area concentrates FPs: refuted; sign FPs
  here are *smaller*, median 35.6 px² vs 74.8 px² for true ones) and the **rendering** (G1's tight
  4× LANCZOS crop is harder to read: refuted, and on one detection strictly better). What remains
  uncontrolled is the **corpus**, so the honest statement is *"not on aug120"*, **not** *"G1 was
  wrong"*. **Open work item, 0 GPU, ~2 h: run the same adjudication on the `w120val` sign leg**
  before any val-side sign label is trusted.
- ⚠️ **The operating point that follows** (same study, §5/§6): `traffic sign` is 🟨 **PRESENCE ONLY
  at the 0.5 default** — adequate for a presence flag, and ⛔ **never admissible as evidence of a
  sign's KIND or TEXT** (the G1 text gate is CLOSED at 0/31, `Project Steering/G1_RESULT.md`).
  Raising to **≥0.70** lifts sample precision to 0.967 but retains only **274/538 (50.9 %)** of the
  class, so it is worth paying **only if the 4,472 build makes signs PER-DETECTION supervision**
  rather than a presence flag. Decide which, in writing, before the build — the threshold is a
  consumer decision, not a corpus property.
- **No metric 3D** (strategy doc §7) — track dynamics stay ordinal. Unchanged.

## 4. Cost shape to expect (ESTIMATED, from this pass's measured rates)

- **Labels are tiny**: 201 clips → 7.8 MB of v2+sam3 JSON and 4.26 MB of fused records
  (median 18 KB/record). 4,472 clips ⇒ **~95 MB fused**, **~175 MB of raw labels**. Storage is a
  non-issue; the build's cost is entirely GPU + the source-chunk pull.
- **Fusion itself is free**: 201 records fused in seconds on a dev box, CPU-only, no GPU. 4,472
  ⇒ minutes. **The fuse is never the bottleneck — do not schedule a pod for it.**
- **Ego npz is ~5 KB/clip**; the whole 4,472 ego spine is ~22 MB.
- The binding costs are (a) building w120 caches from the chunked source, and (b) the VLM+SAM3
  passes. Neither was measured by this run; **do not quote a figure for them from here.**

## 5. The one-line recommendation

**Do not start the 4,472 build until items 1–4 of §2 are settled in writing** — the `--n`
assertion and the per-stage coverage record are 20 minutes of work that this pass proved are worth
more than the pass itself, and the parity question decides whether the output is usable at all.
Close the 115-clip SAM3 gap first as a rehearsal: it exercises the fixed pipeline end-to-end on a
population whose answer is already known.
