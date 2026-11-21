# Colab S2 label lab — built, smoke-tested end-to-end, ready for the PI's T4

**2026-08-16.** Implements the PI's directive: *"review if the pipeline is
extracting the right information and optimizing it until the correct
strategic vocabulary is generated in the right format, then we can scale"* —
as two Colab notebooks + runner scaffolding under `colab/`, using the free
T4 (unsloth 4-bit for the Qwen3.5-9B fit, per the PI's tip). Every number
below is **MEASURED on the dev box this session** unless stamped otherwise;
smoke transcripts are reproducible via `colab/smoke_run.py`.

## 1. Deliverable manifest (all in-repo, staged; nothing stranded)

| artifact | where | what |
|---|---|---|
| `SAM3_BACKFILL_115.ipynb` | `colab/` | the queued 115-clip SAM3 re-run (~30 GPU-min), deliberately the first workload — validates auth → pull → GPU → bank → far-side verify → resume on a small known shape |
| `STRATEGIC_LABEL_LAB.ipynb` | `colab/` | the lab: VLM (unsloth 4-bit) + SAM3 + ego legs, sequential with `del`/`empty_cache` + `max_memory_allocated` prints between; ph1_fuse fusion → `g_str`/`a_str` with **per-token provenance**; per-clip **review sheet** (frames + tokens + provenance + confidence + corroborations); per-clip banking with run manifest |
| `s2_schema.py` | `colab/` | the **one-file schema swap point** — PROVISIONAL diagram vocabulary (`g_str`: FOLLOW_MAIN_ROAD default · ROUTE_TO · LANE_TARGET · TURN; `a_str`: PREPARE_LANE_CHANGE · REDUCE_TO · PREPARE_EXIT · PREPARE_STOP; categorical args), v6 pins with automatic drift check, goal/situation disjointness assert, full record validation |
| `s2_lab_lib.py` | `colab/` | shared plumbing both notebooks call: auth (Colab Secrets → env → Keys.txt in place), gap derivation, banking + far-side byte verify + resume, unsloth loader around `ph0_v2.ConstrainedVLM`, leg wrappers, fusion, review sheet, smoke stubs |
| `nb_build.py` / `smoke_run.py` | `colab/` | notebook generator (cells are magic-free pure Python) / CPU smoke driver that **executes the shipped .ipynb's cells** — the tested artifact is the shipped artifact |
| `fixtures/sam3_backfill_expected.json` | `colab/fixtures/` | the MEASURED 115-clip gap list (see §3) the notebook cross-checks its live derivation against |
| `RUNNER.md` | `colab/` | operator doc: Drive-open path, Secrets setup, session-death/resume semantics, T4 memory budget table, parameter reference |
| this doc | `…/2026-08-16-colab-s2-lab/` | result + decisions + escalations |
| far side (verified) | HF `Sayood/tanitad-s2-lab` (created, private) | `smoke/` tree from the smoke runs: 2 backfill records + 1 lab record + review sheet + 3 run manifests, each byte-round-trip verified |

## 2. The smoke results — both notebooks EXECUTED end-to-end (CPU, GPU stubbed)

⚠️ *A notebook that has never executed is a hypothesis.* These two are not:
`smoke_run.py` execs the shipped `.ipynb` code cells in order (they are
magic-free precisely for this), with `S2_SMOKE=1` stubbing only the GPU
models and the video frames. Everything else — auth, HF pulls, derivations,
the real `ph1_fuse` fusion, schema validation, banking, resume — ran for
real against the far side.

**`SAM3_BACKFILL_115.ipynb` — PASS (9/9 cells, 13 s), and resume PROVEN:**
run 1 derived the gap (limited to 12 records in smoke → 6 absent,
cross-check vs `_label_sources.json` agreed), assembled v2 records
(**25 far-side files → 353 records → the 1 wanted clip** — independently
reproducing the fusion run's own 353 count), banked clip `0089a096…`
(1 213 B, far-side byte-verified). **Run 2 found 1 done on the far side,
skipped it, and banked the NEXT clip `00d05901…`** — find-what-is-done-then-
continue, demonstrated, not asserted.

**`STRATEGIC_LABEL_LAB.ipynb` — PASS (11/11 cells, 6 s):** real ego leg on
the real `bridged_w120train_2400/ego/01b24287….npz` (vote `TURN(left)` at
net dyaw **+100.32°**, real speed spine v 2.66→8.41 m/s), REAL VLM
resolution (`unsloth/Qwen3.5-9B`, arch `Qwen3_5ForConditionalGeneration`),
real fusion → valid S2 record `g_str FOLLOW_MAIN_ROAD [vlm]`,
`a_str [REDUCE_TO(crawl) [vlm], PREPARE_STOP [ego]]`, banked 5 532 B
far-side-verified; review sheet rendered and banked; end-of-run resume
assertion passed. The sheet correctly flags the stub-VLM vs real-ego
disagreement (`ego-yaw vote … DISAGREES with g_str`) — exactly the
review-sheet behaviour the lab exists for.

**One real bug found and fixed by the smoke run:** the goal/situation
disjointness assert initially scanned the whole record and **matched its own
explanatory note** (which contains the word "situation") — CLAUDE.md's
polling-monitor self-match trap in miniature. Fixed by scoping the scan to
the goal payload (`g_str`/`a_str`/`g_tac`) and attaching the prose note
after the assert; the incident is documented in `s2_schema.assert_disjoint`'s
docstring.

## 3. The 115-clip gap list is MEASURED, not inherited

`derive_sam3_gap()` pulled all **201** `fused_aug120/<clip>.json` records
and read each record's own `perception.absent` marker (**count records, not
files — C18**; the gap was invisible to every file count):
**115 absent / 86 covered / 201 checked**, in 70 s. Triple cross-check
agreed: the per-clip `_label_sources.json` map, and `_summary.json
sam3_missing = 115`. Banked as `colab/fixtures/sam3_backfill_expected.json`;
the notebook re-derives live on every run and diffs against the fixture
loudly (a drift after a partial backfill is EXPECTED and says re-derive,
not force).

## 4. Design decisions that need to be visible

1. **VLM model:** the PI's "qwen3.5 9B VL" resolves to `Qwen/Qwen3.5-9B` —
   MEASURED on HF 2026-08-16: it exists, ungated, arch
   `Qwen3_5ForConditionalGeneration` (the multimodal arch `ph0_v2.py` drives
   via `AutoModelForImageTextToText`; it IS the production ph0 arm, so the
   lab reviews the very model the production labels came from). An
   `unsloth/Qwen3.5-9B` mirror exists; **no pre-quantised 4-bit variant
   exists**, so 4-bit is applied at load (`load_in_4bit=True`). Resolution
   happens at runtime from a candidate list, prints what loaded, records it
   in the run manifest, and an older-generation fallback
   (`Qwen3-VL-8B-Instruct` 4-bit) requires an explicit
   `S2_ALLOW_FALLBACK=1` — failing loud beats silently substituting.
   If unsloth rejects the Qwen3.5 arch, a wired-in transformers+
   bitsandbytes nf4 fallback keeps the identical T4 fit (printed which).
2. **Reuse, not reinvention:** the legs are `ph0_v2.run_clip` (B1–B4
   grammar-constrained calls; `ConstrainedVLM.ask` and the enforcer are
   inherited, only the loader is swapped), `ph0_sam3.run_clip_frames` (the
   per-clip block of its `main()` mirrored so the processor loads once per
   session; **the clip count is explicit always** — the `--n` default of 4
   is the measured root cause of the gap), `ph0_pilot` engine A,
   `ph1_fuse` fusion (2-of-3 voting, corroborations, named partials), and
   the `strategic_gt.py` yaw thresholds (25°/150°) for the ego vote —
   carrying that module's own caveat that yaw cannot separate a turn from a
   curving road, which is why ego is one vote, never the sole source.
3. **Provenance per token** (`ego`/`vlm`/`sam3`/`alpamayo`/`map`/
   `fusion_default`) is REQUIRED by the schema validator — the S-S gate's
   goal-provenance audit reads it. Labels may use ego (labels-may-use-ego
   rule); `inference_admissible` still whitelists only vision fields in the
   fused layer, unchanged from `ph1_fuse`.
4. **Banking discipline:** per clip, immediately, far-side verified by byte
   round-trip (`force_download=True` + exact compare — the push log is
   never trusted); resume = far-side listing with a sampled
   filename-vs-content check; smoke banks to `smoke/` in the lab repo,
   never the production label repo.
5. **Headless honesty:** the repo being ON the PI's Drive makes the
   notebooks instantly openable, but free Colab has **no official headless
   execution** — the RUNNER states plainly that execution is Colab-UI-driven
   (PI, or the orchestrator's browser pane after the PI authenticates).

## 5. What this did NOT do

1. **No GPU leg ran** — no GPU on this box, Thor untouched, no pod rented.
   The unsloth load path, the sam3 wheel install, and real per-leg peak
   memory/wall-clock are **T4-unconfirmed**; the memory table in RUNNER.md
   is labelled ESTIMATED until the first real run prints
   `max_memory_allocated`.
2. **The 115 production records are still absent** — the backfill notebook
   is ready-to-run, but a T4 session needs the PI (Secrets + Drive auth).
3. **No re-fuse** — after the backfill, the 115 fused records still carry
   `perception.absent` and must be re-emitted. That step is owned by the
   aug120-fusion package (`AUG120_FUSION_RESULT.md` §9 items 1–2); the
   backfill notebook prints this escalation at the end of every run.
4. **Schema is PROVISIONAL** — the S2-gap agent's
   `…/2026-08-16-s2-strategic-gap/S2_STRATEGIC_GAP.md` was checked
   (twice, start and end of this task) and has not landed yet; the swap is
   one file (`colab/s2_schema.py`), and every lossy v6→S2 mapping edge is
   flagged per record until then.
5. **No commit, no push** (stage only, per the operating standard). Nothing
   in `stack/` was touched; the suite was not affected (all new files live
   under `colab/` and this hub package).

## 6. Escalations

* **→ PI:** the backfill is one *Runtime → Run all* away on the T4
  (RUNNER.md §2 has the two-minute setup). It also doubles as the
  cheapest confirmation of the unsloth/T4 unknowns in §5.1.
* **→ aug120-fusion package owner:** re-fuse of the 115 once backfilled
  (§5.3) — escalated here and printed by the notebook, not buried in a doc.
* **→ S2-gap agent:** when `S2_STRATEGIC_GAP.md` lands, replace
  `colab/s2_schema.py`'s vocabulary/args/mapping and re-run
  `python colab/smoke_run.py colab/STRATEGIC_LABEL_LAB.ipynb` — the
  validator + drift checks make a partial swap fail loud.
