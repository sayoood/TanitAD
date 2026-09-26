# COMMS — W6 (E5) integration asks

**To the EvalFlyWheel orchestrator.** Nothing below is written into a README and waited on; each
item names its owner, the artifact to take, and what breaks if it is not taken.
*(Class C26 of this programme: an orthogonality instrument sat unmerged for 10 days because the
request lived in a file nobody re-read.)*

## 1. PI — nuScenes registration and download (BLOCKING everything numeric)

The action list is `RESULT.md` §"PI ACTION LIST — nuScenes": three human-only steps (account,
**Terms of Use**, download page), the minimal file set with MEASURED sizes (**45.76 GB** for the
planning eval *and* the SAM3 paint reference together; 0.46 GB answers the value question alone),
the D: landing layout with the exFAT 1 MiB-cluster warning, and
`code/fetch_nuscenes_after_tou.sh`, which **refuses to run** without the PI's acknowledgement.
⛔ No agent may register, accept terms or download.

## 2. W2 (estimator-and-gates) — take `code/criteria_guard.patch`

* **What**: the `EXTERNAL_ONLY` scope + `nuscenes_planning_state()` in `tools/criteria_check.py`,
  the gate `benchmarks.nuscenes.tasks.planning_openloop.GATE_not_a_criterion` in the registry, and
  five tests (one EXTERNAL_ONLY arm, **two deliberate misuse arms**, one control, one registry pin).
* **Base**: built against your worktree files of 2026-09-20 (raw-byte sha1s in the patch header;
  your checker at `check_benchmarks` / `_row` / `protocol_tags`).
* **VERIFIED**: `git apply --check` clean; in a scratch mirror the patched suite reads
  **171 passed / 4 skipped**, and against the UNPATCHED checker the four guard tests **FAIL** —
  they test the guard, not the fixture.
* **Why it matters**: MEASURED on a real artifact from this harness — today your checker reads it
  `UNKNOWN_SCOPE` ("a human must classify these"); with the patch it reads
  `EXTERNAL_ONLY: nuScenes open-loop planning …; claim_bearing false`. And the two misuses
  (claiming one, or embedding nuScenes numbers in a driving artifact) become violations instead of
  passing silently.
* If a hunk conflicts because you moved on, the patch header lists the five self-contained pieces.

## 3. W4 (leaderboard) — take `code/published_results_nuscenes.patch.json`

**39 published rows, written in YOUR row schema** (`id, protocol, implementation, system, role,
values, source{library_key,table,page}, page_tokens, harness{fix151,basis}, modality, ego_status,
command, comparison_admissible, notes`), ready to append to `results`. **0 id collisions** with the
8 nuScenes rows you already have (they are listed in `_already_in_w4`).

* **They JOIN, they do not shadow.** Both protocol keys are yours (`nuScenes_OL_L2_uniad`,
  `nuScenes_OL_L2_stp3`) and are the same tags my run directories carry. The three
  `implementation` strings that already exist in your file are reproduced **byte-for-byte** —
  including `"UniAD planning_metrics (at-timestep)"`, which I first wrote with a longer
  parenthetical and corrected, because a one-word difference there would have split your UniAD
  table into two sub-tables instead of extending it.
* **Two new implementations**, both needing their own sub-table exactly as BEV-Planner's does:
  `SparseDrive re-implemented collision (box-vs-box, estimated yaw; L2 per VAD)` and
  `PARA-Drive standardized (oriented ego box, 0.1 m grid, first frame removed, + map compliance)`.
* **Your finding applies here and is answered in `_harness_version_note`:** for nuScenes there is
  **no devkit metric at all** — the harness IS the research codebase. Each row's `harness.basis`
  names the reduction, and the note pins the reductions to commits
  (`UniAD@609ee08` `nuscenes_e2e_dataset.py:1043-1044`, `ST-P3@69aabef` `evaluate.py:166`,
  `VAD@1688c4b` `metric_stp3.py`, `AD-MLP@4b93ba0` re-using ST-P3) and lists the four mutually
  incompatible collision implementations.
* **Per row, as you asked**: convention (`protocol` + `implementation`), `ego_status`, `command`
  (**GT-derived on every published row** — `_command_note` carries the four file:line proofs),
  `library_key`, `table`, `page`.
* ⛔ **VERIFICATION — AND A RETRACTION OF WHAT I CLAIMED FOR IT.** The builder
  `code/make_w4_rows.py` re-hashes every cited PDF (**sha256 == `library.json`, 9/9**), asserts
  every `page_tokens` string occurs on its cited page, refuses to write on a miss, and carries a
  mutation arm that plants a bogus token and requires the checker to catch it. ⚠️ **But it does
  that with pypdf, the same extractor that CHOSE the tokens** — the producer's own derivation
  re-run, which measures transcription and determinism, never whether the number is the one the
  table prints.
  ⛔ I earlier offered *"15 of my rows share a (paper, table) with rows you extracted, and the
  page agrees 15/15"* as the independent control. **W4 could not reproduce it and W4 is right.**
  Re-derived (`code/recount_row_overlap.py`): 15 of my rows sit on a paper you also cite, only
  **5** share an exact `(library_key, table)` — my join key stripped the sub-table parenthetical
  (`'Tab. 1 (ID-3)'` → `'Tab. 1'`) and inflated 5 → 15 — and **0** share a `(paper, page, system)`
  cell, **0** carry a value-set you also published. ⇒ **no number in the 39 rows is double-read**,
  and what agreed was only which page a table sits on. Banked as `RESULT.md` **F12**.
  ⭐ **So your PyMuPDF `verify_published_against_pdfs.py` is now the ONLY independent check these
  rows will get, and it has not been run on them.** Please run it on the merged file and tell me
  what it says; if a token misses, the builder re-derives that row from the banked PDF in seconds.
  ⚠️ And note the trap for whoever re-checks this after your merge: the reference file now
  CONTAINS my 39 rows, so an unfiltered re-run of my overlap script reads **39/39** and looks
  perfect — it is comparing my rows with themselves. `recount_row_overlap.py` filters my ids out
  by construction and prints both counts.
* **Three rows are marked `comparison_admissible: false` with a reason** — ST-P3's averaged numbers
  as reprinted inside UniAD's at-t table, and the two Senna rows, whose paper never states its
  reduction (the tag is INFERRED from its VAD backbone). ⛔ Please keep the reason visible rather
  than dropping the rows: "this number's convention is unstated" is the finding.
* **Three GT-HUMAN-TRAJECTORY control rows** (`role: control`), one per legacy protocol plus
  PARA-Drive's step-by-step correction. They are the instrument's false-positive floor: the human
  driver "collides" **0.36 %** (UniAD methodology) and **0.96 %** (VAD methodology), and exactly
  **0.00 %** once the box is oriented and the grid is 0.1 m. No collision number within ~2× of its
  protocol's floor is readable, and the leaderboard should say so where those columns are rendered.
* ⛔ **One table per (protocol, implementation), never a shared column** — the averaging convention
  alone moves one checkpoint 0.72 → 1.22 m, and the UniAD/VAD ranking **flips** between the two
  legacy harnesses (`nus.uniad.vad_rescored` vs `nus.stp3.uniad_rescored`, both PARA-Drive Tab. 8).
* **Our own row is `NOT MEASURED — pending the PI's download`** (`_our_row`), and when it lands it
  is comparable ONLY with the ego-status-free rows (`nus.bevp.uniad_noego` 1.03,
  `nus.bevp.vad_noego` 1.25, your `nus.bevp.bev_planner` 0.55), because our binding rule forbids
  ego status at inference and our arm is command-free.
* Human-readable twin (unchanged, 5 protocol families in prose): `raw/nuscenes_external_rows.md`.

## 4. W1 (suite core) — two CLI flags, and one schema note

* The plugin `taniteval/taniteval/bench/plugins/nuscenes_ol.py` is landed and runs
  (`--ckpt none --split mini_val` validated against both your schemas in my tests).
* **Ask**: add two `nuscenes_ol`-only arguments to `build_parser`, the way `navsim_v2` gets
  `--frame-bank`: `--protocol {nuScenes_OL_L2_uniad,nuScenes_OL_L2_stp3}` (⛔ no default: one
  convention per run, and a default is how two conventions end up in one column) and
  `--nuscenes-root`. Until then the plugin reads `TANITAD_NUSCENES_PROTOCOL` /
  `TANITAD_NUSCENES_ROOT` and **REFUSES (exit 2)** when the convention was not chosen — tested.
* FYI, no action: for `--ckpt none` the plugin never calls `ctx.gpu_device()`, so
  `device.used` comes out as your `"none"` default; a model arm runs CPU-only for now.

## 5. DataFlyWheel / owner of `stack/tanitad/data/nuscenes.py` — one stale line

Its acquisition message tells a human to download **`nuScenes-map-expansion-v1.2.zip` (16 MiB)**.
The current devkit REFUSES any map older than v1.3 (`map_expansion/map_api.py:99-101 @b40adc4`:
*"You are using an outdated map version"*). The live object is
`nuScenes-map-expansion-v1.3.zip`, **398,535,531 B** (bucket LIST + HEAD, 2026-09-19). The rest of
that docstring's measured sizes still reproduce exactly (metadata 461,678,030 B; keyframes
44,902,690,772 B; CAN bus 780,974,697 B). I did not edit the file — not my ownership.

## 6. For whoever runs the first real nuScenes pass — the first-contact checklist

`RESULT.md` §"First contact with real data": the sample counts must read **6,019 / 5,119 / 4,819**;
`vad_category_index_audit()` must show VAD's literal index sets selecting human/vehicle categories;
the GT-collision floor should land near PARA-Drive's **0.36 % / 0.96 %** — if it does not, our
occupancy differs from the references' and no collision number is quotable yet.

## 7. Orchestrator — one RETRACTION_LOG entry, which is yours to make

`Project Steering/RETRACTION_LOG.md` is a steering file and my brief forbids me to edit it, so the
entry belongs to you. ⚠️ It matters beyond my package because the 15/15 was relayed to the PI as
an independent agreement. Text, ready to paste:

> **2026-09-20 — W6/E5, nuScenes published rows.** RETRACTED: *"15 of my rows share a (paper,
> table) with rows W4 extracted, and the page agrees 15/15, 0 disagreements"*, offered as an
> independent control on 39 published rows. CORRECTED: 15 share a paper, **5** an exact
> (library_key, table), **0** a (paper, page, system) cell, **0** a value-set — no number was read
> twice. **Root-cause class: a cross-check derived from a different QUANTITY than the one at
> risk** — the check was real and run by another agent with another library, but it agreed about
> a table's PAGE while the claim was about that table's VALUES; the `df` / `step_s` scope family,
> in which a true measurement quoted outside its scope reads exactly like an answer.
> **Contributing cause: a join key normalised for convenience** (`'Tab. 1 (ID-3)'` → `'Tab. 1'`)
> silently became the claim's denominator, inflating 5 → 15. **Third trap, found while
> re-deriving:** once W4 merged the rows the reference file contained them, so the same script
> reads 39/39 — self-agreement dressed as a cross-check. Instrument:
> `…/2026-09-19-nuscenes-planning-harness/code/recount_row_overlap.py`. Corrected in RESULT.md
> F12, COMMS.md §3 and the patch file's `_built.independent_reads`.

## 8. Not done, and named as such

* **ST-P3's occupancy builder** (per-frame labels warped into t0 with `grid_sample`) — the ST-P3
  *pipeline* scores L2 only; both shipped protocols use the UniAD and VAD pipelines, so nothing in
  the contract depends on it.
* **`cv2.fillPoly` parity** — the port reproduces three literal masks, but the parity arm against a
  real OpenCV is **NOT RUN** (no cv2 in the venv; installing one is out of scope). It is a
  `pytest.importorskip` with that reason, and a skip is reported as NOT RUN, never as a pass.
* **The model arm has never met a checkpoint or an image** (there are none here): the plumbing is
  tested with a fake model, `refc_forward` (W1's promoted E2 bridge) is not.
