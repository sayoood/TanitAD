# COMMS — W1 (suite core + NAVSIM v2)

## ⭐ Integration asks (also in the report headline)

1. **W5 — LANDED and rendering.** The suite calls `python -m taniteval.benchreport <run_dir>` in a
   SUBPROCESS at the end of every COMPLETE/PARTIAL run and records the outcome in
   `bench_run.json.report`. ⚠️ Success is judged on the ARTIFACT: a non-empty `report/index.html` is
   `RENDERED` even when the CLI exits non-zero, and the code is carried as a `warning` with
   `report/verify.json` — MEASURED 2026-09-20: W5 exits **3** when the page is complete but its
   failure GALLERY cannot be built, and my first two runs mis-recorded that as `REPORT_FAILED`.
   On the run of record: **RENDERED**, W5's own verifier PASS (178 numbers, 7 refusals, 0 errors).
   ⛔ The legacy `taniteval/taniteval/report.py` is never invoked on a run dir.
2. **W4 — consume `summary.json`, and check `bench_run.json.status`.** Read only runs whose status
   is `COMPLETE` (or `PARTIAL`, and then say so in the row); `FAILED` / `REFUSED` runs are recorded
   as evidence and are not results. Validators are importable, no dependencies:
   `from taniteval.bench.schema_check import load_schema, validate` and
   `from taniteval.bench.contract import validate_summary, validate_bench_run`.
   Every arm carries `modality` (sensor_set / setting / ego_status_used / vision_only_claimed),
   `declared_inputs`, `statistics.S2_EPDMS_u`, `paired` vs STOP and CV, and `interval` (or its
   refusal) — the columns the leaderboard needs.
3. **W2 — LANDED and integrated (all four of their items).** The interval goes through
   `rows_from_score_frame` + `interval_from_run`; a run that did not capture the PRE-CSV frame writes
   `interval {UNAVAILABLE, reason: "pre-CSV frame not captured"}` structurally, with
   `pre_csv_frame {captured, path}` recorded either way; artifacts use the adapter's DEFAULT
   `route_leak_check()` (the settled verdict) for command-consuming arms; and the protocol set is
   tested EQUAL to `criteria_check.registered_protocol_tags()` (8 tags, no xfail). Their
   route-level-oracle finding travels in every summary
   (`controls.route_command_is_a_route_level_oracle`, and per arm under `caveats`).
4. **E1 — the navhard aggregation crash has machinery now, not a note.** The promoted wrapper gained
   `--dump-preaggregation` (the `worker_map` seam, pool-safe: it returns in the PARENT), which banks
   the per-token frame as CSV **and** pickle before any aggregation; `scoring.py` classifies that
   case as `AGGREGATION_FAILED` (scoring_complete = true) instead of a failed run; and
   `python -m taniteval.bench reaggregate --preagg <…_preaggregation.pkl> --split navhard_two_stage
   --out <dir>` re-runs ONLY the aggregation with the DEVKIT's own functions. ⚠️ Your banked N1 run
   predates the dump, so it cannot be recovered — a re-run through the suite (or with that flag
   added to your own wrapper copy) will be.
5. **Ownership inside `bench/`.** W1 owns everything except
   `bench/plugins/navsim_v1.py` (W3) and `bench/plugins/nuscenes_ol.py` (W6) — both have already
   replaced W1's docstring-only stubs and load correctly through `plugins.load`.
6. **Master Mind — the STOP floor is machinery now.** STOP and CV are added to every NavSim run by
   rule (`added_by_rule: true` in the record), the summary refuses to validate without them, and the
   HUMAN arm is refused on two-stage splits with E1's reason.

7. **Orchestrator + W2 — a W1-owned file was edited by another stream AFTER my code freeze, and
   only the end-of-turn blob check saw it.** `bench/navsim/artifacts.py` (mtime 10:11) carried W2's
   source-level follow-through: now that `four_families.lateral` emits its own refusal,
   `refuse_undefined_lateral` repairs nothing in the normal case, so the edit makes it *count* a
   term already refused upstream instead of re-writing it. ⭐ **I verified rather than trusted or
   reverted**: both branches exercised (legacy `None` → repaired here; W2-refused → passed
   through), then the four affected test files — **91 passed** — and staged
   (`f87ce653b11858a8db9115329f515d4cc1539c08`). ⛔ **No action needed from W2; the note is for the
   orchestrator**, because the general fact matters more than this instance: **an ownership
   boundary is not a lock.** A MISMATCH on a file you did not touch is evidence about the tree, not
   about your edit — so every stream must re-run the blob comparison at the END of its turn, and
   a stream that edits outside its tree should say so in its report rather than relying on the
   owner's check to catch it. (Mine did; that is luck plus a rule, not a mechanism.)

8. **E1 — all four relayed items are DONE, and one of them changed my verdict about my own tool.**
   (a) **`navsim_win.py` RE-PROMOTED** onto E1's current origin (`96d13540` → `11d5e3cc`; product
   `64186cce` → `935fa44c`): RAM-guard `sustain` **3 → 60**, hard floor unchanged. Re-applied by
   **3-way alignment, never by line number** — E1's edit sits ABOVE both W1 ADDITION blocks and
   shifted them +7 — and the binding check ran BEFORE the write: stripping the marked blocks
   reproduces E1's origin **byte-for-byte**. Tool: `code/w1_repromote_navsim_win.py`.
   (b) **Dry-run vocabulary fixed**: a passing dry run now prints **`BENCH_STATUS=DRY_RUN_PASSED`**
   and exits 0. ⭐ Normalised at the **CLI seam**, which fixes **W3's `navsim_v1.py` too without
   editing a file W1 does not own**; `REFUSED` now means only a real refusal. Schema enum extended
   — ⚠️ **W4/W5: `status` can now be `DRY_RUN_PASSED`; treat it as a PASS that produced no scores.**
   Pinned by 4 tests + mutation **M-DRYRUN** (proven RED, then restored).
   (c) **The IDM degenerate-path patch is now in `devkit_patches()`** (6 entries) with a LIVE
   raw-bytes blob, E1's evidence paths, and E1's framing recorded as
   `why_it_is_a_precondition_not_a_footnote` — 166/5,462 stage-2 tokens, one assert takes the whole
   run down, so **without it the navhard numbers do not exist**. The `bench_run.json.devkit.patches[]`
   plumbing already existed (`benchmark.py:241` → `contract.py:329`); the **entry** was what was missing.
   (d) **The stage-1 cross-check is IMPLEMENTED, not noted**: `summarize.stage1_reference_check`,
   banked in `references.json`, wired into the navhard summary path, 5 tests + mutation **M-S1REF**.

   ⚠️ **One correction to the relay, and it matters for anyone verifying these blobs by hand.** I
   first read the IDM file's blob as `214ef5ee…` and E1's record says `b95bcc7f…`. **Both are right.**
   The file is **100 % CRLF (166/166)**, and **`git hash-object` applies the text/eol conversion**, so
   it hashes the LF-normalised bytes; E1's number is the **raw bytes**. My `profiles.git_blob()` reads
   raw bytes and **agrees with E1 exactly** — so the product was never wrong, only my shell probe was.
   ⛔ The trap is live for everyone: **verify these blobs with raw bytes, never `git hash-object`**, or
   a correct file reads as a mismatch. Recorded in the entry's own `reader_note`.

   ⭐ **And E1's published-match claim is now PRIMARY, not inherited.** I read the navhard
   leaderboard column out of the **banked PDF** myself (library key `2506.04218`, p.8 Table 2, column
   `CV [8]`): NC 88.8 · DAC 42.8 · DDC 70.6 · TLC 99.3 · EP 77.5 · TTC 87.3 · LK 78.6 · HC 97.1.
   Against E1's measured values: **8/8 truncating, 4/8 rounding** — E1's methodological catch
   reproduces exactly, and the 4 that fail under rounding are **precisely the cells whose 2nd decimal
   is ≥ 5**. EC is excluded on purpose (masked in the two-stage score).

9. ⛔ **A DURABLE GAP THIS TURN EXPOSED — `suite_code_blobs` IS A RUN-START SNAPSHOT AND CANNOT SEE A
   MID-RUN CHANGE.** `cli.py:153` captures it once; a devkit-side file is then loaded by a **fresh
   subprocess per arm**, so a change landing mid-run silently makes the record false for every arm
   scored afterwards. It bound today: I applied (a) **while E1's navhard run was scoring**, because
   that run was exposed to the very death the fix prevents and waiting meant ~2 more hours of
   exposure. ⭐ **The change cannot move a score** — the diff is 19 lines, entirely inside
   `Monitor.__init__`; the guard decides only WHEN TO ABORT and is not in the scoring path — and a
   running process does not re-read its source, so the CV arm in flight finished on the pre-change
   file. I recorded it **in the run itself** rather than leaving the record false:
   `…/20260920T082848Z-navsim_v2-none-06e257/raw/CODE_CHANGED_MIDRUN.json` (both blobs, both origins,
   which arms are affected, and why no score can differ). **Ask: capture the devkit-side blob PER ARM
   at launch**, which makes that note unnecessary. W1 will implement it unless the orchestrator would
   rather it waited for the navhard run to land.

10. **Orchestrator rulings + E1's two refinements — all adopted, one premise corrected.**
    ⭐ **PER-ARM devkit-side provenance is IMPLEMENTED** (`scoring.score_arm` →
    `devkit_side_blobs_at_launch` / `devkit_side_drift` / `devkit_side_blob_basis` in
    `<arm>.counts.json`, with a WARNING logged the moment a drift is seen).
    `CODE_CHANGED_MIDRUN.json` is now the fallback, not the mechanism. Mutation **M-DRIFT**, whose
    positive case is today's real event and whose **three negative controls** exist because a
    detector that cries wolf on every arm is one nobody reads.
    ⭐ **E1 (1) — the survival ASYMMETRY is recorded in the run** under
    `THE_ASYMMETRY_THAT_MATTERS`: CV ran `sustain=3`, STOP `sustain=60`; CV's number is as valid
    either way, but **"CV survived" is not evidence the old policy was safe**, and a CV death would
    have been the POLICY, not the DATA.
    ⭐ **E1 (2) — both hashes are now carried, labelled**, in every file-backed
    `devkit_patches()` entry. ⭐ It immediately caught a latent ambiguity **in our own record**:
    E1's historically cited `596cb7d` for `dataclasses.py` is the **NORMALISED** id, while the field
    beside it was **raw-byte** (`20673663…`) — two hash spaces in one entry, and a guaranteed false
    "mismatch" for anyone who compared them.
    ⚠️ **One correction, and please relay it to E1:** the refinement says our `git_blob()`
    normalises CRLF→LF. **It does not and never did** — it reads raw bytes, uses no subprocess, and
    returns `b95bcc7f…`, **agreeing with E1's own patch records**. The normalising number came from
    `git hash-object` at a shell. The ask was right in substance; only the attribution was off. The
    asymmetry E1 states is confirmed and now pinned: **raw-byte is the STRONGER probe for "did this
    file change at all", because the normalising hash cannot see a pure line-ending change.**
    ⛔ **Correction to my OWN escalation too:** I said a mid-run change was invisible to the run
    record. Too strong — `wrapper_sha256` was already captured per arm. What was genuinely missing:
    it covered only the wrapper, it is a **SHA-256** (a different hash space from the run-start git
    blobs), and **nothing ever diffed it**. *A value that is stored but never compared is not a guard.*

11. **Report curation — CONFIRMED as ruled, after settling the ruling's own ⚠ clause by measurement.**
    ⭐ **The render IS deterministic: two independent renders of the same run were 11/11
    BYTE-IDENTICAL**, no render-clock stamp, no absolute paths. So "regenerable" is a valid claim
    and the ruling stands. ⚠️ Scope it honestly: deterministic **relative to a RENDERER VERSION**
    (W5 renamed a module mid-stream today), so a regeneration months from now reproduces *that*
    commit's report — the NUMBERS come from `summary.json`, which is in git either way.
    **Kept:** the run of record (`…5e2458`, 22 files) + internal_t1 (3 files).
    **Unstaged + regenerable:** `…9f38f0`, `…ed36c4`, `…8fde2f` — **3.187 MB**, 91 % of it one run.
    **Mechanism:** `taniteval/results/bench/.gitignore` — a LOCAL ignore file in W1's own tree,
    ⛔ deliberately NOT the shared root `.gitignore`. Each affected run carries
    `REPORT_NOT_IN_GIT.json` (why + the regenerate command + the caveat).
    ⛔ **Two corrections to the ruling's premise, both measured, both worth passing on:**
    **(a)** the reports were **not pending — all 67 were ALREADY STAGED by me** (blanket results
    staging; W5 correctly staged none), so the action was *unstage 42*, not *stage one*;
    **(b)** ⛔ **an ignore rule does NOT apply to a file already in the index** — `check-ignore`
    called the drop file **TRACKED** until it was unstaged. ⭐ Only the NEGATIVE half of the test
    caught that; asserting "the kept file is tracked" alone would have shipped a dead rule.
    ⚠️ And the pattern must exclude `report/**` (contents), never `report/` (the directory): git
    **cannot re-include a file whose parent directory is excluded**, so the directory form makes
    every negation silently dead.

12. ⭐ **`code/w1_stage1_check.py` — E1's stage-1 cross-check, runnable on a CSV, no GPU.**
    ⚠️ Needed because the checker was wired into `benchmark.py` **after** the live navhard run's
    process had imported it, so **that run cannot emit `stage1_reference_check` however correct the
    code is** — the same "a process does not re-read its source" shape as the blob capture.
    E1: the moment `CV.csv` lands, run
    `python code/w1_stage1_check.py <CV.csv> --split navhard_two_stage --arm CV`.
    It prints ours vs E1 (identity) and ours-truncated vs published, and exits **1** if not identical
    to E1. ⭐ CV has already finished stage 1 (it is at **1,725/5,462 of stage 2** at 11:08), so the
    stage-1 numbers exist in the frame the moment the arm completes.

13. ⛔ **The navhard report failure was MINE, it is fixed as a CLASS, and that run is RECOVERED.**
    `navsim/plans.py::write_scenes` wrote `"frame_bank": null`; W5's
    `Path(run.scenes.get("frame_bank", ""))` got **None**, because ⛔ **a dict default fires only on
    an ABSENT key, never on a present-but-null one** — `TypeError: Path(None)` took down the whole
    report of a COMPLETE run. ⭐ **It hid behind a default:** `DEFAULT_BANKS` has a **warmup** entry
    and **none for navhard**, so every warmup run wrote a real string. The defect was there from the
    start and *"it works on warmup"* could never have found it.
    **Fixed:** `write_scenes` now drops **every** top-level null (not just this key) and records
    which; `report_hook` now records a **`reason`** + `how_to_retry` instead of only a log path, and
    returns a STATED `UNAVAILABLE…` for an unreadable/empty log rather than an empty string.
    **Recovered:** scenes.json repaired, re-rendered **through the real hook** → **RENDERED**,
    204,420 bytes, W5's verifier **PASS (481 numbers, 5 refusals, 0 errors)**; `rc 3` only because
    navhard has no frame bank, carried as a `warning` beside the page. `bench_run.json.report` holds
    `repaired_after_failure` with the original failure verbatim. ⭐ **W5: a report-hook failure IS
    renderable after the fact — proven on the untouched run dir, no re-scoring.** Pinned by
    **M-NULLSCENE**, asserted on the ARTIFACTS in the tree (the broken run predates the fixed writer,
    so testing the writer alone would pass while the tree stayed broken).
    ⚠️ **W5, one line worth taking on your side too:** `Path(run.scenes.get(k) or "")` would have
    degraded gracefully instead of raising. Your `.get(k, "")` is a fair read of a contract that
    promised a string or no key — I broke that contract, and I have fixed my side.

14. ⭐ **navhard CROSS-CHECK: the suite reproduces E1 EXACTLY.** `code/w1_stage1_check.py` on the
    suite's own `scores/CV.csv`: **identical_to_e1 = True, 8/8 at 2 dp, n = 450 as expected**, and
    **8/8 truncating** vs the published column. Artifact `raw/navhard_CV_stage1_check.json`.
    **Banked:** CV **11.4816** with the **log-cluster interval [8.25, 14.50]** (S1 28.96 · S2 34.25),
    **Δ 0.0000** vs the HF leaderboard under truncation; **STOP 29.8532**; **HUMAN S1 93.4796**
    (stage 1 only — undefined on two-stage, which is why the suite refuses that headline).
    ⭐ **FIRST quotable NavSim interval in the programme** — warmup's 7 log clusters can never carry
    one (floor 8), navhard has 76. ⇒ my open items **2 (navhard)** and **4 (an interval)** are CLOSED.
    ⚠️ **E1's 19/19 SUPERSEDES my 8/8** as the primary read of the truncation convention (mine is the
    CV S1 column minus EC — a SUBSET). Recorded that way in `references.json`, so the record carries
    one convention rather than two competing counts.

15. ⭐ **W4 — `provenance.ckpt` is emitted and the refusal is STRUCTURAL.** It is the **same object**
    `bench_run.json` carries (never a second triple that could disagree), and
    `contract.validate_summary` — which `write_summary` raises on — **refuses** a summary with a
    `model` arm whose checkpoint is unidentified, so such a run FAILS instead of publishing.
    ⚠️ **One refinement, and please sanity-check it:** `path` + `sha256` are a HARD refusal;
    `registry_key` must be **present OR explicitly accounted for** via `registry_key_status` (which
    `cli.py:138` already writes). Reason: a checkpoint identified by sha256 is **not anonymous**, so
    refusing over a missing cross-reference would reject a reproducible result. ⛔ Silence is what is
    forbidden. Say the word if you want the key itself mandatory and I will tighten it.
    ⭐ **It caught a real anonymous row on its first run, in ANOTHER benchmark:** the banked
    `internal_t1` run has a MODEL arm (`os`) and a **null** checkpoint — while the path sat one file
    down at `refcv3.manifest.model.ckpt`, because the mapper read `rec["ckpt"]`, which is `None` on
    every run of that tool. Fixed and backfilled (`…a3-heldout-20260919/run/ckpt.pt`, sha256
    `54320ec2d72a0b6f…`, both files agreeing). **That row was on the leaderboard path.**
    ⇒ W4: every summary in the tree now carries `provenance.ckpt`; floor runs carry the null triple
    plus the note, model rows carry path + sha256.

16. ⛔ **E1 — your `build_artifacts.py` restructure DELETED three functions W1 promoted, and the
    pre-edit version is unrecoverable.** It now defines only `{build_win, gate_rows_from_criteria_check,
    main, read_csv_rows}`; **`short_row`, `navsim_gates`, `gate_mutations` are gone**, the file is
    **not in HEAD** (never committed) and the index holds the new blob `53051383` — so there is no
    blob left to compare against and **W1's copies are the only surviving ones**. No action needed
    from you; the pin has been changed honestly rather than deleted (`ORPHANED_PROMOTIONS`, pinned on
    non-drift of my copy, with a control that goes RED if you ever restore them so they move back
    under the real AST pin). ⚠️ Worth knowing generally: **a banked package file that another stream
    has promoted from is load-bearing** — if you restructure one, a note in COMMS lets the consumer
    re-pin against the version that existed, instead of discovering it as a RED test hours later.

17. ⭐ **W4 — USE `provenance.ckpt.registry_key_display`; never render the key field raw.**
    Orchestrator ruling 2026-09-20: a row without a `registry_key` must show the **sha256 prefix in
    the key's place, never a blank**. The string is computed ONCE by `contract.ckpt_display()` and
    **written into the artifact**, so please read it rather than re-deriving it — two producers
    formatting the same row independently are two things that can disagree.
    | state | the field reads |
    |---|---|
    | key present | the key, e.g. `refcv4b-b1-v72-40k` |
    | no key but sha256 | `sha256:54320ec2d72a` |
    | path only, no sha256 | `UNIDENTIFIED (path only, no sha256)` |
    | devkit floors | `no checkpoint (devkit floors only)` |
    All six banked runs are backfilled; `internal_t1`'s MODEL row now reads `sha256:54320ec2d72a`
    where it previously had nothing. ⛔ A blank cell reads as "no checkpoint" when the truth is
    "identified, not cross-referenced" — different claims, and only one is a problem.
    ⭐ **And the `registry_key` itself stays OPTIONAL-IF-STATED** (ruling confirmed): `path` + `sha256`
    are the hard refusal; a missing key is legal only with `registry_key_status`. So a page must
    never treat "no registry key" as "unusable row".

18. ⛔ **RETRACTION, and it touches anyone quoting a NavSim external number.** I banked the navhard
    CV external as *"HF leaderboard, 11.4, 1 dp, TRUNCATE, delta 0.0000"*. ⛔ **Our value is 11.4816,
    so that delta is 0.0816** — the entry contradicted its own fields and no new data was needed to
    see it. Cause: I merged **two artifacts**. Corrected, and they never disagreed:
    **paper** (arXiv 2506.04218 v3 Table 2) prints **11.4** at **1 dp** and **TRUNCATES**;
    **HF leaderboard** prints **11.4816** at **4 dp** and **ROUNDS** (Δ 0.0000).
    ⭐ **W3's full-scale v1 measurement settles it and supersedes both my 8/8 and E1's 19/19** — both
    of those were about the PAPER: one underlying value `20.65165…` explains all three sources
    (paper 20.6 truncated, leaderboard 20.6517 rounded; rounding to 1 dp would give 20.7, which is
    NOT what the paper prints).
    ⇒ **The rule: STATE THE ARTIFACT BEFORE THE CONVENTION.** "NavSim publishes X" is not a claim
    until you say which publication. ⚠️ **W4/W5/E1: if any page or report says a NavSim external
    number is 'truncated', check whether it is quoting the PAPER or the LEADERBOARD** — quoting one
    convention against the other source manufactures a FALSE AGREEMENT, which is worse than a
    mismatch because nothing looks wrong.
    **Now pinned by M-SELFREF**, which re-derives every external reference from the entry's OWN
    fields and needs no external data — the reason it should have existed before the entry was written.

19. **W3 — your v1 full-split numbers are banked** under `references.json["navtest_v1"]`
    (CV 20.6517 [19.20, 22.22] · STOP 61.8202 [60.70, 63.08] · HUMAN 94.5514 [93.79, 95.23],
    136 logs, 12,146/12,146, PDMS identity max |Δ| = 0.0), and
    **`summary_row_shape('PDMS_v1_navtest')` needed NO change** — `('average',)`, n = 1. Its evidence
    now cites your FULLSPLIT artifact: the 20-token smoke read reproduced at **12,146 tokens on three
    independent arms**. ⭐ A pre-registered scale-up that leaves the stamp unchanged and the evidence
    stronger is exactly the outcome it is for.

## ⭐ The SHARED DEVICE GATE (orchestrator arbitration, 2026-09-20) — ONE implementation

`taniteval.bench.gpu_gap.device_policy(requested, accept_training_box_load=…, probe=…, queue_hint=…)`
is the single implementation for the whole suite (W1's benchmarks **and** W3's / W6's plugins —
import it, do not re-derive it). Decisions, all unit-tested with literals + a RED mutation:

| `--device` | training alive? | decision |
|---|---|---|
| `auto` / `cuda` | — | **CUDA** if a gap is open (no training process AND device memory < 1,024 MiB), else **REFUSE** with the queue command. Model arms WAIT for a gap; they are never silently moved to CPU |
| `cpu` (explicit) | no | **CPU** — the operator has chosen the hours |
| `cpu` (explicit) | yes | **REFUSE** unless `--accept-training-box-load`; with it, **CPU** and the flag **plus the accepted PIDs** are written into `bench_run.json.device_policy` |

⛔ The override accepts CPU load only — it can never take the card from a trainer (`--device cuda`
+ the flag still REFUSES). ⛔ `--ckpt none` runs never consult the gate: the devkit's scorer is CPU
by construction and is not our-model inference (pinned by a test).

## Decisions made in this stream (each reversible; the PI / orchestrator may overrule)

| # | decision | why (evidence) |
|---|---|---|
| D1 | `taniteval.bench` becomes a PACKAGE that DELEGATES every legacy attribute to `bench/_legacy.py` (which executes the pre-existing `bench.py` unchanged); `--model/--all` route to the legacy CLI | the brief's path collides with a tracked module used by `runner.py:39`, `refc_rerank.py:78`, `generalization.py:901`, 6 test files, 5 stack scripts and `rerun_all.sh:7`. Before/after: **81 passed / 81 passed** (`raw/legacy_importers_*.txt`) |
| D2 | ONE W1 ADDITION to the promoted E1 wrapper: `--dump-preaggregation` at the `worker_map` seam (marked; stripping it reproduces E1's file byte-for-byte) | MEASURED 2026-09-19 navhard CV: 5,912 scenarios scored in 68 min, aggregation raised, NO CSV, and `pdm_calls=0` because a process pool's children do not inherit the in-process hook |
| D3 | files > 20,000,000 B move to `C:/Users/Admin/tanitad-bench-offrepo/…` with a `.offrepo.json` stub carrying the sha256 | BUILD_PLAN §1 ("anything > 20 MB off-repo with its sha256 in the manifest") |
| D4 | NavSim experiment outputs go to `C:/Users/Admin/navsim-crun/exp/tanitad_bench/<run_id>` (C:) | `C:/Users/Admin/navsim` is a JUNCTION to D:; A7 reads D: and the Master Mind asked for modest D: I/O |
| D5 | SUPERSEDED by the shared device gate above (orchestrator arbitration 2026-09-20): `auto`/`cuda` REFUSE with the queue command when there is no gap. `GpuGapLauncher(requested="cuda")` still blocks up to 8 h for a caller that chooses to wait, and records `GAP_WAIT_TIMEOUT` rather than waiting forever | PI *"wait for a gap in the gpu"* + the standing "never add load to a box that is training" |
| D6 | a recorded PI submission approval is the line `SUBMISSION-APPROVED: <id> target=<t> benchmark=<b>` in a PI decision record **as committed in HEAD** | an agent's worktree edit must not be able to manufacture an approval; the gate is proven able to PASS (injected record) and to REFUSE everything real today |
| D7 | `internal_t1` protocol tag `TanitAD_T1_refc_physicalai`, floor `ha0`, headline ADE (lower is better), interval admissible only at ≥ 8 episodes (RG-14) | `refcv3_arm.py`'s own arms + `REFCV3_ARM.md` §2; the tool stamps `os` as `T1*` with the ruling OPEN and the suite carries that stamp through |
| D8 | `--dry-run` exits 0, writes to a scratch root, and marks arms `SKIPPED` | a preflight is not a result and must not litter `taniteval/results/` |

## Coordination notes

* E1's and E2's package files were READ and PROMOTED, never edited. Every promoted function is
  AST-pinned to its origin (`test_bench_suite_promotion.py`); the devkit-side files are byte-pinned.
* `taniteval/adapters/navsim.py`, `tools/criteria_check.py`, `products/P7-TanitEval/CRITERIA_REGISTRY.json`
  are W2's: imported and called, never edited. The suite fills E1's gaps G1–G6 in its own artifact
  builder and records them under `_w1_adapter_gaps`.
* ⚠️ **Finding for W2 / the four-families owner:** on a stationary plan (the STOP floor)
  `four_families.lateral` emits `heading_mae_deg` / `curvature_mae_1pm` / `yaw_rate_mae_degps` as
  **`None`**, and `criteria_check.py` reads a `None` as ABSENT — 3 silent-omission VIOLATIONS. The
  suite converts them into refusals with the harness's own counts (`_w1_lateral_refusals`), but the
  honest fix belongs in `four_families.lateral` itself.
* ⚠️ **Read before comparing LATERAL rows across arms:** `cross_mae_m` is computed as
  `P["cross"] - G["cross"]` in the EGO frame (`four_families.py:776`), so ANY plan with `y ≡ 0`
  scores identically — MEASURED: CV and STOP both read **1.0658 m** on warmup stage 1. It is not a
  suite defect and it is not a tie in driving quality.
