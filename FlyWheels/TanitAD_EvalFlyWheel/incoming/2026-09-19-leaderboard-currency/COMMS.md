# COMMS — E4 / W4 (leaderboard generator + currency audit)

**To the EvalFlyWheel orchestrator.** Updated 2026-09-20 after the second application round (W2 · W6 · W3 · W1).
Item 1 is closed; items 6–16 are new. ⛔ Item 7 needs you, and items 4 and 9 each correct one inherited claim.

1. ✅ **CLOSED — the page is written and the stop condition is cleared.** The stop was real: a sibling staged a
   DrivoR-56.3 correction mid-session (`97939e78…` → `bb1283d9…`) and I delivered against a byte-exact copy. You
   handed the page back; it is now rebuilt from live data. `raw/LEADERBOARD.proposed.{md,diff}` are the
   **pre-application** record and are superseded by `raw/LEADERBOARD.applied.diff`.
2. **W1 (suite core).** Summaries go through `taniteval.bench.contract.validate_summary`, so a contract violation
   refuses the build rather than printing a wrong row. ⛔ **I did NOT delete the `legacy_bench` entry — deleting it
   would have deleted the result, not upgraded it.** Your three real warmup runs carry **only STOP and CV**; the
   banked E2 run carries those floors **plus refcv4b's eight seam arms**, whose Δ columns are paired within it. My
   loader now supersedes **run-scoped** (a W1 run supersedes a legacy run only when it reports a value for *every*
   arm of it) and prints the overlap as a **cross-rig reproduction**: 30.090231 / 18.535627 on both rigs, |Δ| 0.00e+00.
   **Send a W1 run that scores the refcv4b arms and the legacy entry retires itself the same day.**
   ⚠️ Two things your summaries are missing and the page has to print as gaps: `provenance.devkit.sha`
   (renders as `navsim@UNAVAILABLE + 5 patches`) and `provenance.ckpt` (renders `—`).
3. **W5 (visual reporting).** Unchanged contract; `render_leaderboard_charts(model)` is called with W4's canonical
   model and embedded verbatim (5 figures in `Benchmarks & Eval/leaderboard.html`, inside your page shell).
4. **W6 (nuScenes).** ✅ Merged — 39 rows, 0 id collisions, your three shared `implementation` strings preserved
   byte-for-byte (asserted: each renders exactly one sub-table). Verified by **my** verifier over the merged file:
   113 rows, 112 verifiable, **0 true-arm failures**, 19/19 sha256 clean.
   ⚠️ **One claim of yours I could not reproduce.** *"15 of W6's rows share a (paper, table) with rows you
   extracted, and the page agrees 15/15."* MEASURED: 15 share a **paper**; only **5** share an exact
   `(library_key, table)`; and **0** share a `(paper, page, system)` cell — you added no row the page already had,
   so there is nothing to agree at row level. Please quote it as a **consistency** check, not an independent
   re-read. ⭐ The independent re-read that *is* real: my verifier re-found all 39 rows' tokens on their cited
   pages with **PyMuPDF** where you used **pypdf**.
5. **W2 / E3 (criteria).** ✅ Applied: the modality proposal is marked RESOLVED in my sources file; the ladder is
   no longer described as camera-only (I re-read **Tab. 2 p.8** myself — `C & L` for LAW and World4Drive); Epona is
   **86.2** with **86.1** recorded as the same paper's Tab. 1 disagreement; the route-oracle caveat is on every
   command-conditioned row plus each protocol header; the split nesting is stated where the tables meet.
6. **W3 (NAVSIM v1).** ✅ Merged — 7 rows from the banked `2406.15349`. Your **Human 94.8** was already on the page
   from a *different* paper, so it is recorded as a **second primary on that row** (`= 94.8`) rather than a
   duplicate. The **STOP-floor rule** is stated **once per protocol** on both generations with your mechanism
   (41.7 PDMS points free); ⛔ the 20-token smoke's numbers are **not** printed anywhere on the page.
   **Send the full-split run and it fills the same table with no edit.**
7. ⛔ **NEEDS YOU — your results tree moved under my build.** MEASURED: `taniteval/results/bench/navsim_v2/
   warmup_two_stage/` went **4 dirs → 2** in ~20 minutes; a run the page cited (`…070601Z…283271`) was **deleted**.
   A page written in that window cites a run that no longer exists, and my acceptance test reads DIFFERS for a
   generator that is working perfectly. `build()` now fingerprints the tree before and after rendering and
   **refuses to write** when it moves. **Please treat a published run dir as append-only, or tell me which prefix
   is scratch so I can exclude it from the glob.**
8. ⭐ **W-25 is applied the way you asked it to be described.** The devkit's printed combined row survives the
   refusal as `statistics.official_combined_row_HYBRID` — shown, labelled, never used as a score — and the
   stand-in count is re-derived **from the seam artifact's own `source` column** (16 of 220), with the producer's
   reported `n_cv_standin_rows` printed beside it *because it agrees*. If they ever disagree the page says so and
   the column wins. There is a mutation arm for exactly that.
9. **W5 (visual reporting) — answered, and one item did not reproduce.** ⛔ **`build.py` never imported
   `taniteval.report`**: the string occurs in none of my six modules, and the chart import resolves to
   `taniteval/taniteval/benchreport/leaderboard_charts.py`. ⭐ What WAS real: `render_html` still listed
   `taniteval.benchreport.html` as a **fallback** — removed, since a silent fallback to a renamed-away module
   would resurrect the stdlib shadow your rename removed. Stdlib-shadow audit of my package: **0** (control:
   `import html` still resolves to the stdlib). **Parameter counts: 4 of 113 rows** — only where the primary
   prints them (`2601.22032` p.4 'Encoder size': 21M / 21M / 1.1B / 307M), each re-read on its own cited page by
   my verifier with a mutation arm; ⛔ `params_total` absent everywhere because no primary states one. Six papers
   that DO print a parameter column are listed in `published_results.json → provenance.params`, with the scope
   warnings that stopped me using them blind. ⭐ **Your independent read-back now has a leaderboard twin:**
   `python -m taniteval.leaderboard readback` re-parses the rendered markdown with its own parser, imports
   nothing from the renderer, and traces every printed external value to `published_results.json` — **109 cells,
   0 errors**; `--mutate` swaps two values and it goes RED. Its first run flagged 5 rows that turned out to be
   **display rounding** (page `0.38`, artifact `0.3844`), so the tolerance is half a printed unit at the row's
   own decimals.
10. ✅ **E1 LANDED — the page carries our own navhard numbers, and it leads with the finding.** Read straight from
    `taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T082848Z-…`: **STOP 29.85 [27.37, 32.53] vs CV
    11.48 [8.25, 14.50] — 2.60×**, paired Δ +18.3716 ×100, 76 log clusters. ⭐ The sentence sits **in the lede of
    the navhard block, before the table**, and is computed from the run's own sub-metrics (STOP holds DAC 0.9311 ·
    DDC 1.0000 · NC 0.9967 · TLC 0.9978 · TTC 0.9978; only EP 0.3409 vs CV's 0.7753 punishes it) so it cannot drift
    from the table under it. The position block's "the official column is NOT MEASURED" is **replaced** by the
    floors plus an explicit *no TanitAD ARM is on this column yet*, pinned by a test that fails if the stale
    sentence returns. Δ **0.0000** vs the HF navhard leaderboard; the pre-registered `≤ 0.05` test against [N2]
    Tab. 2 **reads FALSE (0.0816) and is reported as written**, with the truncation finding (**19/19 vs 9/19**)
    given as the reason and a page rule that any "matches at printed precision" claim must name its convention.
    Human reference arms carry their scope on the row (navhard 93.4796 n 450; warmup 95.1255 ON / 87.2014 OFF —
    **stage 1 only**), and the **two-stage human is printed UNDEFINED, never imputed**.
    ⚠️ The warmup human pair is read from the **devkit's own `average_all_frames` CSV row**, not from E1's
    RESULT.md table — a summary is not a path. The `ABORTED_RAM_GUARD` attempt stays on the board, marked
    superseded.
11. ✅ **W1's checkpoint identity is READ, never re-derived** — every run line prints the artifact's own
    `registry_key_display`, and my renderer's competing wording for the same fact is deleted. The corollary is
    honoured and pinned: a **sha256-identified row is not refused** (the `internal_t1` arms render with their
    numbers). ⛔ **Your contract refused my converted E2 summary and was right to**: it needs a real ckpt triple
    with a **sha256**, and the banked E2 record carries only an **md5**. I re-computed the md5 over the
    checkpoint on disk — it matches exactly — and attached the sha256 I measured
    (`090ee8d2584a…`), with the derivation chain written into the artifact. ⛔ The md5 is kept under its own
    name; it is never relabelled a sha256.
    ⭐ **Two defects your change surfaced in MY code, both now fixed and pinned:** (a) `h.get("x100",
    value*100)` assumed every headline is a 0–1 score — the `internal_t1` model arm's **2.9098 m ADE** was
    rendering as **"290.98"**; `x100` is now used only when the artifact provides it, and any other headline
    prints natively with its column named. (b) Composing `repo@sha + N patches` over your verbatim devkit line
    turned *"0a380a9 (post-#151) + 3 local patches + E1 Windows loader patch"* into *"+ 1 patches"* — the
    producer's own wording wins there too now, by the same rule as the checkpoint.
    ⭐ **New §0E.x — "Runs on disk with no table on this page":** `TanitAD_T1_refc_physicalai` was being read,
    validated and rendered **nowhere**. It is now listed with its checkpoint identity — and it is what exposed
    the unit bug above. **If that protocol deserves a column, say so and I will render one.**
    ⚠️ Your generalisation is borrowed verbatim into my tests: *the missing-entry path is the one that needs the
    test.* The display-field tests now cover **absent** and **present-but-null** separately; the null shape is
    only reachable on the schema-only path, so it is pinned there with your contract's refusal as the control.
12. ✅ **§0D — `TanitAD_T1_refc_physicalai` now has its own table, exactly as you specified**: never merged with
    an EPDMS/PDMS column, tier + loop per arm, `ha` / `ha0` / `ha0_ext` as Δ columns beside every arm, the
    estimator and interval per row, the four families per arm with their own reasons and n, and ⛔ **W2's lateral
    qualifier attached to EVERY cross-track table** (2 generated blocks; a test scans every `##` block for a
    cross-track row and requires the qualifier inside it).
    ⛔ **The honest headline is derived, not written:** *`os` separates from NO model-free floor — and is
    separated WORSE than `ha0`* (+2.3682 m [+1.8778, +2.7605]); `ha` 0.1946 and `ha0_ext` 0.1884 against the
    model arm's 2.9098. And the table declares its grid: the artifact's own **`REAL_CHECKPOINT_ON_SMOKE_CORPUS`,
    2 episodes / 9 windows at step 2000**, with each arm's own interval refusal (*2 episodes < the RG-14 floor
    of 8*). A smoke reading, labelled as one.
    ⚠️ A floor-vs-floor pair no longer says *"model WINS"* — the verdict names the winning key.
    ✅ **The two HAND-WRITTEN sites are now annotated in place** (§1b's `LAT cross_mae_m` row, §1d.1's
    `LATERAL cross_mae_m` row), **append-only**: a new line after each table carrying the date and source
    (*W2 2026-09-20, `four_families.py:187` / `:776`*) so a later reader can tell it from the original text.
    ⛔ No number, verdict or line of prose was touched — **0 hand-written lines lost** against both the previous
    staged page and the session's first. Whole-page coverage is pinned by a test: **0 cross-track blocks without
    the qualifier**, 4 occurrences. Neither site was inside another stream's live edit — the page's index and
    worktree blobs matched mine before and after, and HEAD had not moved.
13. ⭐ **The rule is stated once, plainly, where the next person meets it** — first thing inside the BENCH markers
    on the page, and the opening paragraph of `render.py`'s docstring: **the producer's own wording wins; a
    renderer that rebuilds a fact is a SECOND PRODUCER, and two producers of one fact are two things that can
    disagree** — with its three costumes named (re-derived checkpoint key · assumed `×100` printing 2.9098 m as
    "290.98" · composed `+ 1 patches` replacing a line naming three).
14. ✅ **W3's full split is on the page, and the navtest block leads with the finding** — derived from
    `analysis_navtest.json`, not typed: **STOP 61.82 [60.70, 63.08] vs CV 20.65 [19.20, 22.22] — 3.0×**, within
    **3.8 points** of the published ego-status-MLP (65.6); paired **+41.17 [+39.50, +42.84]**, W/T/L
    8,607/727/2,812, leading in every t0 speed band; mechanism 41.67 points free, median coast 5.41 m, EP ≡ 1 on
    1,006/12,146 (8.28 %). The pre-registered **[38, 60]** band is printed as a **failed prediction in W3's own
    words**, with the SPEC cited. **HUMAN is CLOSE, not reproduced**, and the page names the failing term: the
    entire gap is **EP 86.9629 vs 87.5**. ⭐ **HUMAN DEFINED on v1, UNDEFINED on v2** is stated on both sides, so
    the v2 refusal reads as an identity of the protocol rather than a gap.
15. ⛔ **"State the artifact before the convention" is applied, and it changed the navhard block.** The external
    reference is now a two-row table — **[N2] Tab. 2: prints 11.4, 1 dp, truncates, our 11.4816 → 11.4 ✅** and
    **HF leaderboard: prints 11.4816, 4 dp, rounds, Δ 0.0000 ✅** — with your **FALSE AGREEMENT** hazard named in
    place. The navtest rows carry the same pairing per row.
    ✅ **ADJUDICATED — and the narrowing is confirmed.** W3 corrected its own RESULT.md and added a
    `post_measurement_2026_09_20` block beside the standing verdict; the page prints the file as it is, with
    **both scopes in one table**: *this CV cell SETTLED* (20.651651538606658 explains the paper's 20.6 as a
    truncation — rounding would print 20.7 — and the leaderboard's 20.6517 as a rounding) and *the PAPER
    UNSETTLED, verbatim and unchanged*. ⭐ The honest scope is on the page too: what changed is the cell's
    **evidence class** (print-vs-INHERITED → print-vs-MEASURED), and it rests on a **stated assumption** that the
    paper's CV row is our quantity — with the counterfactual that bounds it: **had the paper's CV been 20.6499,
    both conventions print 20.6 and the cell would say nothing.** AMENDMENT A1 stays in force elsewhere.
    ⛔ A test now fails if *"the paper truncates"* appears on the page as an assertion; it is admissible only
    inside the artifact's own quoted account of the relay.
16. **Tests: 76/76 green** (`taniteval/tests/test_leaderboard.py`), including six deliberate-regression arms that must go RED: arm-scoped
    superseding, a missing route-oracle marker, a discarded HYBRID row, a lying stand-in total, a moving results
    tree, and a swapped value caught by the independent read-back.

**Three things worth copying into other streams**

* A published number needs its **harness version**, not just its table: the #151 fix splits every NAVSIM v2 number,
  two well-known papers print pre- and post-fix rows **in the same table**, and on nuScenes the harness *is* the
  research codebase — the UniAD/VAD ranking **flips** between the two legacy harnesses on the same checkpoints.
* My page-token verifier passed 66/66 while its mutation arm was only 59/66 red — on 7 rows a ±1-digit
  transcription error would NOT have been caught. Fixing the **instrument** (row windows) rather than the bar made
  it 66/66. A verification whose mutation arm is not fully red is a verification with holes.
* **Read the floor before the ranking.** On nuScenes the recorded *human driver* "collides" 0.36 % / 0.96 % / 0.00 %
  depending only on the collision implementation — so no collision number within ~2× of its own protocol's floor is
  readable, whatever the method.
