# RESULT — E4 / W4: a leaderboard that regenerates itself, and what the currency audit found

**Stream** EvalFlyWheel E4, resumed as **W4** (`…/2026-09-19-eval-suite-build/BUILD_PLAN.md`).
**Dates** 2026-09-19 → 2026-09-20. **CPU only, 0 GPU, nothing re-measured.** Every number on the page is
read at build time from the registry, a raw eval JSON, a W1-schema bench summary, or a banked PDF.

---

## 0. HEADLINE — and one thing the orchestrator must decide

1. ✅ **The page is written — after stopping once, on purpose.** At session start the page's worktree blob
   equalled HEAD (`97939e78…`); mid-session a sibling appended a DrivoR-56.3 correction and **staged** it
   (`bb1283d9…`), so I stopped and delivered against a byte-exact copy. The orchestrator then measured the page
   stable and handed it back; I applied my hand edits to the **live** file and re-ran the generator against it.
   ⭐ The sibling's correction and my §0E agree (56.3 = DrivoR **+134k synthetic** + TOAD test-time search;
   DrivoR alone 48.3 navtrain-only / 54.6 +134k).
   **Proofs on the written page** (`Benchmarks & Eval/LEADERBOARD.md`, blob `09c11cd3…`, staged):
   * the sibling's work survives by **positive assertion**: its correction text, its `56.3 = DrivoR (+134k
     SimScale) + TOAD test-time search` line and its package reference each occur exactly once;
   * **nothing hand-written was lost**: of the 1,719 lines of `bb1283d9…`, **5** are not present verbatim — the
     five rows I annotated in place — and for each, **every token survives** in the new file (a superset /
     descendant check; *"differs from the old version" is not the test, because a revert also differs*).
     Lines whose content was LOST: **0**;
   * **acceptance on the live page**: delete the section → build → **byte-identical in both deletion modes**,
     and a rebuild in place reports no change;
   * **1,718 → 2,202 lines** (+484), 150,165 → 213,467 bytes, CRLF preserved (0 lone LF).
   `raw/LEADERBOARD.applied.diff` records exactly what landed (489 additions, 5 in-place row edits).
   ⚠️ `raw/LEADERBOARD.proposed.{md,diff}` are the **pre-application** record, kept for audit and now superseded.
2. ⭐ **The generator exists and meets the acceptance criterion.** `python -m taniteval.leaderboard build`
   (run from `<repo>/taniteval`) writes the marker-delimited section and `Benchmarks & Eval/leaderboard.html`.
   **Delete the section → build → byte-identical: PASS in both deletion modes** (content only; markers-and-all),
   and a rebuild in place reports `md_changed: False`. **42/42** tests green (`taniteval/tests/test_leaderboard.py`).
3. ⭐ **The page gains what it never had:** a *5 October position* block, a **four-families block per arm**
   (ADE alone is not a result), **§0E External field** — one table per protocol, 67 external rows each re-read
   from a banked PDF — and **§0F currency audit**. Rows that existed nowhere on the page: **refcv3 final**,
   **refav1 final**, the **2026-09-04 CLOSED-loop floor panel**, and **NAVSIM v2 warmup** (E1 + E2).
4. ⛔ **The honest headline the page now states:** *no TanitAD arm beats every trivial floor it is read against,
   at any tier, in either loop.* Three arms beat the straight-line floor `ha0`; none beats hold-action or the echo
   control; refav1's planner **is** the `ha0` plan and loses to it; in the only CLOSED-loop panel refcv3 and
   refc-base are indistinguishable from `cl_ha0` while flagship v1 loses to it by **+7.0745 m**; on NAVSIM warmup
   **doing nothing (STOP 30.09) beats constant velocity (CV 18.54)** on the official two-stage EPDMS.
5. ⚠️ **A sign bug I introduced and caught before it landed:** a substring rule (`"_acc"`) matched
   `LON_accel_mae_mps2`, printing a **separated LOSS as a win** on refcv3 and refav1. Fixed with a token-exact
   rule and pinned by a literal test (`test_accel_mae_is_read_as_lower_is_better`). This is the page's own
   "a check that shares the defect it checks for" family — it was caught by reading the rendered table against
   the registry's verdicts, not by the code.

---

## 0b. Second application round (2026-09-20) — W2, W6, W3 landed, and one defect the round exposed

The page now carries **113 external rows** (67 → +39 nuScenes from W6 → +7 NAVSIM v1 from W3), and its live
blob is rebuilt from disk. **All 113 re-verified by my own PDF verifier: 112 verifiable, 0 true-arm failures,
sha256 clean on 19/19 banked PDFs**, mutation arms M1 108/112 RED and M2 93/112 RED. **2,202 → 2,284 lines.**

**W2 — four items, each applied and pinned by a test.**
1. The stale wrong Drive-JEPA modality sentence in `raw/external_field_sources.json` is marked **RESOLVED in
   registry 2.10.1**; the old text is kept only as the record of what was corrected.
2. ⛔ **The perception-free ladder is not camera-only, and the page no longer says it is.** I re-read
   **Table 2, PDF page 8** of the banked primary `2601.22032` myself (sha256 re-checked against `library.json`):
   the Inputs column prints **`C & L` for LAW (83.8) and World4Drive (85.1)**, `Camera` for **Epona (86.2)** and
   Drive-JEPA (89.0). Epona is recorded **86.2** (Tab. 2 p.8) with **86.1** (Tab. 1 p.4) as a recorded
   cross-check — ⭐ *an internal disagreement inside one paper*, and the page prints it in the row rather than
   silently choosing one.
3. The **route-level oracle** caveat rides on the protocol header of all four NavSim tables **and** on every
   command-conditioned row (`⚠️ route oracle`), with a control row that must NOT carry it.
4. The **split nesting** is stated where the tables meet: warmup ⊂ navhard (16/16) and 367/450 navhard stage-1
   tokens are navtest tokens ⇒ the columns are **not independent samples**.

**W6 — 39 nuScenes rows merged.** 0 id collisions; the three `implementation` strings the page already used are
reproduced byte-for-byte, so no sub-table split (asserted: each appears exactly once). Folded into the page text:
the **harness-version** finding (no devkit exists; the reduction is pinned to a commit, and the UniAD/VAD ranking
flips between the two legacy harnesses) and the **GT-human control** (the recorded human "collides" 0.36 % under
UniAD's methodology, 0.96 % under VAD's, **0.00 %** with an oriented box on a 0.1 m grid ⇒ no collision number
within ~2× of its protocol's floor is readable). The three `comparison_admissible: false` rows stay **visible**
with their reasons. A row whose value keys are outside the eight standard metrics prints `—` in the standard
columns and its **own keys** in the note — never relabelled into a standard column.
⚠️ **One inherited claim I could not reproduce, and it matters for how it is quoted:** *"15 of W6's rows share a
(paper, table) with rows you extracted, and the page agrees 15/15."* MEASURED: **15 share a PAPER**, only **5**
share an exact `(library_key, table)` string, and **0** share a `(paper, page, system)` cell — W6 deliberately
added no row the page already had. So there is nothing to agree or disagree on at row level; it is a
**consistency** check, not an independent re-read. ⭐ What IS an independent re-read is my verifier: all 39 rows'
tokens were re-found on their cited pages with **PyMuPDF**, where W6 used **pypdf** — a different extractor.

**W3 — 7 NAVSIM v1 rows added, 1 turned into a cross-check.** CV 20.6, Ego-MLP 65.6, LTF 83.8, TransFuser 84.0,
UniAD 83.4, PARA-Drive 84.0 (Tab. 1 p.7) and Hydra-MDP 91.3 (Tab. 3 p.9), all from the banked `2406.15349`.
W3's **Human 94.8** is the same value the page already carried from a *different* paper, so it is recorded as a
**second primary on the existing row** (`= 94.8`) rather than a duplicate row — the strongest cheap check there
is, and now visible in the source cell. The **STOP-floor rule** is stated **once per protocol** on both NavSim
generations, with W3's mechanism (navtest keeps only scenes the human passes ⇒ a stopped ego holds
NC = DAC = TTC = 1 = **41.7 PDMS points free**), and ⛔ **the 20-token smoke's numbers are deliberately not
printed** — they are not a navtest result.

**W1's real runs — and the defect the instruction would have caused.** Three real warmup runs now exist and I
read them. ⛔ **I did NOT delete the `legacy_bench` entry as instructed, because that would have deleted the
result.** MEASURED: the W1 runs carry **only the two devkit floors (STOP, CV)** — no refcv4b arm at all — while
the banked E2 run carries those floors **plus refcv4b's eight seam arms**, whose Δ columns are paired *within*
it. My own loader had the same bug in miniature: superseding **arm-by-arm** stripped E2's STOP and CV rows and
left eight arms quoting `Δ vs STOP` against a floor the table no longer showed. The supersede is now
**run-scoped** — a W1 run supersedes a legacy run only when it reports a value for **every** arm of it — and the
overlap is printed as a **cross-rig reproduction**: the W1 bench CLI reads **30.090231 / 18.535627** where the
banked run reads **30.090231 / 18.535627**, |Δ| **0.00e+00**, two producers and one number.
⭐ **W-25 applied as W1 counts it:** the devkit's printed combined row survives the refusal as
`statistics.official_combined_row_HYBRID` (shown, labelled, never used as a score), and the stand-in count is
re-derived **from the seam artifact's own `source` column** — 16 of 220 rows — with the producer's reported
`n_cv_standin_rows` printed beside it *because* it agrees. A disagreement would say so and the column would win.

⛔ **And the round exposed a real defect in my own acceptance: the W1 results tree MOVED UNDER THE BUILD.**
MEASURED 2026-09-20: the warmup run directories went **4 → 2** in ~20 minutes (one run the page cited was
**deleted**, another appeared), so the acceptance round-trip read DIFFERS for a generator that was working
perfectly, and a page written in that window would cite a run that no longer exists. `build()` now fingerprints
the run tree **before and after** rendering and **refuses to write** if it moved (`SourceError`, page untouched).
⚠️ This is the programme's *"a success criterion disconnected from the thing being checked"* family seen from the
other side: the criterion was fine and the **input** was moving.

**W5 — three items, and one of its two claims did not reproduce.**
* ⛔ **`build.py` never imported `taniteval.report`.** Positive check on the live file: the string does not occur
  in **any** of my six modules, and the chart import resolves to
  `taniteval/taniteval/benchreport/leaderboard_charts.py` with `render_leaderboard_charts` callable. W5's item (1)
  is against a stale read. ⭐ What WAS real and is now fixed: `render_html` still carried
  `taniteval.benchreport.html` as a **fallback** to the module W5 renamed away — keeping it would quietly
  resurrect the very stdlib-shadowing hazard the rename removed, and a silent fallback cannot tell you the module
  is gone. One module name only, pinned by a test.
* ✅ **Stdlib-shadow check, with a control:** 0 top-level modules in `taniteval` share a stdlib name, 0 in
  `taniteval/leaderboard`, and `import html` still resolves to `C:\Python314\Lib\html\__init__.py`. Pinned.
* ⚠️ **Parameter counts: 4 of 113 rows, and that is the honest n.** Only where the banked primary PRINTS the
  number: `2601.22032` **p.4** carries an `Encoder size` row — LAW **21M**, World4Drive **21M**, Epona **1.1B**,
  Drive-JEPA **307M** — beside the PDMS row that also prints Epona's 86.1. Those four rows now carry numeric
  `params_encoder` + `params_source` + `params_tokens`, and the PDF verifier **re-reads them on their own cited
  page** (a mutated token goes RED: `308M` → `true-arm failures 1 · VERDICT FAIL`). ⛔ `params_total` is absent
  everywhere: no primary states one. Six other papers do print a parameter column and are named in
  `provenance.params` as the next cheap lever — with the scope warnings that stopped me using them blind
  (`2405.19620`'s Params are **that paper's own reimplementations**; `2411.15139`'s are an **ablation ladder**;
  `2512.07745`'s are a **backbone**, not a model).
* ⭐ **The independent read-back W5 asked for exists:** `python -m taniteval.leaderboard readback`. It re-parses
  the rendered markdown with its own parser, imports **nothing** from `render.py`, and traces every printed
  external value back to `published_results.json` directly: **109 cells checked, 0 errors**, 8 unmatched rows —
  all of them our own `TanitAD — any arm` NOT-MEASURED rows and the navhard `N1` attempt, which have no published
  artifact by construction. `--mutate` swaps two values before rendering and the read-back goes **RED**.
  ⭐ It earned its keep immediately: its first run flagged 5 rows, and all 5 were **display rounding** (the page
  prints `0.38`, the artifact holds `0.3844`) — so the tolerance is now half a printed unit at the row's own
  decimals, which is the only comparison that is about the page rather than about float formatting.

**E1 — our FIRST numbers on the official navhard column, and the finding the page now leads with.**
The suite run `20260920T082848Z-navsim_v2-none-06e257` is read straight from
`taniteval/results/bench/navsim_v2/navhard_two_stage/`; nothing is typed into the renderer.
* ⛔ **STOP 29.85 [27.37, 32.53] beats CV 11.48 [8.25, 14.50] — a factor of 2.60×**, paired Δ **+18.3716** ×100,
  log-cluster bootstrap over **76** `log_name` clusters. The page states this **in the lede of the navhard block,
  before the table**, and the sentence is computed from the run's own sub-metrics, so it cannot drift from the
  numbers underneath it: a stopped ego holds **DAC 0.9311 · DDC 1.0000 · NC 0.9967 · TLC 0.9978 · TTC 0.9978** at
  stage 1 while only **EP 0.3409** (against CV's 0.7753) punishes it — EPDMS *multiplies* its compliance terms.
  ⇒ **"beats CV" is not evidence of driving here; the bar for any TanitAD arm is STOP.**
* The position block's *"the official column is NOT MEASURED"* was true until this run landed and is now **replaced
  by the floors plus an explicit "no TanitAD ARM is on this column yet"** — pinned by a test that fails if the stale
  sentence returns.
* ⭐ **The harness reproduces the HF navhard leaderboard EXACTLY: Δ 0.0000** on CV (11.4816; S1 28.9588, S2 34.2464).
  Against **[N2] 2506.04218v3 Table 2** the pre-registered test (`|100·EPDMS − 11.4| ≤ 0.05`) **reads FALSE**
  (Δ 0.0816) and **is reported as written** — and the *reason* is the printing convention, not the harness:
  **19/19 published terms reproduce exactly under TRUNCATION, only 9/19 under rounding.** The page therefore requires
  any *"matches at printed precision"* claim to name its convention.
* ⚖️ **Reference arms, with their scope on the row:** navhard human **93.4796** (n 450) and warmup human **95.1255**
  (filter ON) vs **87.2014** (filter OFF, the deliberate-regression arm) — all **STAGE 1 ONLY**. ⛔ The **two-stage
  human is UNDEFINED** (every synthetic scene has `num_future_frames = 0`; the official runner exits 1 with no CSV)
  and is printed as undefined, **never imputed**. The warmup pair is read from the **devkit's own
  `average_all_frames` CSV row**, not from E1's RESULT.md table — *a summary is not a path*; that needed a small
  `devkit_csv` adapter which **refuses** a CSV that does not carry exactly one such row.
* The first navhard attempt (`ABORTED_RAM_GUARD`, 4,477 s, no CSV) **stays on the board**, now marked SUPERSEDED —
  the cost of a result includes the runs that did not produce one.

**W1's checkpoint identity — read, never re-derived; and two defects the change surfaced.**
* ⛔ Every run line now prints **the artifact's own `provenance.ckpt.registry_key_display`**. My renderer's
  own wording for the same fact (*"devkit floors only (no checkpoint)"*) was **deleted** — it was exactly the
  second formatter W1 warns about. The three states the page shows today: `refcv4b-b1-v72-40k` (key),
  `sha256:54320ec2d72a` (identified, not cross-referenced), `no checkpoint (devkit floors only)`.
* ⭐ **W1's corollary is honoured and pinned:** a sha256-identified row is **not** treated as unusable — the
  `internal_t1` arms render with their numbers, and a test fails if any of them is refused.
* ⛔ **The contract refused my converted E2 summary, and it was right to.** `provenance.ckpt` had to become the
  real triple, and W1 requires a **sha256** for a run with model arms; the banked E2 record carries only an
  **md5**. ⭐ Rather than relabel the md5 or drop the block, I **re-computed the md5 over the checkpoint on disk**
  (`D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt`, 1,285,301,425 B) — it matches
  `99b573e8…d4b751` **exactly**, which is what licenses attaching the sha256 I measured,
  `090ee8d2584acc81e3738ab28f8f5bf2ed6bfb0d0cc6fba887de2a0cbddcbaa4`. The derivation chain is written **into the
  artifact** (`sha256_provenance`), and the md5 is kept under its own name — ⛔ an md5 is never relabelled a sha256.
* ⛔ **A UNIT BUG OF MY OWN, caught by the new block and fixed:** `off = h.get("x100", h["value"] * 100)` assumed
  every headline is a 0–1 score. The `internal_t1` run's headline is **ADE in metres**, so the model arm rendered
  as **"290.98"** for **2.9098 m**. `x100` is now used **only when the artifact provides it**; any other headline
  prints natively **with its column named** (`2.9098 *(ade_dense_m)*`). Pinned, including a source scan that
  ignores comments — the first version of that test failed against the comment documenting the removed line.
* ⭐ **A run whose protocol has no table is now LISTED, not dropped** (§0E.x). `TanitAD_T1_refc_physicalai` was
  being read, validated and rendered nowhere — the "built, tested and unreachable from its caller" class in
  leaderboard costume, and it is what exposed the unit bug above.
* ⚠️ **And the same over-composing mistake in the devkit cell:** formatting `repo@sha + N patches` over a producer's
  verbatim line turned *"0a380a9 (post-#151) + 3 local patches + E1 Windows loader patch"* into *"+ 1 patches"*.
  The producer's own wording now wins there too, by the same rule as the checkpoint.
* ⭐ **W1's generalisation, borrowed:** *when a lookup table has entries for some keys and not others, the
  missing-entry path is the one that needs the test.* Applied to my `_stage()` spelling table (`stage_one` vs
  `stage1`), which returns `{}` for an unknown name rather than guessing, and to the display-field tests, which
  now cover **absent** and **present-but-null** separately — the second is the shape that bypasses a `.get`
  default, and it is only reachable on the schema-only validation path, so the test pins it *there* with the
  contract's refusal as the control.

**§0D — the internal standard now has its own table, and it does not flatter the programme.**
`TanitAD_T1_refc_physicalai` renders as **its own section, never merged with an EPDMS or PDMS column** — tier and
loop per arm, the model-free floors `ha` / `ha0` / `ha0_ext` as Δ columns beside every arm, the estimator and
interval per row, and the four families per arm with each family's own reason and n.
* ⛔ **The honest headline, derived from the run's own paired verdicts rather than written by hand:** *`os`
  separates from NO model-free floor on this run — and is separated WORSE than `ha0`* (+2.3682 m [+1.8778,
  +2.7605]). Hold-action `ha` (0.1946 m) and the echo floor `ha0_ext` (0.1884 m) both sit far below the model
  arm's 2.9098 m.
* ⛔ **And the table says what grid it is:** the artifact's own `REAL_CHECKPOINT_ON_SMOKE_CORPUS` verdict —
  **2 episodes / 9 windows at step 2000** — is printed above the rows, and every interval carries the arm's own
  refusal (*"2 episodes < the RG-14 floor of 8"*). ⇒ a smoke reading, labelled as one, not a result.
* ⚠️ **A floor-vs-floor pair no longer says "model WINS".** The verdict names the winning KEY — *"⬅ `ha` better"* —
  because three of the four arms on this surface are floors and "model WINS" on a floor-vs-floor row is false.
* ⛔ **W2's LATERAL qualifier now travels with EVERY cross-track table on the PAGE — hand-written and generated.**
  The generator appends it to any block printing a `cross_mae_m` / `cross_bias_m` (2 blocks); the **two
  hand-written sites** (§1b's `LAT cross_mae_m` row and §1d.1's `LATERAL cross_mae_m` row) were annotated
  **in place, on the coordinator's instruction, APPEND-ONLY**: a new line after each table, marked
  *"Annotation appended 2026-09-20 by W4; finding and wording from W2 2026-09-20, `four_families.py:187` /
  `:776`"* so a later reader can tell it from the original text. ⛔ **No number, verdict or line of prose was
  touched** — 2,358 → 2,362 lines, 4 added (two annotations + their blank lines), and the superset check reports
  **0 hand-written lines lost** against both the previous staged page and the first staged page of the session.
  Coverage is pinned by a test that scans the **whole page** (not just the section this stream owns) and fails
  if any `##`/`###` block containing a cross-track row lacks the qualifier: **0 blocks missing**, 4 occurrences.

⭐ **THE RULE, STATED ONCE, WHERE THE NEXT PERSON WILL READ IT — the producer's own wording wins; a renderer that
rebuilds a fact is a SECOND PRODUCER, and two producers of one fact are two things that can disagree.** It is now
the first thing inside the BENCH markers on the page, and the opening paragraph of `render.py`'s docstring, with
the three costumes it wore in one day: a re-derived checkpoint key, an assumed `×100` that printed **2.9098 m as
"290.98"**, and a composed `+ 1 patches` that replaced a verbatim line naming three. ⇒ *if you are formatting a
fact the producer already formatted, read its field instead.*
⭐ **And the provenance-upgrade pattern the same round produced, for the record:** do not relabel an md5 as a
sha256 — re-compute the **md5 over the file on disk**, require an exact match with the banked one, and only then
attach a sha256 **you measured yourself**, with the chain recorded in the artifact (`sha256_provenance`).

**W3's full-split navtest landed — and the page now carries our own numbers on both NavSim generations.**
Read from `…/2026-09-19-navsim-v1-navtest/raw/analysis_navtest.json` (12,146 tokens per arm, 136 log clusters):
* ⛔ **The head finding, derived from the artifact exactly as the navhard one is:** *on navtest a stopped car scores
  **61.82** — **3.0× CV's 20.65**, and within **3.8 points** of the published blind ego-status-MLP baseline (65.6)*.
  Paired on identical tokens **+41.17 [+39.50, +42.84]**, W/T/L **8,607/727/2,812**, leading in **every** t0 speed
  band. Mechanism MEASURED: NC 97.40 · DAC 96.53 · TTC 96.40 make **41.67 points free** (5/12 of PDMS), the braking
  coast earns a median **5.41 m**, and on **1,006/12,146 tokens (8.28 %)** the best compliant progress is ≤ 5 m so
  **EP ≡ 1 by rule**. ⇒ **CV is not a floor on navtest either** — the STOP-floor rule now carries a measured pair on
  BOTH generations (navhard 29.85 vs 11.48; navtest 61.82 vs 20.65).
* ⚠️ **W3's pre-registered range is printed as a FAILED PREDICTION, in its own words** — SPEC §6 committed
  STOP ∈ [38, 60] point ≈ 45, measured **61.82 is above it** — with both committed inequalities still holding. ⛔ Not
  re-fitted, and the SPEC is cited so a reader can check the commitment rather than the retelling.
* ⛔ **HUMAN is CLOSE, not reproduced, and the page names the term that fails:** 94.5514 vs the paper's 94.8, with
  NC / DAC / TTC / C all exact and **the ENTIRE gap in EP (86.9629 vs 87.5 — NOT REPRODUCED)**.
* ⭐ **HUMAN is DEFINED here and UNDEFINED on v2** — navtest is single-stage, so the ceiling is a number; on the
  two-stage splits the suite refuses that headline because every synthetic scene has `num_future_frames = 0`. The
  page says so on both sides, so the v2 refusal reads as an identity of the protocol rather than a gap in our work.

⛔ **STATE THE ARTIFACT BEFORE THE CONVENTION — W1's self-retraction, applied to every external NavSim number.**
The navhard reference is now a two-row table naming the artifact, what it prints, its dp, its rule and our reading:
**[N2] Tab. 2 prints 11.4 at 1 dp and TRUNCATES** (our 11.4816 → 11.4 ✅) · **the HF leaderboard prints 11.4816 at
4 dp and ROUNDS** (Δ **0.0000** ✅). They never disagreed. ⭐ **Quoting one artifact's convention against the other's
printed row manufactures a FALSE AGREEMENT — worse than a mismatch, because nothing looks wrong**, and the page says
that in place. Our navtest rows carry the same pairing per row (`paper 20.6 → REPRODUCED_UNDER_TRUNCATION ·
leaderboard 20.6517 → REPRODUCED_UNDER_ROUNDING`).
⚠️ **And one inherited framing I narrowed — W3 then ADJUDICATED and confirmed the narrowing.** The message that
reached me said the convention question is *"SETTLED by a primary measurement — the PAPER TRUNCATES"*. Checking that
phrasing against the artifact showed the opposite about the paper, so I did not publish it. W3 corrected its own
RESULT.md rather than defending it and added a `post_measurement_2026_09_20` block **beside** the standing verdict;
the page now prints the file as it is, **both scopes in one table**:
* **this CV cell — SETTLED:** one underlying **20.651651538606658** explains the paper's **20.6** (a truncation;
  rounding would have printed 20.7) and the leaderboard's **20.6517** (a rounding of the same number);
* **the PAPER — UNSETTLED, unchanged and verbatim:** *"the 4 informative leaderboard cells split 3 ROUNDING /
  1 TRUNCATION … ⛔ No convention may be assumed"*. TransFuser 83.9 vs 83.8822, Ego-MLP 66.4 vs 66.3989 and
  LTF ±0.6 vs 0.552 still read as rounding, and the paper's own seed/std test implies rounding — a paper that
  truncated everywhere would have printed 83.8, 66.3, 0.5.
⭐ **The honest scope, carried onto the page:** what changed is the cell's **EVIDENCE CLASS** — from *print vs an
INHERITED leaderboard value* to *print vs a value WE measured* — and even that rests on a **stated assumption**, that
the paper's CV row is the same underlying quantity as ours (well supported: five sub-scores match Table 1, the PDMS
matches the leaderboard at 4 dp, and the paper says Table 3 IS that leaderboard — but not itself measured). ⛔ The
counterfactual bounds it: **had the paper's CV been 20.6499, both conventions print 20.6 and the cell would say
nothing.** AMENDMENT A1 stays in force for every other cell: print both readings, never assume.
⭐ **Root-cause class, logged by W3: *true but wrong for the reader* — a claim true of ONE CELL, phrased as a claim
about the PAPER.** It never reached the record because the relayed phrasing was checked against the artifact instead
of accepted. ⛔ A test now fails if the over-broad phrase appears on the page as an assertion; it is admissible only
inside the artifact's own quoted account of the relay.

**Deliberate-regression arms (a guard that has never failed is not a guard).** Each defect was reintroduced and
the guard required to go RED: arm-scoped superseding → **RED** (floors gone, eight arms orphaned); route-oracle
marker removed → **RED** (0/3 command rows marked); HYBRID printed row discarded → **RED**; a reported stand-in
total that contradicts its own rows → **RED** (`REPORTED 999 … the column wins`); a moving results tree → **RED**.
Controls read OK in every case.

---

## 1. Currency audit — what was stale, contradicted, or missing

Full table: `raw/currency_audit.json` (rendered into §0F of the proposed page). Scope: every registry section
carrying an eval result, all 138 entries of `taniteval/results/` (131 paths added since 2026-09-03), and the
284 dated package directories located by `git log --since=2026-09-03 --name-only`. The 2026-09-07…19
Architecture & Inference packages were classified by a read-only sub-agent (report banked verbatim at
`raw/audit_eval_packages_since_0903.md`, **INHERITED**); every number that reaches a row was re-read by me.

| finding | status |
|---|---|
| **refcv3 final (step 40,284)** — four families, T1, registry §4.5, raw `taniteval/results/refcv3-40284-openloop.ARM.json` | page said **ALL PENDING** since 2026-09-03 → **STALE**; now a generated row + family table |
| **refav1 final (step 21,109)** — registry §2.4, raw `taniteval/results/refav1-21109-openloop.json` | page said *final read PENDING*, *registry row ABSENT*, W-20 open → **STALE ×3**; W-20 closed |
| **CLOSED-loop floor panel (2026-09-04)** — Thor NuRec-gsplat, 1 empty scene, 9 rollout starts, registry §1.12 banner, raw `taniteval/results/2026-09-04-closedloop-floor/FLOOR_SUMMARY.json` | **MISSING from the page entirely** — the page believed its only CLOSED rows were from 2026-07-22/23; now three T2 · CLOSED rows with both floors and all four families |
| **NAVSIM v2 warmup (E1 + E2, 2026-09-19)** | **MISSING** (§9.1 said *NOT COMPUTABLE TODAY*) → now §0E.2 with STOP and CV on every row and no interval |
| **v7-tiny** — registry §13 exists (incl. the T1 read §13.10 of 2026-08-30) | page's §11-D / §12.2 / W-12 say *no registry row at all* → **STALE**; still no row here because the raws are Thor-only → **W-24** |
| **§9.1 external numbers** | **three wrong**: PDM-Closed 51.3 is **pre-#151** (post-fix 56.6); DrivoR 56.3 is +134k-synthetic **+ TOAD**; Drive-JEPA 93.3 is the paper's own checklist misquote of 93.7 → corrected in §0E |
| **§0.8 / §0.1 loop vocabulary** | the doctrine source was relabelled at `7ad8522` (2026-09-03) and that sweep stopped there; **no commit since has touched this page's 13 sites** ⇒ UNOWNED → **W-23** (listed, not silently edited, per the brief) |
| **refcv4b `os` rig** | page quotes 0.2965 (dev-box/Thor) while registry §4.6 quotes 0.2975 (A40) — both right, neither named its rig (registry `D-OS-ADE-CROSSRIG-RESOLVED`); the generated row names the rig and leads with rig-free paired margins |
| **§1d.6 vs §1e** | two blocks for the same two arms written 55 min apart by two streams; numbers agree → cross-referenced, not merged |
| **DD-v2 RL arms (09-15 release recipe, 09-17 L1 Stage A)** | **BURNED** — trained on 100 of the 141 eval clips (INHERITED: sub-agent report + memory note) ⇒ never a leaderboard row |
| **WP-D BEV-aux, P1 agent gate, P-RC21 GT-bar pilots, refcv6 W0–W3 + A3** | no admissible driving row: 4k-step test model / trained on eval clips / T0 training-side / instrument work (A3 is a perception metric trained on half the eval set) |

---

## 2. External field — what re-reading the primaries changed

`products/P7-TanitEval/benchmarks/published_results.json` (new, **67 rows**) + per-row reasoning in
`raw/external_field_sources.json`. Verifier: `code/verify_published_against_pdfs.py` → `raw/verify_published_against_pdfs.json`.

* **Verification result:** 66 verifiable rows, **0 true-arm failures**, sha256 OK on 13/13 PDFs;
  deliberate-regression arms **M1 (every value perturbed by one unit in its last decimal) RED 66/66** and
  **M2 (wrong page) RED 56/66**. ⚠️ **M1 first read 59/66 — below the 90 % power bar I had pre-stated.** I did
  **not** move the bar: the 7 blind rows sat in dense tables where the ±1 neighbour is also on the page
  (e.g. DrivoR 94.6 / +TOAD 94.7 / human 94.8), so I made the instrument specific — those rows now carry a
  `row_window` (anchor → next-row anchor) and the mutation goes red everywhere.
* **Findings that change how the field must be quoted:**
  1. **Two EPDMS harness versions.** The #151 human-filter fix splits every navhard and navtest number.
     `2601.05083` Tab. 14 prints a pre- and post-fix block; `2605.09701` Tab. 2 prints `EPDMS*` beside `EPDMS`.
     ⛔ TOAD Tab. 6 and DriveFuture Tab. 1 **mix versions** (their "TransFuser 23.1" is the pre-fix LTF), so
     single-source rows from those tables are `fix151: unverified` and stay out of comparison cells.
     HAD's navhard table (`2604.03581` Tab. 11) is pre-fix throughout (its PDM-Closed 51.3 / LTF 23.1 / GTRS-Dense 41.7 match the pre-fix block).
  2. **"DrivoR 56.3" is three things at once**: DrivoR + **134k SimScale synthetic training samples** + **TOAD
     test-time search**. Established by matching the 18 sub-metrics of TOAD's DrivoR row to DrivoR Tab. 3's
     "+134k" row. Navtrain-only DrivoR is **48.3**.
  3. **The brief's navtest ladder mixes backbones**: Transfuser 76.7 and Hydra-MDP++ 81.4 are ResNet34 rows;
     DriveSuprim 87.1 and Drive-JEPA 87.8 are ViT/L rows — DriveSuprim at ResNet34 reads **83.1**.
  4. **The perception-free ladder is not "front-camera-only"**: Drive-JEPA Tab. 2's Inputs column prints
     **"C & L"** for LAW and World4Drive. Epona reads **86.1** in Tab. 1 and **86.2** in Tab. 2 — same paper.
  5. **GuideFlow is an unresolved 24-point conflict** (51.5 benchmark snapshot vs 27.1 in two method papers) → out of comparison.
  6. **The benchmark authors disown navtest-EPDMS** (`2506.04218` p.7) — kept as orientation, never a target.
  7. **Banked on the way:** `2603.25946` (TF++ w/ VLAAD-MIL 86.97 / 71.97, Tab. 5 p.18 — the page had it as
     INHERITED) and `2604.03581` (HAD). `library.json` verified afterwards: my two entries are present and the
     concurrent bankings of other streams were not clobbered.

---

## 3. Proposals for files I did not edit (as required)

**`Project Steering/MODEL_REGISTRY.md`**
1. §4.8.4's *"there is no B1-scoped `obstacle.offline` join"* is contradicted by
   `…/2026-09-07-join-repro/raw/repro_result_20260907.json` (the B1 TRAIN agent join, 317 MB, rebuilt
   **byte-identical**). Restate the WP-6 blocker precisely.
2. §4.7 / §4.8 both carry refcv5-v2's result under two "4.7" headers — make §4.7 the stopped run only.
3. §4.8.5 says the artifacts are single-disk; `panel.log`, `panel_seed1.log` and both panel JSONs are banked
   in-repo (`…/2026-09-07-refcv5-v2-comparison/raw/`). Only the checkpoint and videos remain single-disk.
4. §4.6: quote `os` **with its rig** in the row headline (A40 0.2975 vs dev-box/Thor 0.2965), not only in the
   2026-09-07 cross-rig note.
5. No registry row exists for the NAVSIM warmup bridge; none is needed while it is not claim-bearing — but if
   a navhard number lands, it needs one.

**`products/P7-TanitEval/CRITERIA_REGISTRY.json`** (owned by W2/E3 this session)
6. `navsim.GATE_modality_label.why` states Drive-JEPA's 93.7 headline stacks FRONT+LEFT+RIGHT at 1024×256.
   The primary says the opposite (Tab. 8 lists "Ours" at 2×512×256; §4.2 and App. B: front camera only, no
   LiDAR); the 1024×256 stacking belongs to Transfuser / HydraMDP++ / DriveSuprim / GoalFlow.
7. Rename `comparable_ladder_perception_free_front_only` → `…_perception_free` and record each row's printed
   inputs (LAW and World4Drive are "C & L") plus the Epona 86.1/86.2 inconsistency.
8. `published_reference_numbers._harness_pin` can now record the #151 side per number (finding 2.1).
9. Consider adopting the protocol tags of `published_results.json` (they already match W1's closed set;
   `Bench2Drive_closed_loop` is marked EXTERNAL-ONLY).

**`Project Steering/GOALS_AND_CLAIMS.md`** — G3 ("beat published SOTA on community benchmarks") now has a
measured starting point: our harness reproduces the HF warmup leaderboard's CV row to Δ 2.7e-05, and the
navhard bar is DriveFuture 55.5 / DrivoR+TOAD 56.3 / PDM-Closed 56.6 against a CV floor of 11.4.

---

## 4. Integration asks (orchestrator)

1. **Land the page**: confirm the sibling's append is final, then apply `raw/LEADERBOARD.proposed.diff`
   (or tell me to re-run `python -m taniteval.leaderboard build` on the live page — it is idempotent and
   touches nothing outside the markers).
2. **W1**: my reader validates every summary with `taniteval.bench.contract.validate_summary`. When the first
   real `navsim_v2` run lands, its arms supersede my converted E2 rows automatically (dedupe is per
   `(protocol, arm)`); the legacy entry in `leaderboard_sources.json` can then be deleted.
3. **W5**: `render_leaderboard_charts(model)` is called with W4's canonical model and its output is embedded
   verbatim (5 figures in `leaderboard.html`). W5 reads `official_x100` / `hybrid` / `kind` / `key` /
   `declared` / `tier_loop` from our rows — if that contract changes, tell me.
4. **W6 (nuScenes)**: send the published rows as a patch against `published_results.json`; the tags
   `nuScenes_OL_L2_stp3` / `nuScenes_OL_L2_uniad` are already in place, and BEV-Planner's re-implementation is
   carried as an `implementation` sub-table, not a separate protocol.
5. **W-25**: the warmup model arms have **no** official two-stage EPDMS (E2's stage 1 was a devkit-CV stand-in
   ⇒ HYBRID, refused). One W1 run with a real stage 1 fixes it — GPU-gap work.

---

## 5. Tests

* **`taniteval/tests/test_leaderboard.py` — 25/25 pass.** Acceptance (both deletion modes), idempotency,
  CRLF preservation, hand-written bytes untouched, four mutation guards (protocol mixing, warmup interval,
  missing floor, cached-value), literal margin expectations, the W1-contract fixture, and the W5 import.
* **Full suite `python -m pytest -q` in `taniteval/`: 1593 passed, 11 failed, 24 skipped (391 s).**
  ⛔ **No failure is in this stream's scope**, and I checked each rather than asserting it:
  * 2 × `test_benchreport_golden.py` — **pass when re-run alone** (10/10). They failed inside the full run
    because W5 was rewriting `taniteval/taniteval/benchreport/` at that moment (it renamed `html.py` → `page.py`
    mid-session). A race with a live sibling, not a defect.
  * 9 × `test_render_openloop_video.py` — **pre-existing and environmental**:
    `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f` (the cp1252 dev-box trap; the file was last
    touched 2026-08-05 and none of my changes are on its import path). Flagged as a separate task.

## 6. Deliverable manifest

All paths staged (`git add`) and blob-verified — index blob == worktree blob, both 40 chars: **26/26 VERIFIED,
0 MISMATCH, 0 INCONCLUSIVE**. ⛔ **Nothing committed, nothing pushed.**
⚠️ **Staging is not a latch, and the end-of-turn re-check proved it twice:** a sibling held `.git/index.lock`
(waited it out in a retry loop — ⛔ never unlink a lock while git may be alive), and `library.json` had moved
under me between staging and verification (another stream banked a paper). Re-added and re-verified; my two
entries are present and sha-clean at 529 entries.

| artifact | where | note |
|---|---|---|
| the generator | `taniteval/taniteval/leaderboard/{__init__,__main__,build,sources,render,legacy,readback}.py` | `python -m taniteval.leaderboard build` (from `<repo>/taniteval`) |
| its source map (pointers only) | `taniteval/taniteval/leaderboard/leaderboard_sources.json` | no measured value is typed into it |
| external numbers | `products/P7-TanitEval/benchmarks/published_results.json` | **113 rows** (67 + W6's 39 nuScenes + W3's 7 NAVSIM v1), protocol tags joined to W1's closed set |
| tests | `taniteval/tests/test_leaderboard.py` | **76** tests incl. the acceptance round-trip and 6 deliberate-regression arms |
| W1-schema fixture | `taniteval/tests/fixtures/leaderboard/bench/navsim_v2/warmup_two_stage/e2_refcv4b_warmup/summary.json` | converted from E2's banked output, validated by `taniteval.bench.contract` |
| HTML leaderboard | `Benchmarks & Eval/leaderboard.html` | 5 figures from W5's `render_leaderboard_charts`, inside W5's page shell |
| **proposed page** | `…/2026-09-19-leaderboard-currency/raw/LEADERBOARD.proposed.md` | byte-exact copy of the sibling's staged page + my edits + the generated section |
| **the page itself** | `Benchmarks & Eval/LEADERBOARD.md` | **2,397 lines**, generated section rebuilt from live data; hand-written half byte-identical to the previous staged version |
| applied diff (record) | `…/raw/LEADERBOARD.applied.diff` | what actually landed in the first application round |
| **superseded proposal** | `…/raw/LEADERBOARD.proposed.diff` | pre-application record only — ⛔ do not apply |
| currency audit | `…/raw/currency_audit.json` | rendered as §0F of the proposed page |
| external provenance | `…/raw/external_field_sources.json` | per-row reasoning, cross-checks, conflicts |
| PDF verification | `…/code/verify_published_against_pdfs.py` + `…/raw/verify_published_against_pdfs.json` | **112/112 verifiable rows pass · 0 failures · sha256 19/19** · M1 108/112 RED · M2 93/112 RED · params tokens re-read on their own page (mutation RED) |
| inherited sub-agent report | `…/raw/audit_eval_packages_since_0903.md` | classification of the A&I packages 09-07…19 |
| package docs | `…/SPEC.md`, `…/PLAN.md`, `…/RESULT.md`, `…/COMMS.md` | |
| banked primaries | `TanitAD Research Lab/Library/papers/2603.25946_*.pdf`, `…/2604.03581_*.pdf`, `library.json`, `LIBRARY.md` | ⚠️ `library.json` / `LIBRARY.md` are shared: staging them necessarily carries three concurrent entries from other streams (verified present, my two entries sha-clean) |

⛔ **Single-location risk:** none introduced. Everything above is in the repo. The artifacts I could NOT put on
the page for that reason are named as work items: v7-tiny's T1 dumps (Thor only, **W-24**) and refcv5-v2's
checkpoint/videos (dev box, registry §4.8.5).

## 7. What I did not do

* No edit to `LEADERBOARD.md` (stop condition), `MODEL_REGISTRY.md`, `GOALS_AND_CLAIMS.md`,
  `CRITERIA_REGISTRY.json`, `taniteval/taniteval/bench/**` or `taniteval/taniteval/benchreport/**`.
* No re-measurement, no GPU, no submission, no download beyond two arXiv PDFs banked with `kb_add.py`.
* The 13 T1 "closed-loop" mislabel sites are **not** rewritten (W-23), per the brief.
* The v7-tiny T1 numbers are **not** on the page: their raws live only on Thor (W-24).
