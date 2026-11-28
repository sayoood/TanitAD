# CONFLICTS found while drafting §7.20

Registry (`MODEL_REGISTRY.md`), register (`GOALS_AND_CLAIMS.md`, "GAC"), paper (`Paper/TANITAD_PAPER.md`) and
raw artifacts, all read at mirror commit `a5a4ee8`. Ordered by what they cost if left. "Section does" says how
`section_7_20.md` already handles it, so the Master Mind knows what is and is not left to decide.

## Needs a decision or an edit before the section lands

**C1. The paper contradicts itself once §7.20 lands (paper vs registry).**
* `Paper/TANITAD_PAPER.md` §12.4, lines 3908–3922, says the programme *"contains zero `import navsim`"* and that
  *"no number we have ever produced is comparable to a published one"*. REFe has scored the full NAVSIM v1.1
  navtest since 2026-10-04 (`REG` §14.1–14.2). Line 3287 (§10 roadmap) still reads *"NAVSIM/Bench2Drive entries
  once closed-loop gates pass"*.
* The class is the one `CLAUDE.md` rule 2 names: a stale absence-claim. §12.4's own reasoning (benchmark → reward
  → RL) is still sound; only its premise changed.
* Section does: states the premise as "§12.4 records … no NAVSIM implementation" and calls §7.20 the benchmark
  step. It does **not** edit §12.4. Recommended: add a dated pointer in §12.4 to §7.20 rather than deleting it.
* Also owed to the paper: the status line and changelog stop at **v1.3 (2026-08-29)**; the last commit touching the
  file is 2026-09-05 (`aad4088`, a path/rename sweep, no new content). The next entry would be **v1.4**.
* Also owed: §5 does not yet state the three-variance rule (*another draw of logs / another training run / another
  inference run*, H-ESTIM-SEED-1, 2026-09-05). §7.20 therefore states it in-line and says §5 owes it.

**C2. The brief calls u400 "deployed"; the registry and register do not.**
* `REG` 5275 calls SFT-4 B u400 "⭐ best REFe". `GAC` 16247 (D-REFE-SFT-4): *"A DEPLOYMENT CANDIDATE … deploying the
  candidate is the PI's call"*. In `RESULT_SFT4.md` and `GAC` 16701 the word "deployed" means `model_final` + route
  fix (84.46 on the full navtest, 85.61 on the 923 tokens).
* Section does: "best REFe checkpoint". If the PI deployed u400 after 2026-10-11, the registry needs a status line.

**C3. The registry lags the register (registry vs register).**
* `MODEL_REGISTRY.md` §14.2 has no P2 (88.82), no P2-s1 (89.15), no P6, and no zero-tail row at `a5a4ee8`
  (`grep -c "88\.82"` = 0; `grep -ci "zero tail"` = 0). The registry is "the ONLY quotable source for model facts".
* Section does: quotes P2 from `GAC` 16702 and the raw `RESULT_P2REP.md` / `P2_MEASURED.json`, which are
  primary measurements, and says so. Recommended: add P2-s0, P2-s1 and the NOT-REPLICATED verdict to §14.2 before
  the paper is read as registry-consistent.

**C4. Two register rows share the id `D-REFE-GAP-1` (GAC lines 16235 and 16250).**
* Row 1 (2026-10-04) starts from `model_final` 83.24: gap "~5.5–6 from zero-score picks + ~5.3 EP", ESTIMATED,
  *"mostly SELECTION"*. Row 2 (2026-10-11) starts from u400 88.25: exact decomposition 4.15 / 1.31 / 0.86 / −0.01.
  Different baselines and methods, not contradictory, but a citation by id alone is ambiguous.
* Section does: cites row 2 only and says "second row of that id". Recommended: rename row 1.

**C5. The R47 correction lives in the register and the log, not in the banked artifact.**
* `…/raw/2026-10-17-zero-tail-mechanism/RESULT.json` `report.lever_coverage_u400.proposal_side` still prints
  "Y 42.0 %, Y-PDMS 1.07" with no gross/net label, and the ZT3 register row's title still bolds *"P6's trunk
  gradient reaches 58 %"*. The correction is in the row's body, `RETRACTION_LOG.md` R47 (line 18242) and
  D-REFE-ZEROTAIL-4.
* Banked artifacts are not rewritten (CLAUDE.md: a banked artifact says what it said). Section does: labels every
  "reach" gross and prints the correction as (e). A one-line note beside `RESULT.json` would help the next reader.

## Wording hazards the section already avoids (flagged so they are not reintroduced)

**C6. "The tracker saves about as many as it breaks" is a point-estimate statement.** Dev N-s0: saved 27, made 20,
difference +0.28 pp [−0.88, +1.24]; navtest saved ESTIMATED 131.8 [43.8, 236.6] vs 144 made
(`RESULT.json` `posthoc_tracker_saved`). Both intervals include "equal" and include zero. The register's *"THE
EXECUTION-SIDE LEVER IS CLOSED"* is therefore stronger than the evidence for every conceivable fix: F_c's upper
bound is +0.04, on a 13-log dev set, against a bar of lower bound > 0 with point ≥ +0.05. Section does: "closes
*those* fixes", and rests the closure on the pre-registered failures, not on the saved-vs-made count.

**C7. Two "seed floors" are in circulation and they are different instruments.**
* `RESULT.json` (ZT3) `next_experiment_ZT4.caveat`: *"seed floor 0.352"*. That is the largest pairwise |ΔPDMS| among
  three N seeds in the S1 experiment (`raw/2026-10-15-s1-read/S1_READ.json` line 1922; T0-like held-out selection on
  teacher-forced banked sets).
* `GAC` 16703 / `RESULT_P2REP.md`: **F = 0.33**, P2-s1 − P2-s0 on the full navtest, T1.
* Section does: quotes only 0.33 (T1), as "one draw of a seed difference, never a variance".

**C8. Two different quantities are both 94.55.** The published *student* row (Table A13, DINOv3 ViT-L, navtrain only,
no SimScale; verified in the banked PDF, sha256 `82a00048…`) and our harness's *human* row. The published human is
94.8 (Tables 1 and 4). `REG` 5282 says the bar "equals our harness's human score", which is true and easy to
misread. Section does: calls it out explicitly in (b).

**C9. A9 reads a map graph at inference while A8 was refused as a map use.** A9's route repair uses NAVSIM's
roadblock-graph topology (`REG` 5246); the PI adopted it on 2026-10-04 as a navigation input, after the 2026-09-28
"no hd map as input" ruling that kept A8's lane-graph fallback OFF (`REG` 5251). Not a record conflict. A reader
will see the tension, so the section discloses it in (a). Please confirm the 2026-10-04 ruling is meant to stand
as written.

## Minor, with no downstream effect on the section

**C10. Label error in a raw output.** `raw/2026-10-11-ep-gap-decomposition/ep_decomp.out` prints u400 *"tokens zeroed
by NC*DAC 4.43 %"*. `ep_decomp.py` lines 24–28 compute `mean(1 − NC×DAC)`, which also counts the half-penalty
(NC = 0.5, 0.21 % of tokens). The zeroed share is 529 / 12,146 = 4.36 % (`ep_lowtail.out`; `GAC` 16250; re-measured
in `gap_decomp_ci.json`). Nothing quotes 4.43 %.

**C11. Registry epoch count.** `REG` 5246 says "10,075 steps, 27 epochs"; `REG` 5260 and the argv say 25 epochs
(25 × ceil(103,039 / 256) = 10,075). Section quotes the 10,075 steps only.

**C12. Two intervals for the same human score.** `GAC` 16235 and `RESULT_P4_TEACHER_NAVTEST.md` col. 1: 94.55
[93.80, 95.24]; the same file's col. 2 and this author's recompute: 94.55 [93.78, 95.23]. Section quotes the
latter. Likewise the route-fix row: banked [83.18, 85.77] vs recompute [83.16, 85.69] (a different resampler);
section quotes the banked one and says so.

**C13. Rounding.** 4.15 + 1.31 + 0.86 − 0.01 = 6.31 against a printed 6.30; unrounded 4.1546 + 1.3053 + 0.8556 −
0.0110 = 6.3045 exactly (`gap_decomp_ci.json`). Not a conflict.

**C14. The recompute inputs.** `navtest_full_tokens.json` on the dev box (sha256 prefix `927bbe38…`) is not the
copy ZT3 hashed (`c6b5cae6…`). Both give 12,146 tokens / 136 logs and the controls reproduce, so the log
assignment is the same; recorded so the difference is not mistaken for drift.
