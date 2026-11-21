# C2 — the ONE navtest look of H-SEARCH-P2-1: **C2_FAIL_LOWER_CI** (2026-10-22 07:53 Berlin)

**Verdict:** the arm FAILED its sealed rule (`AMENDMENT_C2.md` sha256 `a248ec27…`, §11).

**What was tested:**
- Arm `P2HEAD`: the CEM search (C8 config `299c6e63…`) on P2-s0's own, unchanged scorer, with no head substitution.
- Data: all of navtest, 12,146 tokens over 136 logs, in 14 chunks (data sha `b3b609192fcd`).

**Stamps:**
- Tier **T1** = NAVSIM v1.1 PDMS pseudo-simulation of an open-loop plan against logged agents. It is NOT closed loop.
- Interval: the paired log-cluster bootstrap (B = 10,000) answers only "another draw of LOGS".
- Seeds: ONE training seed (`H-ESTIM-SEED-1` open) and 3 search (inference) seeds, whose spread is reported beside the result.

**Evidence class:** MEASURED. The numbers below come from `RESULT_NAVPIPE.json` (sha256 `c04ce9e1…`).

## The decision rule, as sealed

The rule was: lower CI > 0, AND point ≥ +0.10, AND point > the 3-seed range, AND each of dNC/dDAC/dTTC/dC ≥ −0.25.

| term | value | met |
|---|---|---|
| dPDMS (points) | **−0.155 [−0.350, +0.037]** | lower CI > 0: **NO** |
| point ≥ +0.10 | −0.155 | NO |
| point > 3-seed range (0.032) | — | NO |
| per-seed points | −0.172, −0.139, −0.154 (all three negative) | — |
| safety terms (each point ≥ −0.25) | dNC −0.106, dDAC +0.126, dTTC −0.239, dC 0.000 | yes (all four) |
| NEGATIVE flag (upper CI < 0) | upper +0.037 | not flagged |

The search is **not inert**: 19.8 % of outputs moved by more than 0.5 m (per seed 19.6 / 19.6 / 20.1 %).

## Four families (per family, never pooled)

| family | result |
|---|---|
| **LONGITUDINAL** | dEP **−0.223 [−0.427, −0.030]** (separated, worse). dTTC **−0.239 [−0.494, −0.012]** (separated, worse). Search outputs that **slow down** (15.8 %): **−1.18** points on average. Outputs that **speed up** (12.7 %): +0.25 (descriptive). Target-speed and headway are not instrumented in this harness; the PDMS EP/TTC terms stand in. |
| **LATERAL** | dDAC +0.126 [−0.016, +0.282] (not separated). dDDC −0.023 [−0.115, +0.055]. Mean \|offset\| given moved 0.250 m; heading deviation at the end 0.0067 rad; max yaw-rate deviation 0.124 rad/s (mean). |
| **TACTICAL** | Command-class change share 0.0069. This is a proxy and a HYPOTHESIS: the nav_command bearing rule applied to the end point. |
| **STRATEGIC** | **PAUSED** (PI ruling 2026-09-19, PI_DECISION_QUEUE item 25): reported as PAUSED, not as a pass. |

**Latency line:** **UNMEASURED** against the PI budget D-LATENCY-BUDGET (p99 ≤ 100 ms end to end on Thor, 4 cameras; ruled 2026-10-20). It decides nothing until the optimisation phase. (Amendment §7: no Thor number exists for P2 or for the search.)

## Where the gain went: C1 against C2 (both pre-registered outputs, nothing new mined from navtest)

| read | tokens | stratum | dPDMS | given slow-down | given speed-up | moved |
|---|---|---|---|---|---|---|
| C1 (`../../c1/RESULT_SPC.json`) | 905 B-REST | **SEEN-SESSION** (21 sessions, all trained on) | **+0.558 [+0.106, +0.925]** | **+2.46** | +1.96 | 17.3 % |
| C2 (this) | 12,146 navtest | **UNSEEN** sessions (44, 0 shared) | **−0.155 [−0.350, +0.037]** | **−1.18** | +0.25 | 19.8 % |

**What changed:** the same search on the same scorer moves a similar share of plans. Its slow-downs flip from the largest gain to the largest loss.

**HYPOTHESIS, post hoc, not registered:** the search **exploits the scorer's errors**.
- On tokens from drive **sessions** P2 was trained on (C1's 24 logs span 21 sessions, all of them in P2's own PDM fine-tune; `D-REFE-SESSION-EXPOSURE-1`), the scorer's ranking holds, so selecting on it pays. navtest's 44 sessions are all unseen.
- On unseen scenes, the plans the search moves to are the ones whose scores are most over-estimated, and slow-downs most of all.
- This fits the dose read's descriptive O4 − C8 = −0.119 [−0.227, −0.021]: the wider search lost more.
- ⛔ It is not a refutation of search at a larger dose or on a better scorer (§11, verbatim).

## Committed next arms (§11, `C2_FAIL_LOWER_CI`), and their state on 2026-10-22

1. **The search is CLOSED for P2's unchanged scorer on T1 at one training seed.**
2. **The P6 trunk.** Authorised, and being rebuilt now on the new pod: the 3-C bank, then the resume from HF's 6/25-epoch checkpoint (`../../../2026-10-22-p6-newpod-rebuild/RESULT.md`). The search is re-read on P6's scorer when P6 exists. This is the lever with the largest measured effect: the seen-to-unseen drop is **−0.71 points**, and it lives in the scorer.
3. **L2-BRANCH with a TTC-protecting term.** It needs its own amendment and a `launch_gate.py` profile.
   - Its EP-head form traded TTC (`D-REFE-L2-STEPB-1`), and C2 shows the unchanged scorer losing EP and TTC on unseen scenes too.
   - So a TTC term alone does not address the seen/unseen gap.
   - It stays DEPRIORITISED behind (2) and (4).
4. **The SEARCH-DOSE (M = 256) read:** already banked (`../../dose/read/`). It chose C8.
5. **New, from this read (named here, not yet registered):** a selection rule that is **pessimistic about the scorer's error**. Examples: a penalty on the distance from P2's pick, or a lower-confidence-bound score.
   - It must be fitted and judged on an **unseen-scene fit set**, never on navtest.
   - There is **no public unseen-scene pool outside navtest**: `D-DATA-U-FRAMES-1` / R56 (2026-10-19) showed that the public nuPlan val camera set holds frames for 11 of the 1,167 unseen logs, so the public sensor set is essentially navtrain ∪ navtest. The honest route is to make scenes unseen for the **scorer**: log-held-out (out-of-fold) scorer heads on cached trunk features, then the search on each held-out fold. The trunk has still seen those logs, so this tests HEAD-level error exploitation only, and it must say so. (Corrected 2026-10-22, R64.)
   - ⛔ navtest was used up by this look for H-SEARCH-P2-1 (one-run marker `navpipe_c2_ONE_RUN_H-SEARCH-P2-1.json`). A successor gets its own registration and its own look.

## Deliverable manifest

| artifact | where |
|---|---|
| `RESULT_NAVPIPE.json`, `bars.json`, `navpipe_state.json` (artifact sha256s), plans, `licence_c1.json`, `logs/` (analysis, per-chunk gate and GPU logs, run and supervisor logs), `chunks/c*/` (identity, bank manifest, sample) | repo, this directory |
| merged npz files (`bundle_merged`, `scores_merged`, `search_merged`) | sha256 in `final_npz.sha256`; inside the tar below |
| the whole work directory (1,513 files: chunks, T1 CSVs, harness logs, final), plus the dev-box run logs | `D:/Projects/TanitAD-artifacts/c2_navtest_2026-10-22.tar`, 453,918,720 B, sha256 `619bcf26c91bb988eb316e157ef71a2223c76fa72a2b965baed1d8d6bd32557e` |
| the sealed amendment, fixture and launch record | `../AMENDMENT_C2.md`, `../run/` (landed `bd49eaa`) |

## Erratum (2026-10-22, R64)

Item 5 first said that *"the only such pool is the 1,167 nuPlan val logs (BACKLOG `UNSEEN-STRATUM-DATA`); the S2/S4 fetch is the PI's call"*. That route was CLOSED on 2026-10-19 (`D-DATA-U-FRAMES-1`, R56).
- The MM read that BACKLOG row truncated, missed its own CLOSED note, and asked the PI for the fetch. The PI approved it; nothing was fetched.
- Item 5 now names the route that exists: log-held-out scorer heads.
