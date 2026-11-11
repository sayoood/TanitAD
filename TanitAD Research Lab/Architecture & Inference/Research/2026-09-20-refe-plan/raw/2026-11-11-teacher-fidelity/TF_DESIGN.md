<title>Teacher fidelity design</title>

# TF_DESIGN: the lever ladder from our DriveRL teacher (89.81 / DAC 95.23) to the paper's 95.8 / 99.9 on navtest

TF design agent, 2026-11-11. CPU only, no navtest read, no GPU, no git writes. `PK` =
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

**Evidence classes:** MEASURED (ours, with artifact), PUBLISHED (DriveZero arXiv 2609.06055, banked PDF
`Library/papers/2609.06055_…pdf`), ESTIMATED, HYPOTHESIS.

**Estimators:**
* navtest: paired log-cluster bootstrap over 136 logs.
* navtrain held-out: log-cluster bootstrap over 11 logs.
* Both use 10,000 draws, seed 20260927, and a 95 % percentile interval, and both answer *another draw of LOGS*.
* Teacher, labels, tracker and scorer are deterministic.

**Tier:** NAVSIM v1.1 PDMS (non-reactive pseudo-simulation).

**All navtest numbers are exploratory.**

## 0 · Headline

1. **The largest lever is already measured.** L1, the tracker-aware target, moved navtest from 89.81 to **94.10**
   (F2, the 8-pose form of Tc20s; +4.30 [+3.33, +5.32], exploratory). PREREG_TF_N1 makes that read registered. It will
   not reach 95.8 alone.
2. **The second lever is DAC-certified fallback (L3, CERT).** On navtrain held-out it removes **all 10** remaining DAC
   zeros: 95.18 [94.02, 96.20], DAC **100.0**, +0.82 [0.00, +1.53] over Tc20s. The registered outcome is NOT SHOWN,
   because the CI lower bound is exactly 0. CERT cannot lower a row, and its gains sit in 3 of the 11 logs. The fix is
   more logs (TF-2b, about 1 CPU-h).
3. **Refuted today, with a successor named.**
   * Ungated margin shaping (L2b): −2.35, because 26 of its 32 new zero rows are slow cars pushed off a kerb. Its useful
     part survives inside CERT.
   * The 20-pose-submission explanation of the paper's number: T_sub20 equals T, −0.18 [−0.62, +0.26].
4. **What remains after DAC is EP.** Our teacher's EP is the same under every conversion, tracker and sampling (within
   0.4 points). The remaining EP shortfall to the paper (about 1.3 points alive, ESTIMATED about 0.5 to 1.1 PDMS) is
   upstream of NAVSIM: checkpoint (E3), rollout protocol or route.
5. **Protocol.** Under every reading the paper's DAC (99.9) needs fewer policy-owned off-road rows than our checkpoint
   produces (0.59 %). TanitAD's "paper level" is the LQR-tracked reading.

## 1 · Where we stand (all MEASURED)

| population | arm | PDMS | DAC | source |
|---|---|---|---|---|
| navtest 12,110 | T: teacher plan, 8 poses, LQR (registered) | 89.81 [88.56, 91.00] | 95.23 | `PK/eval/RESULT_P4_TEACHER_NAVTEST.md` §2 |
| navtest 12,110 | F2: same plans, 8-pose tracker compensation | **94.10** [93.59, 94.60] | 99.01 | `PK/raw/2026-10-05-teacher-navtest/f2_result.json` |
| navtest 12,110 | human, same harness | 94.55 [93.78, 95.23] | 100.0 | P4 C2 |
| navtrain held-out 1,018 | T tracked | 86.73 [79.87, 92.83] | 91.95 | TF-1 |
| navtrain held-out 1,018 | **Tc20s tracked** | **94.36** [93.07, 95.79] | 99.02 | TF-1 |
| navtrain held-out 1,018 | T transparent (no tracker) | 94.95 [93.78, 96.15] | 99.41 | E2 / TF-1 |
| navtrain held-out 1,018 | human tracked | 94.53 [93.07, 96.15] | 100.0 | E2 / TF-1 |
| PUBLISHED | DriveRL, Table 4 p. 11 | 95.8 | 99.9 | NC 99.8, TTC 99.1, C 99.0, EP 91.5 |

⭐ **L1 is already measured on navtest, in its 8-pose form.** On navtrain, Tc20s and Tc8 (F2's method) have the same
sim8 failure rate (0.98 %), and their labels differ by a median of 0.0056 m (`p5s_cubic.json`). ⇒ F2's +4.30
[+3.33, +5.32] → 94.10 is the navtest read of L1 (exploratory). The registered N1 is a confirmation; it is not a
discovery.

Our harness reads the paper's human row (94.8, EP 87.5) as 94.55 and EP 86.94. So "paper level" expressed in our
harness units is probably about 95.55 (ESTIMATED; it assumes the offset is a property of the harness). We bar on the
literal 95.8.

## 2 · The protocol question: is the paper's 95.8 tracker-transparent or LQR-tracked?

**What the paper says (PUBLISHED):**
* the score comes with "4-s rollouts from ground-truth symbolic inputs" (p. 11);
* the teacher "rolls out a 20-step trajectory" (Fig. 2, p. 5).

Nothing more: no rate, tracker, sampling or route. Every other row of Table 4 (PDM-Closed, human, all camera methods)
comes from NAVSIM's standard LQR-tracked scorer. The release ships no NAVSIM export
(`grep -il navsim DriveRL/` = 1 comment). **The paper alone cannot settle the question.**

**Four facts that bound it:**
1. **A 20-step submission IS standard LQR scoring.** NAVSIM v1.1's `pdm_score` → `transform_trajectory` uses the
   submitted trajectory's own `trajectory_sampling` (`navsim/evaluate/pdm_score.py:24-49`, vendored @ 3e8291b,
   MEASURED by reading the source).
   * So there is a **third reading**: the 20 poses at 0.2 s submitted as they are and LQR-tracked. It needs neither
     pre-distortion nor a transparent scorer.
   * TF-2 arm **P-S20** measures this reading (§3).
   * Our W3 harness accepts only 8 × 0.5 s seams (`w3_agents_v1.py:94-97`).
2. **The DAC gap is in the policy under every reading.** With the tracker bypassed, our navtest DAC ceiling is ≤ 99.41
   (class A+B = 74 of 12,110, D1). The policy's own off-road rate is 0.59 % on both navtest (72 / 12,110) and navtrain
   (6 / 1,018). The paper's 99.9 implies ≤ about 0.1 %. ⇒ **no protocol reading closes DAC; L2 is needed in all
   three.**
3. **Comfort is a weak fingerprint.** The paper's C = 99.0 is the lowest in Table 4: the camera rows read 99.3 to 100
   and the human 99.9 (the printed PDM-Closed row is internally inconsistent in our text copy).
   * Ours: tracked 8-pose 99.92 (T), 99.82 (F2), 99.90 (Tc20s on navtrain); transparent 100.0 (E2's builder).
   * So none of our 8-pose or transparent paths reproduces a 1 % comfort loss. A pipeline that tracks finer-grained
     poses could (HYPOTHESIS, read in P-S20's C column, §3).
4. **EP is above the human in both the paper's row and ours.** The paper: 91.5 vs 87.5 (+4.0). Ours transparent on
   navtrain: 90.08 vs 87.06 (+3.0). Ours F2 on navtest: 88.76 vs 86.94 (+1.8). ⇒ our teacher is also slower than
   theirs on surviving tokens, by about 1 to 2 EP points, whatever the reading.

**What each reading implies for "paper level":**

| reading | what reaching 95.8 means | our nearest MEASURED number | levers needed |
|---|---|---|---|
| **LQR-tracked, 8-pose plan** (the student's own protocol) | our plan reaches 95.8 under the same tracker the student faces | F2 navtest 94.10 | L1 + L2 (DAC) + an EP lever |
| **transparent** (rollout scored as executed) | a number the student can never be scored on; it does not improve REFe's targets as the student is scored | navtrain 94.95; navtest ESTIMATED ≈ 94.9 | L2 + an EP lever |
| **LQR-tracked, 20-pose submission** (standard API) | reproducible with stock NAVSIM; REFe's student emits 20 poses too, so the same submission is open to it | navtrain P-S20 (§3) | per P-S20 |

⇒ **TanitAD's definition of "paper level" is the LQR-tracked reading.** The student is scored that way, and teacher
targets matter only as the student's labels. The transparent read is reported beside it and never in its place.
Whether the paper used 8 or 20 poses is UNVERIFIED. Only the authors' NAVSIM code could settle it, and that code is
not in the release.

## 3 · Residual-failure anatomy (MEASURED, `tf2/tf2_result.json` → `anatomy`; navtrain held-out)

The 11 Tc20s-tracked zero rows are 10 DAC and 1 NC. **9 of the 11 sit in 2 logs** (veh-40_005: 5, veh-50_014: 4). The
6 transparent DAC-zero rows are all among them.

| owner | n | kind | where on the path | margin | human at the token |
|---|---|---|---|---|---|
| **policy** (fails transparent too) | 6 | DAC | 5 are sharp 80 to 97° turns at 3.6 to 4.5 m/s, 4 of them in intersections; 1 is near standstill, straight. First violation at 1.5 to 3.6 s, on a right corner (FR or RR) | −0.06 to −0.30 m (shallow) | 0.06 to 0.67 m; two rows have only 0.06 to 0.09 m |
| **tracker** | 4 | DAC grazes | 2.4 to 3.9 s. Two were broken by Tc20s itself (T passes) | **−0.6 to −2.8 cm** tracked; transparent +0.12 to +0.34 m | 0.15 to 1.34 m |
| **tracker** | 1 | NC | standstill start (v0 0.02); every tracked arm collides at 3.3 s, the transparent path does not | n/a | n/a |

**Protocol probe P-S20 (registered S20-NO).**
* MEASURED: submitting the 20 poses at 0.2 s through stock NAVSIM gives DAC 91.95, identical to the 8-pose plan, and
  PDMS −0.18 [−0.62, +0.26].
* So the tracker gap is the LQR's, not the sampling's.
* Its comfort is 100.0, not the paper's 99.0.

## 4 · The lever ladder, ranked by measured effect

| rank | lever | effect | evidence | cost | status, and what unblocks it |
|---|---|---|---|---|---|
| **1** | **L1 Tc20s** (tracker-aware target) | navtest **+4.30 [+3.33, +5.32] → 94.10**, DAC 99.01 (F2 form); navtrain +7.64 [+2.29, +13.60] | MEASURED, exploratory (F2, TF-1) | 0 GPU; about 2.5 CPU-h for N1 | **PREREG_TF_N1_DRAFT, ready for the MM to freeze** |
| **2** | **L3 CERT** (DAC-certified fallback over [Tc20s, T, M]) | navtrain **+0.82 [0.00, +1.53]**, DAC-zero 10 → 0, PDMS 95.18; navtest ESTIMATED +0.6 to +0.9 (F2's DAC-zero share is 0.99 %) | MEASURED navtrain, registered NOT SHOWN (CI lower bound = 0) | about 1 CPU-h for TF-2b; about 5 CPU-h on navtest | **TF-2b next**: 687 un-banked held-out samples in other logs, the same bar on fresh logs. If it WORKS, CERT becomes N1's candidate |
| 3 | **EP lever** (the largest term left once DAC is closed) | ESTIMATED **+0.5 to +1.1 PDMS** (alive EP about 1.3 points below the paper's against the human; the review's swap gave +1.1) | the mechanism is unmeasured; TF-2 eliminates the tracker, the conversion and the sampling (all within 0.4 EP) | — | candidates in order: **E3** checkpoint identity (running); release TTS N = 8 on held-out alive rows (F1's pass set +0.51 [−0.07, +1.72], descriptive); route / centerline agreement |
| 4 | L4 E3, checkpoint identity | decides whether 3 is fixable without training | — | CPU, running | `e3_identity/HANDOFF.md`. Outcome B (≥ 2 points below 94.50) closes "same teacher" |
| 5 | L2a privileged TTS selection | navtrain: CERT already repairs all 6 policy rows, so the measured marginal value is about 0 | F1 (critic TTS) 0 / 18 on class A | 1 to 1.7 CPU-h (§6) | **only if TF-2b leaves policy rows that CERT cannot fix** |
| ✗ | L2b ungated margin shaping | **−2.35 [−5.15, +0.02]**, REFUTED in this form | MEASURED, TF-2 | — | it lives on gated, inside CERT |
| ✗ | P-S20, a 20-pose submission | −0.18 [−0.62, +0.26] | MEASURED, TF-2 | — | eliminated as the explanation of 95.8 |
| ✗ | sim rate (E1) | 20 Hz worse, +1.84 DAC-fail points | MEASURED, E1 | — | eliminated |

**Arithmetic toward 95.8 on navtest (ESTIMATED, one point at a time):**
* F2 94.10;
* + CERT ≈ +0.7 → **≈ 94.8, DAC ≈ 99.8**;
* + EP lever ≈ +0.5 to +1.1 → **95.3 to 95.9**.

⇒ paper level is **reachable without training only if the EP lever is real**, so E3 decides where to go next. Under the
human-anchored reading of our harness (95.55), the ladder through rung 2 is about 0.75 short.

## 5 · Admissibility (PI 2026-08-03: privileged map and agent geometry at t0 are admissible for teacher LABELS)

| lever | reads | log's future? | the paper's teacher does it? |
|---|---|---|---|
| L1 Tc20s | t0 ego, NAVSIM's public LQR + bicycle model | no | no: a conversion to NAVSIM's interface |
| L2b margin shaping (M) | static map (the scorer's DAC polygons around the t0 ego), t0 ego | no | no: a label-quality lever |
| L3 CERT | map + t0 ego + tracker; uses only the tracked `dac` verdict, which is map-only | no | no. ⚠ It selects by part of the test metric, so its DAC is partly guaranteed by construction |
| L2a privileged TTS selection | the release's own candidates; map margin over its 5-step look-ahead; agents by the release's CTRA extrapolation from the current state | no | its TTS yes; the privileged scoring no |
| P-S20 | nothing new: submission format only | no | unknown (HYPOTHESIS) |
| ⛔ selecting by NC / TTC / EP verdicts | the log-replayed future agents | **YES** | a leak relative to the paper's teacher. No lever here does it |

The REFe student stays vision-only at inference. Every lever above acts on its training labels.

## 6 · L2a run plan (privileged selection among DriveRL's own test-time samples): written, not run

**Why it is not run first.** The closest measured analogue is F1 on navtest: the release's own TTS at N = 16 repaired
1 of 120 failures and **0 of 18 class-A rows**, and 119 of 120 were byte-tied (`PK/raw/2026-10-05-teacher-navtest/f1/f1_result.json`).
That run used the critic's score and a 0.03 switching margin.

**What is unmeasured.** Whether the candidate SET ever contains a compliant path. F1 banked only the selected
trajectory, not the candidates.

**Cost.** About 60 to 120 s per token on CPU (F1: 120 s per token at N = 16). L2a costs far more per row than L2b.

**Plan:**
1. **Code.** `l2a_priv_tts.py`. Subclass `driverl.nuplan.tts.NuPlanTTSSelector` (`C:/Users/Admin/dz/DriveZero/DriveRL/src/driverl/nuplan/tts.py:83`)
   and override `_select_baseline_priority_by_total_return` (line 354) so that it scores each candidate by the minimum
   footprint margin. That margin is measured to the drivable union (the scorer's DAC polygons around the t0 ego) over
   the candidate's L = 5-step look-ahead states from `_rollout_candidates` (line 391).
   * Switch away from the mode only when the mode's margin is < 0.60 m and the best candidate gains ≥ 0.10 m.
   * Break ties by the release's own S(i).
   * Background agents keep the release's CTRA extrapolation from the current state, so the log's future is never read.
2. **Rollout.** Through `PK/eval/teacher_navtest_seam.py --heldout` (the C3 path, which rolls navtrain held-out
   samples), with `tts_enabled` True, `tts_num_candidates` 8 and their seed. Interpreter:
   `C:/Users/Admin/venvs/driverl-eval`, CPU, ≤ 3 processes, and FreeVirtualMemory > 10 GB **first**: these are torch
   processes, the WinError-1455 class.
3. **Population.** The policy-own rows left after TF-2, plus 40 pass rows (`random.Random(20261111)`) as the harm
   control. That is ≤ 50 rows: **≈ 1 to 1.7 CPU-h**.
4. **Controls.**
   * Identity: with N = 1 the path must reproduce the banked t20 to ≤ 0.026 m (C3's bar).
   * Regression: select the MIN-margin candidate; it must break ≥ 5 of the 40 pass rows.
5. **Read.** The new 20-pose path goes through P5-S Tc20s and is scored as in TF-2.
   * **Works:** ≥ 4 of 6 policy rows repaired, 0 pass rows broken.
   * **Otherwise refuted.** The remaining policy rows then need a teacher re-train or a margin term in the policy, both
     blocked on the PI and GPU.

## 7 · What the Master Mind must rule

1. **PREREG_TF_N1 freeze.** Resolve its candidate rule from `tf2/RESULT_TF2.md`, then freeze, then append the ledger
   entry (K 40 → 41), then run. Also rule on whether its two control arms count as separate K reads.
2. **Families read on the executed vs the reference trajectory** for tracker-aware labels. This is P4's open PI
   decision (a); N1 needs it to report the four families.
3. **Whether a 20-pose submission may be used** (if P-S20 reads CONSISTENT). It changes the submission format against
   every earlier navtest read, so comparability and the student's protocol are the PI's call. It also needs a W3
   harness change (8-pose-only check).
4. **CERT is selection by part of the test metric** (DAC, map-only). It is admissible as a label privilege, but a CERT
   number must never be quoted as "our checkpoint equals the paper's".
