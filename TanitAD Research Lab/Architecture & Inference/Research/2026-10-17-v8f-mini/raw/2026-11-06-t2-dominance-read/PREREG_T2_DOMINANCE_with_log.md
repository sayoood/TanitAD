# PREREG_T2_DOMINANCE: is v8F's composite DOMINANT over the P10 floor under T2 re-perception? (Master Mind, 2026-11-06; frozen before any stage-2, stage-3 or stage-4 rollout is scored)

## 0. What was seen before this card was frozen (disclosure)
* **Stage 2, run 2** (42 launch scenes, box 2, from 16:37) is running.
  * Its rollouts are NOT scored, and no row of any arm has been computed or read.
  * What has been seen: cell keys and DONE / FAIL states in the heartbeat; run 1's failure counts (14 of 588 DONE, every failure a missing
    reference frame); and the canary cell `e848c843|T2|C0_s0:DONE`.
* **Stage 3** (the 20 guard scenes) is built and not run.
* **The composite arm C0C** is not built (AGENT L, `t2v8f/BRIEF_T2V8F_C0C_L.md`).
* **The read script** for the crawl card (`t2v8f/code/read_stage2.py`) was pinned before data. This card adds a read of its own (section 6).

## 1. Why
* `PREREG_T2_CRAWL.md` (frozen `2c4c6557`) section 6, outcome 1, commits this read: a T2 dominance read of C0's composite against P10 over
  all 62 scenes, all four families, with Read A's rules. If D holds at both seeds, that is a counted DOMINANT at T2 on these scenes and the
  scale signal to the PI.
* Dominance over the floor is the programme's success criterion (PI, 2026-10-25: *validate the v8f by real dominant driving results*).
  At T1-kin the composites fail Read A's D2 on one row, the M1 crawl share. T2 is the first tier that re-renders the camera at the
  executed pose.

## 2. Hypothesis
**H-T2-DOM-1.** v8F's adopted composite, C0C, is DOMINANT over P10 at T2 by Read A at both training seeds.
* **The composite:** C0, then `sg72` smoothing over its own prior, then the `kbox` curvature projection.
* **Scope:** the 62 screened NuRec scenes.

## 3. Design (zero training)
* **Scenes and starts.**
  * The 62 scenes of `t2_scenes/SCREEN_RULE.md`, frozen before any scene log was read: 42 launch and 20 guard.
  * 121 starts: stage 2's 81 on the launch scenes and stage 3's 40 on the guard scenes (`DEFINITIONS_K.md`, written before its draw).
* **Disclosure.** The scenes were selected on the human's logs only and exclude v8F's training clips. Launch scenes are over-represented
  by design (42 of 62). A result here is a result on these 62 scenes at T2, never on PhysicalAI-AV at large.
* **Condition.** T2, as `PREREG_T2_CRAWL.md` section 3 defines it: the camera is rendered at the executed pose and time, NuRec agents are
  replayed at log time (`objects`), and the executor is playback (`base`).
* **Arms.**
  * **C0C_s0 and C0C_s1 (deciding):** the composite at every replan. It must use the same arithmetic as the T1-kin tool
    (`motion_c/ri/code/v8f_t1_eval_ri.py`, sha256 `0417c838...`, `smooth_plan` then `feasible_plan(kbox)`); gate G1 proves it.
  * **P10:** the frozen prior from the same forward, at T2. Its rollouts are reused from stages 2 and 3, on the same windows.
* **Rows.**
  * The 19 rows of the T1 reader. They are built by the pinned reader `recover_next/read/code/recover_read.py` (sha256
    `de5cfbf572ba2184e5f85a942ac79a6a6fa4df422f83367bbd9fa0e6966e95d1`) through its own `rows_of`: `gapdecomp.decompose`, with the reader's
    class split and per-label pooling.
  * The cluster `eid` is the SCENE index, so the starts of one scene form one cluster. B = 2000, seed 0.
* **Read A.** `read_a_block` of the same reader, per seed:
  * **D1:** ADE6 is separated-better, with point ≤ −0.2415 m.
  * **D2:** no row of the 19 is separated-worse.
  * **D3:** every one of the four families has at least one separated-better row. STRATEGIC is as Read A defines it. The three-family
    form is reported beside, per PI ITEM 25.
* **What the interval answers.** Another draw of SCENES only.
  * The training-seed question is answered by requiring both seeds.
  * The planner is deterministic, and the renderer is checked deterministic (gate T5 of the crawl card).
  * There is no replicate arm on this rig. That limit is named in section 7.

## 4. Gates before any row is read (each with a deliberate-regression arm that must go RED)
* **G1, composite identity (C0C-ID).** On banked cached features, C0C's tick-0 plan equals the T1-kin tool's composite tick-0 plan for the
  same window.
  * RED arms: smooth off, feasible off, and the prior swapped for zeros.
  * With smooth and feasible both off, the arm reproduces C0 exactly.
* **G2, the crawl card's gates.** T0 to T5 and the actor falsifier, as recorded PASS in each stage's `STAGE*_RESULT.json`.
* **G3, completeness and pairing.**
  * Every C0C(T2) and P10(T2) cell on the 121 starts is DONE, with `overall` COMPLETE.
  * The window keys pair exactly (`t2v8f_score.pair_check`).
* **G4, the reader.**
  * Its sha256 matches the pin.
  * It reproduces one banked T1-kin Read A block bit for bit from its banked arrays, as a control. If none can be reproduced, the control
    names why, and no row is read.
* **If any gate fails,** the instrument is fixed first and no row is read.

## 5. Outcomes, committed now (Rule Zero; read in this order)
1. **DOMINANT-T2.** Read A holds (D1, D2, D3) at BOTH seeds and all gates PASS.
   * This is a counted DOMINANT at T2 on these 62 scenes. It goes to the PI as the scale signal, with:
     * the scene-selection disclosure of section 3;
     * the video evidence: T2 renders of C0C beside P10 on launch and guard scenes;
     * the per-family table.
   * It counts whatever the crawl card's outcome. Read A's D2 already contains the M1 row, so a crawl that persists against P10 makes D2 fail.
2. **NOT-DOMINANT.** Read A fails at either seed. Report `failing_terms` per seed, verbatim. The next card targets the failing term with the
   largest measured margin, and it is drafted in the same turn as the read:
   * **D2 on M1 or on another LONGITUDINAL row:** the crawl card's committed lever for its outcome. That is the perception lever under
     2(a) or 3, or T2 launch states as DAgger data under 2(b).
   * **D2 on LATERAL rows only:** a card for the `hold` executor at T2.
     * Why: `RESULT_EXEC_STANDSTILL.md` MEASURED that `hold` shrinks C0's lateral deficit against P10 at T1-kin. The yaw-rate row of C0
       against P10 is not separated-worse under `hold`.
     * `hold` is not adopted there, because that outcome is UNCOMMITTED. It needs this card's evidence first.
   * **D1 only** (ADE6 is better, but its point is above −0.2415 m): report the margin. The next card is the longitudinal row with the
     largest separated-worse or unseparated deficit.
   * **D3 only** (a family with no separated-better row): that family's lever, named from its rows.
3. **UNDERPOWERED.** No row of the 19 is separated at either seed.
   * Extend to the next 42 launch scenes in the frozen screening permutation, within the PI's 80 GB approval.
   * Not a result.
4. **Any gate fails.** The instrument is fixed first. No row is read.

## 6. The read is code, pinned before data
`t2v8f/code/read_t2_dominance.py`, written and sha-pinned before any C0C or P10 row is computed:
* It takes the stage-2, stage-3 and stage-4 pulls.
* It builds the rows through the pinned reader's `rows_of` with `eid` set to the scene, and applies `read_a_block`.
* It writes `DOMINANCE_T2.json` and `DOMINANCE_T2.md`.

Its sha256 is appended to this card's log in section 8 BEFORE it is run on data. A later edit of the script is a refused control.

## 7. Variance and validity, named
* **Episodes.** The scene-cluster bootstrap answers another draw of SCENES. 62 scenes are few. A separated row here is weaker than one on
  the 141-episode T1-kin set.
* **Training.** Both seeds must hold. There is no third seed and no replicate arm on this rig (`H-ESTIM-SEED-1`). A row that clears at both
  seeds by a margin smaller than |X_s1 − X_s0| is reported with that gap beside it.
* **Inference.** Deterministic. T5 pins it.
* **Render gap.** The encoder's rel-L2 / cosine on rendered frames against logged frames (gate T1 of the crawl card) is reported beside. It
  is never deciding.
* **Tier.** T2 re-perception: camera re-rendered, agents replayed at log time, so agents do not react to the ego. A T2 DOMINANT is not a
  reactive-traffic result, and the report says so.

## 8. Log (append-only; every entry dated)
* 2026-11-06: drafted and frozen by the Master Mind before any stage-2, stage-3 or stage-4 row was computed. The sha256 of this file as
  frozen is in `FROZEN_SHA_DOMINANCE.txt`. Any later change is a dated entry below this line, never an edit above it.
* 2026-11-06 17:42, Master Mind. Three readings of this card, fixed before any row of stage 2, 3 or 4 was computed (raised by AGENT L, `t2v8f/HANDOFF_L.md`):
  1. **Outcome order.** Section 5 lists UNDERPOWERED after NOT-DOMINANT, but a block with no separated row fails D1, so the literal order never reaches UNDERPOWERED. Reading: UNDERPOWERED (no row of the 19 separated at either seed) is tested BEFORE NOT-DOMINANT; DOMINANT-T2 stays first and GATE-FAIL overrides all.
  2. **G3 stays as written.** Every stage's `overall` must be COMPLETE. A stage left INCOMPLETE for a skipped beside pass is relaunched (DONE cells are skipped) until COMPLETE; the deciding cells alone do not satisfy G3.
  3. **G2, where each gate is recorded.** Stage 2's `gate` job records T2g only (`STAGE2_RESULT.json`: logged plan played back to 1e-06 m, max 1.4e-14 m, 81 starts). G2 is read as all of: (a) T1 (render gap, reported), T5 and the actor falsifier: `STAGE1_RESULT.json` on box 2, `overall` ALL_GATES_PASS, on the same cell code, which the read checks by the package hashes; (b) T0 and T4: the dev-box gate tests in `t2v8f/tests` green at each package's build; (c) T2g: each stage's own `gate` PASS; (d) T3: recomputed on each stage's own data, ZERO(T2) equal to P10(T2) on every window, exactly; (e) the per-scene actor verdict: every deciding cell's `actor_attach.json` present, its verdict reported per scene. Any of (a) to (e) failing is GATE-FAIL.
  * The read script as first pinned, `t2v8f/code/read_t2_dominance.py` sha256 `df4e863649d6d84d78d3f8999b21287bd945a47d60120455ca37c979028efd24` (AGENT L), implements G2 as (c) only. It is re-pinned once, before data, with (a) to (e); that second sha256 is a dated entry below.
* 2026-11-06, Master Mind. The read script re-pinned, before any row of stage 2, 3 or 4 was computed: `t2v8f/code/read_t2_dominance.py` sha256 `6345d4c704ea46806271761ad3b53e376228f942e605da140fff3057cb9d2f5c` (AGENT M). It adds G2 parts (a) to (e) of the 17:42 reading, each with a RED arm (18 tests). It checks this card by the sha256 of its frozen first 8,497 bytes (`38aa9df3...`), because the dated entries below the log line change the whole-file hash. A per-scene actor verdict other than ACCEPTED is tabulated per scene, as reading 3(e) says, and is not a gate failure. G1 (`C0C_ID_RESULT.json`, sha256 `c8df31eb...`) PASSES at its exact tiers; its end-to-end comparison with the banked dump uses tolerances declared after the float32 batch-size noise (up to 1.1e-5 m) was measured, and is reported as such, never as the identity.
* 2026-11-06 19:31, Master Mind. Rule R-SCN of `AMENDMENT_T2_SCENE_REFUSAL.md` (frozen `2167a7eb`) binds this card, written before any row of stage 2, 3 or 4 was computed.
  The renderer refuses some scenes before any planner runs: 7 of 42 launch scenes in stage 2, and stage 3's first guard scene. A refused scene leaves the read for BOTH arms, per
  the amendment's rule 4, and G3 is read as: every cell of every INCLUDED scene DONE, the failed cells exactly the excluded scenes' cells, `EXCLUSIONS*.json` ATTRIBUTED for each stage.
  The read script `read_t2_dominance.py` is re-pinned once more with this reading, before it runs on data; that sha256 is a dated entry below.
* 2026-11-06 20:20, Master Mind. The read script re-pinned with rule R-SCN, as the 19:31 entry above fixed it: `t2v8f/code/read_t2_dominance.py` sha256
  `d66647180a040318fc48b65bbdd068b637cf5d5a3e8b2ca0f91c4cee6bf1280a` (AGENT P; the second pin was `6345d4c7`). It takes `--exclusions2`, `--exclusions3` and `--exclusions4` (all three or none; with none it reads as the second pin).
  Each file must carry its own stage label and the sha256 of the result file passed beside it. The union of the excluded scenes leaves both C0C and P10 before pairing.
  Tests: 167 passed in the full suite (`t2v8f/logs/pytest_full_P.log`); 40 of 40 deleted guards turn a test RED (`logs/mutations_dom_p.log`).
  * **What had been seen when this pin was entered (disclosure).** No C0C row exists: stage 4 launched at 20:18 and none of its cells is scored. Stage 3 is pulled and
    its EXCLUSIONS file built; none of its rows is computed. Stage 2's rows WERE computed at 19:3x for the crawl card: C0, P10, ZERO, BLIND and DAGOACC at T2 and RR,
    including P10's means on the 35 included launch scenes (`RESULT_T2_STAGE2.md`). The read's content was fixed in the 19:31 entry, before any of those rows; this pin
    implements that entry and changes no threshold, row, estimator or outcome of sections 3 to 5.
  * The renderer refused 14 of the 62 scenes in stages 2 and 3 (7 launch, 7 guard; `EXCLUSIONS2.json`, `EXCLUSIONS3.json`, both ATTRIBUTED). The read covers the 48 others.
* 2026-11-06 22:05, Master Mind. **The read of record: NOT-DOMINANT** (`t2v8f/stage4/score/DOMINANCE_T2.{json,md}`), the pinned script `d6664718...` on the final pulls of stages 2, 3 and 4, with
  `--exclusions2/3/4`. Gates G4, G1, G2 and G3 PASS. Stage 4 finished on box 2 at 21:53; its `EXCLUSIONS4.json` is ATTRIBUTED and names exactly the 14 scenes stages 2 and 3 refused
  (no new refusal). The read covers 48 of 62 scenes, 93 paired starts per arm.
  * **Two invocation errors before the run of record, disclosed; neither changed an input and neither computed a row.** (1) The first call used a Python without numpy and died at
    import (`logs/read_dominance_numpy_missing.log`). (2) The second set `T=$PWD`, which Git Bash spells `/e/...`; MSYS rewrites that form for plain arguments but not inside the
    `tar::member` argument, so G2(a) read `G2A_MANIFEST1_MISSING` and the read stopped at GATE-FAIL with `per_seed` null (output kept in `stage4/score_gatefail_msyspath/`). The run of
    record spells every path `E:/...`, exactly as `STAGE4_LAUNCH.md` section 2 does.
  * **failing_terms, verbatim.** s0: `D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.7893)`; `D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)`;
    `D3 (TACTICAL: no separated-better row)`. s1: `D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.6399)`; `D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE)`;
    `D3 (TACTICAL: no separated-better row)`; `D3 (STRATEGIC: no separated-better row)` (the strategic family is PAUSED, PI item 25; the three-family form is also not met).
  * **Committed next (section 5, 2):** D2 fails on M1 and on LONGITUDINAL rows, so the next card is the crawl card's lever for its outcome, CRAWL-PERSISTS (a): the perception lever
    (`PREREG_V8F_LORA`) with T2 as its evaluation and this read as its case. Drafted the same turn: `t2_prereg/DRAFT_PREREG_T2_DOM_LORA.md`. The PI provided a GPU pod for that run on 2026-11-06.
