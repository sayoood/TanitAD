# PREREG_WSP_GAIN_D: the weight edit aimed at the DATA's own longitudinal gains, not the kinematic base's (Master Mind, 2026-11-07, written before any edit of this card is built or scored)

## 1. The case ([M] unless named)
* **WSP-GAIN** (`PREREG_WSP_GAIN.md`, frozen `92d61dcb`) moved the learned gains toward the non-learned base: ds/dv0 1.00, ds/da0 1.899. READ 2 HARMS in all four cells: M1 separated-better, every cell adding separated-worse guard rows (`D-V8F-WSPGAIN-1`).
* **P5** (`THEORY_CRAWL_AMEND_3.md`, frozen `dcb82ae9`) FAILS. Dilution of a0 is small, λ = 0.968 [0.966, 0.970] (`p5/P5_RESULT.md`).
* **Beside P5, post hoc** (`p5/raw/P5_NEXT_LEVER.json`): the DATA's own least-squares gains of the realised +2 s speed on the measured inputs, on 20,000 train windows at 3–8 m/s, are ds/dv0 = 0.936 [0.919, 0.952] and ds/da0 = 1.384 [1.360, 1.408] (R² 0.783).
* **The model's gains on the edit's own 20,000 train windows** (`wsp_edit/raw/FIT_RESULT.json`): ds/dv0 0.7705 and ds/da0 1.2080 (s0); 0.7536 and 1.1310 (s1).
* **Reading [H]:**
  * Both banked doses aimed past the data: the base's ds/da0 of 1.899 is 0.5 above what the logged drivers do. Restoring it makes the plan follow an acceleration response the data does not have.
  * The model's shortfall against the data is about +0.17 on v0 and +0.18 / +0.25 on a0. A crawl in moving windows is what a too-small gain on the present speed produces.
  * An edit to the data's gains is the smallest edit that removes the model's under-response without exceeding the drivers'.

## 2. The arm
* **E-GAIN-D, on C0_s for s in {0, 1}.** It is the banked edit (`wsp_edit/wsp_edit.py`, the pinned code of `PREREG_WSP_GAIN.md` §6) with only its targets changed:
  * ΔW = u · (Δg_a · w_a + Δg_v · w_v)ᵀ, where Δg = **g_data − g_model**, both measured on the edit's own train windows (`raw/train_window_ids.txt`, sha256 `948a2a1d…`). α = 1, one dose.
  * Δb = u · Δm, where Δm makes the edited model's train-window band-mean plan speed equal the data's band-mean realised speed on the same windows.
  * w, u, the layer (the last linear of `head_o`, longitudinal coefficients only), the ridge rule and the lateral bit-identity are the banked code's, unchanged.
* **g_data:** the OLS coefficients of the realised plan speed on (1, v0, a0, κ0) on those train windows. The plan-speed definition is the edit code's own (`wsp_common.py`). If it differs from P5's secant (knots 8–12), the edit code's definition decides, and P5's 0.936 / 1.384 are reported beside.
* **Fitted on train windows only.** No eval-141 window is used to fit.
* **Admissibility:** inputs unchanged (vision plus measured ego history); no logged-future or oracle channel at inference (the data gains are fitted offline from labels, which the PI's label rule allows); no situation-classifier output in the goal or the plan.

## 3. Reads, in order (the banked harness: `wsp_edit/read1_eval.py`, `wspedit_read.py`, `read2_chain.sh`)
1. **Mechanism:** the edited model's median ds/dv0 and ds/da0 on the TRAIN fit windows are within 10 % of g_data. The eval-141 gains (3–8 m/s) are reported beside. A miss is MECH-FAIL and nothing else is read.
2. **T1-kin on eval-141**, exactly as WSP-GAIN READ 2: the four families (STRATEGIC PAUSED, PI item 25), M1 `share_closed`, the paired episode-cluster bootstrap B 2,000 seed 0, both seeds, against C0 and against P10. Distance keeping per `raw/2026-11-07-dk141/` beside.
3. **T2 (box 2, after the queued BPC stage):** only on MOVES-M1 with guards holding. The arm is EGAIND through the stage-5 driver, read by the pinned dominance read (pin 4) with `--arm EGAIND --arm-stage 5`.

## 4. Outcomes, committed now (the WSP-GAIN table, unchanged)
| T1-kin read | next, executed in the same turn |
|---|---|
| MECH-FAIL | the edit is wrong, not the reading: name the defect and rebuild; no T2 stage |
| HARMS (more separated-worse guard rows than C0) | the data-matched point joins P3's chain beside WSP-GAIN's; no T2 stage; the perception lever (LoRA) stays the route |
| MOVES-M1, guards hold (no new separated-worse row) | the T2 stage on box 2; DOMINANT-T2 there is the scale signal, with video, after a replicate as `PREREG_T2_ATTRIB.md` §3 requires for a second arm |
| NO-EFFECT (M1 not separated) | the gain shortfall is not the crawl's lever at T1-kin; logged; the LoRA route stays |

* **The theory's prediction [H]:** T2 places any downstream edit on the frontier. The data-matched dose is smaller than both banked doses, so it should move M1 less and cost fewer guard rows. A MOVES-M1 with guards holding would show the banked doses failed by overshoot, not by the frontier.
* **Prior for MOVES-M1 with guards holding [E]:** low to moderate. No dose is tuned on eval-141; this card fixes one.

## 5. Pinned before the build
This card's sha256 is in `FROZEN_SHA_PREREG_WSP_GAIN_D.txt`. Before any eval-141 window is scored through an E-GAIN-D checkpoint, a dated log entry below pins the fitted g_data, Δg, Δm and the edited checkpoints' sha256. The banked tests must pass on the target-changed code, plus one new test: with g_data := g_model the edit reproduces C0 byte for byte.

## 6. Log (append-only, dated)
