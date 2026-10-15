# REFe vs the reference paper, and the A40 plan (Master Mind, 2026-10-15 21:45 Berlin)

**Answering the PI:** *"where we stand in achieving the reference paper results and surpass them? what is the plan for the pod and is it used efficiently?"*

**Ground rules for this report:**
- **Tier.** Every PDMS here is T1, NAVSIM v1.1 pseudo-simulation of an open-loop plan. **Not closed loop.**
- **Paths.** PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`.

## 1. Where we stand against the paper

**The bar is 94.55 PDMS:** DriveZero, Table A13, DINOv3 ViT-L, navtrain, no SimScale (PUBLISHED, p. 31; `PK/SPEC_P6_PAPER_FAITHFUL.md`).

| | PDMS | EP | DAC | NC | TTC | source |
|---|---|---|---|---|---|---|
| paper, Table A13 | **94.55** | 92.72 | 99.25 | 98.80 | 95.74 | PUBLISHED (`PK/P6_CONFIDENCE.md`) |
| **P2, our best** (joint PDM fine-tune, 1 seed) | **88.82** | 82.34 | 96.87 | 98.81 | 95.32 | MEASURED, full navtest, 12,146 tokens / 136 logs (`PK/eval/RESULT_P2.md`) |
| SFT-4 u400 (the deployed system before P2) | 88.25 | 81.20 | 96.58 | 98.94 | 95.63 | MEASURED (`PK/eval/RESULT_SFT4.md`) |
| the pre-SFT-4 system | 84.46 | | | | | MEASURED (`D-REFE-SFT-4`) |

**Progress.**
- 84.46 → 88.25 → 88.82 is **+4.36 PDMS in about two weeks**.
- P2's step is separated: **+0.57 [+0.02, +1.18]** over u400, with a paired log-cluster bootstrap. All of it is ego progress: EP +1.14 [+0.61, +1.70].

**The gap is 5.73 PDMS.**

**Where the gap is, in PDMS terms.**
- `D-REFE-GAP-1` decomposes u400's gap exactly, gate first, on all 12,146 tokens. Of the 6.30 PDMS:
  - **4.15 is tokens zeroed by a gate**: DAC = 0 on 3.42 %, NC = 0 on 0.96 %, 529 tokens exactly zeroed;
  - **TTC is 1.31**;
  - **EP is 0.86**.
- P2 cut the zeroed tokens only from **529 to 509**.
- ⇒ **The largest lever is the zero tail: drivable-area and collision failures. It is not average progress.** The sub-score means mislead here: EP's mean gap (−10.4) is large, but a zeroed token costs its whole PDMS.

**The proposals are already good enough.**
- **Best of 64 = 97.07**, MEASURED (REVIEW_7 §2.4). That is above the paper.
- So the trajectory the paper would pick is usually among our 64 proposals. **The bottleneck is the scorer's choice.**
- Whether that also holds for the zero tail is being measured now; see section 4, stream 3.

**The longitudinal family moved against us.** P2's progress gain came with worse speed tracking: 3 of 21 longitudinal components are adverse (`RESULT_P2.md` §3). That read carries into P6.

## 2. What we tried this week, and what it left us

All outcomes MEASURED, from the register rows named.

| lever | outcome | what it left |
|---|---|---|
| **P2**, joint PDM fine-tune | **GO, +0.57** (`D-REFE-P2-R2`) | deployed candidate; seed replicate P2-s1 running (gives P6's seed floor) |
| **RL-S1 R0**, EPO full dose | **KILLED by K3**: TTC fell more than 0.5 against the control (`D-REFE-RLS1-R0`) | diagnosis: scorer generalisation, not the reward |
| **RL-S1b R0b**, quarter dose | **KILLED by K3**. PDMS +0.53 / +0.90 at u250 / u500 (RL-val selection tier), but TTC fails (`D-REFE-RLS1B-R0B`) | the EPO trade is not dose-separable |
| TTC/NC calibration (zero-A40) | **FAILS** (`D-REFE-RLS1-O2-CALIB`) | eliminated |
| H-ADAM | **REFUTED**: the TTC gap is mostly the CONTROL's gain (`D-REFE-RLS1-HADAM`) | ⇒ **R0c** (`--epo-ref armA`): built, verified, **launch-ready**, about 3.5 A40-h, **awaiting the PI** |
| **S1 aux drivable-area head** (aimed at the zero tail) | **NOT PROMOTED**. Navtest drivable IoU 0.50 / 0.52 / 0.49 against K2's 0.80; head AUC below the scorer's (`D-REFE-S1-READ`) | the frozen visual features do not carry the road ⇒ **the scene-tap probe is running** |

## 3. The plan to reach, then surpass, 94.55

**P6, the paper-faithful full run, is the main bet.**
- **Design:**
  - our DINOv3 ViT-L encoder with a trained LoRA;
  - 25 scene-epochs from step 0 = 10,075 steps;
  - online PDM labels on its own proposals, at lag 0.
- **Status:** every PI decision is taken, and the pre-registration is **IN FORCE** (`PK/eval/PREREG_P6.md`, landed `9539a95` before its first training step).
- **Expected:** 88.6–95.1 PDMS, midpoint about 91.8 (ESTIMATED; overlapping lever ranges, `P6_CONFIDENCE.md`).
- **Confidence:** about beating 88.25, MODERATE; about **reaching 94.55, LOW**. Every lever would have to land at the top of its range.
- **Its committed next arms:**
  - **GOAL MET** ⇒ a seed replicate ⇒ **P7.1**, a training-only map-supervised drivable-area/occupancy head (admissible, PI 2026-10-11) ⇒ **P7.2**, a scorer ensemble.
  - **ADOPT below the goal** ⇒ deploy P6. If the EP gap is largest, the next lever is selection and progress calibration (0 GPU); if DAC/NC/TTC is largest, it is P7.1, then P7.2.
  - **NOT PROVEN** ⇒ a scorer-only PDM fine-tune on P6 (that recipe gave SFT-4 +3.79).

**Surpassing 94.55 means closing the zero tail.** That is where our levers have failed so far: on frozen features (S1), and through RL (TTC). Two things change that:
- **P6 trains the trunk.** S1 could not.
- **P7.1 adds map supervision to a trained trunk.**

The scene-tap probe and the zero-tail anatomy (below) are the cheap reads that decide how P7.1 is built.

## 4. Live streams (2026-10-15 21:45 Berlin)

| # | stream | where | state | ETA |
|---|---|---|---|---|
| 1 | **P2-s1**, the P2 seed replicate | A40 | step 191 of 1,209 at 19:23Z, 58.5 s/step, GPU 100 %, 23.9 of 48 GB | ~2026-10-16 11:50Z |
| 2 | **P6 D9 metric cache** (CPU, beside P2-s1) | A40 pod CPU | 36,428 parts, ~191/min, 8 workers at nice 19 | ~2026-10-16 01:30Z (ESTIMATED) |
| 3 | **zero-tail anatomy of P2**: SELECTION vs PROPOSAL failure | dev-box CPU | pre-registered first, then run | tonight |
| 4 | **scene-tap vs visual-tap probe** (pre-registered `66fc3bb`) | dev-box GPU | resumed at 21:25 after a crash (section 6): 739 of 1,058 tokens extracted, 35 training units to go | ~00:30 Berlin |
| 5 | **R0c** | — | launch-ready | **awaiting the PI** |

## 5. The A40: plan, and is it used efficiently?

### Plan (dates ESTIMATED from measured rates)

1. **Now → ~10-16 11:50Z.** P2-s1 trains, with P6's CPU prep beside it.
2. **~10-16 12:00Z.** P2-s1 final and egress. Its navtest read then runs on the dev box (5.3 h, MEASURED for u400) and gives P6's seed floor F.
3. **Decision window: any short A40 arm goes HERE or waits a week.**
   - Candidates: R0c (about 3.5 A40-h), the arm the scene-tap probe names (S1b or S1c), and the N-s0 identity replicate (about 1.5 A40-h, low priority).
   - ⚠️ **Inserting them needs a hold on P6's GPU stage.** Today P6 takes the GPU as soon as P2-s1 is done. That is a small chain change, done on the PI's yes.
4. **P6's GPU stages.** Launch gate (about 40 min), then **about 7.0 days**. The rate is MEASURED: the final REFe run did 25 scene-epochs in 168.97 h at 0.2227 s/sample, and P2-s1 runs at 0.2264 today. That makes it **~10-23**, with the dev-box navtest eval and the **P6 verdict ~10-24**.

### Efficiency: yes inside a run, not between runs

**Inside a run it is efficient.**
- P2-s1 runs at **0.2264 s/sample**, within 1.7 % of the reference run's 0.2227, at 100 % GPU utilisation.
- **The overlap works.** P6's CPU prep runs beside training, and the guard reads the trainer at **57.97 s/step against its 57.90 baseline, a 0.1 % cost**. That takes about 6 h of CPU work (ESTIMATED) off P6's critical path.

**Between runs it was not efficient.** MEASURED from the chain-log timestamps:
- **Window:** 10-12 16:44 → 10-15 19:23 (74.7 h).
- **Idle:** the GPU sat idle for **at least ~15 h, about 20 %**.
- **Concentration:** almost all of it falls in one 30 h stretch (10-14 10:07 → 10-15 16:21), where it idled about half the time.

| cause | idle |
|---|---|
| RL-S1 R0's K3 kill. Downstream chains **refused** instead of taking the GPU (the 10-14 incident) | ~9.5 h |
| launch/preflight refusals: S1 H-arms (the G12 bf16 defect, found and fixed) and P2-s1 (test shims without an exec bit on Linux, fixed) | ~2.1 h |
| per-arm launch gate + preflight between the 9 S1 arms (15–49 min each) | ~3.3 h |

**Fixed since:**
- an idle watcher, which wakes on no GPU process or 0 % utilisation for 40 min;
- both shim defects;
- the overlap design.

**Still open, ranked:**
1. **Downstream refusal must hand the GPU on.** A killed arm must release the queue, never stall it. This is the 9.5 h class, and it is built before the next multi-arm series.
2. **Run arm k+1's CPU-side gate checks while arm k trains.** Worth about 15 min per arm: about 15 % of a 1.5 h arm.
3. **One GPU is the structural limit.** P6 will hold the A40 for 7 days, and every other A40 experiment queues behind it. A second GPU for the week (RunPod or HF; spend = the PI's decision) would let R0c, S1b/S1c and the replicates run in parallel with P6 instead of serially.

## 6. Incident tonight: the artifacts drive changed letter

- **What happened.** The external SanDisk 4 TB (exFAT), which holds `TanitAD-artifacts`, now mounts as **E:**. Every `D:/Projects/TanitAD-artifacts` path broke.
  - The scene-tap probe died. Its process was also killed when the previous Claude session ended.
  - Windows reports the volume's health as **Warning**, consistent with an unclean disconnect.
- **What I did.**
  - Restored the paths with a session-only alias, `subst D: E:\` (undo: `subst D: /d`; it is lost on reboot).
  - Relaunched the probe through WMI (`Win32_Process.Create`) so it survives session ends. It resumed from its banked state, losing no work.
- **For the PI.** Check the drive's USB connection. When nothing is writing to it, consider `chkdsk E:`. ⚠️ Not now: the probe writes there until about 00:30.

## 7. Decisions for the PI
1. **A40 order:** R0c (and the probe's arm, if it names one) **before** P6, at a cost of hours, or P6 first, which makes them wait about 7 days. Recommendation: before.
2. **A second GPU for P6's week** (spend), so the short arms run in parallel with P6.
