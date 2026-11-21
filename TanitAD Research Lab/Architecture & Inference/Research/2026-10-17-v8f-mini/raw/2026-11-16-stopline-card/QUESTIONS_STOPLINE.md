# QUESTIONS_STOPLINE: what only the PI can decide for the map stop-line card

Card: `F:/TanitAD-scratch/stopline/PREREG_STOPLINE_DRAFT.md` (DRAFT, 2026-11-16). Authority so far: the PI's *"3 yes draft"* (2026-11-16), which covers drafting only. Recommendations are the drafter's; the decisions are the PI's. Each question stands on its own.

## Q1. May the REFe planner take a map-carried stop line as an INFERENCE input?

**What the input is.** 13 numbers per frame, derived only from:
* the static nuPlan HD map;
* the car's current position on it;
* the navigation command the model already receives (LEFT / STRAIGHT / RIGHT).

The 13 numbers: is a controlled stop line ahead within 80 m; how far along the lane path; traffic light or stop sign; lateral offset.

It does NOT carry:
* the light's colour;
* anything from the future;
* the logged route;
* any other agent.

**Why it needs a ruling.** No ruling names map inputs, and the rules read both ways.
* **For:**
  * nav is an input at training and inference (*"Nav is an INPUT at training AND inference"*, `CLAUDE.md`);
  * the goal-input rule forbids only the situation classifier's output in the goal (*"we need to be careful not to include the result of the situation classification in the goal input"*, 2026-08-03), and this input cannot be computed from it;
  * the vision-only rule is written for the scenario classifier (*"for ground truth data of scenario classification you can use both ego and other label, for inference only vision"*), and REFe has none.
* **Against:**
  * a painted stop line is visible, and the PI's time-of-day ruling asks for *"pure images and past frames no additional informaio"*;
  * REFe's own code says *"the map is never a model input"* (`refe/navsim_lane.py`);
  * the DriveZero student it reproduces uses cameras, ego kinematics and a command only;
  * NAVSIM gives a sensor agent no map and no global pose, so the car's map position is an extra input (on a real car it comes from localisation).

**One finding the ruling should know.** The behaviour scorecard's own stop line cannot be the input. On the 55 navtest rows where the human waits at a red light to the end of the scenario, the scorecard finds the line in 55 of 55 only by following the human's path up to 30 s into the future (MEASURED, re-count of `feasibility_rows.csv.gz`). As an input that is the same ego-future echo that ruled the PhysicalAI label out. The card therefore selects the line from the current lane and the command only, and tests that rebuilding with shuffled future frames changes nothing.

**Options.**
* (a) Admissible on the REFe planner under four conditions:
  * location only, no signal state;
  * never an input to a situation or scenario classifier;
  * reported as a separate "map-equipped" configuration, never pooled with the paper reproduction or camera-only rows;
  * transfer to the product only as a mechanism claim.
* (b) Admissible only as a training label: a vision stop-location head learns the map line and the car uses vision at inference. That is a different card, and the drafter's named follow-on.
* (c) Not admissible in any form.

**Recommendation: (a).** It is the cheapest experiment that can tell whether stop coverage is an information gap or a capacity gap, after four perception levers failed on PhysicalAI-AV. Under (a), a positive result goes straight into (b), which is the form the product could actually use.

## Q2. Approve the spend for the first two stages, on REFe's L40 pod after P5-C?

**What it is.**
* **Stage 0, no GPU:**
  * build the stop-line feature for navtest and the training rows;
  * check that the teacher's training targets actually stop at red-light lines (if they do not, the GPU stage does not launch and a label card comes back to you instead);
  * count how many navtest windows the read can use.
* **Stage 1:** two or three 2,000-step fine-tunes from the P5-C base (P7 if its read passes, else P6):
  * the stop-line arm;
  * a matched no-input control at seed 1 for the training noise floor;
  * the seed-0 control. It can be P5-C's own C0 checkpoint if a bit-identity check passes, which saves one arm.
* **Stage 2:** navtest reads on the dev box. They include a donor-shuffled and a withheld pass, so that a gain from the line's location is separated from a gain from its presence.

**Cost (ESTIMATED from measured rates: 51.2-52 s/step on the L40, 1.35 s per navtest token on the dev box).**
* L40: 57.2 L40-h (reusing C0) or 85.8 L40-h (own control), plus ≤ 2 L40-h of gates.
* Dev-box GPU: about 23 h.
* Calendar: the L40 pod is busy with P5-C (~57 h, plus C0r if it runs), so stage 1 starts no earlier than ~3 days out. Stage 0 can start on the freeze.

**Who else is involved.** The REFe session owns the pod, the trainer and its launch-gate profiles. This card asks that session for one thing: a slot after P5-C's last arm, with the trainer wiring landed after that arm so no P5-C record is invalidated. It takes none of that session's jobs, GPUs or files.

**Recommendation: yes.** Stage 0 starts now. Stage 1 runs after P5-C, under launch-gate PASS tokens. Any overrun above 100 L40-h, or any change of arm, dose or labels, comes back to you.

## Q3. If the stop-line arm passes its bar (SL-ADDS), approve the follow-on now, conditionally?

**What it is.**
* **(i) The seed-1 replicate of the stop-line arm:** +28.6 L40-h, plus ~5.6 dev-box GPU-h. A lever claim needs both seeds, because a same-flag replicate showed "separated" differences on 9.5 % of cells on the v7-tiny rig.
* **(ii) The re-rendered sim read** on the scorecard's 4 traffic-light scenarios, stop-line arm against control, with the baseline twice: about 12 thor3-h.
  * Thor3 and the suite belong to the sim session, whose suite is paused today.
  * With 4 scenarios the read can reject the arm, never adopt it.

**Why now.** It prevents a stall between the verdict and the next arm. If the arm does not reach SL-ADDS, nothing in Q3 runs and nothing is spent.

**Recommendation: yes, conditionally.** (i) runs only on SL-ADDS at seed 0. (ii) runs only if both seeds agree, and only once the sim session has the suite running. Any cost above 35 L40-h or 15 thor3-h comes back to you.
