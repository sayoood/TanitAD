<title>A zero-label clip-value signal for H-DATA-1, and its kinematic floor (2026-09-09)</title>

# The clip-value signal H-DATA-1 asks for already exists, is zero-label, and is 30 % constant-velocity

`TanitAD Research Lab - Data Engineering - daily pass 2026-09-09 (LAB-RUN-010).`
`Serves backlog row 22 (H-DATA-1), row 27, the DataFlyWheel moat mechanism, and corrects backlog row 4.`

---

## Findings

### F1 - The signal is the JEPA's own temporal prediction error. No labels, at any stage.

`lib 2606.28383 "Zero-Label Driving Scenario Complexity Detection via JEPA". FULL TEXT. Class PUBLISHED.`

Backlog row 22 asks: *does frozen-feature leverage predict a clip's value for WM training?* This paper
answers the neighbouring question with a simpler signal.

**Score:** `s = || z_hat_tgt - z_tgt ||_2` - the L2 distance between the predicted and the actual target
latent. That is the model's own surprise. Verbatim: *"no labels are used during training or in defining
the surprise score itself."*

**It ranks the scenarios we would want it to rank.** Highest surprise: *"stopping at traffic light
without lead"* (2.008), *"starting unprotected cross turn"* (1.978). Lowest: lane-following and
stationary traffic. Downstream anomaly detection reaches **AP 0.512 against a 0.436 chance baseline**.

⭐ **For us the operational point is that this requires nothing we do not already have**: a trained
predictor and a forward pass. We compute `|| z_hat - z ||` every training step and throw it away.

### F2 - ⛔ THE WARNING, AND IT IS THE MORE IMPORTANT HALF: 0.314 of the signal is constant velocity

Their four ablations are exactly the control panel `CLAUDE.md` mandates:

| ablation | Spearman rho | reading |
|---|---|---|
| shuffled scores | **-0.126** | the no-information value, read correctly |
| **random encoder** | **0.024** | the untrained floor reads approximately zero |
| ⛔ **constant-velocity baseline** | **0.314** | **a constant-velocity model recovers a large share of "complexity"** |
| **no-EMA training** | **44-fold collapse** | EMA is the mechanism, not a refinement |

⛔⛔ **The constant-velocity row is our CTRV floor in a new costume.** A clip-value score that looks like
learned scene understanding is substantially predictable from kinematics alone. **Any TanitAD clip-value
number reported without a constant-velocity floor beside it is uninterpretable** - and this is the same
family as the ridge-probe failures where the latent, `[z, dz]` and raw pixels all scored +0.54 because
each merely reproduced the mean.

⭐ The paper also notes the constant-velocity baseline has **30 % lower score spread**, which is the
discriminator: the learned score is not *better on average* so much as *more spread out*.

### F3 - It corrects one of our own records: fixed tau = 0.996 DOES have a published operating point

⛔ Backlog row 4 / MM-E1 states: *"our fixed tau=0.996 has no published operating point - every
EMA-teacher line ramps tau."*

`MEASURED from the primary:` the target encoder *"receives no gradients and updates only through
exponential moving average: theta_tgt <- alpha * theta_tgt + (1 - alpha) * theta_ctx with **alpha =
0.996**"* - **fixed, not ramped** - and removing EMA collapses discriminative power **44-fold**.

⚠️ **Scope it honestly, because the temptation to over-read this is exactly the failure class we log.**
This is a **1,289,130-parameter** model on **structured agent-state vectors** from **nuPlan mini**
(1,322 scenarios, 64 SQLite files, 50 timesteps at 10 Hz) - **not pixels, not our corpus, not our
scale.** What is retracted is only the sentence *"no published operating point exists."* **Whether to
ramp tau at v7 scale on video latents is untouched, and row 4's arm still needs to run.**

### F4 - An independent second source says the same thing about WHAT to select on

`lib 2505.24808 RealDrive. Class PUBLISHED, abstract-level, declared.` They *"leverage embeddings derived
from a **planning model** to more effectively capture task-specific scenario similarities"*, explicitly
because prior work *"focused on interaction between agents rather than **decision-relevant**
information."*

⭐ **Two independent lines converge: select on decision-relevance, not on visual or scene statistics.**
A third (Kairos `2606.16533`) reportedly makes the same two-level argument - ⛔ **but its curation
section failed at three retrieval routes today, so that claim is `RELAYED` and decides nothing here.**

---

## Five-dimension analysis - `2606.28383` (the deep-read item)

| dimension | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct lever on backlog row 22 / H-DATA-1 and on the DataFlyWheel moat (MOSAIC-class: 80 % less data at matched performance). Our corpus is parity-locked at 2,376 episodes, so **curation is the free lever** and this is a curation signal that costs one forward pass. |
| **CONSEQUENCE** | If it transfers, clip value becomes a **byproduct of training rather than a separate instrument** - we already compute the quantity. It also supplies row 22's missing control design. And it **retracts one sentence of backlog row 4**. |
| **COMBINATION** | ⭐ Combines with today's A2 finding in a way neither has alone: **`s = ||z_hat - z||` is the numerator-side quantity, and SR (Separation Rate) is the denominator-side one.** MM-1 MEASURED that our h=1 ratio moved mostly through its denominator; a clip-value score built on prediction error alone would inherit exactly that confound - **a clip could score "valuable" because the scene drifted, not because it was informative.** ⇒ **Our port must read surprise AND separation, not surprise alone.** That is a combination our own MM-1 arithmetic forces and neither paper states. |
| **CHANCES / RISKS** | **Upside:** a free, zero-label selection signal on a fixed corpus, compounding on every future corpus. **Risks:** (a) **0.314 of it is constant velocity** - the dominant risk; (b) their validation is against *scenario tags*, and a tag is not training value - the paper itself concedes it *"does not demonstrate whether scores predict downstream training value"*; (c) 1,322 scenarios on agent-state vectors is far from 2,376 episodes of video latents; (d) high surprise may select **unlearnable** clips (sensor noise, occlusion) as readily as informative ones, which is the classic hard-example-mining failure. |
| **EXPERIMENT** | **Pre-registerable, 0 GPU for the scoring half.** Score all parity-train episodes with `s = ||z_hat - z||` from a banked v7 checkpoint. Report **four arms**: the score, a **constant-velocity floor**, a **shuffled-score control** (must read the no-information value), and a **random-encoder floor** (must read approximately zero). ⛔ **Committed in advance: if the constant-velocity floor recovers more than ~0.5 Spearman of the learned ranking, the signal is predominantly kinematic and H-DATA-1 must be re-scoped to ask what the LATENT adds over kinematics - not abandoned, re-aimed.** Only a learned ranking clearly above the kinematic floor earns the **non-parity** leverage-K vs random-K training arm row 22 proposes. |

---

## What this changes for TanitAD - at most three recommendations

1. ⭐⭐ **Re-scope H-DATA-1 (row 22) to score-and-floor before any training arm.** The scoring half is
   free and the floors are mandatory. Running the leverage-K vs random-K arm without a constant-velocity
   floor would produce a number we could not interpret.
2. ⭐⭐ **Score on surprise AND separation, not surprise alone** - the MM-1 denominator argument in F-note
   above. This is our own contribution, not either paper's.
3. ⚠️ **Correct backlog row 4's premise in the same turn** (done below): fixed tau = 0.996 has a
   published operating point; the scale-transfer question is what remains open.

## Limits of this package

`Literature only - no TanitAD measurement was taken.` Every number above is the paper's, on nuPlan mini
agent-state vectors, and **none is a TanitAD number.** Kairos's curation claim is `RELAYED` after three
failed retrieval routes and decides nothing. RealDrive is abstract-level and declared. **The primaries ARE banked** (`kb_add`, rc=0); debt D-12 is discharged.
