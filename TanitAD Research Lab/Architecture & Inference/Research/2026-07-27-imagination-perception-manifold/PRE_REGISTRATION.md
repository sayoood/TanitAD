# PRE-REGISTRATION — imagination/perception manifold mismatch, literature review

**Written:** 2026-07-27 (Europe/Berlin), **before** any fix-evaluation literature was read.
**Disclosure of exactly what preceded it (so the mtime claim is honest):** I had already read our own
primary artifacts (`BLIND_IMAGINATION.md`, `TBLIND_LADDER.md`, `stack/tanitad/models/metric_dynamics.py`,
`stack/tanitad/train/flagship_losses.py`) and run **two orienting web searches** to check that the topic
has a literature at all. **No paper on any candidate fix had been read, and no ranking existed.**
This file is not edited after research begins; deviations go in the amendments table of the main report.

**Stream:** Architecture & Inference — research (CPU/web only, no pod touched).
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

---

## 1. The defect, as MEASURED by us — the thing every candidate must be checked against

`MEASURED` · `…/Architecture & Inference/Implementation/incoming/2026-07-26-blind-imagination/BLIND_IMAGINATION.md` §3.5 ·
`artifacts/horizon_curve.json` · v1 = `flagship4b-speedjerk-30k` @ step 29999 · 599 windows / 596 episode
clusters · paired episode-cluster bootstrap B=2000 seed 0.

| what `grounding.step["op"]` is fed | `ade_0_2s` | `de@0.5s` | longitudinal share @0.5 s |
|---|---:|---:|---:|
| `(ẑ_t, ẑ_{t+1})` — **imagined pair**, its training distribution | **0.3839** | 0.065 | 0.937 |
| `(z_t, ẑ_{t+1})` — half real | 0.5167 | 0.122 | — |
| `(z_t, z_{t+1})` — **real pair** | **3.6093** | **1.510** | **1.000** |

**The defect is a 9.4× decode penalty on real latent pairs**, monotone in how real the pair is, **pure
longitudinal**, and **variance not bias** (signed mean 0.118 m vs σ 2.083 m at 0.5 s). It reproduces on
the 20-step readout (−0.029 m at 2 s), so it is a property of the **grounding recipe**, not of one head.

**A candidate fix is only admissible if its published evidence speaks to THAT** — a decoder that cannot
read the observation distribution — **not to "distribution shift" in the abstract.** I will mark any
candidate whose evidence is only about generic compounding error or generic domain shift as
**MECHANISM-PLAUSIBLE, EVIDENCE-ADJACENT**, and rank it below anything with a matched measurement.

## 2. What I will and will not count as evidence

* `PUBLISHED (cited — specific paper)` for every literature claim, with the paper named inline.
* ⚠️ **DEMONSTRATED vs ASSERTED must be separated on every row.** "The KL ties prior and posterior" is
  an assertion about a mechanism; "ablating the KL raises return by X" is a demonstration; "decoding a
  prior latent vs a posterior latent costs Y" is the only kind of demonstration that is *matched* to our
  defect. I expect the third kind to be rare and I will say so if it is.
* Tiers: `PROVISIONAL` (single source or my inference) · `CONFIRMED` (≥2 independent sources, or a
  source + our own code) · `DECISION-GRADE` (CONFIRMED + falsifier stated + cost estimated).
* Where sources disagree, both sides are given. Where I could not establish something, it is
  **UNVERIFIED**, not smoothed over.

## 3. ⛔ PRE-REGISTERED FALSIFIER — what would make me conclude the gap is NOT fixable without a retrain

I commit to this **before** looking, and I commit to saying it plainly if the literature points that way.
I will write **"NOT FIXABLE WITHOUT A RETRAIN"** if **any two** of the following four hold:

* **F1 — no frozen-weight repair exists in the literature.** Every published repair of a prior/posterior
  or imagined/observed decode gap requires a **training-time** term (a KL, a consistency loss, a mixed
  decoder objective, scheduled sampling in latent space). If no paper repairs it by re-fitting only a
  small head on a **frozen** trunk, then a frozen-checkpoint fix has no precedent and our ladder cannot
  start at zero GPU.
* **F2 — the gap is located in the TRUNK, not the head.** If the published diagnosis is that the *encoder*
  produces a latent geometry the *predictor* never visits (i.e. the two occupy disjoint regions of the
  state space), then no head re-fit can bridge them, because the head would have to be two different
  functions on two disjoint supports — and our own §3.5 monotonicity (0.3839 → 0.5167 → 3.6093 in the
  *fraction of the pair that is real*) is consistent with exactly that.
* **F3 — the fix is reported to cost rollout quality.** If tying the decoder to the observation manifold
  is published to *degrade* imagined-rollout accuracy by more than it gains on observed decoding (a real
  trade-off, not a free lunch), then our best current number (`ade_0_2s` 0.1865 with the calibrated
  readout) is at risk and the fix is not a repair, it is a different operating point — which cannot be
  chosen without a retrain to measure both ends.
* **F4 — our own already-measured structure refuses the head hypothesis.** If a zero-GPU probe on our
  own checkpoint shows the **`MetricInverseDynamics` heads — which ARE trained on real pairs**
  (`flagship_losses.grounding_losses` term (a)) — **also fail on real pairs at horizon 1**, then the
  information is not in the latent difference at all, the head is not the locus, and only a retrain that
  changes the *representation* can help. **This is the single most decisive cheap test and I pre-commit
  to naming it as the #1 experiment regardless of what the literature says**, because it discriminates
  F2 from the head hypothesis on our own weights in minutes.

**Conversely, I will write "FIXABLE ON A FROZEN CHECKPOINT" only if:** a repair exists whose published
evidence is matched (not merely adjacent) to a decode-source gap, **and** it is implementable as a
re-fit of ≤ a few M parameters against latents we already cache, **and** it has a falsifier we can run
in hours.

## 4. The ranking rule, fixed in advance

Candidates are ranked by **(expected gap closure) / (GPU-hours to a DECIDING result)**, with ties broken
toward the one whose falsifier is **cheapest and most likely to fire negative**. A fix that needs a
GPU-week to return its first bit ranks below a fix that needs an hour, even if its expected effect is
larger — the program's standing rule (`BOOST_PROGRAM` §3.4, `CLAUDE.md`) is that no GPU-week is
committed until an hours-scale ladder returns.

## 5. What is deliberately out of scope

* No pod is touched (pod1 training, pod2 owed-controls, pod3 classifier build, eval trafficsim).
* **No experiment is launched by this stream.** The deliverable is the ranked ladder and the design of
  its first rung; running it is a separate, owned job.
* I do not re-measure v1 or v4. Every one of our numbers is quoted from a committed artifact with its path.

## 6. Known confound in my own reading, declared up front

Our §3.5 arm (c2) decodes `(z_t, z_{t+1})` at a **1-step** separation, and the `op` readout was trained
on **imagined** 1-step transitions. The two differ in **two** ways at once — *source* (real vs imagined)
**and** *the statistics of the latent difference* (an imagined `ẑ` is a smoothed conditional mean; a real
`z` carries encoder noise and appearance change). ⚠️ **Our measurement therefore does not, by itself,
separate "the decoder cannot read perception" from "the decoder cannot read HIGH-VARIANCE latent
differences of any origin".** The (c2) error being **pure variance** (σ 2.083 m vs bias 0.118 m) is
positive evidence for the second reading. I will look specifically for literature that separates these,
and I will design the cheap experiment so that it separates them too.
