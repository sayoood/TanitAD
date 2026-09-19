<title>LEDGER B7 — diffusion models and flow matching</title>

# LEDGER B7 — diffusion models + flow matching

`APPEND-ONLY. Created 2026-09-13 (LAB-RUN-012).`

## 2026-09-13-00 — ⛔ DATED NOTE: this file did not exist although TRACKS.md linked it

`TRACKS.md` (seventh pass, 2026-09-10) links `LEDGER_B7_diffusion_flow.md` for B7; the file was never written — the **FS-6 class again** (a link written in the same turn as the intention). The 09-10 scan items it would have held, reconstructed from TRACKS.md: FlowR2A `2606.24231` (banked, unread), WAM-Flow `2512.06112` (banked, unread), FlowDrive `2509.21961`, GoalFlow PDMS 90.3 (RELAYED).

## 2026-09-13-01 — scan

`2609.04921` *One Diffusion Model, Two Roles* — a diffusion-transformer decoder as both ego planner and safety-critical scenario generator with training-free guidance (DAPSE) — **scan only, unread**. No B7 deep read this pass. FlowR2A remains the standing FS9-5 candidate.


---

## 2026-09-17 (LAB-RUN-014) — the first NUMBER attaching flow matching to a driving planner's quality, from an A-band paper

⭐ **WA-JEPA `2608.20974` ablation (Tab. 4b-c): replacing deterministic regression with conditional
flow matching over latent futures is worth **+1.0 EPDMS**, against **+0.4** for its masking change.**
This track has carried flow-matching *methods* since 08-31 without a controlled number separating
the generative objective from everything else shipped alongside it; this is that number.

⛔ **Stamp: `navtest`** (WA-JEPA 91.7 · Discrete-WAM 90.4 · SparseDriveV2 90.1), **not navhard**.
⛔ No parameter counts published ⇒ not quotable as a matched-params result.

⭐ **Our position.** B7's transfer question is *"multi-modal futures without the blurry mean"*. Today
that question acquires a mechanism on our side: a **deterministic regressor against a stochastic
future converges to the conditional mean**, and a conditional mean **cannot be action-sensitive** —
which is a candidate explanation for our measured h=1 action/scene ratio of **0.004**. The blurry
mean is not merely a visual defect; it may be the action-insensitivity defect wearing another name.
⛔ Candidate, not established: nothing here measures our predictor.

⚠️ **Honesty on this track's depth today:** B7 was **scanned**, and its DEEP content arrives through
an **A2 paper's ablation** rather than a dedicated B7 primary. Counted as a partial Band-B deep-read
in the day's completeness statement, not as a clean one.

**Scan, same pass:** FlowR2A `2606.24231` (⛔ **still banked-unread** — pre-committed rotation item 4
from 09-13, missed again) · WAM-Flow `2512.06112` (banked, unread) · GuideFlow `2511.18729` ·
MeanFuser `2602.20060` (one-step multi-modal trajectory generation) · Adaptive Time Step Flow
Matching `2602.10285` · FlowDrive `2509.21961` · `2606.09962` (optimality of FSQ tokens for
continuous diffusion — a B7/B8 crossover).

→ `Frontier Scan/Daily/2026-09-17/RESULT.md` FS17-2

## 2026-09-18-01 — FlowR2A read in full (rotation debt, missed twice): reward-as-condition, and the reward LEAK it had to fix

`FULL TEXT (local pypdf)` · arXiv **2606.24231**. `p(a|r)` by flow matching on dense (trajectory, reward) pairs. navtest v1 **92.8 PDMS**, navtest v2 **88.9 EPDMS** (⛔ navtest one-stage). Per-timestep reward conditioning TTC **88.8 → 94.9**; reward-noise σ 0 → 0.05: PDMS **84.8 → 89.4** — without noise *"the decoder treats rewards as trajectory identifiers"*. Latency 91.3 ms on an H20 (K = 10: 53.2 ms, −0.6 PDMS).
**Our position:** a conditioning signal that identifies its target is a leak (our nav-echo, in a reward costume) ⇒ noise-on-condition arms by default. Blocked for us by the reward oracle: no map ⇒ TTC/collision-only reward from `obstacle.offline` (FS18-4 prices it).
