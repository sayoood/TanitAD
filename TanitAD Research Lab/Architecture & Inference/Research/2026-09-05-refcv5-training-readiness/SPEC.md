# SPEC — refcv5 training-readiness validation (`V-RC5-READY`)

**Written and banked BEFORE any arm ran.** Both outcomes are committed below.
Stream: Architecture & Inference · 2026-09-05 · branch `agent/arch-inf-20260803`
Rig: **v7-tiny** (CPU, synthetic episodes, `--smoke`). ⛔ No GPU is used; the
dev-box 4060 is saturated by refav1 and `tanitad-refcv3` is training refcv4b.

---

## The question

**Would a refcv5 arm that trains, converges and writes a checkpoint be caught
if one of its stamped seams were never supervised?**

This is not hypothetical. The refcv5 stream has already banked **two** arms
that did exactly that:

1. `--agents head --w-agent 1.0` on a corpus with no join — trained, converged,
   stamped `w_agent: 1.0`, **detector never supervised**.
2. `--agent-w-project 0.2` with `model._rig_camera = None` — stamped, computed
   nothing (M18).

Both were closed by **flag guards**. A flag guard cannot see a third case,
which is what this validation exists to find: a flag that is set correctly, a
camera that is really built, a term that really runs — **and a gradient of
zero**.

---

## Arms

| id | argv (beyond `--arm hier --smoke --device cpu --synth-episodes 4`) | role |
|---|---|---|
| **A0** | `--steps 3` | baseline, every seam off |
| **A0r** | `--steps 3` (identical flags, identical seed, run again) | ⭐ **replicate control** — the rig's own run-to-run noise floor |
| **A1** | `--steps 3 --agents oracle --agent-queries 8` | the agent-token seam ON and supervised |
| **R1** | `--agents oracle --agent-rig-camera nominal --agent-w-ground 0.5` | ⛔ **DELIBERATE REGRESSION** — a stamped weight whose term has no gradient |
| **R2** | `--agents head --w-agent 1.0` (no `--agent-join`) | ⛔ deliberate regression — labels absent |
| **R3** | `--agents off --w-agent 1.0 --agent-join <join>` | ⛔ deliberate regression — seam absent, weight stamped |
| **R4** | `--sampler ddim --w-u0 0` | ⛔ deliberate regression — denoiser with no loss on it |
| **C1** | `--agents oracle --agent-rig-camera nominal --agent-w-project 0.2` | ⚠️ **converse control** — a LIVE monocular term with the same camera must NOT be refused |

## Gate

`stack/scripts/refcv5_preflight.py` plus the trainer's own startup refusals.

**PASS** requires **all** of:

* **G1** every one of **R1–R4** is REFUSED before a checkpoint exists, and the
  refusal message names the defect;
* **G2** **C1** is **not** refused — the gate fires on the dead configuration
  only, not on the camera or the seam;
* **G3** **A0, A0r, A1** all reach `summary.json`;
* **G4** `A0` and `A0r` produce **byte-identical** `ckpt.pt` — the replicate
  control. If two identical runs differ, no per-arm difference on this rig
  means anything, and every other reading here is void;
* **G5** the preflight's `every weighted term has a gradient` check reports
  `agent_w_ground` as DEAD **and** `agent_w_project` as LIVE — a probe where
  every term reads flat is INCONCLUSIVE, not a finding.

---

## ⛔ BOTH OUTCOMES, COMMITTED IN ADVANCE

**If the gate PASSES** — refcv5's seams are protected against the
never-supervised class, `--agent-w-ground` is a MEASURED dead term and is
refused rather than silently stamped, and the remaining blockers to a real arm
are DATA and COMPUTE, not code. The verdict written in RESULT.md is:
*"the code is ready; here is exactly what is missing."*

**If the gate FAILS** — I state which regression slipped through and I do
**not** report refcv5 as ready. Specifically:

* if **any of R1–R4 reaches a checkpoint**, refcv5 can still manufacture a
  refutation and the readiness claim is withdrawn until that guard exists;
* if **C1 is refused**, the guard is over-broad — it would block the live
  projection arm, which is a capability loss dressed as safety, and it must be
  narrowed before it ships;
* if **A0 and A0r differ**, the tiny rig is non-deterministic and **G1–G3 are
  reported as UNVERIFIED** rather than as passes, because on a rig whose
  replicate moves, a refusal that fired once is not evidence it fires;
* if **the gradient probe reads every term flat**, the result is
  **INCONCLUSIVE** — a broken probe, not a finding about the ground prior —
  and the `--agent-w-ground` refusal is withdrawn until the probe is fixed.

⚠️ **No goalpost moves after seeing data.** A regression that is caught by a
mechanism other than the one specified is recorded as caught *by that
mechanism*, and the gap in the specified one is reported.

---

## What this validation CANNOT say

* It does not roll a real corpus, so it says nothing about whether the agent
  head or the sampler **helps**. It says only that an arm which trains and
  stamps a weight cannot do so with that weight unsupervised.
* The tiny rig is 64×64 synthetic. The monocular terms **cannot run there** —
  a 64 px grey crop has no honest `f_ref` and the trainer refuses to invent
  one — so R1/C1 are driven at the corpus geometry (256×640) through the
  shipped `_build_rig_camera` path, not through the tiny rig's own frame.
* ⛔ A separated CI on this rig is **necessary and not sufficient** for any
  lever claim (`H-ESTIM-SEED-1`). Nothing here claims a lever effect; every
  gate is a refusal or an identity.
