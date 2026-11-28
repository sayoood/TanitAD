# PI rulings, 2026-09-05 — recorded by the Master Mind in the same turn

`Source: the PI's message of 2026-09-05 (morning, Europe/Berlin), quoted verbatim where it
binds. Each ruling names the decision it supersedes in
Project Steering/Decisions/2026-09-04-refcv4b-decisions.md and the stream that carries it.`

## R1. The route metric is a COMPLIANCE metric, not a reproduction metric — supersedes D-1

**PI, verbatim:** *"we are not evaluating the nav command itself, we are evaluating the fact
that the model is following the nav command in consistency to the strategic goals."*

⇒ The question is not "does the strategic head emit the token it was fed" (unfalsifiable —
refcv3's 141/141 bijection). The question is **"does the model's BEHAVIOUR follow the
command, and is that behaviour consistent with the strategic goal it set"**. That is
falsifiable: a model given *turn left* that drives straight fails it, and a model that
ignores nav shows NO drop under a nav-shuffle control. The metric therefore carries two
intervention arms (nav-shuffle, nav-zero) as admissibility conditions, and reports plan-
compliance and `g_str`-compliance separately so a seam failure is not billed as a nav
failure. Stream: *nav-compliance metric* (Benchmarks & Eval), which also registers it as
a criterion.

## R2. The hierarchy claim for refcv4b IS to be tested — after training — supersedes D-2

**PI, verbatim:** *"why we can not prove the claim of the hierarchy in refcv4, let it do it
after finished training."*

⇒ My statement that refcv4b "cannot test the hierarchy" was too narrow. What it cannot test
is the core route READOUT (dangling, `graft_route` False). What it CAN test, post-training,
by eval-time ablation on the same windows: H19 (tactical → anchor prior) on/off; E7
`target_latent` and E9 on/off; nav-zero / nav-shuffle; `g_str`-zero — each paired, four
families + compliance + echo gate, T1. Pre-registered with both outcomes before the
checkpoint lands. Stream: *nav-compliance metric* (same package, `PREREG_REFCV4B_HIERARCHY_EVAL.md`).

## R3. The withheld-row anchor bank: research first, then decide — sharpens D-4

**PI, verbatim:** *"do research how it is done in typical similar successful works and the
original paper of refc."*

⇒ Two streams: the LITERATURE (DiffusionDrive v1 as REF-C's origin, plus TransFuser,
PlanTF, PLUTO, DRAMA, Hydra-MDP, UniAD, VAD, ParaDrive, GenAD — what ego state enters,
whether it is dropped, how anchors are speed-conditioned, with the code read, not only
the paper) and the MEASUREMENT (is the 7.59 m geometric burden covered by the offset head
on the live checkpoint, split kept/withheld, with smoothness in the programme's units).
The live run is NOT touched by either.

## R4. Agent fan-out is NOT capped — overrules D-5

**PI, verbatim:** *"dont cap agent fan."*

⇒ D-5's default of four concurrent agents is withdrawn. Nine streams launched this
morning. ⚠️ Recorded plainly: the weekly API allowance is an ACCOUNT constraint I cannot
override — when it is exhausted, agents die mid-turn (four did on 2026-09-04 evening) and
their unbanked analysis is lost. The mitigation that remains available to me is banking
incrementally, which every brief now demands; the exposure itself is accepted by this ruling.

## R5. Strategic supervision = the 30 s label; nav = conditioning input — confirms D-7

**PI, verbatim:** *"Use the 30 s label as the strategic supervision; keep nav as a
conditioning input."*

⇒ E19 (the 30 s strategic heads, `STRATEGIC_S = (8, 30)`) becomes the strategic
supervision target for refcv5; nav remains an input (E20 already ON in refcv4b; E15/E16/
E17/E18/E21 are the refcv5 delta). The 94.6 %-`follow` corpus limit on nav
(`refb_labels.py:13-14`: 15 s minimum future, 25 s horizon, 19.9 s episodes) is thereby
routed around rather than fought.

## R6. DiffusionDriveV2 — analysis and implementation audit commissioned

**PI, verbatim:** *"analyze again the successor of the original paper: DiffusionDriveV2 …
check any differences, benefit from their results, consistency of implementation in the
common parts, potential improvements, review if we implemented all important parts like
the truncated diffusion decoder, the generation of multimodal trajectories, transfuser,
etc.. did we implemented the correct diffusion mechanisms. Review both v1 and v2
diffusiondrive papers, did we implement the cross attention mechanism?"*

⇒ Two streams: (a) the authoritative V2 paper + repo analysis with primaries banked;
(b) the component-by-component audit of REF-C against v1 and v2 — IMPLEMENTED / PARTIAL /
MISSING / DIVERGED with `file:line` both sides, plus a numerical check that our decoder
actually denoises. The cross-attention question is answered directly in (b).

## R7. REF-C RL experiment — locate the FlyWheel's prep, assess, prepare; launch is mine

**PI, verbatim:** *"can we run the rl experiments of refc to see the effect, the training
fly wheel agent prepared this in the past."*

⇒ Stream: find what was prepared (three locations minimum), assess readiness honestly,
name the reward and its echo-trap exposure (we have no NAVSIM; a reward read from the GT
future teaches reproduction of the GT future), prepare a pre-registered one-variable run
on a frozen base with a deliberate-regression arm, verify the launch script imports, and
escalate the cost. ⛔ No launch without the Master Mind's call.

## Status corrections owed with these rulings

- **refav1 eval after the `ccos` fix: NOT performed as of this ruling.** The code is in
  HEAD (`cee5d99`, 18 tests); the measurement and the eval died with the agent. Re-commissioned
  today as a single stream; the PI's question is answered "no, and here is when it becomes
  yes".
- Four streams lost on 2026-09-04 (ego-dropout burden, pod currency audit, units/retraction,
  ccos measurement) are all re-launched.
