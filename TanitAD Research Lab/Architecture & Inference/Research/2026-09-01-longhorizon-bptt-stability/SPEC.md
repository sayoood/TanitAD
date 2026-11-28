<title>SPEC - long-horizon BPTT stability (ASK-1)</title>

# SPEC - E-ARCH-LHB-1: does any published recipe train a 60-step full-chain BPTT rollout stably?

**Trigger.** `LAB_ASKS.md` **ASK-1** (TrainingFlyWheel, 2026-08-31), OPEN. Answering
asks precedes pulling backlog seeds (LAB_BACKLOG update contract, row "answer asks FIRST").
Also addresses `LAB_BACKLOG` PROPOSED **P-8** (k=60 full-chain BPTT gradient-unstable at
clip 1.0; gnorm median 5.71 -> 2.1e9; killed at step 9,000 on a pre-committed criterion).

**Question, verbatim from ASK-1.** *"Does any published recipe train a 60-step full-chain
BPTT rollout stably, and with what clip/schedule? Our k=60 arm diverged at clip 1.0
(gnorm 5.71 median -> 2.1e9) while k=8 was stable."*

## Hypotheses, both outcomes committed IN ADVANCE

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| H-LHB-1 | A published recipe exists that back-propagates a **full chain** of >=60 latent rollout steps | we adopt its clip rule + schedule directly and re-run k=60 | **no such recipe** -> the k=60 failure is not a bug to fix but a configuration nobody uses; report what IS used instead |
| H-LHB-2 | The published stability mechanism is a **larger/different scalar gradient clip** | tune the clip and re-run | the mechanism is **architectural/normalisational or curricular**, and raising the clip is the wrong lever |

**Method.** Literature only; every cited number read from a PDF banked in
`TanitAD Research Lab/Library/` (no aggregator summaries -> nothing PUBLISHED-SECONDARY).
0 GPU. Reference class = our four closest relatives (latent world models that roll out and
train through the rollout) plus driving-specific long-horizon video world models.

**Falsifier for the headline.** A single primary showing >=60 untruncated backprop steps
through a shared-parameter latent transition, with a stable training curve, refutes the
headline finding. Named searches that returned nothing are recorded in `raw/search_log.md`.
