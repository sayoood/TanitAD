# ALPS-4B AD_TRANSFER_RESEARCH v1.1 (2026-07-07) — delta analysis & adoption

Source: `Ressources/AD_TRANSFER_RESEARCH.md` (v1.1, supersedes the v1.0 ingested at kickoff).
The v1.1 additions are fresh measured results from the ALPS-4B testbed. Every item below is a DELTA
against what our plan already incorporated. Adoption decisions: D-017.

## The new measured assets and what they change for TanitAD

### A11 — Egocentric observation makes control learnable (0.69/0.76 vs 0.19 top-down)
The toy's hardest failure (predictor ignoring the action, direction_acc 0.19) **dissolved** when the
observation became egocentric — every action scrolls the whole field, so the action is the dominant
predictable change. This retracts v1.0's "egocentric Two-Rooms failed" (that verdict used the wrong
metric, pre-instrument-fix).
**For us: strong confirmation, zero code change.** Our entire pipeline is egocentric front-camera by
construction (D-009/D-015) — the toy's worst regime is our default. Adopted rule for the CARLA-on-pod
work: **control gates run on the egocentric camera track only; BEV is the planning-isolation track**
(D5/D6 topology) — never gate control on top-down.

### A13 — Action-discrimination ≠ imagination fidelity ⇒ **I4 is demoted from gate to diagnostic**
Control was USABLE at `imag rel = 1.27` (raw imagination worse than persistence) because selection
reads the *action contrast in decoded-state space*, not the full future. Driving's horizon always
imports unseen content, so `imag rel` may stay > 1 forever without blocking maneuver selection.
**For us, operationally important right now:** when p0-sB01 finishes, a final I4 > 1 does NOT by
itself condemn the run. The valid readings:
- I4 > 1 **with healthy geometry** (erank high, dim_std ~1, step_ratio high) → normal for driving;
  judge by probes + action discrimination (D2).
- I4 ≫ 1 **with collapsed geometry** (the F-2 pattern: erank 23, dim_std 1e-3) → collapse; the
  geometry rows are the collapse instrument, not I4 alone.
The binding Stage-0/B control gate becomes **D2: calibrated direction_acc > 0.7 OR forward-dynamics
direction_acc > 0.7 (P4)**, with the I1 oracle row first. (Note: v1.1's §5.2 still lists the old
"imag rel < 1" wording — internal inconsistency; the changelog + §3.5.4 + the new D2 definition are
the operative doctrine, adopted as such.)

### P4 — Forward-dynamics probe: a new first-class control readout
Frozen ridge `g(decoded_state, action) → next decoded_state`, rank actions by predicted next state.
**Strongest readout in the egocentric run (0.76) and cheaper than P1** (low-D → low-D, no per-action
predictor imagination; cannot overfit the high-D latent). P2 (pure latent goal-matching, 0.24) is
confirmed weakest/latest — never gate on it early.
**For us:** the pending gate-runner intake package (2026-07-14) must add P4 alongside P1/P3 before
integration — recorded as an integration condition. P4 also becomes the fast-path operative
controller candidate between imaginations (redundancy channel for the safety case, H11-relevant).

### A12 — The object-binding laws: slots are a real-video technique, scheduled accordingly
Appearance/recon-driven slot binding provably cannot discover a small mover in a low-entropy scene
(two scale-free laws: consequence-dominance at the binding level; the decomposition threshold). Both
laws INVERT on real driving video (high scene entropy forces decomposition; ego-motion makes movers
carry dominant loss mass). **For us:** the object-centric branch (relevant to object-level LOPS/H15
and the H13 behavior heads) is scheduled for **after** the world model trains on real video —
Phase 1, with the two laws as its go/no-go criterion. No toy-scale slot experiments, ever.

### I7 — Task-identity assertion (new doctrine item; the contamination-bug lesson)
A silent env-flag bug made ALPS fit probes on one environment and run control in another for weeks —
invisible to every downstream metric. AD analog: a probe fit on one camera's intrinsics/frame-rate/
action convention silently applied to another stream. **Adopted mechanically:** every corpus loader
now exports a `CORPUS_META` fingerprint (channels, input size, effective focal after D-016
canonicalization, Hz, action convention); the I7 check asserts fit-set and eval-set fingerprints are
IDENTICAL before any probe result counts. This composes with D-016: canonicalization makes corpora
compatible, I7 *proves* per-run that they are.

### I8 — Inference-memory reality (batch-1 profiling at the finest shipped config)
Predictor attention is O((W·N)²); envelopes tuned at coarse configs OOM at fine ones — our own F-5
was this lesson at training time. **Adopted:** the efficiency ledger requires a batch-1 streaming
memory+latency profile at the finest intended config (256 px now; multi-camera later) per checkpoint.

## Consequences applied

1. D-017 logged (adoptions above).
2. Phase 0 Plan §4: D2 gate redefined (calibrated OR P4 > 0.7; imag-rel = diagnostic); I4 row
   reworded in the instrument list.
3. `stack`: I4 docstring reframed (diagnostic + geometry-paired collapse signal); `CORPUS_META`
   fingerprints in comma2k19/physicalai loaders + `i7_task_identity` check + tests.
4. Gate-runner intake: integration conditioned on adding P4 + wiring I7 (noted for triage).
5. Hypothesis ledger: H3 evidence strengthened (A11 = consequence-dominance validated in the driving
   observation model); H11 gains P4 as a redundancy channel; H15/H13 object-centric branch scheduled
   per A12.
6. p0-sB01 (in flight): training losses unaffected; **evaluation practice updated** — final I4 read
   as diagnostic next to geometry rows; checkpoint judged by D1–D3 with the new D2 definition.
