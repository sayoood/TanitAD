# WM-PHYSICS PROOF BATTERY — does the world model learn the physical world, and the RIGHT part of it?

**Drafted 2026-08-10 (PI question, verbatim: "how can prove that wm incl. encoder and predictor
is learning correctly the physical world and predicting the right relevant part of the physical
world?"). Probes P1–P7, each pre-registered with its gate BEFORE running. All are T0/T1-computable
with existing corpora — no new data, no T2 provisioning. Existing assets folded in, not
re-invented: MPC_WM_DESIGN.md E0 (decodability probe), stage-A L_ctrl design, §1.10
swap-latents vision attribution, v5f imagination probes, `obstacle.offline` 3D agent tracks
(97.44 % of corpus).**

The question decomposes into three falsifiable properties:
- **(A) Physical correctness** — predicted futures obey the physics that generated the data.
- **(B) Relevance** — the latent carries the variables driving needs, and NOT nuisance detail.
- **(C) Causal grounding** — the prediction responds to ACTIONS the way the physical system
  would (the property T0 evals cannot see; §1.12 proved how expensive that blindness is).

## The probes

**P1 — Physical-state decodability curve (A).** Linear probes on the PREDICTED latent ẑ_{t+k}
(k = 5, 10, 15, 20) for: ego speed, yaw rate, path curvature, and lead-vehicle gap/TTC (labels
from `obstacle.offline`). Gate: probe R² at k=10 within 15 % of the same probe on the ENCODED
true frame z_{t+k}; error grows monotonically and smoothly with k (no cliff). *Instrument: E0
probe (MPC_WM_DESIGN) generalised across horizons. A WM that "predicts pixels" but loses TTC
fails here; one that carries the driving state passes.*

**P2 — Nuisance non-retention (B, the "right part" half nobody tests).** Same probe protocol,
but for variables a DRIVING abstraction should discard: clip identity (N-way), static background
texture (patch-shuffled control), weather/exposure class. Gate: clip-ID decodability from ẑ_{t+k}
DROPS with k and sits far below chance-adjusted memorisation ceiling; if clip-ID stays highly
decodable at k=20, the predictor is carrying appearance, not dynamics — FAIL with class
"appearance shortcut". *(Together, P1↑ and P2↓ are the information-bottleneck signature of
learning the RELEVANT physical state.)*

**P3 — Counterfactual action-response sign map (C).** Roll the predictor under swapped/held/
amplified actions (the §1.12 battery, but scored on the WM roll itself, not the readout):
left-steer counterfactual ⇒ decoded ego displacement moves LEFT of the factual roll, brake ⇒
gap to lead GROWS, throttle ⇒ shrinks. Gate: sign-correctness ≥ 95 % of windows per channel;
response GAIN within [0.5×, 2×] of the unicycle-analytic prediction at 1 s. *This is stage-A/W3's
probe pack, promoted to a named proof. A WM failing P3 does not model the world — it replays it.*

**P4 — Agent permanence under occlusion (A+B, v5f-specific strength).** Using `obstacle.offline`
tracks: select windows where an agent visible at t is occluded at t+k but present (track
continues). Probe ẑ_{t+k} for the occluded agent's position. Gate: position error of the
occluded-agent probe < 2× the visible-agent probe error at the same k; both far better than a
"agent vanished" null. *Object permanence is the sharpest test that the model predicts the
WORLD and not the IMAGE — an image predictor has nothing to carry the hidden agent with.*

**P5 — Compounding-error boundedness (A, integral test).** T1 rollout ADE growth rate vs
horizon compared against CV and unicycle-extrapolation baselines. Gate: v5.8f's T1 error curve
stays below the CV curve through 2 s and its LOG-slope does not exceed the baselines' (no
super-linear compounding — the signature of physically inconsistent predictions feeding back).

**P6 — Ego/scene factorisation (B+C).** Perturb the action; measure latent change energy inside
the ego-dynamics subspace (top PCs of action-induced variation) vs outside it. Gate: ≥ 80 % of
action-induced latent change lives in a ≤ 32-dim subspace, and scene probes (P1 lead-gap on a
STATIONARY lead) are invariant (< 5 % shift) under ego-action perturbation. *The L_scene design
made measurable: actions must move the ego through the world, not repaint the world.*

**P7 — Uncertainty calibration of the fan (B).** Rank correlation between fan spread
(per-window candidate dispersion / selector entropy) and realised GT error. Gate: Spearman
ρ ≥ 0.3 with the CI excluding 0. *A model that knows WHAT it cannot predict is evidence the
rest is not accident.*

## P8/P9 — DECODING THE ENVIRONMENT PART OF THE LATENT (PI ask 2026-08-10: "not only the
ego dynamics but also the environment related part")

**P8 — BEV occupancy readout (the environment made VISIBLE).** Train a small frozen-latent
decoder `z → BEV occupancy raster` (ego-frame, 60×32 m grid, agents rasterised from
`obstacle.offline` cuboids; ~1M params, trunk NEVER updated — a readout, so it measures what
the latent already carries, the §1.10 latents-only discipline). Then apply it to PREDICTED
latents ẑ_{t+k}: `decode(ẑ_{t+k})` vs the GT raster at t+k IS the picture of "what the WM
believes the world will look like". Metrics: occupancy IoU and per-agent position error vs
horizon k, split visible/occluded (the P4 join reused). Gates: (a) IoU(ẑ_{t+k}) ≥ 0.8 ×
IoU(z_{t+k}) at k=10 (prediction retains the scene, not just ego); (b) occluded-agent
positional error < 2× visible (permanence, same gate as P4 but now VISUALISED). Deliverable
includes a reel: camera | decoded-BEV-from-ẑ | GT-BEV, side by side over the horizon.

**P9 — probe-gradient saliency (WHERE in the image the environment knowledge comes from).**
For each P1/P8 probe output (lead gap, occupancy cell, curvature), backprop through predictor
+ encoder to the input frames; render the saliency as a camera overlay. Sanity gates
(qualitative, pre-stated): lead-gap saliency concentrates on the lead vehicle region;
curvature saliency on lane/road geometry; if saliency is diffuse or sits on sky/hood, the
probe is reading a shortcut — flag, don't narrate around it. Cheap (one backward per probe),
runs on the P8 harness.

## I4 — VALIDATING AND LEVERAGING THE IMAGINATION CHANNEL (PI ask 2026-08-11: "how can we
validate and leverage the imagination capability included in v5f and thus in v5.8f?")

v5f trains with `--cond-imagination`; the trunk carries an imagination-conditioned rollout
path. Whether that channel CONTRIBUTES (vs rides along) has never been isolated. Three
probes, pre-registered:

**I4a — Imagination ablation attribution (mandatory FIRST — the attribution gate).**
Re-run the banked 881-grid eval three ways on the SAME checkpoint: (i) intact, (ii)
imagination input ZEROED, (iii) imagination input SHUFFLED across windows (keeps marginal
statistics, breaks correspondence). Gate: if intact vs zeroed differ by <5 % on every
family, the channel is inert — every "imagination" claim is retired until a training-side
fix lands (result class: rider, not driver). If zeroed degrades ≥5 % on any family, the
delta IS the measured imagination contribution, quoted per family with cluster CIs. ~1
GPU-h off existing dumps + one eval pass. *Same admissibility family as the nav-echo test:
an input's value is what breaks when you break it, not what the architecture diagram says.*

**I4b — Occluded-split prediction (imagination's PHYSICAL test; rides P4/P8, 0 extra GPU).**
The P4/P8 occluded-agent split IS the imagination testbed: predicting an agent the pixels
cannot see is exactly what an imagination channel is FOR. Report the P4 and P8-occluded
gates stratified intact-vs-zeroed (from I4a's runs): if the occluded-split error is where
the zeroed arm loses most, imagination is doing object permanence — the strongest possible
validation, and it is VISIBLE in the P8 reel.

**I4c — Occlusion-stress windows (targeted, only if I4a passes).** Mine the val corpus for
the ~top-100 hardest occlusion windows (lead agent occluded ≥5 of 20 future steps, from the
P8 join's occlusion flags); run the I4a triplet there. Gate: imagination delta on stress
windows ≥ 2× the full-grid delta (the channel concentrates where imagination is needed).

**Leverage paths (run in this order, each gated on the previous):** (1) imagination-closed
W7 roll-cost — ALREADY LIVE (`w7_gate*.json` roll.imagination_closed=true); its calibration
ρ is I4's downstream consumer. (2) Per-candidate imagination axis: expose the top-8
candidates' imagined consequences as the selection feature (the documented no-candidate-axis
limitation, closed with imagination rather than a learned scorer). (3) φ_tac conditioning:
feed the tactical goal-fan the imagined 1-s state instead of the encoded present (E5.x,
after E4.4's oracle-goal finding).

**Latent-geometry views (supporting, not gated):** PCA/2-D embedding of ẑ coloured by speed,
curvature, lead-gap, road class — the organisation-at-a-glance figure; and counterfactual
latent surgery along probe directions (+gap ⇒ plan relaxes braking) linking interpretation
back to control. Both fall out of the P8 harness for free.

## Discipline

Every probe: 881-window grid (or the labelled subset for P4, n reported), episode-cluster
bootstrap CIs, tier stamp (P1/P2/P4/P6/P7 are T0-diagnostic BY DESIGN — they interrogate
representations; P3/P5 are T1), evidence class MEASURED with artifact paths, verdicts appended
here per probe. A probe that cannot run (missing label, n too small) is reported per-probe with
reason — never silently dropped. Failing gates get a root-cause class in RETRACTION_LOG if they
overturn a standing claim.

## Cost & order

P1/P2 share one probe harness (~2 GPU-h). P3 = W3 stage-A probes (~3 h, already in the ladder).
P4 needs the obstacle.offline join (~1 h prep 0-GPU + 1 GPU-h). P5 falls out of the E1.4 T1
runs. P6 piggybacks P3's perturbation rolls. P7 is 0-GPU off existing dumps (x0_fan_dump.npz has
fan+scores+GT for the OLD fan; rerun on the unicycle fan's eval windows). Order: P7 (tonight,
0-GPU) → P1/P2 harness → P3/P6 (W3) → P4 → P5 (with E1.4).

- [x] **P7 RUN 2026-08-10 — GATE PASSED on the v5f-30k original fan (first probe of the
  battery):** endpoint-dispersion Spearman ρ = **0.4915**, selector-entropy ρ = **0.3954**
  (gate ≥ 0.3), permutation p ≈ 0, n = 881. The fan's uncertainty is calibrated — where it
  spreads, it errs. Caveat carried: dump lacks episode ids → permutation p, not the cluster
  CI; registry-grade rerun binds to the v5.8f eval windows. Artifacts: `p7_calibration.json`
  (+ pod5 authoritative copy), `tools_p7_calibration.py`.
- [x] **P7 REGISTRY-GRADE RERUN 2026-08-10 ~23:35Z (eid-clustered CIs, v5.8f arms) — BOTH
  v5.8f ARMS FAIL:** frozen-argmax scores on the unicycle fan ρ = 0.2622 [0.091, 0.410]
  (positive but below gate — miscalibrated by the re-parameterisation, not dead); the W4b
  rescorer ρ = **0.0542 [−0.140, 0.239]** with near-uniform entropy 5.41 — **no uncertainty
  information at all**, consistent with the memorisation verdict. The original pass holds
  ONLY for the v5f fan+selector pairing. Consequence: v5.8f's calibrated-uncertainty
  property must be restored by W4c (if it learns real scores) or carried by W7's roll-based
  costs — P7 re-runs on whichever selection mechanism ships. Artifact: `p7_regrade.json`.
- [x] **P3/P6 (=W3) RUN 2026-08-11 ~01:20Z — P3 FAIL with load-bearing structure, P6 PASS
  decisively** (artifact `w3_gate.json`, 881 grid, tier T1-diagnostic): LATERAL sign
  99.5 %/99.2 % (gate ≥95 % ✓) but **gain median 0.27/0.23 vs [0.5, 2.0] — the WM turns the
  RIGHT WAY at ~¼ the physical magnitude** (the §1.12 near-straight closed-loop driving,
  now mechanistic); LONGITUDINAL sign only 74.5 %/78.7 % ✗. **P6: action-induced latent
  change lives in a 3-DIM subspace** (gate ≤32) with 0.94/0.91 lateral energy — the ego
  interface is real, low-rank, correctly signed, and MUFFLED. ⇒ stage-A post-training
  (L_ctrl) now has a measured target: raise lateral gain into [0.5, 2], fix longitudinal
  sign to ≥95 %, preserve the 3-dim factorisation. E1.4 T1 rows for v5.8f re-run after.
- [x] P1/P2 harness built (committed 3eed42f; RUNNING on pod5 now) · [x] P4 join built
  (39/40 eps, 195,805 boxes) · [x] P5 instrument validated (E1.4 byte-close PASSED 713b9d1)
- [x] **P1 LEAD-GAP RERUN WITH THE VEHICLE-CLASS FILTER — MEASURED 2026-08-11 ~16:15Z
  (pod4, 881 grid, filter provenance stamped in-artifact): THE CLASS FILTER DID NOT
  DISSOLVE THE FAILURE.** With vehicle-only lead candidates (n(lead) ≈ 266/881 per k,
  ~557 labelled-clear), the ENCODED-latent lead-gap probe still reads **R²(enc) ≤ 0 at
  k=10** — retention undefined, per the artifact's own reason string ("the encoded-latent
  probe itself failed; fix the probe/target before gating the predictor"). Control that the
  instrument works elsewhere, same run: speed R²(pred) 0.993 / (enc) 0.744; two further
  driving targets 0.759/0.551 and 0.865/0.728 (pred/enc). ⇒ the class-agnostic join was a
  REAL defect but not the root cause here. Next discriminating step (cheap, pre-registered
  here before running): (a) TARGET TRANSFORM — probe log-gap, inverse-gap and TTC instead
  of metres (a latent plausibly codes nearness, not metres); (b) CAPABILITY CEILING — a
  2-layer MLP probe on the encoded latent as the non-linearity control: if even the MLP
  stays ≤ 0, the latent genuinely lacks lead distance and that becomes a MODEL verdict
  (P1 partial-fail, class "missing state variable") with direct consequences for the
  LONGITUDINAL family's headway/TTC instruments. Artifact: `p12_gate_clsfilter.json`
  (pod4 `p1-rerun-clsfilter/`; probe arrays banked beside it).
- [x] **P1 LEAD FOLLOW-UP RUN 2026-08-11 ~17:15Z — H-ABSENT SUPPORTED (MODEL VERDICT,
  class "missing state variable").** All three transform probes (log1p / inverse /
  TTC-proxy) AND the 2-layer-MLP capability ceiling read nothing from the ENCODED latent
  at k=10 (MLP R² **−0.334**; linear transforms −5.7 to −154, all far below the 0.30
  gate) — episode-disjoint OOF on the same 266-lead-window subset. *(Stated caveat:
  n=266 against a high-dim latent limits linear-probe power — the wildly negative linear
  values are overfit symptoms — but the regularised MLP agreeing at −0.33 and the parent
  run's ≤0 make the direction unambiguous; and the aux-readout lever below is justified
  under EITHER reading, since it shapes the representation as well as reads it.)* ⇒ **v5f's latent does not carry a readable
  lead-distance variable, in any parameterization, at any probe capacity tested.** This is
  load-bearing for the programme: 88.7 % of the oracle gap is LONGITUDINAL, and the state
  variable longitudinal control most needs is the one the WM cannot surface. Pre-registered
  consequences now ACTIVE: (1) headway/TTC remain GT-join instruments (never latent
  probes); (2) ⚠️ RETRACTED SAME DAY (PI): the auxiliary lead-readout loss — labels into
  the trunk break the JEPA self-supervised thesis; replaced by the LABEL-FREE program in
  `JEPA_PHYSICS_SURVEY.md` (LF0 locate-first on pre-pool spatial tokens, then
  interaction-weighted sampling / masked-latent objectives / dense near-field loss
  shaping, all gated on this same frozen lead battery);
  (3) P8's decoded-BEV lead read-off is the convergent test — if the occupancy decode
  (attempt-2) shows the lead vehicle, the information enters the latent but dies before
  the pooled readout, which localises the defect to the readout path rather than the
  encoder. Artifact: `p1_lead_transforms.json` (+ instrument `p1_lead_transforms.py`,
  4 CPU tests).
