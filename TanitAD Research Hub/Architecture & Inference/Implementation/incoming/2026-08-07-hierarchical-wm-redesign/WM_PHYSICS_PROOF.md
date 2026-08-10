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

- [ ] P7 run · [ ] P1/P2 harness built · [ ] P3/P6 (=W3) · [ ] P4 join built · [ ] P5 (=E1.4)
