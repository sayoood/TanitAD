<title>LEDGER A2 — JEPA family, joint-embedding and predictive architectures</title>

# LEDGER A2 — JEPA · joint-embedding · predictive architectures

⛔ **APPEND-ONLY.**

`⭐ CREATED 2026-09-02. TRACKS.md had linked this file since 2026-08-31 without it existing (backlog FS-6,`
`root-cause class: a link written in the same turn as the intention to write the file, never verified).`
`The 2026-08-31 A2 DEEP is recorded below from that pass's RESULT.md as INHERITED, and is marked as such.`

## Our position in this track

**TanitAD's objective family is JEPA.** The v7 predictor learns action-conditioned dynamics in a latent
space rather than in pixels, which is what makes the programme's sub-300M ambition arithmetically possible.

⛔ **And the programme's gating defect lives exactly here.** `V7_LAUNCH_GATE` P1: **no v7 arm has ever
beaten its own hold-action control** — all three arms `LOSES_TO_HOLDV0` on every distance metric, heading
error ~95 deg (chance). P2: **the predictor does not use its actions**; h=1 action/scene ratio **0.004**.
The one positive is that `copy_detector` is **CLEAN** (echo 0.0000) — fake-skill-by-echo was traded for an
honest absence of skill.

⇒ **This track's job is to answer whether that is our bug or the field's.** As of 2026-09-02 the evidence
says: **the field's** — and the field has published fixes.

---

## Entry 2026-08-31-01 — anti-collapse landscape (INHERITED from the 2026-08-31 pass)

⚠️ `INHERITED — recorded here for continuity when this ledger was created on 2026-09-02; NOT re-verified.`

- `2605.09701` **LatentAlign** (full text, that pass): sigmoid-anneal of the prediction target from
  grounded/teacher-forced toward self-predicted. Produced backlog row **P-1**.
- `2602.03604` **EB-JEPA** (abstract-only): energy-based JEPA library, published 97 % Two-Rooms planning
  success. Produced backlog row **P-4** (use it as an external anti-collapse control).

---

## Entry 2026-09-02-01 — ⭐⭐⭐ Delta-JEPA: our P-1/P-2 defect is a documented field failure, with a fix

`PUBLISHED lib 2606.31232 · FULL TEXT READ 2026-09-02 · Zhang et al. (UCAS, IIE-CAS, HIT, IA-CAS) · arXiv 2026-06-30`
`⚠️ Was ALREADY BANKED and unread before this pass — see FS-5.`

### The field-level finding that reframes our gate

⭐⭐⭐ **`LeWorldModel` (`2603.19312`) — a strong, current end-to-end JEPA world model — is measured
ACTION-INSENSITIVE on the same diagnostic TanitAD built for v7.** Delta-JEPA perturbs the action and
measures the predictor's displacement from its own **zero-action** prediction:

> *"Delta-JEPA produces well-separated action-wise mean responses, with larger action magnitudes generally
> inducing larger predicted shifts. In contrast, **LeWM's action-wise means remain concentrated near the
> origin and substantially overlap**, indicating that changing the action does not induce a stable
> directional change in its prediction."*

⇒ **That is our `actdiv` / hold-action control, built independently by another group, and a well-regarded
published model fails it.** TanitAD is not uniquely broken. *(Same reframing as B13's geometry ceiling and
Waymo's long-horizon concession — three in one pass-series.)*

### The mechanism they diagnose (verbatim)

> *"when trained end-to-end with only latent prediction objectives, JEPA-based world models can easily
> collapse to trivial constant representations... the model achieves deceptively low prediction loss while
> destroying the representation structure needed for planning."*

⛔⛔ **The leak argument — this lands on INSTRUMENTS, not architecture:**

> *"its inverse dynamics module decodes actions from concatenated adjacent latent states [z_t, z_t+1].
> Because the forward predictor is itself conditioned on the executed action, end-to-end optimization can
> make the next-state representation z_t+1 absorb action-correlated cues that are easy for the inverse
> decoder to exploit, **without requiring the model to represent the actual transition between the two
> states**."*

⇒ **An action-decodability probe reading concatenated endpoints can pass on leaked endpoint cues.** This is
the same family as the programme's own leak rules (nav-echo bijection scoring 1.0000; label-provenance in
`situations.py`; the action echo). **Any TanitAD probe of this shape is confounded by construction.**

### The method — LDAD

Decode the executed action from the latent **displacement** `Dz_t = z_t+1 - z_t`. Two loss terms total:
`L = L_pred + L_action`. No pixel reconstruction, no distribution-matching regulariser, no frozen encoder,
no stop-gradient branches.

### Numbers (planning success %, 4 visual continuous-control envs, 3 seeds, 50 epochs, lr 5e-5)

| result | value |
|---|---|
| OGB-Cube vs strongest baseline | **+15.14 pp** |
| Two-Room vs PLDM | **+6.27 pp** |
| Push-T vs LeWM | **+4.54 pp** |
| ⭐ Ablation `Dz` vs `concat[z_t,z_t+1]` (Table 2) | **`Dz` wins on all four** — Push-T **+12.60**, Two-Room **+4.07** |
| ⭐ Control: lambda=0 (LDAD removed) | *"nearly collapses, yielding only a negligible planning success rate"* |
| Decoding-target ablation (Table 3) | raw action best; D-joint-position comparable; D-finger-position much worse |
| lambda sensitivity | poor at 0 and 0.1; stable over a broad range; best at **50.0** |

⭐ **The lambda=0 row is a control that must read the no-information value, and does.** That is why this
ablation is admissible under our own probe-panel rule rather than merely suggestive.

### Baselines and their anti-collapse mechanisms (useful map for backlog row 9)

| model | mechanism | Delta-JEPA's critique |
|---|---|---|
| **LeWM** `2603.19312` | SIGReg Gaussian latent regulariser | *"does not explicitly constrain the latent space to be sensitive to executed actions"* |
| **PLDM** | VICReg + inverse dynamics on `[z_t, z_t+1]` | multi-loss, hyperparameter-sensitive; **endpoint-cue leak** |
| **DINO-WM** | frozen DINOv2 features | *"limits task-specific adaptation of the representation"* |
| **Sub-JEPA** `2605.09241` | subspace Gaussian regularisation | strong on Reacher |

### ⚠️ Why this may NOT transfer to us — the risk that decides the line

⛔ **Our action channel IS realised motion (r 0.9988 with realised motion).** Decoding realised motion from
`Dz` may be **near-tautological** in any competent encoder — LDAD could sit near zero loss from
initialisation and buy nothing. **That is gate-problem P-2 branch (b) biting the fix rather than the
model**, and it must be tested before an arm is designed.

Secondary limits: four continuous-control benchmarks, **no driving**; encoder trained **from scratch**
where ours is pretrained/frozen; planning by CEM in a reward-free offline setting.

### Consequences for live rows

| row | movement |
|---|---|
| **P-2** (predictor does not use actions) | branch (c) *representation* gains a **named published mechanism** (endpoint leak), and F4's FiLM entanglement gives it a second. |
| **backlog row 13** (H-RANK-17, `z_t + m_t ~ z_t+1`) | ⭐ same latent-difference family — **now has an external primary with an ablation table.** |
| **backlog row 9** (VICReg placement / anti-collapse three-way) | the baseline map above is directly reusable; PLDM's endpoint decoder is a **design to avoid**, not to copy. |
| **instrument audit (new)** | ⛔ every action-decodability probe must be checked for the concat leak. |

`Next in this track: (1) the 0-GPU LDAD tautology check on banked v7 latents with constant-only and`
`shuffled-action controls; (2) LeWorldModel 2603.19312 and Sub-JEPA 2605.09241 are UNBANKED and unread —`
`both are current anti-collapse primaries and both are owed.`


---

## Entry 2026-09-02-02 — ⛔ LDAD IS AN ENCODER-SHAPING LOSS: the transfer blocker that outranks the tautology risk

`Analysis entry, same day as 2026-09-02-01. Produced while briefing the Master Mind for the v7f design freeze.`
`⚠️ This CORRECTS THE EMPHASIS of the entry above, which named the tautology risk as the decisive one. It is not.`

### The architectural fact, from the paper's own figure

`Dz_t = z_t+1 - z_t` is built from **two ENCODER outputs** — `z_t = f(o_t)` and `z_t+1 = f(o_t+1)` — **not**
from the predictor's output. The predictor path is separate and carries `L_pred` only:

> *"In the forward path, the dynamics predictor forecasts the subsequent representation z_t+1-hat from z_t
> and the action a_t, guided by the prediction loss L_pred. **Concurrently, the Latent Difference Action
> Decoder receives the latent displacement Dz_t to reconstruct the action a_t-hat**, supervised by the
> action loss L_action... the entire framework is optimized end-to-end."*

⇒ **LDAD shapes the LATENT GEOMETRY of the encoder.** The predictor benefits **second-hand**, because its
target space has been made action-structured.

### ⛔ Why this blocks transfer harder than the tautology risk

**Our trunk is pretrained and frozen.** A loss whose entire mechanism is encoder adaptation has **nothing
to shape** on a frozen trunk. The paper is explicit that this is the point, and names our design as the
thing it improves on:

- it lists *"avoiding frozen encoders"* among its virtues;
- on DINO-WM: it *"stabilizes latent dynamics learning by using frozen DINOv2 visual features, but **this
  limits task-specific adaptation of the representation**."*

⇒ **Delta-JEPA is, in part, an argument against the frozen-trunk design** — not merely a bolt-on we can
adopt. Applying LDAD to the **predicted** `Dz-hat` instead would be a **different objective that we would
own and have to justify**, and it would NOT inherit their ablation table.

⚠️ **Ordering of the two transfer risks, corrected:**
1. ⛔ **Frozen encoder** — structural; decides whether the objective has a mechanism at all.
2. ⚠️ **Tautology** — our action channel is realised motion (r 0.9988), so the decode may be free at init.
3. Domain — four continuous-control benchmarks, no driving.

⇒ **Backlog GS-2 is UNDER-SPECIFIED as written**: it tests (2) and not (1). Revised in the same turn.

### ⭐ HYPOTHESIS — this supplies a mechanism for our own o11 anomaly (gate problem P-3)

`HYPOTHESIS. Not established, and the arm that motivates it is confounded four ways.`

`o11p30k` is the **only** TanitAD arm that ever became action-sensitive — at the provable no-information
floor for 5,200 steps, then `pick_acc 1.000` for 12 consecutive rows — and it was abandoned at 25 %. It was
**not one-variable: four diffs, including SCRATCH INIT**, i.e. **a free encoder**.

⭐ **If action-sensitivity is encoder-shaped, then *"the encoder was able to adapt"* is a coherent mechanism
for why that arm and no other developed it.** Read from the other side, it is the same statement as REF-A's
frozen-encoder ceiling (2.14 m plateau, speed R2 0.61).

⚠️ **Falsifiers, stated so this cannot harden into folklore:** a scene-matching shortcut was available to
o11; the arm carries four simultaneous differences; and the hypothesis predicts nothing yet that a frozen
arm could not also produce. ⇒ **It does not license a design change.** What it does is upgrade the banked
4-minute actdiv probe on `o11p30k` from a curiosity to **a test of a named mechanism**, which is worth
running before v7f freezes.

### Three levers that survive a frozen trunk (the part we can use either way)

| lever | what it is | why it survives |
|---|---|---|
| **Anchored actdiv** | measure `z_t+1-hat(a) - z_t+1-hat(0)` — displacement against the **zero-action** prediction — and read **MONOTONICITY** in action magnitude, not just separation | a diagnostic change, no training change. Their LeWM result (*"means remain concentrated near the origin and substantially overlap"*) is what this measurement looks like when it fails |
| **Transition-level probing** | predict `Dx` from `Dz` | our ladder probes **states** only; this probes **transitions**, which is where **P-4 / L3** (*does the predictor add anything over `z_t`*) actually lives. A missing rung, not a refinement |
| **Decode-target rule** | if any action decode is supervised, target **(a, kappa)** and **EXCLUDE v** | `v` IS our realised-motion channel and the leak-prone one. Their Table 3: raw action best, and **concatenating extra state-change signals made it WORSE** (*"redundant or less action-aligned information"*) — alignment beats breadth |

### ⭐ Composition with H-RANK-17 (backlog row 13)

Row 13's additive motion latent (`z_t + m_t ~ z_t+1`) and LDAD are the **same family**: row 13 makes the
displacement **explicit by construction**, LDAD **supervises** it. ⇒ `m_t` is LDAD's natural attachment
point, and the two **compose rather than compete**. If v7f keeps an `m_t` slot, the line stays reachable
even under a frozen trunk.

### Escalated to the Master Mind 2026-09-02 for the v7f design freeze

Four asks, in order: **(1)** run the endpoint-leak probe audit **before** the freeze — it can invalidate a
prior P-2 elimination that v7f may rest on; **(2)** state explicitly whether v7f freezes the trunk, and if
so record that the LDAD line is inapplicable **as published** — or keep a partial-unfreeze arm reachable at
~zero cost (the FS-2 asymmetry: cheap now, expensive after the geometry freezes); **(3)** reserve the `m_t`
slot; **(4)** exclude `v` from any action-decode target.


---

## Entry 2026-09-05-01 — ⭐⭐⭐ ATM `2606.09028` (FULL TEXT): our probe ladder reads the leaky cell

`Action-Consistency Transfer Matrix for Diagnosing and Improving Latent World Models. Jiaheng Chen,`
`arXiv 2606.09028, submitted 2026-06-08. PUBLISHED, FULL TEXT (arxiv HTML v1, method + experiments).`
`Cited by: Frontier Scan/Daily/2026-09-05/RESULT.md F2. Banked — already held; citation updated (V-1).`

### The construct

`D[i,j] = E‖h_i(ξ_j) − a_t‖²` — rows = probe TRAINING domain, columns = probe EVALUATION domain, over
domains `T` (true encoded transitions) and `P` (model-predicted transitions). Probe input, verbatim:

> `ξ_t^T = [z_t, z_{t+1}, z_{t+1} − z_t]` and `ξ_t^P = [z_t, ẑ_{t+1}, ẑ_{t+1} − z_t]`

Diagonal = in-domain decodability. Off-diagonal = cross-domain transfer.
Screening score: `S_ATM = −D_{T,T} − λ₁·|G_{T→P}|` with `G_{T→P} = (D_{T,P} − D_{T,T})/(D_{T,T} + ε)`.

### ⛔ The finding that matters for us

**The diagonal carries the GS-1 endpoint-concatenation confound by construction.** `[z_t, z_{t+1}]` is
exactly the shape Delta-JEPA (entry 2026-09-02-01) warns can succeed on action-correlated cues absorbed
into `z_{t+1}` *"without requiring the model to represent the actual transition"*.

⭐ **The off-diagonal does not.** `G_{T→P}` fits the probe on REAL transitions and evaluates it on
PREDICTED ones. An endpoint cue living in the real `z_{t+1}` need not survive into the model's `ẑ_{t+1}`,
so the gap isolates whether the **predictor** reproduces action-carrying structure.
⇒ **`G_{T→P}` is the leak-resistant statistic, and it is the one TanitAD has never computed.**
Our anchored-actdiv work (GS-8) and the transition rung (GS-9) both sit on the diagonal.

### MEASURED (theirs, PUBLISHED — not our units)

| quantity | value |
|---|---|
| correlation with true planning success, `D_{T,T}` | **ρ = 0.813** |
| correlation with true planning success, prediction loss `−L_pred` | **ρ = 0.498** |
| pairwise ranking accuracy, all pairs | 82.84 % |
| pairwise ranking accuracy, >5 % success margin | **98.81 %** |
| speedup vs CEM (LeWM) | 3–7 min → **33–55 s** |
| speedup vs CEM (DINO-WM) | 1.4–2.8 h → seconds (**>100×**) |
| AITS planning gain | TwoRoom 87→92 · PushT 96→100 · OGBench-Cube 74→84 |

⇒ **A published measurement says an inverse-decodability statistic ranks checkpoints better than the
prediction loss does.** Our v7 ladder ranks arms substantially on `o5_loss`.

### ⛔⛔ AITS does not transfer as published — and that is now a PATTERN

`L_AITS = E‖h_ψ(ξ_t^T) − a_t‖²` is an inverse head added **during world-model training, on the ENCODER's
real transitions**, then discarded. **Our trunk is frozen.**

⭐⭐ **This is the SECOND consecutive action-identifiability objective blocked by the frozen trunk.** LDAD
(`2606.31232`, entry 2026-09-02-02) was the first, and that entry called the frozen-encoder mismatch *"the
decisive one"*. **Two independent papers, same structural blocker: the published mechanisms for making a
latent action-aware all act on the ENCODER.** ⇒ This is no longer a per-paper caveat; it is a **v7f design
input**, and it sharpens escalated design constraint 2 (*record the trunk-freeze decision explicitly*) from
a hygiene ask into a **line-blocking** one.

### Their stated limits, carried honestly

Within-task ranking only, not absolute cross-task comparison. Cross-family (LeWM↔DINO-WM) ranking falls to
**68.54 %** without lightweight coefficient refitting (**89.90 %** with). AITS-P induces *"domain-specific
action codes"* that do not transfer symmetrically on complex tasks. The linear screening score assumes the
true→predicted gap does not dominate nonlinearly. ⛔ **No explicit chance floor is reported** — we must
supply our own constant-only and shuffled-action controls or the number is uninterpretable.

### TanitAD's position in this track after today

The **diagnostic** half is available to us: 0 GPU, post-hoc probes, frozen-trunk compatible, on latents we
already hold. The **training-signal** half (AITS) is not. ⇒ proposed row **FS5-2**.
⚠️ Risk carried forward: our action channel is realised motion (`r 0.9988`), so `D_{T,T}` may sit at the
floor at initialisation — tautological, exactly as GS-2 (REVISED) warned. **The off-diagonal is precisely
the way to see through that**, which is why the experiment reads both cells and commits to both outcomes.

`Next in this track: (1) the 2×2 ATM on banked v7 latents with both controls (FS5-2); (2) LeWorldModel`
`2603.19312 and Sub-JEPA 2605.09241 — ⛔ CORRECTION: both ARE BANKED (library.json, verified 2026-09-05);`
`register debt D-9's "UNBANKED" half is FALSE. Both remain UNREAD, and that half of D-9 stands.`

---

## 2026-09-09-01 - ACPC: the family's first instrument that a FROZEN TRUNK does not block

`lib 2608.12939 "Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency". FULL TEXT. Class PUBLISHED. Retrieved 2026-09-09.`

**The entry that matters is a correction to this ledger's own running conclusion.** Entries
`2026-09-02-01/-02` (Delta-JEPA / LDAD) and `2026-09-05-01` (ATM / AITS) each concluded that an
action-identifiability method was blocked by our frozen trunk, and `2026-09-05-01` generalised it to
*"SECOND consecutive action-identifiability objective blocked by our frozen trunk"*.

**That generalisation was over-broad and is corrected here.** It holds for the OBJECTIVES in this
literature. It does not hold for its MEASUREMENTS. Verbatim from the primary: *"Let F-theta denote the
**frozen** action-conditioned predictor"*; the encoder is frozen throughout; and ACPC is explicitly
contrasted with MWM, which *"enforces action-conditioned rollout consistency **during training**"*.

**Criterion (bisimulation), verbatim:** *"two observations should be treated as the same state only
when their action-conditioned consequences agree."*

| quantity | ASCII transcription | reads |
|---|---|---|
| ACPC | `ACPC_H(h, h~, a) = || G_a(E(h)) - G_a(E(h~)) ||_2`, weighted H-step | divergence of clean vs perturbed history **under the same actions** |
| IR | `IR_q(theta) = Q_q({R_i})` over anchors | clean-vs-perturbed rollout spread |
| SR | `SR_q,delta(theta) = mean 1[ D_diff > IR_q(theta) + delta ]` | whether **different states stay distinguishable after rollout** |

**Proved:** divergence bounds the perturbation-induced change in multi-step prediction error **and
planner cost**. **Measured:** 55.9 +/- 4.7 % prediction-error MAE reduction; 15.2 +/- 2.0 % CEM
selection-regret reduction.

**The collapse signature, and why we want it:** *"we train LeWM on TwoRoom with four SIGReg weights...
the representation collapses... its **SR falls to 0.066**, compared with **0.967-0.984** for nonzero
SIGReg."* SR is a RATE in [0,1] against a self-calibrated threshold (`IR_q + delta`), i.e.
**rig-relative by construction** - the exact property H-RANK-16's 8.56 participation floor lacks
(20.52 / 40.77 / 8.56 on three rigs).

**Protocol:** H=8 diagnostics (H=5 planning), 100 anchor histories, 5 Gaussian draws per anchor, 100
episodes per seed, 3 seeds.

**Our position in this track, updated.** SR reads the DENOMINATOR that MM-1 showed our h=1 ratio was
moving through: `o5_k` 8 to 60 moved action spread 0.83x but scene spread 1.71x. A rollout whose scene
spread grows while SR falls is **diffusing, not discriminating** - a distinction MM-1 needed and could
not name.

**Debt D-9 hardens.** ACPC is the **THIRD** independent critic of LeWM we have read (after Delta-JEPA
and ATM) while `2603.19312` itself remains unread. Three restatements through channels we did not
choose is not a substitute for the primary.

**Cost of adoption:** *"No random baselines or chance-level controls provided."* Any port supplies a
constant-only control and a raw-input floor, or it inherits their gap.

## 2026-09-10-01 — Value-guided JEPA: the winner is an ENCODER loss, and joint training is worse

`FULL TEXT` · arXiv **2601.00844v1** (retrieved 2026-09-10) · Destrade, Bounou, Le Lidec, Ponce, **LeCun** · World Modeling Workshop 2026 poster, 7 pp.
Method: train the state encoder so that `V(s,g) = −‖E(s) − E(g)‖` approximates the negative goal-conditioned value for a reaching cost `C(s,a,g) = 1[s ≠ g]`, via an IQL expectile loss with stop-gradient target and discount γ. Optional quasi-distance in place of the Euclidean. Planner: MPC with an **MPPI** optimiser; success over **200** wall instances / **80** maze instances.

**Table 2, verbatim (planning success; columns WS · WB · Maze):** Contrastive .49/.59/.50 · Regressive .54/.57/.46 · **pred VCReg .55/.89/.54** · **pred EMA .46/.43/.04** · VF .63/.94/.49 · **VF quasi .71/.96/.63** · VF pred .55/.75/.49 · VF quasi pred .61/.85/.43 · VF VCReg .49/.75/.39 · VF VCReg pred .47/.67/.39.

⛔ **The two winners are "Sep" — the state encoder is trained with the value loss alone.** On a **frozen trunk this has nothing to shape.** ⭐ **THIRD consecutive action/value-identifiability family blocked by our frozen encoder** (LDAD encoder-shaping 09-02 → AITS encoder-side 09-09 → IQL/value today). **The trunk-freeze is not neutral; it is a measured exclusion, three deep.**

⛔ **Joint training loses in 5 of 6 cells.** Authors verbatim: *"Learning representations using both a prediction loss and an IQL loss is less effective than using the latter loss alone."* ⇒ **the naive "bolt a value head onto the predictor objective" plan is the published worst variant.**

⭐⭐ **`pred EMA` = 0.04 on Maze vs `pred VCReg` 0.54 — a 13× gap and the table's worst cell.** Their EMA carries **no τ ramp**, exactly backlog row 4 / MM-E1's caveat. ⇒ row 4's τ-ramp precondition now has a published failure number, not only an argument. ⚠️ This measures **planning**, not drift; it does not refute EMA's measured drift benefit (0.4531 → 0.36–0.40). **Both can hold — EMA may buy drift and cost planning.**

⭐ **They independently propose our hierarchy, with a mechanism.** Verbatim: *"using a hierarchy of representation spaces, where higher levels model longer-range transitions or more coarsely sampled trajectories, may better capture distant relationships."* Named cause: distant (s, s′, g) triplets are sparsely sampled and the discounted-value gradient vanishes far from goal, so signal-to-noise collapses at range.

**Author-stated risks:** *"our IQL approach is known to be biased in random environments"*; *"prediction-based methods are indeed expected to be more robust to stochasticity in non-deterministic environments."* ⚠️ **Driving is highly stochastic — a named reason this may not transfer.**

## 2026-09-10-02 — LeWM read at last: our conditioning interface, and no action control (debt D-9)

`FULL TEXT` · arXiv **2603.19312v1** (retrieved 2026-09-10) · Maes, Le Lidec, Scieur, **LeCun**, Balestriero · 2026-03-13.
~15 M params; two loss terms only — prediction MSE + **SIGReg** (isotropic-Gaussian latent regulariser via statistical testing of random projections), λ = 0.1: `L_LeWM ≜ L_pred + λ·SIGReg(Z)`. Planner: **CEM, 300 samples, 30 iterations, horizon 5.** PushT +18 % over PLDM, beats DINO-WM without proprioception; **48× faster planning than DINO-WM**; DINO-WM slightly ahead on OGBench-Cube; PLDM and DINO-WM ahead on Two-Room.

⭐⭐ **Action conditioning, verbatim:** *"Actions are incorporated into the predictor through Adaptive Layer Normalization (AdaLN) applied at each layer."* ⇒ **layer-internal conditioning, the FiLM family — our v7 interface.**

⛔ **LeWM reports NO action-sensitivity control** — no hold-action, no zero-action, no shuffled-action. ⇒ **the action-insensitivity we have cited three times (Delta-JEPA, ATM, ACPC) is a THIRD-PARTY measurement, never a self-report. Every TanitAD citation must now say so.**

⭐⭐ **Two independent systems, one interface, one symptom.** LeWM: AdaLN, insensitive per three critics. v7: FiLM, insensitive (h=1 action/scene ratio 0.004; every arm `LOSES_TO_HOLDV0`). **Our elimination of FiLM rested on "it trained", which was never evidence of innocence.** ⇒ **GS-3 (FiLM vs action-as-token at matched params) rises to the top of the Architecture queue — the only P-2 branch with an independent corroborating instance.**

⚠️ **Opponent strength, recorded fairly:** LeWM's design thesis is that EMA and pre-trained encoders are fragile crutches, and it removes **both** while staying competitive at 15 M params. Our architecture builds on one of them.

**Conceded limitation:** *"Planning with current latent world models remains restricted to short horizons."*

→ `Architecture & Inference/Research/2026-09-10-hierarchy-edge-external-datapoint/RESULT.md` · `Deployment & Optimization/Research/2026-09-10-i1-value-guided-latent/RESULT.md`
