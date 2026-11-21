# The T1 floor: is it "never trained to drive", or is the action input not being used?

**Package** `Architecture & Inference/Research/2026-08-30-t1-floor-action-sensitivity` ·
**Author** Research Lab agent (daily run 004, 2026-08-30) · literature + source reading, 0 GPU.
Seeds: `D-T1-V7-READ` (both v7-tiny arms lose to their own hold-action control on every
distance metric; `copy_detector` CLEAN at `echo_index 0.0000`) and the still-open drift
attractor (`MM-E4`, `MM-E5`; frozen-teacher feature target is the pre-committed next lever).

⛔ **SCOPE — this package deliberately does NOT touch the input pipeline.** Image resolution,
projection/distortion, patch size, temporal context, FOV/camera count and cycle time are the
subject of a separate PI-commissioned review and are excluded here to avoid duplicate work.

---

## F1 — ⭐⭐ THE ARMS ARE ACTION-CONDITIONED **BY CONSTRUCTION** AND ACTION-FREE **BY OBJECTIVE**. That is a third state, and it changes which literature applies.

[MEASURED — source read, this repo] The register's scope note says the arms are *"a world-model
trunk plus a grounding readout, never trained to drive"*, with every planner objective at zero
(`--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0`). Read at source, the picture
is more specific:

| fact | evidence |
|---|---|
| the model's forward **takes actions positionally** | `stack/tanitad/models/v6.py:5455` — `def forward(self, frames: Tensor, actions: Tensor, v0: Tensor, *, …)` |
| the predictor is **built with `action_dim=3`** | `stack/scripts/train_v6_staged.py:4122`, `stack/tanitad/models/v6.py:4007` |
| the input contract carries actions `[batch, window, action_dim]` | `v6.py:5750` (synthetic-batch builder) |
| **O1 is the action-conditioned objective**, and it is ON by default | `train_v6_staged.py:7042` — `ap.add_argument("--w-o1-ctrl", type=float, default=1.0)` |
| O1's documented purpose | *"O1 action-conditioned prediction with L_ctrl in RESPONSE FORM from step 0 — the **ANTI-ACTION-ECHO** measure"* (trainer header, quoted in `…/2026-08-19-simwam-analysis/code/v7tiny.py:38`) |

⇒ These arms have a **fully wired action input pathway and no training signal that requires
using it.** Nothing in `latent prediction + SIGReg` is made worse by ignoring the action
channel entirely.

**Why the distinction is load-bearing.** The literature splits on precisely this axis, and the
two branches give opposite verdicts:

- **Action-FREE trunk** ⇒ the T1 number is a *null measurement*. [PUBLISHED, banked
  `2506.09985`] V-JEPA 2 finds 1 M+ hours of action-free video **insufficient** for control;
  planning requires an action-conditioned post-training stage. **SUPPORTS** "expected floor".
- **Action-CONDITIONED trunk** ⇒ "planner weights were zero" does **not** explain the floor.
  [PUBLISHED, banked `2411.04983` DINO-WM; `2412.03572` Navigation World Models] both plan
  closed-loop with **no policy and no reward model at all** — the planner lives entirely in a
  test-time trajectory optimiser. **UNDERCUTS** "expected floor".

Our arms are **neither cleanly**: conditioned in the graph, unconditioned in the loss. ⛔ **The
comfortable reading — "of course it cannot drive, we never trained it to" — is therefore
NOT established by the configuration alone**, and should not be quoted as if it were. The
honest statement is that O1, *the term whose stated job is to stop action-echo and force
action-response*, was switched off, so the model was free to become **action-insensitive**;
whether it did is an open, cheaply-measurable question (F2, F4).

---

## F2 — ⭐ THE MEASURED cl−ha GAP IS ~1 %, WHICH LOOKS LIKE ACTION-INSENSITIVITY RATHER THAN BAD DRIVING

[DERIVED from the register's own MEASURED T1 numbers — arithmetic only, no new measurement]
Recomputing the closed-loop vs hold-action gap as a **fraction of the hold-action value**:

| arm | metric | closed-loop | hold-action | Δ / ha |
|---|---|---|---|---|
| `emao14_30k` | ade | 14.069 | 13.879 | **+1.37 %** |
| `emao14_30k` | fde | 26.297 | 26.131 | **+0.64 %** |
| `emao14_30k` | LAT_heading | 94.63° | 95.13° | −0.53 % |
| `emao14_30k` | LON_speed | 10.703 | 10.900 | −1.81 % |
| `o14fut30k` | ade | 14.293 | 14.116 | **+1.25 %** |
| `o14fut30k` | fde | 26.690 | 26.442 | **+0.94 %** |
| `o14fut30k` | LAT_cross | 1.461 | 1.426 | +2.45 % |

**Handing the model the wheel changes the rollout by ~1 % on every distance metric.** A model
that were *driving badly* would diverge from a frozen-action rollout substantially — bad
steering takes you somewhere visibly different. A model whose predicted future barely moves
when the action changes is **not driving at all**; its action input is close to decorative.

⚠️ **HYPOTHESIS, not a finding — registered as `H-ARCH-ACTINS`.** A competing account fits the
same evidence and I cannot separate them from banked numbers: at **heading MAE ≈ 95° (chance)**
and ADE ≈ 14 m the rollout may already be so degenerate that *no* action choice would move the
metrics. Both accounts predict a small cl−ha gap. ⛔ **The gap alone therefore does not
establish action-insensitivity**, and no verdict is drawn here — F4 names the discriminating
test. *(Recorded this way deliberately: the sign of `S-rate(masked)` already disagrees with the
distance metrics in this read, per C160.)*

---

## F3 — THE FAILURE MODE HAS A PUBLISHED NAME, AND `echo_index 0.0000` IS ITS OTHER HALF

[PUBLISHED, banked] The copycat/echo half of our story is well named — [`2010.14876`] defines
the copycat problem (the imitator predicts *the expert's previous action*) and fixes it by
adversarially removing previous-action information; [`1905.11979`] is the causal-confusion
foundation. Our `echo_index 0.0000 vs GT 0.2113` says we have achieved what those papers
prescribe. **None of them claims removing the echo is *sufficient*.**

The other half is named too, and it is new work: [`2607.26712`, ActSWM] defines **"Context
Collapse"** — *"autoregressive latent predictors maintain high similarity to future states
while producing nearly indistinguishable futures under different action sequences."* That is
**exactly the signature F2 describes**. [`2606.31232`, Delta-JEPA] states the mechanism:
*"reconstruction-free joint-embedding objectives can collapse to action-insensitive
representations"*, and proposes decoding the executed action from the **latent displacement
between consecutive observations** so adjacent embeddings cannot collapse without losing
action information. ⇒ **"Non-echoing but non-predictive" is not an anomaly; it is the
predicted endpoint of removing the echo without adding an action-response term** — which is
precisely what setting `w-o1-ctrl 0` did.

---

## F4 — THE DISCRIMINATING TESTS ARE CHEAP, AND ONE OF THEM NEEDS NO NEW ARCHITECTURE

[PUBLISHED, banked + MEASURED (repo)] In cost order, all pre-registrable with both outcomes:

1. ⭐ **Action-divergence probe** [`2607.26712`]: roll the *same* initial state under *distinct*
   action sequences; measure the spread of predicted futures. **Spread ≈ 0 ⇒ Context Collapse**
   (F2's hypothesis confirmed, and the frozen-teacher lever is aimed at the wrong target).
   **Spread large ⇒ the trunk does respond to actions** and the floor is a planner/decoder
   problem. This is the single test that separates F2's two accounts, it is T0, and it needs
   **no labels and no training**.
2. ⭐ **Action recoverability of the latent** [`2606.07687` — *"models with strong pixel decoding
   quality can exhibit near-zero action recoverability"*]. **MEASURED: we already have the
   head** — `InverseDynamicsHead` is instantiated in the sibling 4-brain model
   (`stack/tanitad/models/inverse_dynamics.py`; used at `fourbrain.py:461`). ⚠️ **It is NOT
   wired in `models/v6.py`** — the v7 line's model — so this is a port, not a build.
3. **iKCE** [`2607.05966`]: per-step departure from a closed-form kinematic null. Its measured
   warning transfers directly to us — across a sweep crossing a gait-collapse boundary
   **iKCE stayed flat while the policy's reward collapsed**, i.e. *the world model's own
   diagnostic reads fine while control fails*. That is the shape of our T0-good / T1-floor gap.

⛔ **Per the CLAUDE.md probe rule, each of these ships with (a) a constant-only control that
must read the no-information value exactly, (b) a raw-input floor, and (c) printed n and d.**
The 2026-08-22 ridge-probe episode produced four confident publishable-looking numbers without
them.

---

## F5 — DRIFT: the τ-ramp is closed, and the frozen-teacher lever is genuinely un-answered by the literature

[MEASURED — relayed from the Master Mind this run, τ-ramp arm at 30k] The cosine τ-ramp is
**NEUTRAL vs fixed τ**: drift **0.6936 vs 0.6952**, nrmse **0.7408 vs 0.7466**, cos **0.7513 vs
0.7524** — all ~**20× smaller than the only seed spread we have measured**, so the comparison
cannot resolve them. ⇒ **EMA teacher stays in the recipe at FIXED τ = 0.996; the cosine ramp is
dropped.** T0-DIAGNOSTIC.

[PUBLISHED, banked] On **frozen vs EMA teacher the literature is in a genuine three-way
disagreement, and none of the three has run our ablation**:

| position | evidence | its named cost |
|---|---|---|
| frozen works | `2411.04983` (DINO-WM: frozen DINOv2 target, no collapse, no regulariser); `2509.10156` (LayerLock: progressive freezing, non-collapsing, scaled to 4B) | frozen features cannot adapt to the task; both results are manipulation/maze/video, **not driving** |
| EMA is right | `2404.08471` (V-JEPA); `2608.19085` (DA-WAM) — momentum target kept *specifically so* future representations co-evolve with the task | DA-WAM's stated reason for rejecting frozen |
| neither is needed | `2511.08544` (LeJEPA — SIGReg makes collapse statistically impossible, *"no stop-gradient, no teacher-student, no schedulers"*); `2011.10566` (SimSiam — **stop-gradient**, not the momentum encoder, is what prevents collapse) | — |

⇒ **Empty search, named:** no head-to-head frozen-vs-EMA teacher ablation in a *driving* world
model, with a cost figure, was found (two probes: collapse-prevention literature, and driving
world-model literature). **The pre-committed L2 frozen-teacher arm is therefore answering an
open question, not re-deriving a known one** — that is an argument for running it, and a
publishable result either way.

## F6 — `MM-E5`'s coupling hypothesis gains an independent mechanism from the literature

[PUBLISHED, banked `2510.26782`] *"The primary bottleneck for long-horizon fidelity is the
**geometric structure of the latent representation, not the dynamics model itself**."* This
predicts exactly the drift↔prediction coupling `MM-E5` registers as a hypothesis from n = 2
of our own arms. ⛔ **Per `MM-E5`'s own binding clause, no result may be re-read through the
frontier framing until L2/L3/L4 report — this citation does not license it either.** It is
recorded as a *third*, mechanism-independent reason the hypothesis is worth the discriminating
arms, not as support for the hypothesis.

---

## What this changes for TanitAD (≤3)

1. ⭐ **Run the action-divergence probe before the frozen-teacher arm.** It is T0, label-free,
   needs no training, and it decides whether the drift lever is even aimed at the right defect
   (F4.1). If the trunk is action-insensitive, `w-o1-ctrl > 0` — the term we switched off — is
   the indicated lever, not the teacher.
2. **Restate `D-T1-V7-READ`'s scope line.** *"Never trained to drive"* is true of the objective
   but not of the architecture: the action pathway is wired and O1 is on by default (F1). The
   admissible claim is **"the action-response objective was at zero, so action-insensitivity is
   unexcluded"** — which is a sharper and more useful statement than the current one, and it
   keeps `2411.04983`/`2412.03572` from being used to undercut us in review.
3. **Port `InverseDynamicsHead` into `models/v6.py`** (F4.2) — it exists in the 4-brain sibling
   but not in the v7 line. Action recoverability is then a readout, and it converts the
   `echo_index 0.0000` result from "we removed the shortcut" into "we removed the shortcut
   **and** the latent does/does not retain the action".

## Named empty searches

- **A head-to-head frozen-vs-EMA teacher ablation in a driving world model**: NOT FOUND (F5).
- **A published report of a closed-loop rollout losing to its own hold-action control**:
  NOT FOUND — see the Benchmarks package this run; the field publishes no hold-action baseline
  at all, so our T1 read has no comparator and is, as far as two probes establish, novel.
- ⚠️ **Not searched by design:** anything on input resolution, projection, patch size, temporal
  context, FOV/camera count or cycle time — owned by the parallel PI-commissioned review.

## Banked primaries

NEW this package: `2506.09985` · `2411.04983` · `2002.04523` · `2607.26712` · `2606.31232` ·
`2606.07687` · `2510.26782` · `2011.10566` · `2509.10156` · `2511.08544` · `2608.19085` ·
`2010.14876` · `1905.11979` · `1906.08253` · `2607.05966` · `2603.09086` · `2412.03572` ·
`2212.03319`. ⚠️ **Read-depth disclosure:** the search pass opened the arXiv abstract page for
`2607.26712`, `2510.26782`, `2606.07687`, `2606.31232`, `2607.05966`, `2511.08544`,
`2608.19085` and `2603.09086`; the remainder were read from search summaries and are
**PUBLISHED-SECONDARY until their banked PDF is opened**. All are now banked, so the upgrade is
available — but ⛔ **do not quote a body claim from an unopened one into `MODEL_REGISTRY.md` or
the paper.**
