# PRE-REGISTRATION — the three experiments the 2026-07-29 deep research indicates

**Written BEFORE any of these is run.** Both outcomes are committed here in advance, per operating
standard rule 5. Source: `…/incoming/2026-07-29-deep-research-sota/DEEP_RESEARCH_2026-07-29.md`
(215 agents, 12.9 M tokens, 3-vote adversarial verification).

**Estimator for every interval below: the PAIRED episode-cluster bootstrap** (`taniteval/ci.py`,
B=2000) over the 40 val episodes. ⛔ **`overlapping_holdout_se` is forbidden** — it is not a
jackknife, it biases the point estimate bidirectionally (−6.67 % to +11.69 % over 27 arms), and it
manufactured the programme's one "load-bearing" hierarchy seam.

⚠️ **None of these is a retrain.** Neither survey produced admissible evidence for a retrain, a new
architecture, or a quantisation target. Total across all three: **well under 1 GPU-day** on existing
checkpoints.

---

## E-CR — does our imagination decay COMPOUND, or does the task just get harder?

**Resolves C61.** Priority **1**. Cost **ESTIMATED 0–6 GPU-h**.

### The question
On 2026-07-29 we measured pure-imagination ADE at K=4/8/12/16/20 (0.4–2.0 s, n=881 each):
0.0406 / 0.0829 / 0.1607 / 0.2753 / 0.4215, local exponent rising **1.03 → 1.63 → 1.87 → 1.91**,
global OLS 1.456. We described this as *"decay accelerates"*. **That is a mechanism claim the
instrument cannot support** — ADE-vs-horizon confounds two causes:

- **H-TASK**: predicting 2 s ahead is intrinsically harder than 0.4 s ahead.
- **H-COMPOUND**: the rollout feeds its own error forward and amplifies it.

### The instrument
SkyJEPA (arXiv 2606.23444, verified 3-0) — normalise by a teacher-forced arm at the SAME step:

```
CR_k = e_k,rollout / e_k,teacher-forced        # compounding ratio; 1.0 == no compounding
ER_k = E[e_k − e_{k−1}]                        # per-step growth
```

### Protocol
1. Add a teacher-forced path to `taniteval/taniteval/imagination.py`: identical context (8 frames),
   identical actions, but the **true** latent is re-injected at every step instead of the prediction.
2. Re-score the **SAME 40 val episodes / 881 windows** — same windows, same seed, no re-selection.
   ⛔ Anything that re-selects episodes breaks parity and must be refused.
3. Report **CR_k and ER_k at k=4/8/16/20** with the **paired** episode-cluster bootstrap (the two
   arms share windows — a paired interval, never a combination in quadrature).
4. Decompose lateral / longitudinal (`driving.py:201 frenet()`), reported AT the horizon.

### Outcomes, committed in advance
| outcome | reading | consequence |
|---|---|---|
| **CR_k ≈ 1, flat**, CI covering 1 | **H-TASK** | *"Decay accelerates"* is **FALSIFIED**. The acceleration is task difficulty. **NO architecture change is justified** — E-ROLL, rollout-recovery training and the Koopman lever are all unmotivated and must not be run. Amend C61 to a closed retraction. |
| **CR_k rising super-linearly**, CI excluding 1 at k=16/20 | **H-COMPOUND** | Compounding is real. **Rollout-recovery training** (train on prediction-corrupted histories, HorizonDrive's mechanism) becomes the indicated fix — **not** a larger horizon. Only then does the spectral/Koopman lever become a legitimate second experiment. |
| **CR_k rising but CI covers 1** | underpowered | Report as underpowered. Do NOT pick the convenient reading. Increase B or windows; do not proceed to E-ROLL on an ambiguous CR. |

⚠️ **Magnitudes do not transfer.** SkyJEPA's CR ≈ 1.4 at k=60 (vs ≈ 2.4 autoregressive) is on a
low-dimensional quadrotor physics vector with a physics-structured prober. **The metric DESIGN
transfers; their numbers are not a bar for us.** Note also their CR is still *rising* at k=60 — the
word "bounded" in that paper's framing is an editorialisation.

---

## E-DPSI — is our target-speed head keyed to HEADING rather than to the scene?

Priority **2**. Cost **~0.3 GPU-day**, existing checkpoints, no retrain, no renderer, no AlpaSim.

### The question
PlanT 2.0 (Tübingen, verified 3-0 against raw PDF) scores **DS 92.4 ± 1.7** on Bench2Drive — above
every sensor model listed — **while** its predicted speed *"abruptly increases to signal regular
driving"* at ego rotations of **10–15°**. It *"learns a shortcut by using its own rotation as a
predictive signal for when the road is clear"*.

**Why we can test this and a general camera model cannot.** PlanT itself says perturbing sensor
inputs in a controlled fashion is *"time-consuming and non-trivial"* — which is why it used an
object-centric planner. But `taniteval/taniteval/pseudosim.py` already ships `GridSpec.dyaw_deg`, an
**exact** camera rotation `H = K R K⁻¹`, *"exact for ARBITRARY scene depth (max|dH| = 0.000e+00,
30 conditions)"*. PlanT rotates the ego in simulation, moving world pose **and** object input
together; **our warp changes ONLY the observed heading — bit-identical real footage under an exact
homography.** If our speed head moves with dψ, it is unambiguously keyed to heading. No confound.

### Protocol
Sweep `GridSpec.dyaw_deg` across the full validated envelope on the 40-episode val cache, `dlon`
fixed. Plot against dψ: (i) produced `vt_speed`, (ii) `tspeed_5s`, (iii) `long_abs`.
**Look for a STEP, not a slope.** Free rider: repeat on the CL-SFT arm.

### Outcomes, committed in advance
| outcome | consequence |
|---|---|
| **A step in `vt_speed` within \|dψ\| ≤ 12°** | Part of our **88.7 %** longitudinal oracle gap (0.1899 of 0.2140) is a **SHORTCUT, not a capacity or scale limit**. The fix becomes **heading augmentation**, not a new channel and not more capacity. This would be the cheapest large win available to the programme. |
| **Flat / monotone-smooth in dψ** | Strengthens the estimation-problem reading — consistent with our IDM result that monocular speed is **scale-limited** (R² +0.865, smaller-is-better 0.86 M > 2.90 M > 19.98 M). Closes the shortcut hypothesis cheaply. |

⛔ **THE SHARP LIMIT, REGISTERED NOW.** PlanT's onset is **10–15°**; our measurement-grade envelope
is **\|dψ\| ≤ 12°** (`pseudosim` falsifies anything beyond — 0 % out-of-envelope is what makes ≤12°
quotable). **We cover the LOWER EDGE ONLY. A null inside ±12° is NOT "we are clean" — it is
"no shortcut below 12°", and MUST be written up in exactly those words.**

⚠️ A null here would **not** contradict PlanT. Its root cause is CARLA-specific — a scripted expert
that only turns when the road is free, plus a success-only dataset. **We train on HUMAN-DRIVEN
PhysicalAI-AV logs with no such invariant**, so the mechanism may simply be absent by construction.
*(Do not quote PlanT's Town13 data-composition numbers — that ablation was voted 0-3.)*

---

## E-ROLL — does the recursive k=1 head survive past its trained horizon?

Priority **3**, and **GATED ON E-CR RETURNING H-COMPOUND**. Cost **ESTIMATED 2–4 GPU-h**.

### The question
We have described the 2 s bound as architectural — an `index_select` over a learned horizon
embedding. True but incomplete: `stack/tanitad/models/predictor.py` already carries a fixed causal
window (`PredictorConfig.window=6`, learned `self.pos` asserted equal), a **k=1 residual head**, and
an unused projection annotated verbatim
`self.out_proj = nn.Linear(state_dim, d)  # reserved: feed predictions back`.

A rolling-window rollout is therefore **drop oldest → append the k=1 prediction → re-run**: no
retrain, no new parameters, cost flat in horizon (HorizonDrive's bounded-context result).

### Protocol
Recursively apply the existing k=1 head past 20 steps on the deployed arm; measure ADE **and CR_k**
out to **4 s and 6 s** — 2× and 3× the trained horizon, matching SkyJEPA's 3× protocol.

### Outcomes, committed in advance
⚠️ **WE EXPECT IT TO DIVERGE**, because the k=1 head was trained under teacher forcing.

| outcome | consequence |
|---|---|
| **Diverges past 2 s** (expected) | **Informative, not a refutation.** It demonstrates the cap is not the binding constraint — *training* is. Triggers rollout-recovery training as the next experiment. |
| **Degrades gracefully to 4–6 s** | The 2 s cap was costing us horizon for free. Lift it in the eval path immediately and re-run the imagination sweep over the longer range. |

⛔ **Do NOT run E-ROLL if E-CR returns H-TASK.** With no compounding there is nothing for a
rollout-recovery scheme to fix, and a longer horizon would be measuring task difficulty at greater
range — a more expensive way to learn what E-CR already said.

---

## Sequencing, and what is explicitly NOT authorised here

```
E-CR ──H-TASK──────► STOP. Amend C61. No architecture work.
  │
  └──H-COMPOUND──► E-ROLL ──► rollout-recovery training ──► (only then) spectral/Koopman lever

E-DPSI runs INDEPENDENTLY of E-CR — different subsystem, no shared dependency.
```

**NOT authorised by this document** (needs the PI): any retrain; the 30-pod-day X2 verdict run; the
wheelbase fix; deleting anything; new HF publication.

**Also registered:** the Koopman/spectral lever (§8 of the research report) is **LOW confidence** —
proprioceptive domain only, no pixel/ViT observations anywhere, seed count unstated, and its
headline −89.4 % latent MSE is plausibly a units artifact (relay only the −23.2 % observation-space
figure). Its stated trade-off — lowering the spectral radius *erases persistent task information* —
**directly threatens the speed/scale magnitude our speed-channel fix bought (speed R² 0.965)**. If
it is ever run, **CR_k and speed R² must both be reported**, because the mechanism predicts a trade.

## Hypothesis links

- **E-CR** → the imagination-horizon question, and whether H-WM-fidelity has a compounding failure
  mode distinct from capacity.
- **E-DPSI** → directly interrogates the **88.7 % longitudinal** finding and the `tspeed_5s`
  R² 0.7635 / RMSE 4.4545 m/s head. Bears on the longitudinal-blindness mechanism
  (the 5-way maneuver softmax that mixes LAT+LON).
- **E-ROLL** → the architectural-cap claim, and the closed-loop horizon question.

⚠️ **Reporting protocol registered alongside these** (from MoP-JEPA's authorial concession that
*"With 10 % false edges, replanning raises success from 0.40 to 1.00"*): open-loop fidelity (CR_k)
and closed-loop success are **two separate decision inputs**. **A closed-loop improvement with flat
CR_k is an EXECUTOR gain, not a world-model gain, and may not be quoted as world-model progress.**
