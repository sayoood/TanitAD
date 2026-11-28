# The plan to the goal — action-conditioning is the critical path

**Written** 2026-08-24 20:20 · **Author** Master Mind · **Status** live plan, revised
against the day's measurements · **Tier** every claim below is T0-DIAGNOSTIC unless
stamped otherwise

---

## 0. The goal, in the PI's words

> *"solve the collapse, environment learning in the wm in both encoder and
> predictor, and prove that it is a good base to learn the physics of driving"* —
> physics meaning *"if the vehicle in front decelerates, the ego must react on
> it… when the corridor is limited by obstacle vehicles the ego will evade"*.

⭐ **The decisive reframing.** *Physics* as the PI defines it is a
**COUNTERFACTUAL** claim: *if I brake, what happens?* A representation can be
perfectly rich and perfectly predictive and still be useless for it, because
answering it requires the prediction to **change when the action changes**. That
is why the whole plan below has one critical path.

---

## 1. Where we actually are (all MEASURED today)

| goal | status | evidence |
|---|---|---|
| **collapse** | ✅ **SOLVED** | rank 3.80 → 25.58; 5 arms beat the C149 constant floor; Gate B = steps, not data |
| **environment — ENCODER** | ✅ **SUPPORTED HELD-OUT** | lead-matched: `n_agents` **+0.1220** (above DINOv3 +0.0998), `occ_center` **+0.3351**, corridor `n_free_cols` **+0.2080**, `occ_col3/6` +0.3315/+0.3093 — all above the raw-pixel floor |
| **environment — PREDICTOR** | 🔶 **re-measuring lead-matched** | the unmatched read inherits the C153 confound that reversed the encoder verdict |
| **physics base** | ⛔ **NO — and this is the whole problem** | **32 of 32 arms action-independent**: replacing every action with noise changes `nrmse` by ≤ **0.0919 %**, 0/32 above 0.1 % |

**The one number that defines the programme's position:** the largest action
response anywhere in TanitAD moves the prediction **11.6 %**, while a **10 %
perturbation of the latent** moves it **17.7 %**. Our world model responds more to
jitter in its own state than to the driver's command.

---

## 2. Why — and the hypothesis I killed this evening

**The objective explanation (stands).** O5 trains ẑ_{t+k} ≈ z_{t+k}. Over 0.6 s
the scene at t+k is overwhelmingly set by the scene at t and only marginally by
the ego's command. **The loss-minimising solution is to ignore the action.** The
predictor is doing exactly what it was asked.

**The architectural explanation (REFUTED, 2026-08-24 20:15).** I proposed that the
action is structurally suppressed: FiLM `to_scale_shift` is zero-init, and the
branch carrying it is LayerScale-scaled 1e-5, while the state has an identity
residual bypassing both. **Measured on nine banked checkpoints, it is false:**

| arm | steps | `\|FiLM W\|` / `\|act_emb W\|` | LayerScale |
|---|---|---|---|
| `rdw8`, `o5k8`, `splitfrz` | 2,000 | **0.072 – 0.083** | *absent* |
| `postrain10k` | 10,000 | **0.152** | *absent* |
| `rdw8p30k`, `splitp30k`, `scale1`, `champ30k`, `rdw8s30k` | 30,000 | **0.205 – 0.308** | *absent* |

⇒ **The FiLM opens MONOTONICALLY with training and has not plateaued**, and these
arms use `CausalBlock`, which has **no LayerScale at all** — the 1e-5 suppression
does not exist here. ⭐ **This is the strongest available evidence that the lever
is the OBJECTIVE and not the wiring:** the pathway is demonstrably learnable and
is being learned, just slowly, because nothing is pushing it.

⚠️ **A tempting secondary reading that does NOT hold:** FiLM magnitude does not
predict measured action sensitivity (`splitfrz` has the lowest FiLM ratio 0.0747
and among the highest sensitivity 0.0945). Do not use FiLM norm as a proxy for
action-conditioning; measure the response directly with `actchan.py`.

---

## 3. The critical path

```
   O11-CF arm  ──►  action pathway opens?  ──YES──►  physics counterfactual test
   (launches                │                             │
    tonight)                NO                            ▼
                            ▼                     planner on a WM that
                    architectural fix:            answers "if I brake…"
                    give the action an
                    unsuppressed path
```

Everything else is parallel and none of it substitutes for this.

### PHASE 1 — does the objective open the pathway? (tonight → +8 h)

**`o11p30k`**, a matched pair against Gate A `ok8p30k` changing **one thing**:
`--w-o11-cf 1.0 --o11-k 4 --o11-negs 3 --o11-tau 1.0`. Pre-registered with four
outcomes in `PREREG_O11_COUNTERFACTUAL_ACTION.md`. Code shipped and md5-verified
on Thor (`24994d5ce95d736028c913dd7cfd31c1`), both call sites carrying the knobs.

**Primary read:** `d_out(shuffle_all) / d_out(latent +10 % control)`, control
value **8.5 %**. CONFIRMED ≥ 50 % with `o5_loss` within +10 %.

⭐ **Why O11-CF and not "more steps" or "a bigger predictor":** an
action-independent predictor scores **exactly `ln(1+n_neg)`** on O11 and cannot
do better. The no-information floor is *inside the loss*, so the objective cannot
be satisfied without reading the action. Verified: an untrained predictor reads
1.3862943649 against `ln 4` = 1.3862943611.

### PHASE 2 — conditional, and each branch is already decided

| Phase-1 outcome | next action |
|---|---|
| **CONFIRMED** | O11-CF enters the v7 recipe · re-run the 32-arm census with the shuffle control · **proceed to Phase 4** |
| **PARTIAL** (20–50 %) | sweep `--w-o11-cf` 0.3/3.0 and `--o11-k` 2/8 before spending full-scale GPU |
| **DEGENERATE** (ratio up, `o5_loss` down > 10 %) | the ẑ = f(z) + λa solution. O11-CF is wrong for this rig; **go to the architectural fix** |
| **REFUTED** (< 20 %, `o11_excess` ≈ 0) | the deficit is in the action-injection **SITE**. Candidates, in cost order: (a) remove the FiLM zero-init, (b) give the action a direct additive term outside the MLP branch, (c) cross-attend the action instead of FiLM-modulating it |

### PHASE 3 — the DINOv3 gap (parallel, does not block Phase 1)

⛔ **Frozen off-the-shelf DINOv3 beats our trained trunk on 4 of 5 aggregate
spatial targets** (`occ_left` +0.3315 vs −0.0159 · `occ_center` +0.3998 vs
+0.3351 · `occ_right` +0.3477 vs +0.1728 · `n_free_cols` **+0.4857 vs +0.2080**).
We beat it only on `n_agents` and `occ_col6`.

⭐ **The strategic question this forces, and it is the PI's to answer.** If a
frozen encoder we did not train carries more spatial structure than our trained
one, then **our parameter budget is in the wrong place.** The programme's
differentiator is the *world model*, not the *encoder*. The candidate
architecture is **frozen DINOv3 trunk + our action-conditioned predictor**, which
concentrates capacity where we add value. `splitp30k` is already a frozen
*distilled* init, so this is a step along a road we are on, not a reversal.
**Cheapest discriminating experiment:** run the O11 recipe on a frozen-DINOv3
trunk and compare the action pathway and the spatial panel against `o11p30k`.

### PHASE 4 — the physics test itself, pre-registrable NOW

Only meaningful once Phase 1 is CONFIRMED. **The test the PI actually asked for:**

> Take held-out windows where the lead vehicle **decelerates**. Roll the
> predictor twice from the identical state — once with a **braking** action
> sequence, once with **maintain**. Ask two questions:
> 1. **Do the two predictions differ?** (they must; if not, Phase 1 failed)
> 2. **Does the braking rollout's implied headway match the true future better
>    than the maintain rollout's?** — i.e. does the model know that braking
>    *opens the gap*?

Controls, fixed in advance: a **constant** predictor reading the no-information
value exactly · a **raw-pixel** floor · a **time-shuffled** control (structure
surviving a shuffle is leakage, not dynamics) · **n and the function class
stated**. ⚠️ Scope stays **T0**: this is what the representation supports, never
"the car drives".

⭐ **This is the deliverable that closes the PI's mandate**, and it is the first
test in the programme that is a genuine counterfactual rather than a
reconstruction.

---

## 4. What would make me wrong

1. **O11 could open the pathway without making it USEFUL** — the model learns to
   distinguish actions without learning their consequences. Phase 4 catches this;
   Phase 1 alone does not. ⇒ **Phase 1 CONFIRMED is a gate, not a result.**
2. **The lead-matched predictor panel (running now) could show the predictor
   carries nothing** even before O11. Then the problem is upstream of
   action-conditioning and Phase 1 is premature.
3. **The 0.6 s horizon may be too short for actions to matter at all.** If the
   ego's command genuinely cannot move the scene in 6 ticks, no objective will
   make it. ⇒ **A cheap check worth doing before trusting a REFUTED verdict:
   measure how much of z_{t+k} − z_t is explainable from the action at all**, with
   a linear oracle. ⚠️ A linear negative is only a negative about linear maps
   (the standing rule), so a nonlinear probe with a time-shuffled control is the
   honest version.

---

## 5. Three retractions today, one root-cause class

C151, C152, C153 — all the same shape: **a reading published while its own
control was still queued.** C153 is the sharpest: I found the confound, built the
instrument to remove it, and then published the reading that instrument was
designed to test, arguing one component survived. ⇒ **The rule now binding on
this plan: no phase reports a verdict until its committed control has returned.**
