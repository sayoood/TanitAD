<title>Injected row I-1, quality half — the value must shape the latent, and our trunk is frozen</title>

# I-1's quality half: the published answer is a latent-geometry answer, and it is blocked on our frozen trunk

`TanitAD Research Lab · Deployment & Optimization · 2026-09-10 · LAB-RUN-011`
`⭐ SERVES THE TOP INJECTED ROW — LAB_BACKLOG I-1 (MM, 2026-08-29): "Latent model-predictive planning with a LEARNED terminal value over composed h=1 rollouts — does a cheap value head beat fan-scoring at equal rollout budget?"`
`Prior state: COST half served 2026-09-01 (breadth 5.94x cheaper than depth; value head 0.150 ms = 1.2 %). QUALITY half UNTOUCHED and named "the real experiment". Row STAYS OPEN.`
`Search log: raw/search_log.md · Frontier scan F3/F4/F8`

---

## FINDINGS

**1. ⭐⭐⭐ The published record answers I-1's quality half — and changes the question.**
`PUBLISHED` · *Value-guided action planning with JEPA world models*, Destrade, Bounou, Le Lidec, Ponce, **LeCun** — arXiv **2601.00844**, World Modeling Workshop 2026, **full text read**. MPC with an MPPI optimiser; success rate over **200** wall instances / **80** maze instances.

| type | WS | WB | Maze |
|---|---|---|---|
| **pred VCReg** — the standard-JEPA baseline (our family) | 0.55 | 0.89 | 0.54 |
| **pred EMA** | 0.46 | 0.43 | **0.04** |
| VF — IQL value loss on the **encoder**, trained separately | 0.63 | 0.94 | 0.49 |
| **VF quasi** — IQL + quasi-distance, encoder, separate | **0.71** | **0.96** | **0.63** |
| VF pred — value loss **joined** with the prediction loss | 0.55 | 0.75 | 0.49 |
| VF quasi pred — joined | 0.61 | 0.85 | 0.43 |
| VF VCReg | 0.49 | 0.75 | 0.39 |
| VF VCReg pred | 0.47 | 0.67 | 0.39 |

I-1 asks whether a value head beats fan-scoring. **The published answer is that the winning mechanism is neither**: it is shaping the **latent geometry** so that embedding distance *is* the negative goal-conditioned value, learned with an IQL expectile loss on the **state encoder**. The planner is then unchanged — the same MPC — and it simply stops falling into local minima.

**2. ⛔⛔ The winning variant requires training the encoder. Our trunk is frozen.**
The two best rows are both **"Sep"**: the state encoder is trained with `L_VF` alone, then the action encoder and predictor are trained with `L_pred`. **A loss that shapes latent geometry has nothing to shape on a frozen trunk.**

⭐ **This is the THIRD consecutive action/value-identifiability family our frozen encoder excludes:**

| family | source | why it is blocked |
|---|---|---|
| LDAD (Delta-JEPA) | GS-2 REVISED, 09-02 | encoder-shaping — `Dz` is built from two **encoder** outputs |
| AITS | 09-09 | encoder-side |
| **IQL / value-shaped latent** | **today** | **encoder-shaping — `V(s,g) = -‖E(s) - E(g)‖`** |

⇒ **The frozen trunk is not a neutral efficiency choice. It structurally excludes an entire published family of planning improvements, and the exclusion is now measured three times.** This belongs in the v7f freeze decision as a stated cost, not discovered afterwards.

**3. ⛔ The naive I-1 implementation is the variant measured WORST.**
The obvious plan — add a value term to the existing predictor training — is the paper's "pred" (joint) column, and it **loses to the separated variant in 5 of 6 cells**: VF 0.63/0.94/0.49 → 0.55/0.75/0.49; VF quasi 0.71/0.96/0.63 → 0.61/0.85/0.43. Authors, verbatim: *"Learning representations using both a prediction loss and an IQL loss is less effective than using the latter loss alone."* And adding VCReg on top of IQL is worse again (VF VCReg 0.49/0.75/0.39).

**4. ⭐⭐ `pred EMA` = 0.04 on Maze — a live input to backlog row 4 / MM-E1.**
A **13×** gap against `pred VCReg` (0.54) in the same environment, and the worst cell in the table. Their EMA carries **no τ ramp** — exactly row 4's stated caveat (*"our fixed τ=0.996 has no published operating point; every EMA-teacher line ramps τ"*). ⇒ **row 4's τ-ramp precondition now has a published failure number behind it.** ⚠️ This measures *planning*, not drift; it does not refute EMA's measured drift benefit (0.4531 → 0.36–0.40). **Both can be true, and that is the point: EMA may buy drift and cost planning.**

**5. ⭐ FS9-4's pre-committed read has a published precedent.**
FS9-4 committed in advance: *"if the divergence bound for our best arm already admits a planner-cost gap larger than the entire fan-vs-value effect we could hope for, the finding is that I-1 is blocked on REPRESENTATION not on scoring."* **This paper reaches that conclusion independently and supplies the fix.** FS9-4's Stage-0 pre-screen remains the right next step and is now better-aimed.

**6. ⛔ A deployment finding that outranks the value question on our budget.**
`PUBLISHED` · Drive-HWM Table III, verbatim: `Tavg` **84.8 ms**, `Tpeak` **107.2 ms**, `Tf` 81.6, `Ts` 25.6, `N` 8. Cost model: `Tavg = Tf + Ts/N`, `Tpeak = Tf + Ts`.

⭐⭐ This is the published implementation of backlog **P-7**'s requirement (*temporal abstraction is a deployment requirement, not an efficiency preference*). ⛔ **And its warning: the advertised 84.8 ms is a MEAN; the tick on which the slow rung fires costs 107.2 ms and misses a 100 ms budget.** A real-time driving budget is a **bound**. ⇒ **every TanitAD temporal-abstraction design must report and be gated on `Tpeak`, never `Tavg`.**

---

## THE FIVE DIMENSIONS (deep-read item: `2601.00844`)

1. **RELEVANCE ⭐⭐⭐** — injected row **I-1** directly; FS9-4; GS-2; backlog row 4 via F4.
2. **CONSEQUENCE** — I-1's quality half is answered *in the published literature* with a method we cannot run as published. The row does not close; it **re-aims** from "value head vs fan scoring" to "is our latent's geometry the binding constraint, and is the trunk-freeze the reason?"
3. **COMBINATION** — with our own COST measurement (breadth 5.94× cheaper than depth; value head 1.2 % of budget), the picture inverts: **scoring was never the expensive part and is not the weak part either.** With GS-2 and AITS it completes a three-instance pattern about the frozen trunk. With F3's hierarchy remark it links to row 18.
4. **CHANCES / RISKS** — *Chance:* one design decision (a partial or late trunk unfreeze) unblocks three separate published families at once; that is unusual leverage. *Risks, both author-stated:* (a) *"our IQL approach is known to be biased in random environments"* and *"prediction-based methods are indeed expected to be more robust to stochasticity"* — **driving is highly stochastic, a named reason this may not transfer**; (b) evidence is a 7-page workshop poster on a 2-D wall and a maze, so transfer to a 256×640 driving corpus is `HYPOTHESIS`.
5. **EXPERIMENT (pre-registerable, 0 GPU)** — **the value-geometry screen on banked v7 latents.** Fit `V(s,g) = -‖z_s - z_g‖` against a goal-reaching cost on banked latents from the frozen trunk, and measure rank correlation with realised time-to-goal. **Controls, mandatory:** a constant-only predictor that must read the no-information value, a shuffled-goal control that must read chance, and a raw-input floor. ⛔ **Committed in advance: if the frozen latent's distance already correlates with time-to-goal at ρ ≥ 0.5, the geometry is adequate and I-1 returns to being a scoring question; if it reads at or near the shuffled control, the geometry IS the binding constraint, I-1's quality half is blocked on the trunk-freeze, and no value-head arm runs this cycle.** Report `n`.

---

## WHAT THIS CHANGES FOR TANITAD — ≤3 recommendations

1. ⭐⭐⭐ **Do not build the bolted-on value head.** It is the variant the published record measures as worst (5 of 6 cells). Run the 0-GPU value-geometry screen above first.
2. ⭐⭐ **Put the trunk-freeze cost in front of the v7f decision explicitly.** Three published families — LDAD, AITS, IQL-value — are excluded by it. If a partial or late unfreeze is reachable cheaply, keep it reachable.
3. ⭐ **Gate every temporal-abstraction design on `Tpeak`, not `Tavg`.** Add `Tpeak` to the quality-per-FLOP ledger (backlog row 34).

## I-1 backlog motion

**`served 2026-09-10 → TanitAD Research Lab/Deployment & Optimization/Research/2026-09-10-i1-value-guided-latent/RESULT.md`** — ⚠️ **the row STAYS OPEN and is RE-AIMED.** The quality half is answered in the literature but the winning method is inapplicable on a frozen trunk; the open question is now the latent's value-geometry, testable at 0 GPU by the screen above.

## Escalations

- **To the Master Mind (v7f freeze, blocking):** the trunk-freeze now has three measured exclusions. Record the decision and its cost explicitly.
- **To the PI:** nothing blocking. FYI that the top injected row's quality half turned out to be a representation question, which is what FS9-4 pre-registered as one of its two outcomes.
