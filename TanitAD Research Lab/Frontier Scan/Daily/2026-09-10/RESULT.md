<title>Frontier scan 2026-09-10 — the hierarchy edge has a price tag, and a version number nearly cost us a false refutation</title>

# FRONTIER SCAN — 2026-09-10 (LAB-RUN-011)

`TanitAD Research Lab · Charter §2–§4 + v2 amendment §7–§8. Search log: raw/search_log.md`
`Deep-read budget spent: THREE full texts (Drive-HWM 2609.03572 · Value-JEPA 2601.00844 · Kairos 2606.16533 v3) + one HTML full text (LeWM 2603.19312) + one adjudicated Band-D primary.`

---

## FINDINGS FIRST

| # | finding | evidence class | what it moves |
|---|---|---|---|
| **F1** | ⭐⭐⭐ **The newest hierarchical driving WM buys +0.8 PDMS over its own flat control — and is NOT parameter-matched.** Backlog row 18 / H1b **still has no matched-params published datapoint.** | `PUBLISHED` (`2609.03572` Table IV, full text) | row 18 · GS-5 · the three-planner directive |
| **F2** | ⭐⭐⭐ **The navhard/navtest split partition is CLOSED by primary counts** — navhard **450 S1 / 5462 S2**, navtest **≈12,000**. The 84.8–89.3 cluster is **navtest**, not navhard. ⛔ **Residue: PDM-Closed itself reads 51.3 AND 56.6 on "navhard".** | `PUBLISHED` (`2506.04218` §, the NAVSIM v2 authors' own paper) | **D-10** · FS9-2 · row 32 · T-4 |
| **F3** | ⭐⭐⭐ **A learned value function beats prediction-shaped representations for planning — but ONLY as an ENCODER-shaping loss, and combining it with the prediction loss is WORSE in 5 of 6 cells.** Our frozen trunk blocks the winning variant. | `PUBLISHED` (`2601.00844` Table 2, full text) | **injected I-1** · FS9-4 · GS-2 |
| **F4** | ⭐⭐⭐ **`pred EMA` reads 0.04 planning success on Maze** where VCReg reads 0.54 — a non-ramped EMA JEPA is the worst of ten variants for **planning**. | `PUBLISHED` (`2601.00844` Table 2) | **row 4 / MM-E1** (the live EMA-vs-SALT decision) |
| **F5** | ⛔⭐⭐ **An arXiv id is not a document: Kairos v1 (90 pp.) does NOT contain the claim; v3 (119 pp.) does.** Reading v1 would have filed a false `REFUTED`. | `MEASURED` (full-text greps of both versions) | **D-8 discharged** · a new hygiene rule |
| **F6** | ⛔ **Kairos's two-level control-relevant curation is a PROPOSAL, not an implementation** — *"The current pipeline does not yet compute CID directly"*. The RELAYED claim is `UNSUPPORTED-AS-STATED` **as evidence**, but its six-family taxonomy transfers for free. | `PUBLISHED` (`2606.16533` v3, verbatim) | **FS9-3** · row 22 (H-DATA-1) · the 09-09 DataEng package |
| **F7** | ⭐⭐ **LeWM conditions actions by AdaLN — layer-internal, the FiLM family — and reports NO action-sensitivity control at all.** Same interface as v7; same measured symptom, by three independent critics. | `PUBLISHED` (`2603.19312` full text) | **D-9 discharged** · **GS-3 priority RISES** |
| **F8** | ⭐⭐ **Amortisation fixes the AVERAGE, not the BOUND: Drive-HWM is 84.8 ms mean but 107.2 ms PEAK.** A real-time driving budget is a worst-case budget. | `PUBLISHED` (`2609.03572` Table III, verbatim) | **P-7** · the 100 ms budget · B3 |
| **F9** | ⭐ **NVIDIA's closed-loop post-training doctrine post carries ZERO numbers and ZERO ablations** — and ≥2 independent sources contest the mechanism. Verdict `CONTESTED`. | `PUBLISHED-BLOG` + counter-search | register **A16-1…A16-3** |
| **F10** | ⭐ **B3 (efficient decoding) finally has a transfer path after four passes** — and it is temporal abstraction, not token serving. | derived from F8 | B3 ledger |

---

## F1 — The hierarchy edge, priced and unmatched (A1 · `2609.03572` full text)

**Drive-HWM: Hierarchical World Models for Dynamic-Latent Guided Autonomous Driving**, Fan, Zhang, Wu, Wang, Jin, Zhao, Zhu, Yan — arXiv **2609.03572**, submitted **2026-09-03**. A *slow–fast* world model: the slow model predicts multi-step future representations (Dynamic-Aware Latents from optical-flow prediction); the fast model predicts next frame + immediate action every step.

**Table IV (ablation), verbatim numbers:**

| configuration | NC | DAC | PDMS |
|---|---|---|---|
| Fast only | 99.3 | 97.4 | 93.0 |
| Slow only | 98.2 | 97.1 | 90.2 |
| Drive-HWM (K=8) | 99.6 | 99.0 | **93.8** |

1. **RELEVANCE ⭐⭐⭐** — backlog row 18 (*H1b, the hierarchy edge, has never been tested*) and the PI's three-planner-hierarchy directive. This is the closest published matched-system hierarchy-vs-flat ablation that exists.
2. **CONSEQUENCE** — two readings, and **both must be carried**: on the raw scale the hierarchy buys **+0.8 PDMS** (0.86 % relative); against *remaining headroom* it closes **0.8 of 7.0 = 11.4 %**. A programme claiming hierarchy as *the* differentiator must be able to say which reading it means.
3. ⛔ **COMBINATION — the decisive caveat.** The paper **states no parameter matching** for "Fast only" / "Slow only". The +0.8 therefore confounds *hierarchy* with *added capacity*. **This is the same shape as Mobileye's M-1/M-2 — an existence result, not an ablated one — which GS-5 already named.** ⇒ **Row 18 stays OPEN, and F1 is a reason it matters more, not less.**
4. **CHANCES / RISKS** — *Chance:* GS-5's positioning ("the modular-vs-E2E / hierarchy question is unresolved on both sides, and TanitAD is the programme that settles it") is now **strengthened by the newest paper in the field**, not weakened. *Risk:* if a params-matched rerun elsewhere lands near +0.8, our thesis is a small-effect thesis and the paper must say so before a reviewer does.
5. **EXPERIMENT (pre-registerable, 0 GPU to price)** — re-frame PREREG_E_EXP1 as a **params-matched** two-arm v7-tiny test, flat vs slow/fast, with the parameter count of each arm reported to ±1 %. ⛔ **Committed in advance: if the matched hierarchy arm gains < 1 % relative on our four families, "hierarchy is our differentiator" is downgraded from a thesis claim to an efficiency claim, and the paper is rewritten accordingly.**

⚠️ **V-5 unit refusal.** Drive-HWM sets `N = K = 8` but **gives no seconds-per-timestep**. Their K=8 is therefore **not** corroboration of our K=8 (16 s) rung in backlog **P-17**. Recorded as unit-incomparable; do not pair them.

---

## F2 — D-10: the split partition closes, the scoring basis does not (A5 · `2506.04218`)

**Primary, verbatim** (Pseudo-Simulation for Autonomous Driving — the NAVSIM v2 maintainers' own paper):

> *"Our public NAVSIM v2 leaderboard features challenging driving scenarios… It uses a subset of nuPlan that we refer to as navhard, involving 450 Stage 1 and 5462 Stage 2 observations."*

navhard leaderboard anchors from the same primary: **PDM-Closed 51.3 · Latent TransFuser 23.1 · MLP 12.7 · Constant Velocity 10.9.** EPDMS range is **[0, 1]** (so these are ×100). navtest is **≈12,000 samples** (Q8, secondary-confirmed).

⇒ **FS9-2's pre-committed read is MET on its first branch**: the counts partition cleanly into a ~5.5 k navhard population and a ~12 k navtest population. **The 84.8 / 85.1 / 86.1 / 86.4 / 89.3 cluster is navtest-EPDMS and may not be called navhard.**

⛔ **But the residue is real and it is the part that matters.** Our records carry PDM-Closed on navhard at **56.6** (LAB-RUN-008); the maintainers' primary reports **51.3**. **Two values for the same privileged baseline on the same named split.** ⇒ the split name alone does **not** pin the scoring basis — exactly D-10's original suspicion, surviving the count check.

⇒ **D-10: MECHANISM RESOLVED (09-09) · COUNTS RESOLVED (today) · SCORING BASIS STILL OPEN.** Guideline **T-4 stands**: no external EPDMS is quoted without a split stamp *and* a scoring-basis era.

⚠️ **Correction to our own record:** the 09-05 pass wrote *"three independent tables cap navhard at ≤ 45.0"*. The maintainers' primary puts PDM-Closed at **51.3** on navhard. **The ≤45.0 cap is superseded** and must not be re-quoted.

---

## F3 / F4 — Value-guided planning: the winner is an encoder loss (A2 · `2601.00844` full text) — **serves injected I-1**

Destrade, Bounou, Le Lidec, Ponce, **LeCun** — World Modeling Workshop 2026 poster, 7 pp. MPC with an **MPPI** optimiser; planning accuracy over **200** wall instances and **80** maze instances.

**Table 2, verbatim (planning success rate):**

| type | WS | WB | Maze |
|---|---|---|---|
| Contrastive | 0.49 | 0.59 | 0.50 |
| Regressive | 0.54 | 0.57 | 0.46 |
| **pred VCReg** (the standard-JEPA / DINO-WM-family baseline) | 0.55 | 0.89 | 0.54 |
| **pred EMA** | 0.46 | 0.43 | **0.04** |
| VF (IQL, encoder-only) | 0.63 | 0.94 | 0.49 |
| **VF quasi** (IQL + quasi-distance, encoder-only) | **0.71** | **0.96** | **0.63** |
| VF pred (joint) | 0.55 | 0.75 | 0.49 |
| VF quasi pred (joint) | 0.61 | 0.85 | 0.43 |
| VF VCReg | 0.49 | 0.75 | 0.39 |
| VF VCReg pred | 0.47 | 0.67 | 0.39 |

**Three results, in descending order of consequence for us:**

- ⛔ **The winning variants are "Sep" — the STATE ENCODER is trained with the value loss.** The method shapes latent *geometry* so that embedding distance approximates the negative goal-conditioned value. **On a frozen trunk it has nothing to shape.** This is the **third consecutive** action/value-identifiability family blocked by our frozen encoder (LDAD encoder-shaping → AITS encoder-side → now IQL/VF). ⭐ **That is no longer a coincidence; it is a design consequence, and it belongs in the v7f freeze decision.**
- ⛔ **Joint training is measurably WORSE than separate**: VF 0.63/0.94/0.49 → VF pred 0.55/0.75/0.49; VF quasi 0.71/0.96/0.63 → VF quasi pred 0.61/0.85/0.43. **Losing in 5 of 6 cells.** The paper: *"Learning representations using both a prediction loss and an IQL loss is less effective than using the latter loss alone."* ⇒ **the naive I-1 plan — bolt a value head onto the existing predictor objective — is the variant the published record measures as worst.**
- ⭐⭐ **`pred EMA` = 0.04 on Maze** against `pred VCReg` 0.54 on the same environment: a **13×** gap, and the worst cell in the table. Their EMA carries **no τ ramp** — precisely backlog row 4's caveat (*"our fixed τ=0.996 has no published operating point; every EMA-teacher line ramps τ"*). ⇒ **row 4's τ-ramp precondition now has a published failure number behind it, not just an argument.**

⭐ **And the paper independently proposes our hierarchy, with a mechanism.** Verbatim: *"This suggests that using a hierarchy of representation spaces, where higher levels model longer-range transitions or more coarsely sampled trajectories, may better capture distant relationships."* Their named cause: distant (state, next-state, goal) triplets are **sparsely sampled**, and the discounted value gradient **vanishes far from the goal**, so the signal-to-noise ratio collapses at range. **This is the three-planner directive, derived from outside driving, with a falsifiable mechanism attached.**

**CHANCES / RISKS.** *Chance:* a value-shaped latent is the one lever that attacks I-1's quality half at its root. *Risks, both stated by the authors:* (a) *"our IQL approach is known to be biased in random environments"* and *"prediction-based methods are indeed expected to be more robust to stochasticity"* — **driving is highly stochastic, which is a named reason this may not transfer**; (b) the evidence base is a 7-page workshop poster on a 2-D wall and a maze — transfer to a 256×640 driving corpus is `HYPOTHESIS`, not `PUBLISHED`.

**EXPERIMENT** — see the Deployment package (`2026-09-10-i1-value-guided-latent`).

---

## F5 / F6 — Kairos: the version trap, and a taxonomy without a measurement (B9 · `2606.16533`) — **discharges D-8**

**F5 (process).** Three routes failed on 09-09; route 4 failed identically today; **route 5 (byte download + local extraction) succeeded** — the block was always **size**, never access. ⛔ **But route 5 returned `v1`, and the claim is only in `v3`.** Measured greps: v1 (90 pp.) `control-relevant` ×1 (an unrelated VideoDiT ablation), `two-level` ×3 (all *two-level batching*, an I/O technique), `regret` ×**0** — while the arXiv title reads *"A Regret-Aware…"*. v3 (119 pp.) `control-relevant` ×42, claim present verbatim. **A confident false `REFUTED` was one step away.**

**F6 (content), verbatim from v3:**

> *"However, for Physical AI, quality filtering is only the first layer. A visually clean clip is not necessarily control-informative. Therefore, Kairos interprets data curation as a two-level process: basic quality filtering followed by control-relevant event filtering."*

> ⛔ *"The second level **would** prioritize clips according to control information density. **The current pipeline does not yet compute CID directly**; future iterations should explicitly target: …"*

> ⛔ *"In the current Kairos report, these control-relevant filters **should be presented as a guiding extension** built on top of the existing curation pipeline."*

⇒ **Level 2 is unbuilt.** No CID computation, no retention rate, no ablation of curation against downstream model quality. The claim is `PUBLISHED` **as a design intention** and `UNSUPPORTED-AS-STATED` **as evidence that control-relevance filtering works**.

⭐ **What still transfers, and it is worth more than the claim was:** their **six control-relevant event families** — near-boundary failures · recovery events · near-boundary successes · contact transitions · safety/anomaly events · long-horizon dependencies — are a ready-made operational definition of "control-relevant event" that **FS9-3 and row 22 (H-DATA-1) currently lack**, and every family has a driving analogue. ⇒ **TanitAD can run the measurement Kairos deferred.** That is a differentiator in the shape the PI asked for on 09-05: a solve, not a refutation.

⭐ **Kairos's own concession**, verbatim: *"Direct validation of real-robot closed-loop regret reduction… remains an important [direction]"* — an industrial full-stack world-model programme conceding it has **proxy evidence only**, which is our open-vs-closed-loop binding ruling stated by someone else.

---

## F7 — LeWM: the same conditioning interface, the same silence (A2 · `2603.19312`) — **discharges D-9**

**LeWorldModel: Stable End-to-End JEPA from Pixels** — Maes, Le Lidec, Scieur, **LeCun**, Balestriero; 2026-03-13; ~15 M params; two loss terms (prediction + **SIGReg**, isotropic-Gaussian latent regulariser, λ=0.1); CEM planning, 300 samples, 30 iterations, horizon 5.

- ⭐⭐ **Action conditioning is AdaLN applied at each layer** — verbatim: *"Actions are incorporated into the predictor through Adaptive Layer Normalization (AdaLN) applied at each layer."* **That is layer-internal conditioning, the FiLM family — our v7 interface.**
- ⛔ **LeWM reports NO action-sensitivity control**: no hold-action, no zero-action, no shuffled-action. ⇒ **the action-insensitivity we have cited three times is a THIRD-PARTY measurement (Delta-JEPA, ATM, ACPC), never a self-report.** Our citations must now say so.
- ⭐⭐ **Two independent systems, same interface, same symptom.** LeWM: AdaLN, measured action-insensitive by three critics. v7: FiLM, measured action-insensitive (h=1 action/scene ratio **0.004**, all arms `LOSES_TO_HOLDV0`). **Our elimination of FiLM rested on "it trained", which was never evidence of innocence.** ⇒ **GS-3 (FiLM vs action-as-token at matched params) rises in priority: it is now the only branch with an independent corroborating instance.**
- ⚠️ Fair reading of the opponent's strength: LeWM's design thesis is that EMA and pre-trained encoders are **fragile crutches**, and it removes both while staying competitive at 15 M params and planning **48× faster** than DINO-WM. Our architecture uses one of the two crutches (a frozen pre-trained trunk) as a foundation.
- Concession: *"Planning with current latent world models remains restricted to short horizons."*

---

## F8 / F10 — The bound, not the mean (Deploy/B3 · `2609.03572` Table III)

**Table III, verbatim:**

| Method | N | Ts | Tf | Tpeak | Tavg | PDMS |
|---|---|---|---|---|---|---|
| DriveVLA-W0 | – | – | 117.8 | 117.8 | 117.8 | 93.0 |
| Fast Model Only | – | – | 81.6 | 81.6 | 81.6 | 93.0 |
| Drive-HWM (ours) | 8 | 25.6 | 81.6 | **107.2** | 84.8 | 93.8 |

⭐⭐ **This is the published implementation of backlog P-7's requirement** (*"temporal abstraction is a deployment requirement, not an efficiency preference"*) — the slow model runs once per 8 ticks, costing 25.6/8 = 3.2 ms amortised.

⛔ **And it is P-7's warning too.** The advertised **84.8 ms** is a *mean*; the tick on which the slow model fires costs **107.2 ms**. Against a 100 ms real-time budget the system **misses on every 8th tick**. A driving latency budget is a **bound**, not an average. ⇒ **any TanitAD temporal-abstraction design must report Tpeak, and be gated on Tpeak.**

⭐ **F10:** this closes B3's four-pass search for a transfer path. **B3's lever for us is not speculative decoding or KV caching — it is running the expensive rung less often**, and its cost model is `Tavg = Tf + Ts/N`, `Tpeak = Tf + Ts`. Ledgered.

---

## F9 — Band D: NVIDIA closed-loop post-training (adjudicated, 7 steps)

Full adjudication and its three register rows are in `../../../Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md` (rows **A16-1 … A16-3**). Headline: the post is a **how-to with operational specs only — no metric, no ablation, no A/B** — while ≥2 independent 2026 sources contest the mechanism it recommends. Verdict **CONTESTED**; it **binds on us only in its diagnosis, not in its prescription**.

---

## ⛔ COMPLETENESS DECLARATION (charter §6, fail-loudly clause)

| requirement | status |
|---|---|
| Band A deep-read | ✅ A1 (Drive-HWM, full text) · A2 (Value-JEPA full text + LeWM full text) · A5 (navhard primary). ⚠️ **A3 scanned, EMPTY at a 5th probe, no deep-read. A4 scanned only.** |
| Band B ≥3 deep-read | ✅ **B3** (F8/F10), **B5** (Band-D counter-search), **B9** (Kairos v3 full text) |
| Band C sweep | ⚠️ **PARTIAL.** C1, C2, C4 swept. ⛔ **C3 NOT scanned — declared, not concealed** (D-4 is with the PI at five failed routes) |
| Band D adjudication | ✅ one item, seven steps, counter-search recorded |
| ledger appends | ✅ A1, A2, A3, A5, B2→B3, B5, B9, C1/C2 |
| ≥2 full-text reads (V-2) | ✅ **four** |
| V-1 (library before web) | ⚠️ **VIOLATED, and measured: 7 of 9 of today's primaries were ALREADY BANKED**, including `2601.00844`, the pass's second-most consequential paper. Only `2609.03572` and `2606.16533` were new. **The library was checked after the searches, not before.** |
| 22-track scan | ⚠️ **PARTIAL — 18 of 22 tracks carry a query or a named sweep; C3 un-probed; B1/B6/B8/B11 swept only through other result sets.** Stated, not narrowed silently |
