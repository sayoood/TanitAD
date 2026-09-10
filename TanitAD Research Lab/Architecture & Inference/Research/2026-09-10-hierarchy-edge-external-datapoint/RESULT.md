<title>The hierarchy edge finally has an external number — and it is not the one we needed</title>

# The hierarchy edge has an external number, and it is unmatched

`TanitAD Research Lab · Architecture & Inference · 2026-09-10 · LAB-RUN-011`
`Serves: LAB_BACKLOG row 18 (H1b hierarchy-edge UNTESTED) · row 5 (v7 head geometry) · GS-3 · GS-5 · the PI's three-planner-hierarchy directive (2026-07-21)`
`Search log: raw/search_log.md · Frontier scan: ../../../Frontier Scan/Daily/2026-09-10/RESULT.md F1, F7`

---

## FINDINGS

**1. ⭐⭐⭐ The closest published hierarchy-vs-flat ablation reports +0.8 PDMS — and states no parameter matching.**
`PUBLISHED` · Drive-HWM, arXiv **2609.03572**, 2026-09-03, Table IV (full text read).

| configuration | NC | DAC | PDMS |
|---|---|---|---|
| Fast only (the flat control) | 99.3 | 97.4 | 93.0 |
| Slow only | 98.2 | 97.1 | 90.2 |
| Drive-HWM, hierarchical, K=8 | 99.6 | 99.0 | **93.8** |

Two readings, and a paper that uses one without the other is misleading its reader:

| reading | value | what it licenses |
|---|---|---|
| absolute | +0.8 PDMS | "hierarchy helps a little" |
| fraction of remaining headroom (93.0 → 100) | 0.8 / 7.0 = **11.4 %** | "hierarchy takes an eighth of what is left" |

⛔ **The paper states no capacity match between "Fast only" and the full model.** The +0.8 therefore confounds *hierarchy* with *added parameters*. **This is an existence result, not an ablation** — structurally the same as Mobileye's M-1/M-2, which GS-5 already named as the field's characteristic weakness on this question.

⇒ **Backlog row 18 is NOT answered. It is sharpened, and it is now more valuable than it was yesterday**, because the newest and most directly relevant paper in the field reproduces the field's gap rather than closing it.

**2. ⭐⭐ Two independent systems now share our conditioning interface AND our symptom.**
`PUBLISHED` · LeWM, arXiv **2603.19312**, full text — *"Actions are incorporated into the predictor through Adaptive Layer Normalization (AdaLN) applied at each layer."*

| | conditioning | action-sensitivity | measured by |
|---|---|---|---|
| **TanitAD v7** | FiLM, layer-internal | h=1 action/scene ratio **0.004**; every arm `LOSES_TO_HOLDV0` | ourselves (`MEASURED`) |
| **LeWM** | AdaLN, layer-internal | reported action-insensitive | **three third parties** — Delta-JEPA, ATM, ACPC |

⛔ **LeWM itself reports no action-sensitivity control of any kind** — no hold-action, no zero-action, no shuffled-action. So the insensitivity we have cited three times is a critic's measurement, never a self-report, and our citations must say so. **Register debt D-9 is discharged by reading the primary; the correction it produces is a citation-provenance fix, not a retraction.**

⭐ **The consequence is a priority change.** Our elimination of the conditioning interface as a cause of P-2 rested on *"it trained"*, which was never evidence of innocence. **There is now an independent instance of the same interface exhibiting the same defect.** ⇒ **GS-3 (FiLM vs action-as-token at matched params) is the highest-value open Architecture arm, and its pre-committed read is unchanged.**

**3. ⚠️ Their K=8 is not our K=8.**
Drive-HWM sets `N = K = 8` and **gives no seconds-per-timestep**. Backlog **P-17** prices a K=8 (16 s) strategic rung. **The two are unit-incomparable and must not be paired** (rule V-5). Recorded so the coincidence cannot harden into corroboration.

**4. ⭐ Their stated limitation is our open lane.**
Verbatim: *"the current model does not explicitly capture multimodal futures or predictive uncertainty, which may limit its performance in ambiguous and rare driving scenarios."* Our REF-C fan and the B7 diffusion line address exactly this. A differentiator, named by the opponent.

---

## THE FIVE DIMENSIONS (deep-read item: `2609.03572`)

1. **RELEVANCE ⭐⭐⭐** — row 18 (H1b) is the programme's thesis hypothesis and has **no measured datapoint**; the three-planner directive assumes it; row 5 (drop h≥2 heads for composed h=1) sits inside the same design question.
2. **CONSEQUENCE** — row 18 stays OPEN with a sharper specification: the missing artifact is a **params-matched** hierarchy-vs-flat arm. GS-5's positioning claim is strengthened by a fresh instance.
3. **COMBINATION** — pairs with GS-5 (Mobileye existence-result critique) and with F3's independent hierarchy proposal from `2601.00844`, whose *mechanism* (sparse distant triplets, vanishing discounted-value gradient at range) gives the hierarchy a **reason** rather than an intuition. Two papers, two fields, one week apart, both arriving at hierarchy — one by ablation-without-matching, one by diagnosis.
4. **CHANCES / RISKS** — *Chance:* we can be the first to publish the matched test, and the field's newest paper documents that nobody has. *Risk:* a matched rerun may land near +0.8, which would make hierarchy an efficiency claim rather than a thesis claim. **That outcome must be pre-committed to now, before we run it, or we will re-describe it afterwards.**
5. **EXPERIMENT (pre-registered)** — re-frame `PREREG_E_EXP1` as a **two-arm params-matched** v7-tiny test: flat predictor vs slow/fast at equal parameter count (reported to ±1 %), on the four metric families, 3 seeds. ⛔ **Committed in advance: < 1 % relative gain ⇒ "hierarchy is our differentiator" is downgraded to an efficiency claim and the paper is rewritten; ≥ 3 % ⇒ row 18 closes in our favour and becomes the paper's lead result.** Controls: constant-predictor (must read the no-information value) and raw-input floor.

---

## WHAT THIS CHANGES FOR TANITAD — ≤3 recommendations

1. ⭐⭐⭐ **Raise GS-3 (FiLM vs action-as-token, matched params) to the top of the Architecture arm queue.** It is the only P-2 branch that now has an independent corroborating instance, and the interface it tests is shared with LeWM.
2. ⭐⭐ **Keep row 18 open and re-specify it as params-matched.** Add the parameter-count-to-±1 % requirement to the prereg. The field's newest paper shows that omitting it is the default failure.
3. ⚠️ **Fix the LeWM citation provenance wherever it appears.** Every place the programme says "LeWM is action-insensitive" must read "as measured by Delta-JEPA / ATM / ACPC; LeWM reports no action-sensitivity control."

## Escalations

- **To the Master Mind (v7f freeze):** GS-3's priority change, and the params-matching requirement for any hierarchy arm.
- **To the PI:** the pre-committed downgrade rule in Experiment 5 binds how the paper describes hierarchy. It is a positioning decision, not a lab one.
