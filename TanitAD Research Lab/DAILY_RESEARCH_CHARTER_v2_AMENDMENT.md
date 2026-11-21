<title>CHARTER v2 AMENDMENT — Band D, the adjudication protocol, and six value rules</title>

# CHARTER v2 — AMENDMENT

`TanitAD Research Lab · BINDING. Issued by the PI (Sayed), 2026-08-31, hours after v1.`
`Reads together with DAILY_RESEARCH_CHARTER.md — this amendment ADDS Band D and §7–§8; nothing in v1 is repealed.`

---

## 0. The PI's statement, verbatim

> *"now scan the 3 missing tracks and B13 first, try to augment your research strategy, there is a
> room to optimize strategy and value. I discovered e.g. today this article from Waymo describing
> their 10 most important lessons, this type of things are expected from our research lab to be
> identified but also to be critically evaluated both looking for evidence confirming these
> statements and other contradicting their content and derive in a scientific way the strengths and
> weaknesses of our opponents and derive strategic and tactical guidelines for our frontier
> programme."*

⛔ **The defect this fixes, stated plainly: the PI found the Waymo article, not the Lab.** v1's Band C
would have *surfaced* it and filed a one-line finding. That is identification without adjudication —
and a one-line summary of a competitor's doctrine is worth close to nothing. **v1 had no mechanism
for turning an opponent's public claim into a tested position.** Band D is that mechanism.

---

## 7. ⭐⭐ BAND D — OPPONENT DOCTRINE (new; ranks ABOVE Band B on any day a Band-D item exists)

**Trigger.** Any public statement of *doctrine* by a serious opponent — lessons-learned posts,
technical blogs, model cards making architectural arguments, keynote theses, safety frameworks.
Distinct from Band C (a *release*: what shipped) — Band D is *a claim about what is true*.

**Opponents tracked by name:** Waymo · Tesla · Wayve · NVIDIA (Alpamayo) · Waabi · Zoox · Mobileye ·
Chinese AV majors (DeepRoute, Momenta, Horizon) · the frontier labs where they touch embodied AI.

### 7.1 ⛔ The adjudication protocol — every load-bearing claim, no exceptions

A Band-D item is not summarised. Each of its claims goes through **all seven steps**:

| # | step | the rule |
|---|---|---|
| 1 | **CLASSIFY THE DOCUMENT** | who published it, when, into what competitive context, and what its evidence class is. A doctrine post is `PUBLISHED-BLOG`, never `PUBLISHED`. |
| 2 | **IS IT AN EXPERIMENT?** | ⛔ **State explicitly whether the claim is supported by a controlled ablation, an A/B, or a matched comparison — or whether it is attribution-from-exposure.** *"We drove N million miles and concluded X"* has **no counterfactual** and cannot separate *what we built* from *what is necessary*. This single question did more work than any other in the Waymo pass. |
| 3 | **SEEK CONFIRMING EVIDENCE** | ≥1 **independent** source. The opponent's own other publications do **not** count as independent. |
| 4 | ⭐ **SEEK CONTRADICTING EVIDENCE** | ≥1 independent source, **actively sought, not incidentally noticed**. A claim with no counter-search recorded is **not adjudicated** — it is repeated. This step is mandatory and its absence is a defect, not a shortfall. |
| 5 | **VERDICT** | `CONFIRMS-US` · `SUPPORTED` · `CONTESTED` · `UNSUPPORTED-AS-STATED` · `REFUTED`. With the reason in one sentence. |
| 6 | ⭐⭐ **DOES IT BIND ON US?** | **separate from whether it is true.** A true claim can fail to bind because our constraints differ. *(Waymo L1 is a capability argument about sensors; our vision-only rule is a leak-avoidance rule about admissibility. Both can be right — they do not meet. Conflating them would have read as a refutation of our own binding rule.)* |
| 7 | **GUIDELINE** | a **strategic** guideline (what the programme argues/believes) and/or a **tactical** one (what someone does next week), each traceable to the verdict. |

### 7.2 Two asymmetries that must be respected

⭐ **Look hardest for the CONCESSION.** The most valuable sentence in an advocacy document is the one
that costs its author something. *(Waymo L3's headline is "fewer, larger models are better"; the same
paragraph concedes "teacher-student models to optimize onboard compute" — they train large and
**deploy small**, which is our thesis, stated by our largest opponent.)* Concessions are worth more
than headlines because advocacy filters them out and they survive anyway.

⚠️ **Be fair, and record where the opponent is genuinely ahead.** An adjudication that finds only
weaknesses is not analysis, it is comfort. Waymo's exposure record, closed-loop evaluation culture,
and solved compression problem are real advantages and are named as such. **A Band-D package that
lists no opponent strength is INCOMPLETE.**

### 7.3 The durable artifact

`TanitAD Research Lab/Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md` — **append-only**, one row per
adjudicated claim: claim · opponent · date · verdict · binds-on-us · the evidence that would flip it.
⭐ **A verdict is re-opened when new evidence arrives**, which is why the register exists and a
one-off report does not.

---

## 8. ⭐ SIX VALUE RULES — where v1 wasted effort, measured on its own first pass

| # | rule | the waste it removes |
|---|---|---|
| **V-1** | ⛔ **Search the Library BEFORE the web.** `python tools/kb_add.py` reported *"already banked"* for **2 of 10** papers on the first pass — we re-found what we held. At 306 entries this fraction only grows. **Check the register and `library.json` first; a re-find is a free finding.** |
| **V-2** | ⭐ **Depth beats breadth on load-bearing items — and the pass must SAY which items were load-bearing.** The two full-text reads (DriveFuture, DriveWorld-VLA) produced every actionable finding of the first pass; the eight abstract-only banks produced context. ⇒ **Budget: ≥2 FULL-TEXT reads per day**, chosen for decision-relevance, and name them. |
| **V-3** | ⛔ **A claim that cannot be falsified by any measurement we could run does not enter a guideline.** State, per adopted claim, **the measurement that would overturn it**. Without this the Lab imports doctrine and calls it evidence — the exact `INHERITED` failure the operating standard bans. |
| **V-4** | ⭐ **Prefer findings that CHANGE A LIVE DECISION over findings that are merely true.** Rank the day's DEEP list by *"which open gate row, register claim, or PI-queue item does this move?"* An interesting paper touching nothing open is Band-B context, not a deep-read. |
| **V-5** | ⚠️ **Cross-check every opponent number against OUR metric definitions before it enters any table.** `minADE₆` is best-of-6 and is **not** our `fwd_ade`; an EPDMS is comparable only within one scoring-basis era. **The unit error is the most likely way an opponent number corrupts our registry.** |
| **V-6** | ⭐ **One contradiction found is worth more than five confirmations.** Confirmations are cheap and mostly redundant; a contradiction either kills a plan early or reveals a differentiator. **Weight the day's effort accordingly** — and this is why §7.1 step 4 is mandatory rather than encouraged. |

### 8.1 Revised daily budget (replaces v1 §6 step 4's allocation)

| priority | work | why in this order |
|---|---|---|
| **1** | **Band D** — every open opponent-doctrine item, adjudicated in full | highest value per token; it is the only band that changes what the programme *argues* |
| **2** | **≥2 FULL-TEXT deep reads** on the items touching live gate rows (V-2, V-4) | depth is where the actionable findings came from |
| **3** | **Band A** rotation | our own field |
| **4** | **≥3 Band B** neighbouring-discipline tracks, **staleness-ordered** | the PI's transfer mandate |
| **5** | **Band C** sweep | cheap, and it caught Alpamayo |
| **6** | ledgers, `TRACKS.md`, register, backlog motions | same turn, always |

⛔ **Unchanged and still binding: a pass with no Band-B deep-read, no Band-C sweep, or no ledger
append is INCOMPLETE and says so in its own summary.** Band D now joins that list: **an open Band-D
item left un-adjudicated is a failed pass**, because that is precisely the gap the PI had to fill
manually today.
