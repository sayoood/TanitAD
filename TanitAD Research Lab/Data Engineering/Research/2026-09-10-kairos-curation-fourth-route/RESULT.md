<title>Kairos curation, reached at the fifth route — the claim is real, the implementation is not</title>

# Kairos's control-relevant curation: reached at the fifth route, and it is a proposal

`TanitAD Research Lab · Data Engineering · 2026-09-10 · LAB-RUN-011`
`Serves: register debt D-8 (blocking, three failed routes) · FS9-8 · FS9-3 (clip-value score-and-floor) · LAB_BACKLOG row 22 (H-DATA-1) · the 2026-09-09 control-relevant-clip-value package`
`Search log: raw/search_log.md · Frontier scan F5, F6`

---

## FINDINGS

**1. ⛔⭐⭐ AN ARXIV ID IS NOT A DOCUMENT. Reading the wrong version would have filed a false REFUTED.**
`MEASURED` (full-text greps of both PDFs, downloaded and extracted locally).

| | `2606.16533v1` | `2606.16533v3` |
|---|---|---|
| pages | 90 | **119** |
| title | *A Native World Model Stack for Physical AI* | *A **Regret-Aware** Native World-Action Model Stack for Physical AI* |
| `control-relevant` | **1** (an unrelated VideoDiT ablation) | **42** |
| `two-level` | 3 — **all "two-level batching"**, an I/O technique | claim present verbatim |
| `control-informative` | 0 | 1 |
| `regret` | **0** | 96 |

The paper **changed its own title** between versions and grew by 29 pages. A grep of v1 returns a clean, confident, and **wrong** absence for a claim that is verbatim in v3.

⇒ ⛔ **New hygiene rule earned: every banked or cited arXiv primary carries its VERSION (`vN`) and retrieval date, and an absence claim on arXiv is invalid unless it names the version probed.** This is the `w120` identifier class one level up — **a version number is a token silently standing in for "the paper"**, exactly like a suffixed number standing in for a count.

**2. ⭐ THE RETRIEVAL BLOCK WAS SIZE, NOT ACCESS — route 5 works and should be the standard.**
Routes 1–4 (fetch-tool PDF, landing page, HTML, HTML v1) all failed the same way: truncation mid-§4.2. **Route 5 — a byte download via `urllib` followed by local PyMuPDF extraction — succeeded**: 42,560,348 B for v1 and 43,862,955 B for v3, both with a `%PDF-` header and a valid `%%EOF`. ⇒ **the three "unreachable" verdicts across two passes were a size limit in one tool, on a document that was always fully public.**

**3. ⛔⛔ THE CLAIM IS A PROPOSAL. Level 2 is unbuilt.**
`PUBLISHED` · `2606.16533v3`, verbatim:

> *"However, for Physical AI, quality filtering is only the first layer. A visually clean clip is not necessarily control-informative. Therefore, Kairos interprets data curation as a two-level process: basic quality filtering followed by control-relevant event filtering."*

> ⛔ *"The second level **would** prioritize clips according to control information density. **The current pipeline does not yet compute CID directly**; future iterations should explicitly target: …"*

> ⛔ *"In the current Kairos report, these control-relevant filters **should be presented as a guiding extension** built on top of the existing curation pipeline."*

⇒ **There is no CID computation, no retention rate for level 2, and no ablation of curation against downstream model quality anywhere in the paper.** The claim is `PUBLISHED` **as a design intention** and `UNSUPPORTED-AS-STATED` **as evidence that control-relevance filtering works**. ⛔ **It may not decide a GPU-day, and any TanitAD document citing Kairos as evidence for control-relevance filtering must be corrected — including the 2026-09-09 `control-relevant-clip-value` package.**

**4. ⭐⭐ What DOES transfer, and it is worth more than the claim was: the six-family taxonomy.**
Verbatim from v3, the events level 2 *would* target:

| family | Kairos (robotics) | driving analogue on our corpus |
|---|---|---|
| **Near-boundary failures** | marginal grasp failure, near slip that becomes a drop, near collision that becomes contact | TTC below threshold that becomes a contact; lane departure that crosses |
| **Recovery events** | regrasping, repositioning, replanning, human correction, retry | corrective steer after drift; brake-then-resume; aborted lane change |
| **Near-boundary successes** | marginal grasps that stay stable, near slips corrected, near collisions avoided | low-TTC encounters resolved without contact; margin-preserving corrections |
| **Contact transitions** | first contact, loss of contact, grasp closure, slip | lead-vehicle acquisition and loss; gap open/close; merge entry |
| **Safety / anomaly events** | human proximity, excessive force, irreversible state change | VRU proximity; hard decel; irreversible commitment (junction entry) |
| **Long-horizon dependencies** | delayed failure, multi-step dependency, hidden object state | occluded agent that later matters; multi-step route commitment |

Their argument for why these are the valuable clips, verbatim: *"A model that never sees near-boundary failures may generate clean success futures but remain unable to predict where deployment will break. A model that never sees recovery may fail to plan after mistakes."*

⭐ **FS9-3 and row 22 (H-DATA-1) currently lack an operational definition of "control-relevant clip". This supplies one, from a serious industrial programme, with every family mapping onto quantities our corpus already carries.** ⇒ **TanitAD can run the measurement Kairos deferred.** That is a differentiator in the shape the PI asked for on 09-05: a solve, not a refutation.

**5. What the paper DOES implement (level 1), with its numbers.**
`PUBLISHED` · shot segmentation with PySceneDetect, *"over 95 % segmentation precision and 80 % recall"* (77.44 % recall after throughput optimisation); segments kept at **5–40 s**, shots >40 s split into 20-s clips, <5 s discarded; corrupted/duplicate/short clips removed to reach *"several millions of hours of valid raw video"* and *"hundreds of millions of standardized video clips"*. Level-1 filters: aesthetic score (CLIP backbone + MLP head, thresholded), motion score, plus temporal-coherence, safety and redundancy filters. ⚠️ **All thresholds are unstated and no filter is ablated.**

**6. ⭐ Kairos's own concession — our open-vs-closed-loop ruling, stated by an industrial programme.**
Verbatim: *"Direct validation of real-robot closed-loop regret reduction, including rollout correlation, failure prediction, safety filtering, recovery learning, and measurable policy improvement from imagined experience, remains an important [direction]"* — and their results are offered as *"proxy evidence for regret-relevant capabilities"*. ⭐ **A full-stack world-model programme conceding it has proxy evidence only.** Our binding ruling (a planner feeding its own predictor is still open loop) is the same statement.

---

## THE FIVE DIMENSIONS (deep-read item: `2606.16533v3`)

1. **RELEVANCE ⭐⭐⭐** — D-8 blocked FS9-3 and row 22; the DataFlyWheel's selection-signal thesis rests on exactly this question.
2. **CONSEQUENCE** — D-8 discharges. The claim's *evidential* value goes to zero; its *taxonomic* value is high and immediately usable in FS9-3's design. One prior package needs a citation correction.
3. **COMBINATION** — pairs with FS9-3's pre-committed kinematic floor: **their six families are hypotheses about what makes a clip valuable; our score-and-floor design is the instrument that can test them.** Together they make a real experiment out of two halves that were each incomplete. Also pairs with row 22's MOSAIC framing (80 % less data at matched performance) — Kairos supplies the *what*, MOSAIC the *how much*.
4. **CHANCES / RISKS** — *Chance:* an operational definition we did not have, free, from a programme that will not measure it. *Risks:* (a) the taxonomy is robotic-manipulation-native and three of six families (contact transitions above all) map onto driving only by analogy, so the mapping table above is `HYPOTHESIS` and must be validated per family; (b) rare-event families are rare **by construction** — a corpus-rate census must precede any selection arm, or we will design a filter for events that occur too seldom to train on.
5. **EXPERIMENT (pre-registerable, 0 GPU)** — **corpus-rate census of the six families on parity-train.** For each family define one detector from quantities the corpus already carries (TTC, decel, lateral offset, lead acquisition/loss, occlusion), and report the per-family clip rate over all 2,376 parity episodes with `n` per cell. ⛔ **Committed in advance: any family whose rate is < 1 % of clips is dropped from the FS9-3 design as untrainable at our corpus size and we say so; any family above ~5 % becomes a stratum in the FS9-3 score-and-floor arm.** Controls: a random-clip baseline rate and a shuffled-detector control that must read the base rate.

---

## WHAT THIS CHANGES FOR TANITAD — ≤3 recommendations

1. ⭐⭐⭐ **Run the corpus-rate census before FS9-3's scoring arm.** It costs no GPU, it decides which families are trainable at all, and it prevents designing a filter for events we do not have.
2. ⛔ **Correct every citation of Kairos as evidence for control-relevance curation**, starting with the 2026-09-09 `control-relevant-clip-value` package. The paper says the pipeline *does not yet compute CID*.
3. ⭐ **Adopt the version-stamp rule (F1) and route 5 as the standard PDF retrieval path.** Both were paid for today.

## Escalations

- **To the Master Mind:** D-8 discharges; one prior package needs a citation correction; the version-stamp rule is a mechanical change to `kb_add.py`'s record (a `version` field), which is a tooling decision.
- **To the PI:** nothing blocking.
