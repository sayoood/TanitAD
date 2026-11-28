<title>SPEC — frontier action-provenance sweep and NAVSIM comparability (2026-08-31)</title>

# SPEC — E-LAB-OPP-0831

`TanitAD Research Lab · Opponent Analysis (cross-filed to Benchmarks & Evals) · 2026-08-31`
`Seeds: V7_LAUNCH_GATE.md P2(b)/P4/P5 · NAVSIM–nuScenes comparability.`
`Class: COMPETITOR + BENCHMARK sweep, primary sources. 0 GPU, 0 spend, Thor untouched.`

---

## 1. What & why

The v7 gate asks three architecture questions. Opponent Analysis can answer a *fourth* that
none of the three can answer from inside: **what does the rest of the field actually do, and
does our problem look like a defect or like the field norm?** Two questions follow, plus one
that belongs to Benchmarks & Evals:

| Q | question | why it is decision-relevant today |
|---|---|---|
| **Q1** | For each frontier driving world model, is the action channel **measured independently of the realised path**, or **derived from the ego log**? | P2(b) asserts *"we have never given this model a command"*. If that is true of the whole field, the complaint is less novel and the test is more valuable. If competitors have solved it, we are behind on a solved problem. |
| **Q2** | How does the field **demonstrate** the action matters — and does any of them run a **control/null arm**? | ⛔ We claimed on 2026-08-30 that a **hold-action floor** is *"not found in any driving world model."* A novelty claim decays the moment someone publishes; it must be re-checked, not assumed. |
| **Q3** | Which recent work reaches **beyond ~5 s**, and by **temporal abstraction** or a longer flat rollout? | P4 needs a driving-domain precedent; the two 2026 hierarchy papers found by the Architecture package are both manipulation. |
| **Q4** | What has changed in **NAVSIM / nuScenes comparability**? | On 2026-08-29 this programme **pinned navhard-two-stage EPDMS** as a GO target. A pinned target whose scoring basis moved is a silent comparability defect. |

## 2. Hypotheses, with criteria committed IN ADVANCE

**H-LAB-OPP-1.** *Genuine command conditioning is rare-to-absent in frontier driving world models.*
- **SUPPORTED if** ≤1 of the swept models states an independently measured command channel.
- **REFUTED if** several do ⇒ P2(b) is a gap against the field, which is a *worse* finding for
  us and must be reported as such rather than softened.

**H-LAB-OPP-2.** *Our anti-echo novelty claim of 2026-08-30 still holds.*
- **HOLDS if** no swept driving world model runs a floor/null arm the model must beat.
- ⛔ **AT RISK if** any does ⇒ the claim must be **flagged for verification before the paper
  restates it.** ⚠️ Committed in advance: I may **not** resolve this from a relayed quote —
  a novelty claim may only be retracted or upheld against a source I have read myself.

**H-LAB-OPP-3.** *A driving-domain temporal-abstraction precedent exists.*
- **SUPPORTED if** a driving world model publishes an explicit multi-rate / hierarchical latent.

**H-LAB-OPP-4.** *NAVSIM's scoring basis has moved in a way that affects our pinned target.*
- **SUPPORTED if** the official changelog names a dated change touching `navhard_two_stage`
  or the EPDMS metric machinery.
- **REFUTED if** the protocol has been stable ⇒ our pinned target is safe as recorded.

## 3. ⛔ ADMISSIBILITY — the rule that governs this package

Part of this sweep was **delegated**. That creates a specific hazard: a relayed quote reads
exactly like a verified one once it is in a document. So this package uses **three tiers, and
they are never merged in a sentence**:

| tier | meaning | what it may be used for |
|---|---|---|
| ⭐ **VERIFIED** | I opened the primary myself and the quote is from the text I read; the retrieval method (abs / full text / official doc) is stated | anything, including a claim that changes a decision |
| **BANKED** | existence, title and date confirmed because `kb_add.py` fetched the PDF from arXiv | citing the paper as a *pointer*; naming its topic |
| ⚠️ **RELAYED** | a delegated sweep reported it; I have **not** read the source | ⛔ **may not be quoted, may not carry a number, and may not settle a novelty claim.** May only be recorded as *"a verification target"* |

- ⛔ **A novelty claim of ours may not be retracted OR upheld on RELAYED evidence.** If a
  relayed item threatens one, the output is *"flagged for verification"*, not a verdict.
- Vendor documentation (an official repository README, a model card) counts as **VERIFIED**
  when I fetch it myself, with the retrieval date; it is not bankable as a PDF, per the
  precedent set on 2026-08-29.

## 4. Why `tests/` and `code/` are absent

No instrument, no compute, no download. The quotable layer is `raw/QUOTES.md`, which carries
the tier and retrieval method of every passage.
