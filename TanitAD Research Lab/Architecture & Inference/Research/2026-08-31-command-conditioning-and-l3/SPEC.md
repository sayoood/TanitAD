<title>SPEC — command conditioning, the L3 dissociation, and the horizon ladder (2026-08-31)</title>

# SPEC — E-LAB-ARCH-0831

`TanitAD Research Lab · Architecture & Inference · 2026-08-31`
`Seeds: Project Steering/V7_LAUNCH_GATE.md — P2(b), P5/L3, P4.`
`Class: LITERATURE work package. No GPU, no Thor, no spend.`

---

## 1. What & why

`V7_LAUNCH_GATE.md` (written 2026-08-31, PI-binding) names three items this package is
aimed at. Each is stated there as an **open question with no measurement**, which is
exactly the shape a literature pass can move:

| gate item | the gate's own words | what a literature pass can add |
|---|---|---|
| **P2(b)** | *"our 'action' is realised motion, not a command … a genuine command channel has never been tested"* — state ⛔ **NEVER TESTED**; blocked on a corpus (comma2k19) that is neither on Thor nor geometry-compatible | Does the published record contain (i) a driving world model whose action channel IS an independently measured command, and (ii) a documented mechanism by which a low-dimensional command gets ignored, with a fix? |
| **P5 / L3** | *"does `zhat` beat `z_t` on the same targets, paired, \|t\| ≥ 2.9? ⇒ does the PREDICTOR add anything over simply looking at now?"* — the *"dissociation gate, and the actual v7 bar"* | Is there a published name, protocol and control for this comparison, so our bar is precedented rather than invented? Is there a published mechanism for a predictor that restates instead of transports? |
| **P4** | strategic (8–30 s) unreached; *"the temporal-abstraction ladder … not a longer flat rollout"* | What did 2026 measure about temporal abstraction — where it pays, and where it is documented to fail? |

⛔ **Anti-duplication preflight (run before any search).** The programme has already
answered several neighbouring questions and re-reporting one would be worse than silence:

- **MM-E12** already refuted *"the action is redundant given the scene"* — R²(action \| scene)
  sits at the no-information value on both 30k arms, against two controls. ⇒ this package
  must NOT re-argue redundancy-against-the-scene.
- **MM-E11** already tested *"restore the missing objective"* — the ratio **fell 0.40×**.
  ⇒ "add an action objective" is not a new recommendation on its own.
- **2026-08-30-data-efficiency-thesis** already ran the data-efficiency literature,
  including Sorscher (`2206.14486`) and the context/target frame-overlap shortcut.
  ⇒ off-limits here.
- **2026-08-29-ema-warmup-rollout-composition** already ran EMA-teacher scheduling and
  one-step-vs-multi-step rollout composition. ⇒ off-limits here.

## 2. Hypotheses (IDs), with success criteria committed IN ADVANCE

**H-LAB-CMD-1.** *There exists published, primary evidence that a low-dimensional command
channel supplied to a driving policy by concatenation-style conditioning is systematically
under-used, together with a named fix.*
- **SUPPORTED if** a primary source states the failure in its own words AND reports a
  quantitative comparison between two conditioning mechanisms.
- **REFUTED if** the only available statements are secondary summaries, or no quantitative
  comparison exists ⇒ then P2(b)'s premise stands unchallenged and the gate's plan holds.

**H-LAB-CMD-2.** *The provenance of the action channel (command vs realised motion) is the
binding constraint on action-sensitivity.*
- **SUPPORTED if** the literature attributes action-insensitivity to channel provenance.
- **REFUTED if** the literature identifies a different binding constraint that would hold
  even with a perfect command ⇒ then **P2(b) is mis-aimed** and the gate's candidate table
  is incomplete. ⚠️ This is the outcome that would change a spending decision, so it is
  committed here before the search, not after.

**H-LAB-L3-1.** *Our L3 bar (`zhat` must beat `z_t`) is precedented in the literature under
some name, with a control.*
- **SUPPORTED if** a primary source uses a same-shaped baseline and names it.
- **REFUTED if** no such baseline is found ⇒ report that our bar is unprecedented, which is
  a claim about novelty and must then be defensible.

**H-LAB-P4-1.** *Temporal abstraction is measured to help at long horizons.*
- **SUPPORTED if** ≥1 primary reports a quantitative long-horizon gain.
- ⚠️ **Symmetric criterion, committed in advance:** if a primary also reports where
  hierarchy FAILS, that must be reported with equal prominence. A one-sided read of the
  hierarchy literature is the failure mode this criterion exists to prevent.

## 3. Method and admissibility

- **Primary sources only.** arXiv `abs`/full-text (ar5iv/HTML), official vendor docs, or
  source code. ⛔ Not aggregator summaries. Every paper reported here was opened by me.
- **Bank every cited primary** via `tools/kb_add.py` (Agent Operating Standard §Research
  banking). A citation not banked is `PUBLISHED-SECONDARY` and inadmissible.
- **Evidence class on every claim** — `PUBLISHED` (their measurement, quoted verbatim) ·
  `MEASURED` (ours, artifact named) · `INFERRED` (my transfer argument).
- ⛔ **Transfer claims are INFERRED, never PUBLISHED.** A paper measuring a ratio on its own
  benchmark says nothing about our arms; the transfer is an argument I make and own.

## 4. Why `tests/` and `code/` are absent

This package builds no instrument and runs no compute. Per §3 the directories are the
schema for work packages that produce artifacts; here the quotable layer is `raw/QUOTES.md`
— the verbatim passages with source and section — which is what makes every claim in
`RESULT.md` re-checkable without re-opening the web. Stating the absence beats shipping an
empty directory that implies a test exists.
