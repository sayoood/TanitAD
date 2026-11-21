# LAB RUN 001 — Data Engineering — literature pass + ideation

`TanitAD Research Lab, daily run 001. Run label 2026-08-22 (Master Mind trigger);
executed 2026-08-23 wall-clock (gap flagged per the narrative-clock rule).
Bounded first pass: literature only, no GPU (Thor training; dev-box experiment
half starts with run 002). Agenda targets: RESEARCH_AGENDA.md Field 1, items 1–2.`

## Findings (2025–2026 literature vs agenda)

### F1.1 — World modeling AMPLIFIES the data scaling law ⭐ banked
- **DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving** —
  arXiv **2510.12796** (Oct 2025, rev Dec 2025). **PUBLISHED, BANKED
  (library key `2510.12796`, tag `data-scaling`)**.
- **Finding (one line):** adding world-model future-image prediction as a dense
  self-supervised auxiliary to a driving VLA does not just add points — it
  *steepens the data scaling curve* (gains accelerate with corpus size; NAVSIM
  v1/v2 + a 680× larger in-house corpus; beats BEV and VLA baselines).
- **Impact on TanitAD:** direct published support for G1's thesis *from the data
  side* — a WM objective converts additional data into capability at a better
  exponent than action-only supervision. The data moat and the WM architecture
  are multiplicative, not separate bets.
- **What it would change:** agenda item 1's scaling-law fits on OUR corpora must
  be run twice — with and without the WM objective — because the *exponent
  itself* is treatment-dependent. Any such fit obeys the exponent rule (fit
  window, R², n; no bare exponents; per CLAUDE.md).

### F1.2 — Scaling-AWARE data selection (per-domain exponents drive collection)
- **Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems
  (MOSAIC)** — arXiv **2604.08366** (Apr 2026). **PUBLISHED (abstract verified);
  NOT banked — secondary for registry purposes.**
- **Finding:** partition the pool into domains → fit a neural scaling law per
  domain against the eval metric → iteratively collect from the domain with the
  best marginal gain; matches diverse baselines on **EPDMS with up to 80 % less
  data**.
- **Impact:** this is the missing *procedure* for agenda item 1's "minimum data
  for target metric" recipe — data valuation via per-domain scaling slopes, not
  static heuristics.
- **What it would change:** our data-value estimator should output *per-domain
  marginal-gain curves* (domains = our situation/scenario vocabulary), and
  TanitDataSetCreator (P8) should expose "collect-next-from" recommendations.

### F1.3 — Amount scales open-loop; DISTRIBUTION decides closed-loop
- **Data Scaling Laws for Imitation Learning-Based End-to-End Autonomous
  Driving** — arXiv **2412.02689** (Dec 2024; kept despite predating the 2025
  window — it is the foundational AD measurement). **PUBLISHED (abstract-level);
  NOT banked — secondary.**
- **Finding:** ~4M demos / 30k+ h, 23 scenario types: performance follows a
  power law in data amount **open-loop but NOT closed-loop**; small additions of
  long-tail data move their scenarios disproportionately.
- **Impact:** published backing for our T1-primary doctrine from the data angle —
  open-loop scaling fits are the wrong instrument for deciding data spend.
- **What it would change:** every scaling fit we publish carries its eval tier;
  only T1-tier fits are decision-grade for data acquisition.

## Ideation — our own hypothesis

**H-DATA-1 (OPEN, proposed).** *Frozen-feature leverage predicts data value for
WM training.* The marginal value of a clip for our world model is predicted by
the **rank contribution of its frames in frozen-encoder (DINOv3) feature
space**: clips whose latents have high leverage on the corpus covariance (span
under-represented directions) buy more val-side participation AND decodability
per episode than random clips.

- **Cheapest discriminating experiment:** we already hold DINOv3 token caches
  and the participation instrument. (1) Score episodes by mean leverage on the
  frozen covariance; (2) train two v7-tiny arms on top-K-leverage vs random-K
  (same K, same steps) — ⚠️ this re-selects episodes, so it is **explicitly
  flagged NON-PARITY** (experiment-internal comparison only, never cross-arm
  with parity runs; DataFlyWheel boundary respected); (3) read val participation
  + EM vs HOLD + decodability probes, with the constant control and pixel floor.
- **Outcome A (leverage wins by margin):** seed of the data-value estimator
  (agenda items 1+2); promotes to a work package with a SPEC.
- **Outcome B (no separation):** frozen-space leverage is ruled out as a value
  signal — the cheapest coreset family is eliminated, and curation research
  moves to loss/gradient-based scoring.
- Estimated cost: 2 × 29 min v7-tiny (dev-box GPU half, run 002+).

## Transfer note (charter §7 — never silent)

→ **TanitAD_DataFlyWheel:** F1.2's per-domain scaling-aware selection is the
proposed mechanism for the moat criterion ("curated beats random at equal
size"), and H-DATA-1 is the cheapest first test of it on our stack. F1.1 says
the data strategy and the WM objective interact — the moat proof should be run
on a WM-objective model, not a plain BC head. Accept/reject with reason
requested via the Master Mind.

## Evidence discipline

Banked primary verified by `tools/kb_add.py` sha256 at add time; unbanked
citations are marked secondary and are inadmissible for MODEL_REGISTRY.md or
the paper until banked. No numbers in this file enter the registry.
