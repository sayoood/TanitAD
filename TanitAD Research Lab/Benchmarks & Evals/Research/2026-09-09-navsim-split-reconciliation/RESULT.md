<title>Debt D-10: navhard and navtest EPDMS are two disjoint populations (2026-09-09)</title>

# The navhard EPDMS contradiction is resolved in MECHANISM: two populations, one metric name

`TanitAD Research Lab - Benchmarks & Evals - daily pass 2026-09-09 (LAB-RUN-010).`
`Serves register debt D-10 (BLOCKING), backlog row 32, FS5-3, L-15, and rule V-5.`

---

## Findings

### F1 - There are at least two EPDMS populations, and they do not overlap

D-10 was opened on 2026-09-05: three independent 2026 tables cap navhard at **<= 45.0 EPDMS** while our
records carry **55.5** (DriveFuture) and **56.3** (DrivoR). The register's note was *"three independent
papers do not miss a 56.3 that outranks them by 18 points."* Correct - and today a **third** cluster
appeared that explains why.

| cluster | values (EPDMS) | n / basis | source | class |
|---|---|---|---|---|
| **navhard** | 23.1-45.0 · 38.0 · 36.9 | navhard = **450 Stage-1 + 5,462 Stage-2 observations** | GuideFlow `2511.18729` (full text), IDOL `2605.31476`, RAP-DINO `2510.04333` | `PUBLISHED` (read 2026-09-05) |
| **"NAVSIM v2", 12k scenarios** | **89.3** Latent-WAM · **86.1** DriveVLA-W0 · **85.1** Epona · **84.8** World4Drive | *"12k evaluation scenarios"*, verbatim | `2603.24581` Table 1, **full text today** | `PUBLISHED` |
| **ours, unstamped** | 55.5 · 56.3 | **unknown** | `LEADERBOARD.md` / register | ⛔ `INHERITED`, provenance not recorded |

⭐ **The two published clusters are separated by roughly 40 EPDMS and do not overlap at all.** A gap that
size between four-paper and three-paper groups is not a capability spread; it is a **basis** difference.

### F2 - The mechanism, named by the benchmark's own maintainers

`PUBLISHED-BLOG, retrieval 2026-09-09.` The NAVSIM maintainers *"discourage the use of **self-reported
and unofficial 'NAVSIM v2' benchmark splits**"* and direct submitters to the leaderboard *"to ensure
both consistency and visibility of submissions."*

⭐ **The corruption path we hypothesised is documented by the people who own the metric.** That is as
close to a confirming independent source as this question admits.

### F3 - Latent-WAM names no split and claims no leaderboard

`MEASURED from the primary, full text.` The paper reports on *"12k evaluation scenarios"*, states
*"NAVSIM v2 extends PDMS to EPDMS with additional metrics for rule compliance"*, and **never names
navtest, navhard or navsafe**. It makes no reference to the official leaderboard; the numbers read as
self-evaluated. Its own comparison claim is internally consistent - *"outperforming the best prior
perception-free method by 3.2 EPDMS"*, which matches 89.3 against DriveVLA-W0's 86.1.

⛔ **Internally consistent and externally unplaceable.** This is the same shape as the 2026-09-07 label
builder that verified its buckets against its own rounded ladder: correct within itself, wrong for a
consumer who assumes a shared basis.

### F4 - Our own two numbers match NEITHER cluster, which is the part that should worry us

55.5 and 56.3 sit **above** the navhard ceiling (45.0) and **well below** the 12k-scenario floor (84.8).

⚠️ **A number that matches neither published population is not "in dispute" - its provenance is
unknown.** CW-1's resolution (2026-09-02) settled DriveFuture's *internal* consistency, and this is a
different question about the LEVEL. **Neither number is admissible in a comparability table until it
carries a split stamp.**

### F5 - The parameter counts carry the same defect, one axis over

`MEASURED, verbatim from 2603.24581:` *"The model contains **104M parameters at inference time**. During
training, an additional EMA encoder is introduced... **bringing the total to 191M, of which only 104M
are trainable**."*

Backlog row 32 says our efficiency wedge must beat *"~40 M, not 32 B"*. **Both the 40M DrivoR figure
(unverified - L-15) and this 104M are inference-time counts on models with larger training footprints.**
A TanitAD count quoted training-inclusive against either is a V-5 unit error, and it would understate us.

---

## Five-dimension analysis - `2603.24581` (the deep-read item)

| dimension | analysis |
|---|---|
| **RELEVANCE** | ⭐⭐⭐ direct. It is the only source that places a fourth, fifth and sixth number in the disputed band and thereby converts D-10 from a contradiction into a classification. Blocks G3 (beat SOTA on community benchmarks) until settled. |
| **CONSEQUENCE** | Moves debt **D-10** from OPEN-CONTRADICTION to **MECHANISM-RESOLVED / PROVENANCE-OPEN**. Keeps backlog **row 32 BARRED**, and adds a second reason (params) to the existing one (EPDMS). Adds guideline **T-4** and **T-5**. |
| **COMBINATION** | Combines with our 2026-09-05 finding that **navhard Stage 2 is itself a 3DGS counterfactual pseudo-closed-loop evaluation** - so navhard is *harder in kind*, not merely in difficulty, which is a physical reason for a 40-point gap rather than a bookkeeping one. It also combines with backlog row 14 (pseudo-simulation pilot): if navhard is the pseudo-closed-loop tier, **navhard is the tier our T1 doctrine says is comparable to us**, and the 84-89 cluster is the open-loop-flavoured one we would call unevaluated. |
| **CHANCES / RISKS** | **Upside:** we avoid publishing a comparison that a reviewer would dismantle in one line, and we gain a principled reason to report against navhard specifically. **Risks:** (a) the 12k figure is inferred to be navtest-class from its size, **not stated** - that inference could be wrong; (b) we have read Latent-WAM's results section, not its appendix, so a split statement may exist that we missed; (c) treating the 84-89 cluster as "the easy split" could itself be a scope error if those four papers share a different non-standard basis. |
| **EXPERIMENT** | **0 GPU, pre-registerable.** For each of the six external numbers we hold (55.5, 56.3, 89.3, 86.1, 85.1, 84.8), extract from its primary: the split name, the scenario count, the NAVSIM version or commit, and whether the number is leaderboard-submitted. **Committed in advance:** if scenario counts partition cleanly into {~5.9k navhard} and {~12k navtest}, D-10 closes and every number gets a stamp. If any paper reports ~12k **and** calls it navhard, the split names themselves are unreliable and **we stop using published EPDMS for external comparison entirely** and report only our own tier-stamped numbers. |

---

## What this changes for TanitAD - at most three recommendations

1. ⛔⛔ **Adopt T-4 as a hard bar: no external EPDMS number enters any TanitAD table without a SPLIT
   STAMP.** Row 32 and FS5-3 stay barred. This is a solve, not a refusal - the stamp is cheap and the
   six-number extraction above is a 0-GPU afternoon.
2. ⭐ **Adopt T-5: parameter counts carry INFERENCE-TIME or TRAINING-TIME**, the way every metric
   carries its eval tier. Apply it to our own sub-300M claim first, so we are not the ones caught.
3. ⭐ **Prefer navhard as our external comparison tier when we do compare**, because its Stage 2 is a
   3DGS counterfactual pseudo-closed-loop evaluation and therefore the only published tier that speaks
   to our T1 doctrine. Numbers from the 12k cluster should be treated as a different tier, not a
   different leaderboard position.

## Limits of this package

`Literature only.` The clusters are read from four primaries and three secondaries; **the assignment of
"12k scenarios" to navtest is an INFERENCE from the scenario count, not a quoted statement**, and it is
marked as such. Our own 55.5/56.3 remain unexplained rather than corrected - **this package narrows the
question and does not close it.** **The primary IS banked** (`kb_add`, rc=0); debt D-12 discharged.
