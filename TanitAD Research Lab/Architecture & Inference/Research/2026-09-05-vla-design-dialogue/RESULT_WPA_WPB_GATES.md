# RESULT — WP-A and WP-B, the two gates before any reasoner is built

**2026-09-05 · TanitAD_TrainingFlyWheel · Tier T0 · ⚠️ NON-PARITY pilot corpus · Evidence class:
MEASURED (ours).** Both were run to answer a question CHEAPLY that would otherwise be answered by
a training run.

---

## WP-A — do the 11 `cot_tokens` fields cover the free-text Alpamayo trace?

Blob **pinned by md5** `ee44875916…` (the release), 4,719 records. Group clips by their exact
11-field tuple; inside a group the fields are identical, so any remaining text difference is
variance the fields cannot express.

| | |
|---|---|
| distinct field-tuples | 171 (91 with n>1) |
| WITHIN — pairs sharing all 11 fields | 0.3730 |
| RANDOM — chance floor | 0.1387 |
| SELF — ceiling control | **1.0000** ✅ must be exactly 1 |
| **COVERAGE** | **0.2720** |
| SHUFFLED control | **0.0083** ✅ collapses |

⇒ ⛔ **PARTIAL. The fields explain 27.2 % of the explainable text variance.** The signal is real
(shuffled collapses) but rendering a CoT from the fields ALONE would lose most of the trace.

⭐ **Design consequence, applied:** `z` feeds the renderer **directly**, not only through the
decoded fields. The fields remain the supervised, **planner-facing** part — so the safety property
holds, only fields reach the planner — while the continuous latent carries the detail the fields
cannot.

---

## WP-B — is there a driving analogue of TRM's `puzzle_id`?

TRM's own ablation: a blank or random puzzle ID yields **ZERO** accuracy; ARC Prize calls it *"a
strong, limiting dependency"*. In ARC a puzzle's identity appears in train and test; **in driving
every window is a new puzzle**. So: does a context-derived code say WHICH of the top-5 candidates
is right?

n = 675 windows · d = 9 features · K = 5 · 15 episodes · **episode-disjoint** split.

| | |
|---|---|
| UNIFORM (1/K) | 0.2000 |
| **MAJORITY (always rank-1) — the baseline to beat** | **0.4519** |
| CONSTANT-code control | **0.4519** ✅ exactly majority |
| SHUFFLED-label control | 0.3926 ✅ does not beat majority |
| **CONTEXT-DERIVED situation code** | **0.4444** |
| **lift** | **−0.0074** |

⇒ ⛔ **NO SIGNAL. Nine aggregate scene statistics do not say which candidate is right — they do
not even match "always pick rank 1".**

### ⚠️ What this does and does not establish

**Does:** crude *aggregate* scene features (agent count, nearest distance, lateral spread, speed,
lead gap, fan spread, top-2 margin) carry no usable information about the choice, linearly, at this
n. A cheap situation descriptor is **not** a `puzzle_id` substitute.

**Does NOT:**
1. ⚠️ **It tests the wrong ROLE.** TRM's `puzzle_id` does not *predict the answer* — it tells the
   network **which transformation family applies**. I probed the former. A code could condition the
   recursion usefully while failing this test.
2. Nine hand-crafted scalars, **linear**, ~9 training episodes. A learned code from the trunk is a
   different object.
3. Pilot corpus, 15 episodes. Underpowered for a strong negative.

### ⭐ The constructive reading, and it argues FOR v6 rather than against it

**Every one of my nine features is AGGREGATE** — how many agents, how close the nearest, how fast
the ego. **None is RELATIONAL.** That a linear model on aggregates cannot pick the right candidate
is evidence that the choice lives in *which agent interacts with which candidate* — exactly the
information a flat summary destroys and a **typed edge** preserves.

⇒ **WP-B does not kill the recursive direction. It relocates the burden onto the graph**, and makes
**WP-F** (do edges carry anything nodes do not?) the decisive next test rather than an optional one.
⚠️ If WP-F also reads null, the direction is genuinely weak and we should say so.

---

## Process notes — two of my own errors, both caught

* ⚠️ **WP-A first globbed the WRONG BLOB** — `s2_labels_v7_incoming`, the stale-schema copy. Six
  copies exist under three roots with differing md5s and the wrong one loads perfectly. Now pinned
  by md5 with a record-count assert. *(The numbers were identical, but the pin stays: it was luck,
  not correctness.)*
* ⛔ **WP-B's first control spec was WRONG and voided a sound experiment.** I required the
  shuffled-label arm to land near the MAJORITY rate; a noise fit lands near UNIFORM. Two different
  chance levels, conflated. The script refused to report rather than passing quietly — which is the
  behaviour wanted — but the fault was the specification, not the run.
