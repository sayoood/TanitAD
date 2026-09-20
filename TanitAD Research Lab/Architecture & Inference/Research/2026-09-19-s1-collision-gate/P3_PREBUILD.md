# P3 target pre-build — census, one refuted premise, one retracted finding, one real risk

**Agent:** TanitAD_TrainingFlyWheel · **Date:** 2026-09-20 · **GPU used: ZERO** (CPU only;
`CUDA_VISIBLE_DEVICES=""`, the card left free for the gated A7 panel).
**Task as given (Master Mind):** *"pre-build P3's gate-relevant supervision targets on CPU … If the
target set is built and banked before the arm starts, P3 spends its card time training rather than
indexing."*

⛔ **Nothing here ran a P arm.** Every number is a CPU pass over the agent join and the window grid.

---

## 0. Verdict in four lines

| | |
|---|---|
| ⛔ **The premise is REFUTED** | the whole index path costs **2.5 s** (corpus 0.1 + join load 2.3 + window scan 0.1) against an ~11 h arm = **0.006 %**. Pre-building cannot buy card time; there was no indexing cost to remove. |
| ⭐ **The pre-build paid for itself anyway** | it produced the **first verified census of P3's own target population on the TRAIN cache**, and it found a **correction** and a **risk** that change what the arm should expect. |
| ⚠️ **A retraction, before it shipped** | I drafted *"P3 is confounded by the query budget"*. **It is false** and is retracted in §4. |
| ⛔ **A real, unregistered risk to P3** | **24.88 %** of labelled windows carry **zero** gate-relevant targets (vs **2.26 %** under 360°) — an **11.0×** increase in all-negative windows. §5. |

---

## 1. The grid is the trainer's own — proven, not assumed

The window index comes from `refcv3_arm.build_corpus` (the trainer-faithful path), then is
cross-checked against literals **written by runs I did not produce**, on other days:

| cache | n_windows | n_labelled | prefilter boxes | cross-check |
|---|---|---|---|---|
| **halfA** (the P arms' TRAIN cache) | 10,600 | 10,217 | 318,099 | **MATCH 3/3** vs `a3-maphead-halfA-20260919/run/config.json` |
| **halfB** (the eval cache) | 10,600 | 9,929 | 367,351 | **MATCH 3/3** vs `a8-occupancy-5k`, `a3-heldout`, `a2-smoke` (three runs agree) |

⭐ This is the discriminating check, not a determinism check: re-running my own derivation and
finding agreement would measure determinism. These six literals were produced by a different
process on a different day, so agreement is evidence the grid is the consumer's grid.
**A mismatch refuses the bank** (`grid_verdict` → `INCONCLUSIVE`), and so does having compared
**no** literal at all (`UNVERIFIED`) — which fired on its first real use, when I ran halfB before
I had found its literals.

## 2. The census — MEASURED, artifact `raw/p3_census_*.json`

Per **labelled** window (`pad` 32, box3d head `n_queries` **100**):

| | halfA mean | median | p95 | max | windows with 0 |
|---|---|---|---|---|---|
| **360° population** (today's supervision) | **31.13** | 20 | 97 | 149 | 2.26 % |
| **gate-relevant** (`x ∈ [0,60]`, `\|y\| ≤ 16`) | **4.634** | 3 | 16 | 34 | **24.88 %** |
| 360° *delivered* after the pad-32 truncation | **18.95** | 20 | 32 | 32 | 2.26 % |
| gate *delivered* after the pad | **4.632** | 3 | 16 | 32 | 24.88 % |

halfB, same order: **37.00 / 6.639 / 19.79 / 5.600**; zero-gate windows **22.93 %**.

* **Gate share of all supervised boxes: 14.88 % (halfA), 17.94 % (halfB).** ⇒ **~85 % of the boxes
  the head is trained on lie outside the population the gate reads.** That is P3's hypothesis,
  quantified on the train cache for the first time.
* **The pad DOES bind on 360°:** 34.33 % (halfA) / 40.04 % (halfB) of labelled windows carry more
  than 32 raw targets, so their farthest boxes never reach the head at all. The truncation is
  `argsort(hypot(x, y))[:32]` — by distance, which already tilts toward the near field.
* ⚠️ **The pad also bites the gate population on halfB, and not on halfA:** halfB drops
  **10,321 of 65,918** gate boxes (15.7 %) because some windows hold up to **128** gate-relevant
  agents; halfA drops **26 of 47,348** (0.05 %). The two halves are not interchangeable here.

## 3. ⚠️ CORRECTION — the pre-registration's density figures do not reproduce

`PREREG_PERCEPTION_BOX_QUALITY.md:75` says *"targets/window drop from ~18.7 to ~4.6, MEASURED"*
and carries **no artifact path**. Measured now:

| scoping | halfA | halfB |
|---|---|---|
| 360° raw, per labelled window | **31.13** | 37.00 |
| 360° delivered after pad, per labelled window | **18.95** | 19.79 |
| 360° delivered after pad, per **all** windows | **18.27** | 18.54 |
| gate-relevant, per labelled window | **4.634** | 6.639 |

**`4.6` is right — it is halfA's gate density (4.634).** **`18.7` reproduces at none of the four
scopings.** The nearest is halfA's *delivered-after-pad* 18.95. ⇒ the prereg line must be replaced
with **31.13 → 4.634 (raw)** or **18.95 → 4.632 (delivered)**, each naming its scope and artifact;
the drop is **6.7×** on the raw population, not the ~4× the bare pair implies. Amendment drafted
in §7. *(Class: a number without its arm and artifact path — the rule this programme already has,
applied to my own prereg.)*

## 4. ⚠️ RETRACTION — "P3 is confounded by the query budget" is FALSE

**What I drafted:** the A7/P configuration passes `--agent-queries 16` while 360° supervision
delivers ~19 targets/window, so on most windows the set loss can match only 16 of them; P3 would
then remove a query bottleneck as well as change the population, making a positive result
non-attributable. I had the arithmetic ready (56.9 % of windows "over-queried").

**Why it is false — two probes.** `--agent-queries` governs the **2-D agents head**, not the
refcv6 **box3d** decoder that `box_quality` scores and P3 targets. The box decoder is built with
`n_queries=100` (`box3d_head.py:237`, `N_QUERIES_DEFAULT`), and a real run stamps
`refcv6_perception.n_queries: 100` into its own `config.json`. With 100 queries against a pad of
32, **the query budget can never bind**: re-measured with the corrected count,
`frac_labelled_windows_over_queries` is **0.0000 in both populations on both halves**.

⇒ **P3 as pre-registered is NOT confounded by the query budget, and needs no disentangling arm.**
*(Class: a true quantity quoted outside its scope — the `df` / `step_s` / cylindrical-FOV family,
with the scope being **which head the flag configures**. Caught before it reached the MM because
the claim was checked against source and against the run's own stamp instead of being written up.)*

## 5. ⛔ THE REAL RISK TO P3, and it is not in the pre-registration

Under the gate-relevant population, on halfA:

* **24.88 %** of labelled windows carry **zero** targets, against **2.26 %** under 360° — an
  **11.0×** increase in windows that are entirely negative. These are legitimate labelled-clear
  windows (the road ahead really is empty), not NO_LABEL, and the census keeps the two apart.
* The positive rate per query falls from **18.95/100 = 0.190** to **4.63/100 = 0.046** — a **4.1×**
  drop in the fraction of queries with a matched target.

⇒ P3 changes the detector's **positive/negative balance** as well as its population. If the arm
regresses, "the presence term now dominates the loss" is a live explanation that has nothing to do
with P3's hypothesis, so it must be **pre-registered as a named alternative** and read from the
presence/no-object loss terms, not inferred afterwards. This is exactly the kind of thing the arm
would otherwise discover by spending 11 h of card.

## 5b. The risk, now MEASURED THROUGH THE REAL SET LOSS — and the one knob it implies

§5 was a count. `code/p3_presence_balance.py` turns it into the quantity that reaches the
optimiser by calling `agent_slots.slot_set_loss` itself, on **400 real halfA windows** drawn from
the banked census (the real per-window distributions, never their means — a mean would hide
exactly the zero-target windows that are the point). Artifact: `raw/p3_presence_balance_halfA.json`.

| | 360° | gate-relevant |
|---|---|---|
| matched targets / window | **19.02** | **4.855** (ratio **3.92×**) |
| **presence POSITIVE weight share** | **0.7013** | **0.3379** |
| windows with zero matched rows | 2.75 % | **26.75 %** |
| `loss_presence` | 0.2061 | 0.1123 |

⇒ The no-object term goes from a **minority (30 %)** of the presence loss to a **2:1 majority
(66 %)**. `NO_OBJECT_W = 0.1` (`agent_slots.py:232`, the DETR `eos_coef` convention) is implicitly
calibrated for ~19 targets per 100 queries; P3 moves the head to ~4.9 and does not touch it.

⭐ **THE CONTROL ARM THIS IMPLIES, solved rather than tuned.** With
`share = pos / (pos + (Q − pos)·W)`, the weight that holds the gate arm's positive share EQUAL
to the 360° arm's is **`NO_OBJECT_W = 0.02173`** against today's **0.1** — a **4.6×** change.
A P3 run at the current weight and a second at 0.02173 separate *"concentrating supervision on the
gate population helped"* from *"the presence term re-balanced"*. Without it, a P3 regression is
**not attributable**, which is the same defect class as the `--v2` ten-levers-on-two-axes conflation.

⚠️ Note what this is NOT: it is not a claim that P3 will fail, and not a licence to change
`NO_OBJECT_W` in the main arm. The pre-registered P3 stays exactly as written; this adds a
**named alternative** and a control that can test it.

**Controls (both PASS, and the numbers are inadmissible without them):**
* **IDENTITY** — the same count sequence run twice reads EXACTLY equal on every term, so any
  arm-to-arm difference is the population and not the harness.
* **CONSTANT** — an all-zero-target sequence reads the no-information value exactly: matched 0,
  `n_centre` 0, `loss_centre` **0.0**, positive weight share **0.0**.
* Predictions are drawn once from a fixed seed and reused across arms, so the target count
  sequence is the only thing that differs.

## 6. ⚠️ A defect in my own mutation harness, found by its control

The first run of `mutate_prebuild_p3.py` reported **7/7 mutations caught** — with a **RED control**,
so the proof was void and is not quoted. Cause: the scratch copy did not carry `stack/tanitad`, so
`box_quality`'s transitive `import tanitad` was served by the venv's **editable install pointing at
the abandoned G: checkout**, and the read died with `OSError: [Errno 22]`.
⛔ **My wrong-disk conftest guard did not catch it** — it asserted only that the *module under test*
was local, and said nothing about its dependencies. The guard now asserts
`prebuild_p3_targets`, `box_quality` **and** `tanitad` all resolve under the scratch root.
**Re-run: control GREEN, 7/7 caught** (`raw/mutation_proof_prebuild_p3.json`).
*(Generalisation worth carrying: a wrong-disk guard must cover the dependency closure, not the
entry point — the same shape as a positive assertion that is positive in FORM but blind to the
failure that matters.)*

## 7. Proposed prereg amendment (for the MM / PI — NOT applied unilaterally)

1. Replace the density line with the measured pair **31.13 → 4.634 targets/labelled window
   (halfA raw)**, naming `raw/p3_census_*.json`, and state the delivered-after-pad pair separately.
2. Add the **all-negative-window shift** (2.26 % → 24.88 %, 11.0×) as a **named alternative
   explanation** for a P3 regression, read from the presence/no-object terms.
3. Record that the **query budget does not bind** (n_queries 100 ≥ pad 32), so no disentangling arm
   is needed — with §4's retraction, so the question is not re-opened later.
4. Note that **halfA and halfB differ materially** (gate density 4.63 vs 6.64; pad drops 0.05 % vs
   15.7 % of gate boxes), so an eval-cache density may not be quoted for a train-cache claim.

## 8. Deliverable manifest

| artifact | where it lives |
|---|---|
| `stack/scripts/prebuild_p3_targets.py` | repo (D:), staged |
| `stack/tests/test_prebuild_p3_targets.py` | repo (D:), staged — **15 passed bare** (no ambient `PYTHONPATH`) |
| `stack/scripts/mutate_prebuild_p3.py` | repo (D:), staged — control GREEN, **7/7 caught** |
| `raw/mutation_proof_prebuild_p3.json` | repo (D:), staged |
| `raw/p3_census_*.json` (halfA, halfB) | repo (D:), staged |
| `p3_targets_*.npz` (banked selections, 41 KB + 49 KB) | `C:/Users/Admin/tanitad-caches/p3-prebuild-20260920/` — **off-repo, on this dev box only** |
| pinned v8 label copy (md5 `eefc38d1453bd1c73802d44d45affced`, = A7's) | same directory |

⚠️ **The two `.npz` banks are on ONE DISK.** They are cheap to rebuild (2.5 s) and are a
*verification reference* rather than a required input, so I have not pushed them into the repo; say
the word if they should be banked there instead.

**A7 status, unchanged:** still correctly gated — GPU at **3,950 MiB** (the PI's servers), gate
untouched at ≤ 2,500 MiB / ≥ 8 GB, launcher armed.
