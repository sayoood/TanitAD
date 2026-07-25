# PUBLISHED-NUMBER PROVENANCE INVENTORY — 2026-07-25

**Scope.** Every quantitative claim that is or was **decision-grade** in the program's published surface,
traced to its source artifact, with the statistic it names and what it decided. Complementary half of the
sibling `2026-07-25-jack-blast-radius` audit: that one asks *which numbers were **computed** through the
biased estimator*; this one asks *which numbers the program **publishes**, and which of those decided
something*. Joining them on `doc`+`line`+`value` gives the exact blast radius.

**Method.** All six priority documents read **in full** (5,071 lines). Every cited artifact path resolved
against disk. Retraction sweep run **multiline** (`re.DOTALL` over whole-file text, plus the Grep tool with
`multiline: true`) and **including section headers** — a retracted claim once evaded a line-based grep by
wrapping across a newline. Absence probed **twice** before being recorded. `tools/registry_lint.py --strict`
run read-only. **Nothing was edited, staged, committed or pushed; no pod was touched; no GPU was used.**

**Machine-readable companion:** `published_numbers.jsonl` — 188 rows, one JSON object per number, fields
`doc · line · claim · value · cited_source · citation_resolves · statistic · decided · evidence_class · note`.

---

## 0. EXECUTIVE VERDICT

**The program's estimator problem is bigger on the publication side than on the computation side, because
the publication side carries an explicit, load-bearing, and now-false assurance that it is small.**

Five findings, ranked:

**F1 — The paper's instrument-doctrine section licenses exactly the error that was found today.**
`TANITAD_PAPER.md:443` (§5, *The estimator correction*) states: *"The point estimates and every qualitative
verdict in §7.1–§7.5 stand (they were re-verified against the corrected intervals); the **widths** quoted
there are the deprecated statistic."* As of 2026-07-25 this is **false**. `RETRACTION_LOG.md:71` establishes
that `_jack` / `overlapping_holdout_se` **moves the point estimate** — ×2.97 on the ctx→tactical seam, ×3.28
on v4.2b, up to **×4.29 with sign flips** synthetically. The paper's own next sentence already contradicts
the assurance (*"the split-mean compresses between-arm gaps — 0.006 m vs 0.044 m on the full set"*). This
single sentence is what authorises every legacy point estimate in §7.1–§7.5 to stand unrevised, and it sits
in the section whose entire purpose is measurement honesty. **Highest external stakes in the program.**

**F2 — The retracted hierarchy seam is still standing in the paper.**
`TANITAD_PAPER.md:768-770` publishes *"At 30 k **one seam flipped to load-bearing** — ctx→tactical now shows
`content_matters = true` (vs-mean maneuver Δ **+0.044**, CI-separated)"* with **no caveat**. This is the
number retracted today: the true full-set paired delta is **+0.0148**, and under the correct estimator the
seam **fails all three gates on the point estimate alone** (0.0148 < MIN_ACC 0.02 · 0.0050 < MIN_COS 0.01 ·
0.0437 < MIN_ADE_M 0.05). Verified digit-for-digit against both raw artifacts (below). The honest read is
**0 of 3 seams load-bearing, not 1 of 3** — i.e. the paper currently publishes the *only measured leg*
supporting its hierarchy thesis, and that leg does not exist. Same claim also live in
`HYPOTHESIS_LEDGER.md:128` and `:628`.

**F3 — The most-quoted numbers in the program cite raw eval JSONs that are not in the repo.**
`taniteval/results/flagship-30k.json`, `refb-v2-30k.json`, `refa-dynin-30k.json`, `refa-dinov2.json`,
`flagship-nospeed.json`, `flagship-speed.json`, `flagship-v2-6k.json`, `planner_p2_flagship-30k.json` —
**none exist anywhere in the repo** (confirmed with a second, repo-wide probe). They resolve only on
`tanitad-eval:/root/taniteval/results/`. R2 in §7 is marked ✅ **CLOSED** ("TanitEval vendored — 68 files
tracked") — that closed the **code** half; the **results** half is still open and is *not* listed in the
reconstruction-gap register. Most arms are re-derivable from the in-repo `windows_<key>.pt` dumps.
**P2 is not**: it has neither a raw JSON nor a windows dump (`LEADERBOARD.md:88`), and P2's G1/G4 are the
evidence base for **D-033, the v3 pivot** — the largest architecture decision in the program.

**F4 — Three retracted claims are still standing outside the paper, one of them in a standing protocol.**
`GATE_PROTOCOL.md:96-97` publishes *"v1 reached v3enc's current `g_op_fwd_ade_m` (0.422) at step **450** →
~**12×** slower. v2's figure was ~30×."* `MODEL_REGISTRY.md:400-426` retracted "step 450" as a single-row
noise artifact and states the `reached_at_step: 450` field **"is void in both"** gate JSONs. A *binding
standing protocol* quotes a field its own source-of-truth declares void. Also standing:
`MODEL_REGISTRY.md:133` still prints `wallclock_s: 191206.2 (~53 h A40)` after `RETRACTION_LOG` 07-21
settled the true figure at **90.73 h** and established that `wallclock_s` is not the full run's — and the
paper's *"total training cost ≈ $40"* headline (§6.3) still rides that uncorrected 1.71× wall-clock error.

**F5 — Two standing rules are violated by the source of truth itself, on a decision that killed a run.**
`MODEL_REGISTRY.md:310` and `:1586` (D-031) publish *"the decisive read: power-law exponent **v2 = −0.50 vs
v1 = −0.84**"* with **no fit window, no R², no n** — and this was **the read that killed flagship-v2**.
`GATE_PROTOCOL.md:74` (the program's own protocol) states *"v1's famous −0.84 is the 1500–7500 window at
R² 0.541 — below the floor and should never have been quotable."* `CLAUDE.md` forbids it explicitly. The
accompanying *"v1 reached v2's 7.5 k value at step ~250 (~30× faster)"* is the **same estimator** as the
retracted 23×/step-450 claim and was never re-derived, and the *"~9× worse at 30 k"* projection extrapolates
**~4.3×** beyond its fitted range against a **2× cap**.

---

## 1. COUNTS

| | total | unlabelled statistic | non-resolving citation | no source cited at all | retracted-but-still-standing |
|---|---:|---:|---:|---:|---:|
| **ALL DOCUMENTS** | **188** | **91 (48 %)** | **104 (55 %)** | **77 (41 %)** | **7** |
| `MODEL_REGISTRY.md` | 79 | 34 (43 %) | 37 | 17 | 2 |
| `TANITAD_PAPER.md` ⚠️ | 58 | **37 (63 %)** | **51 (88 %)** | **50 (86 %)** | 2 |
| `LEADERBOARD.md` | 25 | 8 (32 %) | 5 | 2 | **0** |
| `GATE_PROTOCOL.md` | 9 | 3 | 3 | 2 | 1 |
| `PROGRAM_OVERVIEW.md` | 9 | 5 | 7 | 6 | 0 |
| `HYPOTHESIS_LEDGER.md` | 8 | 4 | 1 | 0 | 2 |

*Definitions.* **Decision-grade** = ranks an arm, passes/fails a gate, decides a kill / restart / GPU-day
allocation, or is a headline (abstract, section header, or leaderboard rank). **Unlabelled** = the document
does not name which statistic or estimator produced the number (`MODEL_REGISTRY.md:63` mandates naming
`heldout` vs `full_set`: *"Both are published; they differ. Always name which."*). A bare `±` counts as
unlabelled unless the estimator is named nearby. **Non-resolving** = the cited artifact does not exist at
the cited path, or is pod-only, or is stale, or the number is not in the artifact that is cited.

⚠️ **`LEADERBOARD.md` is the cleanest document in the program: 0 retracted claims standing, 32 %
unlabelled, 5 non-resolving.** The derived document out-disciplines both the source of truth and the paper.
That ordering is backwards and is the strongest single argument for propagating LEADERBOARD's citation
discipline upward.

*Scope note, stated plainly:* the `PROGRAM_OVERVIEW` and `HYPOTHESIS_LEDGER` rows in the JSONL are the
**decision-grade subset**. Delegated full sweeps surfaced ~60 and 41 candidate rows in those two documents
respectively; the majority are verbatim restatements of registry numbers already inventoried here, and were
folded rather than duplicated. Their independent findings are carried in §2 and §4 below.

---

## 2. HIGH-VALUE CHECK 1 — UNLABELLED NUMBERS (the ones that cannot be audited after the fact)

**91 of 188 decision-grade numbers (48 %) do not name the statistic or estimator that produced them.**

The registry's own §0.3 line 63 is the rule *and* the worst offender:

> `| Statistic | **8-split episode-disjoint jackknife**, val_frac 0.2 → heldout mean ± CI95. full_set = plain mean over all 881. **Both are published; they differ. Always name which.** |`

That label is **retracted** — §6 line 1441 of the same document says *"This block was historically labelled
'8-split episode-disjoint jackknife'. It is **neither a jackknife nor a valid SE**."* The definitional line
that governs every `heldout` number in the program, and that mandates naming the statistic, **carries the
wrong name, 1,378 lines before its own correction.** Anyone reading §0.3 and stopping there inherits the
retracted label.

**The four gravest unlabelled classes:**

| class | count | example | why it matters |
|---|---:|---|---|
| **`_jack` values published with a bare `±` or "CI-separated"** | 28 | `PAPER:769` Δ +0.044 "CI-separated"; `LEDGER:128`; `PAPER:693` 0.452 ± 0.031 | These are precisely the numbers now known to be biased in the **mean**, not only the width |
| **Trivial floors and bars with no estimator** | 4 | `REGISTRY:64` CTRV 0.523 · best-of-3 0.5005 · ceiling 0.5735 | Every "beats the floor" verdict in the program is measured against these |
| **Ratios / percentages off a single arm or single artifact** | 21 | `REGISTRY:161` "89 % longitudinal"; `PAPER:1096` CNCE median | C5 class; `RETRACTION_LOG:72` retracted a sibling of exactly this ("98.6 % longitudinal", true range 0.607–0.976) |
| **Latency figures without hardware/checkpoint/corpus** | 11 | four different "decision tick" values coexist (below) | The program already retracted one of these for exactly this defect |

**The decision-tick family — four live, mutually inconsistent values for one quantity:**

| value | where | conditions named? |
|---|---|---|
| **11.16 ms** / 89.6 Hz | `REGISTRY:180` | ✅ fp16+CUDA-graph, RTX 4060, step 6,500, comma2k19 (defect corrected 07-20) |
| **15.1 ms** p50 / 17.2 p95 | `PAPER:549` | partially — step-6500, RTX 4060, fp32 |
| **14.331 ms** p50 | `PAPER:1096`, `OVERVIEW:54` | ❌ none; **appears nowhere in `MODEL_REGISTRY`** |
| **17.75 ms** fp32 | `REGISTRY:180` (parenthetical) | ✅ |

`MODEL_REGISTRY` is the only quotable source for model facts. **14.331 ms is not in it**, yet it is
published in the paper *and* the overview as the beyond-ADE suite's headline latency.

**Two param counts published as the model's size, neither matching the registry:**
- `PAPER:20` (abstract), `PAPER:526`, `PAPER:434`: **"261 M parameters"** — this is D-008's *design budget*,
  not a measurement. Registry §1.2 measures `total_model` **263,442,838** / `trainable` **277,404,073**.
- `PAPER:1096` and `OVERVIEW:46`: **"the deployed 262.8 M architecture"** — **262.8 M is REF-B's** count
  (`REGISTRY:1481`), attached to CNCE/TMS/decision-tick numbers measured on the *flagship*.

**And the program's headline open-loop bar has two values in one document:**
`PAPER:36` (abstract) and `PAPER:638` publish **CTRV = 0.544** ("the honest open-loop bar", pre-registered);
`PAPER:694` and `REGISTRY:64` publish **CTRV = 0.523**. Unreconciled.

---

## 3. HIGH-VALUE CHECK 2 — NON-RESOLVING CITATIONS

**104 of 188 (55 %) cite an artifact that does not resolve as cited. 77 (41 %) cite no artifact at all.**

### 3a. The pod-only raw-eval class — the biggest single gap

Every `results/<key>.json` cited by `MODEL_REGISTRY` for a **headline arm** resolves only on the eval pod.
Probed twice (`taniteval/results/` listing + repo-wide `find`):

| cited path | in repo? | re-derivable from in-repo data? |
|---|---|---|
| `results/flagship-30k.json` (the deployed model, 0.4522/0.4271) | ❌ **absent** | ✅ `windows_flagship-30k.pt` + `driving_flagship-30k.json` |
| `results/refb-v2-30k.json` (0.5921, rank 3) | ❌ absent | ✅ `windows_refb-v2-30k.pt` |
| `results/refa-dynin-30k.json` (2.9196, **closes H4**) | ❌ absent | ✅ `windows_refa-dynin-30k.pt` |
| `results/refa-dinov2.json` (2.1322) | ❌ absent | ✅ `windows_refa-dinov2.pt` |
| `results/flagship-nospeed.json` (2.9176, the causal control) | ❌ absent | ✅ `windows_flagship-nospeed.pt` |
| `results/flagship-speed.json` (0.6277, first CV-beater) | ❌ absent | ✅ `windows_flagship-speed.pt` |
| `results/flagship-v2-6k.json` (6.179, **the kill evidence**) | ❌ absent | ✅ `windows_flagship-v2-6k.pt` |
| **`results/planner_p2_flagship-30k.json`** (0.893/1.038, **the v3 pivot**) | ❌ absent | ❌ **NO windows dump either** |

79 JSONs *are* tracked under `taniteval/results/` — the `driving_*`, `eff_*`, and the v3enc/v4/REF-C raw
evals. The gap is precisely the **first-generation headline arms**. R2 in the reconstruction-gap register
reads ✅ CLOSED; it should read *partially* closed.

### 3b. The elided-prefix class — mechanically unresolvable

Registry rows cite `…/incoming/2026-07-24-branchb-transfer-eval/…` with the Research-Hub stream elided.
There are four streams (`Benchmarks & Eval`, `Architecture & Inference`, `Data Engineering`,
`Production & Optimization`) and the prefix is not recoverable from the citation. I resolved 2 of 2 by
second probe (both were under `Architecture & Inference`, not the `Benchmarks & Eval` a reader would guess
from context) — **a linter cannot.** Same defect in `HYPOTHESIS_LEDGER` H7, whose Owner column says
*Data Engineering* while the artifact sits under *Architecture & Inference*.

### 3c. Cited-but-does-not-contain-the-number

- **`+0.0043` seam cosine** — `PAPER:59/833`, `OVERVIEW:51/165`, `LEDGER:129`. The cited folder
  `…/incoming/2026-07-23-planner-wm-gradient-coupling/` **exists but holds only** `DESIGN.md`,
  `PRE_REGISTRATION.md`, `INTAKE.md`. No results artifact carries `+0.0043`. Independently confirmed by a
  second agent. **This number redirected the entire v4 program from gradient surgery to co-evolution
  from random init — a 30k-step GPU-day decision — and it has no resolving artifact.**
- **`taniteval/ci.py`, `taniteval/rollout.py`** cited bare (`GATE_PROTOCOL:42`, `OVERVIEW:21`, `LEDGER` H20):
  the package is nested one level deeper at `taniteval/taniteval/*.py`. Every bare `taniteval/<module>.py`
  citation in the program is wrong by one path segment.
- **`predictor.py`, `calib.py`, `eval_flagship_v4.py`, `COUNTERFACTUAL_ACTION_AUGMENTATION.md`** — cited as
  bare filenames in the paper; do not resolve without a search.

### 3d. Citations that terminate in prose, not data

`CLAUDE.md`'s #1 error class. Live instances:
- `LEDGER:140` (IMP-2) → `PROGRAM_OVERVIEW.md §7`; `LEDGER:141` (IMP-3) → `PROGRAM_OVERVIEW.md §5.2d`,
  which itself cites nothing. **Circular.**
- `REGISTRY:1088` — *"the oracle gap is ~92 % irreducible, across 47 trained arms"* → a dated research note.
  This claim **killed the selection lever on REF-C**.
- `LEADERBOARD:394` — junction departure **0.368** sourced to `CLOSED_LOOP_PLANNER_RESEARCH.md`, conflicting
  with the table's own 0.064. Honestly self-flagged as MUST-RECONCILE-BEFORE-QUOTING.
- `LEADERBOARD:307` — the Orin/Thor precision map sourced to *"vendor specs"* with no doc or URL.

### 3e. Prose and its own cited artifact disagree (R14, still open)

`REGISTRY:181` publishes the flagship planning tick as **103.42 / 93.76 / 104.49 ms**; the committed
`taniteval/results/eff_flagship-30k.json` says **97.32 / 97.70 / 123.83**; `eff_repeatability.json`
(5 clean reps) says **99.03–100.05**. Three values, none matching. Honestly flagged as R14 since 07-20;
still unreconciled. `LEADERBOARD:283` carries the conflict correctly.

---

## 4. HIGH-VALUE CHECK 3 — THE PAPER (external-stakes surface)

`Paper/TANITAD_PAPER.md`, v0.6, 1,450 lines. Its own honesty rule (line 5): *"no number appears here without
its instrument rows in the referenced experiment record."*

**Measured: the paper contains 12 path-like citations in 1,450 lines. 5 of the 12 do not resolve.**
58 decision-grade numbers were inventoried; **50 of them (86 %) cite no artifact at all**, and **37 (63 %)
do not name their statistic.** The paper cites §-references and dated ledger entries — not files. The
honesty rule is aspirational, not enforced.

The 12 citations, resolved:

| resolves | path | line |
|:--:|---|---|
| ❌ | `predictor.py` | 213 |
| ❌ | `COUNTERFACTUAL_ACTION_AUGMENTATION.md` | 332 |
| ❌ | `calib.py` | 585 |
| ❌ | `/root/taniteval/results/` (pod-only) | 616 |
| ❌ | `eval_flagship_v4.py` | 858 |
| ✅ | `Project Steering/FLEET_REVIEW_2026-07-17.md` | 616 |
| ✅ | `Benchmarks & Eval/LEADERBOARD.md` | 618 |
| ✅ | `Architecture & Inference/Implementation/belief_rollout_diagnostic/` | 675 |
| ✅ | `taniteval/generalization.py` | 702 |
| ✅ | `taniteval/pathspeed.py` | 778 |
| ✅ | `TanitAD Research Hub/INITIAL_RESEARCH_SYNTHESIS.md` | 1308 |
| ✅ | `Ressources/AD_TRANSFER_RESEARCH.md` | 1315 |

### The paper's decision-grade / headline numbers, with provenance

*(All 58 rows in `published_numbers.jsonl`; the ones that carry external risk are below.)*

| # | line | claim | value | statistic | source | verdict |
|---|---|---|---|---|---|---|
| P1 | 443 | *"the point estimates … in §7.1–§7.5 stand"* | — | names both estimators | self | 🔴 **NOW FALSE** — `_jack` biases the mean (F1) |
| P2 | 769 | one seam flipped to load-bearing | Δ **+0.044** "CI-separated" | UNLABELLED (`_jack`) | none | 🔴 **RETRACTED-STANDING** — true +0.0148, fails all 3 gates |
| P3 | 763 | H18 grounding dominance | **Δ 2.70 m** at 30 k | UNLABELLED (`_jack`) | none | 🔴 **STALE** — corrected **up** to +2.9568 m |
| P4 | 20 / 526 / 434 | model size | **261 M** | n/a | none | ⚠️ design budget published as a measurement (registry: 263.44 M) |
| P5 | 1096 | beyond-ADE suite | on the *"deployed **262.8 M** architecture"*; tick **14.331 ms** | UNLABELLED | none | ⚠️ REF-B's param count; tick absent from the registry |
| P6 | 36 / 638 vs 694 | CTRV, the pre-registered honest bar | **0.544** vs **0.523** | UNLABELLED | none | ⚠️ two values in one document |
| P7 | 528 | total training cost | **≈ $40** | UNLABELLED | none | ⚠️ rides the retracted ~53 h wall-clock (true 90.73 h, 1.71×) |
| P8 | 537 / 568 | gate D1/D2/D3 verdicts | D2 0.872/0.940 PASS; D1 6.44 ± 0.55 FAIL; D3 1.30 | `±` is `_jack`, unnamed | none | ⚠️ pre-reset `p0-sB01-realmix`, camera-frame, never re-gated (`REGISTRY §8.1 #4`). §7.1 has a supersession note; **§7's table and the gate verdicts do not** |
| P9 | 30 / 423 | spectral fit, the §4 sample-efficiency thesis | R² 0.997, ≈22–35 dims | UNLABELLED | none | ⚠️ step-3000 of the same stale pre-reset run |
| P10 | 678 | I-JEPA beats DINOv2 | fwd-ADE 3.194 vs 3.796 | UNLABELLED | none | ⚠️ registry §2.2/R8: I-JEPA's canonical-val is **~80 % leaked and marked UNUSABLE**; the paper discloses overfitting but **not the leak** |
| P11 | 59 / 833 | seam gradient geometry | cos **+0.0043**, 512 windows | UNLABELLED | folder w/o results | ⚠️ redirected the program; no resolving artifact |
| P12 | 62 / 852 | co-evolution canary + ADE | 15.674 → 1.371; ADE → 0.4788; gate CONTINUE | UNLABELLED, **trainer in-loop** | none | ✅ honestly flagged in §7.6 limit (i); the **abstract** carries it with the caveat deferred to the final clause |
| P13 | 693 | the central result | **0.452 ± 0.031** (plain mean 0.427) | "plain mean" named; `±` is `_jack`, named only in §5 | none | ⚠️ artifact `results/flagship-30k.json` not in repo |
| P14 | 982 | closed-loop n=40 headline | 1.488 vs 0.564, Δ +0.924 [+0.781,+1.065] | paired episode-cluster bootstrap (named) | none | ✅ artifact exists (`lowood_lanekeep_40ep.json`) but is **not cited** |
| P15 | 1043 | v2 corpus turn share | **28.0 %** | n/a | none | ⚠️ registry §1.7: 28.04 % is the **v1 labeler**; under the **v2 labeler now in force** it is **18.83 %**, and the registry instructs quoting 18.83 % for this run |

**What the paper does exceptionally well and should be preserved:** §5's three new failure classes (I8/I9/I10)
with the wrong claim that earned each; §7.6's four-item *Honest limits* block; §7.9's *Honest counterweight*
on the YouTube pilot; §7.7's I9 retraction narrated as method rather than errata; §7.8's explicit statement
of what the lane metric **cannot** measure. Roadmap item **#6** — *"Retire the deprecated interval estimator
from the historical tables and re-publish §7.1–§7.5's widths under the episode-cluster bootstrap"* — is
already the right task, but it is scoped to **widths**; today's finding makes it a **point-estimate** task.

---

## 5. HIGH-VALUE CHECK 4 — DID THE RETRACTIONS LAND EVERYWHERE?

`RETRACTION_LOG.md` holds **46 retracted claims** (per `registry_lint`). Swept multiline, headers included.

### 5a. `registry_lint.py --strict` output (run read-only, unmodified)

```
registry_lint: 1 file(s), 5 pointer(s), 46 retracted claim(s) loaded

[warn]  Project Steering/MODEL_REGISTRY.md:1346: header matches retracted-claim vocabulary,
        but every word is house boilerplate (likely a plain section title): "ref c closed loop"
        matches retracted claim: "flagship v1 beats REF-C closed-loop / drives where REF-C collides"
            rare tokens in the match: (none)
            lines [1346]: ### 4.4 REF-C CLOSED-LOOP - AlpaSim NuRec suite (n = 12) - **RECONSTRUCTION-OOD CONFOUNDED** - MEASURED 202

RESULT: FAIL (0 error(s), 1 warning(s); --strict makes warnings fatal)
```

**Assessment: the single warning is a true negative** — §4.4's header is a plain section title that already
carries the confound label, and the linter itself says so ("every word is house boilerplate"). **The linter
is clean on `MODEL_REGISTRY.md` and found nothing this manual sweep did not.**

**But `registry_lint` scans one file.** All three retracted-still-standing findings below are **outside its
scope**: two in `Paper/TANITAD_PAPER.md`, one in `Project Steering/GATE_PROTOCOL.md`, two in
`HYPOTHESIS_LEDGER.md`. **Extending its file list is the single highest-leverage change available.**

### 5b. Manual multiline sweep — retracted claims still standing

| # | retracted claim | where it still stands | header? | caveat adjacent? |
|---|---|---|:--:|:--:|
| **S1** | *"ctx→tactical is a LOAD-BEARING seam, Δ +0.0439/+0.044 CI-separated"* (C4/C5, 07-25) | `TANITAD_PAPER.md:768-770` · `HYPOTHESIS_LEDGER.md:128` · `HYPOTHESIS_LEDGER.md:628` | no — **a live table row and the paper's §7.5 body** | ❌ **none, in all three** |
| **S2** | *"H18 grounding dominance Δ 2.70 m"* (superseded by +2.9568, same 07-25 entry) | `TANITAD_PAPER.md:763` · `HYPOTHESIS_LEDGER.md:110` · `:631` | no | ❌ none |
| **S3** | *"v1 reached v3enc's value at step 450 → 12×"* + *"v2's was ~30×"* (C5, 07-21; field declared **void**) | `GATE_PROTOCOL.md:96-97` | no — a **binding standing protocol** | ❌ none |
| **S4** | *"wallclock_s 191206.2 (~53 h)"* (C4, 07-21; true 90.73 h) | `MODEL_REGISTRY.md:133` (Status row) | no | ❌ none |
| **S5** | *"8-split episode-disjoint jackknife"* as the statistic's **name** | `MODEL_REGISTRY.md:63` — **the definitional line** | ~ definitional | ❌ correction is 1,378 lines later |
| **S6** | bare exponent *"−0.50 vs −0.84"* + *"step ~250 / ~30×"* + *"9× projection"* | `MODEL_REGISTRY.md:310-313` · `:1586` (D-031) | no | ❌ none — and `GATE_PROTOCOL:74` refutes it |
| **S7** | *"3 of 8 KILL secondaries have no emitter"* | `MODEL_REGISTRY.md:671` · `PROGRAM_OVERVIEW.md:352` · `PAPER:822,1284` | no | ❌ **stale, not retracted**: `v1_g1_dryrun_gate_FIXED.json` shows `kill_adjudicated: 8`; fixed same-day in `3ff5499` |

### 5c. Retractions that DID land cleanly (recorded so the pattern is visible)

`v1.6 "best in the program"` — corrected in the §1.4b **header** *and* the newline-wrapped narrative
instance at `REGISTRY:544`; absent from LEADERBOARD, OVERVIEW, LEDGER and the paper. ·
`"REF-C-XL finishes 0.006 m behind"` — retracted with the full-set gap and paired interval. ·
`"deploy tick 11.16 ms / 89.6 Hz"` — corrected everywhere it propagated, with a two-tick definition table. ·
`"CEM 0.132 = 4.5× headroom"` — retracted in registry, overview **and the paper** (§5 I9, §7.7). ·
`"v1 beats REF-C closed-loop"` — reversed and triple-confirmed across three instruments. ·
`"the closed-loop direction is BOUND"` — reopened, with the horizon confound named. ·
`"98.6 % longitudinal"` — never propagated into any of the six documents.

**Pattern: retractions land reliably in `MODEL_REGISTRY` and `LEADERBOARD`, and unreliably in
`TANITAD_PAPER.md`, `GATE_PROTOCOL.md` and `HYPOTHESIS_LEDGER.md`.** All three of the latter are outside
`registry_lint`'s scan set.

### 5d. Verification of S1, digit-for-digit (both artifacts read read-only)

`…/incoming/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-30k.json`:
```
protocol.ci = "8-split episode-disjoint jackknife (bench.py protocol)"
seam_ctx_to_tactical.maneuver_acc.delta_real_vs_mean = {mean: 0.0439, ci95: 0.031, n: 881, separated: true}
seam_ctx_to_tactical.load_bearing = true , content_matters = true
```
`…/incoming/2026-07-25-jack-blast-radius/jack_hierarchy_recompute.json`:
```
published_jack_split_mean 0.0439 -> true_full_set_delta 0.0148 ; bias_ratio 2.9662
clears_floor_published: true -> clears_floor_true: false ; floor_verdict_flips: TRUE
interval_note: "the hierarchy panel does not persist per-window arrays - a re-run of taniteval.hierarchy is required"
```
⚠️ **The artifact itself still emits `load_bearing: true` and labels its CI with the retracted name.** The
correction exists only in the sibling's recompute file; nothing downstream of the panel has been re-emitted.
⚠️ **The interval is not recomputable from disk** — the point estimate is, the CI is not.

---

## 6. RANKED — WHAT MOST URGENTLY NEEDS RE-DERIVATION

Ranked by **external stakes × decision weight × auditability** (auditability = how badly the number resists
after-the-fact checking; a number with no artifact and no windows dump ranks *higher*, not lower).

### Top 5

**1. `TANITAD_PAPER.md:443` — the §5 assurance *"the point estimates … stand"***
Stakes: maximum (published, and it is the sentence that licenses every other legacy number). Decision
weight: maximum (it is why §7.1–§7.5 were never revised). Auditability: the falsifying evidence already
exists on disk. **Fix: rewrite to state that `_jack` biases the *mean* as well as the width (×2.97 measured,
×4.29 with sign flips), and re-scope roadmap item #6 from "widths" to "widths and point estimates."**
This is a text fix, not a re-derivation, and it unblocks 2–4.

**2. `TANITAD_PAPER.md:769` + `HYPOTHESIS_LEDGER.md:128`/`:628` — ctx→tactical Δ +0.044**
Stakes: maximum (the paper's only measured support for its hierarchy thesis). Decision weight: maximum
(quoted forward into `PROGRAM_OVERVIEW`, the R3 audit, HPP-0). Auditability: **partial — the point estimate
is already re-derived (+0.0148) but the CI is NOT recomputable from disk**; `taniteval.hierarchy` does not
persist per-window arrays. **Fix: (a) correct the point estimate and the "1 of 3 → 0 of 3" reading today;
(b) re-run `taniteval.hierarchy` with per-window persistence to get a paired episode-cluster interval.**
⚠️ Note the second lesson preserved in `RETRACTION_LOG:71`: `_jack`'s `separated` was **one-sided**; a naive
two-sided port would flip the *harmful* intent→operative seam to LOAD-BEARING. Read `separated_positive`.

**3. `MODEL_REGISTRY.md:1470-1489` — the entire cross-arm rank table**
Stakes: high (it is the program's leaderboard, mirrored into `PROGRAM_OVERVIEW:126`). Decision weight:
maximum (every ranking claim). Auditability: **good — `windows_<key>.pt` exists for 14 of 15 arms**, so the
whole table can be re-emitted under the episode-cluster bootstrap without a GPU. It is currently 100 % the
deprecated split-**mean**, which is now known to compress and shift, not merely narrow. **Fix: re-emit the
ADE/FDE/miss columns from the windows dumps; keep the legacy column labelled, as `LEADERBOARD.md` already
does.**

**4. `MODEL_REGISTRY.md:1413-1414` — P2's G1/G4 (0.893 ± 0.114 · 1.038 ± 0.202)**
Stakes: high. Decision weight: **maximum — this is D-033, the v3 pivot, the largest architecture decision in
the program.** Auditability: **WORST IN THE INVENTORY — no raw JSON and no windows dump exist anywhere in
the repo.** It cannot be re-derived at any price short of re-running `planner_p2.py` on the eval pod. Both
values are the deprecated statistic. **Fix: rescue `results/planner_p2_flagship-30k.json` off
`tanitad-eval` and persist a windows dump on the next P2 run.** ⚠️ This is a **stranding escalation**, not
just a statistics one.

**5. `MODEL_REGISTRY.md:310`/`:1586` + `GATE_PROTOCOL.md:96-97` — the flagship-v2 kill evidence**
Stakes: medium-high (it justifies a completed kill and is quoted in a binding protocol). Decision weight:
maximum (killed a run; set the D-A7 restart). Auditability: good — the raw `train_log.jsonl` exists and
`run_gate.py`'s fixed `reference_reached_at` (k-consecutive crossings + mandatory `estimator` field) is the
tool that owns the fact. **Fix: replace the bare exponent with the matched-step ratio + bucket interval,
exactly as §1.4 already did for v3enc (that row is the template); strike or re-derive `step ~250 / ~30×`;
delete the >2× projection. Same edit removes the `GATE_PROTOCOL:96-97` void field.**

### 6–12 (lower rank, still owed)

6. **`+0.0043` seam cosine** (`PAPER:59/833`, `OVERVIEW:51/165`, `LEDGER:129`) — redirected the v4 program; **no resolving artifact**. Stage the results file or re-run the ~0-GPU probe.
7. **The seven pod-only headline raw evals** (§3a) — rescue into `taniteval/results/`; reopen R2 as *partially* closed.
8. **`PAPER:763` / `LEDGER:110` / `:631` — H18 Δ 2.70 m** → +2.9568 m. Corrects **upward**; strengthens the claim, so low risk but currently wrong.
9. **CTRV 0.544 vs 0.523** (`PAPER:36/638` vs `PAPER:694`/`REGISTRY:64`) — the program's pre-registered honest bar has two values in one paper.
10. **"261 M" and "262.8 M"** (`PAPER:20/526/434`, `PAPER:1096`, `OVERVIEW:46`) — replace with the registry's measured 263,442,838 / 277,404,073; move CNCE/TMS onto the right arm's count.
11. **R14 latency triple-conflict** (`REGISTRY:181` vs two committed JSONs) — open since 07-20; one re-measure on an idle A40 or restate §6 from the artifacts.
12. **`REGISTRY:1488` / `LEADERBOARD:93` v3enc "running / not evaluated"** — the registry contradicts its own §1.4 (1.9654, verdict RESTART); `driving_flagship-v3enc-10k.json` and `windows_*.pt` are both in-repo.

---

## 7. ESCALATIONS (per Agent Operating Standard rule 3 — stated in the headline, not buried)

1. **`Paper/TANITAD_PAPER.md` publishes a retracted number with no caveat** (§7.5, Δ +0.044) **and an
   assurance that the retraction disproves** (§5:443). This is the external surface. **Owner decision needed
   today.**
2. **`Project Steering/GATE_PROTOCOL.md` — a binding protocol — quotes a field its own source of truth
   declares void** (`step 450`, `~12×`, `~30×`). A protocol that carries a void statistic cannot enforce the
   rule it exists to enforce.
3. **`tools/registry_lint.py` scans one file.** All three retracted-still-standing findings live outside its
   scan set. **Extend the file list to `Paper/TANITAD_PAPER.md`, `Project Steering/GATE_PROTOCOL.md`,
   `TanitAD Research Hub/HYPOTHESIS_LEDGER.md`, `Project Steering/PROGRAM_OVERVIEW.md`,
   `Benchmarks & Eval/LEADERBOARD.md`** — cheapest structural fix available, and it would have caught S1–S3.
4. **P2's evidence is stranded** (no JSON, no windows dump in-repo) and it is the basis of the v3 pivot.
5. **`MODEL_REGISTRY.md:63`, the definitional statistic line, carries the retracted label.** Every reader who
   stops at §0.3 inherits it. One-line fix.

---

## 8. DELIVERABLE MANIFEST

| artifact | location | exists elsewhere? |
|---|---|---|
| `PUBLISHED_NUMBER_PROVENANCE.md` (this file) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-published-number-provenance/` | **ONE PLACE ONLY** |
| `published_numbers.jsonl` (188 rows, machine-readable) | same folder | **ONE PLACE ONLY** |
| generator script | `scratchpad:gen_provenance.py` (transient; the JSONL is the artifact) | — |

**Not staged** (`git add` deliberately withheld): the brief's HARD CONSTRAINTS specify READ-ONLY on
everything except this report, and two sibling agents hold the index concurrently. Per `CLAUDE.md`'s git
hygiene rule the safe path is for the orchestrator to stage these two files itself after checking
`git status --short` for foreign staged entries. **Flagging rather than acting.**

**Read-only guarantee:** no file outside this folder was written, edited, staged, committed or pushed; no
pod was contacted; no GPU was used; `tools/registry_lint.py` was executed unmodified.
