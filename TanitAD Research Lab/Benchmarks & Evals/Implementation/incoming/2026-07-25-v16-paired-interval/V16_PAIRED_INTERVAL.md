# v1.6 vs the deployed v1 — the decision-grade paired interval

**Date:** 2026-07-25 · **Arm:** `flagship-v16-ab-ft` (§1.4b, step 5,999) vs `flagship4b-speedjerk-30k`
(§1.2, step 29,999, THE DEPLOYED v1) · **Split:** `physicalai-val-0c5f7dac3b11` (clean; the leaky
`…f1b378f295ae` was never touched) · **Metric:** `ade_0_2s`, metric-BEV ego-frame, metres.

---

## 0. Bottom line

> **The gap this task was created to close was ALREADY CLOSED.** `MODEL_REGISTRY.md` §1.4b lines
> 490–495 already carry a **paired episode-cluster bootstrap**, and it is the correct estimator.
> Per the brief's P1 exit condition I did **not** re-score the checkpoints.
>
> I did **re-derive** it from the persisted per-window artifacts — CPU-only, no pod, no GPU, no
> checkpoint load — because a number that decides a headline must be **MEASURED**, not **INHERITED**.
> **It reproduces exactly, digit for digit.**

**VERDICT (one line, as asked):** v1.6's advantage over the deployed v1 is **not CI-separated — there
is no advantage to separate.** v1.6 is **behind** v1 on the point estimate (0.4375 vs 0.4271) and the
paired interval spans zero, so the two arms are a **statistical tie**. **"Best ADE in the program" is
not merely unsupported by the decision-grade estimator — it is contradicted by the point-estimate
ordering as well.** The registry body already says this; **only the §1.4b headline still does not.**

---

## 1. P1 — what interval is attached to the v1.6 claim, and which estimator produced it

**TWO intervals are attached to this arm, and they are not the same kind of object.**

| where | number | estimator | admissible? |
|---|---|---|---|
| §1.4b table, line 470 | `0.4886 ± 0.0800` (heldout) | **`overlapping_holdout_se`** | ❌ deprecated **and** cross-eid-family |
| §1.4b block, lines 490–495 | `Δ +0.0104 [−0.0888, +0.1147]` | **`paired_episode_cluster_bootstrap`** | ✅ **decision-grade** |

**The `± 0.0800` is `overlapping_holdout_se` — MEASURED, not assumed.** From the raw eval JSON
(`taniteval/results/eval_v16_flagship-v16-ab-ft.json`, `heldout.model.ade_0_2s`): `mean 0.4886`,
`std 0.1155`, `ci95 0.0800`. Reproducing the deprecated formula:

```
1.96 * 0.1155 / sqrt(8)  = 0.0800   ← exact match, n_splits = 8
1.96 * 0.1155 / sqrt(40) = 0.0358   ← what a 40-episode SE would have been
```

That is the pre-2026-07-20 `bench._agg` arithmetic verbatim — the block historically mislabelled
"8-split episode-disjoint jackknife", which is neither a jackknife nor a valid SE.

**So the answer to P1 is: the headline ± IS the bad estimator, but the arm ALREADY has the good one
too.** The paired bootstrap was run 2026-07-21 and recorded. **The task's stated gap does not exist.**

### 1.1 What DOES still exist — and it is not an estimator problem

Three residual defects, all in `MODEL_REGISTRY.md` §1.4b:

1. 🔴 **The section HEADER (line 460) still reads `⭐ best ADE in the program`** — contradicted by its
   own body 14 lines later ("**❌ v1.6 does NOT beat v1**") and 30 lines later ("v1.6 and v1 are
   statistically INDISTINGUISHABLE"). This exact claim was **already retracted on 07-21**
   (`RETRACTION_LOG.md`, class **C1**). The retraction landed in the body and in `CLAUDE.md`; **the
   headline was never edited.**
2. 🔴 **The process note (lines 528–529) repeats it**: "*5,999 is the best ADE in the program*". It
   **evaded the line-based grep** because the phrase wraps across a newline (`the best` / `ADE in the
   program`) — a live instance of *absence found at one location is not absence*, in its
   presence-detection form. Any future sweep for this claim must be **multiline**.
3. 🟡 **No raw JSON existed for the paired run.** The repo held
   `paired_v3enc10k_vs_flagship30k.json` but no v16 equivalent, and the §1.4b block does not state
   **B**. The number was reproducible only from prose. **This deliverable emits the missing artifact.**

---

## 2. P2 — checkpoints: not needed, and one premise of the brief is wrong

**I did not load either checkpoint, and re-scoring would have produced no information.** Both arms'
per-window predictions were persisted by their own canonical eval runs:

- `taniteval/results/windows_flagship-v16-ab-ft.pt` — 881×4×2, written 2026-07-21 04:22 by `eval_flagship_v16.py`
- `taniteval/results/windows_flagship-30k.pt` — 881×4×2, the deployed v1

The paired bootstrap is a **pure function of these two arrays**. Re-scoring would have added GPU load
to a pod for a bit-identical result.

> ⚠️ **CORRECTION TO THE BRIEF'S PREMISE.** The brief states *"v1.6 is also on HF as
> `Sayood/flagship-v16-ab-ft`, though the registry lists it at 0.00 GB."* **Neither half holds.**
> - **§1.4b has no `Location` row at all** — unlike §1.2, this arm's entry never recorded where its
>   weights live. The only path in the section is the *eval JSON* on pod2.
> - **`Sayood/flagship-v16-ab-ft` is not in the HF inventory** (`…/2026-07-25-refc-hf-push/NOTE.md`
>   §5, 9 repos listed). `LOOP_STATE.md:235` says the push is **BLOCKED on a publish decision**
>   (private storage full).
>
> **Consequence — a live stranding risk, unrelated to this measurement:** the v1.6 checkpoint exists
> on **exactly one disk**, `tanitad-pod2:/workspace/experiments/flagship-v16-ab-ft/` — and pod2 is
> currently training flagship-v4 and is off-limits. That is a *finish-before-you-start* exposure and
> a §1.4b registry-completeness gap. **Flagged, not acted on** (publishing is Sayed's call).

---

## 3. P3 — the paired episode-cluster bootstrap

### 3.1 Alignment was PROVEN, not assumed

A paired test is valid only if both arms were scored on the same windows in the same order **and**
clustered on the same episode partition. The naive check *fails* here, and the failure is instructive:

```
eid_labels_identical ............ False     ← the two eid FAMILIES
gt   max abs diff ............... 0.0
cv   max abs diff ............... 0.0
speed max abs diff .............. 0.0
wp_steps identical .............. True
eid relabel is a BIJECTION ...... True      (40 ↔ 40)
EPISODE PARTITION IDENTICAL ..... True      ← the load-bearing invariant
```

`bench.py` labels episodes by **file index 0–39**; `eval_flagship_v16.py` by the **real
`episode_id`** (e.g. `808464434`). Requiring identical *labels* would have refused a perfectly valid
pairing. I verified the **partition** — the sorted set of window-index groups — is identical, and the
relabel is a bijection. **This is checked in code from the data, not inherited from the registry's
"consistent 1-to-1 relabel" note.** The script *refuses to run* if it does not hold.

### 3.2 The result

```
ARM A  flagship-v16-ab-ft (v1.6, step 5,999)   full-set ADE@2s = 0.43746
ARM B  flagship4b-speedjerk-30k (v1, 29,999)   full-set ADE@2s = 0.42711

PAIRED EPISODE-CLUSTER BOOTSTRAP  (taniteval/ci.py :: paired_episode_cluster_bootstrap)
    Δ (v1.6 − v1) = +0.0104 m
    CI95           = [−0.0888, +0.1147]
    separated      = FALSE          ← the interval CONTAINS 0
    p(Δ > 0)       = 0.568
    B = 2000  ·  episodes = 40  ·  windows = 881  ·  seed = 0  ·  reducer = mean
    per-window correlation = 0.4532
```

**Δ is POSITIVE, and for ADE lower is better — so the point estimate favours the DEPLOYED v1.**

**Matches `MODEL_REGISTRY.md` §1.4b exactly** (`+0.0104`, `[−0.0888, +0.1147]`, `separated = FALSE`,
40 episodes, 881 windows, corr `0.453`). The registry's number is hereby **MEASURED-confirmed**.

**Robustness — the estimand does not depend on which eid family you cluster on.** The partition is
identical, but the bootstrap draws resample `np.unique(eid)`, whose sort order the relabel changes.
Both realisations:

| eid family | Δ | CI95 | separated |
|---|---:|---|:--:|
| file-index (`bench.py`) | +0.0104 | [−0.0888, +0.1147] | FALSE |
| real `episode_id` | +0.0104 | [−0.0889, +0.1117] | FALSE |

The spread between them **is** the Monte-Carlo noise at B = 2000: ~0.003 m on the bound, ~30× smaller
than the interval. **The conclusion is not an artifact of the labelling choice.**

### 3.3 OLD interval next to NEW — the width difference, as requested

| arm | OLD `overlapping_holdout_se` (deprecated) | NEW episode-cluster bootstrap | widening |
|---|---|---|---:|
| v1.6 | 0.4886 **± 0.0800** | 0.4375 **[0.3423, 0.5501]** (±0.1039) | **1.30×** |
| v1 | 0.4522 **± 0.0312** | 0.4271 **[0.3675, 0.4871]** (±0.0598) | **1.92×** |

Both land inside the program's documented **1.28–2.06×** too-narrow band (`CI_RECOMPUTE_2026-07-20.json`).

> ⚠️ The two OLD numbers are **also not comparable to each other** — 0.4886 and 0.4522 come from
> *different random episode partitions* (the eid-family split hashes id **values**). The old headline
> comparison was doubly invalid: wrong estimator **and** cross-family.

**The tie is not a power artifact of a sloppy estimator.** The paired test is the *most* powerful
valid option here, and it is **tighter** than the (invalid) quadrature combination:

```
INVALID quadrature on deprecated ±  : ±0.0859
INVALID quadrature on ep-cluster ±  : ±0.1199
VALID   paired half-width           : ±0.1018   ← tighter than quadrature, and still spans 0
```

Pairing cancels the shared per-window difficulty (corr 0.453) and still cannot separate the arms.
**With the best available estimator, there is no ADE difference to detect.**

### 3.4 Secondary — §1.4b states G1 both ways; the paired test settles it

§1.4b contains an **internal contradiction** on its own gates:

- **line 474:** `G1 (beat REF-C 0.458) ❌ · G2 (beat v1 0.4522) ❌`
- **line 520:** `✅ G1 (<0.458) ✅ G2 (<0.4522)`

Both readings are arithmetically correct **on different bases** — ❌ uses heldout 0.4886, ✅ uses
full-set 0.43746 — and the section never says which basis the gate was defined on. Since REF-C-XL's
windows are also persisted and pass the same partition check, the paired test resolves it without
picking a basis:

```
Δ (v1.6 − REF-C-XL) = −0.0340 m   CI95 [−0.1060, +0.0511]   separated = FALSE
                                   (REF-C-XL full-set 0.47144)
```

**G1 is a TIE too.** Neither the ❌ nor the ✅ is right: on the decision-grade estimator **v1.6 is
statistically indistinguishable from both v1 and REF-C-XL**. It is a third member of the existing
three-way tie at the top of the leaderboard — which is precisely why `LEADERBOARD.md` gives it **no
rank**.

---

## 4. Retraction-class assessment

**This is NOT a new retraction.** The claim was retracted on 07-21 (`RETRACTION_LOG.md`, class **C1** —
*faster-moving source than the harness*), and the paired bootstrap that superseded it was recorded the
same day. Nothing in this work overturns a live ordering.

**What it exposes is a different failure mode, and it is worth its own class signal:**

> **The retraction was written into the BODY and the headline was left standing.** For four days the
> program's single source of truth opened §1.4b with `⭐ best ADE in the program` and refuted it
> fourteen lines later. A reader who reads headers — which is what headers are for — got the retracted
> claim. **A retraction that does not edit the headline has not fully landed.**

**Proposed for `RETRACTION_LOG.md`** (append-only; PROPOSED, not written by me):

```
| 07-25 | *"v1.6 — ⭐ best ADE in the program"* SURVIVING in the MODEL_REGISTRY §1.4b **header** and
process note (l.528) for 4 days after the 07-21 C1 retraction corrected the body | **C4 (propagation)**
| Zero decision cost — caught before it left the registry. Root cause is NOT a bad number: the paired
bootstrap was correct and recorded the whole time. It is that **a retraction edited the prose and not
the headline**, and a line-WRAPPED second instance ("the best\nADE in the program") evaded the grep
that would have caught it. **New rule earned: a retraction sweep must (a) re-read the section HEADER,
and (b) be MULTILINE.** |
```

---

## 5. PROPOSED corrections — I did NOT edit these files

### 5.1 `Project Steering/MODEL_REGISTRY.md` §1.4b — line 460

```diff
-### 1.4b flagship-v1.6 — `flagship-v16-ab-ft` — ✅ **COMPLETE at 5,999** · ⭐ best ADE in the program
+### 1.4b flagship-v1.6 — `flagship-v16-ab-ft` — ✅ **COMPLETE at 5,999** · ⚖️ **ADE TIE with the deployed v1**
```

### 5.2 `Project Steering/MODEL_REGISTRY.md` §1.4b — lines 527–530 (process note)

```diff
 ⚠️ **Process note (mine).** At step 2500 a transient spike (oracle 2.08, gnorm 161) plus a monotone
-canary trend led me to report a "decisive failure". **It recovered completely** — 5,999 is the best
-ADE in the program. The confirming-eval discipline saved the run; the premature *communication* did
-not. Second such call this session. **A single post-spike eval is not a verdict.**
+canary trend led me to report a "decisive failure". **It recovered completely** — 5,999 finishes at
+**statistical parity with the deployed v1** (paired Δ +0.0104, CI spans 0). The confirming-eval
+discipline saved the run; the premature *communication* did not. Second such call this session.
+**A single post-spike eval is not a verdict.**
```

### 5.3 `Project Steering/MODEL_REGISTRY.md` §1.4b — the paired block (lines 490–495), add B + artifact

```diff
 **✅ PAIRED EPISODE-CLUSTER BOOTSTRAP — run 2026-07-21, and it settles the arm:**

 ```
 Δ(v1.6 − v1) = +0.0104 m   CI95 [−0.0888, +0.1147]   separated = FALSE
 full-set 0.4375 vs 0.4271 · per-window corr 0.453 · 40 episodes · 881 windows
+B = 2000 · seed 0 · reducer mean · taniteval/ci.py::paired_episode_cluster_bootstrap
+Δ > 0 and lower-is-better ⇒ the point estimate favours the DEPLOYED v1.
 ```
+
+*Re-derived independently 2026-07-25 from the persisted window dumps — reproduces exactly.
+Raw artifact: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/`
+`2026-07-25-v16-paired-interval/v16_vs_v1_paired_bootstrap.json`.*
```

### 5.4 `Project Steering/MODEL_REGISTRY.md` §1.4b — resolve the G1/G2 contradiction (lines 474 / 520)

Replace the bare ❌/✅ on G1 and G2 with the basis-free paired result:

```
⚠️ G1/G2 are stated BOTH ways in this section (l.474 ❌❌ on heldout 0.4886; l.520 ✅✅ on full-set
0.43746). Neither reading is decision-grade — both are point estimates against a threshold. On the
paired episode-cluster bootstrap BOTH gates are TIES:
    vs v1        Δ +0.0104  [−0.0888, +0.1147]  separated = FALSE
    vs REF-C-XL  Δ −0.0340  [−0.1060, +0.0511]  separated = FALSE
v1.6 is statistically indistinguishable from both. Treat G1/G2 as UNRESOLVED, not as pass or fail.
```

### 5.5 Add a `Location` row to §1.4b (currently absent)

```
| **Location** | `tanitad-pod2:/workspace/experiments/flagship-v16-ab-ft/` — **SINGLE DISK.**
  HF push BLOCKED on a publish decision (private storage full, `LOOP_STATE.md:235`). Not in the
  HF inventory. ⚠️ stranding exposure. |
```

### 5.6 `Benchmarks & Eval/LEADERBOARD.md` — **NO CORRECTION NEEDED**

I checked it against this result and it is **already fully correct**. It requires no edit:

- §0 already names the estimator **and states B = 2000**;
- line 78 gives v1.6 `0.4375 [0.3423, 0.5501]` with the deprecated heldout in a column marked
  `(DEPRECATED)` and flagged `⚠`;
- line 96 gives it **no rank**, stating "*an ADE tie with v1, so it cannot be ordered*";
- lines 103–106 carry the paired Δ, the cross-eid-family warning, and `NOT separated`.

**The leaderboard was right and the registry headline was wrong — the opposite of the usual drift
direction.** Worth noting: the derived, agent-maintained document out-disciplined the source of truth.

---

## 6. Reproduce

```bash
# CPU-only, no pod, no GPU, ~10 s. Refuses to run if the arms are not aligned.
/c/Users/Admin/venvs/tanitad/Scripts/python.exe \
  "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-v16-paired-interval/verify_v16_paired.py"
```

## 7. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| `V16_PAIRED_INTERVAL.md` (this file) | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-25-v16-paired-interval/` | ✅ repo, staged |
| `v16_vs_v1_paired_bootstrap.json` (raw result) | same dir | ✅ repo, staged |
| `verify_v16_paired.py` (reproducer) | same dir | ✅ repo, staged |
| §5 correction text for `MODEL_REGISTRY.md` §1.4b | §5.1–5.5 above | 🟡 **PROPOSED — not applied** |
| `RETRACTION_LOG.md` entry | §4 above | 🟡 **PROPOSED — not applied** |
| `LEADERBOARD.md` | — | ✅ **verified correct, no change needed** |

**Nothing was left on a pod.** No pod was touched: `tanitad-pod3` was not needed, and pod1/pod2/eval
were never contacted. No checkpoint was read, modified, or re-scored. The parity corpus was not
re-selected or mutated.

**Escalation for the orchestrator:** §5.1 and §5.2 edit `Project Steering/MODEL_REGISTRY.md`, which
this agent may not modify. They are one-line changes to the program's source of truth and should not
wait — the stale headline is the highest-visibility surface of an already-retracted claim.
