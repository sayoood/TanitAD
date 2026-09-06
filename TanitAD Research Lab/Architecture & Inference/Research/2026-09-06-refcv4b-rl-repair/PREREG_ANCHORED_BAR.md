# PRE-REGISTRATION 6 — an ANCHORED bar on `S6_wmr`'s own scale, and the `progress_lead_cap` arm on a FRESH grid

**status: PRE-REGISTERED. ⛔ WRITTEN AND COMMITTED BEFORE THE FRESH GRID PRODUCED ANY NUMBER.**
**date:** 2026-09-06 · **owner:** RL gate stream (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **GPU: 0** (CPU arithmetic over re-computed geometry).

This file executes the Master Mind's rulings of 2026-09-06:

> **RULING 2.** ⛔ THE 0.30 BAR IS VOID ON THE NEW STATISTIC. Both "FAIL"s are inadmissible as
> written. … Derive a bar on `S6_wmr`'s OWN scale, pre-register it, and only then score. Anchor it
> to something measured, not chosen — the human's own value, a trivial-path reference, or a
> multiple of the statistic's bootstrap SD. **State the anchor and why it is the right one.**
> ⛔ Do not reuse 0.30, and ⛔ do not pick the bar after seeing where the arms land.
>
> **RULING 3.** ⭐ Authorised: map `progress_lead_cap True → "lead"` and evaluate it as a
> pre-registered arm on a FRESH grid, with both outcomes committed in advance.

⛔ **`G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE).** Nothing here re-scores it.
⛔ **`H-RL-GATE-STAT-1` (0.301183) and `H-RL-GATE-STAT-2` (0.456155) are VOID AS SCORED** — see §7.
They are not deleted and their arithmetic is not withdrawn; only the **verdicts** are void, because
the bar they were scored against was transposed from the rate's scale.

---

## 1. ⛔ THE HAZARD THIS DOCUMENT MUST DISARM FIRST, AND HOW

⚠️ **I ALREADY KNOW WHERE THE ARMS LAND ON THE OLD GRID.** `RESULT_GATE_STATISTIC.md` §3.4 publishes
all five cells: `nopert_min` 0.301183, `ship` 0.456155, `lead_min` 0.302476, `nopert` 0.295613,
`both_lq` 0.296900. ⇒ **Any bar I derive today and apply to those rows is a bar chosen with the
answers in view**, whatever procedure produced it. Ruling 2's second prohibition would be violated
in substance while being obeyed in form.

⇒ ⭐⭐ **THE PRIMARY VERDICTS ARE SCORED ON A FRESH GRID WHOSE VALUES DO NOT YET EXIST.** The same
fresh grid Ruling 3 requires for the cap arm carries the anchored bar as well. The old grid is
re-scored too, and that re-scoring is reported as **SECONDARY, with this hazard stamped on it.**

## 2. ⭐ THE FRESH GRID — fixed here, and built before any statistic is computed on it

* **Clips.** `rl_refcv3_min.py --mode fitlist` is `sorted(train_ids − eval_ids)`, shuffled by
  `random.Random(seed)`, first `n_fit` taken. Re-run at **`--seed 0 --n-fit 320`**, the first 120
  entries are **bit-identical to `fitlist_120.txt`** (md5 `f0baf21c935f6d298c4140669aef82f1`,
  MEASURED). ⇒ **the fresh grid is `cand[120:320]` — 200 clips drawn by the IDENTICAL procedure
  under the IDENTICAL seed, differing from the old panel only in the slice index.** There is
  therefore no selection effect of any kind between the two grids.
* **Disjointness**, MEASURED with a non-zero control: `fresh200 ∩ fit120` = **0**;
  control `fit120 ∩ fit120` = **120**. ⛔ A zero from a probe that cannot read is not an absence.
* **Lead block** built by `taniteval/tools/build_lead_block_b1.py` with the SAME flags as the old
  panel (`--k 10 --dt 0.2`), same ego/timestamp tars, same chunk map.
* **Labels** `s2_labels_v7.2_train.jsonl.gz`, md5 `0ff902130ce76886b8a925eceed9e3a5` — the same
  file, byte-for-byte, that the old panel used.
* **Everything else is held**: the same frozen refcv3 @ 40,284 base, the same
  `LEAD_MODE=track`, the same `GRID_S`, the same weights `{progress 0.3, collision 1.0,
  headway 0.3}`, the same `taniteval` episode-cluster bootstrap.
* ⛔ **The fresh grid is scored ONCE. There is no select/score sub-split on it**, because nothing
  is being selected on it — the statistic, the anchor rule and the arms are all fixed by this file.

## 3. ⭐⭐ THE ANCHOR — and why it is the right one

```
    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-3a
    PREREG_STATISTIC: S6_wmr
    PREREG_ANCHOR: signflip_episode_null
    PREREG_ALPHA: 0.05
    PREREG_DIRECTION: <=
    PREREG_RUNG: all
    PREREG_CELL: nopert_min
    PREREG_GRID: fresh200
```
```
    PREREG_HYPOTHESIS_ID: H-RL-GATE-STAT-3b
    PREREG_STATISTIC: S6_wmr
    PREREG_ANCHOR: signflip_episode_null
    PREREG_ALPHA: 0.05
    PREREG_DIRECTION: <=
    PREREG_RUNG: all
    PREREG_CELL: ship
    PREREG_GRID: fresh200
```

⛔ **There is deliberately NO `PREREG_BAR:` line, and the scoring tool REFUSES one.** A literal
number is exactly the object that can be transposed; an anchor rule cannot be.

### 3.1 The anchor, stated as a procedure

`S6_wmr(g) = Σ max(g, 0) / Σ |g|` over the paired per-window composed gaps
`g_i = composed(hold_v0)_i − composed(human)_i`. Lower is better; the reading is *"the share of the
total gap mass the reward pays the trivial constant-velocity path."*

> **ANCHOR — the one-sided α = 0.05 critical value of the EPISODE-CLUSTERED SIGN-FLIP
> PERMUTATION NULL of `S6_wmr`, computed on the rows being scored.**
>
> ```
> for b in 1 .. B          (B = 4000, seed = 0, fixed here)
>     s_e ~ Uniform{-1, +1}, drawn INDEPENDENTLY PER EPISODE e
>     g'_i = s_{e(i)} · g_i
>     null_b = S6_wmr(g')
> crit = percentile(null, 100·α) = the 5th percentile
> ```
>
> **VERDICT: PASS iff `S6_wmr_observed <= crit`**, equivalently permutation
> `p = mean(null <= observed) <= 0.05`.

### 3.2 ⭐ Why THIS anchor and not another

1. **It is MEASURED, not chosen.** Every input is the panel's own data — the real per-window `|g|`
   magnitudes, the real episode clustering, the real episode count. The only constant typed in is
   **α = 0.05**, which is the programme's own alpha, already carried by every episode-cluster
   bootstrap in this package.
2. ⛔ **It is on `S6_wmr`'s own scale BY CONSTRUCTION — it is a quantile of `S6_wmr` itself.** It is
   therefore **structurally incapable of the failure Ruling 2 convicts**: substitute a different
   statistic and the bar recomputes in that statistic's units. A transposed bar is not merely
   forbidden here; it is unrepresentable.
3. **Its meaning is exactly the gate's question.** The sign flip preserves each window's gap
   *magnitude* exactly and randomises only *which side the mass is paid to*, clustered by episode so
   the dependence structure is respected. The null is therefore the distribution of the mass share
   under **a reward with no systematic preference between the human and the trivial path**. ⇒ a
   reward that cannot land below the 5th percentile of its own no-preference null **is not
   distinguishably preferring the human**, which is the one thing the gate exists to establish.
4. ⭐ **It travels with the grid.** It is recomputed from whatever rows it is applied to, so it can
   never ride across a change of split, grid, statistic or episode count. ⚠️ That last one is not
   hypothetical: a critical value derived on the 40-episode SELECT split and applied to the
   33-episode SCORE split would be **too lenient**, because the null widens as episodes are removed.
   A constant would have carried that error silently.
5. **It needs no held-out half.** A permutation critical value is a deterministic function of the
   scored rows under a procedure fixed in advance — nothing is fitted, so nothing is over-fitted.

### 3.3 ⛔ WHAT THIS ANCHOR IS NOT — stated before it can be over-read

⚠️ **It is a FLOOR — a necessary condition, not a sufficient one.** Clearing a no-preference null
establishes that the reward's preference for the human is **detectable**, not that it is **large
enough to train on**. The magnitude is reported beside the verdict, always, and no arm may be
described as "the reward works" on the strength of this bar alone.

⚠️ **It answers ONE variance question: would another draw of EPISODES say this?** ⛔ Not another
training run (`H-ESTIM-SEED-1`), not another inference run. On this panel that is the only question
there is — **no arm is trained and no planner samples**; the cells are deterministic arithmetic
functions of one fixed geometry. ⭐ And the fresh grid is itself the strongest available replicate
for an episode-draw claim: a genuinely independent draw of episodes rather than a re-resampling of
the same ones.

### 3.4 ⛔ THE VACUITY CHECK — committed in advance, because a derived bar can still be useless

The friction-circle lesson: a bar that nothing can fail, and a bar that nothing can pass, are both
non-results. Reported with the verdict, and pre-classified here:

| condition | how it is reported |
|---|---|
| `0.5 − crit` **< 1 bootstrap SD** of `S6_wmr` | ⚠️ **WEAK** — nearly any reward clears it; the pass is stated as near-vacuous |
| `crit` **below** the frozen trivial-path reference (§3.5 anchor B) | ⚠️ **UNREACHABLE** — no reward on this grid could pass; the fail is stated as near-vacuous |
| otherwise | the verdict stands as written |

⛔ Neither classification changes the verdict. It changes what the verdict is allowed to mean.

### 3.5 The two OTHER anchors the ruling named — computed and REPORTED, never a verdict

So that the choice among the three options Ruling 2 offered is visible rather than asserted:

* **Anchor B — the trivial-path reference.** `S6_wmr` on `g = composed(frozen) − composed(human)`:
  the mass share paid to a path that **never moves** and therefore provably cannot drive. It
  calibrates the statistic's low end on this grid.
* **Anchor C — the SD multiple.** `0.5 − 2·SD_boot(S6_wmr)`, neutral minus twice the statistic's own
  episode-cluster bootstrap SD on the scored rows.
* ⛔ **Neither may become the verdict**, and no verdict may be re-read against them after the fact.

### 3.6 The STRICT reading — committed here, reported beside the verdict

`S6_wmr`'s bootstrap CI **upper bound ≤ crit** (whole-interval discipline). ⛔ **This is NOT the
primary verdict**, and the reason is stated in advance rather than discovered: the critical value
*already* carries the null's spread, so requiring the bootstrap interval to clear it too prices the
same uncertainty twice. Both readings are reported; the PRIMARY is the point estimate against
`crit`, equivalently the permutation p-value.

---

## 4. ⭐⭐ `H-RL-CAP-1` — the `progress_lead_cap` arm, on the fresh grid, both outcomes committed

```
    PREREG_HYPOTHESIS_ID: H-RL-CAP-1
    PREREG_STATISTIC: S6_wmr
    PREREG_PAIRED: ship - lead_min
    PREREG_RUNG: all
    PREREG_GRID: fresh200
```

`rewards._progress_cap_mode` maps `progress_lead_cap = True` to **`"achievable"`**. Ruling 3
authorises evaluating the mapping **`True → "lead"`** as a pre-registered arm. The cells are the
reward configurations already implemented and tested; nothing in `rewards.py` is edited by this
document, and ⛔ **no default is changed.**

| cell | `progress_lead_cap` | `headway_reduce` | what it is |
|---|---|---|---|
| `ship` | `"achievable"` (the shipped `True`) | `min` | the reward the code ships today |
| `lead_min` | `"lead"` | `min` | ⭐ **the arm** — the one-line mapping change |
| `nopert_min` | `off` | `min` | pre-repair, the STAT-1 cell |
| `nopert` | `off` | `q0.25` | the headway quantile alone |
| `both_lq` | `"lead"` | `q0.25` | both 2026-09-06 repairs together |

**PRIMARY STATISTIC:** the **paired** episode-cluster bootstrap of
`Δ = S6_wmr(ship) − S6_wmr(lead_min)` on the same fresh windows, `n_boot` 4,000, α 0.05.

⛔ **The discovery grid's `+0.153679 [+0.084053, +0.254735]` IS NOT THIS ARM'S RESULT** and may not
be quoted as one. It is the number that motivated the arm; this file exists to find out whether it
survives contact with episodes that did not produce it.

### 4.1 ⛔ THE OUTCOMES, COMMITTED BEFORE THE GRID IS READ

| # | condition on Δ (fresh grid) | ⛔ the committed reading |
|---|---|---|
| **A** | Δ **> 0** and the CI **excludes 0** | ⭐ **REPLICATES.** The shipped `achievable` mapping costs the reward on episodes that did not discover it. `True → "lead"` is a MEASURED repair; the recommendation goes to the PI **with the fresh number**, and the discovery number is retired to "what motivated the arm". ⛔ Still not folded in silently — the default stays as it is until the PI rules. |
| **B** | the CI **includes 0** | ⛔ **DOES NOT REPLICATE.** The +0.153679 is a property of the discovery grid. ⛔ **The escalation is WITHDRAWN as a proposal** and the discovery number is marked grid-specific in the register, in those words. |
| **C** | Δ **< 0** and the CI **excludes 0** | ⛔ **REVERSES.** The shipped cap is *better* on fresh episodes. The escalation is **retracted outright** and logged as a retraction with its class. |

⛔ There is no fourth outcome, no rung-shopping and no cell-shopping: the rung is `all` and the pair
is `ship − lead_min`, both fixed above. The conflict-time ladder is reported as a DIAGNOSTIC only,
and ⛔ **no bar may be moved to a sub-rung**, exactly as `PREREG_H-RL-GATE-STAT-1.md` §3 forbade.

### 4.2 ⭐ THE MECHANISM CHECK — a replication of the NUMBER is not a replication of the CAUSE

The discovery grid attributed **the whole** of Δ to `progress`, with `headway` and `collision`
identical to the digit across cells. Committed in advance:

* Outcome A is reported as **the same mechanism** only if, on the fresh grid, `headway` and
  `collision` are bit-identical across `ship` / `lead_min` / `nopert_min` **and** the whole of Δ
  sits in `progress`.
* If Δ replicates but the attribution moves, that is reported as **a different mechanism**, and the
  fixture proof (§4.3) is re-examined rather than re-quoted.

### 4.3 The fixture proof is RE-RUN, not inherited

`stack/tests/test_rl_gate_statistic.py`'s five mutation proofs of the cap's asymmetry — that
`ref_free = max(v0·H, min_ref)` truncates a faster-than-`v0` candidate while `hold_v0`, sitting at
exactly `v0`, is never truncated, with a discriminating control at 30 m where the `lead` cap does
bite — are re-run in the same turn and reported green or not at all.

---

## 5. ⛔ CONTROLS THAT MUST READ KNOWN VALUES

The 2026-08-22 rule in full. A probe that tunes on the data it scores manufactures results, and only
a control that must read a **known** value catches it.

| # | control | what it MUST read |
|---|---|---|
| 1 | **NULL** — the clone-and-add perturbation path at `d = 0.0` | max abs component delta **exactly 0.000000** |
| 2 | **CAP INERTNESS** — endpoint pinned, `cap="lead"` | max abs `Δprogress` **exactly 0.000000** |
| 3 | **CAP DISCRIMINATING** — the same edit *plus* the endpoint | `Δprogress` **> 0**, or control 2 is vacuous |
| 4 | ⭐ **CROSS-GRID TOOL IDENTITY** — the OLD fit120 rows re-scored through the tool used here | `S1_rate` on `nopert_min` = **0.441780259484316**, err **0.000e+00** |
| 5 | **GRID DISJOINTNESS** with a non-zero control | fresh ∩ fit120 = **0**; fit120 ∩ fit120 = **120** |
| 6 | **PERMUTATION NULL SANITY** — the null's median | within 0.02 of **0.5**; a null not centred at neutral means the flip is not symmetric and the panel is void |
| 7 | ⭐ **ANCHOR SELF-TEST** — the anchor applied to a DELIBERATELY NEUTRAL gap (`g` replaced by its own sign-flipped draw) | must **FAIL** the bar ~95 % of the time; an anchor a no-preference reward passes is not a bar |
| 8 | **FROZEN** — `frozen >= human` fraction, per cell | reported |
| 9 | `n` and `d` of every panel | printed in the table |

⭐ Control 4 is the one that makes the two grids comparable: it proves the tool did not change
between them, so a difference between grids is a difference in **episodes** and not in code.
⭐ Control 7 is the floor-and-ceiling control the 2026-08-22 rule demands: it forces the anchor to
convict a reward that is neutral by construction.

## 6. Estimator, tier, evidence class, vocabulary

* **Estimator:** episode-cluster bootstrap (`taniteval/ci.py` convention), `n_boot` 4,000, α 0.05;
  **paired** for `H-RL-CAP-1`. ⛔ Never `overlapping_holdout_se`.
* ⛔ **Which variance:** **EPISODES**. Not training (`H-ESTIM-SEED-1`), not inference. No arm is
  trained and no planner samples in this package.
* **Tier: T0** instrument probe, **NON-PARITY** RL-fit windows. ⛔ Not a driving number, and no
  capability claim is made from it.
* **Evidence class:** MEASURED (ours); artifacts under this package's `raw/`.
* ⛔ **Four families:** this document scores a **reward-gate statistic**, not a policy. It is not an
  eval, produces no capability claim, and asserts no four-family table.
* ⛔ **Vocabulary:** the v7.2 nav is a **first-class ROUTE input**; `os` is the deployment arm and
  `os_navzero` a **robustness ablation**. `oracle_sel` / `anchor_acc` / `sel_agrees_oracle` are
  INVALID on refcv4b and appear nowhere here.
* ⛔ **Coordinate discipline:** every ranking term is a **position query**; TTC is a **veto only**.
  The perturbation moves positions; no closing rate is introduced anywhere.

## 7. ⛔ WHAT BECOMES OF THE TWO TRANSPOSED-BAR RESULTS

⚠️ **They are marked VOID AS SCORED. They are NOT deleted**, and `RESULT_GATE_STATISTIC.md` stays in
the repo unedited. Ruling 2: *"Report the transposed-bar results as VOID rather than deleting them —
the honesty of scoring against a stated bar is worth preserving in the record."*

| what | status |
|---|---|
| the **arithmetic** — 0.301183 [0.201040, 0.433903] and 0.456155 [0.309354, 0.630646] | ⭐ **STANDS.** Measured, controlled, reproducible. |
| the **verdicts** — "`H-RL-GATE-STAT-1` FAILED", "`H-RL-GATE-STAT-2` FAILED" | ⛔ **VOID.** The bar was `G-REWARD`'s 0.30, transposed from the RATE's scale onto a different statistic. A threshold carries its regime. |
| the **"misses by 0.0012"** reading | ⛔ **VOID and specifically inadmissible** — it is a distance to a bar that does not apply, and it invites the reader to treat 0.30 as nearly cleared. |

## 8. What is NOT changed by this document

No reward term, no weight, no default, no guard, no banked result, and **no file owned by another
stream**. `progress_lead_cap`'s shipped default remains `True → "achievable"` whatever this arm
reports; changing it is a PI/Master-Mind decision and Ruling 3 authorised the **measurement**, not
the adoption.
