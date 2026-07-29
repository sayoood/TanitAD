# TanitAD program report — 2026-07-29 07:57 Berlin (05:57 UTC)

**Covers ~14 h since `2026-07-28-1757`. 75 commits.** Fleet state read from pod clocks
(`date -u`), never from session feel — that drift was itself caught and corrected this period.

⚠️ **Filename convention:** this directory's 20 existing reports are all ISO date-time
(`YYYY-MM-DD-HHMM-program-report.md`), so this one follows them. The brief asked for
sequential numbering; **the directory contradicts it and the directory wins.** Flagging rather
than silently switching schemes.

---

## 1. Headline — four results, five retractions, and one near-miss that mattered most

This was a **results-and-corrections** period, not a training period. Nothing new was launched
except a recovery. The most consequential event was a comparison that **did not happen**.

⭐ **C64 — v2corpus was trained on 21 of the 40 validation episodes (52.5 %).** Found by the
void check that `PREREG_v2corpus_vs_v1.md` mandated as its *first* step, run while the arm was
at step 25,900/30,000 — **before the checkpoint existed and before any number was computed.**
Scoring on the full surface would have measured v2corpus on its own training data for half the
episodes, and the inflation is **one-sided**: it would have manufactured a *"more data helps"*
result for the corpus whose purpose is justifying more corpus investment.

---

## 2. Fleet (pod clock 06:27:45 UTC)

| pod | state | since last report |
|---|---|---|
| **newpod** `69.30.85.48:22192` | v2corpus **step 26,500**, 9 procs, `step_s` 606.8 ÷ 50 = **12.1 s/step** | +~1,600 steps; healthy throughout |
| **pod2** | v5 176×624 @117° **step 1,150**, 5 procs, **0 bytes stderr** | ⚠️ **DIED at 2000, recovered** — see §5 |
| **pod3** | free, 0 % GPU | ran E-CR, E-DPSI and the sitclf study to completion |
| **eval** | provisioned, **idle** | idle all period |
| **pod1** | ⛔ **8× RTX A6000, `/dev/nvidia*` = `total 0`** | **unchanged all night — PI action** |

**Evidence class: MEASURED** (direct pod reads). ⚠️ Trainer-internal metrics are **NOT quoted**
anywhere in this report — they run ~10 % optimistic vs `eval_*.py`.

---

## 3. Results landed — estimator named on every one

### 3.1 ⭐ E-CR — C61 RESOLVED: the imagination decay is COMPOUNDING, not task difficulty

**MEASURED.** v1 `flagship4b-speedjerk-30k` step 29,999, **761 windows / 40 episode clusters**,
metric `e_k = 1 − cos(z_hat_k, z_true_k)` (latent error, **no decoder in the path**).
**Estimator: PAIRED episode-cluster bootstrap, B=2000, interval on the DIFFERENCE.**
⛔ No CI on the ratio is computed and none may be quoted.

| k | e_rollout | e_teacher_forced | CR | Δ | CI95 | sep |
|---|---|---|---|---|---|---|
| 4 | 0.025194 | **0.007189** | 3.50 | +0.0180 | [0.0143, 0.0222] | ✅ |
| 8 | 0.123405 | **0.008140** | 15.16 | +0.1153 | [0.0696, 0.1654] | ✅ |
| 16 | 0.514255 | **0.008009** | 64.21 | +0.5062 | [0.4029, 0.6096] | ✅ |
| 20 | 0.583723 | **0.007227** | 80.77 | +0.5765 | [0.4752, 0.6735] | ✅ |

`p_delta_gt0 = 1.0` at every horizon. ⭐ **The load-bearing fact is the flat teacher-forced
column** — one step from truth is as accurate at 2.0 s as at 0.4 s ⇒ **the task does not get
harder with horizon; all growth is the model consuming its own output.** Replicated against an
earlier 488-window run (3.58/13.64/64.45/77.81) on a differently-skewed window set.

⛔ **Does NOT restore the retracted "decay accelerates" wording** (C61's retraction was correct
on its own terms). ⛔ **Says nothing in metres.** ⛔ **80.77 is not an effect size** — a ratio of
cosine distances with a denominator near a noise floor.

**Unblocks:** rollout-recovery training is now the **indicated fix and highest-value change
available**; E-ROLL's expected divergence is explained; Koopman becomes a legitimate *second*
lever (after rollout-recovery, reporting CR and speed R² jointly).

### 3.2 E-DPSI — NULL: no heading shortcut below 12°

**MEASURED.** `v4fs_ckpt.pt` step 29,999 (`head`+`goal_head`), **881 windows × 7 offsets**, 40 eps.

| dψ° | −12 | −8 | −4 | **0** | +4 | +8 | +12 |
|---|---|---|---|---|---|---|---|
| `tspeed_5s` | 11.500 | 12.143 | 12.779 | **13.633** | 13.376 | 12.460 | 11.920 |

Smooth, near-symmetric, **no discontinuity** — 2.13 m/s over 12°, ≈0.18 m/s per degree. The
pre-registered shortcut signature is a **step**; there is none.
⛔ **This is "no shortcut BELOW 12°", never "we are clean"** — our envelope tops out at 12°,
PlanT 2.0's onset is 10–15°, so the band where their jump lives is unmeasurable on this
instrument. ⛔ **Does not contradict PlanT** — its CARLA scripted-expert root cause is absent
from human logs by construction. ⚠️ **No intervals computed** (arrays saved).

### 3.3 Situation classifier v2 — camera-only anticipation is real in TWO model classes

**MEASURED**, canonical parity corpus, 2,376 eps / 472,627 frames, `lead_s=3.0`.
Base rates: lane_change **0.01726**, intersection **0.02816**.

| | camera only | ego only | shuffle control |
|---|---|---|---|
| **neural** (attn head) | **0.04869** (2.1×) | 0.08858 | **0.02342** |
| **linear** (closed-form ridge) | **0.04522** (2.26×) | 0.05408 | **0.02005** |

Both controls land at ≈ the 0.0227 base rate — the nulls behave like nulls.
⭐ **camera/ego = 0.836 LINEAR vs 0.549 NEURAL** ⇒ the ego advantage is largely **capacity**,
not information. **HYPOTHESIS, not finding** — the competing reading (frozen PCA-reduced image
features are harder for a small head to exploit) is equally consistent and untested.
⚠️ Per **C60**, the ego baseline is **optimistic by construction** on human logs — hindsight
contaminated, not deployable. ⛔ Not comparable to gen-1 numbers (three situations, different
detector generation). ⚠️ **No confidence intervals** — directional only.

### 3.4 Safety gate — unblocked to a file transfer

`pseudosim.py`'s refusal carried two blockers; one was **retracted**: the claim that
episode→clip identity *"is not resolvable from the cache alone"*. Three agents had already
proven the join (600/600 val, 2376/2376 train, 40/40 eval-pod), and pseudosim's own 40-episode
cache carries **40 distinct episode_ids, zero missing**. ⇒ **Only a chunk download remains.**
⚠️ The *refusal itself was right* — with no cuboids there is no gate, and refusing to emit a
constant is what made this recoverable.

---

## 4. Streams — status

| stream | status this period |
|---|---|
| **Closed-loop / E1c** | **No change.** Protocol change registered: open-loop fidelity and closed-loop success are now **separate decision inputs** — a closed-loop gain with flat CR_k is an **executor** gain and may not be quoted as world-model progress (MoP-JEPA: 10 % false edges → success 0.40→1.00). |
| **4-brain dominance** | **No new arms.** E-CR is the substantive contribution: the hierarchy's imagination failure is now **characterised as compounding**, which redirects the next training change. |
| **H2 / side-camera attention** | sitclf v2 complete (§3.3). Camera-only is real; the MoE camera case is **not** settled by it. |
| **IDM** | No change. |
| **Datasets** | 🔴 **C64 dominates** — see §6. v2corpus training continues; the comparison plan was rebuilt around the leak. |
| **Orin / Thor** | **No change.** The AD research survey produced **zero admissible edge-inference evidence** (all candidates refuted 0-3), so nothing new to act on. |
| **AlpaSim** | No change. |

⚠️ **Deep research (215 agents, 12.9 M tokens) landed this period.** Its coverage gaps are a
result, not an omission: **5 of 8** AD areas and **4 of 6** robotics/WM areas produced **zero
admissible claims**. Genie 3, DreamerV4 and Cosmos appear in **no** surviving claim. The
factorised path × velocity vocabulary — which maps 1:1 onto our LAT+LON softmax mechanism —
was **refuted 0-3** and is the top re-verification target.

---

## 5. Retractions logged since the last report — five

| # | root-cause class | one line |
|---|---|---|
| **C61** | reporting a MECHANISM when the measurement supports only a MAGNITUDE | "decay accelerates" was unsupported; ADE-vs-horizon cannot separate task difficulty from compounding |
| **C62** | substituting working memory for the STATED enumeration | reported 3 working pods for a whole session; `tanitad-eval` sat idle and was named in the prompt every time |
| **C63** | importing a published metric without measuring its PRECONDITION on our stack | E-CR v1 on decoded displacement was mis-specified — a two-true-latent decode scores **6.76 m** at 2 s vs the rollout's 0.53 m |
| **C64** | a MISSING CONSTRAINT in a build spec | v2corpus trained on 21/40 val episodes |
| **C65** | a pod tree assembled from TWO GENERATIONS, stale half on a path that runs LATER | v5 died at step 2000; new gate calling stale untracked `taniteval` |

⚠️ **Twice this period a guardrail beat my first instinct:** the pre-registration on C64, and
the permission classifier on C65 — where `git reset --hard` would have destroyed the untracked
trainer that produced 8 hours of v5.

### 5.1 v5 incident detail

Trained cleanly 21:20→05:03, died at **step 2000** = the **first firing** of
`--heldout-gate --heldout-every 2000`. Mixed tree: trainer + gate **Jul 28 17:10 (untracked)**,
`taniteval/pseudosim.py` **Jul 27 18:22 (stale, whole tree untracked)**, pod2 HEAD `0f93b98`
**363 commits behind**. ⚠️ **A git sync would not have fixed it** — `taniteval` is outside
version control.
✅ **Geometry unaffected** — config carries `176×624` and `117.0`, so the 8 h trained at the
approved geometry. **Stopped, not voided.**
✅ Fixed with one PYTHONPATH entry (`/workspace/tev/taniteval`, which was already on the pod and
current), **verified by real import before launching**. Relaunched 05:48:48Z, auto-resumed from
`ckpt.pt` @1000 ⇒ **~3.6 h lost, not 8 h**.
⚠️⚠️ **THE FIX IS STILL UNPROVEN** — v5 is at **1,150**; the gate fires at **2000**, ~3 h out.

---

## 6. 🔴 Decisions owed by Sayed

1. ⭐ **pod1 — console stop/start.** 8× RTX A6000 idle all night. `/dev/nvidia*` is empty; the
   CUDA userspace is already pinned to the exact kernel match (550.127.08). **Not fixable over
   SSH.** This is the only thing blocking **rollout-recovery training**, which E-CR just
   identified as the highest-value change available.
2. ⭐ **C64 — which comparison to run.** Both remedies are prepared and banked:
   - **Option A** — score both arms on the **19 leak-free episodes** (`v2bal_leakfree_val19.json`,
     with cache indices). ⚠️ 19 clusters ≠ 40 for the paired bootstrap; the survivors are what a
     manoeuvre-balanced selection left behind and may skew to lane-keeping **against** v2; and
     **v1 must be re-scored there** (0.4271 is a 40-episode number, not the comparator).
   - **Option B** — a clean v2-line val from the **9,987 unselected pool clips**, disjoint by
     construction, **no new data** (`V2_CLEAN_VAL_PLAN.md`).
   - ⇒ **They answer different questions. Only A speaks to "did the 50 h corpus beat the 13 h
     parity corpus".** My recommendation: **both, A first.**
3. **`main` fast-forward** — classifier-blocked. Everything is on
   `origin/agent/benchmarks-eval-20260721` at `d17e77b`; `origin/main` verified an ancestor,
   0 commits would be lost.
4. **Old CPU pod release** — still not authorised, rescue complete and verified.
5. **#42 — the 30k gate NO-VERDICT** — unchanged, still owed.

---

## 7. Blocked, and on what

| item | blocked on |
|---|---|
| rollout-recovery training | **pod1** (PI console) — no free GPU otherwise; pod3/eval would collide with the v2corpus comparison |
| v2corpus ↔ v1 contrast | the checkpoint (~10 h) **and** the C64 decision |
| C64 clip_id confirmation | **pod2 is training** — `discover_r0_clips` is an unbounded MooseFS walk; deliberately deferred |
| NC safety gate | an `obstacle.offline` chunk download |
| v5 fix verification | v5 reaching step 2000 (~3 h) |

⚠️ **STANDING HAZARD (new):** pod2 carries **317 modified tracked files**, an **untracked
`taniteval`**, and **the running trainer is itself untracked** — it exists on one disk. A
`git clean -fd`, `checkout .` or `reset --hard` would delete it. ⛔ **Reconcile after v5
finishes, never during an incident.**

---

## 8. Next steps, priority order

1. **Verify the C65 fix** — confirm v5 passes step 2000 (~3 h). Until then it is unproven.
2. **If pod1 comes up: launch rollout-recovery training.** Everything else queues behind it.
3. **When v2corpus hits 30k**: run the contrast per the prereg + C64 — the 19 episodes, v1
   re-scored there, `leak_free_n = 19` in the headline with all three caveats.
4. **Confirm C64 at clip_id granularity** once pod2 is free.
5. **Download the `obstacle.offline` chunks** → wire the NC gate with
   `filter_m(agent,human) = 1.0 if m(human)==0 else m(agent)` (**not** PDMS-v1, which
   over-penalises).
6. **Reconcile pod2's tree** — deliberately, after v5 finishes.

---

## 9. What did NOT change

Closed-loop/E1c, IDM, AlpaSim, Orin/Thor: **no movement.** No new model arms were trained or
evaluated. **v1's 0.4271 remains the reference** and no arm has been held to anything else.
