# Experiment (hardening) — frozen-WM planner W on the FULL 40-ep clean val

**Author:** frozenwm-40ep subagent · **Date:** 2026-07-24 (Europe/Berlin) · **Host:** eval pod
`tanitad-eval` (A40, FREE, under `gpu_lock frozenwm-40ep`, released) · **Evidence:** `MEASURED` — raw
`artifacts/results_40ep.json` (+ `_s300.json`) + `artifacts/perwin_40ep.pt`, harness `artifacts/run40.py`.
Frozen v1 WM read-only; **no WM parameter updated. pod1/pod2/pod3 untouched.**

**Hardening pass, NOT a new arm.** Goal: upgrade the 12-ep W=0.599 prototype to a decision-grade read on
the full 40-ep clean val (`physicalai-val-0c5f7dac3b11`, 40 episodes → **881 windows**).

---

## 1. Result summary — what hardened cleanly, and what could not

| result | status |
|---|---|
| **Frozen-WM SIMULATOR fidelity on all 40 eps** (oracle-action ceiling) = **0.4271** | ✅ **decision-grade, hardened** — reproduces v1's canonical ADE@2s to 4 dp |
| **W (deployable feed-forward planner) on all 40 eps** | ⚠️ **could NOT be hardened on the eval pod** — it has no train corpus, so W is data-starved; its real number stays the 12-ep **0.599** |

**Bottom line:** the *simulator* claim is now decision-grade on the full clean val; the *planner* number
could not be firmed up here because the eval pod carries only the val set, and a frozen-WM planner is
strongly training-data-dependent (§3). **For a from-scratch fallback comparison, use W = 0.599** (12-ep,
adequate train data) and the oracle ceiling **0.4271** as the WM's action→trajectory bound.

## 2. Decision-grade references on all 40 eps (deterministic — no training)

| reference (40-ep, 881 windows) | ADE@2s | CI95 (episode-cluster) | vs registry full-set |
|---|---:|---|---|
| **oracle-action ceiling** (frozen WM under GT actions) | **0.4271** | [0.369, 0.491] | **= 0.4271** ✅ exact |
| hold-v0 (go-straight) | 0.7822 | [0.589, 0.993] | 0.7876 ✅ |
| CV | 0.8377 | [0.621, 1.078] | 0.8377 ✅ exact |

**Apples-to-apples validated:** encoded 40 eps → **881 windows** (matches the gate), and all three
deterministic references reproduce the registry full-set numbers. The frozen-WM rollout under GT actions
*is* v1's operative number — the simulator is excellent on the full clean val, CI-separated far below CV.

## 3. W on the 40-ep val — the honest constraint and the data-limited number

**The eval pod has ONLY the 40-ep physicalai val — no train corpus** (probed: no `physicalai-train-*`, no
`_epcache`; HF cache holds models only; root disk 99 % full). The 12-ep prototype trained W on a *separate*
400-episode train corpus (**8,803** windows). That corpus is not on the eval pod and cannot be regenerated
here. So W was trained by **episode-disjoint 5-fold CV within the 40 val episodes** — each fold trains on
only ~32 episodes ≈ **700 windows**, **12.5× less data** than the prototype.

**W is strongly data-dependent, so this CV number is a data-starvation artifact, not W's capability:**

| W training budget | pooled ADE@2s (all 40, held-out) | CI95 | note |
|---|---:|---|---|
| 12-ep prototype (8,803 train windows) | **0.599** | [0.374, 0.854] | the valid W number (adequate data) |
| 40-ep CV, 1500 steps (~700 train win/fold) | **0.9502** | [0.795, 1.123] | best-fit; data-starved |
| 40-ep CV, 300 steps | 0.9726 | [0.838, 1.113] | fewer steps → *worse* (under-fit, not over-fit) |

Per-fold (1500 steps): **0.707 / 0.952 / 1.024 / 0.882 / 1.187** — high variance, the signature of
under-powered training sets. More steps *help* (300→0.973 vs 1500→0.950), confirming the folds are
data-**starved**, not over-fit. **A frozen-WM feed-forward planner needs more than ~700 windows to beat the
trivial floors reliably.**

**Paired episode-cluster bootstraps (40-ep, 1500-step W, same windows, B=2000):**

| contrast | Δ | CI95 | separated | reading |
|---|---:|---|:--:|---|
| W − oracle | +0.5231 | [+0.374, +0.683] | ✅ | data-starved W is far above its own ceiling |
| W − CV | +0.1125 | [−0.056, +0.295] | ❌ no | data-starved W merely **ties** CV |
| W − hold-v0 | +0.1681 | [+0.014, +0.340] | ✅ | data-starved W is **worse** than hold-v0 |

Contrast with the 12-ep W (0.599), which paired-**beat** CV (−0.247) and hold-v0 (−0.189): the difference
is entirely the **12.5× training-data gap**, not the eval set.

## 4. Verdict and recommendation

- ✅ **The frozen-WM simulator is hardened decision-grade on the full 40-ep clean val: oracle-action
  0.4271**, CI-separated far below every trivial floor — v1's WM reproduces its canonical number on all 881
  windows. This is the solid, quotable 40-ep fact.
- ⚠️ **W's deployable ADE could not be hardened to 40-ep on the eval pod** — it lacks the train corpus, and
  W is training-data-hungry (a 5-fold CV on 700 windows/fold gives ~0.95, a data-starvation artifact that
  merely ties CV). **Do not quote ~0.95 as W's capability.** W's valid fallback number remains the 12-ep
  **0.599** (adequate train data, paired-beats CV/hold-v0).
- **To get a clean decision-grade 40-ep W** (recommended if the fallback matters): put a train-episode
  frozen-state cache on the eval pod — either regenerate from the train corpus (needs the raw train data +
  disk, currently absent) or copy the **400-ep cache already built at `pod1:/root/frozenwm/cache/train`**
  (~0.2 GB) — then train W once on 8,803 windows and evaluate on all 40 val eps (one encode + one train
  pass, ~20 min). That was out of scope here (pod1 excluded by the brief); flagging it as the one-step
  unblock. **Escalation:** reuniting the pod1 train cache with the eval pod is the only missing ingredient.

**Scope note:** this hardened the WM-simulator claim (decision-grade) and diagnosed + quantified why the
planner number cannot be hardened on the eval pod as-provisioned. The controls and oracle ceiling are
final; the W row is explicitly a data-limited lower bound, bracketed by the valid 0.599.
