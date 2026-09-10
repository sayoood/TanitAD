# ⛔ P1 IS OUT — agent conditioning does NOT enter `refcv5-cap-b1-v72-40k`

**`D-P1-AGENTCOND-1`** · **Date** 2026-09-07 (Europe/Berlin) · **Agent** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **Evidence class** MEASURED (ours) ·
**Tier T1 — self-action open loop.** ⛔ Nothing here is closed loop.
**Pre-registered at** `PREREG.md` (banked before any outcome number; Amendment 1 added while the
seed-0 arms were still training).

## 1. The verdict, and the clause it triggers

⛔ **P1 IS OUT.** `PREREG.md` §5 FAIL clause: *"FAIL if the distance-keeping contrast is separated
in the unfavourable direction."* It is, on **all three** distance-keeping metrics.

| the gate's own headline family | `head − off` | verdict |
|---|---|---|
| **min headway (m)** — higher is safer | **−0.5015 [−0.7712, −0.2366]** | ⛔ SEPARATED **WORSE** |
| **min time-gap (s)** — higher is safer | **−0.0702 [−0.1074, −0.0385]** | ⛔ SEPARATED **WORSE** |
| **min TTC (s)** — higher is safer | **−2.1959 [−4.1408, −0.6110]** | ⛔ SEPARATED **WORSE** |

n = 779 windows / 28 episodes (719 for time-gap), paired episode-cluster bootstrap, 10,000
resamples. ⛔ `overlapping_holdout_se` not used anywhere.

**LATERAL, read on curvature MAE with the straight-line floor beside it:** `head − off` curvature
**−0.0001 [−0.0002, +0.0000] — NOT separated.** The lever does not move the family the plan named
as its second headline, in either direction.

## 2. ⭐ The deliberate-regression arm is what makes this verdict *mean* something — and it names the mechanism

⛔ **The entire distance-keeping degradation is reproduced by an arm whose agent tokens carry NO
information about the scene.**

| distance-keeping | `head − off` (real join) | `shuf − off` (deranged join) | `head − shuf` |
|---|---|---|---|
| headway (m) | −0.5015 **SEP WORSE** | −0.4576 **SEP WORSE** | −0.0438 [−0.2456, +0.1191] **not sep** |
| time-gap (s) | −0.0702 **SEP WORSE** | −0.0681 **SEP WORSE** | −0.0022 [−0.0299, +0.0202] **not sep** |
| min TTC (s) | −2.1959 **SEP WORSE** | −1.9334 **SEP WORSE** | −0.2625 [−0.5911, +0.0660] **not sep** |

⇒ **The cost is the AUXILIARY DETECTION TASK, not the agent information.** Adding a supervised
detector to a 17 M-param trunk at a 500-step budget takes capacity away from the planner, and the
planner pays for it in longitudinal accuracy (`head − off` speed MAE **+0.2151 [+0.1332, +0.3016]**,
along-track **+1.3688 [+0.8539, +1.9150]**, both separated worse — and `shuf − off` reads **+0.2369**
and **+1.5823**, i.e. the same penalty from a join that knows nothing).

⛔ **Had I run only `head` vs `off`, the honest-looking conclusion would have been "agent
conditioning hurts distance-keeping."** That conclusion would have been WRONG about the mechanism,
and it would have sent the next arm after the wrong thing. The regression arm is the only reason
this report can separate *the seam costs capacity* from *the seam misleads the planner*.

## 3. ⭐ RULE ZERO — the refutation is a waypoint, and here is what it points at

P1 failed its bar, and the SAME panel contains the one place agent information demonstrably paid:

| `head − shuf` (real join vs its own deranged twin — identical params, identical loss, identical box density) | delta | 95 % CI | verdict |
|---|---|---|---|
| **cross-track MAE (m)** | **−0.1362** | [−0.2110, −0.0660] | ⭐ SEPARATED **BETTER** |
| **heading MAE (rad)** | **−0.00060** | [−0.00120, −0.00002] | ⭐ SEPARATED **BETTER** |
| **yaw-rate MAE (rad/s)** | **−0.00062** | [−0.00122, −0.00004] | ⭐ SEPARATED **BETTER** |
| along-track MAE (m) | −0.2136 | [−0.4981, +0.0533] | not separated |
| speed MAE (m/s) | −0.0218 | [−0.0693, +0.0228] | not separated |
| min TTC / headway / time-gap | see §2 | — | not separated |

⇒ ⭐ **Knowing WHERE the agents actually are is a LATERAL signal in this rig, not a longitudinal
one** — the opposite axis from where the plan expected it, and the opposite axis from P14 (whose
gain is 98.9 % along-track). It is enough to cancel the auxiliary-task cost on cross-track
(`head − off` cross-track −0.0709 [−0.1675, +0.0107], **not separated**, while `shuf − off` is
**+0.0653 [+0.0128, +0.1167] separated WORSE**) — the real join pays back exactly what the deranged
one loses — but **not enough to beat the no-seam control**.

⛔⛔ **AND TWO OF THOSE THREE INTERVALS ARE MARGINAL BY EXACTLY THE AMOUNT THE FALSE-POSITIVE FLOOR
WARNS ABOUT.** Heading clears zero by **0.00002** and yaw-rate by **0.00004**, against a MEASURED
one-seed replicate false-positive rate of **6/42 = 14.3 %**. Only **cross-track −0.1362 [−0.2110,
−0.0660]** is robust to that floor. ⇒ **the lateral finding is a DIRECTION to test, not an
established lever**, and `PREREG.md` Amendment 1 makes a second seed mandatory because of it.

**What enters instead of P1**, named in the same turn as the refutation: the composed arm launches
**without** the agent seam, carrying **P14** (already PASSED, +0.2813 [+0.2127, +0.3543]) and **P4
restricted**. P1 re-enters behind the pod-side arm on the B1 TRAIN join, where the capacity
objection this rig just measured does not apply.

---

## 4. ⛔ THE CONTROLS, AND THE ONE THAT CAPS EVERYTHING ABOVE

| control | expected known value | MEASURED | |
|---|---|---|---|
| **constant-only** (plan never moves) | ADE = mean‖gt‖ exactly | **30.706260** vs **30.706261** (tol 1e-4; the dump is float32) | ⭐ PASS |
| **frame chain** (my lead-track transform) | reproduces the trainer's own `waypoint_targets` | max abs err **0.000019 m** | ⭐ PASS |
| **frame chain NEGATIVE control** (heading forced to 0) | must be LARGE | **29.9704 m** | ⭐ PASS |
| **paired-estimator null** (a dump against itself) | exactly zero, not separated | **+0.0000 [+0.0000, +0.0000]** on every metric | ⭐ PASS |
| **arm-independence of the metric mask** | identical windows for every arm | asserted per metric; run refuses otherwise | ⭐ PASS |
| **regression-arm coverage** (a confound check) | `shuf` must match `head` in coverage | **16,874/17,787 (94.9 %)** vs **16,845/17,787 (94.7 %)**; 578,679 vs 563,180 boxes; 104/104 episodes both | ⭐ PASS |
| **derangement** | no row on its own clip | **0 of 26,394** | ⭐ PASS |
| **effective-weight stamp** | `head`/`shuf` TRAIN, `off` builds no graph | `0 → 1 → 1 · op · yes · TRAINS` vs `0 → 0 → 0 · dflt · no · OFF_BY_DEFAULT`, `agent_join_stats: null` | ⭐ PASS |
| **n and d** | printed | n = **1,497 windows / 35 episodes**; d = **16,989,725** ⇒ **n ≪ d** | ⭐ printed |
| **`grep` under-reports on G:** | absence needs a same-breath non-zero control | repo-wide `--agents` → **0** hits; same pattern under `stack/` → **40**. ⛔ The zero was the mount, not the code | ⭐ demonstrated |

### 4.1 ⛔⛔ THE STRAIGHT-LINE FLOOR BEATS EVERY TRAINED ARM ON EVERY FAMILY

| metric | `straight` (never steers) | `off` | `head` | `shuf` |
|---|---|---|---|---|
| ADE (m) | **4.29227** | 4.90824 | 6.12579 | 6.35562 |
| speed MAE (m/s) | **0.49762** | 0.52829 | 0.74337 | 0.76517 |
| along-track MAE (m) | **3.48056** | 4.05527 | 5.42403 | 5.63760 |
| cross-track MAE (m) | **1.45833** | 1.62945 | 1.55854 | 1.69474 |
| curvature MAE (1/m) | **0.00950** | 0.00980 | 0.00973 | 0.00981 |
| mean min headway (m) | **37.25717** | 36.29252 | 36.01010 | 36.06747 |

⛔ **At 500 steps, a plan that never steers is better than all three arms on all six.** This is the
honest power statement and it is stated *first*, before anyone quotes a contrast:

* It does **NOT** invalidate the contrasts. Every one is **paired** — same windows, same GT, same
  lead track, same floor on both sides — so the floor cancels inside each bootstrap draw. The
  question *"does the seam change this model"* is answered; the question *"is this model any good"*
  is answered **no, not yet, at 500 steps**.
* ⛔ It **DOES** cap the transfer claim. A lever measured on arms that all sit below the raw-input
  floor may rank differently once the arms clear it. **This is a GATE number, not a lever
  magnitude**, and the plan already says so (§2.1: *"a tiny-rig PASS is entry to the composed arm,
  not a published result"*) — the same caution applies to a tiny-rig FAIL.

### 4.2 ⛔ A NUMBER THE BRIEF CARRIED DOES NOT REPRODUCE HERE, AND I AM NOT QUOTING IT

The brief and `REFCV5_MISSING_PIECES_PLAN.md` §8.2 state REF-C is **~84× worse on curvature than a
plan that never steers** (0.02737 vs 2.30973). ⛔ **On this rig the ratio is 1.0×** — arms
0.00973–0.00981 against a floor of 0.00950 — and the floor's *absolute value* differs from the
quoted one by three orders of magnitude (0.0095 vs 2.30973 1/m).

⚠️ **This is NOT a refutation of the 84× figure.** Different corpus (35 B1-eval episodes vs the
landing grid), different checkpoint (500 steps vs 40,284), and — the likely dominant difference —
**a different horizon and grid**: curvature here is read on the **uniform 2 s prefix** (slots
[0,1,2,3], horizons [5,10,15,20], dt 0.5 s), because the v3 horizon set
**[5,10,15,20,30,40,50,60] is NOT uniform** and `_seq_geometry` divides by a single `dt`. Scoring
all eight slots at dt = 0.5 s inflates every *rate* by 2× on the tail. ⇒ **I could not reproduce the
84 % / 84× comparison and I am not asserting it either way**; what I can say is that on this rig,
at this budget, **no arm beats the straight-line curvature floor and none is far from it**.

## 5. Power — what variance the interval answers

Seed 0's intervals answer the **EPISODE-DRAW** question only: they resample the 35 eval episodes,
not the training run, not the inference run, not the rig. ⛔ A one-seed separated CI is **NECESSARY,
NOT SUFFICIENT** — the measured replicate false-positive rate is **6/42 = 14.3 %**.

`PREREG.md` Amendment 1 (committed before any outcome number) makes seed 1 **mandatory** because
primary contrasts separated. **Seed 1 HAS RUN — see §10, which CONFIRMS the verdict and RETRACTS §3's lateral reading.** The verdict above does not depend on it: the
distance-keeping FAIL is large relative to that floor (headway upper bound −0.2366 m, TTC upper
bound −0.6110 s) and is **independently reproduced by `shuf − off`**, which is a second arm reaching
the same conclusion by a different route. What seed 1 is for is the **marginal lateral `head − shuf`
cells**, which is exactly where a 14.3 % floor bites.

---

## 6. What was run, and why it is a legitimate gate

**Rig** — `stack/scripts/refc_v3_train.py`, the REAL trainer, at `--size tiny`
(`V3_RIG_SIZES`, **16,989,725** params). ⛔ Never a bespoke script and never a full-scale run.
**Compute** — the dev-box **RTX 4060 (8.0 GiB)**. ⛔ The **A40 was not touched** — it is the
machine this gate releases. ⛔ Thor was not touched.
**Cost** — ~10 min/arm × 3 arms per seed.

**Corpus** — `refav1-eval141` (141 v2ep clips, 256×640 cylindrical, 3-stack, UUID clip ids), split
**episode-disjoint** by `blake2b("split|"+clip_id)` ascending → **104 train / 35 eval, intersection
0**. The split rule is a hash of the clip id, fixed before any metric, so it cannot select on
outcome. The trainer's own guard (*"REFUSING: N episodes appear in BOTH the train cache and the
eval cache"*) passed.

**The seam** — `--agents head --w-agent 1.0 --agent-join <B1 EVAL join>`. MEASURED attachment:
**104/104 train episodes**, **16,845/17,787 train windows labelled (94.7 %)**, 563,180 target
boxes; eval side **35/35 episodes, 5,818/5,985 windows (97.2 %)**. The effective-weight stamp reads
`--w-agent 0 → 1 → 1 · op · builds-a-graph yes · TRAINS`. The control arm's stamp reads
`0 → 0 → 0 · dflt · no · OFF_BY_DEFAULT` with `agent_join_stats: null` — **no graph is built**, which
is the state §8 of the plan describes.

### 6.1 The one blocker that stood between "gate open" and "arm running", and its fix

The join sidecar carried a digest with **no `digest_scope` block**, and the trainer refuses that by
design (*"A hash without its artifact scope is not a verification, it is a number"*). It is not a
data defect: the trainer names its own one-command migration, `python -m tanitad.data.join_meta
<join> --write`, which **MEASURES** which artifact the recorded digest covers. Run, it declared
`md5(compressed) = 3ddb42ecbd3926066795a94587af2aed`, and the arm launched. **The B1 join itself was
never the blocker; a missing declaration was.**

### 6.2 The preflight is mechanical

`launch_arms.py::preflight` diffs the **actual argv** token by token and exits 3 without launching
if any arm differs from the control outside its declared delta. All three arms report
`prefix_identical=True held=60` — a **60-token** held-constant prefix. Banked at
`raw/preflight_s0.json`.

### 6.3 ⭐ The two contrasts answer different questions — and only one isolates the lever

`--agents head` changes two things at once, unavoidably, because they are the same flag:
the **cross-attention conditioning** and an **extra supervised detection head** (parameters +
a loss term through the shared trunk).

| contrast | what it measures |
|---|---|
| `head − off` | the flag's **TOTAL** effect — conditioning *plus* the auxiliary detection task |
| `head − shuf` | ⭐ the **INFORMATION** — identical parameters, identical loss structure, identical box statistics, identical labelled-window fraction; only the clip→agents association differs |

⇒ **`head − shuf` is the contrast that can attribute a gain to agent conditioning.** A `head − off`
movement with no `head − shuf` movement says the arm was helped by *having an auxiliary task*, not
by *knowing where the cars are* — and that is why the PREREG made `head − shuf` a PASS condition
rather than a nice-to-have.

### 6.4 The deliberate-regression arm, and why it had to attack the information

The two regressions the brief named — "agents wired, weight 0.0" and "`--agents off` with a weight
set" — are **already refused by the trainer's own guards** (`refc_v3_train.py:505`, `:524-556`), so
neither can reach a metric. ⇒ the regression arm attacks the **content**: `shuf` permutes `clip_id`
across the join by a **derangement within equal-`n_frames` groups**, so record count, frame indices,
box counts and labelled fraction are preserved exactly. **MEASURED: 0 of 26,394 rows remain on their
own clip.**

---

## 7. The full four-family panel

⛔ **No number in this section is hand-typed** — `code/make_tables.py` renders it straight from `raw/panel_s0.json`.

n_windows **1497** / n_episodes **35** · d (trainable params) **16,989,725** · rate slots [0, 1, 2, 3] at dt 0.5 s · windows with a lead at t0: **880**

Estimator: `paired_episode_cluster_bootstrap (taniteval.ci)`, 10,000 resamples, 95 %. NOT used: `overlapping_holdout_se -- biases the point estimate`. Tier: T1 self-action open loop (planner rolls its own plan from measured state at t0). NOT closed loop.

### Arm point estimates (mean over the scored windows)

| arm | ade_m | speed_mae_mps | along_mae_m | cross_mae_m | heading_mae | curvature_mae | yaw_rate_mae | dk_mean_headway_m | dk_n |
|---|---|---|---|---|---|---|---|---|---|
| `off` | 4.90824 | 0.52829 | 4.05527 | 1.62945 | 0.04799 | 0.00980 | 0.04759 | 36.29252 | 795 |
| `head` | 6.12579 | 0.74337 | 5.42403 | 1.55854 | 0.04743 | 0.00973 | 0.04688 | 36.01010 | 800 |
| `shuf` | 6.35562 | 0.76517 | 5.63760 | 1.69474 | 0.04803 | 0.00981 | 0.04751 | 36.06747 | 797 |
| `const` | 30.70626 | 11.84068 | 30.41345 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 45.17160 | 805 |
| `straight` | 4.29227 | 0.49762 | 3.48056 | 1.45833 | 0.04632 | 0.00950 | 0.04501 | 37.25717 | 803 |

### LONGITUDINAL + LATERAL contrasts (delta = A − B; **lower is better**)

⚠️ LATERAL is read on **curvature MAE with the straight-line floor beside it** — see the control table row `straight`.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:ade_m` | +1.2175 | [+0.7274, +1.7505] | 1497 / 35 | **SEPARATED WORSE** |
| `head-off:speed_mae_mps` | +0.2151 | [+0.1332, +0.3016] | 1412 / 35 | **SEPARATED WORSE** |
| `head-off:along_mae_m` | +1.3688 | [+0.8539, +1.9150] | 1497 / 35 | **SEPARATED WORSE** |
| `head-off:cross_mae_m` | -0.0709 | [-0.1675, +0.0107] | 1497 / 35 | not separated |
| `head-off:heading_mae` | -0.0006 | [-0.0013, +0.0001] | 1412 / 35 | not separated |
| `head-off:curvature_mae` | -0.0001 | [-0.0002, +0.0000] | 1397 / 35 | not separated |
| `head-off:yaw_rate_mae` | -0.0007 | [-0.0016, +0.0001] | 1397 / 35 | not separated |
| `head-shuf:ade_m` | -0.2298 | [-0.4742, +0.0075] | 1497 / 35 | not separated |
| `head-shuf:speed_mae_mps` | -0.0218 | [-0.0693, +0.0228] | 1412 / 35 | not separated |
| `head-shuf:along_mae_m` | -0.2136 | [-0.4981, +0.0533] | 1497 / 35 | not separated |
| `head-shuf:cross_mae_m` | -0.1362 | [-0.2110, -0.0660] | 1497 / 35 | **SEPARATED BETTER** |
| `head-shuf:heading_mae` | -0.00060 | [-0.00120, -0.00002] | 1412 / 35 | **SEPARATED BETTER** |
| `head-shuf:curvature_mae` | -0.0001 | [-0.0002, +0.0000] | 1397 / 35 | not separated |
| `head-shuf:yaw_rate_mae` | -0.00062 | [-0.00122, -0.00004] | 1397 / 35 | **SEPARATED BETTER** |
| `shuf-off:ade_m` | +1.4474 | [+0.9736, +1.9516] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:speed_mae_mps` | +0.2369 | [+0.1465, +0.3330] | 1412 / 35 | **SEPARATED WORSE** |
| `shuf-off:along_mae_m` | +1.5823 | [+1.0681, +2.1285] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:cross_mae_m` | +0.0653 | [+0.0128, +0.1167] | 1497 / 35 | **SEPARATED WORSE** |
| `shuf-off:heading_mae` | +0.0000 | [-0.0006, +0.0006] | 1412 / 35 | not separated |
| `shuf-off:curvature_mae` | +0.0000 | [-0.0001, +0.0001] | 1397 / 35 | not separated |
| `shuf-off:yaw_rate_mae` | -0.0001 | [-0.0009, +0.0008] | 1397 / 35 | not separated |

### DISTANCE-KEEPING contrasts (headway / time-gap / min-TTC; **higher is safer**)

⭐ This is the family the lever is supposed to move.

| contrast | delta | 95 % CI | n win / ep | verdict |
|---|---|---|---|---|
| `head-off:headway_min_m` | -0.5015 | [-0.7712, -0.2366] | 779 / 28 | **SEPARATED WORSE** |
| `head-off:time_gap_min_s` | -0.0702 | [-0.1074, -0.0385] | 719 / 28 | **SEPARATED WORSE** |
| `head-off:min_ttc_s` | -2.1959 | [-4.1408, -0.6110] | 779 / 28 | **SEPARATED WORSE** |
| `head-shuf:headway_min_m` | -0.0438 | [-0.2456, +0.1191] | 779 / 28 | not separated |
| `head-shuf:time_gap_min_s` | -0.0022 | [-0.0299, +0.0202] | 719 / 28 | not separated |
| `head-shuf:min_ttc_s` | -0.2625 | [-0.5911, +0.0660] | 779 / 28 | not separated |
| `shuf-off:headway_min_m` | -0.4576 | [-0.7242, -0.1829] | 779 / 28 | **SEPARATED WORSE** |
| `shuf-off:time_gap_min_s` | -0.0681 | [-0.1112, -0.0297] | 719 / 28 | **SEPARATED WORSE** |
| `shuf-off:min_ttc_s` | -1.9334 | [-3.7665, -0.4244] | 779 / 28 | **SEPARATED WORSE** |

### Controls at their known values

* **constant-only (never moves)** — ADE **30.706260** vs the no-information value mean‖gt‖ **30.706261**, tol 0.0001 ⇒ **PASS**
* **straight-line floor (never steers)** — curvature MAE **0.00950** 1/m. Arms: `off` **0.00980** · `head` **0.00973** · `shuf` **0.00981**. ⛔ Ratio arm/floor: `off` **1.0×** · `head` **1.0×** · `shuf` **1.0×**
* **n vs d** — n = windows scored, d = trainable params. n << d is UNDERPOWERED BY CONSTRUCTION, not a negative result. Here n = 1497 windows / 35 episodes against d = 16,989,725: **n ≪ d**.
* **windows entering each metric** — {'ade_m': 1497, 'cross_mae_m': 1497, 'along_mae_m': 1497, 'speed_mae_mps': 1412, 'heading_mae': 1412, 'curvature_mae': 1397, 'yaw_rate_mae': 1397}

---

## 8. ⛔ What this result is NOT — stated before anyone quotes it

1. **It is not a held-out driving result.** The corpus is the B1 **EVAL** set split in two. The
   split is episode-disjoint, so no eval episode was trained on — but these same 35 episodes are the
   ones the programme's landing reads use. ⇒ **a GATE number, inadmissible as a published result.**
   The B1 **TRAIN** join (md5 `1c985e6d6ad34e605c4ebd30cb353558`, verified present, 4,427 clips)
   covers **0** of the 141 local clips and **no local episode cache exists for that corpus**, so the
   held-out arm is a POD arm and is named as the follow-up.
2. **It is not closed loop.** T1 = self-action open loop. Per the binding ruling, a planner feeding
   its own predictor is still open loop; closed loop needs AlpaSim or a vehicle.
3. **It is a 500-step rig**, ≈0.22 epochs of the 17,787 training windows. It answers *"does this
   seam do anything detectable at rig scale"*, not *"how much is it worth at 40k"*.
4. **`n ≪ d`** — this is **underpowered BY CONSTRUCTION, not a negative result.** The count is
   printed with every interval so a null cannot be mistaken for a refutation.
5. ⛔ **The tiny rig has a measured false-positive floor.** A pure zero-lever replicate produced
   "separated" differences on **6 of 42** family cells = **14.3 %**. Every separation below is read
   against that floor.

## 9. Follow-ups, in priority order

1. ⭐ **The pod-side arm on the B1 TRAIN join** — 4,427 clips, 42× this corpus, genuinely held out.
   That is the arm that can turn a rig signal into a lever claim. Everything it needs exists: the
   join is built and md5-verified, `--agents head` needs no `agent_gt`, and the digest-scope
   migration is a one-liner that must be run on the TRAIN sidecar as it was on the EVAL one.
2. **The monocular geometry terms** (`--agent-w-project`, `--agent-w-ground`) were deliberately left
   at 0.0 so this arm tested DD's cross-attention and not a projection prior. They are a separate
   pre-registered lever with their own rig camera requirement.
3. **A second seed of whichever arms this gate leaves live**, because one seed answers only the
   episode-draw variance.

---

## 10. ⭐⭐ SEED 1 — THE VERDICT IS CONFIRMED, AND ONE OF MY OWN §3 FINDINGS IS REFUTED BY ITS OWN REPLICATE

**Appended 2026-09-07 after `runs/panel_s1.json`. `PREREG.md` Amendment 1 made this seed mandatory,
and it earned its cost: it confirmed the gate and killed a finding that would otherwise have
directed the next arm.**

### 10.1 What REPLICATES — the gate, and the mechanism

| contrast | seed 0 | seed 1 | replicates? |
|---|---|---|---|
| `head − off` **min headway (m)** | **−0.5015** [−0.7712, −0.2366] SEP WORSE | **−0.4535** [−0.7915, −0.1639] SEP WORSE | ⭐ **YES** |
| `head − off` **min time-gap (s)** | **−0.0702** [−0.1074, −0.0385] SEP WORSE | **−0.0351** [−0.0694, −0.0037] SEP WORSE | ⭐ **YES** |
| `head − off` **min TTC (s)** | **−2.1959** [−4.1408, −0.6110] SEP WORSE | **−1.2529** [−2.2522, −0.3694] SEP WORSE | ⭐ **YES** |
| `head − shuf` distance-keeping, all three | not separated | not separated | ⭐ **YES** |
| `head − off` **curvature MAE** | −0.0001, not sep | −0.0001, not sep | ⭐ **YES** |

⇒ ⛔ **P1 IS OUT stands on TWO SEEDS.** All three distance-keeping metrics separate in the
unfavourable direction at both seeds, same sign, overlapping intervals. This is no longer a
one-seed separation sitting above a 14.3 % floor.
⇒ ⭐ **And the MECHANISM replicates too:** `head − shuf` is not separated on any distance-keeping
metric at either seed, while `shuf − off` separates worse at both (headway −0.4576 / −0.3049; TTC
−1.9334 / −0.9668). **The cost is the auxiliary detection task, not the agent information** — twice.

### 10.2 ⛔⛔ RETRACTION — §3's LATERAL FINDING DOES NOT SURVIVE ITS SECOND SEED

§3 reported, from seed 0 alone, that `head − shuf` separated BETTER on cross-track, heading and
yaw-rate, and read that as *"agent identity is a LATERAL signal in this rig."* I hedged it as a
direction rather than a lever. **That hedge was not enough. The finding is REFUTED:**

| `head − shuf` LATERAL | seed 0 | seed 1 | |
|---|---|---|---|
| cross-track MAE (m) | **−0.1362** [−0.2110, −0.0660] SEP BETTER | −0.0585 [−0.4066, +0.2460] **not sep** | ⛔ gone |
| heading MAE (rad) | **−0.00060** [−0.00120, −0.00002] SEP BETTER | **+0.0002** [−0.0016, +0.0018] **not sep** | ⛔ **SIGN FLIP** |
| yaw-rate MAE (rad/s) | **−0.00062** [−0.00122, −0.00004] SEP BETTER | **+0.0006** [−0.0014, +0.0025] **not sep** | ⛔ **SIGN FLIP** |
| curvature MAE (1/m) | −0.0001, not sep | **+0.000118** [+0.000002, +0.000246] **SEP WORSE** | ⛔ **separates the OTHER WAY** |

⛔⛔ **AND THE ARM THAT SETTLES IT: at seed 1 the DERANGED join `shuf − off` separates BETTER on
FOUR lateral metrics** — cross-track −0.5461 [−1.0457, −0.1397], heading −0.0039 [−0.0071, −0.0012],
curvature −0.0002 [−0.0004, −0.0001], yaw-rate −0.0046 [−0.0082, −0.0015]. **A join that knows
nothing about the scene beat the no-seam control on the entire lateral family.** No lever story
survives that. ⇒ **the lateral cells at this rig and budget are SEED NOISE, and §3's reading is
withdrawn.** The three cells that carried it were exactly the marginal ones (clearing zero by 2e-5
and 4e-5) that the MEASURED 6/42 = 14.3 % replicate false-positive rate exists to catch. **It caught
them.**

⚠️ **ADE FLIPS SIGN TOO**, and it is worth stating because it is the metric everyone reaches for
first: `head − off` ADE is **+1.2175 [+0.7274, +1.7505] SEPARATED WORSE at seed 0** and
**−0.4858 [−0.9161, −0.1088] SEPARATED BETTER at seed 1**. ⇒ **a confidently separated ADE verdict,
in both directions, from the same two arms.** This is the "never ADE alone" rule paying for itself
inside a single package — the four-family read is stable on the family that matters while ADE is
not.

### 10.3 What this leaves standing

* ⭐ **P1 IS OUT — two seeds, gate family, same direction.** Nothing below changes that.
* ⭐ **The auxiliary-task attribution — two seeds.** That is the finding worth carrying forward, and
  it is a claim about *capacity at 17 M params / 500 steps*, not about agent conditioning as a design.
* ⛔ **No admissible claim that agent information helps ANY family on this rig.** The one candidate
  did not replicate. ⇒ the pod arm on the B1 TRAIN join is not chasing a measured direction; it is
  testing the piece at a scale where the capacity objection does not apply. That is still worth
  running — it is just not "confirming a lateral signal", and must not be briefed as such.
* ⛔ **The straight-line floor still beats every arm at seed 1** (ADE 4.29 vs 5.29 / 5.78 / 5.91;
  headway 37.26 vs 36.51 / 36.41 / 36.32). The power cap of §4.1 is unchanged.
