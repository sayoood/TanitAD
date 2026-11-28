# PAIRED OPEN-LOOP COMPARISON — refav1 `cl` vs refcv3 `os`

**Package:** `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-paired-openloop-refav1-vs-refcv3/`
· **owner:** Benchmarks & Evals FlyWheel · **date:** 2026-09-03 · **tier T1 (both arms, matched)**
· **compute:** dev-box RTX 4060 for the refcv3 roll (17 min); the comparison itself is **0 GPU**.
⛔ Thor and `tanitad-refcv3` were never contacted — both are training.

**Evidence class:** every number below is **MEASURED** (ours), with its artifact named. The two
wall-clock estimates for the closed-loop work in §7 are **INHERITED** and stamped as such.

---

## 0. ⛔⛔ LINE ONE — THE CROSS-MODEL READ IS **VOID, NOT NEGATIVE**

**`refav1:cl` is bit-identical to the trivial floor `ha0` on 120 of 120 shared windows.** Its plan is
a straight, constant-speed line on 100 % of them, so its margin over the floor is **exactly 0.0000 on
every metric in every family**. This is not "refav1 loses". The model's emitted trajectory *and the
no-information baseline* are **the same array**, so the instrument saw the baseline, not the model
(`D-REFAV1-PAIRED-READ-VOID`, condition 4 of `D-HF-COMPARABILITY`).

⇒ **Every cross-model row in this package is banked for audit and none of it may be quoted as
"refcv3 beats refav1".** What can be quoted is each arm's own margin over the shared floor.

**And the second line matters as much as the first:**

> **refcv3 beats the constant-velocity floor on ADE only WITH its oracle nav token.**
> `os − ha0` = **−0.2045 m [−0.3997, −0.0178] separated**; with the nav withheld,
> `os_navzero − ha0` = **−0.1349 m [−0.3276, +0.0541] NOT separated.**
> The v7.2 `nav_command` is an **ORACLE** (provenance `ego-future`) that will not exist at
> deployment. ⇒ **on the deployment-relevant arm, neither model is shown to beat a straight line at
> the measured speed on ADE.**

**Third: this is the OPEN-LOOP half of the PI's question only.** See §7.

---

## 1. What was compared, and what the PI's ruling settles

**PI ruling, 2026-09-02** (`Project Steering/VOCABULARY.md`; landed in `t1_eval.py`, commit
`32e319149`): a model that is not controlling the vehicle is **OPEN LOOP**, *including* a planner
whose own predictor consumes its actions — the recorded eval ego data keeps arriving regardless.

⇒ refav1's `cl` and refcv3's `os` are **both open loop**, and that is precisely why they are
comparable at all. ⛔ The phrase "closed loop" describes no arm in this package.

| | A | B |
|---|---|---|
| model | **refav1** — latent world model + iCEM planner, rolls out under a 3-channel action incl. speed | **refcv3** — supervised **ONE-SHOT** 128-anchor × 8-slot trajectory model: **no action input, no rollout, no per-step decode** |
| arm | `cl` (T1) | `os` (T1, ⚠️ `status: UNRULED`) |
| consumes at inference | vision, nav token, measured `v0` | vision, v7.2 nav token, measured `v0` |
| checkpoint | `refav1_eval_slice/ckpt_ep2/ckpt.pt` — clean-epoch, stores `step = 1000` (the counter restarts at the epoch boundary; the file differs from the incumbent's by 29 MB) | `ckpt_30000.pt`, **step 30000**, 107,082,365 params |
| dump grid | dt 0.2 s, K 10 | dt 0.5 s, K 4 (`--grid 2s`), rolled at **`--window-stride 1`** for this comparison |

**Selection:** refcv3's arm is the model's **own** `sel_score_v3`-ranked choice via `out["traj"]`.
⛔ `a_star` / `oracle_sel` — the anchor nearest the ground truth — is refused as a model arm by gate
G2 and appears nowhere in this table.

---

## 2. The intersection, honestly

* **n = 120 shared windows over 20 episodes.** Key: `(clip_id, RAW 10 Hz frame of the window origin)`.
* refav1's dump carried 140 windows / 20 clips; refcv3's 3,419 / 20. **20 refav1 windows and 3,299
  refcv3 windows were dropped** by the intersection.
* **Common instants `{1.0, 2.0} s`** (raw-frame offsets `{10, 20}`). refav1 dropped slots
  `2,4,6,8,12,14,16,18`; refcv3 dropped `5, 15`. Both sides are **index-selected**; there is no
  resampling and no interpolation, and the grids were matched in **integer raw frames**, never in
  seconds.
* ⚠️ **Consequence, stated rather than buried:** the common grid is **coarser than either dump's
  own**. Every LON/LAT rate is computed on it for both arms *and for GT*, so the comparison is fair —
  but ⛔ these levels must not be quoted against a single-arm read on a finer grid.
* 20 episodes clears the power floor (10); a `not separated` row here is a real null, not an
  artefact of sample size.

### 2a. ⚠️ The stride trap that had to be worked around, and it will bite the re-fire

refav1's window origins are RAW `2·ws` (≡ 6 mod 20 here); a refcv3 dump at `--window-stride 5`
reaches only RAW ≡ 2 (mod 5). **Those sets are disjoint — the intersection would be EMPTY.** The
141-clip refcv3 dump produced by the single-arm stream is at stride 5 and is therefore **unusable for
this comparison**; a dedicated **stride-1** roll over the 20 refav1 clips was made for it
(17 min, RTX 4060). The tool refuses an empty intersection with the modulus arithmetic printed.

---

## 3. ⭐ The controls that had to read a known value — all pass

| control | expected | **MEASURED** | |
|---|---|---|---|
| **the window key**: GT identity across two independent pipelines | `0.0` exactly | **`0.000e+00 m`** | ✅ |
| **the window key**: `v0` identity | `0.0` exactly | **`0.000e+00 m/s`** | ✅ |
| **C1** the floor reads the no-information value `x = v0·t, y = 0` (side A) | `0.0` exactly | `dx 7.629e-06 m`, **`|y| 0.000e+00 m`** | ✅ |
| **C1** (side B) | `0.0` exactly | `dx 1.526e-05 m`, **`|y| 0.000e+00 m`** | ✅ |
| **C2** the floor is bit-comparable across the two architectures | `0.0` exactly | `1.526e-05 m` | ✅ |
| **C3** the DERIVED floor vs the DUMPED one, **same side** | `0.0` | `1.526e-05 m` | ✅ |
| declared `lat_label` / `lon_label` agree where both sides label | identical on all 40 | **identical** | ✅ |
| declared `route_label` agree where both sides label | identical on all 35 | ⛔ **5 disagree** | ⛔ |

⭐ **The GT identity result is the load-bearing one.** The two dumps are built by different tools,
from different caches (**DINOv3 fp8 tokens** vs **v2ep pixels**), through different GT functions
(`metric_dynamics.gt_ego_waypoints` vs `refb_labels.waypoint_targets`), on hosts that never met. They
agree **bit for bit** on all 120 shared windows. A frame-offset argument cannot be settled by reading
code; this settles it.

**The floor.** The banked refav1 dumps predate the `ha0` arm, so it was **DERIVED** from the dumped
`v0` through the programme's own integrator (`refav1_arm.hold_v0_controls` →
`paths_from_controls` → `refa_v1_plan.unicycle_paths`) — not re-implemented. C3 checks that derivation
against refcv3's independently *dumped* `ha0` on refcv3's own side. The residual `1.5e-05 m` is
float32 round-tripping.

⚠️ **The algebraic identity, stated rather than hidden.** With the floor verified identical, one array
is used for both margins, and therefore `(os − ha0) − (cl − ha0)` **equals** `os − cl` exactly. The
margin framing is still what is reported — it is what `D-HF-COMPARABILITY` binds and what stays
interpretable when the floor is *not* identical — but a reader is owed the identity.

### 3a. ⭐ Why `ha` is NOT the shared floor — now a number, not an argument

The two dumps' `ha` arms carry the **same name** and are **not the same arm**. Measured over the same
floor and the same 120 windows:

| metric | `refcv3:ha − ha0` minus `refav1:ha − ha0` | 95 % CI | separated |
|---|---|---|---|
| `ade_m` | **−0.6404 m** | [−0.9856, −0.3355] | **YES** |
| `fde_m` | **−1.0630 m** | [−1.6341, −0.5557] | **YES** |
| `LAT_cross_mae_m` | −0.5898 m | [−0.8954, −0.3305] | **YES** |
| `LAT_heading_mae_deg` | −3.3219° | [−5.0718, −1.8999] | **YES** |

They differ in **two** ways at once — the action-unit convention (refcv3's manifest declares
`steer`; refav1's dump carries **no `action_units` field at all**, i.e. the legacy reading that
integrates a road-wheel angle as a curvature and over-rotates by ~L = 2.9×) **and** the hold rule and
the tick it is differenced on. ⛔ The 0.64 m is therefore **not attributable to either cause alone** —
and that is exactly the point: **a same-named control is not a shared floor.** `ha0` is, because it is
*exactly zero in either unit* and depends on nothing but `v0`.

---

## 4. Shape before metrics — the two profiles, and what they show

| arm | straight | const speed | **CONSTANT-VELOCITY** | identical to |
|---|---|---|---|---|
| `refav1:cl` | 1.0000 | 1.0000 | **1.0000** | **`shared:ha0` 120/120**, `refav1:cl_navshuf` 105/120 |
| `refav1:cl_navshuf` | 1.0000 | 0.8750 | **0.8750** | `shared:ha0` 105/120 |
| `refav1:ha` | 0.0250 | 0.0083 | 0.0083 | — |
| `refcv3:os` | 0.0000 | 0.0000 | **0.0000** | `refcv3:os_navshuf` 69/120 |
| `refcv3:os_navzero` | 0.0000 | 0.0000 | **0.0000** | — |
| `shared:ha0` | 1.0000 | 1.0000 | 1.0000 | — |

**Selection profile (refcv3):** 24 distinct anchors over 120 windows, modal `#44` at **0.2083**,
entropy **2.667 nats**, agreeing with the GT-nearest oracle on **0.6167**. ⇒ **not degenerate** — the
gate the trivial profile is blind to is passed.

⭐ **A detail worth the Master Mind's eye.** `refav1:cl_navshuf` is constant-velocity on 87.5 % and
identical to `ha0` on 105/120 — so on **15 windows the nav-shuffled plan is NOT the floor while the
true-nav plan always is**. The planner is not *entirely* insensitive; it is that the shipped
conditioning lands on the floor every time.

---

## 5. The four families, separately — never pooled

`A−floor` / `B−floor` are each arm's margin over `ha0`; `(B−f)−(A−f)` is the cross-model statistic.
Error metrics: **negative is better**. Accuracy rows (`*_correct`): **positive is better**.
⛔ Every cross row carries `VOID: true` — see §0. Full table with all intervals:
`raw/paired_table.md`; machine record: `taniteval/results/paired-openloop-refav1-vs-refcv3-20260903T204638Z.json`.

| family | metric | `refav1:cl` | `refcv3:os` | `ha0` | **A−floor** | **B−floor** | (B−f)−(A−f) | 95 % CI | sep |
|---|---|---|---|---|---|---|---|---|---|
| **ADE** | `ade_m` | 0.8635 | 0.6590 | 0.8635 | **0.0000** | **−0.2045** | −0.2045 | [−0.3997, −0.0178] | **YES** |
| | `fde_m` | 1.3633 | 1.0163 | 1.3633 | **0.0000** | **−0.3470** | −0.3470 | [−0.6606, −0.0447] | **YES** |
| **LONGITUDINAL** | `LON_speed_mae_mps` | 0.4688 | 0.4733 | 0.4688 | 0.0000 | +0.0044 | +0.0044 | [−0.1281, +0.1265] | no |
| | `LON_along_mae_m` | 0.5741 | 0.5903 | 0.5741 | 0.0000 | +0.0162 | +0.0162 | [−0.1342, +0.1651] | no |
| | `LON_accel_mae_mps2` | 0.4350 | 0.4842 | 0.4350 | 0.0000 | +0.0492 | +0.0492 | [−0.0928, +0.1718] | no |
| **LATERAL** | `LAT_cross_mae_m` | 0.4459 | 0.1650 | 0.4459 | 0.0000 | **−0.2808** | −0.2808 | [−0.4276, −0.1450] | **YES** |
| | `LAT_heading_mae_deg` | 2.1379 | 0.7442 | 2.1379 | 0.0000 | **−1.3937** | −1.3937 | [−2.2889, −0.6004] | **YES** |
| | `LAT_yaw_rate_mae_radps` | 0.0381 | 0.0727 | 0.0381 | 0.0000 | +0.0346 | +0.0346 | [−0.0243, +0.1195] | no |
| **TACTICAL** | `TAC_traj_lat_correct` | 0.9417 | 0.9917 | 0.9417 | 0.0000 | +0.0500 | +0.0500 | [0.0000, +0.1083] | no |
| | `TAC_traj_lon_correct` | 0.8417 | 0.7500 | 0.8417 | 0.0000 | −0.0917 | −0.0917 | [−0.2000, +0.0083] | no |
| | `TAC_declared_lat_correct` | 0.6750 | 0.7750 | **0.6500** | +0.0250 | +0.1250 | +0.1000 | [−0.1500, +0.3500] | no |
| | `TAC_declared_lon_correct` | 0.5500 | 0.5250 | **0.4500** | +0.1000 | +0.0750 | −0.0250 | [−0.3000, +0.2500] | no |
| **STRATEGIC** | — | — | — | — | — | — | ⛔ **REFUSED** | — | — |

⭐ **The floor for the two decision families is not `ha0`** — `ha0` is a trajectory and has no head.
It is the **MAJORITY-CLASS rate on exactly those windows** (`lat` class 0 at **0.6500**, n = 40;
`lon` class 1 at **0.4500**, n = 40), which is the no-information value for a classifier and is
shared across both models because it is a property of the **labels** alone.

### 5a. Family verdicts

| family | verdict |
|---|---|
| ADE | **VOID** (a degenerate arm entered it) |
| LONGITUDINAL | **VOID** — and note that **nothing separates** here at all, in either direction |
| LATERAL | **VOID** — refcv3 separates on cross-track and heading, but **not** on yaw-rate |
| TACTICAL | **VOID** — nothing separates, declared or trajectory-derived |
| STRATEGIC | ⛔ **REFUSED** — see §6 |

⇒ **They split.** Even setting the VOID aside, refcv3's own margin over the floor is separated on
LATERAL (2 of 3 metrics) and on ADE/FDE, and **not separated anywhere in LONGITUDINAL or TACTICAL**.
⛔ There is no overall winner to report and none is manufactured.

### 5b. The nav-echo controls (without which the nav-conditioned rows are inadmissible)

Neither model's declared heads separate from their own controls:
`TAC_declared_lat true − navshuffled` = refav1 **+0.1000** [−0.0500, +0.2750] ns / refcv3 **+0.0500**
[0.0000, +0.1500] ns; `true − navzero` = refav1 **+0.0250** ns / refcv3 **+0.0750** ns. Every cross
row is `not separated`. ⇒ **no evidence here that either tactical head is reading *this* window's
nav rather than the marginal.**

---

## 6. ⛔ ESCALATION — the STRATEGIC family is REFUSED, and it is a REPO defect, not a model result

`refav1_arm.py` and `refcv3_arm.py` **assign different `route_label`s to the same
(clip_id, RAW frame)**:

* refav1 labels **40** of the 120 shared windows; refcv3 labels **105**.
* On the **35** both label, **5 disagree**: `1→2` ×2 and `2→1` ×3.

This is **not** a coverage difference. Scoring two route heads against "the label" would score two
different questions, so the tool refuses the family with its reason and its `n` rather than printing a
number. ⇒ **WORK ITEM: reconcile `route_label` between the two adapters.** Until then the programme
has **no cross-model strategic row**, and — since the hierarchy is the programme's thesis — that is
worth fixing before the final re-fire. *(The declared tactical labels `lat_label`/`lon_label` are
identical on all 40, so only the route derivation is implicated.)*

### 6a. ⚠️ Corpus identity: **UNVERIFIED**, and it must be said

Two arms both named "b1-v72" does **not** establish they saw the same episodes.

* **refcv3's own `config.json`** records `v2_parity: {parity: false, checked: false,
  corpus_key: null, clips_present: 4572, cache_dirs: ["/root/data/train"]}` — **the run is
  NON-PARITY by its own record.**
* **refav1's trainer publishes no parity block at all**; its config names
  `cache=/home/nvidia/data/refav1-fp8-train`, `episodes=/home/nvidia/data/physicalai-b1-w120-256x640cyl`.

⇒ **the TRAIN corpora cannot be shown identical from the configs.** ⭐ What *is* established, far more
strongly than any config field, is the **EVAL** side: the window-key proof (§3) shows both arms are
scored on the **same moments of the same clips**. ⛔ What is **not** established is (a) that the two
runs trained on the same episodes and (b) that either run's train split is disjoint from these 20 eval
clips. Both are properties of the **runs**, not of this adapter; the probe that settles them is an
episode-id set intersection between each run's train cache and this eval split
(`REFCV3_ARM.md` GATE 3). **Neither could be run from this box** — both hosts are training.

---

## 7. ⛔ CLOSED LOOP — the other half of the PI's question

The PI asked for **open AND closed** loop. **Everything above is the open-loop half**, and it must not
be presented as the whole answer.

* **Not measured here, and this tool cannot measure it**: it reads banked `[N, K, 2]` open-loop
  trajectories scored against recorded GT. Nothing in a dump can become a closed-loop number — the
  world never responded.
* ⭐ **But the harness EXISTS**, so this is a NOT-YET-RUN, not a NOT-POSSIBLE. **VERIFIED at source**
  (`stack/experiments/alpasim-gsplat/closedloop_drive.py:518-533`): it steps a kinematic bicycle from
  the model's own `(steer, accel)` — `dyaw = v/WHEELBASE·tan(steer)·DT`, `T_new = T_ego @ D`,
  `v += accel·DT` — and then **re-renders** the next observation from the resulting ego pose
  (`transport.render(T_ego @ Ts_cam, …)`). The next observation is a consequence of the model's own
  output, which is exactly the PI's definition. It has already produced a published panel
  (flagship v1 vs REF-C base, 9 starts × 50 ticks on Thor, 437 paired windows, paired episode-cluster
  bootstrap).
* **refcv3:** NOT YET RUN — **ESTIMATED (INHERITED, not measured here)** ~1.5 engineer-days +
  ~1 GPU-hour, gated on Thor freeing.
* **refav1:** NOT YET RUN — **ESTIMATED (INHERITED)** 1–2 weeks, blocked on the action-unit contract
  decision.

---

## 8. ⛔ What this comparison does NOT establish

1. **Nothing about closed-loop driving.** §7.
2. **Not that one architecture is better than the other.** The two are different *kinds* of system;
   what is measured is each one's margin over the same trivial floor on the same windows.
3. **Not a statement about either model's own grid** — the table is on the common instants only.
4. **Not a deployment number for refcv3 with its nav fed** — the v7.2 nav token is an oracle; the
   nav-withheld margin is the deployment-relevant one, and on ADE it does **not** separate.
5. **Not a route-head capability claim** — nav is an INPUT; flagship v1's route head scored 1.0000 by
   echoing it. And here the strategic family is refused outright (§6).
6. **Not distance-keeping / headway / time-gap / TTC** — the second half of LONGITUDINAL. Those need
   the lead-block join, which is per-dump and is **not** re-derived here. ⛔ **A WORK ITEM, not a pass**;
   read them from each side's own single-arm record.
7. **Not a claim that either model would drive** — an open-loop trajectory error on recorded data does
   not bound compounding error under the model's own control.
8. **Not evidence that the two runs trained on the same corpus** (§6a).

---

## 9. ⭐ A corroborating measurement, banked in passing (0 GPU)

`raw/refav1_two_checkpoints_identity.json`: over the **140** windows of both banked refav1 dumps, the
**incumbent** and **clean-epoch** checkpoints — different files (2,093,802,833 B vs 2,122,973,057 B) —
emit **byte-identical** `cl`, `cl_navshuf`, `ha` and `ol` (max |Δ| = **0.0**), while `cl_oraclegoal` —
the only arm whose goal is not the degenerate imagined one — moves by up to **2.70 m**.

⇒ **the flat plan is not a property of these particular weights.** Independent corroboration of
`D-REFAV1-COST-SURFACE` / `D-REFAV1-GOAL-DEGENERATE`, measured off the banked dumps with no model
loaded. ⚠️ It does **not** predict the final step-21,109 checkpoint — but it does mean a re-fire that
comes back VOID again should be read as a training/planner lever, not as a longer-run problem.

---

## 10. The one-command re-fire against the FINAL checkpoints

```bash
# 1. roll refcv3 on the FINAL checkpoint, at STRIDE 1, over the refav1 eval clips
#    ⛔ `ckpt_40284_FINAL.pt` IS NEVER WRITTEN — refc_v3_train.py:103 MILESTONES =
#       (5000, 15000, 20000, 30000); the final save is the `or step == args.steps`
#       branch at refc_v3_train.py:1191, which writes plain `ckpt.pt`. That file is
#       ALSO the rolling checkpoint, so ASSERT the step after loading:
python -c "import torch,sys; d=torch.load(sys.argv[1],map_location='meta',mmap=True); \
           assert d['step']==40284, d['step']; print('step OK', d['step'])" \
        <refcv3 run>/ckpt.pt

OMP_NUM_THREADS=6 python taniteval/tools/refcv3_arm.py \
  --ckpt <refcv3 run>/ckpt.pt --episodes <the 20-clip eval dir> \
  --labels <v72>/s2_labels_v7.2_eval.jsonl.gz --nav-source v72 --grid 2s \
  --action-units steer --window-stride 1 --with-oracle-sel --no-lead-block \
  --dump-only --dump-dir <D_B> --out <D_B>.json

# 2. roll refav1 on the FINAL checkpoint (NOT on Thor while it trains)
#    — refav1_arm.py, same 20 clips; ~7.3 min/episode at the default search.

# 3. the paired comparison — 0 GPU
python taniteval/tools/paired_openloop.py \
  --a-dump <D_A> --a-name refav1 --a-arm cl --a-extra cl_navshuf --a-extra ha \
  --b-dump <D_B> --b-name refcv3 --b-arm os --b-extra os_navzero --b-extra os_navshuf --b-extra ha \
  --a-run-config <refav1 run>/config.json --b-run-config <refcv3 run>/config.json \
  --floor ha0 --n-boot 2000 --seed 0 \
  --out "taniteval/results/paired-openloop-refav1-vs-refcv3-<UTC>.json" \
  --md  "TanitAD Research Lab/Benchmarks & Evals/Research/<pkg>/raw/paired_table.md"
```

**Preconditions the tool itself enforces or prints:** the step assertion above; `--window-stride 1`
on the refcv3 side (a stride-5 dump gives an **empty** intersection); never roll on a training host;
and `--b-extra os_navzero`, because with an oracle nav `os − ha0` is not the deployment margin and
must not stand alone as the headline.

---

## 11. Deliverable manifest

| artifact | where it lives | bytes |
|---|---|---|
| the harness | `repo:taniteval/tools/paired_openloop.py` | 63,846 |
| its spec | `repo:taniteval/tools/PAIRED_OPENLOOP.md` | 21,685 |
| the pin (23 tests) | `repo:stack/tests/test_paired_openloop.py` | 31,310 |
| the record | `repo:taniteval/results/paired-openloop-refav1-vs-refcv3-20260903T204638Z.json` | 164,461 |
| this report | `repo:TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-paired-openloop-refav1-vs-refcv3/RESULT.md` | 22,522 |
| the generated table | `…/raw/paired_table.md` | 20,758 |
| the two-checkpoint control | `…/raw/refav1_two_checkpoints_identity.json` | 1,095 |
| the run log | `…/raw/paired_stdout.log` | 6,242 |
| ⚠️ **the refcv3 stride-1 dump** | **`devbox:C:\Users\Admin\run_paired\out\refcv3_s1_dump` — ONE PLACE ONLY** | ~1.4 MB (21 files) |
| ⚠️ **the refav1 banked dumps** | **`devbox:C:\Users\Admin\refav1_eval_slice\t1_dump{,_ep2}` — ONE PLACE ONLY** | ~184 KB each |

⚠️ **The two dump directories live on one disk each.** They are small (< 2 MB together) and they are
the ONLY inputs from which this comparison can be reproduced without re-rolling both models
(~2.7 GPU-hours). **Recommend banking them into the repo** — that is an integration decision for the
Master Mind, not a FlyWheel's, because it puts binary artefacts under `taniteval/`.
