# TanitAD — Program Report · 2026-07-26, D-025 slot 12:57 Berlin

**Fleet probed 12:30 UTC · report written 12:40 UTC.** All pod/log timestamps below are **UTC**;
the D-025 slot label is Berlin. *(The dev box's `TZ=Europe/Berlin` did not resolve during this
run and returned UTC — the slot label is therefore the schedule's name, not a measured local
clock. Flagged rather than silently reconciled.)*

**Mode:** autonomous, Sayed away. **Every number below carries its evidence class.**

---

## 1. Headline

**Four pods, four agents, zero idle GPUs.** Two A40s were sitting at 0 MiB at 12:30 UTC and are
now loaded.

**The single most consequential finding this slot is an instrument finding, not a model finding:**
the v4 30k gate's machine verdict was **`NOT_YET`**, not the RESTART that has been quoted in
chat — `run_gate.py` compared a 0-indexed step counter against a 1-indexed count and **refused a
complete 59-hour run**. The gate agent caught this itself (§D3 of its report) and adjudicated by
hand against the card, printing every criterion. The repair has since landed and I have now
**verified it MEASURED** (§6).

---

## 2. Fleet — MEASURED, probed 12:30 UTC, `scratchpad/fleet_probe.py`

| Pod | GPU | State | Evidence |
|---|---|---|---|
| **pod1** (`tanitad-pod`, A6000) | **0/59/100/100/67/59 %** over 6×1 s | **v2corpus training, HEALTHY** | 8 procs, main at 99.8 % CPU, 1 d 09:48 elapsed |
| **pod2** (A40) | 0 MiB, no job | **was idle → now K-sweep agent** | 30k flagship finished |
| **pod3** (A40) | 0 MiB, no job | **was idle → now VectorMap agent** | E1c complete |
| **eval** (A40) | 0 MiB | **was idle → now v4-lever agent** | 4 stale AlpaSim `multiprocessing-fork` workers, 3 d old, 0 % CPU |

⚠️ **pod1's first GPU sample read 0 %.** Per the standing trap-list I re-sampled six times before
writing anything: **0/59/100/100/67/59 %**. The 0 % was a sampling artifact. *This is the third
time a single-sample GPU read has looked like an outage.* No alarm raised.

**pod1 v2corpus — MEASURED** (`/tmp/flagship-v2corpus-30k.log`, mtime 12:26 UTC):
- **step 11,150 / 30,000**
- `step_s: 544.0`, `data_s: 67.1` — these are **ACCUMULATED over `--log-every=50`** → **10.88 s/step**,
  **1.34 s/step in data** (12.3 %). Quoting 544 s/step would be the documented false-alarm.
- **ETA ≈ 57 h** (ESTIMATED: 18,850 remaining × 10.88 s).
- Memory guard active and working: `[guard] 48.6 → 32.4 GB (9000 files)`.
- ⛔ **Not restarted, not modified** — its micro-batch setting is the arm's own instrument.

**Disk:** all four pods pass a real 200 MB `dd` write (1.6 GB/s pod1, 341–422 MB/s others). `df`
was not used — it reports the cluster, not the per-pod MooseFS quota.

---

## 3. Results landed since the 07:57 report

### 3.1 pod2 stood up as the n ≥ 200 eval host — **the ladder's power ceiling is gone**

Committed `fea9070`. All MEASURED, artifacts under
`Benchmarks & Eval/…/2026-07-26-pod2-eval-host/`.

- **Disjointness 0/600 byte overlap** with the parity train. The `episode_id` key appears to
  overlap 20/600 — those are **collisions**, reproduced exactly (2,376 episodes, **2,342 unique
  ids**).
- **Prefix property: 40/40 positions match element-for-element** at [0…39] → scaling 40 → 600
  **adds episodes and re-selects none. Parity holds.**
- Corroborated four further ways, incl. the harness at `--episodes 40` reproducing **881 windows**
  and **0.4271 [0.3675, 0.4871]** — CI bounds identical to the registry.
- **v1 on all 600: `ade_0_2s` 0.4108 [0.3956, 0.4273]**, 13,198 windows / **600 clusters**.
  *Estimator: episode-cluster bootstrap, `taniteval/ci.py`, B=2000.*
  ⛔ **NOT a correction to 0.4271** — different deployment, and the 600 is **easier** (CV floor
  0.8377 → 0.6917). Registry row pending (§7).
- ⭐ **`along_track_vs_cv` flipped from "tie" [−0.0278, +0.5304] to "model wins" [+0.1926, +0.3104]
  while the effect itself moved 0.7 %.** A pure power flip **on our reference arm** — that is what
  n=40 was costing us.

**Horizon recommendation (MEASURED yield):** **K=60 (6.0 s) primary, K=70 hard maximum**, K=185
report-only-pooled. Junction clusters: 232 (K=20) → 204 (K=70) → 196 (K=75) → **58 (K=185)**.
⚠️ **The horizon is the binding wall, not the corpus** — 600 *is* the corpus ceiling
(`randperm(3000, seed 0)`, read at source), so **above K=70, HP-2 is permanently unmeasurable**.

**Per-HP runnability at n ≥ 200:** HP-1 ✅ · HP-2 ⚠️ K≤70 only · HP-3 ✅ (558/520, ~14× margin —
the published 40 gave 37/34, *below* the bar) · HP-4 ⛔ VectorMap instrument, not n · HP-5 ✅ ·
HP-6 ✅ · HP-7 ⛔ (232 ceiling) · HP-8 ⛔ (172 @K=100).

### 3.2 E1c — closed-loop CL-SFT is **BOUND**, and the trade is **real**

MEASURED, `…/2026-07-26-e1c-heldout-gated-clsft/e1c_frontier_result.json`.
*Estimator: paired episode-cluster bootstrap, B=2000, resampling held-out episodes.
`overlapping_holdout_se` used **nowhere**. Bonferroni α = 0.00278 over M=18.*
Held-out val `physicalai-val-heldout-79d4e3d2d4c6`, **n=44 episodes**, corridor ±1.75 m,
junction 10°, stride 8.

- **VERDICT `BOUND`: 17 frontier points, primary fired 15/17, guardrails held 0/17, success points 0
  — the intersection is EMPTY.**
- Base arm re-rolled as a control and **reproduces E1b digit-for-digit**: dep 0.587681 / junction
  0.841441 / peak XTE 38.944473 / open-loop ADE@2s 0.474666.
- ⭐ **The open-loop cost is paid before the gain arrives.** At step 100 the open-loop ADE delta is
  already **+0.4542 [0.3793, 0.5285]** — ~96 % of the eventual loss — while corridor departure is
  still **worse** (+0.2102). **Early stopping cannot rescue this lever.**
- Per the pre-registration §4.2 this means the closed-loop / open-loop trade is **REAL, not a guard
  artifact**.

⚠️ **Caveat I am flagging myself:** E1c's guardrail Gc adjudicates on OOD peak ≈ **1.2664 base /
1.2919 @step100 against a `≤ 1.30` bar** — that is the **C13 void criterion** (`np.interp` clamps
at 3 m / 12°, so the ratio saturates and the guard *cannot fail*). **Gc's passes are therefore not
evidence.** This is exactly what the pod2 agent is re-checking with `ood.py` right now (§4).

### 3.3 v4 30k gate — see §6, this is now partly an instrument story

MEASURED from `…/2026-07-26-v4-30k-gate/raw/`:
`wm_canary_ade_2s` **1.1409** vs ≤0.55 **FAIL** · `miss_at_2m` **0.2123** vs ≤0.10 **FAIL** ·
`ade_0_2s` **0.6423 oracle / 0.8563 produced**, both CI-separated *behind* v1's **0.4271**, and
worse than its own 15k (0.5839) · `seam_norm_ratio_max` 0.1204/0.1208 PASS.
⚠️ The primary is **goal-ORACLE-fed** (three channels; `vt_speed` is overwritten with observed
`v0`) — **not deployed capability**, and must never be worded as such.
⚠️ **`nonav_route_beats_majority` = VOID BY CONSTRUCTION → INSTRUMENT-FAIL, never MODEL-FAIL.**
Printed here explicitly, per GATE_PROTOCOL §0.7: a suppressed criterion that is not printed is
indistinguishable from one that passed.

Checkpoint safe: `8771c1d9d3da696dcde2a745d628f6a8`, 3,243,109,310 bytes, remote LFS sha256 ==
local, on gated HF `Sayood/flagship-v4-fromscratch`.

---

## 4. Streams — status

| Stream | State | Evidence class |
|---|---|---|
| **D-A closed-loop** | **BOUND on this lever family** (§3.2). Next: intervention #3 (drivable-corridor channel) — **now unblocked-in-progress via the VectorMap agent** | MEASURED |
| **4-brain dominance** | Strategic **PASSES** (51/51 two independent probes; `target_branch` 0.9827) needing **~103 scenes**. Tactical **FAILS** on two independent grounds: agents don't react ([−0.21, +0.14] m vs a **4.5 m noise floor**) and conflicts are **~50× too rare** | MEASURED |
| **v2 corpus (pod1)** | step 11,150/30,000, 10.88 s/step, ETA ~57 h | MEASURED |
| **v4 flagship** | 30k gate NOT-CONTINUE; restart budget **0/2**; lever diagnosis **in flight** | MEASURED |
| **IDM / D-B YouTube** | Retry window opened **12:00 UTC**; agent launched 12:29 UTC — **see §5** | in flight |
| **H2 attention-camera** | L2 label defined (`a_req` + agent-removal counterfactual + actual brake) | — |
| **Datasets** | nuScenes ingest code + 22 green tests; **blocked on a human Terms acceptance** | MEASURED |
| **Orin/Thor** | artifacts under `…/2026-07-26-orin-thor-optimization` | — |
| **AlpaSim** | consolidation landed; VectorMap now being stood up as an instrument | in flight |

**Agents running (4):** D-B YouTube retry · pod2 K-sweep + envelope close-out · pod3 VectorMap
corridor instrument · eval-pod v4 restart-lever diagnosis.

---

## 5. ⚠️ D-B YouTube retry — FIRED, outcome pending

The window opened **2026-07-26 12:00 UTC**. The agent was launched **12:29 UTC** with the
pre-registered gentle config **`W=2 TARGET=400 SEEDS=4 --sleep 4`**, GeoCalib geometry, **one run
only**.

**Its brief carries the absolute constraint verbatim: never bypass or evade bot-detection** — no
cookies, no alternate player clients, no proxies, no UA spoofing, no retry storms. **If blocked,
STOP and report; do not adapt.** Being blocked is an acceptable reportable outcome; a bypassed
block would be a program-level failure. It was also told to verify byte count and duration
per clip, because **a partial download must not become a silent success** (yield 12/400 is a
failed run reported as 12/400).

**Outcome not yet known.** It will be reported in the next slot.

---

## 6. ⭐ Retraction / correction this slot — instrument, class C10

**What I have been saying:** "the v4 30k gate returned NOT-CONTINUE → RESTART."

**What the instrument actually returned:** **`NOT_YET`** —
> *"step 29999 < pre-registered gate step 30000."*

A 30,000-step run indexes 0…29999. `run_gate.py` compared the trainer's **0-indexed** counter
against a **1-indexed** count and **refused a complete 59-hour run**. Both projections
(`GATE_30K_verdict_A/B*.json`) carry `NOT_YET` and are explicitly labelled **"NOT the verdict"**.

**Root-cause class: C10 — the evaluator does not implement its pre-registration.** It is also a
**recurrence**: commit `3ff5499` fixed this exact class and *a sibling instance survived the fix*.

**Credit where due:** the gate agent found this itself, documented it as §D3, and adjudicated by
hand against the card with every criterion printed. **The outcome does not change** — two on-card
kill secondaries fail on MEASURED values by ≥2×, and no reading of the card rescues that. But the
*provenance* of the verdict does change, and I was quoting a machine verdict that did not exist.

**Repair verified — MEASURED, by me, this slot:**

| input | `reached` | convention named |
|---|---|---|
| 29999 / 30000 | **True** | `0-indexed (trainer convention)` |
| **29998 / 30000** | **False** | refused under *either* convention |
| 30500 / 30000 | True | `1-indexed (step count)` |

The repaired guard **still refuses a genuinely unfinished run** — i.e. it is not a C13
"guard that cannot fail". Per the gate agent's instruction the v4 gate is **not** re-rendered;
the fix applies to future gates.

---

## 7. Decisions owed by Sayed

1. **v4 restart** — budget 0/2. Lever diagnosis in flight; **do not restart on a guess.**
2. **Wheelbase A/B/D** — measurement recommends **B**. (You chose C = measure first; the
   measurement is done.)
3. **~103 scenes** for the strategic proof (the 4-brain gate that PASSES).
4. **Tactical escalation** — agents don't react, conflicts ~50× too rare. This is a data-strategy
   decision, not a modelling one.
5. **E1c's 17 frontier deltas (935 MB)** — pod3-only storage call.
6. **nuScenes Terms acceptance** (~2 min; no credential exists and I must not create one).
7. **S3 longitudinal R3 at 12 s** — CI-separated but negative; clears at 8 s. Deliberately not
   adjudicated.

---

## 8. Blocked, and on what

- **HP-4** — VectorMap instrument (agent in flight), **not** sample size.
- **HP-7 / HP-8** — corpus ceilings (232 / 172). No compute fixes these.
- **HP-2 above K=70** — **permanently unmeasurable**; 600 episodes is the corpus maximum.
- **nuScenes** — human Terms acceptance.
- **Two live guard constants still adjudicate on the void OOD criterion**: `e1b_eval.py:403`,
  `e1c_common.py:34` (both `<= 1.30`). **14 committed artifacts carry a factually false OOD
  verdict string**; 118 nodes (the E1a family) quote a ratio with no verdict.
- `planner_p2.py` still un-migrated off `_jack_scalar`/`_jack_paired`; MODEL_REGISTRY §1.2 and
  **externally-published HF model cards** carry legacy numbers.
- IDM 0-GPU fixes (drop impossible yaw labels, fix comma heading derivation, remove `long_accel`
  from `SCALAR_NAMES`) — these change **published** numbers and need an owner.
- Three `cam_h` values (1.5 / 1.43 / 1.22 m) unreconciled; FOV conflict 51.4° vs 33.1°.

---

## 9. Next steps, priority order

1. **Land the four running agents** and commit each (branch + main).
2. **Register v1@600 (0.4108) as a NEW registry row** — never as an edit to 0.4271.
3. **Retire the void OOD criterion** at both live call sites once `ood.py`'s verdict is in, and
   correct the 14 artifacts carrying the false verdict string.
4. **Pin the S3 skill bars** — they move ±0.01 on identical code and data while quoted to 4 dp.
   An unpinnable bar must not sit in a kill conjunction.
5. **Register the horizon** (K=60 primary / K=70 hard max) into GATE_PROTOCOL once the envelope
   check confirms or refutes the yield-based recommendation.
6. **D-B outcome** → next slot.

---

*Estimator discipline: every interval above names its estimator. `overlapping_holdout_se` appears
nowhere. Lateral/longitudinal decomposition is carried on trajectory errors. Numbers are
per-corpus, never pooled. Unit of resampling is the **episode cluster**, not the window.*
