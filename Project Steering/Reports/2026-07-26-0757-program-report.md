# TanitAD program report — 2026-07-26 07:45 UTC (09:45 Berlin)

**Mode:** autonomous (PI away). **Since the last report:** an unattended night in which **8 commissioned
agents landed**, **~40 commits** were pushed to branch **and main**, and **the measurement layer was
repaired in three places where it was silently broken.** Evidence class on every number; estimator named.

---

## 1. Fleet

| pod | state | MEASURED |
|---|---|---|
| **pod2** | flagship-v4-fromscratch | **step 29,300 / 30,000**, 7.09 s/step ⇒ **1.4 h to finish**. Trainer PID 108011 alive, restarts 0 |
| **pod1** | flagship-v2corpus-30k | ~step 7,900+/30,000, 10.65 s/step ⇒ ~65 h. **DO NOT RESTART** (§4) |
| **pod3** | **FREE** (0 %, 0 MiB) — released by E1c | — |
| **eval** | **FREE** (0 %, 0 MiB) — and **now synced**, see §3.1 | v1 MODE A reproduces **0.4214799702167511**, bit-identical pre/post sync |

⚠️ **WM canary note (C5 discipline):** pod2's last row reads `wm 2.569` against `1.642` an hour earlier.
**That is ONE point and is NOT a trend** — the same shape as the 07-24 "canary descending CONFIRMED"
retraction. Judge at the gate, on eval output, not on a trainer row.

---

## 2. The headline results

### 2.1 Closed-loop: **E1c = BOUND, and the trade is REAL** (`e1c_frontier_result.json`)
17-point frontier, paired episode-cluster bootstrap (`taniteval/ci.py`, B=2000, 43 clusters / 6 junction).
**Primary fired at 15/17 checkpoints; held-out guardrails held at 0/17 — the intersection is EMPTY.**

The frontier showed what E1b's two endpoints could not: **the open-loop cost is paid BEFORE the gain
arrives.** At step 100 — inside lr warm-up — **96 % of the total ADE loss is already incurred and the
closed loop is CI-separated WORSE than base** (peak |XTE| 38.94 → 47.94 m). The gain separates only from
step 500 and is 98 % saturated by step 1000. **So the pre-registered stopping point is the first
checkpoint that exists: early stopping cannot rescue it.** OOD passed 17/17 and improved monotonically
(1.2919 → 1.1339), so no point is distribution-shift confounded.

**What the direction bought** (best point, step 2250): corridor departure **0.5877 → 0.16** overall,
**0.8414 → 0.42** junction, **peak XTE 38.94 → 3.04 m** — the largest closed-loop movement the program
has measured — at a minimum open-loop cost of **+0.19 m**, which is structural for this recipe.

**⇒ D-A status: significant closed-loop performance IS achieved and characterized.** Getting the gain
*without* the cost needs a **different mechanism**, not another iteration of the same lever. **I did not
launch E1d** — that is a PI decision, not a default.

### 2.2 4-brain dominance: the program **splits cleanly**
- **Strategic — PASSES.** `succ(lane)` readable; **two independent probes agree 51/51 exactly** (trajdata
  object model vs a raw USDZ read that never imports trajdata). 11,877 lanes / 12,030 successor edges /
  **0 dangling**. **`target_branch` = map ∩ realised path is computable**, ego-to-lane match **0.9827**.
  Blocked only on **scene count: 20 clusters vs bars of 40 and 200 ⇒ needs ~103 scenes — a download.**
- **Tactical — FAILS, on two independent grounds.** Agents **do not react to the ego**: effect bounded at
  **[−0.21, +0.14] m against a 4.5 m noise floor**, near-ego stratum **null in 4/4 scenes**, while the ego
  differed by up to 149 m. **And conflicts are ~50× too rare** (4 across 51 scenes; the two-arm bar needs
  ~2,550 scenes vs a 1,606-scene pool). **Escalate, do not scale.**
- ⭐ **`traffic_light.parquet` exists on 51/51 AlpaSim scenes.** "No traffic-light feature anywhere" is
  true of PhysicalAI-AV and **false of AlpaSim** — the capability is reopened.

### 2.3 The ladder's power ceiling: **BROKEN**
A **600-episode CLEAN** `physicalai-val-0c5f7dac3b11` build exists — on pod2 only. Crucially
**12 ⊂ 40 ⊂ 600 as an order-preserving prefix**, so 40→600 **adds episodes and re-selects none: parity
holds.** 600 is the hard maximum, so any two-arm HP-x is feasible **iff its episode yield ≥ 1/3**.
Blocker: a **66 GB move**, ~30–60 min via HF relay, **after pod2 finishes**.

### 2.4 H2 — **GO**, and honestly re-scoped
`L2` held-out lift **2.41× [1.3998, 3.7041]** (1,415 clusters), **leave-one-chunk-out 16/16 exclude 1.0** —
precisely the test `L1_gate` failed. **Quote the speed-adjusted 2.09× [1.19, 3.38].** ⚠️ **Junctions are
null (0.45×)** and **the off-front residual is not separated (1.66×)** ⇒ **the demonstrable capability is
front-periphery attention, not cross-camera switching.** Efficiency stands: **0.67 % residual need-rate
⇒ 84.8–85.6 % of surround-camera compute avoidable**, reproduced out-of-sample to three digits.

### 2.5 IDM — the answer to "more training or better architecture" is **neither, first**
Deleting **9 windows out of 4,195** (physically impossible GT yaw, comma at v≈0) moves pooled yaw R²
**0.105 → 0.497**; **on PhysicalAI the same head reads 0.9035.** Per-channel ceilings: speed = **monocular
scale** (a linear probe matches the 2.9 M head), yaw = **the comma label** (its own ceiling is 0.352),
steer = **redundant** (r = 0.9865 with ω/v), long_accel = **the label** (a perfect estimator caps at
R² 0.188). **"More training" is rejected on all three axes** — data flat past 34 clips, steps flat past 25
epochs, and **a 0.86 M head beats the shipped 2.90 M**.

---

## 3. Where the measurement layer was silently broken

### 3.1 🔴 The eval pod was **62 % stale and MISSING `corridor.py`**
`taniteval` 62.2 % wrong (18 stale, **5 missing**), `stack/scripts` 83.3 %, `tanitad` 52.3 %.
**`corridor.py` — §0's co-primary emitter — did not exist on the pod, so the 30 k gate could not have
rendered a corridor block at all.** `hierarchy_guard.py` and `tanitad/data/parity.py` also missing (the
val-parity guard was **inert**). **And a second stale tree nobody had named** (`/root/TanitAD/stack`,
hard-coded by every `taniteval` submodule via `sys.path.insert`, holding 1 of 16 lake modules) — found
only by an import-order accident; **probing the obvious tree alone would have declared the pod clean.**
All three synced, **141/141 md5-verified**, v1 bit-identical after.

### 3.2 Two live bugs fixed before the gate
- **`paired_cross_track` mislabelled the horizon 5×** (a sparse knot is 0.5 s, not 0.1 s ⇒ it reported
  0.4 s for a 2.0 s surface). Would have corrupted the gate's lateral panel — the exact C9 defect §0.7
  exists to stop. Fixed with explicit `knot_dt` + a `horizon_provenance` stamp; verified 0.4 → **2.0**.
- **`GATE_PROTOCOL` §0.8 — my own text — listed `vt_speed` as an oracle channel. It is not** (`_goal_inputs`
  overwrites it with observed `v0`). Three channels, not four. Corrected in place, with the reason.

### 3.3 🔴 The gate's primary is goal-oracle-fed — now measured
`oracle 0.5839 → produced 0.7577`, paired **Δ +0.1738 [+0.1247, +0.2356] separated**.
**Decisive: oracle beats CV separated (−0.2538); PRODUCED DOES NOT (−0.0800, CI spans 0).**
The gap is **entirely longitudinal** ⇒ what the arm buys from the oracle is **target speed, not route** —
and this explicitly does **not** license the S3 route-channel finding. The produced goal is **worse than no
goal**, because `curv_5s` has R² = 0.0749 and the route collapses to `straight` on 90.6 % of windows.

---

## 4. Decisions not taken (and why)

| | |
|---|---|
| **v2-corpus restart** | **NO.** Prime suspect dead (`--grad-checkpoint` off OOMs at production micro-batch; loses where it fits). No lever certified — `32×2` spans 0.84–1.40×. **And `16×4` is the arm's OWN instrument**: micro-batch is the level at which `decorr` (v2 LEVER B / H25) and SIGReg are estimated. Speed is not worth voiding the experiment. Real finding for the *next* launch: `aten::copy_` is **23 % of CUDA time** — unpinned H2D copy, a two-line fix |
| **E1d** | **NO** — see §2.1 |
| **X2 verdict run** | Not authorized (30 pod-days) |
| **Wheelbase fix** | PI chose C = measure first. **Measured; recommendation is B** (§5) |

---

## 5. Owed to Sayed

1. **Wheelbase — A/B/D.** Measurement says **B (fix forward only)**. ΔADE **+0.0056 [+0.0007, +0.0113]
   separated**, **entirely lateral** (cross-track +8.6 % relative). Threshold: 9.3 % of v1's own CI
   half-width; A would cost **~53 A40-days** to move numbers by less than their own noise. ⚠️ Both premises
   I gave were wrong: there are **five** wheelbases (mode **2.730 m at 47 %**, not 2.85 at 90 %), and my
   "steer is redundant so it's harmless" is **refuted** — **ω is not an input to v1** (`R²(steer | accel, v0) = 0.0002`).
2. **nuScenes** — decided *ingest freely*; blocked on a **~2-minute human Terms acceptance** (no credential exists).
3. **AlpaSim/TanitSim** — recommendation **do not fork** (the renderer licence forbids exactly what a fork
   would buy). **Give the *renderer* an owner** instead.
4. **4-brain strategic** — approve **~103 scenes** (a download) to clear the power bar.
5. **4-brain tactical** — escalation: agents don't react and conflicts are ~50× too rare. **One cheap test
   could reopen it**: a single full closed-loop rollout with `trafficsim=catk` — every other blocker is now
   removed.
6. **S3 longitudinal R3 at 12 s** — CI-separated but **negative**; clears at 8 s. **Deliberately not
   adjudicated post-hoc.**
7. **E1c's 17 frontier deltas (935 MB)** are pod3-only — the scientific object; needs a storage call.

---

## 6. Retractions logged since the last report (root-cause class)

| claim | class |
|---|---|
| "yaw is unrecoverable, R² 0.010" | **C5** — pooled metric destroyed by 9 impossible label rows; pooling across corpora hid it |
| "2.85 m for ~90 % of clips" | **C2** — distribution read from a single (100 %-US) chunk |
| "steer is redundant ⇒ the wheelbase error is harmless" *(mine)* | **C6** — redundancy measured on the DATA, asserted about the MODEL |
| "the trainer's 0.48 is on the descent to 0.427" *(mine)* | **C1** — a metric NAME is not a metric DEFINITION (dense-20 vs 4-waypoint) |
| E1's null blamed on the trigger | **C12** — a composite AND-label's null blamed on the wrong half |
| §0.8's four-channel oracle list *(mine)* | INHERITED into a binding protocol without checking the code |

---

## 7. Next, in priority order

1. **pod2 hits 30k in ~1.4 h → run the formal gate.** Render on **`oracle`** (the only surface comparable
   to the 10k/15k history), word it per **§0.8**, then re-render **`produced`** on the same checkpoint
   (~5 min) and report the **pair**. ⚠️ **§0.7: `nonav_route_beats_majority` is VOID BY CONSTRUCTION —
   adjudicate INSTRUMENT-FAIL, never MODEL-FAIL.** Then back the final ckpt to gated HF.
2. **Move the 600-episode val off pod2** (66 GB, HF relay) — unblocks every n ≥ 200 two-arm comparison.
3. **D-B YouTube retry at 12:00 UTC** (**4.25 h out**) — gentle config, ONE run, never bypass bot-detection.
4. Free pods (pod3, eval) onto: **REF-C-base canonical eval + v1.6 paired bootstrap** (pending exactly on
   "when pod3 frees"), and **re-running the closed-loop artifacts on the corrected estimator** — every
   circulating closed-loop interval *and mean* is still legacy.
5. Adopt **`OMP_NUM_THREADS=8`** for every per-window numpy labeller (a 9× slowdown was measured from
   111 BLAS threads × 2 processes on 96 cores).
