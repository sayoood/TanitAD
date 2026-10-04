# RESULT — WP-B P7 (X3), P8 (Q2 + Q8), P9 (Q1), (B) enc8, X10 merged: one combined landing

*WP-B, 2026-10-04. Built against the Master Mind's items (1)–(4), (A) and (B). Base tip `ce17f6f` (stack / taniteval /
tools subtrees identical to `c39798d`, asserted by tree hash). Tier: label-level and model-mechanism checks — no
driving number is claimed here.*

## P7 — X3: the past-only speed input N2 (SPEC_REFCV8 §8.3, MM ruling Q5)

**What is built.**
- `--r8-speed-input {n2,n3}` is the SOURCE of the `--max-speed-input-v6` 4-way channel: the v9 past-only column.
- `--r8-speed-unknown-p` (unset = 0.45) is the TRAINED unknown row: input dropout drawn from the dedicated r8 generator.
- `--r8-roll-speed-input` is the V-VSHUF roll, training only.
- At eval, `model._r8_speed_eval` = `off` / `shuf` gives VMAX-OFF / VMAX-SHUF.
- The NavSim-LEGAL row is the unknown row.

**The ceiling.** It reaches the EMITTED plan through the existing `e9_rank` path.
- With a past-only source the ceiling is never below the fed value. N2 = 130 km/h (1.98 % of train rows) caps at 36.1 m/s, not at the 4-way top step of 120.
- The inherited source keeps refcv7's rule unchanged.

**Refusals.** The v8 future-max sidecar is refused BY NAME in two independent places:
- the trainer (`_pin_refcv8`, on every refcv8 run);
- the refcv8 launch profile (a forbidden lever).

The mutation that re-attaches it goes RED at both places (`test_refcv8_launch_profile.py`).

**LEAK** (MEASURED; `raw/x3_leak.json`; instrument `stack/tanitad/eval/speed_leak.py` = D4's estimator; n 576,555;
manifest md5 `3c9f8bc8`):

| input | recovered | reference |
|---|---|---|
| D4's N2 recomputed from the poses | 0.0346 | D4 0.0346 |
| WP-A's shipped `speed_n2_kmh` (8-step) | 0.0348 | — |
| WP-A N2 with the unknown row p 0.45 (8-step) | 0.0094 | D4 0.0093 |
| window oracle | 0.5698 | D4 0.5698 |
| refcv7's v8 sidecar (4-way) | **0.2377** | USAGE_AUDIT 23.8 % |
| **fed by refcv8: 4-way, eval** | **0.0193** | bar ≤ 0.05 |
| **fed by refcv8: 4-way, training (unknown row)** | **0.0062** | bar ≤ 0.05 |
| fed by refcv8: 4-way, V-VSHUF roll | -0.000001 | ~0 |
| N3 through the 4-way | 0.0193 | = N2's 4-way |
| constant / target itself (controls) | 0 / 1 | exact |

Tolerances were fixed before the first run: ±0.00005 where the computation is D4's own, ±0.002 for another stream's artifact.

**Finding.** Through the inherited 4-way encoder {30, 50, 100, 120}, N2 and N3 produce the same partition. 70 and 80 km/h
fall into the 100 bin (20.8 % of train rows), so 44 % of N2's information is lost (0.0346 → 0.0193). That loss motivated (B).

**Full trainer.** `train()` on Thor CPU, with real data and the refcv8 argv:
- config.json carries the past-only provenance, no sidecar, the `r8_speed_derivation` stamp, and no v6 oracle stamp;
- census, train: 746,946 / 746,946 windows known, bins [0, .695, .255, .050];
- census, eval: 23,772 / 23,772 known;
- G-DVB: 0 mismatches.

⚠️ **The 2 training steps did NOT run.**
- Attempt 1 died in torch-inductor: a CPU compile needs `Python.h`, which is absent on Thor.
- Attempt 2, without `--trunk-compile`, was killed by a GLOBAL kernel OOM at 22:57:19 (16.4 GB RSS on Thor's unified memory).
- The OOM killer also took desktop session daemons (pipewire, wireplumber, dbus).
- ⚠️ **It also killed one job:** the registered pre-P7 GPU smoke (`r8_smoke_thor.sh`, old tree, old argv) had taken the GPU lock at 22:56:46, and its DataLoader worker exited at 22:57:47, inside the OOM window. That smoke was due to be replaced and is re-run on the L1 tree (`code/r8_smoke_thor_L1.sh`).
- R1 (H4 training), WP-D P-BOX and WP-RL were not hit.
- **A full-size CPU forward on Thor is not safe and was not retried.** Step-level coverage is the rig tests plus the registered GPU launch smoke after landing.

## P8 — Q2: the v9 constraint vectors are supervised; Q8: R8-1-REACH's key

**What is built.** `--w-r8-v9-cons` builds per-QUERY heads on the behaviour decoder:
- `lat_c` (12) on the lateral queries;
- `lon_c` (10) and `speed_goal` (4) on the longitudinal queries.

They are supervised with a Huber loss on the GT-active class's query.
- **Masked:** PARTIAL or absent v9 rows, NaN fields, and turn fields on non-TURN classes.
- **Scales:** INTEGRATION's (50 m / 8 s / 90° / 15 m/s), clamped at ±6.
- The heads are supervision only: no planner input reads them.
- **The `tac` roll carries the constraint targets** together with the classes.

**Analytic control** (MEASURED, the real decoder module): a synthetic STOP at distance d (5–80 m), with d visible in the
scene, is regressed to held-out **R² 0.9994, MAE 0.36 m**. The bars, fixed before the run, were ≥ 0.9 and ≤ 3 m.

**Deliberate regression:** with the targets shuffled across windows, R² is **−0.0035** (bar ≤ 0.1).

**Q8.** The in-run key is **`r8v9_tac_rows`**: the share of the batch's windows whose lateral AND longitudinal target is
supervised (an exact class or a non-empty partial mask). refcv7's `tac_label_rows` is a count of exact lateral rows and is
not this quantity.

**[2, 6] s label ↔ tag agreement, and its decomposition (MM (A)).**
Artifacts: `raw/label_tag_agreement_*.json`, `raw/turn_disagreement_*.json`.
- **The gap.** 80 % of v9 TURN labels have a turn segment that is not fully inside [0, 6] s.
- **Train, PLAN tag:** of 25,141 missed turns,
  - 86.2 % start inside the window but end after 6 s;
  - 9.6 % start after 6 s;
  - 2.6 % come from the tag's threshold definition;
  - 0.8 % are residual.
- **Ruling** (MM, 2026-10-04): R8-4 (iii) consistency uses the PLAN tag, scored on plan-observable rows only (a TURN segment inside [0, 6] s; LANE_KEEP with curve = 0).
- **Agreement on those rows:** TURN 0.9285 / 0.936, LANE_KEEP 0.965 / 0.972 (train / eval139).

## P9 — Q1: `--r8-alloc-emit-start S_emit`

**What is built.**
- The extra candidates (allocated and prior-free) are generated and trained from step 0, but can WIN the argmax only from `S_emit`.
- A warm-started run that emits at step 0 is refused.
- The I-2 row (`refcv8_train.i2_identity_row`) applies the accepted literals: sel_idx 100 %, max |Δtraj| ≤ 1e-3 m, max |Δ base score| ≤ 1e-4.
- It FAILS any argv that emits at step 0, whatever the numbers read.
- The loader sets the emission state from the checkpoint's REFCV8 step; a refcv7 checkpoint counts as step 0.

**S_emit = 2,000, the end of the launch argv's `--warmup 2000`.** The evidence:
- `iw_diag.py` (Thor GPU, 48 windows, unsplit tree): emission OFF moved the plan by ≤ 7.6e-5 m and the base scores by ≤ 3.1e-5, with sel_idx identical on 48 / 48. That is inside the I-2 literals by 13× and 3×.
- The GPU's own TF32 switch moves the plan 3.6e-4 m, also inside the 1e-3 literal.
- With the split decode the emission-OFF case is bit-identical: CPU full size 4 / 4, and the rig.
- So the emission-OFF phase costs nothing in identity at any length. S_emit is a schedule choice: extra candidates start winning once their heads have trained at full LR. It sits under R8-WARMUP for the Master Mind.

**I-2 on the rig.**
- The scheduled build reads bit-identical (0.0) and PASSES.
- The same build emitting from step 0 FAILS the row.

**I-2 at full scale (MEASURED, Thor GPU).** The REGISTERED SPEC_WPB I-W re-ran on the split-decode tree (`raw/I_W.json`, 218 grid windows):
- allocation 32 + prior-free with emission OFF: **0 windows differ** on traj, sel_idx and the base fan, and the max base-score difference is **0.0**, bit-identical;
- seams only: also bit-identical.

This is the emission-OFF state the launch argv is in at step 0 (`--r8-alloc-emit-start 2000`). The dev-box CPU full-size re-check was stopped when the dev box ran out of commit (0.7 GB free); it is superseded by this GPU reading.

## (B) `--r8-speed-enc8` — the full N2 ladder

**What is built.**
- The 8-step one-hot plus a known bit feeds a ZERO-INIT per-layer FiLM on the behaviour decoder's queries, beside the inherited 4-way channel.
- The seam reads the TREATED fed value, so the unknown row and the V-VSHUF roll apply to it.
- With enc8 on, the ceiling is the fed N2 value itself.

**Step-0 identity.**
- Bit-identical on traj, sel_idx, anchor_traj, sel_score_v3 AND the tactical logits, in training and at eval with an unknown speed.
- At eval with a KNOWN speed, the fan, the scores and the tactical logits are bit-identical. The emitted plan moves only where the tighter N2 ceiling binds; that follows from the ceiling ruling.
- A seam bias moved by 1e-3 makes the tactical outputs differ (the regression arm goes RED).

⚠️ **For I-1 with enc8 in the launch argv:** read the emitted plan with the ceiling off, or on rows where neither ceiling binds.

**LEAK** of the fed channel (4-way + enc8):
- **0.0348** at eval, against 0.0193 for the 4-way alone; it recovers N2's full information;
- **0.0094** in training with the unknown row;
- −0.000009 rolled.

All are ≤ 0.05.

**Obedience.** On the extended fan, the emitted plan obeys the N2-value ceiling (70 km/h = 19.4 m/s). The same windows under the
4-way ceiling (100) emit a plan above 70 km/h on at least one row, so the test is not vacuous.

## X10 (Data FlyWheel) — merged, not authored here

- **The merge:** three-way in LF space, with CRLF restored on the trainer. Nothing was dropped from either side (line-multiset check).
- **Registry pin:** set ONCE, to **268** = 256 + 5 ladder support + 3 X3 + 1 P8 + 1 P9 + 1 enc8 + 1 X10.
- **X10's own tests:** 27, green in this tree.
- **The launch argv** carries `--pose-sync-sidecar /home/nvidia/data/refcv6_pose_sync_sidecar.jsonl` (md5 `ce00a130…`), and the refcv8 profile requires it.
- **`refcv7_loader.py`** is untouched by X10, so the eval kit stays on the uncorrected clock.

## Gate (tip vs candidate, every test file importing the trainer / refcv8 / launch gate / loader / X10)

- Before P8: the identical 11 pre-existing failures on both sides, and ZERO failing only on the candidate.
- Combined landing: see LANDING_READY_WPB.txt `## WPB-L`.
