# refcv4 — the two launch blockers, cleared, and the run that started on them

**2026-09-04 · Architecture & Inference FlyWheel · MEASURED unless marked otherwise**
**Status: LAUNCHED.** `tanitad-refcv3` (A40 46 GB), run dir
`/workspace/experiments/refcv4-b1-v72-40k`, supervisor `sup_refcv4.sh` live.

---

## 0. The headline

| | before (refcv3, MEASURED) | after (refcv4, MEASURED) |
|---|---|---|
| anchor vocabulary | `refc.default_anchors` — **SYNTHETIC**, silently, because `--anchors` was never passed | `refc_anchors_6s_b1train_128.pt` — k-means over **635,331 real GT windows / 4,572 TRAIN clips** |
| oracle-in-vocab ADE 0–2 s, **held-out** | **1.0882 m** | ⭐ **0.3796 m** (ALONG 0.3066 / LAT 0.1496) |
| … vs a single straight line (0.6843 m) | **LOSES by 0.4039 m** | **BEATS by 0.3047 m** |
| … vs refc-base's published 0.4369 m | ×2.49 worse | ⭐ **0.87× — better, at a 3× longer horizon** |
| S2 reach band at the planned horizon | ±15.0 m/s, kills **18.02 %** (from a **2 s** statistic quoted at 6 s) | ±12.0 m/s, kills **26.23 %**, GT deletion 0.000 % eval / 0.007 % train, ΔADE **+0.00000 m** |
| tactical aux budget | **0.20** (double the documented `MANEUVER_WEIGHT`) | **0.10** |
| strategic head | **never supervised** — `goal_str` in 0 of 614 rows | supervised — `goal_str` live in row 1 |

⚠️ **The synthetic number is WORSE than the one on the register.**
`D-REFCV3-SMOOTH1` records **0.9433 m** for the synthetic set, measured on a
**4-slot** `default_anchors((5,10,15,20), …)`. The vocabulary refcv3 *actually*
carried is the **8-slot** one (`V3_HORIZONS`), which reads **1.0882 m** on
19,602 held-out windows. Same defect, 15 % larger. The register row should be
read as a lower bound.

---

## 1. ⛔ BLOCKER 1 — the 6 s data-driven anchor vocabulary

**Built by** `raw/scripts/build_anchors6s.py` · **log** `raw/anchors6s.log` ·
**artifact** `raw/refc_anchors_6s_b1train_128.pt` (`[128, 8, 2]` float32, bare
tensor, sha256 `68f81acf83b6b21ce533df282f78fe269e9b3b462eb82c726004821c7c1a806b`)
· **metadata** `…​.pt.json`.

* **Pool:** every window of the **B1 v7.2 TRAIN split**, `/root/data/train`,
  4,572 clips → **635,331** ego-frame GT trajectories at
  `V3_HORIZONS = (5,10,15,20,30,40,50,60)` = 0.5/1/1.5/2/3/4/5/6 s;
  subsampled to 200,000 with seed 0 for the fit.
* ⛔ **Clip-disjointness asserted and printed**, on the `.v2ep.pt` FILENAMES
  (independent of the loader): `train 4572 · eval 141 · INTERSECTION 0 -> []`.
  The gate is scored on the **141 eval clips / 19,602 windows** the vocabulary
  never saw.
* **Ego-frame convention verified against source** (`refb_labels.ego_frame`),
  not re-derived by eye.

### 1.1 The gate (held-out, metres, lower better)

| arm | ADE 0–2 s | ALONG | LAT | 0–6 s ADE |
|---|---|---|---|---|
| CONTROL zero path (no information) | 14.3264 | 14.3064 | 0.3160 | — |
| CONTROL constant-velocity straight line | **0.6843** | 0.4783 | 0.3160 | — |
| INCUMBENT synthetic 128 (refcv3's actual) | **1.0882** | 0.9087 | 0.4135 | — |
| FPS raw *(what `build_refc_anchors.py` does today)* | 0.8144 | 0.6429 | 0.3499 | 3.5758 |
| FPS slot-normalised | 0.8188 | 0.6409 | 0.3500 | 3.7404 |
| FPS slot+axis whitened | 0.8575 | 0.6347 | 0.4228 | 3.5470 |
| FPS whitened, lateral ×0.5 | 0.8125 | 0.5716 | 0.4288 | 3.6827 |
| FPS whitened, lateral ×0.25 | 0.7666 | 0.5478 | 0.3988 | 3.7595 |
| k-means raw | 0.3867 | 0.3068 | 0.1570 | 1.4783 |
| k-medoid raw | 0.3904 | 0.3078 | 0.1633 | 1.4893 |
| ⭐ **k-means slot-normalised — SHIPPED** | **0.3796** | **0.3066** | **0.1496** | **1.4838** |
| k-medoid slot-normalised | 0.3831 | 0.3072 | 0.1553 | 1.4878 |

**GATE: ✅ PASS.** 0.3796 m beats the straight-line control (0.6843 m measured
here, 0.6780 m published — an independent reproduction on different windows)
and beats refc-base's 0.4369 m target at **0.87×**.

### 1.2 The gain is where the deficit is

`D-REFCV3-AXIS1` measured the deficit at **92.2 % along-track**. The rebuild
moves along-track error **0.9087 → 0.3066 m (−0.6021, −66.3 %)** and lateral
**0.4135 → 0.1496 m (−0.2639, −63.8 %)**. In absolute metres **69.5 % of the
summed reduction is along-track**, so the budget did land on the axis that was
losing. Both mechanisms named in the brief are addressed:

* **(a) the 1-D longitudinal half** — `synth_anchor_pool` gives every anchor ONE
  constant acceleration for 6 s. Gone: the pool is real trajectories.
* **(b) the unnormalised FPS metric** — `furthest_point_sample` flattens
  `[M, S·2]` and normalises in `n` but never in `S`, so the 6 s coordinate
  carries ~9× the magnitude (~81× the squared-distance weight) of the 0.5 s one.
  **Fixed by an explicit per-slot metric**, and the metric was *swept*, not
  assumed: `w[s,:] = 1/RMS_s` (slot-normalised) is the shipped choice.

### 1.3 ⭐ The result that changed the construction: FPS cannot clear the floor at 6 s

**Every FPS variant loses to a straight line** (0.77–0.86 m vs 0.6843 m).
`build_refc_anchors.py`'s docstring rejects k-means because it *"collapses
centroids onto the straight mode and starves the turns"* — a statement about
**mode coverage**, which is not the quantity this gate scores. The gate is the
mean min-distance to the vocabulary, and that is **exactly Lloyd's objective**;
FPS minimises the worst-case covering radius instead. Out of sample, k-means is
**2.1× better** than the best FPS arm. ⇒ **the construction changed from FPS to
k-means, decided by the held-out gate rather than by the docstring.**

⚠️ **k-medoid is within 0.9 %** (0.3831 vs 0.3796) and guarantees every anchor
is a really-driven path. The gate winner shipped, because the criterion was
pre-stated; the choice is **not load-bearing** and k-medoid is the drop-in
alternative if averaged anchors are ever suspected.

### 1.4 ⭐ NEW — refcv3's vocabulary is not merely coarse, it is UNFLYABLE

Per-slot finite differences on the anchor sets themselves (`raw/clamp6s.json`
run log):

| vocabulary | max speed | max \|accel\| | segments > 8 m/s² | >30° turns | max turn |
|---|---|---|---|---|---|
| **SYNTHETIC (refcv3's)** | 46.13 m/s | **14.15 m/s²** | **10.71 %** | 61/128 | **74.5°** |
| **NEW (shipped)** | 36.21 m/s | 4.30 m/s² | **0.00 %** | 23/128 | **178.4°** |

The mechanism is in `synth_anchor_pool`: it samples `v ≤ 30 m/s` and
`|yaw_rate| ≤ 0.35 rad/s` **independently**, so it emits 30 m/s at 0.35 rad/s —
an 86 m radius at 108 km/h, **1.07 g lateral**. So the synthetic set is
simultaneously **over-populated with medium turns it cannot fly** and **missing
every sharp turn** (74.5° ceiling vs the real corpus's 178.4°). It is not a
sparser version of the right set; it is a different set.

---

## 2. ⛔ BLOCKER 2 — the reach clamp, re-derived at the horizon actually planned

**Instrument** `raw/scripts/derive_clamp6s.py` · **artifact** `raw/clamp6s.json`.

`refc.py:652` derives `horizon_s = max(trajectory.horizons) × 0.1`, so at
`V3_HORIZONS` the band is `a × 6.0`, and the inherited `sel_accel_max = 2.5`
opens it to **±15.0 m/s** on a corpus whose `v0` mean is 5.24 m/s.
`refc_v3.py:79-82` warned **in writing** not to inherit the 2 s statistics.

### 2.1 The criterion that actually binds is GT DELETION, not kill rate

A band that removes the trajectory the ego **actually flew** is wrong, not
conservative. Measured directly: `|v_mean(GT over 6 s) − v0|` is
p50 **0.866** · p90 **3.235** · p99 **5.688** · p99.9 **8.160** · max **16.956** m/s
on 635,331 train windows (eval: 0.761 / 3.150 / 5.534 / 7.752 / **8.516**).
⇒ the band cannot be tightened below ≈1.42 m/s² without deleting eval truth.

### 2.2 The sweep (19,602 eval + 635,331 train windows, on the NEW anchors)

| a_max | band | kill % | empty % | GT del. eval / train | surv/win | >30° turns/win | ΔADE |
|---|---|---|---|---|---|---|---|
| 0.500 | ±3.0 | 76.08 | 0.00 | 11.009 / 12.012 | 30.6 | 6.6 | +0.01577 |
| 0.750 | ±4.5 | 65.03 | 0.00 | 3.194 / 3.272 | 44.8 | 9.7 | +0.00230 |
| 1.000 | ±6.0 | 54.84 | 0.00 | 0.388 / 0.738 | 57.8 | 12.5 | +0.00018 |
| 1.250 | ±7.5 | 45.94 | 0.00 | 0.117 / 0.183 | 69.2 | 14.8 | +0.00000 |
| 1.500 | ±9.0 | 38.18 | 0.00 | **0.000** / 0.048 | 79.1 | 16.6 | +0.00000 |
| ⭐ **2.000** | **±12.0** | **26.23** | **0.00** | **0.000 / 0.007** | **94.4** | **19.0** | **+0.00000** |
| 2.500 *(inherited)* | ±15.0 | **18.02** | 0.00 | 0.000 / 0.002 | 104.9 | 20.5 | +0.00000 |

**RE-DERIVED VALUE: `sel_accel_max = 2.0`** (band **±12.0 m/s**), selected by the
criterion stated in the script **before** the numbers existed: eval GT deletion
exactly 0, train GT deletion ≤ 1 in 10,000, never empty, ΔADE ≤ 1 mm, then the
tightest such `a`. **Kill rate 26.23 %** (vs 18.02 % inherited), 94.4
survivors/window of which **19.0** are >30° endpoint-bearing turns (29.3 by
terminal heading), **0.00 % empty windows**. `refc_v3.py:448` now carries the
6 s table instead of the 2 s claim.

⭐ **The finding worth keeping: the inherited VALUE was defensible; the
inherited STATISTIC was wrong by 4×.** 2.5 does not delete truth at 6 s — it
simply kills 18 % where its docstring promised 72–77 %. A config that is right
for the wrong reason reads exactly like one that is right.

⚠️ **7.02 % of windows keep no >30° turn at a = 1.5 (4.40 % at 2.0).** These are
high-speed windows where a sharp turn genuinely is unreachable — physics, not a
defect — which is why "a turn survives in EVERY window" was rejected as a gate.

---

## 3. Also folded in — with what changed and why

| item | decision | evidence |
|---|---|---|
| **tactical aux budget 0.20 → 0.10** | **FIXED.** `refc_v3_train.py` spent `LAT_WEIGHT×(loss_lat + loss_lat_tac) + LON_WEIGHT×(…)` = 0.05×2 + 0.05×2 = **0.20**, against `refc_train.py:83-91`'s written invariant that total tactical pressure is held at **exactly** `MANEUVER_WEIGHT = 0.10`. Halved to 0.025×4. | `refc_v3_train.py:654-655` + the comment block beneath it |
| **`--goal-str`** | **PASSED.** refcv3's strategic head was never supervised. Live in row 1: `goal_str 0.76149`. | preflight `goal_str=1.0482 live / 0.0000 under the +inf-guard control (control able to fail: True)`, LAN `any_valid_frac 0.9766` |
| ⚠️ **`--graft-lan`** | ⛔ **REFUSED — the brief's parenthetical would have opened a route leak.** `--graft-lan` is the **supplied corridor as a MODEL INPUT**, refused by **E12** and the vision-only rule; its own help string says *"NOT part of any registered v3 arm; diagnostics only"*. The **label** path is `want_lan = bool(args.graft_lan or args.goal_str)` (`refc_v3_train.py:1084`) — **`--goal-str` alone mints the label** without building the input pathway. Verified on the built config: `graft_lan False`. | `refc_v3_train.py:1084`, `refc.py:904-909`, config check |
| **Defect A** (`refc_v3.py:890`, `[B,3]×[B,3]` into the 8-wide v7.0 heads ⇒ `BRAKE_TO`/`ACCELERATE`/`TURN_L`/`TURN_R` get 0.000 % mass) | **RECORDED, NOT FIXED.** A head-width change is structural and would make this arm non-attributable. Next arm. | brief; unchanged in this run |
| **`(accel, curvature)` through `rollout_unicycle`** | ⛔ **NOT in this run**, as instructed. Recorded as the next arm. | brief |
| **`--echo-base` / E14** | **OFF.** The tiny rig's own measurement moved the verdict `READS_BOTH → ECHOING`. Second arm. | design-package §6.4 / §7 |
| ⭐ **NEW: the anchors + selection are now STAMPED in `config.json`** | `_anchor_stamp()` records the sha256 of the tensor **actually installed in the decoder**, plus the file sha256, shape, and `source` — plus `sel_reach_clamp` / `sel_accel_max` / `horizon_s` / `band_ms`. **refcv3 recorded none of this**, which is precisely why "which vocabulary did it train on?" had to be answered by reading the launch script months later. | `refc_v3_train.py:848` `_anchor_stamp`, `:1309` |

---

## 4. The launch

```
OUT=/workspace/experiments/refcv4-b1-v72-40k          # ⛔ NEW dir; refcv3-b1-v72-30k untouched
cp /workspace/refc_anchors_6s_b1train_128.pt $OUT/anchors.pt
setsid nohup bash /workspace/sup_refcv4.sh > /workspace/sup_refcv4.out 2>&1 < /dev/null &
```

which runs (embedded in the supervisor, so what you read is what runs):

```
PYTHONPATH=/workspace/TanitAD/stack nohup python3 -u \
  /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --anchors /workspace/experiments/refcv4-b1-v72-40k/anchors.pt \
  --sel-accel-max 2.0 \
  --goal-str \
  --ego-state-inject --ego-dropout 0.5 \
  --out /workspace/experiments/refcv4-b1-v72-40k \
  >> $OUT/train.log 2>> $OUT/train.stderr.log 200>&- &
```

**Everything except the four new flags is the incumbent's own command**, so
refcv4-vs-refcv3 stays a lever set and not a bundle.

### 4.1 Verified by CONTENT, not by exit code

* **Pod code currency, before launch.** `/workspace/TanitAD/stack` held the
  **PRE-v4** `refc_v3.py` (39,215 B, `ego_state_inject` ×**0**). All five v4
  files were shipped and md5-matched at both ends, then **grepped in the file
  the launch imports**: `ego_state_inject` ×11, `--sel-accel-max` at
  `:1573`, `LAT_WEIGHT / 2.0` at `:654`.
* **Config, built from the real CLI:** `sel_accel_max 2.0` · `horizon_s 6.0`
  (derived) · band **±12.0 m/s** · `ego_state_inject True` · `echo_base False` ·
  `ego_valid_channel True` · `ego_dropout 0.5` · `graft_lan False` ·
  hier/flat delta `['core.graft_target_latent', 'hier']` **==** registered ·
  v4-vs-v3 delta `['core.ego_valid_channel', 'ego_state_inject']`.
* **Preflight ✅ PASS:** params **107,053,435**, `ego_inject 25,536`,
  freeze-history `pass True`, `E11' OK`, LAN arm OK with a control able to fail.
* **A 4-step and a 2-step REAL-DATA run** on the full corpus with the exact
  final flag set, before touching the run directory — the synthetic preflight
  cannot exercise `--goal-str` × `--v7-labels` × `--u8-batches` × `--nav-from-v7`.
* **`config.json` of the live run** stamps
  `anchors.file_sha256 = 68f81acf…a806b`, identical to `sha256sum` of the built
  file, and `selection.band_ms = 12.0`.
* **First logged row (step 50):** `ego_injected 1.0` · `ego_keep_frac 0.5` ·
  `nav_injected 1.0` · `tac_label_v7 1.0` · **`goal_str 0.76149`** ·
  `goal_gate_grad 0.60824`.
* **Supervisor asserted present** by `ps`, not by the launch's own success.
* **Disk by a real `dd` write** (never `df`): 3.1 GB at **483 MB/s**.

### 4.1b Rate and ETA — PRELIMINARY, and it disagrees with the inherited figure

Launched **2026-09-04 07:10 UTC** (09:10 Europe/Berlin). First rows:
step 50 @ `elapsed_s` 106.7, step 100 @ 204.0 ⇒ **marginal 1.946 s/step**
(Δelapsed/Δstep between two logged rows — NOT `step_s`, which this programme has
twice mis-divided). At that rate 40,284 steps is **~21.8 h**, finishing
**~2026-09-05 05:00 UTC**.

⚠️ **This is 2.4× faster than the ~53 h/epoch the brief carries, so treat it as
preliminary, not as a correction.** It is **n = 2 rows over 50 steps**, it
excludes the periodic eval (8 batches every 500 steps) which has not yet fired,
and the v2 LRU is still filling. ⇒ **re-measure over ≥2,000 steps before
quoting an ETA anywhere that decides anything**; the honest present statement is
*"between ~22 h and ~53 h, first real estimate at step 2,000."*

**Update at step 450** (9 logged rows): marginal **1.949 s/step**, ETA **21.6 h** —
stable, but the 500-step eval had still not fired. Loss 145.4 → **21.42**,
`traj` 20.34 → **1.275**, `cls` 44.34 → **4.010**, `anchor_acc` **0.050**,
`goal_str` **0.6019** (live), `sel_v3` **3.97**.

### 4.2 Known defect in the supervisor — recorded, not patched live

`last_step()` is inherited from `sup_refcv3_v3.sh` and, **while
`metrics.jsonl` does not yet exist**, `set -o pipefail` makes the pipeline fail
after `python3` has already printed `0`, so the `|| echo 0` fallback appends a
second line and the `-ge` test errors with *"integer expression expected"*
(visible once in `sup_refcv4.out`). It **self-heals** the moment the first row
is written, and the failure mode is safe: the test **errors** rather than
passing, so it can never declare a false DONE. ⛔ Not patched in place — editing
a running bash script makes a live shell execute garbage from mid-line.

---

## 5. Deliverable manifest

| artifact | where it lives |
|---|---|
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-launch/RESULT.md` |
| anchor vocabulary `[128,8,2]` | `repo:…/raw/refc_anchors_6s_b1train_128.pt` **and** `tanitad-refcv3:/workspace/refc_anchors_6s_b1train_128.pt` **and** `tanitad-refcv3:/workspace/experiments/refcv4-b1-v72-40k/anchors.pt` |
| its build metadata + gate | `repo:…/raw/refc_anchors_6s_b1train_128.pt.json`, `repo:…/raw/anchors6s.log` |
| clamp sweep | `repo:…/raw/clamp6s.json` |
| builder / clamp instruments | `repo:…/raw/scripts/build_anchors6s.py`, `…/derive_clamp6s.py` (also `tanitad-refcv3:/workspace/`) |
| supervisor | `repo:…/raw/sup_refcv4.sh` **and** `tanitad-refcv3:/workspace/sup_refcv4.sh` |
| trainer changes (`--sel-accel-max`, tactical /2, `_anchor_stamp`) | `repo:stack/scripts/refc_v3_train.py` **and** shipped to the pod |
| clamp docstring re-derivation | `repo:stack/tanitad/refs/refc_v3.py` **and** shipped to the pod |
| the run itself | `tanitad-refcv3:/workspace/experiments/refcv4-b1-v72-40k` — **ONE PLACE ONLY** until the first checkpoint is pulled |

## 6. Escalations for the Master Mind

1. **`build_refc_anchors.py` saves a DICT; `refc_v3_train.py:1032-1034` does
   `torch.load(...).to(device)` and would crash on it** (`'dict' object has no
   attribute 'to'`). Sidestepped here by saving a bare tensor. **The repo tool
   and the repo trainer do not agree about the file format** — one of them must
   change before anyone uses `build_refc_anchors.py --v2-cache` for a launch.
2. **`build_refc_anchors.py` should gain the k-means path and the per-slot
   metric**, or the next person rebuilding a vocabulary reproduces the FPS
   result that loses to a straight line. The instrument here is standalone.
3. **`GOALS_AND_CLAIMS.md` `D-REFCV3-SMOOTH1` quotes 0.9433 m** for a 4-slot
   synthetic set; refcv3's real 8-slot set reads **1.0882 m**. Register row
   needs the correction.
4. **Defect A is still live** and is the next arm, together with
   `(accel, curvature)` through `rollout_unicycle`.
