# PREREG — REF-C v4, the combined-lever arm (2026-09-04)

**Status:** the arm is **LIVE**. Launched 2026-09-04 ~07:10 UTC on `tanitad-refcv3`
(A40, 69.30.85.211:22001) into `/workspace/experiments/refcv4-b1-v72-40k`, supervised by
`/workspace/sup_refcv4.sh`. This prereg is written **after** the launch and says so; it was
not available beforehand because the launch and the prereg were carried by two different
agents that could not reach each other (see §7).

**PI decision, verbatim (2026-09-04):** *"combine the levers we dont have time."*

---

## 1. ⛔ ATTRIBUTION IS FORFEITED BY PI DECISION

Four levers move at once in this arm. **A regression is therefore NOT attributable to any
single lever**, and neither is an improvement. The PI was told the attribution cost and
reaffirmed the decision; it is recorded here rather than re-litigated.

The practical consequence, stated in advance so it is not discovered later: if this arm
loses to `refcv3-b1-v72-30k`, the programme learns *"this combination is worse"* and
nothing about **which** of ego-state, the vocabulary, the clamp or the goal-string caused
it. Recovering attribution would need a leave-one-out panel of four further arms at
~23 h each.

---

## 2. WHAT ACTUALLY LAUNCHED — the exact command (MEASURED from `ps`)

```
python3 -u /workspace/TanitAD/stack/scripts/refc_v3_train.py \
  --arm hier --size base \
  --v2-cache /root/data/train \
  --v7-labels /workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz \
  --eval-cache /root/data/eval \
  --eval-labels /workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz \
  --eval-every 500 --eval-batches 8 \
  --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --nav-from-v7 --u8-batches \
  --anchors /workspace/experiments/refcv4-b1-v72-40k/anchors.pt \
  --sel-accel-max 2.0 --goal-str \
  --ego-state-inject --ego-dropout 0.5 \
  --out /workspace/experiments/refcv4-b1-v72-40k
```

* PID 2330189 (+6 dataloader workers). Supervisor PID 2330171.
* First logged step **50** (`loss 140.0261 traj 21.3272`); by step 250 `loss 28.8173 traj 1.9656`.
* **Rate 2.052 s/step** over the first 513 s ⇒ **~23.0 h total** — an EARLY estimate that
  still contains model build and loader warm-up, so treat it as an upper bound on the mean.
* `--steps 40284` = one full epoch at batch 20 on the 4,572-clip train split. ⛔ Not a
  portable constant — it is windows/batch and must be recomputed per arm.
* ⚠️ The log records `[parity] ⚠ NON-PARITY v2 corpus` for `/root/data/train`. Results off
  this arm are **not cross-arm comparable with the parity arms** and must not be quoted as
  though they were.

**Supervisor hygiene — verified by content, and it is correct.** `sup_refcv4.sh` carries
`200>&-` on the trainer (`:101`) **and on every `sleep`** (`:131`, `:133`), and writes the
done-marker from a data-derived condition (`:119`, last step in `metrics.jsonl >= TARGET`).
Both of the 2026-09-02 supervisor defects are closed.

---

## 3. THE FIVE LEVERS — status, honestly

| | lever | in this arm? | evidence |
|---|---|---|---|
| **L1** | ego state + anti-echo | ✅ | `--ego-state-inject --ego-dropout 0.5`. `--echo-base` correctly OFF. |
| **L2** | 6 s data-driven anchor vocabulary | ✅ | `--anchors …/anchors.pt`, the k-means/slotnorm bank. Gate §4. |
| **L3** | reach clamp re-derived for the 6 s horizon | ✅ | `--sel-accel-max 2.0`. Kill rate §5. |
| **L4** | `(accel, curvature)` through `rollout_unicycle` + jerk / curvature-rate costs | ⛔ **NO** | §6 — and the reason is not an oversight. |
| **L5** | `tac_vocab_version` pinned, head/label widths agree, `goal_str` supervised, aux budget restored | ✅ | Pin at `refc_v3_train.py:160-161`, width refusal `:592-599`, `--goal-str` passed. Tactical aux budget **0.20 → 0.10** (`compute_losses_v3` was spending `0.05×2 + 0.05×2` against the invariant `refc_train.py:83-91` states in writing). `goal_str` is now really supervised: first row **0.76149**, where refcv3 had it in **0 of 614 rows**. |

So the live arm is **4 of 5**.

**Two further corrections the launching agent measured, recorded here because both
contradict the standing brief:**

* ⚠️ **`--graft-lan` is NOT `--goal-str`'s label path.** `--graft-lan` supplies the
  corridor as a **model INPUT** and is refused by E12 and the vision-only rule; its own
  help string says it is *"NOT part of any registered v3 arm"*. `--goal-str` **alone**
  mints the label (`refc_v3_train.py:1084`, `want_lan = bool(args.graft_lan or
  args.goal_str)`). The arm launched with `--goal-str` only, `graft_lan False` verified on
  the built config. Passing `--graft-lan` as the brief suggested would have opened a
  privileged-input back door.
* The run stamps `anchors.file_sha256` of the tensor **actually installed in the decoder**
  into `config.json` — not merely the presence of a flag. refcv3 recorded neither, which is
  why *"which vocabulary did it train on?"* had to be answered later by reading a launch
  script. **908 tests pass.**

---

## 4. GATE L2 — the anchor vocabulary. **PASS.**

Built by clustering real GT from the **B1 v7.2 TRAIN split**, scored on **held-out eval
clips** (n = **19,602 windows / 141 clips**), oracle-in-vocabulary over 0–2 s, **split
along vs lateral** as required:

| arm | ADE 0–2 s | ALONG | LAT | 0–6 s ADE | anchors used |
|---|---|---|---|---|---|
| CONTROL: zero path (no information) | 14.3264 | 14.3064 | 0.3160 | — | — |
| CONTROL: constant-velocity straight line | 0.6843 | 0.4783 | 0.3160 | — | — |
| INCUMBENT: synthetic 128 (refcv3's actual) | 1.0882 | 0.9087 | 0.4135 | — | — |
| best FPS (slotaxis_along0.25) | 0.7666 | 0.5478 | 0.3988 | 3.7595 | 98/128 |
| **WINNER: k-means, slotnorm** | **0.3796** | **0.3066** | **0.1496** | **1.4838** | **121/128** |

* **Beats the published floor 0.6780 m** (gate condition) and the straight-line control
  0.6843 m. **2.87× better than the synthetic incumbent**, +0.7086 m.
* ⭐ The brief anticipated fixing FPS's axis weighting. The measurement went further: **FPS
  is the wrong algorithm here regardless of metric** — every FPS variant lands 0.77–0.86 m
  while every k-means/k-medoid variant lands 0.38–0.39 m. Per-slot normalisation
  (`slotnorm`) is a second-order improvement *within* k-means (0.3796 vs 0.3867 raw).
* Both controls read sensibly (the zero path is catastrophic and almost purely along-track;
  the straight line reproduces the lateral floor exactly at 0.3160), so the instrument is
  measuring what it claims.

---

## 5. GATE L3 — the reach clamp, re-derived at the ACTUAL 6 s horizon

Chosen operating point `a_max = 2.0` ⇒ band ±12.0 m/s:

| quantity | value |
|---|---|
| kill rate | **26.23 %** |
| empty windows | **0.00 %** |
| GT deleted (eval / train) | **0.00 % / 0.0068 %** |
| survivors per window | 94.42 |
| turn candidates per window (ep / tm) | 19.04 / 29.28 |
| Δ ADE vs unclamped | **0.0000** |

The inherited justification ("inert on ADE, deletes 72.08 % / 77.28 %") was measured at
`horizon_s = 2.0` and does **not** transfer. `refc.py:652` derives
`horizon_s = max(horizons) × 0.1 = 6.0 s`, so the **inherited `sel_accel_max 2.5`** opens
the band to ±15.0 m/s and kills only **18.02 %** — not 72 %, and not the 37.1 % the brief
carried either. Re-deriving was correct and `refc_v3.py:79-82`'s written warning was
well-founded. Turns survive (0.00 % empty windows, 19.04 turn candidates/window >30°).

⭐ **The keeper, and it generalises beyond this knob:** the binding criterion is **GT
DELETION, not kill rate** — a band that removes the trajectory the ego actually **flew** is
wrong, not conservative. And the inherited **value** (2.5) was defensible while the
inherited **statistic** (72.08 %) was wrong by 4×. *A config that is right for the wrong
reason reads exactly like one that is right*, which is why the re-derivation had to happen
even though the outcome barely moved.

---

## 6. ⛔ L4 IS NOT IN THIS ARM — and three measured findings say it must not be bolted on

L4 was absent from the build when the launch fired. Verified by content, three ways:
`grep -c -E "jerk|unicycle"` = **0** in `tanitad/refs/refc_v3.py` and **1** in
`scripts/refc_v3_train.py` (line 396, a docstring about the CI synthetic corpus); no
`rollout_unicycle`, no `W_JERK`, no import from `tanitad.models.kinematic`; and the smoke
run's `config.json` carries no jerk/unicycle/kinematic key.

The standing instruction was to kill a 4/5 arm and relaunch. **That instruction was
conditioned on facts that measurement has since changed.** All three findings below are
MEASURED 2026-09-04 (`stack/tanitad/models/kinematic.py`, probes in §8):

**(a) The stated rationale for the cost terms is INVERTED.** The brief held that refav1's
jerk term sits on "channel 0, the steer channel", penalising lateral jerk and never
acceleration jerk, and that this is why our longitudinal deficit needs a new term. From
source, `refa_v1.py` uses the **same `(accel, curvature)` convention as this module**:

```
jerk = (controls[:, 1:, 0] - controls[:, :-1, 0]) / pc.dt     # channel 0
c += 0.05 * controls[..., 1].pow(2).mean(-1)   # curvature    <- channel 1 IS lateral
v_end = v0 + controls[..., 0].sum(-1) * pc.dt                 # channel 0 integrates to SPEED
```

⇒ refav1 **already penalises LONGITUDINAL jerk**. What it has never had is a
**curvature-RATE (lateral jerk)** term. The two-channel instruction is still right; its
justification was backwards, and building a 23 h arm on the inverted version would have
aimed the cost budget at the axis that is already winning.

**(b) L4 as briefed is not implementable on REF-C's grid.** The fan is emitted on
`horizons = [5,10,15,20,30,40,50,60]` frames at 10 Hz ⇒ slot times 0.5 … 6.0 s ⇒ per-slot
spacing **[0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0] s — NOT UNIFORM**. But
`rollout_unicycle`, `unicycle_controls_from_path` and `unicycle_decode` all take a **scalar
`dt`**. Reconstructing **real human GT** through the scalar path (n = 8,336 windows):

| dt used | reconstruction error of real GT (mean) | p99 |
|---|---|---|
| 0.75 s (the naive "6 s / 8 slots") | **10.9527 m** | 50.4340 m |
| 1.00 s | 22.0296 m | 101.4049 m |
| 0.50 s | 0.5232 m | 3.5000 m |
| **correct per-step dt** | **0.2574 m** | — |

And it fails **silently** — shape unchanged, loss still falls, ADE still computes. This is
the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s` family in an integrator's
costume.

**(c) L2 has already discharged L4's feasibility motivation.** The brief's case rested on
"41/128 anchors break a dry-road Kamm circle" — a property of the **synthetic** pool that L2
has now replaced. On the same slot grid, with real human GT as the control that must read a
known value:

| | n | \|accel\| p99 | \|kappa\| p99 | jerk barrier | curv-rate barrier |
|---|---|---|---|---|---|
| **data-driven anchors** | 128 | **1.9881** | **0.1203** | **0.0000** | **0.0000** |
| CONTROL: real human GT | 8,336 | 2.8860 | 0.1631 | 0.0014 | 0.0008 |

The new vocabulary is **already cleaner than the humans it was clustered from**. The
residual L4 motivation is only the free-form per-waypoint offset — real, but smaller than
advertised.

⭐ **Independently corroborated by a second instrument, from a different agent, on a
different statistic.** The launching agent measured the OLD synthetic vocabulary as not
merely coarse but **UNFLYABLE**: max `|accel|` **14.15 m/s²** with **10.71 %** of segments
above 8 m/s², and **no turn sharper than 74.5°**. Root cause named in source:
`synth_anchor_pool` samples `v ≤ 30 m/s` and `|yaw_rate| ≤ 0.35 rad/s`
**INDEPENDENTLY**, so it emits an 86 m radius at 108 km/h = **1.07 g lateral**. The new
set reads max `|accel|` **4.30 m/s²**, **0.00 %** above 8, and turns to **178.4°**.
Two instruments that share no code — recovered unicycle controls plus smoothness
barriers here, raw accel/turn statistics there — agree that L2 fixed the feasibility
problem L4 was partly meant to fix. That is a genuine second probe, not the same query
run twice.

**(d) A control-space decode breaks a documented architectural invariant.**
`refc_v3.py:383` / `:763` rely on a **zero-init delta head so the model STARTS at the
kinematic extrapolation**. `unicycle_decode` seeds the rollout with the ego's **measured
v0**, while the inverse map recovers speed from the path — and those differ (p50 0.1196,
p95 0.4489, p99 0.6837 m/s). Round-tripping an anchor through the decode with measured v0
costs **0.6069 m mean / 3.8328 m p99**: the emitted path no longer starts at its anchor.
Not a defect — it is `entry_speed_mismatch` being applied — but it is a real change to a
load-bearing property and must be a deliberate choice, not a side effect.

⇒ **L4 is deferred to arm 2, and must clear the `TanitAD_ValidateAIDesign` tiny ladder
before it earns 23 h of A40.** Killing a live, gate-passing arm to force in a lever whose
rationale is inverted, whose feasibility case is already satisfied, and whose implementation
needs an unvalidated design change would not have been defensible.

---

## 7. ACCEPTANCE CRITERIA — committed in advance

⛔ **The bar is NOT ADE.** An image-blind arm beat `ha` by +9.42 % and `ha0_ext` by +4.67 %,
and `ego_dropout 0.5` was the only scene-reading lever while being the **worst** on ADE.

* **PRIMARY / decisive — the scene-reading echo gate** (`stack/tanitad/eval/echo_gate.py`).
  The arm **PASSES** iff it reads `READS_BOTH` (not `ECHOING`). An arm that improves ADE
  while reading `ECHOING` is a **FAILURE** and must be reported as one.
* **SECONDARY / reported, not decisive — ADE** against the standing reference points:
  `ha` **0.2996**, `os` @40,284 **0.4419**, `ha0` **0.6723**.
* **Estimator, committed now:** paired **episode-cluster bootstrap** over the eval episodes
  (`taniteval/ci.py`), paired on identical windows. ⛔ Never `overlapping_holdout_se`, which
  biases the point estimate as well as the interval.
* **The four binding metric families** (LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC) are
  reported per family with their own CIs. An ADE-only table is an incomplete result.
* **Failure is reportable.** A regression is a result; with four levers moving it is a
  result about the *combination* only (§1).

---

## 8. ARTIFACTS

| artifact | where |
|---|---|
| L4 kernel (7 additive functions) | `repo: stack/tanitad/models/kinematic.py` |
| Anchor bank (k-means/slotnorm, 128×8×2) | `pod:/workspace/refc_anchors_6s_b1train_128.pt` + `.json`, copied to the run dir as `anchors.pt` |
| Anchor gate log | `pod:/workspace/anchors6s.log` |
| Clamp derivation | `pod:/workspace/clamp6s.json`, `pod:/workspace/derive_clamp6s.py` |
| L4 validation probes | `pod:/workspace/l4_probe.py`, `pod:/workspace/l4_probe2.py` |
| Live run | `pod:/workspace/experiments/refcv4-b1-v72-40k` |

⚠️ **Stranded — exists in ONE place only:** the anchor bank, the anchor gate log, the clamp
derivation and both builder scripts live **only on the pod**. They are the evidence for §4
and §5 and should be pulled into the repo.
