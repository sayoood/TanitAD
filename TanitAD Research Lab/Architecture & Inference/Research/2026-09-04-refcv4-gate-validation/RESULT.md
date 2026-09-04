# The refcv4 abort gate, recomputed — and the reason it must not fire

**2026-09-04 · independent validator with abort authority · MEASURED unless marked**
**VERDICT: ⛔ DO NOT ABORT.** The run is healthy and continues. The **acceptance
number was applied to the wrong quantity**, and I can falsify its premise with a
number banked in this repo.

---

## 0. The one-line answer

| | |
|---|---|
| gate as issued | raw **oracle-in-vocabulary** ADE (0–2 s) must beat `ha` = **0.2996 m** |
| recomputed, refcv4's shipped vocabulary | **0.3773 m** — ALONG **0.3041** / LAT **0.1503** |
| paired vs `ha` | **+0.0777 m [+0.0528, +0.1044]**, separated ⇒ **fails as issued** |
| ⭐ but the premise *"the arm cannot beat hold-action however well it trains"* | **FALSE, MEASURED** |

**Why.** The 0.2996 bar was justified by `oracle_sel` = 0.3668 — a **model-inclusive**
quantity — and then applied to the raw, **model-free** min-over-anchors ADE. Those are
different numbers about different objects. On the identical 4,823 windows:

| | raw oracle-in-vocab | refined oracle (`oracle_sel`) |
|---|---|---|
| **refcv3** (synthetic 128) | **1.0838 m** | **0.3668 m** |
| **refcv4** (shipped 128) | **0.3773 m** | not yet measurable (needs the checkpoint) |

refc's decoder moved refcv3's oracle-selected path by **−0.7170 m (−66.2 %)** from its
raw vocabulary. ⇒ **Applying the 0.2996 bar to the raw quantity would also have aborted
refcv3** — the arm that nonetheless reached 0.3668 refined. The raw vocabulary is an
**initialisation**, not a ceiling.

**Mechanism, read from the tree the live run imports** (`refc.py`, pod md5
`7b1f359bb07dfd5650d65be658d08c55`):

```
refc.py:1400   x = anchors[None] + offset            # emitted fan = anchors + offset
refc.py:_decode  offset = self.offset_head(q).reshape(b, n, self.n_steps, 2)
```

`offset` is a free-form linear head — unclamped, unscaled. `out["anchor_traj"]` is `x`,
the **refined** fan, and `refcv3_arm.py:1051-1058` builds `oracle_sel` as
`anchor_traj[0, a_star]` where `a_star` is the argmin over the **raw** bank. So
`oracle_sel` measures selection **plus** refinement; the raw min-over-anchors measures
neither.

---

## 1. The recompute — surface, controls, numbers

⛔ **I did not trust the sibling's self-report; I recomputed on the surface the
0.2996 was measured on.** The sibling scored 19,602 windows; `ha` = 0.2996 was
measured on **4,823**. A cross-surface comparison is not a paired one.

**Surface:** the banked per-window dump
`taniteval/results/refcv3-40284-openloop-dump.tar.gz` — **n = 4,823 windows / 141
episodes**, `--window-stride 5`, K = 4 instants **0.5/1.0/1.5/2.0 s**, ego frame at t0,
metres. **Estimator:** paired episode-cluster bootstrap, `n_boot 2000`, seed 0, cluster =
the episode. **Zero GPU, no model, no forward pass** — minutes on CPU.

### 1.1 Controls that must read known values (and do)

| control | expected | read |
|---|---|---|
| `ha` reproduced from the dump | 0.2996 | **0.299618** (Δ +1.8e-05) |
| `ha0` | 0.6723 | **0.672286** (Δ −1.4e-05) |
| `os` | 0.4419 | **0.441935** (Δ +3.5e-05) |
| `oracle_sel` | 0.3668 | **0.366789** (Δ −1.1e-05) |
| frame/units: `ha0` == `(v0·t, 0)` | 0 | max abs err **2.3e-05 m** ⇒ x is ALONG, y is LATERAL, metres |
| zero-path (no information) | large | **14.2483 m** |

All four published arms reproduce to **< 4e-5 m**. If they had not, nothing below
would count.

### 1.2 The gate

| vocabulary | ADE 0–2 s | ALONG | LAT | distinct anchors used | paired Δ vs `ha` |
|---|---|---|---|---|---|
| refcv3 SYNTHETIC (incumbent) | 1.0838 | 0.9090 | 0.4089 | 59 / 128 | +0.7842 [+0.7278, +0.8628] |
| ⭐ **refcv4 NEW (shipped)** | **0.3773** | **0.3041** | **0.1503** | **122 / 128** | **+0.0777 [+0.0528, +0.1044]** |

* The sibling's own numbers (0.3796 / 0.3066 / 0.1496 on **19,602** windows) reproduce
  here to **0.0023 m** on a different window set. **Their measurement is right; only the
  acceptance number was wrong.**
* ⭐ **The gain is on the right axis.** ALONG-track falls **0.9090 → 0.3041 (−0.6049)**
  and LATERAL **0.4089 → 0.1503 (−0.2586)**; **70.1 % of the summed reduction is
  along-track**, which is where `D-REFCV3-AXIS1` put 92.2 % of the deficit.
* **Coverage tripled**: 59 → **122** of 128 anchors are the nearest for at least one
  window. refcv3 was carrying **69 dead anchors**.

### 1.3 The corrected reading

refcv4's **raw** vocabulary (0.3773) already sits **at** refcv3's **refined** ceiling
(0.3668) *before any refinement at all*. To clear `ha` the refinement head must buy
**0.794×** (a 20.6 % improvement on its own starting point); refcv3's head demonstrably
bought **0.338×**. `ESTIMATED`, from one arm — a scale statement, not a promise.

⚠️ **The honest residual risk, stated plainly:** `os` (0.4419) and `oracle_sel` (0.3668)
differ by only 0.0751 m while the raw anchor behind `oracle_sel` was 1.0838 — i.e. on
refcv3 the offset head was already delivering nearly all of the performance and the
vocabulary's raw quality contributed little to the *final* numbers. It is therefore
**possible** that the binding constraint is perception, not vocabulary, and that refcv4's
refined oracle lands near refcv3's. That is a real hypothesis. It is **not** decidable
from the anchor set, and it is **not** grounds to destroy 21.8 h of A40 before the first
checkpoint can answer it.

### ⇒ 1.4 The gate that replaces it (mid-run, cheap, decisive)

**At the first pulled checkpoint, score refcv4's `oracle_sel` on these same 4,823
windows** (`taniteval/tools/openloop_suite.py --with-oracle-sel`, the instrument that
produced the 0.3668). **If refined `oracle_sel` ≥ 0.2996 separated, the arm cannot beat
hold-action and should be killed then.** That is the model-inclusive ceiling the 0.2996
was always about, and it is answerable from a checkpoint rather than from an anchor file.

---

## 2. The run, verified by content

| check | reading |
|---|---|
| supervisor | PID **2330171** `bash /workspace/sup_refcv4.sh` — alive |
| trainer | PID **2330189** + 6 workers — alive, A40 **42,669 / 46,068 MiB, 100 %** |
| anchors installed | `sha256 68f81acf…a806b` — **identical** at `…/refcv4-b1-v72-40k/anchors.pt`, `/workspace/refc_anchors_6s_b1train_128.pt` and my local copy. This is the file I scored. |
| `config.json` | `sel_accel_max 2.0` · `horizon_s 6.0` · `band_ms 12.0` · `tac_vocab_version "v7.0"` |
| rate | step 300 @ `elapsed_s` 593.8 ⇒ **1.95 s/step** (Δelapsed/Δstep, NOT `step_s`) ⇒ **~21.8 h**, not 53 h |

---

## 3. Zero-GPU checks, on the same 4,823 windows

Run through **the programme's own** `refc_select.anchor_reachability_mask` →
`flagship_v15.reachability_mask` (`v_mean = ‖wp_last‖ / 6.0`, band `v0 ± a·6.0`), never a
re-implementation.

### 3.1 Reach clamp

| vocabulary | a_max | band | kill % | empty % | surv/win | >30° turns/win (end-bearing / terminal-heading) | windows with NO >30° turn |
|---|---|---|---|---|---|---|---|
| refcv3 | 2.5 *(shipped)* | ±15.0 | **37.10** | 0.00 | 80.5 | 40.4 / 57.5 | **0.00 %** |
| refcv3 | 2.0 | ±12.0 | 48.36 | 0.00 | 66.1 | 33.2 / 47.2 | 0.00 % |
| refcv4 | 2.5 | ±15.0 | 18.41 | 0.00 | 104.4 | 20.4 / 31.4 | 2.53 % |
| ⭐ **refcv4** | **2.0** *(shipped)* | **±12.0** | **26.60** | **0.00** | **94.0** | **19.0 / 29.2** | **4.62 %** |
| refcv4 | 1.5 | ±9.0 | 38.46 | 0.00 | 78.8 | 16.6 / 25.4 | 7.40 % |

* **refcv3's shipped clamp reproduces exactly**: 37.10 % killed (brief: 37.1 %), **40.4**
  turning survivors/window (brief: 40.4), **0.00 %** empty (brief: 0.00 %). The
  instrument is reading the right thing.
* refcv4's shipped clamp reproduces the sibling's sweep to within rounding (26.60 vs
  26.23 %, 94.0 vs 94.4 surv/win, 19.0 vs 19.0 turns/win).
* ⚠️ **One genuine regression, and it is not in the sibling's headline.** refcv3 kept a
  >30° turn in **every** window; refcv4 leaves **4.62 %** of windows with **none**
  (7.40 % at a = 1.5). The cause is vocabulary composition, not the clamp: refcv4 has
  **23** anchors with a >30° end-bearing against refcv3's **61**. The sibling's
  "physics, not a defect" reading is defensible at high `v0` — but the fact that the
  incumbent never had this hole should be on the record.

### 3.2 Kamm circle (finite differences over the anchor's own 0.5–6 s slots)

| vocabulary | μ=0.7 | μ=0.8 | μ=0.9 | μ=1.0 | peak | max \|a_lon\| | max \|a_lat\| | v_max | turn-while-stalled | max turn |
|---|---|---|---|---|---|---|---|---|---|---|
| refcv3 | **43**/128 | 29 | 20 | 12 | **1.50 g** | 2.99 | **14.38** | 46.13 m/s | 2 | 74.5° |
| ⭐ **refcv4** | **0**/128 | **0** | **0** | **0** | **0.44 g** | 2.15 | 4.29 | 36.21 m/s | **1** | **178.4°** |

⇒ **refcv4's vocabulary is entirely flyable** — nothing breaks even a μ=0.7 circle, at a
peak of 0.44 g — while refcv3's broke a dry-road circle on 43 anchors and peaked at
1.50 g. (The brief's "41 of 128, peak 1.4 g, 5 stalled" is the same finding at a
slightly different threshold/segmentation; my definitions are stated above so the two
are reconcilable rather than merely close.)

### 3.3 Head width vs label width — ✅ **PINNED AND ASSERTED, on both surfaces**

| check | reading |
|---|---|
| version pinned in `config.json` | `tac_vocab_version = "v7.0"`, derived at `refc_v3_train.py:160` from `--v7-labels` |
| v7.2 label widths | `tac_lat` **8**, `tac_lon` **8** (`tanitad/data/v7_labels.py::HEADS`) |
| model z_tac heads | `lat_head_tac` **(8, 512)**, `lon_head_tac` **(8, 512)** — **match** |
| model core heads | `core.lat_head` **(3, 384)**, `core.lon_head` **(3, 384)** — kin3, **match** |
| assertion | `refc_v3_train.py` raises on **both** — the core check *and* a mirrored z_tac check whose own message names the 2026-09-01 defect ("a v3 build once passed the core check and trained 8-wide z_tac heads against 3-class labels") |
| passed by content | the live run has taken **300+ steps** with `tac_label_v7 = 1.0`, i.e. it went through this exact branch with v7.2 labels and did not raise |

### ⛔ 3.4 ESCALATION — Defect A is live, and it is worse than "four classes get 0 % mass"

`refc_v3.py:903-905` feeds the **8-wide** v7.2 `z_tac` logits into
`refc_tactical.derive_man5_logprobs`, which is **defined on `[B,3] × [B,3]`** and indexes
fixed kin3 slots. Resolved numerically against the live vocabulary:

| the 5-way slot | reads v7.2 token | should be |
|---|---|---|
| `turn_left` | **`LANE_CHANGE_L`** | `TURN_L` |
| `turn_right` | **`LANE_CHANGE_R`** | `TURN_R` |
| `steady` | `CRUISE` | (defensible) |
| `accelerate` | ⛔ **`YIELD_MERGE`** | `ACCELERATE` |
| `brake_stop` | ⛔ **`FOLLOW`** | `BRAKE_TO` |

**Never read at all:** lat `ABORT_LC, NUDGE_L, NUDGE_R, TURN_L, TURN_R`; lon
`BRAKE_TO, CREEP, HOLD, ADAPT_SPEED_FOR_CURVE, ACCELERATE` — **10 of 16 classes**.

⚠️ **This is not inert telemetry.** The hook returns `{"maneuver_logits": man5}`
(`refc_v3.py:944`), and `refc.py:1405-1407` feeds it through
`maneuver_to_anchor(log_softmax(maneuver_logits))` — the **H19 anchor prior that
reweights anchor confidences**. So a semantically scrambled 5-way vector is perturbing
anchor selection in the live run, and three of its five slots are wired to the wrong
token rather than merely to an empty one. The sibling recorded Defect A as *"four classes
get 0.000 % mass"*; the sharper statement is that the **surviving** slots are mislabelled
too. Recorded, not patched — patching a live run is worse.

---

## 4. Deliverable manifest

| artifact | where it lives |
|---|---|
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-gate-validation/RESULT.md` |
| gate artifact (both vocabularies, controls, CI) | `repo:…/raw/REFCV4_ANCHOR_GATE.json` |
| clamp + Kamm artifact | `repo:…/raw/ZERO_GPU_CHECKS.json` |
| the two instruments | `repo:…/raw/scripts/gate_both.py`, `…/zero_gpu_checks.py` |
| refcv3's ACTUAL anchor bank, extracted from `ckpt_40284_FINAL.pt` | `repo:…/raw/refcv3_anchors.pt` (sha256 `b31a17f4…1412`) — was **checkpoint-only** before this |
| tiny-rig panel + arms | `devbox:C:\Users\Admin\run_refcv4v` — see §5 |

## 5. Escalations for the Master Mind

1. ⛔ **The abort gate is mis-scoped and must be restated** before anyone else applies
   it. Raw oracle-in-vocabulary is not a ceiling for a select-**then-refine** decoder.
   The replacement is §1.4.
2. ⛔ **Defect A perturbs anchor selection through a mislabelled H19 prior** (§3.4),
   not merely a zero-mass one.
3. ⚠️ **refcv4 loses the "a sharp turn exists in every window" property** refcv3 had
   (§3.1). Cheap fix if wanted: k-medoid + a turn-coverage constraint at rebuild.
4. **ETA is ~21.8 h, not ~53 h** (MEASURED at step 300).
