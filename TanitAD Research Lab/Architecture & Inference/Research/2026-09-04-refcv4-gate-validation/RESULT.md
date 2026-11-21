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

### ⛔ 1.3b And the quantity called "the fan's ceiling" is not the fan's ceiling

`oracle_sel` picks `a_star` = the argmin over the **RAW anchor bank**
(`refcv3_arm.py:1054`) and then reads **that anchor's refinement**. The true
oracle over the emitted fan is `min` over all 128 **refined** trajectories, which is
**≤ `oracle_sel` by definition**. So refcv3's real fan ceiling is *below* 0.3668 by an
unmeasured amount, and "the fan's ceiling is below the bar" rests on an **upper bound
on the ceiling**, not the ceiling. `refc.py:1454` records the quantity by name —
*"the published oracle-in-fan (0.1914 base / 0.1640 XL)"*, `INHERITED` from a code
comment, not re-verified here — and **0.1914 is well below `ha` = 0.2996**.

⚠️ **There is no oracle-in-fan arm in the harness.** `oracle_in_fan`, `best_in_fan`
and `fan_oracle` all return 0 matches in `taniteval/tools/refcv3_arm.py`. Building one
is ~2 lines beside `:1051` (argmin over `out["anchor_traj"]` instead of over
`anchors_bank`) and it is the instrument this whole argument has been missing.

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
| rate | step 1150 @ `elapsed_s` 2372.1 ⇒ **2.059 s/step** over 1,100 steps (Δelapsed/Δstep, ⛔ NOT `step_s`); recent 250 steps **2.191** ⇒ **ETA ~23.8 h**, not 53 h |
| health at 1150 | `tac_label_v7 1.0` · `goal_str 0.78825` · `anchor_acc 0.15` (chance 1/128 = 0.0078, so **19× chance**) |
| ⭐ supervisor lock-fd trap | **CLEAN.** `sup_refcv4.sh` carries `200>&-` on the trainer (`:101`) **and on both `sleep`s** (`:131`, `:133`), and the `/proc/*/fd` scan names exactly **one** holder of `/workspace/.sup_refcv4.lock` — PID 2330171, the supervisor itself. Neither the trainer nor its six workers hold it, so a supervisor restart is possible without killing the run |
| ⚠️ parity | `v2_parity.parity false`, `checked false`, `corpus_key null` — refcv4 is **non-parity like refcv3**. refcv4-vs-refcv3 is a valid arm delta (same `/root/data/train`); a LEVEL against `refc-base` is not |
| route leak | **none.** `graft_lan` absent; `provenance_roles.refused_edges` names `lan -> inference (E12; label-only)` and `situation classifier output -> any goal node` |

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

## ⭐ 3.5 PRE-REGISTRATION FOR THE NEXT ARM — the deferred lever is worth more than the one that shipped

The shipped vocabulary is **128 fixed ego-frame paths in absolute metres**; it is
**not conditioned on the window's speed**, while `ha` is. The sibling deferred
*"(accel, curvature) through `rollout_unicycle`"* to the next arm. That lever is
priceable **now, zero GPU**, on the same 4,823 windows, through the programme's own
integrator (`refa_v1_plan.unicycle_paths`, `action_units="kappa"`).

⛔ **Control that must read a known value:** a **1-point** grid `{a=0, κ=0}` must
reproduce `ha0`. It reads **0.672288** against the published **0.6723** — Δ **−1.2e-05**.

| v0-conditioned (a, κ) grid over a ∈ [−4, 3] m/s², κ ∈ [−0.06, 0.06] 1/m | M | ADE | ALONG | LAT | paired vs `ha` |
|---|---|---|---|---|---|
| 13 × 9 | 117 | ⭐ **0.2572** | 0.1419 | 0.1703 | **−0.0424 [−0.0685, −0.0146] BEATS** |
| 11 × 11 | 121 | **0.2610** | 0.1598 | 0.1567 | **−0.0386 [−0.0642, −0.0116] BEATS** |
| 17 × 7 | 119 | 0.2666 | 0.1295 | 0.1930 | −0.0330 [−0.0634, +0.0022] tied |
| 9 × 15 | 135 | 0.2759 | 0.1991 | 0.1361 | −0.0237 [−0.0491, +0.0025] tied |
| 7 × 19 | 133 | 0.3487 | 0.2922 | 0.1257 | +0.0491 loses |
| *(the SHIPPED fixed-path vocabulary, for scale)* | 128 | **0.3773** | 0.3041 | 0.1503 | **+0.0777 loses** |

⇒ **At the SAME 128-candidate budget, a v0-conditioned kinematic vocabulary CLEARS
0.2996 where the shipped fixed-path one does not** — same surface, same metric, both
raw min-over-vocabulary, no model in either. The ordering is driven by **accel**
resolution, i.e. the **along-track** axis, which is where 92.2 % of the deficit lives.

⛔⛔ **A trap worth pinning: the vocabulary must contain the straight-ahead control
EXACTLY.** My first grids used even counts, so `np.linspace(−0.06, 0.06, n)` omitted
`κ = 0`. That alone read **1.2768 m** (LAT 1.1859) against **0.2608 m** for an
otherwise-comparable grid containing zero — a **4.9×** difference that looks exactly
like a resolution finding and is not one. Every grid in the table above contains
`a = 0` and `κ = 0` exactly.

> ⛔⛔ **RETRACTED 2026-09-05 (RETRACTION_LOG #22) — the last sentence is FALSE for the accel axis.**
> Every grid in the table above used `np.linspace(-4.0, 3.0, na)` (`raw/scripts/kinvocab_probe.py:102`),
> whose step is `7/(na−1)`; for na ∈ {7, 9, 11, 13, 17} it contains **no** `0.0` (nearest node −0.5,
> +0.375, +0.2, **+0.0833**, −0.0625 — MEASURED, numpy). Only the κ axis, `np.linspace(−0.06, 0.06, nk)`
> with odd nk, was symmetric-and-odd and contained zero. The 1-point `{a=0, κ=0}` control was a SEPARATE
> grid (`kinvocab_probe.py:85`), which is why it read the right value while this sentence did not hold.
> ⇒ **0.2572 / 0.2610 / 0.2666 / 0.2759 / 0.3487 are numbers for grids WITHOUT the constant-velocity
> control**, and "odd counts" is not the rule — *0.0 is a node of BOTH axes* is (assert it, or re-centre an
> asymmetric range as `emit_anchors_alat.py` does). The verdict of §3.5 survives in direction:
> `PREREG_REFC_V4.md` §A10.2 re-measured a RE-CENTRED 13 × 9 (zero asserted as a node) at **0.2610** and the
> live `alat` 117 family reads **0.1987**, both separated below `ha`. The trainer's refusal text that
> inherited "Rebuild with odd counts" from this paragraph is corrected in `refc_v3_train.py`.

⚠️ **Scope:** constant `(a, κ)` held for **2 s**, scored at 0–2 s, because that is
where `ha` = 0.2996 was measured. The run plans **6 s**; a constant-control family
will be weaker there and this does **not** transfer to the 6 s horizon unmeasured.

## ⭐ 3.6 THE TINY RIG — the combined arm READS THE SCENE, and the gate proves it can fail

**5 arms × 2,000 steps**, `--size tiny --episodes 48 --batch 12 --lr 1e-4 --warmup 250
--seed 0`, dev-box RTX 4060, **sequential** (`OMP_NUM_THREADS=6`). ⛔ **Scored with the
anti-echo gate, NOT with ADE.** The rig tree was built by overlaying **the POD's OWN
`stack/`** (tar md5 `3c7ef7dc94c504fac79b42a3f4b62f61`, per-file md5 matched:
`refc_v3_train.py` `bb97c5d3…`, `refc_v3.py` `3861a33b…`, `refc.py` `7b1f359b…`) so the
rig runs **the bytes the A40 imports**, verified by `import tanitad; print(__file__)`.

⚠️ **RIG SCOPE (H-SCALE-2):** 17 M params on the **NON-PARITY** local epcache
(`physicalai-val-bb543bdf7836`), `tac_label_v7 = 0.0` (the epcache has no v7.2 labels,
where the A40 has `1.0`). This validates the **design and the gate**, never a model claim.

**The ladder — each rung adds ONE lever group** (⚠️ the tactical-aux `/2` is
*unconditional* in this trainer, so it is HELD CONSTANT across all five, not tested):

| arm | levers added |
|---|---|
| `A_v3` | — (incumbent) |
| `B_ego` | `--ego-state-inject --ego-dropout 0.5` |
| `C_anch` | + `--anchors <new> --sel-accel-max 2.0` |
| **`D_full`** | + `--goal-str` — ⭐ **the LAUNCHED lever set** |
| **`E_regress`** | + `--ablate-frames` — ⛔ **the deliberate regression (image-blind)** |

### 3.6.1 ⭐ The verdict — GATE 2b, source ablation (scene ⊥ ego)

n = **640 windows / 40 episodes**, manoeuvre census **189 straight / 451 non-straight**.

| arm | scene used? | scene degradation | ego used? | ego degradation | **verdict** |
|---|---|---|---|---|---|
| `A_v3` | True | 0.1568 | False | 0.0000 | `IGNORES_EGO` — **correct by construction**, it has no ego input |
| `B_ego` | True | 0.4247 | True | 0.1168 | `READS_BOTH` |
| `C_anch` | True | 0.3621 | True | 0.1190 | `READS_BOTH` |
| ⭐ **`D_full`** | **True** | ⭐ **0.6028** | True | 0.0661 | ⭐ **`READS_BOTH`** |
| ⛔ **`E_regress`** | **False** | ⛔ **0.0000** | True | **2.4875** | ⛔ **`ECHOING`** |

⭐⭐ **THE GATE IS ABLE TO FAIL, AND IT FAILS EXACTLY THE ARM IT MUST.** `E_regress` is
the **only** arm called `ECHOING` (scene degradation **0.0000** against a required 0.05).
Without that arm a PASS would mean nothing; with it, the PASS on `D_full` means something.

⭐ **The five levers together INCREASE scene reading rather than degrading it.**
`D_full`'s scene degradation **0.6028** is **1.42×** `B_ego`'s and **3.84×** `A_v3`'s —
the highest of any arm. **No adverse lever interaction is visible at 2,000 steps.**

### 3.6.2 ⛔ Two instrument findings the panel produced for free

1. **`gate2` (ego intervention) reads `READS_BOTH` for the IMAGE-BLIND arm.** Only
   `gate2b` (source ablation) catches it. ⇒ **`gate2` alone is not a gate** — a design
   validated on it would pass an arm with literally no scene input.
2. ⭐⭐ **The mission's warning reproduced in-panel: an ADE gate selects for the echo.**
   On the composite gate's ADE clause (GATE 1, 6 s), the **image-blind** `E_regress` is
   the **BEST** arm — relative margin **+0.0970 vs `ha`** — while every scene-reading arm
   is **negative** (`D_full` −0.1221, `C_anch` −0.1476, `B_ego` −0.1384). **The arm that
   reads nothing wins on ADE and loses on scene.** ⇒ GATE 1 raised for all four ego arms;
   at 2,000 steps on a 17 M rung that is expected and is **not** the acceptance criterion.

Controls, same windows: `constant_only` **6.4448 / 13.2513 / 20.3085** at 2/4/6 s (the
no-information value, above every arm); `ha` 0.9922 / 5.7637 / 15.1481; `ha0` 1.9666 /
7.3733 / 15.4740; `ha0_ext` 1.3290 / 6.0900 / 14.5494.

### 3.6.3 ⚠️ Reported honestly: one arm died on the first pass, and it was probably my fault

`C_anch` exited **rc 127 at step 300** on pass 1, with **no traceback** — its `.log` held
only the 4 startup lines, so stdout was never flushed, which is the signature of a hard
kill rather than a Python exception. **It coincides exactly (09:56–09:58) with a
CPU-heavy probe I was running on the same 31.8 GB box**, so the leading hypothesis is
self-inflicted host contention; I did **not** establish the cause and do not claim it.
Re-run alone on pass 2: **rc 0, 2,000 steps, `ckpt.pt` 204,389,943 B**. All five arms in
the table above carry a `summary.json` with `"done": true` and a checkpoint —
**verified by content, not by the driver's exit code**, which is what caught this.

## 4. Deliverable manifest

| artifact | where it lives |
|---|---|
| this report | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-04-refcv4-gate-validation/RESULT.md` |
| gate artifact (both vocabularies, controls, CI) | `repo:…/raw/REFCV4_ANCHOR_GATE.json` |
| clamp + Kamm artifact | `repo:…/raw/ZERO_GPU_CHECKS.json` |
| the instruments | `repo:…/raw/scripts/gate_both.py`, `…/zero_gpu_checks.py`, `…/gate_recompute.py`, `…/kinvocab_probe.py` |
| kinematic-vocabulary pre-registration | `repo:…/raw/KINVOCAB_PROBE.json` |
| tiny-rig gate report + panel + scorer | `repo:…/raw/rig/gate_report.json`, `…/rig/panel.sh`, `…/rig/gate_score.py`, `…/rig/arm_configs/*.json` — **and `devbox:C:\Users\Admin\run_refcv4v`** (the 5 checkpoints, ~1 GB, live in ONE place only and are rig-scope screening artifacts) |
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
4. **ETA is ~23.8 h, not ~53 h** (MEASURED over 1,100 steps).
5. ⭐ **The deferred `(accel, curvature)` lever beats the one that shipped** (§3.5) — a
   v0-conditioned vocabulary of the same budget clears 0.2996 where the fixed-path one
   does not. That is the next arm, and it is now pre-registered with both outcomes.
6. ⛔ **`gate2` alone is not a gate** (§3.6.2) — it passes an image-blind arm. Any design
   validated on the ego-intervention probe without the source ablation must be re-read.
7. ⚠️ **RESOLVED BY 11:20, RECORDED BECAUSE THE FAILURE MODE IS INVISIBLE.** A 0-byte
   `.git/index.lock` (created 09:52:44) blocked every agent's `git add` for ~90 minutes.
   ⛔ **`git add` returned exit 0 and staged nothing** — the *"Another git process seems
   to be running"* message goes to **stderr**, so my `git add … 2>&1 | Out-Null` loop
   reported ten clean passes while the index never moved. The check that caught it was a
   **blob comparison against the index** (`git ls-files --stage` vs `git cat-file blob`),
   not the exit code, and the fresh-inode un-poison did **not** help because the file was
   never the problem. I did **not** clear the lock (memory `git-index-corruption-on-gdrive`
   records corruption from removing a lock while an orphaned git process lived, and a
   **hung `git grep` from 07:54:57, PID 24768, is still alive**); it cleared on its own.
   `mm_commit.py` is unaffected — private index — which is how this landed.
8. ⚠️ **`mm_commit.py` must be run FROM THE REPO ROOT and with a private `TEMP`.** Three
   attempts failed: from another cwd `git hash-object` cannot open the work-tree paths,
   and with the shared `%TEMP%` `read-tree` failed 8/8 on `Unable to create <scratch>`.
   Repo-root cwd + `TEMP`/`TMP` pointed at a session-private directory committed first try.
